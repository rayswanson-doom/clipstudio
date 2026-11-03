"""
backend/visual_cues.py  -  turn written visual directions ("zoom in", "pan right", "flash") into
real camera moves on a finished video. CPU only: OpenCV warps each frame, ffmpeg re-encodes.

    apply_cues_to_video("in.mp4", cues, "out.mp4")      # cues come from scorer.build_narration()

A cue is {"effect": "zoom_in", "start": 2.0, "end": 5.5, ...}. Effects:
  zoom_in / zoom_out   gentle push in / pull out over the cue
  pan_left / pan_right slide across the picture over the cue
  fade                 fade in from black at the cue start
  flash                short white camera-flash at the cue start
  shake                short impact shake at the cue start
Zoom and pan ease back to normal after the cue ends, so nothing snaps. Cues with effect "none"
are notes only and change nothing.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    from .avatar_sync import open_encoder
    from .video_engine import find_ffmpeg
except ImportError:  # pragma: no cover
    from avatar_sync import open_encoder
    from video_engine import find_ffmpeg

EFFECTS = ("zoom_in", "zoom_out", "pan_left", "pan_right", "fade", "flash", "shake")
ZOOM_AMOUNT = 0.18  # +18 % at full zoom
PAN_AMOUNT = 0.07  # fraction of the width travelled
RELAX_SECONDS = 0.4  # how long zoom/pan take to ease back after a cue
FADE_SECONDS = 0.7
FLASH_SECONDS = 0.28
SHAKE_SECONDS = 0.6


def _ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def has_effects(cues: Optional[Sequence[dict]]) -> bool:
    return any(c.get("effect") in EFFECTS for c in cues or [])


def frame_state(cues: Sequence[dict], t: float) -> dict:
    """Combine every cue into {"scale","tx","brightness","white","shake"} at time t.
    scale: 1.0 = unchanged; tx: horizontal shift as a fraction of the width; shake: 0..1."""
    scale, tx, bright, white, shake = 1.0, 0.0, 1.0, 0.0, 0.0
    for c in cues:
        eff = c.get("effect")
        if eff not in EFFECTS:
            continue
        s, e = float(c["start"]), max(float(c["end"]), float(c["start"]) + 0.2)
        dt = t - s
        if dt < 0:
            continue
        inside = t <= e
        # 0->1 while the cue runs, then 1->0 while it relaxes
        level = _ease((t - s) / (e - s)) if inside else max(0.0, 1.0 - (t - e) / RELAX_SECONDS)
        if eff == "zoom_in":
            scale *= 1.0 + ZOOM_AMOUNT * level
        elif eff == "zoom_out":  # starts zoomed in and pulls back to normal
            if inside:
                scale *= 1.0 + ZOOM_AMOUNT * (1.0 - level) * min(1.0, dt / 0.25)
        elif eff in ("pan_left", "pan_right"):
            ramp = min(1.0, dt / 0.25) if inside else max(0.0, 1.0 - (t - e) / RELAX_SECONDS)
            pos = (level * 2 - 1) if inside else 1.0  # -1 .. +1 across the cue, then holds and relaxes
            scale *= 1.0 + PAN_AMOUNT * 1.6 * ramp  # extra size so the edges never show
            tx += (-1.0 if eff == "pan_left" else 1.0) * PAN_AMOUNT * pos * ramp
        elif eff == "fade" and dt < FADE_SECONDS:
            bright *= _ease(dt / FADE_SECONDS)
        elif eff == "flash" and dt < FLASH_SECONDS:
            white = max(white, 0.85 * (1.0 - dt / FLASH_SECONDS))
        elif eff == "shake" and dt < SHAKE_SECONDS:
            shake = max(shake, 1.0 - dt / SHAKE_SECONDS)
    return {"scale": scale, "tx": tx, "brightness": bright, "white": white, "shake": shake}


def render_frame(frame, state: dict, t: float):
    """Apply one frame_state() to a BGR frame (returns a new frame, or the same one when unchanged)."""
    import cv2

    if (state["scale"] == 1.0 and state["tx"] == 0.0 and state["brightness"] == 1.0
            and state["white"] == 0.0 and state["shake"] == 0.0):
        return frame
    h, w = frame.shape[:2]
    sc = state["scale"]
    dx = state["tx"] * w
    dy = 0.0
    if state["shake"]:
        amp = 0.012 * w * state["shake"]
        dx += amp * math.sin(t * 61.0)
        dy += amp * math.cos(t * 47.0)
    out = frame
    if sc != 1.0 or dx or dy:
        m = np.array([[sc, 0, (1 - sc) * w / 2 + dx], [0, sc, (1 - sc) * h / 2 + dy]], dtype=np.float32)
        out = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if state["brightness"] < 1.0:
        out = (out * state["brightness"]).astype(np.uint8)
    if state["white"] > 0.0:
        out = cv2.addWeighted(out, 1 - state["white"], np.full_like(out, 255), state["white"], 0)
    return out


def apply_cues_to_video(input_path: str, cues: Sequence[dict], output_path: str,
                        ffmpeg_path: Optional[str] = None, progress=None) -> dict:
    """Re-render `input_path` with the cues applied, keeping its audio. Returns {"output_path","frames","cues_applied"}.
    With no usable cue the input is NOT touched: output_path == input_path."""
    import cv2

    usable = [c for c in cues or [] if c.get("effect") in EFFECTS]
    if not usable:
        return {"output_path": str(input_path), "frames": 0, "cues_applied": 0}
    src = Path(str(input_path))
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")
    cap = cv2.VideoCapture(str(src))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open the video: {input_path}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration = total / fps
        ffmpeg = find_ffmpeg(ffmpeg_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        enc = open_encoder(ffmpeg, str(src), str(output_path), (w, h), int(round(fps)), duration)
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = n / fps
                enc.stdin.write(np.ascontiguousarray(render_frame(frame, frame_state(usable, t), t)).tobytes())
                n += 1
                if progress and n % 25 == 0:
                    progress(min(1.0, n / total))
            enc.stdin.close()
            err = enc.stderr.read().decode(errors="replace")
            code = enc.wait()
            enc.stderr.close()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed while applying the visual cues: {err[-400:]}")
        except BrokenPipeError:
            raise RuntimeError("ffmpeg stopped early while applying the visual cues") from None
        finally:
            if enc.poll() is None:
                enc.kill()
    finally:
        cap.release()
        gc.collect()
    if progress:
        progress(1.0)
    return {"output_path": str(output_path), "frames": n, "cues_applied": len(usable)}
