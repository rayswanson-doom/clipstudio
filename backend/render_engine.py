"""
render_engine.py - a small programmatic video composition manager for CinePulse AI.

Ideas borrowed (re-implemented in Python, CPU only, no Node / browser / GPU):
  * Remotion  - a video is a pure function of the FRAME NUMBER: `interpolate()`, `spring()`, `Series`-style
                sequencing, every layer decides for itself what to draw at frame i.
  * Revideo   - a multi-track timeline where clips are placed relative to each other (`series`, `align`),
                B-roll / audio / text / avatar tracks that are rendered together and muxed with ffmpeg.
  * GSAP      - named easings ("power3.out", "back.out", "elastic.out" ...) and a keyframe `Timeline`
                (`Timeline().to("x", start, duration, value, ease)`), used for the overlay entrances/exits.
  * Motion Canvas / Manim - the procedural animated background (a generated "B-roll" with no footage needed).
  * LTX-Video - generative B-roll is a GPU diffusion model; `ltx_video_status()` reports honestly whether it
                could run here. It cannot on a CPU laptop, so the procedural background is the built-in provider.

Tracks and z-order (bottom -> top):  B-roll  <  text overlays (OCR captions, titles)  <  3-D avatar.
Voiceover and music are audio-only items mixed by ffmpeg (music is ducked under the voice).

    comp = Composition(size=(1080, 1920), fps=30)
    comp.add_audio("voice.mp3", role="voiceover")
    comp.add_broll("loop.mp4")                          # or omit the source: animated procedural background
    comp.add_ocr_overlays(segments, source_duration=12)  # text read from the video's frames, shown in time
    comp.add_avatar("avatar.mp4", "Bottom-Right", scale=0.36)
    comp.render("final.mp4")
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # package import (backend.render_engine) or plain module (render_engine)
    from . import avatar_sync as _avs
    from . import video_engine as _ve
except ImportError:  # pragma: no cover - exercised when backend/ is on sys.path
    import avatar_sync as _avs
    import video_engine as _ve

logger = logging.getLogger(__name__)

ProgressCb = Callable[[float], None]
DEFAULT_SIZE = (1080, 1920)
DEFAULT_FPS = 30
ANIMATIONS = ("none", "fade", "slide_up", "slide_down", "slide_left", "slide_right", "pop", "zoom")
BROLL_STYLES = ("aurora", "sunset", "mono", "day", "night", "interior")
Z_ORDER = {"broll": 0, "text": 10, "avatar": 20}


# --------------------------------------------------------------------------- #
# Easing (GSAP names)
# --------------------------------------------------------------------------- #
def _clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else float(t)


def _bounce_out(t: float) -> float:
    n, d = 7.5625, 2.75
    if t < 1 / d:
        return n * t * t
    if t < 2 / d:
        t -= 1.5 / d
        return n * t * t + 0.75
    if t < 2.5 / d:
        t -= 2.25 / d
        return n * t * t + 0.9375
    t -= 2.625 / d
    return n * t * t + 0.984375


def _elastic_in(t: float) -> float:
    if t <= 0.0 or t >= 1.0:
        return _clamp01(t)
    return -(2 ** (10 * (t - 1))) * math.sin((t - 1.075) * 2 * math.pi / 0.3)


_IN: Dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "power1": lambda t: t ** 2,
    "power2": lambda t: t ** 3,
    "power3": lambda t: t ** 4,
    "power4": lambda t: t ** 5,
    "quad": lambda t: t ** 2, "cubic": lambda t: t ** 3, "quart": lambda t: t ** 4, "quint": lambda t: t ** 5,
    "sine": lambda t: 1 - math.cos(t * math.pi / 2),
    "expo": lambda t: 0.0 if t <= 0 else 2 ** (10 * (t - 1)),
    "circ": lambda t: 1 - math.sqrt(max(0.0, 1 - t * t)),
    "back": lambda t: t * t * (2.70158 * t - 1.70158),
    "elastic": _elastic_in,
    "bounce": lambda t: 1 - _bounce_out(1 - t),
}
EASING_FAMILIES = tuple(k for k in _IN if k != "linear")


def ease(name: str, t: float) -> float:
    """GSAP-style easing: "power2.out", "back.inOut", "elastic.out", "linear" ... t in [0,1] (clamped).
    A bare family ("power2") means ".out", like GSAP."""
    t = _clamp01(t)
    key = str(name or "linear").strip()
    if key in ("none", "linear"):
        return t
    fam, _, mode = key.partition(".")
    fam = fam.strip()
    mode = (mode or "out").strip()
    if fam not in _IN:
        raise ValueError(f"Unknown easing '{name}'. Families: linear, {', '.join(EASING_FAMILIES)}; modes: in, out, inOut.")
    f = _IN[fam]
    if mode == "in":
        return float(f(t))
    if mode == "out":
        return float(1 - f(1 - t))
    if mode in ("inOut", "inout"):
        return float(f(2 * t) / 2 if t < 0.5 else 1 - f(2 - 2 * t) / 2)
    raise ValueError(f"Unknown easing mode '{mode}' in '{name}' (use in, out or inOut).")


def interpolate(x: float, input_range: Sequence[float], output_range: Sequence[float], easing: str = "linear",
                extrapolate_left: str = "clamp", extrapolate_right: str = "clamp") -> float:
    """Remotion's interpolate(): map x through a (multi-point, increasing) input range to an output range.
    extrapolate_* is "clamp" (default), "extend" or "identity"."""
    xs, ys = list(input_range), list(output_range)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("input_range and output_range must have the same length (>= 2).")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("input_range must be strictly increasing.")
    if x < xs[0]:
        if extrapolate_left == "clamp":
            return float(ys[0])
        if extrapolate_left == "identity":
            return float(x)
        i = 0
    elif x > xs[-1]:
        if extrapolate_right == "clamp":
            return float(ys[-1])
        if extrapolate_right == "identity":
            return float(x)
        i = len(xs) - 2
    else:
        i = next(k for k in range(len(xs) - 1) if x <= xs[k + 1])
    p = (x - xs[i]) / (xs[i + 1] - xs[i])
    if 0.0 <= p <= 1.0:
        p = ease(easing, p)
    return float(ys[i] + (ys[i + 1] - ys[i]) * p)


def spring(frame: float, fps: float, damping: float = 10.0, stiffness: float = 100.0, mass: float = 1.0,
           overshoot_clamping: bool = False) -> float:
    """Remotion's spring(): physical 0 -> 1 motion evaluated at a frame (closed form, no simulation)."""
    t = max(0.0, frame / fps)
    w0 = math.sqrt(stiffness / mass)
    zeta = damping / (2 * math.sqrt(stiffness * mass))
    if zeta < 1:
        wd = w0 * math.sqrt(1 - zeta * zeta)
        v = 1 - math.exp(-zeta * w0 * t) * (math.cos(wd * t) + (zeta * w0 / wd) * math.sin(wd * t))
    elif zeta == 1:
        v = 1 - math.exp(-w0 * t) * (1 + w0 * t)
    else:
        s = w0 * math.sqrt(zeta * zeta - 1)
        r1, r2 = -zeta * w0 + s, -zeta * w0 - s
        v = 1 - (r2 * math.exp(r1 * t) - r1 * math.exp(r2 * t)) / (r2 - r1)
    return min(v, 1.0) if overshoot_clamping else float(v)


# --------------------------------------------------------------------------- #
# GSAP-like keyframe timeline
# --------------------------------------------------------------------------- #
@dataclass
class _Tween:
    start: float
    duration: float
    to: float
    from_: float
    easing: str


class Timeline:
    """Per-property tweens on one clock:  tl.to("y", 0.0, 0.5, 0, from_=40, ease="power3.out")."""

    def __init__(self):
        self._props: Dict[str, List[_Tween]] = {}

    def to(self, prop: str, start: float, duration: float, to: float, from_: Optional[float] = None,
           ease: str = "power2.out") -> "Timeline":
        if duration < 0:
            raise ValueError("duration must be >= 0")
        ease_check = globals()["ease"]
        ease_check(ease, 0.5)  # validates the name now, not at render time
        tweens = self._props.setdefault(prop, [])
        if from_ is None:
            from_ = tweens[-1].to if tweens else to
        tweens.append(_Tween(float(start), float(duration), float(to), float(from_), ease))
        tweens.sort(key=lambda w: w.start)
        return self

    def value(self, prop: str, t: float, default: float = 0.0) -> float:
        tweens = self._props.get(prop)
        if not tweens:
            return default
        if t < tweens[0].start:
            return tweens[0].from_
        current = tweens[0]
        for w in tweens:
            if w.start <= t:
                current = w
        if current.duration <= 0 or t >= current.start + current.duration:
            return current.to
        return current.from_ + (current.to - current.from_) * ease(current.easing, (t - current.start) / current.duration)

    @property
    def duration(self) -> float:
        return max((w.start + w.duration for ws in self._props.values() for w in ws), default=0.0)


def overlay_state(animation: str, t: float, length: float, in_dur: float = 0.40, out_dur: float = 0.30):
    """Entrance / exit of an overlay at local time t (seconds): (alpha, dx, dy, scale), with dx/dy in units of the
    overlay's own width/height. Built on the GSAP-style easings above."""
    if animation not in ANIMATIONS:
        raise ValueError(f"Unknown animation '{animation}'. Use one of: {', '.join(ANIMATIONS)}.")
    if animation == "none":
        return (1.0 if 0 <= t <= length else 0.0), 0.0, 0.0, 1.0
    in_dur = min(in_dur, length / 2) if length > 0 else 0.0
    out_dur = min(out_dur, length / 2) if length > 0 else 0.0
    p = _clamp01(t / in_dur) if in_dur > 0 else 1.0
    q = ease("power2.in", _clamp01((t - (length - out_dur)) / out_dur)) if out_dur > 0 else 0.0
    alpha = ease("power1.out", p) * (1.0 - q)
    dx = dy = 0.0
    scale = 1.0
    if animation == "slide_up":
        dy = (1 - ease("power3.out", p)) * 0.9 - q * 0.4
    elif animation == "slide_down":
        dy = -(1 - ease("power3.out", p)) * 0.9 + q * 0.4
    elif animation == "slide_left":
        dx = (1 - ease("power3.out", p)) * 0.9 - q * 0.4
    elif animation == "slide_right":
        dx = -(1 - ease("power3.out", p)) * 0.9 + q * 0.4
    elif animation == "pop":
        scale = (0.80 + 0.20 * ease("back.out", p)) * (1.0 - 0.10 * q)
    elif animation == "zoom":
        scale = 1.0 + 0.30 * (1 - ease("power3.out", p)) - 0.08 * q
    return float(alpha), float(dx), float(dy), float(scale)


# --------------------------------------------------------------------------- #
# generative B-roll availability (LTX-Video)
# --------------------------------------------------------------------------- #
def ltx_video_status() -> dict:
    """Can Lightricks LTX-Video generate B-roll on this computer? It is a multi-GB diffusion model that needs an
    NVIDIA GPU, so on a normal laptop the answer is no and the procedural background is used instead."""
    if importlib.util.find_spec("ltx_video") is None:
        return {"available": False, "reason": "LTX-Video is not installed (it also needs an NVIDIA GPU with several GB of memory)."}
    try:
        import torch  # imported only if LTX-Video itself is installed

        if not torch.cuda.is_available():
            return {"available": False, "reason": "LTX-Video is installed but no NVIDIA GPU (CUDA) was found."}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"LTX-Video is installed but PyTorch failed to load ({exc})."}
    return {"available": True, "reason": "LTX-Video and a CUDA GPU were found (generation is not wired into the renderer yet)."}


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    kind: str  # "broll" | "audio" | "text" | "avatar"
    start: float
    end: Optional[float]
    name: str = ""
    data: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        return None if self.end is None else max(0.0, self.end - self.start)


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = (
    os.environ.get("CINEPULSE_FONT", ""),
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans-Bold.ttf", "arialbd.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont

    for cand in _FONT_CANDIDATES:
        if not cand:
            continue
        try:
            return ImageFont.truetype(cand, size)
        except Exception:  # noqa: BLE001 - try the next one
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> List[str]:
    lines: List[str] = []
    for para in str(text).splitlines() or [""]:
        cur = ""
        for word in para.split():
            trial = (cur + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return [l for l in lines if l] or [""]


def render_text_layer(text: str, canvas_w: int, font_px: int = 0, max_width_frac: float = 0.86,
                      color=(255, 255, 255), background=(0, 0, 0, 165), radius_frac: float = 0.35) -> np.ndarray:
    """A caption pill as a BGRA uint8 array (rounded semi-transparent box + wrapped, centred text)."""
    from PIL import Image, ImageDraw

    font_px = font_px or max(18, int(canvas_w * 0.042))
    font = _load_font(font_px)
    pad = int(font_px * 0.55)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    max_w = int(canvas_w * max_width_frac) - 2 * pad
    lines = _wrap(probe, text, font, max_w)
    line_h = int(font_px * 1.25)
    text_w = int(max(probe.textlength(l, font=font) for l in lines))
    w, h = text_w + 2 * pad, line_h * len(lines) + 2 * pad - int(font_px * 0.1)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(min(w, h) * radius_frac) if len(lines) == 1 else int(font_px * 0.6),
                        fill=tuple(background))
    for i, line in enumerate(lines):
        lw = d.textlength(line, font=font)
        d.text(((w - lw) / 2, pad + i * line_h - int(font_px * 0.05)), line, font=font, fill=tuple(color) + (255,))
    rgba = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])  # RGBA -> BGRA


def _blend(canvas: np.ndarray, premult_bgr: np.ndarray, alpha: np.ndarray, x: int, y: int, gain: float = 1.0) -> None:
    """canvas = canvas * (1 - a*gain) + premult * gain, clipped to the canvas. alpha is (h,w,1) float 0..1."""
    h, w = alpha.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1 or gain <= 0:
        return
    sx, sy = x0 - x, y0 - y
    a = alpha[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * gain
    p = premult_bgr[sy:sy + (y1 - y0), sx:sx + (x1 - x0)] * gain
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + p, 0, 255).astype(np.uint8)


def _scale_sprite(pm: np.ndarray, al: np.ndarray, s: float):
    if abs(s - 1.0) < 1e-3:
        return pm, al
    import cv2

    h, w = al.shape[:2]
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    return (cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_LINEAR),
            cv2.resize(al, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None])


# --------------------------------------------------------------------------- #
# B-roll sources
# --------------------------------------------------------------------------- #
_PALETTES = {  # BGR
    "aurora": [(70, 20, 20), (180, 120, 20), (150, 60, 190), (200, 180, 60)],
    "sunset": [(40, 20, 50), (60, 70, 220), (120, 140, 250), (180, 60, 200)],
    "mono": [(26, 24, 24), (70, 66, 64), (110, 104, 100), (50, 48, 46)],
    "day": [(200, 170, 120), (235, 215, 170), (170, 190, 120), (245, 235, 210)],      # sky blue / light
    "night": [(40, 18, 12), (90, 40, 20), (70, 30, 60), (50, 50, 30)],               # deep navy
    "interior": [(30, 40, 60), (60, 100, 170), (40, 70, 120), (90, 130, 190)],       # warm amber
}


class ProceduralBackground:
    """Slowly drifting colour blobs (a Motion-Canvas / Manim-flavoured motion-graphics backdrop). Computed on a
    tiny grid and scaled up, so it costs about a millisecond per frame."""

    def __init__(self, size: Tuple[int, int], style: str = "aurora", seed: int = 7):
        import cv2

        if style not in _PALETTES:
            raise ValueError(f"Unknown background style '{style}'. Use one of: {', '.join(BROLL_STYLES)}.")
        self.size = size
        self.pal = np.array(_PALETTES[style], np.float32)
        lw = 72
        lh = max(8, int(round(lw * size[1] / size[0])))
        ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
        self.xs, self.ys = xs / (lw - 1), ys / (lh - 1)
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * math.pi, (3, 2)).astype(np.float32)
        self.speed = rng.uniform(0.18, 0.42, (3, 2)).astype(np.float32)
        vy, vx = np.mgrid[0:lh, 0:lw].astype(np.float32)
        r = np.sqrt(((vx / lw) - 0.5) ** 2 + ((vy / lh) - 0.5) ** 2)
        self.vignette = (1.0 - 0.55 * np.clip(r * 1.6, 0, 1) ** 2)[..., None]
        self._cv2 = cv2

    def frame(self, t: float) -> np.ndarray:
        field = np.tile(self.pal[0], (*self.xs.shape, 1)).astype(np.float32)
        for k in range(3):
            cx = 0.5 + 0.38 * math.sin(t * self.speed[k, 0] * 2 * math.pi + self.phase[k, 0])
            cy = 0.5 + 0.38 * math.cos(t * self.speed[k, 1] * 2 * math.pi + self.phase[k, 1])
            w = np.exp(-(((self.xs - cx) ** 2) + ((self.ys - cy) ** 2)) / 0.075)[..., None]
            field += w * (self.pal[k + 1] - self.pal[0]) * 0.9
        small = np.clip(field * self.vignette, 0, 255).astype(np.uint8)
        return self._cv2.resize(small, self.size, interpolation=self._cv2.INTER_CUBIC)


class _VideoReader:
    """Sequential reader of a video by time; holds the last frame once the video has ended."""

    def __init__(self, path: str):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.pos = -1
        self.last = None

    def frame_at(self, t: float):
        idx = max(0, int(t * self.fps + 1e-6))
        while self.pos < idx:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.pos += 1
            self.last = fr
        return self.last

    def close(self) -> None:
        self.cap.release()


class _StillReader:
    """A single image used as an always-available video frame."""

    def __init__(self, path: str):
        import cv2

        self.img = cv2.imread(str(path))
        if self.img is None:
            raise ValueError(f"Could not open image: {path}")

    def frame_at(self, t: float):
        return self.img

    def close(self) -> None:
        pass


def _video_seconds(ffmpeg: str, path: str) -> float:
    return float(_ve._probe_media(ffmpeg, path).get("duration") or 0.0)


# --------------------------------------------------------------------------- #
# the composition
# --------------------------------------------------------------------------- #
class Composition:
    def __init__(self, size: Tuple[int, int] = DEFAULT_SIZE, fps: int = DEFAULT_FPS, duration: Optional[float] = None,
                 ffmpeg_path: Optional[str] = None):
        if min(size) < 16 or fps < 1:
            raise ValueError("size must be at least 16x16 and fps at least 1.")
        self.size = (int(size[0]) - int(size[0]) % 2, int(size[1]) - int(size[1]) % 2)
        self.fps = int(fps)
        self._duration = duration
        self.items: List[Item] = []
        self.ffmpeg = _ve.find_ffmpeg(ffmpeg_path)
        self.warnings: List[str] = []

    # ---- timeline ------------------------------------------------------- #
    @property
    def duration(self) -> float:
        if self._duration:
            return float(self._duration)
        voice = [i.end for i in self.items if i.kind == "audio" and i.data.get("role") == "voiceover" and i.end is not None]
        if voice:
            return float(max(voice))
        ends = [i.end for i in self.items if i.end is not None]
        return float(max(ends)) if ends else 0.0

    def _resolved_end(self, item: Item) -> float:
        return self.duration if item.end is None else item.end

    def _add(self, item: Item) -> Item:
        if item.start < 0:
            raise ValueError("start must be >= 0")
        if item.end is not None and item.end <= item.start:
            raise ValueError("end must be after start")
        self.items.append(item)
        return item

    def series(self, items: Sequence[Item], start: float = 0.0, gap: float = 0.0) -> float:
        """Remotion <Series>: place the items one after another (each keeps its own length). Returns the end time."""
        t = float(start)
        for it in items:
            if it.duration is None:
                raise ValueError("series() needs items with a known end")
            length = it.duration
            it.start, it.end = t, t + length
            t = it.end + gap
        return t - gap if items else float(start)

    def align(self, item: Item, to: Item, edge: str = "end", offset: float = 0.0, keep_length: bool = True) -> Item:
        """Revideo-style relative placement: start `item` at the `edge` ("start" or "end") of `to`, plus an offset."""
        if edge not in ("start", "end"):
            raise ValueError("edge must be 'start' or 'end'")
        anchor = to.start if edge == "start" else self._resolved_end(to)
        length = item.duration
        item.start = max(0.0, anchor + offset)
        if length is not None and keep_length:
            item.end = item.start + length
        return item

    # ---- tracks --------------------------------------------------------- #
    def add_audio(self, path: str, start: float = 0.0, end: Optional[float] = None, volume: float = 1.0,
                  role: str = "voiceover", fade_in: float = 0.0, fade_out: float = 0.0, loop: bool = False) -> Item:
        """role "voiceover" (defines the video length when none is set) or "music" (ducked under the voice)."""
        if role not in ("voiceover", "music", "sfx"):
            raise ValueError("role must be 'voiceover', 'music' or 'sfx'")
        if not Path(path).is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        length = _video_seconds(self.ffmpeg, str(path))
        if end is None and not loop and length > 0:
            end = start + length
        return self._add(Item("audio", start, end, Path(path).name,
                              {"path": str(path), "volume": volume, "role": role, "fade_in": fade_in,
                               "fade_out": fade_out, "loop": loop, "source_seconds": length}))

    def add_broll(self, source: Optional[str] = None, start: float = 0.0, end: Optional[float] = None,
                  motion: str = "ken_burns", style: str = "aurora", fade: float = 0.0, loop: bool = True,
                  seed: Optional[int] = None) -> Item:
        """Background clip: a video, a still image, or (source=None) the animated procedural backdrop.
        Overlapping clips cross-fade over their overlap; `fade` adds a fade-in at the clip start."""
        if motion not in ("ken_burns", "none"):
            raise ValueError("motion must be 'ken_burns' or 'none'")
        kind = "procedural"
        if source:
            if not Path(source).is_file():
                raise FileNotFoundError(f"B-roll not found: {source}")
            kind = "image" if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else "video"
        return self._add(Item("broll", start, end, Path(source).name if source else f"procedural:{style}",
                              {"source": source, "source_kind": kind, "motion": motion, "style": style, "fade": fade,
                               "loop": loop, "seed": seed}))

    def add_text(self, text: str, start: float, end: Optional[float] = None, position="bottom",
                 animation: str = "slide_up", font_px: int = 0, group: str = "text", slot: int = 0,
                 color=(255, 255, 255), background=(0, 0, 0, 165)) -> Item:
        """A caption. position: "top" | "center" | "bottom" | "auto" (away from the avatar) | (cx, cy) pixels."""
        if animation not in ANIMATIONS:
            raise ValueError(f"Unknown animation '{animation}'. Use one of: {', '.join(ANIMATIONS)}.")
        if not str(text).strip():
            raise ValueError("Caption text is empty.")
        return self._add(Item("text", start, end, str(text)[:40],
                              {"text": str(text), "position": position, "animation": animation, "font_px": font_px,
                               "group": group, "slot": slot, "color": color, "background": background}))

    def add_ocr_overlays(self, segments: Sequence[dict], source_duration: Optional[float] = None, loop: bool = True,
                         offset: float = 0.0, max_lines: int = 40, position="auto", animation: str = "slide_up") -> List[Item]:
        """Show the text that OCR read from the video's frames at the moment it appears on screen.
        `segments` are ocr_engine segments ({"text","start","end"}). When the background video is shorter than the
        composition and loops, the overlays repeat with it (`source_duration` = the background's length)."""
        segs = [s for s in segments if str(s.get("text", "")).strip()][:max_lines]
        out: List[Item] = []
        total = self.duration
        loops = 1
        if loop and source_duration and source_duration > 0 and total > source_duration:
            loops = int(math.ceil(total / source_duration))
        placed: List[Tuple[float, float, int]] = []  # (start, end, slot) for vertical stacking
        for k in range(loops):
            shift = offset + k * (source_duration or 0.0)
            for s in segs:
                a, b = float(s["start"]) + shift, float(s["end"]) + shift
                if source_duration and loops > 1:
                    b = min(b, shift + source_duration)  # never run into the next loop's copy
                if total:
                    b = min(b, total)
                if b - a < 0.3 or (total and a >= total):
                    continue
                used = {sl for (pa, pb, sl) in placed if pa < b and pb > a}
                slot = next(i for i in range(len(segs) + 1) if i not in used)
                placed.append((a, b, slot))
                out.append(self.add_text(str(s["text"]).strip(), a, b, position, animation, group="ocr", slot=slot))
        return out

    def add_avatar(self, video: str, start: float = 0.0, end: Optional[float] = None, position: str = "Bottom-Right",
                   xy: Optional[Tuple[float, float]] = None, scale: float = _ve.PIP_DEFAULT_SCALE,
                   animation: str = "pop", border_color=(255, 255, 255), shadow: bool = True) -> Item:
        """The talking avatar in the rounded, bordered, shadowed podcast frame (same geometry as place_podcast_avatar)."""
        if animation not in ANIMATIONS:
            raise ValueError(f"Unknown animation '{animation}'. Use one of: {', '.join(ANIMATIONS)}.")
        if not Path(video).is_file():
            raise FileNotFoundError(f"Avatar video not found: {video}")
        preset = _ve.normalize_position_preset(position)
        if preset == "Custom Coordinates" and xy is None:
            raise ValueError("Custom Coordinates needs an (X, Y) position in pixels.")
        is_image = Path(video).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        if is_image:  # a still portrait (e.g. a character anchor) shown for the whole clip
            import cv2

            img = cv2.imread(str(video))
            if img is None:
                raise RuntimeError("The avatar image could not be read.")
            info = {"has_video": True, "width": img.shape[1], "height": img.shape[0], "duration": 0.0}
        else:
            info = _ve._probe_media(self.ffmpeg, video)
        if not info["has_video"] or info["width"] <= 0:
            raise RuntimeError("The avatar video could not be read.")
        rect = _ve.compute_pip_rect(self.size, (info["width"], info["height"]), preset, xy, scale)
        if end is None and info["duration"] > 0:
            end = start + info["duration"]
        return self._add(Item("avatar", start, end, Path(video).name,
                              {"video": str(video), "is_image": is_image, "preset": preset, "rect": rect, "animation": animation,
                               "border_color": border_color, "shadow": shadow, "source_seconds": info["duration"]}))

    # ---- checks --------------------------------------------------------- #
    def validate(self) -> List[str]:
        """Human-readable problems with the timeline (empty list = fine)."""
        problems: List[str] = []
        total = self.duration
        if total <= 0:
            return ["The composition is empty (add a voiceover or set a duration)."]
        voices = sorted((i for i in self.items if i.kind == "audio" and i.data["role"] == "voiceover"), key=lambda i: i.start)
        if not voices:
            problems.append("There is no voiceover track; the video will be silent.")
        for a, b in zip(voices, voices[1:]):
            if a.end is not None and b.start < a.end - 1e-6:
                problems.append(f"Voiceovers '{a.name}' and '{b.name}' overlap.")
        for it in self.items:
            if it.start >= total:
                problems.append(f"{it.kind} '{it.name}' starts at {it.start:.1f}s, after the end of the video ({total:.1f}s).")
            elif it.end is not None and it.end > total + 0.05 and it.kind != "audio":
                problems.append(f"{it.kind} '{it.name}' runs {it.end - total:.1f}s past the end and will be cut.")
        if not any(i.kind == "broll" for i in self.items):
            problems.append("There is no B-roll track; a plain background is used.")
        if sum(1 for i in self.items if i.kind == "avatar") > 1:
            problems.append("More than one avatar track: they are drawn on top of each other.")
        return problems

    def timeline_rows(self) -> List[dict]:
        """Flat description of every track item for display / debugging."""
        return [{"track": i.kind, "name": i.name, "start": round(i.start, 2), "end": round(self._resolved_end(i), 2)}
                for i in sorted(self.items, key=lambda i: (Z_ORDER.get(i.kind, 5), i.start))]

    # ---- audio ---------------------------------------------------------- #
    def build_audio_command(self, out_path: str) -> List[str]:
        total = self.duration
        audios = [i for i in self.items if i.kind == "audio"]
        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if not audios:
            return cmd + ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{total:.3f}", out_path]
        for a in audios:
            cmd += ["-i", a.data["path"]]
        chains, voice_labels, music_labels, other = [], [], [], []
        for k, a in enumerate(audios):
            d = a.data
            end = self._resolved_end(a)
            parts = [f"[{k}:a]aresample=44100,aformat=channel_layouts=stereo"]
            if d["loop"]:
                parts.append("aloop=loop=-1:size=2000000000")
            parts.append(f"atrim=0:{max(0.01, end - a.start):.3f},asetpts=PTS-STARTPTS")
            if d["volume"] != 1.0:
                parts.append(f"volume={d['volume']:.3f}")
            if d["fade_in"] > 0:
                parts.append(f"afade=t=in:st=0:d={d['fade_in']:.3f}")
            if d["fade_out"] > 0:
                parts.append(f"afade=t=out:st={max(0.0, end - a.start - d['fade_out']):.3f}:d={d['fade_out']:.3f}")
            delay = int(round(a.start * 1000))
            if delay > 0:
                parts.append(f"adelay={delay}|{delay}")
            label = f"a{k}"
            chains.append(",".join(parts) + f"[{label}]")
            (voice_labels if d["role"] == "voiceover" else music_labels if d["role"] == "music" else other).append(label)
        mix_inputs: List[str] = []
        if voice_labels and music_labels:  # duck the music under the voice
            vmix = "[vmix]"
            if len(voice_labels) > 1:
                chains.append("".join(f"[{l}]" for l in voice_labels) + f"amix=inputs={len(voice_labels)}:normalize=0:duration=longest{vmix}")
            else:
                chains.append(f"[{voice_labels[0]}]anull{vmix}")
            chains.append("[vmix]asplit=2[vo][vside]")
            mm = "[mmix]"
            if len(music_labels) > 1:
                chains.append("".join(f"[{l}]" for l in music_labels) + f"amix=inputs={len(music_labels)}:normalize=0:duration=longest{mm}")
            else:
                chains.append(f"[{music_labels[0]}]anull{mm}")
            chains.append("[mmix][vside]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=450[duck]")
            mix_inputs = ["[vo]", "[duck]"] + [f"[{l}]" for l in other]
        else:
            mix_inputs = [f"[{l}]" for l in voice_labels + music_labels + other]
        chains.append("".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:normalize=0:duration=longest,"
                      f"apad,atrim=0:{total:.3f}[mix]")
        return cmd + ["-filter_complex", ";".join(chains), "-map", "[mix]", "-c:a", "pcm_s16le", out_path]

    # ---- rendering ------------------------------------------------------ #
    def render(self, output_path: str, progress: Optional[ProgressCb] = None) -> dict:
        problems = self.validate()
        total = self.duration
        if total <= 0:
            raise ValueError(problems[0])
        warnings = [p for p in problems if "no voiceover" not in p and "no B-roll" not in p] + self.warnings
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        n_frames = max(1, int(round(total * self.fps)))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryDirectory(prefix="render_") as tmp:
            audio = os.path.join(tmp, "mix.wav")
            proc = subprocess.run(self.build_audio_command(audio), capture_output=True, text=True, creationflags=flags)
            if proc.returncode != 0 or not os.path.isfile(audio):
                raise RuntimeError(f"ffmpeg could not mix the audio: {proc.stderr[-600:]}")
            r = _FrameRenderer(self, n_frames)
            enc = _avs.open_encoder(self.ffmpeg, audio, str(out), self.size, self.fps, total)
            try:
                for i in range(n_frames):
                    enc.stdin.write(np.ascontiguousarray(r.frame(i)).tobytes())
                    if progress and i % 5 == 0:
                        progress(i / n_frames)
                enc.stdin.close()
                err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
                code = enc.wait(timeout=300)
            except BrokenPipeError:
                err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
                code = enc.wait(timeout=30) or 1
            except Exception:
                enc.kill()
                raise
            finally:
                r.close()
                if enc.stderr:
                    enc.stderr.close()
        if code != 0 or not out.is_file():
            if out.is_file():
                out.unlink()
            raise RuntimeError(f"ffmpeg could not encode the video ({code}): {err[-600:]}")
        if progress:
            progress(1.0)
        av = next((i for i in self.items if i.kind == "avatar"), None)
        return {"output_path": str(out.resolve()), "duration": round(total, 2), "frames": n_frames, "fps": self.fps,
                "size": self.size, "tracks": self.timeline_rows(), "warnings": warnings,
                "avatar": None if av is None else {"preset": av.data["preset"], "rect": av.data["rect"]},
                "avatars": [{"name": i.name, "preset": i.data["preset"], "rect": i.data["rect"]}
                            for i in self.items if i.kind == "avatar"]}


class _FrameRenderer:
    """Pure function of the frame number: draws every active item bottom to top."""

    def __init__(self, comp: Composition, n_frames: int):
        self.c = comp
        self.W, self.H = comp.size
        self.n = n_frames
        self.broll = sorted((i for i in comp.items if i.kind == "broll"), key=lambda i: i.start)
        self.texts = sorted((i for i in comp.items if i.kind == "text"), key=lambda i: i.start)
        self.avatars = [i for i in comp.items if i.kind == "avatar"]
        self._src: Dict[int, object] = {}
        self._text_layers: Dict[int, np.ndarray] = {}
        self._sprite: Dict[int, tuple] = {}
        self._avatar_rects = [a.data["rect"] for a in self.avatars]
        self.plain = np.full((self.H, self.W, 3), (28, 24, 22), np.uint8)

    # -- b-roll ---------------------------------------------------------- #
    def _source(self, item: Item):
        key = id(item)
        if key not in self._src:
            d = item.data
            if d["source_kind"] == "procedural":
                self._src[key] = ProceduralBackground(self.c.size, d["style"], seed=d.get("seed") if d.get("seed") is not None else 7 + len(self._src))
            elif d["source_kind"] == "image":
                import cv2

                img = cv2.imread(d["source"])
                if img is None:
                    raise ValueError(f"Could not read image: {d['source']}")
                self._src[key] = _avs.cover_crop(img, self.W, self.H)[0]
            else:
                self._src[key] = _avs.LoopingBackground(d["source"], self.W, self.H, self.c.fps)
        return self._src[key]

    def _broll_frame(self, item: Item, t: float):
        d = item.data
        src = self._source(item)
        if d["source_kind"] == "procedural":
            fr = src.frame(t - item.start)
        elif d["source_kind"] == "image":
            fr = src
        else:
            fr = src.next()
            if fr is None:
                return None
        if d["motion"] == "ken_burns":
            end = self.c._resolved_end(item)
            p = ease("sine.inOut", _clamp01((t - item.start) / max(0.1, end - item.start)))
            z = 1.0 + 0.10 * p
            if z > 1.001:
                import cv2

                cw, ch = int(self.W / z), int(self.H / z)
                x0, y0 = (self.W - cw) // 2, (self.H - ch) // 2
                fr = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return fr

    def _background(self, t: float) -> np.ndarray:
        active = [b for b in self.broll if b.start - 1e-6 <= t < self.c._resolved_end(b)]
        if not active:
            return self.plain.copy()
        frames = [(b, self._broll_frame(b, t)) for b in active]
        frames = [(b, f) for b, f in frames if f is not None]
        if not frames:
            return self.plain.copy()
        base = frames[0][1]
        for b, f in frames[1:]:  # cross-fade over the overlap with the clip below
            prev_end = self.c._resolved_end(frames[0][0])
            overlap = max(0.05, prev_end - b.start)
            a = ease("power2.inOut", (t - b.start) / overlap)
            base = (base.astype(np.float32) * (1 - a) + f.astype(np.float32) * a).astype(np.uint8)
        first = frames[0][0]
        fade = first.data["fade"]
        if fade > 0 and t - first.start < fade:
            a = ease("power1.out", (t - first.start) / fade)
            base = (base.astype(np.float32) * a + self.plain.astype(np.float32) * (1 - a)).astype(np.uint8)
        return base.copy()

    # -- text ------------------------------------------------------------ #
    def _text_xy(self, item: Item, layer_h: int, layer_w: int):
        pos = item.data["position"]
        slot = item.data["slot"]
        gap = int(layer_h * 0.25)
        if pos == "auto":
            cy = np.mean([r[1] + r[3] / 2 for r in self._avatar_rects]) if self._avatar_rects else 0
            pos = "top" if (not self._avatar_rects or cy > self.H / 2) else "bottom"
        if isinstance(pos, (tuple, list)):
            return int(pos[0] - layer_w / 2), int(pos[1] - layer_h / 2 + slot * (layer_h + gap))
        x = (self.W - layer_w) // 2
        if pos == "top":
            return x, int(self.H * 0.07) + slot * (layer_h + gap)
        if pos == "center":
            return x, int(self.H * 0.45) + slot * (layer_h + gap)
        return x, int(self.H * 0.80) - layer_h - slot * (layer_h + gap)

    def _draw_text(self, canvas: np.ndarray, item: Item, t: float) -> None:
        key = id(item)
        if key not in self._text_layers:
            d = item.data
            self._text_layers[key] = render_text_layer(d["text"], self.W, d["font_px"], color=d["color"],
                                                      background=d["background"])
        layer = self._text_layers[key]
        length = self.c._resolved_end(item) - item.start
        alpha, dx, dy, scale = overlay_state(item.data["animation"], t - item.start, length)
        if alpha <= 0.003:
            return
        a = layer[..., 3:4].astype(np.float32) / 255.0
        pm = layer[..., :3].astype(np.float32) * a
        pm, a = _scale_sprite(pm, a, scale)
        h, w = a.shape[:2]
        x, y = self._text_xy(item, layer.shape[0], layer.shape[1])
        x += int((layer.shape[1] - w) / 2 + dx * layer.shape[1])
        y += int((layer.shape[0] - h) / 2 + dy * layer.shape[0])
        _blend(canvas, pm, a, x, y, alpha)

    # -- avatar ---------------------------------------------------------- #
    def _avatar_sprite(self, item: Item):
        key = id(item)
        if key not in self._sprite:
            d = item.data
            x, y, w, h = d["rect"]
            deco, mask = _ve.build_pip_frame_layers(self.c.size, d["rect"], d["border_color"], shadow=d["shadow"])
            ys, xs = np.nonzero(deco[..., 3] > 0)
            x0, y0, x1, y1 = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if xs.size else (x, y, x + w, y + h)
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
            region = deco[y0:y1, x0:x1].astype(np.float32)
            a1 = region[..., 3:4] / 255.0
            m = np.zeros((y1 - y0, x1 - x0, 1), np.float32)
            m[y - y0:y - y0 + h, x - x0:x - x0 + w, 0] = mask.astype(np.float32) / 255.0
            self._sprite[key] = (region[..., :3] * a1, a1, m, (x0, y0, x1, y1))
            self._src[key] = _StillReader(d["video"]) if d.get("is_image") else _VideoReader(d["video"])
        return self._sprite[key]

    def _draw_avatar(self, canvas: np.ndarray, item: Item, t: float) -> None:
        import cv2

        c_pm, a1, m, (x0, y0, x1, y1) = self._avatar_sprite(item)
        x, y, w, h = item.data["rect"]
        frame = self._src[id(item)].frame_at(t - item.start)
        if frame is None:
            return
        tile = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        full_tile = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        full_tile[y - y0:y - y0 + h, x - x0:x - x0 + w] = tile
        pm = full_tile * m + c_pm * (1.0 - m)
        al = m + a1 * (1.0 - m)
        length = self.c._resolved_end(item) - item.start
        alpha, dx, dy, scale = overlay_state(item.data["animation"], t - item.start, length, out_dur=0.0)
        if alpha <= 0.003:
            return
        pm, al = _scale_sprite(pm, al, scale)
        sh, sw = al.shape[:2]
        px = x0 + int(((x1 - x0) - sw) / 2 + dx * (x1 - x0))
        py = y0 + int(((y1 - y0) - sh) / 2 + dy * (y1 - y0))
        _blend(canvas, pm, al, px, py, alpha)

    # -- frame ----------------------------------------------------------- #
    def frame(self, i: int) -> np.ndarray:
        t = i / self.c.fps
        canvas = self._background(t)
        for it in self.texts:
            if it.start - 1e-6 <= t < self.c._resolved_end(it):
                self._draw_text(canvas, it, t)
        for it in self.avatars:
            if it.start - 1e-6 <= t < self.c._resolved_end(it) or t >= it.start:
                self._draw_avatar(canvas, it, t)
        return canvas

    def close(self) -> None:
        for s in self._src.values():
            if hasattr(s, "close"):
                s.close()
        self._src.clear()


# --------------------------------------------------------------------------- #
# one-call helper used by the pipeline
# --------------------------------------------------------------------------- #
def render_commentary(
    voiceover_path: str,
    output_path: str,
    background_video: Optional[str] = None,
    avatar_video: Optional[str] = None,
    position: str = "Bottom-Right",
    xy: Optional[Tuple[float, float]] = None,
    scale: float = _ve.PIP_DEFAULT_SCALE,
    screen_segments: Optional[Sequence[dict]] = None,
    show_screen_text: bool = True,
    music_path: Optional[str] = None,
    music_volume: float = 0.16,
    broll_style: str = "aurora",
    size: Tuple[int, int] = DEFAULT_SIZE,
    fps: int = DEFAULT_FPS,
    ffmpeg_path: Optional[str] = None,
    progress: Optional[ProgressCb] = None,
) -> dict:
    """voiceover + B-roll (or the animated backdrop) + OCR text overlays + avatar in a corner -> one video.
    Every track is aligned to the voiceover, which sets the length."""
    comp = Composition(size, fps, ffmpeg_path=ffmpeg_path)
    comp.add_audio(voiceover_path, role="voiceover")
    if music_path:
        comp.add_audio(music_path, role="music", volume=music_volume, loop=True, fade_out=1.0)
    bg_len = None
    if background_video:
        bg_len = _video_seconds(comp.ffmpeg, background_video) or None
        comp.add_broll(background_video, motion="none", loop=True)
    else:
        comp.add_broll(None, style=broll_style, motion="ken_burns")
    if avatar_video:
        comp.add_avatar(avatar_video, 0.0, comp.duration, position, xy, scale)
    if screen_segments and show_screen_text:
        comp.add_ocr_overlays(screen_segments, source_duration=bg_len, loop=True, position="auto")
    return comp.render(output_path, progress)
