"""
backend/avatar_sync.py  -  CPU-only AI avatar lip-sync (no GPU, no API keys, offline)

    create_avatar_video(image_path, audio_path, output_path, words=None, ...) -> dict

Takes a still picture of an avatar (.jpg / .png) and a voiceover, and renders a
vertical MP4 where the avatar "talks" in time with the audio.

How it works (deliberately light, so a low-end laptop can run it):
  1. MediaPipe BlazeFace finds the face and its MOUTH keypoint (the same bundled
     model the clip cropper uses).
  2. The audio is decoded with the bundled ffmpeg and turned into a mouth-openness
     curve: loudness envelope x (optional) word gate. The gate comes from the
     word-level timestamps of faster-whisper, so the mouth closes in the pauses
     BETWEEN words instead of flapping on background noise.
  3. Each frame, a small region around the mouth is warped (jaw drops, a soft
     mouth-cavity is drawn) with one cv2.remap on a few hundred pixels, plus a
     subtle head sway / scale pulse. Pre-computed maps; about 5 ms per frame.
  4. Frames stream straight into ffmpeg (libx264 + the original audio).

Fallback: when no face can be found (cartoon, back view, tiny face) the mouth is
NOT guessed - that would smear a random part of the picture. Instead the whole
avatar does a rhythmic scale/bob pulse driven by the same word-synced curve.
Result["mode"] says which was used: "mouth-warp" or "pulse".

This is 2-D image warping, not neural rendering: it works best with a front-facing
portrait or a flat illustration, and gives a stylised, subtle motion.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:  # package import (backend.avatar_sync) or plain module (avatar_sync)
    from .video_engine import _resolve_model_path, find_ffmpeg
except ImportError:  # pragma: no cover
    from video_engine import _resolve_model_path, find_ffmpeg

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
DEFAULT_FPS = 25
DEFAULT_SIZE = (720, 1280)  # (width, height): 9:16, light enough for slow CPUs
SAMPLE_RATE = 16000
MAX_IMAGE_SIDE = 2048
BACKDROP_BGR = (30, 20, 20)  # behind transparent PNG pixels
CAVITY_BGR = (28, 22, 62)  # dark mouth interior (BGR)

ProgressCb = Callable[[float], None]


# --------------------------------------------------------------------------- #
# audio -> mouth openness (pure numpy, no cv2 needed)
# --------------------------------------------------------------------------- #
def decode_audio(audio_path: str, ffmpeg_path: Optional[str] = None, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 samples in [-1, 1] using ffmpeg."""
    path = Path(str(audio_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ffmpeg = find_ffmpeg(ffmpeg_path)
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, creationflags=flags)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read the audio: {proc.stderr.decode(errors='replace')[-400:]}")
    samples = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError("The audio file contains no sound.")
    return samples


def energy_envelope(samples: np.ndarray, sample_rate: int, fps: int) -> np.ndarray:
    """Per-video-frame loudness in 0..1 (RMS, noise-floor removed, speech-peak normalised)."""
    n_frames = max(1, int(math.ceil(len(samples) / sample_rate * fps)))
    hop = sample_rate / fps
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = int(i * hop)
        b = max(a + 1, int((i + 1) * hop))
        chunk = samples[a:b]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
    peak = float(np.percentile(rms, 95)) if rms.size else 0.0
    if peak < 1e-4:  # silence
        return np.zeros(n_frames, dtype=np.float32)
    env = rms / peak
    env = np.clip((env - 0.08) / 0.92, 0.0, 1.0)  # drop the noise floor
    return env.astype(np.float32)


def word_gate(words: Optional[Sequence[dict]], n_frames: int, fps: int, pad: float = 0.04) -> np.ndarray:
    """1.0 while a word is being spoken, 0.0 in the gaps, with soft edges.

    `words` is the faster-whisper list: [{"word", "start", "end"}, ...]. Without
    usable words the gate is all ones (the mouth then follows loudness alone).
    """
    gate = np.ones(n_frames, dtype=np.float32)
    spans = []
    for w in words or []:
        try:
            s, e = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            spans.append((s - pad, e + pad))
    if not spans:
        return gate
    gate[:] = 0.0
    for s, e in spans:
        a = max(0, int(math.floor(s * fps)))
        b = min(n_frames, int(math.ceil(e * fps)) + 1)
        gate[a:b] = 1.0
    # soften the edges over ~3 frames so the mouth does not snap shut
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    return np.convolve(np.pad(gate, 1, mode="edge"), kernel, mode="valid")


def mouth_openness(envelope: np.ndarray, gate: Optional[np.ndarray] = None,
                   attack: float = 0.65, release: float = 0.35, in_word_floor: float = 0.22) -> np.ndarray:
    """Combine loudness and word timing into a smooth 0..1 mouth-open curve."""
    env = np.asarray(envelope, dtype=np.float32)
    g = np.ones_like(env) if gate is None else np.asarray(gate, dtype=np.float32)
    shaped = np.power(env, 0.7)  # lift quiet syllables a little
    target = g * np.maximum(shaped, in_word_floor * (shaped > 0))  # a word always opens the mouth a bit
    if gate is None:
        target = shaped
    out = np.empty_like(target)
    cur = 0.0
    for i, t in enumerate(target):
        k = attack if t > cur else release  # open fast, close slower: reads as natural speech
        cur += (float(t) - cur) * k
        out[i] = cur
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# face + mouth location
# --------------------------------------------------------------------------- #
@dataclass
class Face:
    """Pixel coordinates in the image the face was found in."""
    x: float
    y: float
    w: float
    h: float
    mouth: Tuple[float, float]

    def scaled(self, s: float, dx: float, dy: float) -> "Face":
        return Face(self.x * s - dx, self.y * s - dy, self.w * s, self.h * s,
                    (self.mouth[0] * s - dx, self.mouth[1] * s - dy))


def _sane_mouth(box: Tuple[float, float, float, float], mouth: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    x, y, w, h = box
    default = (x + w / 2, y + 0.74 * h)
    if mouth is None:
        return default
    mx, my = mouth
    # BlazeFace's mouth must sit in the lower half and inside the box
    if not (x <= mx <= x + w and y + 0.5 * h <= my <= y + 1.05 * h):
        return default
    return (mx, my)


class MediaPipeFaceLocator:
    """Callable(image_bgr) -> Face or None  (largest/most confident face, with mouth)."""

    def __init__(self, model_path: Optional[str] = None, min_confidence: float = 0.3):
        import mediapipe as mp

        self._mp = mp
        self._legacy = None
        self._task = None
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_detection"):
            self._legacy = solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_confidence)
        else:
            path = _resolve_model_path(model_path)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE, min_detection_confidence=min_confidence)
            self._task = vision.FaceDetector.create_from_options(options)

    def __call__(self, image_bgr) -> Optional[Face]:
        import cv2

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            dets = self._legacy.process(rgb).detections or []
            if not dets:
                return None
            best = max(dets, key=lambda d: d.score[0])
            bb = best.location_data.relative_bounding_box
            box = (bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h)
            kps = best.location_data.relative_keypoints
            mouth = (kps[3].x * w, kps[3].y * h) if len(kps) > 3 else None
        else:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            dets = self._task.detect(image).detections
            if not dets:
                return None
            best = max(dets, key=lambda d: d.categories[0].score)
            bb = best.bounding_box
            box = (float(bb.origin_x), float(bb.origin_y), float(bb.width), float(bb.height))
            kps = best.keypoints or []
            mouth = (kps[3].x * w, kps[3].y * h) if len(kps) > 3 else None
        if box[2] < 12 or box[3] < 12:
            return None
        return Face(*box, mouth=_sane_mouth(box, mouth))

    def close(self) -> None:
        for obj in (self._legacy, self._task):
            if obj is not None and hasattr(obj, "close"):
                obj.close()


def find_face(image_bgr, locator: Optional[Callable] = None, warnings: Optional[List[str]] = None) -> Optional[Face]:
    """Run the locator; any failure (no model, no mediapipe) becomes a warning + None."""
    warnings = warnings if warnings is not None else []
    own = None
    try:
        if locator is None:
            own = locator = MediaPipeFaceLocator()
        return locator(image_bgr)
    except Exception as exc:  # model file missing, mediapipe absent, ...
        warnings.append(f"Face detection unavailable ({exc}); using the simple pulse animation instead.")
        return None
    finally:
        if own is not None:
            own.close()


# --------------------------------------------------------------------------- #
# image handling
# --------------------------------------------------------------------------- #
def load_avatar(image_path: str):
    """Read a .jpg/.png into a BGR uint8 array (EXIF-rotated, alpha flattened, <=2048px)."""
    import cv2

    path = Path(str(image_path)).expanduser()
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Avatar must be a .jpg or .png image (got '{path.suffix or 'no extension'}').")
    if not path.is_file():
        raise FileNotFoundError(f"Avatar image not found: {image_path}")
    flag = cv2.IMREAD_UNCHANGED if path.suffix.lower() == ".png" else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)  # fromfile: works with non-ASCII Windows paths
    if img is None:
        raise ValueError("That file could not be read as an image. Is it a valid .jpg or .png?")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # flatten transparency onto a dark backdrop
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        back = np.empty_like(img[:, :, :3], dtype=np.float32)
        back[:] = BACKDROP_BGR
        img = (img[:, :, :3] * alpha + back * (1 - alpha)).astype(np.uint8)
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def cover_crop(img, out_w: int, out_h: int, focus: Optional[Tuple[float, float]] = None):
    """Scale to fill out_w x out_h, cropping around `focus` (x, y). -> (tile, scale, dx, dy)."""
    import cv2

    h, w = img.shape[:2]
    s = max(out_w / w, out_h / h)
    rw, rh = max(out_w, int(round(w * s))), max(out_h, int(round(h * s)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    fx, fy = (focus[0] * s, focus[1] * s) if focus else (rw / 2, rh / 2)
    dx = int(min(max(fx - out_w / 2, 0), rw - out_w))
    dy = int(min(max(fy - out_h / 2, 0), rh - out_h))
    return np.ascontiguousarray(resized[dy:dy + out_h, dx:dx + out_w]), s, dx, dy


# --------------------------------------------------------------------------- #
# per-frame animation
# --------------------------------------------------------------------------- #
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class MouthAnimator:
    """Animates one still picture. Build once, call .frame(openness, t) per video frame."""

    def __init__(self, tile, face: Optional[Face]):
        self.tile = tile
        self.h, self.w = tile.shape[:2]
        self.face = face
        self.roi = None
        if face is not None:
            self._prepare(face)
        self.mode = "mouth-warp" if self.roi is not None else "pulse"

    def _prepare(self, face: Face) -> None:
        fw = max(face.w, 16.0)
        cx, cy = face.mouth
        x0, x1 = int(max(0, cx - 0.8 * fw)), int(min(self.w, cx + 0.8 * fw))
        y0, y1 = int(max(0, cy - 0.5 * fw)), int(min(self.h, cy + 0.95 * fw))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        wx = np.exp(-(((gx - cx) / (0.38 * fw)) ** 2))  # only near the mouth horizontally
        below = _smoothstep((gy - (cy - 0.035 * fw)) / (0.07 * fw))  # 0 above the lip line, 1 just below it
        anchor = _smoothstep((y1 - gy) / (0.45 * fw))  # 0 at the ROI bottom: the edge stays put
        self.weight = (wx * below * anchor).astype(np.float32)
        self.map_x = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)
        self.roi = (x0, y0, x1, y1)
        self.fw, self.cx, self.cy = fw, float(cx), float(cy)
        self.gx_local = (gx - cx).astype(np.float32)
        self.gy_local = (gy - cy).astype(np.float32)

    def _mouth(self, frame, openness: float) -> None:
        import cv2

        x0, y0, x1, y1 = self.roi
        d = openness * 0.11 * self.fw  # jaw drop in pixels
        if d < 0.4:
            return
        map_y = (self.gy - d * self.weight).astype(np.float32)
        warped = cv2.remap(self.tile, self.map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # soft dark mouth cavity inside the stretched gap
        a = max(2.0, 0.24 * self.fw * (0.65 + 0.35 * openness))
        b = max(1.0, d * 0.52)
        ey = d * 0.5 - 0.01 * self.fw
        dist = (self.gx_local / a) ** 2 + ((self.gy_local - ey) / b) ** 2
        mask = np.clip((1.0 - dist) * 3.0, 0.0, 1.0)  # filled ellipse with a short soft edge
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.7, 0.006 * self.fw))[..., None]
        cav = np.empty_like(warped, dtype=np.float32)
        cav[:] = CAVITY_BGR
        warped = (warped * (1 - 0.92 * mask) + cav * (0.92 * mask)).astype(np.uint8)
        frame[y0:y1, x0:x1] = warped

    def frame(self, openness: float, t: float):
        import cv2

        out = self.tile.copy()
        if self.roi is not None:
            self._mouth(out, openness)
        # head sway + a little scale pulse with the voice (bigger when no mouth warp is possible)
        amp = 0.010 if self.roi is not None else 0.032
        scale = 1.0 + amp * openness
        sway = max(self.w, 1) / 720.0
        tx = 1.6 * sway * math.sin(2 * math.pi * 0.31 * t)
        ty = (1.2 if self.roi is not None else 3.0 * openness + 1.2) * sway * math.sin(2 * math.pi * 0.47 * t + 1.1)
        m = np.array([[scale, 0, (1 - scale) * self.w / 2 + tx],
                      [0, scale, (1 - scale) * self.h / 2 + ty]], dtype=np.float32)
        return cv2.warpAffine(out, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class LoopingBackground:
    """Frames from a background video, looped and time-matched to the output fps."""

    def __init__(self, path: str, out_w: int, out_h: int, fps: int):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open background video: {path}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.step = max(0.1, src_fps / fps)  # source frames per output frame
        self.out = (out_w, out_h)
        self.acc = 0.0
        self.last = None

    def next(self):
        import cv2

        self.acc += self.step
        n = int(self.acc)
        self.acc -= n
        if self.last is None:
            n = max(n, 1)
        frame = None
        for _ in range(n):
            ok, fr = self.cap.read()
            if not ok:  # reached the end: loop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = self.cap.read()
                if not ok:
                    break
            frame = fr
        if frame is not None:
            self.last = cover_crop(frame, *self.out)[0]
        return self.last

    def close(self) -> None:
        self.cap.release()


def _rounded_mask(w: int, h: int, radius: int):
    import cv2

    m = np.zeros((h, w), np.float32)
    cv2.rectangle(m, (radius, 0), (w - radius, h), 1.0, -1)
    cv2.rectangle(m, (0, radius), (w, h - radius), 1.0, -1)
    for cx, cy in ((radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(m, (cx, cy), radius, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 1.5)[..., None]


# --------------------------------------------------------------------------- #
# ffmpeg writer
# --------------------------------------------------------------------------- #
def _open_encoder(ffmpeg: str, audio: str, output: str, size, fps: int, duration: float):
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
        "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", output,
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def create_avatar_video(
    image_path: str,
    audio_path: str,
    output_path: str,
    words: Optional[Sequence[dict]] = None,
    auto_words: bool = False,
    background_video: Optional[str] = None,
    size: Tuple[int, int] = DEFAULT_SIZE,
    fps: int = DEFAULT_FPS,
    ffmpeg_path: Optional[str] = None,
    face_locator: Optional[Callable] = None,
    progress: Optional[ProgressCb] = None,
) -> dict:
    """Render a talking-avatar MP4.

    image_path        .jpg / .png picture of the avatar
    audio_path        the voiceover (edge-tts mp3, any ffmpeg-readable audio/video)
    words             faster-whisper word list [{"word","start","end"}]; makes the mouth
                      close between words. Leave None to follow loudness only.
    auto_words        True = transcribe `audio_path` with faster-whisper to get `words`
                      (skipped quietly if it is unavailable)
    background_video  optional: the avatar is shown as a rounded card over this looping
                      video. Without it the avatar fills the whole 9:16 frame.
    Returns {"output_path","duration","frames","fps","size","face_found","mode","layout",
             "used_word_timestamps","warnings"}.
    """
    import cv2

    warnings: List[str] = []
    W, H = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)  # yuv420p needs even sizes
    ffmpeg = find_ffmpeg(ffmpeg_path)

    image = load_avatar(image_path)  # raises with a clear message for bad files
    samples = decode_audio(audio_path, ffmpeg)
    duration = len(samples) / SAMPLE_RATE

    if words is None and auto_words:
        try:
            try:
                from .transcriber import transcribe_audio
            except ImportError:
                from transcriber import transcribe_audio
            words = transcribe_audio(audio_path)
        except Exception as exc:  # no whisper / model: loudness-only still works
            warnings.append(f"Word timing unavailable ({exc}); the mouth follows loudness only.")
            words = None

    env = energy_envelope(samples, SAMPLE_RATE, fps)
    n_frames = len(env)
    gate = word_gate(words, n_frames, fps) if words else None
    openness = mouth_openness(env, gate)

    # --- layout -----------------------------------------------------------
    face = find_face(image, face_locator, warnings)
    bg = None
    if background_video:
        try:
            bg = LoopingBackground(background_video, W, H, fps)
        except ValueError as exc:
            warnings.append(f"{exc}. Using the avatar full-screen instead.")
    if bg is not None:
        tw = int(W * 0.78) // 2 * 2
        th = int(tw * 1.15) // 2 * 2
        place = ((W - tw) // 2, H - th - int(H * 0.06))
        layout = "card"
    else:
        tw, th, place, layout = W, H, (0, 0), "full"
    focus = (face.mouth[0], face.y + face.h * 0.55) if face else None
    tile, s, dx, dy = cover_crop(image, tw, th, focus)
    tile_face = face.scaled(s, dx, dy) if face else None
    if tile_face is not None and not (0 <= tile_face.mouth[0] < tw and 0 <= tile_face.mouth[1] < th):
        tile_face = None  # mouth cropped out of frame
    if face is None:
        warnings.append("No face found in the avatar image, so it uses a simple talking pulse. "
                        "A clear, front-facing picture gives real mouth movement.")
    animator = MouthAnimator(tile, tile_face)
    mask = _rounded_mask(tw, th, max(8, int(tw * 0.05))) if layout == "card" else None

    # --- render -----------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = _open_encoder(ffmpeg, str(audio_path), str(out), (W, H), fps, duration)
    try:
        for i in range(n_frames):
            av = animator.frame(float(openness[i]), i / fps)
            if layout == "card":
                canvas = bg.next().copy()
                x, y = place
                roi = canvas[y:y + th, x:x + tw]
                canvas[y:y + th, x:x + tw] = (roi * (1 - mask) + av * mask).astype(np.uint8)
                frame = canvas
            else:
                frame = av
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if progress and i % 25 == 0:
                progress(i / n_frames)
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        code = enc.wait()
        enc.stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed while writing the avatar video: {err[-500:]}")
    except BrokenPipeError:
        err = enc.stderr.read().decode(errors="replace") if enc.stderr else ""
        raise RuntimeError(f"ffmpeg stopped early: {err[-500:]}") from None
    finally:
        if enc.poll() is None:
            enc.kill()
        if bg is not None:
            bg.close()
    if progress:
        progress(1.0)
    return {
        "output_path": str(out), "duration": round(duration, 2), "frames": n_frames, "fps": fps,
        "size": (W, H), "face_found": tile_face is not None, "mode": animator.mode, "layout": layout,
        "used_word_timestamps": bool(words), "warnings": warnings,
    }
