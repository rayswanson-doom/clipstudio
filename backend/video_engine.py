"""
backend/video_engine.py

Turn a 16:9 video into a 9:16 vertical short that follows the speaker's face,
re-attach the original audio with the bundled ffmpeg, and push the result to
Google Drive.

Pipeline (create_vertical_short):
  1. OpenCV reads the requested time range frame by frame.
  2. MediaPipe BlazeFace finds the face; its horizontal centre is smoothed with
     exponential smoothing (factor 0.1) so the virtual camera glides.
  3. Each frame is cropped to 9:16 around the smoothed centre and resized.
  4. ffmpeg (bin/ffmpeg.exe, bundled; falls back to ffmpeg on PATH) encodes
     H.264 and muxes the matching audio slice from the source file.
  5. uploader.upload_to_google_drive() stores the finished clip in Drive.

BlazeFace needs ONE of:
  * a mediapipe release that still ships `mp.solutions.face_detection`
    (model is built in), or
  * the model file blaze_face_short_range.tflite placed in `models/` next to
    this project (or set BLAZEFACE_MODEL_PATH). Download once from:
    https://storage.googleapis.com/mediapipe-models/face_detector/
    blaze_face_short_range/float16/1/blaze_face_short_range.tflite

Install:  pip install opencv-python mediapipe numpy
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:  # works as a package (backend.video_engine) and as a plain module
    from .uploader import upload_to_google_drive
except ImportError:  # pragma: no cover - exercised when run as a script
    from uploader import upload_to_google_drive

logger = logging.getLogger(__name__)

SMOOTHING_FACTOR = 0.1
# Measured on real faces: 0.5 misses small faces (wide shots) almost entirely, 0.3
# finds them most of the time; smoothing + hold-on-loss absorb the odd false hit.
DEFAULT_MIN_CONFIDENCE = 0.3
TARGET_ASPECT = 9 / 16
OUTPUT_SIZE = (1080, 1920)  # (width, height)
MODEL_FILENAME = "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/" + MODEL_FILENAME
)

FaceDetector = Callable[["object"], Optional[float]]  # frame_bgr -> centre_x in 0..1


# --------------------------------------------------------------------------- #
# Pure helpers (no OpenCV needed)
# --------------------------------------------------------------------------- #
def _even(n: float) -> int:
    n = int(n)
    return n - (n % 2)


def smooth(previous: float, target: float, factor: float = SMOOTHING_FACTOR) -> float:
    """One step of exponential smoothing: previous + factor * (target - previous)."""
    return previous + factor * (target - previous)


class FaceTracker:
    """Keeps the smoothed horizontal face centre (normalised 0..1).

    The first detected face is adopted directly so the clip does not open with
    a slow pan from the middle of the frame; every later detection is smoothed
    with `factor`. When no face is found the centre is held, not reset.
    """

    def __init__(self, factor: float = SMOOTHING_FACTOR):
        if not 0.0 < factor <= 1.0:
            raise ValueError("smoothing factor must be in (0, 1]")
        self.factor = factor
        self.center = 0.5
        self._seen_face = False

    def update(self, detected_center: Optional[float]) -> float:
        if detected_center is not None:
            target = min(1.0, max(0.0, float(detected_center)))
            if not self._seen_face:
                self.center = target
                self._seen_face = True
            else:
                self.center = smooth(self.center, target, self.factor)
        return self.center


def compute_crop_size(frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Largest even-sized 9:16 window that fits inside the frame."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("frame size must be positive")
    if frame_w / frame_h > TARGET_ASPECT:  # wider than 9:16 -> crop the sides
        crop_h = _even(frame_h)
        crop_w = min(_even(round(frame_h * TARGET_ASPECT)), _even(frame_w))
    else:  # already tall/narrow -> crop top and bottom
        crop_w = _even(frame_w)
        crop_h = min(_even(round(frame_w / TARGET_ASPECT)), _even(frame_h))
    return crop_w, crop_h


def crop_origin(center_norm: float, frame_w: int, frame_h: int, crop_w: int, crop_h: int) -> Tuple[int, int]:
    """Top-left corner of the crop window, clamped inside the frame."""
    x = int(round(center_norm * frame_w - crop_w / 2))
    x = max(0, min(x, frame_w - crop_w))
    y = max(0, (frame_h - crop_h) // 2)
    return x, y


def _app_roots() -> List[Path]:
    """Places that may contain bin/ and models/ (dev tree and PyInstaller exe)."""
    roots: List[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)  # project root
    roots.append(Path(__file__).resolve().parent)  # backend/
    return roots


def find_ffmpeg(explicit: Optional[str] = None) -> str:
    """Locate ffmpeg: explicit path > bundled bin/ffmpeg.exe > bin/ffmpeg > PATH."""
    if explicit:
        if Path(explicit).is_file():
            return str(Path(explicit))
        raise FileNotFoundError(f"ffmpeg not found at {explicit}")
    searched: List[str] = []
    for root in _app_roots():
        for name in ("ffmpeg.exe", "ffmpeg"):
            p = root / "bin" / name
            searched.append(str(p))
            if p.is_file():
                return str(p)
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    raise FileNotFoundError("ffmpeg not found. Looked in: " + ", ".join(searched) + " and PATH")


def build_ffmpeg_command(
    ffmpeg: str, silent_video: str, source: str, output: str, audio_start: float, duration: float
) -> List[str]:
    """Mux the cropped silent video with the matching audio slice of `source`."""
    return [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", silent_video,
        "-ss", f"{audio_start:.3f}", "-t", f"{duration:.3f}", "-i", source,
        "-map", "0:v:0", "-map", "1:a:0?",  # '?' = no error if source has no audio
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        output,
    ]


# --------------------------------------------------------------------------- #
# BlazeFace via MediaPipe
# --------------------------------------------------------------------------- #
def _resolve_model_path(explicit: Optional[str]) -> str:
    # A path the caller or user named explicitly must exist: silently using a
    # different model would hide a typo.
    for named in (explicit, os.environ.get("BLAZEFACE_MODEL_PATH")):
        if named:
            if Path(named).is_file():
                return str(named)
            raise FileNotFoundError(
                f"BlazeFace model '{MODEL_FILENAME}' not found at the path you gave: {named}"
            )
    candidates: List[Path] = [r / "models" / MODEL_FILENAME for r in _app_roots()]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        f"BlazeFace model '{MODEL_FILENAME}' not found. Download it from {MODEL_URL} "
        "and place it in the project's models/ folder (or set BLAZEFACE_MODEL_PATH)."
    )


class MediaPipeFaceDetector:
    """Callable(frame_bgr) -> normalised horizontal centre of the best face, or None."""

    def __init__(self, model_path: Optional[str] = None, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        import mediapipe as mp

        self._mp = mp
        self._legacy = None
        self._task = None
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_detection"):
            self._legacy = solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_confidence
            )
        else:
            path = _resolve_model_path(model_path)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
            self._task = vision.FaceDetector.create_from_options(options)

    def __call__(self, frame_bgr) -> Optional[float]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            dets = self._legacy.process(rgb).detections or []
            if not dets:
                return None
            best = max(dets, key=lambda d: d.score[0])
            box = best.location_data.relative_bounding_box
            cx = box.xmin + box.width / 2
        else:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            dets = self._task.detect(image).detections
            if not dets:
                return None
            best = max(dets, key=lambda d: d.categories[0].score)
            bb = best.bounding_box
            cx = (bb.origin_x + bb.width / 2) / w
        return min(1.0, max(0.0, cx))

    def close(self) -> None:
        for obj in (self._legacy, self._task):
            if obj is not None and hasattr(obj, "close"):
                obj.close()


def create_face_detector(
    model_path: Optional[str] = None, min_confidence: float = DEFAULT_MIN_CONFIDENCE
) -> MediaPipeFaceDetector:
    return MediaPipeFaceDetector(model_path, min_confidence)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def create_vertical_short(
    input_path: str,
    output_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    folder_id: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    smoothing: float = SMOOTHING_FACTOR,
    output_size: Tuple[int, int] = OUTPUT_SIZE,
    detector: Optional[FaceDetector] = None,
    detect_every: int = 1,
    upload: bool = True,
    allow_center_fallback: bool = False,
) -> dict:
    """Create a face-tracked 9:16 clip from `input_path` and upload it to Drive.

    start_time / end_time are in seconds (end_time=None -> to the end).
    The Drive folder comes from `folder_id` or the GOOGLE_DRIVE_FOLDER_ID
    environment variable; with neither (or upload=False) the upload is skipped.
    `detector` may be any callable(frame_bgr) -> centre_x in 0..1 or None
    (defaults to MediaPipe BlazeFace). With allow_center_fallback=True a missing
    BlazeFace model degrades to a static centre crop instead of raising.

    Returns {"output_path", "drive_link", "frames", "fps", "crop_size", "faces_found"}.
    """
    import cv2

    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if start_time < 0 or (end_time is not None and end_time <= start_time):
        raise ValueError("require 0 <= start_time < end_time")
    if detect_every < 1:
        raise ValueError("detect_every must be >= 1")

    ffmpeg = find_ffmpeg(ffmpeg_path)  # fail early, before any heavy work
    tracker = FaceTracker(smoothing)

    own_detector = False
    if detector is None:
        try:
            detector = create_face_detector()
            own_detector = True
        except (FileNotFoundError, ImportError) as exc:
            if not allow_center_fallback:
                raise RuntimeError(f"Face detector unavailable: {exc}") from exc
            logger.warning("Face detector unavailable (%s); using centre crop.", exc)
            detector = lambda _frame: None  # noqa: E731

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=str(out.parent))
    tmp.close()
    writer = None
    frames = faces = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or fw <= 0 or fh <= 0:
            raise RuntimeError("Could not read video properties (fps/size)")

        start_frame = int(round(start_time * fps))
        end_frame = total if end_time is None else min(total, int(round(end_time * fps)))
        if start_frame >= end_frame:
            raise ValueError("Requested time range is outside the video")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        crop_w, crop_h = compute_crop_size(fw, fh)
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(output_size))
        if not writer.isOpened():
            raise RuntimeError("Could not open the temporary video writer")

        for i in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            if (i - start_frame) % detect_every == 0:
                found = detector(frame)
                if found is not None:
                    faces += 1
                tracker.update(found)
            x, y = crop_origin(tracker.center, fw, fh, crop_w, crop_h)
            crop = frame[y : y + crop_h, x : x + crop_w]
            writer.write(cv2.resize(crop, tuple(output_size), interpolation=cv2.INTER_AREA))
            frames += 1

        if frames == 0:
            raise RuntimeError("No frames could be read from the requested range")
        writer.release()
        writer = None

        cmd = build_ffmpeg_command(
            ffmpeg, tmp.name, str(src), str(out), start_frame / fps, frames / fps
        )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
        if proc.returncode != 0 or not out.is_file():
            if out.is_file():
                out.unlink()
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-800:]}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if own_detector and hasattr(detector, "close"):
            detector.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    drive_link = None
    target_folder = folder_id or os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    if upload and target_folder:
        drive_link = upload_to_google_drive(str(out), target_folder)
    elif upload:
        logger.warning("No folder_id / GOOGLE_DRIVE_FOLDER_ID set; skipping Drive upload.")

    return {
        "output_path": str(out.resolve()),
        "drive_link": drive_link,
        "frames": frames,
        "fps": fps,
        "crop_size": (crop_w, crop_h),
        "faces_found": faces,
    }


def build_merge_command(
    ffmpeg: str,
    audio_path: str,
    output: str,
    background_video: Optional[str] = None,
    output_size: Tuple[int, int] = OUTPUT_SIZE,
    fps: int = 30,
    background_color: str = "0x101828",
) -> List[str]:
    """ffmpeg command that lays `audio_path` over a 9:16 background.

    With `background_video` the clip is looped as needed, scaled and centre-cropped
    to fill the frame, and its own audio is dropped. Without one, a solid colour
    background is generated. The output ends when the voiceover ends.
    """
    w, h = output_size
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if background_video:
        cmd += ["-stream_loop", "-1", "-i", background_video]
    else:
        cmd += ["-f", "lavfi", "-i", f"color=c={background_color}:s={w}x{h}:r={fps}"]
    cmd += ["-i", audio_path]
    cmd += [
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        output,
    ]
    return cmd


def merge_audio_with_video(
    audio_path: str,
    output_path: str,
    background_video: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    output_size: Tuple[int, int] = OUTPUT_SIZE,
    fps: int = 30,
) -> str:
    """Merge a voiceover with a background video (or a plain background) into a
    vertical MP4 and return its path. Used by the text-to-video (AI Generator Lab) flow."""
    if not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if background_video and not Path(background_video).is_file():
        raise FileNotFoundError(f"Background video not found: {background_video}")

    ffmpeg = find_ffmpeg(ffmpeg_path)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_merge_command(ffmpeg, str(audio_path), str(out), background_video, output_size, fps)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    if proc.returncode != 0 or not out.is_file():
        if out.is_file():
            out.unlink()
        raise RuntimeError(f"ffmpeg merge failed ({proc.returncode}): {proc.stderr[-800:]}")
    return str(out.resolve())



# --------------------------------------------------------------------------- #
# Podcast picture-in-picture: an avatar in a corner of the video
# --------------------------------------------------------------------------- #
PIP_PRESETS = ("Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Custom Coordinates")
PIP_DEFAULT_SCALE = 0.36
PIP_MIN_SCALE, PIP_MAX_SCALE = 0.10, 0.80


def normalize_position_preset(preset: str) -> str:
    """'bottom right', 'bottom_right', 'BR', 'custom' ... -> one of PIP_PRESETS."""
    key = "".join(ch for ch in str(preset or "").lower() if ch.isalnum())
    table = {"bottomright": "Bottom-Right", "br": "Bottom-Right", "bottomleft": "Bottom-Left", "bl": "Bottom-Left",
             "topright": "Top-Right", "tr": "Top-Right", "topleft": "Top-Left", "tl": "Top-Left",
             "custom": "Custom Coordinates", "customcoordinates": "Custom Coordinates", "customxy": "Custom Coordinates",
             "customcoordinatesxypixels": "Custom Coordinates"}
    if key not in table:
        raise ValueError(f"Unknown position '{preset}'. Use one of: {', '.join(PIP_PRESETS)}.")
    return table[key]


def compute_pip_rect(base_size: Tuple[int, int], avatar_size: Tuple[int, int], position_preset: str = "Bottom-Right",
                     custom_coordinates: Optional[Tuple[float, float]] = None, scale: float = PIP_DEFAULT_SCALE,
                     margin: Optional[int] = None) -> Tuple[int, int, int, int]:
    """(x, y, w, h) in base-video pixels of the avatar frame.

    scale   the avatar frame's width as a fraction of the base video's width (0.10 - 0.80)
    Corner presets keep `margin` pixels (default 4 % of the width) from the edges. 'Custom Coordinates'
    puts the frame's top-left corner at (X, Y) pixels; the frame is pushed back inside the picture
    if it would stick out."""
    bw, bh = int(base_size[0]), int(base_size[1])
    aw, ah = int(avatar_size[0]), int(avatar_size[1])
    if min(bw, bh, aw, ah) <= 0:
        raise ValueError("Video sizes must be positive.")
    preset = normalize_position_preset(position_preset)
    scale = float(min(max(scale, PIP_MIN_SCALE), PIP_MAX_SCALE))
    w = max(2, int(round(bw * scale / 2)) * 2)
    h = max(2, int(round(w * ah / aw / 2)) * 2)
    if h > bh:  # tall avatar on a short video: fit by height
        h = max(2, int(bh * 0.9) // 2 * 2)
        w = max(2, int(round(h * aw / ah / 2)) * 2)
    m = max(12, int(bw * 0.04)) if margin is None else int(margin)
    if preset == "Custom Coordinates":
        if custom_coordinates is None:
            raise ValueError("Custom Coordinates needs an (X, Y) position in pixels.")
        x, y = int(round(float(custom_coordinates[0]))), int(round(float(custom_coordinates[1])))
    else:
        x = bw - w - m if preset.endswith("Right") else m
        y = bh - h - m if preset.startswith("Bottom") else m
    x = int(min(max(x, 0), max(bw - w, 0)))
    y = int(min(max(y, 0), max(bh - h, 0)))
    return x, y, w, h


def _rounded_rect_mask(w: int, h: int, radius: int, inset: int = 0, canvas: Optional[Tuple[int, int]] = None,
                       origin: Tuple[int, int] = (0, 0), supersample: int = 3):
    """Float mask (0..1) of a rounded rectangle, drawn supersampled for smooth corners."""
    import cv2
    import numpy as np

    cw, ch = canvas or (w, h)
    big = np.zeros((ch * supersample, cw * supersample), np.uint8)
    x0, y0 = (origin[0] + inset) * supersample, (origin[1] + inset) * supersample
    x1, y1 = (origin[0] + w - inset) * supersample - 1, (origin[1] + h - inset) * supersample - 1
    r = max(0, int((radius - inset) * supersample))
    r = min(r, (x1 - x0) // 2, (y1 - y0) // 2)
    cv2.rectangle(big, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(big, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(big, (cx, cy), r, 255, -1)
    return cv2.resize(big, (cw, ch), interpolation=cv2.INTER_AREA).astype("float32") / 255.0


def build_pip_frame_layers(base_size: Tuple[int, int], rect: Tuple[int, int, int, int],
                           border_color: Tuple[int, int, int] = (255, 255, 255), border_px: Optional[int] = None,
                           radius: Optional[int] = None, shadow: bool = True):
    """(decoration RGBA uint8 (bh,bw,4) in BGRA, avatar mask uint8 (h,w)) for the podcast frame.

    The decoration (drop shadow + border ring) is laid UNDER the avatar; the avatar's mask is inset
    by the border width so the ring stays visible and the avatar's own pixels are never tinted."""
    import cv2
    import numpy as np

    bw, bh = base_size
    x, y, w, h = rect
    border = max(2, int(round(w * 0.014))) if border_px is None else int(border_px)
    rad = max(4, int(round(w * 0.075))) if radius is None else int(radius)
    deco = np.zeros((bh, bw, 4), np.float32)
    outer = _rounded_rect_mask(w, h, rad, 0, (bw, bh), (x, y))
    if shadow:
        blur = max(3, int(w * 0.05)) | 1
        offset = max(2, int(w * 0.025))
        sh = np.zeros((bh, bw), np.float32)
        sx0, sy0 = x + 0, y + offset
        sh_rect = _rounded_rect_mask(w, h, rad, 0, (bw, bh), (min(sx0, bw - 1), min(sy0, bh - 1)))
        sh = cv2.GaussianBlur(sh_rect, (0, 0), blur / 2.0) * 0.60
        deco[..., 3] = sh  # black shadow
    ring_alpha = np.clip(outer - _rounded_rect_mask(w, h, rad, border, (bw, bh), (x, y)), 0, 1)
    col = np.array(border_color[::-1], np.float32) / 255.0  # BGR
    a_old = deco[..., 3:4]
    a_new = ring_alpha[..., None]
    out_a = a_new + a_old * (1 - a_new)
    rgb = (col * a_new + deco[..., :3] * a_old * (1 - a_new)) / np.maximum(out_a, 1e-6)
    deco[..., :3] = rgb
    deco[..., 3:4] = out_a
    mask = _rounded_rect_mask(w, h, rad, border)
    return (deco * 255.0 + 0.5).astype(np.uint8), (mask * 255.0 + 0.5).astype(np.uint8)


def _probe_media(ffmpeg: str, path: str) -> dict:
    """{'width','height','duration','has_audio','has_video'} from `ffmpeg -i` (no ffprobe needed)."""
    import re

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True, creationflags=flags)
    text = proc.stderr
    info = {"width": 0, "height": 0, "duration": 0.0, "has_audio": "Audio:" in text, "has_video": "Video:" in text}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    v = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", text)
    if v:
        info["width"], info["height"] = int(v.group(1)), int(v.group(2))
    return info


def build_pip_command(ffmpeg: str, base: str, avatar: str, deco_png: str, mask_png: str, out: str,
                      rect: Tuple[int, int, int, int], duration: float, audio_source: str, has_avatar_audio: bool,
                      has_base_audio: bool) -> List[str]:
    x, y, w, h = rect
    chain = (
        f"[1:v]scale={w}:{h},setsar=1,format=yuva420p[av0];[3:v]format=gray[m];[av0][m]alphamerge[av];"
        f"[0:v][2:v]overlay=0:0:format=auto[b];[b][av]overlay={x}:{y}:format=auto:eof_action=repeat,format=yuv420p[v]"
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", base, "-i", avatar,
           "-loop", "1", "-framerate", "30", "-i", deco_png, "-loop", "1", "-framerate", "30", "-i", mask_png]
    amap: List[str]
    if audio_source == "mix" and has_base_audio and has_avatar_audio:
        chain += ";[0:a]volume=0.55[ba];[1:a]volume=1.0[aa];[ba][aa]amix=inputs=2:duration=first:normalize=0[a]"
        amap = ["-map", "[a]"]
    elif audio_source == "avatar" and has_avatar_audio:
        amap = ["-map", "1:a:0"]
    elif has_base_audio:
        amap = ["-map", "0:a:0"]
    elif has_avatar_audio:
        amap = ["-map", "1:a:0"]
    else:
        amap = []
    cmd += ["-filter_complex", chain, "-map", "[v]", *amap,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    if amap:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-t", f"{duration:.3f}", "-movflags", "+faststart", out]
    return cmd


def place_podcast_avatar(
    base_video: str,
    avatar_video: str,
    position_preset: str = "Bottom-Right",
    custom_coordinates: Optional[Tuple[float, float]] = None,
    scale: float = PIP_DEFAULT_SCALE,
    output_path: Optional[str] = None,
    audio_source: str = "base",
    border_color: Tuple[int, int, int] = (255, 255, 255),
    shadow: bool = True,
    ffmpeg_path: Optional[str] = None,
) -> dict:
    """Picture-in-picture "podcast" layout: the avatar video in a rounded, bordered, shadowed frame
    in a corner (or at exact pixel coordinates) of the base video.

    position_preset    "Bottom-Right" | "Bottom-Left" | "Top-Right" | "Top-Left" | "Custom Coordinates"
    custom_coordinates (X, Y) pixels of the frame's top-left corner, used with "Custom Coordinates"
    scale              frame width as a fraction of the base video's width (0.10 - 0.80)
    audio_source       "base" (default) | "avatar" | "mix" - which soundtrack the result keeps
    The output is as long as the base video (a shorter avatar holds its last frame).
    Returns {"output_path","rect":(x,y,w,h),"preset","scale","base_size","duration"}."""
    for label, path in (("Base video", base_video), ("Avatar video", avatar_video)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if audio_source not in ("base", "avatar", "mix"):
        raise ValueError("audio_source must be 'base', 'avatar' or 'mix'.")
    preset = normalize_position_preset(position_preset)
    ffmpeg = find_ffmpeg(ffmpeg_path)
    b, a = _probe_media(ffmpeg, base_video), _probe_media(ffmpeg, avatar_video)
    if not b["has_video"] or b["width"] <= 0:
        raise RuntimeError("The base video could not be read.")
    if not a["has_video"] or a["width"] <= 0:
        raise RuntimeError("The avatar video could not be read.")
    rect = compute_pip_rect((b["width"], b["height"]), (a["width"], a["height"]), preset, custom_coordinates, scale)
    out = Path(output_path) if output_path else Path(base_video).with_name(Path(base_video).stem + "_podcast.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = b["duration"] or a["duration"]
    import cv2

    with tempfile.TemporaryDirectory(prefix="pip_") as tmp:
        deco, mask = build_pip_frame_layers((b["width"], b["height"]), rect, border_color, shadow=shadow)
        deco_png, mask_png = os.path.join(tmp, "deco.png"), os.path.join(tmp, "mask.png")
        cv2.imwrite(deco_png, deco)
        cv2.imwrite(mask_png, mask)
        cmd = build_pip_command(ffmpeg, str(base_video), str(avatar_video), deco_png, mask_png, str(out), rect,
                                duration, audio_source, a["has_audio"], b["has_audio"])
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    if proc.returncode != 0 or not out.is_file():
        if out.is_file():
            out.unlink()
        raise RuntimeError(f"ffmpeg could not place the avatar ({proc.returncode}): {proc.stderr[-800:]}")
    return {"output_path": str(out.resolve()), "rect": rect, "preset": preset, "scale": float(scale),
            "base_size": (b["width"], b["height"]), "duration": round(duration, 2)}


def preview_layout(base_size: Tuple[int, int] = OUTPUT_SIZE, avatar_size: Tuple[int, int] = (480, 640),
                   position_preset: str = "Bottom-Right", custom_coordinates: Optional[Tuple[float, float]] = None,
                   scale: float = PIP_DEFAULT_SCALE, base_frame=None, avatar_frame=None, thumb_height: int = 480):
    """A small BGR picture of where the avatar will sit - drawn with the SAME geometry, frame, border
    and shadow as the real render. `base_frame` / `avatar_frame` (BGR arrays) make it a true preview;
    without them a placeholder backdrop and a silhouette are drawn."""
    import cv2
    import numpy as np

    bw, bh = int(base_size[0]), int(base_size[1])
    rect = compute_pip_rect((bw, bh), avatar_size, position_preset, custom_coordinates, scale)
    k = thumb_height / bh
    tw, th = max(2, int(round(bw * k))), thumb_height
    small_rect = tuple(int(round(v * k)) for v in rect)
    x, y, w, h = small_rect
    if base_frame is not None:
        canvas = cv2.resize(base_frame, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        t = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
        canvas = (np.array((70, 52, 40), np.float32) * (1 - t) + np.array((40, 30, 22), np.float32) * t)
        canvas = np.broadcast_to(canvas, (th, tw, 3)).astype(np.uint8).copy()
        for gx in range(1, 3):  # thirds grid so the position reads clearly
            cv2.line(canvas, (tw * gx // 3, 0), (tw * gx // 3, th), (95, 80, 66), 1)
            cv2.line(canvas, (0, th * gx // 3), (tw, th * gx // 3), (95, 80, 66), 1)
    deco, mask = build_pip_frame_layers((tw, th), small_rect)
    if avatar_frame is not None:
        tile = cv2.resize(avatar_frame, (w, h), interpolation=cv2.INTER_AREA)
    else:
        tile = np.full((h, w, 3), (88, 64, 52), np.uint8)
        cv2.circle(tile, (w // 2, int(h * 0.36)), int(w * 0.17), (190, 190, 215), -1)
        cv2.ellipse(tile, (w // 2, int(h * 1.02)), (int(w * 0.34), int(h * 0.38)), 0, 180, 360, (190, 190, 215), -1)
    a = deco[..., 3:4].astype(np.float32) / 255.0
    canvas = (canvas.astype(np.float32) * (1 - a) + deco[..., :3].astype(np.float32) * a).astype(np.uint8)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y:y + h, x:x + w]
    canvas[y:y + h, x:x + w] = (roi * (1 - m) + tile * m).astype(np.uint8)
    return canvas, rect


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Create a 9:16 face-tracked short")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--folder-id")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    print(create_vertical_short(a.input, a.output, a.start, a.end, a.folder_id, upload=not a.no_upload))
