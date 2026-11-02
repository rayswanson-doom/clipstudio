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
