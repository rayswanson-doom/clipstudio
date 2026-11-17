"""
backend/video_engine.py

Turn a 16:9 video into a 9:16 vertical short that follows the speaker's face,
re-attach the original audio with the bundled ffmpeg, burn word-accurate subtitles
in, mix trending background music under the voice, and push the result to Google Drive.

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







# --------------------------------------------------------------------------- #
# Corner-frame geometry (where a reaction meme sits in a corner of the video)
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


def compute_pip_rect(base_size: Tuple[int, int], inset_size: Tuple[int, int], position_preset: str = "Bottom-Right",
                     custom_coordinates: Optional[Tuple[float, float]] = None, scale: float = PIP_DEFAULT_SCALE,
                     margin: Optional[int] = None) -> Tuple[int, int, int, int]:
    """(x, y, w, h) in base-video pixels of the corner frame (`inset_size` = the inserted clip's own size).

    scale   the corner frame's width as a fraction of the base video's width (0.10 - 0.80)
    Corner presets keep `margin` pixels (default 4 % of the width) from the edges. 'Custom Coordinates'
    puts the frame's top-left corner at (X, Y) pixels; the frame is pushed back inside the picture
    if it would stick out."""
    bw, bh = int(base_size[0]), int(base_size[1])
    aw, ah = int(inset_size[0]), int(inset_size[1])
    if min(bw, bh, aw, ah) <= 0:
        raise ValueError("Video sizes must be positive.")
    preset = normalize_position_preset(position_preset)
    scale = float(min(max(scale, PIP_MIN_SCALE), PIP_MAX_SCALE))
    w = max(2, int(round(bw * scale / 2)) * 2)
    h = max(2, int(round(w * ah / aw / 2)) * 2)
    if h > bh:  # tall inserted clip on a short video: fit by height
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






# --------------------------------------------------------------------------- #
# burned-in subtitles
# --------------------------------------------------------------------------- #
def build_burn_command(ffmpeg: str, src: str, ass_name: str, out: str, fonts_dir: Optional[str] = None,
                       crf: int = 20, preset: str = "veryfast") -> List[str]:
    """ffmpeg command that hard-codes an .ass script into the picture (libass `ass=` filter) and copies the audio.
    `ass_name` is relative to the directory ffmpeg runs in (burn_subtitles uses a temp dir), which avoids the Windows
    drive-letter / backslash escaping problems of filter paths."""
    vf = f"ass={ass_name}" + (f":fontsdir={fonts_dir}" if fonts_dir else "")
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", vf, "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-movflags", "+faststart", str(out)]


def _run_ffmpeg(cmd: List[str], cwd: Optional[str] = None, timeout: float = 1800) -> str:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout, creationflags=flags)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-600:] or f"ffmpeg exited with code {proc.returncode}")
    return proc.stderr


def burn_subtitles(input_path: str, ass_path: str, output_path: str, ffmpeg_path: Optional[str] = None,
                   fonts_dir: Optional[str] = None) -> dict:
    """Burn the .ass subtitles into the (already cropped 9:16) clip. The video is re-encoded once; audio is copied.
    Returns {"output_path"}. Raises RuntimeError with ffmpeg's message if libass / the font is unavailable."""
    ff = find_ffmpeg(ffmpeg_path)
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")
    if not Path(ass_path).is_file():
        raise FileNotFoundError(f"Subtitle file not found: {ass_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="burn_") as tmp:
        shutil.copyfile(ass_path, Path(tmp) / "subs.ass")
        tmp_out = Path(tmp) / "out.mp4"
        try:
            _run_ffmpeg(build_burn_command(ff, str(Path(input_path).resolve()), "subs.ass", str(tmp_out), fonts_dir), cwd=tmp)
        except RuntimeError as exc:
            raise RuntimeError(f"ffmpeg could not burn the subtitles in: {exc}") from None
        shutil.move(str(tmp_out), str(out))
    return {"output_path": str(out.resolve())}


# --------------------------------------------------------------------------- #
# background music under the speaker (ducking)
# --------------------------------------------------------------------------- #
MUSIC_VOLUME_RANGE = (0.0, 1.0)
DEFAULT_MUSIC_VOLUME = 0.15


def build_music_filter(has_voice: bool, music_volume: float = DEFAULT_MUSIC_VOLUME, duck: bool = True, duration: float = 0.0,
                       fade: float = 1.0) -> str:
    """The -filter_complex string that puts music (input 1) under the clip's own audio (input 0).

    music: looped endlessly (aloop), set to `music_volume`, faded in/out at the clip's ends.
    duck=True: sidechaincompress lowers the music while the speaker talks (the voice is the sidechain), so the music
               rises in the pauses and never fights the dialogue; duck=False: a plain constant low volume.
    Both mix with amix and stop with the voice (duration=first). No voice in the clip: the music alone, cut to `duration`."""
    lo, hi = MUSIC_VOLUME_RANGE
    vol = min(max(float(music_volume), lo), hi)
    fades = ""
    if duration and duration > 2 * fade:
        fades = f",afade=t=in:st=0:d={fade:.2f},afade=t=out:st={max(0.0, duration - fade):.2f}:d={fade:.2f}"
    bgm = f"[1:a]aloop=loop=-1:size=2e+09,aresample=44100,aformat=channel_layouts=stereo,volume={vol:.3f}{fades}[bgm]"
    if not has_voice:
        trim = f",atrim=0:{duration:.3f}" if duration else ""
        return bgm.replace("[bgm]", trim + "[aout]")
    voice = "[0:a]aresample=44100,aformat=channel_layouts=stereo"
    if duck:
        return (f"{voice},asplit=2[voice][sc];{bgm};"
                "[bgm][sc]sidechaincompress=threshold=0.03:ratio=10:attack=25:release=450:makeup=1[ducked];"
                "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")
    return (f"{voice}[voice];{bgm};"
            "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")


def mix_background_music(video_path: str, music_path: str, output_path: str, music_volume: float = DEFAULT_MUSIC_VOLUME,
                         duck: bool = True, ffmpeg_path: Optional[str] = None) -> dict:
    """Mix `music_path` under the video's speech: the picture is copied untouched, only the audio is re-encoded (AAC).
    Returns {"output_path","ducked","music_volume"}."""
    ff = find_ffmpeg(ffmpeg_path)
    for label, p in (("Video", video_path), ("Music", music_path)):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{label} file not found: {p}")
    info = _probe_media(ff, video_path)
    if not info["has_video"]:
        raise RuntimeError("The video could not be read.")
    duration = float(info["duration"] or 0.0)
    flt = build_music_filter(info["has_audio"], music_volume, duck, duration)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.stem + ".mixing" + out.suffix)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-i", str(music_path),
           "-filter_complex", flt, "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(tmp_out)]
    try:
        _run_ffmpeg(cmd)
    except RuntimeError as exc:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not mix the background music: {exc}") from None
    tmp_out.replace(out)
    return {"output_path": str(out.resolve()), "ducked": bool(duck and info["has_audio"]), "music_volume": float(music_volume)}


def splice_memes(input_path: str, output_path: str, matches, ffmpeg_path: Optional[str] = None,
                 progress: Optional[Callable[[float], None]] = None, border: int = 3) -> dict:
    """Lay reaction clips over a FINISHED video with ffmpeg alone (no frame-by-frame Python, so it is fast on a CPU).

    `matches` come from meme_matcher.plan_memes(): each has meme_path, start, end, mode ("overlay" | "cutaway"), rect
    (x, y, w, h in this video's pixels; overlays only), trim_start, use_audio, volume. An overlay is a bordered
    secondary frame that fades in and out; a cutaway fills the picture (a transition frame) with a quick fade. The
    clip's own sound is mixed over the video's audio. The original video's picture and sound are otherwise untouched.
    Returns {"output_path","memes_applied","size","duration"}."""
    ff = find_ffmpeg(ffmpeg_path)
    base = _probe_media(ff, input_path)
    if not base["has_video"] or base["width"] <= 0:
        raise RuntimeError("The video could not be read.")
    W, H = base["width"] - base["width"] % 2, base["height"] - base["height"] % 2
    total = float(base["duration"] or 0.0)
    items = []
    for m in matches:
        path = str(m["meme_path"])
        if not Path(path).is_file():
            continue
        start = max(0.0, float(m["start"]))
        end = min(float(m["end"]), total) if total else float(m["end"])
        if end - start < 0.3:
            continue
        items.append({**m, "meme_path": path, "start": start, "end": end})
    if not items:
        shutil.copyfile(input_path, output_path)
        return {"output_path": str(output_path), "memes_applied": 0, "size": (W, H), "duration": total}
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    chains, v_last, audio_labels = [], "0:v", []
    needs_sound = any(it.get("use_audio") for it in items)
    if needs_sound and not base["has_audio"]:
        cmd += ["-f", "lavfi", "-t", f"{total or max(i['end'] for i in items):.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    first_meme = len(cmd)  # inputs added from here on are memes
    n_extra = 1 if (needs_sound and not base["has_audio"]) else 0
    for k, it in enumerate(items):
        length = it["end"] - it["start"]
        cmd += ["-ss", f"{float(it.get('trim_start', 0.0)):.3f}", "-t", f"{length:.3f}", "-i", it["meme_path"]]
        idx = 1 + n_extra + k
        fade_out = min(0.2, length / 3)
        if it.get("mode") == "cutaway":
            x = y = 0
            geom = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
            fade_in = 0.10
        else:
            rx, ry, rw, rh = [int(v) for v in it["rect"]]
            rw, rh = max(16, rw - rw % 2), max(16, rh - rh % 2)
            x, y = max(0, rx - border), max(0, ry - border)
            geom = (f"scale={rw}:{rh},pad={rw + 2 * border}:{rh + 2 * border}:{border}:{border}:color=white")
            fade_in = 0.15
        chains.append(f"[{idx}:v]{geom},setsar=1,format=rgba,fade=t=in:st=0:d={fade_in}:alpha=1,"
                      f"fade=t=out:st={max(0.0, length - fade_out):.3f}:d={fade_out:.3f}:alpha=1,"
                      f"setpts=PTS-STARTPTS+{it['start']:.3f}/TB[mv{k}]")
        out_label = f"v{k}"
        chains.append(f"[{v_last}][mv{k}]overlay={x}:{y}:eof_action=pass:enable='between(t,{it['start']:.3f},{it['end']:.3f})'[{out_label}]")
        v_last = out_label
        if it.get("use_audio"):
            ms = int(round(it["start"] * 1000))
            chains.append(f"[{idx}:a]atrim=0:{length:.3f},asetpts=PTS-STARTPTS,aresample=44100,aformat=channel_layouts=stereo,"
                          f"volume={float(it.get('volume', 0.7)):.3f},adelay={ms}|{ms}[ma{k}]")
            audio_labels.append(f"[ma{k}]")
    if audio_labels:
        src = "[0:a]" if base["has_audio"] else "[1:a]"
        chains.append(f"{src}aresample=44100,aformat=channel_layouts=stereo[ba]")
        chains.append("[ba]" + "".join(audio_labels) + f"amix=inputs={1 + len(audio_labels)}:normalize=0:duration=first[aout]")
    cmd += ["-filter_complex", ";".join(chains), "-map", f"[{v_last}]"]
    cmd += ["-map", "[aout]"] if audio_labels else (["-map", "0:a?"] if base["has_audio"] else [])
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "160k"] if (audio_labels or base["has_audio"]) else ["-an"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(output_path)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags)
    try:
        for line in proc.stdout:  # "out_time_us=123456" lines
            if progress and total and line.startswith("out_time_us="):
                try:
                    progress(min(0.99, int(line.split("=", 1)[1]) / 1e6 / total))
                except ValueError:
                    pass
        err = proc.stderr.read()
        code = proc.wait(timeout=600)
    except Exception:
        proc.kill()
        raise
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream:
                stream.close()
    if code != 0 or not Path(output_path).is_file():
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not add the reaction clips: {err[-500:]}")
    if progress:
        progress(1.0)
    return {"output_path": str(Path(output_path).resolve()), "memes_applied": len(items), "size": (W, H), "duration": total}






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
