"""
backend/transcriber.py

Local speech-to-text with word-level timestamps using faster-whisper.
No API keys, no network calls at inference time, no cost. The "base" model
weights are downloaded once from the Hugging Face hub on first use and cached.

Install:
  pip install faster-whisper

Usage:
  from transcriber import transcribe_audio
  words = transcribe_audio("clip.wav")
  # [{"word": "Hello", "start": 0.0, "end": 0.42, "probability": 0.98}, ...]
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

_model = None
_model_lock = threading.Lock()


def _bundled_model_path():
    """models/faster-whisper-base next to the exe / in the project, if present.
    build.py bundles it so the packaged app transcribes with no download."""
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    for root in roots:
        candidate = root / "models" / "faster-whisper-base"
        if (candidate / "model.bin").is_file():
            return str(candidate)
    return None


def unload_model() -> None:
    """Free the Whisper model's RAM. It is loaded again, lazily, by the next transcribe_audio()."""
    global _model
    with _model_lock:
        _model = None
    import gc

    gc.collect()


def is_loaded() -> bool:
    return _model is not None


def _get_model():
    """Load the model once and reuse it (loading is the slow part)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                model_id = _bundled_model_path() or MODEL_SIZE
                logger.info(
                    "Loading faster-whisper model=%s device=%s compute_type=%s",
                    model_id, DEVICE, COMPUTE_TYPE,
                )
                try:
                    _model = WhisperModel(
                        model_id, device=DEVICE, compute_type=COMPUTE_TYPE
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Could not load the speech-recognition model. The packaged app "
                        "includes it in its models/faster-whisper-base folder; if that folder "
                        "is missing, connect to the internet once so it can be downloaded. "
                        f"Details: {exc}"
                    ) from exc
    return _model


LONG_MEDIA_SECONDS = 20 * 60  # longer than this is transcribed window by window
WINDOW_SECONDS = 10 * 60


def _flags() -> int:
    import os
    import subprocess

    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _media_seconds(path: Path):
    """(ffmpeg path, duration in seconds) or (None, 0.0) when ffmpeg or the duration is not available."""
    try:
        try:
            from . import video_engine as ve
        except ImportError:  # pragma: no cover
            import video_engine as ve
        ff = ve.find_ffmpeg()
        return ff, float(ve._probe_media(ff, str(path))["duration"] or 0.0)
    except Exception:  # noqa: BLE001
        return None, 0.0


def _extract_window(ff: str, src: Path, start: float, length: float, dst: Path) -> None:
    """16 kHz mono WAV of one window (about 19 MB per 10 minutes) - what Whisper wants anyway."""
    import subprocess

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=_flags())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-300:] or "ffmpeg could not extract the audio")


def _collect_words(segments, offset: float = 0.0) -> list:
    words: list = []
    for segment in segments:  # lazy generator: transcription happens as it is consumed
        for w in segment.words or []:
            words.append({"word": w.word.strip(), "start": round(float(w.start) + offset, 3),
                          "end": round(float(w.end) + offset, 3), "probability": round(float(w.probability), 4)})
    return words


def transcribe_audio(audio_path: str) -> list:
    """Transcribe `audio_path` and return a flat list of word dicts:

        {"word": str, "start": float, "end": float, "probability": float}

    Timestamps are in seconds. Video files also work (ffmpeg is bundled with
    faster-whisper via PyAV). Raises FileNotFoundError if the file is missing.
    """
    path = Path(str(audio_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()
    ff, seconds = _media_seconds(path)
    if ff and seconds > LONG_MEDIA_SECONDS:
        return _transcribe_in_windows(model, ff, path, seconds)
    segments, info = model.transcribe(
        str(path),
        word_timestamps=True,
        vad_filter=True,  # skip silence; reduces hallucinated words
        beam_size=5,
    )
    logger.info(
        "Detected language=%s (p=%.2f), duration=%.1fs",
        info.language, info.language_probability, info.duration,
    )

    return _collect_words(segments)


def _transcribe_in_windows(model, ff: str, path: Path, seconds: float) -> list:
    """Long videos: decode and transcribe WINDOW_SECONDS at a time, so memory use does not grow with the video length
    (decoding a 2-hour file in one go needs ~0.5 GB of float audio). Window files live in a temp folder that is
    always removed."""
    import shutil
    import tempfile

    words: list = []
    work = tempfile.mkdtemp(prefix="cp_asr_")
    try:
        start = 0.0
        while start < seconds:
            length = min(WINDOW_SECONDS, seconds - start)
            if length < 0.5:
                break
            wav = Path(work) / "window.wav"
            _extract_window(ff, path, start, length, wav)
            segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True, beam_size=5)
            words.extend(_collect_words(segments, start))
            wav.unlink(missing_ok=True)
            start += length
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return words








if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Word-level transcription")
    parser.add_argument("audio_path")
    print(json.dumps(transcribe_audio(parser.parse_args().audio_path), indent=2))
