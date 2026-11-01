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

    words: list = []
    # `segments` is a lazy generator; transcription happens as it is consumed.
    for segment in segments:
        for w in segment.words or []:
            words.append(
                {
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "probability": round(float(w.probability), 4),
                }
            )
    return words


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Word-level transcription")
    parser.add_argument("audio_path")
    print(json.dumps(transcribe_audio(parser.parse_args().audio_path), indent=2))
