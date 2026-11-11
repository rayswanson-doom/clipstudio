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


def _norm(token: str) -> str:
    import re

    return re.sub(r"[^\w]+", "", str(token).lower().replace("'", "").replace("\u2019", ""))


def align_words(text: str, spoken: list, audio_seconds: float = 0.0, min_match: float = 0.5):
    """Map faster-whisper's spoken words onto the SCRIPT words so captions show the script text but light up exactly
    when each word is spoken.

    `spoken` = transcribe_audio() output. Script words that whisper heard get its real start/end; words it missed or
    heard differently are spread over the gap between their neighbours (by length). Returns
    [{"word","t0","t1","exact"}] for every script word, or None when fewer than `min_match` of the words could be matched
    (the caller then falls back to length-based estimated timing)."""
    import difflib

    toks = str(text).split()
    if not toks or not spoken:
        return None
    a = [_norm(t) for t in toks]
    b = [_norm(w.get("word", "")) for w in spoken]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    t0 = [None] * len(toks)
    t1 = [None] * len(toks)
    hit = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            if not a[blk.a + k]:
                continue
            w = spoken[blk.b + k]
            t0[blk.a + k], t1[blk.a + k] = float(w["start"]), float(w["end"])
            hit += 1
    real = sum(1 for x in a if x)
    if real == 0 or hit / real < min_match:
        return None
    end_all = max(float(audio_seconds or 0.0), max(float(w["end"]) for w in spoken))
    out = []
    i = 0
    while i < len(toks):
        if t0[i] is not None:
            out.append({"word": toks[i], "t0": t0[i], "t1": t1[i], "exact": True})
            i += 1
            continue
        j = i
        while j < len(toks) and t0[j] is None:
            j += 1
        lo = out[-1]["t1"] if out else 0.0
        hi = t0[j] if j < len(toks) else end_all
        hi = max(hi, lo + 0.04 * (j - i))
        wts = [len(a[k]) + 1.0 for k in range(i, j)]
        tot = sum(wts)
        t = lo
        for k, wt in zip(range(i, j), wts):
            dt = (hi - lo) * wt / tot
            out.append({"word": toks[k], "t0": t, "t1": t + dt, "exact": False})
            t += dt
        i = j
    prev = 0.0  # monotonic, no zero-length words
    for w in out:
        w["t0"] = max(prev, w["t0"])
        w["t1"] = max(w["t0"] + 0.03, w["t1"])
        prev = w["t0"]
    return out


def word_timings(audio_path: str, text: str, transcribe_fn=None, audio_seconds: float = 0.0):
    """Real word-boundary timing of `text` spoken in `audio_path` (faster-whisper). Returns (words, source) where source
    is "whisper"; (None, reason) when it could not be done - the caller then estimates. Never raises."""
    try:
        spoken = (transcribe_fn or transcribe_audio)(audio_path)
    except Exception as exc:  # noqa: BLE001 - model missing / not installed / unreadable audio
        return None, f"speech recognition unavailable ({str(exc)[:90]})"
    words = align_words(text, spoken, audio_seconds)
    if words is None:
        return None, "the recognised words did not match the script"
    return words, "whisper"


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Word-level transcription")
    parser.add_argument("audio_path")
    print(json.dumps(transcribe_audio(parser.parse_args().audio_path), indent=2))
