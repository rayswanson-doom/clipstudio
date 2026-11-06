"""
backend/ocr_engine.py  -  read the text that is BURNED INTO a video (subtitles, titles, slides)

    segments = extract_screen_text("video.mp4", interval=1.0)
    # [{"text": "SALE ENDS TODAY", "start": 3.0, "end": 8.0, "confidence": 0.97}, ...]

Pipeline:  sample one frame every `interval` seconds  ->  skip frames that look the same as the
previous one (static overlays are only read once)  ->  CPU OCR  ->  drop low-confidence junk  ->
merge repeats into timed segments.  The segments can then be appended to the transcript
(`merge_transcript`) or narrated.

Backends (all CPU, no API keys, nothing leaves the computer):
  "rapidocr"  - the PP-OCR models of PaddleOCR exported to ONNX and run by onnxruntime; 16 MB with
                models included, no extra OpenCV, works in the packaged exe  (default / bundled)
  "paddleocr" - PaddleOCR itself (https://github.com/PaddlePaddle/PaddleOCR), used when you have it
                installed and ask for it: CINEPULSE_OCR_BACKEND=paddleocr
  "auto"      - (default) the first installed one, in the order above.

STRICT LAZY LOADING: importing this module imports nothing heavy. paddleocr / onnxruntime /
OpenCV load only when extract_screen_text() actually runs, and `unload()` frees the engine again.
Bundled model folders (models/ocr/...) are used when present, so the packaged app needs no download.
"""

from __future__ import annotations

import difflib
import gc
import importlib.util
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 1.0  # seconds between sampled frames
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MAX_FRAMES = 900  # hard cap so a 3-hour video cannot run for hours
MAX_SIDE = 960  # frames are shrunk to this before OCR: much faster, text stays readable
SAME_FRAME_DIFF = 3.0  # mean grey difference (0-255) below which a frame counts as unchanged
SAME_TEXT_RATIO = 0.85  # two readings this similar are the same on-screen text
BACKENDS = ("paddleocr", "rapidocr")  # "auto" tries them in this order: PaddleOCR when it is installed (most accurate), else RapidOCR (the PP-OCR models on ONNX, small, bundled in the exe)
_MODULES = {"paddleocr": ("paddleocr",), "rapidocr": ("rapidocr_onnxruntime", "rapidocr")}

Reading = Tuple[str, float]  # (text line, confidence 0..1)
Segment = Dict[str, object]
ProgressCb = Callable[[float], None]

_engine = None
_engine_name: Optional[str] = None
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# backend discovery / loading (lazy)
# --------------------------------------------------------------------------- #
def _installed(name: str) -> Optional[str]:
    """Module name that provides backend `name`, found WITHOUT importing it."""
    for mod in _MODULES[name]:
        try:
            if importlib.util.find_spec(mod) is not None:
                return mod
        except (ImportError, ValueError):
            continue
    return None


def available_backends() -> List[str]:
    return [b for b in BACKENDS if _installed(b)]


def is_available() -> bool:
    return bool(available_backends())


def _model_root() -> Optional[Path]:
    """models/ocr next to the exe / in the project, if it was bundled."""
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    for r in roots:
        if (r / "models" / "ocr").is_dir():
            return r / "models" / "ocr"
    return None


def _first_dir(root: Optional[Path], *names: str) -> Optional[str]:
    if root:
        for n in names:
            if (root / n).is_dir():
                return str(root / n)
    return None


def _make_paddle():
    from importlib.metadata import version

    from paddleocr import PaddleOCR  # imported only now

    root = _model_root()
    det = _first_dir(root, "paddle_det", "det")
    rec = _first_dir(root, "paddle_rec", "rec")
    major = int(version("paddleocr").split(".")[0])
    if major >= 3:  # PaddleOCR 3.x (PaddleX based)
        kw = dict(lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False,
                  enable_mkldnn=False)  # oneDNN crashes on some Windows CPUs ("ConvertPirAttribute2RuntimeAttribute")
        if det and rec:
            kw.update(text_detection_model_dir=det, text_recognition_model_dir=rec)
        eng = PaddleOCR(**kw)

        def run(img) -> List[Reading]:
            out: List[Reading] = []
            for res in eng.predict(img):
                texts = res["rec_texts"] if "rec_texts" in res else []
                scores = res["rec_scores"] if "rec_scores" in res else [1.0] * len(texts)
                out += [(str(t), float(sc)) for t, sc in zip(texts, scores)]
            return out
        return run
    kw = dict(lang="en", use_angle_cls=False, show_log=False, enable_mkldnn=False)  # 2.x
    if det and rec:
        kw.update(det_model_dir=det, rec_model_dir=rec)
    eng = PaddleOCR(**kw)

    def run(img) -> List[Reading]:
        pages = eng.ocr(img, cls=False) or []
        return [(ln[1][0], float(ln[1][1])) for page in pages if page for ln in page]
    return run


def _make_rapid():
    mod = _installed("rapidocr")
    if mod == "rapidocr_onnxruntime":
        from rapidocr_onnxruntime import RapidOCR

        eng = RapidOCR()

        def run(img) -> List[Reading]:
            result, _ = eng(img)
            return [(str(r[1]), float(r[2])) for r in (result or [])]
    else:
        from rapidocr import RapidOCR

        eng = RapidOCR()

        def run(img) -> List[Reading]:
            res = eng(img)
            txts = list(getattr(res, "txts", None) or [])
            scores = list(getattr(res, "scores", None) or [1.0] * len(txts))
            return [(str(t), float(s)) for t, s in zip(txts, scores)]
    return run


_FACTORIES = {"paddleocr": _make_paddle, "rapidocr": _make_rapid}


def get_engine(backend: str = "auto") -> Tuple[Callable, str]:
    """Load (once) and return (callable(frame_bgr) -> [(text, conf)], backend_name)."""
    global _engine, _engine_name
    with _lock:
        if _engine is not None and backend in ("auto", _engine_name):
            return _engine, _engine_name  # type: ignore[return-value]
        wanted = os.environ.get("CINEPULSE_OCR_BACKEND", "").strip().lower() if backend == "auto" else backend
        order = [wanted] if wanted in BACKENDS else list(BACKENDS)
        errors: List[str] = []
        for name in order:
            if not _installed(name):
                errors.append(f"{name}: not installed")
                continue
            try:
                logger.info("Loading OCR backend %s (lazy)", name)
                _engine, _engine_name = _FACTORIES[name](), name
                return _engine, name
            except Exception as exc:  # broken install / missing model files
                errors.append(f"{name}: {exc}")
                logger.warning("OCR backend %s failed to load: %s", name, exc)
        raise RuntimeError(
            "On-screen text reading is unavailable (" + "; ".join(errors) + "). "
            "Install it with: pip install rapidocr-onnxruntime   (or paddlepaddle + paddleocr)"
        )


def unload() -> None:
    """Drop the OCR engine so its RAM is returned (the next run reloads it)."""
    global _engine, _engine_name
    with _lock:
        _engine, _engine_name = None, None
    gc.collect()


# --------------------------------------------------------------------------- #
# text clean-up helpers (pure Python)
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%$]", " ", text.casefold())).strip()


def _same_text(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= SAME_TEXT_RATIO


def clean_readings(readings: Sequence[Reading], min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> List[Reading]:
    """Keep confident, meaningful lines (>= 2 letters/digits), tidy the whitespace."""
    out: List[Reading] = []
    for text, conf in readings:
        t = re.sub(r"\s+", " ", str(text)).strip()
        if conf >= min_confidence and len(re.findall(r"\w", t)) >= 2:
            out.append((t, float(conf)))
    return out


def merge_samples(samples: Sequence[Tuple[float, List[Reading]]], interval: float) -> List[Segment]:
    """Turn per-frame readings [(time, [(text, conf), ...]), ...] into timed segments.
    A line that stays on screen across consecutive samples becomes ONE segment."""
    segments: List[Segment] = []
    open_lines: List[Segment] = []  # segments still being extended
    for t, readings in samples:
        still_open: List[Segment] = []
        for text, conf in readings:
            match = next((s for s in open_lines if _same_text(str(s["text"]), text)), None)
            if match is not None and float(match["end"]) >= t - 1e-6:
                match["end"] = round(t + interval, 2)
                match["confidence"] = round(max(float(match["confidence"]), conf), 3)
                still_open.append(match)
                open_lines.remove(match)
            else:
                seg = {"text": text, "start": round(t, 2), "end": round(t + interval, 2),
                       "confidence": round(conf, 3)}
                segments.append(seg)
                still_open.append(seg)
        open_lines = still_open
    return segments


# --------------------------------------------------------------------------- #
# frame sampling + extraction
# --------------------------------------------------------------------------- #
def _shrink(frame):
    import cv2

    h, w = frame.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


def _signature(frame):
    import cv2

    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 27), interpolation=cv2.INTER_AREA)


def iter_frames(video_path: str, interval: float = DEFAULT_INTERVAL, start: float = 0.0,
                end: Optional[float] = None, max_frames: int = DEFAULT_MAX_FRAMES):
    """Yield (time_seconds, frame_bgr) about every `interval` seconds. The capture is ALWAYS released."""
    import cv2

    path = Path(str(video_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open the video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * max(0.1, interval))))
        first = int(max(0.0, start) * fps)
        last = int(end * fps) if end is not None else None
        idx, yielded = 0, 0
        while yielded < max_frames:
            if last is not None and idx > last:
                break
            if idx < first:
                if not cap.grab():
                    break
                idx += 1
                continue
            if (idx - first) % step:
                if not cap.grab():  # grab() decodes without the colour conversion: cheaper than read()
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, frame
            yielded += 1
            idx += 1
    finally:
        cap.release()


def extract_screen_text(
    video_path: str,
    interval: float = DEFAULT_INTERVAL,
    start: float = 0.0,
    end: Optional[float] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    backend: str = "auto",
    engine: Optional[Callable] = None,
    progress: Optional[ProgressCb] = None,
) -> List[Segment]:
    """OCR a video every `interval` seconds and return timed text segments.

    engine   optional callable(frame_bgr) -> [(text, confidence)] (tests / custom OCR);
             when None the lazy backend is loaded now.
    """
    ocr = engine or get_engine(backend)[0]
    total = None
    if progress:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            frames_n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25.0
        finally:
            cap.release()
        dur = (frames_n / fps) if frames_n > 0 else 0.0
        total = max(1, min(max_frames, int(((end if end is not None else dur) - start) / interval) + 1))

    samples: List[Tuple[float, List[Reading]]] = []
    prev_sig, prev_read, done, skipped = None, [], 0, 0
    for t, frame in iter_frames(video_path, interval, start, end, max_frames):
        small = _shrink(frame)
        sig = _signature(small)
        del frame
        if prev_sig is not None and float(abs(sig.astype("int16") - prev_sig.astype("int16")).mean()) < SAME_FRAME_DIFF:
            readings, skipped = prev_read, skipped + 1  # unchanged picture: reuse, no OCR
        else:
            try:
                readings = clean_readings(ocr(small), min_confidence)
                prev_sig, prev_read = sig, readings
            except Exception as exc:  # one bad frame must not kill the run...
                logger.warning("OCR failed on the frame at %.1fs: %s", t, exc)
                readings = []  # ...and its empty result is NOT remembered, so the next frame is read again
        del small
        samples.append((t, readings))
        done += 1
        if progress and total:
            progress(min(1.0, done / total))
    logger.info("OCR: %d frames sampled, %d reused without OCR", done, skipped)
    return merge_samples(samples, interval)


# --------------------------------------------------------------------------- #
# using the result
# --------------------------------------------------------------------------- #
def segments_to_words(segments: Sequence[Segment]) -> List[dict]:
    """Spread each segment's words evenly over its time span, in the same shape as faster-whisper's
    words, so on-screen text can be appended to a transcript."""
    words: List[dict] = []
    for seg in segments:
        toks = str(seg["text"]).split()
        if not toks:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        step = (e - s) / len(toks)
        for i, tok in enumerate(toks):
            words.append({"word": tok, "start": round(s + i * step, 3), "end": round(s + (i + 1) * step, 3),
                          "probability": float(seg.get("confidence", 1.0)), "source": "screen"})
    return words


def merge_transcript(words: Sequence[dict], segments: Sequence[Segment]) -> List[dict]:
    """One time-ordered transcript: spoken sentences + on-screen text.
    -> [{"source": "speech"|"screen", "start", "end", "text"}, ...]"""
    items: List[dict] = []
    sentence: List[dict] = []

    def flush():
        if sentence:
            items.append({"source": "speech", "start": float(sentence[0]["start"]), "end": float(sentence[-1]["end"]),
                          "text": " ".join(str(w["word"]).strip() for w in sentence).strip()})
            sentence.clear()

    for w in words:
        sentence.append(w)
        if str(w["word"]).strip().endswith((".", "!", "?")):
            flush()
    flush()
    for seg in segments:
        items.append({"source": "screen", "start": float(seg["start"]), "end": float(seg["end"]),
                      "text": str(seg["text"])})
    return sorted(items, key=lambda i: (i["start"], i["source"] != "speech"))


def narration_queue(segments: Sequence[Segment], max_lines: int = 8, min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                    min_word_chars: int = 2) -> List[dict]:
    """The text read from video frames, as an ordered queue ready to be narrated and/or shown on screen.

    Duplicates (the same line seen in several places) are dropped, single-character noise and low-confidence
    readings are skipped, and the order is the order in which the text appears in the video.
    -> [{"text","start","end","confidence","narration": "The text on screen reads: ..."}]"""
    queue: List[dict] = []
    seen = set()
    for seg in sorted(segments, key=lambda s: (float(s["start"]), -float(s.get("confidence", 1.0)))):
        text = re.sub(r"\s+", " ", str(seg["text"])).strip(" .")
        key = _norm(text)
        if not key or key in seen or len(re.findall(r"\w", text)) < min_word_chars:
            continue
        if float(seg.get("confidence", 1.0)) < min_confidence:
            continue
        seen.add(key)
        queue.append({"text": text, "start": float(seg["start"]), "end": float(seg["end"]),
                      "confidence": float(seg.get("confidence", 1.0)), "narration": f"The text on screen reads: {text}."})
        if len(queue) >= max_lines:
            break
    return queue


def screen_text_in_range(segments: Sequence[Segment], start: float, end: float) -> List[str]:
    """Distinct on-screen lines visible at any time within [start, end]."""
    seen, out = set(), []
    for seg in segments:
        if float(seg["start"]) < end and float(seg["end"]) > start:
            key = _norm(str(seg["text"]))
            if key and key not in seen:
                seen.add(key)
                out.append(str(seg["text"]))
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Read burned-in text from a video (CPU OCR)")
    ap.add_argument("video")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--backend", default="auto", choices=("auto", *BACKENDS))
    a = ap.parse_args()
    print(json.dumps(extract_screen_text(a.video, a.interval, backend=a.backend), indent=2, ensure_ascii=False))
