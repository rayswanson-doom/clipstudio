"""
backend/subtitle_engine.py

Word-accurate burned-in subtitles for the vertical clips.

  words  (faster-whisper, word_timestamps=True: device="cpu", compute_type="int8" - see transcriber.py)
    -> chunk_words()            crisp 3-5 word caption chunks, broken at sentence ends and pauses
    -> cues_for_clip()          the chunks that fall inside a clip, with times relative to the clip start
    -> cues_to_srt()            a plain .srt
    -> cues_to_ass()            a styled .ass: bold text, black outline, bottom-centre inside a safe zone, optional
                                karaoke highlight of the word being spoken
    -> video_engine.burn_subtitles()   ffmpeg `ass=` filter hard-codes it into the 9:16 picture

The cue/ASS generation follows the structure of nick8592/video-subtitler's add_subtitles.py (segments -> .srt -> ffmpeg
subtitles/ass burn-in), re-implemented here around word timestamps so each cue is a short punchy chunk rather than a
whole sentence. Pure Python: it imports nothing heavy, and the Whisper model is only loaded when words are requested.

Safe zone: Reels / Shorts / TikTok put their buttons, caption and progress bar over the bottom ~20 % of the frame and the
right edge. Captions are therefore centred horizontally and lifted to start 22 % above the bottom, with 8 % side margins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MIN_WORDS = 3
MAX_WORDS = 5
MAX_CHARS = 30  # a chunk this long still fits on one or two lines on a 1080-wide vertical frame
PAUSE_BREAK = 0.55  # seconds of silence that ends a chunk
MIN_CUE_SECONDS = 0.45
SAFE_BOTTOM_FRAC = 0.22
SAFE_SIDE_FRAC = 0.08

SIZE_PRESETS: Dict[str, float] = {"Small": 0.050, "Medium": 0.064, "Large": 0.080}  # font height / frame width
# ASS colours are &HAABBGGRR. "highlight" = colour of the word being spoken (karaoke); None = whole cue one colour.
COLOR_PRESETS: Dict[str, dict] = {
    "White": {"text": "&H00FFFFFF", "highlight": None},
    "Yellow": {"text": "&H0000F0FF", "highlight": None},
    "White + yellow active word": {"text": "&H00FFFFFF", "highlight": "&H0000F0FF"},
    "White + green active word": {"text": "&H00FFFFFF", "highlight": "&H0050FF50"},
}
DEFAULT_SIZE_PRESET = "Medium"
DEFAULT_COLOR_PRESET = "White"
_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*$")
_CLAUSE_END = re.compile(r"[,;:—-]['\")\]]*$")


@dataclass
class Style:
    size_preset: str = DEFAULT_SIZE_PRESET
    color_preset: str = DEFAULT_COLOR_PRESET
    font: str = "Arial"
    uppercase: bool = False
    outline: Optional[float] = None  # px; default scales with the font

    def validate(self) -> "Style":
        if self.size_preset not in SIZE_PRESETS:
            raise ValueError(f"Unknown font size '{self.size_preset}'. Use one of: {', '.join(SIZE_PRESETS)}.")
        if self.color_preset not in COLOR_PRESETS:
            raise ValueError(f"Unknown colour preset '{self.color_preset}'. Use one of: {', '.join(COLOR_PRESETS)}.")
        return self


# --------------------------------------------------------------------------- #
# words
# --------------------------------------------------------------------------- #
def transcribe_words(media_path: str, transcribe_fn: Optional[Callable[[str], list]] = None) -> List[dict]:
    """Word-level speech boundaries from faster-whisper: [{"word","start","end","probability"}] (seconds). The model is
    WhisperModel(device="cpu", compute_type="int8") with word_timestamps=True, loaded lazily and shared (transcriber.py)."""
    if transcribe_fn is None:
        try:
            from .transcriber import transcribe_audio as transcribe_fn  # type: ignore
        except ImportError:  # pragma: no cover
            from transcriber import transcribe_audio as transcribe_fn  # type: ignore
    return clean_words(transcribe_fn(media_path))


def clean_words(words: Sequence[dict]) -> List[dict]:
    """Drop empty / untimed words and fix overlaps so every word has start < end and the list is non-decreasing."""
    out: List[dict] = []
    prev_end = 0.0
    for w in words or []:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        try:
            s, e = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        s = max(s, prev_end - 0.02 if out else s, 0.0)
        e = max(e, s + 0.04)
        out.append({"word": text, "start": s, "end": e})
        prev_end = e
    return out


def _split_even(ws: List[dict], min_words: int, max_words: int) -> List[List[dict]]:
    """Cut one run of words into n = ceil(len / max_words) chunks of near-equal size (so 6 words -> 3+3, 7 -> 4+3,
    11 -> 4+4+3), moving each cut to a nearby clause end (a comma) when there is one within one word."""
    n = -(-len(ws) // max_words)
    if n <= 1:
        return [ws]
    base, extra = divmod(len(ws), n)
    sizes = [base + (1 if i < extra else 0) for i in range(n)]
    cuts, pos = [], 0
    for sz in sizes[:-1]:
        pos += sz
        for cand in (pos, pos - 1, pos + 1):  # prefer a clause boundary close to the even cut
            lo = (cuts[-1] if cuts else 0) + min_words
            if lo <= cand <= len(ws) - min_words and _CLAUSE_END.search(ws[cand - 1]["word"]):
                pos = cand
                break
        cuts.append(pos)
    out, prev = [], 0
    for c in cuts + [len(ws)]:
        out.append(ws[prev:c])
        prev = c
    return out


def chunk_words(words: Sequence[dict], min_words: int = MIN_WORDS, max_words: int = MAX_WORDS, max_chars: int = MAX_CHARS,
                pause: float = PAUSE_BREAK) -> List[dict]:
    """Group words into caption chunks of min_words..max_words (3-5 by default). Speech is first split into breaths
    (a sentence end or a pause of `pause` s), a breath too short to stand alone is joined to its neighbour when that
    fits, and each breath is then cut into even 3-5 word chunks - so a chunk never straddles a sentence end unless the
    sentence is shorter than min_words. Returns [{"start","end","text","words"}] with absolute times."""
    if min_words < 1 or max_words < min_words:
        raise ValueError("require 1 <= min_words <= max_words")
    ws = clean_words(words)
    runs: List[List[dict]] = []
    cur: List[dict] = []
    for i, w in enumerate(ws):
        cur.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt is None or _SENTENCE_END.search(w["word"]) or nxt["start"] - w["end"] >= pause:
            runs.append(cur)
            cur = []
    merged: List[List[dict]] = []
    for run in runs:  # join a too-short breath with the previous one when they still fit in one chunk
        if merged and (len(run) < min_words or len(merged[-1]) < min_words) and len(merged[-1]) + len(run) <= max_words \
                and run[0]["start"] - merged[-1][-1]["end"] < 2 * pause:
            merged[-1].extend(run)
        else:
            merged.append(run)
    chunks: List[List[dict]] = []
    for run in merged:
        chunks.extend(_split_even(run, min_words, max_words))
    return [{"start": c[0]["start"], "end": c[-1]["end"], "text": " ".join(x["word"] for x in c),
             "words": [dict(x) for x in c]} for c in chunks if c]


def cues_for_clip(words: Sequence[dict], clip_start: float, clip_end: float, **chunk_kw) -> List[dict]:
    """Caption cues for one clip [clip_start, clip_end] of the source: chunks of the words spoken inside it, with times
    relative to the clip. A cue is held a little (never past the next cue) so a quick word is still readable."""
    if clip_end <= clip_start:
        raise ValueError("clip_end must be after clip_start")
    inside = [w for w in words if float(w["end"]) > clip_start and float(w["start"]) < clip_end]
    rel = [{"word": w["word"], "start": max(0.0, float(w["start"]) - clip_start), "end": min(clip_end - clip_start, float(w["end"]) - clip_start)}
           for w in inside]
    cues = chunk_words(rel, **chunk_kw)
    dur = clip_end - clip_start
    for k, c in enumerate(cues):
        limit = (cues[k + 1]["start"] - 0.03) if k + 1 < len(cues) else dur
        hold = c["start"] + MIN_CUE_SECONDS
        c["end"] = round(max(c["end"], min(hold, limit)), 3)
        c["end"] = round(min(c["end"], limit, dur), 3)
        if c["end"] <= c["start"]:
            c["end"] = round(min(dur, c["start"] + 0.2), 3)
    return cues


# --------------------------------------------------------------------------- #
# formats
# --------------------------------------------------------------------------- #
def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def cues_to_srt(cues: Sequence[dict], uppercase: bool = False) -> str:
    """SubRip text for the cues."""
    blocks = []
    for n, c in enumerate(cues, start=1):
        text = c["text"].upper() if uppercase else c["text"]
        blocks.append(f"{n}\n{_srt_time(c['start'])} --> {_srt_time(c['end'])}\n{text}\n")
    return "\n".join(blocks)


def _ass_escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ").replace("\r", " ")


def ass_style_values(size: Tuple[int, int], style: Style) -> dict:
    """The numbers behind the ASS style (exposed so tests and the UI can show them)."""
    style.validate()
    W, H = size
    fs = max(18, int(round(W * SIZE_PRESETS[style.size_preset])))
    return {"fontsize": fs, "outline": style.outline if style.outline is not None else max(2.0, round(fs * 0.085, 1)),
            "shadow": max(1, int(round(fs * 0.03))), "margin_v": int(round(H * SAFE_BOTTOM_FRAC)),
            "margin_lr": int(round(W * SAFE_SIDE_FRAC))}


def cues_to_ass(cues: Sequence[dict], size: Tuple[int, int] = (1080, 1920), style: Optional[Style] = None) -> str:
    """A styled Advanced SubStation Alpha script: bold, black outline, bottom-centre (Alignment 2) lifted by the safe-zone
    margin. PlayRes equals the video size so the sizes above are real pixels. With a highlight preset every word carries a
    \\kf karaoke tag so the spoken word changes colour as it is said."""
    style = (style or Style()).validate()
    W, H = int(size[0]), int(size[1])
    v = ass_style_values((W, H), style)
    colors = COLOR_PRESETS[style.color_preset]
    primary = colors["highlight"] or colors["text"]  # karaoke: text becomes "primary" as it is sung
    secondary = colors["text"]
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{v['fontsize']},{primary},{secondary},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
        f"{v['outline']},{v['shadow']},2,{v['margin_lr']},{v['margin_lr']},{v['margin_v']},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for c in cues:
        if colors["highlight"] and c.get("words"):
            parts = []
            ws = c["words"]
            for k, w in enumerate(ws):
                end = ws[k + 1]["start"] if k + 1 < len(ws) else max(w["end"], min(c["end"], w["end"] + 0.2))
                cs = max(1, int(round((end - w["start"]) * 100)))
                word = _ass_escape(w["word"].upper() if style.uppercase else w["word"])
                parts.append(f"{{\\kf{cs}}}{word}")
            lead = max(0.0, c["words"][0]["start"] - c["start"])
            text = (f"{{\\k{int(round(lead * 100))}}}" if lead > 0.01 else "") + " ".join(parts)
        else:
            text = _ass_escape(c["text"].upper() if style.uppercase else c["text"])
        lines.append(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Default,,0,0,0,,{text}")
    return head + "\n".join(lines) + "\n"


def write_subtitle_files(cues: Sequence[dict], base_path: str, size: Tuple[int, int] = (1080, 1920),
                         style: Optional[Style] = None) -> Dict[str, str]:
    """Write base_path + ".srt" and ".ass" (UTF-8); returns {"srt","ass"}."""
    style = (style or Style()).validate()
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    srt, ass = str(base.with_suffix(".srt")), str(base.with_suffix(".ass"))
    Path(srt).write_text(cues_to_srt(cues, style.uppercase), encoding="utf-8")
    Path(ass).write_text(cues_to_ass(cues, size, style), encoding="utf-8")
    return {"srt": srt, "ass": ass}
