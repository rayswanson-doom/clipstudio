"""
meme_matcher.py - Auto-Meme Reaction Matching for CinePulse AI (CPU only, no paid API, no internet).

Reads the emotional tone of every script segment / transcript highlight and picks the reaction clip from
`storage/memes/` that fits it best, plus exactly where to put it on the timeline.

Patterns borrowed from the two reference projects (their ideas, not their code):
  * DaBlincx/meme-video-gen  - a folder of your own clips is used first; an archive of what was already used keeps
                               the output from repeating itself; everything is assembled on the CPU with ffmpeg.
  * ssrajadh/sentrysearch    - index the clips once and cache the index on disk; search it with a natural-language
                               query; rank by similarity; apply a confidence threshold; drop near-duplicate results;
                               keep a dead-letter list of files that could not be indexed.

How it matches (all local):
  1. Sentiment: a small emotion lexicon (words, phrases, emoji, negation, intensifiers, "!!!", CAPS) turns a segment
     into {emotion, intensity}. Optionally a Groq / Gemini client can tag the segments instead (structured JSON); any
     failure falls back to the local lexicon.
  2. The library: every clip is described by its folder name (storage/memes/shock/clip.mp4), an optional memes.json
     sidecar (tags / description) and its file name. Those words are matched against the segment's emotion and text
     (emotion weight + hashed-embedding similarity; MiniLM is used when sentence-transformers is installed).
  3. Planning: the strongest reactions are kept, at most N per minute with a minimum gap, no clip is reused while an
     unused one fits, and the usage archive favours clips you have not seen recently.

Honest limits: this does not "watch" the meme videos (that needs a vision model); a clip is understood through its
folder, file name and tags, so name your clips for the feeling they show. Sentiment is lexicon based: good for clear
reactions ("no way!", "I love it", "that failed badly"), weak on sarcasm and subtle tone.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MIN_CLIP_SECONDS = 0.3
INDEX_NAME = ".meme_index.json"
USAGE_NAME = ".meme_usage.json"
SIDECAR_NAME = "memes.json"
EMOTIONS = ("laugh", "shock", "sad", "angry", "cringe", "confused", "win", "love", "fear", "cool")
MODES = ("auto", "overlay", "cutaway")
POSITIONS = ("auto", "Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Custom Coordinates")

# --------------------------------------------------------------------------- #
# emotion lexicon
# --------------------------------------------------------------------------- #
LEXICON: Dict[str, Tuple[str, ...]] = {
    "laugh": ("lol", "lmao", "haha", "hahaha", "funny", "hilarious", "joke", "laugh", "laughing", "laughed", "comedy",
              "silly", "ridiculous", "hysterical", "cracking up", "rofl", "prank", "pun", "giggle", "amusing", "fun",
              "😂", "🤣", "😆", "😹", "kidding", "roast", "meme"),
    "shock": ("wow", "omg", "no way", "unbelievable", "incredible", "insane", "crazy", "shocking", "shocked", "surprised",
              "surprise", "suddenly", "unexpected", "can't believe", "cannot believe", "what the", "wait what",
              "jaw", "mind blown", "mind-blown", "huge", "boom", "whoa", "never seen", "impossible", "😱", "😮", "🤯", "😲"),
    "sad": ("sad", "cry", "crying", "cried", "tears", "lost", "lonely", "alone", "miss", "missed", "heartbroken",
            "heartbreak", "depressed", "sorry", "unfortunately", "died", "death", "gone", "grief", "hopeless", "broke",
            "sadly", "disappointed", "disappointing", "😢", "😭", "💔", "😞", "pain", "hurt"),
    "angry": ("angry", "furious", "hate", "mad", "rage", "annoyed", "annoying", "terrible", "worst", "stupid", "idiot",
              "unacceptable", "disgusting", "outrageous", "ridiculous!", "fed up", "sick of", "scam", "liar", "lied",
              "betrayed", "😡", "🤬", "😠", "damn", "ruined", "hell"),
    "cringe": ("cringe", "awkward", "embarrassing", "embarrassed", "yikes", "oops", "awful", "gross", "ew", "ugh", "weird",
               "uncomfortable", "facepalm", "fail", "failed", "failure", "mistake", "blunder", "bad idea", "tragic",
               "painful to watch", "🤦", "😬", "😖", "disaster", "bombed", "flop", "flopped"),
    "confused": ("confused", "confusing", "why", "what?", "huh", "how?", "doesn't make sense", "does not make sense",
                 "no idea", "puzzled", "strange", "mystery", "unclear", "wonder", "wondering", "weirdly", "makes no sense",
                 "wait", "hmm", "really?", "🤔", "😕", "❓", "?!", "lost me"),
    "win": ("win", "won", "winning", "victory", "success", "succeeded", "nailed", "champion", "best", "amazing",
            "awesome", "great", "perfect", "yes!", "finally", "achieved", "record", "congratulations", "proud", "epic",
            "legend", "goal", "crushed it", "celebrate", "celebration", "🎉", "🏆", "💪", "🥳", "bingo", "jackpot",
            "unbelievable!", "made it"),
    "love": ("love", "loved", "adore", "adorable", "cute", "sweet", "heart", "beautiful", "gorgeous", "kiss", "romantic",
             "romance", "darling", "favorite", "favourite", "crush", "wholesome", "❤", "😍", "🥰", "💕", "😘", "thank you",
             "grateful", "blessed"),
    "fear": ("scared", "scary", "afraid", "fear", "terrified", "terrifying", "horror", "nervous", "panic", "danger",
             "dangerous", "run", "help", "threat", "nightmare", "creepy", "worried", "anxious", "anxiety", "trembling",
             "😨", "😰", "😱!", "spooky", "haunted", "ghost", "monster"),
    "cool": ("cool", "smooth", "easy", "boss", "swag", "deal with it", "classic", "calm", "chill", "effortless", "style",
             "stylish", "legendary", "smug", "obviously", "of course", "naturally", "didn't even", "😎", "sunglasses",
             "like a pro", "no problem", "casually"),
}
INTENSIFIERS = {"very": 1.3, "so": 1.25, "really": 1.25, "extremely": 1.5, "totally": 1.3, "absolutely": 1.5, "completely": 1.3,
                "super": 1.3, "incredibly": 1.5, "literally": 1.2, "insanely": 1.5, "truly": 1.2, "utterly": 1.4, "such": 1.2}
NEGATIONS = ("not", "no", "never", "n't", "without", "hardly", "barely", "isn't", "wasn't", "don't", "didn't", "can't")
ALIASES = {  # words in a folder / file name -> emotion
    "laugh": "laugh", "lol": "laugh", "lmao": "laugh", "haha": "laugh", "funny": "laugh", "joke": "laugh", "laughing": "laugh",
    "happy": "laugh", "giggle": "laugh", "rofl": "laugh",
    "shock": "shock", "shocked": "shock", "omg": "shock", "wow": "shock", "surprise": "shock", "surprised": "shock",
    "whoa": "shock", "pikachu": "shock", "gasp": "shock",
    "sad": "sad", "cry": "sad", "crying": "sad", "tears": "sad", "sob": "sad", "disappointed": "sad",
    "angry": "angry", "anger": "angry", "mad": "angry", "rage": "angry", "furious": "angry",
    "cringe": "cringe", "awkward": "cringe", "facepalm": "cringe", "yikes": "cringe", "fail": "cringe", "oops": "cringe",
    "confused": "confused", "confusion": "confused", "huh": "confused", "thinking": "confused", "hmm": "confused",
    "question": "confused", "what": "confused",
    "win": "win", "victory": "win", "yes": "win", "celebrate": "win", "celebration": "win", "success": "win", "proud": "win",
    "cheer": "win", "clap": "win", "applause": "win", "goal": "win",
    "love": "love", "heart": "love", "hearts": "love", "cute": "love", "kiss": "love", "aww": "love",
    "fear": "fear", "scared": "fear", "afraid": "fear", "terror": "fear", "nervous": "fear", "panic": "fear", "sweat": "fear",
    "cool": "cool", "sunglasses": "cool", "smooth": "cool", "boss": "cool", "swag": "cool", "calm": "cool",
}
_WORD = re.compile(r"[a-zA-Z']+")


def _emoji_chars(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) > 0x2000)


def analyze_text(text: str) -> dict:
    """Local sentiment: {"emotion", "intensity" 0-1, "scores", "valence" -1..1, "matched"}. "neutral" when nothing fits."""
    raw = str(text or "")
    low = raw.casefold()
    tokens = _WORD.findall(low)
    scores = {e: 0.0 for e in EMOTIONS}
    matched: List[str] = []
    for emo, terms in LEXICON.items():
        for term in terms:
            t = term.casefold()
            if not t:
                continue
            if " " in t or not t.isascii() or not t.replace("'", "").isalpha():
                if len(t) <= 2 and t.isascii():
                    continue
                idxs = [m.start() for m in re.finditer(re.escape(t), low)] if (" " in t or not t.isascii() or not t[-1:].isalpha()) else []
                hits = [(i, i) for i in idxs]
                tok_positions = [len(_WORD.findall(low[:i])) for i, _ in hits]
            else:
                tok_positions = [k for k, w in enumerate(tokens) if w == t]
            for pos in tok_positions:
                w = 1.0
                window = tokens[max(0, pos - 3):pos]
                if any(x in INTENSIFIERS for x in window):
                    w *= max(INTENSIFIERS[x] for x in window if x in INTENSIFIERS)
                negated = any(x in NEGATIONS or x.endswith("n't") for x in window[-2:])
                if negated:
                    w *= -0.6
                scores[emo] += w
                if w > 0:
                    matched.append(t)
    bangs = min(3, raw.count("!"))
    caps_words = [w for w in re.findall(r"\b[A-Z]{3,}\b", raw)]
    boost = 1.0 + 0.12 * bangs + (0.15 if caps_words else 0.0)
    if "?!" in raw or "!?" in raw:
        scores["shock"] += 0.6
        scores["confused"] += 0.4
    elif raw.count("?") >= 2:
        scores["confused"] += 0.5
    scores = {e: max(0.0, v) for e, v in scores.items()}
    total = sum(scores.values())
    if total <= 0:
        return {"emotion": "neutral", "intensity": 0.0, "scores": scores, "valence": 0.0, "matched": []}
    top = max(scores, key=lambda e: (scores[e], -EMOTIONS.index(e)))
    strength = scores[top] * boost
    intensity = 1.0 - math.exp(-0.9 * strength)  # 1 hit ~0.6, 2 hits ~0.84, saturates at 1
    if top in ("laugh", "win", "love", "cool"):
        valence = 0.6
    elif top in ("sad", "angry", "cringe", "fear"):
        valence = -0.6
    else:
        valence = 0.0
    return {"emotion": top, "intensity": round(min(1.0, intensity), 3), "scores": {e: round(v, 3) for e, v in scores.items()},
            "valence": valence, "matched": sorted(set(matched))[:8]}


LLM_SCHEMA = {"tags": [{"idx": int, "emotion": str, "intensity": (int, float)}]}


def analyze_with_llm(client, texts: Sequence[str]) -> List[Optional[dict]]:
    """Groq / Gemini structured-JSON emotion tags (vimax_orchestrator clients). One entry per text, None where the
    model gave nothing usable (the caller then uses analyze_text). Never raises."""
    out: List[Optional[dict]] = [None] * len(texts)
    if client is None or getattr(client, "offline", True) or not texts:
        return out
    system = ("You label the emotional tone of short script lines for a reaction-meme matcher. Reply with ONLY JSON "
              '{"tags":[{"idx":0,"emotion":"<one of: ' + ", ".join(EMOTIONS) + ', neutral>","intensity":0.0-1.0}]}. '
              "Use neutral with intensity 0 when no reaction fits.")
    for lo in range(0, len(texts), 25):
        chunk = list(texts[lo:lo + 25])
        user = "LINES: " + json.dumps([{"idx": i, "text": t[:300]} for i, t in enumerate(chunk)])
        try:
            data = client.complete_json(system, user, LLM_SCHEMA, temperature=0.1, max_tokens=1500)
        except Exception as exc:  # noqa: BLE001 - any LLM problem means "use the local lexicon"
            logger.warning("meme emotion tagging failed: %s", exc)
            return out
        for tag in data["tags"]:
            i = tag["idx"]
            emo = str(tag["emotion"]).strip().lower()
            if 0 <= i < len(chunk) and (emo in EMOTIONS or emo == "neutral"):
                out[lo + i] = {"emotion": emo, "intensity": round(max(0.0, min(1.0, float(tag["intensity"]))), 3),
                               "scores": {}, "valence": 0.0, "matched": [], "source": "llm"}
    return out


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def segments_from_text(text: str, total_seconds: float) -> List[dict]:
    """Sentences of a script with times proportional to their word counts (the voiceover paces itself the same way)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if s.strip()]
    if not sents or total_seconds <= 0:
        return []
    weights = [max(1, len(s.split())) for s in sents]
    total_w = float(sum(weights))
    t, out = 0.0, []
    for s, w in zip(sents, weights):
        d = total_seconds * w / total_w
        out.append({"start": round(t, 3), "end": round(t + d, 3), "text": s})
        t += d
    return out


def segments_from_words(words: Sequence[dict], start: float = 0.0, end: Optional[float] = None, max_gap: float = 0.7,
                        max_len: float = 8.0) -> List[dict]:
    """Group transcript words ({"word","start","end"}) into sentence-like segments, times relative to `start`."""
    out: List[dict] = []
    cur: List[dict] = []

    def flush():
        if cur:
            out.append({"start": round(max(0.0, cur[0]["start"] - start), 3), "end": round(max(0.0, cur[-1]["end"] - start), 3),
                        "text": " ".join(str(w["word"]).strip() for w in cur).strip()})
            cur.clear()

    for w in words:
        if end is not None and (w["end"] <= start or w["start"] >= end):
            continue
        if cur and (w["start"] - cur[-1]["end"] > max_gap or w["end"] - cur[0]["start"] > max_len
                    or re.search(r"[.!?]$", str(cur[-1]["word"]).strip())):
            flush()
        cur.append(w)
    flush()
    return [s for s in out if s["text"] and s["end"] > s["start"]]


def segments_from_shots(shots: Sequence[dict]) -> List[dict]:
    """Story storyboard shots -> segments: what is said (or, failing that, what happens) with the shot's timing."""
    out = []
    for s in shots:
        text = s.get("line") or s.get("action") or ""
        if text.strip():
            out.append({"start": float(s["start"]), "end": float(s["end"]), "text": text.strip(), "shot": s.get("idx")})
    return out


# --------------------------------------------------------------------------- #
# settings + locations
# --------------------------------------------------------------------------- #
@dataclass
class MemeSettings:
    enabled: bool = False
    mode: str = "auto"              # auto | overlay | cutaway
    max_per_minute: float = 4.0
    sensitivity: float = 0.5        # 0 = only the strongest reactions ... 1 = react to almost everything
    use_audio: bool = True          # play the meme's own sound
    volume: float = 0.7
    position: str = "auto"          # corner for overlays; "auto" avoids the avatar and other memes
    xy: Optional[Tuple[float, float]] = None
    scale: float = 0.34             # overlay width / video width
    max_seconds: float = 3.5        # longest stretch of one meme that is used
    min_score: float = 0.35         # confidence threshold (sentrysearch-style): weaker matches are dropped
    library_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MemeSettings":
        d = dict(d or {})
        allowed = {k: d[k] for k in cls.__dataclass_fields__ if k in d}  # unknown keys are ignored
        if allowed.get("xy"):
            allowed["xy"] = (float(allowed["xy"][0]), float(allowed["xy"][1]))
        s = cls(**allowed)
        s.validate()
        return s

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"meme mode must be one of: {', '.join(MODES)}")
        if self.position not in POSITIONS:
            raise ValueError(f"meme position must be one of: {', '.join(POSITIONS)}")
        if self.position == "Custom Coordinates" and not self.xy:
            raise ValueError("Custom Coordinates needs an X and a Y position for the meme.")
        self.max_per_minute = float(max(0.5, min(20.0, self.max_per_minute)))
        self.sensitivity = float(max(0.0, min(1.0, self.sensitivity)))
        self.volume = float(max(0.0, min(2.0, self.volume)))
        self.scale = float(max(0.15, min(0.8, self.scale)))
        self.max_seconds = float(max(0.8, min(12.0, self.max_seconds)))
        self.min_score = float(max(0.0, min(1.0, self.min_score)))

    def to_dict(self) -> dict:
        return asdict(self)


def default_meme_dir() -> Path:
    """storage/memes next to the exe when packaged (the temp dir if that is read-only), else in the project folder."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    env = os.environ.get("CINEPULSE_MEMES")
    target = Path(env) if env else base / "storage" / "memes"
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".w"
        probe.write_text("x")
        probe.unlink()
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "CinePulseAI" / "storage" / "memes"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# --------------------------------------------------------------------------- #
# the library (sentrysearch-style cached index)
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z]+", str(text).casefold()) if len(t) > 1]


def _safe_filename(name: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(str(name)))
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._") or "meme"
    return stem[:80] + ext.lower()


class MemeLibrary:
    """The clips in a folder, indexed once and cached in `.meme_index.json` (only new / changed files are re-read)."""

    def __init__(self, folder: Optional[str] = None, ffmpeg_path: Optional[str] = None):
        self.dir = Path(folder) if folder else default_meme_dir()
        self.ffmpeg_path = ffmpeg_path
        self.clips: List[dict] = []
        self.failed: List[dict] = []  # dead-letter list: {"path","reason"}
        self._scanned = False

    # -- scanning ---------------------------------------------------------- #
    def _ffmpeg(self) -> str:
        try:
            from . import video_engine as ve
        except ImportError:  # pragma: no cover
            import video_engine as ve
        return ve.find_ffmpeg(self.ffmpeg_path)

    def _sidecar(self) -> dict:
        p = self.dir / SIDECAR_NAME
        if not p.is_file():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            self.failed.append({"path": str(p), "reason": f"{SIDECAR_NAME} is not valid JSON (tags ignored)"})
            return {}

    def scan(self, force: bool = False) -> "MemeLibrary":
        self.clips, self.failed = [], []
        if not self.dir.is_dir():
            self._scanned = True
            return self
        idx_path = self.dir / INDEX_NAME
        cache: Dict[str, dict] = {}
        if idx_path.is_file() and not force:
            try:
                cache = json.loads(idx_path.read_text(encoding="utf-8")).get("files", {})
            except (OSError, ValueError):
                cache = {}
        sidecar = self._sidecar()
        fresh: Dict[str, dict] = {}
        ff = None
        for f in sorted(self.dir.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS or f.name.startswith("."):
                continue
            rel = f.relative_to(self.dir).as_posix()
            try:
                st = f.stat()
            except OSError as exc:
                self.failed.append({"path": rel, "reason": str(exc)})
                continue
            hit = cache.get(rel)
            if hit and hit.get("size") == st.st_size and int(hit.get("mtime", 0)) == int(st.st_mtime):
                probe = hit
            else:
                ff = ff or self._ffmpeg()
                try:
                    try:
                        from . import video_engine as ve
                    except ImportError:  # pragma: no cover
                        import video_engine as ve
                    info = ve._probe_media(ff, str(f))
                except Exception as exc:  # noqa: BLE001
                    self.failed.append({"path": rel, "reason": f"could not be read ({exc})"})
                    continue
                if not info["has_video"] or info["width"] <= 0:
                    self.failed.append({"path": rel, "reason": "no video stream (not a playable video)"})
                    continue
                if 0 < info["duration"] < MIN_CLIP_SECONDS:
                    self.failed.append({"path": rel, "reason": f"too short ({info['duration']:.2f}s)"})
                    continue
                probe = {"size": st.st_size, "mtime": int(st.st_mtime), "duration": round(info["duration"] or 1.0, 3),
                         "width": info["width"], "height": info["height"], "has_audio": bool(info["has_audio"])}
            fresh[rel] = probe
            self.clips.append(self._describe(rel, f, probe, sidecar))
        try:
            idx_path.write_text(json.dumps({"version": 1, "files": fresh}), encoding="utf-8")
        except OSError:
            pass  # a read-only library still works, it just re-reads the files next time
        self._scanned = True
        return self

    def _describe(self, rel: str, path: Path, probe: dict, sidecar: dict) -> dict:
        folder_words = _tokens(Path(rel).parent.as_posix()) if Path(rel).parent.as_posix() != "." else []
        name_words = _tokens(path.stem)
        extra = sidecar.get(rel) or sidecar.get(path.name) or {}
        tags = [str(t).casefold() for t in (extra.get("tags") or [])] if isinstance(extra, dict) else []
        description = str(extra.get("description", "")) if isinstance(extra, dict) else ""
        emotions: Dict[str, float] = {}

        def add(emo: str, w: float) -> None:
            emotions[emo] = max(emotions.get(emo, 0.0), w)

        for w in folder_words:                      # storage/memes/shock/anything.mp4  (strongest signal)
            if w in EMOTIONS:
                add(w, 1.0)
            elif w in ALIASES:
                add(ALIASES[w], 1.0)
        explicit = extra.get("emotion") if isinstance(extra, dict) else None
        for e in ([explicit] if isinstance(explicit, str) else list(explicit or [])):
            e = str(e).casefold()
            if e in EMOTIONS:
                add(e, 1.0)
        for w in tags + _tokens(description):      # sidecar tags / description
            if w in EMOTIONS:
                add(w, 0.9)
            elif w in ALIASES:
                add(ALIASES[w], 0.9)
        for w in name_words:                        # laugh_cat.mp4, surprised-pikachu.mp4
            if w in EMOTIONS:
                add(w, 0.8)
            elif w in ALIASES:
                add(ALIASES[w], 0.8)
        max_s = extra.get("max_seconds") if isinstance(extra, dict) else None
        return {"path": str(path), "rel": rel, "name": path.name, "duration": probe["duration"], "has_audio": probe["has_audio"],
                "width": probe["width"], "height": probe["height"], "emotions": emotions,
                "tags": sorted(set(folder_words + name_words + tags)), "description": description,
                "tag_text": " ".join(folder_words + name_words + tags + [description]).strip(),
                "max_seconds": float(max_s) if isinstance(max_s, (int, float)) and max_s > 0 else None}

    def ensure_scanned(self) -> "MemeLibrary":
        return self if self._scanned else self.scan()

    # -- management -------------------------------------------------------- #
    def summary(self) -> dict:
        self.ensure_scanned()
        per: Dict[str, int] = {}
        for c in self.clips:
            for e in (c["emotions"] or {"untagged": 1}):
                per[e] = per.get(e, 0) + 1
        return {"folder": str(self.dir), "clips": len(self.clips), "per_emotion": dict(sorted(per.items())),
                "untagged": [c["name"] for c in self.clips if not c["emotions"]], "failed": list(self.failed)}

    def add_file(self, filename: str, data: bytes, emotion: Optional[str] = None) -> dict:
        """Save an uploaded reaction clip (into storage/memes/<emotion>/ when an emotion is given) and index it."""
        ext = Path(filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError(f"'{filename}' is not a supported video ({', '.join(VIDEO_EXTENSIONS)}).")
        if not data:
            raise ValueError(f"'{filename}' is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"'{filename}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        emotion = (emotion or "").strip().casefold() or None
        if emotion and emotion not in EMOTIONS:
            raise ValueError(f"Unknown emotion '{emotion}'. Use one of: {', '.join(EMOTIONS)}.")
        folder = self.dir / emotion if emotion else self.dir
        folder.mkdir(parents=True, exist_ok=True)
        name = _safe_filename(filename)
        target = folder / name
        n = 1
        while target.exists():
            n += 1
            target = folder / f"{Path(name).stem}_{n}{ext}"
        target.write_bytes(data)
        try:
            self._scanned = False
            self.scan()
        except Exception:  # noqa: BLE001
            target.unlink(missing_ok=True)
            raise
        clip = next((c for c in self.clips if Path(c["path"]) == target), None)
        if clip is None:
            reason = next((f["reason"] for f in self.failed if f["path"] == target.relative_to(self.dir).as_posix()), "unreadable")
            target.unlink(missing_ok=True)
            self._scanned = False
            self.scan()
            raise ValueError(f"'{filename}' was not added: {reason}.")
        return clip

    def remove(self, rel: str) -> bool:
        target = (self.dir / rel).resolve()
        if self.dir.resolve() not in target.parents or target.suffix.lower() not in VIDEO_EXTENSIONS or not target.is_file():
            return False
        target.unlink()
        self._scanned = False
        self.scan()
        return True

    # -- usage archive (meme-video-gen's archive.txt idea) ------------------ #
    def usage(self) -> Dict[str, int]:
        p = self.dir / USAGE_NAME
        try:
            return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError, AttributeError):
            return {}

    def record_usage(self, matches: Iterable[dict]) -> None:
        use = self.usage()
        for m in matches:
            rel = m.get("meme_rel") or m.get("meme_name")
            if rel:
                use[rel] = use.get(rel, 0) + 1
        try:
            (self.dir / USAGE_NAME).write_text(json.dumps(use), encoding="utf-8")
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
_EMBEDDER = None


def _embedder(prefer: str = "hashed"):
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from . import novel_adapter as na
        except ImportError:  # pragma: no cover
            import novel_adapter as na
        _EMBEDDER = na.get_embedder(prefer, [])
    return _EMBEDDER


def _cosine_texts(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        v = _embedder().encode([a, b])
        return float(max(0.0, (v[0] * v[1]).sum()))
    except Exception:  # noqa: BLE001 - similarity is a bonus signal; matching still works without it
        return 0.0


def score_clip(clip: dict, emotion: str, text: str, used: int = 0, run_used: int = 0) -> float:
    """0..1: how well one clip fits one emotional reading of a segment."""
    emo_w = clip["emotions"].get(emotion, 0.0)
    sim = _cosine_texts(text, clip["tag_text"]) if clip["tag_text"] else 0.0
    score = 0.65 * emo_w + 0.25 * min(1.0, sim * 2.5) + 0.10
    if not clip["emotions"]:
        score = 0.20 + 0.25 * min(1.0, sim * 2.5)  # an untagged clip can still win on its name / description
    score -= min(0.20, 0.04 * used) + 0.25 * run_used  # prefer clips not used before, never repeat inside one video
    return max(0.0, min(1.0, score))


def plan_memes(segments: Sequence[dict], library: MemeLibrary, settings: Optional[dict] = None, total_seconds: Optional[float] = None,
               avoid_rects: Sequence[Tuple[int, int, int, int]] = (), frame_size: Optional[Tuple[int, int]] = None,
               llm=None, clamp: bool = False) -> dict:
    """Pick reaction clips for the segments. Returns
    {"matches": [...], "skipped": [{"segment","reason"}], "analysis": [...], "warnings": [...]}.
    Each match: segment, start, end (when on screen), duration, text, emotion, intensity, meme_path, meme_rel, meme_name,
    score, mode ("overlay" | "cutaway"), trim_start, use_audio, volume, position, xy, scale, reason.
    clamp=True keeps every meme inside its own segment (used when each segment is rendered on its own, e.g. story shots)."""
    cfg = settings if isinstance(settings, MemeSettings) else MemeSettings.from_dict(settings)
    warnings: List[str] = []
    library.ensure_scanned()
    for f in library.failed:
        warnings.append(f"Meme '{f['path']}' was skipped: {f['reason']}.")
    result = {"matches": [], "skipped": [], "analysis": [], "warnings": warnings}
    if not library.clips:
        warnings.append(f"No reaction clips found in {library.dir}. Add some, or use 'Add starter reactions'.")
        return result
    segs = [dict(s) for s in segments if str(s.get("text", "")).strip() and s["end"] > s["start"]]
    if not segs:
        return result
    total = float(total_seconds or max(s["end"] for s in segs))
    llm_tags = analyze_with_llm(llm, [s["text"] for s in segs]) if llm is not None else [None] * len(segs)
    if llm is not None and not getattr(llm, "offline", True) and all(t is None for t in llm_tags):
        warnings.append("Emotion tagging by the AI service did not work; used the built-in sentiment rules.")
    threshold = 0.55 - 0.40 * cfg.sensitivity
    used_before = library.usage()
    run_used: Dict[str, int] = {}
    candidates = []
    for i, (seg, tag) in enumerate(zip(segs, llm_tags)):
        a = tag if tag and tag["emotion"] != "neutral" else analyze_text(seg["text"])
        if tag and tag["emotion"] == "neutral":
            a = {**analyze_text(seg["text"]), "emotion": "neutral", "intensity": 0.0}
        result["analysis"].append({"segment": i, "start": seg["start"], "end": seg["end"], "text": seg["text"],
                                   "emotion": a["emotion"], "intensity": a["intensity"], "source": a.get("source", "lexicon")})
        if a["emotion"] == "neutral" or a["intensity"] < threshold:
            result["skipped"].append({"segment": i, "reason": "no strong emotion" if a["emotion"] == "neutral" else
                                      f"{a['emotion']} too mild ({a['intensity']:.2f} < {threshold:.2f})"})
            continue
        candidates.append((i, seg, a))
    # the strongest reactions first, within the per-minute budget and the minimum gap (sentrysearch-style dedupe)
    budget = max(1, int(math.ceil(total / 60.0 * cfg.max_per_minute)))
    min_gap = max(2.0, 60.0 / cfg.max_per_minute * 0.6)
    chosen: List[tuple] = []
    for i, seg, a in sorted(candidates, key=lambda c: (-c[2]["intensity"], c[1]["start"])):
        if len(chosen) >= budget:
            result["skipped"].append({"segment": i, "reason": "over the per-minute meme limit"})
            continue
        if any(abs(seg["end"] - c[1]["end"]) < min_gap for c in chosen):
            result["skipped"].append({"segment": i, "reason": "too close to another meme"})
            continue
        chosen.append((i, seg, a))
    placed: List[dict] = []
    for i, seg, a in sorted(chosen, key=lambda c: c[1]["start"]):
        ranked = sorted(((score_clip(c, a["emotion"], seg["text"], used_before.get(c["rel"], 0), run_used.get(c["rel"], 0)), c)
                         for c in library.clips), key=lambda sc: (-sc[0], sc[1]["rel"]))
        best_score, clip = ranked[0]
        if best_score < cfg.min_score:
            result["skipped"].append({"segment": i, "reason": f"no clip fits {a['emotion']} well enough "
                                                              f"(best {best_score:.2f} < {cfg.min_score:.2f})"})
            continue
        run_used[clip["rel"]] = run_used.get(clip["rel"], 0) + 1
        seg_len = seg["end"] - seg["start"]
        length = min(clip["duration"], clip["max_seconds"] or cfg.max_seconds, cfg.max_seconds)
        mode = cfg.mode
        if mode == "auto":  # a big reaction takes the whole screen, a smaller one sits in a corner
            mode = "cutaway" if (a["intensity"] >= 0.8 and length >= 1.0) else "overlay"
        if clamp:
            length = round(min(length, seg_len), 2)
            end = min(seg["end"], total)
            start = max(seg["start"], end - length)  # never before the segment starts
        else:
            length = round(max(0.8, min(length, seg_len + 0.8)), 2)
            end = min(seg["end"] + 0.4, total)      # the reaction lands at the end of the line, on the transition
            start = max(0.0, end - length)
        end = round(min(total, start + length), 3)
        if end - start < 0.3:
            result["skipped"].append({"segment": i, "reason": "no room at the end of the video"})
            continue
        placed.append({"segment": i, "shot": seg.get("shot"), "start": round(start, 3), "end": end, "duration": round(end - start, 3), "text": seg["text"],
                       "emotion": a["emotion"], "intensity": a["intensity"], "meme_path": clip["path"], "meme_rel": clip["rel"],
                       "meme_name": clip["name"], "score": round(best_score, 3), "mode": mode, "trim_start": 0.0,
                       "aspect": round(clip["width"] / max(1, clip["height"]), 4),
                       "use_audio": bool(cfg.use_audio and clip["has_audio"]), "volume": cfg.volume, "position": cfg.position,
                       "xy": cfg.xy, "scale": cfg.scale,
                       "reason": f"{a['emotion']} ({a['intensity']:.2f}) -> {clip['name']} (match {best_score:.2f})"})
    if frame_size:
        assign_corners(placed, frame_size, avoid_rects, cfg.scale)
    result["matches"] = placed
    return result


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _rects_hit(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def assign_corners(matches: List[dict], frame_size: Tuple[int, int], avoid_rects: Sequence[Tuple[int, int, int, int]] = (),
                   scale: float = 0.34) -> None:
    """Give every overlay meme a screen corner (or the requested one) that does not cover the 3D avatar, the character
    portraits or another meme showing at the same time. Sets m["position"] / m["rect"]."""
    try:
        from . import video_engine as ve
    except ImportError:  # pragma: no cover
        import video_engine as ve
    W, H = frame_size
    taken: List[Tuple[Tuple[float, float], Tuple[int, int, int, int]]] = []
    for m in matches:
        if m["mode"] == "cutaway":
            m["rect"] = (0, 0, W, H)
            continue
        ratio = float(m.get("aspect") or 1.0)  # the clip's own width / height
        size = (max(16, int(W * m.get("scale", scale))), max(16, int(W * m.get("scale", scale) / ratio)))
        if m["position"] != "auto":
            preset = ve.normalize_position_preset(m["position"])
            m["rect"] = ve.compute_pip_rect((W, H), size, preset, m.get("xy"), m.get("scale", scale))
            taken.append(((m["start"], m["end"]), m["rect"]))
            continue
        chosen = None
        for preset in ("Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left"):
            rect = ve.compute_pip_rect((W, H), size, preset, None, m.get("scale", scale))
            if any(_rects_hit(rect, tuple(r)) for r in avoid_rects):
                continue
            if any(_overlap((m["start"], m["end"]), t) and _rects_hit(rect, r) for t, r in taken):
                continue
            chosen = (preset, rect)
            break
        if chosen is None:  # every corner is busy: take the one furthest from the avatar rects
            best = None
            for preset in ("Top-Right", "Top-Left", "Bottom-Right", "Bottom-Left"):
                rect = ve.compute_pip_rect((W, H), size, preset, None, m.get("scale", scale))
                hits = sum(_rects_hit(rect, tuple(r)) for r in avoid_rects)
                if best is None or hits < best[0]:
                    best = (hits, preset, rect)
            chosen = (best[1], best[2])
        m["position"], m["rect"] = chosen
        taken.append(((m["start"], m["end"]), m["rect"]))


def plan_for_video(segments: Sequence[dict], settings, frame_size: Tuple[int, int], total_seconds: float,
                   avoid_rects: Sequence[Tuple[int, int, int, int]] = (), llm=None, clamp: bool = False,
                   ffmpeg_path: Optional[str] = None, record: bool = True) -> dict:
    """The one call the pipelines make: open the library named in `settings` (a MemeSettings or its dict), plan the
    reactions for `segments` ({"start","end","text"}), give overlays a free corner of `frame_size`, and remember which
    clips were used. Returns plan_memes()'s dict plus "library": {"folder","clips"}."""
    cfg = settings if isinstance(settings, MemeSettings) else MemeSettings.from_dict(settings)
    lib = MemeLibrary(cfg.library_dir, ffmpeg_path=ffmpeg_path).scan()
    plan = plan_memes(segments, lib, cfg, total_seconds, avoid_rects=avoid_rects, frame_size=frame_size, llm=llm, clamp=clamp)
    if record and plan["matches"]:
        lib.record_usage(plan["matches"])
    plan["library"] = {"folder": str(lib.dir), "clips": len(lib.clips)}
    return plan


# --------------------------------------------------------------------------- #
# starter pack: original, tiny animated stand-ins so the feature works out of the box
# --------------------------------------------------------------------------- #
_STARTER = {  # emotion -> (clip name, caption)
    "laugh": ("starter_laugh", "LOL"), "shock": ("starter_shock", "OMG!"), "sad": ("starter_sad", "sad..."),
    "angry": ("starter_angry", "GRR"), "cringe": ("starter_cringe", "yikes"), "confused": ("starter_confused", "huh?"),
    "win": ("starter_win", "YES!"), "love": ("starter_love", "aww"), "fear": ("starter_fear", "eek"), "cool": ("starter_cool", "cool")}


def _draw_reaction_frame(emotion: str, caption: str, t: float, size: int = 320):
    import cv2
    import numpy as np

    img = np.zeros((size, size, 3), np.uint8)
    bg = {"laugh": (60, 170, 230), "shock": (200, 120, 60), "sad": (150, 90, 60), "angry": (40, 40, 170), "cringe": (90, 150, 110),
          "confused": (140, 110, 160), "win": (60, 190, 120), "love": (150, 90, 210), "fear": (110, 110, 70), "cool": (200, 170, 70)}[emotion]
    img[:] = bg
    cx, cy, r = size // 2, int(size * 0.48), int(size * 0.30)
    wob = math.sin(t * 14) * (6 if emotion in ("laugh", "fear", "angry") else 2)
    cx = int(cx + wob)
    face = (60, 210, 250) if emotion != "angry" else (80, 90, 230)
    if emotion == "fear":
        face = (215, 235, 190)
    cv2.circle(img, (cx, cy), r, face, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (30, 60, 90), 3, cv2.LINE_AA)
    ex, ey = int(r * 0.42), int(r * 0.25)
    eye_r = int(r * (0.16 + (0.10 * abs(math.sin(t * 6)) if emotion == "shock" else 0)))
    for s in (-1, 1):
        if emotion == "cool":
            continue
        if emotion == "laugh":
            cv2.ellipse(img, (cx + s * ex, cy - ey), (eye_r, eye_r), 0, 200, 340, (30, 30, 30), 4, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx + s * ex, cy - ey), eye_r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cx + s * ex + (int(3 * math.sin(t * 5)) if emotion == "confused" else 0), cy - ey), max(2, eye_r // 2), (20, 20, 20), -1, cv2.LINE_AA)
        if emotion in ("angry", "cringe"):
            cv2.line(img, (cx + s * ex - 14, cy - ey - 22 + s * 8), (cx + s * ex + 14, cy - ey - 22 - s * 8), (20, 20, 20), 5, cv2.LINE_AA)
    if emotion == "cool":
        cv2.rectangle(img, (cx - r + 12, cy - ey - 16), (cx + r - 12, cy - ey + 14), (15, 15, 15), -1)
        cv2.line(img, (cx - r + 12, cy - ey - 8), (cx + r - 12, cy - ey - 8), (60, 60, 60), 2)
    my = cy + int(r * 0.45)
    mw = int(r * 0.55)
    if emotion in ("laugh", "win", "love"):
        cv2.ellipse(img, (cx, my - 8), (mw, int(r * (0.30 + 0.08 * abs(math.sin(t * 10))))), 0, 0, 180, (30, 30, 120), -1, cv2.LINE_AA)
    elif emotion == "shock":
        cv2.ellipse(img, (cx, my), (int(r * 0.2), int(r * (0.22 + 0.06 * math.sin(t * 8)))), 0, 0, 360, (30, 30, 90), -1, cv2.LINE_AA)
    elif emotion in ("sad", "fear"):
        cv2.ellipse(img, (cx, my + 14), (mw // 2 + 4, 12), 0, 180, 360, (30, 30, 90), 4, cv2.LINE_AA)
    elif emotion == "cringe":
        pts = np.array([[cx - mw + k * (mw // 2), my + (8 if k % 2 else -8)] for k in range(5)], np.int32)
        cv2.polylines(img, [pts], False, (30, 30, 90), 4, cv2.LINE_AA)
    elif emotion == "angry":
        cv2.line(img, (cx - mw // 2, my + 6), (cx + mw // 2, my + 6), (30, 30, 90), 6, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx, my), (mw // 2, 10), 0, 0, 180, (30, 30, 90), 4, cv2.LINE_AA)
    # one animated prop per feeling
    k = (t * 1.2) % 1.0
    if emotion == "sad":
        cv2.ellipse(img, (cx - ex, cy - ey + 24 + int(k * 80)), (6, 10), 0, 0, 360, (255, 200, 120), -1, cv2.LINE_AA)
    if emotion == "fear":
        cv2.ellipse(img, (cx + r - 6, cy - r // 2 + int(k * 40)), (7, 11), 0, 0, 360, (255, 220, 140), -1, cv2.LINE_AA)
    if emotion == "love":
        for j in range(3):
            hy = int(size * 0.9 - ((k + j / 3.0) % 1.0) * size * 0.8)
            cv2.putText(img, "<3", (int(size * (0.2 + 0.3 * j)), hy), cv2.FONT_HERSHEY_DUPLEX, 1.0, (120, 80, 255), 2, cv2.LINE_AA)
    if emotion == "confused":
        cv2.putText(img, "?", (cx + r - 4, int(cy - r * 0.6 - 10 * math.sin(t * 5))), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    if emotion == "win":
        rng = np.random.default_rng(3)
        for j in range(18):
            px, py0 = int(rng.integers(0, size)), int(rng.integers(0, size))
            py = int((py0 + t * 160 * (0.6 + (j % 3) * 0.3)) % size)
            cv2.circle(img, (px, py), 5, [(40, 80, 240), (250, 220, 60), (255, 255, 255)][j % 3], -1, cv2.LINE_AA)
    if emotion == "angry":
        cv2.putText(img, "#@!", (cx + r - 30, cy - r + 10 + int(4 * math.sin(t * 20))), cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 120), 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)
    cv2.putText(img, caption, ((size - tw) // 2, size - 22), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def ensure_starter_pack(folder: Optional[str] = None, ffmpeg_path: Optional[str] = None, fps: int = 12, seconds: float = 1.8) -> List[str]:
    """Draw ten original silent reaction clips (one per emotion) into storage/memes/<emotion>/. Existing clips are kept;
    returns the files created. They are simple stand-ins: replace them with your own favourite reaction videos."""
    try:
        from . import video_engine as ve
    except ImportError:  # pragma: no cover
        import video_engine as ve
    ff = ve.find_ffmpeg(ffmpeg_path)
    root = Path(folder) if folder else default_meme_dir()
    made: List[str] = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    n = max(2, int(fps * seconds))
    for emo, (name, caption) in _STARTER.items():
        target = root / emo / f"{name}.mp4"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "320x320",
                                 "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        try:
            for k in range(n):
                proc.stdin.write(_draw_reaction_frame(emo, caption, k / fps).tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            code = proc.wait(timeout=60)
        except Exception:
            proc.kill()
            target.unlink(missing_ok=True)
            raise
        finally:
            if proc.stderr:
                proc.stderr.close()
        if code != 0 or not target.is_file():
            target.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg could not write the starter clip '{name}': {err[-300:]}")
        made.append(str(target))
    return made
