"""
novel_adapter.py - the Novel2Video engine (literary adaptation, ViMax's novel-compression workflow) on a CPU.

    long text / .txt file
      -> chapters                      (heading detection, with a size-based fallback)
      -> beats                         (TextTiling over sentence-window EMBEDDINGS: a new beat starts where the
                                        meaning drifts)
      -> character profiles            (names + aliases, descriptions, gender, relationships, a visual anchor)
      -> episodes of scene scripts     (extractive compression to a time budget, written in screenplay format so
                                        script_parser.py / the orchestrator can turn them into shots)

Embeddings: `sentence-transformers` `all-MiniLM-L6-v2` is used when it is installed AND its model can be loaded
(it runs fine on a CPU, ~90 MB). Otherwise a deterministic hashed bag-of-words embedder is used - no download, no
GPU, no extra package - which finds topic shifts well but does not know synonyms. Both are loaded lazily, only
when a novel is actually adapted.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MAX_CHARS = 3_000_000  # ~500k words; more than that is refused with a clear message
STOPWORDS = set("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into
is it its itself just me more most my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some
such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves said asked replied""".split())
NOT_NAMES = set("""The A An And But Or So Then There Here This That These Those It He She They We You I His Her Their Our My Your If When While As
After Before Because Although Though Yet Still Now Once Chapter Part Book Prologue Epilogue Mr Mrs Ms Miss Dr Sir Lady Lord Monday Tuesday Wednesday
Thursday Friday Saturday Sunday January February March April May June July August September October November December God Oh Yes No What Why How
Who Where Which Perhaps Maybe Suddenly Finally Meanwhile However Nothing Everything Something Someone Everyone Nobody Not Never Always Even Just
Only Also Too Very One Two Three Let Come Go Well Please Thank Thanks""".split())
HONORIFICS = ("Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Lady", "Lord", "Captain", "Professor", "Father", "Mother", "Uncle", "Aunt", "Madam")
SPEECH_VERBS = r"(?:said|asked|replied|whispered|shouted|cried|answered|muttered|murmured|exclaimed|called|added|continued|snapped|sighed|laughed|yelled|demanded|suggested|warned|insisted)"
PLACES = {"kitchen": "INT", "bedroom": "INT", "room": "INT", "hall": "INT", "hallway": "INT", "library": "INT", "office": "INT",
          "house": "INT", "cottage": "INT", "castle": "INT", "church": "INT", "shop": "INT", "store": "INT", "cellar": "INT",
          "attic": "INT", "parlour": "INT", "parlor": "INT", "study": "INT", "tavern": "INT", "inn": "INT", "hospital": "INT",
          "school": "INT", "classroom": "INT", "cabin": "INT", "ship": "INT", "carriage": "INT", "train": "INT", "car": "INT",
          "garden": "EXT", "street": "EXT", "road": "EXT", "forest": "EXT", "woods": "EXT", "field": "EXT", "beach": "EXT",
          "shore": "EXT", "village": "EXT", "town": "EXT", "city": "EXT", "square": "EXT", "yard": "EXT", "hill": "EXT",
          "mountain": "EXT", "river": "EXT", "bridge": "EXT", "station": "EXT", "harbor": "EXT", "harbour": "EXT",
          "park": "EXT", "farm": "EXT", "path": "EXT", "lake": "EXT", "desert": "EXT", "rooftop": "EXT", "courtyard": "EXT", "lighthouse": "INT", "tower": "INT", "gallery": "INT", "stairs": "INT", "cellar": "INT", "cliff": "EXT", "sea": "EXT", "cliffs": "EXT", "island": "EXT", "market": "EXT", "inn": "INT", "church": "INT", "dock": "EXT", "quay": "EXT"}
TOD = (("midnight", "MIDNIGHT"), ("night", "NIGHT"), ("evening", "EVENING"), ("dusk", "DUSK"), ("twilight", "DUSK"),
       ("sunset", "SUNSET"), ("dawn", "DAWN"), ("sunrise", "SUNRISE"), ("morning", "MORNING"), ("afternoon", "AFTERNOON"),
       ("noon", "NOON"), ("daylight", "DAY"))
COLORS = ("fair", "black", "brown", "blonde", "blond", "red", "auburn", "grey", "gray", "white", "silver", "dark", "golden", "chestnut", "ginger")
EYE_COLORS = ("blue", "green", "brown", "grey", "gray", "hazel", "amber", "black", "dark")


class NovelError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Deterministic CPU embedder: signed feature hashing of unigrams + bigrams (stop-words removed), sub-linear
    term frequency, L2-normalised. No model, no download."""

    name = "hashed-bow"
    dim = 384

    def encode(self, texts: Sequence[str]):
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = [t for t in re.findall(r"[a-z][a-z']+", str(text).casefold()) if t not in STOPWORDS and len(t) > 2]
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            counts = Counter(feats)
            for f, c in counts.items():
                h = zlib.crc32(f.encode("utf-8"))
                out[r, h % self.dim] += (1.0 if (h >> 16) & 1 else -1.0) * (1.0 + math.log(c))
            n = float(np.linalg.norm(out[r]))
            if n > 0:
                out[r] /= n
        return out


class _MiniLM:
    name = "all-MiniLM-L6-v2"

    def __init__(self, model):
        self.model = model

    def encode(self, texts: Sequence[str]):
        import numpy as np

        v = self.model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)


def get_embedder(prefer: str = "auto", warnings: Optional[List[str]] = None):
    """prefer: "auto" (MiniLM if it loads, else hashed), "minilm" (same, but says so when it falls back) or "hashed"."""
    if prefer not in ("auto", "minilm", "hashed"):
        raise ValueError("prefer must be 'auto', 'minilm' or 'hashed'")
    if prefer != "hashed":
        if importlib.util.find_spec("sentence_transformers") is None:
            reason = "sentence-transformers is not installed"
        else:
            try:
                from sentence_transformers import SentenceTransformer

                return _MiniLM(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"))
            except Exception as exc:  # noqa: BLE001 - offline / model missing / broken install
                reason = f"all-MiniLM-L6-v2 could not be loaded ({str(exc)[:120]})"
        if warnings is not None and prefer == "minilm":
            warnings.append(f"{reason}; using the built-in hashed embedder instead.")
        logger.info("embedder fallback: %s", reason)
    return HashingEmbedder()


# --------------------------------------------------------------------------- #
# reading + splitting
# --------------------------------------------------------------------------- #
def read_text_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in (".txt", ".md", ".text", ".fountain", ""):
        raise NovelError(f"Unsupported file type '{p.suffix}'. Save the novel as plain text (.txt) first.")
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", "replace")


_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Sr|Jr|Prof|Capt|Col|Gen|Lt|Sgt|vs|etc|No)\.$")


def _balanced(chunk: str) -> bool:
    """True when no double quote is left open in `chunk` (so a full stop here really ends a sentence)."""
    return chunk.count('"') % 2 == 0 and chunk.count("\u201c") <= chunk.count("\u201d")


def split_sentences(text: str) -> List[str]:
    """Sentence splitter that never cuts inside a quotation and knows common abbreviations."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    out, start = [], 0
    for m in re.finditer(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*\s+(?=[\"'\u201c\u2018(\[]?[A-Z0-9])", text):
        if _ABBREV.search(text[start:m.start() + 1]) or not _balanced(text[start:m.end()]):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


_NUMWORD = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|[ivxlcdm]+|\d+)"
_CHAPTER_RE = re.compile(rf"^\s*(?:#{{1,3}}\s*)?(?:chapter|book|part|section)\s+{_NUMWORD}\b[^\n]{{0,80}}$|^\s*(?:prologue|epilogue|preface|introduction)\b[^\n]{{0,60}}$|^\s*#{{1,2}}\s+\S[^\n]{{0,80}}$",
                         re.IGNORECASE | re.MULTILINE)
_SCENE_BREAK = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*[-=~#]{3,}\s*$", re.MULTILINE)


def split_chapters(text: str, fallback_words: int = 2500) -> List[dict]:
    """-> [{"index","title","text","word_count"}]. Falls back to ~fallback_words chunks (at paragraph edges) when the
    text has no recognisable chapter headings."""
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise NovelError("The text is empty.")
    if len(text) > MAX_CHARS:
        raise NovelError(f"The text is too long ({len(text):,} characters; the limit is {MAX_CHARS:,}). Split it into volumes.")
    marks = [m for m in _CHAPTER_RE.finditer(text)]
    chapters: List[dict] = []
    if len(marks) >= 2:
        opening = text[:marks[0].start()].strip()
        if len(opening.split()) >= 40:  # a few words before chapter 1 is just the book's title
            chapters.append({"title": "Opening", "text": opening})
        for k, m in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()
            if body:
                chapters.append({"title": m.group(0).strip(" #\t\n"), "text": body})
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        cur, words = [], 0
        for p in paras:
            cur.append(p)
            words += len(p.split())
            if words >= fallback_words:
                chapters.append({"title": f"Section {len(chapters) + 1}", "text": "\n\n".join(cur)})
                cur, words = [], 0
        if cur:
            if chapters and words < fallback_words * 0.25:
                chapters[-1]["text"] += "\n\n" + "\n\n".join(cur)
            else:
                chapters.append({"title": f"Section {len(chapters) + 1}", "text": "\n\n".join(cur)})
    for i, c in enumerate(chapters):
        c["index"] = i + 1
        c["word_count"] = len(c["text"].split())
    return chapters


# --------------------------------------------------------------------------- #
# beats (TextTiling over embeddings)
# --------------------------------------------------------------------------- #
def split_beats(chapter_text: str, embedder, window: int = 4, min_words: int = 90, max_words: int = 420,
                sensitivity: float = 0.6) -> List[dict]:
    """Group sentences into story beats. Embeds a sliding window of sentences, finds the places where the similarity
    between the window before and the window after dips (a topic / scene shift), and cuts there, keeping every beat
    between min_words and max_words. Hard scene breaks ("* * *") are always cuts."""
    import numpy as np

    blocks = [b for b in _SCENE_BREAK.split(chapter_text) if b.strip()]
    beats: List[dict] = []
    for block in blocks:
        sents = split_sentences(block)
        if not sents:
            continue
        wc = [len(s.split()) for s in sents]
        cuts: List[int] = []
        if len(sents) >= 2 * window:
            wins = [" ".join(sents[i:i + window]) for i in range(len(sents) - window + 1)]
            emb = embedder.encode(wins)
            sims = np.array([float(emb[i] @ emb[i + window]) for i in range(len(wins) - window)]) if len(wins) > window else np.array([])
            if sims.size:
                thr = sims.mean() - sensitivity * (sims.std() or 1e-9)
                depth = []
                for i in range(len(sims)):
                    left = sims[max(0, i - 3):i + 1].max()
                    right = sims[i:i + 4].max()
                    depth.append((left - sims[i]) + (right - sims[i]))
                cand = [i + window for i in range(len(sims)) if sims[i] < thr and sims[i] <= sims[max(0, i - 1):i + 2].min()]
                cuts = sorted(cand, key=lambda c: -depth[c - window])
        bounds = [0]
        for c in sorted(cuts):
            if sum(wc[bounds[-1]:c]) >= min_words and sum(wc[c:]) >= min_words:
                bounds.append(c)
        bounds.append(len(sents))
        # enforce the maximum length by cutting at the midpoint of over-long beats
        final: List[int] = [bounds[0]]
        for a, b in zip(bounds, bounds[1:]):
            span = sum(wc[a:b])
            if span > max_words:
                pieces = int(math.ceil(span / max_words))
                acc, target = 0, span / pieces
                for k in range(a, b):
                    acc += wc[k]
                    if acc >= target and k + 1 < b and sum(wc[k + 1:b]) >= min_words // 2:
                        final.append(k + 1)
                        acc = 0
            final.append(b)
        final = sorted(set(final))
        for a, b in zip(final, final[1:]):
            part = sents[a:b]
            if part:
                beats.append({"sentences": part, "text": " ".join(part), "word_count": sum(wc[a:b])})
    for i, b in enumerate(beats):
        b["index"] = i + 1
    return beats


# --------------------------------------------------------------------------- #
# characters
# --------------------------------------------------------------------------- #
def _candidate_names(text: str) -> Counter:
    counts: Counter = Counter()
    for m in re.finditer(rf"\b((?:{'|'.join(HONORIFICS)})\.?[ \t]+)?([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:[ \t]+[A-Z][a-z]+){{0,2}})\b", text):
        hon, name = m.group(1), m.group(2)
        words = name.split()
        while words and words[0] in NOT_NAMES:
            words = words[1:]
        if not words or len(words[0]) < 3 and not hon:
            continue
        full = ((hon.strip().rstrip(".") + " ") if hon else "") + " ".join(words)
        at_start = m.start() == 0 or re.search(r"[.!?\"“”]\s*$", text[max(0, m.start() - 3):m.start()]) is not None
        counts[(full, at_start)] += 1
    names: Counter = Counter()
    for (full, at_start), c in counts.items():
        names[full] += c
    for (full, at_start) in list(counts):  # a capitalised word that ONLY ever opens sentences is probably not a name
        if at_start and counts.get((full, False), 0) == 0 and " " not in full:
            names.pop(full, None)
    return names


_NUMBER_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def _age_in(sentence: str) -> Optional[int]:
    m = re.search(r"\b(\d{2})[- ]year[- ]old\b|\baged (\d{2})\b", sentence)
    if m:
        return int(m.group(1) or m.group(2))
    m = re.search(r"\b(?:aged|of|perhaps|about|around)\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty)(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\b", sentence, re.IGNORECASE)
    if m:
        return _NUMBER_WORDS[m.group(1).lower()] + _UNITS.get((m.group(2) or "").lower(), 0)
    return None


def short_name(name: str) -> str:
    """The name used on screen: 'Elena Marsh' -> 'Elena', 'Mr Darcy' -> 'Darcy'."""
    parts = name.split()
    if parts and parts[0] in HONORIFICS and len(parts) > 1:
        return " ".join(parts[1:]) if len(parts) == 2 else parts[1]
    return parts[0] if parts else name


def extract_characters(text: str, top_k: int = 8, min_mentions: int = 3) -> List[dict]:
    """Primary characters with aliases, mentions, gender guess, a description assembled from the text around the name,
    top co-occurring characters and a deterministic visual anchor. Pure Python - no model."""
    names = _candidate_names(text)
    if not names:
        return []
    # merge aliases: "Darcy" -> "Mr Darcy", "Elena" -> "Elena Marsh" (a single name that is the first/last word of a longer one)
    canon: Dict[str, str] = {}
    longer = sorted((n for n in names if len(n.split()) > 1), key=lambda n: -names[n])
    for n in sorted(names, key=lambda n: -names[n]):
        for other in longer:
            if other != n and len(other.split()) > len(n.split()) and n in other.split():
                canon[n] = other
                break
    merged: Counter = Counter()
    aliases: Dict[str, set] = defaultdict(set)
    for n, c in names.items():
        target = canon.get(n, n)
        merged[target] += c
        if target != n:
            aliases[target].add(n)
    speakers = Counter()
    for m in re.finditer(rf"[\"\u201d]\s*,?\s*(?:{SPEECH_VERBS}\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+{SPEECH_VERBS}\b|\b([A-Z][a-z]+)\s+{SPEECH_VERBS}\s*[,:]?\s*[\"\u201c]|{SPEECH_VERBS}\s+([A-Z][a-z]+)\b", text):
        nm = m.group(1) or m.group(2) or m.group(3)
        if nm:
            speakers[canon.get(nm, nm)] += 1
    def place_like(n: str) -> bool:  # "at Port Wren", "to Gull Point": mostly used after a preposition = a place
        forms = [n] + sorted(aliases.get(n, set()))
        hits = sum(len(re.findall(rf"\b(?:at|to|in|from|near|toward|towards|across|through|reached|reach|leave|left)\s+{re.escape(f)}\b", text)) for f in forms)
        return speakers.get(n, 0) == 0 and hits >= max(2, 0.5 * merged[n])
    ranked = sorted(merged, key=lambda n: -(merged[n] + 3 * speakers.get(n, 0)))
    ranked = [n for n in ranked if merged[n] >= min_mentions and not place_like(n)][:top_k]
    sents = split_sentences(text)
    profiles = []
    for n in ranked:
        forms = {n} | aliases.get(n, set())
        forms.add(short_name(n))
        rx = re.compile(r"\b(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")\b")
        mine = [(i, s) for i, s in enumerate(sents) if rx.search(s)]
        he = she = 0
        for i, s in mine[:200]:
            window = " ".join(sents[i:i + 2]).casefold()
            he += len(re.findall(r"\b(?:he|his|him|himself)\b", window))
            she += len(re.findall(r"\b(?:she|her|hers|herself)\b", window))
        if re.match(r"^(?:Mr|Sir|Lord|Father|Uncle|Captain)\b", n):
            gender = "male"
        elif re.match(r"^(?:Mrs|Ms|Miss|Lady|Mother|Aunt|Madam)\b", n):
            gender = "female"
        else:
            gender = "male" if he > she * 1.3 else "female" if she > he * 1.3 else "unknown"
        desc: List[str] = []
        traits: List[str] = []
        age = None
        for i, s in mine:
            # the sentence that names the character, plus a following sentence that continues with He / She
            window = [s] + [t for t in sents[i + 1:i + 3] if re.match(r"^(?:He|She|His|Her)\b", t)][:1]
            for w in window:
                for m in re.finditer(rf"{rx.pattern},\s+((?:a|an|the)\s[^,.;:]{{3,80}})", w):
                    desc.append(m.group(1))
                for m in re.finditer(rf"(?:{rx.pattern}|\b(?:He|She))\s+(?:was|is|looked|seemed)\s+((?:a|an|the|very|rather|quite)\s[^.;:]{{3,100}})", w):
                    desc.append(re.split(r",?\s+(?:and|but|who|while)\s", m.group(1))[0])
                for m in re.finditer(r"\b(\w+(?:-\w+)?)\s+(hair|eyes|beard|skin)\b", w):
                    if m.group(1).casefold() in COLORS + EYE_COLORS + ("long", "short", "curly", "pale", "dark", "bright", "kind", "sharp", "tired"):
                        traits.append(f"{m.group(1).lower()} {m.group(2)}")
                if age is None:
                    age = _age_in(w)
        counter = Counter()
        for _, s in mine:
            for other in ranked:
                if other != n and re.search(rf"\b{re.escape(short_name(other))}\b", s):
                    counter[other] += 1
        desc_unique = list(dict.fromkeys(d.strip() for d in desc))[:3]
        trait_unique = list(dict.fromkeys(traits))[:5]
        profile = {"name": n, "short": short_name(n), "aliases": sorted(forms - {n}), "mentions": merged[n],
                   "speaking_lines": speakers.get(n, 0), "gender": gender, "age": age,
                   "description": "; ".join(desc_unique)[:240], "traits": trait_unique,
                   "relationships": [o for o, _ in counter.most_common(3)]}
        profile["visual_anchor"] = visual_anchor(profile)
        profiles.append(profile)
    return profiles


_SKIN = ("fair", "light olive", "tan", "warm brown", "deep brown", "pale")
_HAIR = ("black", "dark brown", "chestnut", "auburn", "blonde", "grey", "silver", "red")
_OUTFIT = ("navy coat", "olive jacket", "burgundy dress", "charcoal suit", "mustard sweater", "forest-green cloak",
           "white shirt and brown vest", "grey hoodie", "teal blouse", "black leather jacket")


def _article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def visual_anchor(profile: dict) -> dict:
    """A stable visual identity for the character: any colours the text gave (hair, eyes), otherwise values picked
    deterministically from the name, so the same character always gets the same look in every shot."""
    h = zlib.crc32(profile["name"].encode("utf-8"))
    hair = next((t.split()[0] for t in profile.get("traits", []) if t.endswith("hair") and t.split()[0] in COLORS), None)
    eyes = next((t.split()[0] for t in profile.get("traits", []) if t.endswith("eyes") and t.split()[0] in EYE_COLORS), None)
    gender = profile.get("gender", "unknown")
    who = {"male": "man", "female": "woman"}.get(gender, "person")
    age = profile.get("age")
    hair = hair or _HAIR[h % len(_HAIR)]
    skin = _SKIN[(h >> 3) % len(_SKIN)]
    outfit = _OUTFIT[(h >> 6) % len(_OUTFIT)]
    ident = (f"{'a ' + str(age) + '-year-old ' + who if age else _article(who)} with {hair} hair"
             + (f" and {eyes} eyes" if eyes else "") + f", {skin} skin, wearing {_article(outfit)}")
    return {"hair": hair, "eyes": eyes, "skin": skin, "outfit": outfit, "identifier": ident, "seed": h % 100000}


# --------------------------------------------------------------------------- #
# compression + episodes
# --------------------------------------------------------------------------- #
def _location_of(text: str, previous: Tuple[str, str]) -> Tuple[str, str]:
    low = text.casefold()
    hits = Counter()
    for m in re.finditer(r"\b(?:in|at|into|inside|through|across|on|to|from|outside)\s+(?:the|a|an|her|his|their|my)\s+(?:[a-z]+\s+)?([a-z]+)\b", low):
        if m.group(1) in PLACES:
            hits[m.group(1)] += 1
    for w in re.findall(r"[a-z]+", low):
        if w in PLACES:
            hits[w] += 0.3
    if not hits:
        return previous
    place = hits.most_common(1)[0][0]
    return PLACES[place], place.upper()


def _tod_of(text: str, previous: str) -> str:
    low = text.casefold()
    for word, label in TOD:
        if re.search(rf"\b{word}\b", low):
            return label
    return previous


def _dialogue_split(sentence: str, known: Sequence[str]):
    """'"Come here," said Mara softly.' -> ("Come here.", "Mara", "softly", narration); None when there is no quote.
    All quotations in the sentence are joined. speaker is None when only a pronoun / description was used."""
    quotes = re.findall(r"[\"\u201c]([^\"\u201d]{1,400})[\"\u201d]", sentence)
    if not quotes:
        return None
    quote = " ".join(q.strip() for q in quotes).strip()
    quote = re.sub(r"[,;:]\s*$", ".", quote)
    rest = re.sub(r"[\"\u201c][^\"\u201d]*[\"\u201d]", " ", sentence)
    speaker, manner = None, ""
    sm = re.search(rf"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+{SPEECH_VERBS}\b(?:\s+(\w+ly))?|\b{SPEECH_VERBS}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s+(\w+ly))?", rest)
    if sm:
        speaker = sm.group(1) or sm.group(3)
        manner = sm.group(2) or sm.group(4) or ""
    if speaker:
        speaker = next((k for k in known if speaker == k or speaker in k.split()), speaker)
        if speaker in NOT_NAMES or (speaker not in known and speaker.split()[0] in NOT_NAMES):
            speaker = None
    # what is left of the sentence once the quotes and the "X said" tag are removed is narration
    narration = re.sub(rf"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+{SPEECH_VERBS}\b(?:\s+\w+ly)?|\b(?:he|she|they|the\s+\w+(?:\s+\w+)?)\s+{SPEECH_VERBS}\b(?:\s+\w+ly)?|\b{SPEECH_VERBS}\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?|the\s+\w+(?:\s+\w+)?|he|she)(?:\s+\w+ly)?",
                       " ", rest, flags=re.IGNORECASE)
    narration = re.sub(r"\s+", " ", narration).strip(" ,;.:-")
    narration = re.sub(r"^(?:and|but|then|as|while)\s+", "", narration, flags=re.IGNORECASE)
    return quote, speaker, manner, narration


def _trim(sentence: str, max_words: int = 30) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    cut = " ".join(words[:max_words])
    m = re.search(r"^(.*[,;:])\s", cut)
    return (m.group(1)[:-1] if m and len(m.group(1).split()) > 8 else cut).rstrip(" ,;:") + "."


def compress_beat(beat: dict, embedder, names: Sequence[str], budget_words: int) -> List[str]:
    """Extractive compression: keep the sentences that best represent the beat (closest to its centroid), plus
    dialogue and sentences that name a main character, in the original order, within a word budget."""
    import numpy as np

    sents = beat["sentences"]
    if sum(len(s.split()) for s in sents) <= budget_words:
        return list(sents)
    emb = embedder.encode(sents)
    cen = emb.mean(axis=0)
    cen /= (np.linalg.norm(cen) or 1.0)
    scores = []
    for i, s in enumerate(sents):
        sc = float(emb[i] @ cen)
        if re.search(r"[\"“]", s):
            sc += 0.25
        if any(re.search(rf"\b{re.escape(n.split()[-1])}\b", s) for n in names):
            sc += 0.2
        if i == 0:
            sc += 0.3
        if i == len(sents) - 1:
            sc += 0.1
        scores.append(sc)
    chosen, used = [], 0
    for i in sorted(range(len(sents)), key=lambda k: -scores[k]):
        w = min(len(sents[i].split()), 30)
        if used + w > budget_words and chosen:
            continue
        chosen.append(i)
        used += w
    return [sents[i] for i in sorted(chosen)]


def _gender_hint(sentence: str) -> Optional[str]:
    """'said the young man' / 'she replied' -> "male" / "female" (None when the tag gives no hint)."""
    m = re.search(rf"{SPEECH_VERBS}\s+the\s+(?:\w+\s+){{0,2}}(man|boy|gentleman|fisherman|farmer|king|prince|father|brother|woman|girl|lady|queen|princess|mother|sister)\b", sentence, re.IGNORECASE)
    if m:
        return "male" if m.group(1).lower() in ("man", "boy", "gentleman", "fisherman", "farmer", "king", "prince", "father", "brother") else "female"
    m = re.search(rf"\b(he|she)\s+{SPEECH_VERBS}\b", sentence, re.IGNORECASE)
    if m:
        return "male" if m.group(1).lower() == "he" else "female"
    return None


def beat_to_scene(sentences: Sequence[str], profiles: Sequence[dict], previous: Tuple[str, str, str],
                  state: Optional[dict] = None) -> Tuple[str, Tuple[str, str, str]]:
    """Compressed beat -> screenplay lines. A scene heading is written only when the place or time of day changed
    (or at the start of an episode, state["first"]). state carries the last speaker and introduced characters between
    beats so speakers keep alternating and a name is capitalised only the first time it appears in an episode."""
    state = state if state is not None else {}
    text = " ".join(sentences)
    names = [p["name"] for p in profiles]
    shorts = {p["name"]: p.get("short") or short_name(p["name"]) for p in profiles}
    gender_of = {p["name"]: p.get("gender") for p in profiles}
    int_ext, place = _location_of(text, (previous[0], previous[1]))
    tod = _tod_of(text, previous[2] or "DAY")
    out: List[str] = []
    if state.get("first", True) or (int_ext, place, tod) != tuple(previous):
        out += [f"{int_ext}. {place} - {tod}", ""]
    state["first"] = False
    intro: set = state.setdefault("intro", set())
    recent: List[str] = state.setdefault("recent", [])
    last = state.get("last_speaker")
    cont = False  # the previous sentence was a quote with attribution, so a bare quote continues it

    def name_forms(p: dict) -> List[str]:
        return sorted({p["name"], shorts[p["name"]], *p.get("aliases", [])}, key=len, reverse=True)

    def mention(line: str) -> str:
        for p in profiles:
            forms = name_forms(p)
            found = next((f for f in forms if re.search(rf"\b{re.escape(f)}\b", line)), None)
            if not found:
                continue
            if p["name"] not in intro:
                line = re.sub(rf"\b{re.escape(found)}\b(?:\s+(?:{'|'.join(re.escape(f.split()[-1]) for f in forms if ' ' in f) or 'ZZZ'}))?",
                              shorts[p["name"]].upper(), line, count=1)
                intro.add(p["name"])
            if p["name"] in recent:
                recent.remove(p["name"])
            recent.insert(0, p["name"])
        return line

    for s in sentences:
        d = _dialogue_split(s, names)
        if d:
            quote, speaker, manner, narration = d
            if not speaker:
                hint = _gender_hint(s)
                pool = [r for r in recent if r != last] + [n for n in names if n not in recent and n != last] + [r for r in recent if r == last]
                by_gender = [n for n in pool if hint and gender_of.get(n) == hint]
                if cont and last and not hint:
                    speaker = last
                elif by_gender:
                    speaker = by_gender[0]
                else:
                    speaker = pool[0] if pool else (names[0] if names else "Narrator")
            speaker_full = next((n for n in names if speaker in (n, shorts[n])), speaker)
            if speaker_full in recent:
                recent.remove(speaker_full)
            recent.insert(0, speaker_full)
            if len(narration.split()) >= 4:
                narration = re.sub(r"^(?:he|she|they)\b", shorts.get(speaker_full, speaker), narration, flags=re.IGNORECASE)
                out += [mention(_trim(narration[0].upper() + narration[1:])).rstrip(".") + ".", ""]
            out.append(shorts.get(speaker_full, speaker).upper())
            if manner:
                out.append(f"({manner})")
            out += [quote, ""]
            last, cont = speaker_full, True
        else:
            out += [mention(_trim(s)), ""]
            cont = False
    state["last_speaker"] = last
    return "\n".join(out).rstrip() + "\n", (int_ext, place, tod)


def adapt_novel(text: str, episodes: int = 3, seconds_per_episode: float = 90.0, max_characters: int = 8,
                embedder: str = "auto", words_per_second: float = 2.6, progress=None) -> dict:
    """Novel text -> {"chapters","characters","episodes","embedder","warnings"}.

    Each episode has a screenplay `script` that fits roughly `seconds_per_episode` of screen time (about
    seconds * words_per_second words of action and dialogue) and the characters that appear in it."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    warnings: List[str] = []
    emb = get_embedder(embedder, warnings)
    chapters = split_chapters(text)
    if progress:
        progress(0.05)
    characters = extract_characters(str(text), top_k=max_characters)
    if not characters:
        warnings.append("No recurring character names were found; the script will use a narrator.")
    names = [c["name"] for c in characters]
    beats: List[dict] = []
    for ch in chapters:
        for b in split_beats(ch["text"], emb):
            b["chapter"] = ch["index"]
            beats.append(b)
        if progress:
            progress(0.05 + 0.5 * ch["index"] / len(chapters))
    if not beats:
        raise NovelError("No story text could be extracted.")
    episodes = min(episodes, len(beats))
    # split beats into consecutive, word-balanced episodes
    total = sum(b["word_count"] for b in beats)
    target = total / episodes
    groups: List[List[dict]] = [[] for _ in range(episodes)]
    acc, g = 0.0, 0
    for b in beats:
        if g < episodes - 1 and acc >= target * (g + 1) - b["word_count"] / 2:
            g += 1
        groups[g].append(b)
        acc += b["word_count"]
    groups = [x for x in groups if x]
    budget_total = int(seconds_per_episode * words_per_second)
    out_eps = []
    prev = ("INT", "ROOM", "DAY")
    for ei, grp in enumerate(groups, 1):
        weight = sum(b["word_count"] for b in grp)
        scenes, used_names, words = [], set(), 0
        state: dict = {"first": True}
        for b in grp:
            share = max(25, int(budget_total * b["word_count"] / max(weight, 1)))
            kept = compress_beat(b, emb, names, share)
            script, prev = beat_to_scene(kept, characters, prev, state)
            scenes.append({"beat": b["index"], "chapter": b["chapter"], "script": script})
            words += len(script.split())
            used_names |= {c["name"] for c in characters if re.search(rf"\b{re.escape(c['short'])}\b", script, re.IGNORECASE)}
        full = "\n".join(s["script"] for s in scenes)
        out_eps.append({"index": ei, "title": f"Episode {ei}", "chapters": sorted({b["chapter"] for b in grp}),
                        "scenes": scenes, "script": full, "word_count": words,
                        "est_seconds": round(words / words_per_second, 1), "characters": sorted(used_names)})
        if progress:
            progress(0.55 + 0.45 * ei / len(groups))
    return {"chapters": [{"index": c["index"], "title": c["title"], "word_count": c["word_count"]} for c in chapters],
            "characters": characters, "episodes": out_eps, "embedder": emb.name, "beats": len(beats), "warnings": warnings}
