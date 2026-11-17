"""
backend/scorer.py

Free, fully local, rule-based fallback scorer. No API keys, no network, and no
third-party packages (pure standard library: re, math).

Viral clip scoring
    find_viral_clips(words, ...) takes the word list produced by
    transcriber.transcribe_audio() and returns the best non-overlapping clip
    ranges, scored on: question marks, exclamations, emotional/curiosity
    keywords, numbers/statistics, direct address ("you"), pauses between
    sentences (dramatic beats), sentence length, and filler words.

"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

# =========================================================================== #
# Viral clip scoring
# =========================================================================== #
_EMOTION_PATTERNS = [
    # curiosity / hook
    r"secret\w*", r"truth", r"nobody", r"no one", r"never", r"always", r"mistake\w*",
    r"hidden", r"revealed?", r"exposed?", r"warning", r"stop", r"wait",
    r"here'?s why", r"what if", r"imagine", r"guess what", r"the real reason",
    # strong emotion / shock
    r"insane", r"crazy", r"shock\w*", r"unbelievable", r"incredible", r"amaz\w*",
    r"wow", r"terrifying", r"scary", r"fear\w*", r"hate\w*", r"love\w*",
    r"worst", r"best", r"biggest", r"huge", r"massive", r"genius", r"brilliant",
    r"disaster", r"hell", r"die[sd]?", r"kill\w*", r"destroy\w*", r"epic",
    # value / payoff
    r"free", r"hack\w*", r"trick\w*", r"easy", r"fast", r"instantly", r"proven",
    r"million\w*", r"billion\w*", r"rich", r"win\w*", r"changed? my life",
]
_EMOTION_RE = re.compile(r"\b(?:" + "|".join(_EMOTION_PATTERNS) + r")\b", re.I)
_NUMBER_RE = re.compile(r"\d|\b(?:percent|million|billion|thousand|hundred)\b", re.I)
_YOU_RE = re.compile(r"\b(?:you|your|you're|you'll|yours)\b", re.I)
_FILLER_RE = re.compile(r"\b(?:um+|uh+|er+|ah+|you know|i mean|sort of|kind of)\b", re.I)

_SENTENCE_END = (".", "?", "!", "…")


def _clean(word: str) -> str:
    return str(word).strip()


def build_sentences(words: List[dict], pause_threshold: float = 0.7) -> List[dict]:
    """Group word dicts ({"word","start","end"}) into sentences.

    A sentence ends at terminal punctuation (. ? ! ...) or when the silence
    before the next word is >= pause_threshold seconds.
    """
    sentences: List[dict] = []
    current: List[dict] = []

    def flush():
        if not current:
            return
        text = " ".join(_clean(w["word"]) for w in current).strip()
        sentences.append(
            {
                "text": text,
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "word_count": len(current),
            }
        )
        current.clear()

    for i, w in enumerate(words):
        if not _clean(w.get("word", "")):
            continue
        current.append(w)
        token = _clean(w["word"])
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (float(nxt["start"]) - float(w["end"])) if nxt else None
        if token.endswith(_SENTENCE_END) or nxt is None or (
            gap is not None and gap >= pause_threshold
        ):
            flush()

    # attach pauses
    for idx, s in enumerate(sentences):
        s["pause_before"] = (
            max(0.0, s["start"] - sentences[idx - 1]["end"]) if idx > 0 else 0.0
        )
    return sentences


def score_sentence(sentence: dict) -> dict:
    """Return {"score": float, "reasons": [str, ...]} for one sentence."""
    text = sentence["text"]
    score = 0.0
    reasons: List[str] = []

    if "?" in text:
        score += 2.0
        reasons.append("question")
    if "!" in text:
        score += 1.5
        reasons.append("exclamation")

    emo = len(_EMOTION_RE.findall(text))
    if emo:
        score += min(emo, 4) * 1.0
        reasons.append("emotional_keywords")

    if _NUMBER_RE.search(text):
        score += 1.0
        reasons.append("number_or_stat")

    you = len(_YOU_RE.findall(text))
    if you:
        score += min(you, 3) * 0.5
        reasons.append("direct_address")

    if sentence.get("pause_before", 0.0) >= 0.8:
        score += 1.0
        reasons.append("dramatic_pause")

    wc = sentence.get("word_count", len(text.split()))
    if 6 <= wc <= 20:
        score += 0.5
    elif wc < 3:
        score -= 1.0
        reasons.append("too_short")

    fillers = len(_FILLER_RE.findall(text))
    if fillers:
        score -= min(fillers, 4) * 0.5
        reasons.append("filler_words")

    return {"score": score, "reasons": reasons}


def find_viral_clips(
    words: List[dict],
    top_n: int = 5,
    min_len: float = 15.0,
    max_len: float = 60.0,
    pause_threshold: float = 0.7,
) -> List[dict]:
    """Return up to `top_n` non-overlapping clip ranges, best first:

        {"start": float, "end": float, "duration": float, "score": float,
         "text": str, "reasons": [str, ...]}
    """
    if top_n < 1:
        return []
    if min_len > max_len:
        raise ValueError("min_len must be <= max_len")

    sentences = build_sentences(words, pause_threshold)
    if not sentences:
        return []
    scored = [score_sentence(s) for s in sentences]

    candidates: List[dict] = []
    n = len(sentences)
    for i in range(n):
        total = 0.0
        reasons: List[str] = []
        for j in range(i, n):
            duration = sentences[j]["end"] - sentences[i]["start"]
            if duration > max_len:
                break
            total += scored[j]["score"] + (0.5 * scored[i]["score"] if j == i else 0.0)
            reasons.extend(scored[j]["reasons"])
            if duration >= min_len:
                # normalise to "points per 30 s" so long windows don't win by size
                density = total / math.sqrt(max(duration, 1.0) / 30.0 + 1e-9)
                candidates.append(
                    {
                        "start": sentences[i]["start"],
                        "end": sentences[j]["end"],
                        "duration": duration,
                        "score": density,
                        "text": " ".join(s["text"] for s in sentences[i : j + 1]),
                        "reasons": sorted(set(reasons)),
                    }
                )

    if not candidates:  # transcript shorter than min_len (or one huge sentence)
        start = sentences[0]["start"]
        end = min(sentences[-1]["end"], start + max_len)
        kept = [s for s in sentences if s["end"] <= end] or sentences[:1]
        allr = sorted({r for sc in scored[: len(kept)] for r in sc["reasons"]})
        return [
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "score": round(sum(sc["score"] for sc in scored[: len(kept)]), 2),
                "text": " ".join(s["text"] for s in kept),
                "reasons": allr,
            }
        ]

    candidates.sort(key=lambda c: c["score"], reverse=True)
    chosen: List[dict] = []
    for c in candidates:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
            chosen.append(c)
            if len(chosen) == top_n:
                break

    for c in chosen:
        c["start"] = round(c["start"], 2)
        c["end"] = round(c["end"], 2)
        c["duration"] = round(c["duration"], 2)
        c["score"] = round(c["score"], 2)
    return chosen
