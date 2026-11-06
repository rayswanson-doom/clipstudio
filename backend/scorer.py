"""
backend/scorer.py

Free, fully local, rule-based fallback scorer. No API keys, no network, and no
third-party packages (pure standard library: re, math).

Part 1 - Viral clip scoring
    find_viral_clips(words, ...) takes the word list produced by
    transcriber.transcribe_audio() and returns the best non-overlapping clip
    ranges, scored on: question marks, exclamations, emotional/curiosity
    keywords, numbers/statistics, direct address ("you"), pauses between
    sentences (dramatic beats), sentence length, and filler words.

Part 2 - Prompt template parser (text-to-script)
    render_template()      fills {{var}}, {{var|default}} and {{#if var}}...{{/if}}
    parse_script_prompt()  pulls topic / tone / duration / platform / points
                           out of a plain-English prompt using regex
    generate_script()      builds a hook + body + CTA script from the above
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

# =========================================================================== #
# Part 1 - Viral clip scoring
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


# =========================================================================== #
# Part 2 - Prompt template parser (text-to-script)
# =========================================================================== #
_IF_RE = re.compile(r"\{\{#if\s+(\w+)\s*\}\}(.*?)\{\{/if\}\}", re.S)
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*(?:\|([^}]*))?\}\}")


def render_template(template: str, variables: Optional[Dict[str, object]] = None) -> str:
    """Fill a template.

      {{name}}            required variable
      {{name|fallback}}   uses `fallback` when name is missing/empty
      {{#if name}}..{{/if}}  block kept only when name is truthy (no nesting)

    Raises ValueError listing any required variables that are missing.
    """
    variables = variables or {}

    def _if(m):
        return m.group(2) if variables.get(m.group(1)) else ""

    text = _IF_RE.sub(_if, template)
    missing: List[str] = []

    def _var(m):
        name, default = m.group(1), m.group(2)
        val = variables.get(name)
        if val not in (None, "", []):
            return ", ".join(map(str, val)) if isinstance(val, (list, tuple)) else str(val)
        if default is not None:
            return default.strip()
        missing.append(name)
        return m.group(0)

    out = _VAR_RE.sub(_var, text)
    if missing:
        raise ValueError(f"Missing template variables: {sorted(set(missing))}")
    return out


_TONES = [
    "funny", "humorous", "serious", "motivational", "inspiring", "educational",
    "dramatic", "casual", "professional", "friendly", "urgent", "storytelling",
]
_PLATFORMS = {
    "tiktok": "TikTok", "reels": "Instagram Reels", "instagram": "Instagram",
    "shorts": "YouTube Shorts", "youtube": "YouTube", "linkedin": "LinkedIn",
    "twitter": "X", "facebook": "Facebook",
}
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(seconds?|secs?|minutes?|mins?|s|m)\b", re.I
)
_TOPIC_RE = re.compile(
    r"\b(?:about|on|regarding|explaining|covering|around)\s+(.+?)"
    r"(?=\s+(?:for|in|with|using|that|which|aimed|targeted|under|to be)\b|[.,;!?\n]|$)",
    re.I,
)
_LEAD_RE = re.compile(
    r"^\s*(?:please\s+)?(?:write|create|generate|make|draft|give me)\s+"
    r"(?:me\s+)?(?:a|an|the)?\s*(?:short\s+|quick\s+)?(?:video\s+)?(?:script|video)?\s*",
    re.I,
)


def parse_script_prompt(prompt: str) -> dict:
    """Extract structured fields from a plain-English request.

    Returns {"topic", "tone", "duration_seconds", "platform", "points"}.
    Defaults: tone="engaging", duration_seconds=30, platform=None.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    text = prompt.strip()

    duration = 30
    m = _DURATION_RE.search(text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        duration = int(val * 60) if unit.startswith("m") else int(val)
        duration = max(5, duration)

    low = text.lower()
    platform = next((v for k, v in _PLATFORMS.items() if re.search(rf"\b{k}\b", low)), None)
    tone = next((t for t in _TONES if re.search(rf"\b{t}\b", low)), "engaging")

    points: List[str] = []
    for line in text.splitlines():
        b = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if b:
            points.append(b.group(1).strip().rstrip("."))
    if not points:
        kp = re.search(r"(?:key\s+)?points?\s*:\s*(.+)$", text, re.I | re.S)
        if kp:
            points = [p.strip().rstrip(".") for p in re.split(r";|\n", kp.group(1)) if p.strip()]

    head = text.split("\n")[0]
    tm = _TOPIC_RE.search(head)
    if tm:
        topic = tm.group(1).strip()
    else:
        topic = _LEAD_RE.sub("", head)
        topic = _DURATION_RE.sub("", topic).strip(" .,:;-")
    if not topic:
        topic = "this topic"

    return {
        "topic": topic,
        "tone": tone,
        "duration_seconds": duration,
        "platform": platform,
        "points": points,
    }


_HOOKS = {
    "funny": "Okay, nobody warned me about {{topic}}, and now I have opinions.",
    "humorous": "Okay, nobody warned me about {{topic}}, and now I have opinions.",
    "serious": "Here is what you need to understand about {{topic}}.",
    "motivational": "You can get better at {{topic}}, starting today.",
    "inspiring": "You can get better at {{topic}}, starting today.",
    "educational": "Here is how {{topic}} actually works.",
    "dramatic": "Everything you think you know about {{topic}} is about to change.",
    "urgent": "Stop scrolling. You need to hear this about {{topic}}.",
    "default": "Want to know the truth about {{topic}}?",
}
_GENERIC_BEATS = [
    "First, the basics of {{topic}} that most people get wrong.",
    "Second, the one change that makes the biggest difference with {{topic}}.",
    "Third, a simple way to start with {{topic}} today.",
    "Last, the common mistake to avoid with {{topic}}.",
]
_CTAS = {
    "TikTok": "Follow for more, and drop a comment with your take.",
    "Instagram Reels": "Follow for more and share this with someone who needs it.",
    "Instagram": "Follow for more and share this with someone who needs it.",
    "YouTube Shorts": "Subscribe for more, and tell me what to cover next.",
    "YouTube": "Subscribe for more, and tell me what to cover next.",
    "LinkedIn": "Follow for more, and share your experience in the comments.",
}
_WORDS_PER_SECOND = 2.5  # ~150 wpm


def generate_script(
    prompt: str,
    template: Optional[str] = None,
    variables: Optional[Dict[str, object]] = None,
) -> dict:
    """Build a short-form video script locally from a plain-English prompt.

    If `template` is given it is rendered with the parsed fields (topic, tone,
    duration, platform, points) plus any `variables`; otherwise a built-in
    hook -> body -> call-to-action structure is used.

    Returns {"topic","tone","platform","duration_seconds","hook","body","cta",
             "script","word_count","estimated_seconds"}.
    """
    parsed = parse_script_prompt(prompt)
    ctx: Dict[str, object] = {
        "topic": parsed["topic"],
        "tone": parsed["tone"],
        "duration": parsed["duration_seconds"],
        "platform": parsed["platform"] or "",
        "points": parsed["points"],
    }
    ctx.update(variables or {})

    if template is not None:
        script = render_template(template, ctx)
        wc = len(script.split())
        return {**parsed, "hook": "", "body": [], "cta": "", "script": script,
                "word_count": wc, "estimated_seconds": round(wc / _WORDS_PER_SECOND, 1)}

    hook = render_template(_HOOKS.get(parsed["tone"], _HOOKS["default"]), ctx)
    cta = _CTAS.get(parsed["platform"] or "", "Follow for more, and share this with a friend.")
    budget = int(parsed["duration_seconds"] * _WORDS_PER_SECOND)

    beats = [p if p.endswith((".", "!", "?")) else p + "." for p in parsed["points"]] or [
        render_template(b, ctx) for b in _GENERIC_BEATS
    ]
    used = len(hook.split()) + len(cta.split())
    body: List[str] = []
    for beat in beats:
        wc = len(beat.split())
        if body and used + wc > budget:
            break
        body.append(beat)
        used += wc

    script = " ".join([hook, *body, cta])
    wc = len(script.split())
    return {
        "topic": parsed["topic"], "tone": parsed["tone"], "platform": parsed["platform"],
        "duration_seconds": parsed["duration_seconds"], "points": parsed["points"],
        "hook": hook, "body": body, "cta": cta, "script": script,
        "word_count": wc, "estimated_seconds": round(wc / _WORDS_PER_SECOND, 1),
    }


# --------------------------------------------------------------------------- #
# Custom script + visual action narration
# --------------------------------------------------------------------------- #
# Camera/visual effects the video renderer (visual_cues.py) understands. Order matters:
# the more specific phrase ("zoom out") must be tested before the general one ("zoom").
EFFECT_KEYWORDS = [
    ("zoom_out", ("zoom out", "zoom away", "pull back", "pull out", "wide shot", "step back", "reveal")),
    ("zoom_in", ("zoom in", "zoom", "push in", "move in", "closer", "close up", "close-up", "focus on", "punch in")),
    ("pan_left", ("pan left", "slide left", "move left", "swipe left", "to the left")),
    ("pan_right", ("pan right", "slide right", "move right", "swipe right", "to the right")),
    ("flash", ("flash", "camera flash", "strobe", "pop")),
    ("shake", ("shake", "rumble", "impact", "explosion", "bump", "jolt", "earthquake")),
    ("fade", ("fade", "dissolve", "dim", "darken", "from black", "intro")),
]
_INLINE_CUE = re.compile(r"\[([^\[\]]{1,200})\]")
_BEAT_SPLIT = re.compile(r"[\n;]+|\bthen\b|(?<=[.!?])\s+", re.IGNORECASE)


def effect_for(text: str) -> str:
    """Map a free-text direction ("slowly zoom in on the product") to a known effect, or "none"."""
    low = text.casefold()
    for effect, words in EFFECT_KEYWORDS:
        if any(w in low for w in words):
            return effect
    return "none"


def parse_action_prompt(action_prompt: Optional[str]) -> List[Dict[str, str]]:
    """Split a visual-action description into ordered beats.

    "Zoom in on the logo. Then pan right to the product; flash at the price"
        -> [{"text": "Zoom in on the logo", "effect": "zoom_in"},
            {"text": "pan right to the product", "effect": "pan_right"},
            {"text": "flash at the price", "effect": "flash"}]
    Directions with no known effect are kept (effect "none") so they still show up in the shot list.
    """
    beats: List[Dict[str, str]] = []
    for part in _BEAT_SPLIT.split(action_prompt or ""):
        text = re.sub(r"^[\s\-*\d.)]+", "", part or "").strip(" .,\t")
        if len(re.findall(r"\w", text)) >= 2:
            beats.append({"text": text, "effect": effect_for(text)})
    return beats


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?\u2026])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and re.search(r"\w", p)]


# --------------------------------------------------------------------------- #
# complex scripts: scenes, speaker labels, visual lines, timecodes, pauses
# --------------------------------------------------------------------------- #
_SCENE_RE = re.compile(r"^\s*(#{1,3}\s*)?(?:scene|part|act|chapter|section)\b\s*(\d+)?\s*([:\-\u2013\u2014.])?\s*(.*)$", re.IGNORECASE)
_VISUAL_LABELS = ("visuals", "visual", "video", "action", "on screen", "on-screen", "onscreen", "b-roll", "broll", "show",
                  "cut to", "camera", "shot", "screen", "overlay", "graphic", "graphics", "effect", "fx", "animation")
_VISUAL_RE = re.compile(r"^\s*(?:" + "|".join(re.escape(l) for l in sorted(_VISUAL_LABELS, key=len, reverse=True)) + r")\s*:\s*(.+)$", re.IGNORECASE)
_SPEAKERS = ("narrator", "host", "voiceover", "voice over", "voice-over", "speaker", "vo", "v.o.", "announcer",
             "presenter", "guest", "avatar", "me")
_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .'\-]{0,24}?)\s*(\((?:V\.?O\.?|O\.?S\.?|voice[- ]?over|off[- ]?screen)\))?\s*:\s+(\S.*)$")
_TIMECODE_LEAD = re.compile(r"^\s*[\[(]?\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*[\])]?\s*[-\u2013\u2014:]?\s*")
_TIMECODE_INLINE = re.compile(r"\[\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*\]")
_PAUSE_RE = re.compile(r"[\[(]\s*(?:beat|pause)(?:\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?)?\s*[\])]", re.IGNORECASE)
_WHOLE_BRACKET = re.compile(r"^\s*[\[(](.+)[\])]\s*$")


def _timecode(m) -> float:
    frac = float("0." + m.group(3)) if m.group(3) else 0.0
    return int(m.group(1)) * 60 + int(m.group(2)) + frac


def _is_speaker(label: str, paren: Optional[str]) -> bool:
    low = label.strip().casefold()
    return bool(paren) or low in _SPEAKERS or (label.strip().isupper() and 1 < len(label.strip()) <= 24)


def parse_script_document(text: str) -> dict:
    """Understand a script written like a screenplay / shot list, not only plain sentences.

    Recognised (all optional, mix freely):
      SCENE 2: The reveal        scene headings ("Scene", "Part", "Act", "Chapter", "Section", "## Scene")
      NARRATOR: Hello there.     speaker labels (stripped; names in CAPS, known roles, or "(V.O.)")
      VISUAL: zoom in on logo    visual lines: VISUAL / VIDEO / ACTION / ON SCREEN / B-ROLL / SHOW / CUT TO / CAMERA ...
      (slowly pan right)         a whole line in () or [] is a direction
      > flash at the price       "> " quote lines are directions
      [0:12] Big reveal.         timecodes at the start of a line, or "[0:12]" inside it, pin that line's directions
      [pause 1s] / (beat)        pauses (kept as notes)
      Hello [zoom in] world.     inline [directions] inside a spoken sentence
    Direction lines belong to the NEXT spoken sentence (screenplay order). Returns
      {"sentences": [spoken text], "events": [{"sentence_index","text","source","at"?}],
       "scenes": [{"index","title","sentence_index"}], "sentence_times": {sentence_index: seconds}}"""
    sentences: List[str] = []
    events: List[dict] = []
    scenes: List[dict] = []
    sentence_times: Dict[int, float] = {}
    pending: List[dict] = []  # directions waiting for the next spoken sentence

    def add_event(idx: int, txt: str, source: str, at: Optional[float]) -> None:
        txt = re.sub(r"\s+", " ", txt).strip(" .,;")
        if txt:
            ev = {"sentence_index": idx, "text": txt, "source": source}
            if at is not None:
                ev["at"] = round(at, 2)
            events.append(ev)

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        pin: Optional[float] = None
        m = _TIMECODE_LEAD.match(line)
        if m and line[m.end():].strip():
            pin, line = _timecode(m), line[m.end():].strip()
        m = _TIMECODE_INLINE.search(line)
        if m:
            pin = _timecode(m) if pin is None else pin
            line = _TIMECODE_INLINE.sub(" ", line).strip()
        if not line:
            continue
        sc = _SCENE_RE.match(line)
        if sc and (sc.group(1) or sc.group(2) or sc.group(3)):  # "Part of the reason..." is speech, "Part 2 - Intro" a heading
            scenes.append({"index": len(scenes) + 1, "title": (sc.group(4) or "").strip(" -:."), "sentence_index": len(sentences)})
            continue
        vm = _VISUAL_RE.match(line)
        if vm:
            pending.append({"text": vm.group(1), "source": "visual", "at": pin})
            continue
        if line.startswith(">"):
            pending.append({"text": line.lstrip("> ").strip(), "source": "visual", "at": pin})
            continue
        pm = _PAUSE_RE.fullmatch(line)
        if pm:
            pending.append({"text": f"pause {pm.group(1)}s" if pm.group(1) else "pause", "source": "pause", "at": pin})
            continue
        wb = _WHOLE_BRACKET.match(line)
        if wb and not re.search(r"[.!?]\s+\S", wb.group(1)) and "[" not in wb.group(1):
            pending.append({"text": wb.group(1), "source": "visual", "at": pin})
            continue
        sp = _SPEAKER_RE.match(line)
        if sp and _is_speaker(sp.group(1), sp.group(2)):
            line = sp.group(3)
        first = True
        for sent in split_sentences(line):
            local: List[tuple] = []
            for pm in _PAUSE_RE.finditer(sent):
                local.append((f"pause {pm.group(1)}s" if pm.group(1) else "pause", "pause"))
            sent = _PAUSE_RE.sub(" ", sent)
            for cm in _INLINE_CUE.finditer(sent):
                local.append((cm.group(1).strip(), "inline"))
            spoken = re.sub(r"\s+", " ", _INLINE_CUE.sub(" ", sent)).strip()
            has_speech = bool(re.search(r"\w", spoken))
            idx = len(sentences)
            if has_speech and first:
                for ev in pending:
                    add_event(idx, ev["text"], ev["source"], ev["at"])
                pending = []
            for txt, src in local:
                add_event(idx, txt, src, pin if pin is not None and first else None)
            if has_speech:
                if first and pin is not None:
                    sentence_times[idx] = pin
                sentences.append(spoken)
                first = False
    last = max(0, len(sentences) - 1)
    for ev in pending:
        add_event(last, ev["text"], ev["source"], ev["at"])
    for ev in events:
        ev["sentence_index"] = min(ev["sentence_index"], last)
    for sc in scenes:
        sc["sentence_index"] = min(sc["sentence_index"], last)
    return {"sentences": sentences, "events": events, "scenes": scenes, "sentence_times": sentence_times}


def build_narration(
    prompt: Optional[str] = None,
    custom_script: Optional[str] = None,
    action_prompt: Optional[str] = None,
    screen_text: Optional[List[str]] = None,
    max_screen_lines: int = 8,
) -> dict:
    """The text to speak + a timed list of visual cues, from ANY mix of:

      custom_script  the user's own script; used word for word (generate_script is skipped).
                     "[zoom in]" style directions inside it become cues at that sentence.
      prompt         used to generate a script only when no custom script is given.
      action_prompt  a description of what should happen on screen (see parse_action_prompt);
                     its beats are spread across the sentences in order.
      screen_text    lines read from the video by OCR; narrated after the script.

    Returns {"script","sentences","word_count","estimated_seconds","source","topic",
             "cues":[{"index","sentence_index","text","effect","start","end"}], "shot_list":[str]}.
    Cue times are ESTIMATES from word counts; use align_cues() once the real audio length is known.
    """
    custom = (custom_script or "").strip()
    inline: List[dict] = []  # directions parsed out of the script (visual lines, inline [..], pauses)
    scenes: List[dict] = []
    pinned_times: Dict[int, float] = {}
    if custom:
        doc = parse_script_document(custom)
        sentences = doc["sentences"]
        if not sentences:
            raise ValueError("Your script has no speakable text (only [directions]).")
        inline, scenes, pinned_times = doc["events"], doc["scenes"], doc["sentence_times"]
        source, topic = "custom", (prompt or " ".join(sentences[0].split()[:6])).strip()
    else:
        if not (prompt or "").strip():
            raise ValueError("Write a prompt or paste your own script.")
        gen = generate_script(prompt)
        sentences, source, topic = split_sentences(gen["script"]), "generated", gen["topic"]

    lines = []
    for item in (screen_text or [])[:max_screen_lines]:
        text = item.get("text", "") if isinstance(item, dict) else item  # plain lines or ocr_engine.narration_queue() items
        t = re.sub(r"\s+", " ", str(text)).strip(" .")
        if len(re.findall(r"\w", t)) >= 2:  # drop single-character OCR noise
            lines.append(f"The text on screen reads: {t}.")
    sentences = sentences + lines

    script = " ".join(sentences)
    words_per = [len(s.split()) for s in sentences]
    starts, acc = [], 0.0
    for wc in words_per:
        starts.append(acc)
        acc += wc / _WORDS_PER_SECOND

    cues: List[dict] = []
    beats = parse_action_prompt(action_prompt)
    n_sent = max(1, len(sentences) - len(lines))  # actions belong to the script part, not the OCR part
    for i, beat in enumerate(beats):
        si = min(n_sent - 1, int(i * n_sent / len(beats)))
        cues.append({"sentence_index": si, "text": beat["text"], "effect": beat["effect"]})
    for ev in inline:
        cue = {"sentence_index": min(ev["sentence_index"], len(sentences) - 1), "text": ev["text"],
               "effect": "none" if ev["source"] == "pause" else effect_for(ev["text"])}
        if "at" in ev:
            cue["at"] = ev["at"]  # a timecode written in the script
        cues.append(cue)
    cues.sort(key=lambda c: c["sentence_index"])
    for i, c in enumerate(cues):
        c["index"] = i
        c["pinned"] = "at" in c
        c["start"] = round(c.pop("at") if c["pinned"] else starts[c["sentence_index"]], 2)
        later = [x for x in starts if x > c["start"] + 1e-9]
        c["end"] = round(max(min(later[0] if later else acc, c["start"] + 6.0), c["start"] + 1.0), 2)  # 1-6 s per cue
    wc = sum(words_per)
    return {
        "script": script, "sentences": sentences, "word_count": wc,
        "estimated_seconds": round(wc / _WORDS_PER_SECOND, 1), "source": source, "topic": topic,
        "cues": cues, "scenes": [{**sc, "start": round(starts[min(sc["sentence_index"], len(starts) - 1)], 2)} for sc in scenes],
        "shot_list": [f"{c['start']:.1f}s  {c['effect'].replace('_', ' ') if c['effect'] != 'none' else 'note'}: {c['text']}"
                      for c in cues],
    }


def align_cues(cues: List[dict], estimated_seconds: float, actual_seconds: float) -> List[dict]:
    """Stretch estimated cue times to the real voiceover length (TTS pace differs from 150 wpm)."""
    if estimated_seconds <= 0 or actual_seconds <= 0:
        return [dict(c) for c in cues]
    k = actual_seconds / estimated_seconds
    return [dict(c) if c.get("pinned") else {**c, "start": round(c["start"] * k, 2), "end": round(c["end"] * k, 2)}
            for c in cues]  # cues pinned with a timecode in the script keep their exact time


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Local scorer / script generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("score", help="score a transcript JSON (list of word dicts)")
    s1.add_argument("words_json")
    s1.add_argument("--top", type=int, default=5)
    s2 = sub.add_parser("script", help="generate a script from a prompt")
    s2.add_argument("prompt")
    a = ap.parse_args()
    if a.cmd == "score":
        with open(a.words_json, encoding="utf-8") as fh:
            print(json.dumps(find_viral_clips(json.load(fh), top_n=a.top), indent=2))
    else:
        print(json.dumps(generate_script(a.prompt), indent=2))
