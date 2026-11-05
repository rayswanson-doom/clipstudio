"""
backend/avatar_gestures.py  -  what the avatar does with its head, arms and face while it talks.

Patterns from https://github.com/lhupyn/motion-engine (MIT): a catalogue of HEAD gestures (nod, shake,
tilt, bow, ... - the keyframes below are copied from its gestures.json) and HAND gestures (open palms,
point, wave, thumbs up, shrug, ...), picked for every sentence by a "director" with confidence thresholds,
cool-downs and a "relax after" rule. motion-engine asks an online classifier (Jev) to choose; this
module uses an offline keyword classifier instead, so it works with no internet and no API key.

Hand gestures are NOT copied from motion-engine: its poses are Euler angles for one specific Mixamo
rig. Here every arm pose is a pair of DIRECTIONS in the body frame (outward, up, forward) for the upper
arm and forearm plus a palm roll, which the rig solver (avatar_3d_engine.Rig) turns into bone rotations
for any humanoid skeleton - Ready Player Me, Mixamo, VRoid/VRM. Fingers are not posed individually, so
thumbs-up / OK / pointing are arm poses only.
"""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Optional, Tuple

# --- head gestures: keyframes copied from motion-engine/src/gestures.json (radians; dt in ms) --------------
HEAD_GESTURES: Dict[str, dict] = {
    "nod": {"dt": [260, 240, 240, 320], "vs": {"headRotateX": [0.18, -0.03, 0.13, 0], "headRotateZ": [0]}},
    "nod_small": {"dt": [220, 200, 200, 260], "vs": {"headRotateX": [0.09, 0, 0.07, 0]}},
    "shake": {"dt": [280, 300, 280, 260, 300], "vs": {"headRotateY": [-0.16, 0.16, -0.12, 0.08, 0], "headRotateZ": [0]}},
    "tilt": {"dt": [450, 1800, 600], "vs": {"headRotateZ": [-0.18, -0.18, 0], "headRotateX": [0.04, 0.04, 0]}},
    "bow": {"dt": [500, 600, 600], "vs": {"bodyRotateX": [0.16, 0.16, 0], "headRotateX": [0.06, 0.06, 0]}},
}

# --- hand gestures: arm directions in the body frame (outward, up, forward) -----------------------------
# side: "both" | "right" | "left" (character's own right/left). fade/hold in seconds.
REST = {"arm": (0.14, -0.98, 0.02), "fore": (0.04, -0.80, 0.60), "roll": 0.0}
HAND_GESTURES: Dict[str, dict] = {
    "open_palms": {"side": "both", "hold": 1.9, "arm": (0.38, -0.80, 0.38), "fore": (0.22, -0.12, 0.97), "roll": 1.3},
    "hand_on_heart": {"side": "right", "hold": 1.8, "arm": (0.08, -0.95, 0.28), "fore": (-0.78, 0.30, 0.55), "roll": 0.4},
    "think": {"side": "right", "hold": 2.2, "arm": (0.02, -0.82, 0.40), "fore": (-0.40, 0.88, 0.25), "roll": 0.2},
    "wait": {"side": "right", "hold": 1.5, "arm": (0.38, -0.55, 0.50), "fore": (0.10, 0.50, 0.86), "roll": 0.0},
    "point": {"side": "right", "hold": 1.5, "arm": (0.26, -0.50, 0.82), "fore": (0.15, -0.08, 0.99), "roll": 0.0},
    "wave": {"side": "right", "hold": 1.6, "arm": (0.52, -0.38, 0.42), "fore": (0.18, 0.95, 0.32), "roll": 0.0,
             "osc": {"freq": 2.6, "amp": 0.45, "axis": "out"}},
    "hand_raise": {"side": "right", "hold": 1.8, "arm": (0.42, 0.20, 0.40), "fore": (0.10, 0.98, 0.15), "roll": 0.0},
    "thumbs_up": {"side": "right", "hold": 1.6, "arm": (0.24, -0.70, 0.42), "fore": (0.08, 0.40, 0.91), "roll": 0.0},
    "thumbs_down": {"side": "right", "hold": 1.6, "arm": (0.24, -0.75, 0.40), "fore": (0.10, -0.20, 0.97), "roll": 3.1},
    "ok_sign": {"side": "right", "hold": 1.6, "arm": (0.30, -0.65, 0.45), "fore": (0.12, 0.45, 0.88), "roll": 0.5},
    "shrug": {"side": "both", "hold": 1.3, "arm": (0.55, -0.72, 0.18), "fore": (0.50, -0.05, 0.86), "roll": 1.45, "lift": 0.07},
    "namaste": {"side": "both", "hold": 1.8, "arm": (0.20, -0.90, 0.32), "fore": (-0.62, 0.62, 0.48), "roll": 0.9},
    "facepalm": {"side": "right", "hold": 1.5, "arm": (0.0, -0.80, 0.45), "fore": (-0.15, 0.95, 0.20), "roll": 0.0},
    "celebrate": {"side": "right", "hold": 1.7, "arm": (0.38, 0.50, 0.35), "fore": (0.12, 0.97, 0.18), "roll": 0.0},
    "applause": {"side": "both", "hold": 2.0, "arm": (0.18, -0.86, 0.38), "fore": (-0.35, 0.30, 0.89), "roll": 1.5,
                 "osc": {"freq": 4.0, "amp": 0.35, "axis": "out"}},
}
FADE = 0.45

# --- keyword "director" (offline stand-in for motion-engine's classifier) ----------------------------------
_RULES: List[Tuple[str, str, str, float]] = [  # (regex, head, hand, confidence)
    (r"\b(hello|hi there|hey|welcome|good (morning|evening|afternoon)|greetings)\b", "nod_small", "wave", 0.9),
    (r"\b(bye|goodbye|see you|farewell)\b", "nod_small", "wave", 0.9),
    (r"\b(thank(s| you)?|grateful|appreciate)\b", "bow", "namaste", 0.85),
    (r"\b(sorry|apolog\w+)\b", "bow", "hand_on_heart", 0.8),
    (r"\b(congrat\w*|we won|you won|victory|champion|celebrate|hooray|yes!)\b", "nod", "celebrate", 0.9),
    (r"\b(applaus\w*|clap\w*|bravo)\b", "nod", "applause", 0.9),
    (r"\b(i don'?t know|not sure|maybe|perhaps|who knows|no idea|unclear|uncertain)\b", "tilt", "shrug", 0.85),
    (r"\b(wait|hold on|stop|one moment|pause)\b", "", "wait", 0.8),
    (r"\b(hmm+|let me think|let'?s think|think about|consider|ponder|imagine)\b", "tilt", "think", 0.8),
    (r"\b(oh no|oops|mistake|embarrass\w*|facepalm)\b", "shake", "facepalm", 0.75),
    (r"\b(i mean it|honest(ly)?|sincere(ly)?|from the heart|truly|personally)\b", "nod", "hand_on_heart", 0.75),
    (r"\b(great|awesome|amazing|fantastic|love|perfect|excellent|brilliant|well done|good job|sounds good)\b", "nod", "thumbs_up", 0.8),
    (r"\b(terrible|awful|bad|hate|worst|horrible|disappointing)\b", "shake", "thumbs_down", 0.8),
    (r"\b(okay|ok|all good|alright|understood|got it)\b", "nod_small", "ok_sign", 0.7),
    (r"\b(no|not|never|don'?t|doesn'?t|won'?t|can'?t|nothing|wrong|false|myth)\b", "shake", "", 0.7),
    (r"\b(yes|yeah|yep|exactly|absolutely|definitely|indeed|right|correct|true|sure)\b", "nod", "", 0.75),
    (r"\b(you|your|yours)\b", "", "point", 0.55),
    (r"\b(question|why|how|what if|ever wondered|curious)\b", "tilt", "", 0.65),
    (r"\b(let me explain|here'?s|here is|the thing|because|important|key|basically|so|first|second|third|finally|remember|actually)\b",
     "", "open_palms", 0.6),
]
_COMPILED = [(re.compile(r, re.I), h, g, c) for r, h, g, c in _RULES]
NOTE_ALIASES = {  # words a user may write in [directions] / the action prompt
    "wave": "wave", "waves": "wave", "hello": "wave", "greet": "wave", "point": "point", "nod": "nod", "yes": "nod",
    "shake": "shake", "no": "shake", "shrug": "shrug", "clap": "applause", "applaud": "applause", "applause": "applause",
    "thumbs up": "thumbs_up", "thumbs-up": "thumbs_up", "thumbs down": "thumbs_down", "ok": "ok_sign", "think": "think",
    "ponder": "think", "celebrate": "celebrate", "cheer": "celebrate", "raise hand": "hand_raise", "raise a hand": "hand_raise",
    "hand raise": "hand_raise", "bow": "bow", "namaste": "namaste", "pray": "namaste", "heart": "hand_on_heart",
    "open palms": "open_palms", "open hands": "open_palms", "explain": "open_palms", "gesture": "open_palms",
    "facepalm": "facepalm", "tilt": "tilt", "stop": "wait", "wait": "wait", "hold up": "wait",
}


def classify_sentence(text: str) -> Tuple[str, str, float]:
    """(head gesture or "", hand gesture or "", confidence) for one sentence."""
    head = hand = ""
    conf = 0.0
    for rx, h, g, c in _COMPILED:
        if rx.search(text):
            if h and not head:
                head, conf = h, max(conf, c)
            if g and not hand:
                hand, conf = g, max(conf, c)
    if "?" in text and not head:
        head, conf = "tilt", max(conf, 0.6)
    return head, hand, conf


def gesture_from_note(text: str) -> Optional[str]:
    """A [direction] / action-prompt line -> a gesture name, or None if it is a camera move or unknown."""
    t = text.lower()
    for key in sorted(NOTE_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", t):
            return NOTE_ALIASES[key]
    return None


def plan_gestures(sentences: List[dict], notes: Optional[List[dict]] = None, duration: float = 0.0,
                  seed: int = 7, intensity: float = 1.0, min_confidence: float = 0.55,
                  cooldown: float = 2.2) -> List[dict]:
    """Decide what plays when.

    sentences  [{"text","start","end"}]      notes  [{"start","text"}] explicit user directions
    -> events [{"channel": "head"|"hands", "name", "start", "end", "side", "explicit"}]
    Rules (as in motion-engine's director): confidence threshold, a cool-down per channel, and a
    little variety - an unremarkable sentence now and then gets "open palms" so the body never freezes."""
    rng = random.Random(seed)
    events: List[dict] = []
    last_end = {"head": -9.0, "hands": -9.0}
    rotation = ["open_palms", "nod_small", "open_palms", "point", "open_palms"]
    quiet = 0

    def add(channel: str, name: str, start: float, explicit: bool = False):
        if channel == "head":
            total = sum(HEAD_GESTURES[name]["dt"]) / 1000.0
            end = start + total
        else:
            spec = HAND_GESTURES[name]
            end = start + spec["hold"] + 2 * FADE
        if not explicit and start < last_end[channel] + (cooldown if channel == "hands" else 0.8):
            return False
        events.append({"channel": channel, "name": name, "start": round(start, 3), "end": round(end, 3),
                       "side": HAND_GESTURES[name]["side"] if channel == "hands" else "", "explicit": explicit})
        last_end[channel] = end
        return True

    for note in sorted(notes or [], key=lambda n: n.get("start", 0.0)):
        name = gesture_from_note(note.get("text", ""))
        if not name:
            continue
        add("head" if name in HEAD_GESTURES else "hands", name, float(note.get("start", 0.0)), explicit=True)

    for s in sentences:
        head, hand, conf = classify_sentence(s["text"])
        start = float(s["start"])
        span = max(float(s["end"]) - start, 0.5)
        if conf >= min_confidence:
            quiet = 0
            if hand:
                add("hands", hand, start + min(0.25, span * 0.1))
            if head:
                add("head", head, start + min(0.15, span * 0.05))
        else:
            quiet += 1
            if intensity >= 0.5 and quiet >= 2 and span > 1.4:
                pick = rotation[len(events) % len(rotation)]
                add("head" if pick in HEAD_GESTURES else "hands", pick, start + span * rng.uniform(0.1, 0.4))
                quiet = 0
    events.sort(key=lambda e: e["start"])
    return events


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def hand_envelope(ev: dict, t: float) -> float:
    """0..1: eases in over FADE, holds, eases out."""
    if t < ev["start"] or t > ev["end"]:
        return 0.0
    if t < ev["start"] + FADE:
        return _smooth((t - ev["start"]) / FADE)
    if t > ev["end"] - FADE:
        return _smooth((ev["end"] - t) / FADE)
    return 1.0


def head_angles(ev: dict, t: float) -> Dict[str, float]:
    """Keyframed head/body rotation (radians) of a head gesture at time t - linear keys, eased."""
    spec = HEAD_GESTURES[ev["name"]]
    dts = [d / 1000.0 for d in spec["dt"]]
    local = t - ev["start"]
    if local < 0 or local > sum(dts):
        return {}
    out: Dict[str, float] = {}
    for key, vals in spec["vs"].items():
        prev, acc = 0.0, 0.0
        for i, d in enumerate(dts):
            target = vals[min(i, len(vals) - 1)]
            if local <= acc + d:
                f = _smooth((local - acc) / d) if d > 0 else 1.0
                out[key] = prev + (target - prev) * f
                break
            prev, acc = target, acc + d
    return out
