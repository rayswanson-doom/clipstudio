"""
script_parser.py - the Script2Video engine: a screenplay parser plus a shot-manifest generator.

Reads standard screenplay formatting (Fountain-style plain text):

    INT. KITCHEN - NIGHT                 scene heading (INT. / EXT. / INT./EXT. / I/E, location, time of day)
    MARA (30s) stands at the sink.       action (a first-time name in CAPS is fine)
                                         blank line
    MARA (V.O.)                          character cue, optional (V.O.) (O.S.) (CONT'D) extension
    (whispering)                         parenthetical
    He never came home.                  dialogue
    CUT TO:                              transition

and turns it into a manifest of shots with exact start/end times and camera instructions, deterministically and
offline (no model, no GPU).  `script_to_shots()` is what the orchestrator's Storyboard Artist starts from.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

TIMES_OF_DAY = ("DAY", "NIGHT", "MORNING", "EVENING", "AFTERNOON", "DAWN", "DUSK", "SUNSET", "SUNRISE", "NOON",
                "MIDNIGHT", "LATER", "CONTINUOUS", "SAME", "MOMENTS LATER", "SAME TIME")
_HEADING = re.compile(r"^\s*(?:\d+[A-Z]?\s+)?(INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|I\s*/\s*E\.?|INT\.?|EXT\.?)\s+(.+?)\s*(?:\d+[A-Z]?)?\s*$",
                      re.IGNORECASE)
_TRANSITION = re.compile(r"^\s*(?:[A-Z][A-Z .'\-]*\bTO:|FADE IN:?|FADE OUT\.?|FADE TO BLACK\.?|THE END\.?|FADE TO:|CUT TO BLACK\.?)\s*$")
_CUE_EXT = re.compile(r"\s*\((V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D\.?|CONT\.?|FILTERED|ON PHONE|PRE-?LAP)\)", re.IGNORECASE)
_CUE = re.compile(r"^[A-Z0-9][A-Z0-9 .'\-]{0,38}(?:\s*\([^)]*\))*\s*\^?$")
_PAREN = re.compile(r"^\s*\((.+)\)\s*$")

TRANSITION_KINDS = {"CUT": "cut", "DISSOLVE": "dissolve", "FADE": "fade", "SMASH": "smash_cut", "MATCH": "match_cut",
                    "WIPE": "wipe", "JUMP": "jump_cut", "IRIS": "iris"}
EMOTION_WORDS = ("whisper", "cry", "sob", "tear", "shout", "scream", "yell", "angry", "furious", "scared", "afraid",
                 "trembl", "quiet", "soft", "chok", "weep", "pleading", "desperate", "heartbroken", "stunned")
ACTION_RULES = (  # (regex, shot_type, angle, movement)
    (r"\b(run|runs|ran|chase|chases|sprint|race|flee|flees|fight|fights|punch|crash|crashes|escape)\b", "tracking medium wide", "eye level", "handheld tracking"),
    (r"\b(phone|letter|note|gun|knife|key|keys|clock|watch|ring|photograph|photo|map|screen|button|door ?knob|envelope|book)\b", "insert close-up", "eye level", "static"),
    (r"\b(look|looks|stare|stares|watch|watches|notice|notices|glance|gaze|gazes|realiz\w+|recogni\w+)\b", "close-up", "eye level", "slow push-in"),
    (r"\b(enter|enters|arrive|arrives|walk in|walks in|steps? in|emerges?|appears?)\b", "wide", "eye level", "static"),
    (r"\b(towers?|looming|menacing|giant|looms)\b", "medium", "low angle", "slow tilt up"),
    (r"\b(alone|tiny|lost|small|empty|abandoned)\b", "wide", "high angle", "slow pull-out"),
    (r"\b(explod\w+|burst|storm|lightning|fire|flames)\b", "wide", "low angle", "handheld"),
)
LIGHTING = {"DAY": "bright natural daylight", "MORNING": "soft morning light", "AFTERNOON": "warm afternoon sun",
            "EVENING": "warm low evening light", "DAWN": "cool golden dawn light", "SUNRISE": "golden sunrise light",
            "DUSK": "golden-blue dusk light", "SUNSET": "orange sunset light", "NIGHT": "low-key moody night lighting with practical lights",
            "MIDNIGHT": "deep blue low-key night lighting", "NOON": "hard overhead midday light"}


class ScriptParseError(ValueError):
    pass


def _words(text: str) -> int:
    return len(re.findall(r"[\w'’-]+", text))


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])[\"')\]]*\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _transition_kind(text: str) -> str:
    head = text.strip().split(" ")[0].rstrip(":.").upper()
    return TRANSITION_KINDS.get(head, "cut")


def _split_heading(rest: str):
    """'KITCHEN - NIGHT' -> ('KITCHEN', 'NIGHT'); 'HOUSE - KITCHEN - LATER' -> ('HOUSE - KITCHEN', 'LATER')."""
    pieces = [p.strip() for p in re.split(r"\s+[-–—]\s+|\s*--\s*", rest) if p.strip()]
    if len(pieces) > 1 and pieces[-1].upper().rstrip(".") in TIMES_OF_DAY:
        return " - ".join(pieces[:-1]), pieces[-1].upper().rstrip(".")
    m = re.match(r"^(.*?)[,\s]+(DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|SUNSET|SUNRISE)$", rest.strip(), re.IGNORECASE)
    if m and m.group(1):
        return m.group(1).strip(" ,-"), m.group(2).upper()
    return rest.strip(), ""


def _int_ext(token: str) -> str:
    t = token.upper().replace(" ", "")
    if "/" in t:
        return "INT/EXT"
    return "EXT" if t.startswith("EXT") else "INT"


def _is_cue(line: str, next_line: Optional[str]) -> bool:
    s = line.strip()
    if s.startswith("@"):
        return True
    if not s or next_line is None or not next_line.strip():
        return False
    base = re.sub(r"\([^)]*\)", "", s).strip()
    if not base or not re.search(r"[A-Z]", base) or base != base.upper() or not _CUE.match(s):
        return False
    return not (_HEADING.match(s) or _TRANSITION.match(s))


def parse_script(text: str) -> dict:
    """Screenplay text -> {"title","scenes":[{"index","heading","int_ext","location","time_of_day","elements":[...]}],
    "characters":[names in order of appearance], "warnings":[...]}.
    Elements: {"type":"action","text"} | {"type":"dialogue","character","extension","parenthetical","text"} |
    {"type":"transition","text","kind"}. Text before the first heading is placed in an implicit scene."""
    if not str(text or "").strip():
        raise ScriptParseError("The script is empty.")
    src = re.sub(r"/\*.*?\*/|\[\[.*?\]\]", "", str(text).replace("\r\n", "\n").replace("\r", "\n"), flags=re.S)
    lines = src.split("\n")
    title = ""
    scenes: List[dict] = []
    characters: List[str] = []
    warnings: List[str] = []
    cur: Optional[dict] = None
    leading_transition = ""  # e.g. "FADE IN:" before the first heading

    def new_scene(heading: str, int_ext: str, location: str, tod: str) -> dict:
        nonlocal leading_transition
        sc = {"index": len(scenes) + 1, "heading": heading, "int_ext": int_ext, "location": location,
              "time_of_day": tod, "elements": []}
        if leading_transition and not scenes:
            sc["elements"].append({"type": "transition", "text": leading_transition, "kind": _transition_kind(leading_transition)})
            leading_transition = ""
        scenes.append(sc)
        return sc

    def add_char(name: str) -> None:
        if name and name not in characters:
            characters.append(name)

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()
        nxt = lines[i + 1] if i + 1 < n else None
        prev_blank = i == 0 or not lines[i - 1].strip()
        if not line or re.fullmatch(r"=+", line) or line.startswith(("#", "=")) and not _HEADING.match(line.lstrip("#= ")):
            i += 1
            continue
        tm = re.match(r"^title\s*:\s*(.+)$", line, re.IGNORECASE)
        if tm and not scenes and not title:
            title = tm.group(1).strip()
            i += 1
            continue
        forced = line.startswith(".") and not line.startswith("..")
        hm = _HEADING.match(line[1:].strip() if forced else line)
        if hm or forced:
            if hm:
                loc, tod = _split_heading(hm.group(2))
                cur = new_scene(line.lstrip(".").strip().upper(), _int_ext(hm.group(1)), loc.upper(), tod)
            else:
                body = line[1:].strip()
                loc, tod = _split_heading(body)
                cur = new_scene(body.upper(), "INT", loc.upper(), tod)
            i += 1
            continue
        if cur is None and (line.startswith(">") or _TRANSITION.match(line)):
            leading_transition = line.lstrip("> ").strip()
            i += 1
            continue
        if cur is None:
            cur = new_scene("INT. UNSPECIFIED - DAY", "INT", "UNSPECIFIED", "DAY")
            warnings.append("The script starts without a scene heading; an implicit scene was created.")
        if line.startswith(">") or _TRANSITION.match(line):
            tr = line.lstrip("> ").strip().rstrip("<").strip()
            cur["elements"].append({"type": "transition", "text": tr, "kind": _transition_kind(tr)})
            i += 1
            continue
        if prev_blank and _is_cue(line, nxt):
            cue = line.lstrip("@").strip().rstrip("^").strip()
            ext = "".join(m.group(0) for m in _CUE_EXT.finditer(cue)).strip()
            name = re.sub(r"\s*\([^)]*\)", "", cue).strip()
            if cue.startswith("@") or not name:
                name = name or cue
            name = name.title() if name.isupper() else name
            paren, dialogue = "", []
            j = i + 1
            while j < n and lines[j].strip():
                pm = _PAREN.match(lines[j])
                if pm and not dialogue:
                    paren = (paren + " " + pm.group(1)).strip()
                elif pm:
                    paren = (paren + " " + pm.group(1)).strip()
                else:
                    seg = lines[j].strip()
                    for inline in re.findall(r"\(([^)]*)\)", seg):  # "Wait! (beat) Please." -> parenthetical "beat"
                        paren = (paren + " " + inline).strip()
                    dialogue.append(re.sub(r"\s*\([^)]*\)\s*", " ", seg).strip())
                j += 1
            if dialogue:
                add_char(name)
                cur["elements"].append({"type": "dialogue", "character": name, "extension": ext.strip("() ").upper(),
                                        "parenthetical": paren, "text": " ".join(dialogue)})
                i = j
                continue
        # action: gather until a blank line
        block = [line]
        j = i + 1
        while j < n and lines[j].strip() and not _HEADING.match(lines[j].strip()):
            block.append(lines[j].strip())
            j += 1
        cur["elements"].append({"type": "action", "text": " ".join(block)})
        i = j
    for sc in scenes:
        if not sc["elements"]:
            warnings.append(f"Scene {sc['index']} ({sc['heading']}) has no action or dialogue.")
    if not scenes:
        raise ScriptParseError("No scenes were found in the script.")
    return {"title": title, "scenes": scenes, "characters": characters, "warnings": warnings}


# --------------------------------------------------------------------------- #
# shot manifest
# --------------------------------------------------------------------------- #
def _classify_action(text: str):
    low = text.casefold()
    for rx, shot_type, angle, move in ACTION_RULES:
        if re.search(rx, low):
            return shot_type, angle, move
    return "medium", "eye level", "static"


def _names_in(text: str, known: List[str]) -> List[str]:
    found = []
    for name in known:
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
            found.append(name)
    return found


def _lighting(tod: str) -> str:
    return LIGHTING.get(tod, "natural light" if not tod else "natural light")


def _variation(move: str) -> str:
    m = move.casefold()
    if any(k in m for k in ("tracking", "handheld", "whip", "crane")):
        return "large"
    return "small" if m.startswith("static") else "medium"


def _chunk_sentences(text: str, max_words: int) -> List[str]:
    out, cur = [], ""
    for s in _sentences(text) or [text]:
        if cur and _words(cur + " " + s) > max_words:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def script_to_shots(text: str, words_per_second: float = 2.6, action_words_per_second: float = 3.2,
                    min_shot: float = 1.2, max_shot: float = 8.0, dialogue_pad: float = 0.35,
                    establishing: float = 2.5, screenplay: Optional[dict] = None) -> dict:
    """Screenplay text -> {"shots":[...], "scenes":[...], "characters":[...], "total_duration", "warnings"}.

    Timing is exact and gap-free: shot[k].start == shot[k-1].end, spoken lines last words / words_per_second plus a
    short pad (and 0.6 s per "(beat)"/"(pause)"), action lasts words / action_words_per_second, both clamped to
    [min_shot, max_shot] (long blocks are split at sentence boundaries). Each shot carries a camera angle, shot type,
    movement, lighting, a text prompt that references characters as <Name>, and first/last-frame + motion
    descriptions (the ViMax shot-description fields)."""
    sp = screenplay or parse_script(text)
    chars = list(sp["characters"])
    shots: List[dict] = []
    t = 0.0
    last_speaker: Optional[str] = None
    pending_transition = ""

    def emit(sc: dict, kind: str, dur: float, shot_type: str, angle: str, move: str, characters: List[str], *,
             action: str = "", speaker: str = "", line: str = "", emotion: str = "", extension: str = "",
             near: Optional[str] = None) -> dict:
        nonlocal t, pending_transition
        dur = round(min(max_shot, max(min_shot, dur)), 2)
        loc = sc["location"].title()
        tags = [f"<{c}>" for c in characters]
        light = _lighting(sc["time_of_day"])
        subject = " and ".join(tags) if tags else "the location"
        base = (action or line or "").strip()
        visual = f"{shot_type.capitalize()}, {angle}, {move}. {sc['int_ext'].replace('/', '/')} {loc}, {light}. "
        if speaker and line:
            visual += f'{subject} - <{speaker}> says: "{line}"'
            if emotion:
                visual += f" ({emotion})"
        elif base:
            visual += base if not tags else f"{subject}: {base}"
        shot = {
            "idx": len(shots), "scene_idx": sc["index"], "cam_idx": 0, "kind": kind, "start": round(t, 2),
            "end": round(t + dur, 2), "duration": dur, "shot_type": shot_type, "camera_angle": angle,
            "camera_movement": move, "characters": characters, "speaker": speaker or None, "line": line or None,
            "emotion": emotion or None, "extension": extension or None, "action": action or None,
            "location": sc["location"], "int_ext": sc["int_ext"], "time_of_day": sc["time_of_day"],
            "lighting": light, "near_character": near,
            "transition_in": pending_transition or None,
            "visual_desc": visual.strip(),
            "camera_prompt": f"{shot_type}, {angle}, {move}, 35mm lens, shallow depth of field, {light}, {loc}".strip(),
            "ff_desc": f"Opening frame: {shot_type} of {subject} in {loc}.",
            "lf_desc": f"Closing frame: {shot_type} of {subject} after the {move} move." if move != "static" else
                       f"Closing frame: the same framing of {subject}, expression settled.",
            "motion_desc": f"{move}" + (f"; {subject} speaks" if speaker else ""),
            "variation_type": _variation(move),
            "audio_desc": (f"[Speaker] {speaker}{' (' + emotion + ')' if emotion else ''}: {line}" if speaker and line else
                           "[Sound Effect] Ambient sound"),
            "is_last": False,
        }
        pending_transition = ""
        shots.append(shot)
        t = round(t + dur, 2)
        return shot

    for sc in sp["scenes"]:
        first_in_scene = True
        last_speaker = None
        for el in sc["elements"]:
            if el["type"] == "transition":
                if shots and el["kind"] in ("fade", "dissolve") and el["text"].upper().startswith(("FADE OUT", "FADE TO", "THE END")):
                    shots[-1]["transition_out"] = el["kind"]
                else:
                    pending_transition = el["kind"]
                continue
            if first_in_scene:
                first_in_scene = False
                present = _names_in(el.get("text", "") if el["type"] == "action" else "", chars)
                est = "wide" if sc["int_ext"] != "EXT" else "extreme wide"
                emit(sc, "establishing", establishing, f"establishing {est}", "eye level",
                     "slow push-in" if sc["int_ext"] != "EXT" else "slow crane down", [],
                     action=f"Establishing view of {sc['location'].title()}.")
            if el["type"] == "action":
                shot_type, angle, move = _classify_action(el["text"])
                for piece in _chunk_sentences(el["text"], int(max_shot * action_words_per_second)):
                    shot_type, angle, move = _classify_action(piece)
                    present = _names_in(piece, chars)
                    emit(sc, "action", _words(piece) / action_words_per_second, shot_type, angle, move, present, action=piece)
                continue
            # dialogue
            speaker = el["character"]
            paren = el["parenthetical"]
            emotional = any(w in paren.casefold() for w in EMOTION_WORDS)
            beat = 0.6 * len(re.findall(r"\b(?:beat|pause|long pause|silence)\b", paren, re.IGNORECASE))
            ext = el["extension"]
            in_scene = [c for c in {x["character"] for x in sc["elements"] if x["type"] == "dialogue"}]
            other = next((c for c in reversed([x["character"] for x in sc["elements"] if x["type"] == "dialogue"
                                               and x["character"] != speaker])), None)
            if ext.startswith("V"):
                shot_type, angle, move, who, near = "cutaway", "eye level", "slow push-in", [], None
            elif emotional:
                shot_type, angle, move, who, near = "close-up", "eye level", "slow push-in", [speaker], speaker
            elif last_speaker and last_speaker != speaker and other:
                shot_type, angle, move, who, near = "over-the-shoulder medium close-up", "eye level", "static", [speaker, other], other
            else:
                shot_type, angle, move, who, near = "medium close-up", "eye level", "static", [speaker], speaker
            if ext.startswith("O"):
                shot_type, who = "medium shot (speaker off-screen)", [c for c in [other] if c]
            for piece in _chunk_sentences(el["text"], int(max_shot * words_per_second)):
                emit(sc, "dialogue", _words(piece) / words_per_second + dialogue_pad + beat, shot_type, angle, move, who,
                     speaker=speaker, line=piece, emotion=paren, extension=ext, near=near)
                beat = 0.0
            last_speaker = speaker
    if not shots:
        raise ScriptParseError("The script has scenes but nothing to show (no action or dialogue).")
    shots[-1]["is_last"] = True
    scene_rows = []
    for sc in sp["scenes"]:
        mine = [s for s in shots if s["scene_idx"] == sc["index"]]
        if mine:
            scene_rows.append({"index": sc["index"], "heading": sc["heading"], "int_ext": sc["int_ext"],
                               "location": sc["location"], "time_of_day": sc["time_of_day"], "start": mine[0]["start"],
                               "end": mine[-1]["end"], "shots": [s["idx"] for s in mine]})
    return {"title": sp["title"], "shots": shots, "scenes": scene_rows, "characters": chars,
            "total_duration": round(t, 2), "warnings": sp["warnings"]}


def manifest_to_srt(manifest: dict) -> str:
    """Dialogue subtitles (SRT) from a manifest, on the exact shot timings."""
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    out, n = [], 0
    for s in manifest["shots"]:
        if s.get("line"):
            n += 1
            out.append(f"{n}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['speaker']}: {s['line']}\n")
    return "\n".join(out)
