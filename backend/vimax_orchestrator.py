"""
vimax_orchestrator.py - a multi-agent creative loop inspired by ViMax's agent architecture, for CinePulse AI.

ViMax turns an idea / script / novel into a film with a team of LLM agents. This module keeps the team and the
hand-offs, but every step has a deterministic CPU implementation so the whole loop runs with NO model, NO API key
and NO GPU; when a Groq or Gemini key is available, the agents ask the LLM for STRUCTURED JSON instead (validated
against a schema, retried with the error message, and replaced by the offline result if the model keeps failing).

    Director            idea + options  -> plan (title, logline, style, scene outline)
    Screenwriter        plan / novel    -> screenplay text            (novel: novel_adapter.py, Novel2Video)
    CharacterExtractor  screenplay      -> character profiles + visual anchors
    StoryboardArtist    screenplay      -> shots with camera prompts, first/last-frame + motion text, exact timing
                                           (script_parser.py, Script2Video)
    ReferenceSelector   shots           -> dependency graph: which anchor portraits / earlier keyframes each shot gets
                                           (dependency_graph.py)
    ConsistencyChecker  storyboard      -> issues (timing, unknown characters, scene / lighting / appearance drift) and
                                           automatic fixes; unresolved errors go back to the StoryboardArtist

Modes: "idea2video", "script2video", "novel2video".  `render_storyboard()` then renders the result on the CPU with
render_engine.py: a consistent painted portrait per character, scene-seeded backdrops, subtitles, and the last
frame of every shot handed to the next one (keyframe continuity), with a drift report at the end.

Honest limits: nothing here generates photographic images or video with a diffusion model (that needs a GPU); the
renderer produces a stylised animatic. The offline Screenwriter is rule-based, not creative; the LLM path is written
against the public Groq / Gemini JSON APIs and tested with a mocked transport (no live call is made by the tests).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # package import (backend.vimax_orchestrator) or plain module (vimax_orchestrator)
    from . import dependency_graph as dg
    from . import novel_adapter as na
    from . import script_parser as sp
except ImportError:  # pragma: no cover
    import dependency_graph as dg
    import novel_adapter as na
    import script_parser as sp

logger = logging.getLogger(__name__)

MODES = ("idea2video", "script2video", "novel2video")
MODEL_STYLES = ("human", "ai", "anime")
MAX_SOURCE_CHARS = 3_000_000
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-2.0-flash"


class LLMError(RuntimeError):
    pass


class SchemaError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# structured JSON helpers
# --------------------------------------------------------------------------- #
def parse_json_loose(text: str) -> object:
    """JSON from an LLM reply: tolerates ```json fences, text around the object and trailing commas."""
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b > a:
        chunk = raw[a:b + 1]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise LLMError("The model did not return valid JSON.")


def check_shape(obj, schema, path: str = "$") -> None:
    """Minimal structural validation. schema: a type (or tuple of types), a dict {key: schema} (a key ending in '?'
    is optional), or [schema] (a list whose items all match). Raises SchemaError naming the offending path."""
    if isinstance(schema, dict):
        if not isinstance(obj, dict):
            raise SchemaError(f"{path} must be an object")
        for key, sub in schema.items():
            optional = key.endswith("?")
            k = key.rstrip("?")
            if k not in obj:
                if optional:
                    continue
                raise SchemaError(f"{path}.{k} is missing")
            check_shape(obj[k], sub, f"{path}.{k}")
    elif isinstance(schema, list):
        if not isinstance(obj, list):
            raise SchemaError(f"{path} must be a list")
        for i, item in enumerate(obj):
            check_shape(item, schema[0], f"{path}[{i}]")
    else:
        types = schema if isinstance(schema, tuple) else (schema,)
        ok_num = float in types or int in types
        if isinstance(obj, bool) and bool not in types:
            raise SchemaError(f"{path} has the wrong type")
        if not isinstance(obj, types) and not (ok_num and isinstance(obj, (int, float)) and not isinstance(obj, bool)):
            if not (type(None) in types and obj is None):
                raise SchemaError(f"{path} has the wrong type (expected {'/'.join(t.__name__ for t in types)})")


# --------------------------------------------------------------------------- #
# LLM clients (Groq / Gemini over plain HTTPS; the key is only ever sent to the provider's own URL)
# --------------------------------------------------------------------------- #
def _http_post_json(url: str, headers: dict, payload: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last: Optional[str] = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300] if hasattr(exc, "read") else ""
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                last = f"HTTP {exc.code}"
                continue
            raise LLMError(f"The AI service returned HTTP {exc.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = str(getattr(exc, "reason", exc))
            if attempt < 2:
                time.sleep(1.0)
                continue
            raise LLMError(f"Could not reach the AI service ({last}).") from None
    raise LLMError(f"The AI service kept failing ({last}).")


class LLMClient:
    provider = "base"
    offline = False

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 transport: Optional[Callable[[str, dict, dict, float], dict]] = None, timeout: float = 60.0, retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.transport = transport or _http_post_json
        self.timeout = timeout
        self.retries = retries
        self.calls = 0

    def _call(self, system: str, user: str, temperature: float, max_tokens: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema=None, temperature: float = 0.4, max_tokens: int = 4096) -> dict:
        """Ask for a JSON object; validate it against `schema`; on a bad answer retry with the error as feedback."""
        feedback = ""
        last_err = "no answer"
        for _ in range(self.retries + 1):
            self.calls += 1
            raw = self._call(system, user + feedback, temperature, max_tokens)
            try:
                data = parse_json_loose(raw)
                if schema is not None:
                    check_shape(data, schema)
                return data  # type: ignore[return-value]
            except (LLMError, SchemaError) as exc:
                last_err = str(exc)
                feedback = f"\n\nYour previous answer was rejected: {last_err}. Reply again with ONLY a valid JSON object that follows the format."
        raise LLMError(f"The model could not produce valid structured output ({last_err}).")


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(self, api_key=None, model=None, **kw):
        super().__init__(api_key or os.environ.get("GROQ_API_KEY"), model or GROQ_MODEL, **kw)
        if not self.api_key:
            raise LLMError("No Groq API key (set GROQ_API_KEY or paste one in the app).")

    def _call(self, system, user, temperature, max_tokens) -> str:
        payload = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        res = self.transport(GROQ_URL, {"Authorization": f"Bearer {self.api_key}"}, payload, self.timeout)
        try:
            return res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError("Unexpected response from Groq.") from None


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key=None, model=None, **kw):
        super().__init__(api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
                         model or GEMINI_MODEL, **kw)
        if not self.api_key:
            raise LLMError("No Gemini API key (set GEMINI_API_KEY or paste one in the app).")

    def _call(self, system, user, temperature, max_tokens) -> str:
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                        "responseMimeType": "application/json"}}
        res = self.transport(GEMINI_URL.format(model=self.model), {"x-goog-api-key": self.api_key}, payload, self.timeout)
        try:
            return res["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            block = (res.get("promptFeedback") or {}).get("blockReason") if isinstance(res, dict) else None
            raise LLMError(f"Gemini returned no text{' (blocked: ' + block + ')' if block else ''}.") from None


class OfflineClient(LLMClient):
    """No LLM: every agent uses its built-in rule-based implementation."""
    provider = "offline"
    offline = True

    def complete_json(self, *a, **k):
        raise LLMError("Offline mode has no language model.")


def make_client(provider: str = "auto", api_key: Optional[str] = None, model: Optional[str] = None, **kw) -> LLMClient:
    """provider: "auto" (Groq if GROQ_API_KEY, else Gemini if GEMINI_API_KEY, else offline), "groq", "gemini", "offline"."""
    p = (provider or "auto").lower()
    if p not in ("auto", "groq", "gemini", "offline"):
        raise ValueError("provider must be auto, groq, gemini or offline")
    if p == "offline":
        return OfflineClient()
    if p == "groq" or (p == "auto" and (os.environ.get("GROQ_API_KEY") or (api_key and p == "groq"))):
        return GroqClient(api_key, model, **kw)
    if p == "gemini" or (p == "auto" and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))):
        return GeminiClient(api_key, model, **kw)
    return OfflineClient()


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class Agent:
    name = "agent"

    def __init__(self, llm: LLMClient, trace: List[dict], warnings: List[str]):
        self.llm, self.trace, self.warnings = llm, trace, warnings

    def _try_llm(self, system: str, user: str, schema, note_ok: str = "") -> Tuple[Optional[dict], str]:
        """-> (data or None, note). None means "use the offline implementation"."""
        if self.llm.offline:
            return None, "rule-based (offline)"
        try:
            return self.llm.complete_json(system, user, schema), f"{self.llm.provider}: {note_ok or 'structured JSON'}"
        except LLMError as exc:
            self.warnings.append(f"{self.name}: {exc} Used the built-in rules instead.")
            return None, f"{self.llm.provider} failed; rule-based fallback"

    def _log(self, started: float, used_llm: bool, note: str, ok: bool = True) -> None:
        self.trace.append({"agent": self.name, "seconds": round(time.time() - started, 3), "used_llm": used_llm,
                           "ok": ok, "note": note})


class Director(Agent):
    name = "Director"
    SCHEMA = {"title": str, "logline": str, "genre?": str, "tone?": str, "visual_style?": str, "style?": str,
              "scenes": [{"title": str, "location": str, "summary": str, "characters?": [str]}]}

    def run(self, source: str, mode: str, style: str, target_seconds: float, aspect: str) -> dict:
        t0 = time.time()
        sentences = sp._sentences(re.sub(r"\s+", " ", source)) if mode == "idea2video" else []
        n = max(1, min(8, round(target_seconds / 12)))
        sys_p = ("You are the Director of a short-film team. Plan the film. Reply with ONLY JSON: "
                 '{"title","logline","genre","tone","visual_style","style":"human|ai|anime",'
                 '"scenes":[{"title","location","summary","characters":[names]}]}. '
                 f"Use about {n} scenes for a {int(target_seconds)}-second {aspect} video.")
        data, note = (None, "rule-based (offline)")
        if mode == "idea2video":
            data, note = self._try_llm(sys_p, f"IDEA:\n{source[:4000]}", self.SCHEMA)
        if data is None:
            title = _slug(" ".join(re.findall(r"[\w'’-]+", source)[:6])).title() if source.strip() else "Untitled"
            if mode == "script2video":
                parsed = sp.parse_script(source)
                scenes = [{"title": s["heading"].title(), "location": s["location"].title(), "summary": "", "characters": []}
                          for s in parsed["scenes"]]
                title = parsed["title"] or title
                logline = f"A {len(scenes)}-scene script."
            elif mode == "novel2video":
                scenes, logline = [], "An adaptation of a novel."
                m = re.search(r"^(?:title\s*:\s*)?(.{3,80})$", source.strip().splitlines()[0] if source.strip() else "")
                title = m.group(1).strip() if m and len(source.split()) > 50 else title
            else:
                per = max(1, -(-len(sentences) // n)) if sentences else 1
                scenes = []
                for i in range(0, len(sentences), per):
                    chunk = sentences[i:i + per]
                    loc = next((p.title() for p in na.PLACES if re.search(rf"\b{p}\b", " ".join(chunk), re.IGNORECASE)), "Location")
                    scenes.append({"title": f"Scene {len(scenes) + 1}", "location": loc, "summary": " ".join(chunk)[:160], "characters": []})
                logline = sentences[0][:160] if sentences else ""
            low = source.casefold()
            genre = next((g for g, kw in (("thriller", ("murder", "chase", "gun", "danger")), ("romance", ("love", "kiss", "heart")),
                                          ("comedy", ("funny", "joke", "laugh")), ("sci-fi", ("robot", "space", "future", "alien")),
                                          ("fantasy", ("dragon", "wizard", "magic", "kingdom"))) if any(k in low for k in kw)), "drama")
            data = {"title": title, "logline": logline, "genre": genre, "tone": "cinematic",
                    "visual_style": "anime" if style == "anime" else "cinematic realism", "style": style, "scenes": scenes}
        used = data is not None and not self.llm.offline and note.startswith(self.llm.provider + ":")
        data["style"] = data.get("style") if data.get("style") in MODEL_STYLES else style
        if style in MODEL_STYLES:
            data["style"] = style  # the user's choice wins over the model's
        data.update({"mode": mode, "target_seconds": float(target_seconds), "aspect": aspect})
        self._log(t0, used, note)
        return data


class Screenwriter(Agent):
    name = "Screenwriter"
    SCHEMA = {"script": str}

    def run(self, source: str, mode: str, plan: dict, episode: int, episodes: Optional[int], seconds_per_episode: float,
            embedder: str, progress=None) -> dict:
        t0 = time.time()
        extra: dict = {"novel": None}
        if mode == "script2video":
            script, note, used = source, "passed through (already a script)", False
        elif mode == "novel2video":
            n_eps = episodes or max(1, int(round(len(source.split()) / 1800)))
            ad = na.adapt_novel(source, episodes=n_eps, seconds_per_episode=seconds_per_episode, embedder=embedder, progress=progress)
            idx = min(max(1, episode), len(ad["episodes"])) - 1
            ep = ad["episodes"][idx]
            script, note, used = ep["script"], f"novel_adapter ({ad['embedder']}) episode {idx + 1}/{len(ad['episodes'])}", False
            extra["novel"] = {"characters": ad["characters"], "chapters": ad["chapters"], "episodes": len(ad["episodes"]),
                              "episode": idx + 1, "embedder": ad["embedder"], "beats": ad["beats"]}
            self.warnings.extend(ad["warnings"])
            data, n2 = self._try_llm(
                "You are a screenwriter. Rewrite the rough adaptation as a tight screenplay in standard format "
                "(INT./EXT. headings, CHARACTER cues, parentheticals, dialogue). Keep every character name and the order "
                f"of events. Reply with ONLY JSON: {{\"script\": \"...\"}}. Aim for about {int(seconds_per_episode * 2.6)} words.",
                f"ROUGH ADAPTATION:\n{script[:9000]}", self.SCHEMA)
            if data and self._valid(data["script"]):
                script, note, used = data["script"], f"{n2}; polished the extractive adaptation", True
        else:
            words = int(plan["target_seconds"] * 2.6)
            data, note = self._try_llm(
                "You are a screenwriter. Write a short screenplay for the plan in standard format (INT./EXT. headings with "
                "time of day, CHARACTER cues in capitals, parentheticals, dialogue, action). Reply with ONLY JSON: "
                f"{{\"script\": \"...\"}}. About {words} words.", "PLAN:\n" + json.dumps(plan)[:3000] + "\n\nIDEA:\n" + source[:3000], self.SCHEMA)
            if data and self._valid(data["script"]):
                script, used = data["script"], True
            else:
                if data:
                    self.warnings.append("Screenwriter: the model's script could not be parsed; used the built-in rules.")
                script, used = self._offline_idea(source, plan), False
                note = "rule-based: the idea's sentences become scenes (not creative writing)"
        self._log(t0, used, note)
        return {"script": script.strip() + "\n", **extra}

    @staticmethod
    def _valid(script: str) -> bool:
        try:
            return bool(sp.script_to_shots(script)["shots"])
        except (sp.ScriptParseError, ValueError):
            return False

    @staticmethod
    def _offline_idea(idea: str, plan: dict) -> str:
        sentences = sp._sentences(re.sub(r"\s+", " ", idea)) or [idea.strip() or "A story begins."]
        scenes = plan.get("scenes") or [{"location": "Location"}]
        per = max(1, -(-len(sentences) // len(scenes)))
        caps = re.findall(r"\b([A-Z][a-z]{2,})\b", idea)
        mid = set(re.findall(r"(?<=[a-z,] )([A-Z][a-z]{2,})\b", idea))
        names = [n for n in dict.fromkeys(caps)
                 if n not in na.NOT_NAMES and (n in mid or caps.count(n) >= 2)]
        out = []
        last_head = None
        last_place, last_tod = "Location", "DAY"
        for si, sc in enumerate(scenes):
            chunk = sentences[si * per:(si + 1) * per]
            if not chunk:
                break
            text = " ".join(chunk)
            place = sc.get("location") or "Location"
            if place == "Location":
                place = last_place  # no place named: stay where the story already is
            ie = next((v for k, v in na.PLACES.items() if re.search(rf"\b{k}\b", text, re.IGNORECASE)), "EXT" if "EXT" in place.upper() else "INT")
            tod = na._tod_of(text, last_tod)
            last_place, last_tod = place, tod
            head = f"{ie}. {place.upper()} - {tod}"
            if head != last_head:  # only a new heading when the place or time really changes
                out += [head, ""]
                last_head = head
            for s in chunk:
                q = re.findall(r"[\"“]([^\"”]+)[\"”]", s)
                speaker = next((n for n in names if re.search(rf"\b{n}\b", s)), None)
                if q and speaker:
                    rest = re.sub(r"[\"“][^\"”]+[\"”]", " ", s)
                    rest = re.sub(rf"\b{speaker}\b|\b(?:says|said|asks|asked|replies|replied|whispers|whispered|shouts|shouted)\b|[,.!?]", " ", rest).strip()
                    if len(rest.split()) > 2:  # keep real action that shares the sentence with the quote
                        out += [re.sub(r"[\"“][^\"”]+[\"”],?", "", s).strip(" ,"), ""]
                    out += [speaker.upper(), q[0].strip().rstrip(",") or q[0], ""]
                else:
                    out += [s.strip(), ""]
            if not names:
                out += ["NARRATOR (V.O.)", " ".join(chunk), ""]
        return "\n".join(out)


class CharacterExtractor(Agent):
    name = "CharacterExtractor"
    SCHEMA = {"characters": [{"name": str, "description": str, "gender?": str, "age?": (int, float, str, type(None)),
                              "personality?": str, "visual_identifier": str, "outfit?": str}]}

    def run(self, script: str, parsed: dict, plan: dict, novel_profiles: Optional[List[dict]]) -> List[dict]:
        t0 = time.time()
        style = plan.get("style", "human")
        names = parsed["characters"]
        data, note = self._try_llm(
            "You are a character designer. For every named character in the screenplay give a short description, gender, age, "
            "personality and a concrete visual_identifier (hair, face, build, clothing) that an artist could draw identically "
            'in every shot. Reply with ONLY JSON: {"characters":[{"name","description","gender","age","personality","visual_identifier","outfit"}]}. '
            f"Characters: {', '.join(names)}. Visual style: {plan.get('visual_style', '')}.", "SCREENPLAY:\n" + script[:7000], self.SCHEMA)
        llm_by_name = {}
        if data:
            llm_by_name = {self._canon(c["name"], names): c for c in data["characters"] if self._canon(c["name"], names)}
        novel_by_short = {(p.get("short") or na.short_name(p["name"])).casefold(): p for p in (novel_profiles or [])}
        action_text = " ".join(e["text"] for sc in parsed["scenes"] for e in sc["elements"] if e["type"] == "action")
        sents = na.split_sentences(action_text)
        out = []
        for name in names:
            prof = novel_by_short.get(name.casefold())
            mine = [s for s in sents if re.search(rf"\b{re.escape(name)}\b", s, re.IGNORECASE)]
            he = sum(len(re.findall(r"\b(?:he|his|him)\b", s, re.I)) for s in mine)
            she = sum(len(re.findall(r"\b(?:she|her|hers)\b", s, re.I)) for s in mine)
            base = {"name": name, "short": name, "aliases": [], "mentions": len(mine),
                    "gender": (prof or {}).get("gender") or ("male" if he > she * 1.3 else "female" if she > he * 1.3 else "unknown"),
                    "age": (prof or {}).get("age") or next((na._age_in(s) for s in mine if na._age_in(s)), None),
                    "traits": list((prof or {}).get("traits", []))}
            if not base["traits"]:
                for s in mine:
                    for m in re.finditer(r"\b(\w+)\s+(hair|eyes)\b", s):
                        if m.group(1).casefold() in na.COLORS + na.EYE_COLORS:
                            base["traits"].append(f"{m.group(1).lower()} {m.group(2)}")
            description = (prof or {}).get("description") or (_slug(mine[0])[:200] if mine else "")
            anchor = na.visual_anchor(base)
            ll = llm_by_name.get(name)
            if ll:
                if isinstance(ll.get("gender"), str) and ll["gender"].lower() in ("male", "female"):
                    base["gender"] = ll["gender"].lower()
                anchor = na.visual_anchor(base)
                anchor["identifier"] = _slug(ll["visual_identifier"])
                description = ll["description"]
            out.append({"name": name, "short": name, "description": description, "gender": base["gender"], "age": base["age"],
                        "personality": (ll or {}).get("personality", ""), "visual_identifier": anchor["identifier"],
                        "anchor": {k: anchor[k] for k in ("hair", "eyes", "skin", "outfit", "seed")}, "style": style,
                        "portrait_views": ["front", "side", "back"], "mentions": len(mine)})
        self._log(t0, bool(llm_by_name), note if llm_by_name or self.llm.offline else note + " (not usable)")
        return out

    @staticmethod
    def _canon(name: str, names: Sequence[str]) -> Optional[str]:
        for n in names:
            if name.strip().casefold() == n.casefold() or name.strip().casefold().split()[0:1] == [n.casefold()]:
                return n
        return None


class StoryboardArtist(Agent):
    name = "StoryboardArtist"
    SCHEMA = {"shots": [{"idx": int, "camera_prompt?": str, "visual_desc?": str, "motion_desc?": str, "ff_desc?": str,
                         "lf_desc?": str, "camera_angle?": str, "camera_movement?": str, "duration?": (int, float)}]}
    TEXT_FIELDS = ("camera_prompt", "visual_desc", "motion_desc", "ff_desc", "lf_desc", "camera_angle", "camera_movement")

    def run(self, script: str, characters: List[dict], plan: dict, max_shots: Optional[int]) -> dict:
        t0 = time.time()
        manifest = sp.script_to_shots(script)
        shots = manifest["shots"]
        self.warnings.extend(manifest["warnings"])
        if max_shots and len(shots) > max_shots:
            self.warnings.append(f"The storyboard has {len(shots)} shots; kept the first {max_shots} (max_shots).")
            shots = shots[:max_shots]
            shots[-1]["is_last"] = True
        note, used = "rule-based: script_parser shot manifest with exact timing", False
        refined = self._refine(shots, characters, plan)
        if refined:
            used, note = True, f"{self.llm.provider}: refined camera prompts / descriptions for {refined} shots"
        self._retime(shots)
        self._log(t0, used, note)
        return {"shots": shots, "scenes": manifest["scenes"], "characters": manifest["characters"],
                "total_duration": shots[-1]["end"] if shots else 0.0}

    def _refine(self, shots: List[dict], characters: List[dict], plan: dict, issues: Optional[List[dict]] = None) -> int:
        if self.llm.offline:
            return 0
        looks = {c["name"]: c["visual_identifier"] for c in characters}
        count = 0
        system = ("You are a storyboard artist. For each shot write a vivid camera_prompt, a visual_desc that names characters as "
                  "<Name> and repeats their look exactly as given, motion_desc, ff_desc (first frame) and lf_desc (last frame). "
                  'Reply with ONLY JSON: {"shots":[{"idx","camera_prompt","visual_desc","motion_desc","ff_desc","lf_desc"}]}. '
                  "Do not invent characters. Keep idx values.")
        targets = [s for s in shots if not issues or s["idx"] in {i["shot"] for i in issues if i.get("shot") is not None}]
        for i in range(0, len(targets), 12):
            chunk = targets[i:i + 12]
            brief = [{k: s.get(k) for k in ("idx", "scene_idx", "shot_type", "camera_angle", "camera_movement", "characters", "speaker",
                                              "line", "action", "location", "time_of_day", "duration")} for s in chunk]
            user = ("LOOKS: " + json.dumps(looks) + "\nPLAN: " + json.dumps({k: plan.get(k) for k in ("title", "genre", "tone", "visual_style")})
                    + "\nSHOTS: " + json.dumps(brief)
                    + ("\nFIX THESE PROBLEMS: " + json.dumps([x["message"] for x in issues]) if issues else ""))
            try:
                data = self.llm.complete_json(system, user, self.SCHEMA)
            except LLMError as exc:
                self.warnings.append(f"StoryboardArtist: {exc} Kept the rule-based descriptions.")
                break
            by_idx = {s["idx"]: s for s in chunk}
            for r in data["shots"]:
                s = by_idx.get(r["idx"])
                if not s:
                    continue
                for f in self.TEXT_FIELDS:
                    if isinstance(r.get(f), str) and r[f].strip():
                        s[f] = r[f].strip()
                if isinstance(r.get("duration"), (int, float)) and 0.8 <= r["duration"] <= 12:
                    s["duration"] = round(float(r["duration"]), 2)
                count += 1
        return count

    @staticmethod
    def _retime(shots: List[dict]) -> None:
        t = 0.0
        for s in shots:
            s["start"] = round(t, 2)
            t = round(t + s["duration"], 2)
            s["end"] = t

    def revise(self, shots: List[dict], characters: List[dict], plan: dict, issues: List[dict]) -> int:
        n = self._refine(shots, characters, plan, issues)
        self._retime(shots)
        return n


class ReferenceSelector(Agent):
    name = "ReferenceSelector"

    def run(self, shots: List[dict], characters: List[dict]) -> Tuple[dg.DependencyGraph, List[dict]]:
        t0 = time.time()
        graph = dg.build_from_storyboard(shots, characters)
        plan = dg.render_plan(graph)
        by = {p["shot"]: p for p in plan}
        for s in shots:
            p = by[s["idx"]]
            s["references"] = p["inputs"]
            s["continues_from"] = p["continues_from"]
        self._log(t0, False, f"dependency graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph, plan


class ConsistencyChecker(Agent):
    name = "ConsistencyChecker"
    HAIR = tuple(sorted(set(na.COLORS) | {"dark brown"}, key=len, reverse=True))

    def run(self, shots: List[dict], characters: List[dict], graph: dg.DependencyGraph, fix: bool = True) -> dict:
        t0 = time.time()
        issues = self.check(shots, characters, graph)
        fixed = self.apply_auto_fixes(shots, characters, issues) if fix else 0
        if fixed:
            issues = self.check(shots, characters, graph)
        errors = [i for i in issues if i["severity"] == "error"]
        self._log(t0, False, f"{len(errors)} error(s), {len(issues) - len(errors)} warning(s) after {fixed} automatic fix(es)")
        return {"ok": not errors, "issues": issues, "fixed": fixed}

    def check(self, shots: List[dict], characters: List[dict], graph: dg.DependencyGraph) -> List[dict]:
        issues: List[dict] = []

        def add(sev, code, shot, msg, fixable=False):
            issues.append({"severity": sev, "code": code, "shot": shot, "message": msg, "auto_fixable": fixable})

        known = {c["name"] for c in characters}
        anchors = {c["name"]: c for c in characters}
        t = 0.0
        for k, s in enumerate(shots):
            if s["idx"] != k:
                add("error", "E_IDX", s["idx"], f"Shot numbering is not sequential (expected {k}, got {s['idx']}).", True)
            if s["duration"] <= 0:
                add("error", "E_TIMING", s["idx"], "A shot has no duration.", False)
            if abs(s["start"] - t) > 0.011 or abs(s["end"] - (s["start"] + s["duration"])) > 0.011:
                add("error", "E_TIMING", s["idx"], f"Shot {s['idx']} does not start where the previous shot ends.", True)
            t = s["end"]
            for c in s.get("characters", []):
                if c not in known:
                    canon = next((n for n in known if n.casefold() == c.casefold()), None)
                    add("error", "E_UNKNOWN_CHAR", s["idx"], f"Shot {s['idx']} shows '{c}', who has no character profile.", canon is not None)
            for c in s.get("characters", []):
                if c in anchors and not anchors[c].get("visual_identifier"):
                    add("error", "E_NO_ANCHOR", s["idx"], f"'{c}' has no visual anchor, so their look can drift.", False)
            for c in s.get("characters", []):
                if c in known and f"<{c}>" not in s.get("visual_desc", ""):
                    add("warning", "W_TAG", s["idx"], f"Shot {s['idx']}'s description does not reference <{c}>.", True)
            if len(s.get("references", [])) > dg.MAX_REFERENCES:
                add("warning", "W_REFS", s["idx"], f"Shot {s['idx']} has more than {dg.MAX_REFERENCES} references.", False)
            for c in s.get("characters", []):
                a = anchors.get(c)
                if not a:
                    continue
                hair = a["anchor"].get("hair")
                m = re.search(rf"<{re.escape(c)}>[^.<]{{0,70}}?\b({'|'.join(map(re.escape, self.HAIR))})\s+hair", s.get("visual_desc", ""), re.IGNORECASE)
                if m and hair and m.group(1).casefold() != str(hair).casefold():
                    add("warning", "W_APPEARANCE", s["idx"],
                        f"Shot {s['idx']} gives <{c}> {m.group(1)} hair but their anchor has {hair} hair.", True)
        by_scene: Dict[int, dict] = {}
        for s in shots:
            first = by_scene.setdefault(s["scene_idx"], s)
            if (s["location"], s["time_of_day"]) != (first["location"], first["time_of_day"]):
                add("warning", "W_SCENE_DRIFT", s["idx"], f"Shot {s['idx']} changes location / time of day inside scene {s['scene_idx']}.", True)
            exp = sp.LIGHTING.get(s["time_of_day"])
            if exp and s.get("lighting") != exp:
                add("warning", "W_LIGHTING", s["idx"], f"Shot {s['idx']}'s lighting does not match {s['time_of_day']}.", True)
        if shots and [s["idx"] for s in shots if s.get("is_last")] != [shots[-1]["idx"]]:
            add("error", "E_LAST", shots[-1]["idx"], "Exactly the final shot must be marked as last.", True)
        if graph.has_cycle():
            add("error", "E_CYCLE", None, "The reference graph contains a cycle.", False)
        return issues

    def apply_auto_fixes(self, shots: List[dict], characters: List[dict], issues: List[dict]) -> int:
        known = {c["name"]: c for c in characters}
        n = 0
        codes = {i["code"] for i in issues if i["auto_fixable"]}
        if "E_IDX" in codes:
            for k, s in enumerate(shots):
                s["idx"] = k
            n += 1
        if "E_TIMING" in codes:
            StoryboardArtist._retime(shots)
            n += 1
        if "E_LAST" in codes:
            for s in shots:
                s["is_last"] = s is shots[-1]
            n += 1
        by_scene: Dict[int, dict] = {}
        for s in shots:
            first = by_scene.setdefault(s["scene_idx"], s)
            if "W_SCENE_DRIFT" in codes and (s["location"], s["time_of_day"]) != (first["location"], first["time_of_day"]):
                s["location"], s["time_of_day"] = first["location"], first["time_of_day"]
                n += 1
            exp = sp.LIGHTING.get(s["time_of_day"])
            if "W_LIGHTING" in codes and exp and s.get("lighting") != exp:
                s["lighting"] = exp
                n += 1
            for k, c in enumerate(list(s.get("characters", []))):
                if c not in known:
                    canon = next((x for x in known if x.casefold() == c.casefold()), None)
                    if canon:
                        s["characters"][k] = canon
                        n += 1
            for c in s.get("characters", []):
                if c not in known:
                    continue
                if f"<{c}>" not in s.get("visual_desc", ""):
                    desc = re.sub(rf"(?<![<\w]){re.escape(c)}(?![>\w])", f"<{c}>", s.get("visual_desc", ""))
                    if f"<{c}>" not in desc:
                        desc = f"{desc} (<{c}>: {known[c]['visual_identifier']})".strip()
                    s["visual_desc"] = desc
                    n += 1
                hair = known[c]["anchor"].get("hair")
                if hair:
                    new = re.sub(rf"(<{re.escape(c)}>[^.<]{{0,70}}?)\b({'|'.join(map(re.escape, self.HAIR))})(\s+hair)",
                                 lambda m: m.group(1) + (hair if m.group(2).casefold() != str(hair).casefold() else m.group(2)) + m.group(3),
                                 s.get("visual_desc", ""), flags=re.IGNORECASE)
                    if new != s.get("visual_desc"):
                        s["visual_desc"] = new
                        n += 1
        return n


# --------------------------------------------------------------------------- #
# the orchestration loop
# --------------------------------------------------------------------------- #
class ViMaxOrchestrator:
    def __init__(self, llm: Optional[LLMClient] = None, max_rounds: int = 3, progress: Optional[Callable[[float, str], None]] = None):
        self.llm = llm or OfflineClient()
        self.max_rounds = max(1, int(max_rounds))
        self.progress = progress

    def _p(self, f: float, msg: str) -> None:
        if self.progress:
            try:
                self.progress(f, msg)
            except Exception:  # noqa: BLE001 - UI callbacks must never break the run
                pass

    def run(self, source: str, mode: str = "idea2video", style: str = "human", target_seconds: float = 45.0, episode: int = 1,
            episodes: Optional[int] = None, seconds_per_episode: Optional[float] = None, max_shots: Optional[int] = None,
            aspect: str = "9:16", embedder: str = "auto") -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        if style not in MODEL_STYLES:
            raise ValueError(f"style must be one of: {', '.join(MODEL_STYLES)}")
        source = str(source or "")
        if not source.strip():
            raise ValueError("Please provide an idea, a script or a novel text.")
        if len(source) > MAX_SOURCE_CHARS:
            raise ValueError(f"The text is too long ({len(source):,} characters; limit {MAX_SOURCE_CHARS:,}).")
        trace: List[dict] = []
        warnings: List[str] = []
        started = time.time()
        self._p(0.02, "Director: planning")
        plan = Director(self.llm, trace, warnings).run(source, mode, style, target_seconds, aspect)
        self._p(0.12, "Screenwriter: writing the script")
        spw = float(seconds_per_episode or target_seconds)
        sw = Screenwriter(self.llm, trace, warnings).run(source, mode, plan, episode, episodes, spw, embedder,
                                                         progress=lambda f: self._p(0.12 + 0.2 * f, "Screenwriter: reading the novel"))
        script = sw["script"]
        try:
            parsed = sp.parse_script(script)
        except sp.ScriptParseError as exc:
            raise ValueError(f"The script could not be read: {exc}") from None
        self._p(0.40, "Character extractor: building character anchors")
        characters = CharacterExtractor(self.llm, trace, warnings).run(
            script, parsed, plan, (sw["novel"] or {}).get("characters"))
        self._p(0.55, "Storyboard artist: planning shots")
        artist = StoryboardArtist(self.llm, trace, warnings)
        sb = artist.run(script, characters, plan, max_shots)
        shots = sb["shots"]
        selector = ReferenceSelector(self.llm, trace, warnings)
        checker = ConsistencyChecker(self.llm, trace, warnings)
        rounds, report = 0, {"ok": True, "issues": [], "fixed": 0}
        graph = None
        while True:
            rounds += 1
            self._p(0.70 + 0.05 * rounds, f"Reference selector + consistency checker (round {rounds})")
            graph, rplan = selector.run(shots, characters)
            report = checker.run(shots, characters, graph, fix=True)
            errors = [i for i in report["issues"] if i["severity"] == "error"]
            if not errors or rounds >= self.max_rounds:
                break
            if self.llm.offline:
                break  # nothing more the rules can do
            if not artist.revise(shots, characters, plan, errors):
                break
        remaining = [i for i in report["issues"]]
        if remaining:
            warnings.append(f"{len(remaining)} consistency note(s) remain; see 'issues'.")
        sb["total_duration"] = shots[-1]["end"] if shots else 0.0
        self._p(1.0, "Done")
        return {"mode": mode, "plan": plan, "script": script, "characters": characters,
                "storyboard": {"shots": shots, "scenes": sb["scenes"], "total_duration": sb["total_duration"]},
                "graph": graph.to_dict(), "render_plan": dg.render_plan(graph), "issues": report["issues"],
                "rounds": rounds, "novel": sw["novel"],
                "llm": {"provider": self.llm.provider, "offline": self.llm.offline, "calls": self.llm.calls},
                "trace": trace, "warnings": warnings, "seconds": round(time.time() - started, 2)}


def save_project(result: dict, folder: str) -> List[str]:
    """Write the plan as files: project.json, script.fountain, storyboard.json, characters.json, graph.json, graph.mmd, subtitles.srt."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    files = {"project.json": json.dumps({k: v for k, v in result.items() if k not in ("graph", "render_plan")}, indent=2, ensure_ascii=False),
             "script.fountain": result["script"],
             "storyboard.json": json.dumps(result["storyboard"], indent=2, ensure_ascii=False),
             "characters.json": json.dumps(result["characters"], indent=2, ensure_ascii=False),
             "graph.json": json.dumps(result["graph"], indent=2, ensure_ascii=False),
             "graph.mmd": dg.DependencyGraph.from_dict(result["graph"]).to_mermaid(),
             "subtitles.srt": sp.manifest_to_srt(result["storyboard"])}
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    return sorted(files)


# --------------------------------------------------------------------------- #
# CPU renderer: an animatic with consistent characters
# --------------------------------------------------------------------------- #
_COLORS = {  # BGR
    "black": (30, 28, 28), "dark brown": (30, 45, 70), "brown": (45, 70, 110), "chestnut": (40, 65, 120), "auburn": (35, 55, 130),
    "blonde": (110, 170, 215), "blond": (110, 170, 215), "fair": (140, 190, 225), "red": (40, 50, 170), "ginger": (50, 100, 190),
    "grey": (150, 150, 150), "gray": (150, 150, 150), "white": (235, 235, 235), "silver": (190, 190, 195), "golden": (80, 175, 225),
    "dark": (35, 35, 40), "blue": (170, 110, 50), "green": (90, 140, 60), "hazel": (60, 110, 130), "amber": (40, 140, 210),
    "navy": (90, 50, 30), "olive": (60, 120, 110), "burgundy": (50, 30, 120), "charcoal": (70, 65, 62), "mustard": (50, 190, 225),
    "forest-green": (60, 100, 40), "teal": (130, 130, 40), "pale": (210, 220, 235), "tan": (120, 170, 205), "light olive": (130, 180, 200),
    "warm brown": (70, 110, 160), "deep brown": (45, 70, 105)}


def _color(word: str, default=(120, 120, 120)):
    w = str(word or "").casefold().strip()
    for key in sorted(_COLORS, key=len, reverse=True):
        if w.startswith(key) or key in w.split()[0:2]:
            return _COLORS[key]
    return default


def make_portrait(character: dict, path: str, size: Tuple[int, int] = (360, 480)) -> str:
    """Paint a deterministic front-view anchor portrait from the character's visual anchor (hair / skin / outfit /
    eye colours). The same anchor always produces the same picture, which is what keeps a character recognisable
    in every shot. Style "anime" uses flat cel colours and large eyes; "human" / "ai" use soft shading."""
    import cv2
    import numpy as np

    w, h = size
    a = character["anchor"]
    style = character.get("style", "human")
    rng = np.random.default_rng(int(a.get("seed", 1)))
    bg1, bg2 = np.array((60 + int(rng.integers(40)), 50 + int(rng.integers(30)), 45 + int(rng.integers(30))), np.float32), np.array((25, 22, 30), np.float32)
    img = (bg1[None, None, :] * np.linspace(1, 0, h)[:, None, None] + bg2[None, None, :] * np.linspace(0, 1, h)[:, None, None]).repeat(w, axis=1).astype(np.uint8)
    skin, hair, outfit = _color(a.get("skin"), (150, 180, 220)), _color(a.get("hair")), _color(a.get("outfit"), (90, 90, 90))
    eye = _color(a.get("eyes") or "dark", (40, 35, 30))
    cx = w // 2
    cv2.ellipse(img, (cx, int(h * 1.02)), (int(w * 0.46), int(h * 0.32)), 0, 180, 360, outfit, -1, cv2.LINE_AA)      # shoulders
    cv2.rectangle(img, (cx - int(w * 0.07), int(h * 0.60)), (cx + int(w * 0.07), int(h * 0.74)), skin, -1)            # neck
    cv2.ellipse(img, (cx, int(h * 0.40)), (int(w * 0.24), int(h * 0.20)), 0, 0, 360, hair, -1, cv2.LINE_AA)           # hair back
    cv2.ellipse(img, (cx, int(h * 0.43)), (int(w * 0.20), int(h * 0.17)), 0, 0, 360, skin, -1, cv2.LINE_AA)           # face
    cv2.ellipse(img, (cx, int(h * 0.30)), (int(w * 0.215), int(h * 0.105)), 0, 180, 360, hair, -1, cv2.LINE_AA)       # fringe
    ey, ex, er = int(h * 0.43), int(w * 0.085), int(w * (0.032 if style != "anime" else 0.048))
    for sx in (-1, 1):
        cv2.circle(img, (cx + sx * ex, ey), er + 3, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + sx * ex, ey), er, eye, -1, cv2.LINE_AA)
        cv2.circle(img, (cx + sx * ex, ey), max(2, er // 2), (20, 20, 20), -1, cv2.LINE_AA)
        if style == "anime":
            cv2.circle(img, (cx + sx * ex - er // 3, ey - er // 3), max(2, er // 4), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(img, (cx + sx * ex - er - 4, ey - er - 6), (cx + sx * ex + er + 4, ey - er - 8 * sx), hair, 3, cv2.LINE_AA)  # brow
    cv2.line(img, (cx, ey + er), (cx - 4, int(h * 0.50)), tuple(int(c * 0.8) for c in skin), 2, cv2.LINE_AA)           # nose
    cv2.ellipse(img, (cx, int(h * 0.545)), (int(w * 0.06), int(h * 0.016)), 0, 0, 180, (60, 60, 150), 2, cv2.LINE_AA)  # mouth
    if style != "anime":  # soft shading
        shade = cv2.GaussianBlur(img, (0, 0), 6)
        img = cv2.addWeighted(img, 0.78, shade, 0.22, 0)
    cv2.rectangle(img, (0, int(h * 0.90)), (w, h), (20, 20, 24), -1)
    label = str(character.get("name", ""))[:18].upper()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
    cv2.putText(img, label, ((w - tw) // 2, int(h * 0.90) + (int(h * 0.10) + th) // 2 - 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Could not write the portrait: {path}")
    return str(path)


def _backdrop_style(shot: dict) -> str:
    tod = (shot.get("time_of_day") or "").upper()
    if tod in ("NIGHT", "MIDNIGHT", "DUSK", "EVENING"):
        return "night"
    if shot.get("int_ext") == "EXT":
        return "day"
    return "interior"


def _last_frame(video: str, out_png: str) -> str:
    import cv2

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame = None
    if n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 2)
        ok, frame = cap.read()
        frame = frame if ok else None
    if frame is None:
        cap.release()
        cap = cv2.VideoCapture(video)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Could not read the rendered shot.")
    cv2.imwrite(out_png, frame)
    return out_png


def render_storyboard(result: dict, output_path: str, work_dir: Optional[str] = None, size: Tuple[int, int] = (540, 960),
                      fps: int = 20, voice: bool = False, voice_name: str = "en-US-AriaNeural", max_shots: Optional[int] = None,
                      ffmpeg_path: Optional[str] = None, progress: Optional[Callable[[float, str], None]] = None,
                      drift_threshold: float = 0.35) -> dict:
    """Render the storyboard on the CPU: per-shot clips (scene-seeded backdrop, character portraits, subtitles,
    optional TTS voice), the last frame of each shot handed to the next as its starting keyframe, then joined.
    Returns {"output_path","duration","shots","drift","keyframes","warnings","voice_used"}."""
    import cv2
    import subprocess
    import tempfile

    try:
        from . import render_engine as re_
        from . import video_engine as ve
    except ImportError:  # pragma: no cover
        import render_engine as re_
        import video_engine as ve

    shots = list(result["storyboard"]["shots"])[: max_shots or None]
    if not shots:
        raise ValueError("The storyboard has no shots.")
    ff = ve.find_ffmpeg(ffmpeg_path)
    graph = dg.DependencyGraph.from_dict(result["graph"])
    store = dg.KeyframeStore(graph)
    warnings: List[str] = []
    work = Path(work_dir or tempfile.mkdtemp(prefix="story_"))
    work.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def say(f, m):
        if progress:
            progress(f, m)

    # 1. anchor portraits + their appearance signatures
    chars = {c["name"]: c for c in result["characters"]}
    anchors_sig, portrait = {}, {}
    for name, c in chars.items():
        portrait[name] = make_portrait(c, str(work / f"portrait_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.png"))
        store.put(f"ref:{name}", portrait[name])
        anchors_sig[name] = dg.appearance_signature(portrait[name])
    side = {n: ("Bottom-Left" if i % 2 == 0 else "Bottom-Right") for i, n in enumerate(chars)}  # a character always sits on the same side

    # 2. render in dependency order
    plan = {p["shot"]: p for p in result["render_plan"]}
    order = sorted((s for s in shots), key=lambda s: (plan[s["idx"]]["level"], s["idx"]))
    clips, observations, voice_used = {}, [], False
    tts = None
    if voice:
        try:
            try:
                from . import tts_engine as tts
            except ImportError:  # pragma: no cover
                import tts_engine as tts
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Voice is unavailable ({exc}); rendering silent shots.")
            tts = None
    for n, s in enumerate(order):
        say(0.05 + 0.85 * n / len(order), f"Rendering shot {s['idx'] + 1} of {len(shots)}")
        dur = float(s["duration"])
        audio = None
        text = s.get("line") if s.get("line") else None
        if tts and text:
            try:
                audio = str(work / f"voice_{s['idx']}.mp3")
                tts.generate_voiceover_sync(text, audio, voice=voice_name)
                length = float(ve._probe_media(ff, audio).get("duration") or 0)
                if length > dur:
                    dur = round(length + 0.15, 2)
                voice_used = True
            except Exception as exc:  # noqa: BLE001 - offline TTS must not kill the render
                warnings.append(f"Voice for shot {s['idx']} failed ({str(exc)[:80]}); that shot is silent.")
                audio = None
        comp = re_.Composition(size, fps, duration=dur, ffmpeg_path=ff)
        if audio:
            comp.add_audio(audio, role="voiceover")
        style = _backdrop_style(s)
        seed = zlib.crc32(str(s.get("location", "")).encode("utf-8")) % 10000  # same place -> same backdrop
        cont = next((i for i in store.inputs_for(s["idx"])["available"] if i["role"] == "continuity"), None)
        if cont:
            hold = min(0.9, dur * 0.45)
            comp.add_broll(cont["path"], 0.0, hold, motion="none")
            comp.add_broll(None, hold * 0.5, None, style=style, motion="ken_burns", seed=seed)
        else:
            comp.add_broll(None, 0.0, None, style=style, motion="ken_burns", seed=seed)
        visible = [c for c in s.get("characters", []) if c in portrait][:2]
        for c in visible:
            comp.add_avatar(portrait[c], 0.0, dur, side[c], None, 0.30, animation="pop" if s["kind"] != "dialogue" else "none")
        if s["kind"] == "establishing" or (s["idx"] == 0):
            comp.add_text(f"{s['int_ext']}. {s['location']} - {s['time_of_day'] or 'DAY'}".strip(" -"), 0.0, min(dur, 2.4), "top", "fade")
        if s.get("line"):
            comp.add_text(f"{s['speaker']}: {s['line']}", 0.0, dur, "auto", "none", font_px=max(16, int(size[0] * 0.04)))
        elif s.get("action") and s["kind"] != "establishing":
            comp.add_text(s["action"][:110], 0.0, dur, "center", "fade", font_px=max(15, int(size[0] * 0.034)))
        clip = str(work / f"shot_{s['idx']:03d}.mp4")
        res = comp.render(clip)
        warnings.extend(w for w in res["warnings"] if w not in warnings and "More than one avatar" not in w)
        kf = _last_frame(clip, str(work / f"kf_{s['idx']:03d}_last.png"))
        store.put(f"kf:{s['idx']}:last", kf)
        clips[s["idx"]] = (clip, dur)
        img = cv2.imread(kf)
        rects = {Path(a["name"]).stem: a["rect"] for a in res.get("avatars", [])}
        for c in visible:
            rect = rects.get(Path(portrait[c]).stem)
            if rect:
                x, y, w_, h_ = rect
                inset = (int(w_ * 0.12), int(h_ * 0.12), int(w_ * 0.76), int(h_ * 0.60))  # the face / shoulders, clear of border + label
                observations.append({"shot": s["idx"], "character": c, "signature": dg.appearance_signature(
                    img, (x + inset[0], y + inset[1], inset[2], inset[3]))})
    # 3. join in story order
    say(0.92, "Joining the shots")
    listing = work / "list.txt"
    listing.write_text("".join(f"file '{Path(clips[s['idx']][0]).as_posix()}'\n" for s in shots), encoding="utf-8")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                           "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True, creationflags=flags)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"ffmpeg could not join the shots: {proc.stderr[-400:]}")
    # drift is measured on the avatar region, against the anchor portrait's own face region
    base = {}
    for c, p in portrait.items():
        im = cv2.imread(p)
        h_, w_ = im.shape[:2]
        base[c] = dg.appearance_signature(im, (int(w_ * 0.12), int(h_ * 0.12), int(w_ * 0.76), int(h_ * 0.60)))
    drift = dg.check_drift(base, observations, drift_threshold)
    say(1.0, "Done")
    total = round(sum(clips[s["idx"]][1] for s in shots), 2)
    return {"output_path": str(out.resolve()), "duration": total, "shots": [{"idx": s["idx"], "clip": clips[s["idx"]][0],
            "seconds": clips[s["idx"]][1]} for s in shots], "drift": drift, "keyframes": dict(store.paths),
            "portraits": portrait, "warnings": warnings, "voice_used": voice_used}
