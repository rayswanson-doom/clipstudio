"""Pre-production asset gate for the story modes (Idea2Video / Novel2Video / Script2Video).

Before an episode is rendered, every CHARACTER and every LOCATION in the script gets a "prompt card" (name, visual
traits, attire, emotional role / setting, lighting, atmosphere + a ready-to-use image prompt). The user then either
uploads a full-length character picture and a background picture per card, or lets the app generate them. Only when
every card has an image does the gate open and the episode render; the renderer then uses those exact files in every
shot (see dependency_graph.bind_assets / enforce_locks), which is what keeps a character - and a place - looking the
same in every cut.

Image generation backends, tried in this order for "auto":
  online     a free, key-less text-to-image endpoint (pollinations.ai). Needs internet; any failure falls through.
  diffusers  a local CPU Stable-Diffusion-class model, only when `diffusers` + `torch` are installed and the model is
             already on disk (explicit "diffusers" may also download it).
  draw       built-in OpenCV painter: always works offline; simple flat illustrations, not photographs.

Background removal for characters (`remove_background`): a transparent PNG is kept as is; else `rembg` when it is
installed; else OpenCV GrabCut seeded from the picture's border colour.

Nothing here needs a GPU or an API key. Network use is limited to the optional "online" backend.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SIDE = 2048
MIN_SIDE = 64
BACKENDS = ("auto", "online", "diffusers", "draw")
ONLINE_URL = "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&seed={seed}&nologo=true"
BOOK_FILE = "assets.json"

NEGATIVE_CHARACTER = "cropped, cut off, close-up, multiple people, text, watermark, logo, blurry, extra limbs, deformed hands"
NEGATIVE_LOCATION = "people, characters, text, watermark, logo, blurry, distorted perspective"

_STYLE_WORDS = {"human": "photorealistic", "ai": "polished AI-rendered 3D character", "anime": "anime cel-shaded illustration"}
_TIME_LIGHT = {
    "NIGHT": ("low-key night lighting, practical lights, deep blue shadows", "moody, quiet, tense"),
    "MIDNIGHT": ("low-key night lighting, practical lights, deep blue shadows", "moody, quiet, tense"),
    "DUSK": ("warm low sun, long shadows, orange-purple sky", "wistful, calm"),
    "EVENING": ("warm low sun, long shadows, orange-purple sky", "wistful, calm"),
    "DAWN": ("soft cold pink-gold early light, mist", "fresh, hopeful"),
    "MORNING": ("bright soft morning light", "fresh, hopeful"),
    "DAY": ("bright natural daylight, soft shadows", "open, lively"),
    "AFTERNOON": ("bright natural daylight, soft shadows", "open, lively"),
}


class AssetError(ValueError):
    """A bad upload or an unusable request (the message is meant for the user)."""


class AssetGateError(RuntimeError):
    """The Pre-Production Asset Gate is still closed: some characters / locations have no reference image yet."""

    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__("The asset gate is closed: " + ", ".join(self.missing[:8])
                         + (f" and {len(self.missing) - 8} more" if len(self.missing) > 8 else "")
                         + " still need a reference image (upload one or generate it).")


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #
def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return (s or "item")[:limit]


def location_key(name: str) -> str:
    """One background per place, whatever the time of day: 'Kitchen', 'KITCHEN ' -> 'kitchen'."""
    return re.sub(r"\s+", " ", str(name or "")).strip().casefold() or "unknown location"


# --------------------------------------------------------------------------- #
# trait extraction + prompt cards
# --------------------------------------------------------------------------- #
_ATTIRE = re.compile(r"\b(?:wearing|wears|dressed in|clad in|in (?:a|an|the))\s+([^,.;:\n]{3,60})", re.I)
_HAIR = re.compile(r"\b((?:long|short|curly|straight|wavy|messy|spiky|shoulder-length|braided|bald|dark|light|[a-z]+)\s+(?:[a-z]+\s+)?hair)\b", re.I)
_EYES = re.compile(r"\b([a-z]+)\s+eyes\b", re.I)
_BUILD = re.compile(r"\b(tall|short|petite|slim|slender|stocky|broad-shouldered|muscular|lanky|elderly|old|young|middle-aged|teenage)\b", re.I)
_NOT_ATTIRE = re.compile(r"^(?:the|a|an)?\s*(?:kitchen|room|street|car|house|office|dark|light|middle of|front of)\b", re.I)


def extract_traits(character: dict) -> dict:
    """Visual traits of a character from its description first, then from the anchor the extractor derived."""
    desc = str(character.get("description") or "")
    anchor = character.get("anchor") or {}
    attire = None
    for m in _ATTIRE.finditer(desc):
        cand = m.group(1).strip()
        cand = re.split(r"\s+(?:with|while|who|as|and (?:stirs?|walks?|looks?|sits?|stands?|holds?|waves?))\b", cand)[0].strip()
        if not _NOT_ATTIRE.match(cand) and len(cand.split()) <= 8:
            attire = cand
            break
    hair = None
    hm = _HAIR.search(desc)
    if hm:
        hair = hm.group(1).strip().lower()
    em = _EYES.search(desc)
    build = sorted({m.group(1).lower() for m in _BUILD.finditer(desc)})
    return {
        "hair": hair or (f"{anchor['hair']} hair" if anchor.get("hair") else None),
        "eyes": (f"{em.group(1).lower()} eyes" if em else (f"{anchor['eyes']} eyes" if anchor.get("eyes") else None)),
        "skin": anchor.get("skin"),
        "build": ", ".join(build) if build else None,
        "age": character.get("age"),
        "gender": character.get("gender") if character.get("gender") not in (None, "unknown") else None,
        "attire": attire or anchor.get("outfit"),
    }


def _character_role(c: dict, rank: int, emotions: Sequence[str]) -> str:
    place = "lead character" if rank == 0 else ("main supporting character" if rank < 3 else "supporting character")
    if emotions:
        top = sorted(set(emotions), key=lambda e: -emotions.count(e))[:2]
        return f"{place}; mostly {' and '.join(top)}"
    if c.get("personality"):
        return f"{place}; {c['personality']}"
    return place


def character_prompt(card: dict, style: str = "human") -> str:
    t = card["traits"]
    bits = ["full-length character reference sheet", "full body from head to toe", "standing, front view, relaxed pose"]
    who = " ".join(x for x in (str(t["age"]) + " year old" if t.get("age") else None, t.get("build"), t.get("gender") or "person") if x)
    bits.append(who)
    for k in ("hair", "eyes"):
        if t.get(k):
            bits.append(t[k])
    if t.get("skin"):
        bits.append(f"{t['skin']} skin")
    if t.get("attire"):
        bits.append(f"wearing {t['attire']}")
    if ";" in str(card.get("emotional_role") or ""):
        bits.append(card["emotional_role"].split(";")[-1].strip().replace("mostly ", "") + " expression")
    bits += [_STYLE_WORDS.get(style, _STYLE_WORDS["human"]), "plain neutral light-grey background, even lighting, one single character, consistent character design"]
    return ", ".join(b for b in bits if b)


def location_prompt(card: dict, style: str = "human") -> str:
    where = {"INT": "interior", "EXT": "exterior"}.get(card.get("int_ext"), "")
    bits = [f"{card['name'].title()} {where}".strip(), card.get("description") or "", card.get("lighting") or "",
            card.get("atmosphere") or "", "wide establishing background plate, empty, no people, no text, cinematic composition",
            _STYLE_WORDS.get(style, "photorealistic").replace("character ", "")]
    return ", ".join(b for b in bits if b)


def build_cards(result: dict) -> dict:
    """{"characters": [card...], "locations": [card...]} for one planned episode (the result of ViMaxOrchestrator.run).

    Every card: id, kind, name, prompt, negative_prompt, image (None until supplied), source, status ("missing" | "ready")
    plus kind-specific fields (traits / attire / emotional_role for characters; int_ext / times / lighting / atmosphere /
    description for locations)."""
    shots = result.get("storyboard", {}).get("shots", [])
    plan = result.get("plan", {})
    style = plan.get("style") or "human"
    chars_out: List[dict] = []
    emotions: Dict[str, List[str]] = {}
    for s in shots:
        if s.get("speaker") and s.get("emotion"):
            emotions.setdefault(s["speaker"], []).append(str(s["emotion"]))
    ranked = sorted(result.get("characters", []), key=lambda c: -int(c.get("mentions") or 0))
    order = {c["name"]: i for i, c in enumerate(ranked)}
    for c in result.get("characters", []):
        traits = extract_traits(c)
        card = {"id": f"char:{c['name']}", "kind": "character", "name": c["name"], "style": c.get("style", style),
                "description": c.get("description") or "", "traits": traits, "attire": traits["attire"],
                "emotional_role": _character_role(c, order.get(c["name"], 9), emotions.get(c["name"], [])),
                "negative_prompt": NEGATIVE_CHARACTER, "image": None, "source": None, "status": "missing",
                "shots": [s["idx"] for s in shots if c["name"] in (s.get("characters") or [])]}
        card["prompt"] = character_prompt(card, card["style"])
        chars_out.append(card)
    locs: Dict[str, dict] = {}
    for s in shots:
        name = s.get("location")
        if not name:
            continue
        key = location_key(name)
        loc = locs.setdefault(key, {"id": f"loc:{key}", "kind": "location", "name": str(name).title(), "key": key,
                                    "int_ext": s.get("int_ext"), "times": [], "lighting": None, "atmosphere": None,
                                    "description": "", "style": style, "negative_prompt": NEGATIVE_LOCATION,
                                    "image": None, "source": None, "status": "missing", "shots": []})
        tod = (s.get("time_of_day") or "DAY").upper()
        if tod not in loc["times"]:
            loc["times"].append(tod)
        loc["shots"].append(s["idx"])
        if not loc["lighting"]:
            loc["lighting"] = s.get("lighting") or _TIME_LIGHT.get(tod, _TIME_LIGHT["DAY"])[0]
            loc["atmosphere"] = _TIME_LIGHT.get(tod, _TIME_LIGHT["DAY"])[1]
        act = str(s.get("action") or "")
        names = [c["name"] for c in result.get("characters", [])]
        if (act and not act.startswith("Establishing") and len(loc["description"]) < 200 and s.get("kind") == "action"
                and not any(re.search(rf"\b{re.escape(n)}\b", act, re.I) for n in names)):
            loc["description"] = (loc["description"] + " " + act).strip()  # setting detail only, never a person
    for sc in plan.get("scenes", []):
        key = location_key(sc.get("location"))
        if key in locs and not locs[key]["description"]:
            locs[key]["description"] = str(sc.get("heading") or "")
    tone = " ".join(x for x in (plan.get("tone"), plan.get("genre")) if x)
    for loc in locs.values():
        if tone:
            loc["atmosphere"] = f"{loc['atmosphere']}, {tone}" if loc["atmosphere"] else tone
        loc["prompt"] = location_prompt(loc, loc["style"])
    return {"characters": chars_out, "locations": list(locs.values())}


# --------------------------------------------------------------------------- #
# the asset book
# --------------------------------------------------------------------------- #
class AssetBook:
    """The cards of one project plus where each reference image lives. Saved as <folder>/assets.json."""

    def __init__(self, folder: str, cards: Optional[dict] = None):
        self.folder = Path(folder)
        self.cards: Dict[str, dict] = {}
        for c in (cards or {}).get("characters", []) + (cards or {}).get("locations", []):
            self.cards[c["id"]] = dict(c)

    # -- views ----------------------------------------------------------- #
    @property
    def characters(self) -> List[dict]:
        return [c for c in self.cards.values() if c["kind"] == "character"]

    @property
    def locations(self) -> List[dict]:
        return [c for c in self.cards.values() if c["kind"] == "location"]

    def missing(self) -> List[str]:
        return [cid for cid, c in self.cards.items() if not (c.get("image") and Path(c["image"]).is_file())]

    def ready(self) -> bool:
        return bool(self.cards) and not self.missing()

    def character_images(self) -> Dict[str, str]:
        return {c["name"]: c["image"] for c in self.characters if c.get("image")}

    def location_images(self) -> Dict[str, str]:
        return {c["key"]: c["image"] for c in self.locations if c.get("image")}

    def image_paths(self) -> List[str]:
        return [c["image"] for c in self.cards.values() if c.get("image") and Path(c["image"]).is_file()]

    def merge_cards(self, cards: dict) -> int:
        """Add cards for names / places this book has not seen yet (another episode); returns how many were new."""
        n = 0
        for c in cards.get("characters", []) + cards.get("locations", []):
            if c["id"] not in self.cards:
                self.cards[c["id"]] = dict(c)
                n += 1
        return n

    # -- changes --------------------------------------------------------- #
    def set_image(self, card_id: str, path: str, source: str) -> dict:
        if card_id not in self.cards:
            raise AssetError(f"Unknown card '{card_id}'.")
        c = self.cards[card_id]
        c.update({"image": str(path), "source": source, "status": "ready", "cutout": None})
        return c

    def clear_image(self, card_id: str) -> None:
        c = self.cards[card_id]
        c.update({"image": None, "source": None, "status": "missing", "cutout": None})

    def set_prompt(self, card_id: str, prompt: str) -> None:
        if not str(prompt).strip():
            raise AssetError("The prompt cannot be empty.")
        self.cards[card_id]["prompt"] = str(prompt).strip()

    def asset_dir(self, kind: str) -> Path:
        d = self.folder / ("characters" if kind == "character" else "locations")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- persistence ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"version": 1, "cards": list(self.cards.values())}

    def save(self) -> str:
        self.folder.mkdir(parents=True, exist_ok=True)
        p = self.folder / BOOK_FILE
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, folder: str) -> "AssetBook":
        data = json.loads((Path(folder) / BOOK_FILE).read_text(encoding="utf-8"))
        book = cls(folder)
        for c in data.get("cards", []):
            book.cards[c["id"]] = c
        return book

    def summary(self) -> dict:
        return {"characters": len(self.characters), "locations": len(self.locations), "missing": self.missing(),
                "ready": self.ready()}


# --------------------------------------------------------------------------- #
# uploads
# --------------------------------------------------------------------------- #
def _decode(data: bytes):
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def add_upload(book: AssetBook, card_id: str, filename: str, data: bytes) -> dict:
    """Validate a user's picture, normalise it to a PNG (alpha kept) in the project folder and attach it to the card.
    Returns {"card", "warnings"}. Raises AssetError with a user-friendly message for anything unusable."""
    import cv2

    if card_id not in book.cards:
        raise AssetError(f"Unknown card '{card_id}'.")
    ext = Path(str(filename)).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise AssetError(f"'{filename}' is not a .png / .jpg picture.")
    if not data:
        raise AssetError(f"'{filename}' is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AssetError(f"'{filename}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    img = _decode(data)
    if img is None or img.ndim < 2:
        raise AssetError(f"'{filename}' could not be read as a picture.")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise AssetError(f"'{filename}' is too small ({w}x{h}); use at least {MIN_SIDE} pixels on each side.")
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    card = book.cards[card_id]
    warnings: List[str] = []
    if card["kind"] == "character" and w / h > 0.95:
        warnings.append(f"{card['name']}: the picture is wider than tall; a full-length, standing view works best.")
    if card["kind"] == "location" and h / w > 2.2:
        warnings.append(f"{card['name']}: the picture is very tall and thin; a normal background photo works best.")
    out = book.asset_dir(card["kind"]) / f"{slug(card['name'])}.png"
    if not cv2.imwrite(str(out), img):
        raise AssetError("The picture could not be saved.")
    book.set_image(card_id, str(out), "upload")
    return {"card": book.cards[card_id], "warnings": warnings}


# --------------------------------------------------------------------------- #
# background removal
# --------------------------------------------------------------------------- #
def _largest_component(mask):
    import cv2
    import numpy as np

    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == keep) * 255).astype(np.uint8)


def _grabcut_alpha(bgr):
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    s = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    sh, sw = small.shape[:2]
    m = max(2, int(min(sh, sw) * 0.03))
    border = np.concatenate([small[:m].reshape(-1, 3), small[-m:].reshape(-1, 3),
                             small[:, :m].reshape(-1, 3), small[:, -m:].reshape(-1, 3)]).astype(np.float32)
    bg = np.median(border, axis=0)
    spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 90))
    dist = np.linalg.norm(small.astype(np.float32) - bg, axis=2)
    thr = max(18.0, spread * 1.6)
    mask = np.full((sh, sw), cv2.GC_PR_FGD, np.uint8)
    mask[dist < thr] = cv2.GC_PR_BGD
    mask[dist < thr * 0.5] = cv2.GC_BGD
    mask[:m], mask[-m:], mask[:, :m], mask[:, -m:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    cy0, cy1, cx0, cx1 = int(sh * 0.3), int(sh * 0.7), int(sw * 0.4), int(sw * 0.6)
    core = mask[cy0:cy1, cx0:cx1]
    core[dist[cy0:cy1, cx0:cx1] >= thr] = cv2.GC_FGD  # the middle of the picture, where it differs from the backdrop, is the subject
    if not (mask == cv2.GC_FGD).any():
        raise AssetError("No subject could be told apart from the background.")
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = _largest_component(fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, cnts, -1, 255, -1)  # fill holes (eyes, gaps between arm and body)
    alpha = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(alpha, (0, 0), max(0.8, min(h, w) / 400.0))


def remove_background(image, tight_crop: bool = True, report: Optional[dict] = None):
    """BGR / BGRA array (or a path) -> BGRA with the character cut out and cropped to its outline.

    report (optional dict) receives {"method": "alpha" | "rembg" | "grabcut" | "whole-picture"}. When no subject can be
    told apart from the background (e.g. the picture IS the subject, edge to edge) the whole picture is kept opaque
    instead of failing the episode."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED) if isinstance(image, (str, Path)) else image
    if img is None:
        raise AssetError("The character picture could not be read.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    method = "alpha"
    if img.shape[2] == 4 and (img[..., 3] < 250).mean() > 0.02:
        bgra = img.copy()
    else:
        bgr = img[..., :3].copy()
        alpha = None
        try:  # a proper segmentation model when the user has installed one
            from rembg import remove as _rembg  # type: ignore

            from PIL import Image

            out = np.asarray(_rembg(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).convert("RGBA"))
            alpha = out[..., 3]
            method = "rembg"
        except Exception:  # noqa: BLE001 - not installed, or its model could not be loaded
            alpha = None
        if alpha is None:
            try:
                alpha = _grabcut_alpha(bgr)
                method = "grabcut"
                if (alpha > 24).mean() < 0.01:
                    raise AssetError("empty cut-out")
            except AssetError:
                alpha = np.full(bgr.shape[:2], 255, np.uint8)
                method = "whole-picture"
        bgra = np.dstack([bgr, alpha])
    a = bgra[..., 3]
    ys, xs = np.nonzero(a > 24)
    if xs.size == 0:
        raise AssetError("The character picture is fully transparent.")
    if tight_crop:
        pad = 4
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad), max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
        bgra = bgra[y0:y1, x0:x1]
    bgra = np.ascontiguousarray(bgra)
    bgra.flags.writeable = True
    if report is not None:
        report["method"] = method
    return bgra


def ensure_cutout(card: dict, book: AssetBook) -> str:
    """Path of the card's transparent cut-out PNG (made once, next to the reference image)."""
    import cv2

    src = card.get("image")
    if not src or not Path(src).is_file():
        raise AssetGateError([card["id"]])
    cut = card.get("cutout")
    if cut and Path(cut).is_file() and Path(cut).stat().st_mtime >= Path(src).stat().st_mtime:
        return cut
    out = book.asset_dir("character") / f"{slug(card['name'])}_cutout.png"
    rep: dict = {}
    bgra = remove_background(src, report=rep)
    card["cutout_method"] = rep.get("method")
    if not cv2.imwrite(str(out), bgra):
        raise AssetError("The cut-out could not be saved.")
    card["cutout"] = str(out)
    return str(out)


# --------------------------------------------------------------------------- #
# the offline painter (always available)
# --------------------------------------------------------------------------- #
_COL = {  # BGR
    "black": (30, 28, 28), "dark": (35, 35, 40), "brown": (45, 70, 110), "chestnut": (40, 65, 120), "auburn": (35, 55, 130),
    "blonde": (110, 170, 215), "blond": (110, 170, 215), "golden": (80, 175, 225), "red": (40, 50, 190), "ginger": (50, 100, 190),
    "grey": (150, 150, 150), "gray": (150, 150, 150), "white": (235, 235, 235), "silver": (190, 190, 195), "blue": (190, 110, 40),
    "green": (90, 150, 60), "teal": (130, 130, 40), "navy": (90, 50, 30), "olive": (60, 120, 110), "burgundy": (50, 30, 120),
    "purple": (140, 60, 120), "pink": (170, 120, 230), "orange": (40, 130, 235), "yellow": (60, 215, 245), "charcoal": (70, 65, 62),
    "tan": (120, 170, 205), "cream": (205, 225, 240), "pale": (210, 220, 235), "fair": (140, 190, 225), "warm": (110, 150, 200),
    "olive skin": (110, 150, 180), "hazel": (60, 110, 130), "amber": (40, 140, 210)}


def _col(text: Optional[str], default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    t = str(text or "").casefold()
    best = None
    for key in _COL:  # the colour named FIRST wins ("red apron over black trousers" -> red)
        m = re.search(rf"\b{re.escape(key)}\b", t)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(key) > len(best[1]))):
            best = (m.start(), key)
    return _COL[best[1]] if best else default


def _shade(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


def draw_character(card: dict, path: str, size: Tuple[int, int] = (512, 1024)) -> str:
    """Paint a full-length standing figure (transparent PNG) from the card's traits: hair / skin / attire colours,
    build and, for 'anime', flat cel shading. Deterministic: the same card always gives the same picture."""
    import cv2
    import numpy as np

    w, h = size
    t = card["traits"]
    rng = np.random.default_rng(zlib.crc32(card["name"].encode("utf-8")))
    skin = _col(t.get("skin"), (150, 185, 225))
    hair = _col(t.get("hair"), (35, 45, 70))
    attire = str(t.get("attire") or "")
    top = _col(attire, (110, 90, 70))
    pants = _shade(_col(re.sub(r"\b(?:shirt|blouse|top|jacket|coat|apron|dress|hoodie|sweater)\b", "", attire) if re.search(r"trouser|pants|jeans|skirt", attire, re.I) else "", (70, 60, 55)), 1.0)
    eyes = _col(t.get("eyes"), (40, 35, 30))
    style = card.get("style", "human")
    img = np.zeros((h, w, 4), np.uint8)
    build = str(t.get("build") or "")
    wide = 1.15 if re.search(r"stocky|broad|muscular", build) else (0.85 if re.search(r"slim|slender|petite|lanky", build) else 1.0)
    tall = 1.0 if not re.search(r"short|petite", build) else 0.93
    cx = w // 2
    head_r = int(w * 0.105)
    top_y = int(h * (0.04 + (1 - tall) * 0.6))
    head_c = (cx, top_y + head_r + int(h * 0.01))
    neck_y = head_c[1] + head_r
    hip_y = int(h * 0.55)
    foot_y = int(h * (0.96 if tall == 1.0 else 0.97))
    sw = int(w * 0.20 * wide)  # shoulder half-width
    hw = int(w * 0.15 * wide)  # hip half-width
    def poly(pts, col):
        cv2.fillPoly(img, [np.array(pts, np.int32)], tuple(int(v) for v in col) + (255,), cv2.LINE_AA)
    # legs
    leg_w = int(hw * 0.78)
    for sx in (-1, 1):
        x0 = cx + sx * int(hw * 0.5)
        poly([(x0 - leg_w // 2, hip_y), (x0 + leg_w // 2, hip_y), (x0 + leg_w // 2 - 4, foot_y - 26), (x0 - leg_w // 2 + 4, foot_y - 26)], pants)
        cv2.ellipse(img, (x0 + sx * 8, foot_y - 12), (int(leg_w * 0.72), 16), 0, 0, 360, (28, 26, 26, 255), -1, cv2.LINE_AA)
    # arms
    for sx in (-1, 1):
        a0 = (cx + sx * sw, neck_y + int(h * 0.035))
        a1 = (cx + sx * (sw + int(w * 0.035)), hip_y - int(h * 0.01))
        cv2.line(img, a0, a1, tuple(top) + (255,), int(w * 0.085 * wide), cv2.LINE_AA)
        cv2.circle(img, (a1[0], a1[1] + int(w * 0.04)), int(w * 0.04), tuple(skin) + (255,), -1, cv2.LINE_AA)
    # torso (+ skirt-ish flare for dresses / aprons)
    poly([(cx - sw, neck_y + int(h * 0.02)), (cx + sw, neck_y + int(h * 0.02)), (cx + hw + 6, hip_y + 8), (cx - hw - 6, hip_y + 8)], top)
    if re.search(r"dress|skirt|coat|apron", attire, re.I):
        poly([(cx - hw - 8, hip_y - 10), (cx + hw + 8, hip_y - 10), (cx + hw + int(w * 0.07), hip_y + int(h * 0.17)), (cx - hw - int(w * 0.07), hip_y + int(h * 0.17))], _shade(top, 0.92))
    if style != "anime":
        poly([(cx + sw // 3, neck_y + int(h * 0.02)), (cx + sw, neck_y + int(h * 0.02)), (cx + hw + 6, hip_y + 8), (cx + hw // 3, hip_y + 8)], _shade(top, 0.8))
    # neck + head
    cv2.rectangle(img, (cx - int(head_r * 0.32), neck_y - 8), (cx + int(head_r * 0.32), neck_y + int(h * 0.03)), tuple(skin) + (255,), -1)
    cv2.circle(img, (head_c[0], head_c[1] - 6), int(head_r * 1.12), tuple(hair) + (255,), -1, cv2.LINE_AA)       # hair mass
    cv2.ellipse(img, head_c, (int(head_r * 0.92), int(head_r * 1.08)), 0, 0, 360, tuple(skin) + (255,), -1, cv2.LINE_AA)
    cv2.ellipse(img, (head_c[0], head_c[1] - int(head_r * 0.55)), (int(head_r * 1.0), int(head_r * 0.55)), 0, 180, 360, tuple(hair) + (255,), -1, cv2.LINE_AA)  # fringe
    if re.search(r"long|braid|wavy|curly", str(t.get("hair") or ""), re.I):
        for sx in (-1, 1):
            poly([(head_c[0] + sx * int(head_r * 0.95), head_c[1] - 4), (head_c[0] + sx * int(head_r * 1.15), head_c[1] + int(head_r * 2.2)),
                  (head_c[0] + sx * int(head_r * 0.7), head_c[1] + int(head_r * 2.0)), (head_c[0] + sx * int(head_r * 0.7), head_c[1])], hair)
    er = int(head_r * (0.17 if style != "anime" else 0.26))
    for sx in (-1, 1):
        ex, ey = head_c[0] + sx * int(head_r * 0.40), head_c[1] + int(head_r * 0.05)
        cv2.circle(img, (ex, ey), er + 2, (245, 245, 245, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), er, tuple(eyes) + (255,), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), max(2, er // 2), (20, 20, 20, 255), -1, cv2.LINE_AA)
        if style == "anime":
            cv2.circle(img, (ex - er // 3, ey - er // 3), max(2, er // 3), (255, 255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (head_c[0], head_c[1] + int(head_r * 0.62)), (int(head_r * 0.28), int(head_r * 0.12)), 0, 0, 180, (70, 70, 160, 255), 2, cv2.LINE_AA)
    if style != "anime":
        a = img[..., 3].copy()
        sh = cv2.GaussianBlur(img[..., :3], (0, 0), 4)
        img[..., :3] = cv2.addWeighted(img[..., :3], 0.8, sh, 0.2, 0)
        img[..., 3] = a
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise AssetError("The character picture could not be saved.")
    return str(path)


_PLACE_KINDS = (
    ("beach", ("beach", "shore", "coast", "sea", "ocean", "harbor", "harbour", "dock", "pier")),
    ("forest", ("forest", "wood", "park", "garden", "jungle", "field", "meadow", "farm", "mountain", "hill", "trail", "camp")),
    ("city", ("street", "road", "city", "alley", "rooftop", "plaza", "market", "bridge", "highway", "parking", "town", "square", "station")),
    ("space", ("space", "spaceship", "ship", "lab", "laboratory", "bridge deck")),
    ("room", ("kitchen", "room", "office", "house", "home", "apartment", "bedroom", "cafe", "café", "restaurant", "bar", "school",
              "classroom", "hall", "library", "shop", "store", "studio", "hospital", "bathroom", "living", "hotel", "lobby", "cabin", "garage", "basement")),
)
_SKY = {"DAY": ((235, 200, 150), (250, 235, 205)), "NIGHT": ((50, 25, 15), (95, 55, 35)), "DUSK": ((90, 70, 150), (130, 170, 245)),
        "DAWN": ((190, 165, 150), (215, 205, 245)), "MORNING": ((235, 200, 150), (250, 235, 205))}


def place_kind(name: str, int_ext: Optional[str]) -> str:
    n = str(name or "").casefold()
    for kind, words in _PLACE_KINDS:
        if any(re.search(rf"\b{re.escape(w)}", n) for w in words):
            return kind
    return "room" if int_ext == "INT" else "city"


def draw_location(card: dict, path: str, size: Tuple[int, int] = (720, 1280)) -> str:
    """Paint an empty background plate for the location (room / street / forest / beach / space), lit for the first
    time of day on the card. Deterministic per location name."""
    import cv2
    import numpy as np

    w, h = size
    rng = np.random.default_rng(zlib.crc32(card["name"].encode("utf-8")))
    tod = (card.get("times") or ["DAY"])[0].upper()
    night = tod in ("NIGHT", "MIDNIGHT")
    kind = place_kind(card["name"], card.get("int_ext"))
    top_c, bot_c = _SKY.get({"MIDNIGHT": "NIGHT", "EVENING": "DUSK", "AFTERNOON": "DAY"}.get(tod, tod), _SKY["DAY"])
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (np.array(top_c, np.float32) * (1 - grad) + np.array(bot_c, np.float32) * grad).repeat(w, axis=1)
    img = np.ascontiguousarray(img.astype(np.uint8))
    horizon = int(h * 0.62)
    light = 0.45 if night else 1.0
    if kind == "room":
        wall = tuple(int(v) for v in (np.array((rng.integers(150, 215), rng.integers(160, 215), rng.integers(170, 225))) * light))
        img[:horizon] = wall
        floor = _shade((70 + int(rng.integers(40)), 95 + int(rng.integers(40)), 130 + int(rng.integers(40))), light)
        img[horizon:] = floor
        for i in range(0, w, max(40, w // 9)):  # floorboards in perspective
            cv2.line(img, (i, h), (w // 2 + (i - w // 2) // 3, horizon), _shade(floor, 0.82), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, horizon - 14), (w, horizon), _shade(wall, 0.8), -1)  # skirting
        ww, wh = int(w * 0.34), int(h * 0.20)
        wx, wy = int(w * (0.12 + 0.5 * rng.random())), int(h * 0.16)
        cv2.rectangle(img, (wx - 8, wy - 8), (wx + ww + 8, wy + wh + 8), _shade(wall, 0.6), -1)
        cv2.rectangle(img, (wx, wy), (wx + ww, wy + wh), top_c if not night else (60, 35, 20), -1)
        cv2.line(img, (wx + ww // 2, wy), (wx + ww // 2, wy + wh), _shade(wall, 0.6), 5)
        cv2.line(img, (wx, wy + wh // 2), (wx + ww, wy + wh // 2), _shade(wall, 0.6), 5)
        tx = int(w * (0.55 if wx < w * 0.4 else 0.08))  # a table / counter
        cv2.rectangle(img, (tx, horizon - int(h * 0.07)), (tx + int(w * 0.36), horizon + int(h * 0.03)), _shade((60, 90, 140), light), -1)
        cv2.rectangle(img, (tx + 8, horizon + int(h * 0.03)), (tx + 22, horizon + int(h * 0.12)), _shade((50, 70, 110), light), -1)
        cv2.rectangle(img, (tx + int(w * 0.36) - 22, horizon + int(h * 0.03)), (tx + int(w * 0.36) - 8, horizon + int(h * 0.12)), _shade((50, 70, 110), light), -1)
        if night:
            lamp = (tx + int(w * 0.18), horizon - int(h * 0.12))
            glow = np.zeros_like(img)
            cv2.circle(glow, lamp, int(w * 0.35), (110, 190, 245), -1, cv2.LINE_AA)
            img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(glow, (0, 0), w * 0.12), 0.55, 0)
            cv2.circle(img, lamp, 14, (190, 235, 255), -1, cv2.LINE_AA)
    elif kind in ("city", "forest", "beach", "space"):
        if kind == "space":
            img[:] = (30, 18, 12)
            for _ in range(160):
                cv2.circle(img, (int(rng.integers(w)), int(rng.integers(h))), int(rng.integers(1, 3)), (235, 235, 245), -1)
            cv2.circle(img, (int(w * 0.75), int(h * 0.25)), int(w * 0.16), (120, 90, 60), -1, cv2.LINE_AA)
            img[horizon + int(h * 0.1):] = (55, 50, 48)
            cv2.rectangle(img, (0, horizon + int(h * 0.1)), (w, horizon + int(h * 0.12)), (120, 120, 125), -1)
        else:
            ground = {"city": (80, 78, 76), "forest": (60, 120, 70), "beach": (150, 200, 230)}[kind]
            img[horizon:] = _shade(ground, light)
            if kind == "beach":
                cv2.rectangle(img, (0, int(h * 0.50)), (w, horizon), _shade((170, 130, 40), light), -1)
                for i in range(6):
                    y = int(h * 0.50) + i * int(h * 0.02)
                    cv2.line(img, (0, y), (w, y), _shade((220, 200, 160), light), 2, cv2.LINE_AA)
            elif kind == "forest":
                for _ in range(9):
                    x, tw_ = int(rng.integers(-20, w)), int(rng.integers(30, 70))
                    ty = int(rng.integers(int(h * 0.30), horizon - 20))
                    cv2.rectangle(img, (x, ty), (x + tw_ // 3, horizon + 10), _shade((40, 70, 100), light), -1)
                    cv2.circle(img, (x + tw_ // 6, ty), tw_, _shade((50 + int(rng.integers(40)), 120 + int(rng.integers(40)), 50), light), -1, cv2.LINE_AA)
            else:
                x = -10
                while x < w:
                    bw_, bh_ = int(rng.integers(w // 7, w // 3)), int(rng.integers(int(h * 0.2), int(h * 0.5)))
                    col = _shade((int(rng.integers(90, 160)), int(rng.integers(90, 150)), int(rng.integers(90, 150))), light)
                    cv2.rectangle(img, (x, horizon - bh_), (x + bw_, horizon), col, -1)
                    for wy in range(horizon - bh_ + 16, horizon - 14, 34):
                        for wx in range(x + 10, x + bw_ - 18, 30):
                            lit = night and rng.random() < 0.5
                            cv2.rectangle(img, (wx, wy), (wx + 14, wy + 18), (110, 210, 245) if lit else _shade(col, 0.7), -1)
                    x += bw_ + 6
                cv2.rectangle(img, (0, horizon), (w, h), _shade((72, 70, 68), light), -1)
                for i in range(6):
                    cv2.line(img, (int(w * 0.5), horizon + 6), (int(w * (i / 5)), h), _shade((100, 98, 96), light), 3, cv2.LINE_AA)
    vig = np.zeros((h, w), np.float32)
    cv2.circle(vig, (w // 2, h // 2), int(max(w, h) * 0.62), 1.0, -1)
    vig = cv2.GaussianBlur(vig, (0, 0), max(w, h) * 0.2)
    img = (img.astype(np.float32) * (0.62 + 0.38 * vig[..., None])).astype(np.uint8)
    noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise AssetError("The background picture could not be saved.")
    return str(path)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 75.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "CinePulseAI-asset-studio"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https endpoint
        return resp.read(MAX_UPLOAD_BYTES + 1)


def _generate_online(card: dict, size: Tuple[int, int], transport: Optional[Callable[[str], bytes]]) -> bytes:
    seed = zlib.crc32(card["id"].encode("utf-8")) % 100000
    url = ONLINE_URL.format(prompt=urllib.parse.quote(card["prompt"][:900], safe=""), w=size[0], h=size[1], seed=seed)
    return (transport or _http_get)(url)


def _generate_diffusers(card: dict, size: Tuple[int, int], allow_download: bool, pipe_factory: Optional[Callable] = None) -> bytes:
    """A local CPU diffusion model (default: stabilityai/sd-turbo, ~2 GB on first use). Optional dependency."""
    import io

    if pipe_factory is None:
        import diffusers  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401
        from diffusers import AutoPipelineForText2Image  # type: ignore

        model = __import__("os").environ.get("CINEPULSE_DIFFUSERS_MODEL", "stabilityai/sd-turbo")
        pipe = AutoPipelineForText2Image.from_pretrained(model, local_files_only=not allow_download)
        pipe.to("cpu")
    else:
        pipe = pipe_factory()
    w, h = (max(256, min(768, s // 64 * 64)) for s in size)
    img = pipe(prompt=card["prompt"], negative_prompt=card["negative_prompt"], num_inference_steps=4, guidance_scale=0.0,
               width=w, height=h).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _save_generated(book: AssetBook, card: dict, data: bytes, source: str) -> None:
    add_upload(book, card["id"], "generated.png", data)
    book.cards[card["id"]]["source"] = source


def generate_images(book: AssetBook, card_ids: Optional[Iterable[str]] = None, backend: str = "auto",
                    frame_size: Tuple[int, int] = (720, 1280), only_missing: bool = True,
                    transport: Optional[Callable[[str], bytes]] = None, pipe_factory: Optional[Callable] = None,
                    progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Create the reference picture for each card (character sheets: 2:3, locations: the video's aspect).

    backend: "auto" (online -> local diffusers -> drawn), "online", "diffusers" or "draw". A backend that fails
    (no internet, model missing) hands over to the next one in "auto"; a specific choice falls back to the painter.
    Returns {"generated": {card_id: backend}, "failed": {card_id: reason}, "warnings": [...]}."""
    if backend not in BACKENDS:
        raise AssetError(f"Unknown image backend '{backend}'. Use one of: {', '.join(BACKENDS)}.")
    ids = [c for c in (card_ids if card_ids is not None else book.cards) if c in book.cards]
    if only_missing and card_ids is None:
        ids = [c for c in ids if c in book.missing()]
    generated, failed, warnings = {}, {}, []
    online_down = False
    for n, cid in enumerate(ids):
        card = book.cards[cid]
        if progress:
            progress(n / max(1, len(ids)), f"Creating the {card['kind']} picture for {card['name']}")
        size = (512, 768) if card["kind"] == "character" else (min(frame_size), max(frame_size)) if frame_size[0] < frame_size[1] else frame_size
        chain = {"auto": ("online", "diffusers", "draw"), "online": ("online", "draw"), "diffusers": ("diffusers", "draw"), "draw": ("draw",)}[backend]
        done = False
        for step in chain:
            try:
                if step == "online":
                    if online_down:
                        continue
                    data = _generate_online(card, size, transport)
                    _save_generated(book, card, data, "generated:online")
                elif step == "diffusers":
                    data = _generate_diffusers(card, size, allow_download=(backend == "diffusers"), pipe_factory=pipe_factory)
                    _save_generated(book, card, data, "generated:diffusers")
                else:
                    out = book.asset_dir(card["kind"]) / f"{slug(card['name'])}.png"
                    (draw_character(card, str(out)) if card["kind"] == "character" else draw_location(card, str(out), (size[0], size[1])))
                    book.set_image(cid, str(out), "generated:draw")
                generated[cid] = step
                done = True
                break
            except Exception as exc:  # noqa: BLE001 - every backend failure just hands over to the next one
                msg = str(exc)[:120]
                if step == "online" and isinstance(exc, (OSError, urllib.error.URLError)):
                    online_down = True  # do not wait for a timeout on every card
                if step != chain[-1]:
                    warnings.append(f"{card['name']}: the {step} backend failed ({msg}); trying the next one.")
                else:
                    failed[cid] = msg
        if not done and cid not in failed:
            failed[cid] = "no backend produced a picture"
    if progress:
        progress(1.0, "Pictures ready")
    uniq = []
    for w in warnings:  # one note per backend problem is enough
        key = re.sub(r"^[^:]+:", "", w)
        if key not in [re.sub(r"^[^:]+:", "", u) for u in uniq]:
            uniq.append(w)
    return {"generated": generated, "failed": failed, "warnings": uniq}


def ensure_assets(book: AssetBook, backend: str = "draw", **kw) -> dict:
    """Fill every card that still has no picture (used when a caller renders without going through the studio)."""
    return generate_images(book, None, backend=backend, **kw)


def require_ready(book: AssetBook) -> None:
    """Close the gate unless every card has a picture on disk."""
    miss = book.missing()
    if miss:
        raise AssetGateError(miss)


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
