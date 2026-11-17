"""
backend/social_metadata.py

Ready-to-post caption packages for a clip: YouTube Shorts, Instagram Reels and Facebook Reels.

Works 100 % offline with zero API keys: free local NLP (stop-word-filtered unigram / bigram frequency with a boost for the
topic keywords you give, and the scorer's "hook" scoring to pick the punchiest sentence). If a free-tier key is configured
(GROQ_API_KEY or GEMINI_API_KEY) an LLM can polish the wording; any failure silently falls back to the offline result and
says so in `warnings`. The platform limits are enforced on BOTH paths, so an LLM can never produce a 90-character title or
a 40-hashtag caption.

    generate_packages(transcript_text, topic_keywords=("fitness",))
      -> {"keywords": [...], "hook": "...", "source": "offline" | "groq" | "gemini", "warnings": [],
          "youtube":   {"title","description","tags","hashtags","copy_text"},
          "instagram": {"hook","caption","cta","hashtags","copy_text"},
          "facebook":  {"caption","hashtags","copy_text"}}
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

YT_TITLE_MAX = 60
YT_TAGS = (10, 15)
IG_TAGS = (15, 20)
FB_TAGS = (5, 8)
YT_TAG_CHARS = 480  # YouTube allows 500 characters of tags in total

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further get got had has have having he her here hers herself him himself his how i
if in into is it its itself just like me more most my myself no nor not now of off on once one only or other our ours ourselves out over own
really right same she should so some such than that the their theirs them themselves then there these they this those through to too under
until up us very was we were what when where which while who whom why will with would you your yours yourself yourselves yeah okay oh um uh
gonna wanna going thing things stuff said say says know think want need make made let lets see look way back even much many still well also
actually basically literally enough making matter every single never always anything everything something someone people better
nobody tell tells lot lots kind sort around maybe probably though without within another each different seem seems sure thats theres heres dont cant
wont didnt doesnt isnt wasnt youre theyre ive ill youll""".split())

# niche -> (trigger words, niche hashtags)
NICHES: Dict[str, tuple] = {
    "fitness": (("workout", "gym", "fitness", "muscle", "protein", "training", "exercise", "fat", "weight", "cardio"),
                ["fitness", "workout", "gym", "fitnessmotivation", "healthylifestyle", "gymtok", "bodybuilding", "training"]),
    "finance": (("money", "invest", "investing", "stock", "stocks", "crypto", "bitcoin", "rich", "wealth", "income", "business", "profit"),
                ["money", "investing", "finance", "wealth", "entrepreneur", "passiveincome", "financialfreedom", "business"]),
    "tech": (("ai", "tech", "software", "code", "coding", "app", "phone", "computer", "chatgpt", "robot", "startup", "internet"),
             ["tech", "ai", "technology", "coding", "innovation", "gadgets", "futuretech", "programming"]),
    "food": (("recipe", "cook", "cooking", "food", "eating", "kitchen", "meal", "chef", "delicious", "bake"),
             ["food", "recipe", "cooking", "foodie", "easyrecipes", "homecooking", "foodtok", "kitchenhacks"]),
    "gaming": (("game", "gaming", "gamer", "level", "boss", "play", "stream", "console", "esports", "minecraft", "fortnite"),
               ["gaming", "gamer", "gameplay", "videogames", "twitch", "esports", "gamingcommunity", "pcgaming"]),
    "motivation": (("success", "mindset", "discipline", "goal", "goals", "dream", "motivation", "hustle", "grind", "habit", "habits"),
                   ["motivation", "mindset", "success", "discipline", "selfimprovement", "inspiration", "growthmindset", "habits"]),
    "comedy": (("funny", "joke", "hilarious", "laugh", "comedy", "lol", "prank", "meme"),
               ["funny", "comedy", "humor", "memes", "lol", "relatable", "jokes", "laughing"]),
    "education": (("learn", "explain", "science", "history", "study", "school", "fact", "facts", "research", "brain", "psychology"),
                  ["education", "learnontiktok", "science", "facts", "didyouknow", "knowledge", "psychology", "history"]),
    "news": (("news", "breaking", "election", "government", "president", "war", "economy", "report", "politics"),
             ["news", "breakingnews", "currentevents", "worldnews", "politics", "update", "trending", "headlines"]),
    "music": (("song", "music", "singer", "album", "guitar", "piano", "beat", "lyrics", "concert", "rapper"),
              ["music", "newmusic", "singer", "musician", "songwriter", "musicvideo", "livemusic", "artist"]),
    "travel": (("travel", "trip", "vacation", "flight", "hotel", "country", "city", "tour", "beach", "adventure"),
               ["travel", "wanderlust", "traveltok", "explore", "adventure", "vacation", "travelgram", "bucketlist"]),
    "beauty": (("makeup", "skin", "skincare", "beauty", "hair", "style", "fashion", "outfit", "glow"),
               ["beauty", "makeup", "skincare", "fashion", "style", "grwm", "glowup", "selfcare"]),
    "relationships": (("dating", "relationship", "love", "marriage", "partner", "boyfriend", "girlfriend", "wife", "husband", "breakup"),
                      ["relationships", "dating", "love", "relationshipadvice", "couplegoals", "datingadvice", "marriage", "selflove"]),
    "podcast": (("podcast", "episode", "interview", "guest", "conversation", "host"),
                ["podcast", "podcastclips", "interview", "conversation", "podcasting", "podcastlife", "clips", "mustwatch"]),
}
GENERIC_VIRAL = ["viral", "fyp", "trending", "explore", "foryou", "reels", "shorts", "mustwatch", "reelsinstagram", "instagood",
                 "instareels", "explorepage", "viralreels", "trendingreels", "reelsvideo", "shortvideo", "contentcreator", "dailyreels"]
GENERIC_YT_TAGS = ["Shorts", "youtube shorts", "viral", "trending", "viral video", "fyp", "short video", "must watch",
                   "trending shorts", "viral shorts", "shorts video", "entertainment", "for you", "explore"]
HOOK_OPENERS = ("Nobody tells you this about", "Here's the truth about", "Stop scrolling: ", "You need to hear this about", "The real reason behind")
CTAS = {
    "youtube": "Subscribe for more clips like this and tell me what you think in the comments.",
    "instagram": "Save this for later and send it to someone who needs to hear it. Follow for more.",
    "facebook": "What do you think? Share your take in the comments and tag a friend who should see this.",
}


# --------------------------------------------------------------------------- #
# offline NLP
# --------------------------------------------------------------------------- #
def tokens(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z']+|\d+", str(text).casefold().replace("’", "'"))


def extract_keywords(text: str, topic_keywords: Sequence[str] = (), top_k: int = 15) -> List[str]:
    """Ranked keywords / key phrases: frequency of non-stop-word unigrams and bigrams (bigrams that repeat count double),
    plus a strong boost for the caller's topic keywords. No model, no download."""
    toks = tokens(text)
    content = [t for t in toks if t not in STOPWORDS and len(t) > 2 and not t.isdigit() and "'" not in t]
    uni = Counter(content)
    bi = Counter()
    for a, b in zip(toks, toks[1:]):
        if a not in STOPWORDS and b not in STOPWORDS and len(a) > 2 and len(b) > 2:
            bi[f"{a} {b}"] += 1
    scores: Dict[str, float] = {}
    for w, c in uni.items():
        scores[w] = (1.0 + math.log(c)) * (1.0 + min(len(w), 10) / 20.0)
    for p, c in bi.items():
        if c >= 2:
            scores[p] = 2.0 * (1.0 + math.log(c)) + 0.5
    topics = [t.strip().casefold() for t in topic_keywords if t and t.strip()]
    for t in topics:
        scores[t] = scores.get(t, 0.0) + 10.0
    ranked = sorted(scores, key=lambda k: (-scores[k], k))
    out: List[str] = []
    for k in ranked:  # a single word already inside a chosen phrase adds nothing
        if " " not in k and any(k in o.split() for o in out if " " in o):
            continue
        out.append(k)
        if len(out) >= top_k:
            break
    return out


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", str(text).strip()) if len(s.split()) >= 3]


def pick_hook(text: str, max_chars: int = 110) -> str:
    """The most scroll-stopping sentence of the transcript (questions, numbers, strong words, direct address)."""
    sents = _sentences(text)
    if not sents:
        return ""
    try:
        try:
            from . import scorer
        except ImportError:  # pragma: no cover
            import scorer
        best = max(sents, key=lambda s: scorer.score_sentence({"text": s, "word_count": len(s.split())})["score"] - 0.02 * max(0, len(s) - max_chars))
    except Exception:  # noqa: BLE001
        best = sents[0]
    return _trim(best, max_chars)


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut.rstrip(".") + "…" if len(text) > limit else cut


def _titlecase(s: str) -> str:
    small = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "at", "is", "it", "as", "by"}
    ws = s.split()
    return " ".join(w if (i and w.lower() in small) else w[:1].upper() + w[1:] for i, w in enumerate(ws))


def detect_niches(text: str, topic_keywords: Sequence[str] = ()) -> List[str]:
    toks = set(tokens(text)) | {t.casefold() for t in topic_keywords}
    hits = []
    for name, (words, _tags) in NICHES.items():
        n = sum(1 for w in words if w in toks)
        if n:
            hits.append((n, name))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [n for _c, n in hits[:3]]


def _hashtag(word: str) -> str:
    w = re.sub(r"[^0-9a-z]", "", word.casefold())
    return w


def _unique(seq: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        k = x.casefold()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _hashtags(keywords: Sequence[str], niches: Sequence[str], extra: Sequence[str], n_range) -> List[str]:
    lo, hi = n_range
    pool = [_hashtag(k) for k in keywords if " " not in k] + [_hashtag(k) for k in keywords if " " in k]
    for nn in niches:
        pool += NICHES[nn][1]
    pool += list(extra) + GENERIC_VIRAL
    tags = [t for t in _unique(pool) if 2 < len(t) <= 30]
    return tags[:hi] if len(tags) >= lo else tags


def _yt_tags(keywords: Sequence[str], niches: Sequence[str]) -> List[str]:
    pool = list(keywords)
    for nn in niches:
        pool += [t for t in NICHES[nn][1] if t not in ("reels",)][:4]
    pool += GENERIC_YT_TAGS
    tags, used = [], 0
    for t in _unique(pool):
        t = re.sub(r"[^\w\s'-]", "", t).strip()
        if not t or used + len(t) + 1 > YT_TAG_CHARS:
            continue
        tags.append(t)
        used += len(t) + 1
        if len(tags) >= YT_TAGS[1]:
            break
    tags = ["Shorts" if t.lower() == "shorts" else t for t in tags]
    if "Shorts" not in tags:
        tags = tags[: YT_TAGS[1] - 1] + ["Shorts"]
    return tags


def pick_title(text: str, keywords: Sequence[str], limit: int = YT_TITLE_MAX - 1) -> str:
    """A complete, punchy title under `limit` characters: the best-scoring whole sentence or clause that fits (keyword
    hits help), else a keyword headline. Never a sentence cut off mid-way."""
    try:
        try:
            from . import scorer
        except ImportError:  # pragma: no cover
            import scorer
        score = lambda s: scorer.score_sentence({"text": s, "word_count": len(s.split())})["score"]  # noqa: E731
    except Exception:  # noqa: BLE001
        score = lambda s: 0.0  # noqa: E731
    cands = []
    for sent in _sentences(text):
        for clause in [sent] + [c.strip() for c in re.split(r"[,;:\u2014]\s+", sent)]:
            c = clause.strip().rstrip(".!?… ")
            if 3 <= len(c.split()) and 14 <= len(c) <= limit:
                hits = sum(1 for k in keywords[:6] if k in c.casefold())
                cands.append((score(c) + 1.2 * hits - 0.01 * abs(len(c) - 42), c))
    if cands:
        return max(cands)[1][:1].upper() + max(cands)[1][1:]
    kws = [k for k in keywords if k][:3]
    if len(kws) >= 2:
        return _trim(f"{_titlecase(kws[0])}: {_titlecase(kws[1])}" + (f" & {_titlecase(kws[2])}" if len(kws) > 2 else ""), limit).rstrip("…")
    return _trim(f"{_titlecase(kws[0])}: what you need to know" if kws else "You need to see this", limit).rstrip("…")


def _youtube(hook: str, keywords: List[str], niches: List[str], text: str) -> dict:
    title = pick_title(text, keywords)
    tags = _yt_tags(keywords, niches)
    hashtags = ["#" + _hashtag(k) for k in (keywords[:2] or ["viral"]) if _hashtag(k)]
    hashtags = _unique(hashtags + ["#Shorts"])
    sents = _sentences(text)
    summary = _trim(" ".join(sents[1:3]), 220) if len(sents) > 1 else ""
    desc = "\n\n".join(p for p in (
        _trim(hook, 140) if hook else f"Everything you need to know about {keywords[0] if keywords else 'this'}.",
        summary, CTAS["youtube"],
        " ".join(hashtags[:3]) + (" " + " ".join(h for h in hashtags[3:])) if len(hashtags) > 3 else " ".join(hashtags)) if p)
    pkg = {"title": title, "description": desc, "tags": tags, "hashtags": hashtags}
    pkg["copy_text"] = f"{title}\n\n{desc}\n\nTags: {', '.join(tags)}"
    return pkg


def _instagram(hook: str, keywords: List[str], niches: List[str], text: str) -> dict:
    h = _trim(hook or (f"Nobody talks about {keywords[0]} like this." if keywords else "Wait for it."), 125)
    if not re.search(r"[.!?…]$", h):
        h += "."
    sents = _sentences(text)
    body = _trim(" ".join(s for s in sents[:3] if s not in h), 260)
    focus = keywords[0] if keywords else "this"
    cta = f"Save this & send it to someone who needs it. Follow for more {focus}."
    tags = _hashtags(keywords, niches, ("instareels", "explorepage"), IG_TAGS)
    caption = "\n\n".join(p for p in (h, body, cta) if p)
    ht = " ".join("#" + t for t in tags)
    pkg = {"hook": h, "caption": caption, "cta": cta, "hashtags": ["#" + t for t in tags]}
    pkg["copy_text"] = f"{caption}\n.\n.\n{ht}"
    return pkg


def _facebook(hook: str, keywords: List[str], niches: List[str], text: str) -> dict:
    sents = _sentences(text)
    focus = keywords[0] if keywords else "this"
    opener = _trim(hook, 140) if hook else f"I had to share this about {focus}."
    story = _trim(" ".join(s for s in sents[:4] if s not in opener), 320)
    q = f"Has this ever happened to you? Tell me about your experience with {focus} below."
    tags = _hashtags(keywords, niches, (), FB_TAGS)
    caption = "\n\n".join(p for p in (opener, story, q, CTAS["facebook"]) if p)
    pkg = {"caption": caption, "hashtags": ["#" + t for t in tags]}
    pkg["copy_text"] = f"{caption}\n\n{' '.join(pkg['hashtags'])}"
    return pkg


def build_offline(text: str, topic_keywords: Sequence[str] = ()) -> dict:
    text = str(text or "").strip()
    keywords = extract_keywords(text, topic_keywords, 15)
    niches = detect_niches(text, topic_keywords)
    hook = pick_hook(text)
    return {"keywords": keywords, "niches": niches, "hook": hook, "source": "offline", "warnings": [],
            "youtube": _youtube(hook, keywords, niches, text), "instagram": _instagram(hook, keywords, niches, text),
            "facebook": _facebook(hook, keywords, niches, text)}


# --------------------------------------------------------------------------- #
# optional free-tier LLM polish
# --------------------------------------------------------------------------- #
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
Transport = Callable[[str, dict, dict], dict]  # (url, headers, json_body) -> parsed JSON response


def _post_json(url: str, headers: dict, body: dict, timeout: float = 40.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "CinePulseAI", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https endpoints
        return json.loads(resp.read().decode("utf-8", "replace"))


def llm_provider() -> Optional[str]:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return None


def _prompt(text: str, keywords: Sequence[str]) -> str:
    return (
        "You write social-media posts for a short vertical video. Using the transcript below, return ONLY a JSON object with keys: "
        '"youtube": {"title" (max 55 chars, punchy, SEO), "description" (hook line, 1-2 lines, CTA), "tags" (12 plain keywords, no #)}, '
        '"instagram": {"hook" (scroll-stopping first line, max 120 chars), "caption", "cta", "hashtags" (18, each starting with #)}, '
        '"facebook": {"caption" (conversational, story-driven, ends with a question), "hashtags" (6, each starting with #)}.\n'
        f"Topic keywords: {', '.join(keywords)}\nTranscript:\n{text[:3500]}")


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def llm_packages(text: str, keywords: Sequence[str], provider: str, transport: Optional[Transport] = None) -> dict:
    """Ask a free-tier LLM for the three packages; returns the parsed JSON (unvalidated). Raises on any failure."""
    send = transport or _post_json
    if provider == "groq":
        key = os.environ["GROQ_API_KEY"]
        res = send(GROQ_URL, {"Authorization": f"Bearer {key}"},
                   {"model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"), "temperature": 0.7,
                    "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": _prompt(text, keywords)}]})
        raw = res["choices"][0]["message"]["content"]
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        res = send(GEMINI_URL.format(model=model), {"x-goog-api-key": key},
                   {"contents": [{"parts": [{"text": _prompt(text, keywords)}]}],
                    "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}})
        raw = res["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"unknown provider {provider}")
    return _extract_json(raw)


def _clean_tags(items, prefix: str = "") -> List[str]:
    out = []
    for t in items or []:
        t = re.sub(r"[^0-9A-Za-z_ '-]", "", str(t).lstrip("#")).strip()
        if t:
            out.append(prefix + (t.replace(" ", "") if prefix else t))
    return _unique(out)


def merge_llm(base: dict, llm: dict) -> dict:
    """Overlay the LLM's wording on the offline package, enforcing every platform limit; anything missing or out of range
    keeps the offline value."""
    out = json.loads(json.dumps(base))
    yt, ig, fb = (llm.get("youtube") or {}), (llm.get("instagram") or {}), (llm.get("facebook") or {})
    title = _trim(str(yt.get("title") or ""), YT_TITLE_MAX - 1).rstrip("…")
    if 8 <= len(title) < YT_TITLE_MAX:
        out["youtube"]["title"] = title
    if str(yt.get("description") or "").strip():
        desc = str(yt["description"]).strip()
        if "#Shorts" not in desc:
            desc += "\n\n" + " ".join(out["youtube"]["hashtags"])
        out["youtube"]["description"] = desc[:4800]
    tags = _clean_tags(yt.get("tags"))
    if len(tags) >= YT_TAGS[0]:
        tags = [t for t in tags if t.lower() != "shorts"][: YT_TAGS[1] - 1] + ["Shorts"]
        out["youtube"]["tags"] = tags
    if str(ig.get("hook") or "").strip():
        out["instagram"]["hook"] = _trim(str(ig["hook"]), 125)
    if str(ig.get("caption") or "").strip():
        out["instagram"]["caption"] = str(ig["caption"]).strip()[:2000]
    if str(ig.get("cta") or "").strip():
        out["instagram"]["cta"] = str(ig["cta"]).strip()[:200]
    igt = _clean_tags(ig.get("hashtags"), "#")
    if len(igt) >= IG_TAGS[0]:
        out["instagram"]["hashtags"] = igt[: IG_TAGS[1]]
    if str(fb.get("caption") or "").strip():
        out["facebook"]["caption"] = str(fb["caption"]).strip()[:2000]
    fbt = _clean_tags(fb.get("hashtags"), "#")
    if len(fbt) >= FB_TAGS[0]:
        out["facebook"]["hashtags"] = fbt[: FB_TAGS[1]]
    y, i, f = out["youtube"], out["instagram"], out["facebook"]
    y["copy_text"] = f"{y['title']}\n\n{y['description']}\n\nTags: {', '.join(y['tags'])}"
    if i["cta"] not in i["caption"]:
        i["caption"] = (i["caption"] + "\n\n" + i["cta"]).strip()
    i["copy_text"] = f"{i['caption']}\n.\n.\n{' '.join(i['hashtags'])}"
    f["copy_text"] = f"{f['caption']}\n\n{' '.join(f['hashtags'])}"
    return out


def generate_packages(text: str, topic_keywords: Sequence[str] = (), use_llm: str = "auto",
                      transport: Optional[Transport] = None) -> dict:
    """The three post packages for a clip. use_llm: "auto" (a free-tier LLM only if GROQ_API_KEY / GEMINI_API_KEY is set),
    "off" (offline only), "groq" or "gemini". Never raises for an LLM problem - it falls back offline and warns."""
    base = build_offline(text, topic_keywords)
    if use_llm == "off":
        return base
    provider = llm_provider() if use_llm == "auto" else use_llm
    if not provider or not str(text).strip():
        return base
    try:
        merged = merge_llm(base, llm_packages(text, base["keywords"], provider, transport))
        merged["source"] = provider
        return merged
    except Exception as exc:  # noqa: BLE001 - offline, bad key, quota, malformed JSON ...
        base["warnings"].append(f"The {provider} caption writer could not be used ({str(exc)[:100]}); the offline captions were used.")
        return base
