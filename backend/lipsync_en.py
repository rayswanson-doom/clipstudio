"""
backend/lipsync_en.py  -  English text -> Oculus visemes with relative durations.

A Python port of TalkingHead's lipsync-en.mjs (https://github.com/met4citizen/TalkingHead, MIT,
(c) Mika Suominen), which in turn adapts the NRL letter-to-sound rules (Elovitz, Johnson, McHugh and
Shore, 1976). The RULES table below is generated from that file by tools/gen_lipsync.py, and
tests/test_lipsync_en.py checks this port against the original JavaScript running in node.

    words_to_visemes("hello")        -> {"visemes": [...], "times": [...], "durations": [...]}
    text_to_viseme_timeline(text, total_seconds) -> [{"viseme","start","end","level"}]

Visemes are Oculus names: aa E I O U PP SS TH DD FF kk nn RR CH sil.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

RULES = {
    'A': [
        '[A] =aa',
        ' [ARE] =aa RR',
        ' [AR]O=aa RR',
        '[AR]#=E RR',
        ' ^[AS]#=E SS',
        '[A]WA=aa',
        '[AW]=aa',
        ' :[ANY]=E nn I',
        '[A]^+#=E',
        '#:[ALLY]=aa nn I',
        ' [AL]#=aa nn',
        '[AGAIN]=aa kk E nn',
        '#:[AG]E=I kk',
        '[A]^+:#=aa',
        ':[A]^+ =E',
        '[A]^%=E',
        ' [ARR]=aa RR',
        '[ARR]=aa RR',
        ' :[AR] =aa RR',
        '[AR] =E',
        '[AR]=aa RR',
        '[AIR]=E RR',
        '[AI]=E',
        '[AY]=E',
        '[AU]=aa',
        '#:[AL] =aa nn',
        '#:[ALS] =aa nn SS',
        '[ALK]=aa kk',
        '[AL]^=aa nn',
        ' :[ABLE]=E PP aa nn',
        '[ABLE]=aa PP aa nn',
        '[ANG]+=E nn kk',
        '[A]=aa',
    ],
    'B': [
        ' [BE]^#=PP I',
        '[BEING]=PP I I nn',
        ' [BOTH] =PP O TH',
        ' [BUS]#=PP I SS',
        '[BUIL]=PP I nn',
        '[B]=PP',
    ],
    'C': [
        ' [CH]^=kk',
        '^E[CH]=kk',
        '[CH]=CH',
        ' S[CI]#=SS aa',
        '[CI]A=SS',
        '[CI]O=SS',
        '[CI]EN=SS',
        '[C]+=SS',
        '[CK]=kk',
        '[COM]%=kk aa PP',
        '[C]=kk',
    ],
    'D': [
        '#:[DED] =DD I DD',
        '.E[D] =DD',
        '#^:E[D] =DD',
        ' [DE]^#=DD I',
        ' [DO] =DD U',
        ' [DOES]=DD aa SS',
        ' [DOING]=DD U I nn',
        ' [DOW]=DD aa',
        '[DU]A=kk U',
        '[D]=DD',
    ],
    'E': [
        '#:[E] =',
        "'^:[E] =",
        ' :[E] =I',
        '#[ED] =DD',
        '#:[E]D =',
        '[EV]ER=E FF',
        '[E]^%=I',
        '[ERI]#=I RR I',
        '[ERI]=E RR I',
        '#:[ER]#=E',
        '[ER]#=E RR',
        '[ER]=E',
        ' [EVEN]=I FF E nn',
        '#:[E]W=',
        '@[EW]=U',
        '[EW]=I U',
        '[E]O=I',
        '#:&[ES] =I SS',
        '#:[E]S =',
        '#:[ELY] =nn I',
        '#:[EMENT]=PP E nn DD',
        '[EFUL]=FF U nn',
        '[EE]=I',
        '[EARN]=E nn',
        ' [EAR]^=E',
        '[EAD]=E DD',
        '#:[EA] =I aa',
        '[EA]SU=E',
        '[EA]=I',
        '[EIGH]=E',
        '[EI]=I',
        ' [EYE]=aa',
        '[EY]=I',
        '[EU]=I U',
        '[E]=E',
    ],
    'F': [
        '[FUL]=FF U nn',
        '[F]=FF',
    ],
    'G': [
        '[GIV]=kk I FF',
        ' [G]I^=kk',
        '[GE]T=kk E',
        'SU[GGES]=kk kk E SS',
        '[GG]=kk',
        ' B#[G]=kk',
        '[G]+=kk',
        '[GREAT]=kk RR E DD',
        '#[GH]=',
        '[G]=kk',
    ],
    'H': [
        ' [HAV]=I aa FF',
        ' [HERE]=I I RR',
        ' [HOUR]=aa EE',
        '[HOW]=I aa',
        '[H]#=I',
        '[H]=',
    ],
    'I': [
        ' [IN]=I nn',
        ' [I] =aa',
        '[IN]D=aa nn',
        '[IER]=I E',
        '#:R[IED] =I DD',
        '[IED] =aa DD',
        '[IEN]=I E nn',
        '[IE]T=aa E',
        ' :[I]%=aa',
        '[I]%=I',
        '[IE]=I',
        '[I]^+:#=I',
        '[IR]#=aa RR',
        '[IZ]%=aa SS',
        '[IS]%=aa SS',
        '[I]D%=aa',
        '+^[I]^+=I',
        '[I]T%=aa',
        '#^:[I]^+=I',
        '[I]^+=aa',
        '[IR]=E',
        '[IGH]=aa',
        '[ILD]=aa nn DD',
        '[IGN] =aa nn',
        '[IGN]^=aa nn',
        '[IGN]%=aa nn',
        '[IQUE]=I kk',
        '[I]=I',
    ],
    'J': [
        '[J]=kk',
    ],
    'K': [
        ' [K]N=',
        '[K]=kk',
    ],
    'L': [
        '[LO]C#=nn O',
        'L[L]=',
        '#^:[L]%=aa nn',
        '[LEAD]=nn I DD',
        '[L]=nn',
    ],
    'M': [
        '[MOV]=PP U FF',
        '[M]=PP',
    ],
    'N': [
        'E[NG]+=nn kk',
        '[NG]R=nn kk',
        '[NG]#=nn kk',
        '[NGL]%=nn kk aa nn',
        '[NG]=nn',
        '[NK]=nn kk',
        ' [NOW] =nn aa',
        '[N]=nn',
    ],
    'O': [
        '[OF] =aa FF',
        '[OROUGH]=E O',
        '#:[OR] =E',
        '#:[ORS] =E SS',
        '[OR]=aa RR',
        ' [ONE]=FF aa nn',
        '[OW]=O',
        ' [OVER]=O FF E',
        '[OV]=aa FF',
        '[O]^%=O',
        '[O]^EN=O',
        '[O]^I#=O',
        '[OL]D=O nn',
        '[OUGHT]=aa DD',
        '[OUGH]=aa FF',
        ' [OU]=aa',
        'H[OU]S#=aa',
        '[OUS]=aa SS',
        '[OUR]=aa RR',
        '[OULD]=U DD',
        '^[OU]^L=aa',
        '[OUP]=U OO',
        '[OU]=aa',
        '[OY]=O',
        '[OING]=O I nn',
        '[OI]=O',
        '[OOR]=aa RR',
        '[OOK]=U kk',
        '[OOD]=U DD',
        '[OO]=U',
        '[O]E=O',
        '[O] =O',
        '[OA]=O',
        ' [ONLY]=O nn nn I',
        ' [ONCE]=FF aa nn SS',
        "[ON'T]=O nn DD",
        'C[O]N=aa',
        '[O]NG=aa',
        ' ^:[O]N=aa',
        'I[ON]=aa nn',
        '#:[ON] =aa nn',
        '#^[ON]=aa nn',
        '[O]ST =O',
        '[OF]^=aa FF',
        '[OTHER]=aa TH E',
        '[OSS] =aa SS',
        '#^:[OM]=aa PP',
        '[O]=aa',
    ],
    'P': [
        '[PH]=FF',
        '[PEOP]=PP I PP',
        '[POW]=PP aa',
        '[PUT] =PP U DD',
        '[P]=PP',
    ],
    'Q': [
        '[QUAR]=kk FF aa RR',
        '[QU]=kk FF',
        '[Q]=kk',
    ],
    'R': [
        ' [RE]^#=RR I',
        '[R]=RR',
    ],
    'S': [
        '[SH]=SS',
        '#[SION]=SS aa nn',
        '[SOME]=SS aa PP',
        '#[SUR]#=SS E',
        '[SUR]#=SS E',
        '#[SU]#=SS U',
        '#[SSU]#=SS U',
        '#[SED] =SS DD',
        '#[S]#=SS',
        '[SAID]=SS E DD',
        '^[SION]=SS aa nn',
        '[S]S=',
        '.[S] =SS',
        '#:.E[S] =SS',
        '#^:##[S] =SS',
        '#^:#[S] =SS',
        'U[S] =SS',
        ' :#[S] =SS',
        ' [SCH]=SS kk',
        '[S]C+=',
        '#[SM]=SS PP',
        "#[SN]'=SS aa nn",
        '[S]=SS',
    ],
    'T': [
        ' [THE] =TH aa',
        '[TO] =DD U',
        '[THAT] =TH aa DD',
        ' [THIS] =TH I SS',
        ' [THEY]=TH E',
        ' [THERE]=TH E RR',
        '[THER]=TH E',
        '[THEIR]=TH E RR',
        ' [THAN] =TH aa nn',
        ' [THEM] =TH E PP',
        '[THESE] =TH I SS',
        ' [THEN]=TH E nn',
        '[THROUGH]=TH RR U',
        '[THOSE]=TH O SS',
        '[THOUGH] =TH O',
        ' [THUS]=TH aa SS',
        '[TH]=TH',
        '#:[TED] =DD I DD',
        'S[TI]#N=CH',
        '[TI]O=SS',
        '[TI]A=SS',
        '[TIEN]=SS aa nn',
        '[TUR]#=CH E',
        '[TU]A=CH U',
        ' [TWO]=DD U',
        '[T]=DD',
    ],
    'U': [
        ' [UN]I=I U nn',
        ' [UN]=aa nn',
        ' [UPON]=aa PP aa nn',
        '@[UR]#=U RR',
        '[UR]#=I U RR',
        '[UR]=E',
        '[U]^ =aa',
        '[U]^^=aa',
        '[UY]=aa',
        ' G[U]#=',
        'G[U]%=',
        'G[U]#=FF',
        '#N[U]=I U',
        '@[U]=I',
        '[U]=I U',
    ],
    'V': [
        '[VIEW]=FF I U',
        '[V]=FF',
    ],
    'W': [
        ' [WERE]=FF E',
        '[WA]S=FF aa',
        '[WA]T=FF aa',
        '[WHERE]=FF E RR',
        '[WHAT]=FF aa DD',
        '[WHOL]=I O nn',
        '[WHO]=I U',
        '[WH]=FF',
        '[WAR]=FF aa RR',
        '[WOR]^=FF E',
        '[WR]=RR',
        '[W]=FF',
    ],
    'X': [
        ' [X]=SS',
        '[X]=kk SS',
    ],
    'Y': [
        '[YOUNG]=I aa nn',
        ' [YOU]=I U',
        ' [YES]=I E SS',
        ' [Y]=I',
        '#^:[Y] =I',
        '#^:[Y]I=I',
        ' :[Y] =aa',
        ' :[Y]#=aa',
        ' :[Y]^+:#=I',
        ' :[Y]^#=I',
        '[Y]=I',
    ],
    'Z': [
        '[Z]=SS',
    ],
}

_OPS = {
    "#": "[AEIOUY]+", ".": "[BDVGJLMNRWZ]", "%": "(?:ER|E|ES|ED|ING|ELY)", "&": "(?:[SCGZXJ]|CH|SH)",
    "@": "(?:[TSRDLZNJ]|TH|CH|SH)", "^": "[BCDFGHJKLMNPQRSTVWXZ]", "+": "[EIY]",
    ":": "[BCDFGHJKLMNPQRSTVWXZ]*", " ": r"\b",
}

VISEME_DURATIONS = {"aa": 0.95, "E": 0.90, "I": 0.92, "O": 0.96, "U": 0.95, "PP": 1.08, "SS": 1.23, "TH": 1.0,
                    "DD": 1.05, "FF": 1.00, "kk": 1.21, "nn": 0.88, "RR": 0.88, "CH": 1.0, "sil": 1.0}
SPECIAL_DURATIONS = {" ": 1, ",": 3, "-": 0.5, "'": 0.5}
VISEMES = ("aa", "E", "I", "O", "U", "PP", "SS", "TH", "DD", "FF", "kk", "nn", "RR", "CH", "sil")


def _compile() -> Dict[str, list]:
    out: Dict[str, list] = {}
    for key, rules in RULES.items():
        compiled = []
        for rule in rules:
            l, r_, e = rule.index("["), rule.index("]"), rule.index("=")
            left, letters, right, vis = rule[:l], rule[l + 1:r_], rule[r_ + 1:e], rule[e + 1:]
            ctx = list(letters)
            ctx[0] = ctx[0].lower()
            exp = "".join(_OPS.get(x, re.escape(x) if not x.isalnum() else x) for x in left)
            exp += "".join(ctx) + "".join(_OPS.get(x, re.escape(x) if not x.isalnum() else x) for x in right)
            compiled.append((re.compile(exp), len(ctx), vis.split(" ") if vis else []))
        out[key] = compiled
    return out


_COMPILED = _compile()

_DIGITS = ["oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
        9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
        30: "thirtieth", 40: "fortieth", 50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
        90: "ninetieth"}
_SYMBOLS = {"%": "percent", "\u20ac": "euros", "&": "and", "+": "plus", "$": "dollars"}


def _tens(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    return (_TENS[n // 10] + " " + _ONES[n % 10]).strip()


def _hundreds(n: int) -> str:
    return (_ONES[n // 100] + " hundred " + _tens(n % 100)) if n > 99 else _tens(n)


def _thousands(n: int) -> str:
    return (_hundreds(n // 1000) + " thousand " + _hundreds(n % 1000)) if n >= 1000 else _hundreds(n)


def _millions(n: int) -> str:
    return (_millions(n // 1000000) + " million " + _thousands(n % 1000000)) if n >= 1000000 else _thousands(n)


def _digit_by_digit(s: str) -> str:
    return " ".join(_DIGITS[int(c)] for c in s if c.isdigit())


def number_to_words(num: str, not_special: bool = False) -> str:
    """Read a number the way TalkingHead does (years in pairs, long codes digit by digit)."""
    if num in ("0", "-0"):
        return "zero"
    if num.startswith("-"):
        return "minus " + number_to_words(num[1:], not_special)
    if "." in num:
        a, b = num.split(".", 1)
        return (number_to_words(a, not_special) + " point " + _digit_by_digit(b)).strip()
    if num.startswith("0"):
        return _digit_by_digit(num)
    n = int(num)
    if not not_special and ((99 < n < 1000 and n % 100) or 10000 < n < 1000000):
        return _digit_by_digit(num)
    if not not_special and (1000 < n < 2000 or 2009 < n < 3000):
        if n % 100:
            return (_tens(int(num[:2])) + " " + _tens(int(num[2:4]) if int(num[2:4]) >= 10 or num[2] != "0" else 0)
                    if num[2] != "0" else _tens(int(num[:2])) + " oh " + _ONES[int(num[3])]).strip()
        return _tens(int(num[:2])) + " hundred"
    return _millions(n).strip()


_DECADES = {20: "twenties", 30: "thirties", 40: "forties", 50: "fifties", 60: "sixties", 70: "seventies",
            80: "eighties", 90: "nineties"}


def _decade(decade: str) -> str:
    num = int(decade)
    short, long_ = len(decade) == 2, len(decade) > 2 and 0 < num <= 3000
    thousands = num // 1000 if long_ and num % 1000 == 0 else None
    hundreds = num // 100 if long_ and not thousands else None
    tens = (num % 100) // 10 * 10 if (short or long_) else None
    parts: List[str] = []
    if thousands:
        parts += [number_to_words(str(thousands)).strip(), "thousands"]
    else:
        if hundreds:
            parts.append(number_to_words(str(hundreds)).strip())
        if tens:
            parts.append(_DECADES.get(tens) or number_to_words(str(tens)).strip() + "s")
        elif hundreds:
            parts.append("hundreds")
        else:
            parts.append(decade)
    return " ".join(parts)


def preprocess_text(s: str) -> str:
    """Symbols and numbers to words; drop what must not be spoken."""
    r = re.sub(r"[%\u20ac&+$]", lambda m: " " + _SYMBOLS[m.group(0)] + " ", s)
    if re.search(r"\d", r):
        def decade(m):
            res = _decade(m.group(1))
            return m.group(0) if res == m.group(1) else res

        r = re.sub(r"\b(\d{2,4})['\u2019]?\s?[sS](?=\s|[.,!?;:]|$)", decade, r)

        def ordinal(m):
            n = int(m.group(1))
            if n in _ORD:
                return _ORD[n]
            tens, ones = (n % 100) // 10 * 10, n % 10
            parts = []
            if n // 100:
                parts += [number_to_words(str(n // 100)), "hundred" if (tens or ones) else "hundredth"]
            if tens:
                parts.append(number_to_words(str(tens)) if ones else _ORD[tens])
            if ones:
                parts.append(_ORD[ones])
            return " ".join(parts)

        r = re.sub(r"\b(\d+)\s*(?:st|nd|rd|th)(?=\s|[.,!?;:]|$)", ordinal, r, flags=re.I)
        r = re.sub(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?",
                   lambda m: number_to_words(m.group(0).replace(",", ""), bool(m.group(1)) or "," in m.group(0)), r)
    r = re.sub(r"(\D)\1\1+", r"\1\1", r).replace("  ", " ")
    r = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", r) if not unicodedata.combining(c)))
    return r.strip()


def words_to_visemes(word: str) -> Dict[str, list]:
    """Letter-to-sound rules for ONE word -> visemes with relative times and durations."""
    words = word.upper()
    o = {"words": words, "visemes": [], "times": [], "durations": []}
    t, i = 0.0, 0
    chars = list(words)
    while i < len(chars):
        c = chars[i]
        rules = _COMPILED.get(c)
        if rules is None:
            i += 1
            t += SPECIAL_DURATIONS.get(c, 0)
            continue
        test = words[:i] + c.lower() + words[i + 1:]
        for regex, move, visemes in rules:
            if regex.search(test):
                for v in visemes:
                    if o["visemes"] and o["visemes"][-1] == v:
                        d = 0.7 * VISEME_DURATIONS.get(v, 1)
                        o["durations"][-1] += d
                        t += d
                    else:
                        d = VISEME_DURATIONS.get(v, 1)
                        o["visemes"].append(v)
                        o["times"].append(t)
                        o["durations"].append(d)
                        t += d
                i += move
                break
        else:
            i += 1
    return o


def word_spans(text: str, total_seconds: float, words: Optional[List[dict]] = None, lead: float = 0.0) -> List[dict]:
    """[{"word","start","end","sentence"}] for every spoken word.

    With `words` (faster-whisper style {"word","start","end"}) the real timing is kept (sentences are
    found by matching the text); without them `total_seconds` is shared between the words by how long
    their visemes are, with longer pauses after , ; : and . ! ?"""
    spans: List[dict] = []
    if words:
        for w in words:
            tok = str(w.get("word", "")).strip()
            if tok:
                spans.append({"word": tok, "start": float(w["start"]), "end": float(w["end"]), "sentence": 0})
    else:
        toks = re.findall(r"\S+", text)
        weights = []
        for tok in toks:
            letters = re.sub(r"[^A-Za-z']", " ", preprocess_text(tok) or tok).replace(" ", "")
            v = words_to_visemes(letters)
            w = (v["times"][-1] + v["durations"][-1]) if v["visemes"] else 1.0
            pause = 2.5 if re.search(r"[.!?]\W*$", tok) else 1.5 if re.search(r"[,;:]\W*$", tok) else 0.0
            weights.append((max(w, 1.0), pause))
        total_w = sum(w + p for w, p in weights) or 1.0
        usable = max(total_seconds - lead, 0.1)
        t = lead
        for tok, (w, p) in zip(toks, weights):
            d = usable * w / total_w
            spans.append({"word": tok, "start": t, "end": t + d, "sentence": 0})
            t += d + usable * p / total_w
    sentence = 0
    for sp in spans:
        sp["sentence"] = sentence
        if re.search(r"[.!?\u2026]\W*$", sp["word"]):
            sentence += 1
    return spans


def text_to_viseme_timeline(text: str, total_seconds: float, words: Optional[List[dict]] = None,
                            lead: float = 0.0) -> List[dict]:
    """Spread the visemes of `text` over `total_seconds` -> [{"viseme","start","end","level"}] sorted by time."""
    items: List[dict] = []
    for sp in word_spans(text, total_seconds, words, lead):
        clean = preprocess_text(sp["word"])
        pieces = clean.split()
        for k, piece in enumerate(pieces):
            letters = re.sub(r"[^A-Za-z']", "", piece)
            if not letters:
                continue
            val = words_to_visemes(letters)
            n = len(val["visemes"])
            d_total = (val["times"][-1] + val["durations"][-1]) if n else 0
            if not n or d_total <= 0:
                continue
            dur = max(sp["end"] - sp["start"], 0.05)
            piece_start = sp["start"] + dur * k / len(pieces)
            piece_dur = min(dur / len(pieces), n * 0.2)  # TalkingHead: never slower than 200 ms per viseme
            level = 0.6 + 0.4 * min(max((piece_dur - n * 0.15) / max(piece_dur, 1e-6), 0.0), 1.0)
            for v, vt, vd in zip(val["visemes"], val["times"], val["durations"]):
                s = piece_start + (vt / d_total) * piece_dur
                items.append({"viseme": v, "start": s, "end": s + (vd / d_total) * piece_dur,
                              "level": 0.9 if v in ("PP", "FF") else round(level, 3)})
    items.sort(key=lambda x: x["start"])
    return items
