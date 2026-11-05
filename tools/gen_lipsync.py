"""Generate backend/lipsync_en.py's RULES table from TalkingHead's lipsync-en.mjs (MIT, Mika Suominen).
usage: python tools/gen_lipsync.py path/to/lipsync-en.mjs > rules.txt"""
import re, sys, json
src = open(sys.argv[1], encoding="utf-8").read()
body = src[src.index("this.rules = {"):src.index("const ops = {")]
rules = {}
for m in re.finditer(r"'([A-Z])':\s*\[(.*?)\n\s*\]", body, re.S):
    rules[m.group(1)] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(2))
print(json.dumps(rules, indent=1))
