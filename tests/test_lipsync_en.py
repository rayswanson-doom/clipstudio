"""The Python lip-sync port must give the same visemes as TalkingHead's original JavaScript (run in node)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import lipsync_en as L  # noqa: E402

WORDS = ["hello", "world", "the", "this", "through", "though", "people", "welcome", "podcast", "channel",
         "science", "knowledge", "beautiful", "amazing", "quickly", "thought", "enough", "whole", "queue",
         "extra", "yes", "you", "young", "one", "once", "only", "their", "there", "them", "photograph",
         "machine", "question", "nation", "measure", "usually", "eight", "ninety", "twenty", "dollars", "percent"]
TEXTS = ["In 1995 there were 3.5 percent more", "Call 5551234 at 21st street", "It costs $20 & 5% tax",
         "Chapter 7, 1,250 people in 2005", "the year 1900 and 2024", "The 70s were 100% fun"]
JS_ORACLE = """
import { LipsyncEn } from '%s';
const l = new LipsyncEn();
const words = %s, texts = %s;
console.log(JSON.stringify({
  words: words.map(w => { const o = l.wordsToVisemes(w); return [o.visemes, o.times.map(x => +x.toFixed(6)), o.durations.map(x => +x.toFixed(6))]; }),
  texts: texts.map(t => l.preProcessText(t)),
}));
"""
MJS = next((p for p in (
    Path("/tmp/claude-0/ref/265c94fd-TalkingHead-main/TalkingHead-main/modules/lipsync-en.mjs"),
    ROOT / "third_party" / "lipsync-en.mjs") if p.is_file()), None)


class PortTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node") and MJS, "needs node and TalkingHead's lipsync-en.mjs")
    def test_matches_the_javascript_original(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "oracle.mjs"
            script.write_text(JS_ORACLE % (MJS.as_posix(), json.dumps(WORDS), json.dumps(TEXTS)))
            out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        js = json.loads(out.stdout)
        for w, (vis, times, durs) in zip(WORDS, js["words"]):
            got = L.words_to_visemes(w)
            self.assertEqual(got["visemes"], vis, w)
            self.assertEqual([round(x, 6) for x in got["times"]], times, w)
            self.assertEqual([round(x, 6) for x in got["durations"]], durs, w)
        for t, expected in zip(TEXTS, js["texts"]):
            self.assertEqual(L.preprocess_text(t), expected, t)

    def test_known_words(self):
        self.assertEqual(L.words_to_visemes("hello")["visemes"], ["I", "E", "nn", "O"])
        self.assertEqual(L.words_to_visemes("map")["visemes"][0], "PP")
        self.assertEqual(L.words_to_visemes("")["visemes"], [])

    def test_numbers(self):
        self.assertEqual(L.preprocess_text("1995"), "nineteen ninety five")
        self.assertEqual(L.preprocess_text("5%"), "five percent")

    def test_timeline_covers_the_audio_and_is_ordered(self):
        tl = L.text_to_viseme_timeline("Welcome to the show. Today we talk about science!", 6.0)
        self.assertGreater(len(tl), 15)
        self.assertEqual([x["start"] for x in tl], sorted(x["start"] for x in tl))
        self.assertLessEqual(tl[-1]["end"], 6.2)
        self.assertGreaterEqual(tl[0]["start"], 0.0)
        self.assertTrue(all(x["viseme"] in L.VISEMES for x in tl))

    def test_real_word_timestamps_are_respected(self):
        words = [{"word": "Hello", "start": 1.0, "end": 1.5}, {"word": "world", "start": 2.0, "end": 2.6}]
        tl = L.text_to_viseme_timeline("Hello world", 4.0, words=words)
        self.assertGreaterEqual(tl[0]["start"], 1.0)
        self.assertTrue(all(not (1.55 < x["start"] < 1.95) for x in tl))  # silent between the words

    def test_odd_input_never_crashes(self):
        for t in ("", "...", "😀😀", "x" * 500, "ÀÉÎ ñandú", "123 456", "a-b c'd"):
            L.text_to_viseme_timeline(t, 1.0)


if __name__ == "__main__":
    unittest.main()
