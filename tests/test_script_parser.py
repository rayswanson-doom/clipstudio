"""Complex scripts (scenes, speakers, visual lines, timecodes, pauses) and the OCR narration queue."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import ocr_engine  # noqa: E402
import scorer  # noqa: E402

DOC = """SCENE 1: Hook
VISUAL: zoom in on the logo
NARRATOR: Stop scrolling! This is [flash] huge.
[0:06] Did you know? Fun fact: octopuses have three hearts.
(slowly pan right)
[pause 1s]
HOST (V.O.): It cost 5 million dollars.
> flash at the price
Scene 2 - Outro
Subscribe now.
"""


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.d = scorer.parse_script_document(DOC)

    def test_speech_is_clean(self):
        self.assertEqual(self.d["sentences"], ["Stop scrolling!", "This is huge.", "Did you know?",
                                               "Fun fact: octopuses have three hearts.", "It cost 5 million dollars.",
                                               "Subscribe now."])

    def test_directions_attach_to_the_next_spoken_sentence(self):
        ev = {(e["text"], e["sentence_index"], e["source"]) for e in self.d["events"]}
        self.assertIn(("zoom in on the logo", 0, "visual"), ev)
        self.assertIn(("flash", 1, "inline"), ev)
        self.assertIn(("slowly pan right", 4, "visual"), ev)
        self.assertIn(("pause 1s", 4, "pause"), ev)
        self.assertIn(("flash at the price", 5, "visual"), ev)  # trailing line lands on the last sentence

    def test_scenes_and_timecodes(self):
        self.assertEqual([(s["title"], s["sentence_index"]) for s in self.d["scenes"]], [("Hook", 0), ("Outro", 5)])
        self.assertEqual(self.d["sentence_times"], {2: 6.0})

    def test_speech_that_looks_like_labels_is_kept(self):
        d = scorer.parse_script_document("Part of the reason is simple.\nAct now: it ends soon.\nFun fact: cats purr.\n"
                                         "Show: notes are not narrated\nNote: this one is speech.")
        self.assertEqual(d["sentences"], ["Part of the reason is simple.", "Act now: it ends soon.", "Fun fact: cats purr.",
                                          "Note: this one is speech."])
        self.assertEqual(d["scenes"], [])
        self.assertEqual([e["text"] for e in d["events"]], ["notes are not narrated"])

    def test_timecode_formats(self):
        d = scorer.parse_script_document("[1:05.5] First.\nSecond [0:10] line.\n00:20 - Third.")
        self.assertEqual(d["sentence_times"], {0: 65.5, 1: 10.0, 2: 20.0})

    def test_only_directions_is_an_error_in_build(self):
        with self.assertRaises(ValueError):
            scorer.build_narration(custom_script="[zoom in]\nVISUAL: pan right")

    def test_plain_scripts_still_work(self):
        d = scorer.parse_script_document("Hello there. [zoom in] How are you?")
        self.assertEqual(d["sentences"], ["Hello there.", "How are you?"])
        self.assertEqual([(e["text"], e["sentence_index"]) for e in d["events"]], [("zoom in", 1)])

    def test_empty(self):
        self.assertEqual(scorer.parse_script_document("")["sentences"], [])


class BuildNarrationTests(unittest.TestCase):
    def setUp(self):
        self.n = scorer.build_narration(custom_script=DOC, action_prompt="fade in", screen_text=[
            {"text": "SALE ENDS TODAY"}, "Free shipping"])

    def test_script_has_no_directions_but_has_screen_text(self):
        self.assertNotIn("VISUAL", self.n["script"])
        self.assertNotIn("NARRATOR", self.n["script"])
        self.assertTrue(self.n["script"].endswith("The text on screen reads: Free shipping."))
        self.assertIn("The text on screen reads: SALE ENDS TODAY.", self.n["script"])

    def test_effects_and_notes(self):
        effects = {c["text"]: c["effect"] for c in self.n["cues"]}
        self.assertEqual(effects["zoom in on the logo"], "zoom_in")
        self.assertEqual(effects["slowly pan right"], "pan_right")
        self.assertEqual(effects["flash at the price"], "flash")
        self.assertEqual(effects["pause 1s"], "none")
        self.assertEqual(effects["fade in"], "fade")
        for c in self.n["cues"]:
            self.assertGreater(c["end"], c["start"])

    def test_scenes_reported_with_start_times(self):
        self.assertEqual([s["title"] for s in self.n["scenes"]], ["Hook", "Outro"])
        self.assertEqual(self.n["scenes"][0]["start"], 0.0)
        self.assertGreater(self.n["scenes"][1]["start"], 0.0)

    def test_timecode_pins_a_cue_and_survives_alignment(self):
        n = scorer.build_narration(custom_script="[0:06] [wave] Hello world. Another sentence here.")
        wave = next(c for c in n["cues"] if c["text"] == "wave")
        self.assertTrue(wave["pinned"])
        self.assertEqual(wave["start"], 6.0)
        aligned = scorer.align_cues(n["cues"], n["estimated_seconds"], 20.0)
        self.assertEqual(next(c for c in aligned if c["text"] == "wave")["start"], 6.0)
        other = scorer.build_narration(custom_script="Hello world. [zoom in] More words follow here.")
        a = scorer.align_cues(other["cues"], other["estimated_seconds"], other["estimated_seconds"] * 2)
        self.assertGreater(a[0]["start"], other["cues"][0]["start"])  # un-pinned cues still stretch


class NarrationQueueTests(unittest.TestCase):
    def test_dedup_order_confidence_and_noise(self):
        segs = [{"text": "second line", "start": 3.0, "end": 4.0, "confidence": 0.9},
                {"text": "First  line.", "start": 1.0, "end": 2.0, "confidence": 0.8},
                {"text": "first line", "start": 5.0, "end": 6.0, "confidence": 0.99},
                {"text": "x", "start": 0.5, "end": 1.0, "confidence": 0.9},
                {"text": "blurry text", "start": 2.0, "end": 3.0, "confidence": 0.2}]
        q = ocr_engine.narration_queue(segs)
        self.assertEqual([i["text"] for i in q], ["First line", "second line"])
        self.assertEqual(q[0]["narration"], "The text on screen reads: First line.")
        self.assertEqual(len(ocr_engine.narration_queue(segs, max_lines=1)), 1)
        self.assertEqual(ocr_engine.narration_queue([]), [])

    def test_auto_backend_prefers_paddleocr_when_installed(self):
        self.assertEqual(ocr_engine.BACKENDS[0], "paddleocr")
        self.assertIn("rapidocr", ocr_engine.BACKENDS)


if __name__ == "__main__":
    unittest.main()
