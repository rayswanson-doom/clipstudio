"""Custom script + action narration (scorer), camera-move rendering (visual_cues), and the pipeline hookups
for OCR and narration."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import ocr_engine  # noqa: E402
import pipeline  # noqa: E402
import scorer  # noqa: E402
import visual_cues as V  # noqa: E402
from tests.test_pipeline import FakeVoice, three_sentence_words  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, probe, square_detector  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class ActionParsingTests(unittest.TestCase):
    def test_beats_and_effects(self):
        beats = scorer.parse_action_prompt("Zoom in on the logo. Then pan right to the product; flash at the price\nFade to black")
        self.assertEqual([b["effect"] for b in beats], ["zoom_in", "pan_right", "flash", "fade"])
        self.assertEqual(beats[0]["text"], "Zoom in on the logo")

    def test_specific_phrases_win_over_general_ones(self):
        self.assertEqual(scorer.effect_for("zoom out slowly"), "zoom_out")
        self.assertEqual(scorer.effect_for("Pull back to reveal the room"), "zoom_out")
        self.assertEqual(scorer.effect_for("pan left"), "pan_left")
        self.assertEqual(scorer.effect_for("the drone looks great"), "none")

    def test_empty_and_noise(self):
        self.assertEqual(scorer.parse_action_prompt(None), [])
        self.assertEqual(scorer.parse_action_prompt("  ;; . \n"), [])
        self.assertEqual(scorer.parse_action_prompt("1. zoom in\n2) pan left")[1]["text"], "pan left")


class NarrationTests(unittest.TestCase):
    SCRIPT = "Welcome to my channel. [zoom in] Today we test a drone! It flies fast. Buy it now."

    def test_custom_script_is_used_verbatim_without_directions(self):
        n = scorer.build_narration(custom_script=self.SCRIPT)
        self.assertEqual(n["source"], "custom")
        self.assertEqual(n["script"], "Welcome to my channel. Today we test a drone! It flies fast. Buy it now.")
        self.assertNotIn("[", n["script"])

    def test_inline_direction_becomes_a_cue_on_its_sentence(self):
        n = scorer.build_narration(custom_script=self.SCRIPT)
        (cue,) = n["cues"]
        self.assertEqual((cue["effect"], cue["sentence_index"]), ("zoom_in", 1))
        self.assertGreater(cue["start"], 0)

    def test_action_beats_spread_across_sentences_in_order(self):
        n = scorer.build_narration(custom_script="One. Two. Three. Four.", action_prompt="zoom in; pan right; flash; shake")
        self.assertEqual([c["sentence_index"] for c in n["cues"]], [0, 1, 2, 3])
        self.assertEqual([c["effect"] for c in n["cues"]], ["zoom_in", "pan_right", "flash", "shake"])
        starts = [c["start"] for c in n["cues"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(c["end"] > c["start"] for c in n["cues"]))

    def test_prompt_only_still_generates_like_before(self):
        n = scorer.build_narration("Make a motivational 20 second TikTok script about running")
        self.assertEqual(n["source"], "generated")
        self.assertEqual(n["cues"], [])
        self.assertIn("running", n["script"].lower())
        self.assertGreater(n["word_count"], 10)

    def test_custom_script_wins_over_prompt(self):
        n = scorer.build_narration("script about tea", custom_script="Only my words here.")
        self.assertEqual(n["script"], "Only my words here.")

    def test_screen_text_is_narrated_after_the_script_and_not_given_actions(self):
        n = scorer.build_narration(custom_script="Look at this.", action_prompt="zoom in; flash",
                                   screen_text=["SALE 50% OFF", "  ", "x"])
        self.assertTrue(n["script"].endswith("The text on screen reads: SALE 50% OFF."))
        self.assertTrue(all(c["sentence_index"] == 0 for c in n["cues"]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            scorer.build_narration()
        with self.assertRaises(ValueError):
            scorer.build_narration(custom_script="[zoom in] [pan left]")

    def test_align_cues_stretches_to_real_audio_length(self):
        n = scorer.build_narration(custom_script="One two three four. Five six seven eight.", action_prompt="zoom in; flash")
        out = scorer.align_cues(n["cues"], n["estimated_seconds"], n["estimated_seconds"] * 2)
        self.assertEqual(out[1]["start"], round(n["cues"][1]["start"] * 2, 2))
        self.assertEqual(scorer.align_cues(n["cues"], 0, 5)[0]["start"], n["cues"][0]["start"])


class EffectCurveTests(unittest.TestCase):
    def test_zoom_in_ramps_then_relaxes(self):
        c = [{"effect": "zoom_in", "start": 1, "end": 3}]
        s = lambda t: V.frame_state(c, t)["scale"]
        self.assertEqual(s(0.5), 1.0)
        self.assertLess(s(1.5), s(2.5))
        self.assertAlmostEqual(s(3.0), 1 + V.ZOOM_AMOUNT, places=2)
        self.assertLess(s(3.2), s(3.0))
        self.assertEqual(s(3.6), 1.0)  # fully eased back: no snap

    def test_zoom_out_starts_zoomed_and_ends_normal(self):
        c = [{"effect": "zoom_out", "start": 0, "end": 2}]
        self.assertGreater(V.frame_state(c, 0.4)["scale"], 1.1)
        self.assertEqual(V.frame_state(c, 2.5)["scale"], 1.0)

    def test_pan_direction_and_relax(self):
        r = lambda e: V.frame_state([{"effect": e, "start": 0, "end": 2}], 1.9)["tx"]
        self.assertGreater(r("pan_right"), 0)
        self.assertLess(r("pan_left"), 0)
        self.assertEqual(V.frame_state([{"effect": "pan_right", "start": 0, "end": 2}], 3.0)["tx"], 0.0)

    def test_transients(self):
        self.assertGreater(V.frame_state([{"effect": "flash", "start": 1, "end": 2}], 1.05)["white"], 0.5)
        self.assertEqual(V.frame_state([{"effect": "flash", "start": 1, "end": 2}], 1.5)["white"], 0.0)
        self.assertLess(V.frame_state([{"effect": "fade", "start": 0, "end": 2}], 0.1)["brightness"], 0.3)
        self.assertGreater(V.frame_state([{"effect": "shake", "start": 0, "end": 2}], 0.1)["shake"], 0.5)

    def test_notes_do_nothing(self):
        c = [{"effect": "none", "start": 0, "end": 5}]
        self.assertFalse(V.has_effects(c))
        self.assertEqual(V.frame_state(c, 1)["scale"], 1.0)
        self.assertTrue(V.has_effects([{"effect": "zoom_in"}]))


@unittest.skipUnless(cv2 and FFMPEG, "needs opencv + ffmpeg")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.d.name, "src.mp4")
        make_source(cls.src)  # 640x360, 3 s, with audio

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def frame(self, path, n):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        cap.release()
        self.assertTrue(ok)
        return fr

    def test_flash_and_zoom_change_the_picture_and_keep_audio_and_length(self):
        out = os.path.join(self.d.name, "cued.mp4")
        r = V.apply_cues_to_video(self.src, [{"effect": "flash", "start": 1.0, "end": 2.0},
                                             {"effect": "zoom_in", "start": 2.0, "end": 3.0}], out, ffmpeg_path=FFMPEG)
        self.assertEqual(r["cues_applied"], 2)
        info = probe(out)
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertAlmostEqual(float(info["format"]["duration"]), 3.0, delta=0.25)
        fps = cv2.VideoCapture(self.src).get(cv2.CAP_PROP_FPS)
        flash = self.frame(out, int(1.05 * fps))
        plain = self.frame(self.src, int(1.05 * fps))
        self.assertGreater(flash.mean(), plain.mean() + 40)  # white flash is clearly brighter
        late_src, late_out = self.frame(self.src, int(2.9 * fps)), self.frame(out, int(2.9 * fps))
        self.assertGreater(np.abs(late_src.astype(int) - late_out.astype(int)).mean(), 2.0)  # zoomed

    def test_no_usable_cues_leaves_the_file_alone(self):
        out = os.path.join(self.d.name, "none.mp4")
        r = V.apply_cues_to_video(self.src, [{"effect": "none", "start": 0, "end": 1}], out)
        self.assertEqual(r["output_path"], self.src)
        self.assertFalse(os.path.exists(out))

    def test_missing_and_corrupt_input(self):
        with self.assertRaises(FileNotFoundError):
            V.apply_cues_to_video("/no/file.mp4", [{"effect": "flash", "start": 0, "end": 1}], "/tmp/x.mp4")
        junk = os.path.join(self.d.name, "junk.mp4")
        open(junk, "wb").close()
        with self.assertRaises(ValueError):
            V.apply_cues_to_video(junk, [{"effect": "flash", "start": 0, "end": 1}], os.path.join(self.d.name, "j.mp4"))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PipelineOcrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    SEGS = [{"text": "BIG SALE TODAY", "start": 0.0, "end": 2.0, "confidence": 0.9},
            {"text": "LAST ONE", "start": 2.0, "end": 3.0, "confidence": 0.9}]

    def run_clips(self, name, words=None, ocr=None, **kw):
        words = three_sentence_words() if words is None else words
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), mock.patch.object(
            pipeline, "transcribe_audio", return_value=words
        ), mock.patch.object(ocr_engine, "extract_screen_text", return_value=self.SEGS if ocr is None else ocr) as ex:
            res = pipeline.run_vugola_pipeline(
                self.src, "", top_n=2, min_len=1, max_len=2.5, output_dir=os.path.join(self.dir.name, name),
                ffmpeg_path=FFMPEG, detector=square_detector, **kw)
        return res, ex

    def test_ocr_off_by_default(self):
        res, ex = self.run_clips("o1")
        ex.assert_not_called()
        self.assertEqual(res["screen_text"], [])
        self.assertTrue(all(c["screen_text"] == [] for c in res["clips"]))

    def test_ocr_text_is_appended_to_the_transcript_and_clips(self):
        res, ex = self.run_clips("o2", read_screen_text=True)
        ex.assert_called_once()
        kinds = {i["source"] for i in res["transcript"]}
        self.assertEqual(kinds, {"speech", "screen"})
        self.assertEqual([i["start"] for i in res["transcript"]], sorted(i["start"] for i in res["transcript"]))
        self.assertTrue(any("BIG SALE TODAY" in c["screen_text"] for c in res["clips"]))

    def test_silent_video_with_text_uses_the_screen_text(self):
        res, _ = self.run_clips("o3", words=[], read_screen_text=True)
        self.assertGreaterEqual(len(res["clips"]), 1)
        self.assertTrue(any("on-screen" in w or "screen" in w for w in res["warnings"]))

    def test_silent_video_without_text_still_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "No speech"):
            self.run_clips("o4", words=[], ocr=[], read_screen_text=True)

    def test_ocr_failure_is_a_warning_not_a_crash(self):
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), mock.patch.object(
            pipeline, "transcribe_audio", return_value=three_sentence_words()
        ), mock.patch.object(ocr_engine, "extract_screen_text", side_effect=RuntimeError("no backend")):
            res = pipeline.run_vugola_pipeline(self.src, "", top_n=1, min_len=1, max_len=2.5,
                                               output_dir=os.path.join(self.dir.name, "o5"), ffmpeg_path=FFMPEG,
                                               detector=square_detector, read_screen_text=True)
        self.assertTrue(any("Could not read on-screen text" in w and "no backend" in w for w in res["warnings"]))
        self.assertTrue(res["clips"][0]["output_path"])


@unittest.skipUnless(cv2 and FFMPEG, "needs opencv + ffmpeg")
class PipelineNarrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.bg = os.path.join(cls.dir.name, "bg.mp4")
        make_source(cls.bg, with_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def lab(self, name, prompt="", **kw):
        said = {}

        async def voice(text, path, voice="x", **k):
            said["text"] = text
            return await FakeVoice.generate(text, path)

        FakeVoice.seconds = 3
        with mock.patch.object(pipeline, "generate_voiceover", voice), mock.patch.object(pipeline, "drive_status",
                                                                                           return_value="ready"):
            res = pipeline.run_viewmax_pipeline(prompt, "", output_dir=os.path.join(self.dir.name, name),
                                                ffmpeg_path=FFMPEG, **kw)
        return res, said["text"]

    def test_custom_script_is_spoken_exactly_and_prompt_is_optional(self):
        res, spoken = self.lab("n1", custom_script="Hello there. This is my own script.")
        self.assertEqual(spoken, "Hello there. This is my own script.")
        self.assertEqual(res["script"]["source"], "custom")
        self.assertEqual(res["cues_applied"], 0)

    def test_actions_become_camera_moves_in_the_final_video(self):
        res, _ = self.lab("n2", custom_script="First part here. Second part here. Third part here.",
                          action_prompt="zoom in; flash; shake", background_video=self.bg)
        self.assertEqual(res["cues_applied"], 3)
        self.assertEqual(len(res["script"]["shot_list"]), 3)
        self.assertTrue(os.path.isfile(res["video_path"]))
        self.assertFalse(os.path.exists(res["video_path"] + ".cues.mp4"))  # temp file cleaned up
        self.assertAlmostEqual(float(probe(res["video_path"])["format"]["duration"]), 3.0, delta=0.3)

    def test_cue_failure_keeps_the_plain_video_and_warns(self):
        with mock.patch.object(pipeline, "apply_cues_to_video", side_effect=RuntimeError("encoder died")):
            res, _ = self.lab("n3", custom_script="One two. Three four.", action_prompt="zoom in")
        self.assertEqual(res["cues_applied"], 0)
        self.assertTrue(any("encoder died" in w for w in res["warnings"]))
        self.assertTrue(os.path.getsize(res["video_path"]) > 1000)

    def test_background_text_is_narrated_after_the_script(self):
        segs = [{"text": "FLASH SALE", "start": 0, "end": 2, "confidence": .9}]
        with mock.patch.object(ocr_engine, "extract_screen_text", return_value=segs):
            res, spoken = self.lab("n4", custom_script="Check this out.", background_video=self.bg,
                                   read_background_text=True)
        self.assertEqual(spoken, "Check this out. The text on screen reads: FLASH SALE.")
        self.assertEqual(res["screen_text"], ["FLASH SALE"])

    def test_background_text_without_a_video_warns(self):
        res, _ = self.lab("n5", custom_script="Hi there.", read_background_text=True)
        self.assertTrue(any("Add a background video" in w for w in res["warnings"]))

    def test_nothing_to_say_is_rejected_early(self):
        with self.assertRaises(ValueError):
            pipeline.run_viewmax_pipeline("  ", None, custom_script=" ", output_dir=self.dir.name)


if __name__ == "__main__":
    unittest.main()
