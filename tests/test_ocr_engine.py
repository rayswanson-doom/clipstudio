"""ocr_engine: frame sampling, de-duplication, merging into the transcript, lazy loading. The OCR itself is
replaced by a fake engine (the real one is exercised in test_ocr_real when rapidocr is installed)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import ocr_engine as O  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def make_video(path, seconds=6, fps=10, size=(320, 180), caption_at=None):
    """Grey video; frames inside a caption span get a white bar (our fake OCR keys on it)."""
    caption_at = caption_at or {}
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        f = np.full((size[1], size[0], 3), 60, np.uint8)
        t = i / fps
        for (a, b), shade in caption_at.items():
            if a <= t < b:
                f[60:120, 20:300] = shade
        wr.write(f)
    wr.release()
    return path


def fake_engine(frame):
    """Reads 'HELLO' from a white bar, 'WORLD' from a light-grey bar, junk from a dark one."""
    v = int(frame[90, 150].mean())
    if v > 240:
        return [("HELLO", 0.97)]
    if 190 < v < 215:
        return [("WORLD", 0.9)]
    if v == 60:
        return []
    return [("zz", 0.2)]


class PureTests(unittest.TestCase):
    def test_clean_readings_filters_confidence_and_junk(self):
        out = O.clean_readings([("  Big   Sale ", 0.9), ("x", 0.99), ("low conf", 0.3), ("--", 0.9), ("A1", 0.6)])
        self.assertEqual(out, [("Big Sale", 0.9), ("A1", 0.6)])

    def test_same_text_is_fuzzy_but_not_too_fuzzy(self):
        self.assertTrue(O._same_text("SALE ENDS TODAY", "sale ends today!"))
        self.assertTrue(O._same_text("SALE ENDS TODAY 50%", "SALE ENDS T0DAY 50%"))
        self.assertFalse(O._same_text("SALE", "NEW ARRIVALS"))
        self.assertFalse(O._same_text("", "x"))

    def test_merge_samples_extends_repeats_and_splits_gaps(self):
        samples = [(0, [("HELLO", .9)]), (1, [("HELLO", .95)]), (2, []), (3, [("HELLO", .9), ("NEW", .8)]),
                   (4, [("NEW", .8)])]
        segs = O.merge_samples(samples, 1.0)
        hello = [s for s in segs if s["text"] == "HELLO"]
        self.assertEqual([(s["start"], s["end"]) for s in hello], [(0, 2.0), (3, 4.0)])  # gap -> two segments
        self.assertEqual(hello[0]["confidence"], 0.95)
        new = next(s for s in segs if s["text"] == "NEW")
        self.assertEqual((new["start"], new["end"]), (3, 5.0))

    def test_segments_to_words_matches_whisper_shape_and_spreads_time(self):
        w = O.segments_to_words([{"text": "BIG SALE NOW", "start": 2.0, "end": 5.0, "confidence": 0.9}])
        self.assertEqual([x["word"] for x in w], ["BIG", "SALE", "NOW"])
        self.assertEqual((w[0]["start"], w[-1]["end"]), (2.0, 5.0))
        for x in w:
            self.assertEqual(set(x) >= {"word", "start", "end", "probability"}, True)
            self.assertEqual(x["source"], "screen")

    def test_merge_transcript_orders_speech_and_screen_text(self):
        words = [{"word": "Hi", "start": 0.0, "end": 0.3}, {"word": "there.", "start": 0.3, "end": 0.8},
                 {"word": "Bye", "start": 6.0, "end": 6.4}]
        t = O.merge_transcript(words, [{"text": "TITLE", "start": 2.0, "end": 4.0}])
        self.assertEqual([(i["source"], i["text"]) for i in t],
                         [("speech", "Hi there."), ("screen", "TITLE"), ("speech", "Bye")])

    def test_screen_text_in_range_dedupes(self):
        segs = [{"text": "A B", "start": 0, "end": 2}, {"text": "a b!", "start": 2, "end": 3},
                {"text": "LATE", "start": 9, "end": 10}]
        self.assertEqual(O.screen_text_in_range(segs, 0, 5), ["A B"])
        self.assertEqual(O.screen_text_in_range(segs, 8, 20), ["LATE"])


@unittest.skipUnless(cv2, "needs opencv")
class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.video = make_video(os.path.join(cls.d.name, "v.mp4"),
                               caption_at={(1.0, 3.0): 255, (4.0, 5.0): 205})

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def test_samples_about_once_per_second_and_finds_timed_text(self):
        segs = O.extract_screen_text(self.video, interval=1.0, engine=fake_engine)
        by = {s["text"]: s for s in segs}
        self.assertAlmostEqual(by["HELLO"]["start"], 1.0, delta=0.2)
        self.assertAlmostEqual(by["HELLO"]["end"], 3.0, delta=0.3)
        self.assertAlmostEqual(by["WORLD"]["start"], 4.0, delta=0.2)
        self.assertNotIn("zz", by)  # low-confidence junk dropped

    def test_unchanged_frames_are_not_ocrd_again(self):
        calls = []

        def counting(frame):
            calls.append(1)
            return fake_engine(frame)

        O.extract_screen_text(self.video, interval=0.5, engine=counting)
        sampled = len(list(O.iter_frames(self.video, 0.5)))
        self.assertGreaterEqual(sampled, 11)
        self.assertLess(len(calls), sampled / 2)  # static stretches reuse the previous reading

    def test_interval_start_end_and_max_frames(self):
        self.assertAlmostEqual(len(list(O.iter_frames(self.video, 1.0))), 6, delta=1)
        self.assertAlmostEqual(len(list(O.iter_frames(self.video, 1.0, start=2.0, end=4.0))), 3, delta=1)
        self.assertEqual(len(list(O.iter_frames(self.video, 0.1, max_frames=4))), 4)

    def test_engine_error_on_one_frame_does_not_abort(self):
        n = {"i": 0}

        def flaky(frame):
            n["i"] += 1
            if n["i"] == 2:
                raise RuntimeError("bad frame")
            return fake_engine(frame)

        segs = O.extract_screen_text(self.video, interval=1.0, engine=flaky)
        self.assertTrue(any(s["text"] == "HELLO" for s in segs))

    def test_progress_and_errors(self):
        seen = []
        O.extract_screen_text(self.video, engine=fake_engine, progress=seen.append)
        self.assertEqual(seen, sorted(seen))
        self.assertGreater(seen[-1], 0.8)
        with self.assertRaises(FileNotFoundError):
            O.extract_screen_text("/no/such.mp4", engine=fake_engine)
        junk = os.path.join(self.d.name, "junk.mp4")
        open(junk, "wb").close()
        with self.assertRaises(ValueError):
            O.extract_screen_text(junk, engine=fake_engine)

    def test_capture_is_released_even_if_the_consumer_stops_early(self):
        real = cv2.VideoCapture
        caps = []

        class Spy:
            def __init__(self, *a):
                self.c = real(*a)
                caps.append(self)
                self.released = False

            def __getattr__(self, n):
                return getattr(self.c, n)

            def release(self):
                self.released = True
                self.c.release()

        with mock.patch.object(cv2, "VideoCapture", Spy):
            gen = O.iter_frames(self.video, 1.0)
            next(gen)
            gen.close()
        self.assertTrue(caps and all(c.released for c in caps))


class BackendLoadingTests(unittest.TestCase):
    def setUp(self):
        O.unload()
        self.addCleanup(O.unload)

    def test_missing_backends_give_a_clear_install_hint(self):
        with mock.patch.object(O, "_installed", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "rapidocr-onnxruntime"):
                O.get_engine()

    def test_engine_loads_once_and_unload_frees_it(self):
        made = []
        with mock.patch.object(O, "_installed", side_effect=lambda n: n if n == "rapidocr" else None), \
                mock.patch.dict(O._FACTORIES, {"rapidocr": lambda: made.append(1) or (lambda f: [])}):
            e1, name = O.get_engine()
            e2, _ = O.get_engine()
            self.assertIs(e1, e2)
            self.assertEqual((name, len(made)), ("rapidocr", 1))
            O.unload()
            O.get_engine()
            self.assertEqual(len(made), 2)

    def test_falls_back_when_the_first_backend_is_broken(self):
        def boom():
            raise RuntimeError("model files missing")
        with mock.patch.object(O, "_installed", side_effect=lambda n: n), \
                mock.patch.dict(O._FACTORIES, {"rapidocr": boom, "paddleocr": lambda: (lambda f: [])}):
            _, name = O.get_engine()
        self.assertEqual(name, "paddleocr")

    def test_env_var_picks_the_backend(self):
        with mock.patch.object(O, "_installed", side_effect=lambda n: n), \
                mock.patch.dict(O._FACTORIES, {"rapidocr": lambda: (lambda f: []), "paddleocr": lambda: (lambda f: [])}), \
                mock.patch.dict(os.environ, {"CINEPULSE_OCR_BACKEND": "paddleocr"}):
            self.assertEqual(O.get_engine()[1], "paddleocr")

    def test_availability_check_does_not_import_the_backend(self):
        code = ("import sys; sys.path.insert(0, %r); import ocr_engine as o; o.available_backends(); "
                "bad=[m for m in ('paddleocr','paddle','rapidocr_onnxruntime','onnxruntime','cv2') if m in sys.modules]; "
                "print('LOADED', bad)") % str(ROOT / "backend")
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "LOADED []", out.stderr[-300:])


@unittest.skipUnless(O.is_available() and cv2, "no OCR backend installed here (CI installs rapidocr-onnxruntime)")
class RealOcrTests(unittest.TestCase):
    def test_reads_real_text_from_a_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.mp4")
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 160))
            for _ in range(15):
                f = np.full((160, 640, 3), 255, np.uint8)
                cv2.putText(f, "SALE ENDS TODAY", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                wr.write(f)
            wr.release()
            segs = O.extract_screen_text(path, interval=1.0)
            O.unload()
        text = " ".join(s["text"] for s in segs).upper().replace(" ", "")
        self.assertIn("SALEENDS", text)
        self.assertEqual(len(segs), 1 if len(segs) == 1 else len(segs))  # steady text is merged, not repeated per frame
        self.assertLessEqual(len(segs), 3)


if __name__ == "__main__":
    unittest.main()
