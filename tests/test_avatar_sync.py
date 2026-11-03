"""avatar_sync: pure audio/word logic, real rendering with OpenCV + ffmpeg, and the pipeline hookup."""
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import avatar_sync as A  # noqa: E402
import pipeline  # noqa: E402
from tests.test_pipeline import FakeVoice  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, probe  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

FACE = A.Face(120, 150, 460, 600, (350, 640))  # matches cartoon() below
WORDS = [{"word": "a", "start": 0.0, "end": 0.5}, {"word": "b", "start": 1.0, "end": 1.6}]


class FakeLocator:
    def __call__(self, im):
        return FACE

    def close(self):
        pass


def cartoon(path):
    img = np.full((900, 700, 3), (200, 170, 120), np.uint8)
    cv2.ellipse(img, (350, 450), (230, 300), 0, 0, 360, (150, 190, 235), -1)
    for ex in (270, 430):
        cv2.circle(img, (ex, 380), 28, (255, 255, 255), -1)
        cv2.circle(img, (ex, 385), 12, (40, 30, 20), -1)
    cv2.ellipse(img, (350, 640), (60, 14), 0, 0, 180, (60, 60, 170), -1)
    cv2.imwrite(path, img)


def speech_wav(path, seconds=3.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    env = ((np.sin(2 * np.pi * 2.2 * t) > 0.1) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)
    x = 0.4 * env * (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 26000).astype("<i2").tobytes())


class PureLogicTests(unittest.TestCase):
    def test_silence_gives_closed_mouth(self):
        env = A.energy_envelope(np.zeros(16000 * 2, np.float32), 16000, 25)
        self.assertEqual(len(env), 50)
        self.assertEqual(float(A.mouth_openness(env).max()), 0.0)

    def test_loud_audio_opens_mouth_and_stays_in_range(self):
        x = (0.5 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
        o = A.mouth_openness(A.energy_envelope(x, 16000, 25))
        self.assertGreater(o.max(), 0.6)
        self.assertTrue(((o >= 0) & (o <= 1)).all())

    def test_word_gate_closes_gaps_and_ignores_bad_entries(self):
        g = A.word_gate(WORDS + [{"word": "x"}, {"start": "n/a", "end": 1}], 75, 25)
        self.assertGreater(g[5], 0.9)  # inside word a
        self.assertLess(g[18], 0.05)  # gap between words (0.5s - 1.0s)
        self.assertTrue((A.word_gate(None, 10, 25) == 1).all())
        self.assertTrue((A.word_gate([], 10, 25) == 1).all())

    def test_words_silence_the_mouth_between_words_even_with_noise(self):
        env = np.ones(75, np.float32)  # constant loud noise
        o = A.mouth_openness(env, A.word_gate(WORDS, 75, 25))
        self.assertGreater(o[8], 0.5)
        self.assertLess(o[22], 0.1)

    def test_attack_faster_than_release(self):
        env = np.array([0, 1, 1, 0, 0, 0], np.float32)
        o = A.mouth_openness(env)
        self.assertGreater(o[1], 0.5)  # opens within one frame
        self.assertGreater(o[3], 0.2)  # still partly open one frame after

    def test_mouth_keypoint_sanity(self):
        box = (100, 100, 200, 260)
        self.assertEqual(A._sane_mouth(box, (200, 330)), (200, 330))
        self.assertEqual(A._sane_mouth(box, (200, 120))[1], 100 + 0.74 * 260)  # eye-level: rejected
        self.assertEqual(A._sane_mouth(box, None)[0], 200)

    def test_face_scaling(self):
        f = FACE.scaled(2.0, 10, 20)
        self.assertEqual((f.x, f.y, f.w), (230, 280, 920))
        self.assertEqual(f.mouth, (690, 1260))


@unittest.skipUnless(cv2 and FFMPEG, "needs opencv + ffmpeg")
class FileHandlingTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.p = lambda n: os.path.join(self.d.name, n)

    def test_rejects_wrong_extension_and_missing_and_corrupt(self):
        open(self.p("a.gif"), "wb").write(b"GIF89a")
        with self.assertRaisesRegex(ValueError, r"\.jpg or \.png"):
            A.load_avatar(self.p("a.gif"))
        with self.assertRaises(FileNotFoundError):
            A.load_avatar(self.p("nope.png"))
        open(self.p("bad.png"), "wb").write(b"not an image")
        with self.assertRaisesRegex(ValueError, "could not be read"):
            A.load_avatar(self.p("bad.png"))

    def test_png_transparency_and_jpg_load(self):
        rgba = np.zeros((40, 30, 4), np.uint8)
        rgba[10:20, 10:20] = (0, 0, 255, 255)
        cv2.imwrite(self.p("t.png"), rgba)
        img = A.load_avatar(self.p("t.png"))
        self.assertEqual(img.shape, (40, 30, 3))
        self.assertEqual(tuple(img[0, 0]), A.BACKDROP_BGR)  # transparent -> backdrop
        self.assertEqual(tuple(img[15, 15]), (0, 0, 255))
        cv2.imwrite(self.p("a.jpg"), np.full((50, 50, 3), 90, np.uint8))
        self.assertEqual(A.load_avatar(self.p("a.jpg")).shape, (50, 50, 3))

    def test_huge_image_is_downscaled(self):
        cv2.imwrite(self.p("big.png"), np.zeros((3000, 2500, 3), np.uint8))
        self.assertLessEqual(max(A.load_avatar(self.p("big.png")).shape[:2]), A.MAX_IMAGE_SIDE)

    def test_audio_errors(self):
        with self.assertRaises(FileNotFoundError):
            A.decode_audio(self.p("none.wav"))
        open(self.p("junk.wav"), "wb").write(b"junk")
        with self.assertRaises(RuntimeError):
            A.decode_audio(self.p("junk.wav"), FFMPEG)


@unittest.skipUnless(cv2 and FFMPEG, "needs opencv + ffmpeg")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cls.p = lambda self, n: os.path.join(cls.d.name, n)
        cartoon(cls.p(None, "face.png"))
        speech_wav(cls.p(None, "voice.wav"))
        cls.bg = cls.p(None, "bg.mp4")
        make_source(cls.bg, with_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def render(self, name, **kw):
        kw.setdefault("face_locator", lambda im: FACE)
        kw.setdefault("ffmpeg_path", FFMPEG)
        return A.create_avatar_video(self.p("face.png"), self.p("voice.wav"), self.p(name), **kw)

    def frame(self, path, n):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        cap.release()
        self.assertTrue(ok)
        return fr

    def test_full_frame_video_matches_audio_and_size(self):
        r = self.render("full.mp4", words=WORDS)
        self.assertEqual((r["mode"], r["layout"], r["face_found"], r["used_word_timestamps"]),
                         ("mouth-warp", "full", True, True))
        info = probe(r["output_path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), A.DEFAULT_SIZE)
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertAlmostEqual(float(info["format"]["duration"]), 3.0, delta=0.25)

    def test_mouth_really_moves_with_the_voice(self):
        r = self.render("move.mp4", words=WORDS)
        env = A.energy_envelope(A.decode_audio(self.p("voice.wav"), FFMPEG), A.SAMPLE_RATE, A.DEFAULT_FPS)
        o = A.mouth_openness(env, A.word_gate(WORDS, len(env), A.DEFAULT_FPS))
        open_i, shut_i = int(o[:12].argmax()), 19  # frame 19 = gap between the words
        self.assertLess(o[shut_i], 0.15)
        a, b = self.frame(r["output_path"], open_i), self.frame(r["output_path"], shut_i)
        mouth = (slice(800, 1000), slice(150, 600))  # lower face in the 720x1280 frame
        diff = np.abs(a[mouth].astype(int) - b[mouth].astype(int)).mean()
        self.assertGreater(diff, 3.0, "open and closed mouth frames look identical")
        dark_open = (a[mouth].sum(axis=2) < 200).sum()
        dark_shut = (b[mouth].sum(axis=2) < 200).sum()
        self.assertGreater(dark_open, dark_shut + 500)  # mouth cavity is visible when open

    def test_no_face_falls_back_to_pulse_with_warning(self):
        r = self.render("pulse.mp4", face_locator=lambda im: None)
        self.assertEqual((r["mode"], r["face_found"]), ("pulse", False))
        self.assertTrue(any("No face found" in w for w in r["warnings"]))
        self.assertTrue(os.path.getsize(r["output_path"]) > 1000)
        # the pulse must still move the picture
        self.assertGreater(np.abs(self.frame(r["output_path"], 4).astype(int)
                                  - self.frame(r["output_path"], 19).astype(int)).mean(), 0.3)

    def test_detector_failure_becomes_warning_not_crash(self):
        def boom(im):
            raise FileNotFoundError("model missing")
        r = self.render("boom.mp4", face_locator=boom)
        self.assertEqual(r["mode"], "pulse")
        self.assertTrue(any("model missing" in w for w in r["warnings"]))

    def test_background_video_gives_card_layout(self):
        r = self.render("card.mp4", words=WORDS, background_video=self.bg)
        self.assertEqual(r["layout"], "card")
        fr = self.frame(r["output_path"], 10)
        self.assertEqual(fr.shape[:2], (1280, 720)[::1])
        top = fr[:300].mean(axis=(0, 1))  # background video shows above the card
        self.assertLess(abs(float(top.mean()) - 30.0), 25.0)

    def test_bad_background_video_warns_and_goes_full_screen(self):
        open(self.p("junk.mp4"), "wb").write(b"junk")
        r = self.render("badbg.mp4", background_video=self.p("junk.mp4"))
        self.assertEqual(r["layout"], "full")
        self.assertTrue(any("background" in w.lower() for w in r["warnings"]))

    def test_auto_words_uses_transcriber_and_survives_its_failure(self):
        import transcriber
        with mock.patch.object(transcriber, "transcribe_audio", return_value=WORDS) as t:
            r = self.render("auto.mp4", auto_words=True)
        t.assert_called_once()
        self.assertTrue(r["used_word_timestamps"])
        with mock.patch.object(transcriber, "transcribe_audio", side_effect=RuntimeError("no model")):
            r = self.render("auto2.mp4", auto_words=True)
        self.assertFalse(r["used_word_timestamps"])
        self.assertTrue(any("Word timing unavailable" in w for w in r["warnings"]))

    def test_custom_size_is_made_even(self):
        r = self.render("small.mp4", size=(361, 641))
        self.assertEqual(r["size"], (360, 640))

    def test_progress_reaches_one(self):
        seen = []
        self.render("prog.mp4", progress=seen.append)
        self.assertEqual(seen[-1], 1.0)


@unittest.skipUnless(cv2 and FFMPEG, "needs opencv + ffmpeg")
class PipelineHookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.TemporaryDirectory()
        cartoon(os.path.join(cls.d.name, "me.png"))
        cls.img = os.path.join(cls.d.name, "me.png")

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()

    def go(self, **kw):
        out = os.path.join(self.d.name, "out")
        FakeVoice.seconds = 2
        with mock.patch.object(pipeline, "generate_voiceover", FakeVoice.generate), mock.patch.object(
            pipeline, "drive_status", return_value="ready"
        ), mock.patch.object(pipeline, "upload_to_google_drive", return_value="https://drive/v") as up, \
                mock.patch.object(A, "MediaPipeFaceLocator", side_effect=lambda: FakeLocator()), \
                mock.patch("transcriber.transcribe_audio", return_value=[{"word": "x", "start": 0.0, "end": 1.5}]):
            res = pipeline.run_viewmax_pipeline("A 20 second script about running", "FOLD", output_dir=out,
                                                ffmpeg_path=FFMPEG, **kw)
        return res, up

    def test_avatar_video_is_produced_and_uploaded(self):
        res, up = self.go(avatar_image=self.img, avatar_enabled=True)
        self.assertEqual(res["avatar"]["mode"], "mouth-warp")
        self.assertTrue(res["avatar"]["used_word_timestamps"])
        self.assertAlmostEqual(float(probe(res["video_path"])["format"]["duration"]), 2.0, delta=0.25)
        up.assert_called_once()
        self.assertEqual(res["drive_link"], "https://drive/v")

    def test_disabled_keeps_the_old_behaviour(self):
        res, _ = self.go(avatar_image=self.img, avatar_enabled=False)
        self.assertIsNone(res["avatar"])
        v = next(s for s in probe(res["video_path"])["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (1080, 1920))

    def test_enabled_without_image_fails_before_any_work(self):
        with mock.patch.object(pipeline, "generate_voiceover") as gv:
            with self.assertRaisesRegex(ValueError, "avatar image"):
                pipeline.run_viewmax_pipeline("p", None, avatar_enabled=True, output_dir=self.d.name)
        gv.assert_not_called()

    def test_bad_image_fails_before_voiceover(self):
        bad = os.path.join(self.d.name, "bad.gif")
        open(bad, "wb").write(b"GIF89a")
        with mock.patch.object(pipeline, "generate_voiceover") as gv:
            with self.assertRaisesRegex(ValueError, r"\.jpg or \.png"):
                pipeline.run_viewmax_pipeline("p", None, avatar_image=bad, avatar_enabled=True,
                                              output_dir=self.d.name)
        gv.assert_not_called()

    def test_render_failure_falls_back_to_plain_video(self):
        with mock.patch.object(pipeline, "create_avatar_video", side_effect=RuntimeError("encoder died")):
            res, _ = self.go(avatar_image=self.img, avatar_enabled=True)
        self.assertIsNone(res["avatar"])
        self.assertTrue(any("encoder died" in w for w in res["warnings"]))
        self.assertTrue(os.path.getsize(res["video_path"]) > 1000)


if __name__ == "__main__":
    unittest.main()
