"""Pipeline tests with real OpenCV + ffmpeg. Only the things that need the
network or heavy models are replaced: transcription, edge-tts, Drive upload."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pipeline  # noqa: E402
import video_engine as ve  # noqa: E402
from tests.test_scorer import make_words  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, probe, square_detector  # noqa: E402


def three_sentence_words():
    # ~1s per sentence -> ~3s, matching the 3s synthetic source video
    return make_words(
        [("Stop this is crazy!", 0), ("Did you know this secret?", 0), ("It cost 5 million dollars.", 0)],
        wps=0.2, gap=0.05,
    )


class FakeVoice:
    seconds = 2
    calls = []

    @classmethod
    async def generate(cls, text, output_path, voice="en-US-AriaNeural", **kw):
        cls.calls.append({"text": text, "voice": voice})
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             f"sine=frequency=330:duration={cls.seconds}", "-f", "wav", output_path],
            check=True,
        )
        return output_path


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class VugolaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_pipeline(self, name, source=None, folder="F", **kw):
        out = os.path.join(self.dir.name, name)
        ready = mock.patch.object(pipeline, "drive_status", return_value="ready")
        ready.start()
        self.addCleanup(ready.stop)
        with mock.patch.object(pipeline, "transcribe_audio", return_value=kw.pop("words", three_sentence_words())):
            return pipeline.run_vugola_pipeline(
                source or self.src, folder, top_n=kw.pop("top_n", 2), min_len=1, max_len=2.5,
                output_dir=out, ffmpeg_path=FFMPEG, detector=square_detector, **kw,
            )

    def test_happy_path_uploads_every_clip(self):
        seen = []
        with mock.patch.object(ve, "upload_to_google_drive", return_value="https://drive/x") as up:
            res = self.run_pipeline("v1", progress=lambda s, f: seen.append((s, f)))
        self.assertGreaterEqual(len(res["clips"]), 1)
        self.assertEqual(up.call_count, len(res["clips"]))
        for clip in res["clips"]:
            self.assertTrue(os.path.isfile(clip["output_path"]))
            self.assertEqual(clip["drive_link"], "https://drive/x")
            self.assertIsNone(clip["error"])
            info = probe(clip["output_path"])
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual((v["width"], v["height"]), (1080, 1920))
        self.assertEqual(up.call_args[0][1], "F")
        fractions = [f for _, f in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(seen[-1], ("Done", 1.0))
        self.assertEqual(res["warnings"], [])

    def test_url_is_downloaded_first(self):
        with mock.patch.object(pipeline, "download_media", return_value=self.src) as dl, mock.patch.object(
            ve, "upload_to_google_drive", return_value="L"
        ):
            self.run_pipeline("v2", source="https://youtu.be/abc")
        self.assertEqual(dl.call_args[0][0], "https://youtu.be/abc")

    def test_no_speech_raises(self):
        with self.assertRaises(RuntimeError):
            self.run_pipeline("v3", words=[])

    def test_no_folder_skips_upload_with_warning(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_DRIVE_FOLDER_ID"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            ve, "upload_to_google_drive"
        ) as up:
            res = self.run_pipeline("v4", folder="")
        up.assert_not_called()
        self.assertIn(pipeline.NO_FOLDER_WARNING, res["warnings"])
        self.assertTrue(all(c["drive_link"] is None and c["output_path"] for c in res["clips"]))

    def test_upload_failure_is_recorded_not_raised(self):
        with mock.patch.object(ve, "upload_to_google_drive", side_effect=RuntimeError("drive down")):
            res = self.run_pipeline("v5")
        for clip in res["clips"]:
            self.assertIn("drive down", clip["error"])
            self.assertTrue(os.path.isfile(clip["output_path"]))  # clip still saved


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class ViewmaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.bg = os.path.join(cls.dir.name, "bg.mp4")
        make_source(cls.bg, with_audio=False)  # 3 s background

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def setUp(self):
        ready = mock.patch.object(pipeline, "drive_status", return_value="ready")
        ready.start()
        self.addCleanup(ready.stop)
        FakeVoice.calls = []
        FakeVoice.seconds = 2

    def go(self, name, prompt="Make a motivational 20 second TikTok script about running", **kw):
        out = os.path.join(self.dir.name, name)
        kw.setdefault("folder_id", "FOLD")
        with mock.patch.object(pipeline, "generate_voiceover", FakeVoice.generate), mock.patch.object(
            pipeline, "upload_to_google_drive", return_value="https://drive/v"
        ) as up:
            res = pipeline.run_viewmax_pipeline(prompt, output_dir=out, ffmpeg_path=FFMPEG, **kw)
        return res, up

    def test_plain_background(self):
        res, up = self.go("p1")
        info = probe(res["video_path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (1080, 1920))
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertAlmostEqual(float(info["format"]["duration"]), 2.0, delta=0.2)
        up.assert_called_once_with(res["video_path"], "FOLD")
        self.assertEqual(res["drive_link"], "https://drive/v")
        self.assertIn("running", res["script"]["script"])
        self.assertEqual(FakeVoice.calls[0]["text"], res["script"]["script"])

    def test_background_video_loops_to_voiceover_length(self):
        FakeVoice.seconds = 5  # longer than the 3 s background
        res, _ = self.go("p2", background_video=self.bg)
        info = probe(res["video_path"])
        self.assertAlmostEqual(float(info["format"]["duration"]), 5.0, delta=0.3)
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))

    def test_voice_is_passed_through(self):
        self.go("p3", voice="en-GB-RyanNeural")
        self.assertEqual(FakeVoice.calls[0]["voice"], "en-GB-RyanNeural")

    def test_no_folder_skips_upload(self):
        res, up = self.go("p4", folder_id="")
        up.assert_not_called()
        self.assertIsNone(res["drive_link"])
        self.assertIn(pipeline.NO_FOLDER_WARNING, res["warnings"])

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            pipeline.run_viewmax_pipeline("   ", "F")


class AsyncHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_async_inside_running_loop(self):
        async def coro():
            return 42
        self.assertEqual(pipeline._run_async(coro()), 42)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class MergeTests(unittest.TestCase):
    def test_command_variants(self):
        with_bg = ve.build_merge_command("ff", "a.wav", "o.mp4", background_video="bg.mp4")
        self.assertIn("-stream_loop", with_bg)
        self.assertIn("bg.mp4", with_bg)
        plain = ve.build_merge_command("ff", "a.wav", "o.mp4")
        self.assertIn("lavfi", plain)
        self.assertNotIn("-stream_loop", plain)
        for cmd in (with_bg, plain):
            self.assertIn("-shortest", cmd)
            self.assertEqual(cmd[-1], "o.mp4")

    def test_missing_inputs(self):
        with self.assertRaises(FileNotFoundError):
            ve.merge_audio_with_video("/no/a.wav", "/tmp/o.mp4", ffmpeg_path=FFMPEG)
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.wav")
            Path(a).write_bytes(b"x")
            with self.assertRaises(FileNotFoundError):
                ve.merge_audio_with_video(a, os.path.join(d, "o.mp4"), "/no/bg.mp4", ffmpeg_path=FFMPEG)

    def test_ffmpeg_failure_wrapped(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.wav")
            Path(a).write_bytes(b"x")
            fake = os.path.join(d, "bad.sh")
            Path(fake).write_text("#!/bin/sh\necho kaput >&2\nexit 1\n")
            os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
            with self.assertRaises(RuntimeError) as cm:
                ve.merge_audio_with_video(a, os.path.join(d, "o.mp4"), ffmpeg_path=fake)
            self.assertIn("kaput", str(cm.exception))
            self.assertFalse(os.path.exists(os.path.join(d, "o.mp4")))


if __name__ == "__main__":
    unittest.main()
