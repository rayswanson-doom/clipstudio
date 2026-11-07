import io
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import launcher  # noqa: E402
import selftest  # noqa: E402

HAVE_VIDEO = bool(shutil.which("ffmpeg")) and (ROOT / "models" / "blaze_face_short_range.tflite").is_file()
try:
    import cv2  # noqa: F401
    import mediapipe  # noqa: F401
except ImportError:
    HAVE_VIDEO = False


class RealCheckTests(unittest.TestCase):
    """These run the genuine checks that the packaged exe runs, where the sandbox can."""

    def test_scorer(self):
        self.assertIn("clip", selftest.check_scorer())

    @unittest.skipUnless(HAVE_VIDEO, "needs ffmpeg + opencv + mediapipe + model")
    def test_ffmpeg_face_model_and_clip_pipeline(self):
        self.assertIn("ffmpeg", selftest.check_ffmpeg())
        self.assertIn("BlazeFace", selftest.check_face_model())
        self.assertIn("9:16", selftest.check_clip_pipeline())

    @unittest.skipUnless(HAVE_VIDEO, "needs ffmpeg + opencv + mediapipe + model")
    def test_avatar_check_runs_for_real(self):
        self.assertIn("frames", selftest.check_avatar())

    def test_app_page_missing_file(self):
        with self.assertRaises(RuntimeError):
            selftest.check_app_page(Path("/no/app.py"))

    def test_imports_check_reports_every_failure(self):
        with mock.patch.object(selftest, "MODULES", ["json", "no_such_module_a", "no_such_module_b"]):
            with self.assertRaises(RuntimeError) as cm:
                selftest.check_imports()
        self.assertIn("no_such_module_a", str(cm.exception))
        self.assertIn("no_such_module_b", str(cm.exception))
        self.assertNotIn("json (", str(cm.exception))

    def test_whisper_check_fails_loudly_when_frozen_without_bundled_model(self):
        import transcriber
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            transcriber, "_bundled_model_path", return_value=None
        ):
            with self.assertRaises(RuntimeError) as cm:
                selftest.check_whisper()
        self.assertIn("bundled Whisper model", str(cm.exception))


class RunnerTests(unittest.TestCase):
    def patched(self, **overrides):
        names = ["check_imports", "check_ffmpeg", "check_scorer", "check_face_model",
                 "check_whisper", "check_clip_pipeline", "check_avatar", "check_avatar_3d", "check_render_engine", "check_story", "check_ocr", "check_cues", "check_tts_online"]
        stack = [mock.patch.object(selftest, n, return_value="fine") for n in names]
        stack.append(mock.patch.object(selftest, "check_app_page", return_value="fine"))
        return stack

    def run_with(self, failing=None, kind=RuntimeError):
        patches = self.patched()
        mocks = [p.start() for p in patches]
        self.addCleanup(lambda: [p.stop() for p in patches])
        if failing:
            getattr(selftest, failing).side_effect = kind("broken")
        return selftest.run_all(Path("app.py"), log=lambda m: None)

    def test_all_ok(self):
        results = self.run_with()
        self.assertEqual({r[1] for r in results}, {"ok"})
        self.assertEqual(len(results), 14)

    def test_hard_failure_is_reported(self):
        results = dict((r[0], r[1]) for r in self.run_with("check_face_model"))
        self.assertEqual(results["face model"], "fail")

    def test_online_tts_failure_is_only_a_warning(self):
        results = dict((r[0], r[1]) for r in self.run_with("check_tts_online"))
        self.assertEqual(results["edge-tts (online)"], "warn")

    def test_main_exit_codes(self):
        for failing, code in ((None, 0), ("check_whisper", 1), ("check_tts_online", 0)):
            patches = self.patched()
            for p in patches:
                p.start()
            try:
                if failing:
                    getattr(selftest, failing).side_effect = RuntimeError("x")
                with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                    self.assertEqual(selftest.main(Path("app.py")), code)
                self.assertIn("SELFTEST FAILED" if code else "SELFTEST OK", out.getvalue())
            finally:
                for p in patches:
                    p.stop()


class LauncherDispatchTests(unittest.TestCase):
    def test_selftest_flag_runs_selftest_and_not_streamlit(self):
        with mock.patch.object(sys, "argv", ["CinePulseAI.exe", "--selftest"]), mock.patch.object(
            selftest, "main", return_value=0
        ) as m, mock.patch.object(launcher, "open_browser_when_ready") as ob:
            self.assertEqual(launcher.main(), 0)
        m.assert_called_once()
        self.assertTrue(str(m.call_args[0][0]).endswith("app.py"))
        ob.assert_not_called()

    def test_selftest_failure_code_is_returned(self):
        with mock.patch.object(sys, "argv", ["CinePulseAI.exe", "--selftest"]), mock.patch.object(
            selftest, "main", return_value=1
        ):
            self.assertEqual(launcher.main(), 1)

    def test_selftest_is_bundled(self):
        import build
        self.assertIn("selftest", build.HIDDEN_IMPORTS)
        self.assertIn("selftest.py", build.REQUIRED_SOURCES)


if __name__ == "__main__":
    unittest.main()
