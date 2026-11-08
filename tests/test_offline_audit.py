"""Audit: the app must start and work on a plain laptop with NO API keys, NO
Google files, and optional packages missing. Each test proves one fallback."""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import pipeline  # noqa: E402
import scorer  # noqa: E402
import transcriber  # noqa: E402
import uploader  # noqa: E402
from tests.test_scorer import make_words  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, square_detector  # noqa: E402


class ImportTests(unittest.TestCase):
    def test_all_backend_modules_import_with_optional_packages_blocked(self):
        """Run in a clean interpreter where every cloud/heavy package is unimportable."""
        code = (
            "import sys\n"
            "for m in ('yt_dlp','googleapiclient','google','google_auth_oauthlib','faster_whisper',"
            "'edge_tts','streamlit','rapidocr_onnxruntime','onnxruntime'):\n"
            "    sys.modules[m] = None\n"
            f"sys.path.insert(0, {str(ROOT / 'backend')!r})\n"
            "import scorer, uploader, transcriber, tts_engine, video_engine, avatar_sync, ocr_engine, visual_cues, pipeline\n"
            "import avatar_3d_engine, glb_model, soft_raster, lipsync_en, avatar_gestures, demo_avatars, render_engine\n"
            "import vimax_orchestrator, novel_adapter, script_parser, dependency_graph, meme_matcher\n"
            "print('imports-ok')\n"
        )
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-600:])
        self.assertIn("imports-ok", out.stdout)

    def test_scorer_needs_no_network_or_keys(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network used")), mock.patch.dict(
            os.environ, {k: "" for k in os.environ if "KEY" in k or "TOKEN" in k}
        ):
            words = make_words([("Did you know this secret?", 0), ("It is insane!", 0)])
            self.assertTrue(scorer.find_viral_clips(words, min_len=1, max_len=30))
            self.assertIn("script", scorer.generate_script("script about tea"))


class DriveFallbackTests(unittest.TestCase):
    def test_status_levels(self):
        with tempfile.TemporaryDirectory() as d:
            tok, cred = Path(d) / "token.json", Path(d) / "credentials.json"
            with mock.patch.object(uploader, "TOKEN_FILE", tok), mock.patch.object(uploader, "CREDENTIALS_FILE", cred):
                self.assertEqual(uploader.drive_status(), "unavailable")
                cred.write_text("{}")
                self.assertEqual(uploader.drive_status(), "needs_signin")
                tok.write_text("{}")
                self.assertEqual(uploader.drive_status(), "ready")

    def test_local_media_works_without_yt_dlp(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f, mock.patch.dict(sys.modules, {"yt_dlp": None}):
            self.assertEqual(uploader.download_media(f.name), str(Path(f.name).resolve()))

    def test_upload_without_any_auth_files_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d, tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
            with mock.patch.object(uploader, "TOKEN_FILE", Path(d) / "t.json"), mock.patch.object(
                uploader, "CREDENTIALS_FILE", Path(d) / "c.json"
            ):
                try:
                    uploader.upload_to_google_drive(clip.name, "FOLDER")
                except FileNotFoundError as exc:
                    self.assertIn("credentials.json", str(exc))
                except ModuleNotFoundError:
                    pass  # google libs not installed in this sandbox: still a clean, catchable error


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PipelineWithoutGoogleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_vugola_without_google_files_saves_clips_and_warns_once(self):
        words = make_words([("Stop this is crazy!", 0), ("Did you know this secret?", 0)], wps=0.2, gap=0.05)
        with mock.patch.object(pipeline, "drive_status", return_value="unavailable"), mock.patch.object(
            pipeline, "transcribe_audio", return_value=words
        ), mock.patch("video_engine.upload_to_google_drive") as up:
            res = pipeline.run_vugola_pipeline(
                self.src, "SOME_FOLDER", top_n=1, min_len=1, max_len=2.5,
                output_dir=os.path.join(self.dir.name, "o1"), ffmpeg_path=FFMPEG, detector=square_detector,
            )
        up.assert_not_called()
        self.assertEqual(res["warnings"], [pipeline.NO_AUTH_WARNING])
        self.assertTrue(res["clips"][0]["output_path"] and not res["clips"][0]["error"])

    def test_no_face_in_video_warns_and_still_produces_clip(self):
        words = make_words([("Stop this is crazy!", 0), ("Did you know this secret?", 0)], wps=0.2, gap=0.05)
        with mock.patch.object(pipeline, "transcribe_audio", return_value=words):
            res = pipeline.run_vugola_pipeline(
                self.src, "", top_n=1, min_len=1, max_len=2.5,
                output_dir=os.path.join(self.dir.name, "o2"), ffmpeg_path=FFMPEG, detector=lambda f: None,
            )
        self.assertIn(pipeline.NO_FACE_WARNING, res["warnings"])
        self.assertTrue(os.path.isfile(res["clips"][0]["output_path"]))

    def test_broken_face_model_degrades_to_centre_crop(self):
        words = make_words([("Stop this is crazy!", 0), ("Did you know this secret?", 0)], wps=0.2, gap=0.05)
        with mock.patch.object(pipeline, "transcribe_audio", return_value=words), mock.patch.dict(
            os.environ, {"BLAZEFACE_MODEL_PATH": "/no/such/model.tflite"}
        ):
            res = pipeline.run_vugola_pipeline(
                self.src, "", top_n=1, min_len=1, max_len=2.5,
                output_dir=os.path.join(self.dir.name, "o3"), ffmpeg_path=FFMPEG, allow_center_fallback=True,
            )
        self.assertTrue(os.path.isfile(res["clips"][0]["output_path"]))
        self.assertIn(pipeline.NO_FACE_WARNING, res["warnings"])


class TranscriberFallbackTests(unittest.TestCase):
    def test_model_load_failure_is_a_readable_error(self):
        class Boom:
            def __init__(self, *a, **k):
                raise OSError("offline and no local model")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = Boom
        transcriber._model = None
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, mock.patch.dict(sys.modules, {"faster_whisper": fake}):
            with self.assertRaises(RuntimeError) as cm:
                transcriber.transcribe_audio(f.name)
        transcriber._model = None
        self.assertIn("speech-recognition model", str(cm.exception))
        self.assertIn("internet", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
