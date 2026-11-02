"""Tests for build.py and the packaging-related runtime behaviour (frozen paths,
no-console stdio, bundled Whisper model, Quit button). PyInstaller itself is not
run here - it cannot build a Windows exe off Windows - so the command line, the
preflight scan and the dist-folder assembly are what get verified.
"""
import io
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import build  # noqa: E402
import launcher  # noqa: E402


def make_project(root: Path, ffmpeg=True, blaze=True, whisper=True, token=None):
    for rel in build.REQUIRED_SOURCES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub")
    if ffmpeg:
        (root / "bin").mkdir(exist_ok=True)
        (root / build.FFMPEG_REL).write_bytes(b"MZ" + b"\0" * 50)
    if blaze:
        (root / "models").mkdir(exist_ok=True)
        (root / build.BLAZEFACE_REL).write_bytes(b"tflite")
    if whisper:
        (root / build.WHISPER_REL).mkdir(parents=True, exist_ok=True)
        (root / build.WHISPER_REL / "model.bin").write_bytes(b"w")
    if token is not None:
        (root / "token.json").write_text(json.dumps(token))


GOOD_TOKEN = {"token": "t", "refresh_token": "r", "client_id": "c"}


class PreflightTests(unittest.TestCase):
    def pf(self, root, **kw):
        kw.setdefault("allow_non_windows", True)
        with mock.patch.object(build, "_has_module", return_value=True):
            return build.preflight(root, **kw)

    def test_clean_project_passes_with_token_warning(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            r = self.pf(Path(d))
        self.assertTrue(r.ok, r.render())
        self.assertIn(build.TOKEN_WARNING, r.warnings)

    def test_everything_missing_is_reported_at_once(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.pf(Path(d))
        text = r.render()
        for expected in ("launcher.py", "frontend/app.py", "backend/pipeline.py", "ffmpeg.exe", "blaze_face"):
            self.assertIn(expected, text)
        self.assertFalse(r.ok)

    def test_missing_packages_have_fix_commands(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            with mock.patch.object(build, "_has_module", side_effect=lambda n: n not in ("cv2", "streamlit", "PyInstaller")):
                r = build.preflight(Path(d), allow_non_windows=True)
        text = "\n".join(r.errors)
        self.assertIn("pip install opencv-python", text)
        self.assertIn("pip install streamlit", text)
        self.assertIn("pip install pyinstaller", text)

    def test_missing_optional_package_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            with mock.patch.object(build, "_has_module", side_effect=lambda n: n != "yt_dlp"):
                r = build.preflight(Path(d), allow_non_windows=True)
        self.assertTrue(r.ok)
        self.assertTrue(any("yt_dlp" in w for w in r.warnings))

    def test_non_windows_ffmpeg_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            (Path(d) / build.FFMPEG_REL).write_bytes(b"\x7fELF....")
            r = self.pf(Path(d))
        self.assertTrue(any("not a Windows executable" in e for e in r.errors))

    def test_allow_missing_downgrades_ffmpeg_and_blaze(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), ffmpeg=False, blaze=False, token=GOOD_TOKEN)
            strict = self.pf(Path(d))
            lax = self.pf(Path(d), allow_missing=True)
        self.assertFalse(strict.ok)
        self.assertTrue(lax.ok)

    def test_token_cases(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d))
            self.assertTrue(self.pf(Path(d)).ok)  # missing token = warning
            self.assertFalse(self.pf(Path(d), require_token=True).ok)
            (Path(d) / "token.json").write_text(json.dumps({"token": "t"}))
            self.assertTrue(any("refresh_token" in e for e in self.pf(Path(d)).errors))
            (Path(d) / "token.json").write_text("{not json")
            self.assertTrue(any("valid JSON" in e for e in self.pf(Path(d)).errors))
            r = self.pf(Path(d), include_token=False)  # --no-token ignores a bad token
            self.assertTrue(r.ok)
            self.assertTrue(any("NOT be bundled" in w for w in r.warnings))

    def test_windows_requirement(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            with mock.patch.object(build, "_has_module", return_value=True), mock.patch.object(build.os, "name", "posix"):
                r = build.preflight(Path(d))
        self.assertTrue(any("not Windows" in e for e in r.errors))


class CommandTests(unittest.TestCase):
    def args(self):
        with mock.patch.object(build, "_has_module", return_value=True):
            return build.build_pyinstaller_args(ROOT)

    def test_required_flags(self):
        a = self.args()
        for flag in ("--onedir", "--noconsole", "--noconfirm", "--clean"):
            self.assertIn(flag, a)
        self.assertNotIn("--onefile", a)
        self.assertEqual(a[a.index("--name") + 1], "CinePulseAI")
        self.assertEqual(a[-1], str(ROOT / "launcher.py"))

    def test_explicit_collect_all(self):
        a = self.args()
        collected = [a[i + 1] for i, x in enumerate(a) if x == "--collect-all"]
        for pkg in ("streamlit", "mediapipe", "faster_whisper", "cv2"):
            self.assertIn(pkg, collected)
        self.assertIn("ctranslate2", collected)

    def test_uninstalled_extras_are_skipped(self):
        with mock.patch.object(build, "_has_module", side_effect=lambda n: n not in build.EXTRA_COLLECT_ALL):
            a = build.build_pyinstaller_args(ROOT)
        collected = [a[i + 1] for i, x in enumerate(a) if x == "--collect-all"]
        self.assertEqual(collected, build.COLLECT_ALL)

    def test_frontend_data_hidden_imports_metadata(self):
        a = self.args()
        data = a[a.index("--add-data") + 1]
        self.assertTrue(data.endswith(os.pathsep + "frontend"))
        self.assertIn(str(ROOT / "frontend"), data)
        hidden = [a[i + 1] for i, x in enumerate(a) if x == "--hidden-import"]
        for mod in build.BACKEND_MODULES:
            self.assertIn(mod, hidden)
        self.assertEqual(a[a.index("--copy-metadata") + 1], "streamlit")
        self.assertEqual(a[a.index("--paths") + 1], str(ROOT / "backend"))


class AssembleTests(unittest.TestCase):
    def build_dir(self, **proj):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        root, app = Path(d.name) / "proj", Path(d.name) / "dist" / "CinePulseAI"
        root.mkdir()
        make_project(root, **proj)
        return root, app

    def test_assemble_copies_everything_next_to_exe(self):
        root, app = self.build_dir(token=GOOD_TOKEN)
        copied = build.assemble(app, root, include_token=True)
        self.assertEqual(set(copied), {"bin/ffmpeg.exe", "models/", "token.json"})
        self.assertTrue((app / "bin" / "ffmpeg.exe").is_file())
        self.assertTrue((app / "models" / "faster-whisper-base" / "model.bin").is_file())
        self.assertEqual(json.loads((app / "token.json").read_text()), GOOD_TOKEN)
        self.assertIn("Quit CinePulse AI", (app / "README.txt").read_text())

    def test_no_token_flag_leaves_token_out(self):
        root, app = self.build_dir(token=GOOD_TOKEN)
        build.assemble(app, root, include_token=False)
        self.assertFalse((app / "token.json").exists())
        self.assertIn("run the sign-in once", (app / "README.txt").read_text())

    def test_verify_dist(self):
        root, app = self.build_dir(token=GOOD_TOKEN)
        build.assemble(app, root)
        self.assertIn("was not produced", " ".join(build.verify_dist(app)))
        (app / build.exe_name()).write_text("exe")
        self.assertEqual(build.verify_dist(app), [])
        (app / "token.json").unlink()
        self.assertIn("token.json", " ".join(build.verify_dist(app)))
        self.assertEqual(build.verify_dist(app, include_token=False), [])

    def test_file_names_match_what_the_app_searches(self):
        # video_engine / uploader / transcriber look in exactly these places
        import video_engine
        self.assertEqual(video_engine.MODEL_FILENAME, build.BLAZEFACE_REL.name)
        self.assertEqual(str(build.FFMPEG_REL.parent), "bin")
        self.assertEqual(build.WHISPER_REL, Path("models") / "faster-whisper-base")


class MainTests(unittest.TestCase):
    def test_dry_run_prints_command_without_building(self):
        with mock.patch.object(build, "run_pyinstaller") as run, mock.patch.object(
            build, "preflight", return_value=build.Report()
        ), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            code = build.main(["--dry-run"])
        run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("--onedir", out.getvalue())
        self.assertIn("--noconsole", out.getvalue())

    def test_errors_stop_the_build(self):
        bad = build.Report(errors=["boom"])
        with mock.patch.object(build, "preflight", return_value=bad), mock.patch.object(
            build, "run_pyinstaller"
        ) as run, mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(build.main([]), 1)
            self.assertEqual(build.main(["--dry-run"]), 1)
        run.assert_not_called()

    def test_full_flow_with_stubbed_pyinstaller(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d)
            make_project(proj, token=GOOD_TOKEN)

            def fake_pyinstaller(args):
                app = proj / "dist" / build.APP_NAME
                app.mkdir(parents=True)
                (app / build.exe_name()).write_text("exe")
                return 0

            with mock.patch.object(build, "ROOT", proj), mock.patch.object(
                build, "_has_module", return_value=True
            ), mock.patch.object(build, "run_pyinstaller", side_effect=fake_pyinstaller), mock.patch.object(
                build, "fetch_whisper_model"
            ) as fetch, mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = build.main(["--allow-non-windows"])
            app = proj / "dist" / build.APP_NAME
            self.assertEqual(code, 0, out.getvalue())
            fetch.assert_called_once()
            self.assertTrue((app / "bin" / "ffmpeg.exe").is_file())
            self.assertTrue((app / "token.json").is_file())
            self.assertIn("revoke", out.getvalue())

    def test_pyinstaller_failure_code_returned(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d)
            make_project(proj, token=GOOD_TOKEN)
            with mock.patch.object(build, "ROOT", proj), mock.patch.object(
                build, "_has_module", return_value=True
            ), mock.patch.object(build, "run_pyinstaller", return_value=7), mock.patch.object(
                build, "fetch_whisper_model"
            ), mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(build.main(["--allow-non-windows"]), 7)

    def test_whisper_prefetch_skipped_when_present_and_failure_is_soft(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d))
            self.assertTrue(build.fetch_whisper_model(Path(d)))
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, {"huggingface_hub": None}), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertFalse(build.fetch_whisper_model(Path(d)))


class FrozenRuntimeTests(unittest.TestCase):
    def test_launcher_redirects_none_stdio_to_log(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "executable", str(Path(d) / "CinePulseAI.exe")
        ), mock.patch.object(sys, "stdout", None), mock.patch.object(sys, "stderr", None):
            log = launcher.redirect_stdio_if_needed()
            self.assertEqual(log, Path(d) / "cinepulseai.log")
            print("hello from a no-console exe")
            sys.stderr.write("an error\n")
            sys.stdout.flush()
            text = log.read_text()
            sys.stdout.close()
        self.assertIn("hello from a no-console exe", text)
        self.assertIn("an error", text)

    def test_launcher_leaves_real_stdio_alone(self):
        self.assertIsNone(launcher.redirect_stdio_if_needed())

    def test_writable_dir_falls_back_when_read_only(self):
        with mock.patch.object(Path, "mkdir", side_effect=[PermissionError(), None]):
            got = launcher.writable_dir(Path("/proc/nope"))
        self.assertIn("CinePulseAI", str(got))

    def test_uploader_looks_next_to_exe_when_frozen(self):
        import importlib
        import uploader
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "executable", str(Path(d) / "CinePulseAI.exe")
        ), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_TOKEN_FILE", None)
            frozen = importlib.reload(uploader)
            self.assertEqual(frozen.TOKEN_FILE, Path(d).resolve() / "token.json")
        os.environ.pop("GOOGLE_TOKEN_FILE", None)
        dev = importlib.reload(uploader)
        self.assertEqual(dev.TOKEN_FILE, ROOT / "token.json")  # project root, same place build.py reads

    def test_transcriber_prefers_bundled_model(self):
        import transcriber
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "models" / "faster-whisper-base"
            model.mkdir(parents=True)
            (model / "model.bin").write_bytes(b"w")
            with mock.patch.object(sys, "_MEIPASS", d, create=True):
                self.assertEqual(transcriber._bundled_model_path(), str(model))

            captured = {}

            class FakeModel:
                def __init__(self, name, **kw):
                    captured["name"] = name

            fake = types.ModuleType("faster_whisper")
            fake.WhisperModel = FakeModel
            transcriber._model = None
            with mock.patch.dict(sys.modules, {"faster_whisper": fake}), mock.patch.object(sys, "_MEIPASS", d, create=True):
                transcriber._get_model()
            transcriber._model = None
            self.assertEqual(captured["name"], str(model))

    def test_quit_button_only_when_frozen(self):
        from tests.test_app_logic import FakeSt, run_app
        plain = run_app(FakeSt())
        self.assertNotIn("Quit CinePulse AI", [c[1][0] for c in plain.names("button")])
        with mock.patch.object(sys, "frozen", True, create=True):
            frozen = run_app(FakeSt())
        self.assertIn("Quit CinePulse AI", [c[1][0] for c in frozen.names("button")])
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch("os._exit") as ex:
            run_app(FakeSt(clicked=["quit_app"]))
        ex.assert_called_once_with(0)

    def test_frozen_output_dir_is_next_to_exe(self):
        from tests.test_app_logic import FakeSt
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "executable", str(Path(d) / "CinePulseAI.exe")
        ):
            with mock.patch.dict(sys.modules, {"streamlit": FakeSt()}):
                ns = runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="not_main")
            self.assertEqual(ns["OUTPUT_DIR"], Path(d).resolve() / "outputs")
            self.assertTrue((Path(d) / "outputs").is_dir())


if __name__ == "__main__":
    unittest.main()
