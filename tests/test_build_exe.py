"""build_exe.py: arguments, in-process PyInstaller call, ffmpeg/model auto-placement,
dist finishing and zip. PyInstaller itself is stubbed (it can't run off Windows)."""
import io
import shutil
import subprocess
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import build  # noqa: E402
import build_exe  # noqa: E402
from tests.test_build import GOOD_TOKEN, make_project  # noqa: E402


def make_ffmpeg_zip(path: Path, exe_bytes=b"MZ" + b"\0" * 64, inner="ffmpeg-7.0-essentials_build/bin/ffmpeg.exe"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(inner, exe_bytes)
        zf.writestr("ffmpeg-7.0-essentials_build/bin/ffplay.exe", b"MZ")


class ArgsTests(unittest.TestCase):
    def args(self, root=ROOT):
        with mock.patch.object(build, "_has_module", return_value=True):
            return build_exe.pyinstaller_args(root)

    def test_requested_flags_and_collect_all(self):
        a = self.args()
        for flag in ("--onedir", "--noconsole"):
            self.assertIn(flag, a)
        collected = [a[i + 1] for i, x in enumerate(a) if x == "--collect-all"]
        for pkg in ("streamlit", "mediapipe", "faster_whisper", "cv2", "googleapiclient"):
            self.assertIn(pkg, collected)

    def test_add_data_for_frontend_models_and_bin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_project(root)
            a = self.args(root)
        data = [a[i + 1] for i, x in enumerate(a) if x == "--add-data"]
        targets = sorted(x.rsplit(os.pathsep, 1)[1] for x in data)
        self.assertEqual(targets, ["bin", "frontend", "models"])
        self.assertEqual(a[-1], str(root / "launcher.py"))  # script stays last

    def test_missing_folders_are_not_added(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.args(Path(d))
        data = [a[i + 1] for i, x in enumerate(a) if x == "--add-data"]
        self.assertEqual(len(data), 1)  # only frontend

    def test_googleapiclient_is_a_required_package_with_correct_pip_name(self):
        self.assertIn("googleapiclient", build.COLLECT_ALL)
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d), token=GOOD_TOKEN)
            with mock.patch.object(build, "_has_module", side_effect=lambda n: n != "googleapiclient"):
                r = build.preflight(Path(d), allow_non_windows=True)
        self.assertTrue(any("pip install google-api-python-client" in e for e in r.errors))


class InProcessRunTests(unittest.TestCase):
    def stub(self, behaviour):
        main = types.ModuleType("PyInstaller.__main__")
        main.run = behaviour
        pkg = types.ModuleType("PyInstaller")
        pkg.__main__ = main
        return {"PyInstaller": pkg, "PyInstaller.__main__": main}

    def test_calls_pyinstaller_main_run_with_args(self):
        seen = []
        with mock.patch.dict(sys.modules, self.stub(lambda args: seen.append(list(args)))):
            self.assertEqual(build_exe.run_pyinstaller_inprocess(["--onedir", "x.py"]), 0)
        self.assertEqual(seen, [["--onedir", "x.py"]])

    def test_exit_codes(self):
        def fail(args):
            raise SystemExit(5)

        def fail_text(args):
            raise SystemExit("boom")

        with mock.patch.dict(sys.modules, self.stub(fail)):
            self.assertEqual(build_exe.run_pyinstaller_inprocess([]), 5)
        with mock.patch.dict(sys.modules, self.stub(fail_text)):
            self.assertEqual(build_exe.run_pyinstaller_inprocess([]), 1)


class PlacementTests(unittest.TestCase):
    def test_extract_ffmpeg_from_zip(self):
        with tempfile.TemporaryDirectory() as d:
            z, dest = Path(d) / "f.zip", Path(d) / "bin" / "ffmpeg.exe"
            make_ffmpeg_zip(z)
            self.assertTrue(build_exe.extract_ffmpeg_from_zip(z, dest))
            self.assertEqual(dest.read_bytes()[:2], b"MZ")

    def test_non_windows_binary_in_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "f.zip"
            make_ffmpeg_zip(z, exe_bytes=b"\x7fELF")
            self.assertFalse(build_exe.extract_ffmpeg_from_zip(z, Path(d) / "bin" / "ffmpeg.exe"))

    def test_zip_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "f.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("readme.txt", "x")
            self.assertFalse(build_exe.extract_ffmpeg_from_zip(z, Path(d) / "ffmpeg.exe"))

    def test_ensure_ffmpeg_already_present_does_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            make_project(Path(d))
            with mock.patch.object(build_exe, "download") as dl:
                self.assertTrue(build_exe.ensure_ffmpeg(Path(d)))
            dl.assert_not_called()

    def test_ensure_ffmpeg_uses_local_zip_before_downloading(self):
        with tempfile.TemporaryDirectory() as d:
            make_ffmpeg_zip(Path(d) / "ffmpeg-release-essentials.zip")
            with mock.patch.object(build_exe, "download") as dl:
                self.assertTrue(build_exe.ensure_ffmpeg(Path(d)))
            dl.assert_not_called()
            self.assertTrue((Path(d) / "bin" / "ffmpeg.exe").is_file())

    def test_ensure_ffmpeg_downloads_and_falls_back_to_second_mirror(self):
        calls = []

        def fake_download(url, dest, timeout=120):
            calls.append(url)
            if url == build_exe.FFMPEG_URLS[0]:
                raise OSError("mirror down")
            Path(dest).parent.mkdir(parents=True, exist_ok=True)  # the real download() does this
            make_ffmpeg_zip(Path(dest))

        with tempfile.TemporaryDirectory() as d, mock.patch.object(build_exe, "download", fake_download), mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            self.assertTrue(build_exe.ensure_ffmpeg(Path(d)))
            self.assertTrue((Path(d) / "bin" / "ffmpeg.exe").is_file())
            self.assertFalse((Path(d) / "build" / "ffmpeg_download.zip").exists())
        self.assertEqual(calls, build_exe.FFMPEG_URLS)

    def test_ensure_ffmpeg_no_download_and_all_mirrors_failing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(build_exe.ensure_ffmpeg(Path(d), allow_download=False))
            with mock.patch.object(build_exe, "download", side_effect=OSError("offline")), mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                self.assertFalse(build_exe.ensure_ffmpeg(Path(d)))

    def test_ensure_blazeface(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_download(url, dest, timeout=120):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(b"x" * 20000)

            with mock.patch.object(build_exe, "download", fake_download):
                self.assertTrue(build_exe.ensure_blazeface(Path(d)))
            self.assertTrue((Path(d) / build.BLAZEFACE_REL).is_file())
            self.assertFalse(build_exe.ensure_blazeface(Path(d) / "other", allow_download=False))


class FinishTests(unittest.TestCase):
    def dist(self, d, internal=True):
        app = Path(d) / "dist" / "CinePulseAI"
        base = app / "_internal" if internal else app
        (base / "bin").mkdir(parents=True)
        (base / "bin" / "ffmpeg.exe").write_bytes(b"MZ")
        (base / "models" / "faster-whisper-base").mkdir(parents=True)
        (base / build.BLAZEFACE_REL).write_bytes(b"t")
        (base / build.WHISPER_REL / "model.bin").write_bytes(b"w")
        (app / build.exe_name()).write_text("exe")
        return app

    def test_finalize_places_token_and_readme_with_optional_google_instructions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            root.mkdir()
            make_project(root, token=GOOD_TOKEN)
            app = self.dist(d)
            copied = build_exe.finalize(app, root)
            self.assertEqual(copied, ["token.json"])
            self.assertEqual(json.loads((app / "token.json").read_text()), GOOD_TOKEN)
            text = (app / "README.txt").read_text()
            for phrase in ("optional", "credentials.json", "token.json", "cinepulseai.log"):
                self.assertIn(phrase, text)

    def test_finalize_without_token(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            root.mkdir()
            make_project(root, token=GOOD_TOKEN)
            app = self.dist(d)
            self.assertEqual(build_exe.finalize(app, root, include_token=False), [])
            self.assertFalse((app / "token.json").exists())

    def test_verify_finds_files_in_internal_or_beside_exe(self):
        for internal in (True, False):
            with tempfile.TemporaryDirectory() as d:
                app = self.dist(d, internal)
                self.assertEqual(build_exe.verify(app, include_token=False), [])

    def test_verify_reports_missing(self):
        with tempfile.TemporaryDirectory() as d:
            app = Path(d) / "CinePulseAI"
            app.mkdir()
            text = " ".join(build_exe.verify(app, include_token=True))
            for expected in ("was not produced", "ffmpeg.exe", "blaze_face", "model.bin", "token.json"):
                self.assertIn(expected, text)

    def test_make_zip_has_single_top_folder_with_exe(self):
        with tempfile.TemporaryDirectory() as d:
            app = self.dist(d)
            z = build_exe.make_zip(app, Path(d) / "dist" / "out.zip")
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
            self.assertTrue(all(n.startswith("CinePulseAI/") for n in names))
            self.assertIn(f"CinePulseAI/{build.exe_name()}", names)
            self.assertIn("CinePulseAI/_internal/bin/ffmpeg.exe", names)


class MainTests(unittest.TestCase):
    def test_dry_run_downloads_nothing_and_prints_run_call(self):
        with mock.patch.object(build_exe, "download") as dl, mock.patch.object(
            build, "preflight", return_value=build.Report()
        ), mock.patch.object(build, "_has_module", return_value=True), mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ) as out:
            code = build_exe.main(["--dry-run"])
        dl.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("PyInstaller.__main__.run", out.getvalue())
        self.assertIn("'--noconsole'", out.getvalue())

    def test_full_flow_with_stubbed_pyinstaller(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d)
            make_project(proj, token=GOOD_TOKEN)

            def fake_run(args):
                app = proj / "dist" / "CinePulseAI"
                (app / "_internal").mkdir(parents=True)
                for rel in (build.FFMPEG_REL, build.BLAZEFACE_REL, build.WHISPER_REL / "model.bin"):
                    (app / "_internal" / rel).parent.mkdir(parents=True, exist_ok=True)
                    (app / "_internal" / rel).write_bytes(b"x")
                (app / build.exe_name()).write_text("exe")
                ocr = app / "_internal" / "rapidocr_onnxruntime" / "models"
                ocr.mkdir(parents=True)
                for m in ("det.onnx", "rec.onnx", "cls.onnx"):
                    (ocr / m).write_bytes(b"x")
                (app / "_internal" / "frontend").mkdir()
                shutil.copy2(ROOT / "frontend" / "app.py", app / "_internal" / "frontend" / "app.py")
                return 0

            with mock.patch.object(build_exe, "ROOT", proj), mock.patch.object(build, "ROOT", proj), mock.patch.object(
                build, "_has_module", return_value=True
            ), mock.patch.object(build_exe, "run_pyinstaller_inprocess", side_effect=fake_run), mock.patch.object(
                build, "fetch_whisper_model"
            ), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                code = build_exe.main(["--allow-non-windows", "--no-download"])
            self.assertEqual(code, 0, out.getvalue())
            z = proj / "dist" / build_exe.ZIP_NAME
            self.assertTrue(z.is_file())
            with zipfile.ZipFile(z) as zf:
                self.assertIn("CinePulseAI/token.json", zf.namelist())
            self.assertNotIn("WARNING: ffmpeg", out.getvalue())
            self.assertIn("Zip validation passed", out.getvalue())
            self.assertEqual(build_exe.validate_zip(z), [])
            self.assertEqual(build_exe.ZIP_NAME, "CinePulseAI-Ultimate-Portable.zip")

    def test_errors_stop_before_pyinstaller(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(build_exe, "ROOT", Path(d)), mock.patch.object(build_exe, "run_pyinstaller_inprocess") as run, mock.patch(
                "sys.stdout", new_callable=io.StringIO
            ), mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(build_exe.main(["--no-download", "--allow-non-windows"]), 1)
            run.assert_not_called()

    def test_pyinstaller_failure_code_returned(self):
        with tempfile.TemporaryDirectory() as d:
            proj = Path(d)
            make_project(proj, token=GOOD_TOKEN)
            with mock.patch.object(build_exe, "ROOT", proj), mock.patch.object(build, "ROOT", proj), mock.patch.object(
                build, "_has_module", return_value=True
            ), mock.patch.object(build_exe, "run_pyinstaller_inprocess", return_value=9), mock.patch.object(
                build, "fetch_whisper_model"
            ), mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(build_exe.main(["--allow-non-windows", "--no-download"]), 9)


class WorkflowFileTests(unittest.TestCase):
    def test_workflow_is_valid_and_builds_with_build_exe(self):
        text = (ROOT / ".github" / "workflows" / "build-windows-exe.yml").read_text()
        for needle in ("windows-latest", "workflow_dispatch", "python build_exe.py", "_stcore/health",
                       "upload-artifact", "GOOGLE_TOKEN_JSON", "requirements-build.txt"):
            self.assertIn(needle, text)
        try:
            import yaml
        except ImportError:
            return
        self.assertIn("build", yaml.safe_load(text)["jobs"])

    def test_every_collect_all_package_is_in_requirements_build(self):
        reqs = (ROOT / "requirements-build.txt").read_text().lower()
        for pkg in ("streamlit", "mediapipe", "faster-whisper", "opencv-python", "google-api-python-client", "pyinstaller"):
            self.assertIn(pkg, reqs)


if __name__ == "__main__":
    unittest.main()


REAL_APP = (ROOT / "frontend" / "app.py").read_text(encoding="utf-8")


def make_zip_file(path, skip=(), app_text=None):
    top = build.APP_NAME
    members = {
        f"{top}/{build.exe_name()}": b"exe",
        f"{top}/_internal/{build.FFMPEG_REL.as_posix()}": b"x",
        f"{top}/_internal/{build.BLAZEFACE_REL.as_posix()}": b"x",
        f"{top}/_internal/{(build.WHISPER_REL / 'model.bin').as_posix()}": b"x",
        f"{top}/_internal/frontend/app.py": (app_text if app_text is not None else REAL_APP).encode(),
        f"{top}/README.txt": b"hi",
        f"{top}/_internal/rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx": b"x",
        f"{top}/_internal/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx": b"x",
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            if not any(k in name for k in skip):
                zf.writestr(name, data)
    return path


class ValidateZipTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.z = Path(self.d.name) / "x.zip"

    def test_good_zip_passes(self):
        self.assertEqual(build_exe.validate_zip(make_zip_file(self.z)), [])

    def test_each_missing_piece_is_reported(self):
        for skip, text in ((build.exe_name(), build.exe_name()), ("ffmpeg.exe", "ffmpeg"),
                           (".tflite", "tflite"), ("model.bin", "model.bin"), ("app.py", "app.py"),
                           ("README.txt", "README"), (".onnx", "OCR models")):
            problems = build_exe.validate_zip(make_zip_file(self.z, skip=(skip,)))
            self.assertTrue(any(text in p for p in problems), (skip, problems))

    def test_old_ui_in_the_zip_is_caught(self):
        problems = build_exe.validate_zip(make_zip_file(self.z, app_text="old UI Vugola AI Mode"))
        self.assertTrue(any("Enable AI Avatar Lip-Sync Commentary" in p for p in problems))

    def test_missing_and_corrupt_zip(self):
        self.assertTrue(build_exe.validate_zip(Path(self.d.name) / "none.zip"))
        self.z.write_bytes(b"not a zip")
        self.assertTrue(any("not a valid zip" in p for p in build_exe.validate_zip(self.z)))

    def test_cli_flag(self):
        make_zip_file(self.z)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(build_exe.main(["--validate-zip", str(self.z)]), 0)
        self.assertIn("passed", out.getvalue())
        make_zip_file(self.z, skip=("README",))
        with mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(build_exe.main(["--validate-zip", str(self.z)]), 1)


class GitPublishTests(unittest.TestCase):
    """Real git in a temp repo with a bare 'origin' - no network."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        base = Path(self.d.name)
        self.remote, self.work = base / "origin.git", base / "work"
        for args, cwd in ((["init", "-q", "--bare", "-b", "main", str(self.remote)], base),
                          (["init", "-q", "-b", "main", str(self.work)], base)):
            subprocess.run(["git", *args], cwd=cwd, check=True)
        for args in (["config", "user.email", "t@example.com"], ["config", "user.name", "T"],
                     ["remote", "add", "origin", str(self.remote)]):
            subprocess.run(["git", *args], cwd=self.work, check=True)
        (self.work / ".gitignore").write_text("token.json\ncredentials.json\ndist/\nbuild/\n")
        (self.work / "a.py").write_text("print(1)\n")

    def remote_log(self):
        return subprocess.run(["git", "log", "-1", "--format=%B", "main"], cwd=self.remote, capture_output=True,
                              text=True, check=True).stdout

    def test_add_commit_push_with_the_requested_message(self):
        r = build_exe.git_publish(self.work)
        self.assertTrue(r["committed"] and r["pushed"])
        self.assertIn("a.py", r["files"])
        log = self.remote_log()
        self.assertTrue(log.startswith("feat: integrate Remotion, Revideo, LTX-Video paradigms, advanced OCR, 3D avatar corner placement for human/anime models, and automated build"))
        self.assertIn("Claude-Session:", log)

    def test_second_run_with_nothing_new_does_not_fail_or_commit(self):
        build_exe.git_publish(self.work)
        r = build_exe.git_publish(self.work)
        self.assertFalse(r["committed"])
        self.assertTrue(r["pushed"])

    def test_ignored_secrets_are_left_out(self):
        (self.work / "token.json").write_text("{}")
        (self.work / "dist").mkdir()
        (self.work / "dist" / "x.zip").write_bytes(b"x")
        r = build_exe.git_publish(self.work)
        self.assertEqual(sorted(r["files"]), [".gitignore", "a.py"])

    def test_unignored_secret_is_refused_before_staging(self):
        (self.work / ".gitignore").write_text("")
        (self.work / "token.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "token.json"):
            build_exe.git_publish(self.work)
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=self.work, capture_output=True, text=True)
        self.assertEqual(staged.stdout.strip(), "")

    def test_not_a_repo_and_push_failure(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "not a git repository"):
                build_exe.git_publish(Path(d))
        subprocess.run(["git", "remote", "set-url", "origin", str(self.remote) + "-missing"], cwd=self.work, check=True)
        with self.assertRaisesRegex(RuntimeError, "git push .*failed"):
            build_exe.git_publish(self.work)

    def test_cli_git_only(self):
        with mock.patch.object(build_exe, "ROOT", self.work), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(build_exe.main(["--git-only"]), 0)
        self.assertIn("pushed to origin/main", out.getvalue())
        (self.work / "token.json").write_text("{}")
        (self.work / ".gitignore").write_text("")
        with mock.patch.object(build_exe, "ROOT", self.work), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(build_exe.main(["--git-only"]), 1)
        self.assertIn("GIT STEP FAILED", err.getvalue())


class PackagingContentsTests(unittest.TestCase):
    def test_ocr_and_paddle_flags(self):
        args = build.build_pyinstaller_args(ROOT)
        for mod in ("ocr_engine", "visual_cues", "rapidocr_onnxruntime"):
            self.assertIn(mod, args)
        self.assertNotIn("paddleocr", args)  # opt-in only
        pad = build.build_pyinstaller_args(ROOT, with_paddleocr=True)
        for pkg in ("paddleocr", "paddle", "paddlex"):
            self.assertIn(pkg, pad)

    def test_pyinstaller_args_bundle_everything_new(self):
        args = build_exe.pyinstaller_args(ROOT)
        for flag in ("--onedir", "--noconsole"):
            self.assertIn(flag, args)
        self.assertNotIn("--onefile", args)
        pairs = list(zip(args, args[1:]))
        self.assertIn(("--hidden-import", "avatar_sync"), pairs)
        self.assertIn(("--hidden-import", "selftest"), pairs)
        datas = [b for a, b in pairs if a == "--add-data"]
        for folder in ("frontend", "models", "bin"):
            self.assertTrue(any(d.endswith(f"{os.pathsep}{folder}") for d in datas), (folder, datas))
        self.assertTrue(args[-1].endswith("launcher.py"))

    def test_ultimate_flag_bundles_paddle_and_torch(self):
        import argparse
        src = (ROOT / "build_exe.py").read_text(encoding="utf-8")
        self.assertIn("--ultimate", src)
        self.assertIn("a.with_paddleocr or a.ultimate, a.with_torch or a.ultimate", src)
        wf = (ROOT / ".github" / "workflows" / "build-windows-exe.yml").read_text(encoding="utf-8")
        self.assertIn("bundle_paddleocr", wf)
        self.assertIn("bundle_torch", wf)
        self.assertIn("CinePulseAI-Ultimate-Portable.zip", wf)

    def test_3d_modules_are_bundled_and_torch_is_opt_in(self):
        for m in ("avatar_3d_engine", "glb_model", "soft_raster", "lipsync_en", "avatar_gestures", "demo_avatars", "render_engine"):
            self.assertIn(f"backend/{m}.py", build.REQUIRED_SOURCES)
            self.assertTrue((ROOT / "backend" / f"{m}.py").is_file())
            self.assertIn(m, build_exe.pyinstaller_args(ROOT))
        self.assertNotIn("torch", build_exe.pyinstaller_args(ROOT))
        on = build_exe.pyinstaller_args(ROOT, with_torch=True)
        self.assertIn("torch", on)
        args = build_exe.pyinstaller_args(ROOT)
        self.assertIn("--onedir", args)
        self.assertIn("--noconsole", args)

    def test_avatar_module_is_a_required_source_and_exists(self):
        self.assertIn("backend/avatar_sync.py", build.REQUIRED_SOURCES)
        self.assertTrue((ROOT / "backend" / "avatar_sync.py").is_file())

    def test_readme_mentions_background_backup_and_avatar(self):
        with tempfile.TemporaryDirectory() as d:
            app = Path(d)
            build_exe.finalize(app, ROOT, include_token=False)
            text = (app / "README.txt").read_text(encoding="utf-8")
        for needle in ("lip-sync", "background", "token.json", "credentials.json", "CinePulseAI.exe"):
            self.assertIn(needle, text)
