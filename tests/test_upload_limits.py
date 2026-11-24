"""500 MB / 1 GB upload cap (settings, launcher flag, config.toml) and the chunked, low-RAM disk write."""
import gc
import io
import json
import os
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import launcher  # noqa: E402
import upload_limits as ul  # noqa: E402
import uploader  # noqa: E402


class LimitSettingsTests(unittest.TestCase):
    def test_only_500_and_1024_are_allowed(self):
        self.assertEqual(ul.CHOICES, {"500 MB": 500, "1 GB (1024 MB)": 1024})
        for raw, want in ((500, 500), ("500", 500), ("500 MB", 500), (1024, 1024), ("1024mb", 1024), (2048, 1024),
                          (0, 1024), (None, 1024), ("abc", 1024), (-5, 1024)):
            self.assertEqual(ul.normalize(raw), want, raw)

    def test_default_is_one_gb(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(ul.read_limit_mb(Path(d) / "none.json", env={}), 1024)

    def test_saved_choice_round_trips_and_keeps_other_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "storage" / "settings.json"
            p.parent.mkdir()
            p.write_text(json.dumps({"other": 1}))
            self.assertEqual(ul.write_limit_mb("500 MB", p), 500)
            self.assertEqual(ul.read_limit_mb(p, env={}), 500)
            self.assertEqual(json.loads(p.read_text()), {"other": 1, "max_upload_mb": 500})
            self.assertEqual(ul.write_limit_mb(777, p), 1024)  # bad value -> safe default
            self.assertFalse(list(p.parent.glob("*.tmp")))

    def test_environment_beats_the_saved_setting_and_corrupt_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            ul.write_limit_mb(1024, p)
            self.assertEqual(ul.read_limit_mb(p, env={ul.ENV_VAR: "500"}), 500)
            p.write_text("{not json")
            self.assertEqual(ul.read_limit_mb(p, env={}), 1024)

    def test_size_check_and_message(self):
        self.assertFalse(ul.too_big(500 * ul.MB, 500))
        self.assertTrue(ul.too_big(500 * ul.MB + 1, 500))
        self.assertFalse(ul.too_big(1024 * ul.MB, 1024))
        self.assertTrue(ul.too_big(1025 * ul.MB, 1024))
        self.assertFalse(ul.too_big(None, 500))
        msg = ul.too_big_message("big.mp4", 700 * ul.MB, 500)
        self.assertIn("700 MB", msg)
        self.assertIn("500 MB", msg)


class LauncherAndConfigTests(unittest.TestCase):
    def test_streamlit_gets_the_max_upload_size_flag(self):
        for mb in (500, 1024):
            argv = launcher.build_streamlit_argv(Path("frontend/app.py"), 8501, mb)
            self.assertEqual(argv[argv.index("--server.maxUploadSize") + 1], str(mb))
        self.assertEqual(launcher.build_streamlit_argv(Path("a.py"), 1).count("--server.maxUploadSize"), 1)
        self.assertEqual(launcher.build_streamlit_argv(Path("a.py"), 1)[launcher.build_streamlit_argv(Path("a.py"), 1).index("--server.maxUploadSize") + 1], "1024")

    def test_upload_limit_command_line_option(self):
        self.assertEqual(launcher.parse_upload_limit(["--upload-limit", "500"], default=1024), 500)
        self.assertEqual(launcher.parse_upload_limit(["--upload-limit=1024"], default=500), 1024)
        self.assertEqual(launcher.parse_upload_limit(["--upload-limit", "9999"], default=500), 1024)
        self.assertEqual(launcher.parse_upload_limit(["--selftest"], default=500), 500)
        with mock.patch.dict(os.environ, {ul.ENV_VAR: "500"}):
            self.assertEqual(launcher.parse_upload_limit([]), 500)

    def test_config_toml_sets_one_gb(self):
        text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        import re
        self.assertRegex(text, r"(?m)^\[server\]\s*$")
        self.assertEqual(re.search(r"(?m)^maxUploadSize\s*=\s*(\d+)", text).group(1), "1024")
        try:
            import tomllib
            self.assertEqual(tomllib.loads(text)["server"]["maxUploadSize"], 1024)
        except ImportError:
            pass


class StreamedSaveTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, True)

    def test_writes_in_four_mb_chunks(self):
        self.assertEqual(uploader.UPLOAD_CHUNK, 4 * 1024 * 1024)
        sizes = []

        class Spy(io.BytesIO):
            def read(self, n=-1):
                sizes.append(n)
                return super().read(n)

        data = os.urandom(9 * 1024 * 1024 + 123)
        dest = os.path.join(self.d, "v.mp4")
        out = uploader.save_uploaded_video_streamed(Spy(data), dest)
        self.assertEqual(Path(out).read_bytes(), data)
        self.assertTrue(sizes and all(n == 4 * 1024 * 1024 for n in sizes))
        self.assertEqual(len(sizes), 4)  # 4 MB + 4 MB + 1 MB + the empty read that ends the loop

    def test_memory_is_released_and_gc_runs(self):
        f = io.BytesIO(b"x" * 100_000)
        with mock.patch("gc.collect") as collect:
            uploader.save_uploaded_video_streamed(f, os.path.join(self.d, "a.mp4"), chunk_size=10_000)
        self.assertTrue(f.closed)
        collect.assert_called()

    def test_peak_extra_memory_stays_near_one_chunk(self):
        data = os.urandom(64 * 1024 * 1024)
        f = io.BytesIO(data)
        gc.collect()
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            uploader.save_uploaded_video_streamed(f, os.path.join(self.d, "big.mp4"))
            peak = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 12 * 1024 * 1024, f"peak extra memory {peak / 1e6:.1f} MB for a 64 MB upload")
        self.assertEqual(os.path.getsize(os.path.join(self.d, "big.mp4")), len(data))

    def test_too_big_upload_leaves_nothing_behind(self):
        dest = os.path.join(self.d, "x.mp4")
        f = io.BytesIO(b"x" * 50_000)
        with self.assertRaises(ValueError):
            uploader.save_uploaded_video_streamed(f, dest, chunk_size=4096, max_bytes=10_000)
        self.assertEqual(os.listdir(self.d), [])
        self.assertTrue(f.closed)

    def test_read_error_removes_the_partial_file(self):
        class Broken(io.BytesIO):
            calls = 0

            def read(self, n=-1):
                Broken.calls += 1
                if Broken.calls > 1:
                    raise OSError("disk went away")
                return b"x" * 100

        with self.assertRaises(OSError):
            uploader.save_uploaded_video_streamed(Broken(), os.path.join(self.d, "y.mp4"))
        self.assertEqual(os.listdir(self.d), [])

    def test_existing_file_is_replaced_atomically_and_parents_created(self):
        dest = os.path.join(self.d, "new", "deep", "z.mp4")
        uploader.save_uploaded_video_streamed(io.BytesIO(b"one"), dest)
        uploader.save_uploaded_video_streamed(io.BytesIO(b"two!"), dest)
        self.assertEqual(Path(dest).read_bytes(), b"two!")
        self.assertEqual(os.listdir(os.path.dirname(dest)), ["z.mp4"])
        with self.assertRaises(ValueError):
            uploader.save_uploaded_video_streamed(io.BytesIO(b"x"), dest, chunk_size=0)

    def test_upload_names_cannot_escape_the_folder(self):
        for bad in ("../../etc/passwd", "..\\..\\evil.mp4", "/abs/path/v.mp4", "C:\\x\\v.mp4", "a/../../b.mp4", ""):
            n = uploader.safe_upload_name(bad)
            self.assertNotIn("/", n)
            self.assertNotIn("\\", n)
            self.assertFalse(n.startswith("."))
            self.assertTrue(n)
        self.assertEqual(uploader.safe_upload_name("my clip.mp4"), "my clip.mp4")


if __name__ == "__main__":
    unittest.main()
