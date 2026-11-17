"""Automatic Drive backup: retries, the background worker, and 'Drive trouble never loses output'."""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import pipeline  # noqa: E402
import uploader  # noqa: E402
from tests.test_scorer import make_words  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, probe, square_detector  # noqa: E402


class HttpErr(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.resp = type("R", (), {"status": status})()


class ServerNotFoundError(Exception):  # same name httplib2 uses when offline
    pass


class RetryTests(unittest.TestCase):
    def test_transient_classification(self):
        for exc in (HttpErr(503), HttpErr(429), ConnectionResetError(), TimeoutError(), ServerNotFoundError()):
            self.assertTrue(uploader.is_transient_error(exc), exc)
        for exc in (HttpErr(403), HttpErr(404), FileNotFoundError("no creds"), ValueError("bad id"),
                    PermissionError(), RuntimeError("boom")):
            self.assertFalse(uploader.is_transient_error(exc), exc)

    def test_retries_transient_then_succeeds_with_backoff(self):
        fn = mock.Mock(side_effect=[HttpErr(503), ConnectionResetError(), "https://drive/ok"])
        sleeps = []
        link = uploader.upload_with_retry("f.mp4", "F", attempts=3, base_delay=2, sleep=sleeps.append, upload_fn=fn)
        self.assertEqual(link, "https://drive/ok")
        self.assertEqual(fn.call_count, 3)
        self.assertEqual(sleeps, [2, 4])

    def test_gives_up_after_attempts(self):
        fn = mock.Mock(side_effect=HttpErr(500))
        with self.assertRaises(HttpErr):
            uploader.upload_with_retry("f", "F", attempts=3, sleep=lambda s: None, upload_fn=fn)
        self.assertEqual(fn.call_count, 3)

    def test_permanent_errors_are_not_retried(self):
        fn = mock.Mock(side_effect=FileNotFoundError("No valid token.json"))
        with self.assertRaises(FileNotFoundError):
            uploader.upload_with_retry("f", "F", sleep=lambda s: self.fail("slept"), upload_fn=fn)
        self.assertEqual(fn.call_count, 1)

    def test_default_function_is_looked_up_at_call_time(self):
        with mock.patch.object(uploader, "upload_to_google_drive", return_value="L") as up:
            self.assertEqual(uploader.upload_with_retry("a", "B"), "L")
        up.assert_called_once_with("a", "B")


class DriveBackupTests(unittest.TestCase):
    def test_runs_off_the_calling_thread_in_order(self):
        seen = []

        def fn(path, folder):
            seen.append((path, folder, threading.current_thread().name))
            return f"link:{path}"

        with uploader.DriveBackup(" FOLD ", upload_fn=fn) as b:
            futures = [b.submit(p) for p in ("a.mp4", "b.mp4", "c.mp4")]
            links = [f.result() for f in futures]
        self.assertEqual(links, ["link:a.mp4", "link:b.mp4", "link:c.mp4"])
        self.assertEqual([s[0] for s in seen], ["a.mp4", "b.mp4", "c.mp4"])
        self.assertTrue(all(s[1] == "FOLD" for s in seen))
        self.assertTrue(all(s[2].startswith("drive-backup") for s in seen))
        self.assertNotEqual(seen[0][2], threading.current_thread().name)

    def test_submit_returns_before_the_upload_finishes(self):
        gate = threading.Event()
        with uploader.DriveBackup("F", upload_fn=lambda p, f: (gate.wait(5), "L")[1]) as b:
            t0 = time.time()
            fut = b.submit("x")
            self.assertLess(time.time() - t0, 0.5)
            self.assertFalse(fut.done())
            gate.set()
            self.assertEqual(fut.result(), "L")

    def test_failure_surfaces_on_the_future_only(self):
        with uploader.DriveBackup("F", upload_fn=mock.Mock(side_effect=RuntimeError("down")), attempts=1) as b:
            fut = b.submit("x")
            with self.assertRaisesRegex(RuntimeError, "down"):
                fut.result()

    def test_empty_folder_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                uploader.DriveBackup(bad)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PipelineBackupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def clips(self, name, up):
        words = make_words([("Stop this is crazy!", 0), ("Did you know this secret?", 0),
                            ("It cost 5 million dollars.", 0)], wps=0.2, gap=0.05)
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), mock.patch.object(
            pipeline, "transcribe_audio", return_value=words
        ), mock.patch.object(pipeline, "upload_to_google_drive", up):
            return pipeline.run_vugola_pipeline(
                self.src, "FOLD", top_n=3, min_len=1, max_len=2.5,
                output_dir=os.path.join(self.dir.name, name), ffmpeg_path=FFMPEG, detector=square_detector)

    def test_every_clip_is_backed_up_on_the_background_thread(self):
        threads = []

        def up(path, folder):
            threads.append(threading.current_thread().name)
            return "https://drive/" + os.path.basename(path)

        res = self.clips("bg1", up)
        self.assertGreaterEqual(len(res["clips"]), 2)
        self.assertEqual(len(threads), len(res["clips"]))
        self.assertTrue(all(t.startswith("drive-backup") for t in threads))
        for c in res["clips"]:
            self.assertTrue(c["drive_link"].endswith(os.path.basename(c["output_path"])))
            self.assertIsNone(c["error"])

    def test_upload_overlaps_with_cropping_the_next_clip(self):
        started = []

        def slow(path, folder):
            started.append(time.time())
            time.sleep(0.4)
            return "L"

        t0 = time.time()
        res = self.clips("bg2", slow)
        # first upload must start before the whole pipeline is over (it ran while later clips were cropped)
        self.assertLess(started[0] - t0, time.time() - t0 - 0.3)
        self.assertTrue(all(c["drive_link"] == "L" for c in res["clips"]))

    def test_one_failed_upload_does_not_affect_other_clips(self):
        calls = []

        def flaky(path, folder):
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("quota exceeded")
            return "L"

        res = self.clips("bg3", flaky)
        bad = [c for c in res["clips"] if c["error"]]
        self.assertEqual(len(bad), 1)
        self.assertIn("quota exceeded", bad[0]["error"])
        self.assertTrue(os.path.isfile(bad[0]["output_path"]))  # still saved locally
        self.assertTrue(all(c["drive_link"] == "L" for c in res["clips"] if not c["error"]))


class BackupFilesTests(unittest.TestCase):
    def test_finished_clips_are_collected_for_backup(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.mp4")
            Path(a).write_bytes(b"x")
            res = {"clips": [{"output_path": a}, {"output_path": None}, {"output_path": a}]}
            self.assertEqual(uploader.collect_final_videos(res), [a])


if __name__ == "__main__":
    unittest.main()
