"""uploader.backup_files / zip_folder: Drive backup of every finished render (mocked upload, no network)."""
import os
import tempfile
import unittest
import zipfile

from backend import uploader


class BackupFilesTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.d, True)

    def make(self, name, data=b"x"):
        p = os.path.join(self.d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_uploads_each_file_once_in_order(self):
        sent = []
        a, b = self.make("a.mp4"), self.make("b.zip")
        out = uploader.backup_files([a, b, a], "FOLDER", upload_fn=lambda p, f: sent.append((os.path.basename(p), f)) or "link:" + os.path.basename(p))
        self.assertEqual(sent, [("a.mp4", "FOLDER"), ("b.zip", "FOLDER")])
        self.assertEqual(out["links"], {a: "link:a.mp4", b: "link:b.zip"})
        self.assertEqual(out["errors"], {})

    def test_one_failure_does_not_stop_the_rest_and_missing_files_are_reported(self):
        a, b = self.make("a.mp4"), self.make("b.mp4")

        def fn(p, f):
            if p == a:
                raise PermissionError("no access")
            return "ok"

        out = uploader.backup_files([a, b, os.path.join(self.d, "gone.mp4")], "F", upload_fn=fn)
        self.assertEqual(out["links"], {b: "ok"})
        self.assertIn("no access", out["errors"][a])
        self.assertEqual(out["errors"][os.path.join(self.d, "gone.mp4")], "file not found")
        self.assertTrue(os.path.isfile(a))  # the local copy is never touched

    def test_transient_errors_are_retried(self):
        a = self.make("a.mp4")
        calls = []

        def fn(p, f):
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("flaky")
            return "done"

        out = uploader.backup_files([a], "F", upload_fn=fn, base_delay=0.0)
        self.assertEqual(out["links"][a], "done")
        self.assertEqual(len(calls), 3)

    def test_nothing_to_do_and_bad_folder(self):
        self.assertEqual(uploader.backup_files([], "F"), {"links": {}, "errors": {}})
        with self.assertRaises(ValueError):
            uploader.backup_files([self.make("a.mp4")], "  ", upload_fn=lambda p, f: "x")

    def test_zip_folder_skips_scratch_and_keeps_structure(self):
        self.make("proj/script.fountain", b"INT.")
        self.make("proj/sub/graph.json", b"{}")
        self.make("proj/work/big.png", b"junk")
        z = uploader.zip_folder(os.path.join(self.d, "proj"), os.path.join(self.d, "out", "p.zip"))
        with zipfile.ZipFile(z) as zf:
            self.assertEqual(sorted(zf.namelist()), ["script.fountain", "sub/graph.json"])
        with self.assertRaises(FileNotFoundError):
            uploader.zip_folder(os.path.join(self.d, "nope"), os.path.join(self.d, "n.zip"))


if __name__ == "__main__":
    unittest.main()
