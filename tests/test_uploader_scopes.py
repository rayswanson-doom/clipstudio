import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import uploader  # noqa: E402


class ScopeTests(unittest.TestCase):
    def test_default_is_least_privilege(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_DRIVE_FULL_ACCESS", None)
            self.assertEqual(uploader._default_scopes(), [uploader.FILE_SCOPE])

    def test_env_switch(self):
        with mock.patch.dict(os.environ, {"GOOGLE_DRIVE_FULL_ACCESS": "1"}):
            self.assertEqual(uploader._default_scopes(), [uploader.FULL_SCOPE])

    def test_full_access_flag_resets_old_token_and_widens_scope(self):
        with tempfile.TemporaryDirectory() as d:
            token = Path(d) / "token.json"
            token.write_text("{}")
            seen = {}

            def fake_get_credentials():
                seen["scopes"] = list(uploader.SCOPES)
                seen["token_existed"] = token.exists()

            old = list(uploader.SCOPES)
            try:
                with mock.patch.object(uploader, "TOKEN_FILE", token), mock.patch.object(
                    uploader, "get_credentials", fake_get_credentials
                ), mock.patch.object(sys, "argv", ["uploader.py", "--auth", "--full-access"]), mock.patch("builtins.print"):
                    uploader._main()
            finally:
                uploader.SCOPES = old
        self.assertEqual(seen["scopes"], [uploader.FULL_SCOPE])
        self.assertFalse(seen["token_existed"])

    def test_plain_auth_keeps_existing_token_and_scope(self):
        with tempfile.TemporaryDirectory() as d:
            token = Path(d) / "token.json"
            token.write_text("{}")
            with mock.patch.object(uploader, "TOKEN_FILE", token), mock.patch.object(
                uploader, "get_credentials"
            ), mock.patch.object(sys, "argv", ["uploader.py", "--auth"]), mock.patch("builtins.print"):
                uploader._main()
            self.assertTrue(token.exists())


if __name__ == "__main__":
    unittest.main()
