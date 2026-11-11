"""Uploader tests. Plain unittest (pytest runs them too), with hermetic Google / yt-dlp stand-ins from tests/_google_stubs.py,
so the suite passes with or without pytest and without the Google client packages or any network access."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uploader  # noqa: E402
import _google_stubs as gs  # noqa: E402


class _Base(unittest.TestCase):
    def setUp(self):
        self.mods = gs.install(self)
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)

    def write(self, name, data=b"x"):
        p = self.tmp / name
        p.write_bytes(data)
        return p

    def token(self, expiry="2000-01-01T00:00:00Z", **extra):
        p = self.tmp / "token.json"
        p.write_text(json.dumps({"token": "old", "refresh_token": "r", "client_id": "c", "client_secret": "s",
                                 "token_uri": "https://oauth2.googleapis.com/token", "scopes": uploader.SCOPES,
                                 "expiry": expiry, **extra}))
        return p


class DownloadTests(_Base):
    def test_local_path_returned(self):
        f = self.write("clip.mp4")
        self.assertEqual(uploader.download_media(str(f)), str(f.resolve()))

    def test_missing_local_path(self):
        with self.assertRaises(FileNotFoundError):
            uploader.download_media(str(self.tmp / "nope.mp4"))

    def test_empty_source(self):
        with self.assertRaises(ValueError):
            uploader.download_media("  ")

    def test_url_download_uses_ytdlp(self):
        out = self.write("video [abc].mp4", b"data")
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.extract_info.return_value = {"requested_downloads": [{"filepath": str(out)}]}
        with mock.patch("yt_dlp.YoutubeDL", return_value=fake) as ydl:
            got = uploader.download_media("https://youtu.be/abc", str(self.tmp))
        self.assertEqual(got, str(out.resolve()))
        self.assertIs(ydl.call_args[0][0]["noplaylist"], True)

    def test_url_download_failure(self):
        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.extract_info.side_effect = Exception("boom")
        with mock.patch("yt_dlp.YoutubeDL", return_value=fake):
            with self.assertRaises(RuntimeError):
                uploader.download_media("https://youtu.be/abc", str(self.tmp))


def _service(*chunks):
    service = mock.MagicMock()
    service.files.return_value.create.return_value.next_chunk.side_effect = list(chunks)
    return service


OK = (None, {"id": "FID", "webViewLink": "https://drive.google.com/file/d/FID/view"})


class UploadTests(_Base):
    def test_upload_returns_link(self):
        f = self.write("clip.mp4", b"x" * 100)
        service = _service((mock.MagicMock(progress=lambda: 0.5), None), OK)
        with mock.patch.object(uploader, "get_drive_service", return_value=service):
            link = uploader.upload_to_google_drive(str(f), "FOLDER")
        self.assertTrue(link.endswith("/FID/view"))
        self.assertEqual(service.files.return_value.create.call_args.kwargs["body"],
                         {"name": "clip.mp4", "parents": ["FOLDER"]})

    def test_upload_validation(self):
        with self.assertRaises(FileNotFoundError):
            uploader.upload_to_google_drive(str(self.tmp / "x.mp4"), "F")
        f = self.write("a.mp4")
        with self.assertRaises(ValueError):
            uploader.upload_to_google_drive(str(f), " ")

    def test_expired_access_token_mid_upload_is_refreshed_and_retried(self):
        f = self.write("clip.mp4")
        first = _service(gs.FakeHttpError(401, "Invalid Credentials"))
        second = _service(OK)
        creds = mock.MagicMock(refresh_token="r")
        with mock.patch.object(uploader, "get_drive_service", side_effect=[first, second]), \
                mock.patch.object(uploader, "get_credentials", return_value=creds), \
                mock.patch.object(uploader, "_save_token") as save:
            link = uploader.upload_to_google_drive(str(f), "F")
        self.assertTrue(link.endswith("/FID/view"))
        creds.refresh.assert_called_once()
        save.assert_called_once()

    def test_401_twice_gives_a_clear_auth_error(self):
        f = self.write("clip.mp4")
        svc = lambda: _service(gs.FakeHttpError(401, "Invalid Credentials"))  # noqa: E731
        with mock.patch.object(uploader, "get_drive_service", side_effect=[svc(), svc()]), \
                mock.patch.object(uploader, "get_credentials", return_value=mock.MagicMock(refresh_token="r")), \
                mock.patch.object(uploader, "_save_token"):
            with self.assertRaises(uploader.DriveAuthError):
                uploader.upload_to_google_drive(str(f), "F")

    def test_403_is_an_auth_error_not_retried(self):
        f = self.write("clip.mp4")
        with mock.patch.object(uploader, "get_drive_service", return_value=_service(gs.FakeHttpError(403, "forbidden"))):
            with self.assertRaises(uploader.DriveAuthError) as cm:
                uploader.upload_to_google_drive(str(f), "F")
        self.assertIn("folder ID", str(cm.exception))
        self.assertFalse(uploader.is_transient_error(cm.exception))

    def test_dropped_connection_becomes_offline_error_and_is_retryable(self):
        f = self.write("clip.mp4")
        with mock.patch.object(uploader, "get_drive_service", return_value=_service(ConnectionResetError("reset"))):
            with self.assertRaises(uploader.DriveOfflineError) as cm:
                uploader.upload_to_google_drive(str(f), "F")
        self.assertTrue(uploader.is_transient_error(cm.exception))

    def test_retry_wrapper_retries_offline_then_succeeds(self):
        calls = []

        def flaky(path, folder):
            calls.append(1)
            if len(calls) < 3:
                raise uploader.DriveOfflineError("offline")
            return "LINK"

        sleeps = []
        self.assertEqual(uploader.upload_with_retry("f", "F", sleep=sleeps.append, upload_fn=flaky), "LINK")
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_retry_wrapper_does_not_retry_auth_errors(self):
        def denied(path, folder):
            raise uploader.DriveAuthError("revoked")

        sleeps = []
        with self.assertRaises(uploader.DriveAuthError):
            uploader.upload_with_retry("f", "F", sleep=sleeps.append, upload_fn=denied)
        self.assertEqual(sleeps, [])


class CredentialsTests(_Base):
    def patch_files(self, token=None, creds=None):
        return mock.patch.multiple(uploader, TOKEN_FILE=token or (self.tmp / "token.json"),
                                   CREDENTIALS_FILE=creds or (self.tmp / "credentials.json"))

    def test_refresh_token_flow(self):
        token = self.token()
        with self.patch_files(token=token):
            creds = uploader.get_credentials(reachable=lambda: True)
        self.assertEqual(creds.token, "new")
        self.assertEqual(json.loads(token.read_text())["token"], "new")

    def test_valid_token_needs_no_network(self):
        self.token(expiry="2999-01-01T00:00:00Z")
        with self.patch_files():
            creds = uploader.get_credentials(reachable=lambda: self.fail("network must not be touched"))
        self.assertEqual(creds.token, "old")

    def test_offline_refresh_raises_offline_error_without_opening_a_browser(self):
        self.token()
        with self.patch_files():
            with self.assertRaises(uploader.DriveOfflineError):
                uploader.get_credentials(reachable=lambda: False)
        self.mods["google_auth_oauthlib.flow"].InstalledAppFlow.from_client_secrets_file.assert_not_called()

    def test_network_failure_during_refresh_is_offline_not_signout(self):
        self.token()

        def boom(creds):
            raise gs.TransportError("connection aborted")

        gs.FakeCredentials.refresh_behaviour = boom
        with self.patch_files():
            with self.assertRaises(uploader.DriveOfflineError):
                uploader.get_credentials(reachable=lambda: True)

    def test_revoked_refresh_token_without_client_file_is_a_clear_auth_error(self):
        self.token()

        def revoked(creds):
            raise gs.RefreshError("invalid_grant: Token has been expired or revoked.")

        gs.FakeCredentials.refresh_behaviour = revoked
        with self.patch_files():
            with self.assertRaises(uploader.DriveAuthError) as cm:
                uploader.get_credentials(reachable=lambda: True)
        self.assertIn("credentials.json", str(cm.exception))

    def test_revoked_refresh_token_with_client_file_signs_in_again(self):
        self.token()
        cred_file = self.write("credentials.json", b"{}")

        def revoked(creds):
            raise gs.RefreshError("invalid_grant")

        gs.FakeCredentials.refresh_behaviour = revoked
        fresh = mock.MagicMock()
        fresh.to_json.return_value = "{}"
        flow = self.mods["google_auth_oauthlib.flow"].InstalledAppFlow.from_client_secrets_file.return_value
        flow.run_local_server.return_value = fresh
        with self.patch_files(creds=cred_file):
            got = uploader.get_credentials(reachable=lambda: True)
        self.assertIs(got, fresh)
        self.assertEqual(flow.run_local_server.call_args.kwargs["timeout_seconds"], 300)  # a sign-in can never hang forever

    def test_corrupted_token_is_set_aside(self):
        (self.tmp / "token.json").write_text("{not json")
        with self.patch_files():
            with self.assertRaises(FileNotFoundError):
                uploader.get_credentials(reachable=lambda: True)
        self.assertTrue((self.tmp / "token.json.bad").is_file())

    def test_missing_credentials_file(self):
        with self.patch_files():
            with self.assertRaises(FileNotFoundError):
                uploader.get_credentials()


class LiveCheckTests(_Base):
    def test_no_files(self):
        with mock.patch.multiple(uploader, TOKEN_FILE=self.tmp / "t.json", CREDENTIALS_FILE=self.tmp / "c.json"):
            self.assertEqual(uploader.drive_live_check()["stage"], "files")

    def test_offline(self):
        self.token()
        with mock.patch.multiple(uploader, TOKEN_FILE=self.tmp / "token.json", CREDENTIALS_FILE=self.tmp / "c.json"):
            res = uploader.drive_live_check(reachable=lambda: False)
        self.assertEqual((res["ok"], res["stage"]), (False, "network"))

    def test_auth_failure_is_reported_not_raised(self):
        self.token()
        with mock.patch.multiple(uploader, TOKEN_FILE=self.tmp / "token.json", CREDENTIALS_FILE=self.tmp / "c.json"):
            res = uploader.drive_live_check(reachable=lambda: True,
                                            service_factory=mock.Mock(side_effect=uploader.DriveAuthError("revoked")))
        self.assertEqual((res["ok"], res["stage"]), (False, "auth"))

    def test_ok_with_folder(self):
        self.token()
        svc = mock.MagicMock()
        svc.about.return_value.get.return_value.execute.return_value = {"user": {"emailAddress": "a@b.c"}}
        with mock.patch.multiple(uploader, TOKEN_FILE=self.tmp / "token.json", CREDENTIALS_FILE=self.tmp / "c.json"):
            res = uploader.drive_live_check("FOLDER", reachable=lambda: True, service_factory=lambda: svc)
        self.assertTrue(res["ok"])
        self.assertIn("a@b.c", res["detail"])
        self.assertEqual(svc.files.return_value.get.call_args.kwargs["fileId"], "FOLDER")

    def test_folder_failure(self):
        self.token()
        svc = mock.MagicMock()
        svc.files.return_value.get.return_value.execute.side_effect = gs.FakeHttpError(404, "not found")
        with mock.patch.multiple(uploader, TOKEN_FILE=self.tmp / "token.json", CREDENTIALS_FILE=self.tmp / "c.json"):
            res = uploader.drive_live_check("F", reachable=lambda: True, service_factory=lambda: svc)
        self.assertEqual((res["ok"], res["stage"]), (False, "folder"))


if __name__ == "__main__":
    unittest.main()
