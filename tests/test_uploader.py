import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import uploader  # noqa: E402


def test_local_path_returned(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    assert uploader.download_media(str(f)) == str(f.resolve())


def test_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        uploader.download_media(str(tmp_path / "nope.mp4"))


def test_empty_source():
    with pytest.raises(ValueError):
        uploader.download_media("  ")


def test_url_download_uses_ytdlp(tmp_path):
    out = tmp_path / "video [abc].mp4"
    out.write_bytes(b"data")
    fake = mock.MagicMock()
    fake.__enter__.return_value = fake
    fake.extract_info.return_value = {"requested_downloads": [{"filepath": str(out)}]}
    with mock.patch("yt_dlp.YoutubeDL", return_value=fake) as ydl:
        got = uploader.download_media("https://youtu.be/abc", str(tmp_path))
    assert got == str(out.resolve())
    opts = ydl.call_args[0][0]
    assert opts["noplaylist"] is True


def test_url_download_failure(tmp_path):
    fake = mock.MagicMock()
    fake.__enter__.return_value = fake
    fake.extract_info.side_effect = Exception("boom")
    with mock.patch("yt_dlp.YoutubeDL", return_value=fake):
        with pytest.raises(RuntimeError):
            uploader.download_media("https://youtu.be/abc", str(tmp_path))


def test_upload_returns_link(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x" * 100)
    service = mock.MagicMock()
    req = service.files.return_value.create.return_value
    req.next_chunk.side_effect = [
        (mock.MagicMock(progress=lambda: 0.5), None),
        (None, {"id": "FID", "webViewLink": "https://drive.google.com/file/d/FID/view"}),
    ]
    with mock.patch.object(uploader, "get_drive_service", return_value=service):
        link = uploader.upload_to_google_drive(str(f), "FOLDER")
    assert link.endswith("/FID/view")
    body = service.files.return_value.create.call_args.kwargs["body"]
    assert body == {"name": "clip.mp4", "parents": ["FOLDER"]}


def test_upload_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        uploader.upload_to_google_drive(str(tmp_path / "x.mp4"), "F")
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        uploader.upload_to_google_drive(str(f), " ")


def test_refresh_token_flow(tmp_path, monkeypatch):
    token = tmp_path / "token.json"
    token.write_text(json.dumps({
        "token": "old", "refresh_token": "r", "client_id": "c",
        "client_secret": "s", "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": uploader.SCOPES, "expiry": "2000-01-01T00:00:00Z",
    }))
    monkeypatch.setattr(uploader, "TOKEN_FILE", token)

    def fake_refresh(self, request):
        import datetime
        self.token = "new"
        self.expiry = datetime.datetime.utcnow() + datetime.timedelta(hours=1)

    with mock.patch("google.oauth2.credentials.Credentials.refresh", fake_refresh):
        creds = uploader.get_credentials()
    assert creds.token == "new"
    assert json.loads(token.read_text())["token"] == "new"


def test_missing_credentials_file(tmp_path, monkeypatch):
    monkeypatch.setattr(uploader, "TOKEN_FILE", tmp_path / "token.json")
    monkeypatch.setattr(uploader, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    with pytest.raises(FileNotFoundError):
        uploader.get_credentials()
