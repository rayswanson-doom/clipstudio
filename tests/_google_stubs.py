"""Hermetic stand-ins for the Google client libraries and yt-dlp, so the uploader tests need neither those packages nor
network access (the pattern of googleworkspace/python-samples: build the service with unittest.mock and assert on the
requests it receives). Use as a context manager / `install()` inside a TestCase."""
import datetime
import json
import sys
import types
from unittest import mock


class FakeHttpError(Exception):
    """googleapiclient.errors.HttpError look-alike: `.resp.status`."""

    def __init__(self, status, message="error"):
        super().__init__(message)
        self.resp = types.SimpleNamespace(status=status)


class RefreshError(Exception):
    pass


class TransportError(Exception):
    pass


class FakeCredentials:
    """google.oauth2.credentials.Credentials look-alike backed by a token.json."""

    refresh_behaviour = None  # callable(self) or None; tests set it

    def __init__(self, info):
        self.token = info.get("token")
        self.refresh_token = info.get("refresh_token")
        self.scopes = info.get("scopes")
        exp = info.get("expiry")
        self.expiry = datetime.datetime.fromisoformat(exp.replace("Z", "")) if exp else None
        self._info = info

    @classmethod
    def from_authorized_user_file(cls, path, scopes=None):
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)  # ValueError on a corrupted file, like the real class
        if "refresh_token" not in info and "token" not in info:
            raise ValueError("The authorized user info is missing fields")
        return cls(info)

    @property
    def expired(self):
        return self.expiry is not None and self.expiry < datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    @property
    def valid(self):
        return bool(self.token) and not self.expired

    def refresh(self, request):
        if FakeCredentials.refresh_behaviour:
            FakeCredentials.refresh_behaviour(self)
            return
        self.token = "new"
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=1)

    def to_json(self):
        return json.dumps({**self._info, "token": self.token, "refresh_token": self.refresh_token})


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def stub_modules():
    FakeCredentials.refresh_behaviour = None
    ytdlp = _module("yt_dlp", YoutubeDL=mock.MagicMock())
    return {
        "google": _module("google"),
        "google.oauth2": _module("google.oauth2"),
        "google.oauth2.credentials": _module("google.oauth2.credentials", Credentials=FakeCredentials),
        "google.auth": _module("google.auth"),
        "google.auth.transport": _module("google.auth.transport"),
        "google.auth.transport.requests": _module("google.auth.transport.requests", Request=mock.MagicMock()),
        "google_auth_oauthlib": _module("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": _module("google_auth_oauthlib.flow", InstalledAppFlow=mock.MagicMock()),
        "googleapiclient": _module("googleapiclient"),
        "googleapiclient.discovery": _module("googleapiclient.discovery", build=mock.MagicMock()),
        "googleapiclient.http": _module("googleapiclient.http", MediaFileUpload=mock.MagicMock()),
        "yt_dlp": ytdlp,
    }


def install(testcase):
    """Patch sys.modules for the duration of one test; returns the stub dict."""
    mods = stub_modules()
    p = mock.patch.dict(sys.modules, mods)
    p.start()
    testcase.addCleanup(p.stop)
    return mods
