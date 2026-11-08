"""
backend/uploader.py

Two responsibilities, using only free packages (yt-dlp, google-api-python-client):

1. download_media(source)  -> local file path
   Accepts a YouTube/video URL (downloaded with yt-dlp) or an existing local path.

2. upload_to_google_drive(file_path, folder_id) -> shareable link
   Authenticates against Google Drive with a local token.json. The token stores
   an offline refresh token, so after the first one-time browser consent the
   script refreshes access silently with no further prompts.

Setup (one time):
  1. In Google Cloud Console, create a project, enable the "Google Drive API",
     and create an OAuth client ID of type "Desktop app".
  2. Download it as credentials.json and place it in the project root (next to
     launcher.py; next to the exe when packaged) or set GOOGLE_CREDENTIALS_FILE.
  3. Run:  python backend/uploader.py --auth
     A browser opens once; token.json is then saved in the same folder.

Install:
  pip install yt-dlp google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def _data_dir() -> Path:
    """Where token.json / credentials.json live: next to the exe when packaged
    (PyInstaller), otherwise the project root (the folder above backend/)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return BASE_DIR.parent


DATA_DIR = _data_dir()

# Default is drive.file (least privilege): the app only sees files/folders it
# created itself. That can make uploads into a folder YOU created by hand fail
# with "File not found". If that happens, sign in again with
#   python backend/uploader.py --auth --full-access
# (or set GOOGLE_DRIVE_FULL_ACCESS=1), which uses the broader "drive" scope.
FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
FULL_SCOPE = "https://www.googleapis.com/auth/drive"


def _default_scopes() -> list:
    return [FULL_SCOPE if os.environ.get("GOOGLE_DRIVE_FULL_ACCESS") == "1" else FILE_SCOPE]


SCOPES = _default_scopes()

TOKEN_FILE = Path(os.environ.get("GOOGLE_TOKEN_FILE", DATA_DIR / "token.json"))
CREDENTIALS_FILE = Path(
    os.environ.get("GOOGLE_CREDENTIALS_FILE", DATA_DIR / "credentials.json")
)
DEFAULT_DOWNLOAD_DIR = DATA_DIR / "downloads"


# --------------------------------------------------------------------------- #
# 1. Download / accept local media
# --------------------------------------------------------------------------- #
def _is_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def download_media(source: str, output_dir: Optional[str] = None) -> str:
    """Return a local file path for `source`.

    * If `source` is an existing local file, its absolute path is returned.
    * If `source` is an http(s) URL (YouTube or any site yt-dlp supports),
      the video is downloaded into `output_dir` and the final path returned.

    Raises FileNotFoundError for missing local paths and RuntimeError when the
    download fails.
    """
    if not source or not str(source).strip():
        raise ValueError("source must be a non-empty URL or file path")
    source = str(source).strip().strip('"').strip("'")

    if not _is_url(source):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Local media file not found: {source}")
        return str(path.resolve())

    import yt_dlp  # imported lazily so local-path use needs no yt-dlp

    out_dir = Path(output_dir) if output_dir else DEFAULT_DOWNLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(out_dir / "%(title).150B [%(id)s].%(ext)s"),
        # Prefer mp4 so downstream clipping/upload is predictable.
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "restrictfilenames": False,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)
            if info is None:
                raise RuntimeError("yt-dlp returned no information")
            if "entries" in info and info["entries"]:  # playlist fallback
                info = info["entries"][0]
            downloads = info.get("requested_downloads") or []
            filepath = (downloads[0].get("filepath") if downloads else None) or (
                ydl.prepare_filename(info)
            )
    except Exception as exc:  # yt_dlp.utils.DownloadError and friends
        raise RuntimeError(f"Failed to download {source}: {exc}") from exc

    final = Path(filepath)
    if not final.is_file():
        # merge step may have changed the extension
        merged = final.with_suffix(".mp4")
        if merged.is_file():
            final = merged
        else:
            raise RuntimeError(f"Download finished but file not found: {filepath}")
    return str(final.resolve())


# --------------------------------------------------------------------------- #
# 2. Google Drive auth + upload
# --------------------------------------------------------------------------- #
def drive_status() -> str:
    """Cheap, offline check of what Google files exist. Never raises.

      "ready"        token.json present  -> uploads work with no prompt
      "needs_signin" only credentials.json -> first upload opens a Google sign-in page
      "unavailable"  neither file         -> Drive backup is off; clips stay local
    """
    if TOKEN_FILE.is_file():
        return "ready"
    if CREDENTIALS_FILE.is_file():
        return "needs_signin"
    return "unavailable"


def get_credentials():
    """Load credentials from token.json, refreshing via the offline refresh
    token when expired. Runs the one-time browser consent only if no usable
    token exists. Writes the refreshed token back to token.json."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if TOKEN_FILE.is_file():
        # No scopes passed: use the ones stored in token.json, so a token created
        # with --full-access keeps working without any environment variable.
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except Exception as exc:
            logger.warning("Token refresh failed (%s); re-authorizing.", exc)
            creds = None

    if not CREDENTIALS_FILE.is_file():
        raise FileNotFoundError(
            f"No valid token.json and no OAuth client file at {CREDENTIALS_FILE}. "
            "Download credentials.json (Desktop app) from Google Cloud Console."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    # access_type=offline + prompt=consent guarantees a refresh token is issued.
    creds = flow.run_local_server(
        port=0, access_type="offline", prompt="consent", open_browser=True
    )
    _save_token(creds)
    return creds


def _save_token(creds) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass


def get_drive_service():
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=get_credentials(), cache_discovery=False)


def upload_to_google_drive(file_path: str, folder_id: str) -> str:
    """Upload `file_path` into the Drive folder `folder_id` and return the
    file's web link (https://drive.google.com/file/d/<id>/view).

    Uses a resumable upload so large clips survive network hiccups.
    """
    from googleapiclient.http import MediaFileUpload

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File to upload not found: {file_path}")
    if not folder_id or not folder_id.strip():
        raise ValueError("folder_id must be a non-empty Google Drive folder ID")

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    service = get_drive_service()

    metadata = {"name": path.name, "parents": [folder_id.strip()]}
    media = MediaFileUpload(
        str(path), mimetype=mime_type, resumable=True, chunksize=8 * 1024 * 1024
    )
    request = service.files().create(
        body=metadata, media_body=media, fields="id, webViewLink"
    )

    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=3)
        if status:
            logger.info("Upload %s: %d%%", path.name, int(status.progress() * 100))

    link = response.get("webViewLink") or (
        f"https://drive.google.com/file/d/{response['id']}/view"
    )
    logger.info("Uploaded %s -> %s", path.name, link)
    return link


# --------------------------------------------------------------------------- #
# Automatic background backup
# --------------------------------------------------------------------------- #
_RETRY_HTTP_STATUS = {429, 500, 502, 503, 504}


def is_transient_error(exc: BaseException) -> bool:
    """True for temporary problems worth retrying (flaky network, Drive 5xx / rate limit).
    False for things a retry cannot fix: missing files, missing credentials, bad arguments,
    permission or not-found errors."""
    if isinstance(exc, (FileNotFoundError, PermissionError, ValueError, KeyboardInterrupt)):
        return False
    status = getattr(getattr(exc, "resp", None), "status", None)  # googleapiclient HttpError
    if status is not None:
        return int(status) in _RETRY_HTTP_STATUS
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    # DNS failure / SSL hiccup / dropped socket (names checked so we need no optional imports)
    return type(exc).__name__ in {"ServerNotFoundError", "SSLError", "RemoteDisconnected", "IncompleteRead",
                                  "ReadTimeout", "ConnectTimeout", "timeout"}


def upload_with_retry(
    file_path: str,
    folder_id: str,
    attempts: int = 3,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    upload_fn: Optional[Callable[[str, str], str]] = None,
) -> str:
    """upload_to_google_drive with automatic retries (2 s, 4 s, ...) for temporary errors only."""
    fn = upload_fn or upload_to_google_drive
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn(file_path, folder_id)
        except Exception as exc:
            if attempt >= attempts or not is_transient_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Drive upload of %s failed (%s); retry %d/%d in %.0fs",
                           Path(str(file_path)).name, exc, attempt, attempts - 1, delay)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


class DriveBackup:
    """Uploads finished files to Drive on ONE background thread, so the next clip can be
    processed while the previous one is still uploading.

        with DriveBackup(folder_id) as backup:
            future = backup.submit("clip1.mp4")      # returns immediately
            ...                                       # keep working
            link = future.result()                    # raises if the upload finally failed

    A failed upload never touches the local file: the caller keeps it and just reports the error.
    One worker keeps the (non-thread-safe) Google client and the one-time sign-in single-threaded.
    """

    def __init__(self, folder_id: str, upload_fn: Optional[Callable[[str, str], str]] = None,
                 attempts: int = 3, base_delay: float = 2.0):
        if not folder_id or not str(folder_id).strip():
            raise ValueError("folder_id must be a non-empty Google Drive folder ID")
        self.folder_id = str(folder_id).strip()
        self._upload_fn = upload_fn
        self._attempts, self._base_delay = attempts, base_delay
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drive-backup")

    def submit(self, file_path: str) -> Future:
        return self._pool.submit(
            upload_with_retry, str(file_path), self.folder_id, self._attempts, self._base_delay,
            time.sleep, self._upload_fn,
        )

    def close(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "DriveBackup":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def backup_files(paths, folder_id: str, upload_fn: Optional[Callable[[str, str], str]] = None,
                 attempts: int = 3, base_delay: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Back up several finished files (e.g. every episode of a story and its project archive) to one Drive folder.

    Returns {"links": {path: link}, "errors": {path: message}}. One failing file never stops the others and never
    touches the local copy; missing paths are reported as errors. Uploads run one at a time on a single background
    worker (the Google client is not thread-safe) with the same retry rules as DriveBackup.
    """
    out = {"links": {}, "errors": {}}
    todo = []
    for p in paths:
        p = str(p)
        if p in out["links"] or p in todo:
            continue
        if not Path(p).is_file():
            out["errors"][p] = "file not found"
        else:
            todo.append(p)
    if not todo:
        return out
    with DriveBackup(folder_id, upload_fn=upload_fn, attempts=attempts, base_delay=base_delay) as backup:
        futures = [(p, backup.submit(p)) for p in todo]
        for p, fut in futures:
            try:
                out["links"][p] = fut.result()
            except Exception as exc:  # noqa: BLE001 - report it, keep going
                out["errors"][p] = str(exc) or type(exc).__name__
    return out


def zip_folder(folder: str, zip_path: str, skip=("work",)) -> str:
    """Zip a project folder (script, storyboard, characters, graph, subtitles...) for backup, leaving out the
    scratch sub-folders named in `skip`. Returns zip_path."""
    import zipfile

    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    dest = Path(zip_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if f.is_file() and rel.parts[0] not in skip:
                zf.write(f, rel.as_posix())
    return str(dest)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download media and upload to Drive")
    parser.add_argument("--auth", action="store_true", help="run one-time Google sign-in")
    parser.add_argument("--full-access", action="store_true",
                        help="sign in with full Drive access (needed to upload into a folder "
                             "you created by hand if the default scope gives 'File not found')")
    parser.add_argument("--source", help="YouTube URL or local file path")
    parser.add_argument("--folder-id", help="Google Drive folder ID to upload into")
    args = parser.parse_args()

    if args.full_access:
        global SCOPES
        SCOPES = [FULL_SCOPE]
        if TOKEN_FILE.is_file():  # an old narrower token must not be reused
            TOKEN_FILE.unlink()
    if args.auth:
        get_credentials()
        print(f"Authorized. Token saved to {TOKEN_FILE}")
        return
    if args.source:
        local = download_media(args.source)
        print(f"Media ready: {local}")
        if args.folder_id:
            print(f"Uploaded: {upload_to_google_drive(local, args.folder_id)}")
        return
    parser.print_help()


if __name__ == "__main__":
    _main()
