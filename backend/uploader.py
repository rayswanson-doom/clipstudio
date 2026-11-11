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
class DriveAuthError(RuntimeError):
    """Google Drive sign-in is unusable (revoked / expired token, no client file) - the message says what to do."""


class DriveOfflineError(ConnectionError):
    """Google cannot be reached (no internet); the work stays local and the upload can be retried later."""


def _google_reachable(timeout: float = 4.0) -> bool:
    import socket

    try:
        with socket.create_connection(("oauth2.googleapis.com", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _is_network_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) and not isinstance(exc, (FileNotFoundError, PermissionError)):
        return True
    return type(exc).__name__ in {"TransportError", "ServerNotFoundError", "ConnectTimeout", "ReadTimeout", "SSLError",
                                  "RemoteDisconnected", "ConnectionError", "timeout"}


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


def get_credentials(reachable: Optional[Callable[[], bool]] = None):
    """Load credentials from token.json, refreshing via the offline refresh token when expired. Runs the one-time
    browser consent only if no usable token exists. Writes the refreshed token back to token.json.

    Failure handling (never a silent hang):
      - no internet while refreshing        -> DriveOfflineError (no browser is opened; retry when online)
      - refresh token revoked / expired     -> sign in again if credentials.json exists, else DriveAuthError
      - token.json unreadable / corrupted   -> treated as signed out (the file is kept as token.json.bad)
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if TOKEN_FILE.is_file():
        try:
            # No scopes passed: use the ones stored in token.json, so a token created
            # with --full-access keeps working without any environment variable.
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("token.json could not be read (%s); signing in again.", exc)
            try:
                TOKEN_FILE.replace(TOKEN_FILE.with_name(TOKEN_FILE.name + ".bad"))
            except OSError:
                pass
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        if not (reachable or _google_reachable)():
            raise DriveOfflineError("Google Drive cannot be reached (no internet). The file stays on this computer; "
                                    "upload it again when you are online.")
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except Exception as exc:
            if _is_network_error(exc):
                raise DriveOfflineError(f"Google Drive could not be reached while refreshing the sign-in ({str(exc)[:100]}). "
                                        "Try again when the connection is back.") from None
            logger.warning("Token refresh failed (%s); re-authorizing.", exc)
            creds = None
            if not CREDENTIALS_FILE.is_file():
                raise DriveAuthError("Google Drive access was revoked or has expired and there is no credentials.json to "
                                     "sign in again. Download credentials.json (Desktop app) and run the sign-in once more.") from None

    if not CREDENTIALS_FILE.is_file():
        raise FileNotFoundError(
            f"No valid token.json and no OAuth client file at {CREDENTIALS_FILE}. "
            "Download credentials.json (Desktop app) from Google Cloud Console."
        )
    if not (reachable or _google_reachable)():
        raise DriveOfflineError("Google sign-in needs an internet connection; none was found.")

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    # access_type=offline + prompt=consent guarantees a refresh token is issued.
    creds = flow.run_local_server(
        port=0, access_type="offline", prompt="consent", open_browser=True, timeout_seconds=300
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
    metadata = {"name": path.name, "parents": [folder_id.strip()]}
    response = None
    for auth_try in (1, 2):
        service = get_drive_service()
        media = MediaFileUpload(
            str(path), mimetype=mime_type, resumable=True, chunksize=8 * 1024 * 1024
        )
        request = service.files().create(
            body=metadata, media_body=media, fields="id, webViewLink"
        )
        try:
            response = None
            while response is None:
                status, response = request.next_chunk(num_retries=3)
                if status:
                    logger.info("Upload %s: %d%%", path.name, int(status.progress() * 100))
            break
        except Exception as exc:
            code = getattr(getattr(exc, "resp", None), "status", None)
            if code == 401 and auth_try == 1:  # the access token died mid-upload: refresh and start the upload again
                logger.warning("Drive rejected the access token (401); refreshing and retrying once.")
                try:
                    creds = get_credentials()
                    if getattr(creds, "refresh_token", None):
                        from google.auth.transport.requests import Request

                        creds.refresh(Request())
                        _save_token(creds)
                except Exception as exc2:  # noqa: BLE001
                    raise DriveAuthError(f"Google Drive rejected the sign-in and it could not be refreshed ({str(exc2)[:100]}).") from None
                continue
            if code in (401, 403) and "rate" not in str(exc).lower() and "quota" not in str(exc).lower():
                raise DriveAuthError(f"Google Drive refused the upload (HTTP {code}). Check that the folder ID is right and that this "
                                     "account may add files to it.") from None
            if _is_network_error(exc) and not isinstance(exc, DriveOfflineError):
                raise DriveOfflineError(f"The upload was interrupted: Google Drive could not be reached ({str(exc)[:100]}).") from None
            raise

    link = response.get("webViewLink") or (
        f"https://drive.google.com/file/d/{response['id']}/view"
    )
    logger.info("Uploaded %s -> %s", path.name, link)
    return link


def drive_live_check(folder_id: str = "", service_factory: Optional[Callable] = None,
                     reachable: Optional[Callable[[], bool]] = None) -> dict:
    """A live, read-only check of Google Drive: {"ok": bool, "stage": str, "detail": str}. Never raises.
    stages: files (no token / credentials) -> network -> auth (token loads / refreshes) -> api (account) -> folder."""
    st = drive_status()
    if st == "unavailable":
        return {"ok": False, "stage": "files", "detail": "No token.json or credentials.json: Drive backup is off."}
    if not (reachable or _google_reachable)():
        return {"ok": False, "stage": "network", "detail": "No internet connection to Google."}
    try:
        service = (service_factory or get_drive_service)()
    except DriveAuthError as exc:
        return {"ok": False, "stage": "auth", "detail": str(exc)}
    except DriveOfflineError as exc:
        return {"ok": False, "stage": "network", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "auth", "detail": f"Sign-in failed: {str(exc)[:160]}"}
    try:
        about = service.about().get(fields="user(emailAddress)").execute()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "stage": "api", "detail": f"Drive API call failed: {str(exc)[:160]}"}
    if folder_id.strip():
        try:
            service.files().get(fileId=folder_id.strip(), fields="id,name", supportsAllDrives=True).execute()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "stage": "folder", "detail": f"The folder could not be opened ({str(exc)[:120]})."}
    who = ((about or {}).get("user") or {}).get("emailAddress", "")
    return {"ok": True, "stage": "folder" if folder_id.strip() else "api", "detail": f"Connected{(' as ' + who) if who else ''}."}


# --------------------------------------------------------------------------- #
# Automatic background backup
# --------------------------------------------------------------------------- #
_RETRY_HTTP_STATUS = {429, 500, 502, 503, 504}


def is_transient_error(exc: BaseException) -> bool:
    """True for temporary problems worth retrying (flaky network, Drive 5xx / rate limit).
    False for things a retry cannot fix: missing files, missing credentials, bad arguments,
    permission or not-found errors."""
    if isinstance(exc, (FileNotFoundError, PermissionError, ValueError, KeyboardInterrupt, DriveAuthError)):
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


def collect_final_videos(result: dict) -> list:
    """Every FINISHED video a pipeline result points to (clips, the classic generator's video, story episodes) -
    the files that already contain meme reactions / the 3D avatar and are what gets backed up to Google Drive.
    Existing files only, no duplicates, in order."""
    found: list = []
    for key in ("video_path", "output_path"):
        if result.get(key):
            found.append(result[key])
    for clip in result.get("clips") or []:
        if clip.get("output_path"):
            found.append(clip["output_path"])
    for ep in result.get("episodes_out") or []:
        if ep.get("video_path"):
            found.append(ep["video_path"])
    seen, out = set(), []
    for p in found:
        p = str(p)
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            out.append(p)
    return out


def collect_asset_files(book) -> list:
    """The pre-production assets worth keeping on Drive: every character sheet / location picture that exists on
    disk, plus the prompt cards (assets.json). `book` is an asset_manager.AssetBook."""
    files = [p for p in book.image_paths() if os.path.isfile(p)]
    card_file = os.path.join(str(book.folder), "assets.json")
    if os.path.isfile(card_file):
        files.append(card_file)
    return files


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
    parser.add_argument("--check", action="store_true", help="live check of Google Drive sign-in, network and folder access")
    parser.add_argument("--source", help="YouTube URL or local file path")
    parser.add_argument("--folder-id", help="Google Drive folder ID to upload into")
    args = parser.parse_args()

    if args.full_access:
        global SCOPES
        SCOPES = [FULL_SCOPE]
        if TOKEN_FILE.is_file():  # an old narrower token must not be reused
            TOKEN_FILE.unlink()
    if args.check:
        res = drive_live_check(args.folder_id or "")
        print(f"[{'OK' if res['ok'] else 'FAIL'}] {res['stage']}: {res['detail']}")
        sys.exit(0 if res["ok"] else 1)
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
