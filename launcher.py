"""
launcher.py  -  double-click entry point for Clip Studio.

Starts the Streamlit app (frontend/app.py) on a free local port and opens it in
the default browser once the server answers. Works from source
(`python launcher.py`) and when frozen by PyInstaller (the exe).

Design notes
  * Streamlit runs in the main thread via streamlit.web.cli.main (it installs
    signal handlers, which only work there). A small background thread waits for
    the server's health endpoint and then calls webbrowser.open.
  * Streamlit's own browser launch is disabled (--server.headless true) so the
    page is never opened twice.
  * bin/ (bundled ffmpeg) is added to PATH and the project root is made
    importable, for both source and frozen layouts.
  * If the app is already running on the preferred port, the browser is simply
    pointed at it instead of starting a second copy.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

HOST = "127.0.0.1"
PREFERRED_PORT = 8501
APP_RELATIVE = Path("frontend") / "app.py"
STARTUP_TIMEOUT = 90.0


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def resource_root() -> Path:
    """Folder that holds frontend/, backend/, bin/ and models/."""
    meipass = getattr(sys, "_MEIPASS", None)  # PyInstaller one-file / one-dir bundle
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def app_path() -> Path:
    path = resource_root() / APP_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"Could not find the app at {path}")
    return path


def writable_dir(preferred: Path) -> Path:
    """`preferred` if we can write there, else a folder in the temp directory."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return preferred
    except OSError:
        import tempfile

        fallback = Path(tempfile.gettempdir()) / "ClipStudio"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def redirect_stdio_if_needed() -> Optional[Path]:
    """A --noconsole exe starts with sys.stdout/sys.stderr = None, which makes
    Streamlit and logging misbehave. Point them at clipstudio.log instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else resource_root()
    log_path = writable_dir(base) / "clipstudio.log"
    stream = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return log_path


def prepare_environment(root: Optional[Path] = None) -> None:
    """Make bundled tools and modules visible to the app."""
    root = root or resource_root()
    bin_dir = root / "bin"
    if bin_dir.is_dir():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    for p in (root, root / "backend"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("PYTHONUTF8", "1")


# --------------------------------------------------------------------------- #
# networking
# --------------------------------------------------------------------------- #
def port_is_free(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) != 0


def pick_port(preferred: int = PREFERRED_PORT, tries: int = 20) -> int:
    """The preferred port if free, else the next free one, else an OS-chosen port."""
    for port in range(preferred, preferred + tries):
        if port_is_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def is_streamlit_running(port: int, host: str = HOST, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/_stcore/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(port: int, timeout: float = STARTUP_TIMEOUT, host: str = HOST, interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_streamlit_running(port, host):
            return True
        time.sleep(interval)
    return False


def open_browser_when_ready(port: int, timeout: float = STARTUP_TIMEOUT) -> threading.Thread:
    """Open the default browser as soon as the server is up (background thread)."""

    def _worker():
        if wait_until_ready(port, timeout):
            webbrowser.open(f"http://localhost:{port}")
        else:
            print(f"Clip Studio did not start within {int(timeout)} seconds.", file=sys.stderr)

    thread = threading.Thread(target=_worker, name="open-browser", daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# streamlit
# --------------------------------------------------------------------------- #
def build_streamlit_argv(app: Path, port: int) -> list:
    return [
        "streamlit", "run", str(app),
        "--server.port", str(port),
        "--server.address", HOST,
        "--server.headless", "true",  # we open the browser ourselves
        "--server.fileWatcherType", "none",  # no reload watcher in a packaged app
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",  # required when frozen
    ]


def main() -> int:
    redirect_stdio_if_needed()
    prepare_environment()
    try:
        app = app_path()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if "--selftest" in sys.argv[1:]:  # ClipStudio.exe --selftest : check the build, then exit
        import selftest

        return selftest.main(app)

    if is_streamlit_running(PREFERRED_PORT):  # already running: just show it
        webbrowser.open(f"http://localhost:{PREFERRED_PORT}")
        return 0

    port = pick_port()
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Streamlit is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"Starting Clip Studio on http://localhost:{port}  (close this window to stop)")
    open_browser_when_ready(port)
    sys.argv = build_streamlit_argv(app, port)
    try:
        stcli.main()  # blocks until the server stops; normally ends via sys.exit
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
