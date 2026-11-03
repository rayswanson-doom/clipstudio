"""
build.py  -  package CinePulse AI into a double-clickable Windows folder.

    python build.py              # preflight + PyInstaller build + assemble
    python build.py --dry-run    # preflight only, prints the PyInstaller command

Run it ON WINDOWS (PyInstaller cannot cross-compile) from the project root, in
an environment where `pip install -r requirements.txt pyinstaller` has been done.

What it does
  1. Preflight scan: missing source files, missing Python packages, a missing or
     non-Windows ffmpeg.exe, missing models, an unusable token.json. Every
     problem is listed at once, with the fix, before anything is built.
  2. Pre-fetches the faster-whisper "base" model into models/ (needs internet at
     BUILD time only) so transcription works offline on the target laptop.
  3. Runs PyInstaller with --onedir --noconsole and --collect-all for streamlit,
     mediapipe, faster_whisper and cv2 (plus the other packages the pipeline
     needs: ctranslate2, googleapiclient, yt_dlp, ...).
  4. Copies bin/ffmpeg.exe, models/ and token.json NEXT TO the exe in
     dist/CinePulseAI/, which is where the app looks for them, and verifies it.

Internet is still needed at RUN time for the parts that are online by nature:
YouTube downloads, edge-tts voices, and the Google Drive upload.

SECURITY: token.json lets whoever holds it upload to and manage the files this
app created in YOUR Google Drive. Bundling it makes the folder work on a laptop
without a sign-in, but only hand the folder to people you trust. Revoke access
any time at https://myaccount.google.com/permissions. Use --no-token to leave it out.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
APP_NAME = "CinePulseAI"

BACKEND_MODULES = ["pipeline", "uploader", "scorer", "transcriber", "tts_engine", "video_engine", "avatar_sync"]
REQUIRED_SOURCES = [
    "launcher.py", "selftest.py", "frontend/app.py",
    *[f"backend/{m}.py" for m in BACKEND_MODULES],
]

# Requested explicitly; each must be installed or the build is refused.
COLLECT_ALL = ["streamlit", "mediapipe", "faster_whisper", "cv2", "googleapiclient"]
# Also needed at run time; added only when installed so a missing optional
# package cannot break the PyInstaller command line.
EXTRA_COLLECT_ALL = [
    "ctranslate2", "av", "tokenizers", "onnxruntime",
    "google_auth_oauthlib", "yt_dlp", "edge_tts",
]
PIP_NAMES = {  # import name -> pip package name, where they differ
    "cv2": "opencv-python",
    "googleapiclient": "google-api-python-client",
    "faster_whisper": "faster-whisper",
}
HIDDEN_IMPORTS = [
    *BACKEND_MODULES, "selftest",
    "google.auth.transport.requests", "google.oauth2.credentials",
    "googleapiclient.discovery", "googleapiclient.http",
    "google_auth_oauthlib.flow", "mediapipe.tasks.python.vision",
]
COPY_METADATA = ["streamlit"]  # streamlit reads its own version via importlib.metadata

FFMPEG_REL = Path("bin") / "ffmpeg.exe"
BLAZEFACE_REL = Path("models") / "blaze_face_short_range.tflite"
WHISPER_REL = Path("models") / "faster-whisper-base"
WHISPER_REPO = "Systran/faster-whisper-base"
TOKEN_REL = Path("token.json")

TOKEN_WARNING = (
    "token.json is bundled. It gives whoever holds this folder upload/manage access to the "
    "files this app created in your Google Drive. Share the folder only with people you trust; "
    "revoke access at https://myaccount.google.com/permissions if it leaks."
)


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"ERROR:   {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines) if lines else "Preflight passed with no problems."


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def exe_name() -> str:
    return APP_NAME + (".exe" if os.name == "nt" else "")


# --------------------------------------------------------------------------- #
# 1. preflight
# --------------------------------------------------------------------------- #
def preflight(
    root: Path = ROOT,
    include_token: bool = True,
    require_token: bool = False,
    allow_missing: bool = False,
    allow_non_windows: bool = False,
) -> Report:
    """Scan for everything the build and the finished app depend on."""
    r = Report()
    soft = r.warnings if allow_missing else r.errors  # "missing but you said continue"

    if os.name != "nt" and not allow_non_windows:
        r.errors.append(
            "This is not Windows. PyInstaller cannot produce a Windows .exe from here. "
            "Run build.py on a Windows PC (or pass --allow-non-windows for a test build)."
        )

    for rel in REQUIRED_SOURCES:
        if not (root / rel).is_file():
            r.errors.append(f"Missing source file: {rel}")

    if not _has_module("PyInstaller"):
        r.errors.append("PyInstaller is not installed. Fix: pip install pyinstaller")
    for pkg in COLLECT_ALL:
        if not _has_module(pkg):
            fix = PIP_NAMES.get(pkg, pkg.replace("_", "-"))
            r.errors.append(f"Python package '{pkg}' is not installed. Fix: pip install {fix}")
    for pkg in EXTRA_COLLECT_ALL:
        if not _has_module(pkg):
            r.warnings.append(f"Optional package '{pkg}' is not installed and will not be bundled.")

    ffmpeg = root / FFMPEG_REL
    if not ffmpeg.is_file():
        soft.append(
            f"Missing {FFMPEG_REL.as_posix()}. Download a Windows build (e.g. from gyan.dev or "
            "BtbN FFmpeg-Builds), then put ffmpeg.exe in the bin/ folder."
        )
    else:
        with ffmpeg.open("rb") as fh:
            if fh.read(2) != b"MZ":
                r.errors.append(
                    f"{FFMPEG_REL.as_posix()} is not a Windows executable (wrong file type). "
                    "A Linux/Mac ffmpeg will not run inside the Windows build."
                )

    if not (root / BLAZEFACE_REL).is_file():
        soft.append(
            f"Missing {BLAZEFACE_REL.as_posix()} (BlazeFace model). Download "
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/"
            "float16/1/blaze_face_short_range.tflite into models/. Without it the app falls back "
            "to a static centre crop."
        )

    if not (root / WHISPER_REL / "model.bin").is_file():
        r.warnings.append(
            f"{WHISPER_REL.as_posix()} not found; build.py will try to download it now "
            "(needs internet during the build). Without it the app downloads the model on first "
            "transcription instead."
        )

    token = root / TOKEN_REL
    if not include_token:
        r.warnings.append("token.json will NOT be bundled (--no-token); users must run the sign-in once.")
    elif not token.is_file():
        msg = ("token.json not found. Create it once with `python backend/uploader.py --auth` "
               "(needs your credentials.json), then rebuild.")
        (r.errors if require_token else r.warnings).append(msg)
    else:
        try:
            data = json.loads(token.read_text(encoding="utf-8"))
            if not data.get("refresh_token"):
                r.errors.append(
                    "token.json has no refresh_token, so it would stop working after about an hour. "
                    "Delete it and run `python backend/uploader.py --auth` again."
                )
            else:
                r.warnings.append(TOKEN_WARNING)
        except (OSError, ValueError):
            r.errors.append("token.json is not valid JSON. Delete it and run the sign-in again.")

    return r


# --------------------------------------------------------------------------- #
# 2. optional model pre-fetch
# --------------------------------------------------------------------------- #
def fetch_whisper_model(root: Path = ROOT) -> bool:
    """Download the faster-whisper base model into models/ (build-time internet)."""
    target = root / WHISPER_REL
    if (target / "model.bin").is_file():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=WHISPER_REPO, local_dir=str(target))
    except Exception as exc:
        print(f"Could not pre-download the Whisper model ({exc}). "
              "It will be downloaded on first use instead.", file=sys.stderr)
        return False
    return (target / "model.bin").is_file()


# --------------------------------------------------------------------------- #
# 3. PyInstaller
# --------------------------------------------------------------------------- #
def build_pyinstaller_args(root: Path = ROOT, dist: Optional[Path] = None, work: Optional[Path] = None) -> List[str]:
    dist = dist or root / "dist"
    work = work or root / "build"
    sep = os.pathsep  # ';' on Windows, ':' elsewhere - what --add-data expects
    args = [
        "--noconfirm", "--clean",
        "--onedir", "--noconsole",
        "--name", APP_NAME,
        "--distpath", str(dist), "--workpath", str(work), "--specpath", str(work),
        "--paths", str(root / "backend"),
        # Streamlit runs frontend/app.py from disk, so it must exist as a real file.
        "--add-data", f"{root / 'frontend'}{sep}frontend",
    ]
    for pkg in COLLECT_ALL:
        args += ["--collect-all", pkg]
    for pkg in EXTRA_COLLECT_ALL:
        if _has_module(pkg):
            args += ["--collect-all", pkg]
    for pkg in COPY_METADATA:
        args += ["--copy-metadata", pkg]
    for mod in HIDDEN_IMPORTS:
        args += ["--hidden-import", mod]
    args.append(str(root / "launcher.py"))
    return args


def run_pyinstaller(args: List[str]) -> int:
    return subprocess.call([sys.executable, "-m", "PyInstaller", *args])


# --------------------------------------------------------------------------- #
# 4. assemble + verify
# --------------------------------------------------------------------------- #
def assemble(app_dir: Path, root: Path = ROOT, include_token: bool = True) -> List[str]:
    """Copy ffmpeg, models and token.json next to the exe. Returns what was copied."""
    copied: List[str] = []
    app_dir.mkdir(parents=True, exist_ok=True)

    src = root / FFMPEG_REL
    if src.is_file():
        (app_dir / "bin").mkdir(exist_ok=True)
        shutil.copy2(src, app_dir / FFMPEG_REL)
        copied.append(FFMPEG_REL.as_posix())

    models = root / "models"
    if models.is_dir():
        shutil.copytree(models, app_dir / "models", dirs_exist_ok=True)
        copied.append("models/")

    token = root / TOKEN_REL
    if include_token and token.is_file():
        shutil.copy2(token, app_dir / TOKEN_REL)
        copied.append(TOKEN_REL.as_posix())

    (app_dir / "README.txt").write_text(
        f"CinePulse AI\n===========\n\nDouble-click {exe_name()}. Your browser opens the app at "
        "http://localhost:8501 (the next free port is used if that one is busy).\n"
        "There is no console window; use the 'Quit CinePulse AI' button in the sidebar to stop it.\n"
        "A log is written to cinepulseai.log in this folder.\n\n"
        "Needs internet for: YouTube downloads, edge-tts voices, Google Drive upload.\n"
        "Transcription, scoring, cropping and merging run fully offline.\n\n"
        "Google Drive: enter your Drive Folder ID in the sidebar. "
        + ("Your sign-in (token.json) is already in this folder - keep the folder private.\n"
           if (app_dir / TOKEN_REL).is_file() else
           "No token.json is included: run the sign-in once before uploading.\n"),
        encoding="utf-8",
    )
    return copied


def verify_dist(app_dir: Path, include_token: bool = True) -> List[str]:
    """Return a list of problems with the finished folder (empty = good)."""
    problems: List[str] = []
    if not (app_dir / exe_name()).is_file():
        problems.append(f"{exe_name()} was not produced")
    if not (app_dir / FFMPEG_REL).is_file():
        problems.append("bin/ffmpeg.exe is missing from the distribution folder")
    if not (app_dir / BLAZEFACE_REL).is_file():
        problems.append("models/blaze_face_short_range.tflite is missing (face tracking falls back to centre crop)")
    if not (app_dir / WHISPER_REL / "model.bin").is_file():
        problems.append("Whisper model is not bundled (it will download on first transcription)")
    if include_token and not (app_dir / TOKEN_REL).is_file():
        problems.append("token.json is missing from the distribution folder")
    return problems


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the CinePulse AI distribution folder")
    ap.add_argument("--dry-run", action="store_true", help="run the preflight and print the command only")
    ap.add_argument("--no-token", action="store_true", help="do not bundle token.json")
    ap.add_argument("--require-token", action="store_true", help="fail if token.json is missing")
    ap.add_argument("--allow-missing", action="store_true",
                    help="downgrade missing ffmpeg.exe / BlazeFace model to warnings")
    ap.add_argument("--allow-non-windows", action="store_true", help="permit a test build off Windows")
    ap.add_argument("--skip-model-download", action="store_true", help="do not pre-fetch the Whisper model")
    a = ap.parse_args(argv)
    include_token = not a.no_token

    report = preflight(ROOT, include_token, a.require_token, a.allow_missing, a.allow_non_windows)
    print(report.render())
    args = build_pyinstaller_args(ROOT)
    if a.dry_run:
        print("\nPyInstaller command:\n  python -m PyInstaller " + " ".join(args))
        return 0 if report.ok else 1
    if not report.ok:
        print("\nBuild stopped. Fix the errors above and run build.py again.", file=sys.stderr)
        return 1

    if not a.skip_model_download:
        fetch_whisper_model(ROOT)

    code = run_pyinstaller(args)
    if code != 0:
        print(f"PyInstaller failed with exit code {code}.", file=sys.stderr)
        return code

    app_dir = ROOT / "dist" / APP_NAME
    copied = assemble(app_dir, ROOT, include_token)
    print("Copied into the distribution folder: " + ", ".join(copied))
    problems = verify_dist(app_dir, include_token)
    for p in problems:
        print(f"WARNING: {p}")
    if include_token and (app_dir / TOKEN_REL).is_file():
        print("\n" + TOKEN_WARNING)
    print(f"\nDone: {app_dir / exe_name()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
