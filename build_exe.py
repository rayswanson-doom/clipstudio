"""
build_exe.py  -  build the CinePulse AI Windows distribution with PyInstaller.

    python build_exe.py --dry-run     # scan + show the PyInstaller arguments, build nothing
    python build_exe.py               # build dist/CinePulseAI/ and dist/CinePulseAI-ViMax-Portable.zip
    python build_exe.py --ci          # same, for an unattended CI machine
    python build_exe.py --git-push    # ...then `git add .`, `git commit`, `git push origin main`
    python build_exe.py --git-only    # only the git step (no build)
    python build_exe.py --with-torch       # also bundle PyTorch (+1-2 GB; not needed by the built-in 3-D renderer)
    python build_exe.py --with-paddleocr   # also bundle PaddleOCR itself (+~540 MB, see below)

Must run on WINDOWS (PyInstaller cannot cross-compile). No manual setup is
needed beyond `pip install -r requirements-build.txt`: if bin/ffmpeg.exe or the
BlazeFace model are missing they are downloaded automatically, and the Whisper
"base" model is fetched into models/ so the finished app transcribes offline.

PyInstaller is invoked in-process with PyInstaller.__main__.run().
Flags: --onedir --noconsole --collect-all streamlit/mediapipe/faster_whisper/
cv2/googleapiclient (plus other runtime packages) and --add-data for frontend/,
models/ and bin/.

On-screen text reading uses RapidOCR (the PP-OCR models of PaddleOCR running on
ONNX Runtime, ~16 MB, CPU only). It is bundled by default. PaddleOCR itself is
supported when installed, and --with-paddleocr adds it to the bundle (it adds
~540 MB and replaces opencv-python with opencv-contrib-python).

First boot never crashes on missing Google files: token.json and credentials.json
are optional. Without them the app still runs and keeps clips on the computer;
see README.txt in the output folder.

SECURITY: if token.json exists it is copied next to the exe so uploads work with
no sign-in. Anyone holding the folder can then use your Drive access. Use
--no-token to leave it out, and never publish a build that contains it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import build  # shared constants, preflight scan and helpers

ROOT = Path(__file__).resolve().parent
ZIP_NAME = "CinePulseAI-ViMax-Portable.zip"
COMMIT_MESSAGE = ("feat: integrate ViMax multi-agent engine, Novel2Video, Script2Video, graph consistency, "
                  "and automated build package")
COMMIT_TRAILERS = (
    "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>",
    "Claude-Session: https://claude.ai/code/session_01DZZzH74mQWkTiSWum5df7Q",
)
SECRET_FILES = {"token.json", "credentials.json"}  # must never reach a public repository
NEVER_COMMIT_DIRS = ("dist/", "build/")

FFMPEG_URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]
BLAZEFACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


# --------------------------------------------------------------------------- #
# automatic placement of binaries / models
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "CinePulseAI-build"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def extract_ffmpeg_from_zip(zip_path: Path, dest: Path) -> bool:
    """Pull bin/ffmpeg.exe out of a standard Windows FFmpeg zip. True on success."""
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.replace("\\", "/").lower().endswith("/bin/ffmpeg.exe"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                return dest.read_bytes()[:2] == b"MZ"  # real Windows executable?
    return False


def ensure_ffmpeg(root: Path = ROOT, allow_download: bool = True) -> bool:
    """Make sure bin/ffmpeg.exe exists. Order: already there -> a local ffmpeg*.zip
    in the project folder -> download. Returns True when a valid file is in place."""
    target = root / build.FFMPEG_REL
    if target.is_file() and target.read_bytes()[:2] == b"MZ":
        return True
    for local_zip in sorted(root.glob("ffmpeg*.zip")):
        if extract_ffmpeg_from_zip(local_zip, target):
            print(f"Placed ffmpeg.exe from {local_zip.name}")
            return True
    if not allow_download:
        return False
    for url in FFMPEG_URLS:
        tmp = root / "build" / "ffmpeg_download.zip"
        try:
            print(f"Downloading ffmpeg from {url} ...")
            download(url, tmp)
            if extract_ffmpeg_from_zip(tmp, target):
                print("Placed bin/ffmpeg.exe")
                return True
        except Exception as exc:  # try the next mirror
            print(f"  failed: {exc}", file=sys.stderr)
        finally:
            tmp.unlink(missing_ok=True)
    return False


def ensure_blazeface(root: Path = ROOT, allow_download: bool = True) -> bool:
    target = root / build.BLAZEFACE_REL
    if target.is_file():
        return True
    if not allow_download:
        return False
    try:
        print("Downloading the BlazeFace model ...")
        download(BLAZEFACE_URL, target)
        return target.is_file() and target.stat().st_size > 10_000
    except Exception as exc:
        print(f"  failed: {exc}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# PyInstaller
# --------------------------------------------------------------------------- #
def pyinstaller_args(root: Path = ROOT, with_paddleocr: bool = False, with_torch: bool = False) -> List[str]:
    """build.py's arguments plus --add-data for models/ and bin/."""
    args = build.build_pyinstaller_args(root, with_paddleocr=with_paddleocr, with_torch=with_torch)
    sep = build.os.pathsep
    extra: List[str] = []
    for folder in ("models", "bin"):
        if (root / folder).is_dir():
            extra += ["--add-data", f"{root / folder}{sep}{folder}"]
    return args[:-1] + extra + [args[-1]]  # the script path must stay last


def run_pyinstaller_inprocess(args: List[str]) -> int:
    """Invoke PyInstaller.__main__.run(); returns an exit code instead of raising."""
    import PyInstaller.__main__ as pyi_main

    try:
        pyi_main.run(args)
    except SystemExit as exc:
        return int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
    return 0


# --------------------------------------------------------------------------- #
# finishing the dist folder
# --------------------------------------------------------------------------- #
def locate(app_dir: Path, rel: Path) -> Optional[Path]:
    """PyInstaller 6 puts --add-data files in _internal/; older versions beside the exe."""
    for base in (app_dir, app_dir / "_internal"):
        if (base / rel).exists():
            return base / rel
    return None


def finalize(app_dir: Path, root: Path = ROOT, include_token: bool = True) -> List[str]:
    """Add the files that must sit NEXT TO the exe (token.json, README) - the
    embedded bin/ and models/ are already inside the bundle."""
    copied: List[str] = []
    token = root / build.TOKEN_REL
    if include_token and token.is_file():
        shutil.copy2(token, app_dir / build.TOKEN_REL)
        copied.append("token.json")
    (app_dir / "README.txt").write_text(
        "CinePulse AI\n===========\n\n"
        f"Double-click {build.exe_name()}. Your browser opens the app at http://localhost:8501.\n"
        "There is no console window: use 'Quit CinePulse AI' in the sidebar to stop it.\n"
        "If nothing opens, read cinepulseai.log in this folder.\n\n"
        "Works offline: transcription, highlight scoring, face-tracked cropping, AI avatar lip-sync,\n"
        "story videos from an idea, a novel or a screenplay (Idea2Video / Novel2Video / Script2Video, CPU only),\n"
        "3D avatars (.glb / .vrm: phoneme lip-sync, head + upper-body gestures) shown in a podcast corner frame,\n"
        "merging, and reading on-screen text (OCR). Needs internet: YouTube downloads, edge-tts voices, Google Drive backup.\n\n"
        "GOOGLE DRIVE AUTO-BACKUP (optional)\n"
        "  The app runs fine without it - everything is saved in the 'outputs' folder here.\n"
        "  With a Folder ID set, every finished clip, avatar video and story episode (plus its project files) is\n"
        "  uploaded automatically in the\n"
        "  background (temporary network errors are retried). If Drive fails, your files stay on this computer.\n"
        "  token.json       -> put it in THIS folder (next to CinePulseAI.exe) for uploads with no sign-in.\n"
        "  credentials.json -> alternatively put your OAuth client file here; the first upload\n"
        "                      then opens a Google sign-in page and creates token.json.\n"
        "  Then type your Drive Folder ID in the sidebar.\n"
        + ("\nThis folder already contains a token.json: keep it private.\n"
           if (app_dir / build.TOKEN_REL).is_file() else ""),
        encoding="utf-8",
    )
    return copied


def verify(app_dir: Path, include_token: bool) -> List[str]:
    problems: List[str] = []
    if not (app_dir / build.exe_name()).is_file():
        problems.append(f"{build.exe_name()} was not produced")
    for rel, why in ((build.FFMPEG_REL, "bin/ffmpeg.exe"),
                     (build.BLAZEFACE_REL, "models/blaze_face_short_range.tflite"),
                     (build.WHISPER_REL / "model.bin", "models/faster-whisper-base/model.bin")):
        if locate(app_dir, rel) is None:
            problems.append(f"{why} is not in the distribution")
    if include_token and not (app_dir / build.TOKEN_REL).is_file():
        problems.append("token.json is not in the distribution (fine if you never created one)")
    return problems


def validate_zip(zip_path: Path, top: Optional[str] = None) -> List[str]:
    """Open the FINAL archive and check that what the user will unzip is complete.
    Returns a list of problems (empty = good)."""
    import zipfile

    top = top or build.APP_NAME
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        return [f"{zip_path} does not exist"]
    problems: List[str] = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        return [f"not a valid zip file: {exc}"]
    with zf:
        bad = zf.testzip()
        if bad:
            problems.append(f"corrupt entry in the zip: {bad}")
        names = set(zf.namelist())

        def has(*options: str) -> bool:
            return any(f"{top}/{o}" in names for o in options)

        required = [
            (build.exe_name(),), (f"_internal/{build.FFMPEG_REL.as_posix()}", build.FFMPEG_REL.as_posix()),
            (f"_internal/{build.BLAZEFACE_REL.as_posix()}", build.BLAZEFACE_REL.as_posix()),
            (f"_internal/{(build.WHISPER_REL / 'model.bin').as_posix()}", (build.WHISPER_REL / "model.bin").as_posix()),
            ("_internal/frontend/app.py", "frontend/app.py"), ("README.txt",),
        ]
        onnx = [n for n in names if n.startswith(f"{top}/") and n.lower().endswith(".onnx") and "rapidocr" in n.lower()]
        if len(onnx) < 2:
            problems.append("the OCR models (rapidocr_onnxruntime/models/*.onnx) are missing from the zip")
        for options in required:
            if not has(*options):
                problems.append(f"missing from the zip: {top}/{options[0]}")
        app_py = next((f"{top}/{o}" for o in ("_internal/frontend/app.py", "frontend/app.py") if f"{top}/{o}" in names), None)
        if app_py:
            text = zf.read(app_py).decode("utf-8", "replace")
            for needle in ("CinePulse AI", "Clip Repurposer Studio", "AI Generator Lab",
                           "Enable AI Avatar Lip-Sync Commentary", "Your own script", "Visual actions",
                           "Model type", "Avatar position", "Text reader (OCR)", "Animated backdrop",
                           "Creation mode", "Idea2Video", "Novel2Video", "Script2Video",
                           "Character consistency", "Camera angle preferences", "Podcast corner avatar overlay"):
                if needle not in text:
                    problems.append(f"frontend/app.py in the zip lacks '{needle}'")
        if not any(n.startswith(f"{top}/") for n in names):
            problems.append(f"the zip has no top-level '{top}' folder")
    return problems


# --------------------------------------------------------------------------- #
# git: add / commit / push
# --------------------------------------------------------------------------- #
def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {(proc.stderr or proc.stdout).strip()}")
    return proc


def git_publish(root: Path = ROOT, message: str = COMMIT_MESSAGE, trailers=COMMIT_TRAILERS,
                remote: str = "origin", branch: str = "main", push: bool = True) -> dict:
    """`git add .`, `git commit -m <message>`, `git push <remote> <branch>` through subprocess.

    Safety: refuses (before anything is staged) when a token/credential file or a build folder
    is not git-ignored, and un-stages and refuses if one still slipped in. Nothing to commit is
    not an error - the push still runs so earlier local commits are published too."""
    root = Path(root)
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or not on PATH")
    if _git(root, "rev-parse", "--is-inside-work-tree", check=False).stdout.strip() != "true":
        raise RuntimeError(f"{root} is not a git repository")

    def risky(paths: List[str]) -> List[str]:
        return [p for p in paths if Path(p).name in SECRET_FILES or p.replace("\\", "/").startswith(NEVER_COMMIT_DIRS)]

    pending = _git(root, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    tracked = _git(root, "ls-files").stdout.splitlines()
    bad = risky(pending + tracked)
    if bad:
        raise RuntimeError("refusing to commit secrets/build output (add them to .gitignore): " + ", ".join(bad))
    _git(root, "add", ".")
    staged = _git(root, "diff", "--cached", "--name-only").stdout.splitlines()
    bad = risky(staged)
    if bad:  # defensive: add -f style leaks
        _git(root, "reset", "-q")
        raise RuntimeError("refusing to commit secrets/build output: " + ", ".join(bad))

    committed = bool(staged)
    if committed:
        cmd = ["commit", "-m", message]
        if trailers:
            cmd += ["-m", "\n".join(trailers)]
        _git(root, *cmd)
    pushed = False
    if push:
        _git(root, "push", remote, branch)
        pushed = True
    head = _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    return {"committed": committed, "pushed": pushed, "files": staged, "head": head}


def make_zip(app_dir: Path, zip_path: Path) -> Path:
    zip_path.unlink(missing_ok=True)
    made = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=app_dir.parent, base_dir=app_dir.name)
    return Path(made)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the CinePulse AI distribution")
    ap.add_argument("--dry-run", action="store_true", help="scan and print the arguments only")
    ap.add_argument("--ci", action="store_true", help="unattended build (implies downloads allowed)")
    ap.add_argument("--no-token", action="store_true", help="do not bundle token.json")
    ap.add_argument("--require-token", action="store_true", help="fail if token.json is missing")
    ap.add_argument("--no-download", action="store_true", help="never download ffmpeg / models")
    ap.add_argument("--skip-model-download", action="store_true", help="do not pre-fetch the Whisper model")
    ap.add_argument("--allow-non-windows", action="store_true", help="permit a test build off Windows")
    ap.add_argument("--no-zip", action="store_true", help="do not create the final zip")
    ap.add_argument("--with-paddleocr", action="store_true", help="also bundle PaddleOCR (+~540 MB; default is RapidOCR)")
    ap.add_argument("--ultimate", action="store_true", help="bundle PaddleOCR AND PyTorch (equivalent to --with-paddleocr --with-torch; very large)")
    ap.add_argument("--with-torch", action="store_true", help="also bundle PyTorch (+1-2 GB; the 3-D avatar renderer does not need it)")
    ap.add_argument("--git-push", action="store_true", help="after a successful build: git add ., commit, push origin main")
    ap.add_argument("--git-only", action="store_true", help="only run the git add/commit/push step")
    ap.add_argument("--validate-zip", metavar="ZIP", help="only check an existing final zip, then exit")
    a = ap.parse_args(argv)
    if a.validate_zip:
        issues = validate_zip(Path(a.validate_zip))
        for i in issues:
            print(f"PROBLEM: {i}", file=sys.stderr)
        print("Zip validation passed." if not issues else f"Zip validation FAILED ({len(issues)} problem(s)).")
        return 1 if issues else 0
    if a.git_only:
        return _run_git_step()
    include_token = not a.no_token
    # A plain --dry-run changes nothing; a CI dry-run may fetch ffmpeg/models like the real build.
    can_download = not a.no_download and (not a.dry_run or a.ci)

    ensure_ffmpeg(ROOT, can_download)
    ensure_blazeface(ROOT, can_download)

    report = build.preflight(ROOT, include_token, a.require_token, False, a.allow_non_windows)
    print(report.render())
    args = pyinstaller_args(ROOT, a.with_paddleocr or a.ultimate, a.with_torch or a.ultimate)
    if a.dry_run:
        print("\nPyInstaller.__main__.run([\n  " + ",\n  ".join(repr(x) for x in args) + "\n])")
        return 0 if report.ok else 1
    if not report.ok:
        print("\nBuild stopped. Fix the errors above and run build_exe.py again.", file=sys.stderr)
        return 1

    if not a.skip_model_download and not a.no_download:
        build.fetch_whisper_model(ROOT)

    code = run_pyinstaller_inprocess(args)
    if code != 0:
        print(f"PyInstaller failed with exit code {code}.", file=sys.stderr)
        return code

    app_dir = ROOT / "dist" / build.APP_NAME
    finalize(app_dir, ROOT, include_token)
    for p in verify(app_dir, include_token):
        print(f"WARNING: {p}")
    print(f"\nBuilt: {app_dir / build.exe_name()}")
    if not a.no_zip:
        z = make_zip(app_dir, ROOT / "dist" / ZIP_NAME)
        print(f"Archive: {z} ({z.stat().st_size / 1e6:.0f} MB)")
        issues = validate_zip(z)
        for i in issues:
            print(f"PROBLEM: {i}", file=sys.stderr)
        if issues:
            return 1
        print("Zip validation passed.")
    if include_token and (app_dir / build.TOKEN_REL).is_file():
        print("\n" + build.TOKEN_WARNING)
    if a.git_push:
        return _run_git_step()
    return 0


def _run_git_step() -> int:
    try:
        r = git_publish(ROOT)
    except RuntimeError as exc:
        print(f"GIT STEP FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"git: {'committed ' + str(len(r['files'])) + ' file(s)' if r['committed'] else 'nothing new to commit'}; "
          f"pushed to origin/main at {r['head']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
