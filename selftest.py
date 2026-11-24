"""
selftest.py  -  runs INSIDE the packaged exe:   CinePulseAI.exe --selftest

Proves the build actually works, not just that a web server starts: every
bundled library imports, ffmpeg runs, the face model and the Whisper model
load, a real clip is cropped and encoded, and the Streamlit page renders.
Output goes to cinepulseai.log (the exe has no console); exit code 0 = all good.
Network-only features are never exercised here, so the self-test needs no internet.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

Result = Tuple[str, str, str]  # (name, "ok" | "warn" | "fail", detail)

MODULES = [
    "streamlit", "cv2", "numpy", "mediapipe", "mediapipe.tasks.python.vision",
    "faster_whisper", "ctranslate2", "googleapiclient.discovery", "googleapiclient.http",
    "google_auth_oauthlib.flow", "google.oauth2.credentials", "yt_dlp",
    "pipeline", "uploader", "scorer", "transcriber", "video_engine", "ocr_engine", "rapidocr_onnxruntime",
    "subtitle_engine", "music_finder", "social_metadata", "meme_matcher", "story_recap", "upload_limits",
]


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def check_imports() -> str:
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want every failure listed
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
    if failed:
        raise RuntimeError("cannot import: " + "; ".join(failed))
    return f"{len(MODULES)} modules"


def check_ffmpeg() -> str:
    import video_engine

    ff = video_engine.find_ffmpeg()
    out = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=60, creationflags=_flags())
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {out.returncode}: {out.stderr[-200:]}")
    return f"{ff} -> {out.stdout.splitlines()[0][:60]}"


def check_scorer() -> str:
    import scorer

    words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3}
             for i, w in enumerate("Did you know this secret is insane?".split())]
    clips = scorer.find_viral_clips(words, min_len=1, max_len=30)
    if not clips:
        raise RuntimeError("no clip returned")
    return f"{len(clips)} clip(s)"


def check_face_model() -> str:
    import numpy as np
    import video_engine

    det = video_engine.create_face_detector()
    try:
        det(np.zeros((360, 640, 3), np.uint8))
    finally:
        det.close()
    return "BlazeFace loaded and ran"


def _make_wav(path: str) -> None:
    import video_engine

    ff = video_engine.find_ffmpeg()
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", path],
                   check=True, timeout=60, creationflags=_flags())


def check_whisper() -> str:
    import transcriber

    bundled = transcriber._bundled_model_path()
    if getattr(sys, "frozen", False) and bundled is None:
        raise RuntimeError("the packaged app has no bundled Whisper model (models/faster-whisper-base)")
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "t.wav")
        _make_wav(wav)
        words = transcriber.transcribe_audio(wav)
    return f"model={bundled or 'download-on-demand'}, words in test tone={len(words)}"


def check_clip_pipeline() -> str:
    import cv2
    import numpy as np
    import video_engine

    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "src.mp4"), os.path.join(d, "out.mp4")
        silent = os.path.join(d, "silent.mp4")
        wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
        for i in range(60):
            f = np.full((360, 640, 3), 30, np.uint8)
            f[150:210, 100 + i * 6 : 160 + i * 6] = 255
            wr.write(f)
        wr.release()
        ff = video_engine.find_ffmpeg()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i", "sine=duration=2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
                       check=True, timeout=120, creationflags=_flags())
        res = video_engine.create_vertical_short(src, out, upload=False)  # real BlazeFace + real ffmpeg
        size = os.path.getsize(out)
        if size < 1000:
            raise RuntimeError("output clip is empty")
    return f"{res['frames']} frames cropped to 9:16, {size // 1024} KB encoded"





def check_subtitles() -> str:
    """Chunk words into 3-5 word cues, write .srt/.ass and burn them into a real clip with the bundled ffmpeg."""
    import subtitle_engine as se
    import video_engine

    words = [{"word": w, "start": i * 0.35, "end": i * 0.35 + 0.3}
             for i, w in enumerate("Here is the truth about building muscle fast today".split())]
    cues = se.cues_for_clip(words, 0.0, 4.0)
    if not cues or any(not 1 <= len(c["text"].split()) <= 5 for c in cues):
        raise RuntimeError(f"bad cues: {cues}")
    ff = video_engine.find_ffmpeg()
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "s.mp4"), os.path.join(d, "o.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=15:duration=3",
                        "-pix_fmt", "yuv420p", src], check=True, timeout=60, creationflags=_flags())
        files = se.write_subtitle_files(cues, os.path.join(d, "subs"), size=(270, 480))
        video_engine.burn_subtitles(src, files["ass"], out, ffmpeg_path=ff)
        if os.path.getsize(out) < 1000:
            raise RuntimeError("burned clip is empty")
    return f"{len(cues)} cues, .srt + .ass written and burned in"


def check_music() -> str:
    """Mix a cached tone under a clip with ducking (no network); then confirm yt-dlp's search API is importable."""
    import yt_dlp
    import music_finder
    import video_engine

    ff = video_engine.find_ffmpeg()
    with tempfile.TemporaryDirectory() as d:
        src, bgm, out = (os.path.join(d, n) for n in ("s.mp4", "m.mp3", "o.mp4"))
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=270x480:rate=15:duration=3",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest", "-pix_fmt", "yuv420p", src],
                       check=True, timeout=60, creationflags=_flags())
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=500:duration=1", bgm],
                       check=True, timeout=60, creationflags=_flags())
        video_engine.mix_background_music(src, bgm, out, music_volume=0.15, duck=True, ffmpeg_path=ff)
        if os.path.getsize(out) < 1000:
            raise RuntimeError("mixed clip is empty")
    return f"BGM mixed with ducking; yt-dlp {yt_dlp.version.__version__}; cache={music_finder.DEFAULT_CACHE}"


def check_recap_and_uploads() -> str:
    """Chunked 1 GB-safe upload copy + upload-limit helpers + recap ordering (no encoding needed)."""
    import io
    import tempfile

    import story_recap
    import upload_limits
    import uploader

    if upload_limits.normalize(1024) != 1024 or upload_limits.normalize(500) != 500:
        raise RuntimeError("upload limit choices are wrong")
    with tempfile.TemporaryDirectory() as d:
        class _Up:
            name = "t.mp4"

            def __init__(self):
                self._b = io.BytesIO(b"x" * (9 * 1024 * 1024))
                self.closed = False

            def read(self, n=-1):
                return self._b.read(n)

            def seek(self, *a):
                return self._b.seek(*a)

            def close(self):
                self.closed = True

        up = _Up()
        dest = uploader.save_uploaded_video_streamed(up, str(__import__("pathlib").Path(d) / "t.mp4"))
        import os
        if os.path.getsize(dest) != 9 * 1024 * 1024 or not up.closed:
            raise RuntimeError("chunked upload copy did not finish cleanly")
        if [p for p in os.listdir(d) if p.endswith(".part")]:
            raise RuntimeError("partial file left behind")
    with tempfile.TemporaryDirectory() as d2:
        import pathlib

        a, b = pathlib.Path(d2) / "a.mp4", pathlib.Path(d2) / "b.mp4"
        a.write_bytes(b"1")
        b.write_bytes(b"2")
        order = story_recap.order_clips([{"start": 30, "output_path": str(b)}, {"start": 5, "output_path": str(a)}])
        if [pathlib.Path(c["output_path"]).name for c in order] != ["a.mp4", "b.mp4"]:
            raise RuntimeError("recap clips are not in source order")
    return "9 MB streamed to disk in chunks; limits 500/1024 MB; recap order OK"


def check_social() -> str:
    import social_metadata as sm

    p = sm.generate_packages("You need to eat more protein every single day. Protein and sleep build muscle.", use_llm="off")
    yt, ig, fb = p["youtube"], p["instagram"], p["facebook"]
    if len(yt["title"]) >= 60 or "#Shorts" not in yt["description"] + " ".join(yt.get("tags", [])):
        raise RuntimeError(f"bad YouTube package: {yt}")
    if not 15 <= len(ig["hashtags"]) <= 20 or not 5 <= len(fb["hashtags"]) <= 8:
        raise RuntimeError("hashtag counts out of range")
    return f"YT title {len(yt['title'])} chars, {len(ig['hashtags'])} IG tags, {len(fb['hashtags'])} FB tags"


def check_memes() -> str:
    """Auto-meme matcher on the CPU: starter pack -> emotion match -> splice into a real video, all offline."""
    import cv2
    import numpy as np
    import meme_matcher
    import video_engine

    with tempfile.TemporaryDirectory() as d:
        ff = video_engine.find_ffmpeg()
        meme_dir = os.path.join(d, "memes")
        made = meme_matcher.ensure_starter_pack(meme_dir, ffmpeg_path=ff)
        lib = meme_matcher.MemeLibrary(meme_dir, ffmpeg_path=ff).scan()
        if len(lib.clips) < 5:
            raise RuntimeError(f"starter pack is incomplete ({len(lib.clips)} clips)")
        base = os.path.join(d, "base.mp4")
        vw = cv2.VideoWriter(base, cv2.VideoWriter_fourcc(*"mp4v"), 10, (270, 480))
        for _ in range(40):
            vw.write(np.full((480, 270, 3), 60, np.uint8))
        vw.release()
        cfg = meme_matcher.MemeSettings(enabled=True, mode="overlay", min_score=0.2, max_per_minute=20)
        plan = meme_matcher.plan_memes([{"start": 0.2, "end": 2.5, "text": "LOL that was hilarious!!"}], lib, cfg, 4.0,
                                       frame_size=(270, 480))
        if not plan["matches"] or plan["matches"][0]["emotion"] != "laugh":
            raise RuntimeError(f"emotion matching failed: {plan}")
        out = os.path.join(d, "out.mp4")
        res = video_engine.splice_memes(base, out, plan["matches"], ffmpeg_path=ff)
        if res["memes_applied"] != 1 or os.path.getsize(out) < 1000:
            raise RuntimeError(f"splice failed: {res}")
    return f"{len(lib.clips)} starter clips, laugh matched, reaction spliced"





def check_ocr() -> str:
    """Read burned-in text from a rendered video with the real OCR engine, then unload it again."""
    import cv2
    import numpy as np
    import ocr_engine

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
        for _ in range(30):
            f = np.full((360, 640, 3), 255, np.uint8)
            cv2.putText(f, "SALE ENDS TODAY", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)
            vw.write(f)
        vw.release()
        try:
            segs = ocr_engine.extract_screen_text(path, interval=1.0)
        finally:
            ocr_engine.unload()
    text = " ".join(s["text"] for s in segs).upper().replace(" ", "")
    if "SALE" not in text or "TODAY" not in text:
        raise RuntimeError(f"OCR did not read the text: {segs}")
    return f"read {len(segs)} line(s): {segs[0]['text']!r}"



def check_app_page(app_path: Optional[Path]) -> str:
    if app_path is None or not Path(app_path).is_file():
        raise RuntimeError(f"app file not found: {app_path}")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app_path), default_timeout=90).run()
    if at.exception:
        raise RuntimeError("; ".join(str(e.value) for e in at.exception))
    keys = {w.key for w in list(at.text_input) + list(at.checkbox) + list(at.button)}
    for need in ("vugola_url", "subs_on", "music_on", "drive_on", "music_scan", "vugola_run"):
        if need not in keys:
            raise RuntimeError(f"Viral Clip Studio is missing the control {need}")
    return "Viral Clip Studio renders with subtitle, music and Drive controls"



def run_all(app_path: Optional[Path] = None, log: Callable[[str], None] = print) -> List[Result]:
    checks = [
        ("imports", "fail", check_imports),
        ("ffmpeg", "fail", check_ffmpeg),
        ("scorer", "fail", check_scorer),
        ("face model", "fail", check_face_model),
        ("whisper model", "fail", check_whisper),
        ("clip pipeline", "fail", check_clip_pipeline),
        ("burned-in subtitles (.srt/.ass)", "fail", check_subtitles),
        ("background music mixer", "fail", check_music),
        ("YT/IG/FB captions & tags", "fail", check_social),
        ("summary compilation + chunked uploads", "fail", check_recap_and_uploads),
        ("auto-meme reactions", "fail", check_memes),
        ("on-screen text (OCR)", "fail", check_ocr),
        ("app page", "fail", lambda: check_app_page(app_path)),
    ]
    results: List[Result] = []
    for name, severity, fn in checks:
        try:
            detail = fn()
            results.append((name, "ok", detail))
        except Exception as exc:  # noqa: BLE001
            results.append((name, severity, f"{type(exc).__name__}: {exc}"))
            if severity == "fail":
                log(traceback.format_exc())
        status = results[-1][1].upper()
        log(f"[{status}] {name} - {results[-1][2]}")
    return results


def main(app_path: Optional[Path] = None) -> int:
    print("CinePulse AI self-test")
    results = run_all(app_path)
    failed = [r for r in results if r[1] == "fail"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (" - SELFTEST FAILED" if failed else " - SELFTEST OK"))
    sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
