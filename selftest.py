"""
selftest.py  -  runs INSIDE the packaged exe:   CinePulseAI.exe --selftest

Proves the build actually works, not just that a web server starts: every
bundled library imports, ffmpeg runs, the face model and the Whisper model
load, a real clip is cropped and encoded, and the Streamlit page renders.
Output goes to cinepulseai.log (the exe has no console); exit code 0 = all good.
Online-only features (edge-tts) are reported as warnings, never failures.
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
    "google_auth_oauthlib.flow", "google.oauth2.credentials", "yt_dlp", "edge_tts",
    "pipeline", "uploader", "scorer", "transcriber", "tts_engine", "video_engine", "avatar_sync",
    "ocr_engine", "visual_cues", "rapidocr_onnxruntime",
    "avatar_3d_engine", "glb_model", "soft_raster", "lipsync_en", "avatar_gestures", "demo_avatars", "render_engine",
    "vimax_orchestrator", "novel_adapter", "script_parser", "dependency_graph", "meme_matcher",
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
    return f"{len(clips)} clip(s); script words={scorer.generate_script('script about tea')['word_count']}"


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


def check_avatar() -> str:
    """Render a 2 s talking avatar with the real BlazeFace model, ffmpeg and OpenCV."""
    import cv2
    import numpy as np
    import avatar_sync
    import video_engine

    with tempfile.TemporaryDirectory() as d:
        img, wav, out = (os.path.join(d, n) for n in ("a.png", "a.wav", "a.mp4"))
        pic = np.full((640, 480, 3), (120, 150, 190), np.uint8)
        cv2.ellipse(pic, (240, 300), (140, 190), 0, 0, 360, (150, 190, 235), -1)
        cv2.imwrite(img, pic)
        subprocess.run([video_engine.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "sine=frequency=200:duration=2", wav], check=True, timeout=60, creationflags=_flags())
        res = avatar_sync.create_avatar_video(img, wav, out, size=(360, 640))
        size = os.path.getsize(out)
        if size < 1000 or res["frames"] < 40:
            raise RuntimeError(f"avatar video is empty or short: {res}")
    return f"{res['frames']} frames, {res['mode']} animation, {size // 1024} KB"


def check_avatar_3d() -> str:
    """Render a talking built-in 3-D avatar (phoneme lip-sync + gestures), then place it in a podcast corner frame."""
    import avatar_3d_engine
    import demo_avatars
    import video_engine

    with tempfile.TemporaryDirectory() as d:
        wav, av, base, out = (os.path.join(d, n) for n in ("a.wav", "av.mp4", "base.mp4", "out.mp4"))
        ff = video_engine.find_ffmpeg()
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1.5", wav],
                       check=True, timeout=60, creationflags=_flags())
        model = demo_avatars.ensure_demo_avatars(d)["human"]
        res = avatar_3d_engine.create_3d_avatar_video(model, wav, av, text="Hello there, welcome to the show.",
                                                      size=(240, 320), fps=20, quality="fast", ffmpeg_path=ff)
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=navy:s=360x640:r=20:d=1.5",
                        "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", base],
                       check=True, timeout=60, creationflags=_flags())
        placed = video_engine.place_podcast_avatar(base, av, "Bottom-Right", None, 0.4, out, ffmpeg_path=ff)
        size = os.path.getsize(out)
        if size < 1000 or res["frames"] < 25:
            raise RuntimeError(f"3-D avatar video is empty or short: {res}")
    return f"{res['frames']} frames, {res['lip_sync']} lip-sync, placed at {placed['rect'][:2]}"


def check_render_engine() -> str:
    """Compose a video from tracks: voiceover + animated backdrop + timed on-screen text (render_engine)."""
    import render_engine
    import video_engine

    with tempfile.TemporaryDirectory() as d:
        wav, out = os.path.join(d, "v.wav"), os.path.join(d, "o.mp4")
        ff = video_engine.find_ffmpeg()
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1.5", wav],
                       check=True, timeout=60, creationflags=_flags())
        res = render_engine.render_commentary(wav, out, None, None,
                                              screen_segments=[{"text": "SELF TEST", "start": 0.2, "end": 1.2}],
                                              size=(270, 480), fps=20, ffmpeg_path=ff)
        if os.path.getsize(out) < 1000 or res["frames"] < 25:
            raise RuntimeError(f"composed video is empty or short: {res}")
    return f"{res['frames']} frames, tracks: {sorted({t['track'] for t in res['tracks']})}"


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


def check_story() -> str:
    """Story Studio on the CPU: screenplay -> agents -> consistency graph -> 2-shot animatic (no network, no GPU)."""
    import vimax_orchestrator as vo

    script = "INT. KITCHEN - NIGHT\n\nBOB\nHello there.\n\nAMY\nHi Bob.\n"
    r = vo.ViMaxOrchestrator(vo.OfflineClient()).run(script, mode="script2video")
    errors = [i for i in r["issues"] if i["severity"] == "error"]
    if errors or len(r["storyboard"]["shots"]) < 3:
        raise RuntimeError(f"storyboard check failed: {errors or 'too few shots'}")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "s.mp4")
        res = vo.render_storyboard(r, out, work_dir=os.path.join(d, "w"), size=(270, 480), fps=8, max_shots=3)
        if os.path.getsize(out) < 1000 or not res["drift"]:
            raise RuntimeError(f"animatic is empty: {res}")
        bad = [x for x in res["drift"] if x["status"] != "ok"]
        if bad:
            raise RuntimeError(f"character appearance drifted: {bad}")
    return f"{len(r['storyboard']['shots'])} shots, {len(r['graph']['edges'])} graph edges, {len(res['drift'])} appearances consistent"


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


def check_cues() -> str:
    """Apply a real zoom + flash to a clip (decode, warp, re-encode, audio kept)."""
    import cv2
    import numpy as np
    import video_engine
    import visual_cues

    ff = video_engine.find_ffmpeg()
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "s.mp4"), os.path.join(d, "o.mp4")
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=2",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-shortest", "-pix_fmt", "yuv420p", src],
                       check=True, timeout=60, creationflags=_flags())
        res = visual_cues.apply_cues_to_video(src, [{"effect": "zoom_in", "start": 0.0, "end": 1.0},
                                                    {"effect": "flash", "start": 1.0, "end": 1.5}], out, ffmpeg_path=ff)
        cap = cv2.VideoCapture(out)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 16)
        ok, frame = cap.read()
        cap.release()
        if res["cues_applied"] != 2 or not ok or float(np.mean(frame)) < 140:
            raise RuntimeError(f"cues were not rendered: {res}")
    return f"{res['cues_applied']} cues over {res['frames']} frames"


def check_app_page(app_path: Optional[Path]) -> str:
    if app_path is None or not Path(app_path).is_file():
        raise RuntimeError(f"app file not found: {app_path}")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app_path), default_timeout=90).run()
    if at.exception:
        raise RuntimeError("; ".join(str(e.value) for e in at.exception))
    box = at.selectbox(key="lab_mode")
    modes = list(box.options)
    if modes != ["Idea2Video", "Novel2Video", "Script2Video", "Clip Repurposer"]:
        raise RuntimeError(f"unexpected creation modes: {modes}")
    expect = {"Idea2Video": "story_text_idea2video", "Novel2Video": "story_text_novel2video",
              "Script2Video": "story_text_script2video", "Clip Repurposer": "vugola_url"}
    for mode in modes:  # every mode must draw without errors and show its own main field
        at.selectbox(key="lab_mode").select(mode).run()
        if at.exception:
            raise RuntimeError(f"{mode}: " + "; ".join(str(e.value) for e in at.exception))
        keys = {w.key for w in list(at.text_area) + list(at.text_input)}
        if expect[mode] not in keys:
            raise RuntimeError(f"{mode}: missing the field {expect[mode]}")
    at.selectbox(key="lab_mode").select("Idea2Video").run()
    at.selectbox(key="idea_flavor").select("Narrated video with a 3D avatar (classic)").run()
    if at.exception or "viewmax_prompt" not in {w.key for w in at.text_area}:
        raise RuntimeError("the classic narrated generator did not draw")
    return f"all {len(modes)} creation modes (+ classic generator) render"


def check_tts_online() -> str:
    import asyncio
    import edge_tts

    async def go(path):
        await edge_tts.Communicate("Self test.", "en-US-AriaNeural").save(path)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mp3")
        asyncio.run(asyncio.wait_for(go(p), timeout=40))
        return f"voice generated ({os.path.getsize(p)} bytes)"


def run_all(app_path: Optional[Path] = None, log: Callable[[str], None] = print) -> List[Result]:
    checks = [
        ("imports", "fail", check_imports),
        ("ffmpeg", "fail", check_ffmpeg),
        ("scorer", "fail", check_scorer),
        ("face model", "fail", check_face_model),
        ("whisper model", "fail", check_whisper),
        ("clip pipeline", "fail", check_clip_pipeline),
        ("avatar lip-sync", "fail", check_avatar),
        ("3D avatar + podcast layout", "fail", check_avatar_3d),
        ("multi-track render engine", "fail", check_render_engine),
        ("story studio (agents + consistency graph)", "fail", check_story),
        ("auto-meme reactions", "fail", check_memes),
        ("on-screen text (OCR)", "fail", check_ocr),
        ("visual action cues", "fail", check_cues),
        ("app page", "fail", lambda: check_app_page(app_path)),
        ("edge-tts (online)", "warn", check_tts_online),
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
