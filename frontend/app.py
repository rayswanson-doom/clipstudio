"""
frontend/app.py  -  CinePulse AI  (Streamlit UI)

Run:  streamlit run frontend/app.py          (or double-click the packaged exe)

  🎬 Clip Repurposer Studio : YouTube URL / file -> local transcription -> local
       highlight scoring -> MediaPipe face-tracked 9:16 crop -> Google Drive backup
  ⚡ AI Generator Lab        : text prompt -> script -> local edge-tts voiceover
       -> video merge -> Google Drive upload
  Sidebar (collapsible): Google Drive Folder ID + connection status.

Design: dark glassmorphism, violet->cyan->pink accent gradients, card containers
for outputs, metric badges. Responsive down to phone width. All styling is plain
CSS injected below; no external fonts or network assets, so it works offline.

Add ?demo=1 to the address to preview the layout with sample results.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import streamlit as st  # noqa: E402

import pipeline  # noqa: E402
import uploader  # noqa: E402

APP_NAME = "CinePulse AI"
TAB_CLIP = "🎬 Clip Repurposer Studio"
TAB_LAB = "⚡ AI Generator Lab"


def _output_dir() -> Path:
    """outputs/ next to the exe when packaged (or the temp dir if that folder is
    read-only, e.g. under Program Files); the project folder otherwise."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
    target = base / "outputs"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "CinePulseAI" / "outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


OUTPUT_DIR = _output_dir()
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cinepulse_uploads"
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
AVATAR_TYPES = ["glb", "vrm", "jpg", "jpeg", "png"]  # 3D models first, pictures too
MODEL_TYPES = {"Photorealistic Human": "human", "AI Model": "ai", "Anime": "anime"}
QUALITY_CHOICES = {"Balanced": "balanced", "Fast (low-spec laptops)": "fast", "High (slower, smoother edges)": "high"}
VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural", "en-IN-NeerjaNeural",
]

# --------------------------------------------------------------------------- #
# design system (CSS)
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root{
  --cp-bg:#0b0b14; --cp-card:rgba(255,255,255,.05); --cp-card-2:rgba(255,255,255,.08);
  --cp-border:rgba(255,255,255,.11); --cp-text:#eceef8; --cp-muted:#9aa0b8;
  --cp-a1:#8b5cf6; --cp-a2:#22d3ee; --cp-a3:#f472b6; --cp-ok:#34d399; --cp-warn:#fbbf24;
  --cp-grad:linear-gradient(135deg,var(--cp-a1) 0%,var(--cp-a2) 100%);
  --cp-grad-hot:linear-gradient(135deg,var(--cp-a1) 0%,var(--cp-a3) 100%);
}
.stApp{
  background:
    radial-gradient(1100px 560px at 8% -8%, rgba(139,92,246,.28), transparent 62%),
    radial-gradient(900px 520px at 100% 0%, rgba(34,211,238,.18), transparent 58%),
    radial-gradient(800px 600px at 50% 120%, rgba(244,114,182,.12), transparent 60%),
    var(--cp-bg);
  color:var(--cp-text);
}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1180px;}
h1,h2,h3,h4{letter-spacing:-.01em;}

/* ---------- sidebar (collapsible natively via the chevron) ---------- */
[data-testid="stSidebar"]{
  background:rgba(14,14,28,.72);
  backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
  border-right:1px solid var(--cp-border);
}
.cp-brand{display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.1rem;}
.cp-brand .logo{width:38px;height:38px;border-radius:11px;background:var(--cp-grad);display:grid;place-items:center;font-size:1.25rem;box-shadow:0 6px 22px rgba(139,92,246,.45);}
.cp-brand b{font-size:1.1rem;} .cp-brand small{display:block;color:var(--cp-muted);font-size:.72rem;margin-top:-2px;}
.cp-side-title{color:var(--cp-muted);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;margin:1.1rem 0 .4rem;}

/* ---------- hero ---------- */
.cp-hero{
  display:flex;align-items:center;gap:1.1rem;flex-wrap:wrap;
  padding:1.4rem 1.6rem;margin-bottom:1.3rem;border-radius:22px;
  background:linear-gradient(135deg,rgba(139,92,246,.18),rgba(34,211,238,.08));
  border:1px solid var(--cp-border);
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  box-shadow:0 18px 50px rgba(0,0,0,.35);
}
.cp-hero .logo{width:58px;height:58px;border-radius:17px;background:var(--cp-grad);display:grid;place-items:center;font-size:1.9rem;box-shadow:0 10px 30px rgba(139,92,246,.5);flex:none;}
.cp-hero h1{margin:0;padding:0;font-size:2.15rem;line-height:1.1;font-weight:800;}
.cp-hero h1 span{background:var(--cp-grad-hot);-webkit-background-clip:text;background-clip:text;color:transparent;}
.cp-hero p{margin:.3rem 0 0;color:var(--cp-muted);font-size:.98rem;}
.cp-hero .grow{flex:1 1 260px;min-width:0;}
.cp-pills{display:flex;gap:.45rem;flex-wrap:wrap;}

/* ---------- badges, pills, stat tiles ---------- */
.cp-pill,.cp-badge{
  display:inline-flex;align-items:center;gap:.35rem;padding:.28rem .7rem;border-radius:999px;
  font-size:.78rem;font-weight:600;color:var(--cp-text);
  background:var(--cp-card-2);border:1px solid var(--cp-border);white-space:nowrap;
}
.cp-badge.score{background:var(--cp-grad);border-color:transparent;color:#fff;}
.cp-badge.time{color:var(--cp-a2);} .cp-badge.ok{color:var(--cp-ok);} .cp-badge.warn{color:var(--cp-warn);}
.cp-badge.face{color:var(--cp-a3);}
.cp-dot{width:8px;height:8px;border-radius:50%;background:var(--cp-ok);box-shadow:0 0 10px var(--cp-ok);}
.cp-chips{display:flex;gap:.4rem;flex-wrap:wrap;margin:.55rem 0 .2rem;}
.cp-chip{font-size:.72rem;color:var(--cp-muted);padding:.16rem .55rem;border-radius:8px;background:rgba(255,255,255,.05);border:1px solid var(--cp-border);}
.cp-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:.4rem 0 1.1rem;}
.cp-stat{padding:.9rem 1rem;border-radius:16px;background:var(--cp-card);border:1px solid var(--cp-border);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}
.cp-stat .v{font-size:1.6rem;font-weight:800;line-height:1.1;background:var(--cp-grad);-webkit-background-clip:text;background-clip:text;color:transparent;}
.cp-stat .l{color:var(--cp-muted);font-size:.76rem;margin-top:.15rem;}

/* ---------- section headers + step flow ---------- */
.cp-section h3{margin:.2rem 0 .1rem;font-size:1.35rem;} .cp-section p{margin:0 0 .8rem;color:var(--cp-muted);}
.cp-steps{display:flex;flex-wrap:wrap;gap:.45rem;margin:.2rem 0 1rem;}
.cp-step{font-size:.78rem;padding:.3rem .75rem;border-radius:999px;background:var(--cp-card);border:1px solid var(--cp-border);color:var(--cp-muted);}
.cp-step b{color:var(--cp-a2);margin-right:.3rem;}

/* ---------- tabs ---------- */
[role="tablist"]{
  gap:.4rem;padding:.35rem;border-radius:16px;background:var(--cp-card);
  border:1px solid var(--cp-border);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  flex-wrap:wrap;
}
[role="tablist"] [role="tab"]{height:auto;padding:.6rem 1.15rem;border-radius:12px !important;color:var(--cp-muted);font-weight:600;transition:all .2s ease;}
[role="tablist"] [role="tab"]:hover{color:var(--cp-text);background:var(--cp-card-2);}
[role="tablist"] [role="tab"][aria-selected="true"]{background:var(--cp-grad);color:#fff !important;box-shadow:0 8px 24px rgba(139,92,246,.4);}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"],[role="tablist"] [data-testid="stTabHighlight"]{display:none !important;}
[role="tablist"] [role="tab"]{border-bottom:0 !important;transition:none !important;}
[role="tablist"] > :not([role="tab"]){display:none !important;}
[role="tablist"] [role="tab"][aria-selected="true"],[role="tablist"] [role="tab"][aria-selected="true"] *{color:#fff !important;}
[role="tablist"]::after,[role="tablist"] [role="tab"]::after{display:none !important;}
[role="tabpanel"]{padding-top:1.3rem;}

/* ---------- inputs ---------- */
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div,[data-baseweb="input"]{
  background:rgba(255,255,255,.06) !important;border-radius:12px !important;border:1px solid var(--cp-border) !important;color:var(--cp-text) !important;
}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--cp-a1) !important;box-shadow:0 0 0 3px rgba(139,92,246,.25) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--cp-card);border:1.5px dashed rgba(139,92,246,.55) !important;border-radius:16px !important;transition:all .2s ease;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cp-a2);background:var(--cp-card-2);}

/* ---------- buttons ---------- */
.stButton>button,.stLinkButton>a{border-radius:12px;font-weight:700;transition:transform .15s ease,box-shadow .2s ease,filter .2s ease;border:1px solid var(--cp-border);}
.stButton>button[kind="primary"]{background:var(--cp-grad-hot);border:0;color:#fff;padding:.65rem 1.4rem;box-shadow:0 10px 28px rgba(139,92,246,.4);}
.stButton>button:hover,.stLinkButton>a:hover{transform:translateY(-1px);filter:brightness(1.08);}
.stButton>button[kind="primary"]:hover{box-shadow:0 14px 34px rgba(244,114,182,.45);}

/* ---------- glass cards (bordered containers, expanders, alerts) ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--cp-card);border:1px solid var(--cp-border) !important;border-radius:20px !important;
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 14px 40px rgba(0,0,0,.28);
  transition:transform .2s ease,border-color .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover{border-color:rgba(139,92,246,.55) !important;}
[data-testid="stExpander"]{background:var(--cp-card);border:1px solid var(--cp-border);border-radius:16px;}
[data-testid="stAlert"]{border-radius:14px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid var(--cp-border);}
video,audio{border-radius:14px;max-width:100%;}
.cp-thumb{aspect-ratio:9/16;max-height:340px;width:100%;border-radius:16px;display:grid;place-content:center;text-align:center;gap:.3rem;color:var(--cp-muted);
  background:linear-gradient(160deg,rgba(139,92,246,.25),rgba(34,211,238,.10) 60%,rgba(244,114,182,.15));border:1px solid var(--cp-border);}
.cp-thumb span{font-size:2.2rem;}
.cp-clip-head{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.4rem;}
.cp-rank{font-weight:800;font-size:1.1rem;background:var(--cp-grad-hot);-webkit-background-clip:text;background-clip:text;color:transparent;}
.cp-quote{margin:.5rem 0;padding:.7rem .9rem;border-left:3px solid var(--cp-a1);border-radius:0 12px 12px 0;background:rgba(255,255,255,.04);color:var(--cp-text);font-size:.95rem;line-height:1.5;}
.cp-foot{color:var(--cp-muted);font-size:.74rem;text-align:center;margin-top:2rem;}

/* ---------- responsive ---------- */
@media (max-width:900px){ .cp-stats{grid-template-columns:repeat(2,minmax(0,1fr));} }
@media (max-width:640px){
  .block-container{padding:1rem .8rem 3rem;}
  .cp-hero{padding:1rem 1.05rem;gap:.8rem;border-radius:18px;}
  .cp-hero h1{font-size:1.6rem;} .cp-hero .logo{width:46px;height:46px;font-size:1.5rem;}
  [role="tablist"] [role="tab"]{padding:.5rem .8rem;font-size:.9rem;flex:1 1 auto;justify-content:center;}
  .cp-stat .v{font-size:1.3rem;} .cp-thumb{max-height:220px;aspect-ratio:16/9;}
}
@media (prefers-reduced-motion:reduce){ *{transition:none !important;} }
</style>
"""


def _e(value) -> str:
    """HTML-escape anything that came from a user, a file or a model."""
    return html.escape(str(value), quote=True)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text, tone: str = "") -> str:
    return f'<span class="cp-badge {tone}">{_e(text)}</span>'


def stat_tiles(items) -> str:
    """items: [(value, label), ...] -> a responsive row of metric tiles."""
    tiles = "".join(f'<div class="cp-stat"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div></div>' for v, l in items)
    return f'<div class="cp-stats">{tiles}</div>'


def steps_html(steps) -> str:
    inner = "".join(f'<span class="cp-step"><b>{i}</b>{_e(s)}</span>' for i, s in enumerate(steps, start=1))
    return f'<div class="cp-steps">{inner}</div>'


def section_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="cp-section"><h3>{_e(title)}</h3><p>{_e(subtitle)}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def save_upload(uploaded) -> str:
    """Write a Streamlit UploadedFile to disk and return its path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{int(time.time())}_{Path(uploaded.name).name}"
    dest.write_bytes(uploaded.getbuffer())
    return str(dest)


def cleanup_uploads(paths) -> None:
    """Delete the temporary copies of uploaded files as soon as a run is over (frees disk, closes handles)."""
    for p in paths:
        try:
            if p:
                Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def make_progress():
    """Return (callback, bar) wiring pipeline progress to a Streamlit bar."""
    bar = st.progress(0.0, text="Starting...")

    def callback(stage: str, fraction: float) -> None:
        bar.progress(fraction, text=stage)

    return callback, bar


def show_warnings(warnings) -> None:
    for w in warnings or []:
        st.warning(w)


def _demo_mode() -> bool:
    try:
        return str(st.query_params.get("demo", "")) == "1"
    except Exception:
        return False


DEMO_CLIPS = {
    "source_path": "demo.mp4", "word_count": 1842, "warnings": [],
    "clips": [
        {"start": 42.0, "end": 71.5, "duration": 29.5, "score": 14.8, "faces_found": 884,
         "reasons": ["question", "emotional_keywords", "dramatic_pause"], "output_path": None,
         "drive_link": "https://drive.google.com/", "error": None,
         "text": "Did you know the biggest mistake nobody warns you about is trying to do it all at once?"},
        {"start": 190.0, "end": 228.0, "duration": 38.0, "score": 11.2, "faces_found": 0,
         "reasons": ["number_or_stat", "direct_address"], "output_path": None, "drive_link": None,
         "error": "Upload skipped: Google Drive is not connected.",
         "text": "In three weeks you can change your whole routine with just 5 minutes a day."},
    ],
}
DEMO_LAB = {
    "script": {"script": "Stop scrolling. Here is how a morning routine really works. First, one small habit. "
                         "Second, make it easy. Follow for more and share this with a friend.",
               "word_count": 31, "estimated_seconds": 12.4, "topic": "a morning routine"},
    "audio_path": None, "video_path": None, "drive_link": "https://drive.google.com/", "warnings": [],
    "avatar": {"mode": "mouth-warp", "face_found": True, "used_word_timestamps": True, "layout": "full"},
}


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div class="cp-brand"><div class="logo">🎬</div><div><b>{APP_NAME}</b>'
            '<small>Local video studio</small></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="cp-side-title">Cloud backup</div>', unsafe_allow_html=True)
        folder_id = st.text_input(
            "Google Drive Folder ID",
            value=os.environ.get("GOOGLE_DRIVE_FOLDER_ID", ""),
            key="drive_folder_id",
            placeholder="e.g. 1AbCdEfGh...",
            help="The long ID at the end of your Drive folder's URL: "
                 "drive.google.com/drive/folders/<FOLDER_ID>",
        ).strip()
        status = uploader.drive_status()
        if status == "ready":
            st.success("Google account connected (token.json found).")
        elif status == "needs_signin":
            st.info("credentials.json found - the first upload will open a Google sign-in page.")
        else:
            st.warning(
                "Google account not connected. Drive backup is off; clips are still saved on "
                "this computer. To turn it on, put credentials.json or token.json next to the "
                "app (see README.txt)."
            )
        if not folder_id:
            st.info("Enter a Folder ID to turn on automatic Drive backup.")

        st.markdown('<div class="cp-side-title">Privacy</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="cp-pill"><span class="cp-dot"></span>Runs on this computer</span> '
            '<span class="cp-pill">No API keys</span>',
            unsafe_allow_html=True,
        )
        st.caption("Only YouTube downloads, edge-tts voices and the Drive upload use the internet.")
        if getattr(sys, "frozen", False):  # packaged exe has no console window to close
            if st.button("Quit CinePulse AI", key="quit_app"):
                st.write("Closing CinePulse AI. You can close this tab.")
                os._exit(0)
    return folder_id


# --------------------------------------------------------------------------- #
# clip repurposer
# --------------------------------------------------------------------------- #
def render_clip_card(i: int, clip: dict) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            if clip.get("output_path"):
                st.video(clip["output_path"])
            else:
                st.markdown('<div class="cp-thumb"><span>🎞️</span><small>Preview unavailable</small></div>',
                            unsafe_allow_html=True)
        with right:
            badges = [badge(f"★ {clip['score']}", "score"),
                      badge(f"{clip['duration']}s", "time"),
                      badge(f"{clip['start']}s → {clip['end']}s", "time")]
            faces = clip.get("faces_found")
            if faces is not None:
                badges.append(badge("face tracked" if faces else "centre crop", "face" if faces else "warn"))
            badges.append(badge("backed up" if clip.get("drive_link") else "local only",
                                "ok" if clip.get("drive_link") else "warn"))
            st.markdown(f'<div class="cp-clip-head"><span class="cp-rank">#{i}</span>{"".join(badges)}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="cp-quote">{_e(clip["text"])}</div>', unsafe_allow_html=True)
            if clip.get("reasons"):
                chips = "".join(f'<span class="cp-chip">{_e(r.replace("_", " "))}</span>' for r in clip["reasons"])
                st.markdown(f'<div class="cp-chips">{chips}</div>', unsafe_allow_html=True)
            if clip.get("screen_text"):
                lines = "".join(f'<span class="cp-chip">{_e(t)}</span>' for t in clip["screen_text"][:6])
                st.markdown(f'<div class="cp-chips"><span class="cp-chip">🔤 on screen</span>{lines}</div>',
                            unsafe_allow_html=True)
            if clip.get("drive_link"):
                st.link_button("Open in Google Drive", clip["drive_link"])
            if clip.get("error"):
                st.error(clip["error"])


def render_clip_tab(folder_id: str) -> None:
    section_header("Clip Repurposer Studio",
                   "Turn a long video into short vertical clips that follow the speaker's face.")
    st.markdown(steps_html(["Source", "Transcribe", "Score highlights", "Face-tracked crop", "Drive backup"]),
                unsafe_allow_html=True)

    url = st.text_input("YouTube URL", key="vugola_url",
                        placeholder="https://www.youtube.com/watch?v=...")
    uploaded = st.file_uploader("...or upload a video file", type=VIDEO_TYPES, key="vugola_file")

    with st.expander("Clip options"):
        c1, c2, c3 = st.columns(3)
        with c1:
            top_n = st.slider("Number of clips", 1, 10, 3, key="vugola_top_n")
        with c2:
            min_len = st.slider("Min length (s)", 5, 60, 15, key="vugola_min_len")
        with c3:
            max_len = st.slider("Max length (s)", 15, 180, 45, key="vugola_max_len")
        fallback = st.checkbox(
            "Use a centre crop if the face model can't run", value=True, key="vugola_fallback"
        )
        ocr_on = st.checkbox(
            "Also read on-screen text (OCR) - burned-in subtitles and titles are added to the transcript",
            value=False, key="vugola_ocr",
            help="Looks at one frame per second on your CPU. Slower on long videos; loaded only when you click Generate.",
        )

    if st.button("Generate viral clips", type="primary", key="vugola_run"):
        if not url.strip() and uploaded is None:
            st.warning("Paste a YouTube URL or upload a video first.")
        elif min_len > max_len:
            st.error("Min length must not be greater than max length.")
        else:
            callback, bar = make_progress()
            temp = []
            try:
                source = url.strip()
                if uploaded is not None:
                    source = save_upload(uploaded)
                    temp.append(source)
                st.session_state["vugola_result"] = pipeline.run_vugola_pipeline(
                    source, folder_id, top_n=top_n, min_len=min_len, max_len=max_len,
                    output_dir=str(OUTPUT_DIR), progress=callback, allow_center_fallback=fallback,
                    read_screen_text=ocr_on,
                )
            except Exception as exc:
                st.session_state.pop("vugola_result", None)
                bar.empty()
                st.error(f"Something went wrong: {exc}")
            finally:
                cleanup_uploads(temp)

    result = st.session_state.get("vugola_result")
    if result:
        show_warnings(result.get("warnings"))
        clips = result["clips"]
        uploaded_n = sum(1 for c in clips if c.get("drive_link"))
        st.success(f"Found {len(clips)} clip(s) from {result['word_count']} transcribed words.")
        best = max((c["score"] for c in clips), default=0)
        st.markdown(stat_tiles([(len(clips), "Clips created"), (result["word_count"], "Words transcribed"),
                                (best, "Top score"), (f"{uploaded_n}/{len(clips)}", "Backed up to Drive")]),
                    unsafe_allow_html=True)
        for i, clip in enumerate(clips, start=1):
            render_clip_card(i, clip)


# --------------------------------------------------------------------------- #
# generator lab
# --------------------------------------------------------------------------- #
def render_layout_preview(position: str, xy, scale: float, avatar_file) -> None:
    """Show where the avatar will sit - the same geometry, frame and shadow as the finished video."""
    try:
        import cv2
        import numpy as np
        import video_engine

        avatar_frame = None
        if avatar_file is not None and avatar_file.name.lower().endswith(("jpg", "jpeg", "png")):
            avatar_frame = cv2.imdecode(np.frombuffer(avatar_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img, rect = video_engine.preview_layout(position_preset=position, custom_coordinates=xy, scale=scale,
                                                avatar_frame=avatar_frame)
        left, right = st.columns([1, 2])
        with left:
            st.image(img[:, :, ::-1], caption="Layout preview (1080 x 1920 video)", width=180)
        with right:
            x, y, w, h = rect
            st.markdown(f"**Placement:** {position}  \n**Frame:** {w} x {h} px at X {x}, Y {y}")
    except Exception as exc:  # a preview must never block the form
        st.caption(f"Layout preview unavailable ({exc}).")


def render_lab_tab(folder_id: str) -> None:
    section_header("AI Generator Lab",
                   "Describe a video. Get a script, a natural voiceover and a finished vertical video.")
    st.markdown(steps_html(["Prompt", "Script", "Voiceover", "Avatar / video merge", "Drive upload"]),
                unsafe_allow_html=True)

    prompt = st.text_area(
        "What should the video be about?", key="viewmax_prompt", height=120,
        placeholder="Write a motivational 30 second TikTok script about building a morning routine",
    )
    with st.expander("Script & action narration (optional)", expanded=False):
        custom_script = st.text_area(
            "Your own script", key="viewmax_script", height=150,
            placeholder="Paste the exact words to speak. Leave empty to let the app write a script from the prompt above.\n"
                        "You can add directions inside the script, like: Welcome back! [zoom in] Today we test a drone.",
        )
        action_prompt = st.text_area(
            "Visual actions", key="viewmax_actions", height=110,
            placeholder="Describe what should happen on screen, in order. Example:\n"
                        "Zoom in on the product. Then pan right to the price; flash when I say 'now'.",
        )
        st.caption("Understood directions: zoom in / zoom out, pan left / pan right, fade, flash, shake. "
                   "They become camera moves timed to your voiceover.")
    background = st.file_uploader(
        "Background video (optional - a plain dark background is used if empty)",
        type=VIDEO_TYPES, key="viewmax_bg",
    )
    read_text = st.checkbox(
        "Read & narrate the on-screen text of the background video (OCR)", value=False, key="viewmax_read_text",
        help="Finds burned-in text in your background video and speaks it after the script.",
    )
    voice = st.selectbox("Voice", VOICES, index=0, key="viewmax_voice")

    with st.container(border=True):
        st.markdown('<div class="cp-clip-head"><span class="cp-rank">AI Avatar</span>'
                    + badge("3D · runs on CPU", "ok") + badge("no GPU needed", "time") + '</div>',
                    unsafe_allow_html=True)
        avatar_on = st.checkbox("Enable AI Avatar Lip-Sync Commentary", value=False, key="viewmax_avatar_on",
                                help="Your avatar talks in time with the voiceover, in a corner of the video.")
        model_label = st.selectbox("Model type", list(MODEL_TYPES), key="viewmax_model_type", disabled=not avatar_on,
                                   help="Sets the look: realistic lighting, a cool AI-model glow, or cel-shaded anime.")
        avatar_file = st.file_uploader(
            "Your model: a 3D file (.glb or .vrm) for body + head motion, or a front-facing picture (.jpg / .png). "
            "Leave empty to use the built-in 3D avatar.",
            type=AVATAR_TYPES, key="viewmax_avatar", disabled=not avatar_on,
        )
        if avatar_on and avatar_file is not None and avatar_file.name.lower().endswith(("jpg", "jpeg", "png")):
            st.image(avatar_file, width=140, caption="Your avatar")
        elif avatar_on and avatar_file is None:
            st.caption("No file added: the built-in 3D avatar for the chosen model type is used.")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            position = st.selectbox("Avatar position", list(pipeline.PIP_PRESETS), key="viewmax_position",
                                    disabled=not avatar_on, help="Which corner of the video the avatar sits in.")
        with pcol2:
            scale_pct = st.slider("Avatar size (% of the video width)", 15, 60, 36, key="viewmax_scale",
                                  disabled=not avatar_on)
        custom_xy = None
        if position == "Custom Coordinates":
            xcol, ycol = st.columns(2)
            with xcol:
                cx = st.number_input("X (pixels from the left)", 0, 1080, 40, key="viewmax_x", disabled=not avatar_on)
            with ycol:
                cy = st.number_input("Y (pixels from the top)", 0, 1920, 1200, key="viewmax_y", disabled=not avatar_on)
            custom_xy = (int(cx), int(cy))
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            gestures_on = st.checkbox("Body & head gestures", value=True, key="viewmax_gestures", disabled=not avatar_on,
                                      help="Nods, hand gestures and breathing chosen from what is being said (3D models).")
        with gcol2:
            quality_label = st.selectbox("Render quality", list(QUALITY_CHOICES), key="viewmax_quality",
                                         disabled=not avatar_on)
        if avatar_on:
            render_layout_preview(position, custom_xy, scale_pct / 100.0, avatar_file)
        st.caption("3D models get phoneme-level lip sync, blinking, breathing, head nods and arm gestures; the "
                   "video is 1080 x 1920 and the avatar sits in a rounded podcast frame.")

    if st.button("Generate video", type="primary", key="viewmax_run"):
        if not prompt.strip() and not custom_script.strip():
            st.warning("Enter a prompt or paste your own script first.")
        else:
            callback, bar = make_progress()
            temp = []
            try:
                bg_path = None
                if background is not None:
                    bg_path = save_upload(background)
                    temp.append(bg_path)
                extra = {}
                if avatar_on:
                    style = MODEL_TYPES[model_label]
                    if avatar_file is not None:
                        avatar_path = save_upload(avatar_file)
                        temp.append(avatar_path)
                    else:
                        avatar_path = f"demo:{style}"
                    extra = {"avatar_image": avatar_path, "avatar_enabled": True, "avatar_type": style,
                             "avatar_position": position, "avatar_scale": scale_pct / 100.0,
                             "avatar_gestures": gestures_on, "avatar_quality": QUALITY_CHOICES[quality_label]}
                    if custom_xy:
                        extra["avatar_xy"] = custom_xy
                if custom_script.strip():
                    extra["custom_script"] = custom_script
                if action_prompt.strip():
                    extra["action_prompt"] = action_prompt
                if read_text:
                    extra["read_background_text"] = True
                st.session_state["viewmax_result"] = pipeline.run_viewmax_pipeline(
                    prompt, folder_id, background_video=bg_path, voice=voice,
                    output_dir=str(OUTPUT_DIR), progress=callback, **extra,
                )
            except Exception as exc:
                st.session_state.pop("viewmax_result", None)
                bar.empty()
                st.error(f"Something went wrong: {exc}")
            finally:
                cleanup_uploads(temp)

    result = st.session_state.get("viewmax_result")
    if result:
        show_warnings(result.get("warnings"))
        script = result["script"]
        avatar = result.get("avatar")
        tiles = [(script["word_count"], "Words"), (f"{script['estimated_seconds']}s", "Est. length"),
                 ("Yes" if result.get("drive_link") else "No", "Backed up to Drive")]
        if avatar:
            tiles.append(({"mouth-warp": "Mouth sync", "3d": "3D"}.get(avatar["mode"], "Pulse"), "Avatar animation"))
            if isinstance(avatar.get("layout"), dict):
                tiles.append((avatar["layout"]["preset"], "Avatar position"))
            if avatar.get("gestures"):
                tiles.append((len(avatar["gestures"]), "Gestures played"))
        if result.get("cues_applied"):
            tiles.append((result["cues_applied"], "Visual actions"))
        st.markdown(stat_tiles(tiles), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="cp-clip-head"><span class="cp-rank">Script</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cp-quote">{_e(script["script"])}</div>', unsafe_allow_html=True)
            if result.get("audio_path"):
                st.audio(result["audio_path"])
            if result.get("video_path"):
                st.video(result["video_path"])
            if result.get("screen_text"):
                lines = "".join(f'<span class="cp-chip">{_e(t)}</span>' for t in result["screen_text"][:8])
                st.markdown(f'<div class="cp-chips"><span class="cp-chip">🔤 read from video</span>{lines}</div>',
                            unsafe_allow_html=True)
            if script.get("shot_list"):
                with st.expander("Shot list (visual actions)"):
                    for line in script["shot_list"]:
                        st.markdown(f"- {_e(line)}")
            if result.get("drive_link"):
                st.link_button("Open in Google Drive", result["drive_link"])


# --------------------------------------------------------------------------- #
def render_hero() -> None:
    st.markdown(
        '<div class="cp-hero"><div class="logo">🎬</div><div class="grow">'
        f'<h1>CinePulse <span>AI</span></h1>'
        '<p>Repurpose long videos into viral shorts, and turn ideas into finished videos, '
        'right on your own computer.</p></div>'
        '<div class="cp-pills"><span class="cp-pill"><span class="cp-dot"></span>Local &amp; private</span>'
        '<span class="cp-pill">Free · no API keys</span></div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide", initial_sidebar_state="auto")
    inject_css()
    if _demo_mode():  # ?demo=1 -> sample cards for previewing the layout
        st.session_state.setdefault("vugola_result", DEMO_CLIPS)
        st.session_state.setdefault("viewmax_result", DEMO_LAB)
    folder_id = render_sidebar()
    render_hero()
    tab_clip, tab_lab = st.tabs([TAB_CLIP, TAB_LAB])
    with tab_clip:
        render_clip_tab(folder_id)
    with tab_lab:
        render_lab_tab(folder_id)
    st.markdown(f'<div class="cp-foot">{APP_NAME} · everything runs locally · your files never leave this computer '
                'except the Drive backup you enable</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
