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
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg"]
OCR_CHOICES = {"Auto (PaddleOCR if installed)": "auto", "PaddleOCR": "paddleocr", "RapidOCR (built in)": "rapidocr"}
LAB_MODES = {"Idea2Video": "idea2video", "Novel2Video": "novel2video", "Script2Video": "script2video",
             "Clip Repurposer": "clips"}
MODE_BLURB = {
    "idea2video": "One prompt in, a whole episodic video out: the agents write, cast, storyboard and render it.",
    "novel2video": "Upload a long text or novel: it is read in chapters, boiled down to episode scripts and rendered as episodes.",
    "script2video": "Paste a screenplay (INT./EXT., names, parentheticals, dialogue): it becomes a timed multi-shot video.",
    "clips": "Cut a long video into short vertical clips with MediaPipe face tracking and OCR text reading.",
}
IDEA_FLAVORS = {"Episodic story video (ViMax agents)": "story", "Narrated video with a 3D avatar (classic)": "classic"}
STORY_PROVIDERS = {"Offline - built-in rules (no key, no internet)": "offline", "Groq (needs an API key)": "groq",
                   "Gemini (needs an API key)": "gemini"}
CAMERA_SIZES = {"Auto (let the script decide)": "auto", "Close-ups": "close", "Medium shots": "medium", "Wide shots": "wide"}
CAMERA_ANGLES = {"Auto": "auto", "Eye level": "eye", "Low angle (heroic)": "low", "High angle": "high", "Dutch tilt (tense)": "dutch"}
CAMERA_MOVES = {"Auto": "auto", "Static": "static", "Slow push-in": "push", "Handheld": "handheld", "Slow pan": "pan"}
HOST_POSITIONS = list(pipeline.PIP_PRESETS)
STORY_TEXT_TYPES = ["txt", "md", "fountain"]
BACKDROPS = {"Aurora (blue / magenta)": "aurora", "Sunset": "sunset", "Mono (dark grey)": "mono"}
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

/* ---------- motion (GSAP-inspired easings: power3.out, back.out; transform/opacity only) ---------- */
:root{--ease-power3-out:cubic-bezier(.215,.61,.355,1);--ease-back-out:cubic-bezier(.34,1.56,.64,1);--ease-power2-inout:cubic-bezier(.645,.045,.355,1);}
@keyframes cp-rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
@keyframes cp-pop{from{opacity:0;transform:translateY(8px) scale(.96);}to{opacity:1;transform:none;}}
@keyframes cp-float{0%,100%{transform:translateY(0);}50%{transform:translateY(-3px);}}
[role="tabpanel"]{animation:cp-rise .55s var(--ease-power3-out) both;}
[role="tabpanel"] [data-testid="stVerticalBlockBorderWrapper"]{animation:cp-rise .6s var(--ease-power3-out) both;}
[role="tabpanel"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(2){animation-delay:.06s;}
[role="tabpanel"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(3){animation-delay:.12s;}
[role="tabpanel"] [data-testid="stVerticalBlockBorderWrapper"]:nth-of-type(n+4){animation-delay:.18s;}
.cp-stat{animation:cp-pop .5s var(--ease-back-out) both;}
.cp-stat:nth-child(2){animation-delay:.05s;} .cp-stat:nth-child(3){animation-delay:.10s;} .cp-stat:nth-child(4){animation-delay:.15s;}
.cp-stat:nth-child(n+5){animation-delay:.20s;}
.cp-hero .logo{animation:cp-float 4.5s var(--ease-power2-inout) infinite;}
[role="tablist"] [role="tab"]{transition:transform .25s var(--ease-back-out),background-color .25s var(--ease-power3-out),color .25s ease !important;}
[role="tablist"] [role="tab"]:hover{transform:translateY(-1px);}
.stButton>button{transition:transform .25s var(--ease-back-out),box-shadow .25s var(--ease-power3-out),filter .2s ease;}
.stButton>button:active{transform:scale(.98);}
@media (prefers-reduced-motion:reduce){ *,*::before,*::after{animation:none !important;transition:none !important;} }
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


MEME_MODES = {"Auto (cut away on big reactions, otherwise a corner tile)": "auto", "Corner tile only": "overlay",
              "Full-screen cutaway only": "cutaway"}
MEME_POSITIONS = ["Auto (avoid the avatar)"] + list(pipeline.PIP_PRESETS)


def render_meme_panel(prefix: str) -> dict | None:
    """'Enable Contextual Meme Reactions' + the reaction-clip library (upload / list / delete / starter pack).
    Returns the settings dict for the pipelines, or None when switched off. All matching runs locally."""
    mm = pipeline.meme_matcher
    with st.expander("😂 Contextual meme reactions", expanded=False):
        on = st.checkbox("Enable Contextual Meme Reactions", value=False, key=f"{prefix}_meme_on",
                         help="Matches the mood of each line (laugh, shock, sad, ...) to a clip in your meme folder and "
                              "splices it in as a corner tile or a quick cutaway. Runs on the CPU, no API key.")
        lib = mm.MemeLibrary(mm.default_meme_dir())
        info = lib.summary()
        st.caption(f"Meme folder: {info['folder']} - {info['clips']} clip(s). Matching uses folder names "
                   "(laugh/, shock/, ...), file names and tags, not the picture itself.")
        c1, c2, c3 = st.columns(3)
        with c1:
            mode_label = st.selectbox("Style", list(MEME_MODES), key=f"{prefix}_meme_mode", disabled=not on)
        with c2:
            per_min = st.slider("Max reactions per minute", 1, 12, 4, key=f"{prefix}_meme_rate", disabled=not on)
        with c3:
            sens = st.slider("Sensitivity (%)", 0, 100, 50, key=f"{prefix}_meme_sens", disabled=not on,
                             help="Higher reacts to milder moments.")
        d1, d2 = st.columns(2)
        with d1:
            pos_label = st.selectbox("Tile corner", MEME_POSITIONS, key=f"{prefix}_meme_pos", disabled=not on)
        with d2:
            use_audio = st.checkbox("Play the meme's own sound", value=True, key=f"{prefix}_meme_audio", disabled=not on)
        xy = None
        if on and pos_label == "Custom Coordinates":
            e1, e2 = st.columns(2)
            with e1:
                mx = st.number_input("Tile X (pixels)", 0, 3000, 40, key=f"{prefix}_meme_x")
            with e2:
                my = st.number_input("Tile Y (pixels)", 0, 3000, 200, key=f"{prefix}_meme_y")
            xy = (int(mx), int(my))

        st.markdown("**Reaction clip library**")
        emo_label = st.selectbox("File the upload under", ["(let the file name decide)"] + list(mm.EMOTIONS),
                                 key=f"{prefix}_meme_emo")
        ups = st.file_uploader("Add your own reaction videos", type=[e.lstrip(".") for e in mm.VIDEO_EXTENSIONS],
                               accept_multiple_files=True, key=f"{prefix}_meme_up")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Add to library", key=f"{prefix}_meme_add", disabled=not ups):
                for f in ups:
                    try:
                        lib.add_file(f.name, f.getvalue(), None if emo_label.startswith("(") else emo_label)
                        st.success(f"Added {f.name}")
                    except Exception as exc:
                        st.error(str(exc))
        with b2:
            if st.button("Create the starter pack", key=f"{prefix}_meme_starter",
                         help="10 simple original animated reactions, one per mood."):
                try:
                    mm.ensure_starter_pack(str(lib.dir))
                    st.success("Starter pack created.")
                except Exception as exc:
                    st.error(f"Could not create the starter pack: {exc}")
        if info["per_emotion"]:
            st.markdown('<div class="cp-chips">' + "".join(
                f'<span class="cp-chip">{_e(k)}: {v}</span>' for k, v in info["per_emotion"].items()) + "</div>",
                unsafe_allow_html=True)
        for bad in info["failed"][:5]:
            st.warning(f"Skipped {bad.get('path')}: {bad.get('reason')}")
        names = [c["rel"] for c in lib.clips] if lib.clips and "rel" in lib.clips[0] else [c["name"] for c in lib.clips]
        if names:
            gone = st.selectbox("Remove a clip", ["-"] + names, key=f"{prefix}_meme_del")
            if gone != "-" and st.button("Delete this clip", key=f"{prefix}_meme_delbtn"):
                st.success("Deleted." if lib.remove(gone) else "That clip could not be deleted.")
    if not on:
        return None
    cfg = {"enabled": True, "mode": MEME_MODES[mode_label], "max_per_minute": float(per_min),
           "sensitivity": sens / 100.0, "use_audio": bool(use_audio)}
    if not pos_label.startswith("Auto"):
        cfg["position"] = pos_label
        if xy:
            cfg["xy"] = xy
    return cfg


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
        clip_ocr_label = st.selectbox("Text reader (OCR)", list(OCR_CHOICES), key="vugola_ocr_backend", disabled=not ocr_on,
                                      help="Which engine reads the text burned into the video frames.")

    meme_cfg = render_meme_panel("vugola")

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
                    read_screen_text=ocr_on, memes=meme_cfg,
                    **({"ocr_backend": OCR_CHOICES[clip_ocr_label]} if ocr_on and OCR_CHOICES[clip_ocr_label] != "auto" else {}),
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


def render_narrated_generator(folder_id: str) -> None:
    """The original generator: prompt -> script -> voiceover -> video with an optional lip-synced 3D avatar."""
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
                        "You can add directions inside the script, like: Welcome back! [zoom in] Today we test a drone.\n\n"
                        "Longer scripts work too:\nSCENE 1: Hook\nVISUAL: zoom in on the logo\nNARRATOR: Stop scrolling!\n"
                        "[0:06] Did you know? [wave]\n(slowly pan right)\n[pause 1s]",
        )
        st.caption("Script format: SCENE headings, NARRATOR: labels, VISUAL: / (parenthesis) / > direction lines, "
                   "[m:ss] timecodes, [pause] and inline [directions]. Directions apply to the next spoken sentence.")
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
    ocol1, ocol2 = st.columns(2)
    with ocol1:
        ocr_label = st.selectbox("Text reader (OCR)", list(OCR_CHOICES), key="viewmax_ocr_backend", disabled=not read_text,
                                 help="Which engine reads the text burned into the video frames.")
    with ocol2:
        show_text = st.checkbox("Also show the text as captions", value=True, key="viewmax_show_text", disabled=not read_text,
                                help="Displays each line on screen at the moment it appears in your background video.")
    music = st.file_uploader("Background music (optional - kept quiet under the voice)", type=AUDIO_TYPES, key="viewmax_music")
    backdrop_label = st.selectbox("Animated backdrop (used when there is no background video)", list(BACKDROPS),
                                  key="viewmax_backdrop")
    ltx = pipeline.render_engine.ltx_video_status()
    st.caption("Readers installed: " + (", ".join(pipeline.ocr_engine.available_backends()) or "none")
               + "  ·  AI-generated B-roll (LTX-Video): " + ("ready" if ltx["available"] else "not available here - " + ltx["reason"]))
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

    meme_cfg = render_meme_panel("viewmax")

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
                    extra["ocr_backend"] = OCR_CHOICES[ocr_label]
                    extra["show_screen_text"] = bool(show_text)
                if music is not None:
                    music_path = save_upload(music)
                    temp.append(music_path)
                    extra["music_path"] = music_path
                if BACKDROPS[backdrop_label] != "aurora":
                    extra["broll_style"] = BACKDROPS[backdrop_label]
                st.session_state["viewmax_result"] = pipeline.run_viewmax_pipeline(
                    prompt, folder_id, background_video=bg_path, voice=voice,
                    output_dir=str(OUTPUT_DIR), progress=callback, memes=meme_cfg, **extra,
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


def _story_step1(folder_id: str, mode: str, wkey: str) -> None:
    """Step 1 - the story: idea / screenplay / novel -> narrative beats, characters, locations, storyboard (no render yet)."""
    hints = {"idea2video": "Describe the story in a few sentences. Each episode takes its share of it.",
             "script2video": "Paste a screenplay: INT./EXT. headings, CHARACTER names, (parentheticals) and dialogue.",
             "novel2video": "Paste a chapter or a whole novel, or upload a .txt file below."}
    text = st.text_area({"idea2video": "Your idea", "script2video": "Your screenplay", "novel2video": "Your novel or long text"}[mode],
                        key=f"story_text_{mode}", height=200, placeholder=hints[mode])
    upload = None
    if mode != "idea2video":
        upload = st.file_uploader("...or upload a text file", type=STORY_TEXT_TYPES, key=f"story_file_{mode}")

    with st.expander("Story options", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            style_label = st.selectbox("Character style", list(MODEL_TYPES), key="story_style")
            seconds = st.slider("Length of each video (seconds)", 15, 180, 45, key="story_seconds")
        with c2:
            provider_label = st.selectbox("Writing assistant", list(STORY_PROVIDERS), key="story_provider",
                                          help="Offline uses fixed rules. Groq or Gemini can rewrite and enrich the "
                                               "storyboard; only that text is sent to the provider.")
        api_key = st.text_input("API key (used for this run only, never saved)", value="", key="story_api_key",
                                type="password", disabled=STORY_PROVIDERS[provider_label] == "offline")
        episodes, episode, all_eps = 1, 1, False
        if mode in ("idea2video", "novel2video"):
            e1, e2, e3 = st.columns(3)
            with e1:
                episodes = st.number_input("Episodes", 1, 12, 3 if mode == "novel2video" else 1, key="story_episodes")
            with e2:
                all_eps = st.checkbox("Make every episode", value=mode == "novel2video", key="story_all_episodes")
            with e3:
                episode = st.number_input("...or just episode", 1, 12, 1, key="story_episode", disabled=bool(all_eps))
        st.caption("Next you will review every character and location and give each a reference picture, then render. "
                   "The video is a stylised 2D scene animation (background plate + cut-out characters + camera moves), "
                   "not photorealistic AI video. Finished videos and the asset pictures are backed up to the Google "
                   "Drive folder from the sidebar.")

    with st.expander("Character consistency", expanded=False):
        portraits_on = st.checkbox("Show each character's portrait in every shot they appear in", value=True, key="story_portraits",
                                   help="Only used by the older portrait-tile animatic; the scene renderer always uses the locked pictures.")
        continuity_on = st.checkbox("Start each shot from the previous shot's last frame (scene continuity)", value=True,
                                    key="story_continuity",
                                    help="Only used by the older portrait-tile animatic.")
        tolerance = st.slider("Appearance drift tolerance (%)", 10, 60, 35, key="story_tolerance",
                              help="How much a character may differ from their locked reference picture in the rendered "
                                   "frame before the drift report flags it. Lower is stricter.")

    with st.expander("Camera angle preferences", expanded=False):
        k1, k2, k3 = st.columns(3)
        with k1:
            size_label = st.selectbox("Shot size", list(CAMERA_SIZES), key="story_cam_size")
        with k2:
            angle_label = st.selectbox("Camera angle", list(CAMERA_ANGLES), key="story_cam_angle")
        with k3:
            move_label = st.selectbox("Camera movement", list(CAMERA_MOVES), key="story_cam_move")
        st.caption("Applied to every shot except the opening establishing shots, which stay wide. Close-ups also get larger "
                   "character portraits and static shots hold the backdrop still.")

    with st.expander("Podcast corner avatar overlay", expanded=False):
        host_on = st.checkbox("Show a host / narrator avatar in a corner", value=False, key="story_host_on")
        host_file = st.file_uploader("Host picture (optional - a drawn host is used when empty)", type=["png", "jpg", "jpeg"],
                                     key="story_host_file", disabled=not host_on)
        h1, h2 = st.columns(2)
        with h1:
            host_pos = st.selectbox("Corner", HOST_POSITIONS, key="story_host_pos", disabled=not host_on)
        with h2:
            host_scale = st.slider("Avatar size (% of the video width)", 15, 50, 28, key="story_host_scale", disabled=not host_on)
        host_xy = None
        if host_on and host_pos == "Custom Coordinates":
            x1, x2 = st.columns(2)
            with x1:
                hx = st.number_input("X (pixels from the left)", 0, 1080, 40, key="story_host_x")
            with x2:
                hy = st.number_input("Y (pixels from the top)", 0, 1920, 1200, key="story_host_y")
            host_xy = (int(hx), int(hy))
        st.caption("The host is a framed still portrait. The lip-synced 3D avatar is in Idea2Video > "
                   "'Narrated video with a 3D avatar (classic)'. Story characters keep clear of the host's corner.")

    if st.button("Parse the story and continue", type="primary", key="story_run"):
        temp = []
        try:
            source = text
            if upload is not None:
                path = save_upload(upload)
                temp.append(path)
                import novel_adapter
                source = novel_adapter.read_text_file(path)
                cleanup_uploads([path])
                temp.clear()
            if not source.strip():
                st.warning("Type or upload some text first.")
            else:
                callback, bar = make_progress()
                provider = STORY_PROVIDERS[provider_label]
                host = None
                if host_on:
                    host = {"style": MODEL_TYPES[style_label], "position": host_pos, "xy": host_xy, "scale": host_scale / 100.0}
                    if host_file is not None:
                        hp = save_upload(host_file)  # kept until the wizard ends: Step 3 renders with it
                        temp.append(hp)
                        host["image"] = hp
                common = dict(
                    mode=mode, folder_id=folder_id or None, output_dir=str(OUTPUT_DIR),
                    provider=provider, api_key=(api_key.strip() or None) if provider != "offline" else None,
                    style=MODEL_TYPES[style_label], target_seconds=float(seconds), episode=int(episode),
                    episodes=int(episodes) if mode in ("idea2video", "novel2video") and (all_eps or mode == "idea2video") else None,
                    all_episodes=bool(all_eps),
                    camera={"size": CAMERA_SIZES[size_label], "angle": CAMERA_ANGLES[angle_label], "movement": CAMERA_MOVES[move_label]},
                    use_portraits=bool(portraits_on), continuity=bool(continuity_on), drift_threshold=tolerance / 100.0,
                    host_avatar=host)
                plan = pipeline.run_story_pipeline(source, progress=callback, render=False, **common)
                _wizard_cleanup(st.session_state.get(wkey))
                st.session_state[wkey] = {"step": 2, "source": source, "common": common, "plan": plan, "temp": list(temp),
                                          "seen": {}, "backend": "auto"}
                st.session_state.pop("story_result", None)
                temp.clear()
                st.rerun()
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        finally:
            cleanup_uploads(temp)

ASSET_BACKENDS = {"Auto: free online generator, then local CPU model, then built-in painter": "auto",
                  "Free online generator (needs internet)": "online",
                  "Local CPU diffusers model (optional install)": "diffusers",
                  "Built-in painter (offline, simple flat art)": "draw"}
WIZARD_STEPS = ["Story & script", "Asset studio", "Render episode"]


def wizard_steps_html(current: int) -> str:
    inner = "".join(
        f'<span class="cp-step"{" style=&quot;border-color:var(--cp-a2)&quot;" if i == current else ""}><b>{i}</b>{_e(name)}'
        f'{" (current)" if i == current else ""}</span>' for i, name in enumerate(WIZARD_STEPS, start=1))
    return f'<div class="cp-steps">{inner}</div>'


def _wizard_cleanup(wiz) -> None:
    if wiz:
        cleanup_uploads(wiz.get("temp") or [])


def _asset_book(wiz):
    """The asset book of the planned episode (kept on disk in the project's assets folder)."""
    return pipeline.asset_manager.AssetBook.load(wiz["plan"]["assets_dir"])


def _asset_card(book, card: dict, mode: str, wiz: dict, backend_key: str) -> None:
    am = pipeline.asset_manager
    cid, sid = card["id"], card["id"].replace(":", "_").replace(" ", "_")
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            if card.get("image") and Path(card["image"]).is_file():
                st.image(card["image"], width=150)
            else:
                st.markdown('<span class="cp-chip">No picture yet</span>', unsafe_allow_html=True)
        with right:
            kind = "Character" if card["kind"] == "character" else "Location"
            st.markdown(f"**{_e(card['name'])}**  " + badge(kind) + "  "
                        + badge("Ready" if card.get("image") else "Needs a picture", "ok" if card.get("image") else "warn"),
                        unsafe_allow_html=True)
            if card["kind"] == "character":
                t = card["traits"]
                bits = [("Hair", t.get("hair")), ("Eyes", t.get("eyes")), ("Skin", t.get("skin")), ("Build", t.get("build")),
                        ("Attire", card.get("attire")), ("Emotional role", card.get("emotional_role"))]
            else:
                bits = [("Setting", (card.get("int_ext") or "") + " " + ", ".join(card.get("times") or [])),
                        ("Lighting", card.get("lighting")), ("Atmosphere", card.get("atmosphere")),
                        ("Description", card.get("description"))]
            st.markdown("<br>".join(f"<b>{_e(k)}:</b> {_e(v)}" for k, v in bits if v), unsafe_allow_html=True)
        new_prompt = st.text_area("Image prompt (edit it if you like)", value=card["prompt"], key=f"asset_prompt_{mode}_{sid}", height=96)
        up = st.file_uploader("Upload your own picture (.png / .jpg)" + (" - full length, standing" if card["kind"] == "character" else ""),
                              type=["png", "jpg", "jpeg"], key=f"asset_up_{mode}_{sid}")
        g1, g2 = st.columns(2)
        with g1:
            gen = st.button("Generate this picture", key=f"asset_gen_{mode}_{sid}")
        with g2:
            clear = st.button("Remove the picture", key=f"asset_clear_{mode}_{sid}", disabled=not card.get("image"))
        try:
            if new_prompt and new_prompt.strip() and new_prompt.strip() != card["prompt"]:
                book.set_prompt(cid, new_prompt)
                book.save()
            if up is not None:
                token = (getattr(up, "name", ""), len(up.getvalue()) if hasattr(up, "getvalue") else 0)
                if wiz["seen"].get(cid) != token:
                    res = am.add_upload(book, cid, up.name, up.getvalue())
                    wiz["seen"][cid] = token
                    book.save()
                    show_warnings(res["warnings"])
                    st.rerun()
            if clear:
                book.clear_image(cid)
                book.save()
                wiz["seen"].pop(cid, None)
                st.rerun()
            if gen:
                callback, bar = make_progress()
                rep = am.generate_images(book, [cid], backend=backend_key, only_missing=False,
                                         frame_size=tuple(wiz["plan"].get("frame_size") or (720, 1280)),
                                         progress=lambda f, m: callback(m, f))
                book.save()
                show_warnings(rep["warnings"])
                for why in rep["failed"].values():
                    st.error(f"No picture could be made: {why}")
                st.rerun()
        except am.AssetError as exc:
            st.error(str(exc))


def _story_step2(folder_id: str, mode: str, wkey: str, wiz: dict) -> None:
    """Step 2 - Interactive Asset Studio: a prompt card per character / location; upload pictures or generate them."""
    import vimax_orchestrator

    am = pipeline.asset_manager
    book = _asset_book(wiz)
    plan = wiz["plan"]
    n_c, n_l, miss = len(book.characters), len(book.locations), len(book.missing())
    st.markdown(stat_tiles([(len(plan["storyboard"]["shots"]), "Shots"), (n_c, "Characters"), (n_l, "Locations"),
                            (miss, "Pictures still needed")]), unsafe_allow_html=True)
    st.markdown("**Pre-production asset gate.** Rendering stays locked until every character and every location has a "
                "reference picture. Upload your own, or let the app generate them from the prompts below. The same "
                "picture is then used in every shot, so nobody changes face between cuts.")
    show_warnings(plan.get("warnings"))
    label = st.selectbox("How should pictures be generated?", list(ASSET_BACKENDS), key=f"asset_backend_{mode}")
    backend_key = ASSET_BACKENDS[label]
    wiz["backend"] = backend_key
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Generate Reference Visuals", type="primary", key="asset_gen_all", disabled=not miss,
                     help="Creates a picture for every card that has none yet."):
            try:
                callback, bar = make_progress()
                rep = am.generate_images(book, None, backend=backend_key, frame_size=tuple(plan.get("frame_size") or (720, 1280)),
                                         progress=lambda f, m: callback(m, f))
                book.save()
                show_warnings(rep["warnings"])
                for why in rep["failed"].values():
                    st.error(f"No picture could be made: {why}")
                st.rerun()
            except am.AssetError as exc:
                st.error(str(exc))
    with b2:
        if st.button("Back to step 1", key="wiz_back1"):
            wiz["step"] = 1
            st.rerun()
    st.markdown("#### Characters")
    for card in book.characters:
        _asset_card(book, card, mode, wiz, backend_key)
    st.markdown("#### Locations")
    for card in book.locations:
        _asset_card(book, card, mode, wiz, backend_key)
    if book.missing():
        st.info(f"{len(book.missing())} picture(s) still needed before you can continue.")
    if st.button("Lock the assets and continue to step 3", type="primary", key="wiz_to3", disabled=bool(book.missing())):
        try:
            vimax_orchestrator.close_asset_gate(plan, book)
            wiz["step"] = 3
            st.rerun()
        except am.AssetGateError as exc:
            st.error(str(exc))


def _story_step3(folder_id: str, mode: str, wkey: str, wiz: dict) -> None:
    """Step 3 - storyboard preview with the character / location pictures locked, then the final render."""
    import render_engine
    am = pipeline.asset_manager
    plan = wiz["plan"]
    book = _asset_book(wiz)
    shots = plan["storyboard"]["shots"]
    chars, locs = book.character_images(), book.location_images()
    gate_ok = not book.missing()
    st.markdown(stat_tiles([(len(shots), "Shots"), (f"{plan['storyboard']['total_duration']:.0f}s", "Planned length"),
                            (len(chars), "Characters locked"), (len(locs), "Locations locked")]), unsafe_allow_html=True)
    if not gate_ok:
        st.error("Some pictures are missing - go back to the Asset Studio.")
    st.markdown("#### Storyboard (assets locked)")
    for s in shots[:40]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                lp = locs.get(am.location_key(s.get("location")))
                if lp and Path(lp).is_file():
                    st.image(lp, width=90, caption=str(s.get("location") or "").title())
            with c2:
                for name in s.get("characters", [])[:3]:
                    if name in chars and Path(chars[name]).is_file():
                        st.image(chars[name], width=60, caption=name)
            with c3:
                cam = render_engine.camera_for_shot(s, seed=s["idx"])
                move = ("push-in" if cam["zoom"][1] > cam["zoom"][0] + 0.01 else "pull-out" if cam["zoom"][1] < cam["zoom"][0] - 0.01
                        else "pan" if cam["pan"][0] != cam["pan"][1] else "tilt" if cam["tilt"][0] != cam["tilt"][1] else "static")
                st.markdown(f"**{s['idx'] + 1}.** {_e(s['shot_type'])}, {_e(s['camera_angle'])}, camera: {move} - {s['end'] - s['start']:.1f}s")
                if s.get("line"):
                    st.caption(f"{s.get('speaker') or ''}: {s['line']}")
                elif s.get("action"):
                    st.caption(str(s["action"])[:140])
    if len(shots) > 40:
        st.caption(f"... and {len(shots) - 40} more shots.")
    voice_on = st.checkbox("Add a narrator voice (needs internet)", value=False, key="story_voice")
    meme_cfg = render_meme_panel("story")
    r1, r2 = st.columns(2)
    with r1:
        go = st.button("Render the episode", type="primary", key="wiz_render", disabled=not gate_ok)
    with r2:
        if st.button("Back to the Asset Studio", key="wiz_back2"):
            wiz["step"] = 2
            st.rerun()
    if go:
        try:
            callback, bar = make_progress()
            result = pipeline.run_story_pipeline(
                wiz["source"], progress=callback, render=True, voice=bool(voice_on), memes=meme_cfg, plan_result=plan,
                asset_book=book, asset_backend=wiz.get("backend", "draw"), **wiz["common"])
            st.session_state["story_result"] = result
            _wizard_cleanup(wiz)
            wiz["temp"] = []
            wiz["step"] = 4
            st.rerun()
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


def render_story_panel(folder_id: str, mode: str) -> None:
    """Idea2Video / Novel2Video / Script2Video as a 3-step wizard: story -> asset studio -> render (then Drive keeps it)."""
    wkey = f"wiz_{mode}"
    wiz = st.session_state.get(wkey)
    if wiz and wiz.get("step") == 4:  # finished: show the result, offer a new episode
        wiz = None
    step = wiz["step"] if wiz else 1
    st.markdown(wizard_steps_html(step), unsafe_allow_html=True)
    if step == 1:
        _story_step1(folder_id, mode, wkey)
    elif step == 2:
        _story_step2(folder_id, mode, wkey, wiz)
    else:
        _story_step3(folder_id, mode, wkey, wiz)
    if wiz:
        if st.button("Start over", key="wiz_reset"):
            _wizard_cleanup(wiz)
            st.session_state.pop(wkey, None)
            st.rerun()
    else:
        show_story_result(st.session_state.get("story_result"), mode)


def show_story_result(result, mode: str) -> None:
    if not result or result.get("mode") != mode:
        return
    show_warnings(result.get("warnings"))
    shots = result["storyboard"]["shots"]
    eps = result.get("episodes_out") or ([{"episode": 1, "title": result["plan"].get("title"), "video_path": result.get("video_path"),
                                           "shots": len(shots), "seconds": result["storyboard"]["total_duration"],
                                           "drive_link": result.get("drive_link"), "project_link": result.get("project_link")}]
                                         if result.get("video_path") else [])
    errors = [i for i in result["issues"] if i["severity"] == "error"]
    saved = len(result.get("drive_links") or [])
    st.markdown(stat_tiles([(len(eps) or 1, "Episodes"), (len(shots), "Shots (ep. 1)"),
                            (f"{result['storyboard']['total_duration']:.1f}s", "Length (ep. 1)"),
                            (len(result["characters"]), "Characters"),
                            ("Yes" if not errors else f"{len(errors)} left", "Consistency OK"),
                            (saved, "Videos backed up to Drive")]), unsafe_allow_html=True)
    for ep in eps:
        with st.container(border=True):
            st.markdown(f'<div class="cp-clip-head"><span class="cp-rank">Episode {int(ep["episode"])}</span></div>',
                        unsafe_allow_html=True)
            st.markdown(f"**{_e(ep.get('title') or 'Untitled')}** - {ep['shots']} shots, {ep['seconds']:.1f}s")
            if ep.get("video_path"):
                st.video(ep["video_path"])
            if ep.get("drive_link"):
                st.link_button("Open in Google Drive", ep["drive_link"])
            if ep.get("project_link"):
                st.link_button("Open the project files on Drive", ep["project_link"])
    with st.container(border=True):
        st.markdown('<div class="cp-clip-head"><span class="cp-rank">Characters</span></div>', unsafe_allow_html=True)
        for c in result["characters"]:
            st.markdown(f"**{_e(c['name'])}** - {_e(c['visual_identifier'])}")
    if result.get("assets"):
        with st.expander("Locked reference pictures (used in every shot)"):
            book_dir = result["assets"].get("dir")
            try:
                book = pipeline.asset_manager.AssetBook.load(book_dir)
                cols = st.columns(4)
                for n, card in enumerate(book.cards.values()):
                    with cols[n % 4]:
                        if card.get("image") and Path(card["image"]).is_file():
                            st.image(card["image"], width=110, caption=card["name"])
            except (OSError, ValueError, KeyError):
                st.caption(f"Pictures are in: {book_dir}")
            links = result.get("asset_links") or {}
            st.caption(f"{len(links)} asset file(s) backed up to Google Drive." if links else
                       "Asset pictures are saved with the project; pick a Drive folder in the sidebar to back them up too.")
            if result.get("conditioning"):
                st.caption(f"{len(result['conditioning'])} character appearances were drawn from their locked reference pictures.")
    with st.expander("Shot list (episode 1)", expanded=True):
        for s in shots:
            who = ", ".join(s["characters"]) or "-"
            st.markdown(f"**{s['idx'] + 1}.** {s['start']:.1f}-{s['end']:.1f}s - {_e(s['shot_type'])}, "
                        f"{_e(s['camera_angle'])}, {_e(s['camera_movement'])} - {_e(who)}")
            st.caption(s["camera_prompt"])
    with st.expander("Script (episode 1)"):
        st.code(result["script"], language=None)
    with st.expander("How the agents worked"):
        for t in result["trace"]:
            st.markdown(f"- **{_e(t['agent'])}** - {_e(t['note'])}")
        for i in result["issues"]:
            st.markdown(f"- {_e(i['severity'])}: {_e(i['message'])}")
        if result.get("drift"):
            worst = max(d["distance"] for d in result["drift"])
            st.markdown(f"- Appearance drift check: worst distance {worst:.2f} over {len(result['drift'])} appearances")
    st.caption(f"Project files saved in: {result['project_dir']}")


def render_lab(folder_id: str) -> None:
    section_header("AI Generator Lab", "Pick what you want to make. Everything runs on your computer's CPU.")
    mode_label = st.selectbox("Creation mode", list(LAB_MODES), key="lab_mode")
    mode = LAB_MODES[mode_label]
    st.caption(MODE_BLURB[mode])
    if mode == "clips":
        render_clip_tab(folder_id)
    elif mode == "idea2video":
        flavor = st.selectbox("Output style", list(IDEA_FLAVORS), key="idea_flavor")
        if IDEA_FLAVORS[flavor] == "classic":
            render_narrated_generator(folder_id)
        else:
            render_story_panel(folder_id, mode)
    else:
        render_story_panel(folder_id, mode)


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
    render_lab(folder_id)
    st.markdown(f'<div class="cp-foot">{APP_NAME} · everything runs locally · your files never leave this computer '
                'except the Drive backup you enable</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
