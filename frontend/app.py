"""
frontend/app.py  -  CinePulse AI  (Streamlit UI)  -  Viral Clip Studio

Run:  streamlit run frontend/app.py          (or double-click the packaged exe)

  YouTube URL / video file -> local transcription (faster-whisper, CPU, word timestamps) -> highlight scoring
  -> MediaPipe face-tracked 9:16 crop -> burned-in word-accurate subtitles -> contextual meme reactions (optional)
  -> trending YouTube background music ducked under the speaker (optional) -> YouTube / Instagram / Facebook
  captions + tags -> Google Drive backup (optional).
  Sidebar (collapsible): Google Drive Folder ID + connection status.

Heavy libraries (faster-whisper, mediapipe, cv2, yt-dlp) are imported lazily inside the backend, only when you click
Generate or Scan, so the page opens instantly on a low-spec laptop. All styling is plain CSS injected below; no external
fonts or network assets, so it works offline.

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

import music_finder  # noqa: E402
import pipeline  # noqa: E402
import subtitle_engine  # noqa: E402
import uploader  # noqa: E402

APP_NAME = "CinePulse AI"


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


def _music_cache_dir() -> str:
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
    target = base / "storage" / "music_cache"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return str(target)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "CinePulseAI" / "music_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


OUTPUT_DIR = _output_dir()
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cinepulse_uploads"
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "aac"]
OCR_CHOICES = {"Auto (PaddleOCR if installed)": "auto", "PaddleOCR": "paddleocr", "RapidOCR (built in)": "rapidocr"}
SUBTITLE_SIZES = list(subtitle_engine.SIZE_PRESETS)
SUBTITLE_COLORS = list(subtitle_engine.COLOR_PRESETS)
CAPTION_WRITERS = {"Offline (no key needed)": "off", "Auto: free Groq / Gemini key if I set one": "auto"}
DEFAULT_MUSIC_QUERY = music_finder.DEFAULT_QUERY

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


DEMO_SOCIAL = {
    "keywords": ["fitness", "build muscle", "protein"], "hook": "Protein and sleep matter more than the gym.", "source": "offline",
    "warnings": [],
    "youtube": {"title": "Protein and sleep matter more than the gym", "description": "Protein and sleep matter more than the gym.\n\n"
                "Subscribe for more clips like this.\n\n#fitness #buildmuscle #Shorts", "tags": ["fitness", "build muscle", "protein", "Shorts"],
                "hashtags": ["#fitness", "#buildmuscle", "#Shorts"],
                "copy_text": "Protein and sleep matter more than the gym\n\nSubscribe for more clips like this.\n\nTags: fitness, build muscle, protein, Shorts"},
    "instagram": {"hook": "Protein and sleep matter more than the gym.", "caption": "Protein and sleep matter more than the gym.\n\nSave this & send it to a friend.",
                  "cta": "Save this & send it to a friend.", "hashtags": ["#fitness", "#buildmuscle", "#protein", "#reels"],
                  "copy_text": "Protein and sleep matter more than the gym.\n\nSave this & send it to a friend.\n.\n.\n#fitness #buildmuscle #protein #reels"},
    "facebook": {"caption": "Protein and sleep matter more than the gym. Has this ever happened to you?", "hashtags": ["#fitness", "#protein"],
                 "copy_text": "Protein and sleep matter more than the gym. Has this ever happened to you?\n\n#fitness #protein"},
}
DEMO_CLIPS = {
    "source_path": "demo.mp4", "word_count": 1842, "warnings": [],
    "music": {"title": "Demo track", "channel": "Demo channel", "url": "https://www.youtube.com/", "source": "youtube",
              "licence_note": music_finder.LICENCE_NOTE},
    "clips": [
        {"start": 42.0, "end": 71.5, "duration": 29.5, "score": 14.8, "faces_found": 884,
         "reasons": ["question", "emotional_keywords", "dramatic_pause"], "output_path": None,
         "drive_link": "https://drive.google.com/", "error": None, "subtitled": True, "music_mixed": True,
         "social": DEMO_SOCIAL,
         "text": "Did you know the biggest mistake nobody warns you about is trying to do it all at once?"},
        {"start": 190.0, "end": 228.0, "duration": 38.0, "score": 11.2, "faces_found": 0,
         "reasons": ["number_or_stat", "direct_address"], "output_path": None, "drive_link": None,
         "error": "Upload skipped: Google Drive is not connected.", "subtitled": False, "music_mixed": False, "social": DEMO_SOCIAL,
         "text": "In three weeks you can change your whole routine with just 5 minutes a day."},
    ],
}


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div class="cp-brand"><div class="logo">🎬</div><div><b>{APP_NAME}</b>'
            '<small>Viral Clip Studio</small></div></div>',
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
        st.caption("Only YouTube downloads, the trending-music search, optional AI caption polish and the Drive upload "
                   "use the internet.")
        if getattr(sys, "frozen", False):  # packaged exe has no console window to close
            if st.button("Quit CinePulse AI", key="quit_app"):
                st.write("Closing CinePulse AI. You can close this tab.")
                os._exit(0)
    return folder_id


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def render_social_tabs(i: int, social: dict | None) -> None:
    """YouTube / Instagram / Facebook post packages; st.code gives every block a copy button."""
    if not social:
        st.caption("No captions were generated for this clip.")
        return
    t_yt, t_ig, t_fb = st.tabs(["▶ YouTube Shorts", "📸 Instagram Reels", "👍 Facebook Reels"])
    yt, ig, fb = social["youtube"], social["instagram"], social["facebook"]
    with t_yt:
        st.caption(f"Title ({len(yt['title'])}/60 characters)")
        st.code(yt["title"], language=None)
        st.caption("Description")
        st.code(yt["description"], language=None)
        st.caption(f"Tags ({len(yt['tags'])})")
        st.code(", ".join(yt["tags"]), language=None)
    with t_ig:
        st.caption(f"Caption + hashtags ({len(ig['hashtags'])} hashtags)")
        st.code(ig["copy_text"], language=None)
    with t_fb:
        st.caption(f"Caption + hashtags ({len(fb['hashtags'])} hashtags)")
        st.code(fb["copy_text"], language=None)
    if social.get("source") not in (None, "offline"):
        st.caption(f"Wording polished by {social['source']} (free tier).")


def render_clip_card(i: int, clip: dict) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            path = clip.get("output_path")
            if path and os.path.isfile(path):
                st.video(path)
                try:
                    with open(path, "rb") as fh:
                        st.download_button("⬇ Download clip", fh.read(), file_name=Path(path).name, mime="video/mp4",
                                           key=f"dl_{i}_{Path(path).name}", use_container_width=True)
                except OSError as exc:
                    st.caption(f"Download unavailable ({exc}).")
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
            badges.append(badge("subtitled" if clip.get("subtitled") else "no subtitles", "ok" if clip.get("subtitled") else "warn"))
            if clip.get("music_mixed"):
                badges.append(badge("♪ music", "ok"))
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
            elif clip.get("error"):
                st.error(clip["error"])
            files = clip.get("subtitle_files")
            if files:
                st.caption(f"Subtitles saved next to the clip: {Path(files['srt']).name}, {Path(files['ass']).name}")
        render_social_tabs(i, clip.get("social"))


MEME_MODES = {"Auto (cut away on big reactions, otherwise a corner tile)": "auto", "Corner tile only": "overlay",
              "Full-screen cutaway only": "cutaway"}
MEME_POSITIONS = ["Auto (avoid other tiles)"] + list(pipeline.PIP_PRESETS)


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


def _track_label(t: dict) -> str:
    views = t.get("view_count") or 0
    v = f"{views / 1e6:.1f}M" if views >= 1e6 else f"{views / 1e3:.0f}K" if views >= 1e3 else str(views)
    mins, secs = divmod(int(t.get("duration") or 0), 60)
    when = f" · {t['upload_date']}" if t.get("upload_date") else ""
    return f"{t.get('title', '?')[:70]} - {t.get('channel', '')[:28]} · {v} views{when} · {mins}:{secs:02d}"


def render_music_panel() -> dict | None:
    """Trending-YouTube background music: scan, choose among the top 5 (or take the top one / your own file), volume."""
    with st.expander("🎵 Background music", expanded=False):
        on = st.checkbox("Add Trending YouTube Background Music", value=False, key="music_on",
                         help="Searches YouTube (no API key) for fresh, fast-growing tracks, downloads the one you pick and "
                              "mixes it quietly UNDER the speaker, lowering itself while they talk.")
        query = st.text_input("Mood or niche", value=DEFAULT_MUSIC_QUERY, key="music_query", disabled=not on,
                              help="For example: lofi chill, epic motivational, upbeat fitness, cinematic.")
        c1, c2 = st.columns(2)
        with c1:
            volume = st.slider("Music volume (%)", 0, 60, 15, key="music_volume", disabled=not on,
                               help="How loud the music is. 10-20 % sits under speech.")
        with c2:
            duck = st.checkbox("Lower the music while someone speaks", value=True, key="music_duck", disabled=not on)
        if st.button("🔎 Scan Trending Tracks", key="music_scan", disabled=not on):
            with st.spinner("Asking YouTube for trending tracks..."):
                try:
                    st.session_state["music_tracks"] = music_finder.scan_trending_music(query, limit=5)
                    st.session_state["music_scan_query"] = query
                except music_finder.MusicError as exc:
                    st.session_state.pop("music_tracks", None)
                    st.warning(str(exc))
        tracks = st.session_state.get("music_tracks") or []
        pick = None
        if tracks:
            labels = [_track_label(t) for t in tracks]
            idx = st.radio("Pick a track (the best one is first)", range(len(tracks)), format_func=lambda k: labels[k],
                           key="music_pick", disabled=not on)
            pick = tracks[idx]
            st.markdown(f"[Listen on YouTube]({pick['url']})")
        own = st.file_uploader("...or use your own music file", type=AUDIO_TYPES, key="music_file", disabled=not on)
        st.caption(music_finder.LICENCE_NOTE)
    if not on:
        return None
    cfg = {"enabled": True, "volume": volume / 100.0, "duck": bool(duck), "query": query.strip() or DEFAULT_MUSIC_QUERY,
           "cache_dir": _music_cache_dir()}
    if own is not None:
        cfg["path"] = save_upload(own)
    elif pick:
        cfg["track"] = pick
    return cfg


def render_subtitle_panel() -> dict | None:
    with st.expander("💬 Burned-in subtitles", expanded=False):
        on = st.checkbox("Burn-in Subtitles", value=True, key="subs_on",
                         help="Word-accurate captions in 3-5 word chunks, bold with a black outline, kept above the area that "
                              "the Reels / Shorts buttons cover. An .srt and .ass are saved next to each clip.")
        c1, c2 = st.columns(2)
        with c1:
            size = st.selectbox("Font size", SUBTITLE_SIZES, index=SUBTITLE_SIZES.index(subtitle_engine.DEFAULT_SIZE_PRESET),
                                key="subs_size", disabled=not on)
        with c2:
            color = st.selectbox("Colour", SUBTITLE_COLORS, index=SUBTITLE_COLORS.index(subtitle_engine.DEFAULT_COLOR_PRESET),
                                 key="subs_color", disabled=not on)
        upper = st.checkbox("UPPERCASE", value=False, key="subs_upper", disabled=not on)
    return {"enabled": True, "size": size, "color": color, "uppercase": bool(upper)} if on else None


def render_clip_tab(folder_id: str) -> None:
    section_header("Viral Clip Studio",
                   "Turn a long video into vertical clips with subtitles, trending music and ready-to-post captions.")
    st.markdown(steps_html(["Source", "Transcribe", "Score highlights", "Face-tracked crop", "Subtitles + music", "Captions + Drive"]),
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

    subs_cfg = render_subtitle_panel()
    music_cfg = render_music_panel()
    meme_cfg = render_meme_panel("vugola")

    with st.expander("📝 Captions & tags"):
        topics = st.text_input("Topic keywords (optional, comma separated)", key="social_topics",
                               placeholder="fitness, nutrition",
                               help="Steers the titles, tags and hashtags. Without them they are worked out from what is said.")
        writer_label = st.selectbox("Caption writer", list(CAPTION_WRITERS), key="social_writer",
                                    help="Offline works with no key. If you set GROQ_API_KEY or GEMINI_API_KEY (both have free tiers) "
                                         "the wording can be polished; platform limits are enforced either way.")
    drive_on = st.checkbox("Auto-Upload to Google Drive", value=bool(folder_id), key="drive_on",
                           help="Uploads each finished clip to the Folder ID in the sidebar. Needs the Google sign-in files.")
    if drive_on and not folder_id:
        st.info("Enter a Google Drive Folder ID in the sidebar, or the clips are only saved on this computer.")

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
                if music_cfg and music_cfg.get("path"):
                    temp.append(music_cfg["path"])
                st.session_state["vugola_result"] = pipeline.run_vugola_pipeline(
                    source, folder_id, top_n=top_n, min_len=min_len, max_len=max_len,
                    output_dir=str(OUTPUT_DIR), progress=callback, allow_center_fallback=fallback,
                    read_screen_text=ocr_on, memes=meme_cfg, subtitles=subs_cfg, music=music_cfg, upload=bool(drive_on),
                    social={"topic_keywords": [t.strip() for t in topics.split(",") if t.strip()],
                            "use_llm": CAPTION_WRITERS[writer_label]},
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
        music = result.get("music")
        if music:
            who = f" - {music['channel']}" if music.get("channel") else ""
            link = f" ([source]({music['url']}))" if music.get("url") else ""
            st.info(f"♪ Background music: **{music['title']}**{who}{link}. {music.get('licence_note', '')}")
        for i, clip in enumerate(clips, start=1):
            render_clip_card(i, clip)


# --------------------------------------------------------------------------- #
def render_hero() -> None:
    st.markdown(
        '<div class="cp-hero"><div class="logo">🎬</div><div class="grow">'
        f'<h1>CinePulse <span>AI</span></h1>'
        '<p>Viral Clip Studio: cut long videos into vertical shorts with burned-in subtitles, trending background '
        'music and ready-to-post captions, right on your own computer.</p></div>'
        '<div class="cp-pills"><span class="cp-pill"><span class="cp-dot"></span>Local &amp; private</span>'
        '<span class="cp-pill">Free · no API keys</span></div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🎬", layout="wide", initial_sidebar_state="auto")
    inject_css()
    if _demo_mode():  # ?demo=1 -> sample cards for previewing the layout
        st.session_state.setdefault("vugola_result", DEMO_CLIPS)
    folder_id = render_sidebar()
    render_hero()
    render_clip_tab(folder_id)
    st.markdown(f'<div class="cp-foot">{APP_NAME} · everything runs locally · your files never leave this computer '
                'except the Drive backup you enable</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
