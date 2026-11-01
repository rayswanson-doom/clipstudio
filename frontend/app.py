"""
frontend/app.py  -  Streamlit UI

Run:  streamlit run frontend/app.py

Tab 1  Vugola AI Mode : YouTube URL / file -> local transcription -> local
       highlight scoring -> MediaPipe face-tracked crop -> Google Drive backup
Tab 2  Viewmax Mode   : text prompt -> script -> edge-tts voiceover
       -> video merge -> Google Drive upload
Sidebar: Google Drive Folder ID (plus Google sign-in status).
"""

from __future__ import annotations

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

def _output_dir() -> Path:
    """outputs/ next to the exe when packaged (or the temp dir if that folder is
    read-only, e.g. under Program Files); the project folder otherwise."""
    base = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else ROOT
    target = base / "outputs"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ClipStudio" / "outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


OUTPUT_DIR = _output_dir()
UPLOAD_DIR = Path(tempfile.gettempdir()) / "clipstudio_uploads"
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
VOICES = [
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-AU-NatashaNeural", "en-IN-NeerjaNeural",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def save_upload(uploaded) -> str:
    """Write a Streamlit UploadedFile to disk and return its path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{int(time.time())}_{Path(uploaded.name).name}"
    dest.write_bytes(uploaded.getbuffer())
    return str(dest)


def make_progress():
    """Return (callback, bar) wiring pipeline progress to a Streamlit bar."""
    bar = st.progress(0.0, text="Starting...")

    def callback(stage: str, fraction: float) -> None:
        bar.progress(fraction, text=stage)

    return callback, bar


def show_warnings(warnings) -> None:
    for w in warnings or []:
        st.warning(w)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    with st.sidebar:
        st.header("Settings")
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
        st.caption("Everything runs locally, free, with no API keys. "
                   "Only the Drive upload and edge-tts voices use the internet.")
        if getattr(sys, "frozen", False):  # packaged exe has no console window to close
            if st.button("Quit Clip Studio", key="quit_app"):
                st.write("Closing Clip Studio. You can close this tab.")
                os._exit(0)
    return folder_id


# --------------------------------------------------------------------------- #
# tab 1 - Vugola AI mode
# --------------------------------------------------------------------------- #
def render_vugola_tab(folder_id: str) -> None:
    st.subheader("Vugola AI Mode")
    st.caption("YouTube URL or video file → local transcription → highlight scoring "
               "→ face-tracked 9:16 crop → automatic Google Drive backup")

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

    if st.button("Generate viral clips", type="primary", key="vugola_run"):
        if not url.strip() and uploaded is None:
            st.warning("Paste a YouTube URL or upload a video first.")
        elif min_len > max_len:
            st.error("Min length must not be greater than max length.")
        else:
            callback, bar = make_progress()
            try:
                source = save_upload(uploaded) if uploaded is not None else url.strip()
                st.session_state["vugola_result"] = pipeline.run_vugola_pipeline(
                    source, folder_id, top_n=top_n, min_len=min_len, max_len=max_len,
                    output_dir=str(OUTPUT_DIR), progress=callback, allow_center_fallback=fallback,
                )
            except Exception as exc:
                st.session_state.pop("vugola_result", None)
                bar.empty()
                st.error(f"Something went wrong: {exc}")

    result = st.session_state.get("vugola_result")
    if result:
        show_warnings(result.get("warnings"))
        st.success(f"Found {len(result['clips'])} clip(s) from {result['word_count']} transcribed words.")
        for i, clip in enumerate(result["clips"], start=1):
            with st.container(border=True):
                left, right = st.columns([1, 2])
                with left:
                    if clip.get("output_path"):
                        st.video(clip["output_path"])
                with right:
                    st.markdown(f"**Clip {i}** · score {clip['score']} · "
                                f"{clip['start']}s - {clip['end']}s ({clip['duration']}s)")
                    if clip.get("reasons"):
                        st.caption("Why: " + ", ".join(clip["reasons"]))
                    st.write(clip["text"])
                    if clip.get("drive_link"):
                        st.link_button("Open in Google Drive", clip["drive_link"])
                    if clip.get("error"):
                        st.error(clip["error"])


# --------------------------------------------------------------------------- #
# tab 2 - Viewmax mode
# --------------------------------------------------------------------------- #
def render_viewmax_tab(folder_id: str) -> None:
    st.subheader("Viewmax Mode")
    st.caption("Text prompt → script generator → local edge-tts voiceover "
               "→ video merge → Google Drive upload")

    prompt = st.text_area(
        "What should the video be about?", key="viewmax_prompt", height=120,
        placeholder="Write a motivational 30 second TikTok script about building a morning routine",
    )
    background = st.file_uploader(
        "Background video (optional - a plain dark background is used if empty)",
        type=VIDEO_TYPES, key="viewmax_bg",
    )
    voice = st.selectbox("Voice", VOICES, index=0, key="viewmax_voice")

    if st.button("Generate video", type="primary", key="viewmax_run"):
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            callback, bar = make_progress()
            try:
                bg_path = save_upload(background) if background is not None else None
                st.session_state["viewmax_result"] = pipeline.run_viewmax_pipeline(
                    prompt, folder_id, background_video=bg_path, voice=voice,
                    output_dir=str(OUTPUT_DIR), progress=callback,
                )
            except Exception as exc:
                st.session_state.pop("viewmax_result", None)
                bar.empty()
                st.error(f"Something went wrong: {exc}")

    result = st.session_state.get("viewmax_result")
    if result:
        show_warnings(result.get("warnings"))
        script = result["script"]
        with st.expander("Generated script", expanded=True):
            st.write(script["script"])
            st.caption(f"{script['word_count']} words, about {script['estimated_seconds']}s")
        st.audio(result["audio_path"])
        st.video(result["video_path"])
        if result.get("drive_link"):
            st.link_button("Open in Google Drive", result["drive_link"])


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Clip Studio", page_icon="🎬", layout="wide")
    st.title("Clip Studio")
    folder_id = render_sidebar()
    tab_vugola, tab_viewmax = st.tabs(["Vugola AI Mode", "Viewmax Mode"])
    with tab_vugola:
        render_vugola_tab(folder_id)
    with tab_viewmax:
        render_viewmax_tab(folder_id)


if __name__ == "__main__":
    main()
