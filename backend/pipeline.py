"""
backend/pipeline.py

The two end-to-end workflows behind the Streamlit UI. All the real work lives
in the other backend modules; this file only wires them together and reports
progress, so it can be tested without a browser.

  run_vugola_pipeline : URL/file -> transcribe -> score highlights
                        -> face-tracked 9:16 crops -> Google Drive
  run_viewmax_pipeline: prompt -> script -> edge-tts voiceover
                        -> merge with background video -> Google Drive
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

try:  # package import (backend.pipeline) or plain module (pipeline)
    from .avatar_sync import create_avatar_video, load_avatar
    from .scorer import find_viral_clips, generate_script
    from .transcriber import transcribe_audio
    from .tts_engine import DEFAULT_VOICE, generate_voiceover
    from .uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from .video_engine import create_vertical_short, merge_audio_with_video
except ImportError:  # pragma: no cover
    from avatar_sync import create_avatar_video, load_avatar
    from scorer import find_viral_clips, generate_script
    from transcriber import transcribe_audio
    from tts_engine import DEFAULT_VOICE, generate_voiceover
    from uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from video_engine import create_vertical_short, merge_audio_with_video

logger = logging.getLogger(__name__)

ProgressCb = Callable[[str, float], None]

NO_FOLDER_WARNING = "No Google Drive Folder ID set - the finished file was NOT uploaded."
NO_AUTH_WARNING = (
    "Google Drive is not connected (no token.json or credentials.json found) - the finished "
    "file was saved on this computer but NOT uploaded."
)
NO_FACE_WARNING = (
    "No face was detected, so clips use a plain centre crop. Face tracking works best when "
    "the speaker's face is clearly visible."
)


def _usable_folder(folder_id: Optional[str], warnings: list) -> Optional[str]:
    """The folder to upload to, or None (with a warning) when Drive can't be used.
    Keeps the app working on a laptop that has no Google sign-in files."""
    if not folder_id:
        warnings.append(NO_FOLDER_WARNING)
        return None
    if drive_status() == "unavailable":
        warnings.append(NO_AUTH_WARNING)
        return None
    return folder_id


def _emit(progress: Optional[ProgressCb], stage: str, fraction: float) -> None:
    if progress:
        progress(stage, max(0.0, min(1.0, fraction)))


def _safe_name(text: str, limit: int = 50) -> str:
    return re.sub(r"[^\w\-.]+", "_", text).strip("_")[:limit] or "clip"


def _run_async(coro):
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# --------------------------------------------------------------------------- #
# Tab 1 - Clip Repurposer Studio
# --------------------------------------------------------------------------- #
def run_vugola_pipeline(
    source: str,
    folder_id: Optional[str],
    top_n: int = 3,
    min_len: float = 15.0,
    max_len: float = 45.0,
    output_dir: str = "outputs",
    progress: Optional[ProgressCb] = None,
    ffmpeg_path: Optional[str] = None,
    detector=None,
    allow_center_fallback: bool = False,
) -> dict:
    """Find the best highlights in `source` and export each as a vertical short.

    Returns {"source_path", "word_count", "clips": [...], "warnings": [...]}.
    Each clip: start, end, duration, score, reasons, text, output_path,
    drive_link, error (None on success). One failing clip does not stop the rest.
    """
    warnings = []
    folder_id = _usable_folder(folder_id, warnings)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress, "Getting the video", 0.02)
    local = download_media(source, str(out_dir / "downloads"))

    _emit(progress, "Transcribing locally (this can take a while on CPU)", 0.10)
    words = transcribe_audio(local)
    if not words:
        raise RuntimeError("No speech was detected in this video, so there is nothing to score.")

    _emit(progress, "Scoring highlights", 0.45)
    picks = find_viral_clips(words, top_n=top_n, min_len=min_len, max_len=max_len)
    if not picks:
        raise RuntimeError("Could not find any highlight ranges in this transcript.")

    stem = _safe_name(Path(local).stem)
    clips = []
    pending = []  # (clip, future) - Drive uploads running in the background
    backup = DriveBackup(folder_id, upload_fn=upload_to_google_drive) if folder_id else None
    try:
        for idx, pick in enumerate(picks, start=1):
            _emit(progress, f"Cropping clip {idx} of {len(picks)}" + (" (earlier clips upload in the background)" if backup and idx > 1 else ""),
                  0.50 + 0.40 * (idx - 1) / len(picks))
            clip = {**pick, "output_path": None, "drive_link": None, "error": None, "faces_found": None}
            target = out_dir / f"{stem}_clip{idx}_{int(pick['start'])}-{int(pick['end'])}.mp4"
            try:
                res = create_vertical_short(
                    local, str(target),
                    start_time=pick["start"], end_time=pick["end"],
                    ffmpeg_path=ffmpeg_path, detector=detector,
                    allow_center_fallback=allow_center_fallback,
                    upload=False,  # the background DriveBackup uploads it below
                )
                clip["output_path"] = res["output_path"]
                clip["faces_found"] = res["faces_found"]
                if backup:
                    pending.append((clip, backup.submit(res["output_path"])))
            except Exception as exc:  # keep going; report per clip
                logger.exception("Clip %d failed", idx)
                clip["error"] = str(exc)
            clips.append(clip)

        if pending:
            _emit(progress, "Finishing the Google Drive backup", 0.92)
        for clip, future in pending:
            try:
                clip["drive_link"] = future.result()
            except Exception as exc:  # the clip stays safe on this computer
                logger.warning("Drive backup failed for %s: %s", clip["output_path"], exc)
                clip["error"] = str(exc)
    finally:
        if backup:
            backup.close()

    made = [c for c in clips if c["faces_found"] is not None]
    if made and all(c["faces_found"] == 0 for c in made):
        warnings.append(NO_FACE_WARNING)

    _emit(progress, "Done", 1.0)
    return {"source_path": local, "word_count": len(words), "clips": clips, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Tab 2 - AI Generator Lab
# --------------------------------------------------------------------------- #
def run_viewmax_pipeline(
    prompt: str,
    folder_id: Optional[str],
    background_video: Optional[str] = None,
    voice: str = DEFAULT_VOICE,
    output_dir: str = "outputs",
    progress: Optional[ProgressCb] = None,
    ffmpeg_path: Optional[str] = None,
    avatar_image: Optional[str] = None,
    avatar_enabled: bool = False,
) -> dict:
    """Prompt -> script -> voiceover -> merged vertical video -> Drive.

    With avatar_enabled, the video shows `avatar_image` talking in sync with the
    voiceover (backend/avatar_sync.py) instead of a plain audio+background merge.

    Returns {"script": {...}, "audio_path", "video_path", "drive_link", "avatar", "warnings"}.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a prompt first.")
    if avatar_enabled:
        if not avatar_image:
            raise ValueError("Upload an avatar image (.jpg or .png) to use AI Avatar Lip-Sync.")
        load_avatar(avatar_image)  # fail fast on a wrong/unreadable file, before any voiceover work
    warnings = []
    folder_id = _usable_folder(folder_id, warnings)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())

    _emit(progress, "Writing the script", 0.05)
    script = generate_script(prompt)

    _emit(progress, "Generating the voiceover (edge-tts, needs internet)", 0.25)
    audio_path = out_dir / f"voice_{stamp}.mp3"
    audio_path = _run_async(generate_voiceover(script["script"], str(audio_path), voice=voice))

    out_name = str(out_dir / f"viewmax_{_safe_name(script['topic'], 30)}_{stamp}.mp4")
    video_path, avatar_info = None, None
    if avatar_enabled:
        _emit(progress, "Animating the avatar (lip-sync)", 0.50)
        try:
            avatar_info = create_avatar_video(
                avatar_image, audio_path, out_name, auto_words=True, background_video=background_video,
                ffmpeg_path=ffmpeg_path,
                progress=lambda f: _emit(progress, "Animating the avatar (lip-sync)", 0.50 + 0.33 * f),
            )
            video_path = avatar_info["output_path"]
            warnings.extend(avatar_info["warnings"])
        except Exception as exc:  # still hand back a video rather than nothing
            logger.warning("avatar lip-sync failed: %s", exc)
            warnings.append(f"Avatar lip-sync failed ({exc}); made a plain video instead.")
            avatar_info = None
    if video_path is None:
        _emit(progress, "Merging audio and video", 0.60)
        video_path = merge_audio_with_video(
            audio_path, out_name, background_video=background_video, ffmpeg_path=ffmpeg_path,
        )

    drive_link = None
    if folder_id:
        _emit(progress, "Backing up to Google Drive", 0.85)
        try:
            with DriveBackup(folder_id, upload_fn=upload_to_google_drive) as backup:
                drive_link = backup.submit(video_path).result()
        except Exception as exc:  # never lose the finished video because Drive is unreachable
            logger.warning("Drive backup failed: %s", exc)
            warnings.append(f"Google Drive backup failed ({exc}). Your video is saved on this computer.")

    _emit(progress, "Done", 1.0)
    return {
        "script": script, "audio_path": audio_path, "video_path": video_path,
        "drive_link": drive_link, "avatar": avatar_info, "warnings": warnings,
    }
