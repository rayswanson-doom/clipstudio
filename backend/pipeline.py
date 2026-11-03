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
import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

try:  # package import (backend.pipeline) or plain module (pipeline)
    from .avatar_sync import create_avatar_video, load_avatar
    from . import ocr_engine
    from . import transcriber as _transcriber_mod
    from .visual_cues import apply_cues_to_video, has_effects
    from .scorer import align_cues, build_narration, find_viral_clips
    from .transcriber import transcribe_audio
    from .tts_engine import DEFAULT_VOICE, generate_voiceover
    from .uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from .video_engine import create_vertical_short, merge_audio_with_video
except ImportError:  # pragma: no cover
    from avatar_sync import create_avatar_video, load_avatar
    import ocr_engine
    import transcriber as _transcriber_mod
    from visual_cues import apply_cues_to_video, has_effects
    from scorer import align_cues, build_narration, find_viral_clips
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
def release_memory(unload_models: bool = True) -> None:
    """Hand RAM back right after a run (low-spec laptops): drop the Whisper and OCR models, collect garbage.
    Models load lazily, so the next run simply loads them again."""
    if unload_models:
        for fn in (_transcriber_mod.unload_model, ocr_engine.unload):
            try:
                fn()
            except Exception:  # never let cleanup break a finished run
                logger.debug("model unload failed", exc_info=True)
    gc.collect()


def _remove(path) -> None:
    try:
        if path and os.path.isfile(path):
            os.unlink(path)
    except OSError:
        logger.debug("could not remove %s", path)


def _read_screen_text(video, interval, max_frames, warnings, progress, stage, lo, hi) -> list:
    """OCR a video; any failure becomes a warning (the feature is optional). Frees the OCR engine after."""
    try:
        return ocr_engine.extract_screen_text(
            video, interval=interval, max_frames=max_frames,
            progress=lambda f: _emit(progress, stage, lo + (hi - lo) * f),
        )
    except Exception as exc:
        logger.warning("On-screen text reading failed: %s", exc)
        warnings.append(f"Could not read on-screen text ({exc}). Continuing without it.")
        return []
    finally:
        ocr_engine.unload()


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
    read_screen_text: bool = False,
    ocr_interval: float = 1.0,
    keep_download: bool = False,
    keep_models: bool = False,
) -> dict:
    """Find the best highlights in `source` and export each as a vertical short.

    read_screen_text   also OCR the video every `ocr_interval` s and append the burned-in text
                       (subtitles, titles) to the transcript. If the video has NO speech but does
                       have on-screen text, that text is used to pick the highlights.
    keep_download      keep the YouTube download in outputs/downloads (default: delete it, it is
                       only a working copy and can be huge)
    keep_models        keep Whisper/OCR in RAM after the run (default: free them at once)

    Returns {"source_path", "word_count", "clips": [...], "transcript": [...], "screen_text": [...],
             "warnings": [...]}.
    Each clip: start, end, duration, score, reasons, text, screen_text, output_path,
    drive_link, error (None on success). One failing clip does not stop the rest.
    """
    warnings = []
    folder_id = _usable_folder(folder_id, warnings)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = None
    downloaded = False
    try:
        _emit(progress, "Getting the video", 0.02)
        local = download_media(source, str(out_dir / "downloads"))
        downloaded = os.path.abspath(local) != os.path.abspath(str(source))

        _emit(progress, "Transcribing locally (this can take a while on CPU)", 0.10)
        words = transcribe_audio(local)
        if not keep_models:
            _transcriber_mod.unload_model()  # Whisper is not needed again: free its RAM now

        screen_segments: list = []
        if read_screen_text:
            _emit(progress, "Reading on-screen text (OCR)", 0.30)
            screen_segments = _read_screen_text(local, ocr_interval, ocr_engine.DEFAULT_MAX_FRAMES, warnings,
                                                progress, "Reading on-screen text (OCR)", 0.30, 0.44)
        spoken_words = words
        if not words and screen_segments:
            words = ocr_engine.segments_to_words(screen_segments)
            warnings.append("No speech was detected, so the text read from the screen was used to pick highlights.")
        if not words:
            raise RuntimeError("No speech was detected in this video, so there is nothing to score.")
        transcript = ocr_engine.merge_transcript(spoken_words, screen_segments)

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
                clip = {**pick, "output_path": None, "drive_link": None, "error": None, "faces_found": None,
                        "screen_text": ocr_engine.screen_text_in_range(screen_segments, pick["start"], pick["end"])}
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
        return {"source_path": local, "word_count": len(words), "clips": clips, "transcript": transcript,
                "screen_text": screen_segments, "warnings": warnings}
    finally:
        if downloaded and not keep_download:
            _remove(local)  # temporary working copy of the download
        release_memory(not keep_models)


# --------------------------------------------------------------------------- #
# Tab 2 - AI Generator Lab
# --------------------------------------------------------------------------- #
def _media_seconds(path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    finally:
        cap.release()


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
    custom_script: Optional[str] = None,
    action_prompt: Optional[str] = None,
    read_background_text: bool = False,
    keep_models: bool = False,
) -> dict:
    """Prompt / your own script -> voiceover -> video (+ avatar, + visual action cues) -> Drive.

    custom_script         narrate exactly this text (the prompt is then optional)
    action_prompt         describes what happens on screen ("zoom in on the logo, then pan right");
                          rendered as camera moves in time with the narration
    read_background_text  OCR the background video and narrate the text found in it after the script
    avatar_enabled        show `avatar_image` talking in sync with the voiceover (avatar_sync.py)

    Returns {"script": {...narration incl. cues + shot_list...}, "audio_path", "video_path",
             "drive_link", "avatar", "cues_applied", "screen_text", "warnings"}.
    """
    if not (prompt or "").strip() and not (custom_script or "").strip():
        raise ValueError("Please enter a prompt first (or paste your own script).")
    if avatar_enabled:
        if not avatar_image:
            raise ValueError("Upload an avatar image (.jpg or .png) to use AI Avatar Lip-Sync.")
        load_avatar(avatar_image)  # fail fast on a wrong/unreadable file, before any voiceover work
    warnings = []
    folder_id = _usable_folder(folder_id, warnings)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    try:
        screen_lines: list = []
        if read_background_text:
            if not background_video:
                warnings.append("Add a background video to read its on-screen text.")
            else:
                _emit(progress, "Reading on-screen text from the background video (OCR)", 0.03)
                segs = _read_screen_text(background_video, 1.0, 180, warnings, progress,
                                         "Reading on-screen text from the background video (OCR)", 0.03, 0.12)
                screen_lines = ocr_engine.screen_text_in_range(segs, 0.0, float("inf"))
                if not screen_lines and not any("Could not read" in w for w in warnings):
                    warnings.append("No on-screen text was found in the background video.")

        _emit(progress, "Preparing the narration", 0.14)
        script = build_narration(prompt, custom_script, action_prompt, screen_text=screen_lines)

        _emit(progress, "Generating the voiceover (edge-tts, needs internet)", 0.25)
        audio_path = out_dir / f"voice_{stamp}.mp3"
        audio_path = _run_async(generate_voiceover(script["script"], str(audio_path), voice=voice))

        out_name = str(out_dir / f"viewmax_{_safe_name(script['topic'], 30)}_{stamp}.mp4")
        video_path, avatar_info = None, None
        if avatar_enabled:
            _emit(progress, "Animating the avatar (lip-sync)", 0.45)
            try:
                avatar_info = create_avatar_video(
                    avatar_image, audio_path, out_name, auto_words=True, background_video=background_video,
                    ffmpeg_path=ffmpeg_path,
                    progress=lambda f: _emit(progress, "Animating the avatar (lip-sync)", 0.45 + 0.25 * f),
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

        cues_applied = 0
        if has_effects(script["cues"]):
            _emit(progress, "Adding the visual actions (camera moves)", 0.72)
            tmp_out = out_name + ".cues.mp4"
            try:
                aligned = align_cues(script["cues"], script["estimated_seconds"], _media_seconds(video_path))
                res = apply_cues_to_video(video_path, aligned, tmp_out, ffmpeg_path=ffmpeg_path,
                                          progress=lambda f: _emit(progress, "Adding the visual actions (camera moves)", 0.72 + 0.12 * f))
                os.replace(tmp_out, video_path)
                cues_applied = res["cues_applied"]
            except Exception as exc:  # the un-animated video is still good
                logger.warning("visual cues failed: %s", exc)
                warnings.append(f"Could not add the visual actions ({exc}); the video has no camera moves.")
            finally:
                _remove(tmp_out)

        drive_link = None
        if folder_id:
            _emit(progress, "Backing up to Google Drive", 0.88)
            try:
                with DriveBackup(folder_id, upload_fn=upload_to_google_drive) as backup:
                    drive_link = backup.submit(video_path).result()
            except Exception as exc:  # never lose the finished video because Drive is unreachable
                logger.warning("Drive backup failed: %s", exc)
                warnings.append(f"Google Drive backup failed ({exc}). Your video is saved on this computer.")

        _emit(progress, "Done", 1.0)
        return {
            "script": script, "audio_path": audio_path, "video_path": video_path,
            "drive_link": drive_link, "avatar": avatar_info, "cues_applied": cues_applied,
            "screen_text": screen_lines, "warnings": warnings,
        }
    finally:
        release_memory(not keep_models)
