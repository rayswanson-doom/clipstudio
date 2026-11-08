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
    from . import avatar_3d_engine
    from . import demo_avatars
    from . import meme_matcher
    from . import ocr_engine
    from . import render_engine
    from . import transcriber as _transcriber_mod
    from .visual_cues import apply_cues_to_video, has_effects
    from .scorer import align_cues, build_narration, find_viral_clips
    from .transcriber import transcribe_audio
    from .tts_engine import DEFAULT_VOICE, generate_voiceover
    from . import uploader as _uploader_mod
    from .uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from .video_engine import create_vertical_short, merge_audio_with_video, normalize_position_preset, place_podcast_avatar, splice_memes, PIP_PRESETS
except ImportError:  # pragma: no cover
    from avatar_sync import create_avatar_video, load_avatar
    import avatar_3d_engine
    import demo_avatars
    import meme_matcher
    import ocr_engine
    import render_engine
    import transcriber as _transcriber_mod
    from visual_cues import apply_cues_to_video, has_effects
    from scorer import align_cues, build_narration, find_viral_clips
    from transcriber import transcribe_audio
    from tts_engine import DEFAULT_VOICE, generate_voiceover
    import uploader as _uploader_mod
    from uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from video_engine import create_vertical_short, merge_audio_with_video, normalize_position_preset, place_podcast_avatar, splice_memes, PIP_PRESETS

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


def _read_screen_text(video, interval, max_frames, warnings, progress, stage, lo, hi, backend="auto") -> list:
    """OCR a video; any failure becomes a warning (the feature is optional). Frees the OCR engine after."""
    try:
        return ocr_engine.extract_screen_text(
            video, interval=interval, max_frames=max_frames, backend=backend,
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
    ocr_backend: str = "auto",
    keep_download: bool = False,
    keep_models: bool = False,
    memes: Optional[dict] = None,
) -> dict:
    """Find the best highlights in `source` and export each as a vertical short.

    read_screen_text   also OCR the video every `ocr_interval` s and append the burned-in text
                       (subtitles, titles) to the transcript. If the video has NO speech but does
                       have on-screen text, that text is used to pick the highlights.
    keep_download      keep the YouTube download in outputs/downloads (default: delete it, it is
                       only a working copy and can be huge)
    memes              meme_matcher.MemeSettings dict: when enabled, each clip gets contextual reaction clips (matched to
                       what is said / shown) BEFORE the Drive upload, so the uploaded file contains them
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
                                                progress, "Reading on-screen text (OCR)", 0.30, 0.44, ocr_backend)
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
                    clip["memes"] = []
                    if memes and memes.get("enabled"):
                        segs = meme_matcher.segments_from_words(words, pick["start"], pick["end"])
                        for sc in screen_segments:  # on-screen text (PaddleOCR / RapidOCR) can carry the emotion too
                            if float(sc["start"]) < pick["end"] and float(sc["end"]) > pick["start"]:
                                segs.append({"start": max(0.0, float(sc["start"]) - pick["start"]),
                                             "end": min(pick["end"], float(sc["end"])) - pick["start"],
                                             "text": str(sc["text"])})
                        segs.sort(key=lambda x: x["start"])
                        applied = _apply_memes(res["output_path"], segs, memes, [], ffmpeg_path, warnings, progress,
                                               f"Adding meme reactions to clip {idx}")
                        clip["memes"] = _meme_summary(applied)
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


def _frame_size(path: str) -> tuple:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()


def _apply_memes(video_path: str, segments: list, settings, avoid_rects, ffmpeg_path, warnings: list,
                 progress=None, stage: str = "Adding meme reactions", llm=None, clamp: bool = False) -> list:
    """Contextual meme reactions on a FINISHED video: plan from the text segments (emotion -> clip), splice the clips
    in with ffmpeg and replace `video_path` in place, so whatever is uploaded next already contains them.
    Never raises: a problem becomes a warning and the original video is kept. Returns the applied matches."""
    if not settings:
        return []
    try:
        cfg = settings if isinstance(settings, meme_matcher.MemeSettings) else meme_matcher.MemeSettings.from_dict(settings)
        if not cfg.enabled:
            return []
        if not segments:
            warnings.append("Meme reactions: there was no text to match, so none were added.")
            return []
        _emit(progress, stage, 0.0)
        w, h = _frame_size(video_path)
        dur = _media_seconds(video_path)
        plan = meme_matcher.plan_for_video(segments, cfg, (w, h), dur, avoid_rects=avoid_rects or (), llm=llm,
                                           clamp=clamp, ffmpeg_path=ffmpeg_path)
        warnings.extend(x for x in plan.get("warnings", []) if x not in warnings)
        if not plan["matches"]:
            if not plan["library"]["clips"]:
                warnings.append(f"Meme reactions are on but the meme folder is empty ({plan['library']['folder']}). "
                                "Add reaction clips (or create the starter pack) on the Generator tab.")
            else:
                warnings.append("Meme reactions: no moment was emotional enough for a reaction.")
            return []
        tmp = str(video_path) + ".memes.mp4"
        try:
            res = splice_memes(video_path, tmp, plan["matches"], ffmpeg_path=ffmpeg_path,
                               progress=lambda f: _emit(progress, stage, f))
            if res["memes_applied"] <= 0:
                return []
            os.replace(tmp, video_path)
        finally:
            _remove(tmp)
        return plan["matches"][: res["memes_applied"]] if res["memes_applied"] < len(plan["matches"]) else plan["matches"]
    except Exception as exc:  # the un-reacted video is still good
        logger.warning("meme reactions failed: %s", exc)
        warnings.append(f"Could not add the meme reactions ({exc}); the video has none.")
        return []


def _meme_summary(matches: list) -> list:
    return [{k: m.get(k) for k in ("start", "end", "emotion", "meme_name", "mode", "position", "text", "score")} for m in matches]


def _podcast_commentary(avatar_image, avatar_type, script, audio_path, out_name, out_dir, background_video, position, xy,
                       scale, gestures, quality, audio_source, ffmpeg_path, progress, warnings, segments=None,
                       music_path=None, broll_style="aurora"):
    """Voiceover + B-roll (or the animated backdrop) + OCR text overlays + avatar in a corner/at X,Y, composed by
    render_engine. Falls back to the simple ffmpeg layout, and then to the plain voiceover video, if a stage fails,
    so the user always gets a video. Returns (video_path, avatar_info or None)."""
    position = position or "Bottom-Right"
    avatar_clip, info = out_name + ".avatar.mp4", None
    if avatar_image:
        try:
            model = avatar_image
            if str(avatar_image).startswith("demo:"):
                style = str(avatar_image).split(":", 1)[1] or "human"
                model = demo_avatars.ensure_demo_avatars(str(Path(out_dir) / "avatars"))[style if style in demo_avatars.DEMO_STYLES else "human"]
                avatar_type = style if style in demo_avatars.DEMO_STYLES else avatar_type
            is_model = Path(str(model)).suffix.lower() in avatar_3d_engine.MODEL_EXTENSIONS
            label = "Animating the 3D avatar" if is_model else "Animating the avatar (lip-sync)"
            _emit(progress, label, 0.40)
            cb = lambda f: _emit(progress, label, 0.40 + 0.30 * f)  # noqa: E731
            if is_model:
                actual = _media_seconds(audio_path)
                notes = [{"start": c["start"], "text": c["text"]}
                         for c in align_cues(script["cues"], script["estimated_seconds"], actual) if c.get("effect") == "none"]
                info = avatar_3d_engine.create_3d_avatar_video(
                    model, audio_path, avatar_clip, text=script["script"], notes=notes, style=avatar_type,
                    gestures=gestures, quality=quality, ffmpeg_path=ffmpeg_path, progress=cb)
            else:
                info = create_avatar_video(model, audio_path, avatar_clip, auto_words=True, size=(480, 640),
                                           ffmpeg_path=ffmpeg_path, progress=cb)
            warnings.extend(info.get("warnings", []))
        except Exception as exc:
            logger.warning("podcast avatar failed: %s", exc)
            warnings.append(f"The avatar could not be added ({exc}); made a plain video instead.")
            info = None
    base = None
    try:
        _emit(progress, "Composing the video (B-roll, text, avatar)", 0.75)
        try:
            res = render_engine.render_commentary(
                audio_path, out_name, background_video=background_video, avatar_video=avatar_clip if info else None,
                position=position, xy=xy, scale=scale, screen_segments=segments, music_path=music_path,
                broll_style=broll_style, ffmpeg_path=ffmpeg_path,
                progress=lambda f: _emit(progress, "Composing the video (B-roll, text, avatar)", 0.75 + 0.12 * f))
            warnings.extend(res.get("warnings", []))
            if info is not None and res.get("avatar"):
                info["layout"] = {"preset": res["avatar"]["preset"], "rect": res["avatar"]["rect"], "scale": float(scale)}
            return res["output_path"], info
        except Exception as exc:  # fall back to the simple ffmpeg layout
            logger.warning("composition failed: %s", exc)
            warnings.append(f"The multi-track renderer failed ({exc}); used the simple layout.")
        base = out_name + ".base.mp4"
        merge_audio_with_video(audio_path, base, background_video=background_video, ffmpeg_path=ffmpeg_path)
        if info is not None:
            try:
                placed = place_podcast_avatar(base, avatar_clip, position, xy, scale, output_path=out_name,
                                              audio_source=audio_source, ffmpeg_path=ffmpeg_path)
                info["layout"] = {"preset": placed["preset"], "rect": placed["rect"], "scale": placed["scale"]}
                return placed["output_path"], info
            except Exception as exc:
                logger.warning("avatar placement failed: %s", exc)
                warnings.append(f"The avatar could not be added ({exc}); made a plain video instead.")
                info = None
        _remove(out_name)
        os.replace(base, out_name)
        base = None
        return out_name, None
    finally:
        _remove(avatar_clip)
        if base:
            _remove(base)


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
    avatar_type: str = "human",
    avatar_position: Optional[str] = None,
    avatar_xy: Optional[tuple] = None,
    avatar_scale: float = 0.36,
    avatar_gestures: bool = True,
    avatar_quality: str = "balanced",
    avatar_audio: str = "base",
    ocr_backend: str = "auto",
    show_screen_text: bool = True,
    music_path: Optional[str] = None,
    broll_style: str = "aurora",
    memes: Optional[dict] = None,
) -> dict:
    """Prompt / your own script -> voiceover -> video (+ avatar, + visual action cues) -> Drive.

    avatar_image          a picture (.jpg/.png: 2-D lip-sync), a 3-D model (.glb/.vrm: phoneme lip-sync + body
                          gestures) or "demo:human" / "demo:ai" / "demo:anime" for the built-in 3-D avatars
    avatar_type           "human" | "ai" | "anime"  (3-D: lighting / shading style)
    avatar_position       "Bottom-Right" | "Bottom-Left" | "Top-Right" | "Top-Left" | "Custom Coordinates" -> podcast
                          picture-in-picture over the video (video_engine.place_podcast_avatar). None keeps the
                          older full-card layout for picture avatars; 3-D models always use the podcast layout.
    avatar_xy             (X, Y) pixels for "Custom Coordinates";  avatar_scale  frame width / video width

    custom_script         narrate exactly this text (the prompt is then optional)
    action_prompt         describes what happens on screen ("zoom in on the logo, then pan right");
                          rendered as camera moves in time with the narration
    read_background_text  OCR the background video and narrate the text found in it after the script
    avatar_enabled        show `avatar_image` talking in sync with the voiceover (avatar_sync.py)
    ocr_backend           "auto" (PaddleOCR if installed, else RapidOCR) | "paddleocr" | "rapidocr"
    show_screen_text      with read_background_text: also show the text read from the frames as captions, at the
                          moment it appears (render_engine). The text is narrated either way.
    music_path            optional background music, ducked under the voice;  broll_style  animated backdrop used
                          when there is no background video: "aurora" | "sunset" | "mono"

    Returns {"script": {...narration incl. cues + shot_list...}, "audio_path", "video_path",
             "drive_link", "avatar", "cues_applied", "screen_text", "warnings"}.
    """
    if not (prompt or "").strip() and not (custom_script or "").strip():
        raise ValueError("Please enter a prompt first (or paste your own script).")
    avatar_kind = None
    if avatar_enabled:
        if not avatar_image:
            raise ValueError("Upload an avatar image (.jpg or .png) or a 3D model (.glb / .vrm) to use the AI avatar.")
        ext = Path(str(avatar_image)).suffix.lower()
        if str(avatar_image).startswith("demo:"):
            avatar_kind = "3d"
        elif ext in avatar_3d_engine.MODEL_EXTENSIONS:
            avatar_kind = "3d"
            avatar_3d_engine.G.load_model(avatar_image)  # fail fast on a damaged / compressed model
        else:
            avatar_kind = "image"
            load_avatar(avatar_image)  # fail fast on a wrong/unreadable file, before any voiceover work
        if avatar_position is not None:
            normalize_position_preset(avatar_position)  # fail fast on a typo
            if normalize_position_preset(avatar_position) == "Custom Coordinates" and not avatar_xy:
                raise ValueError("Custom Coordinates needs an X and a Y position in pixels.")
    warnings = []
    folder_id = _usable_folder(folder_id, warnings)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    try:
        screen_lines: list = []
        screen_segments: list = []
        screen_queue: list = []
        if read_background_text:
            if not background_video:
                warnings.append("Add a background video to read its on-screen text.")
            else:
                _emit(progress, "Reading on-screen text from the background video (OCR)", 0.03)
                segs = _read_screen_text(background_video, 1.0, 180, warnings, progress,
                                         "Reading on-screen text from the background video (OCR)", 0.03, 0.12, ocr_backend)
                screen_segments = segs
                screen_queue = ocr_engine.narration_queue(segs)
                screen_lines = [q["text"] for q in screen_queue]
                if not screen_lines and not any("Could not read" in w for w in warnings):
                    warnings.append("No on-screen text was found in the background video.")

        _emit(progress, "Preparing the narration", 0.14)
        script = build_narration(prompt, custom_script, action_prompt, screen_text=screen_lines)

        _emit(progress, "Generating the voiceover (edge-tts, needs internet)", 0.25)
        audio_path = out_dir / f"voice_{stamp}.mp3"
        audio_path = _run_async(generate_voiceover(script["script"], str(audio_path), voice=voice))

        out_name = str(out_dir / f"viewmax_{_safe_name(script['topic'], 30)}_{stamp}.mp4")
        video_path, avatar_info = None, None
        overlay_segments = screen_segments if show_screen_text else []
        if avatar_enabled and (avatar_kind == "3d" or avatar_position is not None):
            video_path, avatar_info = _podcast_commentary(
                avatar_image, avatar_type, script, audio_path, out_name, out_dir, background_video, avatar_position,
                avatar_xy, avatar_scale, avatar_gestures, avatar_quality, avatar_audio, ffmpeg_path, progress, warnings,
                segments=overlay_segments, music_path=music_path, broll_style=broll_style)
        elif not avatar_enabled and (overlay_segments or music_path):  # no avatar, but text overlays / music
            video_path, _ = _podcast_commentary(
                None, avatar_type, script, audio_path, out_name, out_dir, background_video, None, None, avatar_scale,
                False, avatar_quality, avatar_audio, ffmpeg_path, progress, warnings, segments=overlay_segments,
                music_path=music_path, broll_style=broll_style)
        elif avatar_enabled:
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

        meme_applied: list = []
        if memes and memes.get("enabled"):
            dur = _media_seconds(video_path)
            segs = meme_matcher.segments_from_text(script["script"], dur)
            rects = [avatar_info["layout"]["rect"]] if avatar_info and avatar_info.get("layout") else []
            meme_applied = _apply_memes(video_path, segs, memes, rects, ffmpeg_path, warnings, progress,
                                        "Adding meme reactions")

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
            "memes": _meme_summary(meme_applied), "screen_text": screen_lines, "screen_queue": screen_queue, "warnings": warnings,
        }
    finally:
        release_memory(not keep_models)


def run_story_pipeline(
    source: str,
    mode: str = "script2video",
    folder_id: Optional[str] = None,
    output_dir: str = "outputs",
    progress: Optional[ProgressCb] = None,
    ffmpeg_path: Optional[str] = None,
    provider: str = "offline",
    api_key: Optional[str] = None,
    style: str = "human",
    target_seconds: float = 45.0,
    episode: int = 1,
    episodes: Optional[int] = None,
    all_episodes: bool = False,
    max_shots: Optional[int] = None,
    render: bool = True,
    voice: bool = False,
    voice_name: str = DEFAULT_VOICE,
    size: tuple = (540, 960),
    fps: int = 20,
    camera: Optional[dict] = None,
    use_portraits: bool = True,
    continuity: bool = True,
    drift_threshold: float = 0.35,
    host_avatar: Optional[dict] = None,
    transport=None,
    memes: Optional[dict] = None,
) -> dict:
    """Idea / screenplay / novel -> multi-shot storyboard (+ CPU animatic) -> Google Drive, on any computer (no GPU).

    Modes: "idea2video", "novel2video", "script2video". The ViMax-style agents (backend/vimax_orchestrator.py)
    run first; `provider` "offline" uses built-in rules, "groq" / "gemini" / "auto" ask that LLM for structured JSON
    (falls back to the rules with a warning when the key is missing or a call fails). The API key is used only for
    those calls and is never stored or returned.

    `all_episodes` renders every episode of an idea (`episodes` of them) or a novel; otherwise just `episode`.
    `camera` = {"size","angle","movement"} preferences; `use_portraits` / `continuity` / `drift_threshold` are the
    character-consistency switches; `host_avatar` puts a podcast-style narrator portrait in a screen corner
    ({"image","style","position","xy","scale"}).

    Every finished render is backed up: each episode's video and a zip of its project files are uploaded to the Drive
    folder as soon as that episode is done (a missing folder / sign-in or an upload failure becomes a warning; the
    local files are never lost).

    Returns the first episode's result ({"plan","script","characters","storyboard","graph","issues","trace",
    "video_path","drift","drive_link", ...}) plus "episodes_out": one summary per episode ({"episode","title",
    "video_path","project_dir","drive_link","project_link","shots","seconds","warnings"}), "drive_links" and "warnings".
    """
    try:
        from . import vimax_orchestrator as vo
    except ImportError:  # pragma: no cover
        import vimax_orchestrator as vo

    warnings: list = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        llm = vo.make_client(provider, api_key, **({"transport": transport} if transport else {}))
    except vo.LLMError as exc:
        warnings.append(f"{exc} Used the built-in rules instead.")
        llm = vo.OfflineClient()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    first: Optional[dict] = None
    summaries: list = []
    drive_links: list = []
    folder_checked, folder = False, None
    total = max(1, int(episodes or 1)) if all_episodes else 1
    e = int(episode) if not all_episodes else 1
    done = 0
    while True:
        label = f"Episode {e}: " if total > 1 else ""
        base, span = done / total, 1.0 / total

        def prog(stage, f, _b=base, _s=span, _l=label):
            _emit(progress, _l + stage, _b + _s * f)

        _emit(progress, label + "Planning the story", base + span * 0.02)
        result = vo.ViMaxOrchestrator(llm, progress=lambda f, m: prog(m, 0.02 + 0.38 * f)).run(
            source, mode=mode, style=style, target_seconds=target_seconds, episode=e,
            episodes=(total if (all_episodes and mode == "idea2video") else episodes), max_shots=max_shots, camera=camera)
        if all_episodes and mode == "novel2video" and done == 0:
            total = max(1, int((result.get("novel") or {}).get("episodes") or 1))
            base, span = 0.0, 1.0 / total
        ep_warn = [w for w in result["warnings"]]
        name = f"story_{_safe_name(result['plan'].get('title') or 'story', 30)}_{stamp}" + (f"_e{e}" if total > 1 else "")
        project_dir = out_dir / name
        vo.save_project(result, str(project_dir))
        video_path, drift = None, []
        if render:
            prog("Rendering the animatic (CPU)", 0.42)
            try:
                rr = vo.render_storyboard(
                    result, str(out_dir / f"{name}.mp4"), work_dir=str(project_dir / "work"), size=tuple(size), fps=fps,
                    voice=voice, voice_name=voice_name, ffmpeg_path=ffmpeg_path, drift_threshold=drift_threshold,
                    use_portraits=use_portraits, continuity=continuity, host=host_avatar,
                    progress=lambda f, m: prog(m, 0.42 + 0.46 * f))
                video_path, drift = rr["output_path"], rr["drift"]
                ep_warn.extend(w for w in rr["warnings"] if w not in ep_warn)
                bad = [d for d in drift if d["status"] != "ok"]
                if bad:
                    ep_warn.append(f"Appearance drift was measured in {len(bad)} shot(s); see the drift report.")
            except Exception as exc:  # the plan is still valuable when rendering fails
                logger.warning("story render failed: %s", exc)
                ep_warn.append(f"The video could not be rendered ({exc}). The storyboard files were still saved.")
        meme_applied: list = []
        if video_path and memes and memes.get("enabled"):
            shots = result["storyboard"]["shots"]
            rects = []
            if host_avatar and host_avatar.get("image"):  # keep reactions off the narrator's corner
                w, h = _frame_size(video_path)
                sc = float(host_avatar.get("scale") or 0.3)
                bw = int(w * sc)
                pos = host_avatar.get("position") or "Bottom-Right"
                xy = host_avatar.get("xy") or (0, 0)
                x = {"Bottom-Right": w - bw - 12, "Top-Right": w - bw - 12}.get(pos, 12) if pos != "Custom Coordinates" else int(xy[0])
                y = {"Top-Right": 12, "Top-Left": 12}.get(pos, h - bw - 12) if pos != "Custom Coordinates" else int(xy[1])
                rects = [(x, y, bw, bw)]
            meme_applied = _apply_memes(video_path, meme_matcher.segments_from_shots(shots), memes, rects, ffmpeg_path,
                                        ep_warn, lambda st, f: prog(st, 0.88 + 0.03 * f), "Adding meme reactions",
                                        llm=llm, clamp=True)
        drive_link = project_link = None
        if video_path:  # every finished render is backed up, together with its project files
            if not folder_checked:
                folder, folder_checked = _usable_folder(folder_id, ep_warn), True
            if folder:
                prog("Backing up to Google Drive", 0.92)
                try:
                    archive = _uploader_mod.zip_folder(str(project_dir), str(out_dir / f"{name}_project.zip"))
                except Exception as exc:  # noqa: BLE001
                    archive = None
                    ep_warn.append(f"The project files could not be zipped for backup ({exc}).")
                sent = _uploader_mod.backup_files([p for p in (video_path, archive) if p], folder, upload_fn=upload_to_google_drive)
                drive_link = sent["links"].get(video_path)
                project_link = sent["links"].get(archive) if archive else None
                for path, msg in sent["errors"].items():
                    ep_warn.append(f"Google Drive backup of {Path(path).name} failed ({msg}). It is saved on this computer.")
                if drive_link:
                    drive_links.append(drive_link)
        result.update({"project_dir": str(project_dir), "video_path": video_path, "drift": drift,
                       "drive_link": drive_link, "project_link": project_link, "rendered": bool(video_path),
                       "memes": _meme_summary(meme_applied)})
        summaries.append({"episode": e, "title": result["plan"].get("title"), "video_path": video_path,
                          "project_dir": str(project_dir), "drive_link": drive_link, "project_link": project_link,
                          "memes": _meme_summary(meme_applied), "shots": len(result["storyboard"]["shots"]), "seconds": result["storyboard"]["total_duration"],
                          "drift": drift, "warnings": ep_warn})
        warnings.extend(f"{label}{w}" for w in ep_warn if f"{label}{w}" not in warnings)
        if first is None:
            first = result
        done += 1
        e += 1
        if not all_episodes or done >= total:
            break
    _emit(progress, "Done", 1.0)
    first["warnings"] = warnings
    first.update({"episodes_out": summaries, "drive_links": drive_links})
    return first
