"""
backend/pipeline.py

The end-to-end Viral Clip Studio workflow behind the Streamlit UI. All the real work lives in the other backend modules;
this file only wires them together and reports progress, so it can be tested without a browser.

  run_vugola_pipeline : URL / file -> transcribe (faster-whisper, word timestamps, CPU int8) -> score highlights
                        -> face-tracked 9:16 crops -> burned-in word-accurate subtitles (subtitle_engine)
                        -> contextual meme reactions (optional) -> trending YouTube background music ducked under the
                        speaker (music_finder, optional) -> YouTube / Instagram / Facebook captions + tags
                        (social_metadata) -> Google Drive (optional)
"""

from __future__ import annotations

import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

try:  # package import (backend.pipeline) or plain module (pipeline)
    from . import meme_matcher
    from . import music_finder
    from . import ocr_engine
    from . import social_metadata
    from . import story_recap
    from . import subtitle_engine
    from . import transcriber as _transcriber_mod
    from .scorer import find_viral_clips
    from .transcriber import transcribe_audio
    from . import uploader as _uploader_mod
    from .uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from .video_engine import PIP_PRESETS, burn_subtitles, create_vertical_short, mix_background_music, splice_memes
except ImportError:  # pragma: no cover
    import meme_matcher
    import music_finder
    import ocr_engine
    import social_metadata
    import story_recap
    import subtitle_engine
    import transcriber as _transcriber_mod
    from scorer import find_viral_clips
    from transcriber import transcribe_audio
    import uploader as _uploader_mod
    from uploader import DriveBackup, download_media, drive_status, upload_to_google_drive
    from video_engine import PIP_PRESETS, burn_subtitles, create_vertical_short, mix_background_music, splice_memes

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
                                "Add reaction clips (or create the starter pack) in the Meme reactions panel.")
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


# --------------------------------------------------------------------------- #
# Viral Clip Studio
# --------------------------------------------------------------------------- #
def _resolve_music(music: Optional[dict], ffmpeg_path, warnings: list, progress) -> Optional[dict]:
    """One background track for the whole run: the user's own file, a track chosen from the scan, or the top trending
    track for the query. Any failure becomes a warning (the clips are then made without music)."""
    if not music or not music.get("enabled"):
        return None
    cache = music.get("cache_dir") or music_finder.DEFAULT_CACHE
    try:
        own = music.get("path")
        if own:
            if not os.path.isfile(own):
                raise music_finder.MusicError(f"The music file was not found: {own}")
            return {"path": own, "title": Path(own).name, "channel": "", "url": "", "source": "file"}
        _emit(progress, "Getting the background music", 0.06)
        track = music.get("track")
        if track:
            path = music_finder.download_track(track, cache, ffmpeg_path)
            return {"path": path, "title": track.get("title", ""), "channel": track.get("channel", ""),
                    "url": track.get("url", ""), "source": "youtube", "licence_note": music_finder.LICENCE_NOTE}
        res = music_finder.fetch_trending_music(music.get("query") or music_finder.DEFAULT_QUERY, count=5, pick=0,
                                                cache_dir=cache, ffmpeg_path=ffmpeg_path)
        warnings.extend(w for w in res["warnings"] if w not in warnings)
        if not res["path"]:
            return None
        sel = res["selected"]
        return {"path": res["path"], "title": sel["title"], "channel": sel["channel"], "url": sel["url"], "source": "youtube",
                "licence_note": res["licence_note"]}
    except music_finder.MusicError as exc:
        warnings.append(f"Background music was skipped: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("music failed: %s", exc)
        warnings.append(f"Background music was skipped: {exc}")
        return None


def _make_recap(clips, spoken_words, out_dir, stem, cfg, music_info, social_cfg, ffmpeg_path, backup, warnings, progress):
    """Story Overview: join the finished clips in source order. Never raises - problems become warnings.
    Returns (info dict | None, Drive-upload future | None)."""
    ready = [c for c in clips if c.get("output_path") and os.path.isfile(c["output_path"])]
    if len(ready) < story_recap.MIN_CLIPS:
        warnings.append("The story overview was skipped: it needs at least 2 finished clips (raise 'Number of clips').")
        return None, None
    target = out_dir / f"{stem}_story_overview.mp4"
    info = {"output_path": None, "drive_link": None, "error": None, "clip_count": len(ready), "duration": None,
            "social": None, "subtitled": False, "music_mixed": False, "subtitle_file": None}
    # the clips already carry their own music; only add it again when some clip is missing it
    music_path = None
    if music_info and music_info.get("path") and not all(c.get("music_mixed") for c in ready):
        music_path = music_info["path"]
    try:
        _emit(progress, "Building the story overview video", 0.90)
        burn = bool(cfg.get("burn_subtitles", True)) and bool(spoken_words)
        path = story_recap.build_story_overview_video(
            ready, spoken_words, str(target), music_path=music_path,
            music_volume=float(cfg.get("music_volume", 0.15)), burn_subtitles=burn, ffmpeg_path=ffmpeg_path, progress=None)
        info["output_path"] = path
        info["music_mixed"] = bool(music_path) or all(c.get("music_mixed") for c in ready)
        srt = Path(path).with_suffix(".srt")
        info["subtitle_file"] = str(srt) if srt.is_file() else None
        info["subtitled"] = bool(info["subtitle_file"]) or all(c.get("subtitled") for c in ready)
        ordered = story_recap.order_clips(ready)
        info["clips"] = [{"start": c["start"], "end": c["end"]} for c in ordered]
        info["duration"] = round(sum(float(c.get("duration") or 0.0) for c in ordered), 1)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Story overview failed")
        warnings.append(f"The story overview could not be built ({exc}).")
        info["error"] = str(exc)
        return info, None
    if social_cfg.get("enabled", True):
        try:
            text = story_recap.recap_text(ordered, spoken_words)
            info["social"] = social_metadata.generate_packages(text, social_cfg.get("topic_keywords") or (),
                                                               social_cfg.get("use_llm", "auto"))
            warnings.extend(w for w in info["social"]["warnings"] if w not in warnings)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Captions and tags for the story overview could not be generated ({exc}).")
    future = backup.submit(info["output_path"]) if backup else None
    return info, future


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
    subtitles: Optional[dict] = None,
    music: Optional[dict] = None,
    social: Optional[dict] = None,
    upload: bool = True,
    recap: Optional[dict] = None,
) -> dict:
    """Find the best highlights in `source` and export each as a vertical short.

    subtitles          {"enabled": bool, "size": "Small|Medium|Large", "color": subtitle_engine.COLOR_PRESETS key,
                       "uppercase": bool}: burn 3-5 word, word-accurate captions into every clip (.srt/.ass kept next to it)
    music              {"enabled": bool, "volume": 0..1 (default 0.15), "duck": bool, "path": own audio file,
                       "track": a track from music_finder.scan_trending_music, "query": search text}: background music
                       mixed UNDER the speaker. With neither path nor track the top trending track for `query` is used.
    social             {"topic_keywords": [...], "use_llm": "auto|off|groq|gemini"}: YouTube / Instagram / Facebook caption
                       packages per clip (always offline-capable). Pass {"enabled": False} to skip.
    recap              {"enabled": bool, "burn_subtitles": bool (default True), "music_volume": 0..1}: ALSO join the finished
                       clips, in the order they happen in the source video, into one Story Overview video (needs 2+ clips),
                       with combined YouTube / Instagram / Facebook captions + tags, uploaded to Drive like the clips.
    upload             False = never upload to Drive (the user turned Auto-Upload off)
    read_screen_text   also OCR the video every `ocr_interval` s and append the burned-in text to the transcript. If the
                       video has NO speech but does have on-screen text, that text is used to pick the highlights.
    keep_download      keep the YouTube download in outputs/downloads (default: delete it, it is only a working copy)
    memes              meme_matcher.MemeSettings dict: contextual reaction clips (matched to what is said / shown)
    keep_models        keep Whisper/OCR in RAM after the run (default: free them at once)

    Returns {"source_path", "word_count", "clips": [...], "transcript": [...], "screen_text": [...], "music": {...} | None,
             "warnings": [...]}.
    Each clip: start, end, duration, score, reasons, text, screen_text, output_path, drive_link, subtitled,
    subtitle_files, music_mixed, social, memes, error (None on success). One failing clip does not stop the rest.
    """
    warnings = []
    folder_id = _usable_folder(folder_id, warnings) if upload else None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = None
    downloaded = False
    sub_cfg = subtitles if (subtitles and subtitles.get("enabled")) else None
    style = None
    if sub_cfg:
        style = subtitle_engine.Style(size_preset=sub_cfg.get("size", subtitle_engine.DEFAULT_SIZE_PRESET),
                                      color_preset=sub_cfg.get("color", subtitle_engine.DEFAULT_COLOR_PRESET),
                                      uppercase=bool(sub_cfg.get("uppercase"))).validate()
    social_cfg = {"enabled": True, **(social or {})}
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
        if sub_cfg and not spoken_words:
            warnings.append("Subtitles were skipped: no speech was detected (the video's own on-screen text is not re-captioned).")
            sub_cfg = None

        _emit(progress, "Scoring highlights", 0.45)
        picks = find_viral_clips(words, top_n=top_n, min_len=min_len, max_len=max_len)
        if not picks:
            raise RuntimeError("Could not find any highlight ranges in this transcript.")

        music_info = _resolve_music(music, ffmpeg_path, warnings, progress)
        stem = _safe_name(Path(local).stem)
        clips = []
        pending = []  # (clip, future) - Drive uploads running in the background
        backup = DriveBackup(folder_id, upload_fn=upload_to_google_drive) if folder_id else None
        try:
            for idx, pick in enumerate(picks, start=1):
                _emit(progress, f"Cropping clip {idx} of {len(picks)}" + (" (earlier clips upload in the background)" if backup and idx > 1 else ""),
                      0.50 + 0.40 * (idx - 1) / len(picks))
                clip = {**pick, "output_path": None, "drive_link": None, "error": None, "faces_found": None,
                        "screen_text": ocr_engine.screen_text_in_range(screen_segments, pick["start"], pick["end"]),
                        "subtitled": False, "subtitle_files": None, "music_mixed": False, "social": None, "memes": []}
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
                    if sub_cfg:  # 1) burned-in word-accurate subtitles
                        _emit(progress, f"Burning subtitles into clip {idx}", 0.50 + 0.40 * (idx - 0.6) / len(picks))
                        try:
                            cues = subtitle_engine.cues_for_clip(spoken_words, pick["start"], pick["end"])
                            if cues:
                                files = subtitle_engine.write_subtitle_files(cues, str(target.with_suffix("")), _frame_size(res["output_path"]), style)
                                tmp = str(target) + ".subs.mp4"
                                try:
                                    burn_subtitles(res["output_path"], files["ass"], tmp, ffmpeg_path)
                                    os.replace(tmp, res["output_path"])
                                finally:
                                    _remove(tmp)
                                clip["subtitled"], clip["subtitle_files"] = True, files
                            else:
                                warnings.append(f"Clip {idx}: no words were spoken in this range, so it has no subtitles.")
                        except Exception as exc:  # noqa: BLE001 - the un-subtitled clip is still good
                            logger.warning("subtitles failed for clip %d: %s", idx, exc)
                            warnings.append(f"Clip {idx}: subtitles could not be added ({exc}).")
                    if memes and memes.get("enabled"):  # 2) contextual reaction clips
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
                    if music_info:  # 3) trending background music, ducked under the speaker
                        _emit(progress, f"Mixing background music into clip {idx}", 0.50 + 0.40 * (idx - 0.3) / len(picks))
                        tmp = str(target) + ".music.mp4"
                        try:
                            mix_background_music(res["output_path"], music_info["path"], tmp,
                                                 float(music.get("volume", 0.15)), bool(music.get("duck", True)), ffmpeg_path)
                            os.replace(tmp, res["output_path"])
                            clip["music_mixed"] = True
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("music mix failed for clip %d: %s", idx, exc)
                            warnings.append(f"Clip {idx}: the background music could not be mixed in ({exc}).")
                        finally:
                            _remove(tmp)
                    if social_cfg["enabled"]:  # 4) YouTube / Instagram / Facebook captions + tags
                        try:
                            text = " ".join(str(w["word"]).strip() for w in spoken_words
                                            if pick["start"] <= float(w["start"]) < pick["end"]) or pick.get("text", "")
                            clip["social"] = social_metadata.generate_packages(
                                " ".join([text] + list(clip["screen_text"] or [])) if clip["screen_text"] else text,
                                social_cfg.get("topic_keywords") or (), social_cfg.get("use_llm", "auto"))
                            warnings.extend(w for w in clip["social"]["warnings"] if w not in warnings)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("captions failed for clip %d: %s", idx, exc)
                            warnings.append(f"Clip {idx}: captions and tags could not be generated ({exc}).")
                    if backup:
                        pending.append((clip, backup.submit(res["output_path"])))
                except Exception as exc:  # keep going; report per clip
                    logger.exception("Clip %d failed", idx)
                    clip["error"] = str(exc)
                clips.append(clip)

            recap_info = None
            recap_future = None
            recap_cfg = {"enabled": False, **(recap or {})}
            if recap_cfg["enabled"]:
                recap_info, recap_future = _make_recap(clips, spoken_words, out_dir, stem, recap_cfg, music_info, social_cfg,
                                                       ffmpeg_path, backup, warnings, progress)

            if pending:
                _emit(progress, "Finishing the Google Drive backup", 0.92)
            for clip, future in pending:
                try:
                    clip["drive_link"] = future.result()
                except Exception as exc:  # the clip stays safe on this computer
                    logger.warning("Drive backup failed for %s: %s", clip["output_path"], exc)
                    clip["error"] = str(exc)
            if recap_future is not None:
                try:
                    recap_info["drive_link"] = recap_future.result()
                except Exception as exc:  # the recap stays safe on this computer
                    logger.warning("Drive backup failed for the story overview: %s", exc)
                    recap_info["error"] = str(exc)
        finally:
            if backup:
                backup.close()

        made = [c for c in clips if c["faces_found"] is not None]
        if made and all(c["faces_found"] == 0 for c in made):
            warnings.append(NO_FACE_WARNING)

        _emit(progress, "Done", 1.0)
        return {"source_path": local, "word_count": len(words), "clips": clips, "transcript": transcript,
                "screen_text": screen_segments, "music": music_info, "recap": recap_info, "warnings": warnings}
    finally:
        if downloaded and not keep_download:
            _remove(local)  # temporary working copy of the download
        release_memory(not keep_models)
