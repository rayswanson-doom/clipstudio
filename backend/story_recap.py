"""
backend/story_recap.py

Story Overview / Summary Compilation: stitch the finished clips into ONE recap video that plays in the order the
moments happen in the source video, so it reads as a short beginning-to-end overview.

How it stays reliable and light on a low-spec CPU laptop
  * every clip is first normalised ON ITS OWN to the same picture (size, fps, yuv420p, square pixels) and the same
    sound (44.1 kHz stereo AAC; clips without sound get silence), so clips of different origin never fail to join or
    drift out of sync;
  * the normalised parts are then joined with ffmpeg's concat demuxer using stream copy (no second re-encode);
  * only one ffmpeg process at a time, limited threads, nothing is held in RAM - all work happens through files in a
    temporary folder that is removed at the end (also on errors).

The ffmpeg-python recipe (`ffmpeg.input` -> filters -> `ffmpeg.concat` -> `ffmpeg.output` -> `ffmpeg.run`) is the model
for the argument lists below; they are built directly as `subprocess` argument lists (no shell, no extra dependency),
so the packaged exe needs nothing beyond the bundled bin/ffmpeg.exe.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

try:  # package import (backend.story_recap) or plain module (story_recap)
    from . import subtitle_engine
    from . import video_engine as _ve
except ImportError:  # pragma: no cover
    import subtitle_engine
    import video_engine as _ve

logger = logging.getLogger(__name__)

TARGET_SIZE = (1080, 1920)  # 9:16, like the clips
TARGET_FPS = 30
AUDIO_RATE = 44100
MIN_CLIPS = 2
ENCODE_THREADS = 2  # keeps a low-spec laptop responsive
ProgressCb = Callable[[str, float], None]


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
def order_clips(clip_records: Iterable[dict]) -> List[dict]:
    """The usable clips (an existing output file) in the order they occur in the source video (by start time).

    Ties keep their original order; records without a usable file or start time are dropped.
    """
    usable = []
    for i, rec in enumerate(clip_records or []):
        path = rec.get("output_path")
        if not path or not Path(str(path)).is_file():
            continue
        try:
            start = float(rec.get("start"))
        except (TypeError, ValueError):
            continue
        usable.append((start, i, rec))
    usable.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in usable]


def flatten_words(transcript_segments: Optional[Sequence[dict]]) -> List[dict]:
    """Word dicts ({"word","start","end"}) from either a word list or a list of segments that carry "words"."""
    words: List[dict] = []
    for item in transcript_segments or []:
        if not isinstance(item, dict):
            continue
        if "word" in item:
            words.append(item)
        elif isinstance(item.get("words"), list):
            words.extend(w for w in item["words"] if isinstance(w, dict) and "word" in w)
    return words


# --------------------------------------------------------------------------- #
# ffmpeg argument lists (pure functions, easy to test)
# --------------------------------------------------------------------------- #
def build_normalize_command(ffmpeg: str, src: str, dst: str, has_audio: bool,
                            size: Tuple[int, int] = TARGET_SIZE, fps: int = TARGET_FPS) -> List[str]:
    """Re-encode one clip to the common picture + sound format (silence is added when the clip has none)."""
    w, h = size
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},format=yuv420p")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if has_audio:
        af = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"
        cmd += ["-vf", vf, "-af", af]
    else:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo", "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", str(ENCODE_THREADS),
            "-c:a", "aac", "-b:a", "128k", "-ar", str(AUDIO_RATE), "-ac", "2", "-movflags", "+faststart", str(dst)]
    return cmd


def concat_list_text(parts: Sequence[str]) -> str:
    """Body of an ffmpeg concat-demuxer list file (single quotes in names are escaped)."""
    lines = []
    for p in parts:
        escaped = str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    return "\n".join(lines) + "\n"


def build_concat_command(ffmpeg: str, list_file: str, dst: str) -> List[str]:
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", "-movflags", "+faststart", str(dst)]


# --------------------------------------------------------------------------- #
# subtitles on the recap timeline
# --------------------------------------------------------------------------- #
def recap_cues(ordered: Sequence[dict], offsets: Sequence[float], words: Sequence[dict]) -> List[dict]:
    """Subtitle cues for the joined video: each clip's words, shifted to where that clip sits in the recap.

    Clips that already have burned-in subtitles (`subtitled`) are skipped so nothing is captioned twice.
    """
    cues: List[dict] = []
    for rec, offset in zip(ordered, offsets):
        if rec.get("subtitled"):
            continue
        try:
            part = subtitle_engine.cues_for_clip(words, float(rec["start"]), float(rec["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        dur = float(rec.get("duration") or (float(rec["end"]) - float(rec["start"])))
        for c in part:
            cues.append({**c, "start": c["start"] + offset, "end": min(c["end"], dur) + offset,
                         "words": [{**w, "start": w["start"] + offset, "end": w["end"] + offset} for w in c.get("words", [])]})
    cues.sort(key=lambda c: c["start"])
    return [c for c in cues if c["end"] > c["start"]]


def recap_text(ordered: Sequence[dict], words: Sequence[dict]) -> str:
    """What is said across the chosen moments, in recap order (used to write the combined captions and tags)."""
    chunks = []
    for rec in ordered:
        try:
            s, e = float(rec["start"]), float(rec["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = " ".join(str(w["word"]).strip() for w in words if s <= float(w.get("start", -1)) < e)
        chunks.append(text or str(rec.get("text") or ""))
    return " ".join(c for c in chunks if c).strip()


# --------------------------------------------------------------------------- #
# the recap video
# --------------------------------------------------------------------------- #
def build_story_overview_video(
    clip_records: List[dict],
    transcript_segments: Optional[List[dict]],
    output_path: str,
    music_path: Optional[str] = None,
    music_volume: float = 0.15,
    burn_subtitles: bool = True,
    ffmpeg_path: Optional[str] = None,
    size: Tuple[int, int] = TARGET_SIZE,
    fps: int = TARGET_FPS,
    duck: bool = True,
    progress: Optional[ProgressCb] = None,
) -> str:
    """Join the clips, in source-video order, into one cohesive recap video and return `output_path`.

    clip_records         dicts with "output_path" (the finished clip) and "start"/"end" (seconds in the source video);
                         "subtitled" (bool) marks clips whose subtitles are already burned in.
    transcript_segments  the word list of the source video ({"word","start","end"}), or segments carrying "words";
                         used for the story-overview subtitles.
    music_path           optional background-music file, mixed under the speech at `music_volume` (0..1)
    burn_subtitles       burn synchronised subtitles for the clips that have none yet
    """
    ff = _ve.find_ffmpeg(ffmpeg_path)
    ordered = order_clips(clip_records)
    if len(ordered) < MIN_CLIPS:
        raise ValueError(f"A story overview needs at least {MIN_CLIPS} finished clips (got {len(ordered)}).")
    if music_path and not Path(music_path).is_file():
        raise FileNotFoundError(f"Music file not found: {music_path}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(msg: str, frac: float) -> None:
        if progress:
            try:
                progress(msg, frac)
            except Exception:  # noqa: BLE001 - progress display must never break the build
                pass

    work = tempfile.mkdtemp(prefix="recap_", dir=str(out.parent))
    try:
        parts, offsets, t = [], [], 0.0
        n = len(ordered)
        for i, rec in enumerate(ordered):
            emit(f"Preparing clip {i + 1} of {n} for the story overview", 0.1 + 0.6 * i / n)
            info = _ve._probe_media(ff, rec["output_path"])
            if not info["has_video"]:
                raise RuntimeError(f"Clip {i + 1} could not be read: {rec['output_path']}")
            dst = os.path.join(work, f"part{i:03d}.mp4")
            _ve._run_ffmpeg(build_normalize_command(ff, rec["output_path"], dst, bool(info["has_audio"]), size, fps))
            parts.append(dst)
            offsets.append(t)
            t += float(_ve._probe_media(ff, dst)["duration"] or info["duration"] or 0.0)

        emit("Joining the clips in story order", 0.72)
        lst = os.path.join(work, "list.txt")
        Path(lst).write_text(concat_list_text(parts), encoding="utf-8")
        current = os.path.join(work, "joined.mp4")
        _ve._run_ffmpeg(build_concat_command(ff, lst, current))
        for p in parts:  # free disk space early
            try:
                os.remove(p)
            except OSError:
                pass

        if burn_subtitles:
            cues = recap_cues(ordered, offsets, flatten_words(transcript_segments))
            if cues:
                emit("Burning story-overview subtitles", 0.82)
                files = subtitle_engine.write_subtitle_files(cues, os.path.join(work, "recap"), size)
                nxt = os.path.join(work, "subtitled.mp4")
                _ve.burn_subtitles(current, files["ass"], nxt, ff)
                os.replace(nxt, current)
                shutil.copyfile(files["srt"], str(out.with_suffix(".srt")))

        if music_path:
            emit("Mixing the background music", 0.92)
            nxt = os.path.join(work, "music.mp4")
            _ve.mix_background_music(current, music_path, nxt, music_volume, duck, ff)
            os.replace(nxt, current)

        os.replace(current, str(out))
        emit("Story overview ready", 1.0)
        return str(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
