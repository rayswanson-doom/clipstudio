"""
backend/music_finder.py

Find and download currently popular background music from YouTube with the embedded yt-dlp Python API - no API key,
no paid service.

  scan_trending_music(query)      yt_dlp.YoutubeDL(extract_flat=...).extract_info("ytsearch10:<query>") plus a
                                  "ytsearchdate10:" pass (newest first); flat entries carry duration and view_count;
                                  the best few are enriched (one cheap metadata request each) for the upload date.
                                  Ranked by view VELOCITY (views per day) so a fresh track that is taking off beats an
                                  old giant. Returns the top tracks to choose from.
  download_track(track)           bestaudio/best + the FFmpegExtractAudio post-processor -> storage/music_cache/<id>.mp3
                                  (bundled ffmpeg, cached: a track is fetched once).
  fetch_trending_music(query)     scan + pick (the top one, or `pick`=index) + download, in one call.

Honest notes: "trending" here means recent + fast-growing among YouTube's own search results for your query - YouTube
offers no public trending-music API without a key. Most popular tracks are COPYRIGHTED: using them under your clip may be
muted, blocked or monetised by the rights holder on YouTube / Instagram / Facebook. The result carries a licence note;
prefer royalty-free / "no copyright" tracks for anything you publish. Needs internet; offline raises MusicError.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_QUERY = "trending shorts background music"
DEFAULT_CACHE = "storage/music_cache"
MIN_SECONDS, MAX_SECONDS = 20, 600  # a usable background track; hour-long mixes and 5 s jingles are skipped
LICENCE_NOTE = ("Popular tracks are usually copyrighted: the platform may mute, block or monetise a video that uses them. "
                "For anything you publish, prefer royalty-free / 'no copyright' music.")
_BAD_TITLE = re.compile(r"\b(\d+\s*hours?|hour\s*loop|full\s*album|compilation|podcast|interview|reaction|tutorial|mix\s*\d+)\b", re.I)


class MusicError(RuntimeError):
    """Music search / download failed (offline, blocked, no usable result)."""


def _is_network_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("urlopen error", "getaddrinfo", "name resolution", "timed out", "connection", "network is unreachable",
                                   "unable to download webpage", "transporterror", "no route to host"))


def _make_ydl(opts: dict, factory: Optional[Callable[[dict], object]] = None):
    if factory is not None:
        return factory(opts)
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - listed in requirements
        raise MusicError("yt-dlp is not installed, so trending music cannot be searched.") from exc
    return yt_dlp.YoutubeDL(opts)


def _upload_date(entry: dict) -> Optional[datetime]:
    d = entry.get("upload_date") or entry.get("release_date")
    if d and re.fullmatch(r"\d{8}", str(d)):
        return datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=timezone.utc)
    ts = entry.get("timestamp") or entry.get("release_timestamp")
    if ts:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None
    return None


def score_track(track: dict, now: Optional[datetime] = None) -> float:
    """Higher = more 'trending'. With an upload date: log10(views per day) + a small (0.1x) bonus for total views. Without one
    (flat results often have none): log10(views) scaled down so dated tracks are preferred."""
    views = max(0, int(track.get("view_count") or 0))
    up = _upload_date(track)
    if up:
        now = now or datetime.now(timezone.utc)
        days = max(1.0, (now - up).total_seconds() / 86400.0)
        return math.log10(views / days + 1.0) + 0.1 * math.log10(views + 1.0)
    return 0.8 * math.log10(views + 1.0)


def _track(entry: dict, now: Optional[datetime]) -> Optional[dict]:
    vid = entry.get("id")
    if not vid:
        return None
    dur = entry.get("duration")
    title = str(entry.get("title") or "")
    if dur is not None and not (MIN_SECONDS <= float(dur) <= MAX_SECONDS):
        return None
    if _BAD_TITLE.search(title):
        return None
    up = _upload_date(entry)
    t = {"id": str(vid), "title": title, "channel": entry.get("channel") or entry.get("uploader") or "",
         "url": entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
         "duration": float(dur) if dur is not None else None, "view_count": int(entry.get("view_count") or 0),
         "upload_date": up.strftime("%Y-%m-%d") if up else None}
    t["score"] = round(score_track({**entry, "view_count": t["view_count"]}, now), 4)
    return t


def scan_trending_music(query: str = DEFAULT_QUERY, limit: int = 5, pool: int = 10, enrich: bool = True,
                        ydl_factory: Optional[Callable[[dict], object]] = None, now: Optional[datetime] = None) -> List[dict]:
    """The `limit` most trending usable tracks for `query`, best first:
    [{"id","title","channel","url","duration","view_count","upload_date","score"}]. Raises MusicError (offline / nothing)."""
    query = (query or DEFAULT_QUERY).strip() or DEFAULT_QUERY
    flat_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True, "noplaylist": True,
                 "socket_timeout": 15}
    entries: Dict[str, dict] = {}
    errors: List[BaseException] = []
    for prefix in (f"ytsearch{pool}", f"ytsearchdate{pool}"):  # YouTube's relevance order, then newest first
        try:
            with _make_ydl(flat_opts, ydl_factory) as ydl:
                info = ydl.extract_info(f"{prefix}:{query}", download=False) or {}
        except MusicError:
            raise
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            continue
        for e in info.get("entries") or []:
            if e and e.get("id") and e["id"] not in entries:
                entries[e["id"]] = e
    if not entries:
        if errors and any(_is_network_error(e) for e in errors):
            raise MusicError("YouTube could not be reached (no internet connection?). Trending music needs the internet; "
                             "you can use your own music file instead.") from None
        if errors:
            raise MusicError(f"The YouTube search failed: {str(errors[0])[:160]}") from None
        raise MusicError(f"No tracks were found for '{query}'.")
    tracks = [t for t in (_track(e, now) for e in entries.values()) if t]
    tracks.sort(key=lambda t: t["score"], reverse=True)
    if enrich:  # flat results rarely include the upload date: look the best candidates up properly (one light request each)
        deep = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "socket_timeout": 15}
        for t in tracks[: max(limit * 2, limit)]:
            if t["upload_date"]:
                continue
            try:
                with _make_ydl(deep, ydl_factory) as ydl:
                    full = ydl.extract_info(t["url"], download=False) or {}
            except Exception as exc:  # noqa: BLE001 - a missing date only lowers the ranking accuracy
                logger.info("could not enrich %s: %s", t["id"], exc)
                continue
            merged = {**entries.get(t["id"], {}), **full}
            better = _track(merged, now)
            if better:
                t.update({k: better[k] for k in ("view_count", "upload_date", "duration", "score")})
        tracks.sort(key=lambda t: t["score"], reverse=True)
    if not tracks:
        raise MusicError(f"No usable background-music tracks (20 s - 10 min) were found for '{query}'.")
    return tracks[:limit]


def cached_track_path(track_id: str, cache_dir: str = DEFAULT_CACHE) -> Optional[str]:
    p = Path(cache_dir) / f"{track_id}.mp3"
    return str(p) if p.is_file() and p.stat().st_size > 1000 else None


def download_track(track: dict, cache_dir: str = DEFAULT_CACHE, ffmpeg_path: Optional[str] = None,
                   ydl_factory: Optional[Callable[[dict], object]] = None) -> str:
    """Download the track's audio (bestaudio/best, converted to mp3 by FFmpegExtractAudio with the bundled ffmpeg) into
    the cache and return the file path. A track already in the cache is not downloaded again."""
    tid = str(track.get("id") or "")
    if not re.fullmatch(r"[\w-]{3,40}", tid):
        raise MusicError("That track has no valid id.")
    hit = cached_track_path(tid, cache_dir)
    if hit:
        return hit
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    try:
        try:
            from .video_engine import find_ffmpeg
        except ImportError:  # pragma: no cover
            from video_engine import find_ffmpeg
        ff = find_ffmpeg(ffmpeg_path)
    except Exception as exc:  # noqa: BLE001
        raise MusicError(f"ffmpeg was not found, so the audio cannot be converted: {exc}") from None
    opts = {
        "format": "bestaudio/best", "outtmpl": str(cache / "%(id)s.%(ext)s"), "noplaylist": True, "quiet": True,
        "no_warnings": True, "socket_timeout": 20, "retries": 3, "restrictfilenames": True,
        "match_filter": _duration_filter,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
    }
    if os.path.exists(ff):  # a real file / folder (bundled bin/ffmpeg.exe); a bare "ffmpeg" is found on PATH by yt-dlp itself
        opts["ffmpeg_location"] = str(ff)
    url = track.get("url") or f"https://www.youtube.com/watch?v={tid}"
    try:
        with _make_ydl(opts, ydl_factory) as ydl:
            ydl.download([url])
    except MusicError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_network_error(exc):
            raise MusicError("The track could not be downloaded: YouTube is not reachable (no internet connection?).") from None
        raise MusicError(f"The track could not be downloaded: {str(exc)[:160]}") from None
    path = cached_track_path(tid, cache_dir)
    if not path:  # some post-processors leave another extension
        for p in cache.glob(f"{tid}.*"):
            if p.suffix.lower() in (".mp3", ".m4a", ".opus", ".webm", ".aac", ".wav") and p.stat().st_size > 1000:
                return str(p)
        raise MusicError("The download finished but no audio file was produced.")
    return path


def _duration_filter(info, *, incomplete=False):
    d = info.get("duration")
    if d and d > MAX_SECONDS:
        return f"longer than {MAX_SECONDS // 60} minutes"
    return None


def fetch_trending_music(mood_or_niche: str = DEFAULT_QUERY, count: int = 5, pick: int = 0, download: bool = True,
                         cache_dir: str = DEFAULT_CACHE, ffmpeg_path: Optional[str] = None,
                         ydl_factory: Optional[Callable[[dict], object]] = None) -> dict:
    """Scan YouTube for trending background music matching `mood_or_niche`, choose track number `pick` (0 = the top one)
    of the best `count`, and download it.

    Returns {"query","tracks": [top tracks],"selected": track | None,"path": local audio file | None,"warnings": [],
    "licence_note": str}. Never raises for "offline": the failure is returned as a warning with selected/path None."""
    out = {"query": mood_or_niche, "tracks": [], "selected": None, "path": None, "warnings": [], "licence_note": LICENCE_NOTE}
    try:
        tracks = scan_trending_music(mood_or_niche, limit=max(1, count), ydl_factory=ydl_factory)
    except MusicError as exc:
        out["warnings"].append(str(exc))
        return out
    out["tracks"] = tracks
    idx = pick if 0 <= pick < len(tracks) else 0
    out["selected"] = tracks[idx]
    if download:
        try:
            out["path"] = download_track(tracks[idx], cache_dir, ffmpeg_path, ydl_factory)
        except MusicError as exc:
            out["warnings"].append(str(exc))
    return out


def prune_cache(cache_dir: str = DEFAULT_CACHE, max_files: int = 30) -> int:
    """Keep the cache small: delete the oldest tracks beyond `max_files`. Returns how many were removed."""
    files = sorted((p for p in Path(cache_dir).glob("*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for p in files[max_files:]:
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed
