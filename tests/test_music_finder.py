"""music_finder: yt-dlp search/ranking/download, tested with a stand-in YoutubeDL (no network, no API key)."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import music_finder as mf  # noqa: E402

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)
FFMPEG = shutil.which("ffmpeg")


def entry(vid, views, date=None, dur=120, title=None, **kw):
    e = {"id": vid, "title": title or f"Track {vid}", "channel": "Chan", "view_count": views, "duration": dur,
         "url": f"https://www.youtube.com/watch?v={vid}"}
    if date:
        e["upload_date"] = date
    e.update(kw)
    return e


class FakeYDL:
    """Records every use; search results come from `results[prefix]`, details from `details[id]`."""
    instances = []

    def __init__(self, opts, results=None, details=None, fail=None, downloads=None):
        self.opts, self.results, self.details, self.fail, self.downloads = opts, results or {}, details or {}, fail, downloads
        self.calls = []
        FakeYDL.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, query, download=False):
        self.calls.append(("extract_info", query, download))
        if self.fail:
            raise self.fail
        if query.startswith("ytsearch"):
            prefix = query.split(":", 1)[0]
            return {"entries": self.results.get(prefix, [])}
        vid = query.rsplit("=", 1)[-1]
        return self.details.get(vid, {})

    def download(self, urls):
        self.calls.append(("download", urls))
        if self.fail:
            raise self.fail
        if self.downloads:
            self.downloads(self.opts, urls)


def factory(**kw):
    FakeYDL.instances = []
    return lambda opts: FakeYDL(opts, **kw)


class ScanTests(unittest.TestCase):
    def test_uses_ytsearch10_with_extract_flat_and_no_api_key(self):
        f = factory(results={"ytsearch10": [entry("aaa111", 1000)], "ytsearchdate10": []})
        mf.scan_trending_music("lofi chill", ydl_factory=f, now=NOW, enrich=False)
        queries = [c[1] for y in FakeYDL.instances for c in y.calls]
        self.assertIn("ytsearch10:lofi chill", queries)
        self.assertIn("ytsearchdate10:lofi chill", queries)
        flat = FakeYDL.instances[0].opts
        self.assertIs(flat["extract_flat"], True)
        self.assertIs(flat["skip_download"], True)
        self.assertFalse(any("key" in k.lower() for k in flat))

    def test_fresh_fast_growing_track_beats_an_old_giant(self):
        f = factory(results={"ytsearch10": [entry("old1111", 80_000_000, "20150101"), entry("new2222", 400_000, "20261110")]})
        tracks = mf.scan_trending_music(ydl_factory=f, now=NOW, enrich=False)
        self.assertEqual(tracks[0]["id"], "new2222")
        self.assertEqual(tracks[0]["upload_date"], "2026-11-10")

    def test_results_merge_both_searches_without_duplicates_and_limit_to_five(self):
        a = [entry(f"rel{i:04d}", 1000 * (i + 1), "20261101") for i in range(6)]
        b = [entry("rel0001", 1000, "20261101")] + [entry(f"new{i:04d}", 500 * (i + 1), "20261112") for i in range(4)]
        tracks = mf.scan_trending_music(ydl_factory=factory(results={"ytsearch10": a, "ytsearchdate10": b}), now=NOW, enrich=False)
        ids = [t["id"] for t in tracks]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual([t["score"] for t in tracks], sorted((t["score"] for t in tracks), reverse=True))

    def test_unusable_results_are_filtered(self):
        res = [entry("short11", 9e6, dur=8), entry("long111", 9e6, dur=7200), entry("mix1111", 9e6, title="10 hours lofi loop"),
               entry("ok11111", 10, dur=90), {"title": "no id"}, None]
        tracks = mf.scan_trending_music(ydl_factory=factory(results={"ytsearch10": res}), now=NOW, enrich=False)
        self.assertEqual([t["id"] for t in tracks], ["ok11111"])

    def test_results_carry_views_duration_and_date(self):
        f = factory(results={"ytsearch10": [entry("abc1234", 12345, "20261101", dur=95.0)]})
        t = mf.scan_trending_music(ydl_factory=f, now=NOW, enrich=False)[0]
        self.assertEqual((t["view_count"], t["duration"], t["channel"]), (12345, 95.0, "Chan"))
        self.assertTrue(t["url"].startswith("https://www.youtube.com/watch"))

    def test_missing_dates_are_filled_in_by_a_second_lookup(self):
        f = factory(results={"ytsearch10": [entry("nodate1", 500_000), entry("dated11", 800_000, "20150101")]},
                    details={"nodate1": {"upload_date": "20261115", "view_count": 500_000, "duration": 100}})
        tracks = mf.scan_trending_music(ydl_factory=f, now=NOW, enrich=True)
        self.assertEqual(tracks[0]["id"], "nodate1")
        self.assertEqual(tracks[0]["upload_date"], "2026-11-15")

    def test_enrich_failure_is_ignored(self):
        class Flaky(FakeYDL):
            def extract_info(self, query, download=False):
                if not query.startswith("ytsearch"):
                    raise RuntimeError("HTTP 429")
                return super().extract_info(query, download)
        tracks = mf.scan_trending_music(ydl_factory=lambda o: Flaky(o, results={"ytsearch10": [entry("abc1234", 99)]}), now=NOW)
        self.assertEqual(tracks[0]["id"], "abc1234")

    def test_offline_gives_a_friendly_error(self):
        f = factory(fail=OSError("<urlopen error [Errno -3] Temporary failure in name resolution>"))
        with self.assertRaises(mf.MusicError) as cm:
            mf.scan_trending_music(ydl_factory=f)
        self.assertIn("internet", str(cm.exception))

    def test_other_failures_and_empty_results(self):
        with self.assertRaises(mf.MusicError) as cm:
            mf.scan_trending_music(ydl_factory=factory(fail=RuntimeError("Sign in to confirm")))
        self.assertIn("search failed", str(cm.exception))
        with self.assertRaises(mf.MusicError):
            mf.scan_trending_music("zzz", ydl_factory=factory(results={}))

    def test_blank_query_falls_back_to_the_default(self):
        f = factory(results={"ytsearch10": [entry("abc1234", 5)]})
        mf.scan_trending_music("   ", ydl_factory=f, now=NOW, enrich=False)
        self.assertIn(f"ytsearch10:{mf.DEFAULT_QUERY}", [c[1] for y in FakeYDL.instances for c in y.calls])

    def test_score_prefers_dated_recent_growth(self):
        recent = mf.score_track({"view_count": 1_000_000, "upload_date": "20261110"}, NOW)
        old = mf.score_track({"view_count": 1_000_000, "upload_date": "20160101"}, NOW)
        undated = mf.score_track({"view_count": 1_000_000}, NOW)
        self.assertGreater(recent, old)
        self.assertGreater(recent, undated)
        self.assertEqual(mf.score_track({}, NOW), 0.0)


def write_mp3(opts, urls):
    out = Path(opts["outtmpl"]).parent
    vid = urls[0].rsplit("=", 1)[-1]
    (out / f"{vid}.mp3").write_bytes(b"x" * 5000)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_download_options_are_bestaudio_to_mp3_via_bundled_ffmpeg(self):
        f = factory(downloads=write_mp3)
        fake_ff = os.path.join(self.d, "ffmpeg.exe")
        Path(fake_ff).write_bytes(b"")
        path = mf.download_track({"id": "abc1234", "url": "https://www.youtube.com/watch?v=abc1234"},
                                 os.path.join(self.d, "cache"), ffmpeg_path=fake_ff, ydl_factory=f)
        self.assertTrue(path.endswith("abc1234.mp3"))
        opts = FakeYDL.instances[0].opts
        self.assertEqual(opts["format"], "bestaudio/best")
        pp = opts["postprocessors"][0]
        self.assertEqual((pp["key"], pp["preferredcodec"]), ("FFmpegExtractAudio", "mp3"))
        self.assertEqual(opts["ffmpeg_location"], fake_ff)
        self.assertTrue(opts["outtmpl"].replace("\\", "/").endswith("cache/%(id)s.%(ext)s"))

    def test_cached_track_is_not_downloaded_twice(self):
        cache = os.path.join(self.d, "c")
        t = {"id": "abc1234"}
        mf.download_track(t, cache, ydl_factory=factory(downloads=write_mp3))
        FakeYDL.instances = []
        again = mf.download_track(t, cache, ydl_factory=factory(downloads=write_mp3))
        self.assertTrue(again.endswith("abc1234.mp3"))
        self.assertEqual(FakeYDL.instances, [])

    def test_failures_are_friendly(self):
        for exc, word in ((OSError("getaddrinfo failed"), "internet"), (RuntimeError("Video unavailable"), "could not be downloaded")):
            with self.assertRaises(mf.MusicError) as cm:
                mf.download_track({"id": "abc1234"}, os.path.join(self.d, "x"), ydl_factory=factory(fail=exc))
            self.assertIn(word, str(cm.exception))
        with self.assertRaises(mf.MusicError):
            mf.download_track({"id": "abc1234"}, os.path.join(self.d, "y"), ydl_factory=factory())  # "worked" but no file
        for bad in ({}, {"id": "../etc/passwd"}, {"id": "a b"}):
            with self.assertRaises(mf.MusicError):
                mf.download_track(bad, self.d, ydl_factory=factory())

    def test_long_videos_are_refused_by_the_match_filter(self):
        opts = {}
        mf.download_track({"id": "abc1234"}, os.path.join(self.d, "z"), ydl_factory=lambda o: (opts.update(o), FakeYDL(o, downloads=write_mp3))[1])
        self.assertIn("longer", mf._duration_filter({"duration": 9999}))
        self.assertIsNone(mf._duration_filter({"duration": 120}))
        self.assertIs(opts["match_filter"], mf._duration_filter)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_returns_the_top_track_downloaded(self):
        f = factory(results={"ytsearch10": [entry("best111", 9_000_000, "20261110"), entry("meh2222", 100, "20261110")]}, downloads=write_mp3)
        res = mf.fetch_trending_music("epic motivational", cache_dir=self.d, ydl_factory=f)
        self.assertEqual(res["selected"]["id"], "best111")
        self.assertTrue(res["path"].endswith("best111.mp3"))
        self.assertEqual(res["warnings"], [])
        self.assertEqual(res["licence_note"], mf.LICENCE_NOTE)
        self.assertIsInstance(res, dict)

    def test_user_can_pick_from_the_top_five(self):
        f = factory(results={"ytsearch10": [entry(f"trk{i:04d}", 1000 * (10 - i), "20261110") for i in range(8)]}, downloads=write_mp3)
        res = mf.fetch_trending_music(count=5, pick=2, cache_dir=self.d, ydl_factory=f)
        self.assertEqual(len(res["tracks"]), 5)
        self.assertEqual(res["selected"]["id"], res["tracks"][2]["id"])
        self.assertTrue(res["path"].endswith(res["tracks"][2]["id"] + ".mp3"))
        res = mf.fetch_trending_music(count=5, pick=99, cache_dir=self.d, ydl_factory=f)  # out of range -> the top one
        self.assertEqual(res["selected"]["id"], res["tracks"][0]["id"])

    def test_signature_default_and_offline_never_raises(self):
        import inspect
        self.assertEqual(inspect.signature(mf.fetch_trending_music).parameters["mood_or_niche"].default, "trending shorts background music")
        res = mf.fetch_trending_music(cache_dir=self.d, ydl_factory=factory(fail=OSError("getaddrinfo failed")))
        self.assertIsNone(res["path"])
        self.assertIsNone(res["selected"])
        self.assertTrue(res["warnings"])

    def test_scan_only_does_not_download_and_download_error_is_a_warning(self):
        f = factory(results={"ytsearch10": [entry("abc1234", 50, "20261110")]}, downloads=write_mp3)
        res = mf.fetch_trending_music(download=False, cache_dir=self.d, ydl_factory=f)
        self.assertIsNone(res["path"])
        self.assertFalse(any(c[0] == "download" for y in FakeYDL.instances for c in y.calls))

        class DownloadFails(FakeYDL):
            def download(self, urls):
                raise RuntimeError("blocked")
        res = mf.fetch_trending_music(cache_dir=self.d, ydl_factory=lambda o: DownloadFails(o, results={"ytsearch10": [entry("abc1234", 50, "20261110")]}))
        self.assertIsNone(res["path"])
        self.assertEqual(res["selected"]["id"], "abc1234")
        self.assertTrue(any("could not be downloaded" in w for w in res["warnings"]))


class CacheTests(unittest.TestCase):
    def test_prune_keeps_the_newest(self):
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                p = Path(d, f"t{i}.mp3")
                p.write_bytes(b"x" * 2000)
                os.utime(p, (time.time() - (5 - i) * 100,) * 2)
            self.assertEqual(mf.prune_cache(d, max_files=2), 3)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["t3.mp3", "t4.mp3"])

    def test_missing_yt_dlp_is_reported_clearly(self):
        saved = sys.modules.get("yt_dlp", "absent")
        sys.modules["yt_dlp"] = None  # makes `import yt_dlp` raise ImportError
        try:
            with self.assertRaises(mf.MusicError) as cm:
                mf.scan_trending_music()
            self.assertIn("yt-dlp", str(cm.exception))
        finally:
            if saved == "absent":
                del sys.modules["yt_dlp"]
            else:
                sys.modules["yt_dlp"] = saved


if __name__ == "__main__":
    unittest.main()
