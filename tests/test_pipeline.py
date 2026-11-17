"""Pipeline tests with real OpenCV + ffmpeg. Only the things that need the
network or heavy models are replaced: transcription, yt-dlp, Drive upload."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pipeline  # noqa: E402
import video_engine as ve  # noqa: E402
from tests.test_scorer import make_words  # noqa: E402
from tests.test_video_engine import FFMPEG, make_source, probe, square_detector  # noqa: E402


def three_sentence_words():
    # ~1s per sentence -> ~3s, matching the 3s synthetic source video
    return make_words(
        [("Stop this is crazy!", 0), ("Did you know this secret?", 0), ("It cost 5 million dollars.", 0)],
        wps=0.2, gap=0.05,
    )


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class VugolaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_pipeline(self, name, source=None, folder="F", **kw):
        out = os.path.join(self.dir.name, name)
        ready = mock.patch.object(pipeline, "drive_status", return_value="ready")
        ready.start()
        self.addCleanup(ready.stop)
        with mock.patch.object(pipeline, "transcribe_audio", return_value=kw.pop("words", three_sentence_words())):
            return pipeline.run_vugola_pipeline(
                source or self.src, folder, top_n=kw.pop("top_n", 2), min_len=1, max_len=2.5,
                output_dir=out, ffmpeg_path=FFMPEG, detector=square_detector, **kw,
            )

    def test_happy_path_uploads_every_clip(self):
        seen = []
        with mock.patch.object(pipeline, "upload_to_google_drive", return_value="https://drive/x") as up:
            res = self.run_pipeline("v1", progress=lambda s, f: seen.append((s, f)))
        self.assertGreaterEqual(len(res["clips"]), 1)
        self.assertEqual(up.call_count, len(res["clips"]))
        for clip in res["clips"]:
            self.assertTrue(os.path.isfile(clip["output_path"]))
            self.assertEqual(clip["drive_link"], "https://drive/x")
            self.assertIsNone(clip["error"])
            info = probe(clip["output_path"])
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual((v["width"], v["height"]), (1080, 1920))
        self.assertEqual(up.call_args[0][1], "F")
        fractions = [f for _, f in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(seen[-1], ("Done", 1.0))
        self.assertEqual(res["warnings"], [])

    def test_url_is_downloaded_first(self):
        with mock.patch.object(pipeline, "download_media", return_value=self.src) as dl, mock.patch.object(
            pipeline, "upload_to_google_drive", return_value="L"
        ):
            self.run_pipeline("v2", source="https://youtu.be/abc")
        self.assertEqual(dl.call_args[0][0], "https://youtu.be/abc")

    def test_no_speech_raises(self):
        with self.assertRaises(RuntimeError):
            self.run_pipeline("v3", words=[])

    def test_no_folder_skips_upload_with_warning(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_DRIVE_FOLDER_ID"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            pipeline, "upload_to_google_drive"
        ) as up:
            res = self.run_pipeline("v4", folder="")
        up.assert_not_called()
        self.assertIn(pipeline.NO_FOLDER_WARNING, res["warnings"])
        self.assertTrue(all(c["drive_link"] is None and c["output_path"] for c in res["clips"]))

    def test_upload_failure_is_recorded_not_raised(self):
        with mock.patch.object(pipeline, "upload_to_google_drive", side_effect=RuntimeError("drive down")):
            res = self.run_pipeline("v5")
        for clip in res["clips"]:
            self.assertIn("drive down", clip["error"])
            self.assertTrue(os.path.isfile(clip["output_path"]))  # clip still saved

    # ---- Viral Clip Studio features -------------------------------------------------------------------------
    def test_subtitles_are_burned_in_and_files_kept(self):
        res = self.run_pipeline("s1", upload=False, subtitles={"enabled": True, "size": "Large", "color": "Yellow"})
        self.assertEqual(res["warnings"], [])
        for clip in res["clips"]:
            self.assertTrue(clip["subtitled"], clip)
            self.assertTrue(os.path.isfile(clip["subtitle_files"]["srt"]))
            ass = Path(clip["subtitle_files"]["ass"]).read_text(encoding="utf-8")
            self.assertIn("[V4+ Styles]", ass)
            self.assertIn("Dialogue:", ass)
            self.assertEqual(probe(clip["output_path"])["streams"][0]["height"], 1920)

    def test_subtitles_off_by_default_and_failure_is_a_warning(self):
        res = self.run_pipeline("s2", upload=False)
        self.assertFalse(any(c["subtitled"] for c in res["clips"]))
        with mock.patch.object(pipeline, "burn_subtitles", side_effect=RuntimeError("libass missing")):
            res = self.run_pipeline("s3", upload=False, subtitles={"enabled": True})
        self.assertTrue(any("subtitles could not be added" in w for w in res["warnings"]))
        for clip in res["clips"]:
            self.assertFalse(clip["subtitled"])
            self.assertIsNone(clip["error"])
            self.assertTrue(os.path.isfile(clip["output_path"]))

    def test_music_from_own_file_is_mixed_under_the_voice(self):
        bgm = os.path.join(self.dir.name, "bgm.mp3")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=500:duration=1", bgm], check=True)
        res = self.run_pipeline("m1", upload=False, music={"enabled": True, "path": bgm, "volume": 0.2, "duck": True})
        self.assertEqual(res["warnings"], [])
        for clip in res["clips"]:
            self.assertTrue(clip["music_mixed"])
            kinds = [x["codec_type"] for x in probe(clip["output_path"])["streams"]]
            self.assertEqual(sorted(kinds), ["audio", "video"])

    def test_music_trending_track_uses_the_finder_and_failure_is_a_warning(self):
        bgm = os.path.join(self.dir.name, "bgm2.mp3")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=400:duration=1", bgm], check=True)
        track = {"id": "abc", "title": "Beat", "channel": "Ch", "url": "https://youtu.be/abc"}
        found = {"query": "lofi", "tracks": [track], "selected": track, "path": bgm, "warnings": [], "licence_note": "check licence"}
        with mock.patch.object(pipeline.music_finder, "fetch_trending_music", return_value=found) as fm:
            res = self.run_pipeline("m2", upload=False, music={"enabled": True, "query": "lofi"})
        self.assertEqual(fm.call_args[0][0], "lofi")
        self.assertTrue(all(c["music_mixed"] for c in res["clips"]))
        self.assertEqual(res["music"]["path"], bgm)
        down = {"query": "", "tracks": [], "selected": None, "path": None, "licence_note": "",
                "warnings": ["Could not reach YouTube: offline"]}
        with mock.patch.object(pipeline.music_finder, "fetch_trending_music", return_value=down):
            res = self.run_pipeline("m3", upload=False, music={"enabled": True})
        self.assertTrue(any("offline" in w for w in res["warnings"]))
        self.assertFalse(any(c["music_mixed"] for c in res["clips"]))
        self.assertTrue(all(os.path.isfile(c["output_path"]) for c in res["clips"]))

    def test_social_packages_are_generated_offline_for_every_clip(self):
        res = self.run_pipeline("c1", upload=False)
        for clip in res["clips"]:
            pk = clip["social"]
            self.assertLess(len(pk["youtube"]["title"]), 60)
            self.assertTrue(10 <= len(pk["youtube"]["tags"]) <= 16)
            self.assertTrue(15 <= len(pk["instagram"]["hashtags"]) <= 20)
            self.assertTrue(5 <= len(pk["facebook"]["hashtags"]) <= 8)
        res = self.run_pipeline("c2", upload=False, social={"enabled": False})
        self.assertTrue(all(c["social"] is None for c in res["clips"]))

    def test_upload_false_never_touches_drive(self):
        with mock.patch.object(pipeline, "upload_to_google_drive") as up:
            res = self.run_pipeline("u1", upload=False)
        up.assert_not_called()
        self.assertTrue(all(c["drive_link"] is None for c in res["clips"]))

    def test_all_features_together(self):
        bgm = os.path.join(self.dir.name, "bgm3.mp3")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=450:duration=1", bgm], check=True)
        with mock.patch.object(pipeline, "upload_to_google_drive", return_value="https://drive/z") as up:
            res = self.run_pipeline("all", subtitles={"enabled": True}, music={"enabled": True, "path": bgm})
        for clip in res["clips"]:
            self.assertTrue(clip["subtitled"] and clip["music_mixed"] and clip["social"])
            self.assertEqual(clip["drive_link"], "https://drive/z")
        self.assertEqual(up.call_count, len(res["clips"]))


if __name__ == "__main__":
    unittest.main()
