"""run_viewmax_pipeline with the 3-D avatar + podcast placement, end to end (real ffmpeg, real renderer)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import pipeline  # noqa: E402
import demo_avatars  # noqa: E402
from tests.test_pipeline import FakeVoice  # noqa: E402
from tests.test_video_engine import FFMPEG, probe  # noqa: E402


def _frame(path, t=0.8):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PodcastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.pic = os.path.join(cls.dir.name, "face.png")
        img = np.full((400, 400, 3), (180, 170, 160), np.uint8)
        cv2.circle(img, (200, 180), 110, (150, 190, 230), -1)
        cv2.circle(img, (165, 155), 12, (30, 30, 30), -1)
        cv2.circle(img, (235, 155), 12, (30, 30, 30), -1)
        cv2.ellipse(img, (200, 240), (40, 14), 0, 0, 360, (60, 60, 160), -1)
        cv2.imwrite(cls.pic, img)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def setUp(self):
        r = mock.patch.object(pipeline, "drive_status", return_value="ready")
        r.start()
        self.addCleanup(r.stop)
        FakeVoice.calls, FakeVoice.seconds = [], 2

    def go(self, name, **kw):
        out = os.path.join(self.dir.name, name)
        kw.setdefault("folder_id", "FOLD")
        kw.setdefault("avatar_enabled", True)
        kw.setdefault("avatar_quality", "fast")
        with mock.patch.object(pipeline, "generate_voiceover", FakeVoice.generate), mock.patch.object(
            pipeline, "upload_to_google_drive", return_value="https://drive/v"
        ) as up:
            res = pipeline.run_viewmax_pipeline("Write a short script about coffee", output_dir=out,
                                                ffmpeg_path=FFMPEG, **kw)
        return res, up, out

    def assert_clean(self, out):
        leftovers = [f for f in os.listdir(out) if f.endswith((".base.mp4", ".avatar.mp4", ".cues.mp4"))]
        self.assertEqual(leftovers, [])

    def test_demo_human_bottom_right_uploaded_to_drive(self):
        res, up, out = self.go("h", avatar_image="demo:human", avatar_type="human", avatar_position="Bottom-Right")
        info = probe(res["video_path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (1080, 1920))
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertEqual(res["avatar"]["mode"], "3d")
        self.assertEqual(res["avatar"]["layout"]["preset"], "Bottom-Right")
        x, y, w, h = res["avatar"]["layout"]["rect"]
        self.assertGreater(x, 400)
        self.assertGreater(y, 900)
        up.assert_called_once_with(res["video_path"], "FOLD")
        self.assertEqual(res["drive_link"], "https://drive/v")
        f = _frame(res["video_path"])
        self.assertGreater(int(f[y + 2, x + w // 2].mean()), 200)  # white podcast border
        self.assert_clean(out)

    def test_anime_custom_coordinates_and_wave_note(self):
        res, up, out = self.go("a", avatar_image="demo:anime", avatar_type="anime", avatar_position="Custom Coordinates",
                               avatar_xy=(100, 300), avatar_scale=0.4, action_prompt="[wave] hello there")
        self.assertEqual(res["avatar"]["layout"]["rect"][:2], (100, 300))
        self.assertEqual(res["avatar"]["style"], "anime")
        self.assertEqual(res["avatar"]["model_format"], "vrm")
        self.assertIn("wave", [g["name"] for g in res["avatar"]["gestures"]])
        self.assert_clean(out)

    def test_custom_coordinates_require_xy(self):
        with self.assertRaises(ValueError):
            self.go("c", avatar_image="demo:ai", avatar_position="Custom Coordinates")

    def test_unknown_position(self):
        with self.assertRaises(ValueError):
            self.go("u", avatar_image="demo:ai", avatar_position="Middle")

    def test_real_glb_file(self):
        glb = demo_avatars.ensure_demo_avatars(os.path.join(self.dir.name, "m"))["ai"]
        res, _, out = self.go("g", avatar_image=glb, avatar_type="ai", avatar_position="Top-Left")
        self.assertEqual(res["avatar"]["model_format"], "glb")
        self.assertEqual(res["avatar"]["layout"]["preset"], "Top-Left")
        self.assert_clean(out)

    def test_picture_avatar_in_podcast_frame(self):
        res, _, out = self.go("p", avatar_image=self.pic, avatar_position="Top-Right", avatar_scale=0.3)
        self.assertEqual(res["avatar"]["layout"]["preset"], "Top-Right")
        self.assertIsNot(res["avatar"].get("mode"), "3d")
        self.assert_clean(out)

    def test_picture_without_position_keeps_legacy_card(self):
        res, _, _ = self.go("l", avatar_image=self.pic)
        self.assertIsNotNone(res["avatar"])
        self.assertEqual(res["avatar"]["layout"], "full")  # old card layout, no podcast frame

    def test_avatar_failure_falls_back_to_plain_video(self):
        with mock.patch.object(pipeline.avatar_3d_engine, "create_3d_avatar_video", side_effect=RuntimeError("boom")):
            res, up, out = self.go("f", avatar_image="demo:human", avatar_position="Bottom-Left")
        self.assertIsNone(res["avatar"])
        self.assertTrue(any("avatar could not be added" in w for w in res["warnings"]))
        self.assertTrue(os.path.isfile(res["video_path"]))
        info = probe(res["video_path"])
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        up.assert_called_once()  # the plain video is still backed up
        self.assert_clean(out)

    def test_broken_model_fails_fast(self):
        bad = os.path.join(self.dir.name, "bad.glb")
        Path(bad).write_bytes(b"nope")
        with self.assertRaises(Exception):
            self.go("b", avatar_image=bad)
        self.assertEqual(FakeVoice.calls, [])  # nothing was synthesised

    def _bg(self, name="bg.mp4", secs=2):
        import subprocess
        path = os.path.join(self.dir.name, name)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=s=320x240:r=25:d={secs}",
                        "-pix_fmt", "yuv420p", path], check=True)
        return path

    SEGS = [{"text": "SALE ENDS TODAY", "start": 0.2, "end": 1.6, "confidence": 0.9},
            {"text": "Free shipping", "start": 0.4, "end": 1.8, "confidence": 0.8}]

    def test_ocr_text_is_narrated_shown_and_uploaded_without_an_avatar(self):
        bg = self._bg()
        with mock.patch.object(pipeline.ocr_engine, "extract_screen_text", return_value=self.SEGS) as ex:
            res, up, out = self.go("o", background_video=bg, read_background_text=True, avatar_enabled=False,
                                   ocr_backend="paddleocr")
        self.assertEqual(ex.call_args[1]["backend"], "paddleocr")
        self.assertEqual(res["screen_text"], ["SALE ENDS TODAY", "Free shipping"])
        self.assertEqual([q["text"] for q in res["screen_queue"]], ["SALE ENDS TODAY", "Free shipping"])
        self.assertIn("The text on screen reads: SALE ENDS TODAY.", res["script"]["script"])
        up.assert_called_once()
        info = probe(res["video_path"])
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assert_clean(out)

    def test_captions_can_be_switched_off(self):
        bg = self._bg("bg2.mp4")
        seen = {}
        real = pipeline.render_engine.render_commentary

        def spy(*a, **k):
            seen.update(k)
            return real(*a, **k)

        with mock.patch.object(pipeline.ocr_engine, "extract_screen_text", return_value=self.SEGS), \
                mock.patch.object(pipeline.render_engine, "render_commentary", side_effect=spy):
            self.go("n", background_video=bg, read_background_text=True, show_screen_text=False, avatar_enabled=False)
            self.assertEqual(seen, {})  # nothing to compose: plain merge path
            self.go("y", background_video=bg, read_background_text=True, show_screen_text=True, avatar_enabled=False)
        self.assertEqual(len(seen["screen_segments"]), 2)

    def test_avatar_background_ocr_all_tracks(self):
        bg = self._bg("bg3.mp4")
        with mock.patch.object(pipeline.ocr_engine, "extract_screen_text", return_value=self.SEGS):
            res, up, out = self.go("all", background_video=bg, read_background_text=True, avatar_image="demo:anime",
                                   avatar_type="anime", avatar_position="Top-Right")
        rows = {r["track"] for r in res["avatar"]["tracks"]} if "tracks" in res["avatar"] else set()
        self.assertEqual(res["avatar"]["layout"]["preset"], "Top-Right")
        self.assertEqual(res["screen_text"][0], "SALE ENDS TODAY")
        up.assert_called_once_with(res["video_path"], "FOLD")
        self.assert_clean(out)

    def test_music_and_backdrop_style(self):
        import subprocess
        music = os.path.join(self.dir.name, "bed.wav")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=90:duration=1", music], check=True)
        res, _, out = self.go("mu", avatar_enabled=False, music_path=music, broll_style="sunset")
        self.assertTrue(os.path.isfile(res["video_path"]))
        v = next(s for s in probe(res["video_path"])["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (1080, 1920))

    def test_composition_failure_falls_back_to_simple_layout(self):
        with mock.patch.object(pipeline.render_engine, "render_commentary", side_effect=RuntimeError("kaput")):
            res, up, out = self.go("fb", avatar_image="demo:ai", avatar_position="Bottom-Right")
        self.assertTrue(any("multi-track renderer failed" in w for w in res["warnings"]))
        self.assertEqual(res["avatar"]["layout"]["preset"], "Bottom-Right")  # ffmpeg PiP fallback still placed it
        self.assertTrue(os.path.isfile(res["video_path"]))
        up.assert_called_once()
        self.assert_clean(out)

    def test_drive_failure_keeps_video(self):
        out = os.path.join(self.dir.name, "d")
        with mock.patch.object(pipeline, "generate_voiceover", FakeVoice.generate), mock.patch.object(
            pipeline, "upload_to_google_drive", side_effect=RuntimeError("offline")
        ):
            res = pipeline.run_viewmax_pipeline("coffee", "FOLD", output_dir=out, ffmpeg_path=FFMPEG,
                                                avatar_enabled=True, avatar_image="demo:ai", avatar_position="Top-Left",
                                                avatar_quality="fast")
        self.assertTrue(os.path.isfile(res["video_path"]))
        self.assertIsNone(res["drive_link"])
        self.assertTrue(any("Drive" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()
