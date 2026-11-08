"""meme_matcher: sentiment, library scan/management, planning rules, LLM tags, starter pack, video splice + pipeline."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import meme_matcher as M  # noqa: E402
import video_engine as V  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def make_video(path, w=270, h=480, seconds=6.0, fps=10, color=(40, 90, 160)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        f = np.full((h, w, 3), color, np.uint8)
        cv2.putText(f, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        vw.write(f)
    vw.release()
    return str(path)


class FakeLLM:
    offline = False

    def __init__(self, reply):
        self.reply = reply

    def complete_json(self, *a, **k):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    chat_json = json_chat = complete_json


class SentimentTests(unittest.TestCase):
    def test_emotions(self):
        self.assertEqual(M.analyze_text("LOL that was hilarious!!")["emotion"], "laugh")
        self.assertEqual(M.analyze_text("No way, I can't believe it!")["emotion"], "shock")
        self.assertEqual(M.analyze_text("ugh what an awkward fail")["emotion"], "cringe")

    def test_negation_and_neutral(self):
        self.assertEqual(M.analyze_text("I was not happy, it's not funny")["emotion"], "neutral")
        self.assertEqual(M.analyze_text("The table has four legs.")["emotion"], "neutral")
        self.assertEqual(M.analyze_text("")["emotion"], "neutral")

    def test_intensity_grows_with_emphasis(self):
        a = M.analyze_text("that was funny")["intensity"]
        b = M.analyze_text("THAT WAS SO FUNNY!!!")["intensity"]
        self.assertGreater(b, a)
        self.assertLessEqual(b, 1.0)

    def test_llm_none_and_offline_fall_back(self):
        self.assertEqual(M.analyze_with_llm(None, ["a", "b"]), [None, None])

        class Off:
            offline = True
        self.assertEqual(M.analyze_with_llm(Off(), ["a"]), [None])

    def test_llm_tags_used_and_bad_ones_ignored(self):
        reply = {"tags": [{"idx": 0, "emotion": "Shock", "intensity": 0.9}, {"idx": 1, "emotion": "banana", "intensity": 1},
                          {"idx": 7, "emotion": "laugh", "intensity": 1}]}
        out = M.analyze_with_llm(FakeLLM(reply), ["a", "b"])
        self.assertEqual(out[0]["emotion"], "shock")
        self.assertEqual(out[0]["source"], "llm")
        self.assertIsNone(out[1])

    def test_llm_failure_never_raises(self):
        out = M.analyze_with_llm(FakeLLM(RuntimeError("boom")), ["hello"])
        self.assertEqual(out, [None])


class SegmentTests(unittest.TestCase):
    def test_text_segments_cover_duration(self):
        segs = M.segments_from_text("One two. Three four five six! Seven?", 10.0)
        self.assertEqual(len(segs), 3)
        self.assertAlmostEqual(segs[-1]["end"], 10.0, places=2)
        self.assertEqual(M.segments_from_text("", 5), [])

    def test_word_segments_relative(self):
        words = [{"word": "Wow.", "start": 10.0, "end": 10.4}, {"word": "Next", "start": 11.5, "end": 11.9}]
        segs = M.segments_from_words(words, 10.0, 12.0)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 0.0)

    def test_shots(self):
        s = M.segments_from_shots([{"idx": 1, "start": 0, "end": 2, "line": "Hi!"}, {"idx": 2, "start": 2, "end": 3}])
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0]["shot"], 1)


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class LibraryAndPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.lib_dir = cls.tmp / "memes"
        M.ensure_starter_pack(str(cls.lib_dir), ffmpeg_path=FFMPEG)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def lib(self):
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        shutil.copytree(self.lib_dir, d / "m")
        return M.MemeLibrary(str(d / "m"), ffmpeg_path=FFMPEG)

    def test_starter_pack(self):
        lib = M.MemeLibrary(str(self.lib_dir), ffmpeg_path=FFMPEG).scan()
        self.assertEqual(len(lib.clips), 10)
        self.assertEqual(len(lib.summary()["per_emotion"]), 10)
        self.assertEqual(lib.failed, [])

    def test_scan_cache_and_failed(self):
        lib = self.lib()
        lib.scan()
        (lib.dir / "broken.mp4").write_bytes(b"not a video")
        lib2 = M.MemeLibrary(str(lib.dir), ffmpeg_path=FFMPEG).scan(force=True)
        self.assertEqual(len(lib2.clips), 10)
        self.assertEqual(len(lib2.failed), 1)
        self.assertTrue((lib.dir / ".meme_index.json").is_file())

    def test_add_and_remove_validation(self):
        lib = self.lib()
        good = (lib.dir / "laugh" / "starter_laugh.mp4").read_bytes()
        clip = lib.add_file("my funny.mp4", good, "laugh")
        self.assertIn("laugh", clip["emotions"])
        with self.assertRaises(ValueError):
            lib.add_file("x.exe", b"abc")
        with self.assertRaises(ValueError):
            lib.add_file("x.mp4", b"")
        with self.assertRaises(ValueError):
            lib.add_file("x.mp4", good, "banana")
        with self.assertRaises(ValueError):
            lib.add_file("junk.mp4", b"garbage bytes")
        self.assertFalse(list(lib.dir.glob("junk*")))
        self.assertFalse(lib.remove("../outside.mp4"))
        self.assertTrue(lib.remove(Path(clip["path"]).relative_to(lib.dir).as_posix()))

    def test_plan_matches_emotion(self):
        lib = self.lib()
        cfg = M.MemeSettings(enabled=True, min_score=0.2, max_per_minute=20)
        segs = [{"start": 0, "end": 3, "text": "LOL that was hilarious!!"},
                {"start": 5, "end": 8, "text": "I am so sad and heartbroken, I cry."}]
        plan = M.plan_memes(segs, lib, cfg, 10.0, frame_size=(270, 480))
        emos = [m["emotion"] for m in plan["matches"]]
        self.assertEqual(emos, ["laugh", "sad"])
        for m in plan["matches"]:
            self.assertTrue(os.path.isfile(m["meme_path"]))
            self.assertLessEqual(m["end"], 10.0)

    def test_budget_threshold_and_neutral(self):
        lib = self.lib()
        segs = [{"start": i * 3, "end": i * 3 + 2.5, "text": "LOL so funny!!!"} for i in range(8)]
        plan = M.plan_memes(segs, lib, M.MemeSettings(enabled=True, max_per_minute=2), 24.0, frame_size=(270, 480))
        self.assertLessEqual(len(plan["matches"]), 1)  # ceil(24/60*2)=1
        self.assertTrue(any("limit" in s["reason"] for s in plan["skipped"]))
        calm = M.plan_memes([{"start": 0, "end": 3, "text": "The meeting is at noon."}], lib,
                            M.MemeSettings(enabled=True), 10.0, frame_size=(270, 480))
        self.assertEqual(calm["matches"], [])

    def test_min_gap_and_no_immediate_repeat(self):
        lib = self.lib()
        segs = [{"start": 0, "end": 2, "text": "LOL!!! hilarious"}, {"start": 2.2, "end": 4, "text": "haha so funny!!!"}]
        plan = M.plan_memes(segs, lib, M.MemeSettings(enabled=True, max_per_minute=20), 5.0, frame_size=(270, 480))
        starts = sorted(m["start"] for m in plan["matches"])
        for a, b in zip(starts, starts[1:]):
            self.assertGreater(b - a, 0.5)
        lib.record_usage(plan["matches"])
        self.assertTrue(lib.usage())

    def test_overlay_avoids_avatar_rect(self):
        lib = self.lib()
        avatar = (150, 330, 110, 140)  # bottom-right
        plan = M.plan_memes([{"start": 0, "end": 3, "text": "I am so sad, crying"}], lib,
                            M.MemeSettings(enabled=True, mode="overlay", min_score=0.2), 5.0,
                            avoid_rects=[avatar], frame_size=(270, 480))
        self.assertEqual(len(plan["matches"]), 1)
        x, y, w, h = plan["matches"][0]["rect"]
        ax, ay, aw, ah = avatar
        self.assertFalse(x < ax + aw and ax < x + w and y < ay + ah and ay < y + h)

    def test_custom_coordinates_and_clamp(self):
        lib = self.lib()
        cfg = M.MemeSettings(enabled=True, mode="overlay", position="Custom Coordinates", xy=(10, 20), min_score=0.2)
        plan = M.plan_memes([{"start": 2, "end": 3.0, "text": "LOL hilarious!!"}], lib, cfg, 8.0, frame_size=(270, 480), clamp=True)
        m = plan["matches"][0]
        self.assertEqual(m["rect"][:2], (10, 20))
        self.assertGreaterEqual(m["start"], 2.0)
        self.assertLessEqual(m["end"], 3.0)

    def test_settings_validation(self):
        with self.assertRaises(ValueError):
            M.MemeSettings.from_dict({"mode": "nope"})
        with self.assertRaises(ValueError):
            M.MemeSettings.from_dict({"position": "Custom Coordinates"})
        s = M.MemeSettings.from_dict({"enabled": True, "bogus": 1, "max_per_minute": 999})
        self.assertTrue(s.enabled)
        self.assertEqual(s.max_per_minute, 20.0)

    def test_empty_library_is_not_an_error(self):
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        lib = M.MemeLibrary(str(d), ffmpeg_path=FFMPEG)
        plan = M.plan_memes([{"start": 0, "end": 2, "text": "LOL"}], lib, M.MemeSettings(enabled=True), 5.0, frame_size=(270, 480))
        self.assertEqual(plan["matches"], [])

    def test_default_dir_env(self):
        with mock.patch.dict(os.environ, {"CINEPULSE_MEMES": str(self.tmp / "envmemes")}):
            self.assertEqual(Path(M.default_meme_dir()), self.tmp / "envmemes")


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class SpliceAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.memes = cls.tmp / "memes"
        M.ensure_starter_pack(str(cls.memes), ffmpeg_path=FFMPEG)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _frames(self, path, t_list):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = []
        for t in t_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            out.append(f if ok else None)
        cap.release()
        return out

    def test_apply_memes_changes_pixels_and_keeps_size(self):
        from backend import pipeline as P
        src = make_video(self.tmp / "base.mp4")
        before = self._frames(src, [1.0])[0].astype(int)
        warnings = []
        segs = [{"start": 0.2, "end": 2.0, "text": "LOL that was hilarious!!!"}]
        cfg = {"enabled": True, "mode": "overlay", "library_dir": str(self.memes), "min_score": 0.2}
        applied = P._apply_memes(src, segs, cfg, [], FFMPEG, warnings)
        self.assertEqual(len(applied), 1, warnings)
        cap = cv2.VideoCapture(src)
        self.assertEqual((int(cap.get(3)), int(cap.get(4))), (270, 480))
        cap.release()
        after = self._frames(src, [1.0])[0].astype(int)
        self.assertGreater(np.abs(after - before).mean(), 1.0)  # a reaction tile is drawn

    def test_apply_memes_disabled_or_empty_keeps_file(self):
        from backend import pipeline as P
        src = make_video(self.tmp / "keep.mp4")
        h = Path(src).read_bytes()
        w = []
        self.assertEqual(P._apply_memes(src, [{"start": 0, "end": 1, "text": "LOL"}], {"enabled": False}, [], FFMPEG, w), [])
        self.assertEqual(P._apply_memes(src, [], {"enabled": True, "library_dir": str(self.memes)}, [], FFMPEG, w), [])
        empty = self.tmp / "emptylib"
        self.assertEqual(P._apply_memes(src, [{"start": 0, "end": 2, "text": "LOL hilarious!!!"}],
                                        {"enabled": True, "library_dir": str(empty)}, [], FFMPEG, w), [])
        self.assertTrue(any("empty" in x for x in w))
        self.assertEqual(Path(src).read_bytes(), h)

    def test_apply_memes_bad_settings_never_raises(self):
        from backend import pipeline as P
        src = make_video(self.tmp / "bad.mp4")
        w = []
        self.assertEqual(P._apply_memes(src, [{"start": 0, "end": 1, "text": "x"}], {"enabled": True, "mode": "zzz"}, [], FFMPEG, w), [])
        self.assertTrue(any("Could not add" in x for x in w))

    def test_vugola_pipeline_uploads_meme_clip(self):
        """The clip uploaded to Drive is the one that already contains the reaction."""
        from backend import pipeline as P
        src = make_video(self.tmp / "long.mp4", seconds=8)
        words = [{"word": w, "start": 0.5 + i * 0.4, "end": 0.8 + i * 0.4} for i, w in enumerate(
            "LOL that was absolutely hilarious!! I cannot stop laughing at this".split())]
        uploaded = {}

        def fake_create(local, target, start_time, end_time, **k):
            shutil.copy(local, target)
            return {"output_path": target, "faces_found": 1}

        class FakeBackup:
            def __init__(self, *a, **k): pass
            def submit(self, path):
                uploaded["bytes"] = Path(path).read_bytes()
                uploaded["path"] = path
                from concurrent.futures import Future
                f = Future(); f.set_result("https://drive/x"); return f
            def close(self): pass

        pick = {"start": 0.0, "end": 7.0, "score": 1, "reasons": [], "text": "x"}
        cfg = {"enabled": True, "mode": "overlay", "library_dir": str(self.memes), "min_score": 0.2}
        with mock.patch.object(P, "download_media", return_value=src), \
             mock.patch.object(P, "transcribe_audio", return_value=words), \
             mock.patch.object(P, "find_viral_clips", return_value=[pick]), \
             mock.patch.object(P, "create_vertical_short", side_effect=fake_create), \
             mock.patch.object(P, "DriveBackup", FakeBackup), \
             mock.patch.object(P, "_usable_folder", return_value="folder"):
            res = P.run_vugola_pipeline(src, "folder", output_dir=str(self.tmp / "out"), ffmpeg_path=FFMPEG, memes=cfg)
        clip = res["clips"][0]
        self.assertIsNone(clip["error"], res["warnings"])
        self.assertGreaterEqual(len(clip["memes"]), 1)
        self.assertEqual(clip["drive_link"], "https://drive/x")
        self.assertEqual(uploaded["bytes"], Path(clip["output_path"]).read_bytes())
        self.assertNotEqual(uploaded["bytes"], Path(src).read_bytes())  # it really contains the reaction


if __name__ == "__main__":
    unittest.main()


class CollectFinalVideosTests(unittest.TestCase):
    def test_collects_unique_existing_videos(self):
        import uploader as U
        d = Path(tempfile.mkdtemp())
        try:
            a, b, c = d / "a.mp4", d / "b.mp4", d / "c.mp4"
            for f in (a, b, c):
                f.write_bytes(b"x")
            res = {"video_path": str(a), "clips": [{"output_path": str(b)}, {"output_path": None}, {"output_path": str(a)}],
                   "episodes_out": [{"video_path": str(c)}, {"video_path": str(d / "gone.mp4")}]}
            self.assertEqual(U.collect_final_videos(res), [str(a), str(b), str(c)])
            self.assertEqual(U.collect_final_videos({}), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)
