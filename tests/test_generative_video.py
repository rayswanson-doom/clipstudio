"""Generative video backends (stub pipelines - no GPU, weights or network), the fallback loop, word-boundary caption timing
and the generative / 2D visual modes of the scenic renderer."""
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import asset_manager as am
from backend import generative_video as gv
from backend import pipeline
from backend import render_engine as R
from backend import transcriber as tr
from backend import vimax_orchestrator as vo

FFMPEG = shutil.which("ffmpeg")


def spoken(text, per=0.3, gap=0.05, t0=0.2):
    out, t = [], t0
    for w in text.split():
        out.append({"word": w, "start": round(t, 3), "end": round(t + per, 3), "probability": 0.9})
        t += per + gap
    return out


class AlignTests(unittest.TestCase):
    def test_exact_match_uses_whisper_times(self):
        words = tr.align_words("Hello big world.", spoken("hello big world"))
        self.assertEqual([w["word"] for w in words], ["Hello", "big", "world."])
        self.assertTrue(all(w["exact"] for w in words))
        self.assertAlmostEqual(words[1]["t0"], 0.55, places=2)
        self.assertAlmostEqual(words[2]["t1"], 1.2, places=2)

    def test_punctuation_and_case_are_ignored(self):
        words = tr.align_words("Don't STOP, believing!", spoken("dont stop believing"))
        self.assertTrue(all(w["exact"] for w in words))

    def test_missed_word_is_interpolated_between_neighbours(self):
        sp = spoken("one two four five")
        words = tr.align_words("one two three four five", sp)
        three = words[2]
        self.assertFalse(three["exact"])
        self.assertGreaterEqual(three["t0"], words[1]["t1"] - 1e-6)
        self.assertLessEqual(three["t1"], words[3]["t0"] + 1e-6)

    def test_times_are_monotonic_and_non_empty(self):
        words = tr.align_words("a b c d e f", spoken("a b x d e f"))
        prev = -1
        for w in words:
            self.assertGreater(w["t1"], w["t0"])
            self.assertGreaterEqual(w["t0"], prev)
            prev = w["t0"]

    def test_mismatch_returns_none_so_the_caller_estimates(self):
        self.assertIsNone(tr.align_words("completely different words here", spoken("nothing in common at all")))
        self.assertIsNone(tr.align_words("hello", []))

    def test_word_timings_never_raises(self):
        def broken(path):
            raise RuntimeError("model missing")

        words, why = tr.word_timings("x.mp3", "hello world", transcribe_fn=broken)
        self.assertIsNone(words)
        self.assertIn("unavailable", why)
        words, src = tr.word_timings("x.mp3", "hello world", transcribe_fn=lambda p: spoken("hello world"))
        self.assertEqual(src, "whisper")
        self.assertEqual(len(words), 2)


class ResampleTests(unittest.TestCase):
    def frames(self, n):
        return [np.full((8, 8, 3), i * 10, np.uint8) for i in range(n)]

    def test_length_and_pingpong(self):
        out = gv.resample_frames(self.frames(5), 20, src_fps=10, fps=10)
        self.assertEqual(len(out), 20)
        vals = [int(f[0, 0, 0]) for f in out]
        self.assertEqual(vals[:5], [0, 10, 20, 30, 40])
        self.assertEqual(vals[5:9], [30, 20, 10, 0])  # plays backward, no jump back to frame 0

    def test_slow_model_is_time_stretched_with_blending(self):
        out = gv.resample_frames(self.frames(3), 9, src_fps=3, fps=9, pingpong=False)
        vals = [int(f[0, 0, 0]) for f in out]
        self.assertEqual(vals[0], 0)
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        self.assertIn(vals[1], range(1, 10))  # an in-between frame, not a repeat

    def test_empty(self):
        with self.assertRaises(gv.GenVideoError):
            gv.resample_frames([], 5)


class FakePipe:
    """Stands in for LTXImageToVideoPipeline / StableVideoDiffusionPipeline: returns moving PIL frames."""

    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def __call__(self, **kw):
        from PIL import Image

        self.calls.append(kw)
        w, h = kw["width"], kw["height"]
        n = kw["num_frames"]
        frames = []
        for i in range(n):
            a = np.zeros((h, w, 3), np.uint8)
            a[:, :] = (30, 60, 90)
            cv2.circle(a, (int(w * (0.2 + 0.6 * i / max(1, n - 1))), h // 2), max(4, w // 8), (255, 200, 50), -1)
            frames.append(Image.fromarray(a))
        return types.SimpleNamespace(frames=[frames])


def init_png(tmp, size=(90, 160)):
    p = os.path.join(tmp, "init.png")
    cv2.imwrite(p, np.full((size[1], size[0], 3), 120, np.uint8))
    return p


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_ltx_adapter_uses_8k_plus_1_frames_and_32_multiples(self):
        pipe = FakePipe("ltx")
        b = gv.DiffusersVideo("ltx", pipe_factory=lambda k: pipe)
        frames = b.generate(init_png(self.tmp), "a prompt", (540, 960), 2.0, 12, seed=3)
        kw = pipe.calls[0]
        self.assertEqual((kw["num_frames"] - 1) % 8, 0)
        self.assertEqual(kw["width"] % 32, 0)
        self.assertEqual(kw["height"] % 32, 0)
        self.assertEqual(kw["prompt"], "a prompt")
        self.assertIn("image", kw)  # image-to-video: the locked first frame is the condition
        self.assertEqual(len(frames), 24)
        self.assertEqual(frames[0].shape[:2], (960, 540))

    def test_svd_adapter_takes_25_frames_and_64_multiples(self):
        pipe = FakePipe("svd")
        frames = gv.DiffusersVideo("svd", pipe_factory=lambda k: pipe).generate(init_png(self.tmp), "p", (540, 960), 1.5, 10)
        kw = pipe.calls[0]
        self.assertEqual(kw["num_frames"], 25)
        self.assertEqual(kw["width"] % 64, 0)
        self.assertEqual(len(frames), 15)

    def test_frames_actually_move(self):
        frames = gv.DiffusersVideo("ltx", pipe_factory=lambda k: FakePipe(k)).generate(init_png(self.tmp), "p", (90, 160), 1.0, 12)
        self.assertGreater(np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean(), 1.0)

    def test_local_backends_are_unavailable_without_packages_or_gpu(self):
        b = gv.DiffusersVideo("ltx")
        with mock.patch.object(gv, "_has", return_value=False):
            with self.assertRaises(gv.BackendUnavailable):
                b.check()
        with mock.patch.object(gv, "_has", return_value=True), mock.patch.object(gv, "_cuda", return_value=False), \
                mock.patch.dict(os.environ, {"CINEPULSE_GENVIDEO_ALLOW_CPU": ""}):
            with self.assertRaises(gv.BackendUnavailable) as cm:
                b.check()
        self.assertIn("GPU", str(cm.exception))
        with mock.patch.object(gv, "_has", return_value=True), mock.patch.object(gv, "_cuda", return_value=True), \
                mock.patch.object(gv, "_weights_cached", return_value=False):
            with self.assertRaises(gv.BackendUnavailable) as cm:
                b.check()
        self.assertIn("not downloaded", str(cm.exception))

    def test_pollinations_needs_key_and_internet(self):
        with mock.patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            with self.assertRaises(gv.BackendUnavailable):
                gv.PollinationsVideo(api_key="").check()
        with self.assertRaises(gv.BackendUnavailable) as cm:
            gv.PollinationsVideo(api_key="k", probe=lambda: False).check()
        self.assertIn("internet", str(cm.exception))

    def test_pollinations_sends_key_in_header_not_url(self):
        seen = {}
        mp4 = os.path.join(self.tmp, "r.mp4")
        gv.write_video([np.full((64, 64, 3), 50 + i * 5, np.uint8) for i in range(12)], mp4, 12, FFMPEG)
        with open(mp4, "rb") as fh:
            data = fh.read()

        def transport(url, headers):
            seen.update(url=url, headers=headers)
            return data

        frames = gv.PollinationsVideo(api_key="SECRET", model="wan", transport=transport).generate("", "a cat", (90, 160), 2.0, 12)
        self.assertNotIn("SECRET", seen["url"])
        self.assertEqual(seen["headers"]["Authorization"], "Bearer SECRET")
        self.assertIn("model=wan", seen["url"])
        self.assertIn("aspectRatio=9:16", seen["url"])
        self.assertEqual(len(frames), 24)

    def test_pollinations_payment_error_is_unavailable(self):
        class Http(Exception):
            code = 402

        def transport(url, headers):
            raise Http("payment required")

        with self.assertRaises(gv.BackendUnavailable):
            gv.PollinationsVideo(api_key="k", transport=transport).generate("", "p", (90, 160), 1.0, 12)


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.init = init_png(self.tmp)

    def run_shot(self, factory, idx=0):
        return factory.shot_video(idx, self.init, "p", (90, 160), 1.0, 12, os.path.join(self.tmp, f"g{idx}.mp4"))

    def test_first_working_backend_wins(self):
        f = gv.VideoFactory("auto", pipe_factory=lambda k: FakePipe(k), ffmpeg=FFMPEG)
        with mock.patch.object(gv.DiffusersVideo, "check", lambda self: None):
            rec = self.run_shot(f)
        self.assertEqual(rec["backend"], "ltx")
        self.assertTrue(rec["image_conditioned"])
        self.assertTrue(os.path.isfile(rec["path"]))
        self.assertEqual(int(cv2.VideoCapture(rec["path"]).get(cv2.CAP_PROP_FRAME_COUNT)), 12)

    def test_falls_through_to_next_backend_and_remembers_failures(self):
        calls = {"ltx": 0}

        def factory(kind):
            if kind == "ltx":
                calls["ltx"] += 1
                raise RuntimeError("CUDA out of memory")
            return FakePipe(kind)

        f = gv.VideoFactory("auto", pipe_factory=factory, ffmpeg=FFMPEG)
        with mock.patch.object(gv.DiffusersVideo, "check", lambda self: None):
            a, b = self.run_shot(f, 0), self.run_shot(f, 1)
        self.assertEqual((a["backend"], b["backend"]), ("svd", "svd"))
        self.assertEqual(calls["ltx"], 1)  # marked down after the first failure
        self.assertTrue(any("CUDA out of memory" in w for w in f.warnings))

    def test_offline_machine_with_nothing_installed_returns_none_with_reasons(self):
        f = gv.VideoFactory("auto", ffmpeg=FFMPEG, probe=lambda: False, api_key="")
        with mock.patch.object(gv, "_has", return_value=False), mock.patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            self.assertIsNone(self.run_shot(f))
            self.assertFalse(f.usable)
        self.assertEqual(set(f.down), {"ltx", "svd", "pollinations"})
        self.assertEqual(len(f.warnings), 3)

    def test_online_backend_losing_internet_falls_back(self):
        f = gv.VideoFactory("pollinations", api_key="k", probe=lambda: False, ffmpeg=FFMPEG)
        self.assertIsNone(self.run_shot(f))
        self.assertIn("internet", f.warnings[0])

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            gv.VideoFactory("sora")


class PromptTests(unittest.TestCase):
    def test_prompt_uses_cards(self):
        p = gv.build_motion_prompt({"action": "Maya stirs a pot.", "location": "Kitchen", "time_of_day": "NIGHT", "kind": "establishing"},
                                   {"name": "Kitchen", "lighting": "warm lamp light", "atmosphere": "cosy"},
                                   [{"name": "Maya", "traits": {"hair": "long black hair"}, "attire": "red apron"}])
        for part in ("Maya stirs a pot", "long black hair", "red apron", "warm lamp light", "NIGHT", "establishing"):
            self.assertIn(part, p)
        self.assertLessEqual(len(p), 900)

    def test_capabilities_lists_all_backends(self):
        with mock.patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            caps = gv.capabilities()
        self.assertEqual([c["backend"] for c in caps], ["ltx", "svd", "pollinations"])
        self.assertTrue(all(c["detail"] for c in caps))


class KineticTimingTests(unittest.TestCase):
    def test_exact_word_times_drive_the_highlight(self):
        comp = R.Composition((180, 320), 12, duration=3.0, ffmpeg_path=FFMPEG)
        wt = [{"word": "Hello", "t0": 0.2, "t1": 0.5}, {"word": "there.", "t0": 0.6, "t1": 1.1}, {"word": "Bye", "t0": 2.0, "t1": 2.4}]
        it = comp.add_kinetic_caption("Hello there. Bye", 0.05, 2.9, word_times=wt)
        self.assertEqual([w["chunk"] for w in it.data["words"]], [0, 0, 1])  # a sentence end starts a new chunk
        self.assertEqual(it.data["words"][2]["t0"], 2.0)  # not spread by length


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class RenderModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def episode(self):
        from tests.test_episode_studio import plan_episode

        plan = plan_episode()
        book = am.AssetBook(os.path.join(self.tmp, "book"), plan["asset_cards"])
        am.generate_images(book, backend="draw")
        vo.close_asset_gate(plan, book)
        return plan, book

    def render(self, mode, factory=None, timer=None, **kw):
        plan, book = self.episode()
        out = os.path.join(self.tmp, f"ep_{mode}.mp4")
        return vo.render_storyboard(plan, out, work_dir=out + "_w", assets=book, size=(180, 320), fps=12, visual_mode=mode,
                                    video_factory=factory, word_timer=timer, max_shots=3, **kw)

    def stub_factory(self):
        f = gv.VideoFactory("ltx", pipe_factory=lambda k: FakePipe(k), ffmpeg=FFMPEG)
        patcher = mock.patch.object(gv.DiffusersVideo, "check", lambda self: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        return f

    def test_2d_mode_is_the_default_and_makes_no_generative_clips(self):
        res = self.render("2d")
        self.assertEqual(res["visual_mode"], "2d")
        self.assertEqual(res["generative"], [])
        self.assertTrue(all(c["visual"] == "2d" for c in res["conditioning"]))

    def test_generative_mode_renders_model_clips_conditioned_on_locked_frames(self):
        res = self.render("generative", self.stub_factory(), timer=lambda *a, **k: (None, "off"))
        self.assertEqual(res["visual_mode"], "generative")
        self.assertEqual(len(res["generative"]), 3)
        g = res["generative"][0]
        self.assertEqual(g["backend"], "ltx")
        self.assertTrue(g["image_conditioned"])
        self.assertTrue(os.path.isfile(g["init_frame"]))
        self.assertTrue(g["init_sha1"])
        self.assertIn("character", g["prompt"].lower() + "character")
        self.assertTrue(all(c["visual"] == "ltx" for c in res["conditioning"]))
        cap = cv2.VideoCapture(res["output_path"])
        self.assertGreater(cap.get(cv2.CAP_PROP_FRAME_COUNT), 10)

    def test_generative_clip_replaces_the_scenery_pan(self):
        a = self.render("2d")
        b = self.render("generative", self.stub_factory())
        ca, cb = cv2.VideoCapture(a["shots"][0]["clip"]), cv2.VideoCapture(b["shots"][0]["clip"])
        ca.set(cv2.CAP_PROP_POS_FRAMES, 8)
        cb.set(cv2.CAP_PROP_POS_FRAMES, 8)
        fa, fb = ca.read()[1], cb.read()[1]
        self.assertGreater(np.abs(fa[100:220].astype(int) - fb[100:220].astype(int)).mean(), 5)  # a different picture

    def test_generative_mode_without_any_backend_falls_back_to_2d_and_says_so(self):
        f = gv.VideoFactory("auto", ffmpeg=FFMPEG, probe=lambda: False, api_key="")
        with mock.patch.object(gv, "_has", return_value=False), mock.patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            res = self.render("generative", f)
        self.assertEqual(res["generative"], [])
        self.assertTrue(os.path.isfile(res["output_path"]))
        text = " ".join(res["warnings"])
        self.assertIn("2D cinematic composition was rendered", text)
        self.assertIn("diffusers", text)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.render("hologram")

    def test_caption_timing_uses_timer_with_stubbed_voice(self):
        def fake_voice(text, path, voice=None, **kw):
            import subprocess

            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1.4", path], check=True)
            return path

        seen = []

        def timer(audio, text, audio_seconds=0.0):
            seen.append(text)
            return tr.align_words(text, spoken(text, per=0.2)), "whisper"

        with mock.patch("backend.tts_engine.generate_voiceover_sync", fake_voice):
            res = self.render("2d", timer=timer, voice=True)
        self.assertTrue(res["voice_used"])
        self.assertTrue(seen)
        self.assertTrue(all(t["source"] == "whisper" for t in res["caption_timing"]))

    def test_caption_timing_falls_back_to_estimate_with_a_warning(self):
        def fake_voice(text, path, voice=None, **kw):
            import subprocess

            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=1.4", path], check=True)
            return path

        with mock.patch("backend.tts_engine.generate_voiceover_sync", fake_voice):
            res = self.render("2d", timer=lambda *a, **k: (None, "speech recognition unavailable (no model)"), voice=True)
        self.assertTrue(all(t["source"] == "estimated" for t in res["caption_timing"]))
        self.assertTrue(any("Caption timing is estimated" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()
