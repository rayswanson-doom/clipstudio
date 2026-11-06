"""render_engine: easings, interpolate/spring/Timeline, timeline alignment, audio mix command and real renders."""
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import render_engine as R  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


class EasingTests(unittest.TestCase):
    def test_endpoints_for_every_family_and_mode(self):
        for fam in R.EASING_FAMILIES:
            for mode in ("in", "out", "inOut"):
                name = f"{fam}.{mode}"
                self.assertAlmostEqual(R.ease(name, 0.0), 0.0, delta=1e-6, msg=name)
                self.assertAlmostEqual(R.ease(name, 1.0), 1.0, delta=1e-6, msg=name)

    def test_known_values_and_defaults(self):
        self.assertEqual(R.ease("linear", 0.3), 0.3)
        self.assertAlmostEqual(R.ease("power2.in", 0.5), 0.125)
        self.assertAlmostEqual(R.ease("power2.out", 0.5), 0.875)
        self.assertEqual(R.ease("power2", 0.4), R.ease("power2.out", 0.4))  # GSAP: bare family = .out
        self.assertAlmostEqual(R.ease("power3.inOut", 0.5), 0.5)
        self.assertGreater(max(R.ease("back.out", t / 100) for t in range(101)), 1.05)  # overshoots
        self.assertGreater(max(R.ease("elastic.out", t / 100) for t in range(101)), 1.0)
        self.assertEqual(R.ease("power2.out", -3), 0.0)
        self.assertEqual(R.ease("power2.out", 9), 1.0)

    def test_monotonic_for_plain_curves(self):
        for name in ("power1.out", "power3.inOut", "sine.out", "expo.out", "circ.in"):
            vals = [R.ease(name, t / 50) for t in range(51)]
            self.assertEqual(vals, sorted(vals), name)

    def test_unknown_names(self):
        with self.assertRaises(ValueError):
            R.ease("wobble.out", 0.5)
        with self.assertRaises(ValueError):
            R.ease("power2.sideways", 0.5)


class InterpolateTests(unittest.TestCase):
    def test_basic_clamp_extend(self):
        self.assertEqual(R.interpolate(5, [0, 10], [0, 100]), 50)
        self.assertEqual(R.interpolate(-5, [0, 10], [0, 100]), 0)
        self.assertEqual(R.interpolate(15, [0, 10], [0, 100]), 100)
        self.assertEqual(R.interpolate(15, [0, 10], [0, 100], extrapolate_right="extend"), 150)
        self.assertEqual(R.interpolate(15, [0, 10], [0, 100], extrapolate_right="identity"), 15)

    def test_multi_point_and_easing(self):
        self.assertEqual(R.interpolate(15, [0, 10, 20], [0, 100, 0]), 50)
        self.assertAlmostEqual(R.interpolate(5, [0, 10], [0, 100], "power2.in"), 12.5)

    def test_bad_ranges(self):
        with self.assertRaises(ValueError):
            R.interpolate(1, [0, 0], [0, 1])
        with self.assertRaises(ValueError):
            R.interpolate(1, [0, 1, 2], [0, 1])


class SpringTests(unittest.TestCase):
    def test_settles_at_one_and_starts_at_zero(self):
        self.assertEqual(R.spring(0, 30), 0.0)
        for damping in (4, 10, 40):
            self.assertAlmostEqual(R.spring(300, 30, damping=damping), 1.0, delta=0.01, msg=damping)

    def test_underdamped_overshoots_and_clamping_prevents_it(self):
        peak = max(R.spring(f, 30, damping=4) for f in range(60))
        self.assertGreater(peak, 1.05)
        self.assertLessEqual(max(R.spring(f, 30, damping=4, overshoot_clamping=True) for f in range(60)), 1.0)

    def test_critically_damped(self):
        v = R.spring(10, 30, damping=20, stiffness=100)
        self.assertTrue(0 < v < 1)


class TimelineTests(unittest.TestCase):
    def test_tween_chain(self):
        tl = R.Timeline().to("y", 0, 1, 0, from_=100, ease="linear").to("y", 2, 1, 50, ease="linear")
        self.assertEqual(tl.value("y", -1), 100)
        self.assertEqual(tl.value("y", 0.5), 50)
        self.assertEqual(tl.value("y", 1.5), 0)       # holds between tweens
        self.assertEqual(tl.value("y", 2.5), 25)      # second tween starts from the first's end value
        self.assertEqual(tl.value("y", 9), 50)
        self.assertEqual(tl.value("nope", 1, default=7), 7)
        self.assertEqual(tl.duration, 3)

    def test_bad_ease_fails_early(self):
        with self.assertRaises(ValueError):
            R.Timeline().to("x", 0, 1, 1, ease="bogus.out")


class OverlayStateTests(unittest.TestCase):
    def test_every_animation_enters_and_leaves(self):
        for name in R.ANIMATIONS:
            a0, _, _, _ = R.overlay_state(name, 0.0, 3.0)
            amid, dx, dy, sc = R.overlay_state(name, 1.5, 3.0)
            self.assertEqual((amid, dx, dy, sc), (1.0, 0.0, 0.0, 1.0), name)
            if name != "none":
                self.assertLess(a0, 0.05, name)
                self.assertLess(R.overlay_state(name, 2.99, 3.0)[0], 0.2, name)

    def test_slide_directions(self):
        self.assertGreater(R.overlay_state("slide_up", 0.05, 3)[2], 0)    # starts below
        self.assertLess(R.overlay_state("slide_down", 0.05, 3)[2], 0)
        self.assertGreater(R.overlay_state("slide_left", 0.05, 3)[1], 0)  # starts to the right
        self.assertLess(R.overlay_state("slide_right", 0.05, 3)[1], 0)
        self.assertLess(R.overlay_state("pop", 0.05, 3)[3], 0.9)

    def test_unknown(self):
        with self.assertRaises(ValueError):
            R.overlay_state("spin", 0, 1)

    def test_ltx_status_is_honest(self):
        st = R.ltx_video_status()
        self.assertIn("available", st)
        self.assertTrue(st["reason"])


class DrawingTests(unittest.TestCase):
    def test_text_layer(self):
        layer = R.render_text_layer("Free shipping on everything over fifty dollars today", 540)
        self.assertEqual(layer.shape[2], 4)
        self.assertLessEqual(layer.shape[1], int(540 * 0.86) + 2)
        self.assertGreater(layer[..., 3].max(), 200)
        self.assertGreater(int(layer[..., :3].max()), 200)  # white text exists
        one = R.render_text_layer("Hi", 540)
        self.assertLess(one.shape[0], layer.shape[0])

    def test_procedural_background_moves_and_is_cheap(self):
        bg = R.ProceduralBackground((270, 480), "aurora")
        a, b = bg.frame(0.0), bg.frame(4.0)
        self.assertEqual(a.shape, (480, 270, 3))
        self.assertGreater(float(np.abs(a.astype(int) - b.astype(int)).mean()), 1.0)
        for style in R.BROLL_STYLES:
            R.ProceduralBackground((64, 96), style).frame(1.0)
        with self.assertRaises(ValueError):
            R.ProceduralBackground((64, 96), "neon")


def _ff(*args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        t = Path(cls.tmp)
        cls.voice, cls.music = str(t / "v.wav"), str(t / "m.wav")
        cls.bg, cls.av, cls.img = str(t / "bg.mp4"), str(t / "av.mp4"), str(t / "i.png")
        _ff("-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-ac", "1", cls.voice)
        _ff("-f", "lavfi", "-i", "sine=frequency=90:duration=1", "-ac", "1", cls.music)
        _ff("-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=2", "-pix_fmt", "yuv420p", cls.bg)
        _ff("-f", "lavfi", "-i", "color=c=red:s=240x320:r=25:d=2", "-pix_fmt", "yuv420p", cls.av)
        cv2.imwrite(cls.img, np.full((100, 100, 3), (0, 255, 0), np.uint8))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def comp(self, **kw):
        return R.Composition((270, 480), 20, **kw)

    def frame(self, path, t):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        cap.release()
        self.assertTrue(ok)
        return f

    def test_duration_comes_from_voiceover(self):
        c = self.comp()
        c.add_audio(self.voice)
        self.assertAlmostEqual(c.duration, 3.0, delta=0.1)
        self.assertEqual(self.comp(duration=5).duration, 5)

    def test_series_and_align(self):
        c = self.comp(duration=10)
        a = c.add_broll(None, 0, 2)
        b = c.add_broll(None, 0, 3)
        end = c.series([a, b], start=1, gap=0.5)
        self.assertEqual((a.start, a.end, b.start, b.end, end), (1, 3, 3.5, 6.5, 6.5))
        t = c.add_text("Hello", 0, 1)
        c.align(t, b, "end", 0.25)
        self.assertEqual((t.start, t.end), (6.75, 7.75))
        c.align(t, a, "start", -5)
        self.assertEqual(t.start, 0.0)
        with self.assertRaises(ValueError):
            c.align(t, a, "middle")
        with self.assertRaises(ValueError):
            c.series([c.add_broll(None, 0, None)])

    def test_validate_reports_problems(self):
        c = self.comp(duration=4)
        self.assertTrue(any("no voiceover" in p for p in c.validate()))
        c.add_audio(self.voice)
        c.add_audio(self.voice, start=1.0)
        c.add_broll(None, 0, 9)
        c.add_text("late", 6, 7)
        c.add_avatar(self.av, 0, None)
        c.add_avatar(self.av, 0, None)
        probs = " | ".join(c.validate())
        for needle in ("overlap", "past the end", "after the end", "More than one avatar"):
            self.assertIn(needle, probs)
        self.assertIn("empty", self.comp().validate()[0])

    def test_argument_errors(self):
        c = self.comp(duration=3)
        with self.assertRaises(FileNotFoundError):
            c.add_audio("/nope.wav")
        with self.assertRaises(FileNotFoundError):
            c.add_broll("/nope.mp4")
        with self.assertRaises(FileNotFoundError):
            c.add_avatar("/nope.mp4")
        with self.assertRaises(ValueError):
            c.add_avatar(self.av, position="Custom Coordinates")
        with self.assertRaises(ValueError):
            c.add_text("x", 0, 1, animation="spin")
        with self.assertRaises(ValueError):
            c.add_text("  ", 0, 1)
        with self.assertRaises(ValueError):
            c.add_text("x", 2, 1)
        with self.assertRaises(ValueError):
            c.add_audio(self.voice, role="narrator")
        with self.assertRaises(ValueError):
            R.Composition((8, 8))

    def test_audio_command_shapes(self):
        c = self.comp()
        c.add_audio(self.voice)
        cmd = " ".join(c.build_audio_command("o.wav"))
        self.assertIn("amix=inputs=1", cmd)
        self.assertNotIn("sidechaincompress", cmd)
        c.add_audio(self.music, role="music", volume=0.2, loop=True, fade_out=1, start=0.5)
        cmd = " ".join(c.build_audio_command("o.wav"))
        self.assertIn("sidechaincompress", cmd)
        self.assertIn("aloop", cmd)
        self.assertIn("adelay=500|500", cmd)
        self.assertIn("volume=0.200", cmd)
        self.assertIn("anullsrc", " ".join(self.comp(duration=2).build_audio_command("o.wav")))

    def test_ocr_overlays_stack_repeat_and_clip(self):
        c = self.comp(duration=7)
        segs = [{"text": "ONE", "start": 0.0, "end": 1.5}, {"text": "TWO", "start": 0.5, "end": 2.0},
                {"text": "TOO SHORT", "start": 0.0, "end": 0.1}, {"text": "  ", "start": 0, "end": 3}]
        items = c.add_ocr_overlays(segs, source_duration=3.0, loop=True)
        self.assertEqual({i.data["text"] for i in items}, {"ONE", "TWO"})
        starts = sorted((round(i.start, 2), i.data["text"], i.data["slot"]) for i in items)
        self.assertEqual([s[:2] for s in starts], [(0.0, "ONE"), (0.5, "TWO"), (3.0, "ONE"), (3.5, "TWO"), (6.0, "ONE"), (6.5, "TWO")])
        self.assertEqual(starts[0][2], 0)
        self.assertEqual(starts[1][2], 1)  # overlapping in time -> stacked
        self.assertLessEqual(max(i.end for i in items), 7.0)
        self.assertTrue(all(i.data["group"] == "ocr" for i in items))
        # without looping only the first pass is shown
        c2 = self.comp(duration=7)
        self.assertEqual(len(c2.add_ocr_overlays(segs, source_duration=3.0, loop=False)), 2)

    def test_full_render_all_tracks(self):
        c = self.comp()
        c.add_audio(self.voice)
        c.add_audio(self.music, role="music", volume=0.2, loop=True)
        c.add_broll(self.bg, motion="none")
        c.add_avatar(self.av, 0.0, None, "Top-Left", None, 0.4, animation="none")
        c.add_text("HELLO WORLD", 0.5, 2.5, "bottom", "fade")
        out = str(Path(self.tmp) / "full.mp4")
        res = c.render(out)
        self.assertEqual(res["size"], (270, 480))
        self.assertAlmostEqual(res["duration"], 3.0, delta=0.1)
        self.assertEqual(sorted({r["track"] for r in res["tracks"]}), ["audio", "avatar", "broll", "text"])
        info = subprocess.run([FFMPEG, "-i", out], capture_output=True, text=True).stderr
        self.assertIn("Audio:", info)
        self.assertIn("270x480", info)
        f = self.frame(out, 1.5)
        x, y, w, h = res["avatar"]["rect"]
        self.assertGreater(int(f[y + h // 2, x + w // 2][2]), 150)       # red avatar
        self.assertGreater(int(f[470, 5][0]), 150)                      # blue B-roll in the far corner
        # caption pixels: the text box exists near 80 % height and is darker/brighter than the plain B-roll
        band = f[int(480 * 0.70):int(480 * 0.82), 20:250].astype(int)
        self.assertGreater(len(np.unique(band.reshape(-1, 3), axis=0)), 5)

    def test_captions_only_when_active(self):
        c = self.comp()
        c.add_audio(self.voice)
        c.add_broll(self.bg, motion="none")
        c.add_text("HELLO WORLD", 1.0, 2.0, "center", "none")
        out = str(Path(self.tmp) / "cap.mp4")
        c.render(out)
        before, during = self.frame(out, 0.5), self.frame(out, 1.5)
        self.assertGreater(float(np.abs(before.astype(int) - during.astype(int)).mean()), 1.0)
        self.assertLess(float(np.abs(self.frame(out, 0.5).astype(int) - self.frame(out, 2.6).astype(int)).mean()), 4.0)

    def test_avatar_animation_slides_in(self):
        c = self.comp()
        c.add_audio(self.voice)
        c.add_broll(self.bg, motion="none")
        res_c = c.add_avatar(self.av, 0.0, None, "Bottom-Right", None, 0.4, animation="slide_up")
        out = str(Path(self.tmp) / "slide.mp4")
        res = c.render(out)
        x, y, w, h = res["avatar"]["rect"]
        early = self.frame(out, 0.05)
        late = self.frame(out, 1.5)
        self.assertLess(int(early[y + h // 2, x + w // 2][2]), 100)   # not there yet
        self.assertGreater(int(late[y + h // 2, x + w // 2][2]), 150)  # settled in place

    def test_image_and_procedural_broll_crossfade(self):
        c = self.comp()
        c.add_audio(self.voice)
        a = c.add_broll(self.img, 0, 2, motion="none")
        b = c.add_broll(None, 1.5, None, style="mono", motion="none")
        out = str(Path(self.tmp) / "xf.mp4")
        c.render(out)
        g = self.frame(out, 0.5)
        self.assertGreater(int(g[240, 135][1]), 200)                    # green image
        self.assertLess(int(self.frame(out, 2.8)[240, 135][1]), 150)    # procedural mono backdrop
        mid = self.frame(out, 1.75)[240, 135]
        self.assertTrue(40 < int(mid[1]) < 250)                         # in the middle of the cross-fade

    def test_render_commentary_helper_with_ocr_loop_and_music(self):
        out = str(Path(self.tmp) / "rc.mp4")
        segs = [{"text": "SALE ENDS TODAY", "start": 0.2, "end": 1.5}]
        res = R.render_commentary(self.voice, out, self.bg, self.av, "Bottom-Left", None, 0.3, segs,
                                  music_path=self.music, size=(270, 480), fps=20)
        self.assertTrue(Path(out).is_file())
        self.assertEqual(res["avatar"]["preset"], "Bottom-Left")
        texts = [r for r in res["tracks"] if r["track"] == "text"]
        self.assertEqual(len(texts), 2)  # the 2 s background loops once inside the 3 s voiceover
        res2 = R.render_commentary(self.voice, str(Path(self.tmp) / "rc2.mp4"), None, None, size=(270, 480), fps=20)
        self.assertIsNone(res2["avatar"])
        self.assertTrue(any(r["name"].startswith("procedural") for r in res2["tracks"]))

    def test_progress_and_errors(self):
        c = self.comp()
        c.add_audio(self.voice)
        seen = []
        c.render(str(Path(self.tmp) / "p.mp4"), progress=seen.append)
        self.assertEqual(seen[-1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(seen, seen[1:])))
        with self.assertRaises(ValueError):
            self.comp().render(str(Path(self.tmp) / "empty.mp4"))


if __name__ == "__main__":
    unittest.main()
