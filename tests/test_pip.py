"""Podcast picture-in-picture: geometry, frame layers and a real ffmpeg composite."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import video_engine as ve  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
BASE = (1080, 1920)
AV = (480, 640)


class RectTests(unittest.TestCase):
    def test_corners(self):
        w = None
        rects = {p: ve.compute_pip_rect(BASE, AV, p, None, 0.36) for p in ve.PIP_PRESETS[:4]}
        x, y, w, h = rects["Bottom-Right"]
        self.assertEqual(x + w + max(12, int(1080 * 0.04)), 1080)
        self.assertEqual(y + h + max(12, int(1080 * 0.04)), 1920)
        self.assertLess(rects["Top-Left"][0], rects["Top-Right"][0])
        self.assertLess(rects["Top-Left"][1], rects["Bottom-Left"][1])
        self.assertEqual(rects["Top-Left"][:2], (43, 43))
        self.assertEqual(rects["Bottom-Left"][0], rects["Top-Left"][0])
        self.assertEqual(rects["Top-Right"][1], rects["Top-Left"][1])
        for r in rects.values():
            self.assertEqual(r[2:], rects["Top-Left"][2:])

    def test_aspect_and_scale(self):
        _, _, w, h = ve.compute_pip_rect(BASE, AV, "Top-Left", None, 0.5)
        self.assertEqual(w, 540)
        self.assertAlmostEqual(h / w, 640 / 480, delta=0.01)
        w2 = ve.compute_pip_rect(BASE, AV, "Top-Left", None, 0.25)[2]
        self.assertLess(w2, w)

    def test_scale_clamped(self):
        self.assertEqual(ve.compute_pip_rect(BASE, AV, "Top-Left", None, 5.0)[2],
                         ve.compute_pip_rect(BASE, AV, "Top-Left", None, ve.PIP_MAX_SCALE)[2])
        self.assertEqual(ve.compute_pip_rect(BASE, AV, "Top-Left", None, 0.0)[2],
                         ve.compute_pip_rect(BASE, AV, "Top-Left", None, ve.PIP_MIN_SCALE)[2])

    def test_custom_exact_and_clamped(self):
        x, y, w, h = ve.compute_pip_rect(BASE, AV, "Custom Coordinates", (100, 200), 0.3)
        self.assertEqual((x, y), (100, 200))
        x, y, w, h = ve.compute_pip_rect(BASE, AV, "Custom Coordinates", (5000, 9000), 0.3)
        self.assertEqual((x + w, y + h), BASE)
        x, y, _, _ = ve.compute_pip_rect(BASE, AV, "Custom Coordinates", (-50, -9), 0.3)
        self.assertEqual((x, y), (0, 0))

    def test_custom_needs_xy(self):
        with self.assertRaises(ValueError):
            ve.compute_pip_rect(BASE, AV, "Custom Coordinates", None, 0.3)

    def test_aliases_and_errors(self):
        for alias, want in (("bottom right", "Bottom-Right"), ("top_left", "Top-Left"), ("BL", "Bottom-Left"),
                            ("custom", "Custom Coordinates"), ("Custom Coordinates (X, Y pixels)", "Custom Coordinates")):
            self.assertEqual(ve.normalize_position_preset(alias), want)
        with self.assertRaises(ValueError):
            ve.normalize_position_preset("middle")

    def test_landscape_base_and_even_sizes(self):
        x, y, w, h = ve.compute_pip_rect((1920, 1080), AV, "Bottom-Right", None, 0.6)
        self.assertLessEqual(y + h, 1080)
        self.assertEqual((w % 2, h % 2), (0, 0))

    def test_bad_sizes(self):
        with self.assertRaises(ValueError):
            ve.compute_pip_rect((0, 10), AV)


class LayerTests(unittest.TestCase):
    def test_hole_border_shadow(self):
        rect = (300, 700, 360, 480)
        deco, mask = ve.build_pip_frame_layers((1080, 1920), rect)
        self.assertEqual(deco.shape, (1920, 1080, 4))
        self.assertEqual(mask.shape, (480, 360))
        self.assertGreater(deco[940, 480, 3], 0)     # shadow sits under the avatar (the avatar is laid on top)
        self.assertEqual(mask[240, 180], 255)        # opaque avatar
        self.assertEqual(mask[0, 0], 0)              # rounded corner is cut
        self.assertGreater(deco[700 + 240, 301, 3], 200)  # border ring on the left edge
        self.assertGreater(int(deco[700 + 240, 301, 0]), 200)  # white-ish
        below = deco[700 + 480 + 8, 480]             # shadow just under the frame
        self.assertGreater(below[3], 20)
        self.assertLess(int(below[0]), 60)           # dark
        self.assertEqual(deco[10, 10, 3], 0)         # far away: nothing

    def test_no_shadow(self):
        deco, _ = ve.build_pip_frame_layers((400, 400), (100, 100, 200, 200), shadow=False)
        self.assertEqual(deco[100 + 200 + 6, 200, 3], 0)

    def test_border_colour(self):
        deco, _ = ve.build_pip_frame_layers((400, 400), (100, 100, 200, 200), border_color=(255, 0, 0))
        px = deco[200, 101]  # BGR(A)
        self.assertGreater(int(px[2]), 200)
        self.assertLess(int(px[0]), 60)


class PreviewTests(unittest.TestCase):
    def test_preview_places_tile(self):
        img, rect = ve.preview_layout(BASE, AV, "Top-Right", None, 0.4, thumb_height=480)
        self.assertEqual(img.shape[0], 480)
        k = 480 / 1920
        x, y, w, h = (int(round(v * k)) for v in rect)
        self.assertGreater(x + w, img.shape[1] * 0.8)
        self.assertLess(y, 480 * 0.2)

    def test_preview_uses_real_frames(self):
        base = np.full((90, 160, 3), (0, 200, 0), np.uint8)
        av = np.full((64, 48, 3), (0, 0, 220), np.uint8)
        img, rect = ve.preview_layout((1920, 1080), AV, "Bottom-Left", None, 0.3, base, av, 270)
        self.assertGreater(int(img[5, img.shape[1] - 5, 1]), 150)  # green backdrop
        x, y, w, h = (int(round(v * 270 / 1080)) for v in rect)
        c = img[y + h // 2, x + w // 2]
        self.assertGreater(int(c[2]), 150)                          # red avatar tile


def _make(path, size, secs, colour, audio=True, fps=25):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s={size[0]}x{size[1]}:r={fps}:d={secs}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


def _frame(path, t=0.5):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PlaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        t = Path(cls.tmp)
        cls.base = t / "base.mp4"
        cls.av = t / "av.mp4"
        cls.av_silent = t / "av_silent.mp4"
        _make(cls.base, (360, 640), 2, "blue", audio=True)
        _make(cls.av, (240, 320), 1, "red", audio=True)
        _make(cls.av_silent, (240, 320), 2, "red", audio=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_preset(self):
        for preset in ve.PIP_PRESETS[:4]:
            out = Path(self.tmp) / f"o_{preset}.mp4"
            r = ve.place_podcast_avatar(str(self.base), str(self.av), preset, None, 0.4, str(out))
            self.assertEqual(r["preset"], preset)
            f = _frame(out)
            self.assertEqual(f.shape[:2], (640, 360))
            x, y, w, h = r["rect"]
            cx, cy = x + w // 2, y + h // 2
            px = f[cy, cx].astype(int)
            self.assertGreater(px[2], 150, preset)   # red avatar inside the rect
            self.assertLess(px[0], 100, preset)
            # opposite corner is still base (blue)
            ox = 5 if x > 180 else 355
            oy = 5 if y > 320 else 635
            self.assertGreater(int(f[oy, ox][0]), 150, preset)

    def test_custom_coordinates_and_border(self):
        out = Path(self.tmp) / "custom.mp4"
        r = ve.place_podcast_avatar(str(self.base), str(self.av), "Custom Coordinates", (60, 100), 0.5, str(out))
        x, y, w, h = r["rect"]
        self.assertEqual((x, y), (60, 100))
        f = _frame(out)
        self.assertGreater(int(f[y + h // 2, x + 1:x + 3].mean()), 180)  # white border ring on the left edge
        self.assertEqual(int(f[y + h // 2, x - 40][0] > 150), 1)           # base outside (blue), shadow is faint

    def test_audio_modes(self):
        def has_audio(p):
            return "Audio:" in subprocess.run([FFMPEG, "-i", str(p)], capture_output=True, text=True).stderr
        for mode in ("base", "avatar", "mix"):
            out = Path(self.tmp) / f"a_{mode}.mp4"
            ve.place_podcast_avatar(str(self.base), str(self.av), "Top-Left", None, 0.4, str(out), audio_source=mode)
            self.assertTrue(has_audio(out), mode)
        out = Path(self.tmp) / "a_silent.mp4"
        ve.place_podcast_avatar(str(self.base), str(self.av_silent), "Top-Left", None, 0.4, str(out), audio_source="avatar")
        self.assertTrue(has_audio(out))  # falls back to the base soundtrack

    def test_length_follows_base_and_short_avatar_holds(self):
        out = Path(self.tmp) / "len.mp4"
        r = ve.place_podcast_avatar(str(self.base), str(self.av), "Top-Right", None, 0.4, str(out))
        self.assertAlmostEqual(r["duration"], 2.0, delta=0.15)
        f = _frame(out, 1.8)  # the avatar clip is only 1 s long: last frame is held
        x, y, w, h = r["rect"]
        self.assertGreater(int(f[y + h // 2, x + w // 2][2]), 150)

    def test_errors(self):
        with self.assertRaises(FileNotFoundError):
            ve.place_podcast_avatar("/nope.mp4", str(self.av))
        with self.assertRaises(ValueError):
            ve.place_podcast_avatar(str(self.base), str(self.av), "Custom Coordinates", None)
        with self.assertRaises(ValueError):
            ve.place_podcast_avatar(str(self.base), str(self.av), "Top-Left", audio_source="x")
        bad = Path(self.tmp) / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(RuntimeError):
            ve.place_podcast_avatar(str(self.base), str(bad))

    def test_default_output_name(self):
        r = ve.place_podcast_avatar(str(self.base), str(self.av))
        self.assertTrue(r["output_path"].endswith("base_podcast.mp4"))
        self.assertTrue(Path(r["output_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
