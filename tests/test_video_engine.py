"""Real OpenCV + ffmpeg tests using a synthetic video (white square moving
left->right, with a sine-wave audio track). A tiny detector that finds the
white square stands in for BlazeFace so no model file is needed.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import video_engine as ve  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
W, H, FPS, SECS = 640, 360, 30, 3


def square_detector(frame):
    ys, xs = np.where(frame[:, :, 0] > 200)
    return float(xs.mean() / frame.shape[1]) if xs.size else None


def none_detector(frame):
    return None


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def make_source(path, with_audio=True):
    silent = path + ".silent.mp4"
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = FPS * SECS
    for i in range(n):
        f = np.full((H, W, 3), 30, np.uint8)
        x = int(100 + (540 - 100) * i / (n - 1))
        f[160:200, x - 20 : x + 20] = 255
        wr.write(f)
    wr.release()
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", silent]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={SECS}", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    os.unlink(silent)


def visible_fraction(path):
    cap = cv2.VideoCapture(path)
    seen = total = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        total += 1
        seen += int((fr[:, :, 0] > 200).sum() > 200)
    cap.release()
    return seen / max(total, 1)


class PureLogicTests(unittest.TestCase):
    def test_smooth_step(self):
        self.assertAlmostEqual(ve.smooth(0.8, 0.2, 0.1), 0.74)
        self.assertEqual(ve.SMOOTHING_FACTOR, 0.1)

    def test_tracker(self):
        t = ve.FaceTracker()
        self.assertEqual(t.update(None), 0.5)  # nothing seen yet -> centre
        self.assertAlmostEqual(t.update(0.8), 0.8)  # first face adopted directly
        self.assertAlmostEqual(t.update(0.2), 0.74)  # then 0.1 smoothing
        held = t.update(None)
        self.assertAlmostEqual(held, 0.74)  # hold on loss
        self.assertAlmostEqual(t.update(5.0), 0.74 + 0.1 * (1.0 - 0.74))  # clamped to 1
        with self.assertRaises(ValueError):
            ve.FaceTracker(0)

    def test_tracker_converges_geometrically(self):
        t = ve.FaceTracker(0.1)
        t.update(0.0)
        vals = [t.update(1.0) for _ in range(30)]
        self.assertAlmostEqual(vals[0], 0.1)
        self.assertAlmostEqual(vals[1], 0.19)
        self.assertGreater(vals[-1], 0.95)

    def test_crop_size(self):
        self.assertEqual(ve.compute_crop_size(1920, 1080), (608, 1080))
        self.assertEqual(ve.compute_crop_size(1080, 1920), (1080, 1920))
        cw, ch = ve.compute_crop_size(1000, 1000)  # square -> crop sides
        self.assertEqual(ch, 1000)
        self.assertEqual(cw % 2, 0)
        cw, ch = ve.compute_crop_size(500, 1200)  # narrower than 9:16 -> crop height
        self.assertEqual(cw, 500)
        self.assertLessEqual(ch, 1200)

    def test_crop_origin_clamped(self):
        self.assertEqual(ve.crop_origin(0.0, 1920, 1080, 608, 1080), (0, 0))
        self.assertEqual(ve.crop_origin(1.0, 1920, 1080, 608, 1080), (1920 - 608, 0))
        self.assertEqual(ve.crop_origin(0.5, 1920, 1080, 608, 1080)[0], 656)

    def test_ffmpeg_command(self):
        cmd = ve.build_ffmpeg_command("ff", "v.mp4", "src.mp4", "out.mp4", 1.5, 2.0)
        self.assertIn("1:a:0?", cmd)
        self.assertLess(cmd.index("-ss"), len(cmd) - cmd[::-1].index("src.mp4") - 1)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "1.500")
        self.assertEqual(cmd[-1], "out.mp4")

    def test_find_ffmpeg_prefers_bundled_bin(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "bin" / "ffmpeg.exe"
            exe.parent.mkdir()
            exe.write_bytes(b"x")
            with mock.patch.object(ve, "_app_roots", return_value=[Path(d)]):
                self.assertEqual(ve.find_ffmpeg(), str(exe))

    def test_find_ffmpeg_errors(self):
        with self.assertRaises(FileNotFoundError):
            ve.find_ffmpeg("/no/such/ffmpeg")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            ve, "_app_roots", return_value=[Path(d)]
        ), mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                ve.find_ffmpeg()

    @unittest.skipIf(hasattr(__import__("mediapipe"), "solutions"), "legacy API present")
    def test_blazeface_model_missing_is_clear(self):
        with self.assertRaises(FileNotFoundError) as cm:
            ve.create_face_detector(model_path="/no/model.tflite")
        self.assertIn(ve.MODEL_FILENAME, str(cm.exception))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.src = os.path.join(cls.dir.name, "src.mp4")
        make_source(cls.src)
        cls.src_noaudio = os.path.join(cls.dir.name, "src_noaudio.mp4")
        make_source(cls.src_noaudio, with_audio=False)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def out(self, name):
        return os.path.join(self.dir.name, "out", name)

    def run_short(self, name, **kw):
        kw.setdefault("detector", square_detector)
        kw.setdefault("ffmpeg_path", FFMPEG)
        kw.setdefault("upload", False)
        return ve.create_vertical_short(self.src, self.out(name), **kw)

    def test_full_clip_vertical_with_audio(self):
        res = self.run_short("full.mp4")
        info = probe(res["output_path"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (1080, 1920))
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        self.assertAlmostEqual(float(info["format"]["duration"]), SECS, delta=0.2)
        self.assertEqual(res["frames"], FPS * SECS)
        self.assertEqual(res["faces_found"], FPS * SECS)
        self.assertEqual(res["crop_size"], (202, 360))
        self.assertIsNone(res["drive_link"])

    def test_subrange(self):
        res = self.run_short("sub.mp4", start_time=1.0, end_time=2.0)
        info = probe(res["output_path"])
        self.assertAlmostEqual(float(info["format"]["duration"]), 1.0, delta=0.15)
        self.assertEqual(res["frames"], FPS)

    def test_tracking_keeps_subject_in_frame_better_than_center_crop(self):
        tracked = self.run_short("tracked.mp4")
        static = self.run_short("static.mp4", detector=none_detector)
        t, s = visible_fraction(tracked["output_path"]), visible_fraction(static["output_path"])
        self.assertGreater(t, 0.95)
        self.assertLess(s, 0.8)

    def test_source_without_audio_still_works(self):
        res = ve.create_vertical_short(
            self.src_noaudio, self.out("noaudio.mp4"), detector=square_detector,
            ffmpeg_path=FFMPEG, upload=False,
        )
        info = probe(res["output_path"])
        self.assertFalse(any(s["codec_type"] == "audio" for s in info["streams"]))

    def test_upload_triggered_with_folder(self):
        with mock.patch.object(ve, "upload_to_google_drive", return_value="https://drive/x") as up:
            res = self.run_short("up.mp4", upload=True, folder_id="FOLDER123")
        up.assert_called_once_with(res["output_path"], "FOLDER123")
        self.assertEqual(res["drive_link"], "https://drive/x")

    def test_upload_uses_env_folder_and_skips_without(self):
        with mock.patch.object(ve, "upload_to_google_drive", return_value="L") as up, mock.patch.dict(
            os.environ, {"GOOGLE_DRIVE_FOLDER_ID": "ENVF"}
        ):
            self.run_short("env.mp4", upload=True)
        up.assert_called_once()
        self.assertEqual(up.call_args[0][1], "ENVF")
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_DRIVE_FOLDER_ID"}
        with mock.patch.object(ve, "upload_to_google_drive") as up2, mock.patch.dict(
            os.environ, env, clear=True
        ):
            res = self.run_short("noenv.mp4", upload=True)
        up2.assert_not_called()
        self.assertIsNone(res["drive_link"])

    def test_upload_failure_propagates_but_clip_is_kept(self):
        with mock.patch.object(ve, "upload_to_google_drive", side_effect=RuntimeError("drive down")):
            with self.assertRaises(RuntimeError):
                self.run_short("keep.mp4", upload=True, folder_id="F")
        self.assertTrue(os.path.isfile(self.out("keep.mp4")))

    def test_validation_errors(self):
        with self.assertRaises(FileNotFoundError):
            ve.create_vertical_short("/no/such.mp4", self.out("x.mp4"), detector=none_detector)
        with self.assertRaises(ValueError):
            self.run_short("bad.mp4", start_time=2.0, end_time=1.0)
        with self.assertRaises(ValueError):
            self.run_short("bad.mp4", start_time=10.0)  # past the end
        with self.assertRaises(ValueError):
            self.run_short("bad.mp4", detect_every=0)

    @unittest.skipIf(hasattr(__import__("mediapipe"), "solutions"), "legacy API present")
    def test_missing_model_raises_unless_fallback(self):
        with mock.patch.dict(os.environ, {"BLAZEFACE_MODEL_PATH": "/no/model.tflite"}):
            with self.assertRaises(RuntimeError):
                ve.create_vertical_short(self.src, self.out("m.mp4"), ffmpeg_path=FFMPEG, upload=False)
            res = ve.create_vertical_short(
                self.src, self.out("m2.mp4"), ffmpeg_path=FFMPEG, upload=False,
                allow_center_fallback=True,
            )
        self.assertEqual(res["faces_found"], 0)
        self.assertTrue(os.path.isfile(res["output_path"]))

    def test_ffmpeg_failure_cleans_up(self):
        fake = os.path.join(self.dir.name, "badffmpeg.sh")
        Path(fake).write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
        outdir = os.path.join(self.dir.name, "failout")
        with self.assertRaises(RuntimeError) as cm:
            ve.create_vertical_short(
                self.src, os.path.join(outdir, "o.mp4"), detector=square_detector,
                ffmpeg_path=fake, upload=False,
            )
        self.assertIn("boom", str(cm.exception))
        self.assertEqual(os.listdir(outdir), [])  # no temp or partial files left


if __name__ == "__main__":
    unittest.main()
