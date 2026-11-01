"""Real BlazeFace (the bundled .tflite) on a real photograph - no stand-in
detector. Uses the public-domain NASA astronaut photo that ships with
scikit-image. Skips if mediapipe, scikit-image, ffmpeg or the model are absent.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
MODEL = ROOT / "models" / "blaze_face_short_range.tflite"

try:
    import cv2
    import numpy as np
    from skimage import data
    import video_engine as ve
    MISSING = None if MODEL.is_file() and shutil.which("ffmpeg") else "model or ffmpeg missing"
except ImportError as exc:
    MISSING = f"dependency missing: {exc.name}"

W, H, FPS, SECS = 640, 360, 30, 3


def talking_head():
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    return cv2.resize(img[0:300, 100:400], (H, H))  # head-and-shoulders framing


def frame_with_face(face, x):
    f = np.full((H, W, 3), 20, np.uint8)
    f[:, x : x + H] = face
    return f


@unittest.skipIf(MISSING, MISSING or "")
class RealBlazeFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ve.create_face_detector(str(MODEL))
        cls.face = talking_head()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.detector.close()
        except Exception:
            pass

    def test_finds_a_real_face_and_reports_its_position(self):
        left = self.detector(frame_with_face(self.face, 0))
        right = self.detector(frame_with_face(self.face, W - H))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertLess(left, 0.5)
        self.assertGreater(right, 0.5)

    def test_no_face_in_empty_frame(self):
        self.assertIsNone(self.detector(np.full((H, W, 3), 20, np.uint8)))

    def test_missing_explicit_model_path_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            ve.create_face_detector("/no/such/model.tflite")

    def test_end_to_end_crop_follows_a_moving_real_face(self):
        with tempfile.TemporaryDirectory() as d:
            silent, src, out = (os.path.join(d, n) for n in ("s.mp4", "src.mp4", "out.mp4"))
            wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            n = FPS * SECS
            for i in range(n):
                wr.write(frame_with_face(self.face, int((W - H) * i / (n - 1))))
            wr.release()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i",
                 f"sine=duration={SECS}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-shortest", src], check=True)

            res = ve.create_vertical_short(src, out, upload=False)  # default = real BlazeFace
            self.assertGreaterEqual(res["faces_found"], int(0.9 * res["frames"]))

            # the face must stay inside the 9:16 crop more often than a fixed centre crop allows
            def face_in_crop_fraction(path):
                cap, seen, total = cv2.VideoCapture(path), 0, 0
                det = ve.create_face_detector(str(MODEL))
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    total += 1
                    seen += det(fr) is not None
                cap.release()
                det.close()
                return seen / max(total, 1)

            self.assertGreater(face_in_crop_fraction(out), 0.85)

            centre = os.path.join(d, "centre.mp4")
            ve.create_vertical_short(src, centre, upload=False, detector=lambda f: None)
            self.assertLess(face_in_crop_fraction(centre), face_in_crop_fraction(out))


if __name__ == "__main__":
    unittest.main()
