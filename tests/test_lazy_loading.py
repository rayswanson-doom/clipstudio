"""Heavy engines (faster-whisper, mediapipe, OCR) must not load until a run needs them, and must be released after."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ["torch", "torchvision", "faster_whisper", "ctranslate2", "mediapipe", "paddleocr", "paddle", "rapidocr_onnxruntime", "onnxruntime"]


class LazyLoadingTests(unittest.TestCase):
    def test_importing_every_backend_module_loads_no_heavy_engine(self):
        code = (
            "import sys; sys.path.insert(0, 'backend')\n"
            "import pipeline, transcriber, video_engine, ocr_engine, scorer, meme_matcher, uploader\n"
            "import subtitle_engine, music_finder, social_metadata\n"
            f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
            "print('LOADED:' + ','.join(bad))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "LOADED:")

    def test_new_modules_use_their_heavy_libraries_only_when_called(self):
        code = (
            "import sys; sys.path.insert(0, 'backend')\n"
            "import subtitle_engine as se, music_finder as mf, social_metadata as sm\n"
            "cues = se.cues_for_clip([{'word': w, 'start': i * .3, 'end': i * .3 + .25} for i, w in enumerate('one two three four five six'.split())], 0, 3)\n"
            "se.cues_to_ass(cues); se.cues_to_srt(cues)\n"
            "sm.generate_packages('Protein builds muscle fast.', use_llm='off')\n"
            f"print('LOADED:' + ','.join(m for m in {HEAVY + ['yt_dlp', 'cv2']!r} if m in sys.modules))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:], "LOADED:")

    def test_availability_checks_do_not_import(self):
        code = (
            "import sys; sys.path.insert(0, 'backend')\n"
            "import ocr_engine, transcriber\n"
            "ocr_engine.available_backends(); ocr_engine.is_available()\n"
            f"print('LOADED:' + ','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "LOADED:")

    def test_release_memory_unloads_everything(self):
        sys.path.insert(0, str(ROOT / "backend"))
        import ocr_engine, pipeline, transcriber
        ocr_engine._engine = (lambda f: [], "fake")
        transcriber._model = object()
        pipeline.release_memory()
        self.assertIsNone(ocr_engine._engine)
        self.assertFalse(transcriber.is_loaded())


if __name__ == "__main__":
    unittest.main()
