"""Heavy engines (faster-whisper, mediapipe, OCR) must not load until a run needs them, and must be released after."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ["faster_whisper", "ctranslate2", "mediapipe", "paddleocr", "paddle", "rapidocr_onnxruntime", "onnxruntime"]


class LazyLoadingTests(unittest.TestCase):
    def test_importing_every_backend_module_loads_no_heavy_engine(self):
        code = (
            "import sys; sys.path.insert(0, 'backend')\n"
            "import pipeline, transcriber, video_engine, avatar_sync, ocr_engine, visual_cues, scorer, tts_engine\n"
            f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
            "print('LOADED:' + ','.join(bad))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "LOADED:")

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
