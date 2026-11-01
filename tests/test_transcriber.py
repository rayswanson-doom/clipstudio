"""Runs with plain `python -m unittest` or pytest. If faster-whisper is not
installed, a stub module is injected so the logic can still be verified; the
real model is only exercised when the package is present and RUN_REAL_WHISPER=1.
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import transcriber  # noqa: E402


class W:
    def __init__(self, word, start, end, p):
        self.word, self.start, self.end, self.probability = word, start, end, p


class Seg:
    def __init__(self, words):
        self.words = words


class FakeModel:
    def __init__(self, *a, **k):
        FakeModel.init_args = (a, k)

    def transcribe(self, path, **kw):
        FakeModel.kw = kw
        info = types.SimpleNamespace(language="en", language_probability=0.99, duration=2.0)
        segs = [Seg([W(" Hello", 0.0, 0.4251, 0.98765)]), Seg([W(" world", 0.5, 1.0, 0.9)]), Seg(None)]
        return iter(segs), info


class TranscriberTests(unittest.TestCase):
    def setUp(self):
        transcriber._model = None
        fake_mod = types.ModuleType("faster_whisper")
        fake_mod.WhisperModel = FakeModel
        self.patcher = mock.patch.dict(sys.modules, {"faster_whisper": fake_mod})
        self.patcher.start()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.tmp.write(b"RIFF")
        self.tmp.close()

    def tearDown(self):
        self.patcher.stop()
        os.unlink(self.tmp.name)
        transcriber._model = None

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            transcriber.transcribe_audio("/no/such/file.wav")

    def test_model_config(self):
        transcriber.transcribe_audio(self.tmp.name)
        args, kwargs = FakeModel.init_args
        self.assertEqual(args, ("base",))
        self.assertEqual(kwargs, {"device": "cpu", "compute_type": "int8"})

    def test_word_timestamps_enabled(self):
        transcriber.transcribe_audio(self.tmp.name)
        self.assertTrue(FakeModel.kw["word_timestamps"])

    def test_output_shape(self):
        out = transcriber.transcribe_audio(self.tmp.name)
        self.assertEqual(
            out,
            [
                {"word": "Hello", "start": 0.0, "end": 0.425, "probability": 0.9877},
                {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.9},
            ],
        )

    def test_model_cached(self):
        transcriber.transcribe_audio(self.tmp.name)
        first = transcriber._model
        transcriber.transcribe_audio(self.tmp.name)
        self.assertIs(first, transcriber._model)


@unittest.skipUnless(os.environ.get("RUN_REAL_WHISPER") == "1", "set RUN_REAL_WHISPER=1")
class RealModelTest(unittest.TestCase):
    def test_real(self):
        sample = os.environ["WHISPER_SAMPLE"]
        out = transcriber.transcribe_audio(sample)
        self.assertTrue(out and {"word", "start", "end"} <= set(out[0]))


if __name__ == "__main__":
    unittest.main()
