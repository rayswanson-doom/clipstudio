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


class WindowedTranscriptionTests(unittest.TestCase):
    """Long media is decoded and transcribed one window at a time (flat RAM); timestamps are shifted back."""

    def test_long_media_is_split_offset_and_cleaned_up(self):
        calls = []

        class Model:
            def transcribe(self, path, **k):
                calls.append(path)
                self_exists = os.path.isfile(path)
                calls.append(self_exists)
                return iter([Seg([W(" hi ", 1.0, 1.5, 0.9)])]), types.SimpleNamespace(language="en", language_probability=1, duration=1)

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "long.mp4"
            src.write_bytes(b"x")
            made = []

            def fake_extract(ff, s, start, length, dst):
                made.append((start, length))
                Path(dst).write_bytes(b"wav")

            with mock.patch.object(transcriber, "WINDOW_SECONDS", 600), \
                    mock.patch.object(transcriber, "_extract_window", side_effect=fake_extract):
                words = transcriber._transcribe_in_windows(Model(), "ffmpeg", src, 1500.0)
        self.assertEqual(made, [(0.0, 600), (600.0, 600), (1200.0, 300.0)])
        self.assertEqual([w["start"] for w in words], [1.0, 601.0, 1201.0])
        self.assertEqual([w["word"] for w in words], ["hi"] * 3)
        self.assertTrue(all(calls[1::2]), "each window file existed while it was transcribed")
        window_dirs = {os.path.dirname(c) for c in calls[0::2]}
        self.assertEqual(len(window_dirs), 1)
        self.assertFalse(os.path.exists(window_dirs.pop()), "temp window folder is removed")

    def test_short_media_uses_the_single_pass_and_long_uses_windows(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "v.mp4"
            src.write_bytes(b"x")
            model = mock.Mock()
            model.transcribe.return_value = (iter([]), types.SimpleNamespace(language="en", language_probability=1, duration=1))
            with mock.patch.object(transcriber, "_get_model", return_value=model), \
                    mock.patch.object(transcriber, "_media_seconds", return_value=("ffmpeg", 60.0)), \
                    mock.patch.object(transcriber, "_transcribe_in_windows") as win:
                transcriber.transcribe_audio(str(src))
            win.assert_not_called()
            with mock.patch.object(transcriber, "_get_model", return_value=model), \
                    mock.patch.object(transcriber, "_media_seconds", return_value=("ffmpeg", 3 * 3600.0)), \
                    mock.patch.object(transcriber, "_transcribe_in_windows", return_value=[]) as win:
                transcriber.transcribe_audio(str(src))
            win.assert_called_once()


if __name__ == "__main__":
    unittest.main()
