"""Uses a stub `edge_tts` module so it runs without the package or internet.
Set RUN_REAL_TTS=1 (with edge-tts installed and internet access) for a live test.
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import tts_engine  # noqa: E402


class FakeCommunicate:
    fail = False
    empty = False
    last = None

    def __init__(self, text, voice, rate="+0%", volume="+0%", pitch="+0Hz"):
        FakeCommunicate.last = dict(text=text, voice=voice, rate=rate, volume=volume, pitch=pitch)

    async def save(self, path):
        Path(path).write_bytes(b"" if FakeCommunicate.empty else b"ID3fakeaudio")
        if FakeCommunicate.fail:
            raise ConnectionError("no network")


async def fake_list_voices():
    return [
        {"ShortName": "en-US-AriaNeural", "Locale": "en-US"},
        {"ShortName": "fr-FR-DeniseNeural", "Locale": "fr-FR"},
        {"ShortName": "en-GB-SoniaNeural", "Locale": "en-GB"},
    ]


class TTSTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeCommunicate.fail = FakeCommunicate.empty = False
        mod = types.ModuleType("edge_tts")
        mod.Communicate = FakeCommunicate
        mod.list_voices = fake_list_voices
        self.p = mock.patch.dict(sys.modules, {"edge_tts": mod})
        self.p.start()
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.p.stop()
        self.dir.cleanup()

    async def test_creates_file_and_nested_dirs(self):
        out = os.path.join(self.dir.name, "a", "b", "v.mp3")
        got = await tts_engine.generate_voiceover("  Hello world  ", out)
        self.assertEqual(got, str(Path(out).resolve()))
        self.assertTrue(Path(out).is_file())
        self.assertFalse(Path(out + ".part").exists())
        self.assertEqual(FakeCommunicate.last["text"], "Hello world")
        self.assertEqual(FakeCommunicate.last["voice"], tts_engine.DEFAULT_VOICE)

    async def test_custom_voice_rate_pitch(self):
        out = os.path.join(self.dir.name, "v.mp3")
        await tts_engine.generate_voiceover("Hi", out, voice="en-GB-SoniaNeural", rate="+10%", pitch="-5Hz")
        self.assertEqual(FakeCommunicate.last["voice"], "en-GB-SoniaNeural")
        self.assertEqual(FakeCommunicate.last["rate"], "+10%")
        self.assertEqual(FakeCommunicate.last["pitch"], "-5Hz")

    async def test_validation(self):
        with self.assertRaises(ValueError):
            await tts_engine.generate_voiceover("   ", "x.mp3")
        with self.assertRaises(ValueError):
            await tts_engine.generate_voiceover("hi", "")

    async def test_failure_cleans_partial_and_wraps(self):
        FakeCommunicate.fail = True
        out = os.path.join(self.dir.name, "v.mp3")
        with self.assertRaises(RuntimeError):
            await tts_engine.generate_voiceover("Hi", out)
        self.assertFalse(Path(out).exists())
        self.assertFalse(Path(out + ".part").exists())

    async def test_empty_audio_is_error(self):
        FakeCommunicate.empty = True
        out = os.path.join(self.dir.name, "v.mp3")
        with self.assertRaises(RuntimeError):
            await tts_engine.generate_voiceover("Hi", out)
        self.assertFalse(Path(out).exists())

    async def test_list_voices_filters_locale(self):
        self.assertEqual(await tts_engine.list_voices("en"),
                         ["en-GB-SoniaNeural", "en-US-AriaNeural"])

    def test_sync_wrapper(self):
        out = os.path.join(self.dir.name, "s.mp3")
        self.assertTrue(Path(tts_engine.generate_voiceover_sync("Hi", out)).is_file())


@unittest.skipUnless(os.environ.get("RUN_REAL_TTS") == "1", "set RUN_REAL_TTS=1")
class RealTTS(unittest.TestCase):
    def test_real(self):
        with tempfile.TemporaryDirectory() as d:
            out = tts_engine.generate_voiceover_sync("This is a test.", os.path.join(d, "t.mp3"))
            self.assertGreater(os.path.getsize(out), 1000)


if __name__ == "__main__":
    unittest.main()
