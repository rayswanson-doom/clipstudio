"""story_recap: chronological order, normalising mixed clips, joining, subtitles/music on the recap timeline."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import story_recap as sr  # noqa: E402
import video_engine as ve  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def words(text, start, step=0.3):
    return [{"word": w, "start": round(start + i * step, 3), "end": round(start + i * step + 0.25, 3)}
            for i, w in enumerate(text.split())]


class PureTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def touch(self, name):
        p = os.path.join(self.d, name)
        Path(p).write_bytes(b"x")
        return p

    def test_clips_are_ordered_by_start_time_in_the_source(self):
        recs = [{"output_path": self.touch("c.mp4"), "start": 300, "end": 330, "tag": "c"},
                {"output_path": self.touch("a.mp4"), "start": 10, "end": 40, "tag": "a"},
                {"output_path": self.touch("b.mp4"), "start": "120.5", "end": 150, "tag": "b"}]
        self.assertEqual([r["tag"] for r in sr.order_clips(recs)], ["a", "b", "c"])  # NOT the score / list order
        self.assertEqual([r["tag"] for r in recs], ["c", "a", "b"], "the input list is not modified")

    def test_unusable_records_are_dropped_and_ties_keep_input_order(self):
        recs = [{"output_path": self.touch("a.mp4"), "start": 5, "tag": 1}, {"output_path": self.touch("b.mp4"), "start": 5, "tag": 2},
                {"output_path": os.path.join(self.d, "missing.mp4"), "start": 1}, {"output_path": None, "start": 2},
                {"output_path": self.touch("c.mp4")}, {"output_path": self.touch("d.mp4"), "start": "x"}]
        self.assertEqual([r["tag"] for r in sr.order_clips(recs)], [1, 2])
        self.assertEqual(sr.order_clips(None), [])

    def test_normalise_command_standardises_picture_and_sound(self):
        cmd = sr.build_normalize_command("ffmpeg", "in.mp4", "out.mp4", True, (1080, 1920), 30)
        j = " ".join(cmd)
        for part in ("scale=1080:1920:force_original_aspect_ratio=decrease", "pad=1080:1920", "setsar=1", "fps=30",
                     "format=yuv420p", "aresample=44100", "channel_layouts=stereo", "-c:a aac", "-ar 44100", "-ac 2", "libx264"):
            self.assertIn(part, j)
        self.assertNotIn("anullsrc", j)
        silent = " ".join(sr.build_normalize_command("ffmpeg", "in.mp4", "out.mp4", False))
        self.assertIn("anullsrc=r=44100:cl=stereo", silent)  # silence is added for clips without sound
        self.assertIn("-shortest", silent)
        self.assertIsInstance(cmd, list)  # an argument list, never a shell string

    def test_concat_list_quotes_awkward_names(self):
        text = sr.concat_list_text([os.path.join(self.d, "it's here.mp4"), os.path.join(self.d, "b.mp4")])
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("file '") and "'\\''" in lines[0])
        cmd = sr.build_concat_command("ffmpeg", "list.txt", "o.mp4")
        self.assertEqual(cmd[cmd.index("-f") + 1], "concat")
        self.assertEqual(cmd[cmd.index("-c") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-safe") + 1], "0")

    def test_flatten_words_accepts_words_or_segments(self):
        w = words("a b", 0)
        self.assertEqual(sr.flatten_words(w), w)
        self.assertEqual(sr.flatten_words([{"text": "x", "words": w}, "junk", {"nothing": 1}]), w)
        self.assertEqual(sr.flatten_words(None), [])

    def test_cues_are_shifted_onto_the_recap_timeline(self):
        spoken = words("alpha beta gamma delta", 100.0) + words("one two three four", 20.0)
        ordered = [{"start": 20.0, "end": 24.0, "duration": 4.0}, {"start": 100.0, "end": 104.0, "duration": 4.0}]
        cues = sr.recap_cues(ordered, [0.0, 4.0], spoken)
        text = [c["text"] for c in cues]
        self.assertEqual(text, ["one two three four", "alpha beta gamma delta"])  # chronological order
        self.assertAlmostEqual(cues[0]["start"], 0.0, delta=0.1)
        self.assertGreaterEqual(cues[1]["start"], 4.0)
        self.assertLessEqual(cues[1]["end"], 8.01)
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)

    def test_clips_that_already_have_subtitles_are_not_captioned_twice(self):
        spoken = words("one two three four", 20.0)
        ordered = [{"start": 20.0, "end": 24.0, "duration": 4.0, "subtitled": True}]
        self.assertEqual(sr.recap_cues(ordered, [0.0], spoken), [])

    def test_recap_text_follows_the_story_order(self):
        spoken = words("late part here", 200.0) + words("early part here", 5.0)
        ordered = [{"start": 5.0, "end": 8.0}, {"start": 200.0, "end": 203.0}]
        self.assertEqual(sr.recap_text(ordered, spoken), "early part here late part here")

    def test_needs_two_clips_and_a_real_music_file(self):
        one = [{"output_path": self.touch("a.mp4"), "start": 1, "end": 2}]
        with mock.patch.object(ve, "find_ffmpeg", return_value="ffmpeg"):
            with self.assertRaises(ValueError):
                sr.build_story_overview_video(one, [], os.path.join(self.d, "o.mp4"))
            two = one + [{"output_path": self.touch("b.mp4"), "start": 3, "end": 4}]
            with self.assertRaises(FileNotFoundError):
                sr.build_story_overview_video(two, [], os.path.join(self.d, "o.mp4"), music_path="/no/music.mp3")


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class RealVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name

        def make(name, size, fps, audio):
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration=2"]
            if audio:
                cmd += ["-f", "lavfi", "-i", f"sine=frequency={audio}:duration=2", "-shortest", "-ar", "48000"]
            subprocess.run(cmd + ["-pix_fmt", "yuv420p", os.path.join(d, name)], check=True)
            return os.path.join(d, name)

        # deliberately different sizes, frame rates, sample rates and one clip with no sound at all
        cls.early = make("early.mp4", "216x384", 25, 300)
        cls.mid = make("mid.mp4", "320x240", 24, None)
        cls.late = make("late.mp4", "180x320", 30, 600)
        cls.music = os.path.join(d, "m.mp3")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=880:duration=1", cls.music], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def recs(self):
        # scrambled on purpose (score order): late, early, mid
        return [{"output_path": self.late, "start": 900.0, "end": 902.0, "duration": 2.0},
                {"output_path": self.early, "start": 10.0, "end": 12.0, "duration": 2.0},
                {"output_path": self.mid, "start": 400.0, "end": 402.0, "duration": 2.0}]

    def info(self, path):
        return ve._probe_media(FFMPEG, path)

    def test_mixed_clips_become_one_uniform_video_in_story_order(self):
        out = os.path.join(self.dir.name, "recap1.mp4")
        res = sr.build_story_overview_video(self.recs(), [], out, burn_subtitles=False, size=(270, 480), fps=15)
        self.assertEqual(res, out)
        i = self.info(out)
        self.assertEqual((i["width"], i["height"]), (270, 480))
        self.assertTrue(i["has_audio"] and i["has_video"])
        self.assertAlmostEqual(i["duration"], 6.0, delta=0.4)  # 3 clips x 2 s, nothing lost or stretched
        # the recap really plays early -> mid -> late: the sound of clip 1 (300 Hz) comes first, clip 3 (600 Hz) last
        self.assertGreater(self.dominant_hz(out, 0.3, 1.2), 0)
        self.assertLess(abs(self.dominant_hz(out, 0.3, 1.2) - 300), 40)
        self.assertLess(abs(self.dominant_hz(out, 4.3, 1.2) - 600), 60)

    def dominant_hz(self, path, start, dur):
        import numpy as np
        raw = subprocess.run([FFMPEG, "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", path, "-vn", "-ac", "1",
                              "-ar", "8000", "-f", "s16le", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, dtype=np.int16).astype(float)
        if a.size < 100 or np.abs(a).max() < 50:
            return 0.0
        spec = np.abs(np.fft.rfft(a * np.hanning(a.size)))
        return float(np.argmax(spec) * 8000 / a.size)

    def test_silent_clip_gets_silence_so_audio_stays_in_sync(self):
        out = os.path.join(self.dir.name, "recap2.mp4")
        sr.build_story_overview_video(self.recs(), [], out, burn_subtitles=False, size=(270, 480), fps=15)
        self.assertEqual(self.dominant_hz(out, 2.3, 1.2), 0.0)  # the middle (silent) clip stays silent
        i = self.info(out)
        self.assertAlmostEqual(i["duration"], 6.0, delta=0.4)

    def test_story_overview_subtitles_are_burned_at_the_right_time(self):
        spoken = words("early words here now", 10.0) + words("late words spoken today", 900.0)
        out = os.path.join(self.dir.name, "recap3.mp4")
        flat = []  # plain dark clips, so any white pixel in the lower part can only be a subtitle
        for name, start in (("f1.mp4", 10.0), ("f2.mp4", 400.0), ("f3.mp4", 900.0)):
            p = os.path.join(self.dir.name, name)
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202020:size=270x480:rate=15:duration=2",
                            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest", "-pix_fmt", "yuv420p", p], check=True)
            flat.append({"output_path": p, "start": start, "end": start + 2.0, "duration": 2.0})
        sr.build_story_overview_video(flat[::-1], spoken, out, burn_subtitles=True, size=(270, 480), fps=15)
        srt = Path(out).with_suffix(".srt").read_text(encoding="utf-8")
        self.assertLess(srt.index("early words"), srt.index("late words"))
        self.assertRegex(srt, r"00:00:0[0-1],\d{3} -->")  # first words at the very start of the recap
        import cv2

        def lower_peak(t):
            png = out + f".{t}.png"
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(t), "-i", out, "-frames:v", "1", png], check=True)
            img = cv2.imread(png)
            return float(img[int(img.shape[0] * 0.6): int(img.shape[0] * 0.8)].min(axis=2).max())  # white pixels need all 3 channels bright

        self.assertGreater(lower_peak(0.5), 230)   # subtitle visible while the first clip plays
        self.assertLess(lower_peak(2.9), 230)      # the silent middle clip has no words

    def test_music_is_mixed_under_the_recap(self):
        out = os.path.join(self.dir.name, "recap4.mp4")
        sr.build_story_overview_video(self.recs(), [], out, music_path=self.music, music_volume=0.3, burn_subtitles=False,
                                      size=(270, 480), fps=15)
        self.assertTrue(self.dominant_hz(out, 2.2, 1.5) > 0, "music must be audible over the silent middle clip")
        self.assertAlmostEqual(self.info(out)["duration"], 6.0, delta=0.4)

    def test_temp_files_are_cleaned_up_even_when_a_clip_is_broken(self):
        sub = os.path.join(self.dir.name, "clean")
        os.makedirs(sub)
        bad = os.path.join(sub, "bad.mp4")
        Path(bad).write_bytes(b"this is not a video")
        recs = self.recs() + [{"output_path": bad, "start": 1.0, "end": 2.0, "duration": 1.0}]
        with self.assertRaises(Exception):
            sr.build_story_overview_video(recs, [], os.path.join(sub, "o.mp4"), burn_subtitles=False, size=(270, 480), fps=15)
        self.assertEqual(sorted(os.listdir(sub)), ["bad.mp4"], "no recap_* work folder or half-written output may remain")

    def test_progress_callback_errors_do_not_break_the_build(self):
        def boom(msg, frac):
            raise RuntimeError("ui gone")
        out = os.path.join(self.dir.name, "recap5.mp4")
        sr.build_story_overview_video(self.recs(), [], out, burn_subtitles=False, size=(270, 480), fps=15, progress=boom)
        self.assertTrue(os.path.isfile(out))


if __name__ == "__main__":
    unittest.main()
