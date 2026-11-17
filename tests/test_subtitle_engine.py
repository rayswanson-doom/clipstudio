"""subtitle_engine: 3-5 word chunking, .srt / styled .ass output, safe zone, and a real ffmpeg burn-in."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import subtitle_engine as se  # noqa: E402
import video_engine as ve  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


def words(text, start=0.0, step=0.3, gap_after=None):
    """Evenly timed words; gap_after = {index: seconds of silence after that word}."""
    out, t = [], start
    for i, w in enumerate(text.split()):
        out.append({"word": w, "start": round(t, 3), "end": round(t + step - 0.05, 3)})
        t += step + (gap_after or {}).get(i, 0.0)
    return out


SPEECH = ("Here's the truth: most people train hard but eat badly. Protein and sleep matter more than the gym. "
          "So eat more protein every single day and sleep eight hours.")


class ChunkTests(unittest.TestCase):
    def test_every_chunk_has_three_to_five_words(self):
        chunks = se.chunk_words(words(SPEECH))
        self.assertGreater(len(chunks), 4)
        for c in chunks:
            self.assertTrue(3 <= len(c["text"].split()) <= 5, c["text"])
            self.assertEqual(c["text"], " ".join(w["word"] for w in c["words"]))
            self.assertLess(c["start"], c["end"])

    def test_no_word_is_lost_or_reordered(self):
        chunks = se.chunk_words(words(SPEECH))
        self.assertEqual(" ".join(c["text"] for c in chunks), SPEECH)

    def test_chunks_do_not_straddle_a_sentence_end(self):
        for c in se.chunk_words(words(SPEECH)):
            body = c["text"].split()[:-1]
            self.assertFalse(any(w.endswith((".", "!", "?")) for w in body), c["text"])

    def test_a_long_pause_ends_the_chunk(self):
        chunks = se.chunk_words(words("one two three four five six", gap_after={2: 1.0}))
        self.assertEqual([c["text"] for c in chunks], ["one two three", "four five six"])

    def test_short_breaths_are_joined_and_single_words_survive(self):
        self.assertEqual([c["text"] for c in se.chunk_words(words("Wow. Really? Yes."))], ["Wow. Really? Yes."])
        self.assertEqual([c["text"] for c in se.chunk_words(words("Hello"))], ["Hello"])
        self.assertEqual(se.chunk_words([]), [])

    def test_custom_limits_and_bad_limits(self):
        for c in se.chunk_words(words("a b c d e f g h i j k l"), min_words=2, max_words=3):
            self.assertLessEqual(len(c["words"]), 3)
        with self.assertRaises(ValueError):
            se.chunk_words([], min_words=5, max_words=3)

    def test_clean_words_repairs_bad_timing(self):
        out = se.clean_words([{"word": " hi ", "start": 1, "end": 1}, {"word": "", "start": 2, "end": 3},
                              {"word": "x"}, {"word": "there", "start": 0.5, "end": 0.9}])
        self.assertEqual([w["word"] for w in out], ["hi", "there"])
        self.assertTrue(all(w["end"] > w["start"] for w in out))
        self.assertGreaterEqual(out[1]["start"], out[0]["end"] - 0.021)

    def test_transcribe_words_uses_the_injected_model(self):
        got = se.transcribe_words("a.mp4", lambda p: [{"word": "yo", "start": 0, "end": 0.2}])
        self.assertEqual(got[0]["word"], "yo")


class CueTests(unittest.TestCase):
    def test_times_are_relative_to_the_clip_and_inside_it(self):
        w = words(SPEECH, start=10.0)
        cues = se.cues_for_clip(w, 10.0, 16.0)
        self.assertTrue(cues)
        self.assertLess(cues[0]["start"], 0.5)
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a["end"], b["start"], "cues must not overlap")
        self.assertTrue(all(0 <= c["start"] < c["end"] <= 6.0 for c in cues))

    def test_words_outside_the_clip_are_ignored(self):
        cues = se.cues_for_clip(words("alpha beta gamma delta epsilon zeta", step=1.0), 2.0, 4.0)
        text = " ".join(c["text"] for c in cues)
        self.assertNotIn("alpha", text)
        self.assertNotIn("epsilon", text)
        self.assertIn("gamma", text)

    def test_quick_cue_is_held_long_enough_to_read(self):
        cues = se.cues_for_clip(words("go go go", step=0.1), 0.0, 5.0)
        self.assertGreaterEqual(cues[0]["end"] - cues[0]["start"], se.MIN_CUE_SECONDS - 0.01)

    def test_bad_range_and_empty_range(self):
        with self.assertRaises(ValueError):
            se.cues_for_clip([], 5, 5)
        self.assertEqual(se.cues_for_clip(words("a b c"), 50, 60), [])


class FormatTests(unittest.TestCase):
    cues = se.cues_for_clip(words(SPEECH), 0.0, 8.0)

    def test_srt_is_valid(self):
        srt = se.cues_to_srt(self.cues)
        blocks = srt.strip().split("\n\n")
        self.assertEqual(len(blocks), len(self.cues))
        self.assertTrue(blocks[0].startswith("1\n00:00:00,"))
        self.assertRegex(blocks[0].splitlines()[1], r"^\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}$")
        self.assertIn("HERE'S", se.cues_to_srt(self.cues, uppercase=True))

    def test_time_formats(self):
        self.assertEqual(se._srt_time(3725.5), "01:02:05,500")
        self.assertEqual(se._ass_time(3725.5), "1:02:05.50")
        self.assertEqual(se._srt_time(-1), "00:00:00,000")

    def test_ass_style_is_bold_outlined_bottom_centre_in_the_safe_zone(self):
        ass = se.cues_to_ass(self.cues, (1080, 1920))
        style = next(l for l in ass.splitlines() if l.startswith("Style: Default"))
        f = style.split(",")
        self.assertEqual(f[7], "-1")                                  # Bold
        self.assertEqual(f[5], "&H00000000")                          # black outline
        self.assertEqual(f[15], "1")                                  # BorderStyle outline + shadow
        self.assertEqual(f[18], "2")                                  # Alignment 2 = bottom centre
        self.assertEqual(int(f[21]), round(1920 * se.SAFE_BOTTOM_FRAC))  # MarginV lifts it above the app buttons
        self.assertEqual(int(f[19]), round(1080 * se.SAFE_SIDE_FRAC))
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("PlayResY: 1920", ass)
        self.assertEqual(ass.count("Dialogue:"), len(self.cues))

    def test_size_presets_scale_with_the_frame(self):
        sizes = [se.ass_style_values((1080, 1920), se.Style(size_preset=p))["fontsize"] for p in ("Small", "Medium", "Large")]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[2])
        small = se.ass_style_values((540, 960), se.Style())["fontsize"]
        self.assertLess(small, se.ass_style_values((1080, 1920), se.Style())["fontsize"])

    def test_colour_presets_and_karaoke(self):
        yellow = se.cues_to_ass(self.cues, style=se.Style(color_preset="Yellow"))
        self.assertIn("&H0000F0FF", yellow)
        self.assertNotIn("\\kf", yellow)
        karaoke = se.cues_to_ass(self.cues, style=se.Style(color_preset="White + yellow active word"))
        self.assertIn("\\kf", karaoke)
        self.assertEqual(karaoke.count("\\kf"), sum(len(c["words"]) for c in self.cues))

    def test_override_braces_in_speech_cannot_inject_ass_tags(self):
        cue = [{"start": 0, "end": 1, "text": "{\\an8}hi\\Nthere", "words": []}]
        line = [l for l in se.cues_to_ass(cue).splitlines() if l.startswith("Dialogue")][0]
        self.assertNotIn("{", line)
        self.assertNotIn("\\", line.split(",,")[-1])

    def test_uppercase(self):
        self.assertIn("HERE'S THE TRUTH", se.cues_to_ass(self.cues, style=se.Style(uppercase=True)))

    def test_unknown_presets_are_rejected(self):
        with self.assertRaises(ValueError):
            se.Style(size_preset="Huge").validate()
        with self.assertRaises(ValueError):
            se.cues_to_ass(self.cues, style=se.Style(color_preset="Pink"))

    def test_write_files(self):
        with tempfile.TemporaryDirectory() as d:
            files = se.write_subtitle_files(self.cues, os.path.join(d, "sub", "clip1"), (540, 960))
            self.assertTrue(files["srt"].endswith("clip1.srt") and files["ass"].endswith("clip1.ass"))
            self.assertIn("PlayResX: 540", Path(files["ass"]).read_text(encoding="utf-8"))
            self.assertIn("-->", Path(files["srt"]).read_text(encoding="utf-8"))


@unittest.skipUnless(FFMPEG, "ffmpeg not installed")
class BurnTests(unittest.TestCase):
    def make(self, d, size="270x480"):
        src = os.path.join(d, "src.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=0x202020:size={size}:rate=15:duration=3",
                        "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-shortest", "-pix_fmt", "yuv420p", src], check=True)
        return src

    def frame_brightness(self, path, t):
        png = path + f".{t}.png"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(t), "-i", path, "-frames:v", "1", png], check=True)
        import cv2
        img = cv2.imread(png)
        h = img.shape[0]
        return float(img[int(h * 0.55): int(h * 0.80)].max()), float(img[: int(h * 0.30)].max())

    def test_text_is_really_drawn_in_the_lower_part_of_the_picture(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(d)
            cues = [{"start": 0.0, "end": 2.5, "text": "BIG WHITE WORDS", "words": []}]
            files = se.write_subtitle_files(cues, os.path.join(d, "s"), (270, 480), se.Style(size_preset="Large"))
            out = os.path.join(d, "out.mp4")
            res = ve.burn_subtitles(src, files["ass"], out, FFMPEG)
            self.assertEqual(res["output_path"], str(Path(out).resolve()))
            lower, upper = self.frame_brightness(out, 1.0)
            self.assertGreater(lower, 230, "white subtitle pixels expected in the lower third")
            self.assertLess(upper, 100, "nothing should be drawn at the top")
            blank_lower, _ = self.frame_brightness(src, 1.0)
            self.assertLess(blank_lower, 100)
            kinds = sorted(s["codec_type"] for s in __import__("tests.test_video_engine", fromlist=["probe"]).probe(out)["streams"])
            self.assertEqual(kinds, ["audio", "video"])

    def test_missing_inputs_and_bad_subtitle_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = self.make(d)
            with self.assertRaises(FileNotFoundError):
                ve.burn_subtitles(src, os.path.join(d, "none.ass"), os.path.join(d, "o.mp4"), FFMPEG)
            with self.assertRaises(FileNotFoundError):
                ve.burn_subtitles(os.path.join(d, "none.mp4"), src, os.path.join(d, "o.mp4"), FFMPEG)

    def test_path_with_spaces_and_special_characters_works(self):
        with tempfile.TemporaryDirectory() as d:
            odd = os.path.join(d, "my clips (1) & more")
            os.makedirs(odd)
            src = self.make(odd)
            files = se.write_subtitle_files([{"start": 0, "end": 2, "text": "hello there friend", "words": []}], os.path.join(odd, "it's"), (270, 480))
            out = os.path.join(odd, "o.mp4")
            ve.burn_subtitles(src, files["ass"], out, FFMPEG)
            self.assertGreater(os.path.getsize(out), 1000)

    def test_burn_command_uses_the_ass_filter_and_copies_audio(self):
        cmd = ve.build_burn_command("ffmpeg", "in.mp4", "subs.ass", "out.mp4")
        joined = " ".join(cmd)
        self.assertIn("ass=", joined)
        self.assertIn("-c:a copy", joined)


if __name__ == "__main__":
    unittest.main()
