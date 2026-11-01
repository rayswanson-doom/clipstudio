import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import scorer  # noqa: E402


def make_words(sentences, wps=0.4, gap=0.1, start=0.0):
    """sentences: list of (text, pause_before_seconds). Returns word dicts."""
    out, t = [], start
    for text, pause in sentences:
        t += pause
        for tok in text.split():
            out.append({"word": tok, "start": round(t, 3), "end": round(t + wps, 3)})
            t += wps + gap
    return out


class SentenceTests(unittest.TestCase):
    def test_split_on_punctuation_and_pause(self):
        words = make_words([("Hello there.", 0), ("Is this working?", 0.2)])
        s = scorer.build_sentences(words)
        self.assertEqual([x["text"] for x in s], ["Hello there.", "Is this working?"])

    def test_split_on_long_pause_without_punctuation(self):
        words = make_words([("first part", 0), ("second part", 1.5)])
        s = scorer.build_sentences(words)
        self.assertEqual(len(s), 2)
        self.assertGreaterEqual(s[1]["pause_before"], 1.4)

    def test_empty(self):
        self.assertEqual(scorer.build_sentences([]), [])


class ScoreTests(unittest.TestCase):
    def sc(self, text, pause=0.0):
        return scorer.score_sentence(
            {"text": text, "word_count": len(text.split()), "pause_before": pause}
        )

    def test_question_beats_plain(self):
        self.assertGreater(self.sc("Did you know this secret?")["score"],
                           self.sc("The meeting is at noon today.")["score"])

    def test_reasons(self):
        r = self.sc("Wow, this insane hack saved you 50 percent!", pause=1.0)["reasons"]
        for tag in ("exclamation", "emotional_keywords", "number_or_stat",
                    "direct_address", "dramatic_pause"):
            self.assertIn(tag, r)

    def test_filler_penalty(self):
        self.assertLess(self.sc("um uh you know I mean sort of like okay")["score"], 0)

    def test_too_short_penalty(self):
        self.assertIn("too_short", self.sc("Okay.")["reasons"])


class ClipTests(unittest.TestCase):
    def build(self):
        boring = ("The report was filed on the usual schedule this week.", 0.2)
        hot = [
            ("Stop, this is the biggest mistake you will ever make!", 1.2),
            ("Did you know nobody tells you this secret?", 0.3),
            ("It cost me 5 million dollars and it was insane!", 0.3),
        ]
        sents = [boring] * 8 + hot + [boring] * 8
        return make_words(sents, wps=0.5, gap=0.1)

    def test_top_clip_contains_hot_section(self):
        clips = scorer.find_viral_clips(self.build(), top_n=1, min_len=10, max_len=30)
        self.assertEqual(len(clips), 1)
        self.assertIn("secret", clips[0]["text"])
        self.assertIn("question", clips[0]["reasons"])

    def test_non_overlapping_and_sorted_and_bounded(self):
        clips = scorer.find_viral_clips(self.build(), top_n=3, min_len=10, max_len=30)
        for c in clips:
            self.assertGreaterEqual(c["duration"], 10 - 1e-6)
            self.assertLessEqual(c["duration"], 30 + 1e-6)
        for a, b in zip(clips, clips[1:]):
            self.assertGreaterEqual(a["score"], b["score"])
        for i, a in enumerate(clips):
            for b in clips[i + 1:]:
                self.assertTrue(a["end"] <= b["start"] or b["end"] <= a["start"])

    def test_short_transcript_fallback(self):
        clips = scorer.find_viral_clips(make_words([("Why is this short?", 0)]),
                                        min_len=15, max_len=60)
        self.assertEqual(len(clips), 1)
        self.assertIn("question", clips[0]["reasons"])

    def test_empty_and_bad_args(self):
        self.assertEqual(scorer.find_viral_clips([]), [])
        self.assertEqual(scorer.find_viral_clips(self.build(), top_n=0), [])
        with self.assertRaises(ValueError):
            scorer.find_viral_clips(self.build(), min_len=50, max_len=10)


class TemplateTests(unittest.TestCase):
    def test_vars_default_and_if(self):
        t = "Hi {{name}}{{#if vip}} (VIP){{/if}}. Mood: {{mood|calm}}"
        self.assertEqual(scorer.render_template(t, {"name": "Sam", "vip": True}),
                         "Hi Sam (VIP). Mood: calm")
        self.assertEqual(scorer.render_template(t, {"name": "Sam"}),
                         "Hi Sam. Mood: calm")

    def test_list_joins(self):
        self.assertEqual(scorer.render_template("{{x}}", {"x": ["a", "b"]}), "a, b")

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            scorer.render_template("{{a}} {{b}}", {"a": 1})


class PromptTests(unittest.TestCase):
    def test_parse_full(self):
        p = scorer.parse_script_prompt(
            "Write a funny 45 second TikTok script about saving money on groceries for students")
        self.assertEqual(p["topic"], "saving money on groceries")
        self.assertEqual(p["tone"], "funny")
        self.assertEqual(p["duration_seconds"], 45)
        self.assertEqual(p["platform"], "TikTok")

    def test_minutes(self):
        self.assertEqual(scorer.parse_script_prompt("script about cats, 2 minutes")["duration_seconds"], 120)

    def test_defaults(self):
        p = scorer.parse_script_prompt("Create a script about sleep")
        self.assertEqual((p["tone"], p["duration_seconds"], p["platform"]), ("engaging", 30, None))
        self.assertEqual(p["topic"], "sleep")

    def test_points_bullets_and_inline(self):
        p = scorer.parse_script_prompt("Script about focus\n- turn off alerts\n- use a timer")
        self.assertEqual(p["points"], ["turn off alerts", "use a timer"])
        q = scorer.parse_script_prompt("Script about focus. Key points: turn off alerts; use a timer")
        self.assertEqual(q["points"], ["turn off alerts", "use a timer"])

    def test_fallback_topic_and_empty(self):
        self.assertEqual(scorer.parse_script_prompt("Write a script gardening tips")["topic"], "gardening tips")
        with self.assertRaises(ValueError):
            scorer.parse_script_prompt("   ")


class ScriptTests(unittest.TestCase):
    def test_structure_and_budget(self):
        r = scorer.generate_script("Make a dramatic 20 second Shorts script about volcanoes")
        self.assertIn("volcanoes", r["hook"])
        self.assertTrue(r["cta"].startswith("Subscribe"))
        self.assertTrue(r["body"])
        self.assertEqual(r["script"], " ".join([r["hook"], *r["body"], r["cta"]]))
        self.assertLessEqual(r["estimated_seconds"], 20 * 1.25)

    def test_uses_points(self):
        r = scorer.generate_script("Script about focus\n- turn off alerts\n- use a timer")
        self.assertEqual(r["body"], ["turn off alerts.", "use a timer."])

    def test_custom_template(self):
        r = scorer.generate_script("Script about tea in 10 seconds",
                                   template="{{topic}} / {{duration}}s / {{who|you}}")
        self.assertEqual(r["script"], "tea / 10s / you")


if __name__ == "__main__":
    unittest.main()
