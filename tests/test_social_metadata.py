"""social_metadata: YouTube Shorts / Instagram Reels / Facebook Reels packages - offline NLP + stubbed free-tier LLM."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
import social_metadata as sm  # noqa: E402

FITNESS = ("You need to eat more protein every single day. Most people train hard in the gym but ignore sleep. "
           "Protein and sleep build muscle faster than any workout. So eat your protein, sleep eight hours and lift heavy.")
TECH = ("Did you know that AI can now write code in seconds? Developers are using ChatGPT to build an app in one afternoon. "
        "This software startup shipped a full product with two engineers.")
KEYLESS = {k: "" for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}


def check_limits(tc, pk):
    yt, ig, fb = pk["youtube"], pk["instagram"], pk["facebook"]
    tc.assertTrue(0 < len(yt["title"]) < 60, yt["title"])
    tc.assertTrue(10 <= len(yt["tags"]) <= 15, yt["tags"])
    tc.assertIn("Shorts", yt["tags"])
    tc.assertIn("#Shorts", yt["description"])
    tc.assertLessEqual(sum(len(t) for t in yt["tags"]) + len(yt["tags"]), sm.YT_TAG_CHARS + 20)
    tc.assertTrue(15 <= len(ig["hashtags"]) <= 20, ig["hashtags"])
    tc.assertTrue(5 <= len(fb["hashtags"]) <= 8, fb["hashtags"])
    for tag in ig["hashtags"] + fb["hashtags"]:
        tc.assertRegex(tag, r"^#[0-9A-Za-z_]+$")
    tc.assertEqual(len({t.lower() for t in ig["hashtags"]}), len(ig["hashtags"]))
    for text in (yt["copy_text"], ig["copy_text"], fb["copy_text"]):
        tc.assertTrue(text.strip())


@mock.patch.dict(os.environ, KEYLESS)
class OfflineTests(unittest.TestCase):
    def test_platform_limits_hold_for_different_topics(self):
        for text in (FITNESS, TECH, "Hello.", "", "lol " * 400):
            pk = sm.generate_packages(text, use_llm="off")
            check_limits(self, pk)
            self.assertEqual(pk["source"], "offline")

    def test_youtube_title_is_a_real_headline_under_60_chars(self):
        title = sm.generate_packages(FITNESS, use_llm="off")["youtube"]["title"]
        self.assertLess(len(title), 60)
        self.assertFalse(title.endswith(("...", "…", " and", " the", " to")))
        self.assertIn("protein", title.lower())

    def test_description_opens_with_the_hook_and_has_a_cta(self):
        pk = sm.generate_packages(FITNESS, use_llm="off")
        self.assertTrue(pk["youtube"]["description"].startswith(pk["hook"]))
        self.assertIn("Subscribe", pk["youtube"]["description"])
        self.assertTrue(pk["instagram"]["caption"].startswith(pk["instagram"]["hook"]))
        self.assertIn(pk["instagram"]["cta"], pk["instagram"]["caption"])
        self.assertIn("Save", pk["instagram"]["cta"])

    def test_facebook_caption_is_conversational_and_ends_with_a_prompt(self):
        cap = sm.generate_packages(FITNESS, use_llm="off")["facebook"]["caption"]
        self.assertTrue(len(cap) > 60)
        self.assertIn("?", cap.splitlines()[-1] if cap.strip() else "")

    def test_keywords_are_content_words_not_filler(self):
        kws = sm.extract_keywords(FITNESS)
        self.assertIn("protein", kws[:3])
        for junk in ("the", "you", "every", "enough", "heres", "more"):
            self.assertNotIn(junk, kws)
        self.assertTrue(all("'" not in k for k in kws))

    def test_hashtags_are_clean_and_niche_aware(self):
        fit = sm.generate_packages(FITNESS, use_llm="off")["instagram"]["hashtags"]
        tech = sm.generate_packages(TECH, use_llm="off")["instagram"]["hashtags"]
        self.assertTrue({"#fitness", "#workout", "#gym"} & set(fit), fit)
        self.assertTrue({"#tech", "#ai", "#coding", "#technology"} & set(tech), tech)
        self.assertFalse({"#enough", "#heres", "#the", "#every"} & set(fit))

    def test_topic_keywords_steer_the_tags(self):
        pk = sm.generate_packages(FITNESS, ["meal prep", "bulking"], use_llm="off")
        self.assertIn("meal prep", [t.lower() for t in pk["youtube"]["tags"]])
        self.assertIn("#mealprep", [t.lower() for t in pk["instagram"]["hashtags"]])
        self.assertIn("#bulking", [t.lower() for t in pk["facebook"]["hashtags"]])

    def test_empty_text_still_yields_usable_generic_packages(self):
        pk = sm.generate_packages("", use_llm="off")
        check_limits(self, pk)

    def test_deterministic(self):
        self.assertEqual(sm.generate_packages(FITNESS, use_llm="off"), sm.generate_packages(FITNESS, use_llm="off"))

    def test_auto_without_a_key_stays_offline_and_never_touches_the_network(self):
        with mock.patch("socket.socket", side_effect=AssertionError("network used")):
            pk = sm.generate_packages(FITNESS, use_llm="auto")
        self.assertEqual(pk["source"], "offline")
        self.assertEqual(pk["warnings"], [])


def good_reply():
    return {
        "youtube": {"title": "Eat More Protein Every Day", "description": "Protein is the lever nobody pulls.\nSubscribe!",
                    "tags": ["protein", "muscle", "gym", "fitness", "sleep", "lifting", "nutrition", "health", "workout", "gains", "diet", "bulk"]},
        "instagram": {"hook": "Your gym routine is missing ONE thing.", "caption": "Protein + sleep beats any program.", "cta": "Save this for later.",
                      "hashtags": [f"#tag{i}" for i in range(18)]},
        "facebook": {"caption": "I used to skip sleep. Big mistake. Do you?", "hashtags": [f"#fb{i}" for i in range(6)]},
    }


def groq_transport(reply):
    def send(url, headers, body):
        send.calls.append((url, headers, body))
        return {"choices": [{"message": {"content": json.dumps(reply)}}]}
    send.calls = []
    return send


class LLMTests(unittest.TestCase):
    def test_groq_wording_is_used_when_a_key_is_set(self):
        send = groq_transport(good_reply())
        with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "gsk_test"}):
            pk = sm.generate_packages(FITNESS, use_llm="auto", transport=send)
        self.assertEqual(pk["source"], "groq")
        self.assertEqual(pk["youtube"]["title"], "Eat More Protein Every Day")
        self.assertEqual(pk["instagram"]["hook"], "Your gym routine is missing ONE thing.")
        check_limits(self, pk)
        url, headers, body = send.calls[0]
        self.assertIn("groq.com", url)
        self.assertEqual(headers["Authorization"], "Bearer gsk_test")
        self.assertIn("protein", body["messages"][0]["content"].lower())

    def test_gemini_path(self):
        reply = {"candidates": [{"content": {"parts": [{"text": json.dumps(good_reply())}]}}]}
        seen = []
        with mock.patch.dict(os.environ, {**KEYLESS, "GEMINI_API_KEY": "g-test"}):
            pk = sm.generate_packages(FITNESS, use_llm="auto", transport=lambda u, h, b: (seen.append((u, h)), reply)[1])
        self.assertEqual(pk["source"], "gemini")
        self.assertIn("generativelanguage.googleapis.com", seen[0][0])
        self.assertEqual(seen[0][1]["x-goog-api-key"], "g-test")  # the key goes in a header, never in the URL
        self.assertNotIn("g-test", seen[0][0])

    def test_llm_cannot_break_platform_limits(self):
        bad = good_reply()
        bad["youtube"]["title"] = "X" * 90
        bad["youtube"]["tags"] = [f"tag{i}" for i in range(40)]
        bad["instagram"]["hashtags"] = [f"#h{i}" for i in range(40)]
        bad["facebook"]["hashtags"] = [f"#f{i}" for i in range(30)]
        with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "k"}):
            pk = sm.generate_packages(FITNESS, use_llm="groq", transport=groq_transport(bad))
        check_limits(self, pk)

    def test_partial_reply_keeps_offline_values_for_the_rest(self):
        with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "k"}):
            pk = sm.generate_packages(FITNESS, use_llm="groq", transport=groq_transport({"facebook": {"caption": "Only this."}}))
        check_limits(self, pk)
        self.assertTrue(pk["facebook"]["caption"].startswith("Only this."))

    def test_any_llm_failure_falls_back_offline_with_a_warning(self):
        def boom(url, headers, body):
            raise OSError("quota exceeded")
        for transport in (boom, lambda u, h, b: {"choices": [{"message": {"content": "not json"}}]}, lambda u, h, b: {}):
            with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "k"}):
                pk = sm.generate_packages(FITNESS, use_llm="auto", transport=transport)
            self.assertEqual(pk["source"], "offline")
            self.assertTrue(any("offline captions were used" in w for w in pk["warnings"]))
            check_limits(self, pk)

    def test_off_never_calls_the_llm_even_with_a_key(self):
        send = groq_transport(good_reply())
        with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "k"}):
            pk = sm.generate_packages(FITNESS, use_llm="off", transport=send)
        self.assertEqual(send.calls, [])
        self.assertEqual(pk["source"], "offline")

    def test_provider_detection(self):
        with mock.patch.dict(os.environ, KEYLESS):
            self.assertIsNone(sm.llm_provider())
        with mock.patch.dict(os.environ, {**KEYLESS, "GEMINI_API_KEY": "x"}):
            self.assertEqual(sm.llm_provider(), "gemini")
        with mock.patch.dict(os.environ, {**KEYLESS, "GROQ_API_KEY": "x", "GEMINI_API_KEY": "y"}):
            self.assertEqual(sm.llm_provider(), "groq")

    def test_json_extraction_tolerates_code_fences(self):
        self.assertEqual(sm._extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            sm._extract_json("no braces")


if __name__ == "__main__":
    unittest.main()
