"""Novel2Video: chaptering, beats, characters, episodic scripts (backend/novel_adapter.py)."""
import os
import unittest

from backend import novel_adapter as na
from backend import script_parser as sp

NOVEL = os.path.join(os.path.dirname(__file__), "data", "sample_novel.txt")


class NovelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = na.read_text_file(NOVEL)
        cls.res = na.adapt_novel(cls.text, episodes=2, seconds_per_episode=40, embedder="hashed")

    def test_hashing_embedder_is_deterministic_and_normalised(self):
        e = na.HashingEmbedder()
        v = e.encode(["the quick brown fox", "the quick brown fox", "an unrelated sentence about trains"])
        self.assertEqual(v.shape[1], 384)
        self.assertTrue((v[0] == v[1]).all())
        self.assertAlmostEqual(float((v[0] ** 2).sum()), 1.0, places=4)
        self.assertGreater(float(v[0] @ v[1]), float(v[0] @ v[2]))

    def test_get_embedder_never_requires_torch(self):
        w = []
        e = na.get_embedder("hashed", w)
        self.assertIsInstance(e, na.HashingEmbedder)

    def test_chapters_and_characters(self):
        self.assertEqual(len(self.res["chapters"]), 2)
        names = [c["name"] for c in self.res["characters"]]
        self.assertTrue({"Elena", "Tomas"} <= {n.split()[0] for n in names})
        self.assertNotIn("Kitchen", names)

    def test_quote_aware_sentence_split(self):
        s = na.split_sentences('He said, "Wait. Please." Then he left. It rained.')
        self.assertEqual(len(s), 3)

    def test_episodes_are_valid_scripts_within_budget(self):
        self.assertEqual(len(self.res["episodes"]), 2)
        for ep in self.res["episodes"]:
            m = sp.script_to_shots(ep["script"])
            self.assertTrue(m["shots"])
            self.assertLess(m["total_duration"], 40 * 2.2)

    def test_compression_is_shorter_than_source(self):
        words = sum(len(ep["script"].split()) for ep in self.res["episodes"])
        self.assertLess(words, len(self.text.split()))

    def test_progress_callback_and_bad_input(self):
        seen = []
        na.adapt_novel(self.text, episodes=1, embedder="hashed", progress=seen.append)
        self.assertTrue(seen and seen[-1] >= 0.99)
        with self.assertRaises(na.NovelError):
            na.adapt_novel("   ")

    def test_missing_minilm_falls_back_with_a_warning(self):
        w = []
        e = na.get_embedder("minilm", w)
        self.assertTrue(hasattr(e, "encode"))
        # sentence-transformers is optional: either it loaded or we were told why not
        self.assertTrue(w or type(e).__name__ != "HashingEmbedder")


if __name__ == "__main__":
    unittest.main()
