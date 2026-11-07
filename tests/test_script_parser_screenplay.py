"""Script2Video: screenplay parsing and shot manifests (backend/script_parser.py)."""
import unittest

from backend import script_parser as sp

SCRIPT = """TITLE: Rain

FADE IN:

INT. KITCHEN - NIGHT

MARA, 30s, stirs a pot. DANIEL enters, soaked.

DANIEL
(quietly)
You didn't answer my calls.

MARA
I was afraid. (beat) Of what you'd say.

CUT TO:

EXT. STREET - DAY

Daniel walks away.
"""


class ScreenplayParserTests(unittest.TestCase):
    def test_scenes_and_characters(self):
        p = sp.parse_script(SCRIPT)
        self.assertEqual(len(p["scenes"]), 2)
        self.assertEqual([s["int_ext"] for s in p["scenes"]], ["INT", "EXT"])
        self.assertEqual(p["scenes"][0]["time_of_day"], "NIGHT")
        self.assertEqual(sorted(p["characters"]), ["Daniel", "Mara"])

    def test_leading_transition_creates_no_phantom_scene(self):
        self.assertEqual(sp.parse_script(SCRIPT)["scenes"][0]["location"].upper(), "KITCHEN")

    def test_parenthetical_is_not_dialogue(self):
        shots = sp.script_to_shots(SCRIPT)["shots"]
        dan = next(s for s in shots if s.get("speaker") == "Daniel")
        self.assertNotIn("quietly", dan["line"])
        self.assertEqual(dan["emotion"], "quietly")
        mara = next(s for s in shots if s.get("speaker") == "Mara")
        self.assertNotIn("(beat)", mara["line"])

    def test_timing_is_gap_free_and_exact(self):
        m = sp.script_to_shots(SCRIPT)
        t = 0.0
        for i, s in enumerate(m["shots"]):
            self.assertEqual(s["idx"], i)
            self.assertAlmostEqual(s["start"], t, places=1)
            self.assertAlmostEqual(s["end"], s["start"] + s["duration"], places=1)
            self.assertTrue(1.2 <= s["duration"] <= 8.0 or s["kind"] == "establishing")
            t = s["end"]
        self.assertAlmostEqual(m["total_duration"], t, places=1)
        self.assertEqual([s["idx"] for s in m["shots"] if s["is_last"]], [len(m["shots"]) - 1])

    def test_longer_line_gets_more_time(self):
        a = sp.script_to_shots("INT. A - DAY\n\nBOB\nHi.\n")["shots"][-1]["duration"]
        b = sp.script_to_shots("INT. A - DAY\n\nBOB\n" + "word " * 14 + "end.\n")["shots"][-1]["duration"]
        self.assertGreater(b, a)

    def test_camera_fields_and_lighting(self):
        for s in sp.script_to_shots(SCRIPT)["shots"]:
            for key in ("shot_type", "camera_angle", "camera_movement", "camera_prompt", "visual_desc", "ff_desc", "lf_desc", "motion_desc"):
                self.assertTrue(s[key], key)
        night = sp.script_to_shots(SCRIPT)["shots"][0]
        self.assertEqual(night["lighting"], sp.LIGHTING["NIGHT"])

    def test_characters_tagged_in_visual_desc(self):
        s = next(s for s in sp.script_to_shots(SCRIPT)["shots"] if s.get("speaker") == "Daniel")
        self.assertIn("<Daniel>", s["visual_desc"])

    def test_srt(self):
        srt = sp.manifest_to_srt(sp.script_to_shots(SCRIPT))
        self.assertIn("-->", srt)
        self.assertIn("Daniel:", srt)

    def test_empty_and_garbage_input(self):
        with self.assertRaises(sp.ScriptParseError):
            sp.parse_script("   \n ")
        m = sp.script_to_shots("Just some prose without any headings at all.")
        self.assertTrue(m["shots"])  # still usable, with a warning
        self.assertTrue(m["warnings"])


if __name__ == "__main__":
    unittest.main()
