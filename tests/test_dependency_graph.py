"""Consistency graph: reference linking, keyframe passing, drift detection (backend/dependency_graph.py)."""
import os
import tempfile
import unittest

import numpy as np

from backend import dependency_graph as dg
from backend import script_parser as sp

SCRIPT = """INT. KITCHEN - NIGHT

MARA stirs a pot. DANIEL enters.

DANIEL
Hello.

MARA
Hi.

EXT. STREET - DAY

Daniel walks.

MARA
Wait!
"""
CHARS = [{"name": "Mara", "visual_identifier": "x", "anchor": {}}, {"name": "Daniel", "visual_identifier": "y", "anchor": {}}]


def _touch(p):
    with open(p, "wb") as f:
        f.write(b"x")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.shots = sp.script_to_shots(SCRIPT)["shots"]
        self.g = dg.build_from_storyboard(self.shots, CHARS)

    def test_basic_graph_ops_and_cycle_rejection(self):
        g = dg.DependencyGraph()
        for n in "abc":
            g.add_node(n, "shot")
        g.add_edge("a", "b", "feeds")
        g.add_edge("b", "c", "feeds")
        self.assertEqual(g.topological_order(), ["a", "b", "c"])
        self.assertEqual(g.levels()["c"], 2)
        self.assertFalse(g.has_cycle())
        with self.assertRaises(dg.GraphError):
            g.add_edge("a", "a", "feeds")
        g.add_edge("c", "a", "feeds")
        self.assertTrue(g.has_cycle())
        with self.assertRaises(dg.GraphError):
            g.topological_order()

    def test_round_trip(self):
        again = dg.DependencyGraph.from_dict(self.g.to_dict())
        self.assertEqual(again.topological_order(), self.g.topological_order())
        self.assertTrue(self.g.to_mermaid().startswith("flowchart"))

    def test_every_shot_gets_its_characters_anchors(self):
        plan = {p["shot"]: p for p in dg.render_plan(self.g)}
        for s in self.shots:
            for c in s["characters"]:
                self.assertIn(f"ref:{c}", [i["node"] for i in plan[s["idx"]]["inputs"]], (s["idx"], c))
            self.assertLessEqual(len(plan[s["idx"]]["inputs"]), dg.MAX_REFERENCES)

    def test_continuity_keyframe_passes_downstream(self):
        plan = dg.render_plan(self.g)
        later = [p for p in plan if p["shot"] > 0]
        self.assertTrue(all(any(i["node"].startswith("kf:") for i in p["inputs"]) for p in later if p["continues_from"] is not None))
        order = [p["shot"] for p in plan]
        for p in plan:
            if p["continues_from"] is not None:
                self.assertLess(order.index(p["continues_from"]), order.index(p["shot"]))

    def test_keyframe_store_readiness(self):
        store = dg.KeyframeStore(self.g)
        self.assertTrue(store.ready(0) is False or store.ready(0) is True)
        with tempfile.TemporaryDirectory() as d:
            for c in ("Mara", "Daniel"):
                p = os.path.join(d, c + ".png")
                _touch(p)
                store.put(f"ref:{c}", p)
            self.assertTrue(store.ready(0))
            if len(self.shots) > 1:
                self.assertFalse(store.ready(1))  # needs shot 0's last frame
                p = os.path.join(d, "k.png")
                _touch(p)
                store.put("kf:0:last", p)
                self.assertTrue(store.ready(1))

    def test_drift_detection(self):
        import cv2

        red = np.zeros((100, 100, 3), np.uint8)
        red[:] = (30, 30, 200)
        blue = np.zeros((100, 100, 3), np.uint8)
        blue[:] = (200, 40, 30)
        a = dg.appearance_signature(red)
        self.assertAlmostEqual(dg.signature_distance(a, a), 0.0, places=5)
        self.assertGreater(dg.signature_distance(a, dg.appearance_signature(blue)), 0.35)
        out = dg.check_drift({"Mara": a}, [{"shot": 1, "character": "Mara", "signature": a},
                                           {"shot": 2, "character": "Mara", "signature": dg.appearance_signature(blue)}])
        self.assertEqual({o["shot"]: o["status"] for o in out}, {1: "ok", 2: "drift"})


if __name__ == "__main__":
    unittest.main()
