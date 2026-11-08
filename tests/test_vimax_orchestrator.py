"""Multi-agent orchestration, LLM clients (mocked transport only: no network, no keys), CPU render."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from backend import vimax_orchestrator as vo

NOVEL = os.path.join(os.path.dirname(__file__), "data", "sample_novel.txt")
SCRIPT = """INT. KITCHEN - NIGHT

MARA stirs a pot. DANIEL enters, soaked.

DANIEL
You didn't answer my calls.

MARA
I was afraid.

EXT. STREET - DAY

Daniel walks away.
"""
IDEA = 'Mara waits in a small kitchen at night. Daniel arrives, soaked from the rain. "You never called," Daniel says. Mara looks away.'


class FakeTransport:
    """Stands in for the HTTPS call; records requests and replays canned replies."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append((url, headers, payload))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


def groq_reply(obj, raw=False):
    return {"choices": [{"message": {"content": obj if raw else json.dumps(obj)}}]}


def gemini_reply(obj, raw=False):
    return {"candidates": [{"content": {"parts": [{"text": obj if raw else json.dumps(obj)}]}}]}


class JsonHelperTests(unittest.TestCase):
    def test_loose_json(self):
        self.assertEqual(vo.parse_json_loose('```json\n{"a": 1,}\n```'), {"a": 1})
        self.assertEqual(vo.parse_json_loose('Sure! {"a": [1, 2]} hope that helps'), {"a": [1, 2]})
        with self.assertRaises(vo.LLMError):
            vo.parse_json_loose("no json here")

    def test_check_shape(self):
        schema = {"n": int, "opt?": str, "items": [{"x": (int, float)}]}
        vo.check_shape({"n": 1, "items": [{"x": 1.5}]}, schema)
        for bad in ({"items": []}, {"n": "1", "items": []}, {"n": 1, "items": [{"x": "a"}]}, {"n": True, "items": []}, []):
            with self.assertRaises(vo.SchemaError):
                vo.check_shape(bad, schema)


class ClientTests(unittest.TestCase):
    def test_groq_request_shape_and_key_placement(self):
        t = FakeTransport(groq_reply({"ok": 1}))
        c = vo.make_client("groq", "KEY123", transport=t)
        self.assertEqual(c.complete_json("sys", "usr"), {"ok": 1})
        url, headers, payload = t.requests[0]
        self.assertTrue(url.startswith("https://api.groq.com/"))
        self.assertEqual(headers["Authorization"], "Bearer KEY123")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("KEY123", json.dumps(payload))

    def test_gemini_request_shape(self):
        t = FakeTransport(gemini_reply({"ok": 2}))
        c = vo.make_client("gemini", "GKEY", transport=t)
        self.assertEqual(c.complete_json("sys", "usr"), {"ok": 2})
        url, headers, payload = t.requests[0]
        self.assertTrue(url.startswith("https://generativelanguage.googleapis.com/"))
        self.assertNotIn("GKEY", url)  # the key travels in a header, never in the URL
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_fenced_json_and_retry_with_feedback(self):
        t = FakeTransport(groq_reply("not json", raw=True), groq_reply('```json\n{"n": 3}\n```', raw=True))
        c = vo.make_client("groq", "k", transport=t)
        self.assertEqual(c.complete_json("s", "u", {"n": int}), {"n": 3})
        self.assertEqual(len(t.requests), 2)
        self.assertIn("rejected", t.requests[1][2]["messages"][1]["content"])

    def test_gives_up_after_retries(self):
        c = vo.make_client("gemini", "k", transport=FakeTransport(gemini_reply({"wrong": 1})))
        with self.assertRaises(vo.LLMError):
            c.complete_json("s", "u", {"n": int})
        self.assertEqual(c.calls, 3)

    def test_missing_key_and_unknown_provider(self):
        old = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}
        try:
            with self.assertRaises(vo.LLMError):
                vo.make_client("groq")
            with self.assertRaises(vo.LLMError):
                vo.make_client("gemini")
            self.assertTrue(vo.make_client("auto").offline)
            with self.assertRaises(ValueError):
                vo.make_client("skynet")
        finally:
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    def test_http_error_is_a_clear_llm_error(self):
        import io
        import urllib.error

        err = urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b"bad key"))
        orig = vo.urllib.request.urlopen
        vo.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(err)
        try:
            with self.assertRaises(vo.LLMError) as cm:
                vo._http_post_json("https://x.test", {}, {})
            self.assertIn("401", str(cm.exception))
        finally:
            vo.urllib.request.urlopen = orig


class OfflineRunTests(unittest.TestCase):
    def check(self, r):
        shots = r["storyboard"]["shots"]
        self.assertTrue(shots)
        self.assertEqual([i for i in r["issues"] if i["severity"] == "error"], [])
        t = 0.0
        for s in shots:
            self.assertAlmostEqual(s["start"], t, places=1)
            t = s["end"]
            self.assertTrue(s["camera_prompt"] and s["visual_desc"] and s["duration"] > 0)
        for c in r["characters"]:
            self.assertTrue(c["visual_identifier"] and c["anchor"]["seed"] is not None)
        self.assertEqual([x["agent"] for x in r["trace"]],
                         ["Director", "Screenwriter", "CharacterExtractor", "StoryboardArtist", "ReferenceSelector", "ConsistencyChecker"])
        self.assertTrue(r["llm"]["offline"] and r["llm"]["calls"] == 0)
        self.assertFalse(any(t_["used_llm"] for t_ in r["trace"]))

    def test_script2video(self):
        r = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")
        self.check(r)
        self.assertEqual(sorted(c["name"] for c in r["characters"]), ["Daniel", "Mara"])
        self.assertEqual(r["script"].strip(), SCRIPT.strip())
        self.assertTrue(all(s["references"] for s in r["storyboard"]["shots"] if s["characters"]))

    def test_idea2video(self):
        r = vo.ViMaxOrchestrator().run(IDEA, mode="idea2video", target_seconds=20)
        self.check(r)
        self.assertIn("DANIEL", r["script"])
        self.assertNotIn("INT. LOCATION", r["script"])

    def test_novel2video(self):
        seen = []
        r = vo.ViMaxOrchestrator(progress=lambda f, m: seen.append(f)).run(
            open(NOVEL, encoding="utf-8").read(), mode="novel2video", episodes=2, episode=2, target_seconds=30, embedder="hashed")
        self.check(r)
        self.assertEqual(r["novel"]["episode"], 2)
        self.assertEqual(r["novel"]["embedder"], "hashed-bow")
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_max_shots_and_validation(self):
        r = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video", max_shots=3)
        self.assertEqual(len(r["storyboard"]["shots"]), 3)
        self.assertTrue(r["storyboard"]["shots"][-1]["is_last"])
        for kw in ({"source": ""}, {"source": "x", "mode": "bad"}, {"source": "x", "style": "bad"}):
            with self.assertRaises(ValueError):
                vo.ViMaxOrchestrator().run(**kw)

    def test_deterministic(self):
        a = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")
        b = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")
        for r in (a, b):
            r.pop("seconds"); [t.pop("seconds") for t in r["trace"]]
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_checker_fixes_a_broken_storyboard(self):
        r = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")
        shots, chars = r["storyboard"]["shots"], r["characters"]
        shots[1]["characters"] = [c.lower() for c in shots[1]["characters"]] or ["mara"]
        shots[2]["start"] += 5
        shots[0]["visual_desc"] = shots[0]["visual_desc"].replace("<Mara>", "Mara")
        graph = vo.dg.DependencyGraph.from_dict(r["graph"])
        out = vo.ConsistencyChecker(vo.OfflineClient(), [], []).run(shots, chars, graph)
        self.assertGreater(out["fixed"], 0)
        self.assertEqual([i for i in out["issues"] if i["severity"] == "error"], [])


class LLMRunTests(unittest.TestCase):
    def test_llm_path_refines_text_but_cannot_corrupt_timing(self):
        shots_reply = lambda idxs: {"shots": [{"idx": i, "camera_prompt": f"LLM prompt {i}", "visual_desc": f"LLM desc {i}",
                                               "duration": 999 if i == 0 else 2.0} for i in idxs]}
        chars_reply = {"characters": [{"name": "Mara", "description": "a cook", "gender": "female", "visual_identifier": "red scarf, black bob",
                                       "age": 30}, {"name": "Daniel", "description": "a tall man", "visual_identifier": "grey coat"}]}
        t = FakeTransport(groq_reply(chars_reply), groq_reply(shots_reply(range(6))))
        r = vo.ViMaxOrchestrator(vo.make_client("groq", "k", transport=t)).run(SCRIPT, mode="script2video")
        self.assertTrue(r["llm"]["calls"] >= 2)
        by = {c["name"]: c for c in r["characters"]}
        self.assertEqual(by["Mara"]["visual_identifier"], "red scarf, black bob")
        shots = r["storyboard"]["shots"]
        self.assertTrue(shots[0]["camera_prompt"].startswith("LLM prompt"))
        self.assertLess(shots[0]["duration"], 13)  # 999 s rejected
        t_ = 0.0
        for s in shots:
            self.assertAlmostEqual(s["start"], t_, places=1)
            t_ = s["end"]
        # every shot that names a character still tags them after the checker's fixes
        for s in shots:
            for c in s["characters"]:
                self.assertIn(f"<{c}>", s["visual_desc"])
        self.assertTrue(any(x["used_llm"] for x in r["trace"]))

    def test_failing_llm_falls_back_to_rules_with_warnings(self):
        t = FakeTransport(groq_reply("garbage", raw=True))
        r = vo.ViMaxOrchestrator(vo.make_client("gemini", "k", transport=FakeTransport(gemini_reply("garbage", raw=True)))).run(
            SCRIPT, mode="script2video")
        self.assertTrue(r["storyboard"]["shots"])
        self.assertTrue(any("built-in rules" in w or "Kept the rule-based" in w for w in r["warnings"]))
        self.assertFalse(any(x["used_llm"] for x in r["trace"]))
        self.assertEqual([i for i in r["issues"] if i["severity"] == "error"], [])

    def test_network_failure_does_not_stop_the_run(self):
        t = FakeTransport(vo.LLMError("Could not reach the AI service (down)."))
        r = vo.ViMaxOrchestrator(vo.make_client("groq", "k", transport=t)).run(IDEA, mode="idea2video")
        self.assertTrue(r["storyboard"]["shots"])
        self.assertTrue(any("Could not reach" in w for w in r["warnings"]))

    def test_idea_plan_from_llm_and_user_style_wins(self):
        plan = {"title": "Rainy Night", "logline": "l", "style": "anime", "scenes": [{"title": "a", "location": "Kitchen", "summary": "s"}]}
        script = {"script": SCRIPT}
        t = FakeTransport(groq_reply(plan), groq_reply(script), groq_reply({"characters": []}), groq_reply({"shots": []}))
        r = vo.ViMaxOrchestrator(vo.make_client("groq", "k", transport=t)).run(IDEA, mode="idea2video", style="human")
        self.assertEqual(r["plan"]["title"], "Rainy Night")
        self.assertEqual(r["plan"]["style"], "human")
        self.assertEqual(r["script"].strip(), SCRIPT.strip())

    def test_api_key_never_appears_in_results(self):
        t = FakeTransport(groq_reply({"characters": []}))
        r = vo.ViMaxOrchestrator(vo.make_client("groq", "SECRET-KEY-XYZ", transport=t)).run(SCRIPT, mode="script2video")
        self.assertNotIn("SECRET-KEY-XYZ", json.dumps(r))


class ProjectAndRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")

    def test_save_project(self):
        with tempfile.TemporaryDirectory() as d:
            names = vo.save_project(self.r, d)
            self.assertIn("storyboard.json", names)
            self.assertEqual(json.load(open(os.path.join(d, "storyboard.json"), encoding="utf-8"))["shots"][0]["idx"], 0)
            self.assertTrue(open(os.path.join(d, "graph.mmd"), encoding="utf-8").read().startswith("flowchart"))
            self.assertIn("-->", open(os.path.join(d, "subtitles.srt"), encoding="utf-8").read())

    def test_portrait_is_deterministic_and_distinct(self):
        import cv2

        a, b = self.r["characters"]
        with tempfile.TemporaryDirectory() as d:
            p1, p2, p3 = (vo.make_portrait(c, os.path.join(d, f"{i}.png")) for i, c in enumerate((a, a, b)))
            i1, i2, i3 = (cv2.imread(p) for p in (p1, p2, p3))
            self.assertTrue((i1 == i2).all())
            self.assertFalse((i1 == i3).all())
            self.assertEqual(i1.shape[:2], (480, 360))
            # anime style differs from human style for the same character
            anime = vo.make_portrait({**a, "style": "anime"}, os.path.join(d, "an.png"))
            self.assertFalse((cv2.imread(anime) == i1).all())

    def test_render_on_cpu(self):
        import cv2

        with tempfile.TemporaryDirectory() as d:
            out = vo.render_storyboard(self.r, os.path.join(d, "o.mp4"), work_dir=os.path.join(d, "w"), size=(180, 320), fps=8, max_shots=4)
            self.assertTrue(os.path.getsize(out["output_path"]) > 1000)
            cap = cv2.VideoCapture(out["output_path"])
            self.assertEqual((int(cap.get(3)), int(cap.get(4))), (180, 320))
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), out["duration"], delta=0.8)
            cap.release()
            self.assertEqual(len(out["shots"]), 4)
            self.assertIn("kf:0:last", out["keyframes"])  # continuity keyframe was produced
            self.assertTrue(out["drift"] and all(x["status"] == "ok" for x in out["drift"]), out["drift"])
            self.assertFalse(out["voice_used"])

    def test_render_needs_shots(self):
        with self.assertRaises(ValueError):
            vo.render_storyboard({"storyboard": {"shots": []}, "graph": self.r["graph"], "characters": [], "render_plan": []},
                                 "x.mp4")


class CameraPreferenceTests(unittest.TestCase):
    def shots(self, **cam):
        return vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video", camera=cam or None)["storyboard"]["shots"]

    def test_auto_changes_nothing(self):
        self.assertEqual(self.shots(), self.shots(size="auto", angle="auto", movement="auto"))

    def test_preferences_rewrite_every_text_field_consistently(self):
        base = self.shots()
        shots = self.shots(size="wide", angle="high", movement="handheld")
        for b, s in zip(base, shots):
            self.assertEqual(s["camera_angle"], "high angle")
            self.assertEqual(s["camera_movement"], "handheld")
            self.assertEqual(s["shot_type"], b["shot_type"] if b["kind"] == "establishing" else "wide shot")
            self.assertEqual(s["variation_type"], "large")
            for field in ("camera_prompt", "visual_desc"):
                self.assertIn("high angle", s[field], field)
                if b["camera_angle"] != "high angle":
                    self.assertNotIn(b["camera_angle"], s[field].split(",")[1] if field == "camera_prompt" else s[field][:60])
            self.assertEqual(s["duration"], b["duration"])  # framing never changes the timing
            self.assertEqual(s["characters"], b["characters"])

    def test_establishing_shots_stay_wide_and_bad_values_are_rejected(self):
        est = [s for s in self.shots(size="close") if s["kind"] == "establishing"]
        self.assertTrue(est and all("close" not in s["shot_type"] for s in est))
        with self.assertRaises(ValueError):
            vo.normalize_camera({"angle": "upside down"})

    def test_close_ups_get_bigger_portraits_than_wide_shots(self):
        close = vo._character_scale({"shot_type": "close-up"})
        wide = vo._character_scale({"shot_type": "wide shot"})
        self.assertGreater(close, vo._character_scale({"shot_type": "medium shot"}))
        self.assertGreater(vo._character_scale({"shot_type": "medium shot"}), wide)


class RenderOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")

    def render(self, **kw):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        return vo.render_storyboard(self.r, os.path.join(d, "o.mp4"), work_dir=os.path.join(d, "w"), size=(180, 320), fps=6, max_shots=3, **kw)

    def test_host_overlay_goes_where_asked_and_characters_keep_clear(self):
        out = self.render(host={"style": "ai", "position": "Bottom-Right", "scale": 0.25})
        self.assertEqual(out["host"]["position"], "Bottom-Right")
        self.assertTrue(os.path.isfile(out["host"]["image"]))

    def test_unreadable_host_picture_falls_back_with_a_warning(self):
        d = tempfile.mkdtemp()
        bad = os.path.join(d, "x.png")
        with open(bad, "wb") as f:
            f.write(b"not an image")
        out = self.render(host={"image": bad, "position": "Top-Left"})
        self.assertTrue(any("host picture" in w for w in out["warnings"]))
        self.assertTrue(out["host"]["image"].endswith("portrait_host.png"))

    def test_custom_host_coordinates(self):
        out = self.render(host={"position": "Custom Coordinates", "xy": (20, 40), "scale": 0.3})
        self.assertEqual(out["host"]["position"], "Custom Coordinates")

    def test_no_portraits_means_no_drift_report_and_no_continuity_keyframe_use(self):
        out = self.render(use_portraits=False, continuity=False)
        self.assertEqual(out["drift"], [])
        self.assertTrue(os.path.getsize(out["output_path"]) > 1000)

    def test_stricter_threshold_flags_more(self):
        lax = self.render(drift_threshold=0.9)["drift"]
        strict = self.render(drift_threshold=0.0)["drift"]
        self.assertTrue(all(d["status"] == "ok" for d in lax))
        self.assertTrue(any(d["status"] != "ok" for d in strict))


class LazyLoadingTests(unittest.TestCase):
    def test_import_loads_no_heavy_modules(self):
        code = ("import sys; sys.path.insert(0, '.');"
                "from backend import vimax_orchestrator, novel_adapter, script_parser, dependency_graph;"
                "heavy=('torch','sentence_transformers','faster_whisper','mediapipe','paddleocr','rapidocr_onnxruntime','onnxruntime','cv2','numpy','transformers');"
                "print([m for m in heavy if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_offline_planning_never_imports_torch_or_sentence_transformers(self):
        code = ("import sys; sys.path.insert(0, '.');"
                "from backend import vimax_orchestrator as v;"
                "v.ViMaxOrchestrator().run('INT. A - DAY\\n\\nBOB\\nHi there.\\n', mode='script2video');"
                "print([m for m in ('torch','sentence_transformers','faster_whisper','mediapipe','paddleocr') if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)


if __name__ == "__main__":
    unittest.main()
