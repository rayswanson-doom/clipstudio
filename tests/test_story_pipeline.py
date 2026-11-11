"""run_story_pipeline: idea/script/novel -> storyboard files (+ CPU animatic), Drive backup, graceful failures."""
import json
import os
import tempfile
import unittest
from unittest import mock

from backend import pipeline

SCRIPT = "INT. KITCHEN - NIGHT\n\nBOB\nHi there.\n\nAMY\nHello Bob.\n"
NOVEL = os.path.join(os.path.dirname(__file__), "data", "sample_novel.txt")


class StoryPipelineTests(unittest.TestCase):
    def run_it(self, source=SCRIPT, **kw):
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        kw.setdefault("size", (180, 320))
        kw.setdefault("fps", 8)
        return pipeline.run_story_pipeline(source, output_dir=d, **kw), d

    def test_script_to_video_and_project_files(self):
        r, d = self.run_it(mode="script2video")
        self.assertTrue(os.path.getsize(r["video_path"]) > 1000)
        for f in ("project.json", "storyboard.json", "characters.json", "graph.json", "graph.mmd", "subtitles.srt", "script.fountain"):
            self.assertTrue(os.path.isfile(os.path.join(r["project_dir"], f)), f)
        self.assertTrue(r["rendered"] and r["drift"])
        self.assertIsNone(r["drive_link"])

    def test_plan_only_has_no_video(self):
        r, _ = self.run_it(mode="script2video", render=False)
        self.assertIsNone(r["video_path"])
        self.assertTrue(r["storyboard"]["shots"])

    def test_novel_episode(self):
        r, _ = self.run_it(open(NOVEL, encoding="utf-8").read(), mode="novel2video", episodes=2, episode=2, render=False)
        self.assertEqual(r["novel"]["episode"], 2)

    def test_missing_key_degrades_with_warning(self):
        keys = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY",)}
        try:
            r, _ = self.run_it(mode="script2video", render=False, provider="groq")
        finally:
            for k, v in keys.items():
                if v is not None:
                    os.environ[k] = v
        self.assertTrue(any("No Groq API key" in w for w in r["warnings"]))
        self.assertTrue(r["llm"]["offline"])

    def test_drive_backup_runs_after_render_and_failures_are_warnings(self):
        sent = []

        def fake_upload(path, folder, **k):
            sent.append((os.path.basename(path), folder))
            return "https://drive/x"

        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", fake_upload):
            r, _ = self.run_it(mode="script2video", folder_id="FOLDER")
        self.assertEqual(r["drive_link"], "https://drive/x")
        self.assertEqual({f for _, f in sent}, {"FOLDER"})
        exts = sorted(n.rsplit(".", 1)[1] for n, _ in sent)
        self.assertEqual([e for e in exts if e in ("mp4", "zip")], ["mp4", "zip"])  # the render AND its project files
        self.assertIn("json", exts)                                                 # assets.json (the prompt cards)
        self.assertGreaterEqual(exts.count("png"), 3)                               # character sheets + location pictures
        self.assertTrue(r["asset_links"])
        self.assertTrue(r["project_link"])
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", side_effect=RuntimeError("boom")):
            r, _ = self.run_it(mode="script2video", folder_id="FOLDER")
        self.assertIsNone(r["drive_link"])
        self.assertTrue(any("Google Drive backup of" in w and "boom" in w for w in r["warnings"]))
        self.assertTrue(os.path.isfile(r["video_path"]))

    def test_no_folder_or_no_sign_in_warns_but_keeps_the_render(self):
        r, _ = self.run_it(mode="script2video", folder_id=None)
        self.assertTrue(any("NOT uploaded" in w for w in r["warnings"]))
        self.assertTrue(os.path.isfile(r["video_path"]))
        with mock.patch.object(pipeline, "drive_status", return_value="unavailable"):
            r, _ = self.run_it(mode="script2video", folder_id="F")
        self.assertTrue(any("not connected" in w for w in r["warnings"]))

    def test_plan_only_uploads_nothing(self):
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", side_effect=AssertionError("no upload expected")):
            r, _ = self.run_it(mode="script2video", folder_id="F", render=False)
        self.assertIsNone(r["drive_link"])

    def test_every_episode_of_a_novel_is_rendered_and_backed_up(self):
        sent = []
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", lambda p, f: sent.append(os.path.basename(p)) or "L" + str(len(sent))):
            r, _ = self.run_it(open(NOVEL, encoding="utf-8").read(), mode="novel2video", episodes=2, all_episodes=True,
                               folder_id="F", size=(120, 214), fps=6, max_shots=3)
        self.assertEqual([e["episode"] for e in r["episodes_out"]], [1, 2])
        self.assertTrue(all(os.path.isfile(e["video_path"]) for e in r["episodes_out"]))
        self.assertEqual(len(r["drive_links"]), 2)
        self.assertEqual(sum(n.endswith(".mp4") for n in sent), 2)
        self.assertEqual(sum(n.endswith("_project.zip") for n in sent), 2)
        self.assertNotEqual(r["episodes_out"][0]["video_path"], r["episodes_out"][1]["video_path"])

    def test_idea2video_makes_episodes_from_one_prompt(self):
        idea = ("Mara waits in a small kitchen at night. Daniel arrives, soaked from the rain. "
                "Mara says nothing. Daniel sits down. They talk until dawn. Morning comes. Mara smiles. Daniel leaves.")
        r, _ = self.run_it(idea, mode="idea2video", episodes=2, all_episodes=True, render=False)
        self.assertEqual(len(r["episodes_out"]), 2)
        scripts = [open(os.path.join(e["project_dir"], "script.fountain"), encoding="utf-8").read() for e in r["episodes_out"]]
        self.assertNotEqual(scripts[0], scripts[1])
        self.assertIn("kitchen", scripts[0].lower())
        self.assertEqual([e["episode"] for e in r["episodes_out"]], [1, 2])

    def test_camera_preferences_reach_the_storyboard(self):
        r, _ = self.run_it(mode="script2video", render=False, camera={"size": "close", "angle": "low", "movement": "push"})
        shots = [s for s in r["storyboard"]["shots"] if s["kind"] != "establishing"]
        self.assertTrue(all(s["shot_type"] == "close-up" and s["camera_angle"] == "low angle" and s["camera_movement"] == "slow push-in" for s in shots))
        self.assertTrue(all("low angle" in s["camera_prompt"] and "close-up" in s["camera_prompt"] for s in shots))
        self.assertEqual(r["camera"], {"size": "close", "angle": "low", "movement": "push"})
        with self.assertRaises(ValueError):
            self.run_it(mode="script2video", render=False, camera={"size": "huge"})

    def test_host_overlay_and_consistency_switches(self):
        r, _ = self.run_it(mode="script2video", host_avatar={"style": "anime", "position": "Top-Right", "scale": 0.3})
        self.assertTrue(os.path.isfile(os.path.join(r["project_dir"], "work", "portrait_host.png")))
        off, _ = self.run_it(mode="script2video", use_portraits=False, continuity=False)
        self.assertTrue(off["drift"])  # the scenic renderer always conditions on the locked pictures (the switches only affect the tile animatic)
        self.assertTrue(os.path.isfile(off["video_path"]))
        strict, _ = self.run_it(mode="script2video", drift_threshold=0.0)
        self.assertTrue(any(d["status"] != "ok" for d in strict["drift"]))

    def test_render_failure_keeps_the_plan(self):
        from backend import vimax_orchestrator as vo

        with mock.patch.object(vo, "render_storyboard", side_effect=RuntimeError("ffmpeg missing")):
            r, _ = self.run_it(mode="script2video")
        self.assertIsNone(r["video_path"])
        self.assertTrue(any("could not be rendered" in w for w in r["warnings"]))
        self.assertTrue(os.path.isfile(os.path.join(r["project_dir"], "storyboard.json")))

    def test_bad_input_raises_clearly(self):
        with self.assertRaises(ValueError):
            self.run_it("   ", mode="script2video")

    def test_progress_is_monotonic_and_ends_at_one(self):
        seen = []
        self.run_it(mode="script2video", progress=lambda s, f: seen.append(f))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_result_is_json_serialisable_without_secrets(self):
        r, _ = self.run_it(mode="script2video", render=False, api_key="SECRET")
        json.dumps(r)
        self.assertNotIn("SECRET", json.dumps(r))


if __name__ == "__main__":
    unittest.main()
