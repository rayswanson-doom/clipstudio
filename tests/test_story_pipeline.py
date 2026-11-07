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
        self.assertEqual(sent[0][1], "FOLDER")
        self.assertTrue(sent[0][0].endswith(".mp4"))
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", side_effect=RuntimeError("boom")):
            r, _ = self.run_it(mode="script2video", folder_id="FOLDER")
        self.assertIsNone(r["drive_link"])
        self.assertTrue(any("Drive backup failed" in w for w in r["warnings"]))
        self.assertTrue(os.path.isfile(r["video_path"]))

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
