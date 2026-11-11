"""Tests frontend/app.py's logic with a stand-in `streamlit` module.

This checks WHAT the app renders and which pipeline calls its buttons make. It
does NOT check visual layout - that is tests/ui/test_layout_playwright.py,
which needs the real Streamlit server and a browser.
"""
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import pipeline  # noqa: E402
import uploader  # noqa: E402

APP = ROOT / "frontend" / "app.py"


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Bar:
    def __init__(self):
        self.updates = []

    def progress(self, fraction, text=None):
        self.updates.append((fraction, text))

    def empty(self):
        pass


class FakeUpload:
    name = "my clip.mp4"

    def __init__(self, name=None, data=b"video-bytes"):
        self.name, self.data = name or self.name, data

    def getbuffer(self):
        return self.data


class FakeSt:
    DEFAULTS = {}

    def __init__(self, inputs=None, clicked=()):
        self.inputs, self.clicked = {**self.DEFAULTS, **(inputs or {})}, set(clicked)
        self.calls, self.session_state, self.sidebar = [], {}, Ctx()
        self.tab_labels = None

    def _log(self, name, *a, **k):
        self.calls.append((name, a, k))

    def names(self, name):
        return [c for c in self.calls if c[0] == name]

    # widgets
    def text_input(self, label, value="", key=None, **k):
        self._log("text_input", label, key=key, value=value)
        return self.inputs.get(key, value)

    def text_area(self, label, key=None, **k):
        self._log("text_area", label, key=key)
        return self.inputs.get(key, "")

    def file_uploader(self, label, type=None, key=None, **k):
        self._log("file_uploader", label, key=key, type=type)
        return self.inputs.get(key)

    def slider(self, label, mn=None, mx=None, value=None, key=None, **k):
        self._log("slider", label, key=key)
        return self.inputs.get(key, value)

    def number_input(self, label, mn=None, mx=None, value=None, key=None, **k):
        self._log("number_input", label, key=key)
        return self.inputs.get(key, value)

    def checkbox(self, label, value=False, key=None, **k):
        self._log("checkbox", label, key=key)
        return self.inputs.get(key, value)

    def selectbox(self, label, options, index=0, key=None, **k):
        self._log("selectbox", label, key=key, options=list(options))
        return self.inputs.get(key, options[index])

    def button(self, label, key=None, **k):
        self._log("button", label, key=key)
        return key in self.clicked

    # layout
    def tabs(self, labels):
        self.tab_labels = list(labels)
        return [Ctx() for _ in labels]

    def columns(self, spec):
        return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, *a, **k):
        return Ctx()

    def container(self, *a, **k):
        return Ctx()

    def progress(self, value, text=None):
        self.bar = Bar()
        self.bar.progress(value, text)
        return self.bar

    def __getattr__(self, name):  # header, caption, warning, error, video, ...
        def f(*a, **k):
            self.calls.append((name, a, k))
            return Ctx()
        return f


class ClipFake(FakeSt):
    DEFAULTS = {"lab_mode": "Clip Repurposer"}


class ClassicFake(FakeSt):
    DEFAULTS = {"lab_mode": "Idea2Video", "idea_flavor": "Narrated video with a 3D avatar (classic)"}


def run_app(fake):
    with mock.patch.dict(sys.modules, {"streamlit": fake}):
        runpy.run_path(str(APP), run_name="__main__")
    return fake


CLIP = {"start": 1.0, "end": 20.0, "duration": 19.0, "score": 9.5, "reasons": ["question"],
        "text": "Did you know?", "output_path": "/tmp/c1.mp4", "drive_link": "https://drive/c1", "error": None}
VUGOLA = {"source_path": "/tmp/s.mp4", "word_count": 120, "clips": [CLIP], "warnings": []}
VIEWMAX = {"script": {"script": "Hello there", "word_count": 2, "estimated_seconds": 0.8},
           "audio_path": "/tmp/a.mp3", "video_path": "/tmp/v.mp4", "drive_link": "https://drive/v",
           "warnings": []}


class StructureTests(unittest.TestCase):
    def test_four_creation_modes_and_sidebar_folder_field(self):
        f = run_app(FakeSt())
        self.assertIsNone(f.tab_labels)  # one Lab with a mode selector, no separate tabs
        mode = [c for c in f.names("selectbox") if c[2]["key"] == "lab_mode"][0]
        self.assertEqual(mode[1][0], "Creation mode")
        self.assertEqual(mode[2]["options"], ["Idea2Video", "Novel2Video", "Script2Video", "Clip Repurposer"])
        labels = [c[1][0] for c in f.names("text_input")]
        self.assertIn("Google Drive Folder ID", labels)
        self.assertEqual(f.names("set_page_config")[0][2]["layout"], "wide")

    def test_each_mode_shows_only_its_own_controls(self):
        def keys(inputs):
            f = run_app(FakeSt(inputs))
            return {c[2].get("key") for n in ("text_area", "text_input", "button", "file_uploader", "selectbox") for c in f.names(n)}
        idea = keys({})
        self.assertIn("story_text_idea2video", idea)
        self.assertNotIn("vugola_url", idea)
        self.assertNotIn("story_file_idea2video", idea)  # ideas are typed, not uploaded
        self.assertIn("story_file_novel2video", keys({"lab_mode": "Novel2Video"}))
        self.assertIn("story_file_script2video", keys({"lab_mode": "Script2Video"}))
        clip = keys({"lab_mode": "Clip Repurposer"})
        self.assertIn("vugola_url", clip)
        self.assertNotIn("story_text_idea2video", clip)
        classic = keys({"idea_flavor": "Narrated video with a 3D avatar (classic)"})
        self.assertIn("viewmax_prompt", classic)
        self.assertNotIn("story_text_idea2video", classic)

    def test_no_widget_key_is_used_twice_on_one_page(self):
        for inputs in ({}, {"lab_mode": "Novel2Video"}, {"lab_mode": "Script2Video"}, {"lab_mode": "Clip Repurposer"},
                       {"idea_flavor": "Narrated video with a 3D avatar (classic)"}):
            f = run_app(FakeSt(inputs))
            ks = [c[2].get("key") for c in f.calls if c[2].get("key")]
            self.assertEqual(len(ks), len(set(ks)), inputs)

    def test_token_status_in_sidebar(self):
        with tempfile.NamedTemporaryFile() as tok, mock.patch.object(uploader, "TOKEN_FILE", Path(tok.name)):
            f = run_app(FakeSt())
        self.assertTrue(any("connected" in c[1][0] for c in f.names("success")))
        with mock.patch.object(uploader, "TOKEN_FILE", Path("/no/token.json")):
            f = run_app(FakeSt())
        self.assertTrue(any("not connected" in c[1][0] for c in f.names("warning")))

    def test_hint_when_folder_empty(self):
        f = run_app(FakeSt())
        self.assertTrue(any("Folder ID" in c[1][0] for c in f.names("info")))
        f = run_app(FakeSt(inputs={"drive_folder_id": "ABC"}))
        self.assertFalse(any("Folder ID" in c[1][0] for c in f.names("info")))


class VugolaTabTests(unittest.TestCase):
    def test_url_run_calls_pipeline_and_shows_results(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            f = run_app(ClipFake({"vugola_url": " https://youtu.be/x ", "drive_folder_id": " FOLDER "},
                               clicked=["vugola_run"]))
        args, kw = rv.call_args
        self.assertEqual(args, ("https://youtu.be/x", "FOLDER"))
        self.assertEqual((kw["top_n"], kw["min_len"], kw["max_len"]), (3, 15, 45))
        self.assertIs(f.session_state["vugola_result"], VUGOLA)
        self.assertEqual(f.names("video")[0][1][0], "/tmp/c1.mp4")
        self.assertEqual(f.names("link_button")[0][1], ("Open in Google Drive", "https://drive/c1"))
        self.assertTrue(f.bar.updates)

    def test_uploaded_file_is_saved_and_preferred_over_url(self):
        seen = {}

        def fake_pipeline(source, *a, **k):
            seen["source"], seen["bytes"] = source, Path(source).read_bytes()  # file exists DURING the run
            return VUGOLA

        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=fake_pipeline):
            run_app(ClipFake({"vugola_file": FakeUpload(), "vugola_url": "https://youtu.be/x"},
                           clicked=["vugola_run"]))
        self.assertTrue(seen["source"].endswith("my clip.mp4"))
        self.assertEqual(seen["bytes"], b"video-bytes")
        self.assertFalse(Path(seen["source"]).exists(), "temporary upload must be deleted after the run")

    def test_nothing_entered_warns_and_skips_pipeline(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline") as rv:
            f = run_app(ClipFake(clicked=["vugola_run"]))
        rv.assert_not_called()
        self.assertTrue(any("YouTube URL" in c[1][0] for c in f.names("warning")))

    def test_min_greater_than_max(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline") as rv:
            f = run_app(ClipFake({"vugola_url": "u", "vugola_min_len": 50, "vugola_max_len": 20},
                               clicked=["vugola_run"]))
        rv.assert_not_called()
        self.assertTrue(f.names("error"))

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=RuntimeError("no speech")):
            f = run_app(ClipFake({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("no speech" in c[1][0] for c in f.names("error")))
        self.assertNotIn("vugola_result", f.session_state)

    def test_clip_error_and_warnings_rendered(self):
        bad = {**VUGOLA, "warnings": ["No Google Drive Folder ID set"],
               "clips": [{**CLIP, "drive_link": None, "error": "drive down"}]}
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=bad):
            f = run_app(ClipFake({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("No Google Drive" in c[1][0] for c in f.names("warning")))
        self.assertTrue(any("drive down" in c[1][0] for c in f.names("error")))
        self.assertFalse(f.names("link_button"))


STORY = {"mode": "script2video", "plan": {"title": "T"}, "script": "INT. A - DAY\n", "warnings": [], "issues": [], "trace": [{"agent": "Director", "note": "n"}],
         "characters": [{"name": "Bob", "visual_identifier": "tall"}], "drift": [], "project_dir": "/tmp/p", "video_path": "/tmp/s.mp4",
         "drive_link": "https://drive/s",
         "storyboard": {"total_duration": 3.0, "shots": [{"idx": 0, "start": 0.0, "end": 3.0, "shot_type": "wide", "camera_angle": "eye",
                                                           "camera_movement": "static", "characters": ["Bob"], "camera_prompt": "cp"}]}}


SCRIPT = """Title: Kitchen Mix-up

INT. KITCHEN - NIGHT

Maya, a tall woman in a red apron with long black hair, stirs a pot.

MAYA
LOL that was hilarious!

BOB
No way, I can't believe it!

EXT. STREET - DAY

Bob, an old man in a grey coat, waves from the street.

BOB
Good morning!
"""


def real_plan(folder, mode="script2video"):
    """A genuine planned episode with its asset gate open and the prompt cards saved in `folder`."""
    import vimax_orchestrator as vo
    plan = vo.ViMaxOrchestrator().run(SCRIPT, mode=mode)
    vo.open_asset_gate(plan, folder)
    plan.update({"assets_dir": folder, "frame_size": [180, 320], "warnings": [], "project_dir": folder + "_p", "video_path": None})
    return plan


def wizard_state(folder, step, mode="script2video", fill=False):
    plan = real_plan(folder, mode)
    if fill:
        am = pipeline.asset_manager
        book = am.AssetBook.load(folder)
        am.generate_images(book, backend="draw", frame_size=(180, 320))
        book.save()
    return {"step": step, "source": SCRIPT, "plan": plan, "temp": [], "seen": {}, "backend": "draw",
            "common": {"mode": mode, "folder_id": None, "output_dir": "/tmp/o", "provider": "offline", "api_key": None, "style": "human",
                       "target_seconds": 30.0, "episode": 1, "episodes": None, "all_episodes": False, "camera": None,
                       "use_portraits": True, "continuity": True, "drift_threshold": 0.35, "host_avatar": None}}


class StoryTabTests(unittest.TestCase):
    def test_controls_exist_in_every_story_mode(self):
        for mode, label in (("idea2video", "Idea2Video"), ("novel2video", "Novel2Video"), ("script2video", "Script2Video")):
            f = run_app(FakeSt({"lab_mode": label}))
            keys = {c[2]["key"] for n in ("selectbox", "text_area", "checkbox", "slider", "file_uploader") for c in f.names(n)}
            self.assertTrue({f"story_text_{mode}", "story_style", "story_provider", "story_portraits", "story_continuity",
                             "story_tolerance", "story_cam_size", "story_cam_angle", "story_cam_move", "story_host_on",
                             "story_host_pos", "story_host_scale", "story_host_file"} <= keys, (mode, keys))
        f = run_app(FakeSt({"lab_mode": "Novel2Video"}))
        up = [c for c in f.names("file_uploader") if c[2]["key"] == "story_file_novel2video"]
        self.assertEqual(up[0][2]["type"], ["txt", "md", "fountain"])
        f = run_app(FakeSt())
        self.assertIn("story_episodes", {c[2]["key"] for c in f.names("number_input")})
        f = run_app(FakeSt({"lab_mode": "Script2Video"}))
        self.assertNotIn("story_episodes", {c[2]["key"] for c in f.names("number_input")})  # a script is already one episode

    def test_empty_text_warns_and_does_not_run(self):
        with mock.patch.object(pipeline, "run_story_pipeline", return_value=STORY) as rs:
            f = run_app(FakeSt(clicked=["story_run"]))
        rs.assert_not_called()
        self.assertTrue(any("Type or upload" in c[1][0] for c in f.names("warning")))

    def test_script_run_forwards_every_option_and_shows_results(self):
        plan = real_plan(tempfile.mkdtemp())
        with mock.patch.object(pipeline, "run_story_pipeline", return_value=plan) as rs:
            f = run_app(FakeSt({"lab_mode": "Script2Video", "story_text_script2video": "INT. A - DAY\n\nBOB\nHi\n",
                                "story_provider": "Groq (needs an API key)", "story_api_key": " k ", "story_style": "Anime",
                                "story_seconds": 30, "drive_folder_id": "F", "story_portraits": False, "story_continuity": False,
                                "story_tolerance": 20, "story_cam_size": "Close-ups", "story_cam_angle": "Low angle (heroic)",
                                "story_cam_move": "Handheld", "story_host_on": True, "story_host_pos": "Top-Left",
                                "story_host_scale": 33}, clicked=["story_run"]))
        kw = rs.call_args[1]
        self.assertEqual((kw["mode"], kw["provider"], kw["api_key"], kw["style"], kw["folder_id"]),
                         ("script2video", "groq", "k", "anime", "F"))
        self.assertEqual(kw["target_seconds"], 30.0)
        self.assertEqual(kw["camera"], {"size": "close", "angle": "low", "movement": "handheld"})
        self.assertEqual((kw["use_portraits"], kw["continuity"], kw["drift_threshold"]), (False, False, 0.2))
        self.assertEqual(kw["host_avatar"], {"style": "anime", "position": "Top-Left", "xy": None, "scale": 0.33})
        self.assertIsNone(kw["episodes"])
        self.assertFalse(kw["all_episodes"])
        self.assertFalse(kw["render"])  # step 1 only plans; rendering waits for the asset gate
        self.assertNotIn("memes", kw)
        wiz = f.session_state["wiz_script2video"]
        self.assertEqual(wiz["step"], 2)
        self.assertEqual(wiz["common"]["host_avatar"]["position"], "Top-Left")

    def test_novel_makes_every_episode_by_default(self):
        with mock.patch.object(pipeline, "run_story_pipeline", return_value={**STORY, "mode": "novel2video"}) as rs:
            run_app(FakeSt({"lab_mode": "Novel2Video", "story_text_novel2video": "x", "story_episodes": 4}, clicked=["story_run"]))
        kw = rs.call_args[1]
        self.assertEqual((kw["mode"], kw["episodes"], kw["all_episodes"]), ("novel2video", 4, True))

    def test_idea_single_episode_and_custom_host_coordinates(self):
        with mock.patch.object(pipeline, "run_story_pipeline", return_value={**STORY, "mode": "idea2video"}) as rs:
            run_app(FakeSt({"story_text_idea2video": "A story.", "story_host_on": True, "story_host_pos": "Custom Coordinates",
                            "story_host_x": 55, "story_host_y": 77}, clicked=["story_run"]))
        kw = rs.call_args[1]
        self.assertEqual((kw["mode"], kw["all_episodes"], kw["episodes"]), ("idea2video", False, 1))
        self.assertEqual(kw["host_avatar"]["xy"], (55, 77))
        self.assertEqual(kw["host_avatar"]["position"], "Custom Coordinates")

    def test_host_off_sends_no_host(self):
        with mock.patch.object(pipeline, "run_story_pipeline", return_value=STORY) as rs:
            run_app(FakeSt({"story_text_idea2video": "x", "story_host_pos": "Top-Left"}, clicked=["story_run"]))
        self.assertIsNone(rs.call_args[1]["host_avatar"])

    def test_uploaded_text_file_is_read_and_deleted(self):
        up = FakeUpload("novel.txt", "Chapter 1\n\nElena walked.".encode())
        with mock.patch.object(pipeline, "run_story_pipeline", return_value={**STORY, "mode": "novel2video"}) as rs:
            run_app(FakeSt({"lab_mode": "Novel2Video", "story_file_novel2video": up}, clicked=["story_run"]))
        self.assertIn("Elena walked", rs.call_args[0][0])

    def test_offline_never_sends_the_key(self):
        with mock.patch.object(pipeline, "run_story_pipeline", return_value=STORY) as rs:
            run_app(FakeSt({"story_text_idea2video": "x", "story_api_key": "leftover"}, clicked=["story_run"]))
        self.assertEqual((rs.call_args[1]["provider"], rs.call_args[1]["api_key"]), ("offline", None))

    def test_result_for_another_mode_is_not_shown(self):
        f = FakeSt({"lab_mode": "Script2Video"})
        f.session_state["story_result"] = {**STORY, "mode": "idea2video"}
        run_app(f)
        self.assertEqual(f.names("video"), [])

    def test_episode_cards_and_drive_links(self):
        res = {**STORY, "mode": "idea2video", "drive_links": ["https://drive/1", "https://drive/2"],
               "episodes_out": [{"episode": i, "title": "T", "video_path": f"/tmp/e{i}.mp4", "shots": 3, "seconds": 9.0,
                                 "drive_link": f"https://drive/{i}", "project_link": f"https://drive/p{i}"} for i in (1, 2)]}
        f = FakeSt()
        f.session_state["story_result"] = res
        run_app(f)
        self.assertEqual([c[1][0] for c in f.names("video")], ["/tmp/e1.mp4", "/tmp/e2.mp4"])
        self.assertEqual(len(f.names("link_button")), 4)

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_story_pipeline", side_effect=ValueError("bad text")):
            f = run_app(FakeSt({"story_text_idea2video": "x"}, clicked=["story_run"]))
        self.assertTrue(any("bad text" in c[1][0] for c in f.names("error")))


class ViewmaxTabTests(unittest.TestCase):
    def test_run_calls_pipeline(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            f = run_app(ClassicFake({"viewmax_prompt": "a script about tea", "drive_folder_id": "FOLDER",
                                "viewmax_voice": "en-GB-RyanNeural"}, clicked=["viewmax_run"]))
        args, kw = rv.call_args
        self.assertEqual(args, ("a script about tea", "FOLDER"))
        self.assertIsNone(kw["background_video"])
        self.assertEqual(kw["voice"], "en-GB-RyanNeural")
        self.assertEqual(f.names("audio")[0][1][0], "/tmp/a.mp3")
        self.assertEqual(f.names("video")[0][1][0], "/tmp/v.mp4")
        self.assertEqual(f.names("link_button")[0][1][1], "https://drive/v")

    def test_background_upload_forwarded(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertTrue(rv.call_args[1]["background_video"].endswith("my clip.mp4"))

    def test_avatar_controls_exist(self):
        f = run_app(ClassicFake())
        boxes = [c for c in f.names("checkbox") if c[2]["key"] == "viewmax_avatar_on"]
        self.assertEqual(boxes[0][1][0], "Enable AI Avatar Lip-Sync Commentary")
        up = [c for c in f.names("file_uploader") if c[2]["key"] == "viewmax_avatar"]
        self.assertEqual(up[0][2]["type"], ["glb", "vrm", "jpg", "jpeg", "png"])

    def test_avatar_off_sends_no_avatar_args(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_avatar": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertNotIn("avatar_image", rv.call_args[1])
        self.assertNotIn("avatar_enabled", rv.call_args[1])

    def test_avatar_on_forwards_image(self):
        up = FakeUpload()
        up.name = "me.png"
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_avatar": up},
                           clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertTrue(kw["avatar_enabled"])
        self.assertTrue(kw["avatar_image"].endswith("me.png"))

    def test_avatar_on_without_file_uses_the_builtin_3d_avatar(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_model_type": "Anime"},
                           clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertEqual((kw["avatar_image"], kw["avatar_type"]), ("demo:anime", "anime"))

    def test_podcast_layout_controls_exist_and_are_forwarded(self):
        f = run_app(ClassicFake())
        labels = [c[1][0] for c in f.names("selectbox")]
        for needle in ("Model type", "Avatar position", "Render quality"):
            self.assertTrue(any(needle in l for l in labels), needle)
        sel = [c for c in f.names("selectbox") if c[2]["key"] == "viewmax_model_type"]
        self.assertTrue(sel)
        up = FakeUpload()
        up.name = "host.glb"
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_avatar": up,
                            "viewmax_model_type": "Photorealistic Human", "viewmax_position": "Top-Left",
                            "viewmax_scale": 30, "viewmax_gestures": False,
                            "viewmax_quality": "Fast (low-spec laptops)"}, clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertTrue(kw["avatar_image"].endswith("host.glb"))
        self.assertEqual((kw["avatar_type"], kw["avatar_position"], kw["avatar_scale"], kw["avatar_gestures"],
                          kw["avatar_quality"]), ("human", "Top-Left", 0.30, False, "fast"))
        self.assertNotIn("avatar_xy", kw)

    def test_custom_coordinates_show_x_y_and_are_forwarded(self):
        inputs = {"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_position": "Custom Coordinates",
                  "viewmax_x": 120, "viewmax_y": 800}
        f = run_app(ClassicFake(inputs))
        self.assertEqual({c[2]["key"] for c in f.names("number_input") if c[2]["key"].startswith("viewmax")}, {"viewmax_x", "viewmax_y"})
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake(inputs, clicked=["viewmax_run"]))
        self.assertEqual(rv.call_args[1]["avatar_xy"], (120, 800))

    def test_x_y_hidden_for_corner_presets(self):
        f = run_app(ClassicFake({"viewmax_avatar_on": True, "viewmax_position": "Bottom-Left"}))
        self.assertEqual([c for c in f.names("number_input") if c[2]["key"].startswith("viewmax")], [])

    def test_layout_preview_is_shown_only_when_the_avatar_is_on(self):
        self.assertFalse(any("Layout preview" in str(c) for c in run_app(ClassicFake()).names("image")))
        f = run_app(ClassicFake({"viewmax_avatar_on": True}))
        self.assertTrue(any("Layout preview" in str(c) for c in f.names("image")))

    def test_3d_result_tiles(self):
        res = {**VIEWMAX, "avatar": {"mode": "3d", "layout": {"preset": "Top-Right"}, "gestures": [{"name": "wave"}]}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        for needle in ("Avatar position", "Top-Right", "Gestures played"):
            self.assertIn(needle, text)

    def test_legacy_card_layout_string_does_not_crash_the_tiles(self):
        res = {**VIEWMAX, "avatar": {"mode": "mouth-warp", "layout": "full"}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        self.assertIn("Mouth sync", text)
        self.assertNotIn("Avatar position", text)

    def test_avatar_result_badge(self):
        res = {**VIEWMAX, "avatar": {"mode": "pulse"}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        self.assertTrue(any("Avatar animation" in str(c) for c in f.names("markdown")))

    def test_script_actions_and_ocr_options_are_forwarded(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_script": "My own words.", "viewmax_actions": "Zoom in then pan right",
                            "viewmax_read_text": True, "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertEqual((kw["custom_script"], kw["action_prompt"], kw["read_background_text"]),
                         ("My own words.", "Zoom in then pan right", True))

    def test_ocr_reader_captions_music_and_backdrop_controls(self):
        f = run_app(ClassicFake())
        keys = {c[2]["key"] for c in f.names("selectbox")} | {c[2]["key"] for c in f.names("checkbox")} | {
            c[2]["key"] for c in f.names("file_uploader")}
        for key in ("viewmax_ocr_backend", "viewmax_show_text", "viewmax_music", "viewmax_backdrop"):
            self.assertIn(key, keys)
        music = FakeUpload()
        music.name = "bed.mp3"
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_read_text": True, "viewmax_ocr_backend": "PaddleOCR",
                            "viewmax_show_text": False, "viewmax_music": music, "viewmax_backdrop": "Sunset"},
                           clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertEqual((kw["ocr_backend"], kw["show_screen_text"], kw["broll_style"]), ("paddleocr", False, "sunset"))
        self.assertTrue(kw["music_path"].endswith("bed.mp3"))

    def test_clip_tab_ocr_reader_is_forwarded(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(ClipFake({"vugola_url": "http://x/y.mp4", "vugola_ocr": True, "vugola_ocr_backend": "PaddleOCR"},
                           clicked=["vugola_run"]))
        self.assertEqual(rv.call_args[1]["ocr_backend"], "paddleocr")

    def test_defaults_do_not_send_optional_render_arguments(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        for k in ("ocr_backend", "show_screen_text", "music_path", "broll_style"):
            self.assertNotIn(k, rv.call_args[1])

    def test_motion_css_is_gsap_style_and_respects_reduced_motion(self):
        css = Path(APP).read_text(encoding="utf-8")
        for needle in ("cubic-bezier(.215,.61,.355,1)", "cubic-bezier(.34,1.56,.64,1)", "@keyframes cp-rise",
                       "prefers-reduced-motion"):
            self.assertIn(needle, css)

    def test_custom_script_alone_is_enough_without_a_prompt(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            f = run_app(ClassicFake({"viewmax_script": "Just my script."}, clicked=["viewmax_run"]))
        rv.assert_called_once()
        self.assertFalse([c for c in f.names("warning") if "prompt" in c[1][0].lower()])

    def test_unused_options_are_not_sent(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        for k in ("custom_script", "action_prompt", "read_background_text"):
            self.assertNotIn(k, rv.call_args[1])

    def test_lab_uploads_are_deleted_after_the_run(self):
        paths = []

        def fake(prompt, folder, background_video=None, **k):
            paths.append(background_video)
            self.assertTrue(Path(background_video).exists())
            return VIEWMAX

        with mock.patch.object(pipeline, "run_viewmax_pipeline", side_effect=fake):
            run_app(ClassicFake({"viewmax_prompt": "p", "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertFalse(Path(paths[0]).exists())

    def test_shot_list_and_screen_text_are_shown(self):
        res = {**VIEWMAX, "screen_text": ["SALE 50%"],
               "script": {**VIEWMAX["script"], "shot_list": ["0.0s  zoom in: Zoom in"]}, "cues_applied": 1}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        self.assertIn("SALE 50%", text)
        self.assertIn("Zoom in", text)
        self.assertIn("Visual actions", text)

    def test_ocr_checkbox_is_forwarded_and_screen_text_chips_shown(self):
        clip = {**CLIP, "screen_text": ["BIG SALE"]}
        res = {**VUGOLA, "clips": [clip]}
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=res) as rv:
            f = run_app(ClipFake({"vugola_url": "https://youtu.be/x", "vugola_ocr": True}, clicked=["vugola_run"]))
        self.assertTrue(rv.call_args[1]["read_screen_text"])
        self.assertIn("BIG SALE", " ".join(str(c) for c in f.names("markdown")))

    def test_empty_prompt_warns(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline") as rv:
            f = run_app(ClassicFake(clicked=["viewmax_run"]))
        rv.assert_not_called()
        self.assertTrue(any("prompt" in c[1][0].lower() for c in f.names("warning")))

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", side_effect=RuntimeError("tts down")):
            f = run_app(ClassicFake({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        self.assertTrue(any("tts down" in c[1][0] for c in f.names("error")))


if __name__ == "__main__":
    unittest.main()


class MemePanelTests(unittest.TestCase):
    def setUp(self):
        import os
        self.tmp = tempfile.mkdtemp()
        self._old = os.environ.get("CINEPULSE_MEMES")
        os.environ["CINEPULSE_MEMES"] = self.tmp

    def tearDown(self):
        import os
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self._old is None:
            os.environ.pop("CINEPULSE_MEMES", None)
        else:
            os.environ["CINEPULSE_MEMES"] = self._old

    def test_toggle_is_present_in_every_generating_mode_and_off_by_default(self):
        for inputs, prefix in (({"lab_mode": "Clip Repurposer"}, "vugola"), ({"idea_flavor": "Narrated video with a 3D avatar (classic)"}, "viewmax"),
                               ({}, "story"), ({"lab_mode": "Script2Video"}, "story"), ({"lab_mode": "Novel2Video"}, "story")):
            f = FakeSt(inputs)
            if prefix == "story":  # the story modes show it in wizard step 3
                mode = {"Script2Video": "script2video", "Novel2Video": "novel2video"}.get(inputs.get("lab_mode"), "idea2video")
                f.session_state[f"wiz_{mode}"] = wizard_state(tempfile.mkdtemp(), 3, mode, fill=True)
            f = run_app(f)
            boxes = [c for c in f.names("checkbox") if c[2]["key"] == f"{prefix}_meme_on"]
            self.assertEqual(len(boxes), 1, inputs)
            self.assertEqual(boxes[0][1][0], "Enable Contextual Meme Reactions")
            ups = [c for c in f.names("file_uploader") if c[2]["key"] == f"{prefix}_meme_up"]
            self.assertEqual(len(ups), 1)

    def test_clip_run_forwards_meme_settings(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(ClipFake({"vugola_url": "https://youtu.be/x", "vugola_meme_on": True, "vugola_meme_mode": "Corner tile only",
                              "vugola_meme_rate": 6, "vugola_meme_pos": "Custom Coordinates", "vugola_meme_x": 11,
                              "vugola_meme_y": 22}, clicked=["vugola_run"]))
        cfg = rv.call_args.kwargs["memes"]
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["mode"], "overlay")
        self.assertEqual(cfg["max_per_minute"], 6.0)
        self.assertEqual(cfg["position"], "Custom Coordinates")
        self.assertEqual(cfg["xy"], (11, 22))

    def test_off_sends_none_and_classic_forwards_when_on(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "hi"}, clicked=["viewmax_run"]))
        self.assertIsNone(rv.call_args.kwargs["memes"])
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(ClassicFake({"viewmax_prompt": "hi", "viewmax_meme_on": True}, clicked=["viewmax_run"]))
        cfg = rv.call_args.kwargs["memes"]
        self.assertTrue(cfg["enabled"])
        self.assertNotIn("position", cfg)  # auto: avoid the avatar

    def test_upload_goes_into_the_meme_folder(self):
        import shutil
        import cv2
        import numpy as np
        ff = shutil.which("ffmpeg")
        if not ff:
            self.skipTest("ffmpeg needed")
        src = Path(self.tmp) / "src.mp4"
        vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for _ in range(15):
            vw.write(np.zeros((64, 64, 3), np.uint8))
        vw.release()

        class Up:
            name = "lol.mp4"

            def getvalue(self):
                return src.read_bytes()
        f = ClipFake({"vugola_meme_up": [Up()], "vugola_meme_emo": "laugh"}, clicked=["vugola_meme_add"])
        run_app(f)
        self.assertTrue(list(Path(self.tmp, "laugh").glob("lol*.mp4")), f.names("error"))


class WizardTests(unittest.TestCase):
    def setUp(self):
        import os
        self.tmp = tempfile.mkdtemp()
        self._old = os.environ.get("CINEPULSE_MEMES")
        os.environ["CINEPULSE_MEMES"] = os.path.join(self.tmp, "memes")
        self.folder = os.path.join(self.tmp, "assets")

    def tearDown(self):
        import os
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self._old is None:
            os.environ.pop("CINEPULSE_MEMES", None)
        else:
            os.environ["CINEPULSE_MEMES"] = self._old

    def app(self, step, inputs=None, clicked=(), fill=False, mode="script2video"):
        f = FakeSt({"lab_mode": "Script2Video", **(inputs or {})}, clicked=clicked)
        f.session_state[f"wiz_{mode}"] = wizard_state(self.folder, step, mode, fill)
        return run_app(f), f.session_state[f"wiz_{mode}"]

    def test_step1_is_the_default_and_has_no_render_controls(self):
        f = run_app(FakeSt({"lab_mode": "Script2Video"}))
        labels = [c[1][0] for c in f.names("button")]
        self.assertIn("Parse the story and continue", labels)
        self.assertNotIn("Render the episode", labels)
        self.assertTrue(any("Story &amp; script" in str(c[1][0]) for c in f.names("markdown")))

    def test_step2_shows_a_card_per_character_and_location(self):
        f, _ = self.app(2)
        keys = {c[2]["key"] for n in ("text_area", "file_uploader", "button") for c in f.names(n)}
        for cid in ("char_Maya", "char_Bob", "loc_kitchen", "loc_street"):
            self.assertIn(f"asset_prompt_script2video_{cid}", keys)
            self.assertIn(f"asset_up_script2video_{cid}", keys)
            self.assertIn(f"asset_gen_script2video_{cid}", keys)
        self.assertIn("asset_gen_all", keys)
        self.assertTrue(any("Generate Reference Visuals" == c[1][0] for c in f.names("button")))

    def test_generate_all_fills_the_book(self):
        _, wiz = self.app(2, clicked=["asset_gen_all"], inputs={"asset_backend_script2video": "Built-in painter (offline, simple flat art)"})
        book = pipeline.asset_manager.AssetBook.load(self.folder)
        self.assertTrue(book.ready())
        self.assertEqual(set(wiz["plan"]["asset_gate"]["required"]), set(book.cards))

    def test_upload_attaches_a_picture_and_bad_files_are_rejected(self):
        import cv2
        import numpy as np
        ok, png = cv2.imencode(".png", np.full((200, 100, 3), 120, np.uint8))
        good = FakeUpload("maya.png", png.tobytes())
        good.getvalue = lambda: png.tobytes()
        _, _ = self.app(2, inputs={"asset_up_script2video_char_Maya": good})
        book = pipeline.asset_manager.AssetBook.load(self.folder)
        self.assertEqual(book.cards["char:Maya"]["source"], "upload")
        bad = FakeUpload("x.png", b"not an image")
        bad.getvalue = lambda: b"not an image"
        f, _ = self.app(2, inputs={"asset_up_script2video_loc_kitchen": bad})
        self.assertTrue(any("could not be read" in c[1][0] for c in f.names("error")))
        self.assertIn("loc:kitchen", pipeline.asset_manager.AssetBook.load(self.folder).missing())

    def test_the_gate_stays_closed_until_every_card_has_a_picture(self):
        f, wiz = self.app(2, clicked=["wiz_to3"])
        self.assertEqual(wiz["step"], 2)
        self.assertTrue(any("asset gate is closed" in c[1][0] for c in f.names("error")))

    def test_closing_the_gate_moves_to_step3_with_locked_assets_in_the_preview(self):
        f, wiz = self.app(2, clicked=["wiz_to3"], fill=True)
        self.assertEqual(wiz["step"], 3)
        self.assertFalse(wiz["plan"]["asset_gate"]["open"])
        f, _ = self.app(3, fill=True)
        self.assertGreaterEqual(len(f.names("image")), 4)  # location + character thumbnails
        self.assertIn("Render the episode", [c[1][0] for c in f.names("button")])

    def test_step3_render_passes_the_locked_book_and_shows_the_result(self):
        res = {**STORY, "mode": "script2video"}
        with mock.patch.object(pipeline, "run_story_pipeline", return_value=res) as rs:
            f, wiz = self.app(3, clicked=["wiz_render"], fill=True, inputs={"story_voice": True, "story_meme_on": True})
        kw = rs.call_args[1]
        self.assertTrue(kw["render"] and kw["voice"])
        self.assertIs(kw["plan_result"], wiz["plan"])
        self.assertTrue(kw["asset_book"].ready())
        self.assertTrue(kw["memes"]["enabled"])
        self.assertEqual(kw["mode"], "script2video")
        self.assertEqual(wiz["step"], 4)
        self.assertEqual(f.session_state["story_result"], res)

    def test_start_over_and_back_buttons(self):
        _, wiz = self.app(2, clicked=["wiz_back1"])
        self.assertEqual(wiz["step"], 1)
        f = FakeSt({"lab_mode": "Script2Video"}, clicked=["wiz_reset"])
        f.session_state["wiz_script2video"] = wizard_state(self.folder, 2)
        run_app(f)
        self.assertNotIn("wiz_script2video", f.session_state)

