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

    def getbuffer(self):
        return b"video-bytes"


class FakeSt:
    def __init__(self, inputs=None, clicked=()):
        self.inputs, self.clicked = inputs or {}, set(clicked)
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
        self._log("selectbox", label, key=key)
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
    def test_two_tabs_and_sidebar_folder_field(self):
        f = run_app(FakeSt())
        self.assertEqual(f.tab_labels, ["🎬 Clip Repurposer Studio", "⚡ AI Generator Lab"])
        labels = [c[1][0] for c in f.names("text_input")]
        self.assertIn("Google Drive Folder ID", labels)
        self.assertIn("YouTube URL", labels)
        self.assertEqual(f.names("text_area")[0][1][0], "What should the video be about?")
        buttons = [c[1][0] for c in f.names("button")]
        self.assertEqual(buttons, ["Generate viral clips", "Generate video"])
        self.assertEqual(f.names("set_page_config")[0][2]["layout"], "wide")

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
            f = run_app(FakeSt({"vugola_url": " https://youtu.be/x ", "drive_folder_id": " FOLDER "},
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
            run_app(FakeSt({"vugola_file": FakeUpload(), "vugola_url": "https://youtu.be/x"},
                           clicked=["vugola_run"]))
        self.assertTrue(seen["source"].endswith("my clip.mp4"))
        self.assertEqual(seen["bytes"], b"video-bytes")
        self.assertFalse(Path(seen["source"]).exists(), "temporary upload must be deleted after the run")

    def test_nothing_entered_warns_and_skips_pipeline(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline") as rv:
            f = run_app(FakeSt(clicked=["vugola_run"]))
        rv.assert_not_called()
        self.assertTrue(any("YouTube URL" in c[1][0] for c in f.names("warning")))

    def test_min_greater_than_max(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline") as rv:
            f = run_app(FakeSt({"vugola_url": "u", "vugola_min_len": 50, "vugola_max_len": 20},
                               clicked=["vugola_run"]))
        rv.assert_not_called()
        self.assertTrue(f.names("error"))

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=RuntimeError("no speech")):
            f = run_app(FakeSt({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("no speech" in c[1][0] for c in f.names("error")))
        self.assertNotIn("vugola_result", f.session_state)

    def test_clip_error_and_warnings_rendered(self):
        bad = {**VUGOLA, "warnings": ["No Google Drive Folder ID set"],
               "clips": [{**CLIP, "drive_link": None, "error": "drive down"}]}
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=bad):
            f = run_app(FakeSt({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("No Google Drive" in c[1][0] for c in f.names("warning")))
        self.assertTrue(any("drive down" in c[1][0] for c in f.names("error")))
        self.assertFalse(f.names("link_button"))


class ViewmaxTabTests(unittest.TestCase):
    def test_run_calls_pipeline(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            f = run_app(FakeSt({"viewmax_prompt": "a script about tea", "drive_folder_id": "FOLDER",
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
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertTrue(rv.call_args[1]["background_video"].endswith("my clip.mp4"))

    def test_avatar_controls_exist(self):
        f = run_app(FakeSt())
        boxes = [c for c in f.names("checkbox") if c[2]["key"] == "viewmax_avatar_on"]
        self.assertEqual(boxes[0][1][0], "Enable AI Avatar Lip-Sync Commentary")
        up = [c for c in f.names("file_uploader") if c[2]["key"] == "viewmax_avatar"]
        self.assertEqual(up[0][2]["type"], ["glb", "vrm", "jpg", "jpeg", "png"])

    def test_avatar_off_sends_no_avatar_args(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_avatar": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertNotIn("avatar_image", rv.call_args[1])
        self.assertNotIn("avatar_enabled", rv.call_args[1])

    def test_avatar_on_forwards_image(self):
        up = FakeUpload()
        up.name = "me.png"
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_avatar": up},
                           clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertTrue(kw["avatar_enabled"])
        self.assertTrue(kw["avatar_image"].endswith("me.png"))

    def test_avatar_on_without_file_uses_the_builtin_3d_avatar(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_model_type": "Anime"},
                           clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertEqual((kw["avatar_image"], kw["avatar_type"]), ("demo:anime", "anime"))

    def test_podcast_layout_controls_exist_and_are_forwarded(self):
        f = run_app(FakeSt())
        labels = [c[1][0] for c in f.names("selectbox")]
        for needle in ("Model type", "Avatar position", "Render quality"):
            self.assertTrue(any(needle in l for l in labels), needle)
        sel = [c for c in f.names("selectbox") if c[2]["key"] == "viewmax_model_type"]
        self.assertTrue(sel)
        up = FakeUpload()
        up.name = "host.glb"
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_avatar_on": True, "viewmax_avatar": up,
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
        f = run_app(FakeSt(inputs))
        self.assertEqual({c[2]["key"] for c in f.names("number_input")}, {"viewmax_x", "viewmax_y"})
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt(inputs, clicked=["viewmax_run"]))
        self.assertEqual(rv.call_args[1]["avatar_xy"], (120, 800))

    def test_x_y_hidden_for_corner_presets(self):
        f = run_app(FakeSt({"viewmax_avatar_on": True, "viewmax_position": "Bottom-Left"}))
        self.assertEqual(f.names("number_input"), [])

    def test_layout_preview_is_shown_only_when_the_avatar_is_on(self):
        self.assertFalse(any("Layout preview" in str(c) for c in run_app(FakeSt()).names("image")))
        f = run_app(FakeSt({"viewmax_avatar_on": True}))
        self.assertTrue(any("Layout preview" in str(c) for c in f.names("image")))

    def test_3d_result_tiles(self):
        res = {**VIEWMAX, "avatar": {"mode": "3d", "layout": {"preset": "Top-Right"}, "gestures": [{"name": "wave"}]}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        for needle in ("Avatar position", "Top-Right", "Gestures played"):
            self.assertIn(needle, text)

    def test_legacy_card_layout_string_does_not_crash_the_tiles(self):
        res = {**VIEWMAX, "avatar": {"mode": "mouth-warp", "layout": "full"}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        self.assertIn("Mouth sync", text)
        self.assertNotIn("Avatar position", text)

    def test_avatar_result_badge(self):
        res = {**VIEWMAX, "avatar": {"mode": "pulse"}}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        self.assertTrue(any("Avatar animation" in str(c) for c in f.names("markdown")))

    def test_script_actions_and_ocr_options_are_forwarded(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_script": "My own words.", "viewmax_actions": "Zoom in then pan right",
                            "viewmax_read_text": True, "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        kw = rv.call_args[1]
        self.assertEqual((kw["custom_script"], kw["action_prompt"], kw["read_background_text"]),
                         ("My own words.", "Zoom in then pan right", True))

    def test_custom_script_alone_is_enough_without_a_prompt(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            f = run_app(FakeSt({"viewmax_script": "Just my script."}, clicked=["viewmax_run"]))
        rv.assert_called_once()
        self.assertFalse([c for c in f.names("warning") if "prompt" in c[1][0].lower()])

    def test_unused_options_are_not_sent(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=VIEWMAX) as rv:
            run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        for k in ("custom_script", "action_prompt", "read_background_text"):
            self.assertNotIn(k, rv.call_args[1])

    def test_lab_uploads_are_deleted_after_the_run(self):
        paths = []

        def fake(prompt, folder, background_video=None, **k):
            paths.append(background_video)
            self.assertTrue(Path(background_video).exists())
            return VIEWMAX

        with mock.patch.object(pipeline, "run_viewmax_pipeline", side_effect=fake):
            run_app(FakeSt({"viewmax_prompt": "p", "viewmax_bg": FakeUpload()}, clicked=["viewmax_run"]))
        self.assertFalse(Path(paths[0]).exists())

    def test_shot_list_and_screen_text_are_shown(self):
        res = {**VIEWMAX, "screen_text": ["SALE 50%"],
               "script": {**VIEWMAX["script"], "shot_list": ["0.0s  zoom in: Zoom in"]}, "cues_applied": 1}
        with mock.patch.object(pipeline, "run_viewmax_pipeline", return_value=res):
            f = run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        text = " ".join(str(c) for c in f.names("markdown"))
        self.assertIn("SALE 50%", text)
        self.assertIn("Zoom in", text)
        self.assertIn("Visual actions", text)

    def test_ocr_checkbox_is_forwarded_and_screen_text_chips_shown(self):
        clip = {**CLIP, "screen_text": ["BIG SALE"]}
        res = {**VUGOLA, "clips": [clip]}
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=res) as rv:
            f = run_app(FakeSt({"vugola_url": "https://youtu.be/x", "vugola_ocr": True}, clicked=["vugola_run"]))
        self.assertTrue(rv.call_args[1]["read_screen_text"])
        self.assertIn("BIG SALE", " ".join(str(c) for c in f.names("markdown")))

    def test_empty_prompt_warns(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline") as rv:
            f = run_app(FakeSt(clicked=["viewmax_run"]))
        rv.assert_not_called()
        self.assertTrue(any("prompt" in c[1][0].lower() for c in f.names("warning")))

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_viewmax_pipeline", side_effect=RuntimeError("tts down")):
            f = run_app(FakeSt({"viewmax_prompt": "p"}, clicked=["viewmax_run"]))
        self.assertTrue(any("tts down" in c[1][0] for c in f.names("error")))


if __name__ == "__main__":
    unittest.main()
