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
import music_finder  # noqa: E402
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

    def radio(self, label, options, index=0, key=None, **k):
        self._log("radio", label, key=key, options=list(options))
        return self.inputs.get(key, list(options)[index])

    def button(self, label, key=None, **k):
        self._log("button", label, key=key, disabled=k.get("disabled"))
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


SOCIAL = {
    "keywords": ["protein"], "hook": "Eat more protein.", "source": "offline", "warnings": [],
    "youtube": {"title": "Eat more protein", "description": "Eat more protein.\n\n#Shorts", "tags": ["protein", "Shorts"],
                "hashtags": ["#Shorts"], "copy_text": "x"},
    "instagram": {"hook": "h", "caption": "c", "cta": "Save this", "hashtags": ["#a"] * 15, "copy_text": "IG TEXT"},
    "facebook": {"caption": "fb", "hashtags": ["#a"] * 5, "copy_text": "FB TEXT"},
}
CLIP = {"start": 1.0, "end": 20.0, "duration": 19.0, "score": 9.5, "reasons": ["question"], "faces_found": 10,
        "text": "Did you know?", "output_path": "/tmp/c1.mp4", "drive_link": "https://drive/c1", "error": None,
        "subtitled": True, "music_mixed": True, "social": SOCIAL, "subtitle_files": {"srt": "/tmp/c1.srt", "ass": "/tmp/c1.ass"}}
VUGOLA = {"source_path": "/tmp/s.mp4", "word_count": 120, "clips": [CLIP], "warnings": [], "music": None}
TRACKS = [{"id": "a1", "title": "Lofi beat", "channel": "Chill", "url": "https://youtu.be/a1", "view_count": 2_500_000,
           "duration": 150, "upload_date": "2026-11-01"},
          {"id": "b2", "title": "Hype", "channel": "Gym", "url": "https://youtu.be/b2", "view_count": 90_000,
           "duration": 95, "upload_date": "2026-10-20"}]


def clip_keys(f):
    return {c[2].get("key") for n in ("text_input", "checkbox", "button", "file_uploader", "selectbox", "slider", "radio")
            for c in f.names(n) if c[2].get("key")}


class StructureTests(unittest.TestCase):
    def test_page_is_the_viral_clip_studio_only(self):
        f = run_app(FakeSt())
        keys = clip_keys(f)
        for need in ("vugola_url", "vugola_file", "subs_on", "subs_size", "subs_color", "music_on", "music_scan",
                     "music_volume", "drive_on", "vugola_run", "drive_folder_id", "vugola_meme_on"):
            self.assertIn(need, keys)
        self.assertFalse({k for k in keys if k.startswith(("story_", "viewmax_", "wiz_", "lab_"))}, keys)
        self.assertEqual(f.names("set_page_config")[0][2]["layout"], "wide")
        self.assertIsNone(f.tab_labels)

    def test_toggle_labels_and_defaults(self):
        f = run_app(FakeSt())
        boxes = {c[2]["key"]: c for c in f.names("checkbox")}
        self.assertEqual(boxes["subs_on"][1][0], "Burn-in Subtitles")
        self.assertEqual(boxes["music_on"][1][0], "Add Trending YouTube Background Music")
        self.assertEqual(boxes["drive_on"][1][0], "Auto-Upload to Google Drive")
        self.assertEqual(boxes["music_on"][2].get("value", False) if "value" in boxes["music_on"][2] else False, False)
        buttons = [c[1][0] for c in f.names("button")]
        self.assertTrue(any("Scan Trending Tracks" in b for b in buttons))
        sizes = [c for c in f.names("selectbox") if c[2]["key"] == "subs_size"][0][2]["options"]
        self.assertEqual(sizes, list(pipeline.subtitle_engine.SIZE_PRESETS))

    def test_no_widget_key_is_used_twice(self):
        for inputs in ({}, {"music_on": True}, {"drive_folder_id": "F"}):
            f = run_app(FakeSt(inputs))
            ks = [c[2].get("key") for c in f.calls if c[2].get("key")]
            self.assertEqual(len(ks), len(set(ks)), inputs)

    def test_no_removed_feature_is_mentioned_in_the_source(self):
        text = APP.read_text(encoding="utf-8").lower()
        for word in ("avatar", "idea2video", "novel2video", "script2video", "episode", "lip-sync", "run_viewmax", "podcast"):
            self.assertNotIn(word, text)

    def test_token_status_in_sidebar(self):
        with tempfile.NamedTemporaryFile() as tok, mock.patch.object(uploader, "TOKEN_FILE", Path(tok.name)):
            f = run_app(FakeSt())
        self.assertTrue(any("connected" in c[1][0] for c in f.names("success")))
        with mock.patch.object(uploader, "TOKEN_FILE", Path("/no/token.json")):
            f = run_app(FakeSt())
        self.assertTrue(any("not connected" in c[1][0] for c in f.names("warning")))

    def test_hint_when_folder_empty(self):
        f = run_app(FakeSt({"drive_on": True}))
        self.assertTrue(any("Folder ID" in c[1][0] for c in f.names("info")))
        f = run_app(FakeSt(inputs={"drive_folder_id": "ABC", "drive_on": True}))
        self.assertFalse(any("Folder ID" in c[1][0] for c in f.names("info")))


class RunTests(unittest.TestCase):
    def test_url_run_calls_pipeline_with_the_new_options_and_shows_results(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv, \
                mock.patch("os.path.isfile", side_effect=lambda p: str(p) == "/tmp/c1.mp4"), \
                mock.patch("builtins.open", mock.mock_open(read_data=b"mp4")):
            f = run_app(FakeSt({"vugola_url": " https://youtu.be/x ", "drive_folder_id": " FOLDER ",
                                "drive_on": True, "social_topics": "fitness, protein ,"}, clicked=["vugola_run"]))
        args, kw = rv.call_args
        self.assertEqual(args, ("https://youtu.be/x", "FOLDER"))
        self.assertEqual((kw["top_n"], kw["min_len"], kw["max_len"]), (3, 15, 45))
        self.assertEqual(kw["subtitles"], {"enabled": True, "size": "Medium", "color": "White", "uppercase": False})
        self.assertIsNone(kw["music"])
        self.assertTrue(kw["upload"])
        self.assertEqual(kw["social"]["topic_keywords"], ["fitness", "protein"])
        self.assertIs(f.session_state["vugola_result"], VUGOLA)
        self.assertEqual(f.names("video")[0][1][0], "/tmp/c1.mp4")
        self.assertEqual(f.names("download_button")[0][2]["file_name"], "c1.mp4")
        self.assertEqual(f.names("link_button")[0][1], ("Open in Google Drive", "https://drive/c1"))
        self.assertTrue(f.bar.updates)

    def test_toggles_off_send_none_and_no_upload(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(FakeSt({"vugola_url": "u", "subs_on": False, "drive_on": False}, clicked=["vugola_run"]))
        kw = rv.call_args.kwargs
        self.assertIsNone(kw["subtitles"])
        self.assertIsNone(kw["music"])
        self.assertFalse(kw["upload"])

    def test_subtitle_style_choices_are_forwarded(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(FakeSt({"vugola_url": "u", "subs_size": "Large", "subs_color": "Yellow", "subs_upper": True},
                           clicked=["vugola_run"]))
        self.assertEqual(rv.call_args.kwargs["subtitles"], {"enabled": True, "size": "Large", "color": "Yellow", "uppercase": True})

    def test_uploaded_file_is_saved_and_preferred_over_url(self):
        seen = {}

        def fake_pipeline(source, *a, **k):
            seen["source"], seen["bytes"] = source, Path(source).read_bytes()
            return VUGOLA

        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=fake_pipeline):
            run_app(FakeSt({"vugola_file": FakeUpload(), "vugola_url": "https://youtu.be/x"}, clicked=["vugola_run"]))
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
            f = run_app(FakeSt({"vugola_url": "u", "vugola_min_len": 50, "vugola_max_len": 20}, clicked=["vugola_run"]))
        rv.assert_not_called()
        self.assertTrue(f.names("error"))

    def test_pipeline_error_is_shown(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=RuntimeError("no speech")):
            f = run_app(FakeSt({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("no speech" in c[1][0] for c in f.names("error")))
        self.assertNotIn("vugola_result", f.session_state)

    def test_clip_error_warnings_and_unavailable_preview(self):
        bad = {**VUGOLA, "warnings": ["Background music was skipped: offline"],
               "clips": [{**CLIP, "drive_link": None, "error": "drive down", "output_path": None, "subtitled": False,
                          "music_mixed": False, "social": None, "subtitle_files": None}]}
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=bad):
            f = run_app(FakeSt({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertTrue(any("music was skipped" in c[1][0] for c in f.names("warning")))
        self.assertTrue(any("drive down" in c[1][0] for c in f.names("error")))
        self.assertFalse(f.names("link_button"))
        self.assertFalse(f.names("video"))
        self.assertTrue(any("No captions" in c[1][0] for c in f.names("caption")))


class ResultCardTests(unittest.TestCase):
    def render(self, clip=None, **extra):
        res = {**VUGOLA, **extra, "clips": [{**CLIP, **(clip or {})}]}
        f = FakeSt()
        f.session_state["vugola_result"] = res
        return run_app(f)

    def test_copyable_caption_tabs_for_every_platform(self):
        f = self.render()
        self.assertEqual(f.tab_labels, ["▶ YouTube Shorts", "📸 Instagram Reels", "👍 Facebook Reels"])
        codes = [c[1][0] for c in f.names("code")]
        for expected in ("Eat more protein", "IG TEXT", "FB TEXT", "protein, Shorts"):
            self.assertIn(expected, codes)

    def test_status_badges_and_subtitle_files(self):
        f = self.render()
        html_ = " ".join(str(c[1][0]) for c in f.names("markdown"))
        self.assertIn("subtitled", html_)
        self.assertIn("♪ music", html_)
        self.assertIn("backed up", html_)
        self.assertTrue(any("c1.srt" in c[1][0] for c in f.names("caption")))
        f = self.render({"subtitled": False, "music_mixed": False, "drive_link": None})
        html_ = " ".join(str(c[1][0]) for c in f.names("markdown"))
        self.assertIn("no subtitles", html_)
        self.assertIn("local only", html_)
        self.assertNotIn("♪ music", html_)

    def test_music_credit_and_licence_note(self):
        f = self.render(music={"title": "Lofi beat", "channel": "Chill", "url": "https://youtu.be/a1", "licence_note": "CHECK LICENCE"})
        info = " ".join(c[1][0] for c in f.names("info"))
        self.assertIn("Lofi beat", info)
        self.assertIn("CHECK LICENCE", info)

    def test_demo_mode_shows_sample_cards(self):
        f = FakeSt()
        f.query_params = {"demo": "1"}
        f = run_app(f)
        self.assertEqual(len(f.names("code")) >= 3, True)


class MusicPanelTests(unittest.TestCase):
    def test_scan_button_lists_tracks_and_stores_them(self):
        with mock.patch.object(music_finder, "scan_trending_music", return_value=TRACKS) as scan:
            f = run_app(FakeSt({"music_on": True, "music_query": "lofi chill"}, clicked=["music_scan"]))
        scan.assert_called_once_with("lofi chill", limit=5)
        self.assertEqual(f.session_state["music_tracks"], TRACKS)
        radio = f.names("radio")[0]
        self.assertEqual(radio[2]["key"], "music_pick")
        self.assertEqual(len(radio[2]["options"]), 2)

    def test_scan_is_disabled_until_music_is_switched_on(self):
        f = run_app(FakeSt())
        scan = [c for c in f.names("button") if c[2]["key"] == "music_scan"][0]
        self.assertEqual(scan[2].get("disabled"), True)

    def test_scan_failure_is_shown_as_a_warning(self):
        with mock.patch.object(music_finder, "scan_trending_music", side_effect=music_finder.MusicError("Could not reach YouTube")):
            f = run_app(FakeSt({"music_on": True}, clicked=["music_scan"]))
        self.assertTrue(any("Could not reach YouTube" in c[1][0] for c in f.names("warning")))
        self.assertNotIn("music_tracks", f.session_state)

    def test_chosen_track_volume_and_ducking_reach_the_pipeline(self):
        f = FakeSt({"vugola_url": "u", "music_on": True, "music_volume": 25, "music_duck": False, "music_pick": 1},
                   clicked=["vugola_run"])
        f.session_state["music_tracks"] = TRACKS
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(f)
        cfg = rv.call_args.kwargs["music"]
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["volume"], 0.25)
        self.assertFalse(cfg["duck"])
        self.assertEqual(cfg["track"]["id"], "b2")
        self.assertTrue(cfg["cache_dir"].replace("\\", "/").endswith("storage/music_cache"))

    def test_own_music_file_wins_over_the_scan(self):
        seen = {}

        def fake_pipeline(source, *a, **k):
            m = k["music"]
            seen.update(path=m["path"], data=Path(m["path"]).read_bytes(), has_track="track" in m)
            return VUGOLA

        f = FakeSt({"vugola_url": "u", "music_on": True, "music_file": FakeUpload("bgm.mp3", b"audio")}, clicked=["vugola_run"])
        f.session_state["music_tracks"] = TRACKS
        with mock.patch.object(pipeline, "run_vugola_pipeline", side_effect=fake_pipeline):
            run_app(f)
        self.assertEqual(seen["data"], b"audio")
        self.assertFalse(seen["has_track"])
        self.assertFalse(Path(seen["path"]).exists())

    def test_track_label_is_readable(self):
        import runpy as _r
        with mock.patch.dict(sys.modules, {"streamlit": FakeSt()}):
            ns = _r.run_path(str(APP), run_name="not_main")
        label = ns["_track_label"](TRACKS[0])
        self.assertIn("2.5M views", label)
        self.assertIn("2:30", label)
        self.assertIn("Lofi beat", label)


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

    def test_toggle_is_present_and_off_by_default(self):
        f = run_app(FakeSt())
        boxes = [c for c in f.names("checkbox") if c[2]["key"] == "vugola_meme_on"]
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][1][0], "Enable Contextual Meme Reactions")
        self.assertEqual(len([c for c in f.names("file_uploader") if c[2]["key"] == "vugola_meme_up"]), 1)
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(FakeSt({"vugola_url": "u"}, clicked=["vugola_run"]))
        self.assertIsNone(rv.call_args.kwargs["memes"])

    def test_clip_run_forwards_meme_settings(self):
        with mock.patch.object(pipeline, "run_vugola_pipeline", return_value=VUGOLA) as rv:
            run_app(FakeSt({"vugola_url": "https://youtu.be/x", "vugola_meme_on": True, "vugola_meme_mode": "Corner tile only",
                            "vugola_meme_rate": 6, "vugola_meme_pos": "Custom Coordinates", "vugola_meme_x": 11,
                            "vugola_meme_y": 22}, clicked=["vugola_run"]))
        cfg = rv.call_args.kwargs["memes"]
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["mode"], "overlay")
        self.assertEqual(cfg["max_per_minute"], 6.0)
        self.assertEqual(cfg["position"], "Custom Coordinates")
        self.assertEqual(cfg["xy"], (11, 22))

    def test_upload_goes_into_the_meme_folder(self):
        import shutil
        import cv2
        import numpy as np
        if not shutil.which("ffmpeg"):
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
        f = FakeSt({"vugola_meme_up": [Up()], "vugola_meme_emo": "laugh"}, clicked=["vugola_meme_add"])
        run_app(f)
        self.assertTrue(list(Path(self.tmp, "laugh").glob("lol*.mp4")), f.names("error"))


if __name__ == "__main__":
    unittest.main()
