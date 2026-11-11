"""Episode Studio: asset gate + prompt cards, uploads, generation backends, cut-outs, graph reference locking,
camera curves, actors, kinetic captions, scenic render (voice sync, host corner, locked references), pipeline + Drive."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from backend import asset_manager as am
from backend import dependency_graph as dg
from backend import pipeline
from backend import render_engine as R
from backend import vimax_orchestrator as vo

FFMPEG = shutil.which("ffmpeg")
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


def png(color=(120, 120, 120), size=(100, 200)):
    ok, buf = cv2.imencode(".png", np.full((size[1], size[0], 3), color, np.uint8))
    return buf.tobytes()


def plan_episode():
    return vo.ViMaxOrchestrator().run(SCRIPT, mode="script2video")


class CardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan_episode()
        cls.cards = am.build_cards(cls.plan)

    def test_gate_is_open_after_planning(self):
        self.assertTrue(self.plan["asset_gate"]["open"])
        self.assertEqual(self.plan["asset_gate"]["status"], "awaiting_assets")
        self.assertEqual(set(self.plan["asset_gate"]["required"]), {"char:Maya", "char:Bob", "loc:kitchen", "loc:street"})

    def test_character_cards_carry_traits_attire_and_role(self):
        maya = next(c for c in self.cards["characters"] if c["name"] == "Maya")
        self.assertIn("red apron", maya["attire"])
        self.assertIn("long black hair", maya["traits"]["hair"])
        self.assertIn("full-length", maya["prompt"])
        self.assertIn("red apron", maya["prompt"])
        self.assertTrue(maya["emotional_role"])
        self.assertEqual(maya["status"], "missing")

    def test_location_cards_have_lighting_atmosphere_and_no_people(self):
        kitchen = next(c for c in self.cards["locations"] if c["key"] == "kitchen")
        street = next(c for c in self.cards["locations"] if c["key"] == "street")
        self.assertIn("night", kitchen["lighting"].lower())
        self.assertIn("daylight", street["lighting"].lower())
        for c in (kitchen, street):
            self.assertIn("no people", c["prompt"])
            self.assertNotIn("Maya", c["prompt"])
            self.assertNotIn("Bob", c["prompt"])
            self.assertTrue(c["atmosphere"])

    def test_location_key_ignores_case_and_spacing(self):
        self.assertEqual(am.location_key("  Kitchen "), am.location_key("KITCHEN"))
        self.assertEqual(am.location_key(None), "unknown location")


class BookAndUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.book = am.AssetBook(self.tmp, am.build_cards(plan_episode()))

    def test_save_load_roundtrip_and_ready(self):
        self.assertFalse(self.book.ready())
        am.generate_images(self.book, backend="draw")
        self.book.save()
        again = am.AssetBook.load(self.tmp)
        self.assertTrue(again.ready())
        self.assertEqual(set(again.character_images()), {"Maya", "Bob"})
        self.assertEqual(set(again.location_images()), {"kitchen", "street"})
        os.remove(again.cards["char:Bob"]["image"])
        self.assertEqual(again.missing(), ["char:Bob"])

    def test_upload_accepts_png_and_jpg_and_keeps_pixels(self):
        r = am.add_upload(self.book, "loc:kitchen", "bg.png", png((10, 20, 30), (300, 500)))
        self.assertEqual(r["card"]["source"], "upload")
        img = cv2.imread(r["card"]["image"])
        self.assertEqual(img.shape[:2], (500, 300))
        ok, jpg = cv2.imencode(".jpg", np.full((200, 100, 3), 99, np.uint8))
        am.add_upload(self.book, "char:Maya", "m.JPG", jpg.tobytes())
        self.assertTrue(Path(self.book.cards["char:Maya"]["image"]).is_file())

    def test_upload_rejections(self):
        for name, data, msg in (("a.gif", b"x", "not a .png"), ("a.png", b"", "empty"), ("a.png", b"junk" * 50, "could not be read"),
                                ("a.png", png(size=(30, 30)), "too small"), ("a.png", b"0" * (am.MAX_UPLOAD_BYTES + 1), "larger than")):
            with self.assertRaises(am.AssetError) as ctx:
                am.add_upload(self.book, "char:Maya", name, data)
            self.assertIn(msg, str(ctx.exception))
        with self.assertRaises(am.AssetError):
            am.add_upload(self.book, "char:Nobody", "a.png", png())
        self.assertIn("char:Maya", self.book.missing())

    def test_big_pictures_are_scaled_down_and_wide_character_warns(self):
        r = am.add_upload(self.book, "char:Maya", "w.png", png(size=(3000, 1000)))
        self.assertLessEqual(max(cv2.imread(r["card"]["image"]).shape[:2]), am.MAX_SIDE)
        self.assertTrue(any("full-length" in w for w in r["warnings"]))

    def test_clear_and_prompt_edit(self):
        am.add_upload(self.book, "char:Maya", "m.png", png())
        self.book.clear_image("char:Maya")
        self.assertIn("char:Maya", self.book.missing())
        self.book.set_prompt("char:Maya", "  my own prompt ")
        self.assertEqual(self.book.cards["char:Maya"]["prompt"], "my own prompt")
        with self.assertRaises(am.AssetError):
            self.book.set_prompt("char:Maya", "   ")

    def test_merge_cards_for_a_later_episode_keeps_pictures(self):
        am.add_upload(self.book, "char:Maya", "m.png", png())
        new = {"characters": [{"id": "char:Zed", "kind": "character", "name": "Zed"}], "locations": []}
        new["characters"].append(dict(self.book.cards["char:Maya"], image=None))
        self.assertEqual(self.book.merge_cards(new), 1)
        self.assertTrue(self.book.cards["char:Maya"]["image"])


class CutoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.book = am.AssetBook(self.tmp, am.build_cards(plan_episode()))
        am.generate_images(self.book, ["char:Maya"], backend="draw")

    def test_transparent_png_is_kept_and_cropped(self):
        out = am.remove_background(self.book.cards["char:Maya"]["image"])
        self.assertEqual(out.shape[2], 4)
        self.assertLess(out.shape[0], 1024 + 1)
        self.assertGreater((out[..., 3] > 200).mean(), 0.15)

    def test_flat_background_is_removed_with_opencv(self):
        src = cv2.imread(self.book.cards["char:Maya"]["image"], cv2.IMREAD_UNCHANGED)
        a = src[..., 3:4] / 255.0
        flat = (src[..., :3] * a + np.full(src.shape[:2] + (3,), (205, 205, 210), np.float32) * (1 - a)).astype(np.uint8)
        path = os.path.join(self.tmp, "flat.png")
        cv2.imwrite(path, flat)
        out = am.remove_background(path, tight_crop=False)
        alpha = out[..., 3] > 128
        truth = src[..., 3] > 128
        iou = (alpha & truth).sum() / max(1, (alpha | truth).sum())
        self.assertGreater(iou, 0.80)
        self.assertLess(alpha[:20, :20].mean(), 0.01)  # the corners are background

    def test_empty_or_unreadable_pictures_raise(self):
        with self.assertRaises(am.AssetError):
            am.remove_background(os.path.join(self.tmp, "nope.png"))
        blank = np.zeros((100, 100, 4), np.uint8)
        with self.assertRaises(am.AssetError):
            am.remove_background(blank)

    def test_a_picture_with_no_separable_subject_is_used_whole(self):
        rep = {}
        out = am.remove_background(np.full((120, 80, 3), 90, np.uint8), report=rep)
        self.assertEqual(rep["method"], "whole-picture")
        self.assertTrue((out[..., 3] == 255).all())

    def test_ensure_cutout_is_cached_and_follows_a_new_upload(self):
        card = self.book.cards["char:Maya"]
        first = am.ensure_cutout(card, self.book)
        m1 = os.path.getmtime(first)
        self.assertEqual(am.ensure_cutout(card, self.book), first)
        self.assertEqual(os.path.getmtime(first), m1)
        with self.assertRaises(am.AssetGateError):
            am.ensure_cutout(self.book.cards["char:Bob"], self.book)  # no picture yet


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.book = am.AssetBook(self.tmp, am.build_cards(plan_episode()))

    def test_draw_fills_everything_and_is_deterministic(self):
        rep = am.generate_images(self.book, backend="draw")
        self.assertEqual(set(rep["generated"].values()), {"draw"})
        self.assertTrue(self.book.ready())
        a = Path(self.book.cards["loc:kitchen"]["image"]).read_bytes()
        am.generate_images(self.book, ["loc:kitchen"], backend="draw", only_missing=False)
        self.assertEqual(a, Path(self.book.cards["loc:kitchen"]["image"]).read_bytes())

    def test_online_backend_uses_the_prompt_and_marks_the_source(self):
        urls = []

        def transport(url):
            urls.append(url)
            return png((50, 60, 70), (256, 384))
        rep = am.generate_images(self.book, ["char:Maya"], backend="online", transport=transport)
        self.assertEqual(rep["generated"], {"char:Maya": "online"})
        self.assertEqual(self.book.cards["char:Maya"]["source"], "generated:online")
        self.assertIn("image.pollinations.ai", urls[0])
        self.assertIn("red%20apron", urls[0])

    def test_online_failure_falls_back_once_and_does_not_retry_every_card(self):
        calls = []

        def down(url):
            calls.append(url)
            raise OSError("no internet")
        rep = am.generate_images(self.book, backend="auto", transport=down, pipe_factory=lambda: (_ for _ in ()).throw(RuntimeError("no model")))
        self.assertTrue(self.book.ready())
        self.assertEqual(set(rep["generated"].values()), {"draw"})
        self.assertEqual(len(calls), 1)  # the first failure marks the backend as down
        self.assertTrue(any("online backend failed" in w for w in rep["warnings"]))

    def test_bad_image_from_the_endpoint_is_rejected_then_drawn(self):
        rep = am.generate_images(self.book, ["loc:street"], backend="online", transport=lambda u: b"<html>rate limited</html>")
        self.assertEqual(rep["generated"], {"loc:street": "draw"})

    def test_local_diffusers_backend(self):
        from PIL import Image

        class Pipe:
            def __call__(self, **kw):
                self.kw = kw
                return type("R", (), {"images": [Image.new("RGB", (kw["width"], kw["height"]), (30, 90, 150))]})()
        pipe = Pipe()
        rep = am.generate_images(self.book, ["loc:kitchen"], backend="diffusers", pipe_factory=lambda: pipe, frame_size=(270, 480))
        self.assertEqual(rep["generated"], {"loc:kitchen": "diffusers"})
        self.assertIn("Kitchen", pipe.kw["prompt"])
        self.assertEqual(self.book.cards["loc:kitchen"]["source"], "generated:diffusers")

    def test_unknown_backend_and_progress(self):
        with self.assertRaises(am.AssetError):
            am.generate_images(self.book, backend="midjourney")
        seen = []
        am.generate_images(self.book, backend="draw", progress=lambda f, m: seen.append(f))
        self.assertEqual(seen[-1], 1.0)

    def test_only_missing_keeps_existing_uploads(self):
        am.add_upload(self.book, "char:Maya", "m.png", png((1, 2, 3)))
        am.generate_images(self.book, backend="draw")
        self.assertEqual(self.book.cards["char:Maya"]["source"], "upload")
        self.assertEqual(self.book.cards["char:Bob"]["source"], "generated:draw")


class GraphLockTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_episode()
        self.shots = self.plan["storyboard"]["shots"]
        self.graph = dg.DependencyGraph.from_dict(self.plan["graph"])
        self.store = dg.KeyframeStore(self.graph)

    def test_every_shot_is_linked_to_its_location_reference(self):
        for s in self.shots:
            key = am.location_key(s["location"])
            self.assertTrue(any(e["src"] == f"locref:{key}" and e["role"] == "location" for e in self.graph.in_edges(f"shot:{s['idx']}")), s["idx"])
        self.assertEqual(len([n for n in self.graph.nodes if n.startswith("locref:")]), 2)

    def test_locks_are_enforced_until_all_references_are_bound(self):
        self.assertTrue(dg.lock_check(self.graph, self.store, self.shots))
        with self.assertRaises(dg.GraphError):
            dg.enforce_locks(self.graph, self.store, self.shots)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        book = am.AssetBook(d, self.plan["asset_cards"])
        am.generate_images(book, backend="draw")
        rep = dg.bind_assets(self.graph, self.store, book.character_images(), {**book.location_images(), "attic": "/x.png"})
        self.assertEqual(rep["unknown"], ["attic"])
        self.assertEqual(dg.lock_check(self.graph, self.store, self.shots), [])
        dg.enforce_locks(self.graph, self.store, self.shots)

    def test_manifest_names_exactly_the_pictures_for_a_shot(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        book = am.AssetBook(d, self.plan["asset_cards"])
        am.generate_images(book, backend="draw")
        dg.bind_assets(self.graph, self.store, book.character_images(), book.location_images())
        shot = next(s for s in self.shots if s["characters"] and s["location"].lower() == "street")
        man = dg.reference_manifest(self.graph, self.store, shot["idx"])
        self.assertEqual(man["location"], book.location_images()["street"])
        self.assertEqual(man["characters"], {c: book.character_images()[c] for c in shot["characters"]})


class MotionTests(unittest.TestCase):
    def test_camera_moves(self):
        push = R.camera_for_shot({"camera_movement": "slow push-in", "shot_type": "medium"})
        self.assertGreater(push["zoom"][1], push["zoom"][0])
        pull = R.camera_for_shot({"camera_movement": "pull out"})
        self.assertLess(pull["zoom"][1], pull["zoom"][0])
        pan = R.camera_for_shot({"camera_movement": "pan left"})
        self.assertLess(pan["pan"][1], pan["pan"][0])
        self.assertGreaterEqual(min(pan["zoom"]), 1.1)  # a pan needs spare picture to travel over
        self.assertEqual(R.camera_for_shot({"camera_movement": "static"})["zoom"][0], R.camera_for_shot({"camera_movement": "static"})["zoom"][1])
        self.assertGreater(R.camera_for_shot({"camera_movement": "handheld"})["shake"], 0)
        self.assertGreater(R.camera_for_shot({"shot_type": "close-up"})["zoom"][0], R.camera_for_shot({"shot_type": "wide"})["zoom"][0])

    def test_camera_pose_stays_in_range(self):
        cam = R.camera_for_shot({"camera_movement": "handheld pan right"})
        for p in np.linspace(-0.5, 1.5, 9):
            z, pan, tilt = R.camera_pose(cam, p, t=p * 3)
            self.assertGreaterEqual(z, 1.0)
            self.assertTrue(-1 <= pan <= 1 and -1 <= tilt <= 1)

    def test_kinetic_timing_covers_the_voice_in_order(self):
        w = R.kinetic_timing("One two, three four five. Six!", 1.0, 5.0, 4)
        self.assertEqual(len(w), 6)
        self.assertAlmostEqual(w[-1]["t1"], 5.0, places=2)
        self.assertTrue(all(a["t1"] <= b["t0"] + 1e-9 for a, b in zip(w, w[1:])))
        self.assertEqual(w[0]["chunk"], 0)
        self.assertGreater(w[-1]["chunk"], w[0]["chunk"])
        self.assertEqual(R.kinetic_timing("", 0, 1), [])

    def test_kinetic_layer_highlights_the_active_word(self):
        a = R.render_kinetic_layer(["hello", "world"], 0, 270)
        b = R.render_kinetic_layer(["hello", "world"], 1, 270)
        self.assertEqual(a.shape[2], 4)
        self.assertFalse(np.array_equal(a, b))


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class ActorAndSceneRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.book = am.AssetBook(cls.tmp + "/a", am.build_cards(plan_episode()))
        am.generate_images(cls.book, backend="draw")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def frame(self, path, t, fps=12):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        cap.release()
        return f

    def test_actor_validation(self):
        c = R.Composition((180, 320), 10, duration=2)
        with self.assertRaises(FileNotFoundError):
            c.add_actor("/nope.png", 0)
        flat = os.path.join(self.tmp, "flat.png")
        cv2.imwrite(flat, np.zeros((50, 50, 3), np.uint8))
        with self.assertRaises(ValueError):
            c.add_actor(flat, 0)  # no alpha channel
        img = self.book.cards["char:Maya"]["image"]
        with self.assertRaises(ValueError):
            c.add_actor(img, 0, entry="up")
        with self.assertRaises(ValueError):
            c.add_actor(img, 0, height=3)
        with self.assertRaises(FileNotFoundError):
            c.add_scenery("/nope.png")
        with self.assertRaises(ValueError):
            c.add_scenery(self.book.cards["loc:street"]["image"], camera={"zoom": (0.5, 1), "pan": (0, 0), "tilt": (0, 0)})

    def test_actor_stays_in_frame_and_slides_in(self):
        c = R.Composition((180, 320), 10, duration=3)
        it = c.add_actor(self.book.cards["char:Maya"]["image"], 0, None, x=0.0, height=0.6, entry="left", name="Maya")
        x, y, w, h = it.data["rect"]
        self.assertGreaterEqual(x, 0)
        early = dict(c.actor_positions(0.05))["Maya"]
        late = dict(c.actor_positions(1.5))["Maya"]
        self.assertLess(early[0] + early[2], late[0] + late[2])  # still sliding in from the left edge
        self.assertEqual(late[:2], (x, y))

    def test_render_has_scenery_actor_and_caption_and_camera_motion(self):
        c = R.Composition((180, 320), 12, duration=2.5)
        c.add_scenery(self.book.cards["loc:street"]["image"], 0, None, {"zoom": (1.2, 1.2), "pan": (-0.9, 0.9), "tilt": (0, 0), "shake": 0, "ease": "linear"})
        c.add_actor(self.book.cards["char:Bob"]["image"], 0, None, x=0.5, height=0.6, speaking=[(0.2, 2.0)], name="Bob")
        c.add_kinetic_caption("hello there my friend", 0.1, 2.3, "top")
        out = os.path.join(self.tmp, "s.mp4")
        r = c.render(out)
        self.assertEqual(r["actors"][0]["name"], "Bob")
        f0, f1 = self.frame(out, 0.9), self.frame(out, 2.2)
        left, right = f0[:60, :40].astype(int), f1[:60, :40].astype(int)
        self.assertGreater(np.abs(left - right).mean(), 2.0)  # the sky corner changed: the camera panned
        bare = R.Composition((180, 320), 12, duration=2.5)
        bare.add_scenery(self.book.cards["loc:street"]["image"], 0, None, {"zoom": (1.2, 1.2), "pan": (-0.9, 0.9), "tilt": (0, 0), "shake": 0, "ease": "linear"})
        bare.render(os.path.join(self.tmp, "bare.mp4"))
        diff = np.abs(f0.astype(int) - self.frame(os.path.join(self.tmp, "bare.mp4"), 0.9).astype(int)).sum(axis=2)
        self.assertGreater((diff[int(0.4 * 320):int(0.9 * 320), 60:120] > 30).mean(), 0.3)  # a figure stands in the middle
        top = f0[10:60, 20:160]
        self.assertGreater(top.std(), 20)  # the caption pill is drawn at the top

    def scenic(self, **kw):
        plan = plan_episode()
        book = am.AssetBook(self.tmp + "/b%d" % len(os.listdir(self.tmp)), plan["asset_cards"])
        am.generate_images(book, backend="draw")
        vo.close_asset_gate(plan, book)
        out = os.path.join(self.tmp, "ep%d.mp4" % len(os.listdir(self.tmp)))
        res = vo.render_storyboard(plan, out, work_dir=out + "_w", assets=book, size=(180, 320), fps=12, **kw)
        return plan, book, res, out

    def test_scenic_render_locks_every_shot_to_its_references(self):
        plan, book, res, out = self.scenic()
        self.assertTrue(res["scenic"])
        self.assertEqual(res["locks"], [])
        refs = {(c["character"], c["reference_sha1"]) for c in res["conditioning"]}
        self.assertEqual(refs, {(n, am.file_sha1(p)) for n, p in book.character_images().items()})  # one picture per character, in every shot
        self.assertTrue(all(d["status"] == "ok" for d in res["drift"]), res["drift"])
        self.assertGreater(len(res["drift"]), 3)
        # the kitchen shots and the street shots really use different backgrounds
        shots = plan["storyboard"]["shots"]
        k = next(s for s in shots if s["location"].lower() == "kitchen")
        st = next(s for s in shots if s["location"].lower() == "street")
        def mid(idx):
            clip = next(x for x in res["shots"] if x["idx"] == idx)
            cap = cv2.VideoCapture(clip["clip"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, 8)
            return cap.read()[1]
        self.assertGreater(np.abs(mid(k["idx"])[:60].astype(int) - mid(st["idx"])[:60].astype(int)).mean(), 10)

    def test_a_closed_gate_blocks_rendering(self):
        plan = plan_episode()
        book = am.AssetBook(self.tmp + "/gate", plan["asset_cards"])
        with self.assertRaises(am.AssetGateError) as ctx:
            vo.render_storyboard(plan, os.path.join(self.tmp, "x.mp4"), assets=book, size=(90, 160))
        self.assertIn("char:Maya", ctx.exception.missing)
        with self.assertRaises(am.AssetGateError):
            vo.close_asset_gate(plan, book)
        self.assertTrue(plan["asset_gate"]["open"])

    def test_voice_sync_extends_the_shot_and_keeps_audio(self):
        def fake_tts(text, path, voice=None):
            import subprocess
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=4", path], check=True)
        with mock.patch("backend.tts_engine.generate_voiceover_sync", fake_tts):
            plan, book, res, out = self.scenic(voice=True)
        self.assertTrue(res["voice_used"])
        spoken = [s for s, x in zip(plan["storyboard"]["shots"], res["shots"]) if s.get("line")]
        self.assertTrue(all(next(x for x in res["shots"] if x["idx"] == s["idx"])["seconds"] >= 4.2 for s in spoken))
        from backend import video_engine as ve
        self.assertTrue(ve._probe_media(FFMPEG, out)["has_audio"])

    def test_host_corner_is_kept_and_characters_avoid_it(self):
        plan, book, res, out = self.scenic(host={"style": "anime", "position": "Bottom-Left", "scale": 0.3})
        self.assertEqual(res["host"]["position"], "Bottom-Left")
        # nobody is placed over the host's corner (left third, bottom): actors use the right-hand range
        clip = res["shots"][-1]["clip"]
        self.assertTrue(os.path.isfile(clip))


@unittest.skipUnless(FFMPEG, "ffmpeg needed")
class PipelineWizardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def plan(self, **kw):
        return pipeline.run_story_pipeline(SCRIPT, mode="script2video", output_dir=self.tmp, size=(180, 320), fps=8, render=False, **kw)

    def test_step1_pauses_at_the_gate(self):
        p = self.plan()
        self.assertIsNone(p["video_path"])
        self.assertTrue(p["asset_gate"]["open"])
        book = am.AssetBook.load(p["assets_dir"])
        self.assertEqual(len(book.cards), 4)
        self.assertEqual(book.missing(), sorted(book.cards, key=list(book.cards).index))

    def test_render_uses_the_users_pictures_and_syncs_assets_to_drive(self):
        p = self.plan()
        book = am.AssetBook.load(p["assets_dir"])
        am.add_upload(book, "char:Maya", "maya.png", png((20, 30, 200), (120, 300)))   # a flat red slab as "Maya"
        am.generate_images(book, backend="draw")
        book.save()
        sent = []

        def fake_upload(path, folder, **k):
            sent.append(os.path.basename(path))
            return "https://drive/" + os.path.basename(path)
        with mock.patch.object(pipeline, "drive_status", return_value="ready"), \
                mock.patch.object(pipeline, "upload_to_google_drive", fake_upload):
            r = pipeline.run_story_pipeline(SCRIPT, mode="script2video", output_dir=self.tmp, size=(180, 320), fps=8, plan_result=p,
                                            asset_book=book, folder_id="F", render=True)
        self.assertTrue(r["rendered"], r["warnings"])
        self.assertTrue(any("Maya: the background could not be removed" in w for w in r["warnings"]))
        self.assertFalse(r["asset_gate"]["open"])
        self.assertEqual(r["project_dir"], p["project_dir"])
        self.assertIn("assets.json", sent)
        self.assertIn("maya.png", sent)
        self.assertIn("kitchen.png", sent)
        self.assertEqual(sum(1 for n in sent if n.endswith(".mp4")), 1)
        self.assertGreaterEqual(len(r["asset_links"]), 5)
        self.assertEqual({c["reference_sha1"] for c in r["conditioning"] if c["character"] == "Maya"}, {am.file_sha1(book.cards["char:Maya"]["image"])})
        self.assertEqual(r["frame_size"], [180, 320])

    def test_rendering_without_pictures_generates_them_and_says_so(self):
        p = self.plan()
        r = pipeline.run_story_pipeline(SCRIPT, mode="script2video", output_dir=self.tmp, size=(180, 320), fps=8, plan_result=p, render=True)
        self.assertTrue(r["rendered"])
        self.assertTrue(any("generated automatically" in w for w in r["warnings"]))
        self.assertTrue(r["assets"]["ready"])

    def test_online_backend_is_used_when_chosen_and_falls_back_offline(self):
        p = self.plan()

        def down(url):
            raise OSError("offline")
        r = pipeline.run_story_pipeline(SCRIPT, mode="script2video", output_dir=self.tmp, size=(180, 320), fps=8, plan_result=p, render=True,
                                        asset_backend="auto", image_transport=down)
        self.assertTrue(r["rendered"])
        self.assertTrue(any("online backend failed" in w for w in r["warnings"]))


class UploaderAssetTests(unittest.TestCase):
    def test_collect_asset_files(self):
        from backend import uploader
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        book = am.AssetBook(d, plan_episode()["asset_cards"])
        self.assertEqual(uploader.collect_asset_files(book), [])
        am.generate_images(book, backend="draw")
        book.save()
        files = uploader.collect_asset_files(book)
        self.assertEqual(len(files), 5)
        self.assertTrue(files[-1].endswith("assets.json"))


if __name__ == "__main__":
    unittest.main()
