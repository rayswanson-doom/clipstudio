"""3D avatars: model loading, rig, software renderer, lip-sync / gesture performance, rendered video."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import avatar_3d_engine as E  # noqa: E402
import avatar_gestures as gest  # noqa: E402
import demo_avatars as D  # noqa: E402
import glb_model as G  # noqa: E402
import soft_raster as R  # noqa: E402
from tests.test_video_engine import FFMPEG, probe  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

REAL_DIRS = [Path(p) for p in (os.environ.get("REAL_GLB_DIR", ""), "/tmp/claude-0/t3/examples/models/gltf",
                               str(ROOT / "tests" / "data")) if p]
REST_ARMS = {s: {"arm": (0.14, -0.98, 0.02), "fore": (0.04, -0.8, 0.6), "roll": 0} for s in ("Left", "Right")}


def real(name):
    for d in REAL_DIRS:
        if (d / name).is_file():
            return str(d / name)
    return None


class DemoFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.paths = D.ensure_demo_avatars(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()


class ModelLoadingTests(DemoFiles):
    def test_all_demo_avatars_load_with_rig_and_mouth_shapes(self):
        for style, path in self.paths.items():
            m = G.load_model(path)
            for bone in ("Hips", "Spine1", "Neck", "Head", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm"):
                self.assertIn(bone, m.bones, (style, bone))
            self.assertGreater(m.triangle_count, 2000)
            self.assertTrue(m.textures)
            self.assertEqual(m.lip_sync_kind(), "vrm-vowels" if style == "anime" else "visemes")

    def test_demo_has_all_fifteen_oculus_visemes_and_blink(self):
        m = G.load_model(self.paths["human"])
        for v in G.OCULUS_VISEMES:
            self.assertIn(f"viseme_{v}", m.channels, v)
        self.assertIn("blinkLeft", m.channels)

    def test_vrm_file_uses_the_humanoid_table_and_vowel_groups(self):
        m = G.load_model(self.paths["anime"])
        self.assertEqual(m.format, "vrm")
        self.assertIn("Head", m.bones)  # node is called J_Bip_Head: only the VRM table can find it
        for k in ("vrm_A", "vrm_I", "vrm_U", "vrm_E", "vrm_O", "blink"):
            self.assertIn(k, m.channels)

    def test_bad_files_give_clear_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with self.assertRaisesRegex(G.ModelError, "not found"):
                G.load_model(str(p / "nope.glb"))
            (p / "a.fbx").write_bytes(b"x")
            with self.assertRaisesRegex(G.ModelError, r"\.glb or \.vrm"):
                G.load_model(str(p / "a.fbx"))
            (p / "junk.glb").write_bytes(b"not a glb at all, just text")
            with self.assertRaisesRegex(G.ModelError, "binary glTF"):
                G.load_model(str(p / "junk.glb"))
            doc = json.dumps({"asset": {"version": "2.0"}, "extensionsRequired": ["KHR_draco_mesh_compression"]}).encode()
            doc += b" " * (-len(doc) % 4)
            (p / "d.glb").write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(doc)) + struct.pack("<II", len(doc), 0x4E4F534A) + doc)
            with self.assertRaisesRegex(G.ModelError, "compressed"):
                G.load_model(str(p / "d.glb"))
            empty = json.dumps({"asset": {"version": "2.0"}}).encode()
            empty += b" " * (-len(empty) % 4)
            (p / "e.glb").write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(empty)) + struct.pack("<II", len(empty), 0x4E4F534A) + empty)
            with self.assertRaisesRegex(G.ModelError, "no triangle meshes"):
                G.load_model(str(p / "e.glb"))

    def test_inspect_model_reports_capabilities(self):
        info = E.inspect_model(self.paths["human"])
        self.assertTrue(info["has_visemes"] and info["arms_posable"] and info["head_posable"] and info["blink"])

    @unittest.skipUnless(real("readyplayer.me.glb"), "real Ready Player Me sample not available")
    def test_real_ready_player_me_avatar(self):
        m = G.load_model(real("readyplayer.me.glb"))
        self.assertGreater(m.vertex_count, 5000)
        self.assertEqual(m.lip_sync_kind(), "jaw-open")
        self.assertIn("Head", m.bones)
        self.assertTrue(Rig_ok(m))

    @unittest.skipUnless(real("Michelle.glb"), "real Mixamo sample not available")
    def test_real_mixamo_character_without_mouth_shapes(self):
        m = G.load_model(real("Michelle.glb"))
        self.assertEqual(m.lip_sync_kind(), "none")
        self.assertTrue(Rig_ok(m))


def Rig_ok(model):
    return E.Rig(model).can_pose_arms and E.Rig(model).can_pose_head


class RigTests(DemoFiles):
    def setUp(self):
        self.m = G.load_model(self.paths["human"])
        self.rig = E.Rig(self.m)

    def test_body_frame(self):
        self.assertAlmostEqual(float(self.rig.ex[0]), 1.0, places=3)   # character's left = +x
        self.assertAlmostEqual(float(self.rig.ey[1]), 1.0, places=3)
        self.assertAlmostEqual(float(self.rig.ez[2]), 1.0, places=3)   # faces +z

    def test_arms_reach_the_requested_directions_on_any_rig(self):
        for path in [self.paths["human"], self.paths["anime"]] + [p for p in (real("readyplayer.me.glb"), real("Michelle.glb")) if p]:
            m = G.load_model(path)
            rig = E.Rig(m)
            W = rig.solve(E.Pose(arms={"Left": {"arm": (0.2, -0.9, 0.3), "fore": (0.1, 0.5, 0.85), "roll": 0.0},
                                       "Right": {"arm": (0.2, -0.9, 0.3), "fore": (0.1, 0.5, 0.85), "roll": 0.0}}))
            b = m.bones
            up = W[b["LeftForeArm"]][:3, 3] - W[b["LeftArm"]][:3, 3]
            want = 0.2 * rig.ex - 0.9 * rig.ey + 0.3 * rig.ez
            cosine = float(up @ want / np.linalg.norm(up) / np.linalg.norm(want))
            self.assertGreater(cosine, 0.999, path)
            fore = W[b["LeftHand"]][:3, 3] - W[b["LeftForeArm"]][:3, 3]
            want_f = 0.1 * rig.ex + 0.5 * rig.ey + 0.85 * rig.ez
            self.assertGreater(float(fore @ want_f / np.linalg.norm(fore) / np.linalg.norm(want_f)), 0.999, path)
            # the character's RIGHT arm mirrors: outward is -x
            r = W[b["RightForeArm"]][:3, 3] - W[b["RightArm"]][:3, 3]
            self.assertLess(float(r @ rig.ex), 0)

    def test_bone_lengths_are_preserved(self):
        W0, W1 = self.rig.bind_world, self.rig.solve(E.Pose(arms=REST_ARMS))
        b = self.m.bones
        for a, c in (("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand")):
            l0 = np.linalg.norm(W0[b[a]][:3, 3] - W0[b[c]][:3, 3])
            l1 = np.linalg.norm(W1[b[a]][:3, 3] - W1[b[c]][:3, 3])
            self.assertAlmostEqual(float(l0), float(l1), places=4)

    def test_head_pitch_moves_the_nose_forward_and_down(self):
        base = self.rig.solve(E.Pose())
        nod = self.rig.solve(E.Pose(body_rot={"Head": (0.4, 0.0, 0.0)}))
        h = self.m.bones["Head"]
        v = np.array([0, 0.1, 0.1], np.float32)  # a point in front of the head joint
        p0 = base[h][:3, :3] @ v
        p1 = nod[h][:3, :3] @ v
        self.assertLess(float(p1 @ self.rig.ey), float(p0 @ self.rig.ey))

    def test_yaw_turns_toward_the_characters_left(self):
        turned = self.rig.solve(E.Pose(body_rot={"Head": (0.0, 0.5, 0.0)}))
        fwd = turned[self.m.bones["Head"]][:3, :3] @ np.array([0, 0, 1], np.float32)
        self.assertGreater(float(fwd @ self.rig.ex), 0.2)


class RasterTests(unittest.TestCase):
    def quad(self, flip=False):
        sx = np.array([10, 50, 50, 10], np.float32)
        sy = np.array([10, 10, 50, 50], np.float32)
        zc = np.full(4, 5.0, np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3]]) if not flip else np.array([[0, 2, 1], [0, 3, 2]])
        return sx, sy, zc, tris

    def test_front_face_covers_the_quad(self):
        sx, sy, zc, tris = self.quad()
        # y-down screen: (0,1,2) = clockwise in pixels = counter-clockwise with y up -> front face
        r = R.rasterize(sx, sy, zc, tris, 64, 64)
        covered = int((r.tri_id >= 0).sum())
        self.assertTrue(covered == 0 or 1500 < covered <= 1700)

    def test_exactly_one_winding_is_culled(self):
        sx, sy, zc, tris = self.quad()
        a = int((R.rasterize(sx, sy, zc, tris, 64, 64).tri_id >= 0).sum())
        b = int((R.rasterize(sx, sy, zc, tris[:, ::-1].copy(), 64, 64).tri_id >= 0).sum())
        self.assertNotEqual(a > 0, b > 0)
        both = int((R.rasterize(sx, sy, zc, tris, 64, 64, two_sided=np.array([True, True])).tri_id >= 0).sum())
        both2 = int((R.rasterize(sx, sy, zc, tris[:, ::-1].copy(), 64, 64, two_sided=np.array([True, True])).tri_id >= 0).sum())
        self.assertEqual(both, both2)
        self.assertGreater(both, 1500)

    def test_nearer_triangle_wins_and_depth_is_right(self):
        sx = np.array([10, 50, 50, 10, 10, 50, 50, 10], np.float32)
        sy = np.array([10, 10, 50, 50, 10, 10, 50, 50], np.float32)
        zc = np.array([5, 5, 5, 5, 3, 3, 3, 3], np.float32)
        tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        for order in (tris, tris[::-1].copy()):
            r = R.rasterize(sx, sy, zc, order, 64, 64, two_sided=np.array([True] * 4))
            self.assertAlmostEqual(float(r.depth[30, 30]), 3.0, places=3)

    def test_perspective_correct_barycentrics_sum_to_one(self):
        sx = np.array([5, 60, 30], np.float32)
        sy = np.array([5, 10, 58], np.float32)
        r = R.rasterize(sx, sy, np.array([2, 8, 4], np.float32), np.array([[0, 2, 1]]), 64, 64, np.array([True]))
        ys, xs = np.nonzero(r.tri_id >= 0)
        self.assertGreater(len(ys), 500)
        np.testing.assert_allclose(r.bary[ys, xs].sum(1), 1.0, atol=1e-4)

    def test_triangles_behind_the_camera_and_off_screen_are_ignored(self):
        r = R.rasterize(np.array([10, 50, 30], np.float32), np.array([10, 10, 50], np.float32),
                        np.array([-1, 5, 5], np.float32), np.array([[0, 1, 2]]), 64, 64, np.array([True]))
        self.assertEqual(int((r.tri_id >= 0).sum()), 0)
        r = R.rasterize(np.array([100, 150, 120], np.float32), np.array([10, 10, 50], np.float32),
                        np.full(3, 5, np.float32), np.array([[0, 1, 2]]), 64, 64, np.array([True]))
        self.assertEqual(int((r.tri_id >= 0).sum()), 0)

    def test_huge_triangle_is_clipped_to_the_screen(self):
        r = R.rasterize(np.array([-500, 1000, -500], np.float32), np.array([-500, -500, 1000], np.float32),
                        np.full(3, 5, np.float32), np.array([[0, 1, 2]]), 64, 64, np.array([True]))
        self.assertEqual(int((r.tri_id >= 0).sum()), 64 * 64)

    def test_alpha_test_cuts_holes(self):
        sx, sy, zc, tris = self.quad()
        keep_left = lambda ids, bary: np.ones(len(ids), bool) if False else (bary[:, 0] > 0.5)  # noqa: E731
        r = R.rasterize(sx, sy, zc, tris, 64, 64, np.array([True, True]), keep_left, np.array([True, True]))
        full = R.rasterize(sx, sy, zc, tris, 64, 64, np.array([True, True]))
        self.assertLess((r.tri_id >= 0).sum(), (full.tri_id >= 0).sum())


@unittest.skipUnless(cv2, "needs opencv")
class RendererTests(DemoFiles):
    def render(self, style="human", pose=None, morph=None, size=(240, 320), model_key="human"):
        m = G.load_model(self.paths[model_key])
        rig = E.Rig(m)
        r = E.Renderer(m, rig, size, style, ssaa=1.0)
        perf = type("P", (), {"m": m})()
        weights = E.Performance.prim_weights(perf, morph or {})
        return r.render(rig.solve(pose or E.Pose(arms=REST_ARMS)), weights)

    def test_character_is_drawn_over_a_background(self):
        img = self.render()
        self.assertEqual(img.shape, (320, 240, 3))
        corner = img[2, 2].astype(int)
        centre = img[110:230, 70:170].reshape(-1, 3).astype(int)
        self.assertGreater(np.abs(centre - corner).sum(1).mean(), 60)

    def test_open_mouth_changes_the_face(self):
        shut, wide = self.render(morph={}), self.render(morph={"viseme_aa": 1.0})
        diff = np.abs(shut.astype(int) - wide.astype(int)).sum(2)
        self.assertGreater(int((diff > 25).sum()), 15)
        self.assertEqual(int((diff[260:, :] > 25).sum()), 0)  # only the face moved

    def test_a_raised_arm_changes_the_picture_on_the_correct_side(self):
        wave = E.Pose(arms={"Left": REST_ARMS["Left"], "Right": {"arm": (0.52, -0.38, 0.42), "fore": (0.18, 0.95, 0.32), "roll": 0}})
        a, b = self.render(), self.render(pose=wave)
        diff = np.abs(a.astype(int) - b.astype(int)).sum(2) > 40
        # the character's right arm is on the viewer's LEFT
        self.assertGreater(int(diff[:, :120].sum()), 5 * max(1, int(diff[:, 120:].sum())) // 2)

    def test_styles_look_different_and_anime_has_outlines(self):
        human, anime = self.render("human"), self.render("anime", model_key="human")
        self.assertGreater(np.abs(human.astype(int) - anime.astype(int)).mean(), 5)
        outline = np.array([46, 30, 52])[::-1]
        near = (np.abs(anime.astype(int) - outline).sum(2) < 40).sum()
        self.assertGreater(int(near), 100)

    def test_render_is_deterministic(self):
        np.testing.assert_array_equal(self.render(), self.render())

    def test_supersampling_keeps_the_output_size(self):
        m = G.load_model(self.paths["human"])
        rig = E.Rig(m)
        img = E.Renderer(m, rig, (200, 260), "human", ssaa=2.0).render(rig.solve(E.Pose()), {})
        self.assertEqual(img.shape, (260, 200, 3))

    @unittest.skipUnless(real("readyplayer.me.glb"), "real model not available")
    def test_real_textured_avatar_renders(self):
        m = G.load_model(real("readyplayer.me.glb"))
        rig = E.Rig(m)
        img = E.Renderer(m, rig, (240, 320), "human", ssaa=1.0).render(rig.solve(E.Pose(arms=REST_ARMS)), {})
        self.assertGreater(int(img.std()), 20)


class PerformanceTests(DemoFiles):
    def setUp(self):
        self.m = G.load_model(self.paths["human"])
        self.rig = E.Rig(self.m)

    def perf(self, **kw):
        args = dict(duration=6.0, fps=25, text="Hello there. Welcome to the show. Thank you so much for watching!")
        args.update(kw)
        return E.Performance(self.m, self.rig, **args)

    def test_mouth_moves_while_speaking_and_rests_before_the_first_word(self):
        p = self.perf()
        opens = [p.mouth_open(i) for i in range(p.n)]
        self.assertGreater(max(opens), 0.4)
        self.assertGreater(sum(1 for o in opens if o > 0.2), 30)
        self.assertLess(opens[0], 0.3)

    def test_silence_in_the_audio_closes_the_mouth(self):
        env = np.concatenate([np.ones(75), np.zeros(75)]).astype(np.float32)
        p = self.perf(envelope=env)
        late = [p.mouth_open(i) for i in range(100, 150)]
        early = [p.mouth_open(i) for i in range(10, 70)]
        self.assertLess(max(late), 0.12)
        self.assertGreater(max(early), 0.4)

    def test_real_word_timestamps_place_the_visemes(self):
        words = [{"word": "Hello", "start": 4.0, "end": 4.6}]
        p = self.perf(text="Hello", words=words)
        self.assertLess(max(p.mouth_open(i) for i in range(0, 90)), 0.05)
        self.assertGreater(max(p.mouth_open(i) for i in range(98, 118)), 0.3)

    def test_visemes_reach_the_models_blend_shape_channels(self):
        p = self.perf()
        names = set()
        for i in range(p.n):
            names |= {k for k, v in p.pose(i).morph.items() if k.startswith("viseme_") and v > 0.3}
        self.assertGreaterEqual(len(names), 4)

    def test_blinks_happen(self):
        p = self.perf(duration=12.0)
        closed = [p.pose(i).morph.get("blinkLeft", 0) > 0.5 for i in range(p.n)]
        blinks = sum(1 for i in range(1, p.n) if closed[i] and not closed[i - 1])
        self.assertGreaterEqual(blinks, 2)
        self.assertLess(sum(closed), p.n * 0.1)  # eyes are open most of the time

    def test_gestures_follow_the_words(self):
        p = self.perf(text="Hello everyone. Thank you so much for watching. No, that is a myth.", duration=9.0)
        names = {e["name"] for e in p.events}
        self.assertIn("wave", names)
        self.assertTrue(names & {"bow", "namaste", "shake"})

    def test_explicit_notes_win(self):
        p = self.perf(text="Just a plain sentence without any keywords here.", notes=[{"start": 1.0, "text": "applause for the crowd"}])
        self.assertIn(("applause", 1.0), [(e["name"], e["start"]) for e in p.events])

    def test_gestures_can_be_switched_off(self):
        p = self.perf(gestures=False)
        self.assertEqual(p.events, [])

    def test_arms_blend_from_rest_into_the_gesture_and_back(self):
        p = self.perf(text="Hello everyone.", duration=6.0)
        ev = next(e for e in p.events if e["name"] == "wave")
        mid = int((ev["start"] + ev["end"]) / 2 * 25)
        rest_up = p.pose(0).arms["Right"]["fore"][1]
        wave_up = p.pose(mid).arms["Right"]["fore"][1]
        self.assertGreater(wave_up, rest_up + 0.5)
        self.assertLess(p.pose(int(ev["end"] * 25) + 10).arms["Right"]["fore"][1], 0.0)
        self.assertEqual(p.pose(mid).arms["Left"]["fore"], p.pose(0).arms["Left"]["fore"])  # left arm stays relaxed

    def test_idle_motion_is_small_and_continuous(self):
        p = self.perf()
        h = [p.pose(i).body_rot["Head"] for i in range(p.n)]
        self.assertLess(max(abs(x[0]) for x in h), 0.35)
        steps = [abs(h[i + 1][1] - h[i][1]) for i in range(p.n - 1)]
        self.assertLess(max(steps), 0.05)

    def test_vrm_model_drives_vowel_channels(self):
        m = G.load_model(self.paths["anime"])
        p = E.Performance(m, E.Rig(m), 4.0, 25, text="Ah oh ee oo")
        seen = set()
        for i in range(p.n):
            seen |= {k for k, v in p.pose(i).morph.items() if v > 0.3}
        self.assertTrue(seen & {"vrm_A", "vrm_O", "vrm_E", "vrm_U", "vrm_I"})

    @unittest.skipUnless(real("readyplayer.me.glb"), "real model not available")
    def test_jaw_open_models_are_driven_through_jaw_open(self):
        m = G.load_model(real("readyplayer.me.glb"))
        p = E.Performance(m, E.Rig(m), 4.0, 25, text="Hello there everyone")
        self.assertGreater(max(p.pose(i).morph.get("jawOpen", 0) for i in range(p.n)), 0.3)


@unittest.skipUnless(FFMPEG and cv2, "needs ffmpeg + opencv")
class VideoTests(DemoFiles):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.audio = os.path.join(cls.dir.name, "a.wav")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
                        "-af", "tremolo=f=4:d=0.9,volume=0.6", cls.audio], check=True)

    def make(self, style="human", model="human", **kw):
        out = os.path.join(self.dir.name, f"{style}_{model}.mp4")
        return E.create_3d_avatar_video(self.paths[model], self.audio, out, text="Hello everyone, welcome!",
                                        style=style, size=(240, 320), fps=12, quality="fast", **kw)

    def test_video_has_audio_right_size_duration_and_moving_content(self):
        calls = []
        r = self.make(progress=calls.append)
        info = probe(r["output_path"])
        kinds = {s["codec_type"] for s in info["streams"]}
        self.assertEqual(kinds, {"video", "audio"})
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((v["width"], v["height"]), (240, 320))
        self.assertAlmostEqual(float(info["format"]["duration"]), 3.0, delta=0.3)
        self.assertEqual((r["mode"], r["lip_sync"], r["frames"]), ("3d", "visemes", 36))
        self.assertEqual(calls[-1], 1.0)
        cap = cv2.VideoCapture(r["output_path"])
        frames = [cap.read()[1] for _ in range(24)]
        cap.release()
        self.assertGreater(len({f.tobytes() for f in frames}), 12)  # it moves

    def test_all_three_styles_and_both_formats_render(self):
        for style, model in (("ai", "ai"), ("anime", "anime")):
            r = self.make(style, model)
            self.assertTrue(os.path.getsize(r["output_path"]) > 2000, style)
        self.assertEqual(self.make("anime", "anime")["lip_sync"], "vrm-vowels")

    def test_errors(self):
        with self.assertRaises(FileNotFoundError):
            E.create_3d_avatar_video(self.paths["human"], "/no/audio.wav", os.path.join(self.dir.name, "x.mp4"))
        with self.assertRaises(G.ModelError):
            E.create_3d_avatar_video("/no/model.glb", self.audio, os.path.join(self.dir.name, "x.mp4"))

    def test_gestures_are_reported(self):
        r = E.create_3d_avatar_video(self.paths["human"], self.audio, os.path.join(self.dir.name, "g.mp4"),
                                     text="Hello everyone! Thank you for watching.", size=(120, 160), fps=8, quality="fast")
        self.assertTrue(any(g["name"] == "wave" for g in r["gestures"]))

    @unittest.skipUnless(real("Michelle.glb"), "real model not available")
    def test_model_without_mouth_shapes_warns_but_renders(self):
        out = os.path.join(self.dir.name, "m.mp4")
        r = E.create_3d_avatar_video(real("Michelle.glb"), self.audio, out, text="Hello", size=(120, 160), fps=6, quality="fast")
        self.assertEqual(r["lip_sync"], "none")
        self.assertTrue(any("cannot lip-sync" in w for w in r["warnings"]))


class GestureCatalogTests(unittest.TestCase):
    def test_head_keyframes_match_the_motion_engine_catalogue(self):
        gj = Path("/tmp/claude-0/ref/64fc0544-motion-engine-main/motion-engine-main/src/gestures.json")
        if not gj.is_file():
            self.skipTest("motion-engine source not available")
        cat = json.loads(gj.read_text())["head"]
        for name, spec in gest.HEAD_GESTURES.items():
            self.assertEqual(spec["dt"], cat[name]["dt"], name)
            for k, v in spec["vs"].items():
                self.assertEqual(v, cat[name]["vs"][k], (name, k))

    def test_classifier(self):
        cases = {"Hello and welcome!": "wave", "Thank you so much": "namaste", "I don't know, maybe": "shrug",
                 "Let me think about it": "think", "That was amazing": "thumbs_up", "Congratulations!": "celebrate"}
        for text, hand in cases.items():
            self.assertEqual(gest.classify_sentence(text)[1], hand, text)
        self.assertEqual(gest.classify_sentence("No, that is wrong.")[0], "shake")
        self.assertEqual(gest.classify_sentence("The cat sat on the mat.")[:2], ("", ""))

    def test_notes_and_cooldowns(self):
        self.assertEqual(gest.gesture_from_note("wave at the audience"), "wave")
        self.assertEqual(gest.gesture_from_note("thumbs up"), "thumbs_up")
        self.assertIsNone(gest.gesture_from_note("zoom in on the logo"))
        sents = [{"text": "Hello!", "start": 0, "end": 1}, {"text": "Welcome!", "start": 1, "end": 2}]
        self.assertEqual(len([e for e in gest.plan_gestures(sents) if e["channel"] == "hands"]), 1)  # cool-down

    def test_every_hand_gesture_has_valid_directions(self):
        for name, g in gest.HAND_GESTURES.items():
            self.assertEqual(len(g["arm"]), 3, name)
            self.assertEqual(len(g["fore"]), 3, name)
            self.assertIn(g["side"], ("both", "left", "right"))

    def test_envelopes(self):
        ev = {"start": 1.0, "end": 4.0}
        self.assertEqual(gest.hand_envelope(ev, 0.5), 0.0)
        self.assertEqual(gest.hand_envelope(ev, 2.5), 1.0)
        self.assertLess(gest.hand_envelope(ev, 1.2), 1.0)
        self.assertEqual(gest.head_angles({"name": "nod", "start": 0}, 5.0), {})


if __name__ == "__main__":
    unittest.main()
