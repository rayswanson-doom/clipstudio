"""
backend/avatar_3d_engine.py  -  talking 3D avatars (realistic human, AI model, anime) rendered on the CPU.

    create_3d_avatar_video(model_path, audio_path, output_path, text="...", style="human"|"ai"|"anime")
        -> {"output_path","duration","frames","fps","size","mode":"3d","lip_sync","gestures","warnings",...}

What it does
  * loads a .glb / .vrm avatar (glb_model.py): Ready Player Me, Mixamo, VRoid/VRM, Blender exports;
  * phoneme-level lip sync: the English letter-to-sound rules and Oculus-viseme timing of TalkingHead
    (lipsync_en.py, verified against the original JavaScript), fitted to the real word timestamps when
    they are known and gated by the loudness of the voiceover, then driven into the model's own
    blend shapes (Oculus visemes / ARKit jaw / VRM vowels), or the jaw bone when the model has no mouth shapes;
  * upper-body motion: breathing, blinking, head sway, nods and head shakes, and arm gestures chosen per
    sentence from the motion-engine catalogue (avatar_gestures.py: open palms, wave, point, shrug, ...);
    the arms are solved from directions, so ANY humanoid rig works, whatever its bone axes are;
  * a software renderer (soft_raster.py): skinning, morph targets, textures, soft lighting, toon shading +
    outline for anime, optional 2x supersampling. No GPU, no OpenGL, no PyTorch - numpy and OpenCV only.

Everything heavy is imported lazily; nothing is loaded until a render is requested.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # package import (backend.x) or plain module (x)
    from . import avatar_gestures as gest
    from . import glb_model as G
    from . import lipsync_en as lip
    from . import soft_raster
    from .avatar_sync import decode_audio, energy_envelope, open_encoder
    from .video_engine import find_ffmpeg
except ImportError:  # pragma: no cover
    import avatar_gestures as gest
    import glb_model as G
    import lipsync_en as lip
    import soft_raster
    from avatar_sync import decode_audio, energy_envelope, open_encoder
    from video_engine import find_ffmpeg

log = logging.getLogger(__name__)

MODEL_EXTENSIONS = (".glb", ".vrm")
STYLES = ("human", "ai", "anime")
STYLE_LABELS = {"human": "Photorealistic Human", "ai": "AI Model", "anime": "Anime"}
DEFAULT_SIZE = (480, 640)
DEFAULT_FPS = 25
ProgressCb = Callable[[float], None]

# --------------------------------------------------------------------------- #
# small maths
# --------------------------------------------------------------------------- #
def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def axis_angle(axis, angle: float) -> np.ndarray:
    a = _unit(axis)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], np.float32)


def rot_between(a, b) -> np.ndarray:
    """Smallest rotation taking direction a onto direction b."""
    a, b = _unit(a), _unit(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 0.999999:
        return np.eye(3, dtype=np.float32)
    if d < -0.999999:
        helper = np.array([1, 0, 0], np.float32) if abs(a[0]) < 0.9 else np.array([0, 1, 0], np.float32)
        return axis_angle(np.cross(a, helper), math.pi)
    return axis_angle(np.cross(a, b), math.acos(d))


# --------------------------------------------------------------------------- #
# rig: forward kinematics + head / arm posing
# --------------------------------------------------------------------------- #
@dataclass
class Pose:
    """Everything that changes from frame to frame."""
    morph: Dict[str, float] = field(default_factory=dict)             # channel -> 0..1
    body_rot: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # bone -> (pitch, yaw, roll) rad, body frame
    arms: Dict[str, dict] = field(default_factory=dict)               # "Left"/"Right" -> {"arm","fore","roll"} body-frame dirs


class Rig:
    def __init__(self, model: G.Model):
        self.m = model
        self.n = len(model.names)
        self._children: Dict[int, List[int]] = {}
        for i, p in enumerate(model.parents):
            if p >= 0:
                self._children.setdefault(p, []).append(i)
        bind = self._solve(Pose(), use_arms=False)
        self.bind_world = bind[0]
        b = model.bones
        pos = lambda name: self.bind_world[b[name]][:3, 3] if name in b else None  # noqa: E731
        ex = ey = ez = None
        la, ra, hips, head = pos("LeftArm"), pos("RightArm"), pos("Hips"), pos("Head")
        if la is not None and ra is not None:
            ex = _unit(la - ra)
        else:
            ex = np.array([1, 0, 0], np.float32)
        ey = _unit(head - hips) if head is not None and hips is not None and np.linalg.norm(head - hips) > 1e-6 else np.array([0, 1, 0], np.float32)
        ey = _unit(ey - ex * float(np.dot(ey, ex)))
        ez = _unit(np.cross(ex, ey))
        self.ex, self.ey, self.ez = ex, ey, ez
        self.frame = np.stack([ex, ey, ez], 1)  # columns: body -> world
        self.can_pose_arms = all(k in b for k in ("LeftArm", "LeftForeArm", "RightArm", "RightForeArm"))
        self.can_pose_head = "Head" in b

    # ---- solver ----
    def _solve(self, pose: Pose, use_arms: bool = True):
        m = self.m
        W = np.zeros((self.n, 4, 4), np.float32)
        WR = np.zeros((self.n, 3, 3), np.float32)
        bones = m.bones
        deltas: Dict[int, np.ndarray] = {}
        F = getattr(self, "frame", None)
        if F is not None:
            for name, (pitch, yaw, roll) in pose.body_rot.items():
                if name in bones and (pitch or yaw or roll):
                    D = axis_angle([0, 0, 1], roll) @ axis_angle([0, 1, 0], yaw) @ axis_angle([1, 0, 0], pitch)
                    deltas[bones[name]] = F @ D @ F.T
        aim: Dict[int, Tuple[np.ndarray, float, int]] = {}
        if use_arms and F is not None and self.can_pose_arms:
            for side, sgn in (("Left", 1.0), ("Right", -1.0)):
                spec = pose.arms.get(side)
                if not spec:
                    continue
                def world(d):  # (outward, up, forward) -> world direction
                    return _unit(sgn * d[0] * self.ex + d[1] * self.ey + d[2] * self.ez)
                aim[bones[f"{side}Arm"]] = (world(spec["arm"]), 0.0, bones[f"{side}ForeArm"])
                hand = bones.get(f"{side}Hand")
                aim[bones[f"{side}ForeArm"]] = (world(spec["fore"]), float(spec.get("roll", 0.0)) * sgn,
                                                hand if hand is not None else -1)
        for i in self.m.order:
            p = self.m.parents[i]
            Wp = W[p] if p >= 0 else np.eye(4, dtype=np.float32)
            Rp = WR[p] if p >= 0 else np.eye(3, dtype=np.float32)
            R = self.m.local_r[i]
            t = self.m.local_t[i]
            if i in deltas:
                R = Rp.T @ deltas[i] @ Rp @ R
            elif i in aim and aim[i][2] >= 0:
                target, roll, child = aim[i]
                Wi = Wp.copy()
                Wi[:3, :3] = Wp[:3, :3] @ (R * self.m.local_s[i][None, :])
                Wi[:3, 3] = Wp[:3, :3] @ t + Wp[:3, 3]
                child_pos = Wi[:3, :3] @ self.m.local_t[child] + Wi[:3, 3]
                cur = child_pos - Wi[:3, 3]
                D = rot_between(cur, target)
                if roll:
                    D = axis_angle(target, roll) @ D
                R = Rp.T @ D @ Rp @ R
            local = np.eye(4, dtype=np.float32)
            local[:3, :3] = R * self.m.local_s[i][None, :]
            local[:3, 3] = t
            W[i] = Wp @ local
            WR[i] = Rp @ R
        return W, WR

    def solve(self, pose: Pose) -> np.ndarray:
        return self._solve(pose)[0]


# --------------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------------- #
STYLE_BACKGROUNDS = {  # (top RGB, bottom RGB)
    "human": ((58, 66, 84), (22, 26, 38)),
    "ai": ((24, 52, 76), (10, 14, 34)),
    "anime": ((246, 214, 226), (176, 170, 240)),
}
OUTLINE_RGB = np.array([46, 30, 52], np.float32)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Renderer:
    def __init__(self, model: G.Model, rig: Rig, size: Tuple[int, int] = DEFAULT_SIZE, style: str = "human",
                 ssaa: float = 1.5, framing: str = "bust", background: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None):
        self.m, self.rig = model, rig
        self.W, self.H = int(size[0]), int(size[1])
        self.style = style if style in STYLES else "human"
        self.ssaa = float(max(1.0, min(ssaa, 3.0)))
        self.rw, self.rh = int(round(self.W * self.ssaa)), int(round(self.H * self.ssaa))
        self.background = background or STYLE_BACKGROUNDS[self.style]
        self.tris_all, self.tri_mat, self.vert_offsets, self.two_sided, self.alpha_tris = [], [], [], [], []
        off = 0
        for pr in model.prims:
            self.vert_offsets.append(off)
            mat = model.materials[pr.material]
            self.tris_all.append(pr.tris + off)
            self.tri_mat.append(np.full(len(pr.tris), pr.material, np.int32))
            self.two_sided.append(np.full(len(pr.tris), mat.double_sided, bool))
            self.alpha_tris.append(np.full(len(pr.tris), mat.alpha_mode in ("MASK", "BLEND") and mat.texture is not None, bool))
            off += len(pr.positions)
        self.tris = np.concatenate(self.tris_all)
        self.tri_mat = np.concatenate(self.tri_mat)
        self.two_sided = np.concatenate(self.two_sided)
        self.alpha_tris = np.concatenate(self.alpha_tris)
        self.n_verts = off
        self.uv = np.concatenate([pr.uvs if pr.uvs is not None else np.zeros((len(pr.positions), 2), np.float32)
                                  for pr in model.prims])
        self.morph_stack = []
        for pr in model.prims:
            self.morph_stack.append(np.stack(pr.morph_deltas) if pr.morph_deltas else None)
        self.zero_pose = Pose()
        self._frame_camera()

    # ---- camera ----
    def _frame_camera(self):
        W0 = self.rig.bind_world
        pos, _ = self.world_vertices(W0, {})
        ex, ey, ez = self.rig.ex, self.rig.ey, self.rig.ez
        up = pos @ ey
        top, bottom = float(up.max()), float(up.min())
        b = self.m.bones
        if "LeftArm" in b and "RightArm" in b:
            sh_y = float((W0[b["LeftArm"]][:3, 3] @ ey + W0[b["RightArm"]][:3, 3] @ ey) / 2)
            sh_w = float(np.linalg.norm(W0[b["LeftArm"]][:3, 3] - W0[b["RightArm"]][:3, 3]))
        else:
            sh_y, sh_w = top - 0.2 * (top - bottom), 0.25 * (top - bottom)
        h2s = max(top - sh_y, 1e-3)
        view_h = 2.75 * h2s
        aspect = self.W / self.H
        view_h = max(view_h, 2.2 * max(sh_w, 1e-3) / aspect)
        cx = float((W0[b["Head"]][:3, 3] @ ex)) if "Head" in b else float((pos @ ex).mean())
        cz = float((W0[b["Head"]][:3, 3] @ ez)) if "Head" in b else float((pos @ ez).mean())
        top_frame = top + 0.22 * h2s
        cy = top_frame - view_h / 2
        self.fov = math.radians(24.0)
        self.dist = (view_h / 2) / math.tan(self.fov / 2)
        self.focal = (self.rh / 2) / math.tan(self.fov / 2)
        self.target = cx * ex + cy * ey + cz * ez
        self.cam_pos = self.target + ez * self.dist
        self.view_h = view_h

    # ---- geometry ----
    def world_vertices(self, W: np.ndarray, morph_weights: Dict[int, np.ndarray]):
        pos_out, nor_out = [], []
        for pi, pr in enumerate(self.m.prims):
            P = pr.positions
            w = morph_weights.get(pi)
            if w is not None and self.morph_stack[pi] is not None:
                nz = np.nonzero(w)[0]
                if len(nz):
                    P = P + np.tensordot(w[nz], self.morph_stack[pi][nz], axes=(0, 0))
            N = pr.normals
            if pr.skin is not None and pr.joints is not None:
                sk = self.m.skins[pr.skin]
                J = W[sk.joints] @ sk.inverse_bind                          # (nj,4,4)
                M = np.einsum("nk,nkij->nij", pr.weights, J[pr.joints])     # (N,4,4)
                pw = np.einsum("nij,nj->ni", M[:, :3, :3], P) + M[:, :3, 3]
                nw = np.einsum("nij,nj->ni", M[:, :3, :3], N)
            else:
                M = W[pr.node]
                pw = P @ M[:3, :3].T + M[:3, 3]
                nw = N @ M[:3, :3].T
            nn = np.linalg.norm(nw, axis=1, keepdims=True)
            pos_out.append(pw.astype(np.float32))
            nor_out.append((nw / np.maximum(nn, 1e-9)).astype(np.float32))
        return np.concatenate(pos_out), np.concatenate(nor_out)

    # ---- texture sampling ----
    def _sample(self, tex_idx: int, uv: np.ndarray) -> np.ndarray:
        tex = self.m.textures[tex_idx]
        h, w = tex.shape[:2]
        u = np.mod(uv[:, 0], 1.0) * w - 0.5
        v = np.mod(uv[:, 1], 1.0) * h - 0.5
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        fx = (u - x0).astype(np.float32)[:, None]
        fy = (v - y0).astype(np.float32)[:, None]
        x0m, x1m = np.mod(x0, w), np.mod(x0 + 1, w)
        y0m, y1m = np.mod(y0, h), np.mod(y0 + 1, h)
        top = tex[y0m, x0m] * (1 - fx) + tex[y0m, x1m] * fx
        bottom = tex[y1m, x0m] * (1 - fx) + tex[y1m, x1m] * fx
        return (top * (1 - fy) + bottom * fy) * np.float32(1 / 255.0)

    # ---- one frame ----
    def render(self, W: np.ndarray, morph_weights: Dict[int, np.ndarray]) -> np.ndarray:
        pos, nor = self.world_vertices(W, morph_weights)
        ex, ey, ez = self.rig.ex, self.rig.ey, self.rig.ez
        rel = pos - self.cam_pos
        xv, yv = rel @ ex, rel @ ey
        zc = -(rel @ ez)
        zc_safe = np.where(zc > 1e-4, zc, 1e-4)
        sx = self.rw / 2 + self.focal * xv / zc_safe
        sy = self.rh / 2 - self.focal * yv / zc_safe
        zc_arr = zc.astype(np.float32)

        def alpha_fn(tri_ids, bary):
            keep = np.ones(len(tri_ids), bool)
            mats = self.tri_mat[tri_ids]
            for mi in np.unique(mats):
                mat = self.m.materials[mi]
                if mat.texture is None or mat.alpha_mode == "OPAQUE":
                    continue
                sel = np.nonzero(mats == mi)[0]
                tv = self.tris[tri_ids[sel]]
                uv = (self.uv[tv] * bary[sel][:, :, None]).sum(1)
                a = self._sample(mat.texture, uv)[:, 3] * mat.base_color[3]
                keep[sel] = a >= (mat.alpha_cutoff if mat.alpha_mode == "MASK" else 0.5)
            return keep

        res = soft_raster.rasterize(sx, sy, zc_arr, self.tris, self.rw, self.rh, self.two_sided,
                                    alpha_fn if self.alpha_tris.any() else None, self.alpha_tris)
        return self._shade(res, nor, zc_arr)

    def _shade(self, res, nor: np.ndarray, zc: np.ndarray) -> np.ndarray:
        rh, rw = self.rh, self.rw
        covered = res.tri_id >= 0
        ys, xs = np.nonzero(covered)
        color = np.zeros((rh, rw, 3), np.float32)
        if len(ys):
            tid = res.tri_id[ys, xs]
            bary = res.bary[ys, xs]
            tv = self.tris[tid]
            N = (nor[tv] * bary[:, :, None]).sum(1)
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
            ez = self.rig.ez
            # back faces of two-sided materials: flip so they are lit from the viewer's side
            N = np.where((N @ ez)[:, None] < -0.0, N, N)
            base = np.ones((len(tid), 3), np.float32)
            uv = (self.uv[tv] * bary[:, :, None]).sum(1)
            mats = self.tri_mat[tid]
            emissive = np.zeros((len(tid), 3), np.float32)
            for mi in np.unique(mats):
                mat = self.m.materials[mi]
                sel = np.nonzero(mats == mi)[0]
                col = np.tile(np.array(mat.base_color[:3], np.float32), (len(sel), 1))
                if mat.texture is not None:
                    col = col * self._sample(mat.texture, uv[sel])[:, :3]
                base[sel] = col
                emissive[sel] = np.array(mat.emissive, np.float32)
            color[ys, xs] = self._light(N, base, emissive)
        alpha = covered.astype(np.float32)
        if self.style == "anime" and len(ys):
            d = res.depth.copy()
            d[~covered] = float(self.dist * 3)
            mat_map = np.where(covered, self.tri_mat[np.maximum(res.tri_id, 0)], -1)
            edge = np.zeros((rh, rw), bool)
            for dy, dx in ((0, 1), (1, 0)):
                a = d[: rh - dy, : rw - dx]
                b = d[dy:, dx:]
                jump = np.abs(a - b) > 0.012 * self.dist
                differ = mat_map[: rh - dy, : rw - dx] != mat_map[dy:, dx:]
                e = jump | differ
                edge[: rh - dy, : rw - dx] |= e
            import cv2
            k = max(1, int(round(self.ssaa * 0.9)))
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            edge &= covered | cv2.dilate(covered.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
            color[edge] = OUTLINE_RGB / 255.0
            alpha = np.maximum(alpha, edge.astype(np.float32))
        return self._compose(color, alpha)

    def _light(self, N: np.ndarray, base: np.ndarray, emissive: np.ndarray) -> np.ndarray:
        ex, ey, ez = self.rig.ex, self.rig.ey, self.rig.ez
        key = _unit(-0.45 * ex + 0.55 * ey + 0.70 * ez)
        fill = _unit(0.65 * ex + 0.10 * ey + 0.60 * ez)
        rim = _unit(0.10 * ex + 0.45 * ey - 0.85 * ez)
        view = ez
        l1 = np.clip(N @ key, 0, 1)
        l2 = np.clip(N @ fill, 0, 1)
        if self.style == "anime":
            band = 0.62 + 0.20 * _smoothstep(0.10, 0.18, l1) + 0.18 * _smoothstep(0.50, 0.58, l1)
            shade = np.clip(band + 0.10 * l2, 0, 1.08)[:, None]
            col = base * shade
            rimv = _smoothstep(0.72, 0.78, 1 - np.clip(N @ view, 0, 1)) * np.clip(N @ rim * 0.5 + 0.5, 0, 1) * 0.18
            col = col + rimv[:, None] * np.array([1.0, 0.95, 1.0], np.float32)
        else:
            amb = 0.40 if self.style == "human" else 0.34
            lit = amb + 0.62 * l1 + 0.24 * l2
            col = base * lit[:, None]
            h = _unit(key + view)
            spec = np.power(np.clip(N @ h, 0, 1), 28) * 0.09
            col = col + spec[:, None]
            rimk = np.power(1 - np.clip(N @ view, 0, 1), 3.0) * np.clip(N @ rim * 0.6 + 0.4, 0, 1)
            tint = np.array([0.95, 1.0, 1.05], np.float32) if self.style == "human" else np.array([0.35, 0.95, 1.15], np.float32)
            col = col + (rimk * (0.10 if self.style == "human" else 0.28))[:, None] * tint
        return np.clip(col + emissive, 0, 1)

    def _compose(self, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        import cv2

        pre = color * alpha[..., None]
        if self.ssaa != 1.0:
            pre = cv2.resize(pre, (self.W, self.H), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (self.W, self.H), interpolation=cv2.INTER_AREA)
        top, bot = self.background
        t = np.linspace(0, 1, self.H, dtype=np.float32)[:, None, None]
        bg = (np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t) / 255.0
        bg = np.broadcast_to(bg, (self.H, self.W, 3))
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        vig = 1.0 - 0.18 * (((xx / self.W - 0.5) ** 2 + (yy / self.H - 0.45) ** 2) * 2.2)
        bg = bg * vig[..., None]
        out = pre + bg * (1 - alpha[..., None])
        out = out[..., ::-1]  # RGB -> BGR
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# performance: what the avatar does over time
# --------------------------------------------------------------------------- #
OPEN = {"aa": 0.85, "O": 0.60, "E": 0.50, "I": 0.30, "U": 0.35, "PP": 0.0, "SS": 0.12, "TH": 0.18, "DD": 0.22,
        "FF": 0.08, "kk": 0.25, "nn": 0.18, "RR": 0.28, "CH": 0.20, "sil": 0.0}
VRM_COMBO = {"aa": {"vrm_A": 1.0}, "E": {"vrm_E": 1.0}, "I": {"vrm_I": 1.0}, "O": {"vrm_O": 1.0}, "U": {"vrm_U": 1.0},
             "SS": {"vrm_I": 0.35}, "TH": {"vrm_E": 0.35}, "DD": {"vrm_E": 0.40}, "FF": {"vrm_I": 0.30},
             "kk": {"vrm_E": 0.35}, "nn": {"vrm_E": 0.25}, "RR": {"vrm_O": 0.35}, "CH": {"vrm_U": 0.35, "vrm_I": 0.25}}


def _split_sentences_with_times(text: str, spans: List[dict]) -> List[dict]:
    out: Dict[int, dict] = {}
    for sp in spans:
        d = out.setdefault(sp["sentence"], {"words": [], "start": sp["start"], "end": sp["end"]})
        d["words"].append(sp["word"])
        d["end"] = max(d["end"], sp["end"])
    return [{"text": " ".join(v["words"]), "start": v["start"], "end": v["end"]} for k, v in sorted(out.items())]


class Performance:
    """Lip-sync + blinks + breathing + head/arm gestures for `duration` seconds at `fps`."""

    def __init__(self, model: G.Model, rig: Rig, duration: float, fps: int, text: Optional[str] = None,
                 words: Optional[Sequence[dict]] = None, envelope: Optional[np.ndarray] = None,
                 notes: Optional[List[dict]] = None, gestures: bool = True, intensity: float = 1.0, seed: int = 7):
        self.m, self.rig = model, rig
        self.fps, self.duration = fps, duration
        self.n = max(1, int(math.ceil(duration * fps)))
        self.intensity = float(max(0.0, min(intensity, 2.0)))
        self.env = envelope
        self.kind = model.lip_sync_kind()
        if self.kind == "none" and "Jaw" in model.bones:
            self.kind = "jaw-bone"
        spans = lip.word_spans(text, duration, list(words) if words else None) if text else []
        self.sentences = _split_sentences_with_times(text or "", spans) if spans else []
        timeline = lip.text_to_viseme_timeline(text, duration, list(words) if words else None) if text else []
        self.has_text = bool(timeline)
        self.vis = self._viseme_curves(timeline)
        self.events = gest.plan_gestures(self.sentences, notes, duration, seed, self.intensity) if (gestures and self.sentences or notes) and gestures else []
        rng = np.random.default_rng(seed)
        self.blinks: List[float] = []
        t = float(rng.uniform(0.8, 2.0))
        while t < duration:
            self.blinks.append(t)
            t += float(rng.uniform(2.4, 5.0))
        self.phase = rng.uniform(0, 6.28, 8)

    # ---- lip sync ----
    def _viseme_curves(self, timeline: List[dict]) -> Dict[str, np.ndarray]:
        n, fps = self.n, self.fps
        curves = {v: np.zeros(n, np.float32) for v in lip.VISEMES}
        t_frames = np.arange(n) / fps
        if timeline:
            for it in timeline:
                s, e = it["start"], it["end"]
                d = max(e - s, 1e-3)
                ts = [s - min(0.060, 2 * d / 3), s + min(0.025, d / 2), s + d + min(0.060, d / 2)]
                a = max(0, int(math.floor(ts[0] * fps)))
                b = min(n, int(math.ceil(ts[2] * fps)) + 1)
                if b <= a:
                    continue
                w = np.interp(t_frames[a:b], ts, [0.0, it["level"], 0.0], left=0.0, right=0.0)
                curves[it["viseme"]][a:b] = np.maximum(curves[it["viseme"]][a:b], w)
        elif self.env is not None:  # no text: open the mouth with the voice
            env = np.resize(self.env, n).astype(np.float32)
            curves["aa"] = np.clip(env * 0.9, 0, 1)
            curves["E"] = np.clip(env * 0.35 * (0.5 + 0.5 * np.sin(t_frames * 9.0)), 0, 1)
        if self.env is not None and timeline:
            env = np.resize(self.env, n).astype(np.float32)
            gate = np.clip(env * 3.0, 0.0, 1.0)
            kernel = np.array([0.25, 0.5, 0.25], np.float32)
            gate = np.convolve(np.pad(gate, 1, mode="edge"), kernel, mode="valid")
            scale = (0.35 + 0.65 * np.clip(env / 0.6, 0, 1)) * gate
            for v in curves:
                curves[v] = curves[v] * scale
        for v, c in curves.items():  # open fast, close a little slower
            out, cur = np.empty_like(c), 0.0
            for i, x in enumerate(c):
                cur += (x - cur) * (0.75 if x > cur else 0.5)
                out[i] = cur
            curves[v] = out
        return curves

    def mouth_open(self, i: int) -> float:
        return float(min(1.0, sum(OPEN[v] * self.vis[v][i] for v in lip.VISEMES)))

    def _channels(self, i: int, t: float, pose: Pose):
        ch = self.m.channels
        mo = pose.morph
        w = {v: float(self.vis[v][i]) for v in lip.VISEMES}
        if self.kind == "visemes":
            for v, x in w.items():
                if x > 0.002 and v != "sil":
                    mo[f"viseme_{v}"] = x
        elif self.kind == "vrm-vowels":
            for v, x in w.items():
                for name, k in VRM_COMBO.get(v, {}).items():
                    mo[name] = min(1.0, mo.get(name, 0.0) + x * k)
        elif self.kind == "jaw-open":
            mo["jawOpen"] = min(1.0, self.mouth_open(i) * 0.9)
            if "mouthFunnel" in ch:
                mo["mouthFunnel"] = min(0.6, (w["O"] + w["U"]) * 0.5)
        elif self.kind == "jaw-bone":
            pose.body_rot["Jaw"] = (self.mouth_open(i) * 0.30, 0.0, 0.0)
        speaking = float(self.env[min(i, len(self.env) - 1)]) if self.env is not None and len(self.env) else self.mouth_open(i)
        smile = 0.10 + 0.05 * math.sin(t * 0.7)
        for ev in self.events:
            if ev["channel"] == "hands" and ev["name"] in ("wave", "thumbs_up", "celebrate", "applause", "namaste", "ok_sign"):
                smile += 0.35 * gest.hand_envelope(ev, t)
        if "smile" in ch:
            mo["smile"] = min(1.0, smile * self.intensity)
        for side in ("Left", "Right"):
            if f"smile{side}" in ch:
                mo[f"smile{side}"] = min(1.0, smile * self.intensity)
        if "browInnerUp" in ch:
            mo["browInnerUp"] = min(1.0, (0.08 + 0.22 * speaking) * self.intensity)
        blink = 0.0
        for b in self.blinks:
            if b - 0.02 <= t <= b + 0.16:
                blink = max(blink, 1.0 - abs((t - b - 0.07) / 0.09))
        blink = float(max(0.0, min(1.0, blink)))
        if blink > 0:
            for name in ("blink", "blinkLeft", "blinkRight"):
                if name in ch:
                    mo[name] = blink

    # ---- body ----
    def pose(self, i: int) -> Pose:
        t = i / self.fps
        k = self.intensity
        pose = Pose()
        self._channels(i, t, pose)
        speaking = float(self.env[min(i, len(self.env) - 1)]) if self.env is not None and len(self.env) else self.mouth_open(i)
        ph = self.phase
        breathe = math.sin(2 * math.pi * 0.27 * t + ph[0])
        pose.body_rot["Spine1"] = (0.010 * breathe * k, 0.015 * math.sin(2 * math.pi * 0.11 * t + ph[1]) * k, 0.0)
        pose.body_rot["Spine2"] = (0.006 * breathe * k, 0.0, 0.012 * math.sin(2 * math.pi * 0.13 * t + ph[2]) * k)
        pose.body_rot["Hips"] = (0.0, 0.012 * math.sin(2 * math.pi * 0.09 * t + ph[3]) * k, 0.0)
        talk = 0.45 + 0.55 * min(1.0, speaking * 1.5)
        pitch = (0.022 * math.sin(2 * math.pi * 0.31 * t + ph[4]) + 0.030 * speaking) * k * talk
        yaw = 0.060 * math.sin(2 * math.pi * 0.17 * t + ph[5]) * k * talk
        roll = 0.022 * math.sin(2 * math.pi * 0.23 * t + ph[6]) * k * talk
        body_pitch = 0.0
        for ev in self.events:
            if ev["channel"] == "head" and ev["start"] <= t <= ev["end"]:
                a = gest.head_angles(ev, t)
                pitch += a.get("headRotateX", 0.0)
                yaw += a.get("headRotateY", 0.0)
                roll += a.get("headRotateZ", 0.0)
                body_pitch += a.get("bodyRotateX", 0.0)
        pose.body_rot["Neck"] = (pitch * 0.4, yaw * 0.4, roll * 0.4)
        pose.body_rot["Head"] = (pitch * 0.6, yaw * 0.6, roll * 0.6)
        if body_pitch:
            sp = pose.body_rot["Spine1"]
            pose.body_rot["Spine1"] = (sp[0] + body_pitch, sp[1], sp[2])
        # arms: relaxed, with the current hand gesture blended in
        arms = {}
        for side in ("Left", "Right"):
            spec = dict(gest.REST)
            sway = 0.025 * math.sin(2 * math.pi * 0.2 * t + (ph[7] if side == "Left" else ph[1]))
            arm = np.array(spec["arm"], np.float32) + np.array([0, 0, sway], np.float32)
            fore = np.array(spec["fore"], np.float32)
            roll_a = 0.0
            for ev in self.events:
                if ev["channel"] != "hands":
                    continue
                e = gest.hand_envelope(ev, t) * min(1.0, 0.6 + 0.4 * k)
                if e <= 0:
                    continue
                g = gest.HAND_GESTURES[ev["name"]]
                if g["side"] != "both" and g["side"].capitalize() != side:
                    continue
                arm = arm * (1 - e) + np.array(g["arm"], np.float32) * e
                f_t = np.array(g["fore"], np.float32)
                osc = g.get("osc")
                if osc:
                    f_t = f_t + np.array([osc["amp"] * math.sin(2 * math.pi * osc["freq"] * (t - ev["start"])), 0, 0], np.float32)
                fore = fore * (1 - e) + f_t * e
                roll_a = roll_a * (1 - e) + g["roll"] * e
            arms[side] = {"arm": tuple(_unit(arm)), "fore": tuple(_unit(fore)), "roll": roll_a}
        pose.arms = arms
        return pose

    def prim_weights(self, morph: Dict[str, float]) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for name, value in morph.items():
            for pi, ti, wt in self.m.channels.get(name, ()):
                arr = out.get(pi)
                if arr is None:
                    arr = out[pi] = np.zeros(len(self.m.prims[pi].morph_deltas), np.float32)
                arr[ti] = min(1.0, arr[ti] + value * wt)
        return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
QUALITY = {"fast": 1.0, "balanced": 1.25, "high": 2.0}
_MODEL_CACHE: Dict[Tuple[str, float], G.Model] = {}


def inspect_model(path: str) -> dict:
    """What a model supports - shown to the user before rendering."""
    m = G.load_model(path)
    rig = Rig(m)
    return {"format": m.format, "vertices": m.vertex_count, "triangles": m.triangle_count,
            "lip_sync": m.lip_sync_kind(), "has_visemes": m.has_visemes(),
            "bones": sorted(m.bones), "arms_posable": rig.can_pose_arms, "head_posable": rig.can_pose_head,
            "blink": any(k in m.channels for k in ("blink", "blinkLeft", "blinkRight")),
            "expressions": sorted(m.channels), "warnings": list(m.warnings)}


def create_3d_avatar_video(
    model_path: str,
    audio_path: str,
    output_path: str,
    text: Optional[str] = None,
    words: Optional[Sequence[dict]] = None,
    notes: Optional[List[dict]] = None,
    style: str = "human",
    size: Tuple[int, int] = DEFAULT_SIZE,
    fps: int = DEFAULT_FPS,
    quality: str = "balanced",
    gestures: bool = True,
    intensity: float = 1.0,
    framing: str = "bust",
    ffmpeg_path: Optional[str] = None,
    progress: Optional[ProgressCb] = None,
    seed: int = 7,
) -> dict:
    """Render a talking 3D avatar MP4 (with the voiceover as its audio track).

    text       what is being said - gives phoneme-level lip sync. Without it the mouth follows the loudness.
    words      faster-whisper word timestamps; keep real timing between words when known
    notes      [{"start": seconds, "text": "wave at the audience"}] explicit gestures from the user's script
    style      "human" (realistic lighting), "ai" (cool rim light), "anime" (cel shading + outline)
    """
    import cv2  # noqa: F401  (fail early with a clear message if OpenCV is missing)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    warnings: List[str] = []
    model = G.load_model(model_path)
    warnings.extend(model.warnings)
    rig = Rig(model)
    ssaa = QUALITY.get(quality, 1.25)
    W, H = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)
    samples = decode_audio(audio_path, find_ffmpeg(ffmpeg_path) if ffmpeg_path else None)
    duration = len(samples) / 16000.0
    if duration <= 0.05:
        raise ValueError("The voiceover is empty.")
    env = energy_envelope(samples, 16000, fps)
    perf = Performance(model, rig, duration, fps, text, words, env, notes, gestures, intensity, seed)
    if perf.kind == "none":
        warnings.append("This model has no mouth shapes or jaw bone, so it cannot lip-sync; it still moves its head and arms.")
    if not rig.can_pose_arms:
        warnings.append("The rig has no recognisable arm bones, so arm gestures are skipped.")
    if not rig.can_pose_head:
        warnings.append("The rig has no head bone, so head motion is skipped.")
    renderer = Renderer(model, rig, (W, H), style, ssaa, framing)
    ffmpeg = find_ffmpeg(ffmpeg_path)
    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    enc = open_encoder(ffmpeg, str(audio_path), out, (W, H), fps, duration)
    try:
        for i in range(perf.n):
            pose = perf.pose(i)
            frame = renderer.render(rig.solve(pose), perf.prim_weights(pose.morph))
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if progress and i % 5 == 0:
                progress(i / perf.n)
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        code = enc.wait()
        enc.stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed while writing the 3D avatar video: {err[-500:]}")
    except BrokenPipeError:
        err = enc.stderr.read().decode(errors="replace") if enc.stderr else ""
        raise RuntimeError(f"ffmpeg stopped early: {err[-500:]}") from None
    finally:
        if enc.poll() is None:
            enc.kill()
    if progress:
        progress(1.0)
    return {"output_path": out, "duration": round(duration, 2), "frames": perf.n, "fps": fps, "size": (W, H),
            "mode": "3d", "style": style, "lip_sync": perf.kind, "model_format": model.format,
            "gestures": [{"name": e["name"], "channel": e["channel"], "start": e["start"]} for e in perf.events],
            "used_word_timestamps": bool(words), "warnings": warnings}
