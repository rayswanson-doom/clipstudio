"""
backend/demo_avatars.py  -  built-in 3D avatars, generated in code (no downloads, no licence worries).

    write_demo_avatar("human" | "anime" | "ai", "out.glb")  ->  path

Low-poly, rigged upper-body characters with the SAME rig and blend shapes a Ready Player Me avatar has:
a Mixamo-named skeleton, the 15 Oculus viseme shapes, ARKit blink / smile / jaw shapes, a painted face
texture. They are what the 3D Studio uses when you do not upload a model of your own, and they let the
whole 3D pipeline be tested without any third-party files. The "anime" one is written as a VRM 0.x file
(humanoid bone table + A/I/U/E/O/blink blend-shape groups) so that path is exercised too.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

VISEME_SHAPES = {  # name: (jaw open, mouth widening, pucker)
    "sil": (0.0, 0.0, 0.0), "PP": (0.0, -0.2, 0.0), "FF": (0.10, 0.1, 0.0), "TH": (0.20, 0.1, 0.0),
    "DD": (0.25, 0.1, 0.0), "kk": (0.30, 0.0, 0.0), "CH": (0.22, -0.3, 0.5), "SS": (0.12, 0.5, 0.0),
    "nn": (0.20, 0.0, 0.0), "RR": (0.28, -0.3, 0.4), "aa": (0.90, 0.2, 0.0), "E": (0.50, 0.6, 0.0),
    "I": (0.30, 0.8, 0.0), "O": (0.65, -0.5, 0.9), "U": (0.35, -0.6, 1.0),
}
JOINTS = {  # name: (parent, world position)
    "Hips": (None, (0, 0.95, 0)), "Spine": ("Hips", (0, 1.05, 0)), "Spine1": ("Spine", (0, 1.18, 0)),
    "Spine2": ("Spine1", (0, 1.32, 0)), "Neck": ("Spine2", (0, 1.48, 0)), "Head": ("Neck", (0, 1.55, 0)),
    "LeftShoulder": ("Spine2", (0.05, 1.42, 0)), "LeftArm": ("LeftShoulder", (0.20, 1.42, 0)),
    "LeftForeArm": ("LeftArm", (0.46, 1.42, 0)), "LeftHand": ("LeftForeArm", (0.70, 1.42, 0)),
    "RightShoulder": ("Spine2", (-0.05, 1.42, 0)), "RightArm": ("RightShoulder", (-0.20, 1.42, 0)),
    "RightForeArm": ("RightArm", (-0.46, 1.42, 0)), "RightHand": ("RightForeArm", (-0.70, 1.42, 0)),
}
ORDER = list(JOINTS)
STYLE = {
    "human": {"skin": (0.86, 0.66, 0.54), "hair": (0.20, 0.13, 0.09), "shirt": (0.74, 0.30, 0.27), "head_r": (0.118, 0.142, 0.122),
              "eye": 0.018, "iris": (60, 90, 120)},
    "ai": {"skin": (0.78, 0.84, 0.90), "hair": (0.10, 0.45, 0.60), "shirt": (0.12, 0.55, 0.70), "head_r": (0.118, 0.142, 0.122),
           "eye": 0.020, "iris": (40, 220, 240)},
    "anime": {"skin": (1.0, 0.88, 0.82), "hair": (0.95, 0.52, 0.70), "shirt": (0.95, 0.95, 1.0), "head_r": (0.132, 0.140, 0.126),
              "eye": 0.034, "iris": (180, 70, 160)},
}


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class _Mesh:
    def __init__(self, material: int):
        self.material = material
        self.pos: List[np.ndarray] = []
        self.nor: List[np.ndarray] = []
        self.uv: List[np.ndarray] = []
        self.jw: List[np.ndarray] = []   # (n,4) joint idx
        self.ww: List[np.ndarray] = []   # (n,4) weights
        self.idx: List[np.ndarray] = []
        self.n = 0
        self.targets: Dict[str, np.ndarray] = {}

    def add(self, pos, nor, uv, joints, weights, idx):
        self.pos.append(np.asarray(pos, np.float32)); self.nor.append(np.asarray(nor, np.float32))
        self.uv.append(np.asarray(uv, np.float32)); self.jw.append(np.asarray(joints, np.uint16))
        self.ww.append(np.asarray(weights, np.float32)); self.idx.append(np.asarray(idx, np.uint32) + self.n)
        self.n += len(pos)

    def done(self):
        return (np.concatenate(self.pos), np.concatenate(self.nor), np.concatenate(self.uv), np.concatenate(self.jw),
                np.concatenate(self.ww), np.concatenate(self.idx).reshape(-1))


def _grid_indices(rows: int, cols: int, wrap: bool = True) -> np.ndarray:
    out = []
    for r in range(rows - 1):
        for c in range(cols):
            c2 = (c + 1) % cols if wrap else c + 1
            if not wrap and c == cols - 1:
                continue
            a, b, d, e = r * cols + c, r * cols + c2, (r + 1) * cols + c, (r + 1) * cols + c2
            out += [[a, d, b], [b, d, e]]
    return np.array(out, np.uint32)


def _ellipsoid(center, radii, rows=22, cols=32):
    v = np.linspace(0, 1, rows)[:, None]
    u = np.linspace(0, 1, cols, endpoint=False)[None, :]
    phi = (u - 0.5) * 2 * np.pi            # 0 = front (+z)
    theta = v * np.pi                       # 0 = top
    x = np.sin(theta) * np.sin(phi) * radii[0]
    y = np.cos(theta) * radii[1] + 0 * phi
    z = np.sin(theta) * np.cos(phi) * radii[2]
    pos = np.stack([x, y, z], -1).reshape(-1, 3)
    nor = pos / np.array(radii, np.float32) ** 2
    nor /= np.linalg.norm(nor, axis=1, keepdims=True)
    uv = np.stack([np.broadcast_to(u, (rows, cols)), np.broadcast_to(v, (rows, cols))], -1).reshape(-1, 2)
    uv[:, 0] = (uv[:, 0] + 0.0)
    return pos + np.array(center, np.float32), nor, uv, _grid_indices(rows, cols)


def _tube(a, b, r0, r1, rows=6, cols=14):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    axis = b - a
    length = np.linalg.norm(axis)
    d = axis / length
    helper = np.array([0, 0, 1], np.float32) if abs(d[2]) < 0.9 else np.array([0, 1, 0], np.float32)
    u1 = np.cross(d, helper); u1 /= np.linalg.norm(u1)
    u2 = np.cross(d, u1)
    s = np.linspace(0, 1, rows)[:, None]
    ang = np.linspace(0, 2 * np.pi, cols, endpoint=False)[None, :]
    rad = r0 + (r1 - r0) * s
    ring = (np.cos(ang)[..., None] * u1 + np.sin(ang)[..., None] * u2)
    pos = (a + axis * s[..., None] + ring * rad[..., None]).reshape(-1, 3)
    nor = np.broadcast_to(ring, (rows, cols, 3)).reshape(-1, 3).copy()
    t = np.broadcast_to(s, (rows, cols)).reshape(-1)
    uv = np.zeros((len(pos), 2), np.float32)
    return pos, nor, uv, _grid_indices(rows, cols), t


def _face_texture(style: str, size=(512, 256)) -> np.ndarray:
    import cv2

    st = STYLE[style]
    W, H = size
    skin = tuple(int(c * 255) for c in st["skin"][::-1])
    hair = tuple(int(c * 255) for c in st["hair"][::-1])
    tex = np.full((H, W, 3), skin, np.uint8)
    # soft shading across the face
    shade = (np.linspace(0, 1, H)[:, None] * 18).astype(np.int16)
    tex = np.clip(tex.astype(np.int16) - shade[..., None], 0, 255).astype(np.uint8)
    # hair on the top of the head and all round the back
    hair_line = int(H * (0.30 if style != "anime" else 0.27))
    tex[:hair_line] = hair
    back = np.ones(W, bool); back[int(W * 0.25):int(W * 0.75)] = False
    tex[:, back] = hair
    # fringe
    for k in range(-3, 4):
        cx = int(W * 0.5 + k * W * 0.045)
        cv2.ellipse(tex, (cx, hair_line), (int(W * 0.03), int(H * 0.06)), 0, 0, 180, hair, -1)
    ex = int(W * 0.045 * (1.0 if style != "anime" else 1.15))
    ey = int(H * 0.47)
    r = max(3, int(W * st["eye"] * 0.55))
    iris = st["iris"][::-1]
    for sgn in (-1, 1):
        cx = int(W * 0.5) + sgn * ex
        if style == "anime":
            cv2.ellipse(tex, (cx, ey), (r, int(r * 1.35)), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(tex, (cx, ey + 2), (int(r * 0.72), int(r * 1.1)), 0, 0, 360, iris, -1)
            cv2.circle(tex, (cx, ey + 2), int(r * 0.35), (30, 20, 40), -1)
            cv2.circle(tex, (cx - r // 3, ey - r // 2), max(2, r // 4), (255, 255, 255), -1)
            cv2.ellipse(tex, (cx, ey - int(r * 1.35)), (int(r * 1.1), int(r * 0.4)), 0, 180, 360, (40, 25, 40), 3)
        else:
            cv2.ellipse(tex, (cx, ey), (r, int(r * 0.62)), 0, 0, 360, (245, 245, 245), -1)
            cv2.circle(tex, (cx, ey), int(r * 0.55), iris, -1)
            cv2.circle(tex, (cx, ey), int(r * 0.25), (15, 15, 15), -1)
            cv2.ellipse(tex, (cx, ey - int(r * 1.1)), (int(r * 1.2), int(r * 0.35)), 0, 190, 350, hair, 4)
    # nose + mouth
    cv2.ellipse(tex, (int(W * 0.5), int(H * 0.56)), (int(W * 0.008), int(H * 0.025)), 0, 0, 360,
                tuple(max(0, c - 28) for c in skin), -1)
    mx, my = int(W * 0.5), int(H * 0.645)
    cv2.ellipse(tex, (mx, my), (int(W * 0.026), int(H * 0.022)), 0, 0, 360, (45, 25, 70), -1)
    cv2.ellipse(tex, (mx, my - 3), (int(W * 0.028), int(H * 0.012)), 0, 180, 360, (95, 85, 190), 3)
    return tex


def _build(style: str):
    st = STYLE[style]
    skin, hair, shirt = st["skin"], st["hair"], st["shirt"]
    jidx = {n: i for i, n in enumerate(ORDER)}
    mats = [{"name": "face", "tex": True}, {"name": "skin", "color": skin}, {"name": "shirt", "color": shirt},
            {"name": "hair", "color": hair}]
    meshes: List[_Mesh] = []
    J = {n: np.array(v[1], np.float32) for n, v in JOINTS.items()}

    # ---- head with morph targets ----
    head = _Mesh(0)
    hc = np.array([0, 1.68, 0.008], np.float32)
    pos, nor, uv, idx = _ellipsoid(hc, st["head_r"])
    n = len(pos)
    head.add(pos, nor, uv, np.tile([jidx["Head"], 0, 0, 0], (n, 1)), np.tile([1, 0, 0, 0], (n, 1)), idx)
    rel = pos - hc
    front = _smooth(-0.02, 0.07, rel[:, 2])
    mouth_y = -0.050
    low = _smooth(mouth_y + 0.035, mouth_y - 0.045, rel[:, 1]) * front * _smooth(-0.15, -0.02, rel[:, 1] - 0.0 * rel[:, 0] + 0.02)
    near_mouth = np.exp(-(((rel[:, 0] / 0.05) ** 2) + (((rel[:, 1] - mouth_y) / 0.05) ** 2))) * front
    for name, (op, wide, pucker) in VISEME_SHAPES.items():
        d = np.zeros_like(pos)
        d[:, 1] = -op * 0.040 * low
        d[:, 0] = rel[:, 0] * wide * 0.28 * near_mouth
        d[:, 2] = pucker * 0.012 * near_mouth
        head.targets[f"viseme_{name}"] = d
    head.targets["jawOpen"] = head.targets["viseme_aa"] * 1.0
    head.targets["mouthSmile"] = np.stack([rel[:, 0] * 0.12 * near_mouth,
                                           0.014 * near_mouth * _smooth(0.015, 0.05, np.abs(rel[:, 0])), 0 * rel[:, 0]], 1)
    eye_y = rel[:, 1] - 0.012
    for nm, sgn in (("eyeBlinkLeft", 1), ("eyeBlinkRight", -1)):
        eyew = np.exp(-(((rel[:, 0] - sgn * 0.045) / 0.035) ** 2 + ((eye_y) / 0.030) ** 2)) * front
        d = np.zeros_like(pos)
        d[:, 1] = -eye_y * 0.95 * eyew
        head.targets[nm] = d
    head.targets["browInnerUp"] = np.stack([0 * rel[:, 0], 0.012 * np.exp(-(((rel[:, 0]) / 0.05) ** 2 + ((rel[:, 1] - 0.045) / 0.03) ** 2)) * front, 0 * rel[:, 0]], 1)
    meshes.append(head)

    # ---- hair tails for the anime / ai look ----
    if style in ("anime", "ai"):
        hairm = _Mesh(3)
        for sgn in (-1, 1):
            a = hc + np.array([sgn * 0.095, 0.02, -0.03], np.float32)
            b = a + np.array([sgn * 0.045, -0.26, -0.02], np.float32)
            p, nr, uv_, ix, t = _tube(a, b, 0.040, 0.012, rows=7, cols=10)
            hairm.add(p, nr, uv_, np.tile([jidx["Head"], 0, 0, 0], (len(p), 1)), np.tile([1, 0, 0, 0], (len(p), 1)), ix)
        meshes.append(hairm)

    # ---- body ----
    def chain(mesh: _Mesh, a_name: str, b_name: str, r0: float, r1: float, rows=6, cols=14):
        p, nr, uv_, ix, t = _tube(J[a_name], J[b_name], r0, r1, rows, cols)
        w_b = _smooth(0.55, 1.0, t)
        jw = np.stack([np.full(len(p), jidx[a_name]), np.full(len(p), jidx[b_name]), np.zeros(len(p)), np.zeros(len(p))], 1)
        ww = np.stack([1 - w_b, w_b, np.zeros(len(p)), np.zeros(len(p))], 1)
        mesh.add(p, nr, uv_, jw, ww, ix)

    torso = _Mesh(2)
    chain(torso, "Hips", "Spine", 0.145, 0.150)
    chain(torso, "Spine", "Spine1", 0.150, 0.150)
    chain(torso, "Spine1", "Spine2", 0.150, 0.165)
    chain(torso, "Spine2", "Neck", 0.165, 0.150, rows=5)
    for side in ("Left", "Right"):
        chain(torso, f"{side}Shoulder", f"{side}Arm", 0.066, 0.064, rows=3)
        chain(torso, f"{side}Arm", f"{side}ForeArm", 0.060, 0.052)
        for joint, radius in ((f"{side}Arm", 0.068), (f"{side}ForeArm", 0.053)):
            p, nr, uv_, ix = _ellipsoid(J[joint], (radius,) * 3, rows=10, cols=12)
            torso.add(p, nr, uv_, np.tile([jidx[joint], 0, 0, 0], (len(p), 1)), np.tile([1, 0, 0, 0], (len(p), 1)), ix)
    meshes.append(torso)
    limbs = _Mesh(1)
    chain(limbs, "Neck", "Head", 0.052, 0.048, rows=3)
    for side in ("Left", "Right"):
        chain(limbs, f"{side}ForeArm", f"{side}Hand", 0.046, 0.034)
        p, nr, uv_, ix = _ellipsoid(J[f"{side}Hand"], (0.040, 0.050, 0.025), rows=10, cols=12)
        limbs.add(p, nr, uv_, np.tile([jidx[f"{side}Hand"], 0, 0, 0], (len(p), 1)), np.tile([1, 0, 0, 0], (len(p), 1)), ix)
    meshes.append(limbs)
    return mats, meshes, jidx, J


def build_glb(style: str = "human") -> bytes:
    import cv2

    style = style if style in STYLE else "human"
    mats, meshes, jidx, J = _build(style)
    bin_chunks: List[bytes] = []
    views, accessors = [], []

    def add_view(data: bytes, target=None) -> int:
        while len(b"".join(bin_chunks)) % 4:
            bin_chunks.append(b"\0")
        off = len(b"".join(bin_chunks))
        bin_chunks.append(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_acc(arr: np.ndarray, ctype: int, typ: str, target=None, minmax=False) -> int:
        bv = add_view(arr.tobytes(), target)
        acc = {"bufferView": bv, "componentType": ctype, "count": int(len(arr)), "type": typ}
        if minmax:
            acc["min"] = arr.min(0).tolist(); acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    nodes: List[dict] = []
    for name in ORDER:
        parent = JOINTS[name][0]
        local = J[name] - (J[parent] if parent else 0)
        nodes.append({"name": name, "translation": [float(x) for x in local], "children": []})
    for name in ORDER:
        parent = JOINTS[name][0]
        if parent:
            nodes[jidx[parent]]["children"].append(jidx[name])
    # inverse bind matrices (column-major), pure translations
    ibm = np.zeros((len(ORDER), 4, 4), np.float32)
    for name, i in jidx.items():
        ibm[i] = np.eye(4); ibm[i][:3, 3] = -J[name]
    ibm_acc = add_acc(ibm.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32), 5126, "MAT4")
    skin = {"joints": list(range(len(ORDER))), "inverseBindMatrices": ibm_acc, "skeleton": 0}
    gl_meshes = []
    for m in meshes:
        pos, nor, uv, jw, ww, idx = m.done()
        attrs = {"POSITION": add_acc(pos, 5126, "VEC3", 34962, True), "NORMAL": add_acc(nor, 5126, "VEC3", 34962),
                 "TEXCOORD_0": add_acc(uv, 5126, "VEC2", 34962), "JOINTS_0": add_acc(jw.astype(np.uint16), 5123, "VEC4", 34962),
                 "WEIGHTS_0": add_acc(ww.astype(np.float32), 5126, "VEC4", 34962)}
        prim = {"attributes": attrs, "indices": add_acc(idx.astype(np.uint32), 5125, "SCALAR", 34963), "material": m.material}
        if m.targets:
            names = list(m.targets)
            prim["targets"] = [{"POSITION": add_acc(m.targets[k].astype(np.float32), 5126, "VEC3")} for k in names]
        mesh = {"name": ["Wolf3D_Head", "Hair", "Body", "Limbs"][min(len(gl_meshes), 3)] if False else f"part{len(gl_meshes)}",
                "primitives": [prim]}
        if m.targets:
            mesh["name"] = "Head"
            mesh["extras"] = {"targetNames": list(m.targets)}
        gl_meshes.append(mesh)
    base = len(nodes)
    for i, mesh in enumerate(gl_meshes):
        nodes.append({"name": mesh["name"] + f"_{i}", "mesh": i, "skin": 0})
    tex_png = cv2.imencode(".png", _face_texture(style))[1].tobytes()
    img_view = add_view(tex_png)
    materials = []
    for m in mats:
        if m.get("tex"):
            materials.append({"name": m["name"], "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.8}})
        else:
            materials.append({"name": m["name"], "pbrMetallicRoughness": {"baseColorFactor": list(m["color"]) + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8}})
    doc = {"asset": {"version": "2.0", "generator": "CinePulse AI demo avatars"},
           "scene": 0, "scenes": [{"nodes": [0] + list(range(base, len(nodes)))}], "nodes": nodes, "skins": [skin],
           "meshes": gl_meshes, "materials": materials, "textures": [{"source": 0}],
           "images": [{"bufferView": img_view, "mimeType": "image/png"}], "accessors": accessors, "bufferViews": views}
    if style == "anime":  # make it a VRM 0.x file: humanoid table + vowel blend-shape groups
        head_mesh = next(i for i, m in enumerate(gl_meshes) if m.get("extras"))
        names = gl_meshes[head_mesh]["extras"]["targetNames"]
        t = {n: i for i, n in enumerate(names)}

        def grp(preset, binds):
            return {"name": preset, "presetName": preset, "binds": [{"mesh": head_mesh, "index": t[k], "weight": w} for k, w in binds]}
        bone_map = {"Hips": "hips", "Spine": "spine", "Spine1": "chest", "Spine2": "upperChest", "Neck": "neck", "Head": "head",
                    "LeftShoulder": "leftShoulder", "LeftArm": "leftUpperArm", "LeftForeArm": "leftLowerArm", "LeftHand": "leftHand",
                    "RightShoulder": "rightShoulder", "RightArm": "rightUpperArm", "RightForeArm": "rightLowerArm", "RightHand": "rightHand"}
        for n in doc["nodes"][:len(ORDER)]:
            n["name"] = "J_Bip_" + n["name"]   # VRoid-style names: only the humanoid table can find them
        doc["extensions"] = {"VRM": {"humanoid": {"humanBones": [{"bone": v, "node": jidx[k]} for k, v in bone_map.items()]},
                                      "blendShapeMaster": {"blendShapeGroups": [
                                          grp("A", [("viseme_aa", 100)]), grp("I", [("viseme_I", 100)]), grp("U", [("viseme_U", 100)]),
                                          grp("E", [("viseme_E", 100)]), grp("O", [("viseme_O", 100)]),
                                          grp("Blink", [("eyeBlinkLeft", 100), ("eyeBlinkRight", 100)]), grp("Joy", [("mouthSmile", 100)])]}}}
        doc["extensionsUsed"] = ["VRM"]
        # a VRM has no viseme_* targets exposed: rename so only the VRM groups provide mouth shapes
        for mesh in gl_meshes:
            if mesh.get("extras"):
                mesh["extras"]["targetNames"] = [f"vrm_target_{i}" for i in range(len(names))]
        # keep the real names reachable for the bind table above by index (already resolved to indices)
    blob = b"".join(bin_chunks)
    blob += b"\0" * (-len(blob) % 4)
    doc["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob


def write_demo_avatar(style: str, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_glb(style))
    return str(p)


DEMO_STYLES = {"human": "Demo human", "ai": "Demo AI model", "anime": "Demo anime (VRM)"}


def ensure_demo_avatars(folder: str) -> Dict[str, str]:
    """Write the three built-in avatars into `folder` (once) and return {style: path}."""
    out = {}
    for style in DEMO_STYLES:
        target = Path(folder) / f"demo_{style}.{'vrm' if style == 'anime' else 'glb'}"
        if not target.is_file():
            write_demo_avatar(style, str(target))
        out[style] = str(target)
    return out
