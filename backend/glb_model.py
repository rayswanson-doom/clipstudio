"""
backend/glb_model.py  -  load a .glb / .vrm 3D avatar (pure numpy + OpenCV; no three.js, no PyTorch).

    model = load_model("avatar.glb")
    model.bones        canonical rig name -> node index   (Hips, Spine, Head, LeftArm, ... Mixamo / Ready Player Me
                       names, "mixamorig:" prefixes, or the humanoid table of a VRM 0.x / 1.0 file)
    model.channels     expression channel -> [(primitive_index, morph_target_index, weight)]   (ARKit and Oculus
                       viseme blend-shape names, or VRM expression presets)
    model.prims        triangle meshes with positions / normals / uvs / skin weights / morph-target deltas
    model.materials, model.textures

Reads: binary glTF 2.0 (.glb) and VRM (.vrm = glTF with extensions). NOT supported: Draco / meshopt
compressed meshes (re-export without compression), separate .gltf + .bin files, sparse accessors on
morph targets are applied, KHR_texture_transform is ignored.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MAX_TEXTURE_SIDE = 1024
CANONICAL_BONES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot", "Jaw",
]
VRM_BONES = {
    "hips": "Hips", "spine": "Spine", "chest": "Spine1", "upperChest": "Spine2", "neck": "Neck", "head": "Head",
    "leftEye": "LeftEye", "rightEye": "RightEye", "jaw": "Jaw",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftArm", "leftLowerArm": "LeftForeArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightArm", "rightLowerArm": "RightForeArm", "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg", "leftLowerLeg": "LeftLeg", "leftFoot": "LeftFoot",
    "rightUpperLeg": "RightUpLeg", "rightLowerLeg": "RightLeg", "rightFoot": "RightFoot",
}
OCULUS_VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U"]
# VRM vowel presets -> our channel names
VRM_PRESET_CHANNEL = {"aa": "vrm_A", "ih": "vrm_I", "ou": "vrm_U", "ee": "vrm_E", "oh": "vrm_O",
                      "a": "vrm_A", "i": "vrm_I", "u": "vrm_U", "e": "vrm_E", "o": "vrm_O",
                      "blink": "blink", "blinkleft": "blinkLeft", "blinkright": "blinkRight",
                      "joy": "smile", "happy": "smile", "fun": "smile", "surprised": "surprised"}
_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class ModelError(ValueError):
    """The file is not a usable 3D avatar (message is meant for the user)."""


@dataclass
class Material:
    name: str = ""
    base_color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    texture: Optional[int] = None
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Primitive:
    mesh_index: int
    node: int
    skin: Optional[int]
    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: Optional[np.ndarray]
    joints: Optional[np.ndarray]
    weights: Optional[np.ndarray]
    tris: np.ndarray
    material: int
    morph_deltas: List[np.ndarray] = field(default_factory=list)
    morph_names: List[str] = field(default_factory=list)


@dataclass
class Skin:
    joints: List[int]
    inverse_bind: np.ndarray  # (J,4,4) column-vector convention


@dataclass
class Model:
    path: str
    names: List[str]
    parents: List[int]
    local_t: np.ndarray
    local_r: np.ndarray  # (N,3,3)
    local_s: np.ndarray
    order: List[int]
    prims: List[Primitive]
    skins: List[Skin]
    materials: List[Material]
    textures: List[np.ndarray]  # RGBA uint8
    bones: Dict[str, int]
    channels: Dict[str, List[Tuple[int, int, float]]]
    format: str = "glb"
    warnings: List[str] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return int(sum(len(p.positions) for p in self.prims))

    @property
    def triangle_count(self) -> int:
        return int(sum(len(p.tris) for p in self.prims))

    def has_visemes(self) -> bool:
        return any(f"viseme_{v}" in self.channels for v in ("aa", "O", "E"))

    def lip_sync_kind(self) -> str:
        if self.has_visemes():
            return "visemes"
        if any(k in self.channels for k in ("vrm_A", "vrm_I", "vrm_U", "vrm_E", "vrm_O")):
            return "vrm-vowels"
        if "jawOpen" in self.channels:
            return "jaw-open"
        return "none"


# --------------------------------------------------------------------------- #
# glTF container
# --------------------------------------------------------------------------- #
def _parse_container(data: bytes) -> Tuple[dict, bytes]:
    if len(data) < 20 or data[:4] != b"glTF":
        raise ModelError("This is not a binary glTF file. Export your avatar as .glb (or .vrm).")
    version, total = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ModelError(f"Unsupported glTF version {version}; export as glTF 2.0.")
    pos, doc, blob = 12, None, b""
    while pos + 8 <= min(len(data), total):
        length, kind = struct.unpack_from("<II", data, pos)
        chunk = data[pos + 8:pos + 8 + length]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942 and not blob:
            blob = bytes(chunk)
        pos += 8 + length + (-length % 4)
    if doc is None:
        raise ModelError("The .glb has no JSON chunk (file is damaged).")
    used = set(doc.get("extensionsRequired", []))
    for ext in ("KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_mesh_quantization"):
        if ext in used and ext != "KHR_mesh_quantization":
            raise ModelError("This model is compressed (%s). Re-export it without Draco/meshopt compression." % ext)
    return doc, blob


class _Reader:
    def __init__(self, doc: dict, blob: bytes):
        self.doc, self.blob = doc, blob

    def view(self, index: int) -> bytes:
        bv = self.doc["bufferViews"][index]
        if self.doc["buffers"][bv.get("buffer", 0)].get("uri"):
            raise ModelError("External .bin files are not supported; export a single .glb.")
        off = bv.get("byteOffset", 0)
        return self.blob[off:off + bv["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.doc["accessors"][index]
        dtype, n = _COMP[acc["componentType"]], _NCOMP[acc["type"]]
        count = acc["count"]
        if "bufferView" in acc:
            bv = self.doc["bufferViews"][acc["bufferView"]]
            raw = self.view(acc["bufferView"])
            start = acc.get("byteOffset", 0)
            item = np.dtype(dtype).itemsize * n
            stride = bv.get("byteStride") or item
            if stride == item:
                arr = np.frombuffer(raw, dtype, count * n, start).reshape(count, n)
            else:
                rows = np.frombuffer(raw, np.uint8, len(raw) - start, start)
                arr = np.stack([np.frombuffer(rows[i * stride:i * stride + item].tobytes(), dtype, n) for i in range(count)])
        else:
            arr = np.zeros((count, n), dtype)
        arr = arr.copy()
        sparse = acc.get("sparse")
        if sparse:
            idx_info, val_info = sparse["indices"], sparse["values"]
            idx = np.frombuffer(self.view(idx_info["bufferView"]), _COMP[idx_info["componentType"]], sparse["count"],
                                idx_info.get("byteOffset", 0))
            vals = np.frombuffer(self.view(val_info["bufferView"]), dtype, sparse["count"] * n,
                                 val_info.get("byteOffset", 0)).reshape(-1, n)
            arr[idx.astype(np.int64)] = vals
        if acc.get("normalized") and dtype != np.float32:
            info = np.iinfo(dtype)
            arr = np.maximum(arr.astype(np.float32) / float(info.max), -1.0) if info.min < 0 else arr.astype(np.float32) / float(info.max)
        return arr


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w or 1.0
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], np.float32)


def _decompose(m: np.ndarray):
    t = m[:3, 3].copy()
    sx, sy, sz = (float(np.linalg.norm(m[:3, i])) or 1.0 for i in range(3))
    r = m[:3, :3] / np.array([sx, sy, sz], np.float32)
    return t, r.astype(np.float32), np.array([sx, sy, sz], np.float32)


def _norm_bone(name: str) -> str:
    n = name.lower()
    n = re.sub(r"^(mixamorig|armature|bip\d*|j_bip_c|j_bip_l|j_bip_r)", "", n) if False else n
    n = n.replace("mixamorig", "").replace(":", "").replace("_", "").replace(" ", "").replace(".", "")
    return n


_BONE_LOOKUP = {b.lower(): b for b in CANONICAL_BONES}
_BONE_ALIASES = {"leftupperarm": "LeftArm", "rightupperarm": "RightArm", "leftlowerarm": "LeftForeArm",
                 "rightlowerarm": "RightForeArm", "chest": "Spine1", "upperchest": "Spine2", "leftupperleg": "LeftUpLeg",
                 "rightupperleg": "RightUpLeg", "leftlowerleg": "LeftLeg", "rightlowerleg": "RightLeg",
                 "jawbone": "Jaw", "lefteyeball": "LeftEye", "righteyeball": "RightEye"}


def _texture_from_view(raw: bytes) -> Optional[np.ndarray]:
    import cv2

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 1.0) * 255.0, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if max(h, w) > MAX_TEXTURE_SIDE:
        s = MAX_TEXTURE_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


# --------------------------------------------------------------------------- #
def load_model(path: str) -> Model:
    p = Path(path)
    if not p.is_file():
        raise ModelError(f"Model file not found: {path}")
    if p.suffix.lower() not in (".glb", ".vrm"):
        raise ModelError("Use a .glb or .vrm 3D avatar (export from Ready Player Me, VRoid Studio, Blender, ...).")
    doc, blob = _parse_container(p.read_bytes())
    rd = _Reader(doc, blob)
    warnings: List[str] = []
    nodes = doc.get("nodes", [])
    n = len(nodes)
    names = [nd.get("name") or f"node{i}" for i, nd in enumerate(nodes)]
    parents = [-1] * n
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parents[c] = i
    lt = np.zeros((n, 3), np.float32)
    lr = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    ls = np.ones((n, 3), np.float32)
    for i, nd in enumerate(nodes):
        if "matrix" in nd:
            lt[i], lr[i], ls[i] = _decompose(np.array(nd["matrix"], np.float32).reshape(4, 4).T)
        else:
            lt[i] = nd.get("translation", (0, 0, 0))
            if "rotation" in nd:
                lr[i] = _quat_to_mat(nd["rotation"])
            ls[i] = nd.get("scale", (1, 1, 1))
    order: List[int] = []
    seen = set()

    def visit(i: int):
        if i in seen:
            return
        seen.add(i)
        order.append(i)
        for c in nodes[i].get("children", []):
            visit(c)

    for i in range(n):
        if parents[i] == -1:
            visit(i)

    textures: List[np.ndarray] = []
    tex_of_gltf: Dict[int, Optional[int]] = {}
    for ti, tex in enumerate(doc.get("textures", [])):
        src = tex.get("source")
        if src is None:
            for ext in tex.get("extensions", {}).values():
                src = ext.get("source", src)
        arr = None
        if src is not None:
            img = doc["images"][src]
            if "bufferView" in img:
                try:
                    arr = _texture_from_view(rd.view(img["bufferView"]))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"texture {ti} could not be decoded ({exc})")
            else:
                warnings.append(f"texture {ti} is stored outside the file and was skipped")
        if arr is None:
            tex_of_gltf[ti] = None
        else:
            tex_of_gltf[ti] = len(textures)
            textures.append(arr)

    materials: List[Material] = []
    for m in doc.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        bt = pbr.get("baseColorTexture")
        materials.append(Material(
            name=m.get("name", ""), base_color=tuple(pbr.get("baseColorFactor", (1, 1, 1, 1))),
            texture=tex_of_gltf.get(bt["index"]) if bt else None,
            alpha_mode=m.get("alphaMode", "OPAQUE"), alpha_cutoff=float(m.get("alphaCutoff", 0.5)),
            double_sided=bool(m.get("doubleSided", False)), emissive=tuple(m.get("emissiveFactor", (0, 0, 0)))))
    materials.append(Material(name="default"))
    default_mat = len(materials) - 1

    skins: List[Skin] = []
    for s in doc.get("skins", []):
        joints = list(s["joints"])
        if "inverseBindMatrices" in s:
            ibm = rd.accessor(s["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float32)
        else:
            ibm = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
        skins.append(Skin(joints, ibm))

    prims: List[Primitive] = []
    mesh_prim_ids: Dict[int, List[int]] = {}
    for ni, nd in enumerate(nodes):
        if "mesh" not in nd:
            continue
        mesh = doc["meshes"][nd["mesh"]]
        mesh_target_names = (mesh.get("extras") or {}).get("targetNames")
        for pi, prim in enumerate(mesh.get("primitives", [])):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = rd.accessor(attrs["POSITION"]).astype(np.float32)
            nv = len(pos)
            if "indices" in prim:
                idx = rd.accessor(prim["indices"]).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(nv, dtype=np.int64)
            tris = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            if "NORMAL" in attrs:
                nor = rd.accessor(attrs["NORMAL"]).astype(np.float32)
            else:
                nor = np.zeros_like(pos)
                fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
                for k in range(3):
                    np.add.at(nor, tris[:, k], fn)
                nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
            uv = rd.accessor(attrs["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attrs else None
            joints = weights = None
            if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs and nd.get("skin") is not None:
                joints = rd.accessor(attrs["JOINTS_0"]).astype(np.int64)
                weights = rd.accessor(attrs["WEIGHTS_0"]).astype(np.float32)
                wsum = weights.sum(1, keepdims=True)
                weights = weights / np.where(wsum > 0, wsum, 1.0)
            deltas, tnames = [], []
            tn = (prim.get("extras") or {}).get("targetNames") or mesh_target_names or []
            for ti, target in enumerate(prim.get("targets", [])):
                deltas.append(rd.accessor(target["POSITION"]).astype(np.float32) if "POSITION" in target
                              else np.zeros_like(pos))
                tnames.append(tn[ti] if ti < len(tn) else f"target{ti}")
            mat = prim.get("material", default_mat)
            prims.append(Primitive(nd["mesh"], ni, nd.get("skin"), f"{mesh.get('name', 'mesh')}#{pi}", pos, nor, uv,
                                   joints, weights, tris, mat if mat is not None else default_mat, deltas, tnames))
            mesh_prim_ids.setdefault(nd["mesh"], []).append(len(prims) - 1)
    if not prims:
        raise ModelError("The model contains no triangle meshes.")

    bones, fmt = _find_bones(doc, names, warnings)
    if fmt:
        pass
    channels = _find_channels(doc, prims, mesh_prim_ids, nodes)
    return Model(str(p), names, parents, lt, lr, ls, order, prims, skins, materials, textures, bones, channels,
                 "vrm" if (doc.get("extensions", {}).get("VRM") or doc.get("extensions", {}).get("VRMC_vrm")) else "glb",
                 warnings)


def _find_bones(doc: dict, names: List[str], warnings: List[str]):
    bones: Dict[str, int] = {}
    ext = doc.get("extensions", {})
    vrm0 = (ext.get("VRM") or {}).get("humanoid", {}).get("humanBones")
    vrm1 = (ext.get("VRMC_vrm") or {}).get("humanoid", {}).get("humanBones")
    if vrm0:
        for hb in vrm0:
            if hb.get("bone") in VRM_BONES and hb.get("node") is not None:
                bones[VRM_BONES[hb["bone"]]] = int(hb["node"])
    if vrm1:
        for k, v in vrm1.items():
            if k in VRM_BONES and isinstance(v, dict) and "node" in v:
                bones[VRM_BONES[k]] = int(v["node"])
    for i, nm in enumerate(names):
        key = _norm_bone(nm)
        canon = _BONE_LOOKUP.get(key) or _BONE_ALIASES.get(key)
        if canon and canon not in bones:
            bones[canon] = i
    return bones, bool(vrm0 or vrm1)


def _find_channels(doc: dict, prims: List[Primitive], mesh_prims: Dict[int, List[int]], nodes: List[dict]):
    ch: Dict[str, List[Tuple[int, int, float]]] = {}

    def add(name: str, pi: int, ti: int, w: float = 1.0):
        ch.setdefault(name, []).append((pi, ti, float(w)))

    arkit_alias = {"eyeBlinkLeft": "blinkLeft", "eyeBlinkRight": "blinkRight", "mouthSmileLeft": "smileLeft",
                   "mouthSmileRight": "smileRight", "mouthSmile": "smile", "browInnerUp": "browInnerUp", "jawOpen": "jawOpen",
                   "mouthOpen": "jawOpen", "mouthClose": "mouthClose", "browOuterUpLeft": "browOuterUpLeft",
                   "browOuterUpRight": "browOuterUpRight", "eyeWideLeft": "eyeWideLeft", "eyeWideRight": "eyeWideRight",
                   "mouthFunnel": "mouthFunnel", "mouthPucker": "mouthPucker", "cheekSquintLeft": "cheekSquintLeft",
                   "cheekSquintRight": "cheekSquintRight"}
    for pi, pr in enumerate(prims):
        for ti, nm in enumerate(pr.morph_names):
            if nm.startswith("viseme_"):
                add(nm, pi, ti)
            elif nm in arkit_alias:
                add(arkit_alias[nm], pi, ti)
    # VRM expressions
    vrm0 = (doc.get("extensions", {}).get("VRM") or {}).get("blendShapeMaster", {}).get("blendShapeGroups")
    for grp in vrm0 or []:
        preset = str(grp.get("presetName", "")).lower()
        name = VRM_PRESET_CHANNEL.get(preset)
        if not name:
            continue
        for bind in grp.get("binds", []):
            for pi in mesh_prims.get(bind.get("mesh"), []):
                if bind["index"] < len(prims[pi].morph_deltas):
                    add(name, pi, bind["index"], float(bind.get("weight", 100)) / 100.0)
    vrm1 = (doc.get("extensions", {}).get("VRMC_vrm") or {}).get("expressions", {}).get("preset")
    for preset, ex in (vrm1 or {}).items():
        name = VRM_PRESET_CHANNEL.get(preset.lower())
        if not name:
            continue
        for bind in ex.get("morphTargetBinds", []):
            mesh = nodes[bind["node"]].get("mesh")
            for pi in mesh_prims.get(mesh, []):
                if bind["index"] < len(prims[pi].morph_deltas):
                    add(name, pi, bind["index"], float(bind.get("weight", 1.0)))
    return ch
