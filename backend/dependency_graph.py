"""
dependency_graph.py - graph-based visual dependency tracking (ViMax pattern) for CinePulse AI.

ViMax keeps a growing library of reference images (character portraits + frames already generated) and, for every
new shot, selects the references the renderer must see so characters and places do not drift. This module is that
idea as a small, dependency-free directed graph:

    char:Mara --anchors--> ref:Mara (portrait) --references--> shot:3
    shot:2 --produces--> kf:2:last --continues--> shot:3          (the keyframe handed to the next render step)
    scene:1 --located_in--> shot:3

An edge  A -> B  means "B needs A first". Shots are therefore rendered in topological order and each shot is handed
exactly the keyframes it depends on (`KeyframeStore.inputs_for`). After rendering, `check_drift` compares how each
character looks in every keyframe with the anchor portrait and flags drift.

Pure Python (+ lazily numpy/OpenCV for the appearance signature). Nothing here needs a GPU.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NODE_KINDS = ("character", "reference", "scene", "shot", "keyframe")
EDGE_KINDS = ("anchors", "references", "located_in", "produces", "continues", "feeds")
MAX_REFERENCES = 8  # ViMax selects at most 8 reference images per frame


class GraphError(ValueError):
    pass


class DependencyGraph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self._out: Dict[str, List[dict]] = defaultdict(list)
        self._in: Dict[str, List[dict]] = defaultdict(list)

    # ---- construction --------------------------------------------------- #
    def add_node(self, node_id: str, kind: str, **attrs) -> str:
        if kind not in NODE_KINDS:
            raise GraphError(f"Unknown node kind '{kind}'. Use one of: {', '.join(NODE_KINDS)}.")
        if node_id in self.nodes:
            self.nodes[node_id].update(attrs)
            return node_id
        self.nodes[node_id] = {"id": node_id, "kind": kind, **attrs}
        return node_id

    def add_edge(self, src: str, dst: str, kind: str, **attrs) -> dict:
        if kind not in EDGE_KINDS:
            raise GraphError(f"Unknown edge kind '{kind}'. Use one of: {', '.join(EDGE_KINDS)}.")
        for n in (src, dst):
            if n not in self.nodes:
                raise GraphError(f"Unknown node '{n}'.")
        if src == dst:
            raise GraphError("A node cannot depend on itself.")
        for e in self._out[src]:
            if e["dst"] == dst and e["kind"] == kind:
                e.update(attrs)
                return e
        edge = {"src": src, "dst": dst, "kind": kind, **attrs}
        self.edges.append(edge)
        self._out[src].append(edge)
        self._in[dst].append(edge)
        return edge

    # ---- queries -------------------------------------------------------- #
    def predecessors(self, node_id: str, kind: Optional[str] = None) -> List[str]:
        return [e["src"] for e in self._in.get(node_id, []) if kind is None or e["kind"] == kind]

    def successors(self, node_id: str, kind: Optional[str] = None) -> List[str]:
        return [e["dst"] for e in self._out.get(node_id, []) if kind is None or e["kind"] == kind]

    def in_edges(self, node_id: str) -> List[dict]:
        return list(self._in.get(node_id, []))

    def ancestors(self, node_id: str) -> List[str]:
        seen, stack, order = set(), [node_id], []
        while stack:
            for p in self.predecessors(stack.pop()):
                if p not in seen:
                    seen.add(p)
                    order.append(p)
                    stack.append(p)
        return order

    def topological_order(self) -> List[str]:
        """Kahn's algorithm, stable (ties keep insertion order). Raises GraphError on a cycle."""
        indeg = {n: len(self._in.get(n, [])) for n in self.nodes}
        queue = deque(n for n in self.nodes if indeg[n] == 0)
        order: List[str] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for e in self._out.get(n, []):
                indeg[e["dst"]] -= 1
                if indeg[e["dst"]] == 0:
                    queue.append(e["dst"])
        if len(order) != len(self.nodes):
            stuck = sorted(n for n in self.nodes if n not in set(order))
            raise GraphError(f"Dependency cycle among: {', '.join(stuck[:6])}")
        return order

    def has_cycle(self) -> bool:
        try:
            self.topological_order()
            return False
        except GraphError:
            return True

    def levels(self) -> Dict[str, int]:
        """Longest-path depth of every node (nodes on the same level do not depend on each other)."""
        depth: Dict[str, int] = {}
        for n in self.topological_order():
            depth[n] = max((depth[p] + 1 for p in self.predecessors(n)), default=0)
        return depth

    # ---- (de)serialisation ---------------------------------------------- #
    def to_dict(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        g = cls()
        for n in data.get("nodes", []):
            n = dict(n)
            g.add_node(n.pop("id"), n.pop("kind"), **n)
        for e in data.get("edges", []):
            e = dict(e)
            g.add_edge(e.pop("src"), e.pop("dst"), e.pop("kind"), **e)
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_mermaid(self) -> str:
        """A Mermaid flowchart of the graph (paste into any Mermaid viewer)."""
        def mid(n: str) -> str:
            return re.sub(r"\W", "_", n)
        lines = ["flowchart LR"]
        for n in self.nodes.values():
            label = n.get("label") or n["id"]
            lines.append(f'  {mid(n["id"])}["{label}"]')
        for e in self.edges:
            lines.append(f'  {mid(e["src"])} -->|{e["kind"]}| {mid(e["dst"])}')
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# building the graph from a storyboard
# --------------------------------------------------------------------------- #
def _key(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def portrait_view_for(shot: dict, character: str) -> str:
    """front / side / back portrait best matching how the camera sees the character (ViMax reference selection)."""
    text = f"{shot.get('camera_angle', '')} {shot.get('shot_type', '')} {shot.get('visual_desc', '')}".casefold()
    if re.search(r"from behind|back to (?:the )?camera|over[- ]the[- ]shoulder", text):
        near = shot.get("near_character")
        if near is None or near == character:
            return "back"
    if re.search(r"\bprofile\b|\bside view\b|in profile", text):
        return "side"
    return "front"


def build_from_storyboard(shots: Sequence[dict], characters: Sequence[dict],
                          max_references: int = MAX_REFERENCES) -> DependencyGraph:
    """Link character anchors, scene backdrops and keyframes of earlier shots to every shot that needs them.

    For each shot, in story order:
      * every character in the shot gets an edge from its anchor portrait  (identity)
      * the shot that CONTINUES the previous one (same scene, shared character or location) gets an edge from the
        previous shot's last keyframe                                        (continuity: the first frame is handed on)
      * every other visible character also gets an edge from the last keyframe in which it appeared (recency),
        so its look is carried across cuts and scenes                         (no drift)
    Edges that would push a shot over `max_references` are dropped, anchors and continuity first."""
    g = DependencyGraph()
    for c in characters:
        name = _key(c["name"])
        g.add_node(f"char:{name}", "character", label=name)
        g.add_node(f"ref:{name}", "reference", label=f"{name} portrait", character=name)
        g.add_edge(f"char:{name}", f"ref:{name}", "anchors")
    known = {_key(c["name"]) for c in characters}
    last_seen: Dict[str, int] = {}
    prev: Optional[dict] = None
    for sh in shots:
        i = int(sh["idx"])
        sid = f"shot:{i}"
        g.add_node(sid, "shot", label=f"Shot {i}", scene=sh.get("scene_idx"), start=sh.get("start"), end=sh.get("end"))
        scene_id = f"scene:{sh.get('scene_idx', 0)}"
        g.add_node(scene_id, "scene", label=f"Scene {sh.get('scene_idx', 0)}")
        g.add_edge(scene_id, sid, "located_in")
        g.add_node(f"kf:{i}:last", "keyframe", label=f"Shot {i} last frame", shot=i)
        g.add_edge(sid, f"kf:{i}:last", "produces")
        refs: List[Tuple[str, str, dict]] = []  # (src, edge kind, attrs) in priority order
        chars = [_key(c) for c in sh.get("characters", []) if _key(c) in known]
        for c in chars:
            refs.append((f"ref:{c}", "references", {"role": "anchor", "character": c, "view": portrait_view_for(sh, c)}))
        continues = False
        if prev is not None and prev.get("scene_idx") == sh.get("scene_idx"):
            shared = set(map(_key, prev.get("characters", []))) & set(chars)
            continues = bool(shared) or prev.get("location") == sh.get("location")
        if continues:
            refs.append((f"kf:{prev['idx']}:last", "continues", {"role": "continuity"}))
        for c in chars:
            j = last_seen.get(c)
            if j is not None and not (continues and prev is not None and j == prev["idx"]):
                refs.append((f"kf:{j}:last", "references", {"role": "recent", "character": c}))
        for src, kind, attrs in refs[:max_references]:
            g.add_edge(src, sid, kind, **attrs)
        g.nodes[sid]["continues_from"] = prev["idx"] if continues else None
        for c in chars:
            last_seen[c] = i
        prev = sh
    g.topological_order()  # raises on cycles
    return g


def render_plan(graph: DependencyGraph) -> List[dict]:
    """Shots in render order, each with the inputs (anchors / keyframes) it must be given."""
    out = []
    depth = graph.levels()
    for n in graph.topological_order():
        if graph.nodes[n]["kind"] != "shot":
            continue
        inputs = [{"node": e["src"], "kind": e["kind"], "role": e.get("role"), "character": e.get("character"),
                   "view": e.get("view")} for e in graph.in_edges(n) if e["kind"] in ("references", "continues")]
        out.append({"shot": int(n.split(":")[1]), "level": depth[n], "inputs": inputs,
                    "continues_from": graph.nodes[n].get("continues_from")})
    return out


# --------------------------------------------------------------------------- #
# keyframes + drift
# --------------------------------------------------------------------------- #
class KeyframeStore:
    """node id -> image file. The renderer puts keyframes in, and asks which ones a shot should receive."""

    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.paths: Dict[str, str] = {}

    def put(self, node_id: str, path: str) -> None:
        if node_id not in self.graph.nodes:
            raise GraphError(f"Unknown node '{node_id}'.")
        self.paths[node_id] = str(path)

    def inputs_for(self, shot_idx: int) -> dict:
        """{"available": [{node, path, role, character, view}], "missing": [node ids not produced yet]}"""
        sid = f"shot:{shot_idx}"
        if sid not in self.graph.nodes:
            raise GraphError(f"Unknown shot {shot_idx}.")
        avail, missing = [], []
        for e in self.graph.in_edges(sid):
            if e["kind"] not in ("references", "continues"):
                continue
            p = self.paths.get(e["src"])
            if p:
                avail.append({"node": e["src"], "path": p, "role": e.get("role"), "character": e.get("character"),
                              "view": e.get("view")})
            else:
                missing.append(e["src"])
        return {"available": avail, "missing": missing}

    def ready(self, shot_idx: int) -> bool:
        """True when every upstream keyframe the shot depends on has been produced (anchors may be absent)."""
        return not [m for m in self.inputs_for(shot_idx)["missing"] if m.startswith("kf:")]


def appearance_signature(image, region: Optional[Tuple[int, int, int, int]] = None):
    """A 60-bin hue/saturation/value histogram (normalised) of an image or of an (x, y, w, h) region of it."""
    import cv2
    import numpy as np

    img = cv2.imread(str(image)) if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
    if img is None:
        raise GraphError(f"Could not read image: {image}")
    if region:
        x, y, w, h = (int(v) for v in region)
        img = img[max(0, y):y + h, max(0, x):x + w]
    if img.size == 0:
        raise GraphError("The region is empty.")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    parts = [cv2.calcHist([hsv], [c], None, [n], [0, r]).ravel() for c, n, r in ((0, 24, 180), (1, 18, 256), (2, 18, 256))]
    sig = np.concatenate(parts).astype(np.float64)
    return sig / max(sig.sum(), 1e-9)


def signature_distance(a, b) -> float:
    """Bhattacharyya distance between two signatures, 0 (identical) .. 1 (nothing in common)."""
    import numpy as np

    bc = float(np.sum(np.sqrt(np.asarray(a) * np.asarray(b))))  # signatures are normalised to sum to 1
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, bc))))


def check_drift(anchors: Dict[str, object], observations: Iterable[dict], threshold: float = 0.35) -> List[dict]:
    """Compare how each character looks in rendered keyframes with its anchor signature.

    anchors: {character: signature};  observations: [{"shot", "character", "signature"}]
    -> [{"shot","character","distance","status": "ok" | "drift"}] (worst first)."""
    out = []
    for ob in observations:
        ref = anchors.get(ob["character"])
        if ref is None:
            continue
        d = signature_distance(ref, ob["signature"])
        out.append({"shot": ob["shot"], "character": ob["character"], "distance": round(d, 4),
                    "status": "drift" if d > threshold else "ok"})
    return sorted(out, key=lambda r: -r["distance"])
