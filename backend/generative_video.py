"""
backend/generative_video.py

Generative AI video for the Episode Studio: turns a shot (locked character + location pictures, action, line)
into MOVING frames produced by a video model, instead of a still plate with a camera move.

Backends, tried in this order when backend="auto" (the first one that is available and succeeds wins):

  ltx          local Lightricks LTX-Video (diffusers LTXImageToVideoPipeline). Image-to-video: the model animates the
               shot's locked first frame (scenery + character reference pictures), so the references stay in force.
  svd          local Stability AI Stable Video Diffusion (diffusers StableVideoDiffusionPipeline), image-to-video.
  pollinations pollinations.ai video (gen.pollinations.ai/image/{prompt}?model=...). It needs an API key
               (POLLINATIONS_API_KEY) and most of its video models are paid. It is TEXT-to-video: the pictures are
               not fed to the model, so it is the weakest tier for character / background consistency.

If none is usable (no GPU / no weights / no key / no internet) the caller falls back to the 2D cinematic
composition and the reason is reported - nothing is faked.

Honest limits: LTX and SVD need a CUDA GPU and several GB of model weights (CPU only when
CINEPULSE_GENVIDEO_ALLOW_CPU=1, which is extremely slow). Weights are never downloaded unless allow_download=True.
Everything heavy is imported lazily; this module imports only numpy at load time.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BACKENDS = ("ltx", "svd", "pollinations")
MODES = ("2d", "generative")
LTX_MODEL = "Lightricks/LTX-Video"
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
POLLINATIONS_VIDEO_URL = "https://gen.pollinations.ai/image/{prompt}?model={model}&duration={duration}&aspectRatio={aspect}"
POLLINATIONS_HOST = "gen.pollinations.ai"
MAX_VIDEO_BYTES = 120 * 1024 * 1024
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, deformed, extra limbs, text, watermark"


class GenVideoError(RuntimeError):
    """A generative-video backend failed."""


class BackendUnavailable(GenVideoError):
    """A backend cannot run here (missing package / GPU / weights / key / internet)."""


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def build_motion_prompt(shot: dict, location_card: Optional[dict] = None, character_cards: Sequence[dict] = ()) -> str:
    """One text prompt per shot from the storyboard shot and the prompt cards of its locked assets."""
    parts: List[str] = []
    if shot.get("action"):
        parts.append(str(shot["action"]).strip().rstrip("."))
    elif shot.get("line"):
        parts.append("a character speaks: " + str(shot["line"]).strip()[:120])
    for c in character_cards:
        bits = [c.get("name", "")]
        traits = c.get("traits") or {}
        bits += [str(v) for k, v in traits.items() if v and k in ("hair", "eyes", "build", "age")]
        if c.get("attire"):
            bits.append(str(c["attire"]))
        parts.append("character " + ", ".join(b for b in bits if b))
    loc = location_card or {}
    place = loc.get("name") or shot.get("location")
    if place:
        extra = ", ".join(str(x) for x in (loc.get("lighting"), loc.get("atmosphere"), shot.get("time_of_day")) if x)
        parts.append(f"set in {place}" + (f", {extra}" if extra else ""))
    if shot.get("kind") == "establishing":
        parts.append("slow establishing camera move")
    parts.append("cinematic, natural motion, consistent characters, high quality")
    return ". ".join(p for p in parts if p)[:900]


# --------------------------------------------------------------------------- #
# frames helpers
# --------------------------------------------------------------------------- #
def resample_frames(frames: Sequence[np.ndarray], n: int, src_fps: float = 24.0, fps: float = 24.0, pingpong: bool = True) -> List[np.ndarray]:
    """n output frames (played at `fps`) from model frames recorded at `src_fps`: frames are blended in time, and a model
    clip shorter than the shot is played forward then backward (ping-pong) instead of jumping back to its first frame."""
    if not frames:
        raise GenVideoError("The video model returned no frames.")
    src = list(frames)
    L = len(src)
    if L == 1:
        return [src[0].copy() for _ in range(n)]
    ratio = float(src_fps) / max(1e-6, float(fps))
    period = 2 * (L - 1) if pingpong else L - 1
    out = []
    for i in range(n):
        p = i * ratio
        if pingpong:
            p = p % period
            if p > L - 1:
                p = period - p
        else:
            p = min(p, L - 1)
        a = int(math.floor(p))
        b = min(L - 1, a + 1)
        w = p - a
        out.append(src[a].copy() if w < 1e-3 else (src[a].astype(np.float32) * (1 - w) + src[b].astype(np.float32) * w).astype(np.uint8))
    return out


def _fit_frame(fr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Cover-crop a BGR frame to size (w, h)."""
    import cv2

    W, H = size
    h, w = fr.shape[:2]
    k = max(W / w, H / h)
    nw, nh = max(W, int(round(w * k))), max(H, int(round(h * k)))
    fr = cv2.resize(fr, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return fr[y0:y0 + H, x0:x0 + W]


def write_video(frames: Sequence[np.ndarray], path: str, fps: float, ffmpeg: Optional[str] = None) -> str:
    """H.264 silent video from BGR frames (ffmpeg if given, else OpenCV's mp4v writer)."""
    import cv2

    if not frames:
        raise GenVideoError("There are no frames to write.")
    h, w = frames[0].shape[:2]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)],
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        try:
            for fr in frames:
                proc.stdin.write(np.ascontiguousarray(fr).tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "ignore")
            rc = proc.wait()
        except BrokenPipeError:
            rc, err = 1, proc.stderr.read().decode("utf-8", "ignore")
        finally:
            for pipe in (proc.stdin, proc.stderr):
                try:
                    pipe.close()
                except Exception:  # noqa: BLE001
                    pass
        if rc != 0 or not out.is_file():
            raise GenVideoError(f"ffmpeg could not write the generated video: {err[-300:]}")
        return str(out)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise GenVideoError("OpenCV could not open a video writer.")
    for fr in frames:
        vw.write(fr)
    vw.release()
    return str(out)


def read_video_frames(path: str, limit: int = 600) -> Tuple[List[np.ndarray], float]:
    """(frames, fps) of a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while len(frames) < limit:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise GenVideoError("The generated video could not be decoded.")
    return frames, src_fps


def _pil_to_bgr(img) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def _round_to(v: float, m: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v / m) * m)))


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def internet_reachable(host: str = POLLINATIONS_HOST, port: int = 443, timeout: float = 4.0) -> bool:
    """A TCP probe; the graceful-fallback loop uses it so that an offline machine fails fast, not after a long timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _weights_cached(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def capabilities(api_key: Optional[str] = None, check_internet: bool = False) -> List[dict]:
    """What each backend needs and whether this machine looks ready: [{"backend","ready","detail"}].
    Cheap (no model is loaded, torch is not imported unless diffusers is present)."""
    out = []
    have_local = _has("diffusers") and _has("torch")
    gpu = have_local and _cuda()
    allow_cpu = os.environ.get("CINEPULSE_GENVIDEO_ALLOW_CPU") == "1"
    for name, repo in (("ltx", LTX_MODEL), ("svd", SVD_MODEL)):
        if not have_local:
            out.append({"backend": name, "ready": False, "detail": "needs the optional packages diffusers + torch"})
        elif not (gpu or allow_cpu):
            out.append({"backend": name, "ready": False, "detail": "needs a CUDA GPU (CPU only with CINEPULSE_GENVIDEO_ALLOW_CPU=1)"})
        elif not _weights_cached(repo):
            out.append({"backend": name, "ready": False, "detail": f"model weights {repo} are not downloaded yet"})
        else:
            out.append({"backend": name, "ready": True, "detail": "local model ready"})
    key = api_key or os.environ.get("POLLINATIONS_API_KEY")
    if not key:
        out.append({"backend": "pollinations", "ready": False, "detail": "needs a pollinations.ai API key (POLLINATIONS_API_KEY)"})
    elif check_internet and not internet_reachable():
        out.append({"backend": "pollinations", "ready": False, "detail": "no internet connection"})
    else:
        out.append({"backend": "pollinations", "ready": True, "detail": "API key set (video models are mostly paid)"})
    return out


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
_PIPES: Dict[str, object] = {}


def unload_pipelines() -> None:
    """Free the video models' memory."""
    _PIPES.clear()
    import gc

    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class DiffusersVideo:
    """Local image-to-video with LTX-Video or Stable Video Diffusion through diffusers."""

    def __init__(self, kind: str, pipe_factory: Optional[Callable[[str], object]] = None, allow_download: bool = False,
                 steps: Optional[int] = None):
        if kind not in ("ltx", "svd"):
            raise ValueError("kind must be 'ltx' or 'svd'")
        self.kind = kind
        self.name = kind
        self.pipe_factory = pipe_factory
        self.allow_download = allow_download
        self.steps = steps

    def check(self) -> None:
        if self.pipe_factory is not None:
            return
        if not (_has("diffusers") and _has("torch")):
            raise BackendUnavailable(f"{self.kind}: the optional packages diffusers and torch are not installed")
        if not (_cuda() or os.environ.get("CINEPULSE_GENVIDEO_ALLOW_CPU") == "1"):
            raise BackendUnavailable(f"{self.kind}: no CUDA GPU found")
        repo = os.environ.get("CINEPULSE_LTX_MODEL", LTX_MODEL) if self.kind == "ltx" else os.environ.get("CINEPULSE_SVD_MODEL", SVD_MODEL)
        if not self.allow_download and not _weights_cached(repo):
            raise BackendUnavailable(f"{self.kind}: model weights {repo} are not downloaded (downloads are off)")

    def _pipe(self):
        if self.pipe_factory is not None:
            return self.pipe_factory(self.kind)
        if self.kind in _PIPES:
            return _PIPES[self.kind]
        import torch  # type: ignore

        gpu = torch.cuda.is_available()
        dtype = torch.bfloat16 if (gpu and self.kind == "ltx") else (torch.float16 if gpu else torch.float32)
        local = not self.allow_download
        if self.kind == "ltx":
            from diffusers import LTXImageToVideoPipeline  # type: ignore

            pipe = LTXImageToVideoPipeline.from_pretrained(os.environ.get("CINEPULSE_LTX_MODEL", LTX_MODEL), torch_dtype=dtype,
                                                           local_files_only=local)
        else:
            from diffusers import StableVideoDiffusionPipeline  # type: ignore

            kw = {"variant": "fp16"} if gpu else {}
            pipe = StableVideoDiffusionPipeline.from_pretrained(os.environ.get("CINEPULSE_SVD_MODEL", SVD_MODEL), torch_dtype=dtype,
                                                                local_files_only=local, **kw)
        if gpu:
            pipe.enable_model_cpu_offload()  # fits consumer GPUs
        _PIPES[self.kind] = pipe
        return pipe

    def generate(self, init_image: str, prompt: str, size: Tuple[int, int], seconds: float, fps: float, seed: int = 0,
                 negative: str = NEGATIVE_PROMPT) -> List[np.ndarray]:
        from PIL import Image

        W, H = size
        img = Image.open(init_image).convert("RGB")
        pipe = self._pipe()
        try:
            gen = None
            try:
                import torch  # type: ignore

                gen = torch.Generator("cpu").manual_seed(int(seed))
            except Exception:  # noqa: BLE001
                pass
            long_side = 768 if self.kind == "ltx" else 1024
            k = min(1.0, long_side / max(W, H))
            if self.kind == "ltx":
                w, h = _round_to(W * k, 32, 256, 1280), _round_to(H * k, 32, 256, 1280)
                native_fps = 24.0
                n = 8 * max(1, min(12, math.ceil(seconds * native_fps / 8))) + 1  # frames = 8k + 1 (LTX requirement)
                res = pipe(image=img.resize((w, h)), prompt=prompt, negative_prompt=negative, width=w, height=h, num_frames=n,
                           num_inference_steps=self.steps or 30, guidance_scale=3.0, generator=gen)
            else:
                w, h = _round_to(W * k, 64, 256, 1024), _round_to(H * k, 64, 256, 1024)
                native_fps = 7.0
                res = pipe(image=img.resize((w, h)), width=w, height=h, num_frames=25, decode_chunk_size=4, motion_bucket_id=127,
                           noise_aug_strength=0.02, fps=7, num_inference_steps=self.steps or 25, generator=gen)
            pil_frames = res.frames[0]
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenVideoError(f"{self.kind} failed: {str(exc)[:200]}") from exc
        frames = [_fit_frame(_pil_to_bgr(f), size) for f in pil_frames]
        return resample_frames(frames, max(2, int(round(seconds * fps))), native_fps, fps, pingpong=True)


def _http_get_video(url: str, headers: dict, timeout: float = 300.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "CinePulseAI-episode-studio", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https endpoint
        return resp.read(MAX_VIDEO_BYTES + 1)


class PollinationsVideo:
    """pollinations.ai text-to-video. Needs an API key; sends it as a Bearer header (never in the URL)."""

    name = "pollinations"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 transport: Optional[Callable[[str, dict], bytes]] = None, probe: Optional[Callable[[], bool]] = None):
        self.api_key = api_key or os.environ.get("POLLINATIONS_API_KEY")
        self.model = model or os.environ.get("CINEPULSE_VIDEO_MODEL", "ltx-2")
        self.transport = transport
        self.probe = probe or internet_reachable

    def check(self) -> None:
        if not self.api_key:
            raise BackendUnavailable("pollinations: no API key (set POLLINATIONS_API_KEY; its video models are mostly paid)")
        if self.transport is None and not self.probe():
            raise BackendUnavailable("pollinations: no internet connection")

    def generate(self, init_image: str, prompt: str, size: Tuple[int, int], seconds: float, fps: float, seed: int = 0,
                 negative: str = NEGATIVE_PROMPT) -> List[np.ndarray]:
        duration = int(max(1, min(10, math.ceil(seconds))))
        aspect = "9:16" if size[1] > size[0] else "16:9"
        url = POLLINATIONS_VIDEO_URL.format(prompt=urllib.parse.quote(prompt[:700], safe=""), model=urllib.parse.quote(self.model),
                                            duration=duration, aspect=aspect)
        try:
            data = (self.transport or _http_get_video)(url, {"Authorization": f"Bearer {self.api_key}"})
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
            if code in (401, 402, 403):
                raise BackendUnavailable(f"pollinations rejected the request (HTTP {code}): check the API key and balance") from None
            raise GenVideoError(f"pollinations request failed: {str(exc)[:160]}") from None
        if not data or len(data) > MAX_VIDEO_BYTES:
            raise GenVideoError("pollinations returned no video (or one that is too large).")
        with tempfile.TemporaryDirectory(prefix="pollvid_") as tmp:
            p = Path(tmp) / "clip.mp4"
            p.write_bytes(data)
            frames, src_fps = read_video_frames(str(p))
        frames = [_fit_frame(f, size) for f in frames]
        return resample_frames(frames, max(2, int(round(seconds * fps))), src_fps, fps, pingpong=True)


def make_backend(name: str, **kw):
    if name in ("ltx", "svd"):
        return DiffusersVideo(name, pipe_factory=kw.get("pipe_factory"), allow_download=bool(kw.get("allow_download")),
                              steps=kw.get("steps"))
    if name == "pollinations":
        return PollinationsVideo(api_key=kw.get("api_key"), transport=kw.get("transport"), probe=kw.get("probe"))
    raise ValueError(f"Unknown video backend '{name}'. Use one of: {', '.join(BACKENDS)}.")


# --------------------------------------------------------------------------- #
# the fallback loop
# --------------------------------------------------------------------------- #
class VideoFactory:
    """Makes the generative clip of each shot, walking the backend list with graceful fallback.

    A backend that is unavailable or fails is marked down for the rest of the episode (so an offline machine or a
    missing GPU is detected once, not once per shot); every reason is kept in `.warnings`. shot_video() returns None
    when no backend produced a clip - the caller then renders that shot as the 2D composition."""

    def __init__(self, backend: str = "auto", allow_download: bool = False, api_key: Optional[str] = None,
                 pipe_factory: Optional[Callable[[str], object]] = None, transport: Optional[Callable[[str, dict], bytes]] = None,
                 probe: Optional[Callable[[], bool]] = None, steps: Optional[int] = None, ffmpeg: Optional[str] = None):
        if backend != "auto" and backend not in BACKENDS:
            raise ValueError(f"Unknown video backend '{backend}'. Use 'auto' or one of: {', '.join(BACKENDS)}.")
        order = list(BACKENDS) if backend == "auto" else [backend]
        kw = dict(allow_download=allow_download, api_key=api_key, pipe_factory=pipe_factory, transport=transport, probe=probe, steps=steps)
        self.backends = [make_backend(n, **kw) for n in order]
        self.down: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.ffmpeg = ffmpeg
        self.used: List[dict] = []

    @property
    def usable(self) -> bool:
        return any(b.name not in self.down for b in self.backends)

    def _mark(self, name: str, why: str) -> None:
        self.down[name] = why
        self.warnings.append(f"Generative video - {why}")

    def shot_video(self, shot_idx: int, init_image: Optional[str], prompt: str, size: Tuple[int, int], seconds: float, fps: float,
                   out_path: str, seed: int = 0) -> Optional[dict]:
        for b in self.backends:
            if b.name in self.down:
                continue
            if b.name in ("ltx", "svd") and not init_image:
                continue
            t0 = time.time()
            try:
                b.check()
                frames = b.generate(init_image or "", prompt, size, seconds, fps, seed=seed)
                write_video(frames, out_path, fps, self.ffmpeg)
            except BackendUnavailable as exc:
                self._mark(b.name, str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                self._mark(b.name, f"{b.name} failed on shot {shot_idx} ({str(exc)[:140]}); skipped for the rest of this episode")
                continue
            rec = {"shot": shot_idx, "backend": b.name, "path": out_path, "frames": len(frames),
                   "seconds_taken": round(time.time() - t0, 2), "image_conditioned": b.name in ("ltx", "svd")}
            self.used.append(rec)
            return rec
        return None

    def summary(self) -> dict:
        return {"clips": list(self.used), "down": dict(self.down)}
