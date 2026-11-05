"""
backend/soft_raster.py  -  a small CPU triangle rasteriser (numpy only; no GPU, no OpenGL, no PyTorch).

    rasterize(sx, sy, zc, tris, width, height, two_sided=None, alpha_test=None)
        -> RasterResult(tri_id (H,W) int32 [-1 = empty], bary (H,W,3) float32, depth (H,W) float32)

Perspective-correct barycentrics + a z-buffer. Triangles are grouped by the size of their screen
bounding box, and every group is tested against a fixed pixel window in ONE vectorised numpy
operation, so a 30k-triangle avatar needs a few dozen numpy calls per frame instead of a Python loop
per triangle. Pixel centres sit at +0.5; front faces are counter-clockwise on screen when y points up
(i.e. a negative signed area in y-down pixel coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

MAX_CANDIDATES = 1_500_000  # pixel tests per numpy call (bounds peak memory)
NEAR = 1e-4


@dataclass
class RasterResult:
    tri_id: np.ndarray
    bary: np.ndarray
    depth: np.ndarray


def rasterize(sx, sy, zc, tris, width: int, height: int, two_sided: Optional[np.ndarray] = None,
              alpha_test: Optional[Callable] = None, alpha_tris: Optional[np.ndarray] = None) -> RasterResult:
    """Rasterise `tris` (T,3 vertex indices) whose vertices sit at pixel (sx,sy) with camera depth zc (>0).

    two_sided   bool (T,) - draw back faces too (hair cards, thin cloth); default culls them
    alpha_test  fn(tri_ids, bary(n,3)) -> bool keep-mask, called only for the triangles flagged in
                `alpha_tris` (bool (T,)) before the depth test, so cut-out textures work"""
    sx = np.asarray(sx, np.float32)
    sy = np.asarray(sy, np.float32)
    zc = np.asarray(zc, np.float32)
    tris = np.asarray(tris, np.int64)
    T = len(tris)
    zbuf = np.full(width * height, np.inf, np.float32)
    tri_buf = np.full(width * height, -1, np.int32)
    bary_buf = np.zeros((width * height, 3), np.float32)
    if T == 0:
        return RasterResult(tri_buf.reshape(height, width), bary_buf.reshape(height, width, 3),
                            zbuf.reshape(height, width))

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ax, ay, bx, by, cx, cy = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    front = area < 0
    keep = (np.abs(area) > 1e-8) & (zc[a] > NEAR) & (zc[b] > NEAR) & (zc[c] > NEAR)
    keep &= front if two_sided is None else (front | np.asarray(two_sided))
    minx = np.minimum(np.minimum(ax, bx), cx)
    maxx = np.maximum(np.maximum(ax, bx), cx)
    miny = np.minimum(np.minimum(ay, by), cy)
    maxy = np.maximum(np.maximum(ay, by), cy)
    x0 = np.ceil(minx - 0.5).astype(np.int64)
    x1 = np.floor(maxx - 0.5).astype(np.int64)
    y0 = np.ceil(miny - 0.5).astype(np.int64)
    y1 = np.floor(maxy - 0.5).astype(np.int64)
    keep &= (x1 >= 0) & (y1 >= 0) & (x0 < width) & (y0 < height) & (x1 >= x0) & (y1 >= y0)
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return RasterResult(tri_buf.reshape(height, width), bary_buf.reshape(height, width, 3),
                            zbuf.reshape(height, width))
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)
    x1 = np.minimum(x1, width - 1)
    y1 = np.minimum(y1, height - 1)
    size = np.maximum(x1 - x0, y1 - y0) + 1
    inv_z = (1.0 / zc).astype(np.float32)
    # barycentric weights are affine in the pixel position: w_i(px,py) = c_i + a_i*px + b_i*py
    inv_area = (1.0 / np.where(keep, area, 1.0)).astype(np.float32)
    coef = {}
    for name, (p, q) in (("0", ((bx, by), (cx, cy))), ("1", ((cx, cy), (ax, ay)))):
        (px_, py_), (qx_, qy_) = p, q
        coef[name] = ((px_ * qy_ - py_ * qx_) * inv_area, (py_ - qy_) * inv_area, (qx_ - px_) * inv_area)

    def run(ids: np.ndarray, ox: np.ndarray, oy: np.ndarray, kx: int, ky: int):
        per = kx * ky
        chunk = max(1, MAX_CANDIDATES // per)
        ox = ox.astype(np.float32)[None, :]
        oy = oy.astype(np.float32)[None, :]
        for s in range(0, len(ids), chunk):
            t = ids[s:s + chunk]
            bx0 = (x0[t] + 0.5).astype(np.float32)
            by0 = (y0[t] + 0.5).astype(np.float32)
            c0, a0, b0 = (v[t] for v in coef["0"])
            c1, a1, b1 = (v[t] for v in coef["1"])
            base0 = (c0 + a0 * bx0 + b0 * by0)[:, None]
            base1 = (c1 + a1 * bx0 + b1 * by0)[:, None]
            w0 = base0 + a0[:, None] * ox + b0[:, None] * oy
            w1 = base1 + a1[:, None] * ox + b1[:, None] * oy
            w2 = 1.0 - w0 - w1
            eps = np.float32(-1e-4)
            inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            inside &= (ox <= (x1[t] - x0[t])[:, None]) & (oy <= (y1[t] - y0[t])[:, None])
            ti, pi = np.nonzero(inside)
            if not len(ti):
                continue
            tt = t[ti]
            W0, W1, W2 = w0[ti, pi], w1[ti, pi], w2[ti, pi]
            q0, q1, q2 = W0 * inv_z[a[tt]], W1 * inv_z[b[tt]], W2 * inv_z[c[tt]]
            z = 1.0 / (q0 + q1 + q2)
            pb = np.stack([q0, q1, q2], 1) * z[:, None]
            key = ((y0[tt] + oy[0, pi].astype(np.int64)) * width + x0[tt] + ox[0, pi].astype(np.int64))
            if alpha_test is not None and alpha_tris is not None:
                flagged = alpha_tris[tt]
                if flagged.any():
                    ok = np.ones(len(tt), bool)
                    ok[flagged] = alpha_test(tt[flagged], pb[flagged])
                    tt, pb, z, key = tt[ok], pb[ok], z[ok], key[ok]
                    if not len(tt):
                        continue
            win = z < zbuf[key]
            if not win.any():
                continue
            tt, pb, z, key = tt[win], pb[win], z[win], key[win]
            order = np.argsort(-z, kind="stable")  # nearest written last -> it wins duplicate pixels
            key, z, tt, pb = key[order], z[order], tt[order], pb[order]
            zbuf[key] = z
            tri_buf[key] = tt
            bary_buf[key] = pb

    # small triangles: one fixed window per power-of-two size class
    for k in (2, 4, 8, 16, 32, 64):
        lo = k // 2 if k > 2 else 0
        sel = idx[(size[idx] > lo) & (size[idx] <= k)]
        if len(sel):
            g = np.arange(k)
            oy, ox = np.meshgrid(g, g, indexing="ij")
            run(sel, ox.ravel(), oy.ravel(), k, k)
    # big triangles: a window per triangle, few of them
    for t in idx[size[idx] > 64]:
        wx, wy = int(x1[t] - x0[t] + 1), int(y1[t] - y0[t] + 1)
        gy, gx = np.meshgrid(np.arange(wy), np.arange(wx), indexing="ij")
        run(np.array([t]), gx.ravel(), gy.ravel(), wx, wy)

    return RasterResult(tri_buf.reshape(height, width), bary_buf.reshape(height, width, 3),
                        zbuf.reshape(height, width))
