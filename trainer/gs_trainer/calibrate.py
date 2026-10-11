"""LiDAR noise, measured: sigma as a function of ARKit confidence, distance and depth edges (plan P4).

The thesis this phase follows gave its Kinect a fixed 10 mm sigma and named the uncalibrated noise model as
its main weakness. The iPad's LiDAR is a different sensor: 256x192 depth upsampled by ARKit from a sparse
dot pattern, with a per-pixel confidence (0/1/2), errors that grow with distance and smeared depth at
object edges. Inverse-variance fusion is only as good as these sigmas, so they are measured here, on our
own scans, rather than guessed.

The reference is the TSDF mesh's depth (the same raycast `gs_trainer.priors` writes). The mesh is built from
the LiDAR itself, but by averaging hundreds of views per voxel, so one frame's deviation from it is mostly
that frame's noise. What it can't separate is the mesh's own error (smoothing at edges, missing thin
things, pose drift between sessions), so the numbers are an upper bound on the LiDAR's noise, which is the
safe side for a fusion weight. Statistics are robust (1.4826 x MAD, p90) so the few pixels where the mesh is
simply absent or wrong don't dominate.

The comparison runs at the LiDAR's native resolution (256x192: no upsampled duplicates), over a sample of
video frames spread evenly through the dataset. Mesh depth comes from `prior/<stem>.npz` when it exists,
else it is raycast on the fly at the depth map's size.

Output `prior/sigma.json`:
- `d_edges`: distance bin edges (m) of the LiDAR reading;
- `sigma[conf]`, `p90[conf]`, `bias[conf]`, `n[conf]`: per confidence level ("0", "1", "2") and distance
  bin, away from depth edges. `sigma` is 1.4826 x MAD of (LiDAR - mesh), `p90` the 90th percentile of
  |LiDAR - mesh - bias|, `bias` the median. Bins with too few pixels borrow from their nearest neighbour;
- `edge_factor[conf]`: sigma at depth edges / sigma away from them (edges found on the mesh depth);
- `edge` and `flat`: the same statistics pooled over distance, for the report.

`lidar_sigma(table, conf, depth, edge)` turns the table into a per-pixel sigma for `gs_trainer.fusion`.

    python -m gs_trainer.calibrate --data data/<project>/splat/datasets/default [--frames 150]
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .data import Frame, Scene, load_depth, load_prior, scaled_k

VERSION = 1
D_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.5)  # LiDAR depth bins (m); ARKit reads to about 5 m
MIN_PIXELS = 200  # a bin with fewer pixels borrows its neighbour's sigma
SIGMA_FLOOR = 0.002  # 2 mm: below that, it's the uint16 mm quantisation and the mesh's own error
EDGE_REL = 0.04  # a neighbour more than 4 % nearer or farther marks a depth edge
MAD_SCALE = 1.4826  # MAD -> sigma for a normal distribution


def edge_mask(depth: np.ndarray, rel: float = EDGE_REL, radius: int = 1) -> np.ndarray:
    """Pixels within `radius` of a depth discontinuity, (H, W) bool; 0 = no depth, never an edge itself.

    A discontinuity is a 4-neighbour whose depth differs by more than `rel` x the pixel's own depth: a
    relative threshold, because a 2 cm step is an edge at 50 cm and noise at 5 m. Holes don't make edges
    (a missing reading next to a valid one says nothing about geometry).
    """
    d = depth.astype(np.float32)
    valid = d > 0
    jump = np.zeros(d.shape, bool)
    for a, b in (
        (np.s_[:, 1:], np.s_[:, :-1]),
        (np.s_[1:, :], np.s_[:-1, :]),
    ):
        both = valid[a] & valid[b]
        diff = np.abs(d[a] - d[b]) > rel * np.minimum(d[a], d[b])
        step = both & diff
        jump[a] |= step
        jump[b] |= step
    if radius > 0:
        k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        jump = cv2.dilate(jump.astype(np.uint8), k).astype(bool)
    return jump & valid


def robust_stats(r: np.ndarray) -> dict[str, float | int]:
    """n, bias (median), sigma (1.4826 x MAD) and p90 of |r - bias| of residuals `r` (metres)."""
    if r.size == 0:
        return {"n": 0, "bias": 0.0, "sigma": 0.0, "p90": 0.0}
    bias = float(np.median(r))
    dev = np.abs(r - bias)
    return {
        "n": int(r.size),
        "bias": round(bias, 5),
        "sigma": round(float(MAD_SCALE * np.median(dev)), 5),
        "p90": round(float(np.percentile(dev, 90)), 5),
    }


def sample_frames(scene: Scene, n: int) -> list[Frame]:
    """Up to `n` video frames with LiDAR depth, evenly spread through `scene.frames` (all sessions)."""
    video = [f for f in scene.frames if f.depth_path is not None]
    if len(video) <= n:
        return video
    pick = np.linspace(0, len(video) - 1, n).round().astype(int)
    return [video[i] for i in pick]


def mesh_depths(scene: Scene, frames: list[Frame], sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """Mesh z-depth (H, W) float32 per frame at `sizes` (W, H): from the priors, else raycast now."""
    out: list[np.ndarray | None] = []
    todo: list[int] = []
    for i, (f, (w, h)) in enumerate(zip(frames, sizes, strict=True)):
        prior = load_prior(f, w, h)
        out.append(prior[0].numpy() if prior is not None else None)
        if prior is None:
            todo.append(i)
    if todo:
        import open3d as o3d  # only here: tests and fusion import this module without needing a raycast

        from . import priors

        tf = json.loads((scene.data_dir / "transforms.json").read_text())
        rc = priors.load_mesh(scene.data_dir / tf.get("mesh_file_path", "mesh.ply"))
        for i in todo:
            w, h = sizes[i]
            r, _ = priors.rays(_resized(frames[i], w, h), max(w, h))  # exactly the depth map's pixels
            t = rc.cast_rays(o3d.core.Tensor.from_numpy(r), nthreads=0)["t_hit"].numpy()
            out[i] = np.where(np.isfinite(t), t, 0.0).astype(np.float32).reshape(h, w)
    return [x for x in out if x is not None]


def _resized(f: Frame, w: int, h: int) -> Frame:
    """`f` as if its image were w x h (K scaled), so `priors.rays` casts exactly at that size."""
    return replace(f, K=scaled_k(f.K, f.w, f.h, w, h), w=w, h=h)


def lidar_sigma(
    table: dict, conf: np.ndarray, depth: np.ndarray, edge: np.ndarray | None = None
) -> np.ndarray:
    """Per-pixel LiDAR sigma (m) from a `sigma.json` table: by confidence and distance bin, x edge factor.

    Confidence levels missing from the table get the worst measured level's sigma; distances past the
    last bin get the last bin's.
    """
    edges = np.asarray(table["d_edges"], np.float64)
    b = np.clip(np.digitize(depth, edges[1:-1]), 0, len(edges) - 2)
    worst = max(table["sigma"], key=lambda c: float(np.mean(table["sigma"][c])))
    sigma = np.zeros(depth.shape, np.float32)
    for c in np.unique(conf):
        row = np.asarray(table["sigma"].get(str(int(c)), table["sigma"][worst]), np.float32)
        m = conf == c
        s = row[b[m]]
        if edge is not None:
            factor = float(table.get("edge_factor", {}).get(str(int(c)), 1.0))
            s = np.where(edge[m], s * factor, s)
        sigma[m] = s
    return np.maximum(sigma, SIGMA_FLOOR)


def _fill(values: list[float], counts: list[int]) -> list[float]:
    """Bins under MIN_PIXELS take the nearest populated bin's value (all populated bins if none is)."""
    ok = [i for i, n in enumerate(counts) if n >= MIN_PIXELS]
    if not ok:
        ok = [i for i, n in enumerate(counts) if n > 0]
    if not ok:
        return values
    return [values[i] if i in ok else values[min(ok, key=lambda j: abs(j - i))] for i in range(len(values))]


def calibrate(data: Path, frames: int = 150, d_edges: tuple[float, ...] = D_EDGES) -> dict:
    """Measure the LiDAR's sigma table on `frames` sampled video frames; write and return `prior/sigma.json`."""
    data = Path(data)
    started = time.monotonic()
    scene = Scene.load(data)
    sample = sample_frames(scene, frames)
    if not sample:
        raise ValueError(f"{data}: no video frames with LiDAR depth to calibrate on")

    lidar: list[tuple[np.ndarray, np.ndarray]] = []
    sizes: list[tuple[int, int]] = []
    for f in sample:
        assert f.depth_path is not None
        raw = cv2.imread(str(f.depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"can't read depth {f.depth_path}")
        h, w = raw.shape[:2]
        got = load_depth(f, w, h)  # native size: no upsampling
        assert got is not None
        lidar.append((got[0].numpy(), got[1].numpy()))
        sizes.append((w, h))
    meshes = mesh_depths(scene, sample, sizes)

    edges = np.asarray(d_edges, np.float64)
    res: dict[int, list[np.ndarray]] = {}  # conf -> per-bin residual lists (flat)
    res_edge: dict[int, list[np.ndarray]] = {}
    res_flat: dict[int, list[np.ndarray]] = {}
    for (d, conf), m in zip(lidar, meshes, strict=True):
        ok = (d > 0) & (m > 0)
        edge = edge_mask(m)
        r = d - m
        b = np.digitize(d, edges[1:-1])
        inside = (d >= edges[0]) & (d < edges[-1])
        for c in (0, 1, 2):
            sel = ok & (conf == c) & inside
            res_edge.setdefault(c, []).append(r[sel & edge])
            res_flat.setdefault(c, []).append(r[sel & ~edge])
            bins = res.setdefault(c, [np.empty(0, np.float32) for _ in range(len(edges) - 1)])
            flat = sel & ~edge
            for k in range(len(edges) - 1):
                bins[k] = np.concatenate([bins[k], r[flat & (b == k)]])

    table: dict = {
        "version": VERSION,
        "reference": "mesh",
        "frames": len(sample),
        "d_edges": [float(x) for x in edges],
        "sigma": {},
        "p90": {},
        "bias": {},
        "n": {},
        "edge_factor": {},
        "edge": {},
        "flat": {},
    }
    for c in (0, 1, 2):
        stats = [robust_stats(x) for x in res[c]]
        counts = [int(s["n"]) for s in stats]
        if sum(counts) == 0:
            continue
        key = str(c)
        table["n"][key] = counts
        table["sigma"][key] = [max(SIGMA_FLOOR, v) for v in _fill([float(s["sigma"]) for s in stats], counts)]
        table["p90"][key] = _fill([float(s["p90"]) for s in stats], counts)
        table["bias"][key] = _fill([float(s["bias"]) for s in stats], counts)
        e = robust_stats(np.concatenate(res_edge[c]))
        fl = robust_stats(np.concatenate(res_flat[c]))
        table["edge"][key], table["flat"][key] = e, fl
        enough = int(e["n"]) >= MIN_PIXELS and float(fl["sigma"]) > 0
        table["edge_factor"][key] = (
            round(max(1.0, float(e["sigma"]) / float(fl["sigma"])), 3) if enough else 1.0
        )
    if not table["sigma"]:
        raise ValueError(f"{data}: no pixel has both a LiDAR reading and a mesh hit")
    table["seconds"] = round(time.monotonic() - started, 2)

    out = data / "prior"
    out.mkdir(exist_ok=True)
    (out / "sigma.json").write_text(json.dumps(table, indent=1) + "\n")
    return table


def format_table(table: dict) -> str:
    """The sigma table as plain text: one row per confidence level, sigma / p90 in mm per distance bin."""
    e = table["d_edges"]
    head = "conf  " + "  ".join(f"{e[i]:.1f}-{e[i + 1]:.1f}m".rjust(13) for i in range(len(e) - 1))
    lines = [head + "   edge x", "      " + "  ".join("sigma/p90 mm".rjust(13) for _ in range(len(e) - 1))]
    for c in sorted(table["sigma"]):
        cells = []
        for i in range(len(e) - 1):
            n = table["n"][c][i]
            cell = f"{table['sigma'][c][i] * 1000:.1f}/{table['p90'][c][i] * 1000:.0f}" if n else "-"
            cells.append(f"{cell} ({n // 1000}k)".rjust(13) if n else cell.rjust(13))
        lines.append(f"{c:>4}  " + "  ".join(cells) + f"   {table['edge_factor'][c]:.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gs_trainer.calibrate", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument("--data", type=Path, required=True, help="dataset v2 directory (depth/, mesh.ply)")
    ap.add_argument("--frames", type=int, default=150, help="video frames to sample (default 150)")
    args = ap.parse_args(argv)
    table = calibrate(args.data, args.frames)
    print(format_table(table))
    print(f"wrote {args.data / 'prior' / 'sigma.json'} ({table['frames']} frames, {table['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
