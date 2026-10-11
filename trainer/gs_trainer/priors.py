"""Mesh priors: depth and normals of the TSDF mesh, raycast into every frame (plan P2).

The LiDAR is the measured truth but it is coarse (256x192), has holes near edges and on dark or shiny
surfaces, and stills have none. The TSDF mesh (`mesh.ply`, from `fuse/dense.ply`) is dense, consistent
across views, sharp at edges and exists for every frame, so it is the depth loss's dense companion and the
source of the normals the normal loss compares against.

For each frame (train, interp and extrap: the evaluator scores depth on the held-out views too) this
writes `prior/<stem>.npz`:
- `depth` float16 (H, W): z-depth in metres along the camera's optical axis (what gsplat's "ED" render
  mode and ARKit report), 0 where the ray misses the mesh;
- `normal` int8 (H, W, 3): the hit triangle's normal in the world (scene) frame, flipped to face the
  camera, times 127; 0 on a miss.
H, W = the image shrunk to max side `side` (`data.scaled_size`, never enlarged). `prior/meta.json` records
the side and a version; a different side or version means the priors are stale and are rebuilt. Frames
whose npz is already there are skipped, so an interrupted run resumes.

Raycasting is Open3D's `RaycastingScene` (Embree BVH, all CPU cores). Rays go through pixel centres
(u + 0.5, v + 0.5) with a camera-frame z of 1, so the hit distance is the z-depth directly. Several frames
share one `cast_rays` call so the threads stay busy.

    python -m gs_trainer.priors --data data/<project>/splat/datasets/default [--side 640]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open3d as o3d

from .data import Frame, Scene, scaled_k, scaled_size

VERSION = 1
RAYS_PER_BATCH = 4_000_000


def rays(frame: Frame, side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """(rays (H*W, 6) float32 origin + direction with camera z = 1, (W, H)) for one frame."""
    w, h = scaled_size(frame.w, frame.h, side)
    k = scaled_k(frame.K, frame.w, frame.h, w, h)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    cam = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    d = cam @ frame.c2w[:3, :3].T
    o = np.broadcast_to(frame.c2w[:3, 3], d.shape)
    return np.concatenate([o, d], 1).astype(np.float32), (w, h)


def encode(t_hit: np.ndarray, normals: np.ndarray, dirs: np.ndarray, w: int, h: int) -> dict[str, np.ndarray]:
    """Raycast results for one frame -> the npz arrays (depth float16, normal int8)."""
    hit = np.isfinite(t_hit)
    depth = np.where(hit, t_hit, 0.0).astype(np.float16)
    flip = np.where((normals * dirs).sum(1) > 0, -1.0, 1.0)[:, None]  # face the camera
    n = np.where(hit[:, None], normals * flip, 0.0)
    normal = np.clip(np.round(n * 127), -127, 127).astype(np.int8)
    return {"depth": depth.reshape(h, w), "normal": normal.reshape(h, w, 3)}


def load_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.t.io.read_triangle_mesh(str(path))
    if mesh.triangle.indices.shape[0] == 0:
        raise ValueError(f"{path} has no triangles")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh.vertex.positions.to(o3d.core.float32), mesh.triangle.indices.to(o3d.core.uint32))
    return scene


def prepare(data: Path, side: int = 640) -> dict:
    """Write `prior/<stem>.npz` for every frame of the dataset in `data` that lacks one; returns the meta."""
    data = Path(data)
    started = time.monotonic()
    scene = Scene.load(data)
    tf = json.loads((data / "transforms.json").read_text())
    mesh_path = data / tf.get("mesh_file_path", "mesh.ply")
    out = data / "prior"
    out.mkdir(exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.is_file():
        old = json.loads(meta_path.read_text())
        if old.get("side") != side or old.get("version") != VERSION:
            for p in out.glob("*.npz"):
                p.unlink()
    todo = [f for f in scene.frames if not (out / f"{f.stem}.npz").is_file()]
    skipped = len(scene.frames) - len(todo)

    raycast_s = 0.0
    load_s = 0.0
    if todo:
        t0 = time.monotonic()
        rc = load_mesh(mesh_path)
        load_s = time.monotonic() - t0

        def write(
            f: Frame, r: np.ndarray, size: tuple[int, int], t_hit: np.ndarray, normals: np.ndarray
        ) -> None:
            arrays = encode(t_hit, normals, r[:, 3:], *size)
            tmp = out / f"{f.stem}.tmp.npz"
            np.savez_compressed(tmp, depth=arrays["depth"], normal=arrays["normal"])
            os.replace(tmp, out / f"{f.stem}.npz")  # whole files only: a killed run resumes cleanly

        # Batches of whole frames, ~RAYS_PER_BATCH rays each. Ray set-up, encoding and zlib run on a thread
        # pool between the casts, not during them: cast_rays keeps every core busy (and the GIL).
        batches: list[list[Frame]] = [[]]
        n_rays = 0
        for f in todo:
            w, h = scaled_size(f.w, f.h, side)
            if batches[-1] and n_rays + w * h > RAYS_PER_BATCH:
                batches.append([])
                n_rays = 0
            batches[-1].append(f)
            n_rays += w * h
        with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
            for batch in batches:
                made = list(pool.map(lambda f: rays(f, side), batch))
                t0 = time.monotonic()
                hits = rc.cast_rays(
                    o3d.core.Tensor.from_numpy(np.concatenate([r for r, _ in made])), nthreads=0
                )
                t_hit = hits["t_hit"].numpy().copy()
                normals = hits["primitive_normals"].numpy().copy()
                raycast_s += time.monotonic() - t0
                ends = np.cumsum([len(r) for r, _ in made])
                starts = np.concatenate([[0], ends[:-1]])
                jobs = [
                    pool.submit(write, f, r, size, t_hit[a:b], normals[a:b])
                    for f, (r, size), a, b in zip(batch, made, starts, ends, strict=True)
                ]
                for j in jobs:
                    j.result()  # re-raise a failed write

    seconds = time.monotonic() - started
    files = list(out.glob("*.npz"))
    meta = {
        "version": VERSION,
        "side": side,
        "frames": len(scene.frames),
        "computed": len(todo),
        "skipped": skipped,
        "mesh": mesh_path.name,
        "seconds": round(seconds, 2),
        "mesh_load_s": round(load_s, 2),
        "raycast_s": round(raycast_s, 2),
        "frames_per_s": round(len(todo) / seconds, 2) if todo and seconds > 0 else None,
        "mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gs_trainer.priors", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument(
        "--data", type=Path, required=True, help="dataset v2 directory (transforms.json, mesh.ply)"
    )
    ap.add_argument("--side", type=int, default=640, help="max image side of the priors (default 640)")
    args = ap.parse_args(argv)
    meta = prepare(args.data, args.side)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
