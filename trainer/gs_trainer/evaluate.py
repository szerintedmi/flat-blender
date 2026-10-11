"""Score any splat PLY (msplat, Brush, gsplat) on the shared split of dataset v2.

    python -m gs_trainer.evaluate --data D --ply P --out O [--sets interp,extrap] [--pose-align-steps 100]
                               [--pose-align-scale 0.5] [--views data/<project>/scene/views.json | --no-views]
                               [--max-side 960]
                               [--backend gsplat|torch]
    python -m gs_trainer.evaluate compare RUN_EVAL_DIR... --out DIR

Per test set (`split.json`: `interp`, `extrap`), every frame is rendered at `max_side` with its raw pose:
  PSNR, SSIM, LPIPS (AlexNet)     against the photo
  AbsRel, δ<1.05                  rendered expected depth against LiDAR, on video frames where
                                  confidence == 2, LiDAR > 0.1 m and rendered alpha > 0.5 (pooled per pixel)
  pose-aligned PSNR               after a per-frame 6-DoF correction fitted with the splat frozen, so runs
                                  that optimise poses during training stay comparable with those that don't;
                                  fitted at `pose_align_scale` x the image size (default half: a ~270-frame
                                  test set at 960 px x 200 steps took too long), scored at full size; it
                                  starts from the re-solved pose where there is one (its basin is ~10 cm)
  re-solved                       PSNR, SSIM, LPIPS and depth again at the frame's pose re-solved against the
                                  training frames (`test_poses.json`, written by `scanmodel splat`: SIFT matches
                                  to training frames lifted with their LiDAR, PnP), so a test frame's pose
                                  error (scan 5's lap drifted 48 cm) stops passing as splat quality. Means
                                  over the re-solved frames only (`n_resolved`; `n_unresolved` failed to
                                  re-solve and are left out), with the dataset-pose PSNR of the same frames
                                  beside it (`psnr_same_frames`); `pose_moved` lists the frames whose
                                  re-solved pose is > 10 cm or 2 deg from the dataset pose

`test_poses.json` is read from the dataset (or `--test-poses FILE`) and refused, with a warning, when the
transforms.json / split.json it was solved from (its `source` SHA-256s) differ from the dataset's.

`out/`: `eval.json`, `index.html`, `views/<name>.png` (the named views of `--views`, e.g.
`data/<project>/scene/views.json`) and
`worst/<set>/<stem>.jpg` (the 6 worst frames by PSNR: photo | render | depth error; `--samples N`: also N evenly
spaced frames in `samples/<set>/`). eval.json keeps every scored frame (`per_frame`, one line each, with its
`session`) and per set a `sessions` breakdown (the set's metrics per capture session, plus the median and lowest
PSNR and the worst frame). `compare` puts several runs' eval dirs on one page (metrics side by side, the
sessions matched across runs by room name, named views in a grid); it also reads eval.json written before
the breakdown (the session from the stem).
"""

import argparse
import hashlib
import heapq
import html
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .cloud import store_path
from .data import GL_CV, Frame, Images, Scene, load_depth, viewmat
from .losses import ssim
from .render import Backend, render

SETS = ("interp", "extrap")
POSE_ALIGN_STEPS = 100
POSE_ALIGN_SCALE = 0.5
WORST = 6
POSE_LR = 1e-3
POSE_SSIM_WEIGHT = 0.2
MIN_DEPTH = 0.1  # m: LiDAR readings closer than this are not trusted
MIN_ALPHA = 0.5
DELTA = 1.05
HEAT_MAX = 0.2  # relative depth error at the top of the heat map's colour scale
SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")
TEST_POSES = "test_poses.json"  # re-solved test poses (`scanmodel splat`, pipeline/scanmodel/testposes.py)
MOVED_CM, MOVED_DEG = 10.0, 2.0  # a re-solved pose this far from the dataset pose: listed in `pose_moved`

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


# --- PLY ---------------------------------------------------------------------------------------------


def read_ply_vertices(path: Path) -> np.ndarray:
    """The vertex element of a binary PLY as a structured array (any property order, any scalar types)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, elements, current = "", [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header has no end_header")
            words = line.decode("ascii", "replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                current = (words[1], int(words[2]), [])
                elements.append(current)
            elif words[0] == "property" and current is not None:
                if words[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported in element {current[0]}")
                current[2].append((words[2], _PLY_TYPES[words[1]]))
        data = f.read()
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"{path}: PLY format {fmt!r} is not supported (binary only)")
    if not elements or elements[0][0] != "vertex":
        raise ValueError(f"{path}: the first PLY element must be 'vertex' (compressed PLYs aren't supported)")
    _, n, props = elements[0]
    order = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(name, order + t) for name, t in props])
    return np.frombuffer(data, dtype=dtype, count=n)


def load_ply(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    """A standard 3DGS PLY as the contract's splat tensors (float32, CPU) plus its SH degree.

    Same layout as `gsplat.exporter.load_ply_to_splats` (f_rest channel-major), without plyfile, and
    tolerant of the extra properties (normals, ...) and property orders other trainers write.
    """
    v = read_ply_vertices(path)
    names = set(v.dtype.names or ())
    need = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "rot_0"}
    if missing := need - names:
        raise ValueError(f"{path}: not a 3DGS PLY, missing {sorted(missing)}")

    def cols(prefix: str) -> list[str]:
        return sorted((n for n in names if n.startswith(prefix)), key=lambda s: int(s.rsplit("_", 1)[1]))

    def stack(keys: Sequence[str]) -> np.ndarray:
        return np.stack([np.asarray(v[k], dtype=np.float32) for k in keys], axis=1)

    rest = cols("f_rest_")
    if len(rest) % 3:
        raise ValueError(f"{path}: {len(rest)} f_rest properties is not a multiple of 3")
    k = len(rest) // 3 + 1
    degree = round(math.sqrt(k)) - 1
    if (degree + 1) ** 2 != k:
        raise ValueError(f"{path}: {len(rest)} f_rest properties is no SH degree")
    n = len(v)
    shn = stack(rest).reshape(n, 3, k - 1).swapaxes(1, 2) if rest else np.zeros((n, 0, 3), np.float32)
    splats = {
        "means": stack(["x", "y", "z"]),
        "scales": stack(cols("scale_")),
        "quats": stack(cols("rot_")),
        "opacities": np.asarray(v["opacity"], dtype=np.float32),
        "sh0": stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :],
        "shN": shn,
    }
    return {key: torch.from_numpy(np.ascontiguousarray(a)) for key, a in splats.items()}, degree


def frozen(splats: Mapping[str, torch.Tensor], device: torch.device | str) -> torch.nn.ParameterDict:
    return torch.nn.ParameterDict({
        k: torch.nn.Parameter(splats[k].to(device), requires_grad=False) for k in SPLAT_KEYS
    })  # fmt: skip


# --- metrics -----------------------------------------------------------------------------------------


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.clamp(0, 1) - b) ** 2).mean())
    return -10 * math.log10(max(mse, 1e-10))


def ssim_hw3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM of two (H, W, 3) images in [0, 1] through `losses.ssim` (which takes (C, H, W, 3))."""
    return ssim(a[None], b[None])


_LPIPS: dict[str, torch.nn.Module] = {}


def lpips_model(device: torch.device) -> torch.nn.Module:
    """LPIPS with AlexNet (the torchvision weights are downloaded once into the torch hub cache)."""
    key = str(device)
    if key not in _LPIPS:
        import lpips

        _LPIPS[key] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[key]


def lpips_score(a: torch.Tensor, b: torch.Tensor) -> float:
    model = lpips_model(a.device)
    with torch.no_grad():
        x = a.clamp(0, 1).permute(2, 0, 1)[None] * 2 - 1
        y = b.permute(2, 0, 1)[None] * 2 - 1
        return float(model(x, y).mean())


def depth_valid(
    depth: torch.Tensor, gt: torch.Tensor, conf: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Pixels the depth metrics count: LiDAR confidence 2, LiDAR > 0.1 m, rendered alpha > 0.5."""
    return (conf == 2) & (gt > MIN_DEPTH) & (alpha > MIN_ALPHA) & torch.isfinite(depth)


def depth_errors(
    depth: torch.Tensor, gt: torch.Tensor, conf: torch.Tensor, alpha: torch.Tensor
) -> tuple[float, int, int]:
    """(sum of |D - gt| / gt, count with max(D/gt, gt/D) < 1.05, valid pixel count), all (H, W)."""
    m = depth_valid(depth, gt, conf, alpha)
    d, g = depth[m], gt[m]
    ratio = torch.maximum(d / g, g / d.clamp(min=1e-6))
    return float(((d - g).abs() / g).sum()), int((ratio < DELTA).sum()), int(m.sum())


# --- rendering and pose alignment --------------------------------------------------------------------


def render_one(
    splats: torch.nn.ParameterDict,
    w2c: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    sh_degree: int,
    backend: Backend,
    render_kw: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rgb (H, W, 3), expected depth (H, W), alpha (H, W)) of one camera."""
    rgb, depth, alpha, _ = render(
        splats, w2c[None], K[None], W, H, sh_degree, mode="RGB+ED", backend=backend, **render_kw
    )
    assert depth is not None
    return rgb[0], depth[0, ..., 0], alpha[0, ..., 0]


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """4x4 transform of a 6-vector (axis-angle, translation), differentiable at zero."""
    w, t = delta[:3], delta[3:]
    z = torch.zeros((), dtype=delta.dtype, device=delta.device)
    twist = torch.stack([
        torch.stack([z, -w[2], w[1], t[0]]),
        torch.stack([w[2], z, -w[0], t[1]]),
        torch.stack([-w[1], w[0], z, t[2]]),
        torch.stack([z, z, z, z]),
    ])  # fmt: skip
    return torch.linalg.matrix_exp(twist)


def align_pose(
    splats: torch.nn.ParameterDict,
    image: torch.Tensor,
    w2c: torch.Tensor,
    K: torch.Tensor,
    sh_degree: int,
    steps: int,
    backend: Backend = "gsplat",
    render_kw: Mapping[str, Any] | None = None,
    lr: float = POSE_LR,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit a camera-frame correction `T` (w2c' = T @ w2c) to the photo with the splat frozen.

    Adam on axis-angle + translation from zero, loss L1 + 0.2 (1 - SSIM). Returns the corrected w2c with
    the lowest loss seen, and the correction's size (`rot_deg`, `shift_cm`: camera centre movement).
    """
    H, W = image.shape[:2]
    kw = render_kw or {}
    delta = torch.zeros(6, device=w2c.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best, best_loss = torch.zeros(6, device=w2c.device), math.inf
    for i in range(steps + 1):
        rgb, _, _ = render_one(splats, se3_exp(delta) @ w2c, K, W, H, sh_degree, backend, kw)
        rgb = rgb.clamp(0, 1)
        loss = (rgb - image).abs().mean() + POSE_SSIM_WEIGHT * (1 - ssim_hw3(rgb, image))
        if float(loss.detach()) < best_loss:
            best_loss, best = float(loss.detach()), delta.detach().clone()
        if i == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        aligned = se3_exp(best) @ w2c
        shift = torch.linalg.inv(aligned)[:3, 3] - torch.linalg.inv(w2c)[:3, 3]
    return aligned, {
        "rot_deg": round(math.degrees(float(best[:3].norm())), 4),
        "shift_cm": round(float(shift.norm()) * 100, 4),
    }


def shrink(image: torch.Tensor, K: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(image (H, W, 3), K) at `scale` x the size (area resampling; K scaled with it). 1 = unchanged."""
    if scale >= 1.0:
        return image, K
    H, W = image.shape[:2]
    h, w = max(1, round(H * scale)), max(1, round(W * scale))
    small = F.interpolate(image.permute(2, 0, 1)[None], size=(h, w), mode="area")[0].permute(1, 2, 0)
    s = torch.tensor([w / W, h / H, 1.0], device=K.device, dtype=K.dtype)
    return small.contiguous(), s[:, None] * K


# --- images ------------------------------------------------------------------------------------------


def to_bgr8(img: torch.Tensor) -> np.ndarray:
    a = (img.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(a[..., ::-1])


def heat_map(depth: torch.Tensor, gt: torch.Tensor, conf: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    """Relative depth error 0..20 % (turbo) where the metrics count; grey without LiDAR, purple where
    the LiDAR is valid but the splat is transparent there."""
    rel = ((depth - gt).abs() / gt.clamp(min=1e-6)).clamp(0, HEAT_MAX) / HEAT_MAX
    u8 = (rel.cpu().numpy() * 255).astype(np.uint8)
    out = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    lidar = ((conf == 2) & (gt > MIN_DEPTH)).cpu().numpy()
    valid = depth_valid(depth, gt, conf, alpha).cpu().numpy()
    out[~lidar] = (48, 48, 48)
    out[lidar & ~valid] = (140, 40, 140)
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    """`text` in the top left corner (none on panels too small to hold it)."""
    if img.shape[1] < 200:
        return img
    scale = max(img.shape[1] / 900, 0.4)
    org = (int(8 * scale) + 2, int(26 * scale) + 4)
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(int(4 * scale), 2), cv2.LINE_AA
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(int(scale), 1), cv2.LINE_AA
    )
    return img


def view_K(view: Mapping[str, Any], max_side: int) -> tuple[torch.Tensor, int, int]:
    """Pinhole intrinsics of a named view (vertical fov, square pixels, centred), capped at `max_side`."""
    w, h = int(view["w"]), int(view["h"])
    s = min(1.0, max_side / max(w, h))
    W, H = max(round(w * s), 1), max(round(h * s), 1)
    f = H / 2 / math.tan(math.radians(float(view["fov_y_deg"])) / 2)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32)
    return K, W, H


# --- evaluate ----------------------------------------------------------------------------------------


def _panels(
    image: torch.Tensor,
    rgb: torch.Tensor,
    heat: np.ndarray | None,
    db: float,
    resolved: tuple[torch.Tensor, float] | None = None,
) -> np.ndarray:
    """photo | render | render at the re-solved pose (if any) | depth error (video frames), BGR."""
    panels = [label(to_bgr8(image), "photo"), label(to_bgr8(rgb), f"render {db:.2f} dB")]
    if resolved is not None:
        panels.append(label(to_bgr8(resolved[0]), f"re-solved pose {resolved[1]:.2f} dB"))
    if heat is not None:
        panels.append(label(heat, f"depth error 0-{HEAT_MAX:.0%}"))
    return np.hstack(panels)


def _mean(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_test_poses(
    data: Path, path: Path | None = None
) -> tuple[dict[str, np.ndarray | None], dict[str, Any], dict[str, dict[str, Any]]]:
    """The re-solved test poses (`test_poses.json` of `data`, or `path`): ({stem: c2w (OpenCV axes), or
    None when the frame failed to re-solve}, info for eval.json, {stem: the file's row}). Empty, with the
    reason in info, when the file is missing or was solved from another transforms.json / split.json."""
    file = Path(path) if path is not None else Path(data) / TEST_POSES
    info: dict[str, Any] = {"file": store_path(file), "status": "missing"}
    if not file.is_file():
        return {}, info, {}
    tp = json.loads(file.read_text())
    info |= {"version": tp.get("version"), "method": tp.get("method")}
    tf_path = Path(data) / "transforms.json"
    split_path = Path(data) / json.loads(tf_path.read_text()).get("split_file_path", "split.json")
    src = tp.get("source") or {}
    here = {
        "transforms_sha256": _sha256(tf_path),
        "split_sha256": _sha256(split_path) if split_path.is_file() else None,
    }
    if any(src.get(k) != v for k, v in here.items()):
        info["status"] = "mismatch"
        print(
            f"gs-trainer: {file} was solved from another transforms.json / split.json: not used", flush=True
        )
        return {}, info, {}
    poses: dict[str, np.ndarray | None] = {}
    for stem, row in tp.get("frames", {}).items():
        ok = row.get("status") == "ok" and "transform_matrix" in row
        poses[stem] = np.asarray(row["transform_matrix"], float) @ GL_CV if ok else None
    info["status"] = "ok"
    return poses, info, dict(tp.get("frames", {}))


def pose_change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(camera centre distance cm, rotation deg) between two camera-to-world matrices."""
    c = (np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 100, math.degrees(math.acos(float(np.clip(c, -1, 1))))


def evaluate_set(
    scene: Scene,
    name: str,
    splats: torch.nn.ParameterDict,
    sh_degree: int,
    out: Path,
    pose_align_steps: int,
    max_side: int,
    backend: Backend,
    device: torch.device,
    render_kw: Mapping[str, Any],
    pose_align_scale: float = POSE_ALIGN_SCALE,
    samples: int = 0,
    test_poses: Mapping[str, np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Score one test set. `test_poses` ({stem: re-solved c2w or None}, see `load_test_poses`) adds the
    `_resolved` metrics; without it (or empty) they are None."""
    indices: list[int] = list(getattr(scene, name))
    worst_dir, sample_dir = out / "worst" / name, out / "samples" / name
    for d in (worst_dir, sample_dir):
        if d.is_dir():
            for old in d.glob("*.jpg"):
                old.unlink()
    have_poses = bool(test_poses)
    tp: Mapping[str, np.ndarray | None] = test_poses or {}
    if not indices:
        return {"n": 0, "psnr": None, "ssim": None, "lpips": None, "psnr_aligned": None,
                "abs_rel": None, "delta105": None, "depth_px": 0, "sessions": {}, "per_frame": [],
                **_resolved_summary([], have_poses, 0.0, 0, 0)}  # fmt: skip
    images = Images(scene, indices, max_side, device, cache=False)
    sample_at = (
        set(np.linspace(0, len(indices) - 1, samples).round().astype(int).tolist()) if samples else set()
    )
    sample_rows: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    worst: list[tuple[float, int, str, np.ndarray]] = []  # min-heap on -PSNR: the 6 lowest PSNRs
    rel_sum, d_count, px = 0.0, 0, 0
    rel_sum_r, d_count_r, px_r = 0.0, 0, 0
    started = time.monotonic()
    for j, i in enumerate(indices):
        frame: Frame = scene.frames[i]
        image, K = images.get(i)
        H, W = image.shape[:2]
        w2c = viewmat(frame.c2w).to(device=device, dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            rgb, depth, alpha = render_one(splats, w2c, K, W, H, sh_degree, backend, render_kw)
            rgb = rgb.clamp(0, 1)
            row: dict[str, Any] = {
                "stem": frame.stem,
                "session": frame.session,
                "camera": "still" if frame.camera_idx == 1 else "video",
                "psnr": round(psnr(rgb, image), 4),
                "ssim": round(float(ssim_hw3(rgb, image)), 4),
                "lpips": round(lpips_score(rgb, image), 4),
            }
        heat = None
        lidar = load_depth(frame, W, H, device) if frame.depth_path is not None else None
        if lidar is not None:
            gt, conf = lidar
            s, c, n = depth_errors(depth, gt, conf, alpha)
            rel_sum, d_count, px = rel_sum + s, d_count + c, px + n
            row.update(abs_rel=round(s / n, 5) if n else None, delta105=round(c / n, 5) if n else None,
                       depth_px=n)  # fmt: skip
            heat = heat_map(depth, gt, conf, alpha)
        else:
            row.update(abs_rel=None, delta105=None, depth_px=0)

        c2w_r = tp.get(frame.stem)
        row["pose"] = None if not have_poses else "resolved" if c2w_r is not None else "unresolved"
        w2c_r = None
        resolved_panel = None
        if c2w_r is not None:
            w2c_r = viewmat(c2w_r).to(device=device, dtype=torch.float32)
            shift_cm, rot = pose_change(c2w_r, frame.c2w)
            with torch.no_grad():
                rgb_r, depth_r, alpha_r = render_one(splats, w2c_r, K, W, H, sh_degree, backend, render_kw)
                rgb_r = rgb_r.clamp(0, 1)
                row.update(
                    psnr_resolved=round(psnr(rgb_r, image), 4),
                    ssim_resolved=round(float(ssim_hw3(rgb_r, image)), 4),
                    lpips_resolved=round(lpips_score(rgb_r, image), 4),
                    pose_shift_cm=round(shift_cm, 2),
                    pose_rot_deg=round(rot, 3),
                )
            resolved_panel = (rgb_r, row["psnr_resolved"])
            if lidar is not None:
                s, c, n = depth_errors(depth_r, lidar[0], lidar[1], alpha_r)
                rel_sum_r, d_count_r, px_r = rel_sum_r + s, d_count_r + c, px_r + n
                row["abs_rel_resolved"] = round(s / n, 5) if n else None
                row["delta105_resolved"] = round(c / n, 5) if n else None
                row["depth_px_resolved"] = n

        if pose_align_steps > 0:
            start = w2c_r if w2c_r is not None else w2c
            small, K_small = shrink(image, K, pose_align_scale)
            aligned, size = align_pose(
                splats, small, start, K_small, sh_degree, pose_align_steps, backend, render_kw
            )
            with torch.no_grad():
                rgb_a, _, _ = render_one(splats, aligned, K, W, H, sh_degree, backend, render_kw)
            row["psnr_aligned"] = round(psnr(rgb_a, image), 4)
            row["pose_correction"] = size
        else:
            row["psnr_aligned"] = None
        per_frame.append(row)

        if j in sample_at:
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(sample_dir / f"{frame.stem}.jpg"),
                _panels(image, rgb, heat, row["psnr"], resolved_panel),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            sample_rows.append(
                {"stem": frame.stem, "psnr": row["psnr"], "file": f"samples/{name}/{frame.stem}.jpg"}
            )
        key = (-row["psnr"], j)
        if len(worst) < WORST or key > (worst[0][0], worst[0][1]):
            item = (key[0], j, frame.stem, _panels(image, rgb, heat, row["psnr"], resolved_panel))
            if len(worst) < WORST:
                heapq.heappush(worst, item)
            else:
                heapq.heapreplace(worst, item)
        if (j + 1) % 25 == 0 or j + 1 == len(indices):
            print(f"  {name}: {j + 1}/{len(indices)} frames, {time.monotonic() - started:.0f} s", flush=True)

    worst_dir.mkdir(parents=True, exist_ok=True)
    worst_rows = []
    for neg, _, stem, img in sorted(worst, reverse=True):  # lowest PSNR first
        cv2.imwrite(str(worst_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        worst_rows.append({"stem": stem, "psnr": -neg, "file": f"worst/{name}/{stem}.jpg"})
    return {
        "n": len(indices),
        "psnr": _mean([r["psnr"] for r in per_frame]),
        "ssim": _mean([r["ssim"] for r in per_frame]),
        "lpips": _mean([r["lpips"] for r in per_frame]),
        "psnr_aligned": _mean([r["psnr_aligned"] for r in per_frame]),
        "abs_rel": round(rel_sum / px, 5) if px else None,
        "delta105": round(d_count / px, 5) if px else None,
        "depth_px": px,
        **_resolved_summary(per_frame, have_poses, rel_sum_r, d_count_r, px_r),
        "sessions": session_breakdown(per_frame),
        "worst": worst_rows,
        "samples": sample_rows,
        "per_frame": per_frame,
    }


def _resolved_summary(
    per_frame: Sequence[Mapping[str, Any]], have_poses: bool, rel_sum: float, d_count: int, px: int
) -> dict[str, Any]:
    """A set's `_resolved` metrics (means over the re-solved frames), counts and `pose_moved`."""
    res = [r for r in per_frame if r.get("pose") == "resolved"]
    unres = [r for r in per_frame if r.get("pose") == "unresolved"]
    moved = [
        {k: r[k] for k in ("stem", "pose_shift_cm", "pose_rot_deg", "psnr", "psnr_resolved")}
        for r in res
        if r["pose_shift_cm"] > MOVED_CM or r["pose_rot_deg"] > MOVED_DEG
    ]
    return {
        "n_resolved": len(res) if have_poses else None,
        "n_unresolved": len(unres) if have_poses else None,
        "psnr_resolved": _mean([r["psnr_resolved"] for r in res]),
        "ssim_resolved": _mean([r["ssim_resolved"] for r in res]),
        "lpips_resolved": _mean([r["lpips_resolved"] for r in res]),
        "abs_rel_resolved": round(rel_sum / px, 5) if px else None,
        "delta105_resolved": round(d_count / px, 5) if px else None,
        "psnr_same_frames": _mean([r["psnr"] for r in res]),
        "psnr_unresolved": _mean([r["psnr"] for r in unres]),
        "pose_moved": sorted(moved, key=lambda m: -m["pose_shift_cm"]),
    }


def frame_session(row: Mapping[str, Any]) -> str:
    """A per-frame row's session: its `session`, or (eval.json written before 2026-10-10) the stem's prefix."""
    return str(row.get("session") or str(row["stem"]).rpartition("_")[0])


def session_label(session: str) -> str:
    """The room part of a session name (`20261007-183715-kitchen` -> `kitchen`): matches sessions across scans."""
    head = session.split("-", 2)
    return head[2] if len(head) == 3 and head[0].isdigit() and head[1].isdigit() and head[2] else session


def _pooled(rows: Sequence[Mapping[str, Any]], key: str, px_key: str) -> float | None:
    """A per-frame ratio pooled per pixel, as the set's own depth metrics (sum of value x pixels / pixels)."""
    pairs = [(r[key], r.get(px_key, r.get("depth_px")) or 0) for r in rows if r.get(key) is not None]
    px = sum(n for _, n in pairs)
    return round(sum(v * n for v, n in pairs) / px, 5) if px else None


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Means of a group of per-frame rows, by the set-level definitions: PSNR / SSIM / LPIPS as means over
    frames, depth pooled per pixel, the `_resolved` metrics over the re-solved frames only. Adds the median
    and lowest PSNR and the worst frame, which a mean over a session hides."""
    psnrs = [r["psnr"] for r in rows]
    res = [r for r in rows if r.get("pose") == "resolved"]
    worst = min(rows, key=lambda r: r["psnr"]) if rows else None
    return {
        "n": len(rows),
        "psnr": _mean(psnrs),
        "psnr_median": round(float(np.median(psnrs)), 4) if psnrs else None,
        "psnr_min": round(float(min(psnrs)), 4) if psnrs else None,
        "worst": worst["stem"] if worst else None,
        "ssim": _mean([r["ssim"] for r in rows]),
        "lpips": _mean([r["lpips"] for r in rows]),
        "psnr_aligned": _mean([r.get("psnr_aligned") for r in rows]),
        "abs_rel": _pooled(rows, "abs_rel", "depth_px"),
        "delta105": _pooled(rows, "delta105", "depth_px"),
        "depth_px": sum(r.get("depth_px") or 0 for r in rows),
        "n_resolved": len(res) if any(r.get("pose") is not None for r in rows) else None,
        "psnr_resolved": _mean([r.get("psnr_resolved") for r in res]),
        "ssim_resolved": _mean([r.get("ssim_resolved") for r in res]),
        "lpips_resolved": _mean([r.get("lpips_resolved") for r in res]),
        "abs_rel_resolved": _pooled(res, "abs_rel_resolved", "depth_px_resolved"),
        "psnr_same_frames": _mean([r["psnr"] for r in res]),
    }


def session_breakdown(per_frame: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """{session: `aggregate` of its frames}, in the order the sessions first appear."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in per_frame:
        groups.setdefault(frame_session(r), []).append(r)
    return {s: aggregate(rows) for s, rows in groups.items()}


def set_sessions(s: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """A set's per-session breakdown: eval.json's `sessions`, or computed from `per_frame` for older files."""
    return dict(s.get("sessions") or session_breakdown(s.get("per_frame") or []))


def render_views(
    views: Path,
    splats: torch.nn.ParameterDict,
    sh_degree: int,
    out: Path,
    max_side: int,
    backend: Backend,
    device: torch.device,
    render_kw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render each named view of `views` (`{"views": [{name, c2w, fov_y_deg, w, h, note}]}`) to views/."""
    spec = json.loads(views.read_text())
    (out / "views").mkdir(parents=True, exist_ok=True)
    rows = []
    for v in spec.get("views", []):
        K, W, H = view_K(v, max_side)
        c2w = np.asarray(v["c2w"], dtype=np.float64).reshape(4, 4)
        w2c = viewmat(c2w).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            rgb, _, _ = render_one(splats, w2c, K.to(device), W, H, sh_degree, backend, render_kw)
        file = f"views/{v['name']}.png"
        cv2.imwrite(str(out / file), to_bgr8(rgb))
        rows.append({"name": v["name"], "note": v.get("note", ""), "file": file, "w": W, "h": H})
    return rows


def evaluate(
    data: Path,
    ply: Path,
    out: Path,
    sets: Sequence[str] = SETS,
    pose_align_steps: int = POSE_ALIGN_STEPS,
    views: Path | None = None,
    max_side: int = 960,
    backend: Backend = "gsplat",
    device: torch.device | str | None = None,
    antialiased: bool = False,
    pose_align_scale: float = POSE_ALIGN_SCALE,
    samples: int = 0,
    test_poses: Path | None = None,
    use_test_poses: bool = True,
) -> dict[str, Any]:
    """Score `ply` on `data`'s test sets; writes `out/eval.json`, `out/index.html` and images. Returns
    the eval.json content. `antialiased` renders with gsplat's antialiased mode (for splats trained so).
    Pose alignment runs `pose_align_steps` steps per frame at `pose_align_scale` x the image size.
    `samples` evenly spaced frames per set also go to `samples/<set>/` (photo | render | depth error).
    `test_poses`: the re-solved test poses (default `data`/test_poses.json when it is there);
    `use_test_poses=False` scores at the dataset poses only."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene.load(Path(data))
    tensors, sh_degree = load_ply(Path(ply))
    splats = frozen(tensors, dev)
    render_kw: dict[str, Any] = {"antialiased": True} if antialiased else {}
    poses: dict[str, np.ndarray | None] = {}
    tp_info: dict[str, Any] = {"status": "off"}
    tp_rows: dict[str, dict[str, Any]] = {}
    if use_test_poses:
        poses, tp_info, tp_rows = load_test_poses(Path(data), test_poses)
    started = time.monotonic()
    result: dict[str, Any] = {
        "ply": store_path(ply),  # repo-relative when local (`store_path`)
        "data": store_path(data),
        "splats": int(tensors["means"].shape[0]),
        "sh_degree": sh_degree,
        "max_side": max_side,
        "pose_align_steps": pose_align_steps,
        "pose_align_scale": pose_align_scale,
        "backend": backend,
        "antialiased": antialiased,
        "test_poses": tp_info,
        "sets": {},
        "views": [],
    }
    for name in sets:
        if name not in SETS:
            raise ValueError(f"unknown set {name!r} (have {', '.join(SETS)})")
        result["sets"][name] = evaluate_set(
            scene,
            name,
            splats,
            sh_degree,
            out,
            pose_align_steps,
            max_side,
            backend,
            dev,
            render_kw,
            pose_align_scale,
            samples,
            poses,
        )
        for m in result["sets"][name]["pose_moved"]:  # how well each pose explains the matches
            row = tp_rows.get(m["stem"], {})
            m.update(reproj_px=row.get("reproj_px"), reproj_dataset_px=row.get("reproj_dataset_px"))
    if views is not None:
        result["views"] = render_views(Path(views), splats, sh_degree, out, max_side, backend, dev, render_kw)
    result["wall_s"] = round(time.monotonic() - started, 1)
    (out / "eval.json").write_text(dumps_eval(result))
    (out / "index.html").write_text(eval_html(result))
    return result


def dumps_eval(result: Mapping[str, Any]) -> str:
    """eval.json's text: indented, but each per-frame row and each session on one line (a few hundred frames
    per set stay readable and a tenth of the lines)."""
    one_line: list[str] = []

    def mark(obj: Any) -> str:
        one_line.append(json.dumps(obj, separators=(", ", ": ")))
        return f"@@row{len(one_line) - 1}@@"

    sets = {
        name: {
            **s,
            "per_frame": [mark(r) for r in s.get("per_frame", [])],
            **({"sessions": {k: mark(v) for k, v in s["sessions"].items()}} if "sessions" in s else {}),
        }
        for name, s in result.get("sets", {}).items()
    }
    text = json.dumps({**result, "sets": sets}, indent=1)
    return re.sub(r'"@@row(\d+)@@"', lambda m: one_line[int(m[1])], text) + "\n"


# --- HTML --------------------------------------------------------------------------------------------

METRICS: tuple[tuple[str, str, int, bool], ...] = (
    # key, label, decimals, higher is better
    ("psnr", "PSNR dB", 2, True),
    ("ssim", "SSIM", 4, True),
    ("lpips", "LPIPS", 4, False),
    ("psnr_aligned", "PSNR aligned dB", 2, True),
    ("abs_rel", "depth AbsRel", 4, False),
    ("delta105", "depth δ<1.05", 4, True),
    ("psnr_resolved", "PSNR re-solved dB", 2, True),
    ("ssim_resolved", "SSIM re-solved", 4, True),
    ("lpips_resolved", "LPIPS re-solved", 4, False),
    ("abs_rel_resolved", "AbsRel re-solved", 4, False),
    ("psnr_same_frames", "PSNR same frames dB", 2, True),
)
# per-frame table: the dataset-pose metrics, then the re-solved ones
FRAME_METRICS = METRICS[:-1]
# per-session tables (index.html: all of them; compare: the first five per run)
SESSION_METRICS: tuple[tuple[str, str, int, bool], ...] = (
    ("psnr", "PSNR dB", 2, True),
    ("ssim", "SSIM", 4, True),
    ("lpips", "LPIPS", 4, False),
    ("abs_rel", "AbsRel", 4, False),
    ("psnr_resolved", "PSNR re-solved dB", 2, True),
    ("psnr_median", "PSNR median dB", 2, True),
    ("psnr_min", "PSNR min dB", 2, True),
    ("psnr_aligned", "PSNR aligned dB", 2, True),
    ("delta105", "δ<1.05", 4, True),
)
COMPARE_SESSION_METRICS = SESSION_METRICS[:5]

CSS = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #ffffff; --muted: #6e6e73; --line: #d2d2d7;
        --best: #e3f4e1; --card: #f5f5f7; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #f5f5f7; --bg: #161617; --muted: #a1a1a6; --line: #3a3a3c; --best: #1f3a24; --card: #1f1f21; }
}
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--fg);
       background: var(--bg); margin: 0 auto; padding: 16px; max-width: 1600px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 8px; }
h3 { font-size: 14px; margin: 18px 0 6px; }
.muted { color: var(--muted); } code { font-size: 12px; word-break: break-all; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
td.best { background: var(--best); font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--card); border-radius: 6px; overflow: hidden; }
figure img { display: block; width: 100%; height: auto; }
figcaption { padding: 6px 8px; font-size: 12px; }
.wide figure { margin-bottom: 12px; }
.views td { vertical-align: top; text-align: left; } .views img { display: block; width: 100%; min-width: 240px; }
"""


def _fmt(x: Any, decimals: int) -> str:
    return "&ndash;" if x is None else f"{x:.{decimals}f}"


def _page(title: str, body: str) -> str:
    return (
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n<body>\n{body}\n</body></html>\n"
    )


def _tp_text(info: Mapping[str, Any] | None) -> str:
    if not info:
        return "none"
    if info.get("status") == "ok":
        return f"{info.get('method')} v{info.get('version')}"
    return {"missing": "none (no test_poses.json)", "mismatch": "refused (solved from another dataset)"}.get(
        str(info.get("status")), str(info.get("status"))
    )


def eval_html(r: Mapping[str, Any]) -> str:
    e = html.escape
    parts = [
        f"<h1>Splat evaluation</h1><p class='muted'><code>{e(r['ply'])}</code><br>"
        f"{r['splats']:,} splats · SH {r['sh_degree']} · {r['max_side']} px · "
        f"pose alignment {r['pose_align_steps']} steps at {r.get('pose_align_scale', 1.0):g}x size · {e(r['backend'])}"
        f" · re-solved test poses: {e(_tp_text(r.get('test_poses')))}</p>",
        "<h2>Metrics</h2><div class='scroll'><table><tr><th>set</th><th>frames</th>"
        + "".join(f"<th>{e(lab)}</th>" for _, lab, _, _ in METRICS)
        + "<th>depth px</th><th>re-solved / unresolved / moved</th></tr>",
    ]
    for name, s in r["sets"].items():
        cells = "".join(f"<td>{_fmt(s.get(k), d)}</td>" for k, _, d, _ in METRICS)
        counts = (
            "&ndash;"
            if s.get("n_resolved") is None
            else f"{s['n_resolved']} / {s['n_unresolved']} / {len(s.get('pose_moved', []))}"
        )
        parts.append(
            f"<tr><td>{e(name)}</td><td>{s['n']}</td>{cells}<td>{s.get('depth_px', 0):,}</td><td>{counts}</td></tr>"
        )
    parts.append("</table></div><p class='muted'>re-solved: at each test frame's pose solved against the "
                 "training frames (means over the re-solved frames only; <i>same frames</i>: their PSNR at the "
                 f"dataset poses). moved: re-solved pose &gt; {MOVED_CM:g} cm or {MOVED_DEG:g}&deg; from the "
                 "dataset pose.</p>")  # fmt: skip
    for name, s in r["sets"].items():
        sessions = set_sessions(s)
        if not sessions:
            continue
        parts.append(
            f"<h3>{e(name)} by session</h3><div class='scroll'><table><tr><th>session</th><th>frames</th>"
            + "".join(f"<th>{e(lab)}</th>" for _, lab, _, _ in SESSION_METRICS)
            + "<th>re-solved</th><th>worst frame</th></tr>"
        )
        for sess, g in sessions.items():
            parts.append(
                f"<tr><td>{e(sess)}</td><td>{g['n']}</td>"
                + "".join(f"<td>{_fmt(g.get(k), d)}</td>" for k, _, d, _ in SESSION_METRICS)
                + f"<td>{_fmt(g.get('n_resolved'), 0)}</td><td>{e(str(g.get('worst') or ''))}</td></tr>"
            )
        parts.append("</table></div>")
    if any(set_sessions(s) for s in r["sets"].values()):
        parts.append("<p class='muted'>Per session: the set's own definitions (means over frames; depth pooled "
                     "per pixel; re-solved over the re-solved frames only), plus the median and lowest PSNR.</p>")  # fmt: skip
    for name, s in r["sets"].items():
        if not s.get("pose_moved"):
            continue
        parts.append(
            f"<details><summary>{e(name)}: {len(s['pose_moved'])} frames whose pose moved</summary>"
            "<div class='scroll'><table><tr><th>frame</th><th>moved cm</th><th>moved &deg;</th>"
            "<th>PSNR dB</th><th>PSNR re-solved dB</th><th>reproj. px re-solved</th><th>at dataset pose</th></tr>"
        )
        for m in s["pose_moved"]:
            parts.append(
                f"<tr><td>{e(m['stem'])}</td><td>{_fmt(m['pose_shift_cm'], 1)}</td><td>{_fmt(m['pose_rot_deg'], 2)}</td>"
                f"<td>{_fmt(m['psnr'], 2)}</td><td>{_fmt(m['psnr_resolved'], 2)}</td>"
                f"<td>{_fmt(m.get('reproj_px'), 2)}</td><td>{_fmt(m.get('reproj_dataset_px'), 2)}</td></tr>"
            )
        parts.append("</table></div></details>")
    if r.get("views"):
        parts.append("<h2>Named views</h2><div class='grid'>")
        for v in r["views"]:
            note = f" <span class='muted'>{e(v['note'])}</span>" if v.get("note") else ""
            parts.append(
                f"<figure><img src='{e(v['file'])}' alt='{e(v['name'])}' loading='lazy'>"
                f"<figcaption><b>{e(v['name'])}</b>{note}</figcaption></figure>"
            )
        parts.append("</div>")
    for name, s in r["sets"].items():
        if not s.get("worst"):
            continue
        parts.append(f"<h2>Worst {e(name)} frames</h2><p class='muted'>photo | render | relative depth "
                     f"error 0-{HEAT_MAX:.0%} (grey: no LiDAR, purple: LiDAR but splat alpha &le; "
                     f"{MIN_ALPHA})</p><div class='wide'>")  # fmt: skip
        for w in s["worst"]:
            parts.append(
                f"<figure><img src='{e(w['file'])}' alt='{e(w['stem'])}' loading='lazy'>"
                f"<figcaption>{e(w['stem'])} · {w['psnr']:.2f} dB</figcaption></figure>"
            )
        parts.append("</div>")
    for name, s in r["sets"].items():
        if not s.get("per_frame"):
            continue
        rows = sorted(s["per_frame"], key=lambda f: f["psnr"])
        cols = [(k, lab, d) for k, lab, d, _ in FRAME_METRICS]
        parts.append(
            f"<details><summary>All {e(name)} frames ({len(rows)}, worst first)</summary>"
            "<div class='scroll'><table><tr><th>frame</th>"
            + "".join(f"<th>{e(lab)}</th>" for _, lab, _ in cols)
            + "<th>pose Δ °</th><th>pose Δ cm</th><th>re-solved Δ cm</th></tr>"
        )
        for f in rows:
            pc = f.get("pose_correction") or {}
            parts.append(
                f"<tr><td>{e(f['stem'])}</td>"
                + "".join(f"<td>{_fmt(f.get(k), d)}</td>" for k, _, d in cols)
                + f"<td>{_fmt(pc.get('rot_deg'), 3)}</td><td>{_fmt(pc.get('shift_cm'), 2)}</td>"
                + f"<td>{_fmt(f.get('pose_shift_cm'), 1) if f.get('pose') != 'unresolved' else 'unresolved'}</td></tr>"
            )
        parts.append("</table></div></details>")
    return _page("Splat evaluation", "\n".join(parts))


# --- compare -----------------------------------------------------------------------------------------


def _labels(runs: Sequence[Path]) -> list[str]:
    """A column label per run: the eval dir's name, or its parent's when the dir is called eval*."""
    labels: list[str] = []
    for run in runs:
        p = run.resolve()
        base = p.parent.name if p.name.startswith("eval") and p.parent.name else p.name
        lab, k = base, 2
        while lab in labels:
            lab, k = f"{base}-{k}", k + 1
        labels.append(lab)
    return labels


def by_room(s: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """A set's per-frame rows grouped by `session_label` (sessions of one room in one run pooled)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in s.get("per_frame") or []:
        groups.setdefault(session_label(frame_session(r)), []).append(r)
    return groups


def _compare_sessions(
    evals: Sequence[Mapping[str, Any]], labels: Sequence[str], set_names: Sequence[str]
) -> list[str]:
    """Per set, a row per room (sessions matched across runs by `session_label`) and a column group per run,
    plus `rooms in every run`: the frames of the rooms every run has, pooled (runs of different scans
    compared on common ground, e.g. without one scan's garden) and `all`."""
    e = html.escape
    cols = COMPARE_SESSION_METRICS
    parts: list[str] = []
    for name in set_names:
        rooms = [by_room(ev["sets"].get(name, {})) for ev in evals]
        order: list[str] = []
        for g in rooms:
            order += [k for k in g if k not in order]
        if not order:
            continue
        common = [k for k in order if all(k in g for g in rooms)]
        rows: list[tuple[str, list[dict[str, Any] | None]]] = [
            (k, [aggregate(g[k]) if k in g else None for g in rooms]) for k in order
        ]
        if len(evals) > 1 and common and len(common) < len(order):
            rows.append((f"rooms in every run ({', '.join(common)})",
                         [aggregate([r for k in common for r in g[k]]) for g in rooms]))  # fmt: skip
        rows.append(("all", [aggregate([r for v in g.values() for r in v]) if g else None for g in rooms]))
        head = "".join(f"<th colspan='{len(cols) + 1}'>{e(lab)}</th>" for lab in labels)
        sub = "".join("<th>frames</th>" + "".join(f"<th>{e(c[1])}</th>" for c in cols) for _ in labels)
        parts.append(f"<h3>{e(name)} by room</h3><div class='scroll'><table><tr><th></th>{head}</tr>"
                     f"<tr><th>room</th>{sub}</tr>")  # fmt: skip
        for room, aggs in rows:
            best: dict[str, float | None] = {}
            for key, _, _, higher in cols:
                vals = [a[key] for a in aggs if a is not None and a.get(key) is not None]
                best[key] = (max(vals) if higher else min(vals)) if len(vals) > 1 else None
            cells = []
            for a in aggs:
                if a is None:
                    cells.append(f"<td class='muted' colspan='{len(cols) + 1}'>&ndash;</td>")
                    continue
                cells.append(f"<td>{a['n']}</td>")
                for key, _, d, _ in cols:
                    v = a.get(key)
                    cls = " class='best'" if v is not None and v == best[key] else ""
                    cells.append(f"<td{cls}>{_fmt(v, d)}</td>")
            bold = room == "all" or room.startswith("rooms in every run")
            parts.append(
                f"<tr><td>{'<b>' if bold else ''}{e(room)}{'</b>' if bold else ''}</td>{''.join(cells)}</tr>"
            )
        parts.append("</table></div>")
    if parts:
        parts.insert(0, "<h2>By room</h2><p class='muted'>Sessions matched across runs by their room name (the "
                        "session name without its date); each run is scored on its own test frames, so across "
                        "scans this compares rooms, not the same photos. Means over frames, AbsRel pooled per "
                        "pixel, re-solved over the re-solved frames only.</p>")  # fmt: skip
    return parts


def compare(runs: list[Path], out: Path) -> Path:
    """One page across runs' eval dirs: metrics side by side, named views in a grid (a row per view, a
    column per run). Images are linked relative to `out`, not copied. Returns the page's path."""
    e = html.escape
    out.mkdir(parents=True, exist_ok=True)
    evals = [json.loads((Path(r) / "eval.json").read_text()) for r in runs]
    labels = _labels([Path(r) for r in runs])
    head = "".join(f"<th>{e(lab)}</th>" for lab in labels)
    parts = [
        f"<h1>Splat comparison</h1><p class='muted'>{len(runs)} runs</p>",
        f"<h2>Metrics</h2><div class='scroll'><table><tr><th>metric</th>{head}</tr>",
        "<tr><td>splats</td>" + "".join(f"<td>{ev['splats']:,}</td>" for ev in evals) + "</tr>",
        "<tr><td>SH degree</td>" + "".join(f"<td>{ev['sh_degree']}</td>" for ev in evals) + "</tr>",
    ]
    set_names = [s for s in SETS if any(s in ev["sets"] for ev in evals)]
    for name in set_names:
        ns = [ev["sets"].get(name, {}).get("n") for ev in evals]
        parts.append(f"<tr><th colspan='{len(evals) + 1}'>{e(name)} "
                     f"<span class='muted'>({' / '.join('&ndash;' if n is None else str(n) for n in ns)} frames)"
                     "</span></th></tr>")  # fmt: skip
        resolved = [ev["sets"].get(name, {}).get("n_resolved") for ev in evals]
        if any(v is not None for v in resolved):
            parts.append("<tr><td>re-solved frames</td>"
                         + "".join(f"<td>{_fmt(v, 0)}</td>" for v in resolved) + "</tr>")  # fmt: skip
        for key, lab, d, higher in METRICS:
            vals = [ev["sets"].get(name, {}).get(key) for ev in evals]
            present = [v for v in vals if v is not None]
            best = (max(present) if higher else min(present)) if len(present) > 1 else None
            cells = "".join(
                f"<td class='best'>{_fmt(v, d)}</td>"
                if v is not None and v == best
                else f"<td>{_fmt(v, d)}</td>"
                for v in vals
            )
            parts.append(f"<tr><td>{e(lab)}</td>{cells}</tr>")
    parts.append("</table></div>")
    parts += _compare_sessions(evals, labels, set_names)

    names: list[str] = []
    notes: dict[str, str] = {}
    for ev in evals:
        for v in ev.get("views", []):
            if v["name"] not in names:
                names.append(v["name"])
                notes[v["name"]] = v.get("note", "")
    if names:
        parts.append(
            f"<h2>Named views</h2><div class='scroll'><table class='views'><tr><th>view</th>{head}</tr>"
        )
        for name in names:
            cells = []
            for run, ev in zip(runs, evals, strict=True):
                v = next((v for v in ev.get("views", []) if v["name"] == name), None)
                if v is None:
                    cells.append("<td class='muted'>&ndash;</td>")
                    continue
                src = os.path.relpath(Path(run).resolve() / v["file"], out.resolve())
                cells.append(
                    f"<td><a href='{e(src)}'><img src='{e(src)}' alt='{e(name)}' loading='lazy'></a></td>"
                )
            note = f"<br><span class='muted'>{e(notes[name])}</span>" if notes[name] else ""
            parts.append(f"<tr><td><b>{e(name)}</b>{note}</td>{''.join(cells)}</tr>")
        parts.append("</table></div>")

    parts.append("<h2>Runs</h2><ul>")
    for run, lab, ev in zip(runs, labels, evals, strict=True):
        page = os.path.relpath(Path(run).resolve() / "index.html", out.resolve())
        parts.append(f"<li><a href='{e(page)}'>{e(lab)}</a> <code class='muted'>{e(ev['ply'])}</code></li>")
    parts.append("</ul>")
    page = out / "index.html"
    page.write_text(_page("Splat comparison", "\n".join(parts)))
    return page


# --- CLI ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["compare"]:
        p = argparse.ArgumentParser(prog="gs_trainer.evaluate compare", description=compare.__doc__)
        p.add_argument("runs", nargs="+", type=Path, help="eval dirs (each with eval.json)")
        p.add_argument("--out", type=Path, required=True)
        a = p.parse_args(args[1:])
        print(compare(a.runs, a.out))
        return 0
    p = argparse.ArgumentParser(prog="gs_trainer.evaluate", description=evaluate.__doc__)
    p.add_argument("--data", type=Path, required=True, help="dataset v2 dir (transforms.json, split.json)")
    p.add_argument("--ply", type=Path, required=True, help="3DGS PLY in the scene frame")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sets", default=",".join(SETS), help="comma-separated: interp,extrap")
    p.add_argument(
        "--pose-align-steps", type=int, default=POSE_ALIGN_STEPS, help="per frame; 0 skips pose-aligned PSNR"
    )
    p.add_argument(
        "--pose-align-scale",
        type=float,
        default=POSE_ALIGN_SCALE,
        help="image size for the pose fit, x --max-side (the aligned PSNR is scored at full size)",
    )
    p.add_argument(
        "--views", type=Path, default=None, help="named views, e.g. data/<project>/scene/views.json"
    )
    p.add_argument(
        "--no-views", action="store_true", help="disable named views even when --views is supplied"
    )
    p.add_argument("--max-side", type=int, default=960)
    p.add_argument(
        "--samples", type=int, default=0, help="also save N evenly spaced frames per set (samples/)"
    )
    p.add_argument("--backend", choices=("gsplat", "torch"), default="gsplat")
    p.add_argument("--device", default=None)
    p.add_argument("--antialiased", action="store_true", help="render in gsplat's antialiased mode")
    p.add_argument(
        "--test-poses",
        type=Path,
        default=None,
        help=f"re-solved test poses (default: the dataset's {TEST_POSES}, when there)",
    )
    p.add_argument("--no-test-poses", action="store_true", help="score at the dataset poses only")
    a = p.parse_args(args)
    r = evaluate(
        a.data, a.ply, a.out,
        sets=tuple(s.strip() for s in a.sets.split(",") if s.strip()),
        pose_align_steps=a.pose_align_steps, views=None if a.no_views else a.views, max_side=a.max_side,
        backend=a.backend, device=a.device, antialiased=a.antialiased, pose_align_scale=a.pose_align_scale,
        samples=a.samples, test_poses=a.test_poses, use_test_poses=not a.no_test_poses,
    )  # fmt: skip
    for name, s in r["sets"].items():
        vals = " ".join(f"{k}={'-' if s.get(k) is None else f'{s[k]:.{d}f}'}" for k, _, d, _ in METRICS)
        print(f"{name:>7}: n={s['n']} {vals}")
        if s.get("n_resolved") is not None:
            print(f"         re-solved {s['n_resolved']}, unresolved {s['n_unresolved']}, "
                  f"moved > {MOVED_CM:g} cm or {MOVED_DEG:g}°: {len(s['pose_moved'])}")  # fmt: skip
    print(a.out / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
