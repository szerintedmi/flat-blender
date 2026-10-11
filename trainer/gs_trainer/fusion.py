"""Sensor + network depth fusion with measured sigmas and gating (plan P4, after Howell's Cal Poly thesis).

Two depth sources per frame, both in metres:
- the LiDAR (`depth/`, upsampled nearest to the prior's size): accurate where it reads, with a sigma from
  `prior/sigma.json` (`gs_trainer.calibrate`) by ARKit confidence and distance, raised at depth edges;
- Depth Anything 3 (`prior/da3/`, `gs_trainer.da3`): dense, everywhere, but with errors of a few per cent.

They are combined by inverse-variance weighting, as in the thesis:
`d = (w_s d_s + w_m d_m) / (w_s + w_m)`, `w = 1 / sigma^2`, `sigma_f = (w_s + w_m)^-1/2`; where only one
source has a value, it is used alone. What differs from the thesis, and why:
- DA3's scale. Even pose-conditioned, a window's scale can be off by tens of per cent (scan 3: a blank
  wall, close views), so each frame's DA3 depth is first scaled to the LiDAR (median ratio over confident,
  non-edge pixels; stills: to the mesh prior). `align=False` keeps DA3's own scale, for comparison.
- DA3's sigma is measured per frame, not assumed: 1.4826 x MAD of its relative residual to the LiDAR (the
  LiDAR's own share removed in quadrature), so `sigma_m = rel x d_m`. Frames with nothing to compare
  against take the median of the measured frames. On top: the thesis' edge term, x (1 + 0.5 g / g_ref)
  with g DA3's smoothed depth gradient (g_ref = its 99th percentile in the frame, a robust version of
  the thesis' max), and x `ceiling_factor` on ceilings (mesh normal pointing down), a known DA3 weakness.
  A frame whose DA3 is off by more than `max_rel` (e.g. 20 %: a textureless wall) is not used at all.
- Gating. Where both read and `|d_s - d_m| > gate x sqrt(sigma_s^2 + sigma_m^2)`, the two disagree beyond
  their noise: trust the sensor at confidence 2, and drop the pixel (no target) below it. The plan wrote
  `gate x sigma_f`, but sigma_f is smaller than either source's sigma (with the LiDAR at 1 cm it is ~1 cm),
  so that test fires on DA3's ordinary few-cm noise: on scan 3 it flagged 21 % of all pixels. The
  difference of two independent readings has sigma sqrt(sigma_s^2 + sigma_m^2), which is what a 3-sigma
  test needs. `gate_sigma="fused"` restores the plan's version for comparison.

Modes, for the P4 ablations; each writes its own keys into `prior/<stem>.npz`, next to the mesh prior's
`depth` / `normal` (which are kept as they are), so all can coexist on one data volume:
- `fused` -> `fused`, `sigma`: the fusion above;
- `lidar_only` -> `fused_lidar_only`, `sigma_lidar_only`: the LiDAR with its calibrated sigma, no DA3;
- `da3_holes_only` -> `fused_da3_holes_only`, `sigma_da3_holes_only`: the LiDAR where it reads, DA3 only
  where it doesn't (and in the stills).
Depth and sigma are float16 (H, W) at the prior's size, metres; 0 in both = no target. A training loss
weights each pixel by 1/sigma, normalised per image (`load_fused` reads them back).

`prior/fusion.json` records the settings and statistics of each mode's last run (pixel shares per source,
gated and masked pixels, DA3's per-frame scales and relative errors).

    python -m gs_trainer.fusion --data data/<project>/splat/datasets/default
        [--mode fused|lidar_only|da3_holes_only|all]
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import calibrate, da3
from .data import Frame, Scene, _resize, load_depth

VERSION = 1
MODES = ("fused", "lidar_only", "da3_holes_only")
KEYS = {
    "fused": ("fused", "sigma"),
    "lidar_only": ("fused_lidar_only", "sigma_lidar_only"),
    "da3_holes_only": ("fused_da3_holes_only", "sigma_da3_holes_only"),
}
MIN_PIXELS = 200  # fewer reference pixels than this: no per-frame scale or sigma
MAD_SCALE = 1.4826


@dataclass
class Options:
    min_conf: int = 1  # LiDAR readings below this ARKit confidence are no reading at all
    gate: float = 3.0  # |d_s - d_m| > gate x sigma of the difference: the sources disagree
    gate_sigma: str = "diff"  # "diff": sqrt(sigma_s^2 + sigma_m^2); "fused": sigma_f, as the plan wrote
    align: bool = True  # scale DA3 to the LiDAR (stills: the mesh) per frame
    edge_weight: float = 0.5  # the thesis' edge term: sigma x (1 + edge_weight x g / g_ref)
    ceiling_factor: float = 2.0  # DA3 sigma x this on ceilings
    ceiling_nz: float = -0.8  # a mesh normal (facing the camera) with z below this is a ceiling
    rel_floor: float = 0.005  # DA3's relative sigma never below 0.5 %
    default_rel: float = 0.05  # DA3's relative sigma when no frame could be measured (thesis: 5 cm + 2 %)
    max_rel: float = 0.2  # frames whose DA3 is worse than this are not used


def robust_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative error of `pred` against `ref` (1-D, same pixels): sqrt(bias^2 + (1.4826 MAD)^2) of (p-r)/r.

    The bias counts as error: with `align` it is ~0 by construction, without it a scale error is exactly
    what the sigma has to describe.
    """
    r = (pred - ref) / ref
    bias = float(np.median(r))
    return float(np.hypot(bias, MAD_SCALE * np.median(np.abs(r - bias))))


def gradient_term(depth: np.ndarray, blur_px: float = 1.5) -> np.ndarray:
    """g / g_ref in [0, 1]: DA3's depth gradient magnitude after a Gaussian blur, over its 99th percentile.

    The thesis divides by the frame's maximum, which one spike sets; the 99th percentile (clipped at 1)
    marks the same edges without that.
    """
    d = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), blur_px) if blur_px > 0 else depth
    gy, gx = np.gradient(d)
    g = np.hypot(gx, gy)
    ref = float(np.percentile(g, 99))
    return np.clip(g / ref, 0.0, 1.0) if ref > 0 else np.zeros_like(g)


def da3_sigma(
    depth: np.ndarray, rel: float, grad: np.ndarray, ceiling: np.ndarray, opts: Options
) -> np.ndarray:
    """DA3's per-pixel sigma (m): rel x depth, x the edge term, x `ceiling_factor` on ceilings."""
    s = max(rel, opts.rel_floor) * depth * (1.0 + opts.edge_weight * grad)
    return np.where(ceiling, s * opts.ceiling_factor, s).astype(np.float32)


def fuse_frame(
    d_s: np.ndarray | None,
    conf: np.ndarray | None,
    sigma_s: np.ndarray | None,
    d_m: np.ndarray | None,
    sigma_m: np.ndarray | None,
    mode: str = "fused",
    min_conf: int = 1,
    gate: float = 3.0,
    shape: tuple[int, ...] | None = None,
    gate_sigma: str = "diff",
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """(depth, sigma, pixel counts) of one frame from the sensor (d_s, conf, sigma_s) and DA3 (d_m, sigma_m).

    0 depth = no value, in the inputs and the output (where the output sigma is 0 too). Either source may
    be None (a still has no LiDAR; `lidar_only` has no DA3); with neither, `shape` gives the (empty)
    output's shape. Counts: `sensor`, `da3` (one source alone),
    `both` (fused), `gated` (disagreement, sensor kept), `masked` (disagreement at low confidence, dropped).
    """
    ref = d_s if d_s is not None else d_m
    if ref is not None:
        shape = ref.shape
    if shape is None:
        raise ValueError("fuse_frame needs a source or a shape")
    zero = np.zeros(shape, np.float32)
    s_ok = np.zeros(shape, bool)
    if d_s is not None and conf is not None and sigma_s is not None:
        s_ok = (d_s > 0) & (conf >= min_conf) & np.isfinite(d_s)
    m_ok = np.zeros(shape, bool)
    if mode != "lidar_only" and d_m is not None and sigma_m is not None:
        m_ok = (d_m > 0) & np.isfinite(d_m) & (sigma_m > 0)
    ds = d_s if d_s is not None and s_ok.any() else zero
    ss = sigma_s if sigma_s is not None and s_ok.any() else zero
    dm = d_m if d_m is not None and m_ok.any() else zero
    sm = sigma_m if sigma_m is not None and m_ok.any() else zero
    cf = conf if conf is not None else np.zeros(shape, np.uint8)

    counts = {"sensor": 0, "da3": 0, "both": 0, "gated": 0, "masked": 0}
    if mode == "lidar_only":
        depth, sigma = np.where(s_ok, ds, 0.0), np.where(s_ok, ss, 0.0)
        counts["sensor"] = int(s_ok.sum())
    elif mode == "da3_holes_only":
        only_m = m_ok & ~s_ok
        depth = np.where(s_ok, ds, np.where(only_m, dm, 0.0))
        sigma = np.where(s_ok, ss, np.where(only_m, sm, 0.0))
        counts["sensor"], counts["da3"] = int(s_ok.sum()), int(only_m.sum())
    elif mode == "fused":
        both = s_ok & m_ok
        with np.errstate(divide="ignore", invalid="ignore"):
            w_s = np.where(s_ok, 1.0 / np.square(np.where(s_ok, ss, 1.0)), 0.0)
            w_m = np.where(m_ok, 1.0 / np.square(np.where(m_ok, sm, 1.0)), 0.0)
            w = w_s + w_m
            fused = np.where(w > 0, (w_s * ds + w_m * dm) / np.where(w > 0, w, 1.0), 0.0)
            sigma_f = np.where(w > 0, 1.0 / np.sqrt(np.where(w > 0, w, 1.0)), 0.0)
        scale = np.hypot(ss, sm) if gate_sigma == "diff" else sigma_f
        disagree = both & (np.abs(ds - dm) > gate * scale)
        keep_sensor = disagree & (cf >= 2)
        drop = disagree & (cf < 2)
        depth = np.where(keep_sensor, ds, fused)
        sigma = np.where(keep_sensor, ss, sigma_f)
        depth = np.where(drop, 0.0, depth)
        sigma = np.where(drop, 0.0, sigma)
        counts["sensor"] = int((s_ok & ~m_ok).sum())
        counts["da3"] = int((m_ok & ~s_ok).sum())
        counts["both"] = int((both & ~disagree).sum())
        counts["gated"], counts["masked"] = int(keep_sensor.sum()), int(drop.sum())
    else:
        raise ValueError(f"unknown mode {mode!r}: one of {', '.join(MODES)}")
    return depth.astype(np.float32), sigma.astype(np.float32), counts


def _resize_np(x: np.ndarray, w: int, h: int, nearest: bool) -> np.ndarray:
    if x.shape[:2] == (h, w):
        return x.astype(np.float32)
    interp = cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR
    return cv2.resize(x.astype(np.float32), (w, h), interpolation=interp)


@dataclass
class FrameInputs:
    """Everything `fuse_frame` needs for one frame, at the prior's size (H, W)."""

    d_s: np.ndarray | None
    conf: np.ndarray | None
    sigma_s: np.ndarray | None
    d_m: np.ndarray | None
    sigma_m: np.ndarray | None
    scale: float | None  # DA3 scale applied (None: no DA3, or not aligned)
    rel: float | None  # DA3's relative sigma used (None: no DA3)
    measured: bool  # rel was measured on this frame (not borrowed)
    dropped: bool  # DA3 was worse than max_rel on this frame


def frame_inputs(
    data: Path,
    frame: Frame,
    prior: dict[str, np.ndarray],
    table: dict,
    opts: Options,
    use_da3: bool,
    fallback_rel: float,
) -> FrameInputs:
    """Load and prepare one frame's sensor and DA3 depth and sigmas at the prior's size."""
    mesh = prior["depth"].astype(np.float32)
    h, w = mesh.shape
    radius = max(1, round(w / 256))  # about one LiDAR pixel, as calibrated
    edge = calibrate.edge_mask(mesh, radius=radius) if (mesh > 0).any() else None

    d_s = conf = sigma_s = None
    lidar = load_depth(frame, w, h)
    if lidar is not None:
        d_s, conf = lidar[0].numpy(), lidar[1].numpy()
        if edge is None:
            edge = calibrate.edge_mask(d_s, radius=radius)
        sigma_s = calibrate.lidar_sigma(table, conf, d_s, edge)
    if edge is None:
        edge = np.zeros((h, w), bool)

    pred = da3.load(data, frame.stem) if use_da3 else None
    if pred is None:
        return FrameInputs(d_s, conf, sigma_s, None, None, None, None, False, False)
    grad = _resize_np(gradient_term(pred[0]), w, h, nearest=False)
    d_m = _resize_np(pred[0], w, h, nearest=True)

    # Reference for DA3's scale and error: confident LiDAR away from edges; for stills, the mesh.
    if d_s is not None and conf is not None and sigma_s is not None:
        ref_ok = (d_s > 0) & (conf >= 2) & ~edge & (d_m > 0)
        ref, ref_rel = d_s, sigma_s / np.maximum(d_s, 1e-6)
    else:
        ref_ok = (mesh > 0) & ~edge & (d_m > 0)
        ref, ref_rel = mesh, np.zeros_like(mesh)
    scale, rel, measured = None, fallback_rel, False
    if ref_ok.sum() >= MIN_PIXELS:
        if opts.align:
            scale = float(np.median(ref[ref_ok] / d_m[ref_ok]))
            d_m = d_m * scale
        meas = robust_rel(d_m[ref_ok], ref[ref_ok])
        own = float(np.median(ref_rel[ref_ok]))
        rel, measured = float(np.sqrt(max(meas * meas - own * own, 0.0))), True
    rel = max(rel, opts.rel_floor)
    if rel > opts.max_rel:
        return FrameInputs(d_s, conf, sigma_s, None, None, scale, rel, measured, True)
    normal = prior["normal"].astype(np.float32) / 127.0
    ceiling = normal[..., 2] < opts.ceiling_nz
    sigma_m = da3_sigma(d_m, rel, grad, ceiling, opts)
    return FrameInputs(d_s, conf, sigma_s, d_m, sigma_m, scale, rel, measured, False)


def _summary(x: list[float]) -> dict[str, float] | None:
    if not x:
        return None
    a = np.asarray(x)
    return {k: round(float(np.percentile(a, q)), 4) for k, q in (("p10", 10), ("median", 50), ("p90", 90))}


def fuse(
    data: Path,
    mode: str = "fused",
    opts: Options | None = None,
    stems: Sequence[str] | None = None,
) -> dict:
    """Write `mode`'s depth and sigma (or every mode's, `mode="all"`) into `prior/<stem>.npz`; return stats.

    Needs the mesh priors (`gs_trainer.priors`) and, except for `lidar_only`, DA3's predictions
    (`gs_trainer.da3`). Runs the LiDAR calibration first if `prior/sigma.json` is missing.
    """
    data = Path(data)
    opts = opts or Options()
    modes = list(MODES) if mode == "all" else [mode]
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown mode {m!r}: one of {', '.join(MODES)} or all")
    started = time.monotonic()
    scene = Scene.load(data)
    frames = [f for f in scene.frames if stems is None or f.stem in set(stems)]
    missing = [f.stem for f in frames if f.prior_path is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frames have no mesh prior (first: {missing[0]}); run python -m gs_trainer.priors first"
        )
    sigma_path = data / "prior" / "sigma.json"
    table = json.loads(sigma_path.read_text()) if sigma_path.is_file() else calibrate.calibrate(data)
    use_da3 = any(m != "lidar_only" for m in modes)
    if use_da3 and not any(da3.da3_path(data, f.stem).is_file() for f in frames):
        raise FileNotFoundError(
            f"no DA3 predictions in {data / 'prior' / 'da3'}; run python -m gs_trainer.da3 first"
        )

    # Frames with LiDAR first, so frames with nothing to measure DA3 against can borrow their median error.
    frames.sort(key=lambda f: f.depth_path is None)
    counts = {m: {"sensor": 0, "da3": 0, "both": 0, "gated": 0, "masked": 0, "none": 0} for m in modes}
    totals = dict.fromkeys(modes, 0)
    scales: list[float] = []
    rels: list[float] = []
    no_da3 = dropped = 0
    for f in frames:
        assert f.prior_path is not None
        with np.load(f.prior_path) as z:
            prior = {k: z[k] for k in z.files}
        fallback = float(np.median(rels)) if rels else opts.default_rel
        fi = frame_inputs(data, f, prior, table, opts, use_da3, fallback)
        if use_da3:
            no_da3 += fi.rel is None
            dropped += fi.dropped
            if fi.scale is not None:
                scales.append(fi.scale)
            if fi.measured and fi.rel is not None and not fi.dropped:
                rels.append(fi.rel)
        for m in modes:
            depth, sigma, c = fuse_frame(
                fi.d_s,
                fi.conf,
                fi.sigma_s,
                fi.d_m,
                fi.sigma_m,
                m,
                opts.min_conf,
                opts.gate,
                prior["depth"].shape,
                opts.gate_sigma,
            )
            for k, v in c.items():
                counts[m][k] += v
            counts[m]["none"] += int((depth <= 0).sum())
            totals[m] += depth.size
            kd, ks = KEYS[m]
            prior[kd], prior[ks] = depth.astype(np.float16), sigma.astype(np.float16)
        tmp = f.prior_path.with_name(f"{f.stem}.tmp.npz")
        np.savez_compressed(tmp, **prior)
        os.replace(tmp, f.prior_path)  # whole files only

    fusion_path = data / "prior" / "fusion.json"
    report = json.loads(fusion_path.read_text()) if fusion_path.is_file() else {}
    report["version"] = VERSION
    shared = {
        "options": asdict(opts),
        "frames": len(frames),
        "sigma_table_frames": table.get("frames"),
        "da3": None,
    }
    if use_da3:
        meta_path = data / "prior" / "da3" / "meta.json"
        shared["da3"] = {
            "meta": json.loads(meta_path.read_text()) if meta_path.is_file() else None,
            "frames_without": no_da3,
            "frames_dropped": dropped,
            "scale": _summary(scales),
            "rel": _summary(rels),
        }
    for m in modes:
        total = max(1, totals[m])
        report[m] = shared | {
            "keys": list(KEYS[m]),
            "pixels": counts[m],
            "share": {k: round(v / total, 4) for k, v in counts[m].items()},
        }
    report["seconds"] = round(time.monotonic() - started, 2)
    fusion_path.write_text(json.dumps(report, indent=1) + "\n")
    return report


def load_fused(
    frame: Frame, W: int, H: int, device: torch.device | str = "cpu", mode: str = "fused"
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(depth (H, W) float32 m, sigma (H, W) float32 m) of `mode` from `prior/<stem>.npz`, or None.

    Nearest resampling, so a target depth is always one of the two sources' values; 0 = no target.
    """
    if frame.prior_path is None or not frame.prior_path.is_file():
        return None
    kd, ks = KEYS[mode]
    with np.load(frame.prior_path) as z:
        if kd not in z.files:
            return None
        d, s = z[kd].astype(np.float32), z[ks].astype(np.float32)
    depth = _resize(torch.from_numpy(d).to(device), W, H, "nearest-exact")
    sigma = _resize(torch.from_numpy(s).to(device), W, H, "nearest-exact")
    return depth, sigma


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m gs_trainer.fusion", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument(
        "--data", type=Path, required=True, help="dataset v2 directory with prior/ (and prior/da3/)"
    )
    ap.add_argument("--mode", default="fused", choices=[*MODES, "all"], help="what to write (default fused)")
    ap.add_argument("--no-align", action="store_true", help="keep DA3's own scale (no per-frame alignment)")
    ap.add_argument("--gate", type=float, default=Options.gate, help="disagreement threshold in sigmas")
    ap.add_argument(
        "--gate-sigma", default="diff", choices=["diff", "fused"], help="sigma of the gate (default diff)"
    )
    ap.add_argument("--min-conf", type=int, default=Options.min_conf, help="lowest ARKit confidence used")
    ap.add_argument(
        "--ceiling-factor", type=float, default=Options.ceiling_factor, help="DA3 sigma x on ceilings"
    )
    ap.add_argument(
        "--max-rel", type=float, default=Options.max_rel, help="drop a frame's DA3 above this error"
    )
    args = ap.parse_args(argv)
    opts = Options(
        min_conf=args.min_conf,
        gate=args.gate,
        gate_sigma=args.gate_sigma,
        align=not args.no_align,
        ceiling_factor=args.ceiling_factor,
        max_rel=args.max_rel,
    )
    report = fuse(args.data, args.mode, opts)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
