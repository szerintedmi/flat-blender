"""Train a Gaussian splat with gsplat on dataset v2: our loop around the gsplat library.

    python -m gs_trainer.train --data <dataset> --out <run dir> [--steps 30000] [--max-side 960] [...]

Why our own loop and not gsplat's `simple_trainer.py` (plan, "Decisions"): it reads only COLMAP, can't
resume, and prints a `\\r` progress bar. This one reads `transforms.json` directly, trains in the scene
frame (metres, z up, never normalised, so the PLY lands on `model.glb` as is), and adds what msplat and
Brush lack: LiDAR free-space pruning, depth and normal supervision, pose and colour optimisation, the
anti-needle regularisers and the surface regularisers (depth variance along the rays, opacity entropy). Each is a `Config` option; the defaults are the P1 baseline (gsplat's defaults
plus our pruning, no depth, no normals). The depth loss targets the raw LiDAR or one of `gs_trainer.fusion`'s
priors (`--depth-target`; P4), with DN-Splatter's EdgeAwareLogL1 or the thesis' per-image normalised
1/sigma-weighted L1 (`--depth-weighting`).

What it writes into `--out`, so `gs-trainer cloud` and the pipeline can follow and collect a run:

- `train.log` and stdout: `step=  1200 loss=0.0412 l1=… depth=… splats=1,204,331 ms=38.2` every 100
  steps (`pipeline/scanmodel/progress.py` parses `step=`), plus pruning rounds and the summary;
- `progress.json` (step, total, splats, loss, elapsed, eta, status) every 100 steps;
- `ckpts/latest.pt` every `ckpt_every` steps, written atomically: splats, every optimizer's moments,
  the strategy's state, pose/colour modules and their optimizers, schedulers, the image order, RNG states
  and the config. `--resume auto` continues from it, so a preempted GPU (Modal retries) loses at most
  `ckpt_every` steps;
- at the end `splat.ply` (standard 3DGS PLY via `gsplat.export_splats`), `metrics.json` (splats, wall_s,
  steps, quick PSNR/SSIM on the interp frames with raw poses and no colour correction, pruned) and
  `config.json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from torch import Tensor

from gs_trainer import data, fusion, losses, prune
from gs_trainer.cloud import store_path
from gs_trainer.modules import AffineColor, BilateralGrid, CameraOptModule
from gs_trainer.render import render

SH_C0 = 0.28209479177387814
DEPTH_TARGETS = ("lidar", *fusion.MODES)


def opt(default: Any, help: str, choices: tuple[str, ...] | None = None) -> Any:
    """A Config field with its CLI help (and choices)."""
    return field(default=default, metadata={"help": help, "choices": choices})


@dataclass
class Config:
    data: Path = field(metadata={"help": "dataset v2 directory (transforms.json, split.json, ...)"})
    out: Path = field(metadata={"help": "run directory: train.log, progress.json, ckpts/, splat.ply, ..."})
    # Run
    steps: int = opt(30_000, "training steps")
    max_side: int = opt(960, "training images' longest side in pixels (never enlarged)")
    seed: int = opt(0, "random seed")
    device: str = opt("auto", "torch device; auto = cuda when available, else cpu")
    backend: str = opt(
        "gsplat", "renderer: gsplat (CUDA) or torch (dense reference, tiny scenes)", ("gsplat", "torch")
    )
    resume: str = opt(
        "no", "'auto' continues from out/ckpts/latest.pt if it exists, 'no' starts fresh, or a path"
    )
    cache_images: str = opt(
        "auto", "keep uint8 images on the device: auto (if they fit), yes, no", ("auto", "yes", "no")
    )
    log_every: int = opt(100, "print a step= line and write progress.json every this many steps")
    ckpt_every: int = opt(2000, "write ckpts/latest.pt every this many steps (0 = never)")
    eval: bool = opt(True, "quick PSNR/SSIM on the interp frames at the end")
    # Initialisation
    max_init_points: int = opt(0, "random subset of points.ply to start from (0 = all)")
    init_opacity: float = opt(0.0, "initial opacity (0 = the strategy's preset: 0.1 default, 0.5 mcmc)")
    init_scale: float = opt(
        0.0, "initial scale x mean distance to 3 neighbours (0 = preset: 1.0 default, 0.1 mcmc)"
    )
    sh_degree: int = opt(3, "spherical harmonics degree")
    sh_degree_interval: int = opt(1000, "raise the SH degree in use every this many steps")
    # Optimisation (gsplat's learning rates)
    means_lr: float = opt(1.6e-4, "means learning rate x scene scale, decays exponentially to 1%")
    scales_lr: float = opt(5e-3, "log-scales learning rate")
    opacities_lr: float = opt(5e-2, "logit-opacities learning rate")
    quats_lr: float = opt(1e-3, "quaternions learning rate")
    sh0_lr: float = opt(2.5e-3, "SH degree 0 learning rate")
    shN_lr: float = opt(2.5e-3 / 20, "higher SH learning rate")
    ssim_lambda: float = opt(0.2, "weight of 1 - SSIM against L1")
    antialiased: bool = opt(False, "Mip-Splatting compensation (gsplat rasterize_mode=antialiased)")
    near_plane: float = opt(0.01, "near clipping plane in metres")
    # Densification
    strategy: str = opt("default", "gsplat densification strategy", ("default", "mcmc"))
    absgrad: bool = opt(False, "AbsGS gradients for splitting (default strategy)")
    grow_grad2d: float = opt(0.0, "split/duplicate threshold (0 = gsplat's: 0.0002, or 0.0008 with absgrad)")
    prune_opa: float = opt(0.005, "opacity below which splats are pruned (mcmc: relocated)")
    refine_start: int = opt(500, "first refine step")
    refine_stop: int = opt(0, "last refine step (0 = gsplat's: 15000 default, 25000 mcmc)")
    refine_every: int = opt(100, "refine every this many steps")
    reset_every: int = opt(3000, "reset opacities every this many steps (default strategy)")
    cap_max: int = opt(1_500_000, "splat budget (mcmc)")
    mcmc_opacity_reg: float = opt(0.01, "opacity regulariser (mcmc only)")
    mcmc_scale_reg: float = opt(0.01, "scale regulariser (mcmc only)")
    # LiDAR free-space pruning
    prune: bool = opt(True, "drop splats in LiDAR free space during training (needs views.npz)")
    prune_every: int = opt(1500, "pruning interval in steps")
    prune_until: int = opt(15_000, "last pruning step")
    min_free: int = opt(3, "free-space votes needed (and more free than seen) to drop a splat")
    prune_max_radius: float = opt(0.10, "cap on the splat radius added to the vote margin, metres")
    views: Path | None = field(
        default=None, metadata={"help": "depth views for pruning (default <data>/views.npz)", "choices": None}
    )
    # Depth supervision (LiDAR, or a gs_trainer.fusion prior)
    depth_loss: bool = opt(False, "supervise the rendered depth (renders RGB+ED); target and loss below")
    depth_lambda: float = opt(0.2, "depth loss weight")
    depth_target: str = opt(
        "lidar",
        "lidar: the raw LiDAR with the confidence schedule; fused / lidar_only / da3_holes_only: that"
        " mode's depth and sigma from gs_trainer.fusion (prior/<stem>.npz), stills included",
        DEPTH_TARGETS,
    )
    depth_weighting: str = opt(
        "edge_log",
        "edge_log: EdgeAwareLogL1; sigma: L1 weighted by 1/sigma, normalised per image (needs a fusion target)",
        ("edge_log", "sigma"),
    )
    depth_conf_until: int = opt(7000, "lidar target: confidence ≥ 1 before this step, then confidence 2 only")
    # Normal supervision (mesh priors from gs_trainer.priors)
    normal_loss: bool = opt(False, "L1 + TV of rendered splat normals against the mesh prior normals")
    normal_lambda: float = opt(0.1, "normal loss weight")
    normal_start: int = opt(7000, "first step of the normal loss")
    # Anti-needle regularisers
    flatten_lambda: float = opt(0.0, "weight of mean(min_i exp(s_i)) (DN-Splatter flattening)")
    scale_ratio_lambda: float = opt(0.0, "weight of the max/min scale ratio cap (splatfacto)")
    max_scale_ratio: float = opt(10.0, "the scale ratio cap")
    # Surface regularisers (P6 a3): splats on the surface, not semi-transparent layers along the rays
    depth_var_lambda: float = opt(
        0.0, "weight of the relative depth variance along each ray, mean(Var[z] / E[z]^2) where alpha > 0.5"
    )
    depth_var_start: int = opt(3000, "first step of the depth variance loss")
    opacity_entropy_lambda: float = opt(0.0, "weight of the mean binary entropy of the opacities")
    opacity_entropy_start: int = opt(7000, "first step of the opacity entropy loss")
    # Poses
    pose_opt: bool = opt(False, "optimise a 9-D pose correction per training image")
    pose_opt_lr: float = opt(1e-5, "pose correction learning rate")
    pose_opt_reg: float = opt(1e-6, "pose correction weight decay")
    # Colour
    color: str = opt("none", "per-image colour correction while training", ("none", "bilagrid", "affine"))
    color_lr: float = opt(2e-3, "colour module learning rate")
    bilagrid_x: int = opt(16, "bilateral grid width")
    bilagrid_y: int = opt(16, "bilateral grid height")
    bilagrid_w: int = opt(8, "bilateral grid luminance bins")
    bilagrid_tv: float = opt(10.0, "bilateral grid total variation weight")
    affine_reg: float = opt(0.1, "per-image affine colour: weight of the distance from identity")


def build_parser() -> argparse.ArgumentParser:
    """`--name value` for every Config field, `--x/--no-x` for booleans."""
    p = argparse.ArgumentParser(prog="python -m gs_trainer.train", description=(__doc__ or "").split("\n")[0])
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        kind = hints[f.name]
        help = f.metadata.get("help", "").replace("%", "%%")  # argparse formats help with %
        if f.default is dataclasses.MISSING:
            p.add_argument(flag, type=Path, required=True, help=help)
        elif kind is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default, help=help)
        else:
            base = next((a for a in typing.get_args(kind) if a is not type(None)), kind)
            p.add_argument(
                flag,
                type=base,
                default=f.default,
                choices=f.metadata.get("choices"),
                help=f"{help} [{f.default}]",
            )
    return p


def parse_config(argv: list[str] | None = None) -> Config:
    return Config(**vars(build_parser().parse_args(argv)))


def config_dict(cfg: Config) -> dict[str, Any]:
    """The config as JSON values; local paths repo-relative (`cloud.store_path`)."""
    return {k: store_path(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}


# --- setup -----------------------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def knn_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest other points (gsplat's initial splat size)."""
    from scipy.spatial import KDTree

    if len(points) <= 1:
        return np.full(len(points), 0.01)
    dist, _ = KDTree(points).query(points, k=min(k, len(points) - 1) + 1)
    return np.sqrt((dist[:, 1:] ** 2).mean(axis=1))


def init_splats(
    scene: data.Scene, cfg: Config, device: torch.device
) -> tuple[torch.nn.ParameterDict, dict[str, torch.optim.Optimizer]]:
    """Splats from points.ply as gsplat's `create_splats_with_optimizers`, and one Adam per tensor."""
    points, colors = scene.points, scene.colors
    if cfg.max_init_points and len(points) > cfg.max_init_points:
        keep = np.random.default_rng(cfg.seed).choice(len(points), cfg.max_init_points, replace=False)
        points, colors = points[keep], colors[keep]
    mcmc = cfg.strategy == "mcmc"
    init_scale = cfg.init_scale or (0.1 if mcmc else 1.0)
    init_opacity = cfg.init_opacity or (0.5 if mcmc else 0.1)
    dist = np.maximum(knn_distance(points.astype(np.float64)), 1e-7)
    n = len(points)
    sh = torch.zeros(n, (cfg.sh_degree + 1) ** 2, 3)
    sh[:, 0] = (torch.from_numpy(colors.astype(np.float32) / 255.0) - 0.5) / SH_C0
    tensors = {
        "means": torch.from_numpy(points.astype(np.float32)),
        "scales": torch.log(torch.from_numpy(dist * init_scale).float())[:, None].repeat(1, 3),
        "quats": torch.rand(n, 4),
        "opacities": torch.logit(torch.full((n,), init_opacity)),
        "sh0": sh[:, :1].contiguous(),
        "shN": sh[:, 1:].contiguous(),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in tensors.items()}).to(device)
    return splats, make_optimizers(splats, cfg, scene.scene_scale, device)


def make_optimizers(
    splats: torch.nn.ParameterDict, cfg: Config, scene_scale: float, device: torch.device
) -> dict[str, torch.optim.Optimizer]:
    lrs = {
        "means": cfg.means_lr * scene_scale,
        "scales": cfg.scales_lr,
        "quats": cfg.quats_lr,
        "opacities": cfg.opacities_lr,
        "sh0": cfg.sh0_lr,
        "shN": cfg.shN_lr,
    }
    fused = device.type == "cuda"
    return {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15, fused=fused)
        for name, lr in lrs.items()
    }


def make_strategy(cfg: Config) -> DefaultStrategy | MCMCStrategy:
    if cfg.strategy == "mcmc":
        return MCMCStrategy(
            cap_max=cfg.cap_max,
            refine_start_iter=cfg.refine_start,
            refine_stop_iter=cfg.refine_stop or 25_000,
            refine_every=cfg.refine_every,
            min_opacity=cfg.prune_opa,
        )
    return DefaultStrategy(
        prune_opa=cfg.prune_opa,
        grow_grad2d=cfg.grow_grad2d or (0.0008 if cfg.absgrad else 0.0002),
        refine_start_iter=cfg.refine_start,
        refine_stop_iter=cfg.refine_stop or 15_000,
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
        absgrad=cfg.absgrad,
    )


def exp_decay(optimizer: torch.optim.Optimizer, steps: int) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01 ** (1.0 / max(steps, 1)))


# --- depth targets ---------------------------------------------------------------------------------


def fused_modes(data_dir: Path) -> list[str]:
    """The gs_trainer.fusion modes written for this dataset (keys of `prior/fusion.json`)."""
    path = Path(data_dir) / "prior" / "fusion.json"
    if not path.is_file():
        return []
    report = json.loads(path.read_text())
    return [m for m in fusion.MODES if m in report]


def check_depth_config(cfg: Config) -> None:
    """Refuse depth settings that can't work, before anything is loaded."""
    if not cfg.depth_loss:
        return
    if cfg.depth_weighting == "sigma" and cfg.depth_target == "lidar":
        raise SystemExit(
            "--depth-weighting sigma needs a sigma: use --depth-target lidar_only (the LiDAR with its"
            " calibrated sigma), fused or da3_holes_only"
        )
    if cfg.depth_target != "lidar" and cfg.depth_target not in fused_modes(cfg.data):
        raise SystemExit(
            f"--depth-target {cfg.depth_target}: no fused depth priors in {Path(cfg.data) / 'prior'}"
            " (prior/fusion.json lacks it); run gs_trainer.priors, calibrate, da3, then"
            " `python -m gs_trainer.fusion --mode all` (on the cloud: `gs-trainer cloud push --da3`)"
        )


def depth_supervision(
    cfg: Config, frame: data.Frame, W: int, H: int, device: torch.device, step: int
) -> tuple[Tensor, Tensor, Tensor | None] | None:
    """(target depth (H, W) m, valid (H, W) bool, weight (H, W) or None) for one frame, or None.

    `lidar`: the raw LiDAR, valid at confidence ≥ 1 before `depth_conf_until`, then 2 (stills: None).
    Fusion targets: valid where the target has a depth, stills included. With `depth_weighting="sigma"`
    the weight is 1/sigma over the valid pixels, divided by its mean there (the thesis' per-image
    normalisation, so the loss scale doesn't depend on how many pixels or which sources a frame has).
    """
    sigma: Tensor | None = None
    if cfg.depth_target == "lidar":
        lidar = data.load_depth(frame, W, H, device)
        if lidar is None:
            return None
        d_gt, conf = lidar
        valid = conf >= (1 if step < cfg.depth_conf_until else 2)
    else:
        fused = fusion.load_fused(frame, W, H, device, cfg.depth_target)
        if fused is None:
            return None
        d_gt, sigma = fused
        valid = (d_gt > 0) & (sigma > 0)
    if cfg.depth_weighting != "sigma" or sigma is None:
        return d_gt, valid, None
    w = torch.where(valid, 1.0 / sigma.clamp_min(1e-6), torch.zeros_like(sigma))
    if bool(valid.any()):
        w = w / w[valid].mean()
    return d_gt, valid, w


# --- small output helpers --------------------------------------------------------------------------


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def psnr(a: Tensor, b: Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b).item()
    return float(-10.0 * math.log10(max(mse, 1e-12)))


# --- training --------------------------------------------------------------------------------------


class Logger:
    """train.log (line-buffered) and stdout, flushed per line so remote logs stream."""

    def __init__(self, path: Path, append: bool) -> None:
        self.f = path.open("a" if append else "w", buffering=1)

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self) -> None:
        self.f.close()


def train(cfg: Config) -> dict[str, Any]:
    """Train, write the outputs into `cfg.out` and return the metrics."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpts" / "latest.pt"
    resume_from: Path | None = None
    if cfg.resume == "auto":
        resume_from = ckpt_path if ckpt_path.is_file() else None
    elif cfg.resume not in ("", "no"):
        resume_from = Path(cfg.resume)
    log = Logger(out / "train.log", append=resume_from is not None)
    try:
        return _train(cfg, out, ckpt_path, resume_from, log)
    except BaseException:
        progress = out / "progress.json"
        state = json.loads(progress.read_text()) if progress.is_file() else {"step": 0, "total": cfg.steps}
        write_json(progress, {**state, "status": "failed"})
        raise
    finally:
        log.close()


def _train(cfg: Config, out: Path, ckpt_path: Path, resume_from: Path | None, log: Logger) -> dict[str, Any]:
    device = resolve_device(cfg.device)
    if cfg.backend == "gsplat" and device.type != "cuda":
        raise SystemExit("--backend gsplat needs a CUDA device; --backend torch runs tiny scenes on the CPU")
    check_depth_config(cfg)
    seed_everything(cfg.seed)
    write_json(out / "config.json", config_dict(cfg))

    scene = data.Scene.load(cfg.data)
    train_ids = list(scene.train)
    if not train_ids:
        raise SystemExit(f"{cfg.data}: no training frames")
    slot = {fi: k for k, fi in enumerate(train_ids)}  # frame index -> per-image module row
    cache = {"auto": None, "yes": True, "no": False}[cfg.cache_images]
    images = data.Images(scene, train_ids, cfg.max_side, device, cache=cache)
    c2ws = {fi: torch.tensor(scene.frames[fi].c2w, dtype=torch.float32, device=device) for fi in train_ids}
    log(
        f"{len(train_ids)} training frames at max side {cfg.max_side}"
        f" ({'cached' if images.is_cached else 'from disk'}), scene scale {scene.scene_scale:.2f} m,"
        f" {cfg.strategy} strategy, backend {cfg.backend} on {device}"
    )

    splats, optimizers = init_splats(scene, cfg, device)
    strategy = make_strategy(cfg)
    strategy.check_sanity(splats, optimizers)
    if isinstance(strategy, DefaultStrategy):
        state: dict[str, Any] = strategy.initialize_state(scene_scale=scene.scene_scale)
    else:
        state = strategy.initialize_state()
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {
        "means": exp_decay(optimizers["means"], cfg.steps)
    }

    pose: CameraOptModule | None = None
    pose_optim: torch.optim.Optimizer | None = None
    if cfg.pose_opt:
        pose = CameraOptModule(len(train_ids)).to(device)
        pose_optim = torch.optim.Adam(pose.parameters(), lr=cfg.pose_opt_lr, weight_decay=cfg.pose_opt_reg)
        schedulers["pose"] = exp_decay(pose_optim, cfg.steps)
    color: BilateralGrid | AffineColor | None = None
    color_optim: torch.optim.Optimizer | None = None
    if cfg.color == "bilagrid":
        color = BilateralGrid(len(train_ids), cfg.bilagrid_x, cfg.bilagrid_y, cfg.bilagrid_w).to(device)
        color_optim = torch.optim.Adam(color.parameters(), lr=cfg.color_lr, eps=1e-15)
        schedulers["color"] = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(color_optim, start_factor=0.01, total_iters=1000),
                exp_decay(color_optim, cfg.steps),
            ]
        )
    elif cfg.color == "affine":
        color = AffineColor(len(train_ids)).to(device)
        color_optim = torch.optim.Adam(color.parameters(), lr=cfg.color_lr)
    color_weight = cfg.bilagrid_tv if cfg.color == "bilagrid" else cfg.affine_reg

    views: prune.Views | None = None
    if cfg.prune:
        views_path = Path(cfg.views) if cfg.views else Path(cfg.data) / "views.npz"
        if views_path.is_file():
            views = prune.load_views(views_path)
        else:
            log(f"no {views_path}: free-space pruning off")
    if cfg.depth_loss:
        how = "EdgeAwareLogL1" if cfg.depth_weighting == "edge_log" else "1/sigma-weighted L1"
        log(f"depth loss: {how} against {cfg.depth_target} depth, lambda {cfg.depth_lambda}")
    if cfg.depth_var_lambda > 0:
        log(f"depth variance loss: lambda {cfg.depth_var_lambda} from step {cfg.depth_var_start}")
    if cfg.opacity_entropy_lambda > 0:
        log(
            f"opacity entropy loss: lambda {cfg.opacity_entropy_lambda} from step {cfg.opacity_entropy_start}"
        )
    if cfg.normal_loss and not any(scene.frames[i].prior_path for i in train_ids):
        log(
            "no mesh priors (gs_trainer.priors) for the training frames: normal loss has nothing to compare with"
        )

    start, order, pos, prior_wall = 0, torch.randperm(len(train_ids)).tolist(), 0, 0.0
    prune_log: list[dict[str, int]] = []
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in ck["splats"].items()})
        optimizers = make_optimizers(splats, cfg, scene.scene_scale, device)
        for k, o in optimizers.items():
            o.load_state_dict(ck["optimizers"][k])
        state = ck["strategy_state"]
        schedulers["means"] = exp_decay(optimizers["means"], cfg.steps)
        for k, s in schedulers.items():
            if k in ck["schedulers"]:
                s.load_state_dict(ck["schedulers"][k])
        for module, optim, key in ((pose, pose_optim, "pose"), (color, color_optim, "color")):
            if module is not None and optim is not None and ck.get(key) is not None:
                module.load_state_dict(ck[key])
                optim.load_state_dict(ck[key + "_optim"])
        start, order, pos = int(ck["step"]), list(ck["order"]), int(ck["pos"])
        prior_wall, prune_log = float(ck["wall_s"]), list(ck["prune_log"])
        set_rng_state(ck["rng"])
        log(f"resumed from {resume_from} at step {start}")

    def save_checkpoint(step: int, wall: float) -> None:
        atomic_save(
            {
                "step": step,
                "splats": {k: v.detach() for k, v in splats.items()},
                "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                "strategy": cfg.strategy,
                "strategy_state": state,
                "schedulers": {k: s.state_dict() for k, s in schedulers.items()},
                "pose": pose.state_dict() if pose is not None else None,
                "pose_optim": pose_optim.state_dict() if pose_optim is not None else None,
                "color": color.state_dict() if color is not None else None,
                "color_optim": color_optim.state_dict() if color_optim is not None else None,
                "order": order,
                "pos": pos,
                "wall_s": wall,
                "prune_log": prune_log,
                "rng": rng_state(device),
                "config": config_dict(cfg),
            },
            ckpt_path,
        )

    def progress(step: int, loss: float | None, wall: float, rate: float, status: str) -> None:
        eta = (cfg.steps - step) / rate if rate > 0 else None
        write_json(
            out / "progress.json",
            {
                "step": step,
                "total": cfg.steps,
                "splats": len(splats["means"]),
                "loss": loss,
                "elapsed": round(wall, 1),
                "eta": round(eta, 1) if eta is not None else None,
                "status": status,
            },
        )

    # gsplat tunes MCMC for a world normalised so the cameras sit within ~1 (scene_scale ~1.1 there); ours is
    # metric. This is the factor between the two.
    world_scale = scene.scene_scale / 1.1
    mode = "RGB+ED" if cfg.depth_loss else "RGB"
    window: dict[str, list[float]] = {}
    session_start = time.monotonic()
    tick, tick_step = session_start, start
    for step in range(start, cfg.steps):
        if pos >= len(order):
            order, pos = torch.randperm(len(train_ids)).tolist(), 0
        fi = train_ids[order[pos]]
        pos += 1
        frame = scene.frames[fi]
        image, K = images.get(fi)
        H, W = image.shape[:2]
        ids = torch.tensor([slot[fi]], device=device)
        c2w = c2ws[fi][None]
        if pose is not None:
            c2w = pose(c2w, ids)
        viewmats = data.viewmat(c2w)

        use_normals = cfg.normal_loss and step >= cfg.normal_start
        prior = data.load_prior(frame, W, H, device) if use_normals else None
        use_dvar = cfg.depth_var_lambda > 0 and step >= cfg.depth_var_start
        # Extra render channels: the splat normals (3) and the depth moments 1, z, z² (3), in that order.
        signals = [losses.splat_normals(splats, c2w)[:1]] if prior is not None else []
        if use_dvar:
            signals.append(losses.depth_moment_signals(splats["means"], viewmats))
        extra = torch.cat(signals, dim=-1) if signals else None
        rgb, depth, alpha, info = render(
            splats,
            viewmats,
            K[None],
            W,
            H,
            min(step // cfg.sh_degree_interval, cfg.sh_degree),
            mode=mode,
            backend=cfg.backend,  # pyright: ignore[reportArgumentType]
            antialiased=cfg.antialiased,
            absgrad=cfg.absgrad and isinstance(strategy, DefaultStrategy),
            near_plane=cfg.near_plane,
            extra_signals=extra,
        )
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        if color is not None:
            rgb = color(rgb, ids)

        gt = image[None]
        terms: dict[str, Tensor] = {}
        terms["l1"] = losses.l1(rgb, gt)
        terms["dssim"] = 1.0 - losses.ssim(rgb, gt)
        loss = (1.0 - cfg.ssim_lambda) * terms["l1"] + cfg.ssim_lambda * terms["dssim"]
        target = depth_supervision(cfg, frame, W, H, device, step) if cfg.depth_loss else None
        if target is not None and depth is not None:
            d_gt, valid, weight = target
            d_gt, valid = d_gt[None, ..., None], valid[None, ..., None]
            if weight is not None:
                terms["depth"] = losses.weighted_l1(depth, d_gt, weight[None, ..., None], valid)
            else:
                terms["depth"] = losses.edge_aware_log_l1(depth, d_gt, gt, valid)
            loss = loss + cfg.depth_lambda * terms["depth"]
        if prior is not None:
            _, n_world = prior
            n_gt = n_world @ c2w[0, :3, :3].detach()  # Rᵀ n per pixel: world -> camera
            rendered = info["render_extra_signals"][..., :3]
            valid = (n_world.norm(dim=-1) > 0.5)[None, ..., None] & (alpha.detach() > 0.1)
            pred = (F.normalize(rendered, dim=-1) + 1) / 2
            terms["normal"] = losses.normal_loss(pred, ((n_gt + 1) / 2)[None], valid)
            loss = loss + cfg.normal_lambda * terms["normal"]
        if use_dvar:
            moments = info["render_extra_signals"][..., -3:]
            terms["dvar"] = losses.depth_variance_loss(moments, alpha)
            loss = loss + cfg.depth_var_lambda * terms["dvar"]
        if cfg.opacity_entropy_lambda > 0 and step >= cfg.opacity_entropy_start:
            terms["ent"] = losses.opacity_entropy(splats["opacities"])
            loss = loss + cfg.opacity_entropy_lambda * terms["ent"]
        if cfg.flatten_lambda > 0:
            terms["flat"] = losses.flatten_loss(splats["scales"])
            loss = loss + cfg.flatten_lambda * terms["flat"]
        if cfg.scale_ratio_lambda > 0:
            terms["ratio"] = losses.scale_ratio_loss(splats["scales"], cfg.max_scale_ratio)
            loss = loss + cfg.scale_ratio_lambda * terms["ratio"]
        if isinstance(strategy, MCMCStrategy):
            loss = loss + cfg.mcmc_opacity_reg * torch.sigmoid(splats["opacities"]).mean()
            loss = loss + cfg.mcmc_scale_reg / world_scale * torch.exp(splats["scales"]).mean()
        if color is not None:
            terms["color"] = color.reg_loss()
            loss = loss + color_weight * terms["color"]

        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        for o in (pose_optim, color_optim):
            if o is not None:
                o.step()
                o.zero_grad(set_to_none=True)
        for s in schedulers.values():
            s.step()

        # Free-space pruning on refine steps: before MCMC's relocation (which then reuses the retired
        # splats), after the default strategy's own grow/prune.
        prune_views = views if 0 < step <= cfg.prune_until and step % cfg.prune_every == 0 else None
        if prune_views is not None and isinstance(strategy, MCMCStrategy):
            prune_log.append(_prune(splats, optimizers, state, strategy, prune_views, cfg, step, log))
        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        else:
            # MCMC's position noise is lr * noise_lr applied through each splat's covariance (m²), tuned
            # for gsplat's normalised world (cameras within ~1). In our metric frame that is world_scale²
            # too strong (both the lr and the covariance carry the scale): the first runs diverged once
            # the budget was full (l1 0.14 -> 0.27). Hand it the normalised-world lr.
            strategy.step_post_backward(
                splats,
                optimizers,
                state,
                step,
                info,
                lr=schedulers["means"].get_last_lr()[0] / world_scale**2,
            )
        if prune_views is not None and isinstance(strategy, DefaultStrategy):
            prune_log.append(_prune(splats, optimizers, state, strategy, prune_views, cfg, step, log))

        window.setdefault("loss", []).append(loss.item())
        for k, v in terms.items():
            window.setdefault(k, []).append(v.item())
        done = step + 1
        if done % cfg.log_every == 0 or done == cfg.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.monotonic()
            ms = 1000.0 * (now - tick) / max(done - tick_step, 1)
            tick, tick_step = now, done
            mean = {k: sum(v) / len(v) for k, v in window.items()}
            parts = [f"step={done:>6}", f"loss={mean.pop('loss'):.4f}"]
            parts += [f"{k}={v:.4f}" for k, v in mean.items()]
            parts += [f"splats={len(splats['means']):,}", f"ms={ms:.1f}"]
            log(" ".join(parts))
            wall = prior_wall + now - session_start
            rate = (done - start) / max(now - session_start, 1e-9)
            progress(done, sum(window["loss"]) / len(window["loss"]), wall, rate, "running")
            window = {}
        if cfg.ckpt_every and done % cfg.ckpt_every == 0 and done < cfg.steps:
            save_checkpoint(done, prior_wall + time.monotonic() - session_start)

    wall = prior_wall + time.monotonic() - session_start
    n_exported = export_ply(splats, out / "splat.ply")
    log(f"Saved {out / 'splat.ply'} ({n_exported:,} splats)")
    metrics: dict[str, Any] = {
        "splats": n_exported,
        "wall_s": round(wall, 1),
        "steps": cfg.steps,
        "pruned": sum(r["removed"] for r in prune_log),
        "prune_rounds": prune_log,
    }
    if cfg.eval and scene.interp:
        metrics |= quick_eval(splats, scene, cfg, device)
        log(
            f"interp: PSNR {metrics['psnr']:.3f} SSIM {metrics['ssim']:.4f} ({metrics['eval_frames']} frames)"
        )
    if pose is not None:
        t, r = pose.magnitudes()
        metrics |= {
            "pose_t_median_m": float(t.median()),
            "pose_t_max_m": float(t.max()),
            "pose_r_median_deg": float(r.median()),
            "pose_r_max_deg": float(r.max()),
        }
        log(
            f"pose corrections: median {metrics['pose_t_median_m'] * 100:.2f} cm / {metrics['pose_r_median_deg']:.3f}°"
        )
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", config_dict(cfg))
    progress(cfg.steps, None, wall, 0.0, "done")
    log(f"Training took {wall:.1f} s")
    return metrics


def _prune(
    splats: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, Any],
    strategy: DefaultStrategy | MCMCStrategy,
    views: prune.Views,
    cfg: Config,
    step: int,
    log: Logger,
) -> dict[str, int]:
    n = len(splats["means"])
    drop = prune.free_space_drop(splats, views, cfg.min_free, cfg.prune_max_radius)
    removed = prune.apply(splats, optimizers, state, drop, strategy)
    log(f"Pruned at step {step}: {removed:,} of {n:,} splats in LiDAR free space")
    return {"step": step, "splats": n, "removed": removed}


@torch.no_grad()
def export_ply(splats: torch.nn.ParameterDict, path: Path) -> int:
    """Standard 3DGS PLY in the scene frame (CPU float32); returns the number written (export_splats
    drops rows with NaN/inf)."""
    from gsplat import export_splats

    t = {k: v.detach().float().cpu() for k, v in splats.items()}
    export_splats(
        means=t["means"],
        scales=t["scales"],
        quats=t["quats"],
        opacities=t["opacities"],
        sh0=t["sh0"],
        shN=t["shN"],
        format="ply",
        save_to=str(path),
    )
    finite = torch.ones(len(t["means"]), dtype=torch.bool)
    for v in t.values():
        finite &= torch.isfinite(v.reshape(len(v), -1)).all(dim=1)
    return int(finite.sum())


@torch.no_grad()
def quick_eval(
    splats: torch.nn.ParameterDict, scene: data.Scene, cfg: Config, device: torch.device
) -> dict[str, Any]:
    """PSNR/SSIM on the interp frames at the training size, raw poses, no colour correction: a sanity
    number for the run, not the shared evaluation (`gs_trainer.evaluate`)."""
    images = data.Images(scene, scene.interp, cfg.max_side, device, cache=False)
    p, s = [], []
    for fi in scene.interp:
        image, K = images.get(fi)
        H, W = image.shape[:2]
        c2w = torch.tensor(scene.frames[fi].c2w, dtype=torch.float32, device=device)
        rgb, _, _, _ = render(
            splats,
            data.viewmat(c2w)[None],
            K[None],
            W,
            H,
            cfg.sh_degree,
            mode="RGB",
            backend=cfg.backend,  # pyright: ignore[reportArgumentType]
            antialiased=cfg.antialiased,
            near_plane=cfg.near_plane,
        )
        rgb = rgb.clamp(0, 1)
        p.append(psnr(rgb, image[None]))
        s.append(losses.ssim(rgb, image[None]).item())
    return {"psnr": float(np.mean(p)), "ssim": float(np.mean(s)), "eval_frames": len(p)}


def main(argv: list[str] | None = None) -> int:
    cfg = parse_config(argv)
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
