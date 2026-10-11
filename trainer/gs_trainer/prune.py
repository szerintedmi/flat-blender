"""LiDAR free-space pruning during training, the msplat behaviour (`pipeline/scanmodel/msplat_train.py`) on gsplat.

Every `prune_every` steps up to `prune_until`, the splats whose centres the LiDAR saw through (see
`pipeline/scanmodel/freespace.py`) are dropped, and training goes on, so the splats on the real surfaces take
over the colour the dropped ones carried. Unlike msplat there is no checkpoint round trip: gsplat's
parameters, Adam moments and strategy state are tensors we hold.

- `DefaultStrategy`: `gsplat.strategy.ops.remove` drops the rows from the parameters, every optimizer's
  moments and the strategy's running state (grad2d, count) at once.
- `MCMCStrategy` keeps a fixed budget and never deletes: it relocates splats whose opacity is at or below
  `min_opacity` onto live ones on its refine steps. So the flagged splats get an opacity below that, and
  pruning runs just before a refine step's `step_post_backward`, which then reuses them.

The splat radius in the votes is `min(max(exp(scales)), 0.10 m)` as in msplat; centres are the means
directly, since we train in the scene frame (no normalisation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove
from numpy.typing import NDArray

Views = tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64]]


def freespace() -> ModuleType:
    """`freespace` ships next to `gs_trainer` on the cloud; locally it lives in `pipeline/scanmodel`."""
    try:
        import freespace as fs  # pyright: ignore[reportMissingImports]
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parents[2] / "pipeline" / "scanmodel"))
        import freespace as fs  # pyright: ignore[reportMissingImports]
    return fs


def load_views(path: Path) -> Views:
    return freespace().load_views(path)


@torch.no_grad()
def free_space_drop(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    views: Views,
    min_free: int = 3,
    max_radius: float = 0.10,
) -> torch.Tensor:
    """Boolean (N,) on the splats' device: True where the LiDAR saw free space at the splat's centre."""
    fs = freespace()
    means = splats["means"]
    centres = means.detach().double().cpu().numpy()
    radius = np.minimum(np.exp(splats["scales"].detach().double().cpu().numpy()).max(axis=1), max_radius)
    free, seen = fs.votes(centres, radius, *views)
    return torch.from_numpy(fs.drop_mask(free, seen, min_free)).to(means.device)


@torch.no_grad()
def apply(
    splats: torch.nn.ParameterDict | dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_state: dict[str, Any],
    drop: torch.Tensor,
    strategy: DefaultStrategy | MCMCStrategy,
) -> int:
    """Drop (Default) or retire for relocation (MCMC) the flagged splats; returns how many."""
    n = int(drop.sum())
    if n == 0:
        return 0
    if isinstance(strategy, MCMCStrategy):
        value = torch.logit(torch.tensor(strategy.min_opacity * 0.5)).item()
        opa = splats["opacities"]
        opa.data[drop] = value
        state = optimizers["opacities"].state.get(opa, {})
        for key, v in state.items():
            if key != "step" and isinstance(v, torch.Tensor) and v.shape[:1] == opa.shape[:1]:
                v[drop] = 0.0
    else:
        remove(params=splats, optimizers=optimizers, state=strategy_state, mask=drop)
    return n
