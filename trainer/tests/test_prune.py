"""Free-space pruning: the votes drop splats the LiDAR saw through, and `apply` keeps parameters, Adam
moments and strategy state the same length (DefaultStrategy) or retires splats for relocation (MCMC)."""

import numpy as np
import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from gs_trainer import prune


def wall_views(n: int = 4, depth: float = 3.0) -> prune.Views:
    """n depth views from near the origin looking down +z at a wall `depth` metres away."""
    h, w = 24, 32
    d = np.full((n, h, w), depth, np.float32)
    k = np.tile(np.array([30.0, 30.0, w / 2, h / 2]), (n, 1))
    w2c = np.tile(np.eye(4), (n, 1, 1))
    w2c[:, 0, 3] = np.linspace(-0.05, 0.05, n)
    return d, k, w2c


def make_splats(means: list[list[float]]) -> torch.nn.ParameterDict:
    n = len(means)
    t = {
        "means": torch.tensor(means),
        "scales": torch.full((n, 3), float(np.log(0.01))),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.zeros(n),
        "sh0": torch.rand(n, 1, 3),
        "shN": torch.zeros(n, 15, 3),
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in t.items()})


def adam_with_state(splats: torch.nn.ParameterDict) -> dict[str, torch.optim.Optimizer]:
    optimizers: dict[str, torch.optim.Optimizer] = {
        k: torch.optim.Adam([{"params": p, "lr": 1e-3, "name": k}]) for k, p in splats.items()
    }
    loss = sum(p.sum() for p in splats.values())
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    for o in optimizers.values():
        o.step()
    return optimizers


# In front of the wall (free), on it (seen), behind it (no evidence), outside the views (no evidence).
MEANS = [[0.0, 0.0, 1.5], [0.2, 0.1, 3.0], [0.0, 0.0, 4.5], [10.0, 0.0, 2.0], [-0.3, 0.2, 1.0]]


def test_free_space_drop() -> None:
    splats = make_splats(MEANS)
    drop = prune.free_space_drop(splats, wall_views(), min_free=3)
    assert drop.dtype == torch.bool
    assert drop.tolist() == [True, False, False, False, True]
    # Too few views to reach min_free: nothing goes.
    assert not bool(prune.free_space_drop(splats, wall_views(n=2), min_free=3).any())


def test_apply_default_strategy_keeps_everything_aligned() -> None:
    splats = make_splats(MEANS)
    optimizers = adam_with_state(splats)
    strategy = DefaultStrategy()
    state = strategy.initialize_state(scene_scale=1.0)
    state["grad2d"] = torch.arange(5.0)
    state["count"] = torch.ones(5)
    drop = prune.free_space_drop(splats, wall_views())
    assert prune.apply(splats, optimizers, state, drop, strategy) == 2
    for k, p in splats.items():
        assert len(p) == 3, k
        opt = optimizers[k]
        assert opt.param_groups[0]["params"][0] is p
        moments = opt.state[p]
        assert moments["exp_avg"].shape == p.shape and moments["exp_avg_sq"].shape == p.shape
    assert state["grad2d"].tolist() == [1.0, 2.0, 3.0]
    assert state["count"].shape == (3,)
    torch.testing.assert_close(splats["means"][0], torch.tensor(MEANS[1]), atol=2e-3, rtol=0)  # one Adam step
    strategy.check_sanity(splats, optimizers)
    assert prune.apply(splats, optimizers, state, torch.zeros(3, dtype=torch.bool), strategy) == 0


def test_apply_mcmc_retires_below_min_opacity() -> None:
    splats = make_splats(MEANS)
    optimizers = adam_with_state(splats)
    strategy = MCMCStrategy(min_opacity=0.005)
    state = strategy.initialize_state()
    drop = prune.free_space_drop(splats, wall_views())
    assert prune.apply(splats, optimizers, state, drop, strategy) == 2
    assert len(splats["means"]) == 5  # MCMC keeps its budget; relocation reuses the slots
    opa = torch.sigmoid(splats["opacities"].detach())
    assert bool((opa[drop] <= strategy.min_opacity).all()) and bool((opa[~drop] > 0.4).all())
    assert bool((optimizers["opacities"].state[splats["opacities"]]["exp_avg"][drop] == 0).all())
    # gsplat's relocation picks exactly these (it's the dead mask `_relocate_gs` builds).
    dead = torch.sigmoid(splats["opacities"].flatten()) <= strategy.min_opacity
    assert dead.tolist() == drop.tolist()
