"""The dense PyTorch renderer: single-splat values, occlusion, gradients, agreement with gsplat's own torch
reference functions (projection, SH), and with the CUDA rasteriser where a GPU is available."""

import math

import pytest
import torch
from gsplat.strategy import DefaultStrategy

from gs_trainer.render import eval_sh, quat_to_rotmat, render

SH_C0 = 0.28209479177387814


def make_splats(means, colors, scale=0.05, opacity=0.9, sh_degree=0) -> torch.nn.ParameterDict:
    means = torch.as_tensor(means, dtype=torch.float32)
    n = len(means)
    sh = torch.zeros(n, (sh_degree + 1) ** 2, 3)
    sh[:, 0] = (torch.as_tensor(colors, dtype=torch.float32) - 0.5) / SH_C0
    t = {
        "means": means,
        "scales": torch.full((n, 3), math.log(scale)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.logit(torch.full((n,), opacity)),
        "sh0": sh[:, :1],
        "shN": sh[:, 1:],
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in t.items()})


W, H = 32, 24
# Principal point on a pixel centre (gsplat samples at x + 0.5), so a splat on the axis hits pixel (16, 12)
# exactly.
K = torch.tensor([[[50.0, 0, 16.5], [0, 50.0, 12.5], [0, 0, 1]]])
EYE = torch.eye(4)[None]


def test_single_splat_centre_pixel_and_depth() -> None:
    splats = make_splats([[0.0, 0.0, 2.0]], [[0.8, 0.4, 0.2]])
    rgb, depth, alpha, info = render(splats, EYE, K, W, H, 0, mode="RGB+ED", backend="torch")
    assert rgb.shape == (1, H, W, 3) and alpha.shape == (1, H, W, 1)
    assert depth is not None and depth.shape == (1, H, W, 1)
    torch.testing.assert_close(alpha[0, 12, 16, 0], torch.tensor(0.9))
    torch.testing.assert_close(rgb[0, 12, 16], 0.9 * torch.tensor([0.8, 0.4, 0.2]))
    torch.testing.assert_close(depth[0, 12, 16, 0], torch.tensor(2.0))
    # Accumulated depth is the weight times z; the corner sees nothing.
    _, d_acc, _, _ = render(splats, EYE, K, W, H, 0, mode="RGB+D", backend="torch")
    assert d_acc is not None
    torch.testing.assert_close(d_acc[0, 12, 16, 0], torch.tensor(1.8))
    assert alpha[0, 0, 0, 0] == 0
    # What the strategies read.
    assert info["means2d"].shape == (1, 1, 2) and info["radii"].shape == (1, 1, 2)
    assert info["radii"].dtype == torch.int32 and bool((info["radii"] > 0).all())
    assert (info["width"], info["height"], info["n_cameras"], info["gaussian_ids"]) == (W, H, 1, None)
    torch.testing.assert_close(info["means2d"][0, 0], torch.tensor([16.5, 12.5]))


def test_front_splat_occludes_and_behind_camera_is_culled() -> None:
    splats = make_splats(
        [[0, 0, 1.0], [0, 0, 3.0], [0, 0, -1.0]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]], opacity=0.99
    )
    rgb, depth, _, info = render(splats, EYE, K, W, H, 0, backend="torch")
    assert depth is not None
    r, g, b = rgb[0, 12, 16].tolist()
    assert r > 0.95 and b < 0.02 and g == 0
    assert 1.0 <= depth[0, 12, 16, 0].item() < 1.05
    assert info["radii"][0, 2].tolist() == [0, 0]  # behind the camera


def test_gradients_flow_to_every_parameter() -> None:
    torch.manual_seed(0)
    splats = make_splats(torch.randn(20, 3) * 0.3 + torch.tensor([0, 0, 2.0]), torch.rand(20, 3), sh_degree=3)
    with torch.no_grad():
        splats["shN"].normal_(0, 0.1)
        splats["quats"].copy_(torch.randn(20, 4))
    rgb, depth, alpha, info = render(splats, EYE, K, W, H, 3, backend="torch", antialiased=True)
    info["means2d"].retain_grad()
    assert depth is not None
    (rgb.sum() + depth.sum() + alpha.sum()).backward()
    for k, p in splats.items():
        assert p.grad is not None and bool(p.grad.abs().sum() > 0), k
    assert info["means2d"].grad.shape == (1, 20, 2)


def test_extra_signals_and_colors_override() -> None:
    splats = make_splats([[0.0, 0.0, 2.0]], [[0.5, 0.5, 0.5]])
    extra = torch.tensor([[[0.0, 0.0, -1.0]]])
    over = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    rgb, _, _, info = render(
        splats, EYE, K, W, H, 0, backend="torch", colors_override=over, extra_signals=extra
    )
    assert rgb.shape == (1, H, W, 4)
    torch.testing.assert_close(rgb[0, 12, 16], 0.9 * over[0])
    torch.testing.assert_close(info["render_extra_signals"][0, 12, 16], torch.tensor([0, 0, -0.9]))


def test_matches_gsplat_torch_projection_and_sh() -> None:
    from gsplat.cuda._torch_impl import _fully_fused_projection, _spherical_harmonics

    from gs_trainer.render import _project

    torch.manual_seed(1)
    n = 50
    means = torch.randn(n, 3) + torch.tensor([0, 0, 3.0])
    rot = quat_to_rotmat(torch.randn(n, 4))
    m = rot * torch.rand(n, 1, 3) * 0.2
    covars = m @ m.transpose(-1, -2)
    view = torch.eye(4)
    view[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    ours = _project(means, covars, view[None], K, W, H, 0.3, 0.01, 1e10, True)
    ref = _fully_fused_projection(means, covars, view[None], K, W, H, 0.3, calc_compensations=True)
    for a, b in zip(ours, ref, strict=True):
        assert a is not None and b is not None
        torch.testing.assert_close(a, b)

    dirs = torch.nn.functional.normalize(torch.randn(2, n, 3), dim=-1)
    coeffs = torch.randn(n, 16, 3)
    for degree in range(4):
        torch.testing.assert_close(eval_sh(degree, dirs, coeffs), _spherical_harmonics(degree, dirs, coeffs))


def test_default_strategy_runs_on_torch_info() -> None:
    torch.manual_seed(2)
    splats = make_splats(torch.randn(30, 3) * 0.3 + torch.tensor([0, 0, 2.0]), torch.rand(30, 3))
    optimizers: dict[str, torch.optim.Optimizer] = {
        k: torch.optim.Adam([{"params": p, "lr": 1e-3, "name": k}]) for k, p in splats.items()
    }
    strategy = DefaultStrategy(refine_start_iter=0, refine_every=2, grow_grad2d=1e-9)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=1.0)
    for step in range(3):
        rgb, _, _, info = render(splats, EYE, K, W, H, 0, mode="RGB", backend="torch", absgrad=True)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        rgb.mean().backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
    assert len(splats["means"]) > 30  # duplicated with a near-zero threshold
    assert state["grad2d"].shape == (len(splats["means"]),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat's rasteriser needs CUDA")
def test_torch_backend_matches_gsplat_cuda() -> None:
    torch.manual_seed(3)
    n = 200
    splats = make_splats(torch.randn(n, 3) * 0.5 + torch.tensor([0, 0, 3.0]), torch.rand(n, 3), sh_degree=3)
    with torch.no_grad():
        splats["shN"].normal_(0, 0.1)
        splats["quats"].copy_(torch.randn(n, 4))
        splats["scales"].copy_(torch.log(torch.rand(n, 3) * 0.1 + 0.01))
        splats["opacities"].copy_(torch.logit(torch.rand(n) * 0.9 + 0.05))
    splats = splats.cuda()
    view, k = EYE.cuda(), K.cuda()
    extra = torch.randn(1, n, 3, device="cuda")
    for aa in (False, True):
        out = [
            render(splats, view, k, W, H, 3, mode="RGB+ED", backend=b, antialiased=aa, extra_signals=extra)
            for b in ("torch", "gsplat")
        ]
        (rt, dt, at, it), (rg, dg, ag, ig) = out
        assert dt is not None and dg is not None
        torch.testing.assert_close(rt, rg, atol=2e-3, rtol=1e-3)
        torch.testing.assert_close(at, ag, atol=2e-3, rtol=1e-3)
        mask = at > 0.2
        torch.testing.assert_close(dt[mask], dg[mask], atol=5e-3, rtol=1e-3)
        torch.testing.assert_close(it["means2d"], ig["means2d"], atol=1e-3, rtol=1e-4)
        if not aa:  # antialiased: gsplat weights extra signals differently from colour (checked on an L40S:
            # RGB/alpha/depth agree, extras don't); training uses gsplat's, the torch path is tests only
            torch.testing.assert_close(
                it["render_extra_signals"], ig["render_extra_signals"], atol=5e-3, rtol=1e-3
            )
        assert it["radii"].shape == ig["radii"].shape
