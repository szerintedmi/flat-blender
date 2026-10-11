"""Losses and the per-image modules on cases with known answers."""

import math

import torch

from gs_trainer import losses
from gs_trainer.modules import AffineColor, BilateralGrid, CameraOptModule


def test_identical_inputs_cost_nothing() -> None:
    torch.manual_seed(0)
    img = torch.rand(1, 16, 20, 3)
    depth = torch.rand(1, 16, 20, 1) + 1.0
    n = torch.rand(1, 16, 20, 3)
    assert losses.l1(img, img).item() == 0
    assert abs(losses.ssim(img, img).item() - 1.0) < 1e-5
    assert losses.edge_aware_log_l1(depth, depth, img).item() == 0
    assert losses.weighted_l1(depth, depth, torch.ones_like(depth)).item() == 0
    flat = torch.full((1, 16, 20, 3), 0.5)
    assert losses.normal_loss(flat, flat).item() == 0
    assert losses.ssim(img, torch.rand_like(img)).item() < 0.5
    assert losses.normal_loss(n, n).item() > 0  # TV of a noisy prediction


def test_edge_aware_log_l1_down_weights_image_edges() -> None:
    gt = torch.full((1, 8, 8, 1), 2.0)
    pred = gt.clone()
    pred[0, :, 3] += 0.5  # the same depth error on one column in both cases
    flat = torch.full((1, 8, 8, 3), 0.5)
    edge = flat.clone()
    edge[0, :, 4:] = 1.0  # a strong photo edge right of that column
    on_flat = losses.edge_aware_log_l1(pred, gt, flat)
    on_edge = losses.edge_aware_log_l1(pred, gt, edge)
    assert 0 < on_edge.item() < on_flat.item()
    # log(1 + 0.5) on 8 of the 56 x pairs (left pixel in column 3) and 7 of the 56 y pairs.
    expected = math.log1p(0.5) * (8 / 56 + 7 / 56)
    assert abs(on_flat.item() - expected) < 1e-5


def test_depth_masks_and_empty_masks() -> None:
    img = torch.rand(1, 6, 6, 3)
    pred = torch.full((1, 6, 6, 1), 1.0, requires_grad=True)
    no_lidar = torch.zeros(1, 6, 6, 1)
    loss = losses.edge_aware_log_l1(pred, no_lidar, img)
    assert loss.item() == 0
    loss.backward()  # still differentiable
    gt = torch.full((1, 6, 6, 1), 2.0)
    mask = torch.zeros(1, 6, 6, 1, dtype=torch.bool)
    assert losses.edge_aware_log_l1(pred, gt, img, mask).item() == 0
    assert losses.edge_aware_log_l1(pred, gt, img).item() > 0
    w = torch.full_like(gt, 3.0)
    assert abs(losses.weighted_l1(pred, gt, w).item() - 3.0) < 1e-6


def test_flatten_and_scale_ratio() -> None:
    s = torch.log(torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]))
    assert abs(losses.flatten_loss(s).item() - 0.3) < 1e-6
    ok = torch.log(torch.tensor([[0.01, 0.05, 0.05]]))  # ratio 5
    needle = torch.log(torch.tensor([[0.01, 0.01, 0.2]]))  # ratio 20
    assert losses.scale_ratio_loss(ok).item() == 0
    assert abs(losses.scale_ratio_loss(needle).item() - 10.0) < 1e-4


def test_splat_normals_face_the_camera_in_camera_frame() -> None:
    splats = {
        "means": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]),
        "scales": torch.log(torch.tensor([[0.1, 0.1, 0.001], [0.001, 0.1, 0.1]])),
        "quats": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
    }
    n = losses.splat_normals(splats, torch.eye(4))
    torch.testing.assert_close(n[0], torch.tensor([0.0, 0.0, -1.0]))  # thin along z, facing the camera
    assert abs(n[1, 0].item()) == 1.0  # thin along x
    # A camera turned 90° about y, below the first splat: its normal is world -z, seen in camera axes.
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    c2w[:3, 3] = torch.tensor([-3.0, 0.0, 0.5])
    nc = losses.splat_normals(splats, c2w[None])
    assert nc.shape == (1, 2, 3)
    torch.testing.assert_close(c2w[:3, :3] @ nc[0, 0], torch.tensor([0.0, 0.0, -1.0]))


def _moments(means: list[list[float]], opacities: list[float], antialiased: bool = False):
    """Render the depth moments of splats on the optical axis (torch renderer), as the training loop does."""
    from gs_trainer.render import render

    n = len(means)
    t = {
        "means": torch.tensor(means),
        "scales": torch.full((n, 3), math.log(0.3)),
        "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.logit(torch.tensor(opacities)),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 0, 3),
    }
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in t.items()})
    view = torch.eye(4)[None]
    K = torch.tensor([[[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]]])
    extra = losses.depth_moment_signals(splats["means"], view)
    _, _, alpha, info = render(
        splats, view, K, 16, 12, 0, backend="torch", extra_signals=extra, antialiased=antialiased
    )
    return splats, info["render_extra_signals"], alpha


def test_depth_variance_is_zero_for_one_surface_and_positive_for_layers() -> None:
    for aa in (False, True):
        _, m, alpha = _moments([[0.0, 0.0, 2.0]], [0.99], antialiased=aa)
        assert bool((alpha > 0.5).any())
        assert losses.depth_variance_loss(m, alpha).item() < 1e-6  # one splat per pixel, whatever its alpha
        _, m, alpha = _moments([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], [0.6, 0.6], antialiased=aa)
        assert losses.depth_variance_loss(m, alpha).item() < 1e-6  # two splats, same depth

    # A semi-transparent layer 1 m in front of an opaque one on the same ray.
    splats, m, alpha = _moments([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], [0.5, 0.99])
    loss = losses.depth_variance_loss(m, alpha)
    # Centre pixel: weights 0.5 and 0.495 at z 2 and 3 -> Var/E² = w1 w2 / (w1 + w2)² / E² ≈ 0.25 / 2.4975²
    c = m[0, 6, 8]
    ez = c[1] / c[0]
    rel = (c[2] / c[0] - ez**2) / ez**2
    assert abs(rel.item() - 0.25 / 2.4975**2) < 2e-3
    assert loss.item() > 0.01
    loss.backward()
    grads = splats["means"].grad, splats["opacities"].grad
    assert grads[0] is not None and grads[1] is not None
    # Descent (-grad) moves the front layer back and the back layer forward: together along the ray.
    assert grads[0][0, 2] < 0 and grads[0][1, 2] > 0
    # Nothing counts where alpha <= min_alpha.
    assert losses.depth_variance_loss(m, alpha, min_alpha=1.0).item() == 0


def test_opacity_entropy() -> None:
    logits = torch.tensor([-20.0, -3.0, 0.0, 3.0, 20.0])
    per = [losses.opacity_entropy(x[None]).item() for x in logits]
    assert per[0] < 1e-6 and per[-1] < 1e-6  # solid or gone costs nothing
    assert abs(per[2] - math.log(2)) < 1e-6  # the maximum at 0.5
    assert per[0] < per[1] < per[2] and per[2] > per[3] > per[4]
    assert abs(per[1] - per[3]) < 1e-6
    x = torch.tensor([-1.0, 1.0, 1e4], requires_grad=True)
    losses.opacity_entropy(x).backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert x.grad[0] > 0 and x.grad[1] < 0  # gradient descent pushes opacities away from 0.5


def test_modules_start_as_identity() -> None:
    torch.manual_seed(1)
    rgb = torch.rand(2, 10, 12, 3)
    ids = torch.tensor([0, 2])
    grid = BilateralGrid(3, 4, 4, 2)
    torch.testing.assert_close(grid(rgb, ids), rgb)
    assert grid.reg_loss().item() == 0
    affine = AffineColor(3)
    torch.testing.assert_close(affine(rgb, ids), rgb)
    assert affine.reg_loss().item() == 0
    pose = CameraOptModule(3)
    c2w = torch.eye(4)[None].repeat(2, 1, 1)
    c2w[:, :3, 3] = torch.randn(2, 3)
    torch.testing.assert_close(pose(c2w, ids), c2w)
    t, r = pose.magnitudes()
    assert t.max().item() == 0 and r.max().item() < 1e-3
    # Gradients reach every module.
    (grid(rgb, ids).sum() + affine(rgb, ids).sum() + pose(c2w, ids).sum()).backward()
    assert (
        grid.grids.grad is not None and affine.mats.grad is not None and pose.embeds.weight.grad is not None
    )
