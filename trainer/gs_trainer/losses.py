"""Training losses: photometric, depth, normals and the anti-needle regularisers.

Images are channel-last like the renderer's output: (C, H, W, 3) colour, (C, H, W, 1) depth and masks.
Masks are boolean; a loss over an empty mask is zero but stays in the graph, so callers never need to
special-case a frame without LiDAR readings.

- `edge_aware_log_l1` is DN-Splatter's depth loss: log(1 + |D - d|), so a few metres of error on a
  window or a mirror don't dominate, weighted by exp(-|∇I|) of the photo, so depth edges, where the
  rendered expected depth blends two surfaces and the 256x192 LiDAR is least reliable, count less.
- `normal_loss` compares normals in DN-Splatter's [0, 1] encoding ((n + 1) / 2): L1 plus total variation
  of the prediction, which smooths the walls the photos can't pin down.
- `flatten_loss` pulls every splat's smallest axis towards zero (DN-Splatter), so splats lie on surfaces
  instead of floating in front of them as blobs.
- `scale_ratio_loss` caps the ratio of the longest to the shortest axis (splatfacto's
  `use_scale_regularization`), the needles seen off the training trajectory.
- `depth_variance_loss` (P6 a3) penalises the spread of the splats' depths along each pixel's ray, a cheap
  3DGS stand-in for 2DGS's depth distortion: the expected-depth loss is satisfied by semi-transparent
  layers spread along a ray (the streaks off the captured path), this one only by one surface. The depth
  moments are rendered as extra signals (`depth_moment_signals`: 1, z, z²) and normalised by their own
  first channel rather than the rendered alpha, because gsplat's antialiased mode weights extra signals
  differently from colour and alpha.
- `opacity_entropy` (P6 a3) is the mean binary entropy of the opacities: splats become solid or vanish,
  so fog dies and surfaces turn opaque.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gs_trainer.render import quat_to_rotmat


def _masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x.mean()
    m = mask.expand_as(x)
    if not bool(m.any()):
        return x.sum() * 0.0
    return x[m].mean()


def l1(a: Tensor, b: Tensor, mask: Tensor | None = None) -> Tensor:
    return _masked_mean(torch.abs(a - b), mask)


def _gaussian_window(size: int, sigma: float, channels: int, like: Tensor) -> Tensor:
    x = torch.arange(size, device=like.device, dtype=like.dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] @ g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(a: Tensor, b: Tensor, window: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean SSIM of two (C, H, W, ch) images in [0, 1], Gaussian window 11, sigma 1.5, zero padding: the
    3DGS reference implementation, so `1 - ssim` matches what gsplat's trainer optimises."""
    x = a.permute(0, 3, 1, 2)
    y = b.permute(0, 3, 1, 2)
    ch = x.shape[1]
    w = _gaussian_window(window, sigma, ch, x)
    pad = window // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=ch)
    mu_y = F.conv2d(y, w, padding=pad, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x = F.conv2d(x * x, w, padding=pad, groups=ch) - mu_x2
    s_y = F.conv2d(y * y, w, padding=pad, groups=ch) - mu_y2
    s_xy = F.conv2d(x * y, w, padding=pad, groups=ch) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu_xy + c1) * (2 * s_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (s_x + s_y + c2))
    return m.mean()


def edge_aware_log_l1(
    depth: Tensor, gt: Tensor, image: Tensor, mask: Tensor | None = None, min_depth: float = 0.1
) -> Tensor:
    """DN-Splatter's EdgeAwareLogL1 on (C, H, W, 1) depths against the (C, H, W, 3) photo.

    Valid where gt > `min_depth` (0 = no LiDAR reading) and `mask`. The x term pairs each pixel with its
    right neighbour, the y term with the one below, as DN-Splatter does; the mask is taken at the left/top
    pixel of each pair."""
    valid = gt > min_depth
    if mask is not None:
        valid = valid & mask
    err = torch.log1p(torch.abs(depth - gt))
    gx = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:]), dim=-1, keepdim=True)
    gy = torch.mean(torch.abs(image[:, :-1] - image[:, 1:]), dim=-1, keepdim=True)
    lx = torch.exp(-gx) * err[:, :, :-1]
    ly = torch.exp(-gy) * err[:, :-1]
    return _masked_mean(lx, valid[:, :, :-1]) + _masked_mean(ly, valid[:, :-1])


def weighted_l1(depth: Tensor, gt: Tensor, weight: Tensor, mask: Tensor | None = None) -> Tensor:
    """Per-pixel weighted L1 (P4: weight = 1/sigma of the fused depth), over gt > 0 and `mask`."""
    valid = gt > 0
    if mask is not None:
        valid = valid & mask
    return _masked_mean(weight * torch.abs(depth - gt), valid)


def normal_loss(pred_n: Tensor, gt_n: Tensor, mask: Tensor | None = None) -> Tensor:
    """L1 + TV on (C, H, W, 3) normals in [0, 1] encoding. TV pairs count where both pixels are in the
    mask, so it doesn't smooth across mesh edges or holes."""
    loss = l1(pred_n, gt_n, mask)
    tv_x = torch.abs(pred_n[:, :, 1:] - pred_n[:, :, :-1])
    tv_y = torch.abs(pred_n[:, 1:] - pred_n[:, :-1])
    if mask is None:
        return loss + tv_x.mean() + tv_y.mean()
    return (
        loss
        + _masked_mean(tv_x, mask[:, :, 1:] & mask[:, :, :-1])
        + _masked_mean(tv_y, mask[:, 1:] & mask[:, :-1])
    )


def flatten_loss(scales: Tensor) -> Tensor:
    """mean(min_i exp(s_i)) over the (N, 3) log scales."""
    return torch.exp(scales.min(dim=-1).values).mean()


def scale_ratio_loss(scales: Tensor, max_ratio: float = 10.0) -> Tensor:
    """splatfacto: mean(max(max_s / min_s, r) - r). Computed from log scales, so no division by zero."""
    ratio = torch.exp(scales.max(dim=-1).values - scales.min(dim=-1).values)
    return (torch.clamp_min(ratio, max_ratio) - max_ratio).mean()


def depth_moment_signals(means: Tensor, viewmats: Tensor) -> Tensor:
    """(C, N, 3) per-splat extra signals [1, z, z²] with z the camera-frame depth of each centre
    (viewmats (C, 4, 4) world-to-camera, OpenCV). Differentiable through the means, so the variance
    loss also pulls splats along the ray towards the pixel's surface."""
    z = torch.einsum("cj,nj->cn", viewmats[:, 2, :3], means) + viewmats[:, 2, 3:4]
    return torch.stack([torch.ones_like(z), z, z * z], dim=-1)


def depth_variance_loss(moments: Tensor, alpha: Tensor, min_alpha: float = 0.5) -> Tensor:
    """mean(Var[z] / E[z]²) over pixels with alpha > `min_alpha`, from the rendered (C, H, W, 3)
    `depth_moment_signals` (Σw, Σwz, Σwz²) and the (C, H, W, 1) alpha.

    Relative, so a far wall isn't punished more than a near one (it is the variance of z / E[z]); the
    denominator is detached so the loss can't shrink by pushing surfaces away."""
    s0, s1, s2 = moments.unbind(-1)
    valid = (alpha.detach()[..., 0] > min_alpha) & (s0.detach() > 1e-3)
    w = s0.clamp_min(1e-6)
    ez = s1 / w
    var = (s2 / w - ez * ez).clamp_min(0.0)
    rel = var / ez.detach().clamp_min(1e-3) ** 2
    return _masked_mean(rel, valid)


def opacity_entropy(opacities: Tensor) -> Tensor:
    """Mean binary entropy (nats, at most ln 2 at 0.5) of sigmoid(`opacities`), the (N,) logits; computed
    with log-sigmoids so it stays finite at saturated opacities."""
    o = torch.sigmoid(opacities)
    return -(o * F.logsigmoid(opacities) + (1 - o) * F.logsigmoid(-opacities)).mean()


def splat_normals(splats: torch.nn.ParameterDict | dict[str, Tensor], c2w: Tensor) -> Tensor:
    """Per-splat normal in the camera frame: the axis of the smallest scale, flipped to face the camera.

    c2w (4, 4) or (C, 4, 4), OpenCV axes; returns (N, 3) or (C, N, 3) unit vectors. Differentiable
    through the quaternions (the choice of axis is not)."""
    single = c2w.dim() == 2
    c2w = c2w[None] if single else c2w
    rot = quat_to_rotmat(splats["quats"])  # (N, 3, 3), columns are the axes
    idx = splats["scales"].argmin(dim=-1)  # (N,)
    n = torch.gather(rot, 2, idx[:, None, None].expand(-1, 3, 1)).squeeze(-1)  # (N, 3) world
    to_cam = c2w[:, None, :3, 3] - splats["means"][None]  # (C, N, 3)
    sign = torch.where((n[None] * to_cam).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    n_world = n[None] * sign
    n_cam = torch.einsum("cji,cnj->cni", c2w[:, :3, :3], n_world)  # Rᵀ n
    return n_cam[0] if single else n_cam
