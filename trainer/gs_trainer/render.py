"""Render splats: gsplat's CUDA rasteriser on the GPU, a dense PyTorch reference everywhere else.

    rgb, depth, alpha, info = render(splats, viewmats, Ks, W, H, sh_degree, mode="RGB+ED")

`splats` holds gsplat's raw parameters (log scales, logit opacities, SH in `sh0`/`shN`, see the README);
this module applies the activations, so training, evaluation and tests all render the same way.

`backend="gsplat"` calls `gsplat.rasterization` (CUDA only: even its "torch" code path still uses the CUDA
tile intersection). `backend="torch"` is a dense O(N·H·W) re-implementation of the same maths in plain
PyTorch, so the training loop, gsplat's densification strategies and the losses run in CPU tests on tiny
scenes. It follows gsplat closely so the two agree to float precision on small scenes (checked by a
CUDA-only test):

- projection: means into the camera, EWA 2D covariance J Σ Jᵀ with the same clamped Jacobian, plus
  `eps2d` (0.3 px²) on the diagonal; `antialiased` multiplies opacities by the Mip-Splatting compensation
  sqrt(det Σ / det(Σ + eps2d·I));
- radii: ceil(3.33·sigma) per axis, zero behind the near plane or off-screen (the strategies read them);
- colour: SH up to degree 3 towards the viewing direction, + 0.5, clamped at 0;
- compositing: front to back by camera z (gsplat's global z order), alpha = min(0.99, o·exp(-½dᵀΣ⁻¹d))
  at pixel centres (x + 0.5), alphas below 1/255 skipped, stopping once transmittance would fall
  to 1e-4;
- depth: "D" accumulates Σwz, "ED" divides by the accumulated alpha.

`info` carries what `DefaultStrategy` / `MCMCStrategy` read in gsplat 512d366 (`means2d` [C, N, 2] in the
graph so `retain_grad()` works, `radii` [C, N, 2] int, `width`, `height`, `n_cameras`, `gaussian_ids` None
because we render unpacked), plus `depths`, `conics`, `opacities`. `absgrad` in the torch backend is an
approximation: |∂L/∂means2d| summed over the image rather than per pixel, which is enough to exercise the
code path.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

Backend = Literal["gsplat", "torch"]
Mode = Literal["RGB", "RGB+D", "RGB+ED"]

MAX_ALPHA = 0.99
ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4
MIN_COMPENSATION = 0.005


# Channel counts gsplat's rasterizer is compiled for (GSPLAT_NUM_CHANNELS in gsplat/cuda/csrc/Config.h @512d366).
_GSPLAT_CHANNELS = frozenset({1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 21, 23, 24, 32, 33, 64, 65})


def render(
    splats: torch.nn.ParameterDict | dict[str, Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    W: int,
    H: int,
    sh_degree: int,
    mode: Mode = "RGB+ED",
    backend: Backend = "gsplat",
    colors_override: Tensor | None = None,
    **kw: Any,
) -> tuple[Tensor, Tensor | None, Tensor, dict[str, Any]]:
    """Render C views of the splats.

    viewmats (C, 4, 4) world-to-camera (OpenCV axes), Ks (C, 3, 3) at WxH. Returns rgb (C, H, W, D)
    (D = 3, or the channels of `colors_override`), depth (C, H, W, 1) or None for mode "RGB", alpha
    (C, H, W, 1) and gsplat's `info`.

    `colors_override` (N, D) or (C, N, D) renders those post-activation values instead of the SH colour
    (`sh_degree` is then ignored). Keyword options: `antialiased` (bool), `absgrad` (bool), `near_plane`,
    `far_plane`, `eps2d`, `backgrounds` (C, D), `extra_signals` (C, N, E) rendered alongside and returned
    in `info["render_extra_signals"]` (C, H, W, E), accumulated like colour (Σ w·e, not normalised).
    """
    if mode not in ("RGB", "RGB+D", "RGB+ED"):
        raise ValueError(f"unsupported render mode {mode!r}")
    antialiased = bool(kw.pop("antialiased", False))
    absgrad = bool(kw.pop("absgrad", False))
    near_plane = float(kw.pop("near_plane", 0.01))
    far_plane = float(kw.pop("far_plane", 1e10))
    eps2d = float(kw.pop("eps2d", 0.3))
    backgrounds: Tensor | None = kw.pop("backgrounds", None)
    extra: Tensor | None = kw.pop("extra_signals", None)
    if kw:
        raise TypeError(f"unknown render options: {sorted(kw)}")

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    if colors_override is not None:
        colors, degree = colors_override, None
    else:
        colors, degree = torch.cat([splats["sh0"], splats["shN"]], 1), sh_degree

    if backend == "gsplat":
        from gsplat import rasterization

        # gsplat's kernels exist only for the channel counts in GSPLAT_NUM_CHANNELS (csrc/Config.h: ..., 6,
        # 8, 9, 16, ...). RGB + expected depth + 3 normal channels is 7, so pad the extra signals with zeros
        # up to a compiled count and cut the padding off the result.
        n_extra = 0 if extra is None else extra.shape[-1]
        if extra is not None:
            base = colors.shape[-1] + (1 if "D" in mode else 0)
            pad = (
                next(k for k in range(64) if base + n_extra + k in _GSPLAT_CHANNELS)
                if base + n_extra < 64
                else 0
            )
            if pad:
                extra = torch.cat([extra, extra.new_zeros((*extra.shape[:-1], pad))], -1)
        out, alpha, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            sh_degree=degree,
            packed=False,
            absgrad=absgrad,
            rasterize_mode="antialiased" if antialiased else "classic",
            render_mode=mode,
            backgrounds=backgrounds,
            extra_signals=extra,
        )
        if extra is not None and "render_extra_signals" in info:
            info["render_extra_signals"] = info["render_extra_signals"][..., :n_extra]
    elif backend == "torch":
        out, alpha, info = _rasterize_torch(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            W,
            H,
            sh_degree=degree,
            mode=mode,
            antialiased=antialiased,
            absgrad=absgrad,
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            backgrounds=backgrounds,
            extra=extra,
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    d = colors.shape[-1]
    rgb = out[..., :d]
    depth = out[..., d : d + 1] if mode != "RGB" else None
    return rgb, depth, alpha, info


# --- the dense PyTorch reference -------------------------------------------------------------------


def quat_to_rotmat(quats: Tensor) -> Tensor:
    """(..., 4) wxyz, not necessarily normalised -> (..., 3, 3)."""
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quats.shape[:-1], 3, 3)


def eval_sh(degree: int, dirs: Tensor, coeffs: Tensor) -> Tensor:
    """SH colour (without the +0.5) for unit `dirs` (..., N, 3) and `coeffs` (N, K, D), degree ≤ 3.

    Same real SH basis and signs as gsplat (Sloan's formulation, as in `_eval_sh_bases_fast`)."""
    x, y, z = dirs.unbind(-1)
    bases = [torch.full_like(x, 0.2820947917738781)]
    if degree >= 1:
        a = -0.48860251190292
        bases += [a * y, -a * z, a * x]
    if degree >= 2:
        z2 = z * z
        b = -1.092548430592079 * z
        a = 0.5462742152960395
        c1 = x * x - y * y
        s1 = 2 * x * y
        bases += [a * s1, b * y, 0.9461746957575601 * z2 - 0.3153915652525201, b * x, a * c1]
    if degree >= 3:
        z2 = z * z
        c1 = x * x - y * y
        s1 = 2 * x * y
        tc = -2.285228997322329 * z2 + 0.4570457994644658
        tb = 1.445305721320277 * z
        ta = -0.5900435899266435
        c2 = x * c1 - y * s1
        s2 = x * s1 + y * c1
        bases += [
            ta * s2,
            tb * s1,
            tc * y,
            z * (1.865881662950577 * z2 - 1.119528997770346),
            tc * x,
            tb * c1,
            ta * c2,
        ]
    if degree > 3:
        raise ValueError("SH degree above 3 is not supported")
    basis = torch.stack(bases, dim=-1)  # (..., N, k)
    return (basis[..., None] * coeffs[:, : basis.shape[-1]]).sum(dim=-2)


def _project(
    means: Tensor,
    covars: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    W: int,
    H: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    antialiased: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """gsplat's `_fully_fused_projection` for a pinhole camera: (radii [C, N, 2] int32, means2d
    [C, N, 2], depths [C, N], conics [C, N, 3], compensations [C, N] or None)."""
    R = viewmats[:, :3, :3]
    t = viewmats[:, :3, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]
    covars_c = R[:, None] @ covars[None] @ R[:, None].transpose(-1, -2)

    tx, ty, tz = means_c.unbind(-1)
    fx, fy = Ks[:, 0, 0, None], Ks[:, 1, 1, None]
    cx, cy = Ks[:, 0, 2, None], Ks[:, 1, 2, None]
    tan_x, tan_y = 0.5 * W / fx, 0.5 * H / fy
    lim_xp, lim_xn = (W - cx) / fx + 0.3 * tan_x, cx / fx + 0.3 * tan_x
    lim_yp, lim_yn = (H - cy) / fy + 0.3 * tan_y, cy / fy + 0.3 * tan_y
    jx = tz * torch.clamp(tx / tz, min=-lim_xn, max=lim_xp)
    jy = tz * torch.clamp(ty / tz, min=-lim_yn, max=lim_yp)
    zero = torch.zeros_like(tz)
    J = torch.stack([fx / tz, zero, -fx * jx / tz**2, zero, fy / tz, -fy * jy / tz**2], -1)
    J = J.reshape(*tz.shape, 2, 3)
    cov2d = J @ covars_c @ J.transpose(-1, -2)
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means_c) / tz[..., None]

    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    cov2d = cov2d + eps2d * torch.eye(2, device=means.device, dtype=means.dtype)
    det = (cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]).clamp(min=1e-10)
    comp = torch.sqrt(torch.clamp(det_orig / det, min=MIN_COMPENSATION**2)) if antialiased else None
    conics = torch.stack(
        [cov2d[..., 1, 1] / det, -(cov2d[..., 0, 1] + cov2d[..., 1, 0]) / 2.0 / det, cov2d[..., 0, 0] / det],
        dim=-1,
    )
    with torch.no_grad():
        radius = torch.stack(
            [
                torch.ceil(3.33 * torch.sqrt(cov2d[..., 0, 0])),
                torch.ceil(3.33 * torch.sqrt(cov2d[..., 1, 1])),
            ],
            dim=-1,
        )
        valid = (tz > near_plane) & (tz < far_plane)
        inside = (
            (means2d[..., 0] + radius[..., 0] > 0)
            & (means2d[..., 0] - radius[..., 0] < W)
            & (means2d[..., 1] + radius[..., 1] > 0)
            & (means2d[..., 1] - radius[..., 1] < H)
        )
        radius = torch.where((valid & inside)[..., None], radius, torch.zeros_like(radius))
        radii = torch.nan_to_num(radius).int()
    return radii, means2d, tz, conics, comp


def _rasterize_torch(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    W: int,
    H: int,
    *,
    sh_degree: int | None,
    mode: Mode,
    antialiased: bool,
    absgrad: bool,
    near_plane: float,
    far_plane: float,
    eps2d: float,
    backgrounds: Tensor | None,
    extra: Tensor | None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    C, N = viewmats.shape[0], means.shape[0]
    rot = quat_to_rotmat(quats)
    M = rot * scales[:, None, :]
    covars = M @ M.transpose(-1, -2)
    radii, means2d, depths, conics, comp = _project(
        means, covars, viewmats, Ks, W, H, eps2d, near_plane, far_plane, antialiased
    )
    if absgrad:

        def _absgrad(grad: Tensor) -> None:
            means2d.absgrad = grad.abs()  # pyright: ignore[reportAttributeAccessIssue]

        means2d.register_hook(_absgrad)

    if sh_degree is None:
        feats = colors if colors.dim() == 3 else colors[None].expand(C, N, -1)
    else:
        campos = torch.linalg.inv(viewmats)[:, :3, 3]
        dirs = F.normalize(means[None] - campos[:, None], dim=-1)
        feats = torch.clamp_min(eval_sh(sh_degree, dirs, colors) + 0.5, 0.0)
    D = feats.shape[-1]
    if extra is not None:
        feats = torch.cat([feats, extra], dim=-1)
    if mode != "RGB":
        feats = torch.cat([feats, depths[..., None]], dim=-1)
    opac = opacities[None].expand(C, N)
    if comp is not None:
        opac = opac * comp

    ys, xs = torch.meshgrid(
        torch.arange(H, device=means.device, dtype=means.dtype) + 0.5,
        torch.arange(W, device=means.device, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    pix = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)  # (P, 2)
    outs, alphas = [], []
    for c in range(C):
        vis = torch.nonzero((radii[c] > 0).all(-1)).squeeze(-1)
        vis = vis[torch.argsort(depths[c, vis].detach())]
        d = pix[None] - means2d[c, vis][:, None]  # (n, P, 2)
        a, b, cc = conics[c, vis].unbind(-1)
        sigma = (
            0.5 * (a[:, None] * d[..., 0] ** 2 + cc[:, None] * d[..., 1] ** 2)
            + b[:, None] * d[..., 0] * d[..., 1]
        )
        alpha = torch.clamp_max(opac[c, vis][:, None] * torch.exp(-sigma), MAX_ALPHA)
        alpha = torch.where((sigma >= 0) & (alpha >= ALPHA_THRESHOLD), alpha, torch.zeros_like(alpha))
        t_incl = torch.cumprod(1.0 - alpha, dim=0)
        t_excl = torch.cat([torch.ones_like(t_incl[:1]), t_incl[:-1]], dim=0)
        w = alpha * t_excl * (t_incl > TRANSMITTANCE_THRESHOLD)
        out = w.transpose(0, 1) @ feats[c, vis]  # (P, F)
        acc = w.sum(0)[:, None]  # (P, 1)
        if mode == "RGB+ED":
            out = torch.cat([out[:, :-1], out[:, -1:] / acc.clamp(min=1e-10)], dim=-1)
        if backgrounds is not None:
            out = torch.cat([out[:, :D] + (1.0 - acc) * backgrounds[c], out[:, D:]], dim=-1)
        outs.append(out.reshape(H, W, -1))
        alphas.append(acc.reshape(H, W, 1))
    out = torch.stack(outs)
    # Keep means2d in the graph even when nothing is visible, so the strategies always find a gradient.
    out = out + 0.0 * means2d.sum()
    info: dict[str, Any] = {
        "batch_ids": None,
        "camera_ids": None,
        "gaussian_ids": None,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opac,
        "width": W,
        "height": H,
        "n_batches": 1,
        "n_cameras": C,
    }
    if extra is not None:
        e = extra.shape[-1]
        info["render_extra_signals"] = out[..., D : D + e]
        out = torch.cat([out[..., :D], out[..., D + e :]], dim=-1)
    return out, torch.stack(alphas), info
