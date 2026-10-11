# Parts of this file are adapted from gsplat's examples (examples/utils.py: CameraOptModule,
# rotation_6d_to_matrix; examples/lib_bilagrid.py: BilateralGrid, slicing, total_variation_loss), commit
# 512d366, https://github.com/nerfstudio-project/gsplat.
#
# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team
# and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Yuehao Wang (https://github.com/yuehaowang), "Bilateral Guided
# Radiance Field Processing" (https://bilarfpro.github.io/).
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Changes: typed, the bilateral grid without tensorly or the CP-4D variant, and a per-image affine colour
# module added.
"""Per-image modules trained next to the splats: pose corrections and colour correction.

- `CameraOptModule`: one 9-D correction per training image (3 translation + 6-D rotation, zero = no
  change), applied on the right of camera-to-world, so the translation is in the camera's own frame.
  Corrections larger than ~2 cm / 0.5° point at the sessions' alignment rather than per-frame noise
  (plan P3), so `magnitudes()` reports them.
- `BilateralGrid`: a 16x16x8 grid of 3x4 affine colour transforms per image, sliced by pixel position
  and luminance (gsplat's `post_processing="bilateral_grid"`). It absorbs exposure and white-balance
  differences (stills vs video frames) so the splat keeps one neutral colour. Regularised by total
  variation.
- `AffineColor`: one 3x4 colour transform per image, identity at the start, the lighter option if the
  grid overfits.

Both colour modules map (C, H, W, 3) renders with the images' indices (C,) to corrected (C, H, W, 3), and
expose `reg_loss()`. They are used for training only: the PLY carries the uncorrected colour.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """6-D rotation (Zhou et al. 2019) -> (…, 3, 3) by Gram-Schmidt, rows b1, b2, b3 as in pytorch3d."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOptModule(nn.Module):
    """Per-image pose deltas: c2w' = c2w @ [R(δrot) | δt]."""

    identity: Tensor

    def __init__(self, n: int) -> None:
        super().__init__()
        self.embeds = nn.Embedding(n, 9)
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.embeds.weight)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """camtoworlds (…, 4, 4), embed_ids (…) -> adjusted camtoworlds (…, 4, 4)."""
        batch = camtoworlds.shape[:-2]
        deltas = self.embeds(embed_ids)
        dx, drot = deltas[..., :3], deltas[..., 3:]
        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch, -1))
        top = torch.cat([rot, dx[..., None]], dim=-1)
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=deltas.device).expand(*batch, 1, 4)
        return camtoworlds @ torch.cat([top, bottom], dim=-2)

    @torch.no_grad()
    def magnitudes(self) -> tuple[Tensor, Tensor]:
        """(translation metres (n,), rotation degrees (n,)) of every image's correction."""
        deltas = self.embeds.weight
        rot = rotation_6d_to_matrix(deltas[:, 3:] + self.identity)
        cos = ((rot.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0)
        return deltas[:, :3].norm(dim=-1), torch.rad2deg(torch.arccos(cos))


def total_variation_loss(x: Tensor) -> Tensor:
    """Mean squared difference between neighbours along every dimension after the first two (B, C, …)."""
    tv = x.new_zeros(())
    for d in range(2, x.dim()):
        n = x.shape[d]
        diff = x.narrow(d, 1, n - 1) - x.narrow(d, 0, n - 1)
        tv = tv + diff.pow(2).sum() / max(diff[0].numel(), 1)
    return tv / x.shape[0]


def _apply_affine(mats: Tensor, rgb: Tensor) -> Tensor:
    """mats (…, 3, 4), rgb (…, 3)."""
    return (mats[..., :3] @ rgb[..., None]).squeeze(-1) + mats[..., 3]


class BilateralGrid(nn.Module):
    """One bilateral grid of affine colour transforms per image, (n, 12, L, H, W), identity at the start."""

    rgb2gray: Tensor

    def __init__(self, n: int, grid_x: int = 16, grid_y: int = 16, grid_w: int = 8) -> None:
        super().__init__()
        eye = torch.tensor([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0])
        grid = eye.repeat(grid_w * grid_y * grid_x, 1).reshape(1, grid_w, grid_y, grid_x, 12)
        self.grids = nn.Parameter(grid.permute(0, 4, 1, 2, 3).tile(n, 1, 1, 1, 1).contiguous())
        self.register_buffer("rgb2gray", torch.tensor([0.299, 0.587, 0.114]))

    def forward(self, rgb: Tensor, ids: Tensor) -> Tensor:
        """rgb (C, H, W, 3) in [0, 1], ids (C,) -> corrected (C, H, W, 3)."""
        C, H, W, _ = rgb.shape
        ys, xs = torch.meshgrid(
            (torch.arange(H, device=rgb.device, dtype=rgb.dtype) + 0.5) / H,
            (torch.arange(W, device=rgb.device, dtype=rgb.dtype) + 0.5) / W,
            indexing="ij",
        )
        xy = torch.stack([xs, ys], -1)[None].expand(C, -1, -1, -1) * 2 - 1  # (C, H, W, 2) in [-1, 1]
        z = (rgb @ self.rgb2gray)[..., None] * 2 - 1  # luminance guidance in [-1, 1]
        coords = torch.cat([xy, z], dim=-1)[:, None]  # (C, 1, H, W, 3)
        mats = F.grid_sample(
            self.grids[ids], coords, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (C, 12, 1, H, W)
        mats = mats[:, :, 0].permute(0, 2, 3, 1).reshape(C, H, W, 3, 4)
        return _apply_affine(mats, rgb)

    def reg_loss(self) -> Tensor:
        return total_variation_loss(self.grids)


class AffineColor(nn.Module):
    """One 3x4 colour transform per image, identity at the start."""

    def __init__(self, n: int) -> None:
        super().__init__()
        eye = torch.eye(3, 4)
        self.mats = nn.Parameter(eye[None].repeat(n, 1, 1))

    def forward(self, rgb: Tensor, ids: Tensor) -> Tensor:
        return _apply_affine(self.mats[ids][:, None, None], rgb)

    def reg_loss(self) -> Tensor:
        """Mean squared distance from identity: keeps the images' transforms from drifting together,
        which would only move colour between the splat and every image's transform."""
        return (self.mats - torch.eye(3, 4, device=self.mats.device)).pow(2).mean()
