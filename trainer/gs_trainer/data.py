"""Dataset v2 for the trainer: frames, split, init points, images, LiDAR depth and mesh priors.

Reads what `scanmodel splat` writes (trainer/README.md, "Dataset v2") directly, with no COLMAP detour:

- Cameras. transforms.json holds camera-to-world in OpenGL axes (x right, y up, z back), as nerfstudio
  wants it; gsplat wants OpenCV axes (x right, y down, z forward), so `c2w_cv = c2w_gl @ diag(1,-1,-1,1)`.
  Each frame keeps its own intrinsics at its own image size: stills and video frames come from different
  cameras and sizes. `camera_idx` (0 video, 1 still) lets colour correction tell the two apart.
- Split. Train = transforms.json frames minus interp (every 8th); the extrap frames (the closing lap)
  exist only in split.json, so msplat and Brush never see them. `Scene.frames` is the transforms.json
  frames followed by the extrap frames, and `train` / `interp` / `extrap` index into it.
- Images. Resized to a max side with INTER_AREA (the training size is a flag, not an `images_N/` folder)
  and K scaled with them. Kept as uint8 on the device when they fit comfortably (no decode per step;
  1,600 images at 1920 px are 13 GB, fine on a 48 GB card), else decoded from disk on every call.
- Depth. LiDAR stays at its native 256x192 on disk; it is upsampled here, nearest (a depth between two
  surfaces is a surface nobody saw), on the device. Its intrinsics are the image's scaled by depth size /
  image size, so upsampling to the image size lines it up with the image.

Pixel centres are at (u + 0.5, v + 0.5) throughout, as in gsplat: scaling an image by s scales K by s.
"""

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

GL_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL <-> OpenCV camera axes (its own inverse)
DEPTH_SCALE = 0.001  # transforms.json `depth_unit_scale_factor`: uint16 millimetres
CACHE_FRACTION = 0.4  # cache images on the device when they need less than this share of its free memory


@dataclass
class Frame:
    stem: str
    image_path: Path
    K: np.ndarray  # 3x3 at w x h
    w: int
    h: int
    c2w: np.ndarray  # 4x4 camera-to-world, OpenCV axes, scene frame (metres, z up)
    camera_idx: int  # 0 video, 1 still
    session: str
    depth_path: Path | None
    confidence_path: Path | None
    prior_path: Path | None
    depth_scale: float = DEPTH_SCALE


def frame_from_entry(e: dict, data_dir: Path, depth_scale: float = DEPTH_SCALE) -> Frame:
    """A transforms.json / split.json frame entry -> Frame."""
    stem = Path(e["file_path"]).stem
    k = np.array([[e["fl_x"], 0.0, e["cx"]], [0.0, e["fl_y"], e["cy"]], [0.0, 0.0, 1.0]])
    c2w = np.asarray(e["transform_matrix"], float) @ GL_CV
    depth = e.get("depth_file_path")
    conf = e.get("confidence_file_path")
    prior = data_dir / "prior" / f"{stem}.npz"
    return Frame(
        stem=stem,
        image_path=data_dir / e["file_path"],
        K=k,
        w=int(e["w"]),
        h=int(e["h"]),
        c2w=c2w,
        camera_idx=1 if e.get("camera") == "still" else 0,
        session=e.get("session") or stem.rsplit("_", 1)[0],
        depth_path=data_dir / depth if depth else None,
        confidence_path=data_dir / conf if conf else None,
        prior_path=prior if prior.is_file() else None,
        depth_scale=depth_scale,
    )


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(xyz (M, 3) float32, rgb (M, 3) uint8) from a binary little-endian PLY's vertex element.

    Enough for `scanmodel splat`'s points.ply (float x/y/z + uchar red/green/blue) and anything with extra
    scalar properties; grey when there are no colours.
    """
    types = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4"}
    types |= {"float": "<f4", "double": "<f8", "float32": "<f4", "float64": "<f8", "uint8": "u1"}
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        n, fields, element = 0, [], ""
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[:1] == ["format"] and parts[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY is supported, not {parts[1]}")
            if parts[:1] == ["element"]:
                element = parts[1]
                if element == "vertex":
                    n = int(parts[2])
            elif parts[:1] == ["property"] and element == "vertex":
                fields.append((parts[2], types[parts[1]]))
        rec = np.frombuffer(f.read(n * np.dtype(fields).itemsize), dtype=fields, count=n)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    names = rec.dtype.names or ()
    if "red" in names:
        rgb = np.stack([rec["red"], rec["green"], rec["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 128, np.uint8)
    return xyz, rgb


@dataclass
class Scene:
    frames: list[Frame]
    train: list[int]
    interp: list[int]
    extrap: list[int]
    points: np.ndarray  # (M, 3) float32
    colors: np.ndarray  # (M, 3) uint8
    scene_scale: float
    data_dir: Path

    @classmethod
    def load(cls, data_dir: Path) -> "Scene":
        data_dir = Path(data_dir)
        tf = json.loads((data_dir / "transforms.json").read_text())
        depth_scale = float(tf.get("depth_unit_scale_factor", DEPTH_SCALE))
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_unit_scale_factor must be finite and positive")
        frames = [frame_from_entry(e, data_dir, depth_scale) for e in tf["frames"]]
        if not frames:
            raise ValueError(f"{data_dir}: transforms.json has no frames")
        split_path = data_dir / tf.get("split_file_path", "split.json")
        split = json.loads(split_path.read_text()) if split_path.is_file() else {}
        # Without split.json (dataset v1): every 8th held out, msplat's and Brush's own rule.
        interp_stems = set(split.get("interp", [f.stem for f in frames[::8]]))
        interp = [i for i, f in enumerate(frames) if f.stem in interp_stems]
        train = [i for i, f in enumerate(frames) if f.stem not in interp_stems]
        extra = [frame_from_entry(e, data_dir, depth_scale) for e in split.get("extrap_frames", [])]
        extrap = list(range(len(frames), len(frames) + len(extra)))
        # gsplat's scene scale (its learning rates and densification thresholds are relative to it) from
        # the cameras it trains and validates on, not from the extrap lap.
        centres = np.array([f.c2w[:3, 3] for f in frames])
        scale = 1.1 * float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
        points, colors = read_ply_points(data_dir / tf.get("ply_file_path", "points.ply"))
        # A stationary capture (e.g. a turntable) has no camera radius. Use its geometry so the
        # means learning rate and densification thresholds do not collapse to zero.
        if scale <= 1e-6:
            if len(points):
                scale = 1.1 * float(np.linalg.norm(points - points.mean(0), axis=1).max())
            scale = max(scale, 1e-6)
        return cls(frames + extra, train, interp, extrap, points, colors, scale, data_dir)


def scaled_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """(W, H) of a w x h image shrunk to fit `max_side` (never enlarged), as `scanmodel splat` resizes."""
    s = min(1.0, max_side / max(w, h))
    return (round(w * s), round(h * s)) if s < 1.0 else (w, h)


def scaled_k(k: np.ndarray, w: int, h: int, new_w: int, new_h: int) -> np.ndarray:
    """K at w x h -> K at new_w x new_h (pixel centres at +0.5, so a plain scale)."""
    return np.diag([new_w / w, new_h / h, 1.0]) @ k


def free_memory(device: torch.device) -> int:
    """Bytes free on `device` (on the CPU: available RAM, else half of physical RAM)."""
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    if device.type == "mps":
        return torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):  # macOS has no SC_AVPHYS_PAGES
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2


def read_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 (H, W, 3) at `size` (W, H), INTER_AREA when shrinking."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"can't read image {path}")
    if (bgr.shape[1], bgr.shape[0]) != size:
        shrink = size[0] < bgr.shape[1]
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Images:
    """The images of `scene.frames[i]` for i in `indices`, at max side `max_side`.

    `get(i)` and `size(i)` take a frame index (an index into `scene.frames`, as in `scene.train`), not a
    position in `indices`. `cache=None` keeps uint8 copies on `device` when they take less than 40 % of
    its free memory; otherwise every `get` decodes the JPEG again.
    """

    def __init__(
        self,
        scene: Scene,
        indices: Sequence[int],
        max_side: int,
        device: torch.device | str = "cpu",
        cache: bool | None = None,
    ) -> None:
        self.scene = scene
        self.indices = list(indices)
        self.device = torch.device(device)
        self.sizes: dict[int, tuple[int, int]] = {}
        self.ks: dict[int, torch.Tensor] = {}
        for i in self.indices:
            f = scene.frames[i]
            size = scaled_size(f.w, f.h, max_side)
            self.sizes[i] = size
            self.ks[i] = torch.tensor(scaled_k(f.K, f.w, f.h, *size), dtype=torch.float32, device=self.device)
        nbytes = sum(w * h * 3 for w, h in self.sizes.values())
        if cache is None:
            cache = nbytes < CACHE_FRACTION * free_memory(self.device)
        self.cached: dict[int, torch.Tensor] = {}
        if cache:
            with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as pool:  # cv2 releases the GIL
                for i, img in zip(self.indices, pool.map(self._decode, self.indices), strict=True):
                    self.cached[i] = torch.from_numpy(img).to(self.device)

    def _decode(self, i: int) -> np.ndarray:
        return read_image(self.scene.frames[i].image_path, self.sizes[i])

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_cached(self) -> bool:
        return bool(self.cached)

    def size(self, i: int) -> tuple[int, int]:
        """(W, H) of frame i's image."""
        return self.sizes[i]

    def get(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(image float32 (H, W, 3) in [0, 1] on the device, K (3, 3) float32 at W x H)."""
        img = self.cached.get(i)
        if img is None:
            img = torch.from_numpy(self._decode(i)).to(self.device, non_blocking=True)
        return img.float() / 255.0, self.ks[i]


def _resize(x: torch.Tensor, w: int, h: int, mode: str) -> torch.Tensor:
    """(h0, w0) or (h0, w0, c) -> (h, w) or (h, w, c), float."""
    chw = x.float()[None, None] if x.dim() == 2 else x.float().permute(2, 0, 1)[None]
    if chw.shape[-2:] != (h, w):
        kw = {"align_corners": False} if mode == "bilinear" else {}
        chw = F.interpolate(chw, size=(h, w), mode=mode, **kw)
    return chw[0, 0] if x.dim() == 2 else chw[0].permute(1, 2, 0)


def load_depth(
    frame: Frame, W: int, H: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(depth (H, W) float32 metres, confidence (H, W) uint8) on `device`, or None without LiDAR depth.

    Nearest upsampling (pixel-centre aligned); 0 where the sensor had no reading. Nothing is masked by
    confidence here: the losses pick their own threshold. Without a confidence file (dataset v1, whose
    depth was masked at export) every reading counts as confidence 2.
    """
    if frame.depth_path is None:
        return None
    raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"can't read depth {frame.depth_path}")
    d = torch.from_numpy(raw.astype(np.float32) * frame.depth_scale).to(device)
    if frame.confidence_path is not None:
        conf_np = cv2.imread(str(frame.confidence_path), cv2.IMREAD_UNCHANGED)
        if conf_np is None:
            raise FileNotFoundError(f"can't read confidence {frame.confidence_path}")
        conf = torch.from_numpy(conf_np).to(device)
    else:
        conf = (d > 0).to(torch.uint8) * 2
    depth = _resize(d, W, H, "nearest-exact")
    return depth, _resize(conf, W, H, "nearest-exact").round().to(torch.uint8)


def load_prior(
    frame: Frame, W: int, H: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(depth (H, W) float32 metres, normal (H, W, 3) float32 world) from `gs_trainer.priors`, or None.

    Depth nearest (0 = the mesh wasn't hit), normals bilinear and renormalised (0 where there's none).
    """
    if frame.prior_path is None or not frame.prior_path.is_file():
        return None
    z = np.load(frame.prior_path)
    d = torch.from_numpy(z["depth"].astype(np.float32)).to(device)
    n = torch.from_numpy(z["normal"].astype(np.float32) / 127.0).to(device)
    depth = _resize(d, W, H, "nearest-exact")
    normal = _resize(n, W, H, "bilinear")
    length = normal.norm(dim=-1, keepdim=True)
    normal = torch.where(length > 1e-3, normal / length.clamp_min(1e-3), torch.zeros_like(normal))
    return depth, normal


def viewmat(c2w: np.ndarray | torch.Tensor) -> torch.Tensor:
    """World-to-camera (4, 4) float32 (OpenCV axes) of a rigid camera-to-world.

    A tensor stays on its device (and in the autograd graph); an array comes back as a CPU tensor.
    """
    m = c2w if isinstance(c2w, torch.Tensor) else torch.from_numpy(np.asarray(c2w, np.float64))
    r, t = m[..., :3, :3], m[..., :3, 3:]
    rt = r.transpose(-1, -2)
    top = torch.cat([rt, -rt @ t], -1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([top, bottom], -2).float()
