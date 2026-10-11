"""A tiny synthetic dataset v2 (see trainer/README.md, "Dataset v2"), for the tests.

A box room 4 x 3 x 2.5 m (scene frame: metres, z up, floor at z = 0) with smooth, colourful procedural
textures on every face, seen from a ring of cameras inside it looking outwards. Everything is rendered
analytically: a ray from the camera centre through each pixel centre leaves the box through exactly one
face, so the hit point, its z-depth and its colour are closed-form. No rasteriser, no mesh library, and
it takes milliseconds, so every test can build its own copy.

What it exercises, on purpose:
- video frames with LiDAR depth at a lower native size than the image (half), uint16 mm, and an ARKit-like
  confidence map that is mostly 2 with a few 0/1 pixels and a few holes (depth 0, "no reading");
- stills without depth, at a different size and focal length than the video frames, so code that assumes
  one camera or one image size breaks here;
- principal points off the image centre, so a half-pixel or cx/cy slip shows up;
- OpenGL camera axes in transforms.json (as `scanmodel splat` writes them), OpenCV axes in views.npz;
- a split with every 8th frame held out (interp) and a closing "-lap" session at a different height whose
  frames live only in split.json (extrap).
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOM_LO = np.array([-2.0, -1.5, 0.0])
ROOM_HI = np.array([2.0, 1.5, 2.5])
GL_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL <-> OpenCV camera axes (its own inverse)
SESSION = "synth-ring"
LAP_SESSION = "synth-lap"
FRAME_STEP = 15  # video frame numbers in the stems, as if sampled every 15th frame


class Texture:
    """Smooth colour per face: a base colour plus a few low-frequency sine patterns (0.4-1.6 cycles/m).

    Smooth enough that a few hundred splats fit it, varied enough that a pose or projection error shows
    up in the photometric loss. Faces: 0/1 = x min/max, 2/3 = y min/max, 4 = floor, 5 = ceiling.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.base = rng.uniform(0.25, 0.75, (6, 3))
        self.freq = rng.uniform(0.4, 1.6, (6, 3, 2, 2))  # face, channel, pattern, axis
        self.phase = rng.uniform(0.0, 1.0, (6, 3, 2))
        self.amp = rng.uniform(0.1, 0.2, (6, 3, 2))

    def __call__(self, face: np.ndarray, p: np.ndarray) -> np.ndarray:
        """RGB in [0, 1] (n, 3) at points `p` (n, 3) on faces `face` (n,)."""
        axis = face // 2
        # The two in-plane coordinates of each point.
        a = np.where(axis == 0, p[:, 1], p[:, 0])
        b = np.where(axis == 2, p[:, 1], p[:, 2])
        ab = np.stack([a, b], -1)[:, None, None, :]  # n, 1, 1, 2
        arg = 2 * np.pi * ((self.freq[face] * ab).sum(-1) + self.phase[face])  # n, 3, 2
        rgb = self.base[face] + (self.amp[face] * np.sin(arg)).sum(-1)
        return np.clip(rgb, 0.03, 0.97)


def ray_box(origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exit of rays from inside the room: (t (n,), face (n,), point (n, 3)), point = origin + t * dir."""
    with np.errstate(divide="ignore"):
        bound = np.where(dirs > 0, ROOM_HI, ROOM_LO)
        t_axis = np.where(dirs != 0, (bound - origin) / dirs, np.inf)
    axis = t_axis.argmin(1)
    t = t_axis[np.arange(len(dirs)), axis]
    face = axis * 2 + (dirs[np.arange(len(dirs)), axis] > 0)
    return t, face, origin + t[:, None] * dirs


def cast(
    c2w_cv: np.ndarray, k: np.ndarray, w: int, h: int, texture: Texture, s: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """(rgb (h*s, w*s, 3), z-depth (h*s, w*s)) of rays through the centres of an s x s grid per pixel."""
    u, v = np.meshgrid((np.arange(w * s) + 0.5) / s, (np.arange(h * s) + 0.5) / s)
    cam = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    # Camera-frame z of each direction is 1, so the ray parameter t is the z-depth.
    t, face, p = ray_box(c2w_cv[:3, 3], cam @ c2w_cv[:3, :3].T)
    return texture(face, p).reshape(h * s, w * s, 3), t.reshape(h * s, w * s)


def render(
    c2w_cv: np.ndarray, k: np.ndarray, w: int, h: int, texture: Texture, supersample: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """(rgb (h, w, 3) float in [0, 1], z-depth (h, w) metres); `k` is 3x3 at w x h.

    Pixel centres are at (u + 0.5, v + 0.5), as in gsplat and `freespace.votes`. Colours are averaged over
    `supersample`^2 sub-pixel rays (antialiasing); the depth is taken at the pixel centres only, as a
    depth sensor samples it.
    """
    rgb, z = cast(c2w_cv, k, w, h, texture)
    if supersample > 1:
        s = supersample
        rgb = cast(c2w_cv, k, w, h, texture, s)[0].reshape(h, s, w, s, 3).mean((1, 3))
    return rgb, z


def look_out(centre: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Camera-to-world, OpenCV axes (x right, y down, z forward), looking along `yaw` (radians, from +x)."""
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c, s = np.cos(roll), np.sin(roll)
    right, down = c * right + s * down, -s * right + c * down
    m = np.eye(4)
    m[:3, :3] = np.stack([right, down, fwd], 1)
    m[:3, 3] = centre
    return m


def intrinsics(w: int, h: int, hfov_deg: float, offset: tuple[float, float]) -> np.ndarray:
    f = w / 2 / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2 + offset[0]], [0, f, h / 2 + offset[1]], [0, 0, 1]])


def write_ply_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """float32 x/y/z + uchar red/green/blue, the layout `scanmodel splat` writes."""
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    path.write_bytes(header.encode() + rec.tobytes())


def write_ply_mesh(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """float32 vertices, int32 triangles, nothing else (dataset v2's mesh.ply)."""
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    faces = np.empty(len(triangles), dtype=[("n", "u1"), ("i", "<i4", 3)])
    faces["n"], faces["i"] = 3, triangles
    path.write_bytes(header.encode() + vertices.astype("<f4").tobytes() + faces.tobytes())


def box_mesh() -> tuple[np.ndarray, np.ndarray]:
    """The room's 8 corners and 12 triangles (winding facing inwards; the priors flip normals anyway)."""
    v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    v = ROOM_LO + v * (ROOM_HI - ROOM_LO)
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return v, np.array(tris, np.int32)


def sample_walls(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` points uniformly over the room's surface (by area) and the face each lies on."""
    size = ROOM_HI - ROOM_LO
    areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]]).repeat(2)
    face = rng.choice(6, n, p=areas / areas.sum())
    p = ROOM_LO + rng.uniform(0, 1, (n, 3)) * size
    axis = face // 2
    p[np.arange(n), axis] = np.where(face % 2 == 1, ROOM_HI[axis], ROOM_LO[axis])
    return p, face


def write_dataset(
    path: Path, n_frames: int = 16, size: tuple[int, int] = (64, 48), extrap: int = 4, seed: int = 0
) -> Path:
    """Write the dataset into `path` (created; existing files overwritten) and return `path`.

    `n_frames` frames go into transforms.json (every 8th position on the ring is a still without depth,
    the rest video frames), plus `extrap` frames of a "-lap" session at a lower height in split.json only.
    `size` is the video frames' (w, h); depth is at half that, stills at 1.5x it.
    """
    rng = np.random.default_rng(seed)
    texture = Texture(rng)
    path.mkdir(parents=True, exist_ok=True)
    for d in ("images", "depth", "confidence"):
        (path / d).mkdir(exist_ok=True)
    w, h = size
    dw, dh = max(1, w // 2), max(1, h // 2)
    sw, sh = round(w * 1.5), round(h * 1.5)
    k_video = intrinsics(w, h, 70.0, (0.7 * w / 64, -0.4 * h / 48))
    k_still = intrinsics(sw, sh, 62.0, (-0.9 * sw / 96, 0.6 * sh / 72))

    views: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # depth (dh, dw), k (4,), w2c

    def frame(session: str, i: int, still: bool, centre: np.ndarray, yaw: float) -> dict[str, object]:
        c2w = look_out(centre, yaw, rng.uniform(-0.15, 0.15), rng.uniform(-0.05, 0.05))
        fw, fh, k = (sw, sh, k_still) if still else (w, h, k_video)
        stem = f"{session}_s{i:02d}" if still else f"{session}_{i * FRAME_STEP:06d}"
        rgb, _ = render(c2w, k, fw, fh, texture, supersample=2)
        bgr = (rgb[..., ::-1] * 255).round().astype(np.uint8)
        cv2.imwrite(str(path / "images" / f"{stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        e: dict[str, object] = {
            "file_path": f"images/{stem}.jpg",
            "transform_matrix": (c2w @ GL_CV).tolist(),
            "fl_x": k[0, 0],
            "fl_y": k[1, 1],
            "cx": k[0, 2],
            "cy": k[1, 2],
            "w": fw,
            "h": fh,
            "camera": "still" if still else "video",
            "session": session,
            "t": i * 0.5,
        }
        if not still:
            kd = k * np.array([[dw / w], [dh / h], [1.0]])  # ARKit's depth covers the image's field of view
            _, z = render(c2w, kd, dw, dh, texture)
            conf = np.full((dh, dw), 2, np.uint8)
            r = rng.uniform(size=(dh, dw))
            conf[r < 0.08] = 1
            conf[r < 0.03] = 0
            mm = np.clip(z * 1000, 0, 65535).round().astype(np.uint16)
            mm[rng.uniform(size=(dh, dw)) < 0.02] = 0  # no reading
            cv2.imwrite(str(path / "depth" / f"{stem}.png"), mm)
            cv2.imwrite(str(path / "confidence" / f"{stem}.png"), conf)
            e["depth_file_path"] = f"depth/{stem}.png"
            e["confidence_file_path"] = f"confidence/{stem}.png"
            d = np.where(conf >= 2, mm / 1000, 0).astype(np.float32)
            views.append((d, kd[[0, 1, 0, 1], [0, 1, 2, 2]], np.linalg.inv(c2w)))
        return e

    step = 2 * np.pi / max(n_frames, 1)
    frames = [
        frame(
            SESSION,
            i,
            i % 8 == 5,
            np.array([0.5 * np.cos(i * step), 0.4 * np.sin(i * step), 1.4 + rng.uniform(-0.05, 0.05)]),
            i * step + rng.uniform(-0.1, 0.1),
        )
        for i in range(n_frames)
    ]
    frames.sort(key=lambda f: str(f["file_path"]))
    lap_step = 2 * np.pi / max(extrap, 1)
    lap = [
        frame(
            LAP_SESSION,
            i,
            False,
            np.array([0.7 * np.cos((i + 0.5) * lap_step), 0.5 * np.sin((i + 0.5) * lap_step), 1.0]),
            (i + 0.5) * lap_step,
        )
        for i in range(extrap)
    ]
    lap.sort(key=lambda f: str(f["file_path"]))

    def stem(f: dict[str, object]) -> str:
        return Path(str(f["file_path"])).stem

    split = {
        "interp": [stem(f) for i, f in enumerate(frames) if i % 8 == 0],
        "extrap": [stem(f) for f in lap],
        "extrap_session": LAP_SESSION if lap else None,
        "extrap_frames": lap,
    }
    (path / "split.json").write_text(json.dumps(split, indent=1) + "\n")
    transforms = {
        "version": 2,
        "camera_model": "OPENCV",
        "depth_unit_scale_factor": 0.001,
        "ply_file_path": "points.ply",
        "mesh_file_path": "mesh.ply",
        "split_file_path": "split.json",
        "frames": frames,
    }
    (path / "transforms.json").write_text(json.dumps(transforms, indent=1) + "\n")

    p, face = sample_walls(rng, 3000)
    write_ply_points(path / "points.ply", p, (texture(face, p) * 255).round().astype(np.uint8))
    write_ply_mesh(path / "mesh.ply", *box_mesh())
    if views:
        depth, k, w2c = (np.stack(x) for x in zip(*views, strict=True))
        np.savez(path / "views.npz", depth=depth, k=k.astype(np.float64), w2c=w2c)
    return path
