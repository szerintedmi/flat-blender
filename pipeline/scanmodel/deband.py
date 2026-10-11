"""LED flicker banding: find it and divide it out of video frames and stills.

Mains-powered LED lights flicker at twice the mains frequency (100 Hz on 50 Hz mains). With a short shutter
(1/240 s) and a rolling-shutter sensor every row integrates a different part of the flicker, so a frame
carries horizontal bright/dark bands: one period every 1 / (100 Hz x row time) rows (982 of 1440 on the M2
iPad Pro, whose 1920x1440 ARKit format reads out in 14.7 ms), moving by 2 pi x 100 Hz x 1/60 s = 240° every
frame. Scan 4 (1/240 s, evening, LED lights): ±18 % median in a single frame, up to ±45 % near a strongly
flickering lamp.

The bands are not just a function of the row: a room is lit by several lamps that flicker by different
amounts (scan 4: the conservatory's lamp ~70 %, the living room's 20-30 %, the hallway's < 5 %), so the
gain is `exp(Re(W(x, y) e^(2 pi i y / P)))` with a complex modulation W that is smooth over the image
but differs between lamps. What makes it measurable: three consecutive frames see each scene point at
three flicker phases ~120° apart, which cancel. Per pixel (after warping the neighbours into the frame by
their rotation), three samples `u_k = s + Re(Z e^(i D_k))` with the known phase offsets D_k (frame time
offset x 100 Hz plus the row offset from the warp) give the scene term s and the flicker Z; Z is then
demodulated by the row phase, smoothed over flat, well-exposed pixels (texture, edges and misregistration
left out; holes filled from wider), and remodulated. Harmonics 1, 2, 4, 5 of the flicker are captured;
the 3rd (300 Hz, ~1° drift per frame at 60 fps, so invisible to any frame-to-frame comparison) is not:
about 0.5-1 % left in scan 4's ceiling frames.

The gain lives in `u = log(V + TRANSFER_OFFSET)` (V the 8-bit value / 255): Apple's video tone curve is
flatter than sRGB's, and this effective curve leaves the least banding on textured areas of scan 4 (sRGB
leaves 45 % more there). RGB share the same gain within ±5 % (B 1.04, R 0.96 relative to G), so each
channel is corrected with its own field.

Stills (ARKit high-res frames) are taken with the nearest stream frame (same timestamp, same readout:
2062 bands-period rows of 3024 = 982 of 1440 scaled), so they get that frame's gain mapped through the
still-to-frame homography, scaled by STILL_SCALE (the photo pipeline's tone curve shows the bands 4 %
weaker).
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import rel
from .models.bundle import Intrinsics
from .models.common import F64

MAINS_HZ = 50.0  # the lights flicker at twice this
READOUT_S = 14.66e-3  # top-to-bottom readout of a video frame (M2 iPad Pro, 1920x1440 ARKit format)
TRANSFER_OFFSET = 0.5  # the gain is applied to (V + this), V in [0, 1]: Apple's video tone curve
WORK_SIDE = 480  # the flicker field is measured at this long side
SMOOTH_PX = 5.0  # Gaussian sigma of the field at WORK_SIDE (~20 px at 1920): keeps lamp boundaries
FILL_PX = 30.0  # wider sigma that fills textured / clipped / unwarped holes
MIN_SUPPORT = 0.3  # local weight below which the wide fill takes over
SMOOTH_GRAD = 0.023  # max gradient of u per working pixel (Sobel / 8) for a pixel to measure the flicker
EXPOSED = (3 / 255, 225 / 255)  # all channels of all three samples within this to measure
MAX_FLICKER = 0.46  # |Z| in u beyond this is misregistration, not flicker
CLIP = (235, 252)  # pixels whose brightest channel is above CLIP[0] are faded to uncorrected by CLIP[1]
MIN_SPREAD_DEG = 60.0  # the three samples' phases must be at least this far apart
STILL_SCALE = 0.96
AUTO_THRESHOLD = 0.02  # `detect` amplitude above which `auto` debands a session
DETECT_SAMPLES = 16
DEBAND_MODES = ("auto", "on", "off")


@dataclass
class View:
    """A video frame or still: BGR image, time (s), camera-to-world (OpenCV axes) and intrinsics."""

    img: np.ndarray
    t: float
    c2w: F64
    k: Intrinsics


@dataclass
class Field:
    """Demodulated flicker of one frame at working resolution: the frame's log gain in u at full-res row y is
    Re(W e^(2 pi i y / period_rows)) with W bilinearly upsampled."""

    w: np.ndarray  # complex (h, w, 3), working resolution
    period_rows: float  # band period in full-res rows
    scale: float  # working / full resolution
    size: tuple[int, int]  # full resolution (width, height)
    amplitude: float  # median |W| over measured pixels (u units), NaN when nothing was measured


def period_rows(height: int, light_hz: float = 2 * MAINS_HZ, readout_s: float = READOUT_S) -> float:
    """Band period in rows of a `height`-row frame."""
    return height / (readout_s * light_hz)


def to_u(img: np.ndarray) -> np.ndarray:
    return np.log(img.astype(np.float32) / 255 + TRANSFER_OFFSET)


def _k(i: Intrinsics, s: float = 1.0) -> F64:
    return np.array([[i.fx * s, 0.0, i.cx * s], [0.0, i.fy * s, i.cy * s], [0.0, 0.0, 1.0]])


def rotation_homography(src: View, dst: View, s_src: float = 1.0, s_dst: float = 1.0) -> F64:
    """Pixels of `src` (scaled by s_src) to pixels of `dst` (scaled by s_dst), rotation only: the frames
    are 17 ms apart, so parallax is a fraction of a pixel at working resolution."""
    r = dst.c2w[:3, :3].T @ src.c2w[:3, :3]
    return _k(dst.k, s_dst) @ r @ np.linalg.inv(_k(src.k, s_src))


def _blur(v: np.ndarray, m: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution: Gaussian average of `v` (h, w, c) over the pixels in mask `m`, and the weight."""
    wsum = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    num = cv2.GaussianBlur((v * m[..., None]).astype(np.float32), (0, 0), sigma)
    if num.ndim == 2:
        num = num[..., None]
    return num / np.maximum(wsum, 1e-6)[..., None], wsum


def phase_offsets(cur: View, others: Sequence[View], light_hz: float) -> list[float]:
    """Flicker phase of each other frame relative to `cur` (radians, wrapped to (-pi, pi])."""
    return [math.remainder(2 * math.pi * light_hz * (o.t - cur.t), 2 * math.pi) for o in others]


def well_spread(offsets: Sequence[float], min_deg: float = MIN_SPREAD_DEG) -> bool:
    """True when 0 and the offsets are pairwise at least `min_deg` apart on the circle."""
    ph = [0.0, *offsets]
    return all(
        abs(math.degrees(math.remainder(a - b, 2 * math.pi))) >= min_deg
        for i, a in enumerate(ph)
        for b in ph[i + 1 :]
    )


def flicker_field(cur: View, others: Sequence[View], light_hz: float = 2 * MAINS_HZ) -> Field:
    """The flicker field of `cur` from two other frames taken a few frames around it (see the module
    docstring). Their phase offsets must be well spread (`well_spread`)."""
    if len(others) != 2:
        raise ValueError("flicker_field needs exactly two other frames")
    h0, w0 = cur.img.shape[:2]
    s = min(1.0, WORK_SIDE / max(h0, w0))
    w, h = round(w0 * s), round(h0 * s)
    p_work = period_rows(h0, light_hz) * s
    small = [cv2.resize(v.img, (w, h), interpolation=cv2.INTER_AREA) for v in (cur, *others)]
    u0 = to_u(small[0])
    lo, hi = EXPOSED
    v0 = small[0].astype(np.float32) / 255
    ok = (v0.min(2) >= lo) & (v0.max(2) <= hi)
    rows = np.arange(h, dtype=np.float64)[:, None]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([xs, ys, np.ones_like(xs)], -1)
    samples, deltas = [u0], [np.zeros((h, w))]
    for o, im, dphi in zip(others, small[1:], phase_offsets(cur, others, light_hz), strict=True):
        hom = rotation_homography(cur, o, s, s)
        warped = cv2.warpPerspective(
            im.astype(np.float32),
            hom,
            (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=(-1, -1, -1),
        )
        valid = warped.min(2) >= 0
        vw = np.maximum(warped, 0) / 255
        ok &= valid & (vw.min(2) >= lo) & (vw.max(2) <= hi)
        q = pix @ hom.T
        src_row = q[..., 1] / q[..., 2]  # the row of `o` the pixel was read at: its own flicker phase
        deltas.append(dphi + 2 * np.pi * (src_row - rows) / p_work)
        samples.append(np.log(vw + TRANSFER_OFFSET).astype(np.float32))
    d = np.stack(deltas, -1)  # (h, w, 3 samples)
    m = np.stack([np.ones_like(d), np.cos(d), -np.sin(d)], -1)  # u_k = s + a cos D_k - b sin D_k
    det_ok = np.abs(np.linalg.det(m)) > 0.5
    m[~det_ok] = np.eye(3)
    sol = np.linalg.solve(m, np.stack(samples, -2))  # (h, w, 3 params, 3 channels)
    z = sol[:, :, 1] + 1j * sol[:, :, 2]  # (h, w, 3 channels)
    g = u0.mean(2)
    grad = cv2.GaussianBlur(
        np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8,
        (0, 0),
        1.5,
    )
    use = ok & det_ok & (grad < SMOOTH_GRAD) & (np.abs(z).max(-1) < MAX_FLICKER)
    wd = z * np.exp(-2j * np.pi * rows / p_work)[..., None]
    parts = np.concatenate([np.real(wd), np.imag(wd)], -1)
    near, support = _blur(parts, use, SMOOTH_PX)
    wide, _ = _blur(parts, use, FILL_PX)
    a = np.clip(support / MIN_SUPPORT, 0, 1)[..., None]
    sm = a * near + (1 - a) * wide
    field = sm[..., :3] + 1j * sm[..., 3:]
    amp = float(np.median(np.abs(field[use]))) if use.any() else math.nan
    return Field(field.astype(np.complex64), period_rows(h0, light_hz), s, (w0, h0), amp)


def log_gain(field: Field, height: int, width: int) -> np.ndarray:
    """The frame's log gain in u (height, width, 3) at its full resolution."""
    parts = np.concatenate([np.real(field.w), np.imag(field.w)], -1).astype(np.float32)
    up = cv2.resize(parts, (width, height), interpolation=cv2.INTER_LINEAR)
    s = field.scale
    y = ((np.arange(height) + 0.5) * s - 0.5) / s  # full-res rows in the working rows' frame
    ph = np.exp(2j * np.pi * y / field.period_rows)[:, None, None]
    return (up[..., :3] * np.real(ph) - up[..., 3:] * np.imag(ph)).astype(np.float32)


def still_log_gain(field: Field, frame: View, still: View, size: tuple[int, int]) -> np.ndarray:
    """The still's log gain at `size` (w, h): the nearest frame's gain at the frame pixel each still pixel
    shows (rotation homography; the still is a narrower crop of the same readout), x STILL_SCALE. Only the
    views' poses and intrinsics are used, not their images."""
    fw, fh = field.size
    g = log_gain(field, fh, fw)
    w, h = size
    hom = rotation_homography(still, frame, w / still.k.width, fw / frame.k.width)
    out = cv2.warpPerspective(
        g, hom, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )
    return out * STILL_SCALE


def apply(img: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Divide the log gain (u) out of `img` (BGR uint8, same size). Near-clipped pixels are left as they were:
    their true brightness is unknown, and dividing would draw the bands into blown highlights."""
    v = img.astype(np.float32) / 255
    out = (v + TRANSFER_OFFSET) * np.exp(-gain) - TRANSFER_OFFSET
    lo, hi = CLIP
    keep = np.clip((img.max(2).astype(np.float32) - lo) / (hi - lo), 0, 1)[..., None]
    out = out * (1 - keep) + v * keep
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def deband(cur: View, others: Sequence[View], light_hz: float = 2 * MAINS_HZ) -> tuple[np.ndarray, Field]:
    """`cur`'s image with the flicker divided out, and its field."""
    f = flicker_field(cur, others, light_hz)
    h, w = cur.img.shape[:2]
    return apply(cur.img, log_gain(f, h, w)), f


def neighbours(idx: int, times: dict[int, float], light_hz: float = 2 * MAINS_HZ) -> tuple[int, int] | None:
    """Two frames near `idx` (indices into `times`, frame index -> time) whose flicker phases are well
    spread with `idx`'s: the previous and next frame when both exist, else others within ±2 frames."""
    if idx not in times:
        return None
    for a, b in ((-1, 1), (1, 2), (-2, -1), (-1, 2), (-2, 1)):
        i, j = idx + a, idx + b
        if i in times and j in times:
            offs = [2 * math.pi * light_hz * (times[x] - times[idx]) for x in (i, j)]
            if well_spread(offs):
                return i, j
    return None


# --------------------------------------------------------------------------- detection


def _row_profile(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        img, (WORK_SIDE, round(WORK_SIDE * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA
    )
    return np.log(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float64).mean(1) + 1)


def pair_amplitude(a: np.ndarray, b: np.ndarray, period: float, step_rad: float) -> float:
    """Single-frame band amplitude (log of the 8-bit row means) from two consecutive frames' row profiles
    (any length), fitted at `period` (rows of the profile): the bands' difference has amplitude
    2 sin(step / 2) x theirs."""
    y = np.arange(len(a), dtype=np.float64)
    r = b - a
    x = 2 * np.pi * y / period
    cols = np.stack([np.ones_like(y), y / len(y), (y / len(y)) ** 2, np.cos(x), np.sin(x)], 1)
    c, *_ = np.linalg.lstsq(cols, r, rcond=None)
    return float(math.hypot(c[3], c[4]) / max(2 * abs(math.sin(step_rad / 2)), 0.2))


def detect(video: Path, light_hz: float = 2 * MAINS_HZ, samples: int = DETECT_SAMPLES) -> dict[str, object]:
    """Cheap banding check of a video: `samples` runs of three consecutive frames spread over it (seeking, no
    poses), the band amplitude at the expected period from each consecutive pair, median over all.
    Camera motion adds texture to the row profiles, so a band-free video reads ~0.5 %, not 0."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    step = 2 * math.pi * light_hz / fps
    amps: list[float] = []
    height = 0
    for start in np.linspace(0.05 * n, 0.95 * n - 3, samples).astype(int) if n > 10 else [0]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        profs = []
        for _ in range(3):
            ok, img = cap.read()
            if not ok:
                break
            height = img.shape[0]
            profs.append(_row_profile(img))
        if len(profs) < 2:
            continue
        period = period_rows(height, light_hz) * len(profs[0]) / height
        pa = [pair_amplitude(profs[i], profs[i + 1], period, step) for i in range(len(profs) - 1)]
        amps.append(float(np.median(pa)))
    cap.release()
    amp = float(np.median(amps)) if amps else 0.0
    return {
        "amplitude": round(amp, 4),
        "samples": len(amps),
        "p90": round(float(np.percentile(amps, 90)), 4) if amps else 0.0,
        "banded": amp > AUTO_THRESHOLD,
    }


def detect_cached(video: Path, cache: Path, light_hz: float = 2 * MAINS_HZ) -> dict[str, object]:
    """`detect`, cached in `cache` (a .json describing the video, like the sharpness cache)."""
    st = video.stat()
    meta = {
        "video": rel(video), "bytes": st.st_size, "mtime": st.st_mtime, "light_hz": light_hz,
        "samples": DETECT_SAMPLES, "threshold": AUTO_THRESHOLD,
    }  # fmt: skip
    if cache.is_file():
        old = json.loads(cache.read_text())
        if old.get("meta") == meta:
            return old["result"]
    result = detect(video, light_hz)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"meta": meta, "result": result}, indent=1) + "\n")
    return result
