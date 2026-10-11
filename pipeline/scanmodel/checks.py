"""Data checks shared by stages."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .geom import apply, backproject, project
from .io import Bundle
from .models.bundle import Frame, TrackingState
from .models.common import F64, mat


def depth_consistency(
    bundle: Bundle,
    c2w: Callable[[Frame], F64],
    pairs: int = 24,
    baseline: tuple[float, float] = (0.08, 0.6),
    min_confidence: int = 1,
) -> float | None:
    """Median |depth disagreement| (m) when one frame's depth is reprojected into a nearby frame.

    With correct poses and intrinsics this is a few millimetres to ~1 cm for LiDAR. It is used to
    check bundle integrity and to detect the camera-axis convention of imported data.
    """
    frames = [f for f in bundle.frames if f.depth is not None]
    if len(frames) < 2:
        return None
    centres = np.array([c2w(f)[:3, 3] for f in frames])
    residuals: list[F64] = []
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(frames))[: pairs * 4]:
        dist = np.linalg.norm(centres - centres[i], axis=1)
        cand = np.flatnonzero((dist > baseline[0]) & (dist < baseline[1]))
        if cand.size == 0:
            continue
        j = int(cand[np.argmin(np.abs(cand - i))])
        fa, fb = frames[i], frames[j]
        da, db = bundle.depth_m(fa, min_confidence), bundle.depth_m(fb, min_confidence)
        if da is None or db is None:
            continue
        pts_world = apply(c2w(fa), backproject(da, fa.intrinsics, stride=4))
        pts_b = apply(np.linalg.inv(c2w(fb)), pts_world)
        h, w = db.shape
        uv, z = project(pts_b, fb.intrinsics, w, h)
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        zb = db[v[ok], u[ok]]
        valid = zb > 0
        if valid.sum() > 50:
            residuals.append(np.abs(zb[valid] - z[ok][valid]))
        if len(residuals) >= pairs:
            break
    return float(np.median(np.concatenate(residuals))) if residuals else None


SETTLE_S = 1.0  # tracking must be clean this long before a session's frames are trusted
MAX_SPEED_M_S = 3.0  # far beyond a hand-held iPad (scans: p99 0.7 m/s): ARKit moved its world (as pose_jumps)


def settled_from(frames: list[Frame], fps: float = 60.0) -> int:
    """Index into `frames` of the first frame of a session's settled world, when the session starts
    unsettled: tracking not normal, or a timestamp gap (> 3 frame intervals), before the first SETTLE_S
    of clean tracking. From that first clean stretch on, also past steps faster than MAX_SPEED_M_S (ARKit's
    snap into the relocalised world and its settling).

    Scan 4: four frames tracked "normal", an 11 s stall, re-initialisation, five more "normal" frames, then
    a 31 cm jump and 10 cm of settling; the frames before it are in a different world than the rest. 0
    when the session starts tracked (a fast step alone is not enough: sparse imports and synthetic data
    move that fast), or never settles (the other checks say what is wrong). Only the start is handled:
    later corrections are drift, which the pose graph corrects."""
    if len(frames) < 2:
        return 0
    t = np.array([f.t for f in frames])
    pos = np.array([mat(f.camera_to_world)[:3, 3] for f in frames])
    normal = np.array([f.tracking in (None, TrackingState.NORMAL) for f in frames])
    dt = np.diff(t)
    gap = dt > 3.0 / fps  # frame i -> i + 1
    fast = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(dt, 1e-3) > MAX_SPEED_M_S

    def first_clean(start: int, bad_step: np.ndarray) -> int | None:
        for k in range(start, len(frames)):
            end = int(np.searchsorted(t, t[k] + SETTLE_S))
            if end >= len(frames):
                return None
            if normal[k : end + 1].all() and not bad_step[k:end].any():
                return k
        return None

    tracked = first_clean(0, gap)
    if not tracked:
        return 0
    settled = first_clean(tracked, gap | fast)
    return tracked if settled is None else settled


EXCURSION_RATIO = 8.0  # a jump's speed vs the median speed of the second either side of it
EXCURSION_SPIN_DEG_S = 300.0  # rotation-only jumps (brisk hand-held turns reach ~155°/s, 2.6° a frame)
EXCURSION_MIN_M, EXCURSION_MIN_DEG = 0.1, 5.0  # smaller jumps aren't paired: the tolerance would cover them
EXCURSION_TOL_M, EXCURSION_TOL_DEG, EXCURSION_TOL_REL = 0.1, 2.0, 0.25  # |out + back| < max(TOL, REL |out|)
EXCURSION_RETURN_S = 5.0  # the jump back comes within this
EXCURSION_MAX_S = 3.0  # longer and it isn't dropped, only warned about


@dataclass(frozen=True)
class Excursion:
    """Frames `first`..`last` (indices into the frames, inclusive, padding included) of an excursion."""

    first: int
    last: int
    kind: str  # "move" (translation) or "turn" (rotation only)
    out_m: float  # the jump out: its net translation, m
    out_deg: float  # and rotation
    dropped: bool  # False: longer than EXCURSION_MAX_S, kept whole (the warning says so)


def _jumps(t: F64, step: F64, min_rate: float, merge: int) -> list[tuple[int, int]]:
    """(first, last) step index of each jump (step i: frame i -> i + 1): steps faster than `min_rate`
    and EXCURSION_RATIO x the median speed within 1 s either side (centred, so the start of a brisk turn
    isn't a jump; speeds, so a dropped frame isn't one), merged when `merge` steps or fewer apart."""
    rate = step / np.maximum(np.diff(t), 1e-3)
    out: list[tuple[int, int]] = []
    for i in np.flatnonzero(rate > min_rate):
        k, e = np.searchsorted(t[:-1], [t[i] - 1.0, t[i] + 1.0])
        around = np.concatenate([rate[k:i], rate[i + 1 : e]])
        if not len(around) or rate[i] <= EXCURSION_RATIO * float(np.median(around)):
            continue
        if out and i - out[-1][1] <= merge:
            out[-1] = (out[-1][0], int(i))
        else:
            out.append((int(i), int(i)))
    return out


def excursions(frames: list[Frame]) -> list[Excursion]:
    """Where ARKit's pose jumped away and, within EXCURSION_RETURN_S, jumped back by about as much, with
    tracking "normal" throughout: the frames in between are in the wrong place. Scan 5 (living): 42/13/4
    cm over three frames, ~60 cm off for 1 s, 60 cm back. The pose graph and refine can't take that out
    (ARKit's relative motion is trusted between neighbours), so these frames are left out like an unsettled
    start. A jump: faster than MAX_SPEED_M_S (or, rotation only, turning faster than EXCURSION_SPIN_DEG_S)
    and EXCURSION_RATIO x the median speed around it; jumps of at least EXCURSION_MIN_M (_DEG) pair. A jump
    without a return is a one-way shift (drift, for the pose graph) and stays. Skipped for sparse imports
    (median frame interval > 0.1 s)."""
    if len(frames) < 3:
        return []
    t = np.array([f.t for f in frames])
    dt = float(np.median(np.diff(t)))
    if dt > 0.1:
        return []
    m = np.array([mat(f.camera_to_world) for f in frames])
    pos, rot = m[:, :3, 3], m[:, :3, :3]
    scale = 1 / (60.0 * dt)  # the frame counts below are for 60 Hz
    merge, pad = max(1, round(3 * scale)), max(1, round(2 * scale))

    def deg(r: F64) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))

    # Per kind: step sizes, the speed of a jump, its net change over steps a..c, size, out + back, floors.
    def move(a: int, c: int) -> F64:
        return pos[c + 1] - pos[a]

    def turn(a: int, c: int) -> F64:
        return rot[c + 1] @ rot[a].T  # world frame: an offset of ARKit's world cancels with its undoing

    kinds = (
        (
            "move",
            np.linalg.norm(np.diff(pos, axis=0), axis=1),
            MAX_SPEED_M_S,
            move,
            lambda x: float(np.linalg.norm(x)),
            lambda out, back: out + back,
            EXCURSION_MIN_M,
            EXCURSION_TOL_M,
        ),
        (
            "turn",
            np.array([deg(r1 @ r0.T) for r0, r1 in itertools.pairwise(rot)]),
            EXCURSION_SPIN_DEG_S,
            turn,
            deg,
            lambda out, back: back @ out,
            EXCURSION_MIN_DEG,
            EXCURSION_TOL_DEG,
        ),
    )
    found: list[Excursion] = []
    taken = np.zeros(len(frames), dtype=bool)
    for kind, step, min_rate, net, size, combine, min_size, tol in kinds:
        jumps = _jumps(t, step, min_rate, merge)
        n = 0
        while n < len(jumps):
            a, c = jumps[n]
            out = net(a, c)
            back, best = None, max(tol, EXCURSION_TOL_REL * size(out))  # the closest return
            if size(out) >= min_size:
                for k in range(n + 1, len(jumps)):
                    if t[jumps[k][0]] - t[c + 1] > EXCURSION_RETURN_S:
                        break
                    ret = net(*jumps[k])
                    if size(ret) >= min_size and (miss := size(combine(out, ret))) < best:
                        back, best = k, miss
            if back is None:
                n += 1
                continue
            # Frames a + 1 .. c' are off (frame a is the last before the jump, c' + 1 the first back).
            first, last = max(0, a + 1 - pad), min(len(frames) - 1, jumps[back][1] + pad)
            if not taken[first : last + 1].any():  # a turn inside a move excursion is that excursion
                taken[first : last + 1] = True
                ok = float(t[last] - t[first]) <= EXCURSION_MAX_S
                found.append(
                    Excursion(first, last, kind, float(np.linalg.norm(move(a, c))), deg(turn(a, c)), ok)
                )
            n = back + 1
    return sorted(found, key=lambda e: e.first)


def pose_jumps(frames: list[Frame], max_step_m: float = 0.25) -> list[int]:
    """Frame idx values where the camera moved implausibly far since the previous frame."""
    out: list[int] = []
    for a, b in itertools.pairwise(frames):
        dt = max(b.t - a.t, 1e-3)
        step = float(np.linalg.norm(mat(b.camera_to_world)[:3, 3] - mat(a.camera_to_world)[:3, 3]))
        if step > max_step_m and step / dt > 3.0:
            out.append(b.idx)
    return out
