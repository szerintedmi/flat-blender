"""Doubled surfaces: do separate stretches of the capture put the same walls in the same place?

Eval's face metrics can't see a capture split into rigid blocks: each RoomPlan face belongs to one session and
is fitted to "the dominant surface within 3 cm" of it, and meshcheck compares each frame's LiDAR with a fused
mesh that holds both copies. Scan 5 (2026-10-09) went unnoticed that way with its hallway, kitchen and lap
0.4-0.5 m off the rest: the living room's south wall was there twice, 40 cm apart, seen from the same side.

This compares the wall points of every 30 s window (`walls/<session>@<s>.npz`, written by `scanmodel poses` and
`scanmodel refine` at their poses: each window's own LiDAR, normals facing the camera that saw them) with every
other window that saw the same surfaces from the same side: windows of different sessions, or of one session
at least a window apart. Per pair, the horizontal shift that puts most of one window's surfaces onto the
other's (`pair_offset`: each point and each same-facing point of the other window within `RADIUS_M` that
lies on the same surface, `LATERAL_M`, give a line t . n = d; the grid cell with most lines through it,
refined by least squares; along a direction no surface pins, t stays 0). Pairs that disagree (a shift of
`OFFSET_M` or more, with clearly more support at the shift than at zero) are grouped by their shift: a group
backed by several window pairs is a split, the same walls twice `offset_m` apart. A real surface a few cm in
front of another (a radiator, a cabinet front) shows within single pairs and is outvoted at zero; windows
that straddle the moment a capture split agree with both sides, so the windows are not clustered into blocks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d.core as o3c

from .models.common import F64
from .points import WINDOW_S, read_wall_points, to_o3d

VOXEL_M = 0.05  # windows are thinned to this before comparing
RADIUS_M = 0.6  # largest offset looked for
LATERAL_M = 0.15  # a point of the other window on "the same surface": this close along the surface
SAME_FACING = 0.9  # cos between the two normals
CELL_M = 0.02  # vote grid
INLIER_M = 0.03  # a line through the peak
MIN_POINTS = 80  # points of the smaller window on surfaces the other one saw, at the peak, at least
WIN_OVER_ZERO = 2.0  # support at the offset / support at zero
AGREE_M = 0.03  # windows this close agree
OFFSET_M = 0.05  # windows this far apart disagree
MAX_POINTS = 1000  # per window, after thinning
SPLIT_MIN_PAIRS = 3  # window pairs behind a split, at least
SPLIT_WARN_M = 0.10  # a split this large is reported as a warning


@dataclass
class PairOffset:
    a: str
    b: str
    offset: F64  # (2,) horizontal: move window b by this to put it on window a
    points: int  # points of the smaller window on the peak
    at_zero: int  # ... and on surfaces where they are now (|d| < INLIER_M)
    overlap: int  # points with a same-facing neighbour within RADIUS_M at all
    pinned: int = 2  # directions the surfaces pin (1: parallel walls only, the offset is along their normal)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.offset))

    @property
    def agrees(self) -> bool:
        return self.points >= MIN_POINTS and self.norm < AGREE_M

    @property
    def disagrees(self) -> bool:
        return (
            self.points >= MIN_POINTS
            and self.norm >= OFFSET_M
            and self.points >= WIN_OVER_ZERO * self.at_zero
        )


def _thin(pts: F64, nrm: F64) -> tuple[F64, F64]:
    n = nrm.copy()
    n[:, 2] = 0.0
    keep = np.linalg.norm(n, axis=1) > 0.9  # walls: the normal near horizontal
    pc = to_o3d(pts[keep], n[keep] / np.linalg.norm(n[keep], axis=1, keepdims=True)).voxel_down_sample(
        VOXEL_M
    )
    q = np.asarray(pc.normals)
    p = np.asarray(pc.points)
    if len(p) > MAX_POINTS:  # evenly, deterministic
        pick = np.linspace(0, len(p) - 1, MAX_POINTS).round().astype(int)
        p, q = p[pick], q[pick]
    return p, q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def pair_offset(a: tuple[F64, F64], b: tuple[F64, F64], names: tuple[str, str] = ("a", "b")) -> PairOffset:
    """The horizontal shift of window b onto window a (thinned points, horizontal unit normals)."""
    (pa, na), (pb, nb) = a, b
    flip = len(pb) < len(pa)  # query with the smaller window
    (qp, qn), (tp, tn) = ((pb, nb), (pa, na)) if flip else ((pa, na), (pb, nb))
    empty = PairOffset(names[0], names[1], np.zeros(2), 0, 0, 0)
    if len(qp) < MIN_POINTS or len(tp) < MIN_POINTS:
        return empty
    nns = o3c.nns.NearestNeighborSearch(o3c.Tensor.from_numpy(np.ascontiguousarray(tp, np.float32)))
    nns.hybrid_index(RADIUS_M)
    idx, _, _ = nns.hybrid_search(o3c.Tensor.from_numpy(np.ascontiguousarray(qp, np.float32)), RADIUS_M, 48)
    idx = idx.numpy()
    qi, k = np.nonzero(idx >= 0)
    ti = idx[qi, k]
    same = np.einsum("ij,ij->i", qn[qi], tn[ti]) > SAME_FACING
    qi, ti = qi[same], ti[same]
    rel = tp[ti] - qp[qi]
    d = np.einsum("ij,ij->i", rel, qn[qi])  # along the query point's normal
    lateral = np.linalg.norm(rel - d[:, None] * qn[qi], axis=1)
    ok = lateral < LATERAL_M
    qi, d = qi[ok], d[ok]
    n2 = qn[qi, :2]
    overlap = len(np.unique(qi))
    if overlap < MIN_POINTS:
        return PairOffset(names[0], names[1], np.zeros(2), 0, 0, overlap)
    # lines t . n = d (t: moves the query window onto the other); vote on a grid, one vote per point and cell
    # Rasterise each line over the vote grid, a point's lines sharing one vote (a point in front of two
    # parallel surfaces votes half for each).
    m = round(RADIUS_M / CELL_M)
    steps = np.arange(-m, m + 1) * CELL_M
    along = np.column_stack([-n2[:, 1], n2[:, 0]])
    pts2 = d[:, None, None] * n2[:, None, :] + steps[None, :, None] * along[:, None, :]  # (K, S, 2)
    ij = np.round(pts2 / CELL_M).astype(int) + m
    inside = np.all((ij >= 0) & (ij <= 2 * m), axis=2)
    per_point = np.bincount(qi, minlength=len(qp)).astype(float)
    w = np.broadcast_to((1.0 / per_point[qi])[:, None], inside.shape)[inside]
    cell_ij = ij[inside][:, 0] * (2 * m + 1) + ij[inside][:, 1]
    votes = np.bincount(cell_ij, weights=w, minlength=(2 * m + 1) ** 2)
    k = int(np.argmax(votes))
    t = (np.array([k // (2 * m + 1), k % (2 * m + 1)]) - m) * CELL_M
    strong = np.eye(2)
    for _ in range(3):  # least squares on the lines through the peak, only along directions they pin
        inl = np.abs(n2 @ t - d) < INLIER_M
        a_mat = n2[inl].T @ n2[inl] / max(1, int(inl.sum()))
        w, v = np.linalg.eigh(a_mat)
        sol = np.linalg.lstsq(n2[inl], d[inl], rcond=None)[0]
        strong = v[:, w > 0.05]
        t = strong @ (strong.T @ sol)
    inl = np.abs(n2 @ t - d) < INLIER_M
    points = len(np.unique(qi[inl]))
    at_zero = len(np.unique(qi[np.abs(d) < INLIER_M]))
    offset = t if flip else -t  # as "move b onto a"
    return PairOffset(names[0], names[1], offset, points, at_zero, overlap, int(strong.shape[1]))


def _session_t(name: str) -> tuple[str, float]:
    s, t = name.rsplit("@", 1)
    return s, float(t)


def compare(windows: dict[str, tuple[F64, F64]]) -> list[PairOffset]:
    """Every pair of windows that may have seen the same surfaces: other sessions, or a window apart."""
    thin = {k: _thin(*v) for k, v in windows.items()}
    names = sorted(k for k, (p, _) in thin.items() if len(p) >= MIN_POINTS)
    lo = {k: thin[k][0].min(axis=0) - RADIUS_M for k in names}
    hi = {k: thin[k][0].max(axis=0) + RADIUS_M for k in names}
    out: list[PairOffset] = []
    for i, a in enumerate(names):
        sa, ta = _session_t(a)
        for b in names[i + 1 :]:
            sb, tb = _session_t(b)
            if sa == sb and abs(ta - tb) < WINDOW_S:
                continue  # overlapping windows of one session share frames
            if np.any(lo[a] > hi[b]) or np.any(lo[b] > hi[a]):
                continue
            out.append(pair_offset(thin[a], thin[b], (a, b)))
    return [p for p in out if p.overlap >= MIN_POINTS]


def splits(pairs: list[PairOffset]) -> list[dict[str, object]]:
    """Disagreeing pairs grouped by their offset (each pair oriented so its offset points the same way as
    its group's): per group the median offset, the pairs and points behind it, and the windows on each
    side (`moved`: the ones that sit `offset_m` away from `reference`). Largest support first."""
    todo = sorted((p for p in pairs if p.disagrees), key=lambda p: (-p.pinned, -p.points))
    out: list[dict[str, object]] = []

    def near(p: PairOffset, seed: F64) -> bool:
        # a pair that pins one direction only has its offset along that direction: compare it there
        diff = p.offset - seed
        if p.pinned < 2 and p.norm > 0:
            diff = np.array([diff @ p.offset / p.norm])
        return float(np.linalg.norm(diff)) <= 2 * OFFSET_M

    while todo:
        seed = todo[0].offset
        group: list[PairOffset] = []
        rest: list[PairOffset] = []
        for p in todo:
            if near(p, seed):
                group.append(p)
            elif near(PairOffset(p.b, p.a, -p.offset, 0, 0, 0, p.pinned), seed):  # the other way round
                group.append(PairOffset(p.b, p.a, -p.offset, p.points, p.at_zero, p.overlap, p.pinned))
            else:
                rest.append(p)
        full = [p.offset for p in group if p.pinned == 2]
        off = np.median(full if full else [p.offset for p in group], axis=0)
        out.append(
            {
                "offset_m": [round(float(x), 3) for x in off],
                "offset_norm_m": round(float(np.linalg.norm(off)), 3),
                "pairs": len(group),
                "points": int(sum(p.points for p in group)),
                "reference": _describe(sorted({p.a for p in group})),
                "moved": _describe(sorted({p.b for p in group})),
            }
        )
        todo = rest
    return sorted(out, key=lambda s: -int(s["points"]))  # type: ignore[call-overload]


def _describe(names: list[str]) -> list[str]:
    """`session@first-last s` per session among some windows."""
    by: dict[str, list[float]] = {}
    for n in names:
        s, t = _session_t(n)
        by.setdefault(s, []).append(t)
    return [f"{s}@{min(ts):.0f}-{max(ts) + WINDOW_S:.0f}s" for s, ts in sorted(by.items())]


def report(walls_dir: Path) -> dict[str, object]:
    """The doubled-surface check over a poses folder's `walls/`: pairs compared, how many agree and disagree,
    the splits (`splits`), and `split_max_m`: the largest split backed by `SPLIT_MIN_PAIRS` pairs or more
    (`split_warning` when it reaches `SPLIT_WARN_M`)."""
    windows = {w.stem: read_wall_points(w) for w in sorted(walls_dir.glob("*.npz"))}
    pairs = compare(windows)
    found = splits(pairs)
    backed = [s for s in found if int(s["pairs"]) >= SPLIT_MIN_PAIRS]  # type: ignore[call-overload]
    split_max = max((float(s["offset_norm_m"]) for s in backed), default=0.0)  # type: ignore[arg-type]
    disagree = sorted((p for p in pairs if p.disagrees), key=lambda p: -p.points)
    return {
        "windows": len(windows),
        "pairs_compared": len(pairs),
        "pairs_agree": sum(p.agrees for p in pairs),
        "pairs_disagree": len(disagree),
        "split_max_m": round(split_max, 3),
        "split_warning": split_max >= SPLIT_WARN_M,
        "splits": found[:10],
        "disagreeing_pairs": [
            {
                "a": p.a,
                "b": p.b,
                "offset_m": [round(float(x), 3) for x in p.offset],
                "points": p.points,
                "at_zero": p.at_zero,
            }
            for p in disagree[:30]
        ],
    }


def warning_line(r: dict[str, object]) -> str | None:
    """One line for the console when the capture is split, else None."""
    if not r.get("split_warning"):
        return None
    s = cast(list[dict[str, Any]], r["splits"])
    top = max((x for x in s if int(x["pairs"]) >= SPLIT_MIN_PAIRS), key=lambda x: float(x["offset_norm_m"]))
    return (
        f"doubled surfaces: {top['pairs']} window pairs see the same walls {100 * float(top['offset_norm_m']):.0f} cm"
        f" apart ({', '.join(top['moved'])} vs {', '.join(top['reference'])}): the poses are split into blocks"
    )
