"""Doubled surfaces (scanmodel.consistency): windows that put the same walls in different places."""

from pathlib import Path

import numpy as np

from scanmodel import consistency
from scanmodel.models.common import F64
from scanmodel.points import write_wall_points


def _room(
    rng: np.random.Generator, shift: tuple[float, float] = (0.0, 0.0), n: int = 4000
) -> tuple[F64, F64]:
    """Wall points of a 4 x 3 m room seen from inside (normals facing in), plus a 1.2 m partition at
    x = 1.5 seen from its west side, all moved by `shift`; 3 mm of LiDAR noise."""
    k = n // 5
    s = rng.uniform(0.05, 0.95, (5, k))
    h = rng.uniform(0.3, 2.3, (5, k))
    walls = [
        (np.column_stack([4 * s[0], np.zeros(k), h[0]]), (0.0, 1.0)),  # south, facing north
        (np.column_stack([4 * s[1], np.full(k, 3.0), h[1]]), (0.0, -1.0)),
        (np.column_stack([np.zeros(k), 3 * s[2], h[2]]), (1.0, 0.0)),
        (np.column_stack([np.full(k, 4.0), 3 * s[3], h[3]]), (-1.0, 0.0)),
        (np.column_stack([np.full(k, 1.5), 1.2 * s[4], h[4]]), (-1.0, 0.0)),
    ]
    pts = np.concatenate([p for p, _ in walls]) + np.array([shift[0], shift[1], 0.0])
    pts += rng.normal(0.0, 0.003, pts.shape)
    nrm = np.concatenate([np.tile([nx, ny, 0.0], (len(p), 1)) for p, (nx, ny) in walls])
    return pts, nrm


def _windows(
    tmp: Path, shifts: dict[str, tuple[float, float]], extra: dict[str, tuple[F64, F64]] | None = None
) -> Path:
    rng = np.random.default_rng(3)
    d = tmp / "walls"
    for name, shift in shifts.items():
        pts, nrm = _room(rng, shift)
        if extra and name in extra:
            pts, nrm = np.vstack([pts, extra[name][0]]), np.vstack([nrm, extra[name][1]])
        write_wall_points(d / f"{name}.npz", pts, nrm)
    return d


def test_pair_offset_finds_the_shift_and_only_pinned_directions() -> None:
    rng = np.random.default_rng(0)
    a = consistency._thin(*_room(rng))  # pyright: ignore[reportPrivateUsage]
    b = consistency._thin(*_room(rng, (0.18, -0.32)))  # pyright: ignore[reportPrivateUsage]
    p = consistency.pair_offset(a, b)
    assert np.allclose(p.offset, [-0.18, 0.32], atol=0.01), p.offset  # move b onto a
    assert p.disagrees and not p.agrees and p.pinned == 2
    assert p.points > 10 * p.at_zero
    # only the north and south walls: x is not pinned, the offset is along y alone
    pts, nrm = _room(rng)
    ns = np.abs(nrm[:, 1]) > 0.5
    pb, nb = _room(rng, (0.18, -0.32))
    q = consistency.pair_offset(
        consistency._thin(pts[ns], nrm[ns]),  # pyright: ignore[reportPrivateUsage]
        consistency._thin(pb[ns], nb[ns]),  # pyright: ignore[reportPrivateUsage]
    )
    assert q.pinned == 1 and abs(q.offset[0]) < 0.01 and abs(q.offset[1] - 0.32) < 0.01, q.offset


def test_split_into_two_blocks_is_reported(tmp_path: Path) -> None:
    """Two sessions, two windows each; the second session sits 0.2 m east and 0.4 m north of the first,
    as scan 5's hallway and kitchen did: the same walls twice, a warning."""
    d = _windows(
        tmp_path,
        {"s1@0000": (0, 0), "s1@0030": (0, 0), "s2@0000": (0.2, 0.4), "s2@0030": (0.2, 0.4)},
    )
    r = consistency.report(d)
    assert r["pairs_compared"] == 6
    assert r["pairs_agree"] == 2 and r["pairs_disagree"] == 4
    assert r["split_warning"] and abs(float(r["split_max_m"]) - np.hypot(0.2, 0.4)) < 0.01  # type: ignore[arg-type]
    top = r["splits"][0]  # type: ignore[index]
    sides = {top["moved"][0].split("@")[0], top["reference"][0].split("@")[0]}  # type: ignore[index]
    assert sides == {"s1", "s2"}
    line = consistency.warning_line(r)
    assert line is not None and "45 cm" in line


def test_consistent_windows_and_a_radiator_stay_quiet(tmp_path: Path) -> None:
    """All windows agree; one also sees a radiator 6 cm in front of the south wall (a real surface in front
    of another, from one window only): no split."""
    rng = np.random.default_rng(5)
    k = 600
    radiator = np.column_stack([rng.uniform(1.0, 2.2, k), np.full(k, 0.06), rng.uniform(0.3, 0.9, k)])
    extra = {"s2@0000": (radiator, np.tile([0.0, 1.0, 0.0], (k, 1)))}
    d = _windows(
        tmp_path,
        {"s1@0000": (0, 0), "s1@0030": (0.005, 0), "s2@0000": (0, 0.005), "s2@0030": (0, 0)},
        extra,
    )
    r = consistency.report(d)
    assert r["pairs_compared"] == 6 and r["pairs_agree"] == 6, r
    assert r["split_max_m"] == 0.0 and not r["split_warning"]
    assert consistency.warning_line(r) is None
