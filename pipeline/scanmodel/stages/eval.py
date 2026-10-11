"""Stage 9, eval (geometric part): how far is the model from the fused points?

- Per wall face: distance from the face plane to the fused points that belong to it: points facing
  the same way, outside openings, within 10 cm, and on the dominant surface there (within 3 cm of it),
  so radiators, pictures and furniture in front of a wall don't count. A face placed off the real wall
  still shows its full error. Mean, 95th percentile and bias. A face nobody scanned is skipped.
  The headline (`observed_*`, pass/fail) is over faces a session saw (`Wall.faces_observed`): the back
  of a wall seen from one side only is placed a default thickness behind it, a guess, and what the
  fused points hold there is mostly TSDF spill. Those faces are reported apart (`unobserved_*`);
  `mean_m` / `p95_m` stay over all faces, as in history.jsonl before the split.
  A seen face too far off its wall to be measured drops out of the headline: `observed_faces` of
  `observed_faces_total` says how many were measured.
- Gross errors: a face 10-50 cm from its wall is outside that 10 cm window, so it scores whatever
  stands nearby or is skipped. Per face, the nearest same-facing surface within 50 cm, looked for in
  a band just below the wall's top (above most furniture) and only where it runs along at least half
  of the face (furniture is patchy, a wall isn't); faces whose nearest surface is 10 cm or more away
  are listed (`gross`, `faces_gross` over seen faces).
- Per opening: `solid_fraction`, how much of the opening's rectangle is covered by wall surface,
  relative to the wall right beside it: occupied 5 cm cells inside / occupied cells in a 30 cm strip
  either side, on the same face (points within 3 cm of the face plane, facing the same way), clipped
  to 0..1. ~0 means the hole is real; ~1 means the opening sits on solid wall. None when the strip
  beside it was barely scanned. Windows and openings above 0.5 are "suspect"; doors are not, since a
  closed door leaf reads as solid.
- Floors (with the poses the points were fused from): per room, each session's own floor height (the
  peak of its LiDAR points over the room's floor, 30 cm in from the walls) and the spread (the largest
  offset between two sessions over the 0.5 m cells both saw, so a sloping floor seen in different parts
  by different sessions doesn't count); and how far
  rooms that should be level (floor within LEVEL_M of the largest room's) sit from it. A spread or offset
  of more than a cm means the poses disagree vertically (refine without floor factors: 2 cm, 1.5 cm).
- Doubled surfaces (`scanmodel.consistency`, with the poses the points were fused from): every two 30 s windows of
  the capture that saw the same walls from the same side, compared by their own LiDAR (`walls/`); a group
  of window pairs that agree on a shift of 10 cm or more means the poses are split into rigid blocks and the
  walls are there twice (`doubled`, `split_max_m`, `split_warning`; a warning after `scanmodel all`). The face
  metrics can't see that: each face is fitted to one copy.
- Heatmap (heatmap.ply + plan.png) and report.html, plus history.jsonl so runs can be compared.

The photometric part (SSIM / LPIPS from held-out stills) is phase 3.
"""

import base64
import html
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from shapely import contains_xy
from shapely.geometry import Polygon

from .. import consistency
from ..boxes import peak
from ..config import from_repo
from ..io import read_jsonl
from ..models import bundle as b
from ..models.common import F64
from ..models.derived import PosedImage, Poses
from ..models.scene import Opening, OpeningKind, Scene, Wall
from ..points import WALLS_DIR, read_points, sample_points

PASS_P95_M = 0.02
GROSS_M = (0.10, 0.50)  # a face whose nearest surface is this far off is a gross error
GROSS_BAND_M = (0.45, 0.10)  # it is looked for between these depths below the wall's top
GROSS_COVER = 0.5  # share of the face's scanned 10 cm columns the surface must reach
LEVEL_M = 0.03  # rooms whose floor is within this of the largest room's are meant to be level with it
FLOOR_WINDOW_M = 0.12  # a session's floor in a room: its points within this of the room's floor_z
FLOOR_CELL_M = 0.5  # sessions' floors are compared per cell of this size


@dataclass
class FaceStats:
    wall: str
    side: int  # +1: the wall's left face (normal side), -1: right face
    n: int
    mean_m: float
    p95_m: float
    bias_m: float
    observed: bool = True  # a session saw this face (Wall.faces_observed)


def _wall_frame(w: Wall, pts: F64) -> tuple[F64, F64, F64]:
    """Points in the wall's frame: along (s), across (d, + = left/normal side), height above base (h)."""
    rel = pts[:, :2] - np.array(w.start)
    return rel @ w.direction(), rel @ w.normal(), pts[:, 2] - w.base_z


def _in_openings(
    s: F64, h: F64, openings: list[tuple[float, float, float, float]], margin: float
) -> NDArray[np.bool_]:
    m = np.zeros(s.shape, bool)
    for offset, width, bottom, height in openings:
        m |= (
            (np.abs(s - offset) < width / 2 + margin) & (h > bottom - margin) & (h < bottom + height + margin)
        )
    return m


CELL_M = 0.05
STRIP_M = 0.30
MIN_STRIP_COVER = 0.2


def _cover(s: F64, h: F64, mask: NDArray[np.bool_], s0: float, s1: float, h0: float, h1: float) -> float:
    """Share of CELL_M cells in [s0, s1] x [h0, h1] that hold at least one masked point."""
    ns, nh = int((s1 - s0) / CELL_M), int((h1 - h0) / CELL_M)
    if ns < 1 or nh < 1:
        return 0.0
    m = mask & (s >= s0) & (s < s0 + ns * CELL_M) & (h >= h0) & (h < h0 + nh * CELL_M)
    cells = np.unique(((s[m] - s0) // CELL_M).astype(int) * nh + ((h[m] - h0) // CELL_M).astype(int))
    return cells.size / (ns * nh)


def _solid_fraction(
    w: Wall,
    o: Opening,
    s: F64,
    d: F64,
    h: F64,
    n_across: F64,
    body: NDArray[np.bool_],
    holes: NDArray[np.bool_],
) -> float | None:
    """Wall-surface cover inside the opening relative to the strips beside it, on the better-scanned face."""
    lo, hi = o.offset - o.width / 2, o.offset + o.width / 2
    h0, h1 = o.bottom + 0.03, o.bottom + o.height - 0.03
    best = (-1.0, 0.0)  # (strip cover, inside cover)
    for side in (1, -1):
        face = (np.abs(d - side * w.thickness / 2) < 0.03) & (side * n_across > 0.8)
        inside = _cover(s, h, face, lo + 0.03, hi - 0.03, h0, h1)
        beside = face & body & ~holes
        left = _cover(s, h, beside, lo - STRIP_M, lo - 0.05, h0, h1)
        right = _cover(s, h, beside, hi + 0.05, hi + STRIP_M, h0, h1)
        strip = max(left, right)
        if strip > best[0]:
            best = (strip, inside)
    if best[0] < MIN_STRIP_COVER:
        return None
    return float(min(best[1] / best[0], 1.0))


def nearest_surface(dist: F64, s: F64) -> float | None:
    """Offset of the surface nearest the face plane among `dist` (same-facing points within GROSS_M[1],
    along-wall positions `s`) that occupies GROSS_COVER of the 10 cm columns holding any of them."""
    cols = np.floor(s / 0.10).astype(int)
    scanned = np.unique(cols).size
    if dist.size < 100 or scanned < 5:
        return None
    edges = np.arange(-GROSS_M[1], GROSS_M[1] + 0.005, 0.01)
    centres = (edges[:-1] + edges[1:]) / 2
    hs = np.convolve(np.histogram(dist, edges)[0], [1, 2, 1], "same")
    for i in np.argsort(np.abs(centres), kind="stable"):  # nearest first
        if hs[i] < 40 or hs[i] < hs[max(i - 1, 0)] or hs[i] < hs[min(i + 1, len(hs) - 1)]:
            continue
        on = np.abs(dist - centres[i]) < 0.015
        if np.unique(cols[on]).size >= GROSS_COVER * scanned:
            return float(dist[on].mean())
    return None


def _suspect(o: dict[str, object]) -> bool:
    f = o["solid_fraction"]
    return isinstance(f, float) and f > 0.5 and o["kind"] != OpeningKind.DOOR.value


def _colour(err: F64) -> F64:
    """0 -> green, 2 cm -> yellow, >= 5 cm -> red."""
    t = np.clip(err / 0.05, 0, 1)
    return np.stack([np.clip(t * 2.5, 0, 1), np.clip(2 - t * 2, 0, 1), np.zeros_like(t)], axis=1)


def _floor_peak(z: F64, centre: float, min_points: int = 1500) -> float | None:
    """The floor's height in `z` (a room's points of one session): the densest 4 mm bin within
    FLOOR_WINDOW_M of `centre`, refined to the median of the points within 8 mm of it."""
    z = z[np.abs(z - centre) < FLOOR_WINDOW_M]
    if len(z) < min_points:
        return None
    hist, edges = np.histogram(z, bins=np.arange(centre - FLOOR_WINDOW_M, centre + FLOOR_WINDOW_M, 0.004))
    i = int(np.argmax(hist))
    mid = (edges[i] + edges[i + 1]) / 2
    return float(np.median(z[np.abs(z - mid) < 0.008]))


def floors(scene: Scene, poses_dir: Path, frames_per_session: int = 200) -> dict[str, object]:
    """Per room: each session's floor height (m; where that session saw it) and the spread: the largest
    offset between two sessions' floors over the cells (FLOOR_CELL_M) both saw (median), so a sloping floor
    seen in different parts doesn't count. Per room meant to be level with the largest room, its floor_z
    offset from that room's (see the module docstring)."""
    poses = Poses.model_validate_json((poses_dir / "poses.json").read_text())
    raw = {s.session: from_repo(s.raw_dir) for s in poses.sessions}
    by: dict[str, list[PosedImage]] = {}
    for f in read_jsonl(poses_dir / "frames.jsonl", PosedImage):
        if f.depth and f.tracking in (None, b.TrackingState.NORMAL):
            by.setdefault(f.session, []).append(f)
    inner = []
    for r in scene.rooms:
        poly = Polygon(r.polygon)
        shrunk = poly.buffer(-0.3)
        inner.append(shrunk if not shrunk.is_empty and shrunk.area > 0.5 else poly)
    rooms: dict[str, dict[str, object]] = {r.name or r.id: {"sessions": {}} for r in scene.rooms}
    cells: dict[str, dict[str, dict[tuple[int, int], float]]] = {k: {} for k in rooms}
    for sid, fs in sorted(by.items()):
        if sid not in raw:
            continue
        pts = sample_points(raw[sid], fs, frames_per_session, stride=2, min_confidence=2, max_depth=3.5)
        for r, poly in zip(scene.rooms, inner, strict=True):
            name = r.name or r.id
            q = pts[contains_xy(poly, pts[:, 0], pts[:, 1])]
            z = _floor_peak(q[:, 2], r.floor_z)
            if z is None:
                continue
            rooms[name]["sessions"][sid] = round(z, 4)  # type: ignore[index]
            keys = np.floor(q[:, :2] / FLOOR_CELL_M).astype(int)
            own: dict[tuple[int, int], float] = {}
            for k in {(int(a), int(c)) for a, c in keys}:
                zc = _floor_peak(q[(keys[:, 0] == k[0]) & (keys[:, 1] == k[1]), 2], r.floor_z, 300)
                if zc is not None:
                    own[k] = zc
            cells[name][sid] = own
    for name, v in rooms.items():
        pairs: dict[str, float] = {}
        sess = sorted(cells[name])
        for i, a in enumerate(sess):
            for c in sess[i + 1 :]:
                shared = cells[name][a].keys() & cells[name][c].keys()
                if len(shared) >= 3:
                    d = np.median([cells[name][a][k] - cells[name][c][k] for k in shared])
                    pairs[f"{a}/{c}"] = round(float(d), 4)
        v["pairs_m"] = pairs
        v["spread_m"] = max((abs(d) for d in pairs.values()), default=None)
    main = max(scene.rooms, key=lambda r: Polygon(r.polygon).area) if scene.rooms else None
    level: dict[str, float] = {}
    if main is not None:
        level = {
            r.name or r.id: round(r.floor_z - main.floor_z, 4)
            for r in scene.rooms
            if r is not main and abs(r.floor_z - main.floor_z) < LEVEL_M
        }
    spreads = {k: v["spread_m"] for k, v in rooms.items() if v["spread_m"] is not None}
    worst = max(spreads, key=lambda k: spreads[k], default=None)  # type: ignore[arg-type]
    off = max(level, key=lambda k: abs(level[k]), default=None)
    return {
        "rooms": rooms,
        "level_with": main.name or main.id if main else None,
        "level_offset_m": level,
        "spread_max_m": spreads[worst] if worst else None,
        "spread_max_room": worst,
        "level_offset_max_m": abs(level[off]) if off else None,
        "level_offset_max_room": off,
    }


def run(scene_path: Path, fuse_dir: Path, out: Path, poses_dir: Path | None = None) -> dict[str, object]:
    scene = Scene.model_validate_json(scene_path.read_text())
    pts, nrm = read_points(fuse_dir / "points.ply")
    out.mkdir(parents=True, exist_ok=True)

    faces: list[FaceStats] = []
    errs: list[F64] = []
    gross: list[dict[str, object]] = []
    near = np.full(len(pts), np.inf)
    openings_out: list[dict[str, object]] = []
    for w in scene.walls:
        s, d, h = _wall_frame(w, pts)
        n_across = nrm[:, :2] @ w.normal()
        ops = [(o.offset, o.width, o.bottom, o.height) for o in scene.openings if o.wall == w.id]
        body = (s > 0.05) & (s < w.length() - 0.05) & (h > 0.10) & (h < w.height - 0.10)
        holes = _in_openings(s, h, ops, 0.05)
        top = min(w.height, w.height_end or w.height)
        band = body & ~holes & (h > top - GROSS_BAND_M[0]) & (h < top - GROSS_BAND_M[1])
        for side, observed in zip((1, -1), w.faces_observed, strict=True):
            dist = d - side * w.thickness / 2  # signed distance from this face, + = in front of it
            wide = band & (np.abs(dist) < GROSS_M[1]) & (side * n_across > 0.8)
            surface_at = nearest_surface(dist[wide], s[wide])
            if surface_at is not None and abs(surface_at) >= GROSS_M[0]:
                gross.append({"wall": w.id, "side": side, "observed": observed, "surface_m": surface_at})
            on = body & ~holes & (np.abs(dist) < 0.10) & (side * n_across > 0.8)
            surface = peak(dist[on]) if on.sum() >= 100 else None
            if surface is not None:
                on[on] = np.abs(dist[on] - surface[0]) < 0.03  # the wall itself, not what is in front of it
            if on.sum() >= 100:
                e = np.abs(dist[on])
                faces.append(
                    FaceStats(
                        w.id,
                        side,
                        int(on.sum()),
                        float(e.mean()),
                        float(np.percentile(e, 95)),
                        float(dist[on].mean()),
                        observed,
                    )
                )
                errs.append(e)
                near[on] = np.minimum(near[on], e)
        for o in [o for o in scene.openings if o.wall == w.id]:
            openings_out.append(
                {
                    "id": o.id,
                    "kind": o.kind.value,
                    "wall": w.id,
                    "is_open": o.is_open,
                    "solid_fraction": _solid_fraction(w, o, s, d, h, n_across, body, holes),
                }
            )

    def stats(keep: list[bool]) -> tuple[int, int, float | None, float | None]:
        e = [x for x, k in zip(errs, keep, strict=True) if k]
        a = np.concatenate(e) if e else np.zeros(0)
        if not a.size:
            return len(e), 0, None, None
        return len(e), int(a.size), float(a.mean()), float(np.percentile(a, 95))

    seen = [f.observed for f in faces]
    n_all, pts_all, mean_all, p95_all = stats([True] * len(faces))
    n_obs, pts_obs, mean_obs, p95_obs = stats(seen)
    n_unobs, _, mean_unobs, p95_unobs = stats([not k for k in seen])
    summary: dict[str, object] = {
        "walls": len(scene.walls),
        # all faces: the headline before seen and unseen faces were told apart (history.jsonl)
        "faces_measured": n_all,
        "points": pts_all,
        "mean_m": mean_all,
        "p95_m": p95_all,
        # the headline: faces a session saw; a seen face that lost its points (moved off its wall) only
        # shows in observed_faces < observed_faces_total and in faces_gross
        "observed_faces": n_obs,
        "observed_faces_total": sum(sum(w.faces_observed) for w in scene.walls),
        "observed_points": pts_obs,
        "observed_mean_m": mean_obs,
        "observed_p95_m": p95_obs,
        "unobserved_faces": n_unobs,
        "unobserved_mean_m": mean_unobs,
        "unobserved_p95_m": p95_unobs,
        "pass": p95_obs is not None and p95_obs < PASS_P95_M,
        "faces_gross": sum(bool(g["observed"]) for g in gross),
        "openings_suspect": sum(_suspect(o) for o in openings_out),
    }
    floor_report = (
        floors(scene, poses_dir) if poses_dir is not None and (poses_dir / "poses.json").is_file() else None
    )
    if floor_report is not None:
        summary |= {
            k: floor_report[k]
            for k in ("spread_max_m", "spread_max_room", "level_offset_max_m", "level_offset_max_room")
        }
    walls_dir = poses_dir / WALLS_DIR if poses_dir is not None else None
    doubled = consistency.report(walls_dir) if walls_dir is not None and walls_dir.is_dir() else None
    if doubled is not None:
        summary |= {
            "split_max_m": doubled["split_max_m"],
            "split_warning": doubled["split_warning"],
            "window_pairs_disagree": doubled["pairs_disagree"],
            "split_warning_line": consistency.warning_line(doubled),
        }

    # Heatmap: wall points coloured by error, everything else grey.
    colours = np.full((len(pts), 3), 0.6)
    measured = np.isfinite(near)
    colours[measured] = _colour(near[measured])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(colours)
    o3d.io.write_point_cloud(out / "heatmap.ply", pc)
    plan_png = _plan_image(scene, pts, colours, measured)
    plan_png.save(out / "plan.png")

    report = {
        "summary": summary,
        "faces": [f.__dict__ for f in faces],
        "gross": gross,
        "openings": openings_out,
        "floors": floor_report,
        "doubled": doubled,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    history = out / "history.jsonl"
    with history.open("a") as fh:
        fh.write(json.dumps({"time": datetime.now(UTC).isoformat(), **summary}) + "\n")
    (out / "report.html").write_text(
        _html(summary, faces, openings_out, plan_png, history, floor_report, gross, doubled)
    )
    return summary


def _plan_image(scene: Scene, pts: F64, colours: F64, measured: F64, px_per_m: float = 100) -> Image.Image:
    lo = pts[:, :2].min(axis=0) - 0.3
    hi = pts[:, :2].max(axis=0) + 0.3
    w, h = (np.ceil((hi - lo) * px_per_m).astype(int) + 1).tolist()
    img = np.full((h, w, 3), 255, np.uint8)
    order = np.argsort(measured)  # measured (coloured) points drawn last
    u = ((pts[order, 0] - lo[0]) * px_per_m).astype(int)
    v = ((hi[1] - pts[order, 1]) * px_per_m).astype(int)
    img[v, u] = (colours[order] * 255).astype(np.uint8)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - lo[0]) * px_per_m, (hi[1] - y) * px_per_m

    for wall in scene.walls:
        a, b2 = np.array(wall.start), np.array(wall.end)
        n = wall.normal() * wall.thickness / 2
        draw.polygon([px(*q) for q in (a + n, b2 + n, b2 - n, a - n)], outline=(40, 40, 160))
    draw.ellipse([*(np.array(px(0, 0)) - 4), *(np.array(px(0, 0)) + 4)], outline=(200, 0, 200), width=2)
    return im


def _sparkline(history: Path) -> str:
    """p95 over the last runs: all faces (every run has it) and seen faces (runs since the split)."""
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("p95_m") is not None][-40:]
    if len(rows) < 2:
        return ""
    top = max(*(r["p95_m"] * 1000 for r in rows), PASS_P95_M * 1000) * 1.1
    step = 300 / (len(rows) - 1)

    def line(key: str, colour: str) -> str:
        pts = " ".join(
            f"{i * step:.1f},{60 - r[key] * 1000 / top * 60:.1f}"
            for i, r in enumerate(rows)
            if r.get(key) is not None
        )
        return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>' if pts else ""

    limit = 60 - PASS_P95_M * 1000 / top * 60
    return (
        f'<svg width="300" height="60" role="img" aria-label="p95 history">'
        f'<line x1="0" x2="300" y1="{limit:.1f}" y2="{limit:.1f}" stroke="#c33" stroke-dasharray="4"/>'
        f"{line('p95_m', '#999')}{line('observed_p95_m', '#236')}</svg>"
        f"<p>p95 over the last {len(rows)} runs (mm): blue seen faces, grey all faces; dashed ="
        f" {PASS_P95_M * 1000:.0f} mm target</p>"
    )


def _floors_html(f: dict[str, object]) -> str:
    def cm(v: object) -> str:
        return f"{float(v) * 100:+.1f}" if isinstance(v, int | float) else "-"

    rooms: dict[str, dict[str, object]] = f["rooms"]  # type: ignore[assignment]
    sessions = sorted({s for r in rooms.values() for s in r["sessions"]})  # type: ignore[union-attr]
    head = "".join(f"<th>{html.escape(s)}</th>" for s in sessions)
    level: dict[str, float] = f["level_offset_m"]  # type: ignore[assignment]
    rows = "".join(
        f"<tr class={'bad' if float(r['spread_m'] or 0) > 0.01 else 'ok'}>"  # type: ignore[arg-type]
        f"<td>{html.escape(name)}</td>"
        + "".join(f"<td>{cm(r['sessions'].get(s))}</td>" for s in sessions)  # type: ignore[union-attr]
        + f"<td>{cm(r['spread_m']).lstrip('+')}</td><td>{cm(level.get(name))}</td></tr>"
        for name, r in rooms.items()
    )
    return f"""<h2>Floors</h2>
<p class="muted">Each session's own floor height per room (cm, LiDAR at the fused poses), the spread across
sessions, and the room's floor_z relative to {html.escape(str(f["level_with"]))} for rooms meant to be level with it.
More than 1 cm means the poses disagree vertically.</p>
<table><tr><th>room</th>{head}<th>spread</th><th>vs {html.escape(str(f["level_with"]))}</th></tr>{rows}</table>
"""


def _doubled_html(r: dict[str, object]) -> str:
    splits = cast(list[dict[str, Any]], r["splits"])
    rows = "".join(
        f"<tr class={'bad' if float(x['offset_norm_m']) >= consistency.SPLIT_WARN_M else 'ok'}>"
        f"<td>{html.escape(', '.join(x['moved']))}</td><td>{html.escape(', '.join(x['reference']))}</td>"
        f"<td>{float(x['offset_norm_m']) * 100:.1f}</td>"
        f"<td>{', '.join(f'{float(v) * 100:+.1f}' for v in x['offset_m'])}</td><td>{x['pairs']}</td></tr>"
        for x in splits
    )
    table = (
        "<table><tr><th>these windows</th><th>sit this far from these</th><th>cm</th><th>x, y cm</th>"
        f"<th>window pairs</th></tr>{rows}</table>"
        if rows
        else "<p>No two windows disagree.</p>"
    )
    head = "Doubled surfaces: the poses are split" if r["split_warning"] else "Doubled surfaces: none"
    return f"""<h2>{head}</h2>
<p class="muted">{r["pairs_compared"]} pairs of 30 s windows that saw the same walls from the same side, compared by
their own LiDAR: {r["pairs_agree"]} agree within {consistency.AGREE_M * 100:.0f} cm, {r["pairs_disagree"]} are
{consistency.OFFSET_M * 100:.0f} cm or more apart. Groups of pairs that agree on one shift (moving the first
windows by "x, y" puts them on the second): {consistency.SPLIT_WARN_M * 100:.0f} cm or more, backed by
{consistency.SPLIT_MIN_PAIRS} pairs or more, means the same walls are there twice.</p>
{table}
"""


def _html(
    summary: dict[str, object],
    faces: list[FaceStats],
    openings: list[dict[str, object]],
    plan: Image.Image,
    history: Path,
    floor_report: dict[str, object] | None = None,
    gross: list[dict[str, object]] | None = None,
    doubled: dict[str, object] | None = None,
) -> str:
    buf = io.BytesIO()
    plan.save(buf, format="PNG")
    png = base64.b64encode(buf.getvalue()).decode()

    def mm(v: object) -> str:
        return f"{float(v) * 1000:.1f}" if isinstance(v, int | float) else "-"

    face_rows = "".join(
        f"<tr class={'bad' if f.p95_m >= PASS_P95_M else 'ok'}><td>{html.escape(f.wall)}</td>"
        f"<td>{'left' if f.side > 0 else 'right'}{'' if f.observed else ' (unseen)'}</td>"
        f"<td>{f.n}</td><td>{mm(f.mean_m)}</td>"
        f"<td>{mm(f.p95_m)}</td><td>{mm(f.bias_m)}</td></tr>"
        for f in sorted(faces, key=lambda f: -f.p95_m)
    )
    op_rows = "".join(
        f"<tr><td>{html.escape(str(o['id']))}</td><td>{o['kind']}</td><td>{html.escape(str(o['wall']))}</td>"
        f"<td>{'-' if o['solid_fraction'] is None else f'{o["solid_fraction"]:.2f}'}</td>"
        f"<td>{'suspect' if _suspect(o) else ''}</td></tr>"
        for o in openings
    )
    floor_html = _floors_html(floor_report) if floor_report else ""
    gross_rows = "".join(
        f"<tr class=bad><td>{html.escape(str(g['wall']))}</td><td>{'left' if g['side'] == 1 else 'right'}"
        f"{'' if g['observed'] else ' (unseen)'}</td><td>{mm(g['surface_m'])}</td></tr>"
        for g in gross or []
    )
    gross_html = (
        f"""<h2>Gross errors</h2>
<p class="muted">Faces whose nearest surface facing the same way is {GROSS_M[0] * 100:.0f}-{GROSS_M[1] * 100:.0f} cm
off (+ = in front of the face), outside the window below: looked for {GROSS_BAND_M[1] * 100:.0f}-{GROSS_BAND_M[0] * 100:.0f}
cm below the wall's top, along at least half of the face. Tall furniture can still pass for a wall.</p>
<table><tr><th>wall</th><th>face</th><th>surface mm</th></tr>{gross_rows}</table>
"""
        if gross_rows
        else ""
    )
    doubled_html = _doubled_html(doubled) if doubled else ""
    verdict = "PASS" if summary["pass"] else "FAIL"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Scan eval</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --fg:#1b1f24; --bg:#fff; --muted:#667; --ok:#e8f5e9; --bad:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6e6e6; --bg:#16181c; --muted:#99a; --ok:#1d3321; --bad:#3a1d1d; }} }}
body {{ font: 15px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 0 auto; max-width: 1100px; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 24px; }}
td, th {{ padding: 4px 8px; border-bottom: 1px solid #8884; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
tr.ok {{ background: var(--ok); }} tr.bad {{ background: var(--bad); }}
img {{ max-width: 100%; border: 1px solid #8884; }} .muted {{ color: var(--muted); }}
</style></head><body>
<h1>Geometric eval: {verdict}</h1>
<p>p95 {mm(summary["observed_p95_m"])} mm, mean {mm(summary["observed_mean_m"])} mm over
{summary["observed_points"]} points on {summary["observed_faces"]} of the {summary["observed_faces_total"]} seen faces of
{summary["walls"]} walls.
Target: p95 &lt; {PASS_P95_M * 1000:.0f} mm. Faces 10-50 cm off: {summary["faces_gross"]}.
Suspect openings: {summary["openings_suspect"]}.</p>
<p class="muted">Unseen faces (the back of a wall seen from one side, a default thickness behind it): p95
{mm(summary["unobserved_p95_m"])} mm, mean {mm(summary["unobserved_mean_m"])} mm on {summary["unobserved_faces"]} faces.
All {summary["faces_measured"]} faces (the headline until 2026-10-09): p95 {mm(summary["p95_m"])} mm, mean
{mm(summary["mean_m"])} mm.</p>
{_sparkline(history)}
<h2>Plan heatmap</h2>
<p class="muted">Wall points coloured by error: green 0, yellow 2 cm, red &ge; 5 cm. Blue: model walls. Magenta: origin.</p>
<img alt="plan heatmap" src="data:image/png;base64,{png}">
{doubled_html}
{floor_html}
{gross_html}<h2>Wall faces</h2>
<table><tr><th>wall</th><th>face</th><th>points</th><th>mean mm</th><th>p95 mm</th><th>bias mm</th></tr>{face_rows}</table>
<h2>Openings</h2>
<p class="muted">solid fraction: wall surface covering the opening, relative to the wall beside it (0 = real hole, 1 = solid; doors are never suspect, a closed leaf reads as solid).</p>
<table><tr><th>opening</th><th>kind</th><th>wall</th><th>solid fraction</th><th></th></tr>{op_rows}</table>
</body></html>
"""
