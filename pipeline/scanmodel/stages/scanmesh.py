"""The fused LiDAR mesh for Blender: a lighter copy of all of it (a reference layer), and cut-outs of the
places the structured model only approximates (stairs: RoomPlan describes a turning staircase as
separate straight boxes; the scan has the real shape).

Writes into the build folder: scan.ply (simplified to `voxel`), stairs.ply (the stair region, if any).
"""

from pathlib import Path

import numpy as np
import open3d as o3d
from shapely import Polygon, contains_xy
from shapely.ops import unary_union

from ..boxes import Box
from ..models.scene import Scene


def _footprint(
    center: tuple[float, float, float], size: tuple[float, float, float], yaw_deg: float
) -> Polygon:
    box = Box.of(center, size, yaw_deg)
    x, y = box.axes
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    return Polygon([box.centre + sx * box.half["+x"] * x + sy * box.half["+y"] * y for sx, sy in corners])


def _on_walls(scene: Scene, centres: np.ndarray, normals: np.ndarray, tol: float = 0.04) -> np.ndarray:
    """Triangles lying on a wall face (within `tol`, parallel to it): the walls around a staircase are in
    the model already."""
    out = np.zeros(len(centres), bool)
    for w in scene.walls:
        a, d, n = np.array(w.start), w.direction(), w.normal()
        rel = centres[:, :2] - a
        s, t = rel @ d, rel @ n
        along = (s > -0.05) & (s < w.length() + 0.05)
        parallel = np.abs(normals[:, :2] @ n) > 0.9
        for side in (1, -1):
            out |= along & parallel & (np.abs(t - side * w.thickness / 2) < tol)
    return out


def prepare(
    dense: Path,
    scene_path: Path,
    out: Path,
    voxel: float = 0.02,
    margin: float = 0.15,
    min_piece: int = 100,
) -> dict[str, object]:
    mesh = o3d.io.read_triangle_mesh(dense)
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"{dense} has no triangles")
    light = mesh.simplify_vertex_clustering(voxel_size=voxel)
    light.compute_vertex_normals()
    light.compute_triangle_normals()
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out / "scan.ply", light, write_ascii=False)
    summary: dict[str, object] = {"scan_triangles": len(light.triangles)}

    stairs_ply = out / "stairs.ply"
    stairs_ply.unlink(missing_ok=True)
    scene = Scene.model_validate_json(scene_path.read_text())
    if scene.stairs:
        region = unary_union(
            [_footprint(st.center, st.size, st.yaw_deg).buffer(margin) for st in scene.stairs]
        )
        z0 = min(st.center[2] - st.size[2] / 2 for st in scene.stairs) - 0.10
        z1 = (
            max(
                [st.center[2] + st.size[2] / 2 for st in scene.stairs if not st.tread_z]
                + [max(st.tread_z) for st in scene.stairs if st.tread_z]
            )
            + 0.30
        )
        v = np.asarray(light.vertices)
        keep = contains_xy(region, v[:, 0], v[:, 1]) & (v[:, 2] > z0) & (v[:, 2] < z1)
        tri = np.asarray(light.triangles)
        inside = keep[tri].all(axis=1)
        normals = np.asarray(light.triangle_normals)
        inside &= ~_on_walls(scene, v[tri].mean(axis=1), normals)
        # A stair is treads, landings and the flight's underside (horizontal), and risers (vertical,
        # facing along the climb of the flight they are in). Side walls face across it.
        centres = v[tri].mean(axis=1)
        horizontal = np.abs(normals[:, 2]) > 0.8
        riser = np.zeros(len(normals), bool)
        for st in scene.stairs:
            here = contains_xy(
                _footprint(st.center, st.size, st.yaw_deg).buffer(margin), centres[:, 0], centres[:, 1]
            )
            climb = np.array([-np.sin(np.radians(st.yaw_deg)), np.cos(np.radians(st.yaw_deg))])
            riser |= here & (np.abs(normals[:, 2]) < 0.3) & (np.abs(normals[:, :2] @ climb) > 0.9)
        inside &= horizontal | riser
        if inside.any():
            cut = o3d.geometry.TriangleMesh(light)
            cut.remove_triangles_by_mask((~inside).tolist())
            cut.remove_unreferenced_vertices()
            # Speckle: small pieces of noisy wall mesh whose triangles happen to look horizontal.
            ids, counts, _ = cut.cluster_connected_triangles()
            small = np.asarray(counts)[np.asarray(ids)] < min_piece
            cut.remove_triangles_by_mask(small.tolist())
            cut.remove_unreferenced_vertices()
            o3d.io.write_triangle_mesh(stairs_ply, cut, write_ascii=False)
            summary["stairs_triangles"] = int(inside.sum())
    return summary
