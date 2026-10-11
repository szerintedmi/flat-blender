"""Build a .blend from a project's scene/model.json (pipeline stage 8, "build"; `scanmodel build` runs it).

Run headless:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python blender/build.py -- \
      --scene data/<project>/scene/model.json --out data/<project>/build/model.blend \
      [--stills data/<project>/derived/refine/stills.jsonl] [--render-preview data/<project>/build/preview.png]

Reads the JSON directly (Blender's Python has no pydantic). The contract is
pipeline/scanmodel/models/scene.py and pipeline/scanmodel/models/derived.py (PosedImage):
- scene frame: metres, +z up, +x east, +y north, finished floor at z = 0; all z values are absolute;
- Wall: centreline start -> end, thickness, base_z, height (at the start; `height_end` if the top
  slopes). Openings sit on a wall at `offset` (wall start -> opening centre along the centreline),
  `bottom` above the wall's base_z;
- Room: floor at floor_z; ceiling at ceiling_z, or on `ceiling_plane` (z = a + b x + c y) if it slopes;
- Object / Stair: oriented box (center, size along local x/y/z, yaw_deg about +z); stairs climb +y;
- View.camera_to_world: Blender/OpenGL camera axes. PosedImage.camera_to_world: OpenCV axes,
  converted here by right-multiplying with diag(1, -1, -1, 1).

Every generated object carries custom properties: element_id, element_kind, source_kind and
model_json (the element as read), so later stages and overrides can find them. Object names are the
element ids (walls, openings, objects, stairs) or derived from them ("<room>.floor", "still:<id>").
"""

import argparse
import json
import math
import os
import sys
import zlib

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blendcommon import log, render_camera, script_argv, setup_cycles  # noqa: E402

EPS = 1e-4  # metres: anything thinner than this is not built

COLLECTIONS = (
    "Walls", "Openings", "Floors", "Ceilings", "Objects", "Stairs", "Stair boxes", "Cameras", "Stills", "Lights",
    "Scan",
)  # fmt: skip
# Hidden in the viewport (eye icon in the Outliner): one camera per still clutters the view; the scan is a
# reference to compare against; stair boxes are RoomPlan's, replaced by the scanned stairs when there are any.
HIDDEN = ("Stills", "Scan", "Stair boxes")

FRAME_W = 0.05  # door/window casing width
FRAME_PROUD = 0.01  # casing stands this far proud of each wall face
LEAF_T = 0.04  # door leaf thickness
GLASS_T = 0.01
STEP_RISE = 0.18  # nominal rise when a stair has no fitted treads
PLAN_CUT = 1.2  # plan camera cuts the model this far above the lowest floor
PREVIEW_LONG_SIDE = 960

# name -> (base colour (linear RGB), roughness)
SURFACES = {
    "wall": ((0.80, 0.79, 0.75), 0.85),
    "floor": ((0.36, 0.22, 0.12), 0.45),
    "ceiling": ((0.88, 0.88, 0.86), 0.9),
    "frame": ((0.92, 0.92, 0.90), 0.4),
    "door": ((0.60, 0.52, 0.42), 0.5),
    "stair": ((0.50, 0.34, 0.20), 0.5),
}

# RoomPlan object categories -> colour; anything else gets a stable colour from its name.
CATEGORY_COLOURS = {
    "storage": (0.55, 0.42, 0.30),
    "refrigerator": (0.85, 0.86, 0.88),
    "stove": (0.20, 0.20, 0.22),
    "oven": (0.20, 0.20, 0.22),
    "dishwasher": (0.75, 0.76, 0.78),
    "washerDryer": (0.80, 0.80, 0.82),
    "sink": (0.70, 0.72, 0.75),
    "bathtub": (0.92, 0.92, 0.92),
    "toilet": (0.92, 0.92, 0.92),
    "bed": (0.55, 0.60, 0.70),
    "sofa": (0.25, 0.35, 0.45),
    "chair": (0.60, 0.35, 0.20),
    "table": (0.45, 0.30, 0.18),
    "television": (0.05, 0.05, 0.06),
    "fireplace": (0.45, 0.25, 0.20),
}


# ------------------------------------------------------------------------------------------------
# scene helpers
# ------------------------------------------------------------------------------------------------
class Builder:
    def __init__(self) -> None:
        self.collections: dict[str, bpy.types.Collection] = {}
        self.materials: dict[str, bpy.types.Material] = {}
        for name in COLLECTIONS:
            coll = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(coll)
            self.collections[name] = coll
            if name in HIDDEN:  # out of the viewport (eye icon) and out of renders (camera icon)
                bpy.context.view_layer.layer_collection.children[name].hide_viewport = True
                coll.hide_render = True

    # -- materials ---------------------------------------------------------------------------------
    def material(self, name: str, colour, roughness: float = 0.5, glass: bool = False) -> bpy.types.Material:
        if name in self.materials:
            return self.materials[name]
        mat = bpy.data.materials.new(name)  # Blender 5.x: always node-based, starts with a Principled BSDF
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            out = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        mat.diffuse_color = (*colour, 0.3 if glass else 1.0)  # viewport colour
        if glass:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
            bsdf.inputs["IOR"].default_value = 1.45
        else:
            # Backfaces render flat black. All meshes are closed, so this only shows where a camera
            # clips into solids (the plan's section cut: walls appear as black poché) and hides light
            # leaking through coplanar faces where boxes overlap (wall corners, T-junctions).
            out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
            geo = nodes.new("ShaderNodeNewGeometry")
            black = nodes.new("ShaderNodeEmission")
            black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            mix = nodes.new("ShaderNodeMixShader")
            links = mat.node_tree.links
            links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
            links.new(bsdf.outputs["BSDF"], mix.inputs[1])
            links.new(black.outputs["Emission"], mix.inputs[2])
            links.new(mix.outputs["Shader"], out.inputs["Surface"])
        self.materials[name] = mat
        return mat

    def surface(self, kind: str) -> bpy.types.Material:
        colour, rough = SURFACES[kind]
        return self.material(kind, colour, rough)

    def glass(self) -> bpy.types.Material:
        return self.material("glass", (0.9, 0.95, 1.0), 0.0, glass=True)

    def category(self, category: str) -> bpy.types.Material:
        colour = CATEGORY_COLOURS.get(category)
        if colour is None:  # stable pastel from the name (not Python's salted hash())
            h = zlib.crc32(category.encode())
            colour = tuple(0.25 + 0.5 * ((h >> s) & 0xFF) / 255 for s in (0, 8, 16))
        return self.material(f"object.{category}", colour, 0.6)

    # -- objects -----------------------------------------------------------------------------------
    def mesh_object(self, name: str, bm: bmesh.types.BMesh, collection: str, mats) -> bpy.types.Object:
        me = bpy.data.meshes.new(name)
        bm.normal_update()
        bm.to_mesh(me)
        bm.free()
        for m in mats:
            me.materials.append(m)
        return self.link(bpy.data.objects.new(name, me), collection, name)

    def link(self, ob: bpy.types.Object, collection: str, wanted: str) -> bpy.types.Object:
        self.collections[collection].objects.link(ob)
        if ob.name != wanted:
            log(f"warning: object {wanted!r} was named {ob.name!r} (duplicate id or name too long)")
        return ob


def tag(ob: bpy.types.Object, element: dict | None, kind: str, element_id: str | None = None) -> None:
    ob["element_id"] = element_id if element_id is not None else (element or {}).get("id", ob.name)
    ob["element_kind"] = kind
    source = (element or {}).get("source") or {}
    if "kind" in source:
        ob["source_kind"] = source["kind"]
    if element is not None:
        ob["model_json"] = json.dumps(element, sort_keys=True)


def add_box(bm: bmesh.types.BMesh, x0, x1, y0, y1, z0, z1, mat_index: int = 0) -> None:
    """Axis-aligned box (in the mesh's local frame) appended to a bmesh, outward normals."""
    if x1 - x0 < EPS or y1 - y0 < EPS or z1 - z0 < EPS:
        return
    v = [
        bm.verts.new(p)
        for p in (
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        )
    ]  # fmt: skip
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in f]).material_index = mat_index


def add_sloped_box(bm: bmesh.types.BMesh, x0, x1, y0, y1, z0, top0, top1, mat_index: int = 0) -> None:
    """Like add_box, but the top is at top0 over x0 and top1 over x1 (a wall under a sloped ceiling)."""
    if x1 - x0 < EPS or y1 - y0 < EPS or min(top0, top1) - z0 < EPS:
        return
    v = [
        bm.verts.new(p)
        for p in (
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, top0), (x1, y0, top1), (x1, y1, top1), (x0, y1, top0),
        )
    ]  # fmt: skip
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([v[i] for i in f]).material_index = mat_index


def yaw_matrix(x: float, y: float, z: float, yaw_rad: float) -> Matrix:
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(yaw_rad, 4, "Z")


# ------------------------------------------------------------------------------------------------
# walls and openings
# ------------------------------------------------------------------------------------------------
def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def wall_pieces(
    u_min: float, u_max: float, height: float, holes: list[tuple[float, float, float, float]]
) -> list[tuple[float, float, float, float]]:
    """Solid boxes (u0, u1, z0, z1) of a wall from u_min..u_max, 0..height minus rectangular holes.

    The wall is cut at every hole edge; in each slice the solid part is the complement of the union
    of the holes covering it (so overlapping openings merge), giving full-height piers between
    openings plus a sill and lintel piece per opening. Neighbouring slices with the same profile
    are merged.
    """
    cuts = sorted({u_min, u_max, *(u for h in holes for u in h[:2] if u_min < u < u_max)})
    slices: list[tuple[float, float, list[tuple[float, float]]]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < EPS:
            continue
        mid = 0.5 * (a + b)
        gaps = merge_intervals([(h[2], h[3]) for h in holes if h[0] < mid < h[1]])
        solid, z = [], 0.0
        for g0, g1 in gaps:
            if g0 - z > EPS:
                solid.append((z, g0))
            z = max(z, g1)
        if height - z > EPS:
            solid.append((z, height))
        key = [(round(s0, 5), round(s1, 5)) for s0, s1 in solid]
        if slices and slices[-1][2] == key and abs(slices[-1][1] - a) < EPS:
            slices[-1] = (slices[-1][0], b, key)
        else:
            slices.append((a, b, key))
    return [(a, b, z0, z1) for a, b, key in slices for z0, z1 in key]


def corner_extensions(walls: list[dict]) -> dict[tuple[str, int], float]:
    """At an L-corner (two walls' centreline endpoints coincide) one wall is extended by half the
    other's thickness so the outside corner is closed. The wall whose id sorts first gets the extension,
    the other stops at the shared centreline point (inside the first): no overlapping faces."""
    ext: dict[tuple[str, int], float] = {}
    ends = [(w, i, Vector((*(w["start"] if i == 0 else w["end"]), 0.0))) for w in walls for i in (0, 1)]
    for a in range(len(ends)):
        wa, ia, pa = ends[a]
        for b in range(a + 1, len(ends)):
            wb, ib, pb = ends[b]
            if wa["id"] == wb["id"] or wa.get("level") != wb.get("level") or (pa - pb).length > 0.01:
                continue
            first, i_first, other = (wa, ia, wb) if wa["id"] < wb["id"] else (wb, ib, wa)
            key = (first["id"], i_first)
            ext[key] = max(ext.get(key, 0.0), 0.5 * other["thickness"])
    return ext


def clamp_opening(op: dict, length: float, height: float) -> tuple[float, float, float, float] | None:
    """Opening rectangle (u0, u1, z0, z1) in wall coordinates, clamped to the wall."""
    u0 = max(0.0, op["offset"] - 0.5 * op["width"])
    u1 = min(length, op["offset"] + 0.5 * op["width"])
    z0 = max(0.0, op["bottom"])
    z1 = min(height, op["bottom"] + op["height"])
    if u1 - u0 < 0.01 or z1 - z0 < 0.01:
        log(f"warning: opening {op['id']} lies outside wall {op['wall']}; skipped")
        return None
    if (u0, u1, z0, z1) != (
        op["offset"] - 0.5 * op["width"],
        op["offset"] + 0.5 * op["width"],
        op["bottom"],
        op["bottom"] + op["height"],
    ):
        log(f"warning: opening {op['id']} clamped to wall {op['wall']}")
    return u0, u1, z0, z1


def build_walls(b: Builder, scene: dict) -> None:
    walls = scene.get("walls", [])
    ext = corner_extensions(walls)
    by_wall: dict[str, list[dict]] = {}
    for op in scene.get("openings", []):
        by_wall.setdefault(op["wall"], []).append(op)
    known = {w["id"] for w in walls}
    for op in scene.get("openings", []):
        if op["wall"] not in known:
            log(f"warning: opening {op['id']} refers to unknown wall {op['wall']}; skipped")

    for w in walls:
        sx, sy = w["start"]
        ex, ey = w["end"]
        length = math.hypot(ex - sx, ey - sy)
        if length < EPS:
            log(f"warning: wall {w['id']} has zero length; skipped")
            continue
        yaw = math.atan2(ey - sy, ex - sx)
        t, h0 = w["thickness"], w["height"]
        h1 = w.get("height_end") or h0  # a sloped top runs linearly from start to end
        h = max(h0, h1)
        # local frame: origin at the wall start on its base, +x along the wall, +y = left normal
        frame = yaw_matrix(sx, sy, w.get("base_z", 0.0), yaw)

        holes = []
        for op in by_wall.get(w["id"], []):
            rect = clamp_opening(op, length, h)
            if rect is None:
                continue
            holes.append(rect)
            build_opening(b, op, w, rect, frame)

        bm = bmesh.new()
        e0, e1 = ext.get((w["id"], 0), 0.0), ext.get((w["id"], 1), 0.0)
        for u0, u1, z0, z1 in wall_pieces(-e0, length + e1, h, holes):
            if abs(z1 - h) < EPS and abs(h1 - h0) > EPS:  # reaches the top: follow the slope

                def top(u: float) -> float:
                    return h0 + (h1 - h0) * min(max(u / length, 0.0), 1.0)

                add_sloped_box(bm, u0, u1, -t / 2, t / 2, z0, top(u0), top(u1))
            else:
                add_box(bm, u0, u1, -t / 2, t / 2, z0, z1)
        ob = b.mesh_object(w["id"], bm, "Walls", [b.surface("wall")])
        ob.matrix_world = frame
        tag(ob, w, "wall")
        ob["length"] = length
        ob["extension_start"] = e0
        ob["extension_end"] = e1


def build_opening(b: Builder, op: dict, wall: dict, rect, wall_frame: Matrix) -> None:
    """Placeholder: casing around the hole (+ glass for windows, a leaf for doors).

    Local frame: origin at the bottom centre of the (clamped) opening on the wall centreline,
    +x along the wall, +y = the wall's left normal, +z up.
    """
    u0, u1, z0, z1 = rect
    w, h = u1 - u0, z1 - z0
    d = 0.5 * wall["thickness"] + FRAME_PROUD
    fw = min(FRAME_W, 0.25 * w, 0.25 * h)
    kind = op["kind"]

    bm = bmesh.new()
    add_box(bm, -w / 2, -w / 2 + fw, -d, d, 0, h)  # left jamb
    add_box(bm, w / 2 - fw, w / 2, -d, d, 0, h)  # right jamb
    add_box(bm, -w / 2 + fw, w / 2 - fw, -d, d, h - fw, h)  # head
    if kind == "window":
        add_box(bm, -w / 2 + fw, w / 2 - fw, -d, d, 0, fw)  # sill
        add_box(bm, -w / 2 + fw, w / 2 - fw, -GLASS_T / 2, GLASS_T / 2, fw, h - fw, mat_index=1)
    mats = [b.surface("frame")] + ([b.glass()] if kind == "window" else [])
    ob = b.mesh_object(op["id"], bm, "Openings", mats)
    ob.matrix_world = wall_frame @ Matrix.Translation((0.5 * (u0 + u1), 0.0, z0))
    tag(ob, op, "opening")
    ob["opening_kind"] = kind
    ob["clamped_width"] = w
    ob["clamped_height"] = h

    if kind == "door":
        leaf_w = w - 2 * fw
        bm = bmesh.new()
        add_box(bm, 0, leaf_w, -LEAF_T / 2, LEAF_T / 2, 0, h - fw)
        leaf = b.mesh_object(f"{op['id']}.leaf", bm, "Openings", [b.surface("door")])
        leaf.parent = ob
        # hinge on the start-side jamb; an open door swings 90 degrees towards the wall's left side
        leaf.matrix_parent_inverse = Matrix.Identity(4)
        leaf.location = (-w / 2 + fw, 0.0, 0.0)
        leaf.rotation_euler = (0.0, 0.0, math.radians(90.0) if op.get("is_open") else 0.0)
        tag(leaf, None, "door_leaf", op["id"])


# ------------------------------------------------------------------------------------------------
# floors, ceilings, objects, stairs
# ------------------------------------------------------------------------------------------------
def signed_area(poly: list[tuple[float, float]]) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]))


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain, counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, c):
        return (a[0] - o[0]) * (c[1] - o[1]) - (a[1] - o[1]) * (c[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


SLAB_T = 0.05  # floors and ceilings are slabs this thick (on the side away from the room)


def polygon_object(b: Builder, name: str, poly, z, up: bool, collection: str, mat) -> bpy.types.Object:
    """A slab under a floor (up) or over a ceiling: its room-side face at height z, or z(x, y) if sloped.

    Closed like every other mesh here, so seen from outside (above a ceiling, below a floor) it shows
    its outer face, not a back face (back faces render black: the plan's section fill).
    """
    height = z if callable(z) else (lambda x, y: z)
    away = -SLAB_T if up else SLAB_T
    pts = [tuple(p) for p in poly]
    if signed_area(pts) < 0:
        pts.reverse()
    bm = bmesh.new()
    near = [bm.verts.new((x, y, height(x, y))) for x, y in pts]
    far = [bm.verts.new((x, y, height(x, y) + away)) for x, y in pts]
    bm.faces.new(near)  # n-gon; Blender tessellates concave ones
    bm.faces.new(list(reversed(far)))
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        bm.faces.new([near[j], near[i], far[i], far[j]])
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))  # closed: normals point out
    return b.mesh_object(name, bm, collection, [mat])


def build_rooms(b: Builder, scene: dict) -> None:
    rooms = scene.get("rooms", [])
    for room in rooms:
        fl = polygon_object(b, f"{room['id']}.floor", room["polygon"], room.get("floor_z", 0.0), True,
                            "Floors", b.surface("floor"))  # fmt: skip
        tag(fl, room, "floor")
        plane = room.get("ceiling_plane")
        z = (lambda x, y, a=plane: a[0] + a[1] * x + a[2] * y) if plane else room["ceiling_z"]
        ce = polygon_object(b, f"{room['id']}.ceiling", room["polygon"], z, False, "Ceilings",
                            b.surface("ceiling"))  # fmt: skip
        tag(ce, room, "ceiling")
    if not rooms and scene.get("walls"):
        pts = [tuple(w[k]) for w in scene["walls"] for k in ("start", "end")]
        hull = convex_hull(pts)
        if len(hull) >= 3:
            # 1 mm low: this floor runs under the walls, and must not be coplanar with their bottoms
            z = min(w.get("base_z", 0.0) for w in scene["walls"]) - 0.001
            ob = polygon_object(b, "floor.hull", hull, z, True, "Floors", b.surface("floor"))
            tag(ob, None, "floor", "floor.hull")
            log("no rooms in the scene: built a convex-hull floor from the wall endpoints")


def build_objects(b: Builder, scene: dict) -> None:
    for o in scene.get("objects", []):
        sx, sy, sz = o["size"]
        bm = bmesh.new()
        add_box(bm, -sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2)
        ob = b.mesh_object(o["id"], bm, "Objects", [b.category(o["category"])])
        ob.matrix_world = yaw_matrix(*o["center"], math.radians(o.get("yaw_deg", 0.0)))
        tag(ob, o, "object")
        ob["category"] = o["category"]


def stair_profile(sy: float, sz: float, cz: float, tread_z: list[float]) -> list[tuple[float, float]]:
    """Closed (y, z) outline of a solid stair in its local frame (centred box, climbing +y)."""
    y0, zb = -sy / 2, -sz / 2
    tops = [min(max(t - cz, zb + EPS), sz / 2) for t in tread_z]  # tread_z is absolute (scene frame)
    if not tops:
        n = max(1, round(sz / STEP_RISE))
        tops = [zb + sz * (i + 1) / n for i in range(n)]
    run = sy / len(tops)
    pts = [(y0, zb), (sy / 2, zb)]
    for i in reversed(range(len(tops))):  # down the stair from the top back edge
        pts += [(y0 + (i + 1) * run, tops[i]), (y0 + i * run, tops[i])]
    return pts


def build_stairs(b: Builder, scene: dict, collection: str = "Stairs") -> None:
    for s in scene.get("stairs", []):
        sx, sy, sz = s["size"]
        prof = stair_profile(sy, sz, s["center"][2], s.get("tread_z", []))
        bm = bmesh.new()
        left = [bm.verts.new((-sx / 2, y, z)) for y, z in prof]
        right = [bm.verts.new((sx / 2, y, z)) for y, z in prof]
        bm.faces.new(left)
        bm.faces.new(list(reversed(right)))
        n = len(prof)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((left[i], right[i], right[j], left[j]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        ob = b.mesh_object(s["id"], bm, collection, [b.surface("stair")])
        ob.matrix_world = yaw_matrix(*s["center"], math.radians(s.get("yaw_deg", 0.0)))
        tag(ob, s, "stair")
        ob["steps"] = (len(prof) - 2) // 2


def import_scan(b: Builder, path: str, name: str, collection: str) -> bpy.types.Object:
    """A fused-mesh .ply (pipeline scene frame, vertex colours from the video) as one object."""
    bpy.ops.wm.ply_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    b.collections[collection].objects.link(ob)
    ob.name = ob.data.name = name
    mat = bpy.data.materials.get("scan") or bpy.data.materials.new("scan")
    if not mat.node_tree or "Color Attribute" not in mat.node_tree.nodes:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        colour = nodes.new("ShaderNodeVertexColor")  # the mesh's (only) colour attribute
        mat.node_tree.links.new(colour.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
        nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    tag(ob, None, "scan", name)
    return ob


# ------------------------------------------------------------------------------------------------
# cameras and lights
# ------------------------------------------------------------------------------------------------
OPENCV_TO_BLENDER = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def new_camera(b: Builder, name: str, collection: str = "Cameras") -> bpy.types.Object:
    cd = bpy.data.cameras.new(name)
    cd.clip_start = 0.05
    cd.clip_end = 200.0
    return b.link(bpy.data.objects.new(name, cd), collection, name)


def apply_intrinsics(cam: bpy.types.Object, k: dict) -> None:
    """Pinhole intrinsics (pixels, OpenCV convention) -> Blender lens, sensor fit and shift.

    The sensor is fitted to the larger image side S; lens = f * sensor / S and shifts are in units of S.
    A principal point right of / below the centre moves the frame left / down.
    """
    cd = cam.data
    w, h = k["width"], k["height"]
    cd.sensor_width = cd.sensor_height = 36.0
    if w >= h:
        cd.sensor_fit, s, f = "HORIZONTAL", w, k["fx"]
    else:
        cd.sensor_fit, s, f = "VERTICAL", h, k["fy"]
    cd.lens = f * 36.0 / s
    cd.shift_x = (0.5 * w - k["cx"]) / s
    cd.shift_y = (k["cy"] - 0.5 * h) / s
    if abs(k["fx"] - k["fy"]) > 0.01 * k["fx"]:
        log(f"warning: {cam.name}: fx != fy ({k['fx']:.1f} vs {k['fy']:.1f}); using square pixels")
    cam["width"], cam["height"] = w, h
    for key in ("fx", "fy", "cx", "cy"):
        cam[key] = k[key]


def read_stills(path: str) -> list[dict]:
    stills = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                stills.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: {e}") from e
    return stills


def build_still_cameras(b: Builder, stills: list[dict]) -> None:
    for st in stills:
        cam = new_camera(b, f"still:{st['id']}", "Stills")
        cam.data.display_size = 0.1
        cam.matrix_world = Matrix(st["camera_to_world"]) @ OPENCV_TO_BLENDER
        apply_intrinsics(cam, st["intrinsics"])
        tag(cam, None, "still", st["id"])
        cam["model_json"] = json.dumps(st, sort_keys=True)
        for key in ("session", "file", "kind"):
            if st.get(key) is not None:
                cam[key] = st[key]


def find_still(stills: list[dict], ref: str | None) -> dict | None:
    """View.still is "<session>/<still id>"; PosedImage ids are "<session>/s<still id>". Accept both."""
    if not ref:
        return None
    session, _, sid = ref.partition("/")
    wanted = {ref, f"{session}/s{sid}"}
    return next((s for s in stills if s["id"] in wanted), None)


def build_view_cameras(b: Builder, scene: dict, stills: list[dict]) -> list[bpy.types.Object]:
    out = []
    for v in scene.get("views", []):
        still = find_still(stills, v.get("still"))
        if v.get("camera_to_world") is not None:
            mw = Matrix(v["camera_to_world"])  # already Blender/OpenGL camera axes
        elif v.get("position") is not None and v.get("look_at") is not None:
            pos = Vector(v["position"])
            look = Vector(v["look_at"]) - pos
            if look.length < EPS:
                log(f"warning: view {v['id']}: look_at == position; skipped")
                continue
            # up hint: +z, unless looking straight up/down
            up = "Y" if abs(look.normalized().z) < 0.999 else "X"
            mw = Matrix.Translation(pos) @ look.to_track_quat("-Z", up).to_matrix().to_4x4()
        elif still is not None:
            mw = Matrix(still["camera_to_world"]) @ OPENCV_TO_BLENDER
        else:
            log(f"warning: view {v['id']} has no pose (camera_to_world, position + look_at or a still); skipped")
            continue
        cam = new_camera(b, f"view:{v['id']}")
        cam.matrix_world = mw
        if still is not None:
            apply_intrinsics(cam, still["intrinsics"])
        else:
            cam.data.sensor_fit = "AUTO"  # fov_deg spans the longer image side
            cam.data.angle = math.radians(v.get("fov_deg", 70.0))
            cam["width"], cam["height"] = 1920, 1080
        tag(cam, v, "view")
        if v.get("name"):
            cam["label"] = v["name"]
        out.append(cam)
    return out


def scene_bounds(scene: dict) -> tuple[float, float, float, float] | None:
    pts: list[tuple[float, float]] = []
    for w in scene.get("walls", []):
        r = 0.5 * w["thickness"]
        for x, y in (w["start"], w["end"]):
            pts += [(x - r, y - r), (x + r, y + r)]
    for room in scene.get("rooms", []):
        pts += [tuple(p) for p in room["polygon"]]
    for o in scene.get("objects", []) + scene.get("stairs", []):
        r = 0.5 * math.hypot(o["size"][0], o["size"][1])
        pts += [(o["center"][0] - r, o["center"][1] - r), (o["center"][0] + r, o["center"][1] + r)]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def build_plan_camera(b: Builder, scene: dict) -> bpy.types.Object | None:
    """Top-down orthographic camera framing everything, cutting the model PLAN_CUT above the floor."""
    bounds = scene_bounds(scene)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    margin = 0.05 * max(x1 - x0, y1 - y0) + 0.2
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    # The frame's origin need not be this scene's floor (e.g. an upper storey or an object-only scene).
    floors = [r.get("floor_z", 0.0) for r in scene.get("rooms", [])]
    if not floors:
        floors = [w.get("base_z", 0.0) for w in scene.get("walls", [])]
    if not floors:
        floors = [o["center"][2] - o["size"][2] / 2 for o in scene.get("objects", []) + scene.get("stairs", [])]
    floor = min(floors, default=0.0)
    cut = floor + PLAN_CUT
    cam = new_camera(b, "plan")
    cam.data.type = "ORTHO"
    cam.data.sensor_fit = "AUTO"
    cam.data.ortho_scale = max(x1 - x0, y1 - y0)
    cam.data.clip_start = 0.001  # everything above the cut is behind the camera
    cam.data.clip_end = cut - floor + 50.0
    cam.location = (0.5 * (x0 + x1), 0.5 * (y0 + y1), cut)
    wx, wy = x1 - x0, y1 - y0
    if wx >= wy:
        cam["width"], cam["height"] = PREVIEW_LONG_SIDE, max(16, round(PREVIEW_LONG_SIDE * wy / wx))
    else:
        cam["width"], cam["height"] = max(16, round(PREVIEW_LONG_SIDE * wx / wy)), PREVIEW_LONG_SIDE
    cam["hide_collections"] = "Ceilings"
    cam["plan_cut_z"] = cut
    tag(cam, None, "plan", "plan")
    return cam


def build_lights(b: Builder, scene: dict) -> None:
    world = bpy.data.worlds.new("scene.world")
    bpy.context.scene.world = world
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.75, 0.80, 0.90, 1.0)
        bg.inputs["Strength"].default_value = 0.6

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(2.0)
    sun = b.link(bpy.data.objects.new("sun", sun_data), "Lights", "sun")
    sun.rotation_euler = (math.radians(25.0), math.radians(10.0), math.radians(30.0))
    tag(sun, None, "light", "sun")

    # One soft ceiling panel per room, so interior views with ceilings on aren't black.
    for room in scene.get("rooms", []):
        xs = [p[0] for p in room["polygon"]]
        ys = [p[1] for p in room["polygon"]]
        sx, sy = max(xs) - min(xs), max(ys) - min(ys)
        ld = bpy.data.lights.new(f"light:{room['id']}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = 0.5 * sx, 0.5 * sy
        ld.energy = 10.0 * sx * sy  # ~10 W per m2 of floor
        ld.color = (1.0, 0.92, 0.82)
        ob = b.link(bpy.data.objects.new(f"light:{room['id']}", ld), "Lights", f"light:{room['id']}")
        # area lights shine down their local -z: no rotation needed
        ob.location = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), room["ceiling_z"] - 0.02)
        tag(ob, None, "light", room["id"])


# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="build.py", description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True, help="data/<project>/scene/model.json")
    ap.add_argument("--out", required=True, help="output .blend")
    ap.add_argument("--stills", help="data/<project>/derived/refine/stills.jsonl (PosedImage lines)")
    ap.add_argument("--scan", help="fused mesh (.ply) for the hidden Scan layer")
    ap.add_argument("--stairs-mesh", help="the stairs cut out of the fused mesh (.ply): replaces the stair boxes")
    ap.add_argument("--render-preview", metavar="PNG", help="render the plan camera to this PNG")
    ap.add_argument("--preview-samples", type=int, default=16)
    args = ap.parse_args(script_argv())

    with open(args.scene, encoding="utf-8") as f:
        scene = json.load(f)
    if scene.get("format") not in (None, "scan-model") or scene.get("version", 1) != 1:
        raise SystemExit(f"{args.scene}: expected format scan-model v1")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    b = Builder()
    build_walls(b, scene)
    build_rooms(b, scene)
    build_objects(b, scene)
    build_stairs(b, scene, "Stair boxes" if args.stairs_mesh else "Stairs")
    if args.stairs_mesh:
        import_scan(b, args.stairs_mesh, "stairs.scan", "Stairs")
    if args.scan:
        import_scan(b, args.scan, "scan", "Scan")
    build_lights(b, scene)
    stills = read_stills(args.stills) if args.stills else []
    build_still_cameras(b, stills)
    views = build_view_cameras(b, scene, stills)
    plan = build_plan_camera(b, scene)

    bl_scene = bpy.context.scene
    bl_scene.unit_settings.system = "METRIC"
    setup_cycles(bl_scene, args.preview_samples)
    bl_scene.camera = views[0] if views else plan
    if bl_scene.camera is not None:
        bl_scene.render.resolution_x = int(bl_scene.camera.get("width", 1920))
        bl_scene.render.resolution_y = int(bl_scene.camera.get("height", 1080))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0  # the .blend is rebuildable: no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    counts = {name: len(b.collections[name].objects) for name in COLLECTIONS}
    log(f"wrote {args.out}: {counts}")

    if args.render_preview:
        if plan is None:
            raise SystemExit("nothing to frame: the scene has no walls, rooms, objects or stairs")
        render_camera(bl_scene, plan, args.render_preview)


if __name__ == "__main__":
    main()
