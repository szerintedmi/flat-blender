"""Export the structure of a .blend written by build.py as glTF binary, for the web viewer.

Run headless:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python blender/export.py -- \
      --blend data/<project>/build/model.blend --out data/<project>/export/model.glb

Walls, openings, floors, ceilings, objects and stairs (the scanned stairs among them) only: cameras,
stills, lights, the scan layer and RoomPlan's stair boxes stay out. glTF is
y up; the exporter turns the scene frame's z up into it, which is what the viewer expects (glbUp=y).
"""

import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blendcommon import log, script_argv  # noqa: E402

KEEP = ("Walls", "Openings", "Floors", "Ceilings", "Objects", "Stairs")


def main() -> None:
    ap = argparse.ArgumentParser(prog="export.py")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--out", required=True, help="output .glb")
    args = ap.parse_args(script_argv())

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    # Not select_all(): it skips hidden collections, and build.py saves the hidden scan still selected.
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    n = 0
    for name in KEEP:
        coll = bpy.data.collections.get(name)
        for ob in coll.all_objects if coll else []:
            if ob.type == "MESH":
                ob.select_set(True)
                n += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=os.path.abspath(args.out),
        export_format="GLB",
        use_selection=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
        export_apply=True,
    )
    log(f"wrote {args.out}: {n} meshes")


if __name__ == "__main__":
    main()
