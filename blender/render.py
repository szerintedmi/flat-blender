"""Render named cameras of a .blend written by build.py.

Run headless:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python blender/render.py -- \
      --blend data/<project>/build/model.blend --view "still:s01/s000001" --out data/<project>/build/renders \
      [--samples 64] [--res 1920x1080] [--percent 25]

--view takes one camera name, a comma-separated list, or `all`. Output files are named after the
camera ("still:s01/s000001" -> still_s01_s000001.png). Without --res, each camera renders at the
size stored on it by build.py (the still's full resolution, 1920x1080 for views); --percent scales that.
"""

import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blendcommon import cameras, log, render_camera, safe_filename, script_argv, setup_cycles  # noqa: E402


def parse_res(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from e
    return w, h


def main() -> None:
    ap = argparse.ArgumentParser(prog="render.py")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--view", required=True, help="camera name, comma-separated names, or 'all'")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", type=parse_res, help="override resolution, e.g. 1920x1080")
    ap.add_argument("--percent", type=int, default=100, help="resolution percentage")
    args = ap.parse_args(script_argv())

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    scene = bpy.context.scene
    setup_cycles(scene, args.samples)
    for cam in cameras(args.view):
        path = os.path.join(args.out, safe_filename(cam.name) + ".png")
        render_camera(scene, cam, path, res=args.res, percent=args.percent)


if __name__ == "__main__":
    main()
