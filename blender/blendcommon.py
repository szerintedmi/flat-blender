"""Helpers shared by build.py and render.py (Blender's bundled Python: stdlib + bpy only)."""

import os
import re
import sys

import bpy


def script_argv() -> list[str]:
    """Arguments after Blender's own `--` separator."""
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def log(msg: str) -> None:
    print(f"[blender] {msg}", flush=True)


def safe_filename(name: str) -> str:
    """Camera names like "still:s01/s000003" -> "still_s01_s000003"."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "camera"


def setup_cycles(scene: bpy.types.Scene, samples: int, denoise: bool = True) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    scene.cycles.device = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        for dev in prefs.devices:
            dev.use = True
        if gpus:
            scene.cycles.device = "GPU"
    except (KeyError, TypeError, ValueError, AttributeError):
        pass


def cameras(view: str) -> list[bpy.types.Object]:
    """`all`, one camera name, or a comma-separated list of names."""
    cams = sorted((o for o in bpy.data.objects if o.type == "CAMERA"), key=lambda o: o.name)
    if view == "all":
        return cams
    by_name = {o.name: o for o in cams}
    missing = [n for n in view.split(",") if n not in by_name]
    if missing:
        raise SystemExit(f"unknown camera(s) {missing}; available: {sorted(by_name)}")
    return [by_name[n] for n in view.split(",")]


def render_camera(
    scene: bpy.types.Scene,
    cam: bpy.types.Object,
    filepath: str,
    res: tuple[int, int] | None = None,
    percent: int = 100,
) -> str:
    """Render one camera to a PNG.

    Resolution comes from the camera's `width`/`height` custom properties (set by build.py from the
    still's intrinsics), unless `res` overrides it. Collections named in the camera's
    `hide_collections` property (comma-separated, e.g. "Ceilings" for the plan) are hidden for the render.
    """
    width = int(cam.get("width", 1920))
    height = int(cam.get("height", 1080))
    if res is not None:
        if abs(res[0] / res[1] - width / height) > 0.01:
            log(
                f"warning: {cam.name}: --res {res[0]}x{res[1]} has a different aspect than the camera "
                f"({width}x{height}); the field of view along the shorter side will differ"
            )
        width, height = res
    scene.camera = cam
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = percent
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"

    hidden = []
    for name in filter(None, str(cam.get("hide_collections", "")).split(",")):
        coll = bpy.data.collections.get(name)
        if coll is not None and not coll.hide_render:
            coll.hide_render = True
            hidden.append(coll)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        scene.render.filepath = os.path.abspath(filepath)
        bpy.ops.render.render(write_still=True)
    finally:
        for coll in hidden:
            coll.hide_render = False
    log(f"rendered {cam.name} -> {scene.render.filepath}")
    return scene.render.filepath
