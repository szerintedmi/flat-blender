"""Framing regressions. Run with Blender -b --factory-startup --python this_file.py."""

import sys
import unittest
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build import Builder, build_plan_camera, scene_bounds  # noqa: E402


class SceneFramingTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.builder = Builder()

    def test_objects_outside_rooms_are_framed(self):
        scene = {
            "rooms": [{"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]], "floor_z": 8}],
            "objects": [{"center": [10, 10, 8.5], "size": [2, 2, 1]}],
            "stairs": [{"center": [-10, -10, 9], "size": [2, 2, 2]}],
        }
        x0, y0, x1, y1 = scene_bounds(scene)
        self.assertLessEqual(x0, -11)
        self.assertLessEqual(y0, -11)
        self.assertGreaterEqual(x1, 11)
        self.assertGreaterEqual(y1, 11)
        self.assertAlmostEqual(build_plan_camera(self.builder, scene)["plan_cut_z"], 9.2)

    def test_floor_fallbacks_follow_scene_elevation(self):
        for scene in (
            {"walls": [{"start": [0, 0], "end": [2, 0], "thickness": 0.2, "base_z": 8}]},
            {"objects": [{"center": [0, 0, 8.5], "size": [1, 1, 1]}]},
            {"stairs": [{"center": [0, 0, 9], "size": [1, 2, 2]}]},
        ):
            with self.subTest(scene=scene):
                self.assertAlmostEqual(build_plan_camera(self.builder, scene)["plan_cut_z"], 9.2)

    def test_empty_scene_has_no_plan(self):
        self.assertIsNone(build_plan_camera(self.builder, {}))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SceneFramingTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
