# Viewer (phase 0)

Minimal web viewer: a Gaussian splat ([Spark](https://github.com/sparkjsdev/spark) 2.x) plus an optional
glTF of the structure, in the pipeline's scene frame. Vite + three.js + TypeScript.

```sh
just export      # from the repo root: the model and every trained splat into public/data
just viewer      # bun install + dev server, http://localhost:5173 (or the next free port)
```

Or in this folder: `bun install`, `bun run dev`, `bun run build` (type check + static build in `dist/`).

## Data

`just export --project NAME` writes `public/data/` (gitignored), one folder per scan: `scans/<scan>/`
(the selected project name, unless `--scan` overrides it) holds `model.glb` (the Blender model), every trained splat
(`data/<project>/splat/runs/<run>/*.ply`, linked as `splats/<id>.ply`) with `splats.json` describing
them (trainer, steps, resolution, splat count, PSNR/SSIM on held-out views, training time and date, and
whether it was trained on the current dataset), and `views.json`, the project's named views
(`data/<project>/scene/views.json`). `scans.json` lists the scans and the default one. Each project exports its own scan
and leaves the others' folders alone:

```sh
just export --trainer my-trained-splat            # a trainer id belonging to the selected project
just export --project another-scan       # adds the other project's scan
just export --viewer-default                     # this project's scan opens without ?scan=
```

Scans have their own scene frames (each project's origin door and yaw), so positions, poses and named views
mean something in one scan only. Without `scans.json` the viewer reads `data/splats.json` and `data/model.glb`;
the old single-splat exports (`data/flat.spz`, `data/flat.ply`) are no longer read: re-export with `just export`.

| Query param | Default | |
|---|---|---|
| `scan` | `scans.json`'s default | a scan id from `scans.json`; the layers panel's Scan list switches (and drops the camera, splat and view, which belong to the old scan's frame) |
| `view` | — | start at a named view of the scan (the panel's Named view list sets it) |
| `splat` | `splats.json`'s default (`just export --trainer`) | an id from the scan's `splats.json` (`msplat`, `brush`, …), any splat URL (`.spz`, `.ply`, `.splat`, `.ksplat`, `.sog`), or `none` |
| `glb` | the scan's `model.glb` | glTF binary; `none` to skip |
| `model` | on only when no splat is shown | `1` / `0`: show the Blender model at start |
| `splatUp` | `z` | up axis of the splat file |
| `glbUp` | `y` | up axis of the glTF file |
| `cam`, `look` | frame everything | camera position and target, scene-frame `x,y,z` |
| `pose` | — | exact camera: 16 numbers, row-major camera-to-world with OpenGL camera axes in the scene frame (a nerfstudio `transform_matrix`); the camera stays put |
| `fov` | `70` | vertical field of view in degrees |

For scripted renders (comparing splats on the same views as the photos), `window.setPose(m)` moves the
camera to a `pose` without reloading the splat.

Examples:

```
http://localhost:5173/?splat=brush
http://localhost:5173/?splat=none&model=1&cam=1.3,0.6,1.5&look=4.2,3.8,1.0
http://localhost:5173/?scan=my-scan&view=my-view
```

URLs are fetched by the browser, so they must be served by the dev server (`public/`) or allow CORS.

## Layers panel

Top right (P collapses it). The scan (reloads the page on the other scan) and its named views (jumps
there, walkable; a still's slight roll is dropped, `?pose=` keeps it). One splat at a time: pick it in the list or with 1–9 (0: none); its training
parameters and metrics show below, with a warning if it was trained on an older dataset. The Blender
model is an overlay (M): on its own it shows the structure; over a splat, lower its opacity or switch to
wireframe (X) to compare the two.

## Copy view

C (or the panel's **Copy view** button) copies a URL of the current view: the page's parameters with the
scan, the shown splat, `model`, `fov` and the exact camera as `pose`; the toast confirms, and the URL is also in the
console. Opened, it shows the same view with the camera fixed. Shift+C (shift+click) copies a walkable
`cam`/`look` URL instead (same view: walking never rolls the camera). `just cli view add NAME "<url>"` takes
either, to evaluate splats on that view.

## Axes

The pipeline's scene frame is metres, z up, dominant walls along x/y, main floor at z = 0. three.js is y up, so
z-up content is rotated -90° about x: scene `(x, y, z)` → three `(x, z, -y)`. Splats trained in the
scene frame are z-up (`splatUp=z`). glTF is y-up by spec and exporters (Blender's included) already
apply the same rotation, so `glbUp=y` is the default; use `glbUp=z` for a file written without it.
The HUD shows the camera position and target in scene-frame coordinates, ready for `?cam=…&look=…`
or a `views` entry in the project's `scene/model.json`. The axes gizmo at the origin is red = +x, green = +y,
blue = up.

## Controls

Game-style, first person. Click the view to capture the mouse, which then looks around; Esc releases it.

| Key | |
|---|---|
| ↑ / ↓ | walk forward / back (horizontally, wherever you look) |
| ← / → | turn left / right |
| Shift + ↑ / ↓ | move up / down |
| Shift + ← / → | strafe left / right |
| W/S, A/D, Q/E | walk, strafe, down/up |
| wheel | step forward / back |
| G | grid and axes |
| F | overview: from above the south side, looking at the middle |
| 1–9 / 0 | show that splat / no splat |
| M | Blender model on / off |
| X | model as wireframe |
| P | layers panel |
