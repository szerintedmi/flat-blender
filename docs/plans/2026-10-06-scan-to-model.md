# Plan: scan-to-model pipeline for the flat

Status: v7, 2026-10-09. Current state, best results, budget and the ordered "Next" list are in
[START-HERE.md](START-HERE.md); the splat trainer is in [2026-10-08-gsplat-cloud.md](2026-10-08-gsplat-cloud.md).
This file holds the design, the phases, the capture protocol, what each scan taught us, and the pipeline's
decisions and negative results.

A clean-slate redesign in a **private repo**, `flat-blender` (local only so far). The photo-based proof of concept
stays in its own folder `~/projects/flat-blender-poc`; it is never published, and this repo only reads from it.

## Goal

A model of the flat that is **metrically accurate, photoreal and editable**, produced by a repeatable pipeline:
scan, process, model, render and view. Measurements come from LiDAR, not guesses. Textures and camera poses are
computed, not picked by hand. Every stage can be re-run without losing manual fixes.

What the prototype taught us: dimensions are the bottleneck and photos can't supply them; hand-matched cameras and
texture quads don't scale and can't be checked; a parametric model (walls, openings, furniture as objects) is the
right thing to edit, but in a data file, not a Python script; clutter is expensive to model and never looks real;
side-by-side images can't measure error, we need numbers.

## Core idea: three layers from one capture

| Layer | What | Source | Used for |
|---|---|---|---|
| **Structure** | Walls, floors, ceilings, openings, stairs, fitted furniture as typed objects with metric dimensions | RoomPlan, refined against the LiDAR fusion | Editing, clean renders, floor plans, relighting |
| **Appearance** | PBR textures on the structure (albedo for now; roughness and normals later) | Posed 12 MP stills projected onto surfaces | Photoreal look on editable geometry |
| **Radiance** | Gaussian splat of the whole floor, plus cut-outs per object | Video frames + stills + LiDAR depth | Ground-truth reference, clutter, photoreal walkthroughs |

The structure is the source of truth. Appearance and radiance hang off it in one shared coordinate frame.

## Tech stack

| Area | Choice | Why |
|---|---|---|
| Capture | **Our own iPadOS app** (Swift 6, SwiftUI, ARKit, RoomPlan). [Stray Scanner](https://github.com/strayrobots/scanner) (MIT) as a reference only | A fork would save only the encoders (~600 lines) and still need a rewrite of UI, persistence and the iOS 14 target; a fork of a public repo is public. Stock Stray / Polycam have no 12 MP stills, exposure lock, world-map continuation or RoomPlan |
| Capture format | **Our own bundle format** (below), versioned, JSON Schema in the repo. One-way Stray importer | Stray's format has no stills, fixed-column CSVs, one intrinsics matrix per session, no tracking state or exposure per frame, no version |
| Device | iPad Pro 11" 4th gen (M2): LiDAR, 12 MP wide, 8 GB RAM, iPadOS 26 | We have it. 12 MP stills during the session work on M1+ |
| Pipeline | **Python 3.14 + uv**, one package with a `scanmodel` CLI (Typer), stage cache keyed by input hash. Pydantic 2.13, pyright strict, ruff | Re-runnable and scriptable; each stage reads and writes files |
| Geometry | numpy, **Open3D 0.20** (tensor `VoxelBlockGrid` TSDF, ICP), trimesh; our own deterministic line/plane fits | Open3D's RANSAC isn't repeatable; the legacy `ScalableTSDFVolume` extracts nothing in 0.20 on macOS arm64 |
| Poses | **gtsam**: keyframe pose graph (`poses`), then landmark bundle adjustment on RootSIFT tracks (`refine`) | See "Pipeline decisions". pycolmap / LightGlue only if `register` needs them |
| Texturing | Our own planar projection for structure surfaces; [mvs-texturing](https://github.com/nmoehrle/mvs-texturing) for free-form meshes if needed | Not started |
| Splats | **gsplat on a Modal GPU** (`trainer/`, our own loop with LiDAR pruning, depth/normal priors); msplat and [Brush](https://github.com/ArthurBrussee/brush) on the Mac; [SuperSplat](https://github.com/playcanvas/supersplat) for cleanup | The Mac trainers have no depth supervision and limited memory; see the gsplat doc |
| Scene format | **`data/<project>/scene/model.json`** (+ JSON Schema generated from Pydantic) as the single source of truth, plus `overrides.json` for manual edits | Diffable. Generated values never overwrite hand fixes |
| Renderer | **Blender 5.2 → 5.3** (native splats in Cycles from 5.3), driven headless | Best free path-tracer; mixes meshes and splats |
| Viewer | Vite + three.js + **[Spark](https://github.com/sparkjsdev/spark)**, mesh as glTF plus splats (PLY now, SPZ planned) | Walkthroughs on any device |
| Eval | `scanmodel eval` → `report.html`; `scanmodel bench` (ARKitScenes + hand-measured checks); the splat evaluator in `trainer/` | Numbers instead of looking at pictures |

Not used: our own SLAM from raw IMU, Android, continuous 4K ARKit, Polycam/Record3D as the main path.

## Repo layout

```
capture/            FlatBlender (Xcode project) + BundleKit (bundle encoders, testable on macOS)
pipeline/scanmodel/ the `scanmodel` CLI: ingest, align (pose graph), refine (BA), fuse, structure, rooms, boxes,
                    eval, bench, splat (dataset v2, frame selection), deband, freespace, meshcheck, stillcal, …
trainer/            gs_trainer: the gsplat trainer, evaluator and Modal app (own uv project, Python 3.12)
blender/            build.py (model.json -> .blend), render.py
viewer/             Vite + three.js + Spark
scene/              JSON Schemas (generated)
docs/plans/         START-HERE.md, this plan, the gsplat doc
docs/research/      background research
data/<project>/     one scan: raw/, scene/ (model.json, overrides.json, views.json), derived/, splat/, build/,
                    export/ (gitignored; back up raw/ and the hand-made overrides.json and views.json)
```

**The proof of concept** (`~/projects/flat-blender-poc`) is no longer read by the pipeline: its 28
phone photos are future `register` input, its `LAYOUT` estimates were a weak sanity check (off by up to 1.1 m against
scan 3; ignore its warnings). Ideas worth rewriting later: the procedural chilli plants and the gait model in
`walk.py`.

## Capture app: FlatBlender

Current behaviour, settings and files are documented in `capture/README.md`. The design:

**Session model**
- One project = one flat, several **sessions** (one room each, 5–10 min, because of heat), all in one frame:
  - **Next room** (in a doorway, while recording): closes this room's session and starts the next on the same
    ARSession with a new `RoomCaptureSession`. No relocalisation; the new session records `world_map_in`.
  - A later visit loads the latest `ARWorldMap` and starts in a doorway that overlaps an earlier room.
- **Doorway sweep** (~10 s, at every Next room): the wall around the door from both sides within seconds.
- **Closing lap** (offered at Finish): walk back through every room to the start on the same ARSession, so the
  end of the recording sees what its start saw. It is also the splat's `extrap` test set.
- **Anchors**: an `ARAnchor` at every still and every 1 s, with ARKit's updates logged (it does move them, up
  to 14 cm on scan 3; the pose graph uses them as priors).
- Session types: **scan** (stream + auto stills + RoomPlan), **lap** (stream + auto stills, no RoomPlan),
  **texture** (relocalise, then tap-shoot close-ups), **ultra-wide** (not built).

**Recording**: world tracking with `sceneDepth` (smoothed only on request), mesh reconstruction, the high-res
video format (1920×1440 @ 60 Hz), HEVC at 0.3 bit/px (~50 Mbit/s). RoomPlan on the same ARSession (depth-enabled
config re-run after `didStart`). Meter, then **lock** exposure, white balance and focus: shutter 1/240 s with ISO
up to 800, falling back to 1/120 s, then underexposure; Flicker-safe 50/60 Hz modes. Auto stills via
`captureHighResolutionFrame` when the camera is nearly still and the view is new (> 0.3 m or > 15°) and steady
(predicted blur < 1 px at 960 px); scan and lap sessions have no shutter button. **Coverage dots** (magenta per
8 cm voxel seen from too few directions, green when covered, mini-map). Export: one zip per session, **Export
all**, or `devicectl` straight from the app container (START-HERE).

**Coverage metric** (app and offline): 32 direction bins per voxel; a view counts at 0.3–4 m, medium+ LiDAR
confidence, > 15° off grazing, predicted blur < 6 px; deficit = min(needed − seen, 3 − seen), where "needed" is
the directions reachable from where the user actually stood (without that the garden view was 97 % flagged);
ceilings excluded. A voxel that later depth sees through by > max(12 cm, 5 %) is dropped; nothing is integrated
until tracking has been normal for 1 s without a > 10 cm pose jump. Chosen over a tint (invisible in the warm
kitchen, or washing out the wireframe), outlines (blocky) and dense stipple (loud).

### App versions and why

| Version | What changed | Why |
|---|---|---|
| 0.2.0 (5) | Doorway sweep, closing lap, anchor log; the position gate (a session waits to relocalise, ingest rejects one that never did) | Scan 2: drift between rooms; one texture session never relocalised |
| 0.3.0 (6) | Coverage dots + mini-map; shutter 1/240 s with an ISO ceiling and flicker-safe modes; 50 Mbit/s video; steady + sharp auto stills; taps never refused; steadiness ring and cues | Scan 3: 1/120 s gave 4–9 px of blur on 21–47 °/s turns; the messiest views lacked viewpoints |
| 0.3.0 (7) | Coverage free-space carving, 1 s settle, green "covered", key; taps fire within 0.5 s or not at all; capture sounds on an `AVAudioEngine`; Run = Release | Scan 4 feedback (below) |
| 0.4.0 (8) | Depth PNGs and frames.jsonl lines on their own queue, 8 pending frames; no Laplacian sharpness or blur flag; no shutter in scan/lap sessions; `.smoothedSceneDepth` only on request; free-disk query off the main thread; flicker prompt at Lock; `timing …` notes + signposts | Performance review after scan 4; stills analysis (taps add nothing) |
| 0.5.0 (9) | One exposure control (meters, locks by itself when settled; ARKit's lock resets re-applied); Next room / Start lap stop the room and its RoomPlan at the tap, save in the background while the next room is named and prepared, record only on Record, errors offer Retry / Continue anyway; pose-guided doorway sweep; frame count after the queue drains; `startup …` note; world map load, sounds, session scan and deletes off the main thread; coverage casts capped | Scan 5 feedback: Meter/Lock unclear, old room kept recording (and RoomPlan mixed rooms) during the name prompt, hidden auto-start, sweeps not done, sluggish start |

What scan 4 showed about the app (0.3.0 (6), Debug):
- **Debug was the root cause of the slow stills**: per-pixel Swift at `-Onone` ran ~270× slower (sharpness 1.64 s
  per 12 MP still vs 6 ms in Release), so stills took 2–6 s to save, the sharpness median never filled, and auto
  stills starved. Run is now Release, and a Debug build shows an orange banner.
- **Taps queued behind slow saves**: they fired up to 6 s later at whatever the camera did then (18 of 35 taps
  blurry), and from ~95 s took almost every slot (30 taps vs 5 autos).
- **Manual taps added nothing**: the video already had every tap's viewpoint within 9 cm / 5.5°, and a 12 MP
  still buys only ~1.0–1.1× the detail at the 960 px we train at. Decision: no shutter in scan/lap sessions; taps
  only in texture sessions.
- **No sound at all**: system sounds are silent in Silent Mode (1104 is the keyboard click, which also needs
  Keyboard Clicks on); iPads have no haptics. Now synthesized tones at media volume.
- **Coverage dots floated** 10–80 cm in front of surfaces (depth integrated at a pose ARKit reported as normal
  just before its 31 cm relocalisation snap, plus long-range LiDAR bias), asked for the current direction and never
  cleared. Fix: carving + settle. Replay: dots asking for the current direction 32 → 6 %.
- **19 one-frame gaps** in frames.jsonl, all right after a depth frame: the PNG job held the video queue. Fixed in
  0.4.0.
- **Performance review**: in Release our own per-frame work is small. Remaining suspects for lag: the RoomPlan
  forward in the delegate, ARKit's own load, the preview GPU (blur materials, wireframe). A lighter preview or a
  lower anchor tick rate are optional, decided from the next scan's `timing` line.
- **Flicker prompt**: probably unnecessary now that `scanmodel splat` debands; answer "Keep 1/240 s" and consider
  turning it into a silent log after the next scan.

What scan 5 showed about the app (0.4.0 (8), Release): **every device check passed.** A perfect 60 Hz stream (0
one-frame gaps, scan 4: 19), tracking normal from frame 0 in every session (no restart after Record), delegate p99
0.8–1.3 ms with 0 frames over 8 ms, arkit gaps 0, our drops 0, still latency max 89 ms (scan 4: 209), 310 auto
stills (15–25 / min, blur max 7.4 px, scan 4: 28), thermal nominal. One-off peaks without a gap: coverage 137 ms
(living), depth png 109 ms (lap). The flicker prompt recorded `50hz` / `declined` at 1/240 s. The exposure lock
set in the first room carried to all (ISO 232): the conservatory ran bright (+1.3 EV median), the hallway dark
(p5 −3.5 EV), the outdoor garden blown out. Debug vs Release is not recorded in the bundle.

## Bundle format v1

Ours; schema in `pipeline/scanmodel/schemas/bundle.v1.json` (from `pipeline/scanmodel/models/bundle.py`). Per file:
`capture/README.md`, "What the app writes".

```
<session>/
  session.json            version, ids + type, device, app version, video format, locked camera (metered,
                          shutter mode, flicker fields), world map in/out, sweeps, notes (stats, timing)
  frames.jsonl            per stream frame: idx, t, camera_to_world, intrinsics, tracking, exposure, depth file
  video.mov               HEVC stream
  depth/, confidence/     uint16 mm / 0-1-2 PNGs, 256×192, every 6th frame (10 Hz)
  stills.jsonl, stills/   12 MP JPEGs: intrinsics (ARKit's, ~4.8 % off: corrected in ingest), pose, exposure,
                          WB, nearest frame, gravity, motion
  mesh.ply                ARMeshAnchors in world coordinates
  room.json, room.usdz    RoomPlan CapturedRoom
  worldmap.arworldmap, anchors.jsonl
```

Raw IMU isn't recorded: ARKit's poses already include it. `calibration.json` isn't written (ARKit doesn't expose
`AVCameraCalibrationData`).

## Pipeline stages (`scanmodel <stage>`)

Inputs and outputs per stage are in `README.md`. In order:

1. **ingest**: check the bundle; ARKit (y up) → pipeline axes (z up, metres); correct the stills' focal length per
   session (`scanmodel.stillcal`); leave out frames and stills before tracking settles (`scanmodel.checks.settled_from`).
2. **poses**: merge sessions; keyframe pose graph (`scanmodel.align`): ARKit odometry, gravity, floor-height links,
   SIFT + LiDAR loop closures gated against ARKit, anchors as priors; scene frame (z = gravity, yaw snapped to
   the walls, origin at the front door, main floor at z = 0).
3. **refine** (default since 2026-10-09; `--poses posegraph` / `poses = "posegraph"` in `settings.toml` skips it and reproduces
   P6-era results exactly): bundle adjustment of thinned depth frames (5 cm / 4° / 0.4 s) and stills on RootSIFT
   tracks across passes and the lap, with LiDAR range priors, ARKit relative motion, gravity (loosened to 2°) and
   **floor heights** (LiDAR floor cells as height landmarks, each level held level; `--floor-sigma`, 0 = off).
   First a **coarse round** finds drift the pose graph left (wide windows, loose gate): **trajectory cuts** where
   ARKit saw nothing or has a gap in its frames and later sessions inherited the jump ("Trajectory cuts" below),
   then consensus per session and 4 s bin; only if there is some, it moves the start poses ("Drift beyond the
   gates" below).
   Corrections are interpolated to every frame; the first 3 s after an unsettled start are dropped. **Fails loudly,
   never falls back** to the pose graph. 2.5 min on scan 4, 8.6 min on scan 3.
4. **fuse**: Open3D TSDF from LiDAR depth (confidence ≥ 1), ~1 cm voxels → `dense.ply`, `points.ply`.
5. **structure**: RoomPlan walls, openings, objects, stairs → typed elements, each refined against the points
   (walls: deterministic line fits, right-angle snap only if a wall's ends move ≤ 1 cm, no refinement that rotates
   a face > 10°; boxes: `scanmodel.boxes`); rooms (`scanmodel.rooms`: outlines snapped to faces, floor per room, sloped
   ceilings); thickness (below); `overrides.json` last → `scene/model.json`.
6. **eval**: distance of each wall face to the dominant surface within 3 cm in `points.ply` (mean, p95), openings,
   per-session floor heights, **doubled surfaces** (30 s windows that put the same walls in different places:
   the poses split into blocks) → `report.html`, `history.jsonl`. `scanmodel meshcheck`: the fused mesh vs each frame's
   own LiDAR. `scanmodel bench`: ARKitScenes rooms + our hand-measured checks.
7. **texture** (not started): score every still per surface (head-on angle × px/m × sharpness × visibility against
   `dense.ply`), pick or blend, warp to 500 px/m (1,500 for artwork), colour-match sessions with a grey card.
   Priority: registered upload > texture session > scan still > video frame. Shiny and see-through surfaces get
   procedural materials.
8. **register** (not started): poseless photos from any camera → poses (retrieval, LightGlue, PnP against depth).
9. **splat**: dataset v2 for any trainer (frame/still selection, debanding, LiDAR depth, mesh, interp/extrap split;
   contract in `trainer/README.md`), then `--train` with msplat, Brush or gsplat (cloud).
10. **build**: Blender headless: `model.json` → `.blend` (stairs as the scanned mesh inside RoomPlan's stair boxes;
    a hidden "Scan" collection with the fused mesh).
11. **export**: `model.glb` + every trained splat into `viewer/public/data`.

Every stage writes to `data/<project>/derived/<stage>/` with a manifest of its input hashes; `scanmodel all` runs only
what changed. Raw bundles are ~1–3 GB per room in `data/<project>/raw/` (back them up; Git LFS isn't worth it).
Back up `scene/overrides.json` and `views.json` too; everything derived is rebuildable.

## Phases

**0. Device spike**: done. Depth survives RoomPlan; tracking 100 % normal over multi-room walks; still latency
~60 ms; exposure lock holds over 41.8k frames; thermal nominal over 11.5 min; still intrinsics 4.8 % off (fixed in
ingest). Remote: private `szerintedmi/flat-blender` (2026-10-09). Open: a backup of the raw scans (`data/` is gitignored).

**1. App MVP**: done (2026-10-06), plus everything in "App versions". 0.4.0 (8) device-tested on scan 5 (2026-10-09); now 0.5.0 (9) (scan 5 feedback, "App versions").

**2. Structure end to end**: done except the bar. Open:
- [ ] Done when: the ground floor from scans with wall error p95 < 2 cm and no hand-written dimensions. Scan 3 now
      17.1 / 64.3 mm (mean / p95).
- [ ] Bench floor check fails (−10.4 vs −6 ± 4 cm; sloped conservatory floor): slope-aware `floor_z` or a
      redefined check (START-HERE, open problem 1).
- [ ] Duplicate per-session walls: one wall per session, so a session's poor fit survives as a duplicate
      (`wall-4c36a2ee`, ~15 cm in front of the living east wall, tilted ~3°).
- [ ] Cross-session thickness: gated on leave-one-pair-out < 1 cm, which the pose graph never reached (2.1 cm on
      scan 3). Re-measure with refined poses (cross-pass ~0.8 px) before building it.
- [ ] Use the doorway sweeps explicitly (recorded, but thickness comes from any 30 s window; of scan 3's three
      sweeps, two walls got a thickness).
- [ ] Audit items for other flats (`docs/research/2026-10-07-threshold-audit.md`): #8 same-face merge tolerance
      from measured drift, #9 attic ceilings with two slopes, #12 per-axis odometry noise, #14 outlines snap only
      to their own room's walls, #19 small floor steps snapped to the room's floor, not 0.

**3. Appearance**: not started. The `texture` stage is the largest feature left; photometric part of `eval`. Done
when every wall, floor and fitted front is textured automatically and held-out side-by-sides pass a look.

**4. Radiance**: gsplat on Modal with LiDAR depth, mesh normals, frame selection, debanding and refined poses (the
gsplat doc). Open: densification, object cut-outs, the views that are off the captured paths. Blender 5.3 hybrid
once 5.3 ships.

**5. Upgrades**: texture sessions in practice, `register` for extra photos, `scanmodel serve` LAN upload.

**6. Viewer**: done: Spark splat + glTF in the scene frame, layers panel (one splat at a time, model overlay),
walk controls, `?pose`/`?fov`, copy view URL. Open: SPZ, stay near captured paths / fade to the mesh, floor-plan
minimap, walk heights, a textured-model layer, camera paths shared with Blender.

**Later, optional**: albedo de-lighting, image-to-3D furniture (licences, CUDA), a VLM for materials and missed
objects, the upper floor.

## Capture protocol

- Daylight plus all lights on. No flash. Blinds against direct sun. Doors open. Nobody in frame, cat included.
- Under LED light keep **Sharp (1/240 s)**: answer "Keep 1/240 s" at the flicker prompt; `scanmodel splat` removes
  the bands.
- First still of each session: a grey card, for colour matching (texture stage; not yet used).
- Start in the hallway. One room per session; between rooms go back through the hallway (hub and spoke).
- Two slow loops per room at chest height. **Keep furniture and edges in view**: scan 5's living session drifted
  55 cm in ~30 s with the camera held high on plain walls and ceiling (tracking "normal"). **Turn slowly** (the app says "Turn a little slower"); pause briefly at
  new viewpoints for auto stills. No taps in scan and lap sessions.
- **Walk the routes the viewer will use, both ways**, at head height; side passes along counters; doorways from
  both sides; near clutter from several angles; tilt up for the ceiling. Follow the coverage dots until they turn
  green, especially kitchen-counter-side and kitchen-to-hallway.
- **Next room** in each doorway, then the **doorway sweep** (~10 s countdown).
- After the last room: **closing lap** back through every room to the start, then Finish. In the lap, **walk
  corridors facing along them, never sideways along a blank wall**, and pause
  and pan in each doorway.
- One tape-measured long distance per scan (e.g. kitchen → living), to tell drift from geometry.
- Texture sessions for close-ups; high-res photos from any camera for the most detailed items (`register`, later).

## What each scan taught us

| Scan | Capture | Lessons |
|---|---|---|
| 1 (2026-10-06) | One 3 min session through four rooms, before auto stills and Next room existed | RoomPlan gives an interior wall one face per session: a measured thickness needs the rooms either side as separate sessions (→ Next room). Right-angle snapping by angle alone undid good fits on long walls (→ bounded by end movement). Stills only happen if auto-shoot does them. Eval 2.1 / 9.4 cm |
| 2 `second` (2026-10-06) | Four rooms with Next room + three texture sessions (app 0.2.0 (2)) | **Measured thickness was mostly drift** (the kitchen ~8 cm off the hallway; the same face twice, 8 cm apart). Each room is accurate on its own (face fits 4–5 mm rms); the error is between rooms. Doorway jambs don't measure thickness. Scanning more carefully wouldn't fix it. A texture session never relocalised (→ position gate). Conservatory: a 9 s tracking stall at the start |
| 3 `third` (2026-10-07) | Four rooms + sweeps + 40 s lap, 186 stills (app 0.2.0 (5)) | **The pose graph got steadier** with the lap, sweeps and floor links (leave-one-pair-out 13 → 2.1 cm), still above the 1 cm bar for cross-session thickness. ARKit moves anchors (≤ 14 cm). RoomPlan reports what it sees through glass and doorways and calls framed gaps doors (→ rooms from camera positions, doors need a leaf). Cabinet fronts became a wall's far face (→ furniture footprints ignored). **Still intrinsics 4.8 % off**. The splat's fog is colour on splats in LiDAR free space (→ pruning during training). 1/120 s blurred turns |
| 4 `fourth` (2026-10-08) | Living room only, 5:19 + 38 s lap, 1/240 s, LED evening light, coverage dots (app 0.3.0 (6), Debug) | **1/240 s cut blur** (frames < 0.6× their neighbours 3.6 → 2.2 %; predicted blur 2–4× smaller; ISO 620 noise not visible). **But 100 Hz LED banding of 18 %** (→ debanding). ARKit can report "normal" before its world settles (11 s stall, then a 31 cm snap) (→ `settled_from`). The app findings above (Debug build, taps, sounds, floating dots). **Ghosting from cross-pass pose error** (→ `scanmodel refine`). A single 5 min pass covered walls and high surfaces better than scan 3 did for this room |
| 5 `fifth` (2026-10-09) | Conservatory, **garden (outdoors)**, living, hallway, kitchen + 47 s lap, 310 auto stills, 1/240 s, one exposure lock (app 0.4.0 (8), Release) | **The app is clean** (above). **ARKit can make a 1 s, 60 cm out-and-back excursion with tracking "normal"** (living; the pose graph and refine kept it, and an anchor placed in it fed a 64 cm "correction" to the pose graph) (→ ingest drops excursions). **ARKit drifted 55 cm / 1.4° in the living session** (t ≈ 106–158 s, camera held high on plain walls and ceiling, tracking "normal"), and the hallway, kitchen and lap, which continued its ARSession, inherited it: the flat came out as two rigid blocks 0.4–0.5 m apart (conservatory + garden + living before the drift vs the rest; the living room's south wall twice, 40 cm apart, from the same side). The 1 s "excursion" at 157.9 s was ARKit briefly relocalising to the right pose (its jump is the drift vector) and jumping back. First read as **the lap drifting 48 cm** along a blank wall (the pose graph had no closure for the lap, refine's 25 cm gate threw away every match that showed the offset; → refine's coarse round, which moved the lap onto the other block and left the split); found on 2026-10-10 while rendering splat walks (`data/derived/fifth/analysis/walks/`) and then by a new eval check (→ refine's trajectory cuts, eval's doubled-surface check; "Trajectory cuts" below). Eval exposed two structure bugs: faces refit onto the far side of thin partitions, and a camera vote counting cameras that couldn't see the face (→ fixed). The garden's facade walls fit well (7–13 mm); its RoomPlan also put a phantom wall through the glass onto living-room furniture. The living room's polygon came out as the hallway (gone after the trajectory cut: it was the split) |

## Pipeline decisions and negative results

Splat-side decisions and negative results are in the gsplat doc. These are the pipeline's, so nobody repeats them.

**Wall thickness and room alignment.** Thickness is the gap between opposite faces, so any misalignment between
sessions lands in it. Decision: thickness only from **one session seeing both faces within a 30 s window** (0.5 m
strips, median; windows must agree within 2 cm), plausible 4–90 cm (flagged over 50 cm), else the 10 cm default
(overridable); a flag when two sessions' faces disagree. Tried and failed (bar: leave-one-out change ≤ 1 cm):
- Rigid per-session (dx, dy, dθ) from shared wall faces: corrections moved up to ~10 cm with small changes to the
  candidate lines; RoomPlan door centres as constraints made one pair worse (1.5 → 11 cm). v2 (per-session line
  fits, no glass/doors): leave-one-face-out 5.6 cm. v3 (joint ICP over shared vertical surfaces): leave-one-pair-out
  10 cm, 28 cm without the conservatory's only link.
- Image features, rigid per session: leave-one-pair-out 22 cm. A 4-DOF drift model with knots every 20 s fits the
  matches (1.3 cm median) but is underdetermined: leaving out one session pair moves walls by up to 47 cm (no
  odometry edges).
- Colored ICP between 20 s fragments as extra pose-graph edges: worse (stability 6.1 cm, p95 9.1 cm).
- Open3D `segment_plane` isn't repeatable even with a fixed seed (→ our own line fits).

**Pose graph** (`scanmodel.align`, the standard fix per `docs/research/2026-10-06-arkit-roomplan-best-practices.md`):
closures are gated against ARKit (3 cm vertical, 15 cm horizontal: stair steps and identical cabinet doors gave
consistent false closures of ~20 cm); floor-height links (horizontal surfaces in the same 0.5 m cell seen ≥ 15 s
apart, outside stair/furniture footprints) fixed vertical drift (scan 2: conservatory +12.7 → −9.1 cm vs living);
moved anchors are priors. ALIKED + LightGlue wasn't needed: closures stopped being the weak spot.

**Floors and structure rules** (audit, checked with `scanmodel bench`): floors are searched relative to each room's
lowest upward surface (split levels). "Largest-area level" was rejected by the bench (floor errors up to 3.4 cm vs
0.9 cm: drift leaves fused floors in thin layers); a large sunken area (shower tray) can still win. Box sides need
half a side's coverage; hidden sides follow the top only where it visibly ends. Rooms drop only slivers (< 0.5 m
wide); objects merge only across sessions with overlapping footprints.

**Ghosting diagnosis → `scanmodel refine`.** Scan 4's splat ghosted near surfaces (the city-map painting). Frames of one
pass agreed to 1–3 mm, but passes past the painting sat 2–6 cm apart (median 29 mm, 6–18 px at 960 px, worse near):
the pose graph's closure error, about 3 cm of cross-pass inconsistency. Rolling shutter was secondary (14.7 ms
readout). **Not "too many images"**: fewer images only help if they all come from one pass. Results of refine:
- Scan 4: cross-pass residual 3.81 → 0.81 px; painting per-view offset 29.3 → 1.8 mm median (p90 47.5 → 4.1); dense
  cross-pass flow (strict mask) 0–1.5 m 12.4 → 2.5 px, overall 7.9 → 1.8. A local msplat of the painting: NCC
  0.62 → 0.90. Corrections 3–4 cm median, up to 15 cm, up to 1.6° yaw after the relocalisation.
- Scan 3 (sessions drift further apart): cross-pass 3.39 → 0.86 px; dense flow 9.3 → 1.1 px, at the noise floor.
- Fused mesh vs each frame's own LiDAR (`scanmodel meshcheck`): priors fused from the pose graph but used at refined
  poses sit ~2 cm off (scan 4 19.1 mm median); fused from refined poses 8.6 mm (scan 3 6.9 mm), with thinner
  surfaces (fewer doubled walls). So fuse must use the same poses as the splat: `scanmodel splat` refuses a mismatch.
- **Floor heights in the BA** (added after refine became the default; they do what the pose graph's floor links
  did): scan 3's floors agree (hallway / kitchen vs living +0.3 / +0.1 cm); against the pose graph, eval 23.3 /
  86.9 → 17.1 / 64.3 mm and meshcheck median 9.1 → 6.5 mm (scan 4: 15.2 / 82.3 → 9.2 / 41.6 mm). The gravity
  prior is loose (2°).
- **Tried, not kept:** a second, narrower matching round for close range (windows 2–4 cm, tighter gates):
  neutral within noise (0–1.5 m 2.35 → 2.02 px in one variant, 2.49 → 2.40 in another, other bands slightly worse;
  mostly a 6–8 mm gauge shift). Most of the close-range "residual" was the flow metric failing on glass, plants,
  thin legs and occlusion edges. Planar targets (the painting) are the better close-range check.
- **Tried, off by default:** rolling-shutter keypoint correction (`--readout-s 0.0147`): BA residual 0.77 → 0.61 px
  but no change on planar targets.
- Landmark BA with native gtsam factors (no Python custom factors): 40–60 s for 1.4k views / 400k observations.

**Drift beyond the gates: refine's coarse round** (`scanmodel.refine`, "coarse alignment"). (2026-10-10: the scan 5 story
below was wrong about who drifted: it was the living session, see "Trajectory cuts"; the per-session vote stays
for a session that drifts against the others, and after the cut it admits nothing on scan 5.) Scan 5's lap "drifted"
48 cm / 1.5° while it slid along a blank wall (t 16–23 s, nothing matched), smoothly, tracking "normal". The pose
graph had no closure for the lap at all (20–350 depth-backed keypoints per frame at contrast 0.04, a closure needs
20 3D inliers; not its 40 cm gate), and refine's guided windows (8 cm) and gate (25 cm / 3°) kept or rejected
matches against the wrong poses (23 accepted pairs at 24–28 s agreed with the wrong pose: a blank wall aliases
along itself). Now refine first matches cross pairs from 50 cm voxels with 0.6 m windows and a 1 m / 6° gate, and
accepts drift only by consensus per session and 4 s bin of the camera being moved: ≥ 5 loose pairs, 60 % within
10 cm + 3 % of the camera distance of their median, median > 10 cm, outnumbering the bin's tight pairs to other
sessions 3:1 (pairs to the session's own other passes don't vote: its views drift together), from two other
sessions or views 45° apart, in ≥ 3 bins in a row; neighbours with the same offset join a run when their loose
pairs merely outnumber the tight ones (tight pairs in admitted bins are dropped as aliased). The drift span runs
from the session's last tight pair to another session before the run to the end of the run's first bin: ARKit's
relative motion there is loosened by the offset and rotation (all six axes, offset / sqrt(links) each; gravity
still holds roll and pitch), a coarse pose graph (loose priors, none in the span; pair poses under Huber, re-gated
to 10 cm / 2°, then Cauchy) gives the start poses, and the fine round and the bundle adjustment run from there
with the loosened odometry. Scan 5: the lap's bins from 20 s on agree on 50–53 cm (79–98 % per session pair),
span 16.3–24 s; late lap frames vs the PnP-corrected poses (`lap5/pnp3.json`) 45.9 → 3.4 cm median (p90 4.8);
eval seen faces 9.1 / 27.3 → 8.3 / 25.4 mm (all faces 10.4 / 31.2 → 9.8 / 34.6); cross-pass 0.80 → 0.79 px; the
lap's walls now sit 6–31 mm from the other sessions' (garden 206 → 20 mm, living 60 → 17 mm: its LiDAR was fused
50 cm off before). Cost: the other sessions bend by ~3.7 cm along the drift (conservatory / garden −15 mm y,
hallway / kitchen +23 mm y vs the old refine) with no change in their own residuals (0.59–0.86 px per session,
as before) or wall-to-wall agreement: an inter-room offset the old solution didn't determine either; and the
lap's floor sits 1–1.6 cm above the others' in garden and conservatory (eval floor spread 0.5 → 1.6 cm). Scans 3
and 4: no bin admitted, poses bit-identical; the coarse round costs ~1 min. Tried, not kept (scan 5): loosening
only along the offset (bend 6 cm, eval 9.3 / 26.8), only horizontally and also after the span (4.2 cm), and
holding the other sessions in the coarse solve (7 cm, eval 10.3 / 37.1: the adjustment starts inconsistent).
`refine.json` `session_pairs` (fine round) and `coarse.session_pairs` report per session pair the accepted and
rejected pairs, matches and the median offset of the rejected (`drift`: they agree on > 10 cm, also a warning);
the pose graph's manifest has `closures_by_session_pair` and `sessions_without_closures` (scan 5: the lap).

**Trajectory cuts** (`scanmodel.refine`, "trajectory cuts"; 2026-10-10). Scan 5 stayed split into two rigid blocks after
the coarse round: conservatory + garden + living 0–106 s vs living 158 s on + hallway + kitchen + lap, about
(−0.19, +0.41) m and 1.2° apart at the refined poses (0.53 m at the pose graph's). Diagnosis
(`data/derived/fifth/analysis/split/`): the living session's own LiDAR puts the living room's south wall at
y = −5.06 for t 30–130 s and −5.59 from 150 s (raw ARKit, one rigid transform; ICP of 4 s windows against its
first 130 s: (−0.16, +0.53) m from 148 s on, fit 0.99); between ~106 and ~156 s the camera was high on plain
surfaces and nothing there matched anything outside that stretch; tracking normal, no pose step over 6 cm. At
157.9 s ARKit jumped by (−0.26, +0.55) m, the drift, and back 1 s later: ingest's "excursion" was a brief
relocalisation. All six sessions are one ARKit trajectory (each continued the previous ARSession), so everything
after the drift inherited it and agrees with itself: the per-session vote saw only the lap's bins disagree with
three other sessions and moved the lap (spuriously; after the cut it finds nothing there), the pose graph's
closures are gated at 40 cm, and nothing else links the blocks. Eval and meshcheck missed it: each RoomPlan face
belongs to one session and is fitted to "the dominant surface within 3 cm" of it, and meshcheck compares each
frame's LiDAR with a fused mesh that holds both copies. Fix: before the vote, a trajectory may be cut where ARKit
had nothing to hold on to (stretches of 8 s or more between views anchored by a coarse pair to a view 30 s or more
away along the trajectory or to another trajectory; 15 s let the drifting views anchor each other) or at a gap in
its frames (ingest's dropped windows, lost tracking). A cut is admitted when the coarse pairs crossing it agree on
one jump (≥ 15 agreeing, 60 % of the loose ones, > 10 cm, 3:1 over the crossing pairs that agree with no jump,
from 3+ bins on each side, two sessions or views 45° apart after it) and the jump makes the most coarse pairs
consistent; the views after it move by its rigid jump (yaw + translation from the agreeing pairs; the
trajectory's start keeps the scene frame), spread linearly over the stretch, ARKit's relative motion there is
loosened (a `Span` of the trajectory), the coarse pairs are matched again and the vote and the coarse solve run as
before. The stretch's frames and stills are left out of the output (their poses are a guess; scan 5: 3,144 frames,
living 106–158 s, mostly ceiling), and RoomPlan moves with its session (`session_to_scene`; a session the cut
splits keeps the larger part's and gets `session_to_scene_parts`, and structure places each RoomPlan element with
the move that puts it on the fused points: 12 of the living session's, its hallway end). Scan 5: one cut, living
105.7 → 157.8 s, 346 crossing pairs agree / 1 against, jump 55 cm / −1.4°; the lap's late frames and the vote's
drift are gone; FRONT moved by (−0.188, +0.405) m / −1.2° relative to the rest vs the old refine (the walk
renderer's hand join was (−0.198, +0.402, 0)); the living room's south wall is one face (living-side −5.005 for
both living parts and the lap; kitchen side −5.095); eval seen faces 8.3 / 26.6–27.0 mm over three fuses (29 of 31 measured; was 8.3 /
25.4, 27 of 30; all faces 10.0 / 40.4, was 9.7 / 33.0: more unseen faces measured), meshcheck median 6.8 mm (was 7.1), RoomPlan's duplicate
living "room 2" over the hallway gone, cross-pass
0.79 px, floors within 0.2 cm across hallway / kitchen / living. Scans 3 and 4: no cut admitted, refined poses bit-identical (scan 3: eval 12.5–13.2 / 52.8–58.2 mm over three fuses, meshcheck 6.5 mm; scan 4: 7.3 / 24.7 mm, meshcheck 8.4 mm); the coarse round now takes ~2 min on scan 5 (matched twice). **Fuse is not deterministic** (Open3D's TSDF): on scan 3's identical poses three fuses gave eval 12.5 / 52.8, 13.1 / 57.9 and 13.2 / 58.2 mm, so compare eval within ~1 / 5 mm (scan 5 three times: 8.3 / 26.6–27.0).
Detector (`scanmodel.consistency`, in eval): every two 30 s windows of the poses' `walls/` that saw the same walls from
the same side, compared by their own LiDAR (a vote over t · n = d lines from same-facing neighbours, least squares
at the peak, only along pinned directions); pairs that disagree by ≥ 5 cm with twice the support at the shift than
at zero are grouped by their shift, and a group of 3+ pairs at ≥ 10 cm is a split: a warning after `scanmodel all` and
a table in the report. Old scan 5 poses: 44 cm, (−0.155, +0.415) m, 46 window pairs (pose graph's: 54 cm); new: 0
disagreeing pairs. Scans 3 and 4 refined: 0 disagreeing pairs (scan 3's pose graph: 6 cm, under the warning).

**Pose excursions** (`checks.excursions`, ingest). A jump faster than 3 m/s and 8× the median speed within ±1 s
(or a turn over 300°/s; brisk hand-held turns reach ~155°/s), at least 10 cm / 5°, followed within 5 s by a jump
back that cancels it (to 0.1 m or 25 %): the frames between (+2 either side) and the stills shot then are left out,
like an unsettled start, and listed in `report.json` `dropped`. Up to 3 s; longer ones and one-way jumps stay (a
warning). Anchor events inside a dropped window are ignored by the pose graph. Scan 5: living 9464–9528 and still
56 (2026-10-10: this one was ARKit briefly relocalising to the right pose after 55 cm of drift and jumping
back; dropping it is still right, the drift is a trajectory cut's job); pose-graph max correction 17.2 → 8.0 cm, anchors moved 64 → 13 cm, eval (all faces) 20.6 / 90.4 → 16.7 /
76.0 mm. Scans 3 and 4 ingest unchanged. Hand-picked ranges: `overrides.json` `drop_frames`. Why not let the BA
fix it: odometry between neighbours is trusted (sigma ∝ the step, no robust loss) and closures are gated, so a
60 cm step is "real motion"; re-aligning 1 s of video isn't worth an estimator.

**Structure: faces fit only surfaces facing their room; a face's own depth outvotes far cameras.** The refit took
points with `|n · interior| > 0.85`, so on a partition thinner than the 8 cm search band the other room's surface
could win, making a duplicate wall 6–9 cm off (scan 5: two, scan 3: `wall-d0f849f1`). And the room side came from
every camera within the face's extent, even 2.7 m behind it. Now `n · interior > 0.85`, and a face is flipped when
its session's own depth points within 15 cm face the other way 2:1 (≥ 200 points). **Eval's headline is over seen
faces** (`Wall.faces_observed`: a back face counts only with a partner face or a measured thickness); unseen faces
and the old all-faces numbers are reported beside it, plus faces whose nearest real surface is 10–50 cm off
(`faces_gross`). Seen-faces mean / p95: scan 5 18.0 / 88.2 → 9.1 / 27.3 mm, scan 3 15.9 / 62.6 → 12.5 / 52.7,
scan 4 7.3 / 24.9 (unchanged within 1 mm). Excluding the outdoor garden from structure made eval worse (good facade
walls dropped out): keep it.

**Tracking settle** (`settled_from`): only when a session starts with non-normal tracking or a timestamp gap. Scan
4's living session lost 127 frames (13.2 s); scan 2's conservatory 145; scan 3 unchanged. Scan 4's unsettled
frame 0 (30 cm off) had made structure split the room in two.

**Debanding** (`scanmodel splat --deband auto|on|off --mains-hz 50`, `pipeline/scanmodel/deband.py`). Under 50 Hz LED light
1/240 s shows 100 Hz flicker as rolling bands (period 982 rows at 14.7 ms readout, 240° further each frame); lamps
flicker by different amounts (conservatory ~70 %, living 20–30 %, hallway < 5 %), so each frame gets a smooth 2D
amplitude map per channel. Method: the neighbouring frames are warped onto the frame by rotation, giving three
samples ~120° apart per pixel with known phases; the fit is smoothed over flat pixels and divided out on (V + 0.5),
Apple's video tone curve measured from the bands. Stills get their nearest frame's map through the homography
(× 0.96). Results: single-frame amplitude **18.3 → 0.4 %**; same surface 0.5 s apart **7.5 / 19.8 → 1.7 / 5.2 %**
(median / p90; scan 3's band-free kitchen 2.1 / 5.6 %); stills 14 → 0.5 %. Left: the 300 Hz harmonic (0.5–1 % on
the ceiling), slightly crushed blacks under strong bands. `auto` (threshold 2 %) also debands scan 3's
conservatory, living room and lap (4–6 %); `--deband off` reproduces P6 datasets. Cost ~0.3 s per image (scan 4's
export 1.5 → 5.5 min). **Decision: keep 1/240 s under LED light** (sharp input, bands removed); the trained splat
shows no banding residue.

**Stills.** ARKit's high-res intrinsics are the stream's scaled to 12 MP; the still is 4.8 % narrower (measured
and corrected per session in ingest). Stills are ~8 % brighter than the video frames (gains matched in `scanmodel
splat`). For splats at 960 px a still adds little over the video; their value is textures.

## Risks

| Risk | Mitigation |
|---|---|
| ARKit drift across rooms (scan 2: ~8 cm by the fourth room; scan 3: pose-graph corrections up to 13 cm) | Pose graph, then `scanmodel refine` (cross-pass ~0.8 px); doorway sweeps; closing lap |
| ARKit reports "normal" before it settles | `settled_from` in ingest, 3 s settle drop in refine, 1 s settle in the app's coverage |
| Debug builds on the device | Run = Release; orange banner in Debug |
| LED flicker at short shutters | Debanding in `scanmodel splat`; Flicker-safe modes in the app |
| RoomPlan misreads odd geometry (winder stairs, conservatory roof, things seen through glass) | Plane refinement, camera-based rooms, typed overrides |
| Colour shifts between sessions | Locked settings, per-image brightness gains, grey card (texture stage) |
| Cloud cost | START-HERE budget rules; `just gs ps`; Modal timeouts and workspace budget |
| No backup of the repo | Private GitHub remote `szerintedmi/flat-blender` (done); the raw scans in `data/` still need one |
| Licences | Our code MIT. Brush Apache-2.0, Spark MIT, Open3D MIT, gsplat Apache-2.0, DA3-BASE Apache-2.0. Avoid non-commercial splat-to-mesh repos; ARKitScenes data never committed |

## Implementation notes

- Data contracts are Pydantic models in `pipeline/scanmodel/models/`. `scanmodel schema` writes
  `pipeline/scanmodel/schemas/bundle.v1.json`, `scene/schema.json`, `scene/overrides.schema.json`; a test fails if
  they are stale.
- Axes: world z up; cameras OpenCV internally, OpenGL for Blender and nerfstudio. Bundles keep ARKit's values.
- Walls in `model.json` are centrelines + thickness; a face seen from one room gets the 10 cm default on its outside.
- Front door (origin): `overrides.json` `frame.origin_door`, else the first session's door nearest its start.
- The Stray importer measures the odometry's camera-axis convention instead of assuming it.
- Synthetic end-to-end test: two rooms, ray-cast depth, RoomPlan with ±3 cm / ±1° errors → faces within 1 cm,
  eval p95 ≈ 6 mm; a one-session variant checks the room split.
- `scanmodel bench`: ARKitScenes rooms and the checks in `data/<project>/checks.json`; results in `data/derived/bench/`.
- Each scan is a project, `data/<project>/`, selected with `--project` (default: `project` in `settings.toml`;
  `docs/scene-portability.md`).

## References

- Stray Scanner (reference + importer): https://github.com/strayrobots/scanner (format: docs/format.md)
- ARKit high-res stills: https://developer.apple.com/documentation/arkit/arconfiguration/recommendedvideoformatforhighresolutionframecapturing,
  https://developer.apple.com/forums/thread/709811
- RoomPlan with your own ARKit session (WWDC23): https://developer.apple.com/wwdc23/10192, depth
  workaround: https://developer.apple.com/forums/thread/763400
- Open3D TSDF: https://www.open3d.org/html/tutorial/t_reconstruction_system/integration.html
- mvs-texturing: https://github.com/nmoehrle/mvs-texturing
- Brush: https://github.com/ArthurBrussee/brush, SuperSplat: https://github.com/playcanvas/supersplat,
  Spark: https://github.com/sparkjsdev/spark
- DN-Splatter (indoor splats with phone-LiDAR depth): https://github.com/maturk/dn-splatter
- Blender 5.3 splats: https://digitalproduction.com/2026/09/22/blender-5-3-brings-gaussian-splats-home/
