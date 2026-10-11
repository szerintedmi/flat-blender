# flat-blender

Scan-to-model pipeline for indoor scenes: LiDAR capture on the iPad → metric, editable `data/<project>/scene/model.json`
→ Blender and a web viewer. For a different scene, read [scene portability](docs/scene-portability.md).
**Current flat experiments:** [docs/plans/START-HERE.md](docs/plans/START-HERE.md) (state, best
results, budget, next steps). Plan and rationale: [docs/plans/2026-10-06-scan-to-model.md](docs/plans/2026-10-06-scan-to-model.md);
the cloud splat trainer: [docs/plans/2026-10-08-gsplat-cloud.md](docs/plans/2026-10-08-gsplat-cloud.md).

| Folder | What |
|---|---|
| `capture/` | FlatBlender iPad app (Swift 6, SwiftUI, ARKit, RoomPlan) + `BundleKit` (bundle format encoders) |
| `pipeline/` | Python 3.14 package `scanmodel` (uv): the `scanmodel` CLI and every stage |
| `blender/` | `build.py` (model.json → .blend), `render.py` |
| `viewer/` | Vite + three.js + Spark splat/glTF viewer |
| `scene/` | Reusable JSON Schemas (generated) |
| `data/<project>/` | One scan's captures, scene, derived files, splat datasets/runs, builds and analysis (gitignored) |

## Setup

Tools: [uv](https://docs.astral.sh/uv/) (Python), [bun](https://bun.sh) (viewer), [just](https://just.systems)
(commands from the repo root), Blender, and msplat or Brush for splats. Optional, for cloud gsplat training:
a [Modal](https://modal.com/docs/guide) account; the `modal` CLI comes with the trainer (`just install`), log in
once with `uv run --project trainer modal setup`.

```sh
just install    # pipeline (Python 3.14, Open3D 0.20, Pydantic 2.13), trainer (Python 3.12, torch), viewer
just            # list the commands
just status     # config, tools, stage state
```

Every command runs from the repo root (or any folder below it); arguments pass through to the CLI, and
`just cli <command> --help` shows its options. Without just: `uv run --project pipeline scanmodel <command>`.

`settings.toml`: default scan and pose options; tool paths if not in the default places.

## Pipeline

**A Blender model from a scan**, nothing else (no splats):

```sh
just model fifth                 # the project data/fifth/ (default: settings.toml project)
just model fifth --open          # …and open model.blend in Blender; --glb also writes model.glb
just model fifth --poses posegraph   # skip refine (~8 min less on a 15 min scan; worse walls, re-fuses when switched back)
just open plan --project fifth   # blend, plan, report, glb or data
```

It runs ingest → poses → refine → fuse → structure → eval → build (the stages of `just all`; only what
changed), prints each stage's time and where the model, plan render and eval report are.

Bundles go in `data/<project>/raw/<session>/`. Select a scan with `--project NAME` on a command, or
the default (`project` in `settings.toml`). No per-scan config file is required; the project folder sets
every default path. `--raw` and `--skip` explicitly select inputs; effective options are recorded in
generated manifests.
Ingest removes the output of sessions it no longer reads, so later stages only see selected sessions.

```sh
just all --project fifth
just splat --project fifth
just gs train --project fifth --name baseline
just export --project fifth
just report --project fifth
```

All stage-table paths below are relative to `data/<project>/`. Structure writes `scene/model.json`;
processing stages use `derived/<stage>/`; splat datasets use `splat/datasets/default/` (or a named
dataset), trained splats use `splat/runs/<run>/`, Blender outputs use `build/`, and viewer-ready artifacts
use `export/`. Named camera views are `scene/views.json`. See the [layout guide](docs/scene-portability.md).

```sh
just all                    # ingest → poses → refine → fuse → structure → eval → build (only what changed; --force)
just all --poses posegraph  # …without refine: the pose graph's poses throughout, as before P7 (`poses` in settings.toml)
just report                 # the eval report; `just blend` opens the .blend
just splat                  # the splat dataset (refined poses, the points + mesh fused from them)
just cli meshcheck         # the fused mesh vs each frame's own LiDAR at its pose (--dataset D: its frames)
just splat --train          # nerfstudio dataset + msplat splats, floaters pruned with the LiDAR
                            # (--trainer brush|gsplat, --name run, -n steps, --no-prune, --trainer-arg=--sh-degree=2)
just cli view add kitchen-door "<viewer URL>" --note "…"   # a named view for splat comparisons
just cli view add sofa --image <session>_002940   # …or at a captured frame/still's pose
just export                 # viewer data: model.glb, every trained splat, named views into viewer/public/data/scans/<scan>/
                            # (each project adds its own scan; --viewer-default: the scan the viewer opens first)
just viewer                 # web viewer at http://localhost:5173 (or the next free port; ?scan=fifth: another scan)
                            # C (or the panel's Copy view): URL of the exact view -> `just cli view add NAME "<url>"`
just cli ingest --from stray ~/Downloads/stray-capture   # import a stock Stray Scanner capture
just cli poses             # any single stage: ingest, poses, refine, fuse, structure, eval, build
```

**Poses.** Every stage after `poses` reads `scanmodel refine`'s bundle-adjusted poses (`derived/refine/`) unless
`--poses posegraph` (on `all`, `fuse`, `splat`, `meshcheck`, `view add`) or `poses = "posegraph"` in `settings.toml`
picks the pose graph's (`derived/poses/`); the outputs go to the same folders either way, and with
`posegraph` the stage keys are the ones from before refine was the default (P6-era results). Structure and
build follow the poses that `derived/fuse/` was fused from; `splat` refuses fused points from other poses.
If refine can't run (a session without `video.mov`, fewer than 200 usable tracks), `scanmodel all` stops with
that message: it never falls back to the pose graph by itself. Fuse's manifest and the splat dataset's
`transforms.json` record `pose_source` / `mesh_pose_source` (`refine`, `arkit-posegraph` or `arkit`).

| Stage | Reads | Writes |
|---|---|---|
| `ingest` | `raw/*/`, `raw/*/*/` (or `--raw`, minus `--skip`); stills' focal length measured against the video and corrected per session (`scanmodel.stillcal`: ARKit's 12 MP intrinsics are ~4.8 % off); frames and stills from before tracking settles at a session's start left out (`scanmodel.checks.settled_from`), and those of a pose excursion (`scanmodel.checks.excursions`: ARKit's pose jumps away and back within 5 s, tracking "normal"; at most 3 s dropped) or of `scene/overrides.json` `drop_frames` (`{"<session>": [[first idx, last idx], …]}`) | `derived/ingest/<session>/report.json` (`dropped`: each window left out, its frames and stills), poses in pipeline axes |
| `poses` | ingest, `scene/overrides.json` (`frame`); anchors logged in a window ingest dropped are ignored; drift correction (`scanmodel.align`: keyframe pose graph with floor-height constraints, SIFT + LiDAR loop closures, anchors; main floor at z = 0; warns when a session pair's rejected closures agree on an offset > 10 cm, or a session has no closure to any other) | `derived/poses/poses.json`, `frames.jsonl`, `stills.jsonl` (scene frame), `walls/<session>@<s>.npz` (wall points per 30 s window); the manifest's `pose_graph.closures_by_session_pair` (candidate, accepted, rejected closures, median offset of the rejected) and `sessions_without_closures` |
| `refine` | poses, ingest's raw ARKit poses, video, stills, depth; bundle adjustment (`scanmodel.refine`): first a coarse round for drift the pose graph left: **trajectory cuts** (a trajectory, sessions that continued one ARSession together, may be cut where ARKit had nothing to hold on to (30 s+ without a coarse pair to a view more than 30 s away along it or to another trajectory, at least 8 s) or has a gap in its frames (ingest's dropped windows, lost tracking); a cut is admitted when the coarse pairs crossing it agree on one jump > 10 cm and outnumber those that agree with no jump 3:1, from 3+ bins on each side and two sessions or views 45° apart after it, and the jump makes the most pairs consistent; everything after it is moved back (the trajectory's start holds the scene frame), ARKit's motion over the stretch is loosened, the stretch's frames and stills are left out (their poses are a guess), and RoomPlan moves with the sessions (`session_to_scene`; a session cut in two gets `session_to_scene_parts`, structure places each element with the part that puts it on the fused points); then (cross pairs matched with wide windows and a 1 m / 6° gate; a session's 4 s bins whose pairs agree on one offset > 10 cm, outnumber its tight pairs, come from two other sessions, and persist over three bins are drift: ARKit's relative motion is loosened over the unseen stretch before them and a coarse pose graph moves the start poses; nothing changes otherwise); RootSIFT matched between neighbours and across passes / the lap (guided by the poses and LiDAR, PnP- and depth-gated), tracks, gtsam BA with Huber reprojection factors, LiDAR range priors, ARKit relative motion, loose gravity, floor heights (LiDAR floor cells as height landmarks, each floor level held level, the main floor back at z = 0: what the pose graph's floor links did; `--floor-sigma 0` turns them off); corrections interpolated to every frame; first 3 s after an unsettled session start dropped. Skipped with `--poses posegraph` | `derived/refine/poses.json`, `frames.jsonl`, `stills.jsonl`, `walls/` (same format as poses), `refine.json` (pairs, tracks, reprojection error before/after per distance band for cross- and same-pass tracks, corrections per session and view, `session_pairs`: per session pair the accepted and rejected cross pairs and matches and the median offset of the rejected ones (`drift`: they agree on > 10 cm, also a warning), `coarse`: the coarse round's pairs, `cuts` (every candidate cut: where, crossing pairs loose / tight / agreeing, the jump, its gain, admitted), its own `session_pairs` (what the fine gate would reject), the bins that agree on drift, the drift `spans`, whether it was applied and how far it moved each session, `floor`: floor residuals, how far apart sessions see the same floor cells (`session_offset_mm`) and the camera height change as a plane over the scan), `refine.png`, `features/` (SIFT cache) |
| `fuse` | refined poses (or `--poses posegraph`) + depth | `derived/fuse/dense.ply`, `points.ply`; `pose_source` in its manifest |
| `structure` | the fused points' poses (+ their `walls/`), points, `room.json`, overrides; RoomPlan walls and enclosed areas in another session's room are left out (`scanmodel.ownership`: votes of the sessions' cameras, thinned by motion, without a session's doorway dwell at its end; RoomPlan keeps adding what it sees from the next room's doorway or through glass), flagged; an object seen by two sessions keeps the copy of the session whose room it is in, and every object gets the room it stands in | `scene/model.json` |
| `eval` | model.json, points, the poses they were fused from | `derived/eval/report.html`, `metrics.json` (walls: the headline `observed_*` over faces a session saw (`faces_observed` in model.json), `unobserved_*` for the back of one-sided walls, `mean_m` / `p95_m` over all faces as before 2026-10-09, `gross`: faces whose nearest surface is 10-50 cm off; `doubled` (`scanmodel.consistency`): every two 30 s windows that saw the same walls from the same side compared by their own LiDAR (`walls/`), groups of pairs agreeing on one shift, `split_max_m` and `split_warning` (>= 10 cm: the poses are split into blocks, the walls are there twice; a warning after `scanmodel all`); `floors`: each session's floor height per room, its spread, and rooms meant to be level vs the largest room; also in `scanmodel status` and after `scanmodel all`), `history.jsonl` |
| `splat` | poses, stills, video, depth, fused points and mesh | `splat/datasets/default/` (dataset v2, contract in `trainer/README.md`): `transforms.json` (training + every-8th `interp` frames), `split.json` (`interp`; `extrap` = the `-lap` session, left out of training; `--extrap-session`; `--poses posegraph` reads `derived/poses/` (needs `derived/fuse/` fused from them too), `--dataset NAME` writes `splat/datasets/NAME/`; `pose_source` / `mesh_pose_source` in `transforms.json`: `refine` or `arkit-posegraph`), native LiDAR `depth/` + `confidence/`, `points.ply`, `mesh.ply`, `views.npz`, `test_poses.json` (each test frame's pose re-solved against the training frames, for the eval; `--no-test-poses`; `scanmodel test-poses` adds it to an older dataset, ~1 min); `--train`: `splat/runs/<name>/splat.ply` (`--name`, default the trainer; scene frame), `train.log`, PSNR/SSIM and splats pruned in `manifest.json`. msplat prunes splats in LiDAR free space during training (`scanmodel.freespace`, every 1500 steps up to 15k; `prune.json`); Brush can't; gsplat trains on a cloud GPU through `gs-trainer cloud train` (`cloud.log`). Named views: `scene/views.json` of the selected project (`scanmodel view add/list --project NAME`) |
| `build` | model.json, stills | `build/model.blend`, `preview.png` |
| `export` | model.blend, splats, `scene/views.json` | `export/model.glb`; in `viewer/public/data/scans/<scan>/` (the viewer of the repo the code runs from; `<scan>` = the project, e.g. `third`, or `--scan`): `model.glb`, `splats/<id>.ply` (links to every trained splat), `splats.json` (their parameters and metrics, for the viewer's layers panel; `--trainer` picks the default, else the previous one stays), `views.json` (the project's `scene/views.json`); links added by hand and their entries are kept while their file exists. `viewer/public/data/scans.json` lists the exported scans and the default one (the first exported, or `--viewer-default`); other scans' folders are left alone |

Scene frame: metres, z up, dominant walls aligned with x/y. The origin uses the door nearest the first
camera (or the first camera's XY if no door exists), with the estimated main floor at z = 0. These axes
are not geographic east/north. Set `frame.origin_door` in overrides to choose a particular door.
Hand fixes go in `scene/overrides.json` (merge patches keyed by element id); they are applied last,
so re-running never loses them. After changing a model in `pipeline/scanmodel/models/`, run `just schema`.

## Checks

```sh
just check                  # pipeline: pytest, pyright, ruff; viewer: type check + build
```

The tests build a synthetic two-room flat (ray-cast LiDAR depth, RoomPlan JSON with a few cm of error,
an arbitrary ARKit world frame) and run the whole chain against ground truth.

## Benchmark

`just bench` checks a rule change against captures with ground truth, not just our own scan
(thresholds and their risks: [docs/research/2026-10-07-threshold-audit.md](docs/research/2026-10-07-threshold-audit.md)).

```sh
just bench-download                 # once: 4 rooms + laser scans, ~11 GB fetched, ~3 GB kept in data/external/
just bench                          # ~10 min: ARKitScenes subset (pose graph off and on) + every project's checks
just bench --project X --no-arkitscenes          # one project's checks, on its scene/model.json (seconds)
just bench --project X --no-arkitscenes --rerun  # the same on a fresh run of its raw/ (no overrides)
```

- **ARKitScenes** (Apple, iPad Pro LiDAR + ARKit poses, Faro laser scans, annotated boxes; non-commercial
  licence, data never committed): ingest → poses → fuse per room in `data/derived/bench/work/`, then
  fused points vs the laser (accuracy, completeness), drift per 20 s window, floor/ceiling heights
  (`scanmodel.rooms.fit_heights`) vs the laser's, and object boxes refitted (`scanmodel.boxes`) after RoomPlan-like
  noise. No RoomPlan here, so walls and rooms aren't tested.
- **Our own scans**: hand-measured facts in `data/<project>/checks.json`, reported pass/fail; then
  `just bench --project X`. They are never read by the pipeline. Kinds: `floor_delta` (room vs
  `relative_to`), `ceiling_height`, `room_width`, `room_length`; rooms by name or id. Facts only: the scene
  and raw captures are the project's (optional `"raw"`: another bundle folder for `--rerun`). Example:
  `{"name": "third", "checks": [{"id": "conservatory-floor", "kind": "floor_delta", "room": "conservatory",
  "relative_to": "living", "expected_m": -0.06, "tolerance_m": 0.04}]}`. A project without `scene/model.json` yet
  needs `just all --project X` first (or `--rerun`).

Output: a table, `data/derived/bench/<date>.json` and a line in `data/derived/bench/history.jsonl`
(with the git commit), so runs before and after a change can be compared. Exit code 1 if a check fails.
