# Running a different scene

The code is reusable, but this repository also contains the history and outputs of one flat. Room names
such as kitchen, living and conservatory in the plans are observations about those scans, not required
room names. The repo's flat-blender and the app's FlatBlender are product names.

## One folder per scan

```text
data/
├── fourth/
│   ├── raw/<session>/          # original capture bundles
│   ├── raw-archive/            # excluded historical captures, when present
│   ├── scene/
│   │   ├── model.json          # generated model
│   │   ├── overrides.json      # manual edits
│   │   └── views.json          # named camera views
│   ├── derived/<stage>/        # ingest, poses, refine, fuse, structure, eval
│   ├── splat/
│   │   ├── datasets/default/  # or another named dataset
│   │   ├── runs/<run>/         # trained splat, config, logs, metrics
│   │   └── analysis/           # historical comparisons, evals and caches
│   ├── build/                  # Blender model and previews
│   ├── export/                 # viewer-ready artifacts
│   ├── analysis/               # screenshots, scripts, walk videos
│   ├── logs/
│   └── checks.json             # measured ground truth, when available
├── fifth/
│   └── ...
└── external/                   # shared benchmark datasets
```

No `project.toml` is needed. Directory names determine paths; command options determine processing
settings, which generated manifests record. Select a scan explicitly:

```sh
just all --project fifth
just splat --project fifth
just gs train --project fifth --name baseline
just export --project fifth
just report --project fifth
just blend --project fifth
just bench --project fifth --no-arkitscenes
```

The default scan is `project` in the tracked `settings.toml`; `--project` overrides it. A project is
required: there is no layout outside `data/<project>/`. Project selection controls pipeline and trainer
defaults. Explicit `--data`, `--out` or `--raw` options still choose a different input/output. Cloud evaluation
with `--run` uses the dataset recorded for that run, unless `--data` explicitly changes it.
Named evaluation views are opt-in: pass `-- --views data/fifth/scene/views.json` to cloud eval.

Put a new capture in `data/my-scene/raw/<session>/` and use `--project my-scene`. Use explicit options
(`--skip`, ...) or scene overrides for corrections. Build/eval/export consume existing outputs: run the new
scene's pipeline before using them.

Create overrides and named views inside the scan. Do not copy old element IDs,
`drop_frames`, camera matrices, or frame overrides across scenes; those refer to a specific capture's
coordinate frame and sessions. Camera views for trainer evaluation must be supplied explicitly with
`--views`; cloud evaluation no longer injects the original flat's views automatically.

## What is intentionally scene-specific in git

| Tracked files | Meaning |
|---|---|
| `trainer/sweeps/p6/`, `p7/`, `p8/`, `r1/`, `r2/` | Recorded experiments; inspect names, settings and dataset arguments before reusing |
| `docs/plans/`, `docs/research/` | Flat-specific results, budgets and rationale, alongside general design notes |
| `blender/tests/`, `pipeline/tests/synth.py`, `trainer/gs_trainer/synth.py` | Deliberate test fixtures, not runtime scene defaults |
| `scene/*.schema.json`, `scene/schema.json` | Reusable format schemas |

All project data and viewer exports are gitignored. Back up raw captures and manual overrides/views;
the latter cannot be regenerated from raw captures. The viewer publication folder under
`viewer/public/data/` is rebuildable from project exports. Shared schemas remain in repo `scene/`.

## Migration (history)

Scans second–fifth were moved from the old shared layout (`data/raw/`, `data/derived/`, the repo's
`scene/`) into this one once, on 2026-10-11, with one-off scripts that are not kept. Scan 3's scene is
in `data/third/scene/` (no tracked copy). Stored local paths in cloud run
metadata are resolved to the new layout when pulled or evaluated. Dated plans, logs and research notes keep
the paths and names of their time.

## Remaining modeling assumptions

The structure layer targets RoomPlan-style indoor architecture: straight wall segments, gravity,
an estimated main floor, room outlines and ceilings. Yaw follows dominant wall directions; it is not
a compass measurement. Without RoomPlan doors, the origin uses the first camera's XY. The automatic
door choice is a heuristic, not detection of a building's front door.

Floor and ceiling fitting uses thresholds for ordinary room heights, horizontal floors and modestly
sloped ceilings. Wall thickness has a default when only one side was observed. These are configurable
algorithm assumptions, not measurements guaranteed for tiny rooms, curved architecture, ramps or
outdoor scenes. The splat/depth stages can represent more general geometry, but successful splat
training does not validate the parametric structure model.

Session registration needs overlap or a shared ARKit world map; disconnected captures cannot be placed
reliably just from room labels. Refine needs video and enough visual tracks; use `--poses posegraph`
explicitly for inputs it cannot refine. Splat's automatic extrapolation holdout recognizes the `-lap`
session naming convention when other sessions exist; use `--extrap-session` to choose another holdout.
Evaluation thresholds and historical best trainer settings should be checked on each new dataset.
