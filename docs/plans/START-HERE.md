# Start here

This page records the original flat's scans and experiments. For another physical scene, start with
[Running a different scene](../scene-portability.md); the paths, views and results below are scene-specific.
Local files now live under `data/third/`, `data/fourth/` and `data/fifth/`. Select one with `--project`
(default: `project` in `settings.toml`). Dated plans/logs retain historical paths and names (`flat` CLI,
`flat_gs`, `flat.json`: now `scanmodel`, `gs_trainer`, `model.json`).

Status 2026-10-11. The entry point for a new session: where things are, the best results, how to work, what's
next. Read this, then the document for what you touch:

- [2026-10-06-scan-to-model.md](2026-10-06-scan-to-model.md): the plan. Design, the capture app and its versions,
  bundle format, pipeline stages, phases, capture protocol, what each scan taught us, pipeline decisions and
  negative results.
- [2026-10-08-gsplat-cloud.md](2026-10-08-gsplat-cloud.md): the gsplat trainer on Modal. Decisions, the current
  recipe, every scored run, what helped and what didn't, open issues, operating notes.
- Commands and contracts: `README.md` (pipeline, stages and their files), `trainer/README.md` (cloud trainer,
  dataset v2), `capture/README.md` (the app, bundle v1, timing line).
- Background research: `docs/research/` (drift and wall thickness, threshold audit, splat trainers).

Git history keeps the step-by-step logs that used to be here. Remote: private `szerintedmi/flat-blender` on GitHub;
commit only as `szerintedmi <szerintedmi@gmail.com>` (the repo's git config; its credential helper pushes as that
account whatever `gh`'s active account is).

## State

| Subsystem | State |
|---|---|
| Capture app (`capture/`, FlatBlender) | **0.5.0 (8)** (version bump only; the code is 0.4.0 (8), **device-tested on scan 5: every check passed**) (perfect 60 Hz stream, no tracking restart, delegate p99 ≤ 1.3 ms, 0 drops; details in the plan's "App versions and why"). Coverage dots with free-space carving; capture sounds; Run = Release; no shutter in scan/lap sessions; flicker prompt at Lock; `timing …` line in the session notes |
| Pipeline (`pipeline/`, `just cli …`) | ingest → poses (pose graph) → **refine** (bundle adjustment) → fuse → structure → eval → build → export. **Refine cuts a trajectory where ARKit drifted unseen** and the sessions after it inherited the jump (scan 5's split, 2026-10-10: plan, "Trajectory cuts"); **eval reports doubled surfaces** (`scanmodel.consistency`: 30 s windows that put the same walls in different places; a warning after `just all`). **Ingest drops ARKit pose excursions** (out and back within 5 s, tracking "normal") and `overrides.json` `drop_frames`. **Structure** fits faces only to surfaces facing their room and lets a face's own depth outvote far cameras; **eval's headline is over seen faces**. `just splat` writes dataset v2 with frame/still selection and LED debanding (`--deband auto`). `data/third/scene/model.json` is scan 3, regenerated with these fixes. Textures (phase 3), `register` and SPZ are not started |
| Splat training (`trainer/`, `just gs …`) | gsplat on Modal, driven from the Mac. Best: **`s3-refined-fused-select-geo`**. msplat and Brush still train on the Mac (`just splat --train`) but are no longer the reference |
| Viewer (`viewer/`) | Spark + three.js, layers panel, walk controls, `?pose`/`?fov`, C copies the view URL. **Several scans side by side**: `just export` writes the project's scan into `viewer/public/data/scans/<scan>/` (model.glb, splats, `views.json` from its `scene/views.json`) and lists it in `scans.json`; `?scan=fifth` or the panel's Scan list switches, the Named view list jumps to a view (`?view=`). Default scan `third` (`--viewer-default` changes it), default splats `s3-refined-fused-select-geo` / `s5-drift-select-geo-bilagrid`. Exports before 2026-10-10 (`data/splats.json`, scan 4's hand links) still load when there's no `scans.json`; scan 4 can now be exported from its own root instead |

Scans in use (the older ones are in the plan's "What each scan taught us"):

| Scan | Raw | Derived root | What |
|---|---|---|---|
| 3 (`third`, 2026-10-07, app 0.2.0 (5)) | `data/third/raw` | `data/third/` (default project) | Ground floor: conservatory → living → hallway → kitchen with Next room, doorway sweeps, a 40 s lap. 186 stills. The scan in `data/third/scene/model.json` |
| 4 (`fourth`, 2026-10-08, app 0.3.0 (6), Debug) | `data/fourth/raw` | `data/fourth/` (`--project fourth`) | Living room only, 5:19 + a 38 s lap, 1/240 s under LED light. Its own `data/fourth/scene/model.json` and 11 named views in `data/fourth/scene/views.json` (gitignored) |
| 5 (`fifth`, 2026-10-09, app 0.4.0 (8)) | `data/fifth/raw` (7.0 GB) | `data/fifth/` (`--project fifth`; logs in `logs/`) | Conservatory, **garden (outdoors)**, living, hallway, kitchen + 47 s lap; 15 min, 310 auto stills, 1/240 s, one exposure lock. **Was split into two rigid blocks until 2026-10-10**: the living session drifted 55 cm / 1.4° unseen (t ≈ 106–158 s, camera high on plain walls and ceiling) and the hallway, kitchen and lap inherited it; its 1 s "excursion" was ARKit briefly relocalising. Refine's trajectory cut now re-joins it (the lap's "48 cm drift" was this split; the lap is fine). The derived root is re-run with the cut (`before-split-fix/` keeps the old summaries, old refined poses and walls); the splat `s5-drift-select-geo-bilagrid` was trained on the split poses. Named views: scan 3's 12 moved into scan 5's frame (`data/fifth/scene/views.json`, Next 1). Kept in its own root; replacing scan 3 is undecided |

Scans first–fourth live in the old app's container, `dev.flatblender.FlatBlender.pp`; builds from 0.3.0 (7) on are
`dev.flatblender.FlatBlender`, a separate container (a wrong id fails with "CoreDevice.ActionError error 3"), where
scan 5 is. Scans are pulled over USB with `devicectl`, no export needed (7 GB for scan 5's 15 min took 4:47):

```sh
xcrun devicectl list devices                      # the iPad's identifier
xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
  --domain-identifier dev.flatblender.FlatBlender --source Documents/Projects/<project id> --destination data/<name>/raw
```

## Best results

**Geometry** (`just cli eval`, seen wall faces vs the fused points; target p95 20 mm, not met; all faces, the
pre-2026-10-09 measure, beside it):

| | seen faces mean / p95 | all faces | Other |
|---|---|---|---|
| Scan 3 | **12.5 / 52.7 mm** (26 of 30 measured; re-run 2026-10-11 with the trajectory cuts: poses bit-identical; fuse isn't deterministic: three fuses gave 12.5 / 52.8, 13.1 / 57.9 and 13.2 / 58.2, the last is on disk) | 13.8 / 59.8 (was 17.1 / 64.3; pose graph only 23.3 / 86.9) | meshcheck (fused mesh vs each frame's own LiDAR) median 6.5 mm; floors consistent: hallway / kitchen vs living +0.3 / +0.1 cm |
| Scan 4 | **7.3 / 24.9 mm** (15 of 16) | 9.2 / 41.4 | re-run 2026-10-11: poses bit-identical, 7.3 / 24.7; meshcheck median 8.4 mm |
| Scan 5 | **8.3 / 26.6 mm** (29 of 31; 26.6–27.0 over three fuses; split poses: 8.3 / 25.4, 27 of 30) | 10.0 / 40.4 (split: 9.7 / 33.0; was 20.6 / 90.4) | joined (trajectory cut, 2026-10-10): no doubled surfaces (split poses: 44 cm, 46 window pairs); meshcheck median 6.8 mm (7.1); floors: sessions within 1.6 cm (garden), hallway / kitchen / living within 0.2 cm, conservatory vs living −10.0 cm (scan 3: −10.4) |

`just cli refine`: cross-pass track residual scan 3 3.39 → 0.86 px, scan 4 3.81 → 0.81 px; the dining-wall painting's
per-view offset 29 → 1.8 mm median.

**Splats** (gsplat, 30k steps at 960 px, `trainer/sweeps/p6/select-geo.toml` settings; scan 4 adds `--color
bilagrid`). interp = every 8th frame, extrap = the closing lap (never trained on). The scans' test sets differ:
compare within a scan only.

| Run | interp PSNR / SSIM / LPIPS / AbsRel | extrap PSNR / SSIM / LPIPS / AbsRel |
|---|---|---|
| scan 3 `p6-select-geo` (pose graph) | 23.07 / 0.827 / 0.416 / 0.034 | 17.72 / 0.692 / 0.464 / 0.076 |
| scan 3 `s3-refined-select-geo` (refined poses, priors fused from the pose graph) | 24.62 / 0.866 / 0.338 / 0.024 | 19.71 / 0.753 / 0.370 / 0.031 |
| **scan 3 `s3-refined-fused-select-geo`** (refined poses and priors: what the default pipeline makes now) | **24.68 / 0.865 / 0.339 / 0.024** | **19.94 / 0.756 / 0.371 / 0.029** |
| scan 4 `s4-select-geo-bilagrid` (pose graph, debanded) | 22.60 / 0.844 / 0.317 / 0.020 | 19.04 / 0.736 / 0.379 / 0.052 |
| scan 4 `s4-refined-select-geo-bilagrid` | 23.85 / 0.878 / 0.264 / 0.015 | 20.69 / 0.802 / 0.306 / 0.029 |
| scan 4 `s4-refined-fused-select-geo-bilagrid` | 23.92 / 0.877 / 0.264 / 0.012 | 20.64 / 0.800 / 0.309 / 0.030 |
| scan 5 `s5-select-geo-bilagrid` (pre-fix dataset: split, plus the lap's late part 48 cm off) | 21.39 / 0.806 / 0.336 / 0.021 | 14.93 / 0.596 / 0.527 / 0.144; at re-solved poses (34 frames) 18.86 / 0.657 / 0.330 / 0.027 |
| **scan 5 `s5-drift-select-geo-bilagrid`** (excursion dropped, the lap moved onto the conservatory block; still split in two blocks 0.45 m apart: the living room's south wall looks 0.5 m thick. The default pipeline now makes a joined dataset, not trained yet) | **21.21 / 0.801 / 0.342 / 0.022** | **17.75 / 0.723 / 0.385 / 0.052**; at re-solved poses (40 frames, none moved > 2 cm) 19.00 / 0.667 / 0.325 / 0.029 |

Still messy in every run: kitchen-counter-side and kitchen-to-hallway (missing viewpoints), kitchen-stairs, and
reflections in the conservatory glass. `pose_opt` on top of the best run was tried and not adopted (gsplat doc).

## How to work

```sh
just model <project> [--open] [--glb]   # data/<project>/raw -> build/model.blend only (no splats)
just all                            # ingest → poses → refine → fuse → structure → eval → build (only what changed)
just splat                          # dataset v2 into splat/datasets/default/ (refined poses, --deband auto)
just bench --project third --no-arkitscenes   # our hand-measured checks (data/<project>/checks.json)
just export --trainer <run>; just viewer
just all --project fourth           # --project on any command for scan 4 or 5
```

Cloud loop (details in `trainer/README.md`):

```sh
just gs ps                                                    # runs, the real Modal bill, live apps: always first
just gs push --data <dataset dir>                             # upload under its hash, make the priors
just gs sweep trainer/sweeps/<x>.toml --data <dataset dir>    # one detached run per config
just gs pull <run> --out data/third/splat/runs/<name>
just gs eval --run <run> --out data/third/splat/analysis/eval/<name>   # ~5 min
uv run --project trainer python -m gs_trainer.evaluate compare data/third/splat/analysis/eval/<a> … --out <dir>
```

- **Scan 4 and other roots:** `gs eval --run` scores on the run's own dataset (before 2026-10-09 it defaulted
  to scan 3's: PSNR ~7). Cloud eval no longer automatically adds scan 3's `data/third/scene/views.json`.
  Render scene-specific named views locally with an explicit views path: `uv run --project trainer python
  data/fourth/analysis/views/scripts/render_views.py PLY VIEWS.json OUT` (5–18 s per view on the CPU).
- **Budget:** October's Modal spend is **$45.81** by the bill (2026-10-09, 23:00), against the **$52 cap**
  the user approved for October ($40 + $6 + $6 for scan 5). Beyond it, ask first. A scan 5 run ≈ $2.3 (57 min,
  2.7M splats) + $0.4 eval. A scan 3 run costs ≈ $1 on an L40S plus
  ≈ $0.25–0.4 for its eval; a scan 4 run ≈ $2 (3.5M splats, ~1 h 45). The bill lags about an hour. One run at a
  time, only through `just gs` (ad-hoc Modal apps once cost ~$13 unseen).
- **GPU fallback caveat:** GPU functions take the first free of L40S → L4 → H100. An H100 costs about 2× an L40S,
  but `just gs ps` estimates and run manifests still say L40S, so fallback runs under-report by ~2× (open bug).
  Pin `gpu = "L40S:1"` in a sweep TOML to avoid the fallback.
- Report a run's end when it happens. Commit at milestones; keep this file current, and put lasting results and
  decisions in the plan or the gsplat doc, not here.

## Next

In order:

1. **Retrain scan 5 on the joined dataset?** (needs the user's OK: ≈ $2.7, October is at $45.81 of $52).
   `data/fifth/splat/datasets/default` is regenerated from the joined poses (trajectory cut, 2026-10-10);
   every scan 5 splat so far (`s5-drift-select-geo-bilagrid`, the viewer's default for scan 5) has the hallway,
   kitchen and stairs 0.45 m off the rest. The walk videos (`data/fifth/analysis/walks/walk_{cat,human}_splat.mp4`,
   Spark + Playwright, `walks/scripts/render_walk.py`) were rendered from a hand-joined copy of it
   (`walks/scripts/fix_splat.py` moved 565k of 2.69M splats by (−0.198, +0.402, 0); POC → splat is then the
   translation (+0.053, −8.853, +0.113)). The pipeline's join agrees with that hand join to 1 cm ((−0.188, +0.405)
   m and −1.2° for the front block vs the rest) but also moved the rest by (−0.005, +0.045) m, so after a retrain
   re-fit POC → splat (`walks/scripts/align_poc.py`) and check the named views (scan 3 → 5 transform below).
2. **Is scan 5 the new default?** Tools are in: the viewer shows both scans (`?scan=fifth` or the Scan list;
   `just export --viewer-default` from a root makes it the default; scan 3 is the default now), scan 5 has scan 3's
   12 named views moved into its frame (`data/fifth/scene/views.json`; scan 3 → 5: −179.55° about z,
   (1.31, 2.00, 0.11) m, residual median 16 mm), and eval.json has per-room numbers. Findings: scan 5's interp PSNR
   is 2.5–4.3 dB lower in **every** room (not the garden), but LPIPS is close or better (conservatory 0.334 vs
   0.439), so colour (daylight vs night, bilagrid) may drive PSNR. Side by side
   (`data/fifth/analysis/compare-s3-s5/sheet_{1,2}.jpg`): scan 5 better at kitchen-window,
   kitchen-counter-side, kitchen-to-hallway, hallway-front-door; scan 3 better at hallway-to-living (scan 5 never
   filmed there) and the kitchen-counter drawers; even elsewhere; the two conservatory views need re-picking in
   scan 5 (doors in another state). Walls: scan 5 clearly better. Open: the user's look in the viewer, a
   colour-matched PSNR, then the decision (model and splat may favour different scans).
3. **Left from the split fix:** the old coarse round's ~3.7 cm "bend" of the other sessions came with its
   spurious lap correction; with the cut the conservatory / garden / early-living block sits (−0.5, +4.5) cm from
   the old refine and the rest moved as one block. The lap's floor in the garden is still 1.3–1.9 cm above the
   conservatory session's (eval floor spread 1.6 cm). The living session's 106–158 s (3,144 frames, mostly ceiling) are left out: their
   poses are a guess. Mid-lap frames 1296–1680 didn't re-solve (blank walls, blur): re-check after a retrain.
   The pose graph still can't link fast, low-texture laps (it warns), and it leaves splits like scan 5's to
   refine (`--poses posegraph` keeps the split; eval's doubled-surface check warns).
4. **Extrap below scan 3** (17.75 vs 19.94) after the fix: the lap walks close and sideways past hallway / living
   walls the training views saw from 0.3–1.8 m away (coverage), plus a person on the sofa in frame 552.
5. **Structure, from the scan 5 diagnosis:** thickness from the partner-face gap when unmeasured (6–45 cm,
   tagged inferred: the 30 cm living / conservatory wall gets 10 cm); drop a RoomPlan face its own session's depth
   doesn't support (the garden's through-glass phantom on living furniture); scan 3's `wall-cfb345a2` has no
   room-facing points within 8 cm (a second refit pass over 10–50 cm); short faces whose refit is rejected as
   "turned 90°" could fit only the offset (`wall-26c8ee07`).
6. **Fix the GPU-fallback cost reporting** (`trainer/gs_trainer/cloud.py`: record the GPU the container really ran
   on). Pin `gpu = "L40S:1"` meanwhile.
7. **Bench floor check** (open problem 1): a slope-aware `floor_z`, or redefine the check.
8. **Densification** (a4): fix MCMC's divergence, or a sparser init, for fine detail.
9. **Viewer:** stay near the captured paths or fade to the mesh where coverage is low; SPZ export (~10× smaller;
   `vite build` copies every linked splat into `dist/`, ~10 GB with two scans).
10. Move `render_views.py` into the repo (`trainer/gs_trainer/`), and let the cloud eval take a views path.
11. Decide the flicker prompt (probably a silent log: debanding handles 1/240 s; scan 5 needed no lightening of the
    preview by its timings).

Then, from the plan: phase 3 textures (the largest feature left), the audit items for other flats, cross-session
wall thickness with the refined poses, and a backup of the raw scans (`data/` is gitignored: 14 GB of originals only on this Mac).

## Open problems

1. **Bench check "conservatory floor vs living −6 ± 4 cm" fails** on scan 3 at −10.4 cm (pose graph −8.5). The
   conservatory floor slopes ~1 % (every session agrees; LiDAR floor normal vs gravity 0.44°), and structure's
   `floor_z` takes the lowest strong level; at the room's centroid it is −9.6 cm. The hand measurement is probably
   at the step. Options: a slope-aware `floor_z`, or redefine the check (`data/third/checks.json`).
2. **`wall-4c36a2ee`** (scan 3, living east wall, the conservatory session's duplicate of it) fits a surface ~15 cm
   in front of the wall, probably furniture, tilted ~3°: the known limitation of one wall per session. Eval's
   gross list now flags it (−13.5 cm).
3. **GPU-fallback cost reporting** under-reports H100 runs ~2× (above).
4. **MCMC diverges** in our trainer once its budget fills (gsplat doc, "Open issues").
5. Eval p95 stays above the 20 mm target (64 mm on scan 3); thickness on unmeasured walls is still the 10 cm
   default.
6. Scan 4 had an 11 s tracking restart right after Record (now dropped by `settled_from`); scan 5 had none.
7. **Scan 5's split** (found 2026-10-10 while rendering splat walks: the living room's south wall twice, 0.5 m
   "thick" in the splat): the living session drifted 55 cm unseen and the sessions after it inherited it; the
   coarse round had blamed the lap. Fixed by refine's trajectory cuts, caught from now on by eval's doubled-surface
   check (plan, "Trajectory cuts"). The splat is not retrained yet (Next 1).
8. Anchors placed during an unsettled start aren't filtered like those in an excursion; a `drop_frames` range from
   idx 0 makes refine also drop 3 s as an unsettled start.
9. **Excursions can be relocalisations**: scan 5's 1 s "excursion" was ARKit briefly at the right pose after
   drifting. Dropping it is still right; a long excursion (> 3 s, kept) could carry the true pose and isn't used.
10. **Fuse isn't deterministic** (Open3D TSDF): identical poses, eval 12.5 / 52.8, 13.1 / 57.9, 13.2 / 58.2 mm on scan 3.
   Compare eval within ~1 / 5 mm, or pin the fusion (single thread / sorted integration) if it matters.

## Housekeeping

- The default splat datasets are regenerated with refine + floors + deband auto: scan 3's
  (`data/third/splat/datasets/default`) is done (`transforms.json` `pose_source` / `mesh_pose_source: refine`,
  `selection.json` debands conservatory, living and lap; 1,883 frames); scan 4's
  (`data/fourth/splat/datasets/default`) is done too (`pose_source` / `mesh_pose_source: refine`, 756
  frames, same 95 / 106 interp / extrap split as `dataset-refined`). Neither has been
  trained on yet (the best runs used the `dataset-refined-fused` datasets, made before the floor factors, and scan 3's with
  `--deband off`).
- Scan 5: `data/fifth/splat/datasets/default` is regenerated from the joined poses (2026-10-11; 2,282 images:
  2,008 frames + 274 stills, 1,900 train / 272 interp / 110 extrap, was 2,413; not trained on; the previous one's `transforms.json`, `split.json`, `selection.json`, `test_poses.json` and manifest are in
  `data/fifth/before-split-fix/splat/`). `data/fifth/before-split-fix/` keeps the split run's eval,
  refine (summaries, `frames.jsonl`, `walls/`: the doubled-surface check fires on them), poses manifest, fuse
  manifest and `model.json`; `analysis/split/` the diagnosis scripts (per-segment walls, ICP over time, the coarse
  pairs, the cut prototype); `analysis/walks/` the walk videos and the hand join. Before that: the hand-trimmed dataset `s5-select-geo-bilagrid` trained on is deleted (its cloud copy
  `d0c78e34877ebfd9` remains, with its `test_poses.json` uploaded for re-scoring). `data/fifth/before-excursion-fix/`
  keeps the first run's eval, refine and poses summaries; `analysis/lap5` and `analysis/eval5` the diagnosis
  scripts and results (caches deleted).
- Deleted 2026-10-09: the historical P7 A/B datasets (`dataset-refined`, `dataset-refined-fused`; their runs keep
  their PLYs), `data/derived/bench/work` (rebuilt by `just bench`) and scan 4's named-view render caches
  (`analysis/views/scripts/` kept). Still on disk if space is needed: the superseded splat runs in
  `data/third/splat/runs/` (msplat*, brush, final-*, r1-*, r2-*, p6-*; ~7.8 GB, linked from the viewer's layers panel).
- Analysis scripts and sheets from the sessions live in `data/fourth/analysis/` (ghosting, deband,
  close, priors, views) and `data/fourth/analysis/stills-analysis/`; they are not in git.
