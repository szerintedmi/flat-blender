# gsplat trainer on a cloud GPU

Started 2026-10-08, consolidated 2026-10-09 (the plan and its to-do list merged; git history has the step-by-step
log). Current state, budget and the "Next" list: [START-HERE.md](START-HERE.md). Commands and the module and
dataset contracts: `trainer/README.md`. Background: `docs/research/2026-10-08-splat-trainers.md`.

Goal: a splat trainer with what the Mac trainers lack (LiDAR free-space pruning, dense depth and normal
supervision, pose and colour correction, anti-needle regularisers) on a rented CUDA GPU, every step scripted
from the Mac (`just gs …`): deploy code, upload data, train, follow, download, stop. Nothing is clicked in a web
console after the one-time `modal setup`.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Trainer base | **Our own training loop** on the gsplat *library*: `rasterization`, `DefaultStrategy` / `MCMCStrategy`, `strategy.ops.remove`, `export_splats`. Pose, bilateral-grid and affine-colour modules adapted from gsplat `examples/` (Apache-2.0) | `simple_trainer.py` reads only COLMAP data, can't resume, prints a `\r` bar, and on `main` depends on unreleased library code |
| gsplat version | Pinned commit of `main` (`512d366`, 2026-09-19, version string 1.6.0). Fallback: release 1.5.3 | 1.5.3 has prebuilt wheels only up to torch 2.4. `main` has PPISP and 3DGUT (rolling shutter) if needed |
| Image | One `trainer/Dockerfile` on `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` (nvcc for gsplat, fused-ssim, fused-bilagrid); Modal builds from it. Kernels compiled for arch 8.9 / 9.0 (L40S, L4, H100; no A100) | One environment for any provider |
| Code layout | Separate uv project `trainer/` (package `gs_trainer`, Python 3.12, torch 2.9.1), `pipeline/scanmodel/freespace.py` shipped with it | `pipeline/` is Python 3.14 with Open3D and no CUDA |
| World frame | `normalize_world_space=False`: train in the scene frame (metres, z up) | The PLY lands in the scene frame; depth and LiDAR in metres; no SH rotation to undo |
| Images in memory | uint8 on the GPU at the training size when they fit, else decoded per step | msplat's RAM limit (450 px) is gone |
| Provider | **Modal** (ranking under "Services"). GPU functions take the first free of **L40S → L4 → H100** (L40S ran out for > 30 min once). The trainer is a plain CLI, so RunPod is a fallback with no trainer changes | Per-second billing, no machine to forget, code shipped on every run |
| Evaluation | One evaluator (`gs_trainer.evaluate`) for every trainer, on a shared split: `interp` (every 8th frame) and `extrap` (the closing lap, never trained on), plus renders of the named views | Each trainer's own PSNR uses different code; every-8th frames sit 0.5 s from a training view |

## Architecture

```
Mac                                                 Modal (CUDA)
just splat                (dataset v2)
just gs push   ── dataset + bulk.tar ───────────▶   volume flat-gs-data:/datasets/<hash>/
                                                    prepare(hash): mesh depth+normals, sigma (+ DA3 on --da3)
just gs train / sweep ── code snapshot, config ─▶   train(hash, run, config)   [timeout 3 h, retries 2]
               ◀── step= lines (logs) ───────────     checkpoints, progress.json, splat.ply
just gs pull   ◀── splat.ply, metrics, manifest ─   volume flat-gs-runs:/runs/<run>/
just gs eval   ── ply ──────────────────────────▶   evaluate(...)
just export    -> a viewer layer per run
```

`scanmodel splat --train --trainer gsplat` calls `gs-trainer cloud train …` and pulls into `data/<project>/splat/runs/<name>/`.

**Deploying while iterating.** Every run ships the current `trainer/gs_trainer` and `freespace.py`; nothing to push.
The image rebuilds only when `trainer/uv.lock`, the Dockerfile or the gsplat pin changes; a detached run keeps
the code it started with. Each run's manifest records the git commit, a hash of the uncommitted diff, the image
id, the GPU and the full config.

**Safety nets.** No machine: the container ends with `train()`. A 3 h timeout per attempt, `retries=2`. `just gs
pull` and `train` list anything still running. The Modal workspace budget stops spending at its limit. Volumes are
free under 1 TiB. Always use `just gs build` / `just gs gpu-test`, never ad-hoc Modal scripts (their apps were
invisible to `ps`; ~$13 of builds went unseen once).

**Privacy.** The dataset is photos of a home: data-centre GPUs only (Modal, RunPod Secure Cloud), private
volumes, no marketplace hosts; `just gs gc --all` deletes everything.

## The current recipe

`trainer/sweeps/p6/select-geo.toml` (as `refined-select-geo.toml`, `p7/refined-fused-select-geo.toml`):
DefaultStrategy, SH 3, 30k steps at 960 px, LiDAR free-space pruning every 1,500 steps to 15k (`min_free 3`),
`--depth-loss --depth-lambda 0.2` (EdgeAwareLogL1 on LiDAR, confidence ≥ 1 then 2 after 7k), `--normal-loss`
(against the mesh normals), `--flatten-lambda 1.0`, `--scale-ratio-lambda 0.1`, `--antialiased`. Scan 4 adds
`--color bilagrid`. Evaluated without `--antialiased` (as all the reference numbers were), `--samples 8`.

Input (`scanmodel splat`, dataset v2): refined poses, points and mesh fused from the same poses; frames selected by
movement (a slot every 0.12 m / 6° or 1 s, the sharpest frame per slot against the median of all frames within
±1 s, blurry picks dropped unless they're the only view); blurry stills dropped (scan 3: 13 of 186); stills'
brightness matched to the nearest video frame (gains ~0.92: stills ~8 % brighter, blue most); `--deband auto`.

Best runs: START-HERE, "Best results".

## Results

**Development runs on scan 3** (pose-graph poses, the first dataset v2; i = interp 188 frames, x = extrap 85;
not comparable with the later test sets):

| Run | Steps | Splats | iPSNR | iSSIM | iLPIPS | iAbsRel | xPSNR | xAbsRel |
|---|---|---|---|---|---|---|---|---|
| r1-base | 7k | 1.01M | 19.71 | 0.786 | 0.531 | 0.180 | 16.80 | 0.118 |
| r1-base-full (1920 px) | 7k | 0.99M | 19.72 | 0.785 | 0.543 | 0.171 | 16.85 | 0.115 |
| r1-depth | 7k | 0.86M | 19.95 | 0.780 | 0.541 | 0.080 | 16.68 | 0.087 |
| r1-geo (depth + normals + flatten + ratio + aa) | 7k | 0.93M | 20.26 | 0.781 | 0.525 | 0.055 | 16.98 | 0.074 |
| r1-geo-pc (+ pose opt + bilagrid) | 7k | 0.82M | 20.12 | 0.777 | 0.517 | 0.044 | 16.43 | 0.076 |
| r1-fused (DA3 fused, σ-weighted) | 7k | 1.03M | 20.07 | 0.775 | 0.521 | 0.056 | 16.68 | 0.076 |
| **final-default-geo** (the recipe) | 30k | 1.21M | 22.76 | 0.819 | 0.425 | 0.033 | 17.53 | 0.076 |
| final-geo-dense (grow_grad2d 0.00008) | 30k | 3.55M | 22.55 | 0.818 | 0.416 | 0.060 | 17.34 | 0.080 |
| final-depth-only (no normals / flatten / ratio / aa) | 30k | – | 22.54 | 0.815 | 0.445 | 0.045 | 17.50 | 0.081 |
| msplat on dataset v2 (fair: no lap frames) | 30k | 1.42M | 22.73 | 0.820 | 0.383 | 0.132 | 17.06 | 0.114 |

**Then, on the later test sets** (interp PSNR / SSIM / LPIPS / AbsRel; extrap the same):

| Run | Change | interp | extrap |
|---|---|---|---|
| final-default-geo | (reference on the P6 split) | 23.59 (biased: trained on many of these frames) | 17.55 / 0.689 / 0.474 / 0.077 |
| p6-select-geo | frame/still selection + brightness (P6 a1+a2) | 23.07 / 0.827 / 0.416 / 0.034 | 17.72 / 0.692 / 0.464 / 0.076 |
| p6-surface | + depth variance 0.5, opacity entropy 0.01 | 16.62 / – / – / 0.51 (collapsed) | 14.40 |
| p6-surface-gentle | + depth variance 0.05, entropy 0.001 | 22.55 / – / 0.436 / 0.082 | 17.43 / – / 0.490 / 0.102 |
| s3-refined-select-geo | refined poses (P7) | 24.62 / 0.866 / 0.338 / 0.024 | 19.71 / 0.753 / 0.370 / 0.031 |
| **s3-refined-fused-select-geo** | + priors fused from refined poses | **24.68 / 0.865 / 0.339 / 0.024** | **19.94 / 0.756 / 0.371 / 0.029** |
| s3-refined-fused-select-geo-poseopt | + `pose_opt` lr 3e-5 | trainer's own: 24.73 → 24.07, SSIM 0.866 → 0.851 | no cloud eval |
| s4-select-geo-bilagrid | scan 4, debanded, `--color bilagrid` | 22.60 / 0.844 / 0.317 / 0.020 | 19.04 / 0.736 / 0.379 / 0.052 |
| s4-refined-select-geo-bilagrid | refined poses | 23.85 / 0.878 / 0.264 / 0.015 | 20.69 / 0.802 / 0.306 / 0.029 |
| s4-refined-fused-select-geo-bilagrid | + priors fused from refined poses | 23.92 / 0.877 / 0.264 / 0.012 | 20.64 / 0.800 / 0.309 / 0.030 |
| s5-select-geo-bilagrid | scan 5 (app 0.4.0, + garden outdoors), the default dataset with 3 excursion images and 2 pruning views trimmed by hand, `--color bilagrid`, L40S pinned (57 min, 2.7M splats, $2.27) | 21.39 / 0.806 / 0.336 / 0.021 | 14.93 / 0.596 / 0.527 / 0.144 (**the lap's poses are 48 cm off from t ≈ 23 s**: ARKit drift; first 8 s 20–23 dB; at corrected poses frames gain 7–9 dB) |
| s5-drift-select-geo-bilagrid | scan 5 after the fixes (ingest drops the excursion, refine's coarse round takes out the lap drift), the default dataset, same settings (57 min, 2.7M splats, $2.28) | 21.21 / 0.801 / 0.342 / 0.022 | 17.75 / 0.723 / 0.385 / 0.052; re-solved (40 frames) 19.00 / 0.667 / 0.325 / 0.029. The first run re-scored at re-solved poses: 18.86 / 0.657 / 0.330 / 0.027 (34 frames) |

Fairness checks: the pose-graph splats were also scored on the refined splits (same frames and poses). The
refined splats still win: scan 3 aligned interp 24.66 vs 25.07, extrap +0.6 dB aligned, LPIPS −0.09, AbsRel
halved; scan 4 extrap aligned +1.05 dB, SSIM +0.07, LPIPS −0.08. Painting crops (band-pass NCC vs the photo):
held-out views 0.61 → 0.88 on scan 4. Named views (`data/derived/splat/eval/compare-s3-refined/`): sharper
everywhere (drawer and fridge handles, spice rack, shoes and coats at the front door, door panels, the painting);
kitchen-to-hallway lost its needles but stays smeared; kitchen-counter-side and kitchen-stairs about the same.

Cost reference: a scan 3 run ≈ $1 (L40S, ~25 min of training, 1.6–1.8M splats) + eval ≈ $0.25–0.4; scan 4 runs
grow to 3.5M splats and ~1 h 45 (≈ $2).

## What helped, what didn't

Helped:
- **Depth + normals** fix geometry at no photometric cost (AbsRel 0.18 → 0.055 / 0.044 at 7k; 4× better than msplat
  at 30k, δ<1.05 84 % vs 30 %). Dropping normals / flatten / ratio / antialiasing makes it worse (LPIPS 0.445 vs
  0.425): the regularisers help.
- **Refined poses** (P7): the largest single gain (scan 3 interp +1.5 dB, extrap +2 dB, LPIPS −0.08 / −0.09).
- **Priors fused from the same poses**: a tie on photometrics, better depth (scan 3 extrap AbsRel 0.031 → 0.029,
  scan 4 interp 0.015 → 0.012); the default now.
- **Frame/still selection** (P6 a1+a2): small (extrap +0.17 dB); the input was only partly blurry.
- **Debanding + bilateral grid** on scan 4: no banding residue in renders (ceiling row-profile std 0.27 % in the
  render vs 0.51 % in the photo). The grid's own effect isn't isolated (no scan 4 run without it).
- **LiDAR free-space pruning during training** (msplat era): removed the fog (489k splats on scan 3). Pruning a
  finished splat leaves holes (−0.55 dB): the wall's colour sat on the removed splats.
- **Tar loading** (`bulk.tar`, `prior.tar`): the container has the dataset in ~15 s instead of ~35 min.

Didn't help (don't repeat without a new idea):
- **7k-step screens**: useless for photometric quality with the default strategy (opacity resets every 3k, the
  last at 6k); 30k is needed to reach msplat's level.
- **Full resolution while underfit** (r1-base-full: same as 960 px).
- **`pose_opt` after refine**: lr 3e-5 learned corrections of 2.4 mm / 0.1° and lowered the trainer's interp PSNR
  (24.73 → 24.07, SSIM 0.866 → 0.851); named views identical (`data/derived/splat/eval/poseopt-views/sheet.jpg`).
  Not adopted. The default lr 1e-5 is a no-op in metric units (~0.3 mm corrections).
- **DA3 fused depth** (r1-fused, 7k): no measurable gain; the P4 ablations were never run.
- **Denser densification**: absgrad didn't densify (0.98M); grow_grad2d 0.00008 did (3.55M) but scored no better.
- **MCMC**: diverges once its budget fills (l1 0.14 → 0.27), with or without depth; scaling the noise lr by world²
  changed nothing. Open issue below.
- **Surface losses** (P6 a3): depth variance fought the depth loss and opacity entropy overlapping the opacity
  resets collapsed the splat (770k → 187k splats); the gentle version trains but AbsRel ×2.4. If revisited: depth
  variance alone, smaller λ, or a proper 2DGS distortion loss; check the moment normalisation with antialiasing.
- **A second, close-range refine round**: neutral (plan, "Pipeline decisions").
- Mac trainers (msplat, Brush, 7k screens): fewer frames at higher resolution −1.1 dB; SH degree 1 −0.3 dB; a
  higher densify threshold ±0; Brush at 960 px the same as msplat, 2.5× slower; Brush can't be pruned (no API).

Why some views stay messy (the user's screenshots, 2026-10-08): good views are frontal, matte, mid-distance and
near the captured path; messy ones are off it (along the counter at a grazing angle, through the kitchen→hallway
doorway with the frame, bin and door close by), close (large parallax), cluttered, shiny, or seen only by walking
frames. Radiating streaks are splats stretched along the few rays that saw a spot; the expected-depth loss fixes
only the mean depth per pixel. The extrap set (the lap) walks the same corridors, so it doesn't measure these
views; the named views do. Coverage (scan 3, mean deficit per named view): hallway-front-door 0.41,
kitchen-counter-side 0.36, kitchen-to-hallway 0.31, garden and pantry 0.08. Fixes: capture (walk the viewer's
routes both ways, side passes, doorways both sides), then densification for detail, then a viewer that stays near
the captured paths.

## Phase status

- **P0 shared evaluation**: done (split, evaluator, named views, compare pages). The single-session msplat test
  (hallway alone, kitchen alone) said resolution and depth mattered more than cross-session poses at 450 px; at
  960 px P7 showed cross-pass poses matter a lot. Not done: scoring the old msplat/Brush PLYs on the cloud.
- **P1 cloud plumbing**: done (Modal app, volumes, `just gs`, full checkpoints and `--resume auto`, `step=`
  lines, manifests, the real bill in `ps`).
- **P2 depth, normals, anti-needle**: done; in the recipe.
- **P3 poses and colour**: pose opt not adopted (above); bilateral grid used on scan 4; per-image affine colour
  implemented, untried.
- **P4 sensor + DA3 fusion** (after Howell's Cal Poly thesis): implemented, not adopted. LiDAR σ measured against
  mesh depth (confidence 2, 1.4826·MAD: 10–13 mm to 2 m, 20 mm at 2–3 m, 41 at 3–4 m, 59 at 4–5.5 m, ×5 at depth
  edges; confidence 1: 23–164 mm). Pose-conditioned DA3-BASE (Apache-2.0) is near metric (LiDAR/DA3 1.00–1.08) but
  up to ±40 % per frame in the hallway and 4–11× on a blank wall at 0.7 m, so fusion rescales it to LiDAR per frame
  (1.3 % median error after); gate 3·√(σ_s²+σ_m²). Open: the ablations (a) `lidar` + EdgeAwareLogL1, (b)
  `lidar_only` + σ-weighted, (c) `fused` + σ, (d) `da3_holes_only` + σ, two seeds each, after `just gs push --da3`.
  Low priority: DA3 mostly fills holes, and LiDAR + mesh already cover most views.
- **P5 resolution and defaults**: 960 px is the working size; full resolution and MCMC wait for densification.
  gsplat is the reference trainer in practice; `scanmodel splat --train` still defaults to msplat.
- **P6 input quality and capture**: a1 selection, a2 stills, a3 surface losses (not kept) done; **a4 densification
  open**; app work b1–b3 done in code (0.4.0 (8), device test pending); b4 guidance is the capture protocol.
- **P7 pose refinement**: done, the pipeline default (plan, "Pipeline decisions").

## Open issues

- **MCMC divergence.** Suspects: the fused CUDA perturb kernel's inputs (log scales / logit opacities?),
  relocation with our free-space pruning, the opacity/scale regularisers in a metric world, the dense 2 cm init.
  Alternative: a sparser init (`--max-init-points 200000` or 5 cm voxels) so the default strategy densifies where
  the detail is (it barely densifies from ~1M tiny splats: gradients spread thin).
- **GPU-fallback cost reporting**: an H100 fallback costs ~2× but `ps` and manifests say L40S. Pin `gpu =
  "L40S:1"` in a sweep TOML until fixed.
- **Views for other scans**: automatic upload of the repo's scan 3 `scene/splat_views.json` was removed
  in the portability audit. Named views are opt-in via `--views`; cloud evaluation uploads that explicit
  local file. Scene-specific views can also be rendered locally with
  `data/fourth/analysis/views/scripts/render_views.py` (CPU, 5–18 s per view).
- **Aligned PSNR is optimistic**: on the synthetic scene it adds ~0.6 dB even with correct poses.
- With antialiasing gsplat weights extra signals differently from colour (normal loss only; not blocking).

## Operating notes

- **Image build**: gsplat's full compile ran > 1.5 h on Modal's builder. Now a `run_function` step on 32 cores with
  `BUILD_2DGS=0 BUILD_3DGUT=0` (+ the same macros in `NVCC_FLAGS`): 162 s. Modal doesn't cache Dockerfile layers
  across Dockerfile edits (a full rebuild ~30 min), so small additions go in `modal_app.py` after the compile
  (e.g. `apt_install libusb-1.0-0` for Open3D). Importing DA3 imports gsplat, which JIT-compiles if the kernels
  aren't built: never import DA3/gsplat in a Dockerfile step.
- **Channels**: gsplat kernels exist only for some channel counts (…6, 8, 9…); RGB + ED + normals = 7, so
  `render.py` pads the extra signals. gsplat's "pure torch" path still calls CUDA for tile intersection, so CPU
  tests use our dense reference renderer. MCMC can't run on CPU. GPU tests: `just gs gpu-test` (79/80 pass).
- **Trainer defaults** = the P1 baseline: antialiased / absgrad / depth / normal losses off, `--resume no` locally
  (the Modal function passes `--resume auto`). Pruning runs on steps divisible by 1,500; with MCMC just before
  `step_post_backward`, with Default after.
- **Evaluator**: pose alignment 100 steps per frame at half size (`--pose-align-steps`, `--pose-align-scale`),
  scored at full size; `--samples N` saves N evenly spaced frames per set; `--no-views` skips named views. LPIPS
  needs torchvision's AlexNet weights (fetched in the Dockerfile).
- **Re-solved test poses** (2026-10-09, START-HERE Next 2): scan 5's lap poses drift 48 cm, and the extrap PSNR
  measured that instead of the splat. `scanmodel splat` now re-solves every test frame against the training frames
  only (`pipeline/scanmodel/testposes.py`: RootSIFT matches to co-visible training frames, lifted with their LiDAR at
  their training poses, PnP RANSAC + LM, then gates; "unresolved" rather than a doubtful pose) into
  `test_poses.json`; the eval scores each set at the dataset poses and at the re-solved ones (`psnr_resolved`,
  `psnr_same_frames`, `n_unresolved`, `pose_moved`) and starts pose alignment from the re-solved pose. Placement
  (with an advisor's second opinion): on the Mac at dataset time (the pose depends only on the dataset; no GPU
  minutes, no new image dependencies), in a side file that is neither hashed nor pushed with the dataset (training
  never reads it, so a newer solve re-scores old runs without a 1–2 GB re-upload; `cloud eval` uploads it beside
  the PLY and the eval refuses it when its `source` hashes don't match the dataset's transforms.json /
  split.json). Caveat: with `--pose-opt` the splat lives in the optimised training poses, which the re-solve
  doesn't see.
- **Priors**, in order, each reading the previous one's `prior/` output: `priors` (mesh depth + normals) →
  `calibrate` (`sigma.json`) → `da3` → `fusion --mode all`. `just gs push` runs the first two (`prepare`, CPU),
  `--da3` the last two (`depth_prior`, GPU). Stills have no LiDAR of their own: they get mesh priors.
- **DA3 in the image**: `--no-deps` at `3d835ec` plus its inference imports, constrained to what's installed (numpy
  2 stays); not in `uv.lock`; weights in `HF_HOME=/opt/hf`.
- The research clones (gsplat, dn-splatter, Depth-Anything-3, the thesis code) were in a session scratchpad;
  re-clone if needed.

## Services

Prices from the providers' pages on 2026-10-08; [U] = not confirmed. Modal per hour: L40S $1.95, L4 $0.80, A100
80 GB $2.50, H100 $3.95, A10 $1.10, plus ~$0.45 for CPU and RAM (`cloud.GPU_PRICES`). The same L40S on RunPod
Secure Cloud is $1.09 (Community $0.79), so Modal is ~2.2× the price.

| | Modal (pick) | RunPod Pods | Vast.ai | Lambda |
|---|---|---|---|---|
| Deploying code | Shipped from the Mac on every run; image cached | Image in a registry, rsync onto the pod | Image + `vastai copy` | Rsync onto a VM |
| Data at rest | Volumes, free to 1 TiB | Network volume $0.07/GB-month (pins a data centre) | Instance disk only | ~$0.20/GB-month [U] |
| A forgotten GPU | Impossible: ends with the function, hard timeout | Ours to build: start command deletes the pod, in-pod deadline, local reaper | Ours: self-destroy with `CONTAINER_API_KEY` | None built in [U] |
| Start-up | Seconds to ~1 min | 2–10 min image pull [U] | Varies by host | Minutes [U] |
| Extra plumbing over Modal | none | ~1–1.5 days | ~1.5–2 days | similar to RunPod |

Ranking: 1. **Modal** (worth ~$10–15 a month more at our volume for one developer). 2. **RunPod** Secure Cloud as
the fallback if Modal's price or stock becomes a problem: the same Dockerfile from GHCR, a network volume, a start
command that trains, uploads and deletes its own pod. 3. Vast.ai for long cheap sweeps if host variance and
photos on a stranger's disk are acceptable. Not considered: Hetzner (monthly only), DigitalOcean, Colab (no job
API), AWS/GCP (quota and IAM overhead), SkyPilot (one more layer).

## Risks

| Risk | Mitigation |
|---|---|
| gsplat `main` API changes | Pinned commit; we use only `rasterization`, strategies, `ops.remove`, `export_splats`. Fallback 1.5.3 |
| Preemption mid-run | Full checkpoints every 2k steps, `--resume auto`, `retries=2` |
| Pose optimisation hides real pose errors | Not used; pose errors are fixed in `scanmodel refine` |
| Bilateral grid absorbs real colour | Compare named-view renders; per-image affine as the lighter option |
| DA3 licence | Apache-2.0 weights (BASE, SMALL, METRIC-LARGE) only; GIANT (CC BY-NC) at most as a reference |
| Cost runaway | START-HERE budget rules, `just gs ps` before every job, timeouts, the workspace budget |

## References

- gsplat: https://github.com/nerfstudio-project/gsplat (`examples/simple_trainer.py`, `gsplat/strategy/ops.py`,
  `gsplat/exporter.py`), docs https://docs.gsplat.studio
- DN-Splatter (losses ported in P2): https://github.com/maturk/dn-splatter, https://arxiv.org/abs/2403.17822;
  AGS-Mesh https://arxiv.org/abs/2411.19271
- Howell, Cal Poly MS thesis 2026: https://digitalcommons.calpoly.edu/theses/3347, code
  https://github.com/bshowell0/thesis (`src/depth_fusion.py`)
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Modal: https://modal.com/pricing, https://modal.com/docs/guide/preemption, https://modal.com/docs/guide/timeouts,
  https://modal.com/docs/guide/budgets, https://modal.com/docs/reference/cli/app
- RunPod: https://www.runpod.io/pricing, https://docs.runpod.io/api-reference/pods/POST/pods
- Vast.ai: https://vast.ai/pricing; Lambda: https://lambda.ai/pricing
