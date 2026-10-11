# gs-trainer: the gsplat trainer

A third splat trainer next to msplat and Brush (decisions, results, recipe: `docs/plans/2026-10-08-gsplat-cloud.md`;
budget and best runs: `docs/plans/START-HERE.md`). It runs on a CUDA GPU on Modal and is driven from the Mac
with `just gs …`. It has its own uv project (Python 3.12,
torch 2.9.1, gsplat pinned to `512d366`), separate from `pipeline/` (Python 3.14, no CUDA).

On the Mac, gsplat installs as pure Python (no CUDA kernels): the strategies, `ops.remove` and the PLY
export work, rasterisation does not. `gs_trainer.render` has a small pure-PyTorch reference renderer so the
whole training loop runs in CPU tests on tiny synthetic scenes. On the GPU image the kernels are compiled
once at image build.

```
uv run --project trainer pytest -q          # CPU tests
uv run --project trainer gs-trainer --help  # train / evaluate / prepare (on a GPU), cloud ...
```

From the repo root, `just gs <cmd>` = `gs-trainer cloud <cmd>` (the list is in the `justfile`). Besides the
run commands (`push`, `train`, `sweep`, `logs`, `pull`, `stop`, `eval`, `gc`):

```
just gs build                 # build (or reuse) the image now, build log shown; prints the image id
just gs gpu-test [-- -k x]    # the test suite on a cloud L40S (pytest args after --), CUDA-only tests included
just gs ps                    # runs + cost estimate, the ACTUAL bill (today / this month, by kind), live apps
```

Use these instead of ad-hoc Modal scripts: their apps (`flat-gs-build`, `flat-gs-gpu-tests`; the Modal app and
volume names keep the trainer's old name, see `cloud.APP_NAME`) show in `ps`
while they run, and their cost in its bill. The bill comes from `modal billing report` (full hours only:
the current hour isn't in it yet); the per-run estimate covers training containers only. Function
timeouts cap a hung job: gsplat compile 20 min, prepare / depth_prior / evaluate 45 min, gpu_test 30 min,
train 3 h per attempt. `cloud eval --run <run>` scores the run on the dataset it trained on (its `run.json`
`dataset`); `--data` overrides that, with a warning when it differs (`--ply` without `--data` scores on the
project's dataset, `data/NAME/splat/datasets/default`). It uploads the re-solved test poses beside the PLY (`--test-poses FILE`, else
the local dataset's `test_poses.json`: `--data`, or the run's `data_dir`; `--no-test-poses` after `--`: none).

### Preparing a dataset's priors

Project selection works after the command, for example `just gs train --project fifth`,
`just gs push --project fifth --da3`, or `just gs pull RUN_ID --project fifth`.
Without `--project`, the tracked repo-root `settings.toml`'s top-level `project` is the default; with
neither, a command that needs a project stops with an error (an empty value counts as unset). No
environment variable or `.env` is read. A selected project's dataset is
`data/NAME/splat/datasets/default`, and downloaded runs are `data/NAME/splat/runs/RUN_ID`.
`--data` and `--out` override those paths. Detached runs record their project and explicit output
directory, so a later `pull` can recover them; selecting a project on `pull` overrides the recorded
destination. `cloud eval --run RUN_ID --project NAME` still scores the dataset the run trained on;
the project selects its local output location. Use `--data` explicitly to score a different dataset.
Local paths the trainer writes (`data_dir` and `local_out` in a new run's `run.json`, eval.json's
`ply`/`data`, `config.json`) are repo-relative when inside the repo, so records work from any checkout;
older records' absolute paths still read. Missing legacy local paths stored in older run metadata are
resolved to their migrated project paths for dataset lookup and output routing; existing paths and
explicit `--data`/`--out` are kept as given.
`train`, `evaluate`, and `prepare` on a local GPU also accept `--project`; local training defaults to
`runs/local` and evaluation to `runs/evaluation`.

Each step reads the previous one's output in `prior/`; all of them skip or redo whole files, so they resume.

| Step | Module | Writes | Where it runs on the cloud |
|---|---|---|---|
| 1 | `gs_trainer.priors` | `prior/<stem>.npz` `depth`, `normal`; `prior/meta.json` | `prepare` (CPU), run by every `cloud push` when missing |
| 2 | `gs_trainer.calibrate` | `prior/sigma.json` (LiDAR sigma by confidence and distance) | `prepare`, after step 1 |
| 3 | `gs_trainer.da3` | `prior/da3/<stem>.npz`, `prior/da3/meta.json` (DA3-BASE, pose-conditioned) | `depth_prior` (GPU), only with `cloud push --da3` |
| 4 | `gs_trainer.fusion --mode all` | the fused depth + sigma of each mode into `prior/<stem>.npz`; `prior/fusion.json` | `depth_prior`, after step 3 |

`prepare` and `depth_prior` end by packing `prior/` into `prior.tar` (see below); `pack_prior` (CPU) does
only that, and `cloud push` runs it for priors made before the tars.

Locally: `gs-trainer prepare --data D` (steps 1–2), then `python -m gs_trainer.da3 --data D` and
`python -m gs_trainer.fusion --data D --mode all`. DA3 needs `depth_anything_3`, which is in the GPU image
only (Dockerfile; not in `uv.lock`: its own metadata pins numpy<2 and xformers, so it goes in with
`--no-deps` plus the short list of what its inference path imports; the same two `uv pip install` lines
work in the Mac venv, until the next `uv sync` or `just install` removes them). Training with a fused target
(`--depth-target fused|lidar_only|da3_holes_only`) needs steps 1–4: `just gs push --da3` once per dataset.
`cloud train`/`sweep` refuse such a target before starting a GPU when `prior/fusion.json` on the volume
lacks it, and `gs_trainer.train` refuses it too.

### The dataset on the volume

`flat-gs-data:/datasets/<hash>/` (hash: `cloud.dataset_hash`, the small files by content and the folders
by name and size):

| Entry | Written by | Content |
|---|---|---|
| the dataset's files | `cloud push` | `transforms.json`, `split.json`, `points.ply`, `mesh.ply`, `views.npz`, `images/`, `depth/`, `confidence/` as below, file by file |
| `bulk.tar` | `cloud push` (built on the Mac in a temp dir) | `images/`, `depth/`, `confidence/` once more, as one uncompressed tar (`gs_trainer.datapack`) |
| `.complete` | `cloud push`, last | JSON: files, bytes, source, time; `bulk_tar` once the tar is there. A dataset without it is re-uploaded; one without `bulk_tar` (pushed before the tars) gets just the tar |
| `prior/` | `prepare`, `depth_prior` | the priors (steps 1–4 above) |
| `prior.tar` | `prepare`, `depth_prior` (at the end), `pack_prior` | `prior/` without `prior/da3/`, uncompressed; deleted when `prepare`/`depth_prior` start, so it never outlives the `prior/` it was made from |

The volume is slow with thousands of small files (a run once spent ~35 of 59 min copying and reading
them), so `train` and `evaluate` work on a copy on the container's disk, `/tmp/dataset/<hash>`
(`modal_app._local_copy`): each tar is copied in one sequential read and unpacked, the other top-level
files are linked to the volume. Without a tar its folders are copied file by file (`depth/`,
`confidence/`, `prior/`) or linked (`images/`). The eval leaves the priors out. The log shows each step's
time (`gs-trainer: bulk.tar 1.2 GB in 14 s (86 MB/s), … unpacked in 3 s`).

## Contracts

These are the interfaces the modules agree on. Change them here first.

### Dataset v2 (written by `scanmodel splat`, `pipeline/scanmodel/stages/splat.py`)

The trainer uses the dataset's world coordinates and `depth_unit_scale_factor`; it does not require
particular room names or camera positions. Stationary cameras use the point-cloud extent for scene
scale. Named evaluation viewpoints are scene-specific: neither local nor cloud evaluation selects
a project's `scene/views.json` automatically. Pass `--views FILE` explicitly (after `--` for cloud commands);
cloud evaluation uploads that local file. `--no-views` disables it.

`trainer/sweeps/p6/`, `p7/`, and `p8/` include historical recipes for particular scans, edited datasets,
and exposure conditions. Their comments document those experiments; they are not selected by default.
Choose a recipe and dataset explicitly and tune supervision and pruning for the new capture. Depth
confidence, calibration distance bins, and pruning defaults target ARKit LiDAR (confidence 0/1/2), so
another depth sensor requires compatible export semantics and calibration.

`data/NAME/splat/datasets/default/` (`scanmodel splat --dataset D`: `datasets/D/`):

| File | Content |
|---|---|
| `transforms.json` | `version: 2`, `camera_model: "OPENCV"`, `depth_unit_scale_factor: 0.001`, `ply_file_path: "points.ply"`, `mesh_file_path: "mesh.ply"`, `split_file_path: "split.json"`, `frames: [...]` sorted by `file_path`. These are the **training + interp** frames, so msplat and Brush (which read only this file) never see the extrap frames |
| frame entry | `file_path: "images/<stem>.jpg"`, `transform_matrix` (camera-to-world, **OpenGL** axes, scene frame: metres, z up), `fl_x fl_y cx cy w h` (for the image as written), `camera: "video" \| "still"`, `session`, `t`; video frames also `depth_file_path: "depth/<stem>.png"` and `confidence_file_path: "confidence/<stem>.png"` |
| `depth/<stem>.png` | LiDAR depth, uint16 mm, **native 256×192**, not upsampled, not masked (0 = no reading). Its intrinsics are the image's scaled by `(depth_w / w, depth_h / h)`: ARKit's depth map covers the image's field of view |
| `confidence/<stem>.png` | uint8 ARKit confidence 0/1/2, same size as the depth |
| `split.json` | `{"interp": [stems], "extrap": [stems], "extrap_session": str \| null, "extrap_frames": [frame entries]}`. `interp` = every 8th frame of `transforms.json` (index % 8 == 0, msplat's and Brush's own rule). `extrap` = the frames of the closing lap session (name ends in `-lap`), else none; their entries live only in `extrap_frames`, their images and depth in the same folders. Train = transforms.json frames minus interp |
| `test_poses.json` | Optional (absent in datasets exported before 2026-10-09; `scanmodel test-poses` adds it). Each test frame's pose re-solved against the **training frames only** (`pipeline/scanmodel/testposes.py`), so the eval can tell a test frame's pose error from splat quality. Eval-only: not in the cloud hash, not pushed with the dataset (`cloud eval` uploads it). `{"version": 1, "method": "sift-pnp", "source": {"transforms_sha256", "split_sha256"} (the files it was solved from; the eval refuses the file when they differ), "params": {...}, "summary": {"interp" \| "extrap": {n, resolved, unresolved, moved, shift_cm_median, shift_cm_max}}, "wall_s", "frames": {stem: entry}}`, an entry for every interp and extrap stem |
| `test_poses.json` entry | Always: `status` (`"ok"` \| `"unresolved"`), `refs` (training stems), `matches`, `inliers`, `reproj_px` (median, px at the image size), `reproj_dataset_px` (the same inliers at the dataset pose), `sigma_cm`, `sigma_deg` (1σ from the PnP Jacobian), `quadrants`, `hull`, `depth_resid_cm` (median vs the frame's own LiDAR; null for stills), `refs_agree`, `refs_disagree`, `ref_checks` ([stem, own inliers, share the pose explains]); diagnostics may be null when no pose was found. `"ok"`: `transform_matrix` (camera-to-world, OpenGL axes, like the frame entries), `shift_cm`, `rot_deg`, `offset_cm` [dx, dy, dz] (world) from the dataset pose; `kept_dataset_pose: true` (+ `solved_shift_cm`, `solved_rot_deg`) when the dataset pose reprojects the matches about as well, so the matrix is the dataset pose. `"unresolved"`: `reason`, no matrix (`rejected_offset_cm` when a pose was found but failed a gate) |
| `points.ply` | init points (float xyz + uchar rgb), as before |
| `views.npz` | LiDAR depth views for free-space pruning, `scanmodel.depthviews.write_views(poses_dir, …)` (read with `freespace.load_views`) |
| `mesh.ply` | the TSDF mesh from `fuse/dense.ply`: float32 vertices, int32 triangles, nothing else |
| `prior/<stem>.npz` | written by `gs_trainer.priors` (on the cloud): `depth` float16 (H, W) metres, 0 = no hit; `normal` int8 (H, W, 3), world frame, unit × 127. H, W = the image scaled to max side `prior/meta.json["side"]` (default 640). `gs_trainer.fusion` adds, per mode, depth + sigma float16 (H, W) metres, 0 in both = no target: `fused`/`sigma`, `fused_lidar_only`/`sigma_lidar_only`, `fused_da3_holes_only`/`sigma_da3_holes_only` (read with `fusion.load_fused(frame, W, H, device, mode)`) |
| `prior/sigma.json`, `prior/da3/`, `prior/fusion.json` | the LiDAR sigma table (`gs_trainer.calibrate`), DA3's raw depth + confidence at its processing size (`gs_trainer.da3`), and the fusion's settings and per-mode statistics; a mode is available for training when it is a key of `fusion.json` |

`stem` = the image file name without `.jpg`, e.g. `20261007-183434-hallway_000120` or
`20261007-183434-hallway_s12`. Stills have no LiDAR depth (only mesh priors).

### Splat tensors

gsplat's convention throughout: `means` (N, 3) scene frame, `scales` (N, 3) **log**, `quats` (N, 4) wxyz
(not necessarily normalised), `opacities` (N,) **logit**, `sh0` (N, 1, 3), `shN` (N, K−1, 3). A
`torch.nn.ParameterDict` with these keys. PLY = the standard 3DGS PLY (`gsplat.export_splats`,
`gsplat.exporter.load_ply_to_splats`), already in the scene frame (we never normalise the world).

### Modules (`trainer/gs_trainer/`)

- `data.py`
  - `Frame` dataclass: `stem, image_path: Path, K: np.ndarray (3×3, at w×h), w, h, c2w: np.ndarray (4×4, OpenCV axes), camera_idx: int (0 video, 1 still), session: str, depth_path: Path | None, confidence_path: Path | None, prior_path: Path | None`.
  - `Scene.load(data_dir: Path) -> Scene` with `frames: list[Frame]` (transforms.json frames, then extrap frames), `train: list[int]`, `interp: list[int]`, `extrap: list[int]` (indices into `frames`), `points: np.ndarray (M,3) float32`, `colors: np.ndarray (M,3) uint8`, `scene_scale: float` (1.1 × max camera distance from the cameras' centroid, as gsplat), `data_dir`.
  - `Images(scene, indices, max_side: int, device, cache: bool | None = None)`: images at max side `max_side`. `cache=None` caches uint8 on `device` when the total fits under ~40 % of free device memory, else decodes from disk each call. `get(i)` / `size(i)` take a frame index into `scene.frames` (not a position in `indices`); images are never enlarged (`scaled_size(w, h, max_side)`). `get(i) -> (image float32 (H, W, 3) in [0, 1] on device, K (3, 3) float32 tensor scaled to H, W)`; `size(i) -> (W, H)`.
  - `load_depth(frame, W, H, device) -> (depth (H, W) float32 metres, confidence (H, W) uint8)`, nearest upsampling, zeros where there's no reading; `None` for stills.
  - `load_prior(frame, W, H, device) -> (depth (H, W) float32, normal (H, W, 3) float32 world)` or `None` if there's no prior (`Frame.prior_path` is set only if the npz existed at `Scene.load`). Depth nearest, normals bilinear and renormalised.
  - `viewmat(c2w) -> torch.Tensor (4, 4)` world-to-camera (OpenCV).
- `synth.py`: `write_dataset(path: Path, n_frames: int = 16, size: tuple[int, int] = (64, 48), extrap: int = 4, seed: int = 0) -> Path`. A tiny v2 dataset of a textured box room seen from a ring of cameras, with depth, confidence, points, mesh and a split. Used by every test.
- `render.py`: `render(splats, viewmats (C,4,4), Ks (C,3,3), W, H, sh_degree, mode="RGB+ED", backend="gsplat" | "torch", colors_override=None, **kw) -> (rgb (C,H,W,3), depth (C,H,W,1) | None, alpha (C,H,W,1), info)`. `info` carries what gsplat's strategies read (`means2d` with retained grad, `radii`, `width`, `height`, `n_cameras`, `gaussian_ids`). `backend="torch"` is a dense O(N·H·W) reference for CPU tests only.
- `losses.py`: `l1`, `ssim` (torch, Gaussian window), `edge_aware_log_l1(depth, gt, image, mask)`, `weighted_l1(depth, gt, weight, mask)`, `normal_loss(pred_n, gt_n, mask)` (L1 + TV), `flatten_loss(scales)` (`mean(min_i exp(s_i))`), `scale_ratio_loss(scales, max_ratio=10)`, `splat_normals(splats, c2w)` (shortest axis, flipped towards the camera).
- `prune.py`: `free_space_drop(splats, views, min_free) -> torch.BoolTensor` through `pipeline/scanmodel/freespace.py` (numpy, shipped alongside), and `apply(splats, optimizers, strategy_state, drop, strategy)` (`ops.remove` for DefaultStrategy; low opacity for MCMC).
- `modules.py`: pose optimisation (per-image 9-D delta) and per-image bilateral grid / affine colour, adapted from gsplat `examples/` (Apache-2.0).
- `train.py`: `Config` dataclass (every option, CLI `--name value` generated from it); `train(cfg) -> dict`; `main(argv) -> int`. Writes to `cfg.out`: `train.log` lines `step=  1200 loss=… splats=… ms=…` every 100 steps, `progress.json`, `ckpts/latest.pt` (full state, every 2,000 steps; `--resume auto`), `splat.ply`, `metrics.json`, `config.json`. Losses beyond L1/SSIM, each off by default: `--depth-loss` (`--depth-target lidar|fused|…`, `--depth-weighting edge_log|sigma`), `--normal-loss`, `--flatten-lambda`, `--scale-ratio-lambda`, `--depth-var-lambda` (variance of depth along each ray, from [1, z, z²] rendered as extra signals; `--depth-var-start`), `--opacity-entropy-lambda` (splats solid or gone; `--opacity-entropy-start`). Free-space pruning: `--prune`, `--prune-every`, `--prune-until`, `--min-free`. Run configs live in `trainer/sweeps/` (TOML, `just gs sweep FILE…`).
  - Depth (`--depth-loss`): `--depth-target lidar` (default: raw LiDAR, confidence ≥ 1 before `--depth-conf-until`, then 2; video frames only) or `fused | lidar_only | da3_holes_only` (that fusion mode's depth, valid where > 0, stills included when the mode has values for them). `--depth-weighting edge_log` (default, EdgeAwareLogL1) or `sigma` (`weighted_l1` with w = 1/σ on the valid pixels, divided by its mean there; needs a fusion target). The P4 ablations: (a) `lidar` + `edge_log`, (b) `lidar_only` + `sigma`, (c) `fused` + `sigma`, (d) `da3_holes_only` + `sigma` (or `edge_log`). `depth_supervision(cfg, frame, W, H, device, step) -> (depth, valid, weight | None) | None`, `check_depth_config(cfg)` (raises `SystemExit` on `sigma` + `lidar`, or a fusion target missing from `prior/fusion.json`).
- `evaluate.py`: `evaluate(data: Path, ply: Path, out: Path, sets=("interp", "extrap"), pose_align_steps=100, views: Path | None = None, ..., pose_align_scale=0.5, samples=0) -> dict`; `main(argv) -> int`. PSNR/SSIM/LPIPS, depth AbsRel and δ<1.05 against LiDAR (confidence 2), pose-aligned PSNR (the pose is fitted at `--pose-align-scale` × the eval size for `--pose-align-steps` steps per frame, then scored at full size), named-view renders (opt-in with `--views FILE`; `--no-views`: none, also on the cloud), the 6 worst frames per set (`--samples N`: also N evenly spaced ones in `samples/<set>/`), `eval.json` and `index.html`; `--antialiased` for splats trained with it. Re-solved test poses (`test_poses=`, default the dataset's `test_poses.json`; `use_test_poses=False` / `--no-test-poses`: none; refused when solved from another transforms.json / split.json, `eval.json` `test_poses.status`: `ok` \| `missing` \| `mismatch` \| `off`): each set is also scored at them: `psnr_resolved`, `ssim_resolved`, `lpips_resolved`, `abs_rel_resolved`, `delta105_resolved` (means over the re-solved frames only), `psnr_same_frames` (their dataset-pose PSNR), `psnr_unresolved` (the dataset-pose PSNR of the frames that failed), `n_resolved`, `n_unresolved` (null without test poses), `pose_moved` (frames re-solved > 10 cm or 2° from the dataset pose: stem, `pose_shift_cm`, `pose_rot_deg`, `psnr`, `psnr_resolved`, `reproj_px`, `reproj_dataset_px`); per frame `pose` (`resolved` \| `unresolved` \| null) and the `_resolved` metrics. Pose alignment starts from the re-solved pose where there is one. Every scored frame is in `per_frame` (one JSON line each: `stem`, `session`, `camera`, `psnr`, `ssim`, `lpips`, `abs_rel`, `delta105`, `depth_px`, `psnr_aligned`, `pose_correction`, and the `_resolved` ones incl. `delta105_resolved`, `depth_px_resolved`); per set `sessions` = {session: `aggregate(rows)`}: `n`, `psnr`, `psnr_median`, `psnr_min`, `worst`, `ssim`, `lpips`, `psnr_aligned`, `abs_rel`, `delta105`, `depth_px`, `n_resolved`, `psnr_resolved`, `ssim_resolved`, `lpips_resolved`, `abs_rel_resolved`, `psnr_same_frames`, by the set's own definitions (frame means, depth pooled per pixel), shown in `index.html`. `compare(runs, out)` / `python -m gs_trainer.evaluate compare EVAL_DIR... --out DIR` builds one page across runs, with a per-room table per set (sessions matched across runs by `session_label`, the name without its date; a `rooms in every run` row pools the rooms all runs have, e.g. leaving out one scan's garden); for eval.json written before `sessions` the session comes from the stem. PLYs are read with our own numpy reader (gsplat's `load_ply_to_splats` needs `plyfile`).
- `priors.py`: `prepare(data: Path, side: int = 640) -> dict` (mesh raycast, Open3D); `main(argv) -> int`.
- `calibrate.py`, `da3.py`, `fusion.py` (P4): the LiDAR sigma table, DA3 inference over windows of frames, and the inverse-variance fusion with gating; see "Preparing a dataset's priors" and each module's docstring.
- `datapack.py`: the dataset tars (`bulk.tar`, `prior.tar`): `tar_members(data, tar)`, `pack(data, tar, dst=None)`, `extract(tar, dst)`, `copy_plan(entries, prior=True)` (which tars to unpack, folders to copy, entries to link). Standard library only.
- `cloud.py`, `modal_app.py`, `cli.py`: the Modal app and the `gs-trainer` CLI (`train`, `evaluate`, `prepare` run locally; `cloud setup|build|push [--da3]|train|sweep|ps|logs|pull|stop|eval|gc|gpu-test`). Modal functions: `prepare(hash, args)` (CPU: priors → calibrate → `prior.tar`), `pack_prior(hash)` (CPU: `prior.tar` only), `depth_prior(hash, model="depth-anything/DA3-BASE")` (GPU: da3 → fusion `--mode all`; the weights are baked into the image under `HF_HOME=/opt/hf`), `train(hash, run_id, args, meta)`, `evaluate(hash, ply, out, args)`, `gpu_test(args)` (pytest on an L40S, `trainer/tests` added at container start).

### `gs-trainer cloud train` (called by `scanmodel splat --train --trainer gsplat`)

```
gs-trainer cloud train --data <dataset dir> --out <local run dir> [--name NAME] [--steps N] [--gpu L40S]
                       [--max-side N] [--detach] [-- <gs_trainer.train options>]
```

Uploads the dataset if needed, runs `gs_trainer.train` on Modal and prints its `step=` lines to stdout as they
come (the pipeline's progress bar parses them). Unless `--detach`, it then downloads into `--out`:
`splat.ply`, `train.log`, `metrics.json` (at least `psnr`, `ssim` on interp, `splats`, `wall_s`),
`config.json`, `progress.json`, `run.json` (run id, Modal app id, git commit, diff hash, GPU, dataset hash,
arguments, start/end). Exit code 0 only if `splat.ply` arrived. The pipeline logs this command's own output
to `cloud.log` (the remote `train.log` is downloaded separately).

`freespace.py` is importable as a top-level module `freespace` on the cloud (shipped next to `gs_trainer`);
locally `gs_trainer.prune` adds `pipeline/scanmodel` to `sys.path` when the import fails.
