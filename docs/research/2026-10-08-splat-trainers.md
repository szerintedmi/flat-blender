# Splat quality: what causes the artifacts, trainers, and what to change

Date: 2026-10-08. Context: scan 3 (`third`). msplat 1.1.4 at 1/4.25 (≈450 px wide), 30k steps, PSNR 23.0 /
SSIM 0.80, with LiDAR free-space pruning. Brush 0.3.0 gives similar results. The splat shows streaks on walls,
needle-like splats, a doubled doorframe and blurry blobs, worst in the kitchen and hallway.

Labels: **[Doc]** = paper, vendor docs or source code (primary source). **[Comm]** = community or vendor report
without a method. **[Inf]** = my inference. **[Checked]** = I checked it myself on our data or on the package
source. **[Unverified]** = I could not confirm it.

## TL;DR

**Is resolution the bottleneck? No, not for these artifacts.** At 450 px one pixel is about 4 mm at 1.5 m. On
the same scale, our pose residuals are 3–6 px inside a session and 14–35 px on the walls where sessions
disagree. Motion blur in the video frames is about 4 px at full resolution (≈1 px at 450 px), and the video
frames are 3× less sharp than the stills. Training at 1920 px multiplies every one of these errors by 4.3 in
pixels, without adding much real detail. The streaks, needles and doubled edges come from geometry the trainer
can't pin down (no depth supervision, grazing views in narrow rooms) and from poses that disagree. Resolution
starts to pay off only after those are fixed (see "Diagnosis" for the numbers). Your own tests agree: Brush at
960 px scored the same as msplat at 450 px, and 1/2 resolution with fewer frames scored −1.1 dB.

Ranked list (try in this order):

| # | What | Fixes | Expected gain | Effort | Where |
|---|---|---|---|---|---|
| 1 | **Diagnostics first.** (a) Hold out an *extrapolation* set: the lap session's frames, or all stills, instead of every 8th frame. (b) Score rendered depth against LiDAR. (c) Train the **hallway session alone** and the kitchen alone, then compare with the all-session splat | Tells you whether the doubling and streaks come from cross-session pose error (single-session splats clean) or from geometry ambiguity (single-session splats still streaky) | Nothing visible. It makes every later step measurable. The current every-8th metric sits 0.5 s from training views and can't see these artifacts | ½–1 day | Mac, msplat as is |
| 2 | **LiDAR depth (+ normal) supervision**, trying DN-Splatter first. Our `transforms.json` already has `depth_file_path` and `depth_unit_scale_factor: 0.001` | Wall streaks, needles, blobs in mid-room, under-observed hallway and ceiling | Large visual gain on walls and floors. PSNR on interpolated views moves little, but depth error drops ~2–3× in the paper (MuSHRoom AbsRel 8.3 → 3.6) | 1–2 days (cloud setup, our pruning ported to Python) | **Cloud CUDA** (DN-Splatter is CUDA-only). No Mac trainer has depth loss today |
| 3 | **Pose refinement**: (a) cross-session BA on the Mac (pycolmap, ARKit/pose-graph priors), which is already "optional BA" in the plan; then (b) per-image pose optimisation in training (`gsplat --pose_opt`, splatfacto `camera_optimizer`) | Doubled doorframe, ghosting, general softness | ScanNet++ iPhone: +0.8 dB from in-training pose optimisation. MobileBrick: ARKit → refined poses recovers +0.9–1.2 dB of a 1.7 dB gap. Our cross-session errors are larger than both | BA: 3–5 days (also helps textures). Pose opt: one flag on cloud | BA on Mac, pose opt on cloud |
| 4 | **Frame selection by blur + higher video bitrate.** Keep the sharpest frame in each 0.5 s window; predicted blur = f·ω·t_exp from the ARKit poses (t_exp = 8.3 ms is logged). Raise HEVC from 16.7 Mbps | Softness, part of the smearing | Small to moderate (+0.2–0.5 dB [Inf]), sharper textures | Hours (selection), an app change (bitrate) | Mac |
| 5 | **Per-image colour correction** (stills vs frames): bilateral grid or PPISP (gsplat), or a per-image affine transform | Blotches where stills and frames overlap. Stills are ~6 % brighter than the video frames even with exposure locked [Checked] | +0.3–1 dB [Inf]. ScanNet++ got +1.4 dB, but with auto-exposure, which scan 3 didn't use | One flag on cloud. A day to add a per-image affine to msplat by hand | Cloud (flag) or Mac (patch) |
| 6 | **Anti-needle regularisation**: scale-ratio cap (`max_gauss_ratio` 10), MCMC with `scale_reg`, Mip-Splatting filter | Needles, spikes when viewed off-trajectory | Visual. PSNR roughly neutral | Flags on cloud. **Brush main** (unreleased) has it built in (#479) | Cloud, or Mac via Brush built from source |
| 7 | **Only then, more resolution**: patch msplat to store images as uint8 or stream them, or use Brush main (streams from disk), at 1/2 to 1/1.5 | Fine texture | +0.3–1 dB once poses are good [Inf]. Possibly negative before | ½ day (free the duplicates) to 2 days (streaming) | Mac |

Shortest path to a visibly better splat: #1 on the Mac (a day), then one cloud run of DN-Splatter
(splatfacto-based, sensor depth + normals, `camera_optimizer` on, bilateral grid on) on the existing dataset. In
parallel, the cross-session BA that the textures need anyway.

## 1. Diagnosis of the artifacts

### Numbers from our own data [Checked]

| Quantity | Value | In pixels at 450 px (fx ≈ 357) | At 1920 px (fx ≈ 1524) |
|---|---|---|---|
| Pose graph closure error (`poses/manifest.json`) | median 2.5 cm, p90 6.2 cm | 6 / 15 px at 1.5 m | 25 / 63 px |
| ICP between sessions | 2.4–2.7 cm rms; kitchen corrected 6.7 cm, hallway 4.7 cm | 6 px at 1.5 m | 25 px |
| One wall's faces from two sessions | 5.7 and 13.6 cm apart | 14–32 px at 1.5 m | 58–138 px |
| Typical ARKit error (ScanNet++ iPhone rooms, [Kyriazis 2026](https://arxiv.org/abs/2608.21008)) | 0.55° / 19 mm | 3.4 px (rotation) | 15 px |
| Motion blur, frames (8.3 ms exposure, measured angular speed p50 / p90 = 21 / 47 °/s in the hallway) | f·ω·t | 1.0 / 2.2 px | 4.1 / 9.4 px |
| Rolling shutter (readout ~20 ms assumed, [Unverified] for the iPad) | f·ω·t_readout | ~2.5 px skew | ~11 px |
| Still vs nearest video frame (40 pairs) | stills 5.6–6.9 % brighter (blue most), Laplacian variance 3.15× higher | | |
| Video | HEVC 16.7 Mbps, 1920×1440 at 60 fps (≈0.1 bit/px/frame) | | |
| Exposure | Locked in all 5 sessions: 1/120 s, ISO 284, WB and focus locked | | |

The pixel is the smallest error source in this table. At 450 px, pose error is 3–30× a pixel, and blur and
rolling shutter are 1–3×. At 1920 px every row grows 4.3×, while the true detail in the video frames is limited
by blur and compression to roughly 500–900 px of effective resolution [Inf].

### Likely causes, ranked for the artifacts in the screenshot

1. **Geometry the photos don't pin down, with no depth supervision.** This explains the wall streaks, needles
   and mid-room blobs [Inf, strong]. In a narrow hallway most views see the walls at grazing angles. A
   photometric loss is then satisfied by splats stretched along the view rays or floating in front of the wall
   (a classic 3DGS failure on low-texture indoor surfaces). Viewed off the training trajectory, they show up
   as streaks and needles. 3DGS drives many Gaussians towards one dominant axis: Brush measured >25 % of
   splats with aspect ratio >10 on bonsai and 53 % on garden ([Brush #479](https://github.com/ArthurBrussee/brush/pull/479)).
   [Effective-rank regularisation (NeurIPS 2024)](https://arxiv.org/abs/2406.11672) names needle artifacts as
   a direct result. Our pruning removes splats in front of surfaces only where the LiDAR saw free space (≤ 4.5 m,
   confidence 2, ≥3 votes). It doesn't pull surface splats flat or onto the surface. Depth supervision with
   phone LiDAR is the standard fix for indoor rooms ([DN-Splatter](https://arxiv.org/abs/2403.17822),
   [AGS-Mesh](https://arxiv.org/abs/2411.19271)).
2. **Pose inconsistency, mainly between sessions.** This explains the doubled doorframe, ghosting and soft
   walls [Inf, strong]. Two groups of views that each agree internally but are offset by several cm make the
   optimiser keep two copies of an edge, or average them into a blur. Kitchen and hallway had the largest ICP
   corrections (6.7 and 4.7 cm), and the wall with faces 13.6 cm apart is the same kind of evidence. 3DGS
   needs roughly pixel-level pose consistency. Published numbers for phone captures: in-training pose
   optimisation adds +0.79 dB on ScanNet++ iPhone ([Robust Gaussian Splatting, Darmon et al. 2024](https://arxiv.org/abs/2404.04211),
   ablation 21.71 → 23.08 with colour → 23.87 with pose). Refining ARKit poses recovers 0.93–1.16 dB of a 1.74 dB
   gap to ground truth (SSIM 0.554 → 0.66–0.70) on MobileBrick ([Kyriazis 2026, Table 4](https://arxiv.org/html/2608.21008)).
   Neither msplat nor Brush can optimise poses.
3. **Motion blur, rolling shutter and HEVC softness in the video frames.** This explains the general softness
   and some smearing [Doc + Checked]. The frames are 3× less sharp than the stills (Laplacian variance), and 90 %
   of frames carry < 2.2 px of rotational blur at 450 px. This is real but secondary at the current
   resolution, and it is the main reason going to full resolution helps less than you'd hope. Modelling blur
   added only +0.21 dB on ScanNet++ once colour and pose were handled ([Darmon et al.](https://arxiv.org/abs/2404.04211)).
   [Gaussian Splatting on the Move](https://arxiv.org/abs/2403.13327) models blur and rolling shutter
   together using VIO velocities, which ARKit gives us.
4. **Photometric differences between stills and frames.** This explains blotchy walls and some floaters [Checked
   + Inf]. Exposure, ISO and WB were locked on scan 3, so the large auto-exposure effect seen in published
   phone data mostly doesn't apply. The stills still come out ~6 % brighter and slightly bluer than the frames,
   because the high-resolution capture goes through a different ISP path. With SH degree 3 the trainer bakes
   this into view-dependent colour or extra splats. This is milder than (1) and (2).
5. **Under-observation.** This explains the mottled ceilings and parts of the hallway [Inf]. Few viewpoints
   make (1) worse. Depth supervision and capture changes (tilt up, slower in narrow spaces) address it.
6. **Densification and regularisation choices in msplat** [Checked in source]. There is no scale or anisotropy
   regularisation and no mip filter. Opacity resets every 3,000 steps until 15k. Densification is skipped
   unless `step % 3000 > num_train_cameras + 100` (the splatfacto rule,
   [model.cpp](https://github.com/rayanht/msplat/blob/main/core/src/model.cpp)). With ~1,400 training cameras
   it runs only in the second half of each 3,000-step cycle, and with ≥ 2,900 cameras it would never run. This
   helps explain why "more frames" behaves differently from what you'd expect.
7. **Resolution.** This explains uniform softness of fine texture, but not structure [Checked: your A/B tests].

What the hanging basket and window have in common [Inf]: high texture, seen frontally by many views within one
session, and (the window) little near geometry. These are the conditions where photometric 3DGS works without
priors and pose drift between sessions doesn't matter.

### How to confirm cheaply (Mac, no new tools)

- **Single-session splats**: build a dataset with only the hallway session (plus its stills) and train it with
  the same settings. If the doubled frames and most streaks disappear, cross-session poses are the main cause
  (do #3 first). If streaks remain, geometry ambiguity is (do #2 first).
- **Extrapolation eval**: train without the lap session's images (keep its poses in the graph) and score on
  them. The lap walks through all rooms on a different path, which is close to how the viewer is used.
- **Depth check**: render depth with `render_from_pose` (msplat renders RGB only, so use the exported PLY and a
  depth renderer, or Brush/gsplat later) and compare it with LiDAR confidence-2 depth on held-out frames. Report
  median |Δ| and the fraction of pixels > 5 cm.
- **Stills-only vs frames-only**: if the frames-only splat is sharper on walls, still/frame photometric or pose
  mismatch matters.

## 2. Memory limit: fixes

### What msplat does [Checked: source of v1.1.4]

- `Dataset::Dataset` decodes every image to **float32 RGB** (`struct Image { std::vector<float> data; }`), resized
  to the downscale factor ([input_data.cpp](https://github.com/rayanht/msplat/blob/main/core/src/input_data.cpp)).
- `splitTrainTest` **copies** the cameras, images included, into `trainCams`/`testCams`. The originals in
  `data.cameras` are never freed, which makes 2 CPU copies ([msplat_api.mm](https://github.com/rayanht/msplat/blob/main/core/src/msplat_api.mm)).
- `getImage(ds)` keeps a CPU pyramid per downscale level, and `getGPUImage(ds)` keeps a GPU copy per level,
  forever (default `num_downscales=2` → +1/4 + 1/16 each).
- Total ≈ 3.6 float copies ≈ **44 bytes per training pixel**. Scan 3 at full resolution is 4.4 Gpx, about
  195 GB. At 1/4.25 it is about 11 GB, which is what `MSPLAT_MAX_PIXELS` already guards.
- No option for lazy loading, uint8 or streaming. Issue [#5 "Memory Issue"](https://github.com/rayanht/msplat/issues/5)
  is open: the maintainer says he never measured memory (128 GB machine) and might do a pass. PR [#6](https://github.com/rayanht/msplat/pull/6)
  (open) fixes a separate **per-step leak** in the C++ CLI (missing `@autoreleasepool`). The Python binding's
  `step()` has no pool either, so watch RSS during long runs [Unverified for Python].

### Patches, smallest first

| Patch | Change | Bytes/pixel | Scan 3 at 1/2 (1.1 Gpx) | At 1/1.5 (2.0 Gpx) | Effort |
|---|---|---|---|---|---|
| A | `std::move` the cameras into train/test (or clear `data.cameras[i].image` after the split). Drop pyramid and GPU entries for levels the schedule has passed, or train with `--num-downscales 0` | ~16 | 18 GB (too tight on 24 GB) | — | 1–2 h |
| B | A + store `Image` as **uint8**, convert to float in `getGPUImage` into one reused per-step staging `MTensor` (unified memory: a ~3 ms CPU loop, or a tiny Metal kernel) | ~3 | 3.3 GB | 5.9 GB | ~1 day |
| C | Keep the **JPEG bytes** (or just paths) and decode per step on a prefetch thread with ImageIO (already used in `image_io.cpp`) | ~0.2 | <1 GB | <1 GB (full res feasible) | 1–2 days |

Build from source with `pip install -e .` (CMake, Xcode). Offer it upstream against issue #5. Others already
did exactly C: OpenSplat 1.2.2 added a "compressed memory cache… keeping images encoded in JPEG and decoding
them in memory" ([PR #250](https://github.com/WebODM/OpenSplat/pull/250)). gsplat's trainer now decodes one image
per step by default, with image caching opt-in (`--cache_images`, Sep 2026, [commits](https://github.com/nerfstudio-project/gsplat/commits/main)).
nerfstudio has `cache_images: disk` and `cache_images_type: uint8` ([full_images_datamanager.py](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datamanagers/full_images_datamanager.py)).

**Zero-code workaround (untested idea)**: two-stage training. Train all frames at 1/4.25 to ~20k steps, then
load the checkpoint into a second run at 1/2 on a subset (stills + the sharpest third of frames). The catch:
msplat normalises positions from the *camera set* (`autoScaleAndCenter`: mean and max-abs of camera
positions), so a different subset means a different internal frame. Positions, log-scales and the position
Adam moments must be remapped between the two frames. `msplat_train.py` already fits this transform
(`internal_frame`), so it is ~½ day.

### Brush

**No such limit.** Brush streams images from disk with a bounded cache (configurable since [#490](https://github.com/ArthurBrussee/brush/pull/490))
and uses u8 on Metal ([#442](https://github.com/ArthurBrussee/brush/pull/442)). Your 3 h full-resolution
run on scan 2 confirms it. `--max-resolution` (default 1920) only sets the training size.

### Other memory levers

- Random crops (`gsplat --patch_size`) and image tiling (nerfstudio [#3685](https://github.com/nerfstudio-project/nerfstudio/pull/3685))
  reduce per-step GPU memory, not the dataset's. They don't help msplat's problem.
- msplat's built-in coarse-to-fine (`num_downscales=2`, `resolution_schedule=3000`) already trains the first
  3k steps at 1/4 and the next 3k at 1/2 of the training size. At 1/4.25 that means a first stage only ~112 px
  wide. Fine for structure, but irrelevant to memory.
- Fewer, better frames (#4) also cut memory: 1,600 → ~900 sharp frames lets patch A alone reach ~1/2.5.

## 3. Tools

Status as of 2026-10-08, checked on GitHub unless noted.

| Tool | Licence | Platform | Depth sup. | Pose opt. | Exposure / appearance | Maintained? | Verdict |
|---|---|---|---|---|---|---|---|
| [msplat](https://github.com/rayanht/msplat) 1.1.4 | Apache-2.0 | Mac (Metal) | No | No | No | One maintainer. v1.1.4 Aug 2026, 95★ | Keep as the fast Mac baseline. Patch memory (§2). No regularisers |
| [Brush](https://github.com/ArthurBrussee/brush) 0.3.0 / main | Apache-2.0 | Mac (wgpu; compiles to MSL on macOS since Aug 2026), Win, Linux, web | No. Depth loss is open PR [#497](https://github.com/ArthurBrussee/brush/pull/497) (L1 on inverse depth, COLMAP-only, TIFF), not merged | No | No (masks only) | Very active (main Oct 2026). **No release since 0.3.0 (Sep 2025)** | Build **main** (`cargo run --release`): needle fixes (covariance-aware split, `--kill-at-screen-size`, area regulariser, lower `lr_scale`) [#479](https://github.com/ArthurBrussee/brush/pull/479), Mip-Splatting 2D/3D filters, faster training (#546, #554, #556), streaming. Still no pruning hook. The crates are a Rust library, so a custom loop is possible but not small. An FFI ([#308](https://github.com/ArthurBrussee/brush/pull/308), Nov 2025) only covers "train and save" [Unverified on main] |
| [OpenSplat](https://github.com/WebODM/OpenSplat) 1.2.2 | **AGPL-3.0** | Mac (MPS), CUDA, ROCm, CPU | No | No | No | Active (Sep 2026) | Memory solved (JPEG cache), masks. Otherwise the same 3DGS as msplat but slower. Not worth switching |
| [MetalSplat](https://github.com/tchauffi/metalsplat) 0.3.0 | MIT | Mac (PyTorch MPS + Metal kernels) | Renders differentiable depth (and 2DGS normals), so a LiDAR loss is a few lines of Python | Could be added in PyTorch | No | New (Sep 2026, 11★) | Promising: the only Mac trainer with Python-level access to depth. Far too slow today (15k steps on 161 garden images in 42 min, 330k splats). Watch it |
| [LuluSplat](https://liquiduslabs.com/lulusplat/) 1.1 | Proprietary (free dmg, $14.99 App Store) | Mac app (msplat + metal-gauss MCMC) | No | No | Not mentioned | Active (Sep 2026) | MCMC and blur rejection on Mac, CLI/MCP. No LiDAR, no pose opt. Not a fit ([article](https://radiancefields.com/lulusplat-trains-gaussian-splats-entirely-on-apple-silicon-macs)) |
| [gsplat](https://github.com/nerfstudio-project/gsplat) 1.5.3 (latest release; `main` says 1.6.0, unreleased) `simple_trainer.py` | Apache-2.0 | **CUDA only**. The MPS fork [gsplat-mps](https://github.com/iffyloop/gsplat-mps) is AGPL and dormant since 2024 | `--depth_loss` uses sparse COLMAP points only. Dense LiDAR needs a custom parser and loss (~100 lines) | `--pose_opt` (`pose_opt_lr 1e-5`) | `--app_opt`, `--post_processing bilateral_grid` or `ppisp` | Active (Sep 2026) | **Best cloud base**: MCMC (`mcmc --scale_reg 0.01 --opacity_reg 0.01`), `--antialiased` (Mip-Splatting), `absgrad` (AbsGS), `--visible_adam` (Taming 3DGS), 3DGUT with rolling-shutter support ([docs/3dgut.md](https://github.com/nerfstudio-project/gsplat/blob/main/docs/3dgut.md)). Direct tensor access, so our LiDAR pruning ports in ~50 lines. The parser is COLMAP-only, so convert or write one |
| [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) splatfacto / splatfacto-big | Apache-2.0 | CUDA (gsplat). No MPS | Not in core | `camera_optimizer` (SO3xR3) | `use_bilateral_grid` | **Stale**: last commit Jul 2025, last release 1.1.5 (Nov 2024) | Reads our `transforms.json` as is. `use_scale_regularization` + `max_gauss_ratio 10` (PhysGaussian cap), `strategy mcmc`, `cache_images disk`, `cache_images_type uint8` ([splatfacto.py](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py)) |
| [DN-Splatter / AGS-Mesh](https://github.com/maturk/dn-splatter) | Apache-2.0 | CUDA (nerfstudio + gsplat 1.0) | **Yes**: sensor depth (`--pipeline.model.use-depth-loss True --pipeline.model.depth-loss-type EdgeAwareLogL1 --pipeline.model.depth-lambda 0.2`), normals from depth or mono (`--pipeline.model.normal-supervision depth`). AGS-Mesh filters bad priors adaptively | Through nerfstudio's camera optimiser [Unverified for these models] | Splatfacto's options [Unverified] | Stale: last feature Nov 2024, pushed Jul 2025, 841★ | **Try first on cloud.** Built for exactly this (phone LiDAR, indoor rooms; MuSHRoom iPhone). Dataparser `normal-nerfstudio` on our dataset. Expect version-pinning friction (old nerfstudio/gsplat). Tested on rooms with iPhone LiDAR ([paper](https://arxiv.org/abs/2403.17822), [AGS-Mesh](https://arxiv.org/abs/2411.19271)) |
| [3dgs-deblur](https://github.com/spectacularai/3dgs-deblur) ("Splatting on the Move") | [Unverified] | CUDA (nerfstudio fork, ≥ 8 GB) | No | Yes, plus per-frame velocity | Motion blur + rolling shutter (both at once need exposure and readout times) | 2024 | Only if diagnostics show blur or rolling shutter matter. We log exposure (8.3 ms) and have ARKit velocities, but not readout time ([paper](https://arxiv.org/abs/2403.13327)) |
| BAD-Gaussians, Deblurring 3DGS | non-commercial (3DGS licence) [Unverified] | CUDA | No | BAD: yes (trajectory in exposure) | No | Research | Superseded for us by sharp-frame selection plus the above |
| 2DGS, PGSR, GOF, RaDe-GS | mostly inherit the Inria non-commercial licence [Unverified per repo]. 2DGS also lives in gsplat (Apache) | CUDA | Geometry-focused. PGSR has exposure compensation | No | PGSR: yes | Research, 2024 | For meshes and accurate surfaces, not prettier splats. gsplat's 2DGS if we ever want splat-derived geometry |
| Mip-Splatting / AbsGS / 3DGS-MCMC / Taming 3DGS | (techniques) | In gsplat (CUDA). Mip filters are also in Brush main | — | — | — | — | Use through gsplat flags. MCMC's scale/opacity L1 controls splat count. It is not a needle fix on its own (no paper shows that) |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | GPL-3.0 | CUDA 12.8+ (Win/Linux; Windows binaries are paid) | [Unverified] | [Unverified] | Bilateral grid | Very active | MCMC, bilateral grid, 3DGUT, Python/MCP. COLMAP input. No LiDAR story. Skip |
| [Postshot](https://www.jawset.com/) | Proprietary | **Windows + NVIDIA only** | No | Imports poses | Not stated | Active | Not usable on the Mac |
| Scaniverse, Polycam, KIRI, Voxelio, SplatKing | Commercial apps | iOS / cloud | Internal | Polycam: globally optimised poses ([nerfstudio docs](https://docs.nerf.studio/quickstart/custom_dataset.html)) | Internal | Active | Useful as a quality reference for one room. They can't use our poses or pose graph, and only some export training data |
| Apple RealityKit `GaussianSplatComponent` (OS 27) | — | Apple platforms | — | — | — | New | **Rendering only**, no training ([docs](https://developer.apple.com/documentation/realitykit/gaussiansplatcomponent), [WWDC26 279](https://developer.apple.com/videos/play/wwdc2026/279/)) |
| Feed-forward: [Depth Anything 3](https://arxiv.org/abs/2511.10647), [MapAnything](https://arxiv.org/abs/2509.13414), [WorldMirror](https://arxiv.org/abs/2510.10726), [AnySplat](https://arxiv.org/abs/2505.23716) | various | CUDA | MapAnything / WorldMirror take poses + depth as inputs | — | — | Research | Not a replacement at 1,600 frames: below per-scene optimisation. Possible side use: dense, multi-view-consistent depth or normals for regions the 256×192 LiDAR misses |
| CF-3DGS, InstantSplat | various | CUDA | — | Pose-free (sparse views) | — | Research | Built for unknown poses and few images. Not our case |
| pycolmap / GLOMAP / hloc | BSD / Apache | **Mac CPU** (pycolmap 4.2.1 wheels) | — | BA with position and gravity priors | — | Active | The way to fix cross-session drift (see `2026-10-06-arkit-roomplan-best-practices.md`). Caution: COLMAP triangulation + prior-seeded BA made ARKit priors *worse* in all 15 ScanNet++ rooms (0.55° → 0.74°) ([Kyriazis 2026](https://arxiv.org/abs/2608.21008)). Gate every BA result with the photometric/depth eval, and keep the pose graph as the prior |

On the Mac, no trainer today has LiDAR depth loss, pose optimisation and colour correction together. The
realistic choice is: (a) cloud CUDA for gsplat/DN-Splatter with everything, or (b) the Mac with pre-fixed poses
(BA), sharp frames and pruning, plus small msplat patches. A per-image affine colour correction or a depth loss
in msplat means Metal kernel work. The rasteriser is "nd" (multi-channel), so an expected-depth channel is
plausible, but the gradient to the means needs care. That is a multi-day job [Inf].

Cloud cost [Comm/Inf, rough]: an RTX 4090 / L40S / A100 rents for about $0.5–2 per hour. A 30k-step
splatfacto/DN-Splatter run on ~1,600 images at ≤ 1/2 resolution should take well under an hour. The work is
in the environment (pin nerfstudio 1.1.x + gsplat 1.0/1.4 for DN-Splatter), not the GPU bill.

## 4. Pipeline and capture: are we doing it the right way?

Mostly yes. The capture protocol, exposure lock, per-image intrinsics, still focal fix, TSDF init and LiDAR
pruning are all what an expert would do. The gaps:

**Evaluation (change first).** Every-8th-frame hold-out with frames 0.5 s apart measures interpolation. Add an
off-trajectory set (lap session or stills) and a depth error against LiDAR, and look at both in the viewer. All
of the screenshot's artifacts are invisible to the current PSNR.

**Poses.** The pose graph got leave-one-pair-out to 2.1 cm, but splats need consistency in pixels, and the
residuals are still 2.5 cm median (6.2 cm p90) at closures. Planned BA with image features is the next real step
for both splats and textures. In-training pose optimisation then cleans up the last pixels (cloud).
Rolling-shutter and blur modelling come last.

**Use the LiDAR for supervision, not only pruning.** `depth/` is already written. Better still is depth (and
normals) rendered from the fused TSDF mesh per view at training resolution. It is denser and higher-resolution
than 256×192, consistent with the poses, and hole-filled [Inf, untested]. Weight it down where the LiDAR is
weak (glass, > 4.5 m, thin objects), as AGS-Mesh does.

**Frames.**
- Choose the sharpest frame per 0.5 s window, not the first. Predicted blur from the poses (f·ω·8.3 ms plus
  translation over depth) is free and better than image measures on blank walls. Drop frames above ~1.5 px at
  the training resolution.
- Weight the stills up (they are 3× sharper) or sample them more often per epoch.
- HEVC at 16.7 Mbps for 1920×1440@60 is thin for fine wall texture. 40–60 Mbps, or 30 fps at the same bitrate,
  is cheap in the app [Inf].

**Photometrics.** Keep the exposure lock (it worked: constant 1/120 s, ISO 284 across all sessions). Correct
stills ↔ frames with a global per-session 3×3 colour matrix fitted on still/nearest-frame pairs (we already
match them in `stillcal.py`), or learn it in training (bilateral grid / PPISP on cloud).

**Undistortion.** The `OPENCV` camera model has no distortion coefficients. ARKit frames are treated as pinhole.
The 0.5–0.8 px homography residual in `stillcal` suggests distortion is small at 1920 px and negligible at
450 px [Inf, unverified at the image corners].

**Training settings (msplat).**
- `--num-downscales 0` once memory allows (the coarse start costs ~0.8 dB at 7k in msplat's own benchmark).
- Lower SH degree for the viewer (SH 1 lost 0.3 dB on the interpolation metric, but may look cleaner off
  trajectory: re-test with the extrapolation set).
- Keep the train-camera count well under 2,900 because of the densification gate (§1.6).

**Capture.**
- Slower in narrow spaces: the hallway p90 is 47 °/s, which means 9 px of blur at full resolution. Aim for
  < 20 °/s.
- More frontal views of hallway walls: walk the hallway twice, once facing each wall, not only along it.
- Tilt up for ceilings.
- Use auto-shoot stills in narrow rooms.

**What an expert would change first, in order:** the eval set and depth metric → single-session test to find
the dominant cause → depth supervision (cloud DN-Splatter) and cross-session BA → sharp-frame selection →
colour correction → regularisers → resolution.

## References

Our code and data: `pipeline/scanmodel/stages/splat.py`, `pipeline/scanmodel/msplat_train.py`, `data/derived/poses/manifest.json`,
`data/raw/third/*/frames.jsonl` (exposure, poses), `docs/plans/2026-10-06-scan-to-model.md` (phase 4, "Third scan").

Trainers and code:
- msplat: https://github.com/rayanht/msplat, PyPI https://pypi.org/project/msplat/, issue #5 https://github.com/rayanht/msplat/issues/5, PR #6 https://github.com/rayanht/msplat/pull/6
- Brush: https://github.com/ArthurBrussee/brush, degenerate splats #479 https://github.com/ArthurBrussee/brush/pull/479, depth PR #497 https://github.com/ArthurBrussee/brush/pull/497, cache #490 https://github.com/ArthurBrussee/brush/pull/490
- OpenSplat: https://github.com/WebODM/OpenSplat, JPEG cache https://github.com/WebODM/OpenSplat/pull/250
- MetalSplat: https://github.com/tchauffi/metalsplat, https://radiancefields.com/metalsplat-v0.3.0-trains-3dgs-and-2dgs-on-apple-silicon
- LuluSplat: https://liquiduslabs.com/lulusplat/, https://radiancefields.com/lulusplat-trains-gaussian-splats-entirely-on-apple-silicon-macs
- gsplat: https://github.com/nerfstudio-project/gsplat (`examples/simple_trainer.py`), paper https://arxiv.org/abs/2409.06765; gsplat-mps https://github.com/iffyloop/gsplat-mps
- nerfstudio: https://github.com/nerfstudio-project/nerfstudio, custom data / Polycam poses https://docs.nerf.studio/quickstart/custom_dataset.html
- DN-Splatter: https://github.com/maturk/dn-splatter, https://arxiv.org/abs/2403.17822; AGS-Mesh https://arxiv.org/abs/2411.19271
- 3dgs-deblur: https://github.com/spectacularai/3dgs-deblur
- LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio; Postshot: https://www.jawset.com/
- Apple RealityKit splats: https://developer.apple.com/documentation/realitykit/gaussiansplatcomponent, https://developer.apple.com/videos/play/wwdc2026/279/

Papers:
- Robust Gaussian Splatting (blur, pose, colour on ScanNet++ iPhone): https://arxiv.org/abs/2404.04211
- Triangulation-free BA from ARKit priors (ARKit error, splat PSNR by pose source): https://arxiv.org/abs/2608.21008
- Gaussian Splatting on the Move (blur + rolling shutter with VIO): https://arxiv.org/abs/2403.13327
- Effective rank regularisation (needles): https://arxiv.org/abs/2406.11672
- 3DGS-MCMC: https://arxiv.org/abs/2404.09591; Mip-Splatting: https://arxiv.org/abs/2311.16493; AbsGS: https://arxiv.org/abs/2404.10484; Taming 3DGS: https://arxiv.org/abs/2406.15643
- Bilateral grid: https://arxiv.org/abs/2406.00448; PPISP (CVPR 2026): https://arxiv.org/abs/2601.18336
- PhysGaussian (scale-ratio regulariser used by splatfacto): https://arxiv.org/abs/2311.12198
- 2DGS: https://arxiv.org/abs/2403.17888; PGSR: https://arxiv.org/abs/2406.06521; GOF: https://arxiv.org/abs/2404.10772; RaDe-GS: https://arxiv.org/abs/2406.01467
- BAD-Gaussians: https://arxiv.org/abs/2403.11831; Deblurring 3DGS: https://arxiv.org/abs/2401.00834
- CF-3DGS: https://arxiv.org/abs/2312.07504; InstantSplat: https://arxiv.org/abs/2403.20309
- Depth Anything 3: https://arxiv.org/abs/2511.10647; MapAnything: https://arxiv.org/abs/2509.13414; WorldMirror: https://arxiv.org/abs/2510.10726; AnySplat: https://arxiv.org/abs/2505.23716
- GLOMAP: https://arxiv.org/abs/2407.20219

Practitioner reports:
- "I Tried to Make Gaussian Splatting 1,000× Faster" (BA from ARKit poses in 0.8 s, msplat vs Brush on Mac): https://radiancefields.com/i-tried-to-make-gaussian-splatting-1-000×-faster
- Mac trainer overview: https://radiancefields.com/tags/mac
- Phone scanning apps 2026 (vendor-written): https://www.kiriengine.app/blog/best-lidar-3d-scanner-apps-iphone-2026
