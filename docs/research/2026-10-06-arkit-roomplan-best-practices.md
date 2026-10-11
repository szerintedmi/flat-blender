# ARKit / RoomPlan multi-room capture: best practices, drift and wall thickness

Date: 2026-10-06

Labels: **[Doc]** = documented by the vendor, a paper or a package (primary source). **[Comm]** = community report (forum,
blog, vendor marketing with no method given). **[Inf]** = my inference. **[Checked]** = I checked it myself (for example
by inspecting a package).

## Executive summary

- **Nothing in Apple's stack fixes our problem.** RoomPlan surfaces have no thickness [Doc]. StructureBuilder merges
  rooms that already share a coordinate frame, but Apple gives no drift correction and no accuracy figure for merged
  structures [Doc]. ARKit has no API for refined past camera poses [Doc/Comm].
- **ARAnchors are the only official way to get loop-closure corrections back.** An Apple engineer says anchor
  transforms follow map refinement. The person who asked tried per-frame anchors and saw no change (Aug 2026) [Comm].
  It costs little to test.
- **Check the iPad's iOS version.** There is a known ARKit drift regression on LiDAR devices in iOS 26.4–26.6, which
  Apple believes is fixed in 26.6.1 [Comm, acknowledged by Apple].
- **Commercial apps mostly assume wall thickness.** Polycam uses 10 cm by default, magicplan one interior and one
  exterior value per floor, and Canvas infers it "from construction standards" [Doc]. Their stated accuracy is
  1–2 % or 1–5 cm, which is far from our 2 cm p95 target over a whole flat [Doc].
- **The standard fix for time-continuous drift is a keyframe pose graph or BA.** Each keyframe is a node, ARKit's
  relative motion gives odometry edges, and visual (plus depth) loop closures connect sessions. This is our time-knot
  model with a principled smoothness prior, which our version lacked [Inf, standard SLAM practice].
- **pycolmap 4.2.1 has wheels for Python 3.14 on macOS arm64.** It includes ALIKED+LightGlue matching, a global mapper
  (GLOMAP), and bundle adjustment with position and gravity priors [Checked]. We don't need PyTorch/MPS for matching.
- **Capture changes are cheap and probably worth more than more pipeline cleverness.** End with a closing lap through
  all rooms. Sweep both faces of each doorway wall in one go. Log anchors. Keep doors open for overlap.
- **Realistic target** [Inf]: about 1 cm within a room (we already have 4–5 mm face rms), 1.5–3 cm between rooms after
  global optimisation. Wall thickness measured to ±1 cm only where one short time window sees both faces.

## 1. ARKit / RoomPlan: what Apple provides and recommends

### Multi-room RoomPlan

- **Two ways to share a frame** [Doc] ([WWDC23 "Explore enhancements to RoomPlan"](https://developer.apple.com/videos/play/wwdc2023/10192/)):
  - Keep one ARSession running: `roomCaptureSession.stop(pauseARSession: false)`, then `run` again. This is what our
    Next room does.
  - Relocalise with an `ARWorldMap` (`initialWorldMap`), recommended for scans taken "on the next day or week".
  - `StructureBuilder(options: [.beautifyObjects]).capturedStructure(from: rooms)` then merges the rooms into a
    `CapturedStructure`. It holds walls, doors, windows, openings, objects, floors and sections, and removes the
    duplicate walls/objects that manual stitching would leave.
- **Recommended scope** [Doc, same session]:
  - single-floor homes, "one to four bedrooms";
  - at most about 186 m² in total;
  - at least 50 lux of light.
- **No drift handling** [Doc]: the session gives no accuracy figures for merged structures and no API to correct drift
  between rooms.
- **Single-room guidance** [Doc] ([WWDC22 10127](https://developer.apple.com/videos/play/wwdc2022/10127/)):
  - rooms up to about 9 × 9 m;
  - avoid full-height mirrors/glass, high ceilings and dark surfaces;
  - "avoid repeated scans or single long scans over 5 minutes" (thermal);
  - close doors to avoid scanning outside the room.

  Closed doors conflict with our need for overlap between rooms [Inf].
- **Apple's published accuracy figures are detection rates, not dimensions** [Doc]
  ([Apple ML Research](https://machinelearning.apple.com/research/roomplan)):
  - about 95 % precision/recall for walls and windows, 90 % for doors;
  - walls are 2D lines detected in a bird's-eye view, then lifted to 3D using the wall height;
  - rooms up to 15 × 15 m.

  **I found no official dimensional accuracy for RoomPlan.**
- **Wall thickness** [Doc]: an Apple DTS engineer says "The surfaces returned by RoomPlan do not provide a thickness,
  they are estimated planar surfaces". The API also does not tell interior from exterior side
  ([forum 837547](https://developer.apple.com/forums/thread/837547)). StructureBuilder therefore deduplicates walls but
  does not estimate thickness. I found no documentation of how it chooses between two disagreeing faces.
- **StructureBuilder side effects** [Comm] ([forum 760952](https://developer.apple.com/forums/thread/760952)):
  - it re-transforms the first room;
  - it creates new UUIDs for merged walls;
  - relocalised world maps can leave earlier frames offset.

  Don't rely on StructureBuilder output for frame-to-room bookkeeping.

### ARKit drift, loop closure and anchors

- **Past poses are never corrected.** ARKit gives no notification of loop-closure jumps and no API for refined
  historical poses ([forum 706223](https://developer.apple.com/forums/thread/706223),
  [forum 842466](https://developer.apple.com/forums/thread/842466)) [Comm, confirmed by an Apple engineer in 842466].
- **The anchor workaround.** The Apple engineer's suggestion in 842466 is to add an `ARAnchor(transform:
  frame.camera.transform)` per frame: "ARKit propagates those corrections into all affected anchors". The asker
  reported that the anchors "were not refined… the value is not changing" (Aug 2026) [Comm]. An older report says
  anchors "do move an appreciable amount when you rescan the same area" (iOS 15)
  ([forum 690668](https://developer.apple.com/forums/thread/690668)) [Comm]. So the evidence conflicts. Worth a
  15-minute test on our device.
- **World maps keep the latest anchor transforms** [Doc] ([Saving and loading world data](https://developer.apple.com/documentation/arkit/saving-and-loading-world-data)):
  `ARWorldMap.anchors` is a snapshot of the session's anchors *with their current transforms* at save time. It also
  carries `rawFeaturePoints`. Relocalisation works best "in areas… visited during the previous session" under the same
  lighting.

  [Inf] So if anchors are refined at all, a world map saved at the end of the capture holds the refined set. We already
  save world maps, so we only need to add anchors.
- **iOS 26.4+ regression** [Comm, acknowledged as a "known issue" by Apple]
  ([forum 827240](https://developer.apple.com/forums/thread/827240)):
  - continuous world-tracking drift on LiDAR devices only;
  - reproducible on 26.6;
  - Apple replied on FB23200349 that it is believed fixed in iOS 26.6.1.

  If the iPad ran 26.4–26.6, part of our 5–15 cm may be this bug.
- **Depth** [Doc] ([ARDepthData.confidenceMap](https://developer.apple.com/documentation/arkit/ardepthdata/confidencemap)):
  confidence 0–2 (low/medium/high), lower on reflective or absorbing surfaces. `smoothedSceneDepth` averages over
  frames. For TSDF the usual practice is to keep only high confidence (and medium within about 3 m) [Inf].
- **Recommended motion.** I found no Apple text beyond "a good scanning motion" [Doc]. Vendors give these tips:
  - Canvas: a "fence painting" pattern ([Canvas accuracy](https://support.canvas.io/article/5-what-kind-of-accuracy-can-i-expect-from-canvas)) [Doc].
  - Polycam:
    - start at recognisable features such as a doorway;
    - move slowly through transitions and keep shared features visible;
    - "rooms may fail to align if doorways or hallways lack overlap";
    - avoid re-scanning the same area, because conflicting data lowers accuracy;
    - drift grows in open, feature-poor areas.

    ([Polycam best practices](https://learn.poly.cam/hc/en-us/articles/34295265957140-Improving-Scan-Accuracy-Best-Practices-for-Interior-Spaces))
    [Doc; the page returned 403 to me, so this is from the search-indexed text].

## 2. How others solve it

### Commercial apps

| App | Alignment approach (known) | Wall thickness | Stated accuracy |
|---|---|---|---|
| Polycam (LiDAR/Room/Space) | On-device or cloud processing. Cloud is "better for complex or multi-room" spaces; no method published | Assumed 10 cm, editable ([exterior measurements](https://learn.poly.cam/hc/en-us/articles/52371263328788-How-Polycam-Calculates-Exterior-Measurements)) [Doc] | Wall-to-wall about 1.3–5 cm against a laser ([accuracy](https://learn.poly.cam/hc/en-us/articles/50434745985556-How-Accurate-Are-Polycam-Scans)) [Doc] |
| magicplan | LiDAR auto-scan of several rooms, plus manual "assemble rooms" | One interior and one exterior value per floor, not per wall ([help](https://help.magicplan.app/change-exterior-and-interior-wall-thickness)) [Doc] | none found |
| Canvas | Cloud "Scan to CAD" with human drafters; scans merged per room/floor | "does not observe [thickness] directly… construction standards"; exterior scans help ([accuracy](https://support.canvas.io/article/5-what-kind-of-accuracy-can-i-expect-from-canvas)) [Doc] | 1–2 % of tape/laser [Doc] |
| Matterport (iPhone) | Stationary "scan points" aligned in the cloud (Cortex). LiDAR shortens point spacing to 2–3 m ([forum](https://www.wegetaroundnetwork.com/topic/13777/capture-41beta-via-testflight-is-using-built-into-ios-devices-lidar)) [Comm] | not documented | iPhone scanner vs Pro3 on a large building: mean cloud-to-cloud 4.1 cm, sd 7.2 cm, max 52 cm at doors ([arXiv 2506.14771](https://arxiv.org/html/2506.14771v1)) [Doc] |
| CubiCasa | Video plus AI, human QA | Assumed (standard values) [Comm] | Vendor says within about 3 %; independent MLS tests found larger errors ([ARMLS](https://armls.com/how-accurate-are-cubicasa-floorplan-scans)) [Comm] |
| Scaniverse, 3D Scanner App, Locometric RoomScan | On-device; no published method for multi-room alignment | not documented | none found |

Takeaways:

- Nobody we can see *measures* wall thickness from consumer LiDAR. They assume it, ask the user, or have a human
  drafter infer it [Doc].
- The tools that do multi-room well either constrain the capture (stationary scan points, overlap at doorways) or use
  cloud/human post-processing [Doc/Comm].
- **I found no public description of Polycam's or Canvas's alignment algorithm.**

### Academic / open-source

- **ARKit pose quality** [Doc]:
  - ScanNet++ iPhone room captures: ARKit poses are about 0.55° / 19 mm scene-mean against laser-registered ground
    truth.
  - Plain COLMAP triangulation plus prior-seeded BA made them *worse* in all 15 test scenes (0.55° → 0.74°). The
    triangulated structure bakes in the prior's error.
  - A triangulation-free BA fixes this (Kyriazis, [arXiv 2608.21008](https://arxiv.org/abs/2608.21008), Aug 2026;
    code "to be released").
  - ScanNet++ itself registers iPhone frames to a laser scan via COLMAP plus dense photometric refinement, and drops
    frames whose depth disagrees by more than 0.3 m ([arXiv 2308.11417](https://arxiv.org/abs/2308.11417)).
    ARKitScenes does the same against a Faro scanner ([arXiv 2111.08897](https://arxiv.org/abs/2111.08897)).

  Lesson [Inf]: the published "good poses" all come from registration against a laser scan, which we don't have.
  Naive BA can hurt rotations. It still helps us, because our error is cross-session translation of 5–15 cm, not
  sub-degree rotation.
- **BundleFusion** [Doc] ([arXiv 1604.01093](https://arxiv.org/abs/1604.01093)):
  - global optimisation of all frames, in two passes: sparse features first, then dense geometric and photometric
    terms;
  - re-integrates the TSDF when poses change.

  This is the reference design for "LiDAR + features + re-fuse". The code is CUDA-only and dated; use it as a recipe,
  not a tool.
- **Open3D reconstruction system** [Doc] ([tutorial](https://www.open3d.org/html/tutorial/reconstruction_system/index.html)):
  - fragments from short frame windows;
  - pairwise fragment registration (global, then ICP/colored ICP);
  - pose-graph optimisation with loop-closure edges marked "uncertain" (robust line process);
  - optional non-rigid **SLAC** with a shared control grid
    ([API](https://www.open3d.org/html/python_api/open3d.t.pipelines.slac.html)), which absorbs low-frequency
    deformation.

  [Inf] This matches our drift better than our per-session rigid fix, because a fragment of about 20–60 s is nearly
  rigid while a 3-minute session is not.
- **ORB-SLAM3 Atlas / Kimera** [Inf]: these use the same multi-session pose graph plus place recognition. They make
  sense only if we wanted to re-track from raw video. With ARKit odometry already recorded, the back-end alone (GTSAM)
  is enough.
- **Plane- or BIM-constrained BA** [Doc]: BIMCaP ([arXiv 2412.03434](https://arxiv.org/abs/2412.03434)) refines mobile
  poses against known BIM planes. That suits a *second* pass once we have walls, adding point-to-plane factors for the
  wall points of each keyframe.
- **Feed-forward models** [Doc]: MapAnything ([arXiv 2509.13414](https://arxiv.org/abs/2509.13414)) accepts images
  plus optional poses and depth and outputs metric geometry. It's interesting as a source of loop-closure candidates,
  but it's unproven at the cm level and needs a GPU [Inf].

### Wall thickness, in practice

- **Volumetric scan-to-BIM.** Methods such as Ochmann et al.
  ([arXiv 1907.00631](https://arxiv.org/abs/1907.00631)) take thickness as the gap between opposite room faces [Doc].
  That needs a *registered* cloud (terrestrial laser scans), which is exactly what we lack.
- **Default thickness.** PointCloud2BIM-style tools use an expected thickness to detect walls [Doc].
- **Consumer apps** assume a value or take user input (see the table above).
- **Door jambs.** I found no paper or app that infers thickness from door jambs. Our own experience says jambs are
  noisy: architraves, door leaves.

## 3. Toolchain review

| Choice | Verdict |
|---|---|
| Own capture app vs RoomPlan export / Polycam / Record3D / Stray / NeRFCapture | **Keep ours.** None of these refine poses: they record ARKit poses, as our app does ([Stray format](https://github.com/strayrobots/scanner), [nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html)) [Doc]. Only our app can add anchors, high-res stills with depth, and a protocol (closing lap, doorway sweeps). |
| Open3D tensor VoxelBlockGrid TSDF | **Fine.** The drift is in the poses, not the fusion. Changes worth making: re-integrate after pose optimisation (BundleFusion-style), and filter on confidence. Open3D runs on CPU on macOS (no Metal), which is fine at our scale [Inf]. |
| Feature matching on Mac | **Use pycolmap 4.2.1**, not hloc+PyTorch. It has cp314 macOS-arm64 wheels, and `FeatureExtractorType` includes `ALIKED_N16ROT/ALIKED_N32/LOMA_*`, `FeatureMatcherType` includes `ALIKED_LIGHTGLUE`, `SIFT_LIGHTGLUE` [Checked]. COLMAP 4.2 added ONNX matching with macOS CoreML execution ([radiancefields](https://radiancefields.com/colmap-4.2.0-adds-multi-component-global-mapping-and-loma-learned-matching)) [Doc]. LightGlue itself publishes no MPS numbers ([repo](https://github.com/cvg/LightGlue)); I did not benchmark either. |
| BA with ARKit priors | pycolmap has `PosePrior` with **position (plus covariance) and gravity only, no full rotation prior**, `create_pose_prior_bundle_adjuster`, `GlobalPipeline` (GLOMAP, merged into COLMAP 4.0), and `triangulate_points` [Checked]. Usable, but mind the Kyriazis result above. |
| Pose graph | GTSAM 4.3.0 has cp314 macOS wheels ([PyPI](https://pypi.org/project/gtsam/)) [Checked]. It offers robust kernels, GNC and `Pose3` between factors. Open3D's `PoseGraph` is simpler but less flexible. |
| Splats | **Brush** works on macOS via wgpu and reads COLMAP/nerfstudio, but its README mentions no depth supervision or pose optimisation ([repo](https://github.com/ArthurBrussee/brush)) [Doc]. Alternatives on Apple Silicon: **msplat** (Metal, about 4× faster than gsplat per its own benchmark, COLMAP/nerfstudio/Polycam input, Apache-2.0, Aug 2026) ([PyPI](https://pypi.org/project/msplat/)) [Doc]; **OpenSplat** (Metal build) ([radiancefields](https://radiancefields.com/opensplat-3dgs-now-on-mac)) [Doc]. nerfstudio splatfacto/gsplat and DN-Splatter need CUDA [Doc/Comm]. Postshot: I found only Windows downloads [Comm]. Scaniverse makes splats on the device but gives no control over poses [Doc]. Recommendation: keep Brush for now and feed it *corrected* poses. That matters more than the trainer choice [Inf]. |
| Blender splats | Blender 5.3 (release expected 10 Nov) imports PLY/SPZ/USD splats natively, renders them in Workbench/EEVEE/Cycles, and can't export them yet. Performance is "not ideal" and colours assume sRGB ([Digital Production](https://digitalproduction.com/2026/09/22/blender-5-3-brings-gaussian-splats-home/), [Cinevva](https://app.cinevva.com/news/2026-09-16-blender-5-3-gaussian-splats)) [Doc/Comm]. Good enough for preview; keep the web viewer (Spark) for delivery. |
| Full BA vs keyframe pose graph vs non-rigid | Start with a **keyframe pose graph**. It needs only relative-pose loop closures (2D–2D matches lifted to 3D with LiDAR depth, then Procrustes/PnP with RANSAC), and handles time-continuous drift directly. Add BA (pycolmap, position priors) as a second-stage polish, or SLAC-style non-rigid fragments if residual bending remains [Inf]. |

## 4. Why our earlier attempts failed

[Inf]

- **Rigid per-session corrections** can't represent drift that grows with time.
- **Wall-only ICP is degenerate.** Planes don't constrain sliding along them. A hallway constrains one axis, and the
  glass conservatory almost none. So leave-one-out swings by 5–28 cm.
- **The time-knot model** has the right shape but no prior tying knots to ARKit's *relative* motion, so it overfits.
  - In a pose graph, odometry edges between consecutive keyframes carry ARKit's local accuracy (on the order of mm to
    1 cm over a few seconds) as their information matrix.
  - Loop closures enter with robust kernels.
  - Knots can then be dense (every 1–2 s) without overfitting, because moving one knot costs odometry residual.
  - Keep 4-DOF (x, y, z, yaw) or fix roll/pitch with strong gravity priors: ARKit observes gravity well.
- **Validation** should stay leave-one-loop-closure-out plus a few independent checks. A laser distance meter across
  2–3 doorway-spanning diagonals is optional, used only for validation and never as model input.

## Recommendations for flat-blender

Ranked by expected impact per effort.

1. **Check the iOS version; update to ≥ 26.6.1 if it is 26.4–26.6.** Effort S.
   - Impact: possibly large, since the known LiDAR-only drift regression could explain part of the 5–15 cm.
   - Try first: read Settings → General → About, then rescan one room pair and compare against the second scan.
2. **Capture: closing lap plus doorway sweeps.** Effort S (protocol), S–M (app prompts).
   - Impact: high. Gives loop closures between early and late frames, and thickness measured within seconds.
   - After the last room, walk back through every room to the start, slowly, at chest height, recording frames.
   - In each doorway, stand in the opening and sweep the wall faces 0.3–1.5 m either side of the door on *both* sides
     within about 10 s. Same ARSession, so drift is negligible.
   - Prefer a hub-and-spoke order (return to the hallway between rooms) so ARKit itself relocalises often.
   - Keep doors open. Pan back into the previous room before pressing Next room.
3. **Log ARAnchors and test if ARKit refines them.** Effort S.
   - Impact: unknown, ranging from zero to "free drift correction".
   - Add an anchor at each still and every 1 s.
   - Record `didUpdate` anchors and the final `getCurrentWorldMap().anchors` per session and at the end.
   - Offline: if anchors moved, interpolate the correction between anchors to adjust frame poses.
   - First test: one loop around the flat, then compare start and end anchor transforms.
4. **Keyframe pose graph with visual+depth loop closures (GTSAM).** Effort M.
   - Impact: high. This is the standard fix for exactly this failure.
   - Keyframes at about 1 s, or every still. Odometry edges come from ARKit relative poses.
   - Loop-closure candidates: by ARKit proximity and a retrieval/ALIKED+LightGlue match via pycolmap. Lift to 3D with
     LiDAR depth (high confidence), then RANSAC Procrustes with ≥ 30 inliers and residual < 2 cm.
   - Robust kernel (Cauchy/GNC). Then re-fuse the TSDF with the corrected poses.
   - First step: run the second scan's existing SIFT matches through this instead of the time-knot fit, and compare
     leave-one-out stability.
5. **Fragment-based registration as a geometric cross-check** (Open3D reconstruction system with ARKit initial guesses,
   colored ICP between overlapping fragments as extra pose-graph edges). Effort M.
   - Impact: medium. Color constrains the sliding along walls that defeated plain ICP.
6. **Thickness policy.** Effort S.
   - Impact: medium, and it gives honest output.
   - Measured only from a ≤ 30 s window that sees both faces (doorway sweep). Otherwise use the default with a flag,
     as now.
   - Never compute thickness from faces of different sessions before item 4 has converged. After it, accept only when
     cross-session and single-window values agree within 1.5 cm.
7. **BA polish with pycolmap** (`PosePrior` positions from the pose-graph result, gravity priors, intrinsics fixed). Use
   the stills only. Effort M.
   - Impact: medium for texture/splat sharpness, low for walls.
   - Watch the documented risk that BA degrades already-good priors; compare against item 4 on held-out loop closures.
8. **Plane factors in the graph** (keyframe wall points to the merged wall planes, BIMCaP-style) once walls are stable.
   Effort M. Impact: low–medium, for the final cm.
9. **Splats: feed corrected poses to Brush, and try msplat on the same nerfstudio export.** Effort S. Impact: visual
   only. Defer depth-supervised splats (DN-Splatter) until a CUDA box is available.
10. **Don't switch to another capture app or to StructureBuilder for geometry.** None refines poses or measures
    thickness. StructureBuilder output is still useful as a sanity check for wall dedupe.

**Realistic accuracy** [Inf, from the sources above]:

- Single room: 0.5–1 cm (already met).
- Whole flat after the pose graph plus closing lap: p95 of about 1.5–3 cm on wall-only points.
- Thickness: ±1 cm where a doorway sweep exists, the default elsewhere.
- Below 2 cm p95 across the flat is plausible but not guaranteed. Commercial consumer-LiDAR tools claim 1–2 % or
  1–5 cm, and the iPhone-vs-laser building study measured a 4 cm mean.
