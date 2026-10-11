# Threshold audit: which rules are tuned to the second scan

Date: 2026-10-07. Code at commit `b7c87b4` (line numbers refer to it; `align.py` and `poses.py` are being
changed in parallel). Every numeric threshold or rule in `pipeline/scanmodel/` that changes the output, ranked by
the risk that it breaks on another flat.

**Why.** In `scanmodel.align` a loop closure that disagrees with ARKit by more than 3 cm vertically is dropped
(`GATE_Z`), a value picked to stop false stair-step matches. The same living-room floor sits at +16, +7
and +6 cm in different sessions (about 10 cm of vertical drift), so the gate threw away exactly the
closures that would have corrected it. `scanmodel.rooms` then put the conservatory floor 16 cm *above* the
living room; it is about 10 cm *below* (`scanmodel bench` check: measured +12.7 cm, expected −10 ± 4 cm, FAIL).
Many values were chosen while looking at this one capture; the synthetic tests only show that the code
does what was intended.

Justification codes: **P** physical bound or sensor spec, **L** literature or vendor documentation,
**S2** picked on the second scan (or the first), **D** design choice without measurement.
Risk: **H** likely wrong on common flats, **M** wrong on some layouts, **L** low.

## Status (2026-10-07, after working through the top of the list and the third scan)

Each change was accepted only if `scanmodel bench` (ARKitScenes) didn't get worse and the second scan's
check passed. Line numbers in the table below are from before these changes.

| # | Status | What changed / why not |
|---|---|---|
| 1 | Done (`5b998ac`) | Floor-height constraints fix vertical drift in pass 1; closures are gated against that, gates growing with time (rates from the bench's worst room). Second scan: conservatory −9.0 cm vs living (was +12.7). Weak spot: leave-one-pair-out 13 cm |
| 2 | Done (`526b8f6`) | Floor searched within 0.8 m of the room's own lowest upward surface |
| 3 | Tried, rejected | Largest-area level: floor errors up to 3.4 cm on the bench (vs 0.9), drift leaves fused floors in thin layers. Lowest strong cluster kept; a large shower tray can still win |
| 4 | Done (`526b8f6`) | Objects snap to their room's `floor_z` |
| 5 | Done (`fcc5de5`) | Side surfaces must cover half the side; a hidden side follows the top only where it visibly ends. Kitchen sides made worse 23 % → 9 % |
| 6 | Done (`30aae66`) | Sliver test: shorter side < 0.5 m |
| 7 | Done (`30aae66`, `c4b4643`) | 0.9 m, flagged over 0.5 m. Follow-up from the third scan: the far-face search ignores RoomPlan's furniture footprints, else cabinet fronts become the far face (a kitchen wall at 78 cm) |
| 10 | Done (`30aae66`) | Merge only on footprint overlap > 50 % and across sessions |
| 8, 9, 11–20 | Open | Next: #8 (merge tolerance from drift), #9 (attic ceilings), #12 (per-axis odometry noise), #14 (snap only to the room's own walls), #19 (thresholds to the room's floor) |
| 21–28 | Keep | Low risk |

Third scan (2026-10-07), what it adds:
- The conservatory floor step was measured by hand: about −6 cm, not the eyeballed −10. The check in
  `data/bench/second.json` and `third.json` is now −6 ± 4 cm. Second scan with the pose graph −9.0 cm,
  third −8.4 cm: both pass. (Three sessions of the third scan each see about −9 cm within their own
  recording, so the hand value may be the less certain one.)
- #8 is real here: one wall's faces from two sessions are 5.7 cm apart, another's 13.6 cm, and stay two
  layers (default thickness). Leave-one-pair-out is 2.1 cm, so the merge tolerance could come from
  the pose graph's per-face residual rather than a fixed 3 cm.
- Not in the table, found on this scan and fixed: RoomPlan walls seen through glass and doorways made
  the conservatory's outline include the garden (`9de1eae`: a session's rooms are the enclosed areas
  holding ≥ 15 % of its cameras, `room_camera_share`, else 1.5 m around the camera path,
  `room_reach_m`; both **D**, risk **L-M** for a room recorded from its doorway only); RoomPlan doors
  without a leaf in the scan become openings (`672bbcd`, middle half of the opening in 10 cm cells, **D**).

New rules from the splat work (2026-10-08), same codes. Not in the ranked table: they only change the
stills' intrinsics and the splat, not the structure.

| Where | Value | What it does | Why | Risk on other flats |
|---|---|---|---|---|
| `stillcal.py` | up to 8 earliest stills per session, ≥ 3 must fit; SIFT 4000, Lowe 0.75, RANSAC homography 3 px, ≥ 40 inliers | Median focal scale of the 12 MP stills against their nearest video frames | L (Lowe, RANSAC) + D; measured ×1.047–1.048 in all five sessions of scan 3, 0.5–0.8 px residual | **L**: a session whose first stills face a blank wall falls back to ×1.0 (no correction, printed). Assumes still and frame share a pose (they do: same ARFrame time) |
| `freespace.py` margin | 4 cm + 1.5 % of depth + the splat's largest axis (capped at 10 cm) | A splat centre this far in front of the LiDAR surface is "free" | P (iPad LiDAR noise ~1–2 cm at ≤ 3 m) + D | **M**: drift between sessions above the margin (scan 3: faces 5.7 and 13.6 cm apart) makes one session's surface "free" for another's depth; the free > seen rule limits it |
| `freespace.py` vote | ≥ 3 free votes and more free than seen | Removes the splat | D | **L-M**: thin objects missed by the 256×192 sensor (cables, plant stems) could go; the 5×5 window minimum counts near-surface as surface |
| `depthviews.py` | confidence 2 only, ≤ 4.5 m, 5×5 window minimum, every 6th depth frame (1,162 on scan 3) | Which LiDAR evidence votes | P (confidence, range) + D | **L**: glass is seen through by the LiDAR, so splats on clear glass go (correct for a window; mirrors untested) |
| `msplat_train.py` schedule | every 1500 steps up to 15k (msplat densifies until 15k) | When to prune | D; 7k A/B: +0.05 dB, 30k: with the still fix +1.26 dB | **L** |

## Ranked table

| # | Where | Value | What it does | Why | Risk on other flats | Replace with |
|---|---|---|---|---|---|---|
| 1 | `align.py:40` `GATE_Z, GATE_XY` | 3 cm, 15 cm | Drops a loop closure if it moves keyframe j away from ARKit by more than this | S2 (stair steps, identical cabinet doors gave consistent false closures of ~20 cm) | **H**. Real drift grows with time and distance: 10 cm vertical over minutes on the second scan, so the gate rejects the true closures and keeps the drift. Long captures, closing laps and multi-storey walks all exceed 3 cm | A gate that grows with the time (or path length) between the two keyframes, e.g. `max(3 cm, k·Δt)` with k from measured ARKit drift (bench `drift` column); plus geometric consistency between closures (pairwise consistency / max clique) instead of agreement with ARKit; floors as height constraints (in progress) |
| 2 | `rooms.py:29` `floor_band` | −0.6 … +0.8 m | A room's floor points must lie in this band of the scene frame | D (scene frame floor = 0) | **H** for split levels, maisonettes, sunken rooms > 60 cm, half-landings: no floor points pass, `floor_z` silently stays 0 | Band relative to the room's own cameras (hand-held: floor ≈ 0.6–1.9 m below the median camera height), not to the flat's floor |
| 3 | `points.py:139-149` `floor_height(min_share=0.2)` (used by `poses.py:221` and `rooms.py:88`) | lowest 1 cm bin with ≥ 20 % of the peak | Floor = the lowest strong cluster of upward-facing points | D | **H**. Lowest-first picks a sunken shower tray, a step down that covers a quarter of the room, a strip of the lower neighbouring room inside a drifted outline, or LiDAR reflections below a glossy floor. In `poses` it also sets the flat's z = 0 from the *whole* flat | Largest-area horizontal cluster (area from a 2D occupancy grid, not point counts, which favour areas scanned longer); report a second level when both clusters exceed ~1 m² |
| 4 | `boxes.py:121` | z0 < 8 cm → 0 | An object whose bottom is within 8 cm of z = 0 is put on z = 0 | D | **H** wherever the room's floor isn't at 0: on the second scan the conservatory furniture is either sunk 16 cm into a raised floor or (truth) floats 10 cm above a lower one | Snap to the room's fitted `floor_z` (or the dominant upward surface under the box) |
| 5 | `boxes.py:138-141` hidden side follows the worktop | 98th percentile of top points within `max_shift` | A side with no points of its own moves to where the top surface ends | S2 (second-scan worktop running past RoomPlan's box) | **H** for kitchens annotated per unit: under one continuous worktop every unit's hidden side grows by the full 15 cm into its neighbour. Bench, kitchen 43828439, fit started *on the true boxes*: 9 of 15 units move a side by 12-15 cm | Only extend when the top surface ends within the window (an edge in the top points), not when it runs on; never past a neighbouring box |
| 6 | `structure.py:92` `min_room_m2` | 1.0 m² | Enclosed areas smaller than this are dropped as slivers | S2 (first scan's room split) | **H** for WCs (0.8–1.2 m²), store cupboards, airing cupboards, which disappear | A sliver test on shape: short side of the minimum rectangle < 0.5 m |
| 7 | `structure.py:82` `max_thickness_m` (also limits the search behind a face in `measure_thickness`, `structure.py:336,355`) | 0.50 m | Opposite faces further apart are not one wall; thickness isn't searched beyond | P (interior walls) | **M-H** for old buildings: external and party walls of 50–70 cm get no partner, the 10 cm default is added and the rooms either side overlap or leave a gap | 0.9 m (solid masonry), and pair only faces whose rooms are on opposite sides; flag > 0.5 m for review |
| 8 | `structure.py:81` `merge_offset_m` | 3 cm | Same-side faces from two sessions merge only within 3 cm | D | **M-H**: with 5–15 cm drift between sessions the same face stays twice (parallel duplicate walls, ghost layers) | Merge on overlap and same room adjacency with a tolerance from the measured drift (pose-graph residual), not a fixed 3 cm |
| 9 | `rooms.py:31-33` `min_room_height`, `slope_max_deg`, single ceiling plane | 2.0 m, 45° | Ceilings lower than 2 m at the centre are rejected (meant: a stair's underside); steeper than 45° counts as flat (median height) | S2 (stair underside, conservatory 4.6°) | **M**: attic rooms (pitches of 45–55°, gables = two planes, collar ceilings), cellars < 2 m; a 50° roof gets a wrong flat height | Reject by *where* (stair footprints, which are already excluded) not by height; fit up to two planes (gable) and allow any pitch < 70° |
| 10 | `structure.py:707` object de-duplication | same category, centres < 0.3 m | Two RoomPlan objects of one category this close are one (bigger wins) | S2 (same object from two sessions) | **M**: a run of narrow identical units (30 cm base units, bar stools 35–40 cm apart, twin bedside tables) loses members | Merge only if the boxes overlap (IoU > 0.5) and come from different sessions |
| 11 | `boxes.py:83-86` `max_shift`, `wall_gap`, `wall_overlap` | 15 cm, 8 cm, 10 cm | How far a box side may move to a surface; when a side is pulled onto a wall | S2 (kitchen counter: 13 cm gap) | **M**: small objects (< 30 cm) can jump to a neighbour's surface; a sofa standing 8 cm off the wall is pulled onto it | `max_shift` from RoomPlan's measured box error, scaled by object size (bench `boxes` column shows the error before/after) |
| 12 | `align.py:331,335` odometry and closure noise | rot 0.05° + 0.02°/s, trans 2 mm + 1 % of distance + 1 mm/s; closures Cauchy, 0.5°, 1.5 cm | Decides how far closures may bend ARKit's trajectory | L-ish ("ARKit ~1 % of distance") + D | **M**: same sigma horizontally and vertically, while the second scan drifted mostly vertically; with the gate removed these numbers decide everything | Fit the per-axis drift rate on the bench (laser-registered windows: `drift` columns), separately for z |
| 13 | `fuse.py:33-34` (and `cli.py:83`) `min_confidence`, `max_depth` | ≥ 1 (medium), 4 m | Depth pixels used in the TSDF | D | **M**: medium-confidence pixels on glass and dark surfaces add noise (conservatory); 4 m drops the far side of open-plan rooms > 8 m unless walked | High confidence everywhere plus medium within 3 m (research doc §1); `max_depth` 5 m with confidence 2 |
| 14 | `rooms.py:28` `snap_m` (+ area guard `rooms.py:154` 30 %) | 0.40 m | An outline edge moves onto a parallel wall face up to 40 cm away | D | **M**: flattens bays and niches < 40 cm; in a narrow room a face of the next room can win | Snap only to faces of walls that list this room (`Wall.rooms`), within the drift tolerance |
| 15 | `structure.py:80,93` `corner_m`, `close_gap_m` | 0.30 m, 1.0 m | Face ends this close are joined; a face is extended up to 1 m to close a room | S2 (first scan's room split) | **M**: nibs and piers < 30 cm vanish; open-plan openings > 1 m never close (no split), 0.8–1 m doorways close | Join by the gap relative to wall length; close openings with RoomPlan's door/opening objects instead of a distance |
| 16 | `structure.py:84` `default_thickness_m` | 0.10 m | Thickness of a wall seen from one side only | L (Polycam 10 cm) | **M**: old brick interior walls are 12–25 cm, so rooms either side overlap or leave gaps | Keep as default but per flat (overrides) and per class (external vs interior); measure where a doorway sweep exists |
| 17 | `points.py:14` `WINDOW_S`, `structure.py:88-90` `strip_m`, `face_search_m`, `thickness_agree_m` | 30 s, 0.5 m, 0.15 m, 2 cm | Thickness only from one 30 s window, per 0.5 m strip; windows must agree within 2 cm | L (research: ≤ 30 s window) + S2 | **M-L**: strict, so mostly falls back to the default; never wrong, often absent | Window length from measured drift per second (bench), agreement from the strip spread |
| 18 | `rooms.py:30` `ceiling_above_floor` | 1.6 m | Downward points must be this far above the floor to be ceiling | D | **M-L**: sloped ceilings reaching below 1.6 m (attic eaves) lose their low part, which biases the plane | Use the room's camera height: ceiling points above the median camera height |
| 19 | `structure.py:445`, `structure.py:495` | `abs(base_z) < 5 cm → 0`, door bottom < 5 cm → 0 | Small wall-base / threshold offsets are rounded to the floor | D | **M-L**: hides 2–5 cm steps between rooms (and drift of the same size) | Snap to the room's `floor_z` instead of 0 |
| 20 | `structure.py:74-79` `band_m`, `min_inliers`, `snap_deg`, `snap_tol_m` | 8 cm, 200, 2°, 1 cm | Points within 8 cm of RoomPlan's face fit its plane; snap to 90° if the ends move ≤ 1 cm | D; `snap_tol_m` from the 2 cm eval budget | **M-L**: a RoomPlan face > 8 cm off (glass, curtains) isn't refined; non-orthogonal flats are protected by the 1 cm rule | `band_m` from RoomPlan's measured face error |
| 21 | `structure.py:85` `max_rotation_deg` | 10° | A fit turned more than this is another surface | S2 (86.5° case) | **L-M**: angled bay walls (45°) are protected since RoomPlan gives their angle | Keep |
| 22 | `structure.py:798-799,819`, `structure.py:833-845` stairs | same flight if Δz < 0.4 m, footprint overlap > 50 %; clip 30 % into walls; treads ≥ 10 cm apart, peak ≥ max(20, 15 % of max) | Stair merge and tread fit | S2 (5 stairs, one seen twice) | **L-M**: spiral / winder stairs and stacked flights in a small footprint | Merge by overlap in 3D (bounding volumes), treads by riser regularity |
| 23 | `align.py:38,143,178,198-228` closure search | MIN_DT 15 s; 25 cm voxels, overlap > 25 %; Lowe 0.75; MAGSAC 1.5 px; 3 cm Kabsch inliers, ≥ 20, rms ≤ 2 cm | Candidate pairs and verification | L (Lowe, MAGSAC) + D | **L-M**: low-texture rooms give no closures, so the graph silently does nothing | Report closures per session pair (already logged); fall back to geometric (floor/wall) links |
| 24 | `rooms.py:74`, `fit_plane` tol 2 cm, inlier share 0.3 | 20 cm inset, 2 cm, 30 % | Floor/ceiling sampled 20 cm inside the outline; ceiling plane needs 30 % of the ceiling points | D | **L-M**: beams/coffers/pendants; corridors < 40 cm wide get nothing | Keep; check on the bench (ceiling column) |
| 25 | `boxes.py:84,97,119,123,134` | 150 points, ±1.5 cm peak, share > 0.3, 2 cm top band | Box side and top fitting details | D | **L** | Keep |
| 26 | `align.py:312` gravity prior | 0.05° | Roll/pitch kept from ARKit | L (ARKit gravity is well observed) | **L**: 0.05° over 10 m = 9 mm | Keep |
| 27 | `poses.py:147` anchors | moved > 5 mm | Anchor priors only if ARKit moved them | D | **L** (never seen moving yet) | Keep |
| 28 | Report-only: `poses.py:212` ICP 3 cm warn, `structure.py:435,1072` flags, `ingest.py:74,91,163`, `checks.py:18,57-63` | various | Warnings and flags | D | none on output | Keep |

### Eval (changes the reported quality, not the model)

`eval.py:137-140`: only points within 10 cm of a face and on the dominant surface within 3 cm of it count.
A face placed more than ~10 cm off the real wall has *no* points and is skipped, so a worse model can score
better. Report the share of faces with no points, and add the bench's laser-based numbers next to it.

## Notes

- **The common failure is "relative to 0".** Rules 2, 3, 4, 18 and 19 measure heights against the flat's
  z = 0 or snap to it. That is right in a single-level flat whose rooms share one floor, and wrong in every
  flat with a step, which is exactly the case the conservatory exposed. Measuring per room (its own floor,
  its own camera height) removes the assumption at no cost.
- **Thresholds that encode drift** (1, 8, 12, 17) are really one unknown: how far ARKit drifts per second
  and per metre, horizontally and vertically. Measure it once (the bench's `drift` columns: 20 s windows
  ICP'd onto a laser scan) and derive all four from it, instead of four hand-picked numbers.
- **Data-driven beats fixed when the scale varies**: room sizes (6, 15), object sizes (5, 10, 11), wall build
  (7, 16). Prefer shape tests and per-object scaling over absolute metres.
- **Values that are fine**: snap tolerance from the eval budget (20), gravity prior (26), the SIFT/MAGSAC
  settings (23) and the box fitting internals (25) follow standard practice or physical bounds.

## Benchmark (`scanmodel bench`)

First run, 2026-10-07 (code `b7c87b4` plus the working tree, including another agent's in-progress
`align.py`). Four ARKitScenes rooms (`scanmodel.stages.arkitscenes.SUBSET`), chosen from 356 3dod videos with laser
scans by screening their ARKit meshes for floor levels, ceiling heights and length; none has stairs (ARKitScenes
has one room per video). Each room once with ARKit's poses (`arkit`) and once with the pose graph (`pg`).

| Room | Poses | Reg. | Acc. median / p95 | Compl. 5 cm | Window corr. xy / z | Spread xy / z | Floor (lowest / densest) | Ceiling | Box sides before → after | Tops before → after |
|---|---|---|---|---|---|---|---|---|---|---|
| 42447329 ordinary room, 234 s | arkit | 82 % | 1.2 / 9.1 | 94 % | 2.6 / 2.3 | 4.4 / 4.1 | −2.7 / +1.1 | +0.2 | 7.6 → 4.1 (1 solid) | 7.8 → 4.4 |
| | pg | 82 % | 1.2 / 9.1 | 94 % | 2.7 / 2.2 | 4.7 / 3.7 | −1.5 / +1.4 | −0.1 | 7.6 → 4.1 | 7.8 → 3.6 |
| 43828439 kitchen, 123 s | arkit | 57 % ? | 2.3 / 12.1 | 84 % | 14.4 / 3.8 | 25.2 / 5.3 | +0.8 / +2.2 | −1.1 | 7.6 → 7.3 (15 solid, 22 % worse) | 7.7 → 3.3 |
| | pg | 56 % ? | 2.3 / 11.4 | 84 % | 12.5 / 4.0 | 22.9 / 5.5 | +0.9 / +2.3 | −1.0 | 7.6 → 7.5 | 7.7 → 3.2 |
| 47204322 dining room, 130 s | arkit | 50 % ? | 2.4 / 9.6 | 80 % | 6.0 / 4.7 | 7.4 / 5.7 | −2.9 / +0.1 | +0.2 | 7.7 → 3.5 (3 solid) | 7.9 → 6.6 |
| | pg | 51 % ? | 2.2 / 9.7 | 81 % | 7.5 / 4.7 | 6.3 / 4.9 | −3.5 / −0.6 | −0.5 | 7.7 → 3.6 | 7.9 → 7.2 |
| 42899822 bedroom, 3.3 m ceiling, 99 s | arkit / pg (0 closures) | 47 % ? | 2.2 / 13.9 | 73 % | 6.6 / 5.7 | 8.0 / 7.5 | +0.6 / +2.1 | −0.4 | 7.5 → 7.8 (7 solid, 24 % worse) | 7.8 → 6.7 |

All values in cm unless %. Reg. = share of our fused points within 2 cm of the aligned laser; "?" marks rooms
below 60 %, where the laser and our cloud overlap less (glass, a lower area seen through an opening, scans
that reach other rooms), so their window corrections mix drift with registration residual.

What it says about the rules:
- **Vertical drift above 3 cm is normal, even in one room.** The 20 s windows of a single 1.5–4 min recording
  disagree vertically by 4.1–7.5 cm (spread z), in all four rooms. A fixed 3 cm `GATE_Z` (#1) sits inside
  ordinary ARKit drift; the window corrections (`per_window` in the JSON) are the data for a time-scaled gate.
- **Floors**: within 3 cm of the laser in all rooms. The lowest-strong-cluster rule (#3) reads 2.7–3.5 cm low
  in the two rooms whose fused floor has several drift layers; the densest layer is within 2.3 cm everywhere
  but not uniformly better. Ceilings are within 1.1 cm; no sloped ceiling was called by either side (none
  in this subset: add an attic room to test #9).
- **Boxes**: the fit fixes tops (7.8 → 3.2–6.7 cm) and solid sides where an object stands alone (7.6 →
  3.5–4.1 cm), but not in a kitchen run or a bedroom with built-ins (7.5 → 7.3–7.8 cm, about a quarter of sides
  worse). Started *on* the true kitchen boxes, the fit moves 9 of 15 units by 12–15 cm: the worktop rule (#5)
  and the 15 cm search window (#11).
- **Pose graph**: small effect on single-room captures (corrections ≤ 3 cm; no closures at all in the
  bedroom). These sequences are too short to show what it does across a flat: our own scans remain the test
  for that.
- **Our scan** (`data/bench/second.json`): conservatory floor vs living measured **+12.7 cm**, expected
  −10 ± 4 cm (eyeballed): **FAIL**, both on `scene/model.json` and on a fresh `--rerun` (pose graph on, no
  overrides) of the working tree.

Limits: half of each laser scan is read (every other 1 MB block, ~2 scan rows): ≥ 99.4 % of the full scan's
1 cm voxels keep a sampled point within 2 cm (88–100 % within 1 cm). The laser is aligned to our cloud by one
rigid transform, so a global error common to all windows is absorbed; window corrections show only the
differences. RoomPlan isn't part of ARKitScenes, so walls, rooms and thickness are only covered by our own
checks.

## ARKitScenes licence

ARKitScenes ([github.com/apple/ARKitScenes](https://github.com/apple/ARKitScenes)) points to its `LICENSE`
for using the dataset: Apple's licence grants a personal, non-exclusive licence for non-commercial use, plus
a commercial licence for licensees with fewer than 700 M monthly active users (before August 2024); Apple's
names and marks may not be used for endorsement; provided "as is". Questions:
ARKitScenes-license@group.apple.com. (The arXiv paper itself is CC BY 4.0.) Our use: local evaluation
only; the data stays in `data/external/` (gitignored) and is never redistributed or used as model input.
