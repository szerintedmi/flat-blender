# FlatBlender (capture app, phase 1 MVP)

iPadOS app that records one room per session (plus a closing lap) as a **capture bundle v1**: the contract defined by
`pipeline/scanmodel/models/bundle.py` (JSON Schema: `pipeline/scanmodel/schemas/bundle.v1.json`). Plan:
`docs/plans/2026-10-06-scan-to-model.md` ("Capture app: FlatBlender" with the versions and why, "Bundle format v1",
"Capture protocol"); the next device test: `docs/plans/START-HERE.md`.

Target: iPad Pro 11" (M2, LiDAR), iPadOS 26+, any orientation and window size, Swift 6 (strict concurrency), SwiftUI, no
third-party code.

```
capture/
  FlatBlender.xcodeproj/      hand-written project (objectVersion 77, synchronized folder, shared scheme)
  FlatBlender-Info.plist      the one Info.plist key Xcode can't generate (UIFileSharingEnabled); rest via INFOPLIST_KEY_*
  FlatBlender/                app sources (synchronized folder: new files need no project edits)
    Capture/               ARKit/RoomPlan engine, recorders, camera lock, mesh/world-map export
    Storage/               projects + sessions as plain folders, settings, device info
    Export/                zip a session for AirDrop / Files
    Views/                 SwiftUI screens
  BundleKit/               Swift package: bundle models + encoders (no ARKit; builds and tests on macOS)
    Sources/BundleKit/       Codable models, Mat4, JSONL writer, depth/confidence PNG, PLY, exposure/flicker
                             policy, motion blur, StageTimer
    Sources/bundlekit-sample/  writes a synthetic bundle for the pipeline contract test
    Tests/BundleKitTests/    `swift test`
```

## Build and run

1. Open `capture/FlatBlender.xcodeproj` in Xcode 27.
2. Create `capture/Local.xcconfig` (gitignored) with your team and, if `dev.flatblender.FlatBlender` is
   taken, a bundle-id suffix:
   ```
   DEVELOPMENT_TEAM = ABCDE12345
   BUNDLE_ID_SUFFIX = .yourname
   ```
   Don't pick the team in *Signing & Capabilities*: that writes it into `project.pbxproj`, which is shared
   (see `Signing.xcconfig`).
3. Connect the iPad (Developer Mode on), choose it as the run destination, Run. Accept the camera prompt.
   The shared scheme's **Run uses the Release configuration** (since 0.3.0 (7)). Scan 4 was a Debug
   (`-Onone`) build: the per-pixel Swift code (then the Laplacian sharpness, and the coverage map) ran ~300×
   slower, so stills took 2–6 s to save and the capture lagged. A Debug build shows an orange banner on the
   capture screen. In Release our own per-frame work is small (0.4.0 writes per-stage timings into the session
   notes to keep it that way, see "Timing").

Command-line build check (no signing):

```sh
xcodebuild -project capture/FlatBlender.xcodeproj -scheme FlatBlender \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

The app needs real hardware: ARKit world tracking with `sceneDepth` + mesh reconstruction (LiDAR). The
simulator won't capture anything.

## Screens

- **Projects** (home). One project = one flat. `+` creates `Documents/Projects/<project_id>/project.json`.
  Gear opens **Settings**. Swipe to delete a project (and all its sessions).
- **Project**. Lists sessions (room, type, start time, duration, stills, map icon if it saved a world map,
  capture stats and warnings from `session.json` notes). Swipe or long-press a session → **Export** (zip +
  share sheet: AirDrop, Save to Files) or **Delete**. **Export all** zips the whole project folder (unzip
  into the pipeline's `data/<project>/raw/`). `+` starts a new session.
- **New session**. Room name (e.g. `hall`), type **Scan** or **Texture** (Ultra-wide is listed but is a
  phase-5 TODO), and **Continue world map** from the latest session that saved one (on by default when there
  is one). Creates `<project>/<yyyyMMdd-HHmmss-room>/`.
- **Capture** (full screen):
  - Live camera with the ARKit mesh overlay (toggle in Settings).
  - HUD: session, tracking state, world-mapping status, elapsed time (orange after 10 min) and recording
    time, recorded frames (+ dropped), depth frames and measured depth rate (red if the frames carry no
    `sceneDepth`), stills (+ last still latency and whether the high-res frame had depth), anchors added,
    camera lock values, RoomPlan element counts, thermal state, free disk (queried off the main thread every
    30 s).
  - Banners: relocalisation status (when a world map was loaded) until tracking is normal again, RoomPlan
    instructions, heat warnings, errors.
  - Controls: **Lock exposure** (one control since 0.5.0; it was Meter, then Lock): the tap switches to
    continuous auto exposure / WB / focus and shows a yellow "Point at a mid-bright wall and hold still" panel
    with a progress bar; it locks by itself once auto exposure has settled (`ExposureSettle` in BundleKit:
    duration, ISO and WB gains steady for 0.6 s with nothing adjusting, or 1.2 s whatever AVFoundation's
    `isAdjusting…` flags say, never before 0.8 s; after 5 s it locks anyway and says so). Tapping "Metering…" locks
    at once. The lock (exposure custom, re-balanced
    by `ExposurePolicy` in BundleKit: shutter 1/240 s with ISO raised to keep the brightness, ISO at most
    800; if that is too dark the shutter goes back up to 1/120 s, then the image is left darker rather than
    blurrier or noisier (orange banner from 1 stop, HUD shows `-x EV`). WB gains locked, focus lens position
    locked; values go to `session.json` `camera`, with the `metered` exposure, `underexposure_ev` and
    `shutter_mode`; Settings › Camera can fix the shutter at 1/100 s or 1/120 s instead) → **Record** (auto
    stills) → **Next room** (scan sessions) → **Finish** (scan sessions: offers the closing lap). Texture
    sessions have a shutter button instead of Record. **Discard** (before recording) deletes the folder. The
    exposure control works only before recording. Metering and locking play a tone. Locked shows green
    **Locked** with shutter and ISO (tap to meter again). When ARKit resets the lock (a configuration re-run) the
    app puts the same values back at once (counted in the notes as "camera lock reset by ARKit N×"); only if that
    fails does the orange **Relock** appear, and its tap re-applies the same values (0.4.0 re-locked from whatever
    auto exposure showed then).
  - **Flicker prompt at Lock** (0.4.0): if the metered shutter is within ~1 % of k/100 s (or k/120 s), auto
    exposure has most likely settled on whole flicker periods of LED/fluorescent light on 50 Hz (60 Hz) mains
    (`ExposurePolicy.suspectedFlicker`; scan 4 metered exactly 0.010002 s at ISO 258). In the Sharp mode a
    non-modal orange panel then asks "LED flicker likely: use Flicker-safe 50 Hz?". **Use 50 Hz** switches the
    shutter mode (this session, the next rooms, and Settings) and re-locks from the same metered exposure;
    **Keep 1/240 s** keeps the lock. Never switched automatically. `session.json` `camera`:
    `flicker_suspected` (`50hz`/`60hz`, any mode) and `flicker_prompt` (`shown` = never answered, `accepted`,
    `declined`).
  - Position gate: Record and the shutter stay disabled until the session has its bearings (relocalised
    against the loaded world map, and tracking normal). An orange "Finding position… m:ss" panel shows
    while ARKit relocalises (with help after 30 s), a chime and a green "Position found" when it does, and
    a red panel if tracking is lost later. A session that never relocalised says so in its notes, and
    `scanmodel ingest` rejects it.
  - Texture sessions don't record video: they load the world map, wait for the position, then enable the
    shutter. **Finish** ends them.
  - **Continue world map** uses the latest *scan* or *lap* session's map (after Next room and the lap it
    covers every room of that walk), not a texture session's.
  - Auto stills (scan and lap sessions, while recording, paused during the doorway sweep): `StillTrigger` shoots
    when the camera has been nearly still for ~0.25 s (< 0.15 m/s, < 0.3 rad/s) and the view is > 0.3 m or
    > 15° from every still so far, and (`MotionMeter`) the camera is steady at that instant (pose-predicted
    motion blur < 1 px at 960 px). Each still records that `motion` (`angular_speed_deg_s`, `speed_m_s`,
    `blur_px`). **Scan and lap sessions have no shutter button** (0.4.0): on scan 4, taps queued behind
    stills, came out blurry and took the slots of auto stills; the pipeline picks images from the video and
    the auto stills anyway. Until 0.3.0 (7) the gate also needed the stream's Laplacian sharpness (every 6th
    frame, 1.5 s median), taps were flagged blurry (`flagged_blurry`) and a blurry still was "forgotten" so an
    auto still could replace it; all of that is gone in 0.4.0 (the pipeline measures sharpness itself).
  - Tapped stills (texture sessions only, `FrameSink.TapRequest`): a tap shoots at the first frame within
    0.5 s where the camera is steady enough (predicted blur < 1.5 px at 960 px: this waits out the jolt of
    the tap) and no other still is in flight. Otherwise it is **not taken** (falling tone, "Not taken: hold
    still, then tap" or "the last still is still saving"): never seconds later while the camera moves.
  - Cues (nothing modal): a ring, green when steady (< 1 px predicted blur at 960 px), amber when moving,
    flashing on each still: round the shutter in texture sessions, round the auto-still count in scans and
    laps; "Turn a little slower" while the predicted blur stays > 2 px at 960 px (too fast for sharp video).
  - Sounds (`CaptureSounds`; Settings › Feedback › Capture sounds, default on): short synthesized tones through
    our own `AVAudioEngine` with the audio session `.playback` + `.mixWithOthers`, so they play at media volume,
    also in Silent Mode, without stopping the user's music. Auto still: a faint, high tick; too fast for sharp
    video: two low notes (on each onset, at most every 8 s, not during the doorway sweep); metering, lock,
    Position found; texture sessions: a click per tapped still, a falling pair when a tap was not taken. Nothing records audio (`video.mov` is video-only, ARKit `providesAudioData` is off), so the tones
    never end up in a recording. Until 0.3.0 (6) these were `AudioServicesPlaySystemSound` system sounds:
    silent in Silent Mode, and the still click (1104, the keyboard click) also needed Settings › Sounds ›
    Keyboard Clicks; auto stills had none. iPads have no Taptic Engine: no haptics.
  - **Coverage** (Settings › Preview, while recording; `CoverageMap`, `CoverageOverlay`, `CoverageMiniMap`):
    magenta dots on surfaces (one per 8 cm voxel) seen from too few directions so far, counting only the
    directions you could have had from where you have walked (bigger = more missing, up to 3). Step sideways
    and look again: a dot that is covered turns green for 1.5 s, then goes. A key under the mini-map says so
    ("Look from another angle" / "Covered"). The mini-map (top right, heading-up, 8 m) shows the walls around
    you: grey covered, magenta wanting 2+ more directions, your path in cyan. Since 0.3.0 (7) voxels that later
    depth sees *through* are dropped (free space) and nothing is integrated until tracking has been normal for
    1 s without a pose jump: on scan 4 about half the dots in view floated 10–80 cm in front of the real
    surfaces (depth integrated with the pose ARKit reported just before its relocalisation snap, long-range
    LiDAR bias) and could never clear.
  - **Next room** (0.5.0): the tap stops this room at once: stream, auto stills, anchor ticks and its
    RoomPlan session (`stop(pauseARSession: false)`). On scan 5 (0.4.0) the old room kept recording, RoomPlan
    included, while the name alert was up: each room ended with 25–40 s looking into the next one, and
    garden's room.json had the living room's TV and sofas. The room is then saved in the background while ARKit
    keeps running (same world frame), and a non-modal panel shows the steps ("Saving living… world map") with a
    name field for the next room (type while it saves). Once the steps that read the ARSession (world map, mesh)
    are done, the next room is prepared: a new RoomCaptureSession on the same ARSession (RoomPlan's export of
    the old room's data runs meanwhile), our configuration again after its didStart, the camera lock re-applied
    and checked (`relockAfterConfiguration`: every 100 ms until it held 0.3 s, ≥ 0.6 s after the run, ≤ 2 s;
    0.4.0 slept a fixed 500 ms and re-applied once). The next session's folder and session.json are made when
    the old one is fully saved and the name is in. **Recording starts only with Record** (disabled, "Saving
    previous room…", until then); a blue panel shows the doorway-sweep steps and the exposure carried over,
    with "very different light (outdoors)? tap Locked to meter this room". Expected gap between two
    recordings: ~4–5 s of saving and setup (RoomPlan's build of room.json, 2.2–3.5 s on scan 5, dominates),
    overlapping the typing, plus the Record tap; 0.4.0's was 3.0–4.25 s, but with the old room still recording.
    A failed step (world map, mesh.ply, room.json/usdz) shows what failed with **Retry** (walk around a little
    first for the world map), **Continue anyway** (it stays a warning in that session's notes) or Finish.
    "Not another room…" offers the closing lap or Finish instead. The new `session.json` has `world_map_in`
    and `continued_from` = the previous session.
  - **Doorway sweep** (start of every Next room's recording): guided by the pose since 0.5.0 (`SweepProgress`
    in BundleKit). Scan 5's sweeps were never really done under 0.4.0's 10 s countdown (0.19–0.45 m of movement,
    7–119° of turn). Steps: 1. look at the wall around the door, in the room you're leaving; 2. step through
    the door (0.5 m, ticked when done); 3. turn round and look at the same wall from this side (150°, ticked). It
    ends by itself 2 s after both, at 20 s anyway, or with **Done** (after 5 s). Recording continues, auto
    stills pause. Its start and end frame timestamps go to `session.json` `sweeps` (`t0`, `t1`), so the
    pipeline can measure that wall's thickness from seconds of recording, with almost no drift.
  - **Closing lap**: Finish in a scan session asks "Closing lap (recommended)". **Start lap** closes the
    room like Next room (saved in the background) and opens a session of type `lap` on the same ARSession
    (`world_map_in` = the last room, no `room_name`, folder `<time>-lap`): stream, depth, auto stills, anchors,
    world map; no RoomPlan. It records when you tap **Record** (0.4.0 started it by itself). Walk back
    through every room to the start, then **Finish** (no second prompt). The end of the recording then sees
    what its start saw: loop closures for the pose graph. **Finish without lap** just finishes.
  - **Anchors**: while recording (texture sessions: once the position is found) an `ARAnchor` at every
    still (`<session_id>/still/<id>`, the still's pose) and every 1 s (`<session_id>/tick/<n>`, the
    camera's pose). `anchors.jsonl` logs them: `add`; `update` when ARKit moves one (any session's: they
    persist across Next room) by > 1 mm or > 0.05° since its last line; `final` for all of ours in the
    world map saved at the end of the session. Tests whether ARKit's loop closures move them.
  - One hint line at the bottom for the current step.
  - Finish (and Next room, Start lap) writes, in order: `video.mov` finalised, stills flushed,
    `worldmap.arworldmap`, `mesh.ply` (the mesh at the tap), `room.json` + `room.usdz` (scan sessions with
    RoomPlan), the `final` lines of `anchors.jsonl`, `session.json` with `ended_at`, with a capture-stats line
    in `notes` (recording time, frames and dropped frames counted after the queue drained (0.4.0 read them
    before, 4–12 low), stills, doorway sweeps, anchors, lock resets, still latency median/max, high-res frames
    with depth, peak thermal state), a `timing …` line (see "Timing") and, in the first session, a `startup …`
    line (seconds from Start to the world map loaded, RoomPlan started, our configuration, first frame,
    tracking normal). World map and mesh files are written off the main thread.
- **Settings**. Shutter at Lock: Sharp (1/240 s, default), Flicker-safe 50 Hz (1/100 s), Flicker-safe 60 Hz
  (1/120 s). Under LED/fluorescent light on 50 Hz mains (UK/EU), if stills or video show horizontal bands,
  switch to Flicker-safe 50 Hz (the lock offers it when the metering points to flicker). RoomPlan on/off (default
  on), depth every N frames (default 6 = 10 Hz at 60 fps), raw vs smoothed depth for the PNGs (default raw;
  ARKit's `.smoothedSceneDepth` is only enabled when this is on, since 0.4.0), live mesh preview, coverage dots
  and mini-map, capture sounds.

The Files app shows the bundles under *On My iPad › FlatBlender › Projects*.

## What the app writes (bundle v1)

| File | Written by | Notes |
|---|---|---|
| `session.json` | `CaptureEngine` | At start (no `ended_at`), after Lock, after each doorway sweep, and at Finish. `type` `scan`, `lap` or `texture`. `world_map_in` = session id of the loaded map, or of the session this one continued (Next room, lap), `world_map_out` = this session id when `worldmap.arworldmap` was saved. `continued_from` (app 0.5+, optional) = the session whose ARSession this one continued without a restart (Next room, closing lap): same frame, one trajectory however long the pause before Record. `sweeps` = `[{t0, t1}]` (ARFrame timestamps; may be missing in older files = `[]`). Warnings go to `notes`. |
| `video.mov` | `StreamRecorder` | HEVC at 0.3 bit/px (~50 Mbit/s at 1920×1440@60, ~3.7 GB per 10 min), every ARFrame's `capturedImage` (copied out of ARKit's pool first), PTS = `ARFrame.timestamp`, no B-frame reordering, 2 s movie fragments (crash-safe). |
| `frames.jsonl` | `StreamRecorder` | One line per frame **in video.mov**: `idx` = index in the movie. Frames dropped under back-pressure get neither a video frame nor a line, so `idx` always matches. Intrinsics at video resolution, `tracking`, `exposure` (duration, offset, ISO). |
| `depth/NNNNNN.png`, `confidence/NNNNNN.png` | `StreamRecorder` | Every Nth recorded frame, named by that frame's `idx`. 16-bit mm (0 = no data), 8-bit 0/1/2. |
| `stills/NNNNNN.jpg`, `stills.jsonl` | `StillRecorder` | `captureHighResolutionFrame`, JPEG q0.95, ids from 1. Intrinsics at still resolution, pose, tracking, exposure, WB gains, `nearest_frame`, `gravity`, `motion` (angular speed, speed, predicted blur at still resolution). Source `auto` (StillTrigger) in scan and lap sessions, `texture` in texture sessions (`tap` only in bundles from app < 0.4). App < 0.4 also wrote `sharpness`, `motion.frame_sharpness`/`median_sharpness` and `flagged_blurry` (still optional in the models, so old bundles validate). Auto stills only while recording (so every still has a `nearest_frame`, and ingest needs frames for AR sessions). |
| `mesh.ply` | `MeshExporter` | All `ARMeshAnchor`s, vertices transformed to world, normals rotated; binary little-endian PLY. |
| `room.json`, `room.usdz` | `CaptureEngine` | `RoomBuilder(.beautifyObjects)` → `CapturedRoom` → `JSONEncoder` / `export(to:)`. |
| `worldmap.arworldmap` | `WorldMapIO` | `NSKeyedArchiver` (secure coding) of `getCurrentWorldMap`. |
| `anchors.jsonl` | `AnchorLog` | One line per anchor event: `t` (latest ARFrame timestamp), `id` (UUID), `name`, `event` (`add`, `update`, `final`), `transform` (row-major like `camera_to_world`). Only our names; RoomPlan's and ARKit's anchors are ignored. |
| `calibration.json` | — | **Not written** (TODO): ARKit doesn't expose `AVCameraCalibrationData`. |

Details worth knowing:
- Poses and intrinsics are exactly ARKit's (`ARCamera.transform`, `ARCamera.intrinsics`), converted to the
  row-major nested lists of the contract (`m[row][col] = transform.columns.col[row]`). Images are in the
  sensor's native landscape orientation, unrotated, whatever the UI orientation.
- `gravity` is the unit gravity direction in camera axes derived from the pose in ARKit's gravity-aligned
  world (`R^T · (0, -1, 0)`), not raw CoreMotion.
- Floats that came from ARKit as `Float` are written with their shortest round-trip decimal (e.g. `0.1`).
- **Still intrinsics are wrong by ~4.8 %.** `captureHighResolutionFrame`'s `ARCamera.intrinsics` are the
  stream's scaled to 4032×3024, but the 12 MP still has a narrower field of view (third scan: true focal
  ×1.048 in every session). The app records ARKit's values as they are; `scanmodel ingest` measures the factor
  per session against the stream frames and corrects the stills (`pipeline/scanmodel/stillcal.py`).

## Contract test (Swift encoding ↔ pipeline models)

```sh
cd capture/BundleKit && swift test
swift run --package-path capture/BundleKit bundlekit-sample /tmp/flatblender-sample
cd pipeline && uv run python -c "
from pathlib import Path
from scanmodel.models.bundle import Session, Frame, Still, AnchorEvent
from scanmodel.stages.ingest import check_bundle
d = Path('/tmp/flatblender-sample')
Session.model_validate_json((d / 'session.json').read_text())
[Frame.model_validate_json(l) for l in (d / 'frames.jsonl').read_text().splitlines()]
[Still.model_validate_json(l) for l in (d / 'stills.jsonl').read_text().splitlines()]
[AnchorEvent.model_validate_json(l) for l in (d / 'anchors.jsonl').read_text().splitlines()]
print(check_bundle(d)[0].model_dump_json(indent=1))"
```

The sample has no `video.mov` (ingest warns about it), synthetic depth, two tiny JPEG stills, six anchor
lines and one sweep.

## Threading (Swift 6 strict concurrency)

- `CaptureEngine` is `@MainActor @Observable`: UI state, session start/finish, camera lock, RoomPlan.
- `FrameSink` is the ARSession delegate on a dedicated serial queue. Per frame it copies the YUV image into
  our own `CVPixelBufferPool` buffer and (every Nth frame) the depth/confidence maps, then returns, so ARKit's
  small buffer pool is never held. At most 8 copied frames (~4.1 MB each, ≤ 33 MB) wait for the video
  encoder (4 until 0.3.0 (7)); beyond that frames are dropped and counted.
- `StreamRecorder` appends to the video on its serial `flatblender.stream` queue and hands each appended frame
  to a second serial queue, `flatblender.stream.lines`, which writes the depth/confidence PNGs and the
  frames.jsonl line (in `idx` order; a line names its PNGs only if they were written). The append never waits
  for PNG encoding or file creation (0.4.0): on scan 4 all 19 one-frame gaps in frames.jsonl came right after
  a depth frame (gaps after `idx % 6 == 3`, depth at `idx % 6 == 0`), i.e. `reserve` refused frames while a
  depth job held the shared queue.
- `StillRecorder` handles one high-res frame at a time on its own queue (JPEG, JSONL); one `CIContext` for the
  app (0.4.0 made one per session).
- `CaptureSounds` sets up the audio session and engine and plays on its own serial queue (0.5.0), pre-warmed when
  a capture starts: 0.4.0 did it on the main thread at the first cue.
- `CoverageMap` casts at most 2 viewpoints (each ~5.4k rays) per depth frame and reuses its scratch
  dictionaries (0.5.0; scan 5's one 137 ms coverage job was several casts due at once).
- Off the main thread since 0.5.0: loading the world map at start (RoomPlan starts meanwhile), the project
  screen's session scan, deleting a session or project. The HUD assigns only changed values (an `@Observable`
  assignment re-renders its views even when equal; the capture screen re-rendered 4×/s).
- `AnchorLog` adds anchors from the delegate queue (ticks) and the high-res frame callback (stills), and
  writes `anchors.jsonl` on its own queue. `FrameSink` passes it `didUpdate anchors` before forwarding.
- ARKit completion handlers are requested from nonisolated code, so they may run on any queue.
- RoomPlan and our recorder share the ARSession. If RoomPlan installs its own session delegate, `FrameSink`
  is re-installed and forwards every callback to RoomPlan's delegate.

## Timing (0.4.0)

`CaptureTimings` (one `StageTimer` per stage: count, total, max, p50/p99 from a 0.1 ms histogram) is reset at
each recording start and written into the session notes at Finish / Next room, e.g.

```
timing delegate p50 0.9 p99 3.1 max 12 ms (n 18476, >8 ms 3), roomplan fwd p50 0.3 max 6.0 ms, stream max 9.8 ms,
depth png max 7.1 ms, coverage max 31 ms, still jpeg max 140 ms, arkit gaps 4 (5 frames), our drops 0
```

- `delegate`: `FrameSink.session(_:didUpdate:)` as a whole (RoomPlan forward included); `>8 ms` counts slow
  callbacks (at 60 fps ARKit has few buffers to spare). `roomplan fwd`: the forward to RoomPlan's delegate.
- `stream`: one video append job; `depth png`: one depth + confidence PNG pair; `coverage`: one depth frame in
  `CoverageMap`; `still jpeg`: one 12 MP JPEG encode + write.
- `arkit gaps`: delegate timestamps more than 1.5 frame intervals apart (frames ARKit never delivered), with
  the frames they skipped. `our drops`: frames `StreamRecorder` refused or dropped (gaps in frames.jsonl
  only). The two are separate causes.

The same stages are `OSSignposter` intervals (subsystem `FlatBlender`, category Points of Interest: `delegate`,
`roomplan forward`, `stream`, `depth png`, `coverage`, `still jpeg`), for Instruments' Points of Interest track.

## Phase 0 device spike: manual test steps

Run these on the M2 iPad before relying on the app (plan, phase 0). Record results in the plan.

(Since 0.4.0 scan sessions have no shutter: steps 1–3 use auto stills, or a texture session for taps.)

1. **High-res still resolution.** Scan session, Lock, Record, pause at a new view for an auto still. Check
   `stills.jsonl` `intrinsics.width/height` and the JPEG size (expected 4032×3024 with the 1920×1440 stream).
2. **Back-to-back stills.** Texture session: tap the shutter as fast as it re-enables, 10 times. HUD "Last
   still" shows the request-to-frame latency; note the median and the stills/s rate (`t` in `stills.jsonl`).
3. **Tracking stutter during stills.** While auto stills fire, watch the HUD frame counter and dropped count;
   afterwards check `frames.jsonl` `t` gaps around each still's `t` (ingest reports gaps > 3 frame
   intervals) and whether the preview stutters.
4. **High-res frame has `sceneDepth`?** HUD "Last still … depth yes/no".
5. **Exposure lock holds while stills run.** Lock exposure, Record, take stills; HUD Camera stays "locked"
   (turns red "lock lost" if ARKit resets it). Afterwards check `frames.jsonl` `exposure.duration_s` and
   `iso` are constant and the stills' `exposure` matches.
6. **Time to heat throttling.** Record continuously with RoomPlan on; note when the HUD thermal state goes
   *fair* → *serious* (minutes since start). Repeat with RoomPlan off and with the mesh preview off.
7. **Depth survives RoomPlan on our own ARKit session.** Scan session with RoomPlan on: after "Configuring
   ARKit…" the HUD Depth row must count up at ~10 Hz (red "no sceneDepth!" means the forum-763400
   workaround failed), and RoomPlan element counts must keep updating (RoomPlan still gets frames).
   On Finish, `room.json`/`room.usdz` must exist.
8. **World map continuation.** Second session in the same project with "Continue world map", started in an
   overlapping doorway: banner goes from "Relocalising…" to "Relocalised"; `session.json` has
   `world_map_in`. Ingest both sessions and check the rooms line up.
9. **Phase 1 done-criterion.** One room captured, exported (AirDrop / Files), unzipped into `data/<project>/raw/`,
   and `scanmodel ingest` passes.

## TODO (later phases)

- Ultra-wide sessions (phase 5): `UltrawideCapture.swift`; disabled in the new-session sheet.
- `calibration.json`: needs `AVCameraCalibrationData`, which ARKit doesn't expose.
- "Upload to Mac" (`scanmodel serve` on the LAN), phase 5.
