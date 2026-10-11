import ARKit
import BundleKit
import Foundation
import Observation
import RoomPlan
import UIKit
import os

/// Everything a capture session needs to start.
struct SessionRequest: Sendable {
    let project: ProjectInfo
    let sessionId: String
    let dir: URL
    let type: SessionType
    let roomName: String
    /// Earlier session whose world map should be loaded as `initialWorldMap` (nil: start a fresh map).
    let worldMapFrom: SessionSummary?
    var settings: CaptureSettings  // the shutter mode can change at Lock (flicker prompt)
}

/// Drives one capture session: ARKit (+ RoomPlan) on a shared ARSession, camera lock, stream recording,
/// stills, anchors, and writing the bundle on finish. Next room and the closing lap continue the same
/// ARSession in a new session folder. Main-actor state for the UI; the heavy lifting runs in
/// FrameSink / StreamRecorder / StillRecorder on their own queues.
@MainActor
@Observable
final class CaptureEngine {
    /// `closing`: Next room / Start lap was tapped; the previous session is being saved while ARKit keeps running
    /// (and the next one is prepared). `finishing`: Finish or Discard, ARKit stops at the end.
    enum Phase: Equatable {
        case idle, starting, ready, recording, closing, finishing, finished
    }

    enum Relocalisation: Equatable {
        case notRequested, relocalising, relocalised
    }

    enum CameraState: Equatable {
        case auto, metering, locked, lockLost
    }

    /// What comes after the session being closed.
    enum AfterClose: Equatable {
        case room, lap, finish
    }

    /// Progress of `closing` / `finishing`, for the panel.
    enum CloseStatus: Equatable {
        case saving(String)  // the step running now
        case failed([String])  // retryable steps that failed: Retry, Continue anyway, Finish
        case preparing(String)  // the previous session is saved; the next one is being set up
        case ready  // saved and set up: waiting for the next room's name
        case setupFailed(String)  // the next session's folder could not be created
    }

    // MARK: UI state

    private(set) var phase: Phase = .idle
    private(set) var progress = ""
    var message: String?

    private(set) var tracking: BundleKit.TrackingState = .notAvailable
    private(set) var mapping = "none"
    private(set) var hasDepth = false
    private(set) var interrupted = false
    private(set) var framesRecorded = 0
    private(set) var framesDropped = 0
    private(set) var depthFrames = 0
    private(set) var depthHz = 0.0
    private(set) var stillCount = 0
    private(set) var stillBusy = false
    private(set) var lastStillInfo: String?
    private(set) var elapsed: TimeInterval = 0
    private(set) var recordingElapsed: TimeInterval = 0
    private(set) var thermal: ProcessInfo.ThermalState = .nominal
    private(set) var freeBytes: Int64?
    private(set) var relocalisation: Relocalisation = .notRequested
    private(set) var cameraState: CameraState = .auto
    /// While metering: 0...1 towards the automatic lock (`ExposureSettle`).
    private(set) var meterProgress = 0.0
    private(set) var lockedCamera: LockedCamera?
    private(set) var roomPlanActive = false
    private(set) var roomPlanInstruction: String?
    private(set) var roomSummary: String?
    private(set) var anchorCount = 0
    /// While the doorway sweep runs (start of every Next room's recording): when it started, and how far the
    /// turn and the step through the door got (`SweepProgress`).
    private(set) var sweepStartedAt: Date?
    private(set) var sweepProgress: SweepProgress?
    /// The next room's recording starts with the doorway sweep (its steps are shown before Record).
    private(set) var sweepPending = false
    /// The latest still captured (for the steadiness ring's flash).
    private(set) var stillCue: StillCue?

    /// While `closing` / `finishing`: what comes next, how far it got, the session being saved, and the name
    /// given for the next room (nil until submitted).
    private(set) var after: AfterClose?
    private(set) var closeStatus: CloseStatus?
    private(set) var closingLabel = ""
    private(set) var nextRoomName: String?

    struct StillCue: Equatable {
        let at: Date
        let source: StillSource
    }

    /// Texture sessions: the latest tap that was not taken (no steady moment within 0.5 s, or the previous still
    /// still saving).
    private(set) var tapMissed: TapMissed?

    struct TapMissed: Equatable {
        let at: Date
        let busy: Bool  // false: the camera kept moving
    }

    /// Live motion (lock-protected, not observed: the cue views poll it).
    var motion: MotionMeter.Reading { sink.motion.current }

    static let sweepMinDuration: TimeInterval = 5  // Done is enabled after this

    /// The session being recorded; replaced when the next room / the lap opens.
    private(set) var request: SessionRequest

    /// Only texture sessions have a shutter button (scan and lap sessions shoot auto stills only, since 0.4.0):
    /// stills without a stream, once the position is found. Stays enabled while a still is in flight: the tap
    /// then waits up to 0.5 s for it (`FrameSink.TapRequest`).
    var hasShutter: Bool { request.type == .texture }
    var canStill: Bool { hasShutter && positioned && phase == .ready }

    /// "LED flicker likely" at Lock (sharp shutter, metered shutter a whole number of flicker periods): which
    /// mains. Non-modal; cleared by an answer, metering, Lock or Record.
    private(set) var flickerPrompt: FlickerMains?

    /// The session is in the project's frame and tracking: relocalised against the loaded world map (or a
    /// fresh map), and tracking normally right now. Recording and stills wait for it.
    var positioned: Bool {
        tracking == .normal && relocalisation != .relocalising
    }

    var canRecord: Bool { phase == .ready && positioned }

    /// When relocalisation started (nil when no world map was loaded), for the "finding position" timer.
    private(set) var relocalisingSince: Date?
    var canMeterOrLock: Bool { (phase == .ready) && camera.isAvailable }

    /// Scan and lap sessions auto-shoot while recording (not during the doorway sweep: fast motion, blur).
    private var autoShoots: Bool { request.type == .scan || request.type == .lap }

    // MARK: Internals

    @ObservationIgnored let arSession = ARSession()
    @ObservationIgnored private let camera = CameraControl()
    @ObservationIgnored private let anchorLog: AnchorLog
    @ObservationIgnored private let sink: FrameSink
    /// Coverage overlay + mini-map (fed by `sink` while recording; kept across Next room and the lap).
    @ObservationIgnored let coverage: CoverageOverlay
    @ObservationIgnored private var stills: StillRecorder
    @ObservationIgnored private let trigger = StillTrigger()
    @ObservationIgnored private var recorder: StreamRecorder?
    @ObservationIgnored private var roomSession: RoomCaptureSession?
    @ObservationIgnored private var configuration: ARWorldTrackingConfiguration?
    @ObservationIgnored private var bundle: BundleKit.Session
    @ObservationIgnored private var hudTask: Task<Void, Never>?
    @ObservationIgnored private var meterTask: Task<Void, Never>?
    @ObservationIgnored private var recordingStartedAt: Date?
    @ObservationIgnored private var depthWindow: [(time: Date, count: Int)] = []
    @ObservationIgnored private var roomEndWaiter: CheckedContinuation<CapturedRoomData?, Never>?
    @ObservationIgnored private var roomEnded = false
    @ObservationIgnored private var roomEndedData: CapturedRoomData?
    @ObservationIgnored private var roomEndToken = 0
    @ObservationIgnored private var maxThermal: ProcessInfo.ThermalState = .nominal
    @ObservationIgnored private var sweepT0: TimeInterval?
    @ObservationIgnored private var closing: Closing?
    @ObservationIgnored private var closeTask: Task<Void, Never>?
    @ObservationIgnored private var nextPrepared = false
    @ObservationIgnored private var prepareStarted = false
    /// Times this session found the lock reset by ARKit and put it back (session notes).
    @ObservationIgnored private var lockResets = 0
    @ObservationIgnored private var startup = StartupMarks()
    @ObservationIgnored private let delegateQueue = DispatchQueue(label: "flatblender.arsession", qos: .userInteractive)
    @ObservationIgnored private let log = Logger(subsystem: "FlatBlender", category: "engine")
    @ObservationIgnored private let sounds = CaptureSounds.shared
    @ObservationIgnored private var wasTooFast = false
    @ObservationIgnored private var lastSlowDownCue: Date = .distantPast
    @ObservationIgnored private var diskQueryRunning = false
    static let slowDownCueEvery: TimeInterval = 8

    init(request: SessionRequest) {
        self.request = request
        anchorLog = AnchorLog(session: arSession)
        sink = FrameSink(camera: camera, anchorLog: anchorLog)
        coverage = CoverageOverlay(map: sink.coverage)
        stills = StillRecorder(dir: request.dir, anchorLog: anchorLog, motion: sink.motion, timings: sink.timings)
        bundle = BundleKit.Session(
            projectId: request.project.projectId, sessionId: request.sessionId, type: request.type,
            device: DeviceInfo.bundleDevice, startedAt: .now,
            roomName: request.roomName.isEmpty ? nil : request.roomName)
        stills.setOnShot(onShot)
        sounds.prewarm()  // audio session + engine off the main thread, so the first cue doesn't hitch
    }

    /// From ARKit's high-res frame callback (any queue): hop to the main actor for the cues.
    private var onShot: @Sendable (StillSource) -> Void {
        { [weak self] source in Task { @MainActor in self?.stillShot(source: source) } }
    }

    /// A still was captured: a faint tick for an auto still, a click for a tapped one (texture sessions), and the
    /// ring flash in the view. (iPads have no haptics.)
    private func stillShot(source: StillSource) {
        stillCue = StillCue(at: .now, source: source)
        sounds.play(source == .auto ? .autoStill : .tapStill)
    }

    private func usesRoomPlan(_ type: SessionType) -> Bool {
        type == .scan && request.settings.roomPlanEnabled && RoomCaptureSession.isSupported
    }

    /// Seconds from the engine's creation to the first ARKit milestones, for the first session's notes
    /// (`startup …`): where the wait after "Start" goes.
    private struct StartupMarks {
        let t0 = ProcessInfo.processInfo.systemUptime
        var marks: [(String, TimeInterval)] = []
        var noted = false

        mutating func mark(_ name: String) {
            guard !marks.contains(where: { $0.0 == name }) else { return }
            marks.append((name, ProcessInfo.processInfo.systemUptime - t0))
        }

        /// Once: the line for the first session's notes.
        mutating func note() -> String? {
            guard !noted, !marks.isEmpty else { return nil }
            noted = true
            return "startup " + marks.map { "\($0.0) \(String(format: "%.1f", $0.1)) s" }.joined(separator: ", ")
        }
    }

    // MARK: Start

    func start() async {
        guard phase == .idle else { return }
        phase = .starting
        UIApplication.shared.isIdleTimerDisabled = true

        guard ARWorldTrackingConfiguration.isSupported,
            ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
            ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
        else {
            message = "This device has no LiDAR world tracking (sceneDepth + mesh). FlatBlender needs an iPad Pro."
            phase = .ready  // allow Discard
            return
        }

        arSession.delegateQueue = delegateQueue
        arSession.delegate = sink
        startHUD()
        writeSessionJSON()
        // RoomPlan starts while the world map loads (both take a moment; the map can be tens of MB).
        if usesRoomPlan(request.type) {
            runRoomPlan()
            // didStart normally arrives within a second. If it doesn't, continue without waiting forever.
            Task { [weak self] in
                try? await Task.sleep(for: .seconds(8))
                while let self, self.phase == .starting, self.configuration == nil {  // the map is still loading
                    try? await Task.sleep(for: .milliseconds(100))
                }
                guard let self, self.phase == .starting, self.configuration != nil else { return }
                self.message = "RoomPlan did not report a start; using the ARKit configuration anyway."
                self.applyConfiguration()
            }
        }

        var worldMap: ARWorldMap?
        if let from = request.worldMapFrom {
            progress = "Loading the world map…"
            let url = from.dir.appending(path: BundleLayout.worldMapFile)
            // Unarchiving a world map (tens of MB after a few rooms) took the main thread for a noticeable moment.
            let loaded = await Task.detached(priority: .userInitiated) { () -> Result<Handoff<ARWorldMap>, any Error> in
                Result { Handoff(value: try WorldMapIO.load(from: url)) }
            }.value
            startup.mark("map")
            switch loaded {
            case .success(let map):
                worldMap = map.value
                bundle.worldMapIn = from.id
                relocalisation = .relocalising
                relocalisingSince = .now
                writeSessionJSON()
            case .failure(let error):
                message = "Could not load the world map of \(from.id): \(error.localizedDescription). Starting a new map."
            }
            progress = ""
        }
        guard phase == .starting else { return }  // discarded meanwhile
        configuration = Self.makeConfiguration(
            worldMap: worldMap, planeDetection: usesRoomPlan(request.type), smoothedDepth: request.settings.smoothedDepth)

        if !usesRoomPlan(request.type) {
            arSession.run(configuration!)
            startup.mark("config")
            phase = .ready
        } else if roomPlanActive {
            applyConfiguration()
        }  // else roomPlanDidStart applies it
    }

    private func runRoomPlan() {
        roomEnded = false
        roomEndedData = nil
        roomPlanActive = false
        let room = RoomCaptureSession(arSession: arSession)
        room.delegate = self
        roomSession = room
        var roomConfig = RoomCaptureSession.Configuration()
        roomConfig.isCoachingEnabled = false
        room.run(configuration: roomConfig)
    }

    /// Re-runs our depth-enabled configuration on the shared session. With RoomPlan this must happen after
    /// RoomPlan's didStart, otherwise sceneDepth is missing from the frames
    /// (https://developer.apple.com/forums/thread/763400). No reset options: RoomPlan's tracking continues;
    /// an `initialWorldMap` in the configuration starts relocalisation against the earlier session.
    private func applyConfiguration() {
        guard let configuration else { return }
        let current = arSession.delegate
        if let current, current !== sink {
            sink.setForward(current)  // RoomPlan took the delegate: keep feeding it
            log.info("forwarding ARSession delegate calls to \(String(describing: type(of: current)))")
        }
        arSession.delegate = sink
        sink.resetRelocalisation()
        arSession.run(configuration)
        startup.mark("config")
        if phase == .starting { phase = .ready }
    }

    /// If something (RoomPlan) replaced the session delegate after we installed ours, take it back and
    /// forward to the replacement, so recording never silently stops.
    private func ensureDelegate() {
        guard phase == .ready || phase == .recording, let current = arSession.delegate, current !== sink else { return }
        sink.setForward(current)
        arSession.delegate = sink
        log.info("re-installed FrameSink as ARSession delegate")
    }

    /// `smoothedDepth`: the settings want `smoothedSceneDepth` for the PNGs; only then is it enabled (ARKit's
    /// temporal filtering is extra work every frame).
    static func makeConfiguration(
        worldMap: ARWorldMap?, planeDetection: Bool, smoothedDepth: Bool
    ) -> ARWorldTrackingConfiguration {
        let c = ARWorldTrackingConfiguration()
        c.worldAlignment = .gravity
        c.videoFormat = videoFormat()
        var semantics: ARConfiguration.FrameSemantics = []
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { semantics.insert(.sceneDepth) }
        if smoothedDepth && ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        }
        c.frameSemantics = semantics
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) { c.sceneReconstruction = .mesh }
        if planeDetection { c.planeDetection = [.horizontal, .vertical] }  // RoomPlan keeps working
        c.environmentTexturing = .none
        c.isAutoFocusEnabled = true
        c.videoHDRAllowed = false  // constant tone mapping across the session
        c.initialWorldMap = worldMap
        return c
    }

    /// ARKit's recommendation for high-res stills (1920x1440 @ 60 expected on the M2 iPad Pro). Fallback: the
    /// largest 4:3 back-camera format at 60 fps below 4K.
    static func videoFormat() -> ARConfiguration.VideoFormat {
        if let f = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing { return f }
        let all = ARWorldTrackingConfiguration.supportedVideoFormats
        let candidates = all.filter { f in
            let w = f.imageResolution.width
            let h = f.imageResolution.height
            return abs(w / h - 4.0 / 3.0) < 0.01 && f.framesPerSecond == 60 && w < 3840 && f.captureDevicePosition == .back
        }
        let area = { (f: ARConfiguration.VideoFormat) in f.imageResolution.width * f.imageResolution.height }
        return candidates.max { area($0) < area($1) } ?? all[0]
    }

    // MARK: Camera (one control: Lock exposure → metering → locked; 0.5.0)

    /// The exposure button: auto or locked → meter (the lock follows by itself once auto exposure settles);
    /// metering → lock now; lock lost → re-apply the same lock.
    func exposureTapped() {
        switch cameraState {
        case .auto, .locked: startMetering()
        case .metering: lock()
        case .lockLost: relock()
        }
    }

    /// Back to auto exposure / white balance / focus, then locks by itself once they have settled
    /// (`ExposureSettle`: steady ~0.6 s, at most 5 s).
    private func startMetering() {
        flickerPrompt = nil
        do {
            try camera.meter()
        } catch {
            message = error.localizedDescription
            return
        }
        cameraState = .metering
        meterProgress = 0
        lockedCamera = nil
        sounds.play(.meter)
        meterTask?.cancel()
        meterTask = Task { [weak self] in
            var settle = ExposureSettle()
            while !Task.isCancelled {
                guard let self, self.cameraState == .metering, self.canMeterOrLock else { return }
                guard let s = self.camera.meteringSample else { return }
                let state = settle.add(
                    t: ProcessInfo.processInfo.systemUptime, durationS: s.durationS, iso: s.iso, gains: s.gains,
                    adjusting: s.adjusting)
                switch state {
                case .metering(let p):
                    if abs(p - self.meterProgress) > 0.01 { self.meterProgress = p }
                case .settled:
                    self.lock()
                    return
                case .timedOut:
                    self.lock()
                    self.message = "Exposure kept changing for 5 s; locked anyway. To meter again: hold still on a mid-bright wall and tap Locked."
                    return
                }
                try? await Task.sleep(for: .milliseconds(66))
            }
        }
    }

    /// Locks with the session's shutter mode. If auto exposure metered a whole number of flicker periods
    /// (`ExposurePolicy.suspectedFlicker`), that goes to session.json (`flicker_suspected`), and in the sharp
    /// mode a non-modal prompt offers the flicker-safe mode (`flicker_prompt`). Never switches by itself.
    private func lock() {
        meterTask?.cancel()
        meterTask = nil
        flickerPrompt = nil
        do {
            var locked = try camera.lock(mode: request.settings.shutterMode)
            if let metered = locked.metered?.durationS, let mains = ExposurePolicy.suspectedFlicker(meteredDurationS: metered) {
                locked.flickerSuspected = mains
                if locked.shutterMode == .sharp {
                    locked.flickerPrompt = .shown
                    flickerPrompt = mains
                }
            }
            applyLock(locked)
        } catch {
            cameraState = .auto
            message = error.localizedDescription
        }
    }

    /// ARKit reset the lock (a configuration re-run): put the same values back, not a new metering, so every
    /// room of the walk keeps one exposure.
    private func relock() {
        guard let locked = lockedCamera else { return startMetering() }
        do {
            try camera.relock(locked)
            cameraState = .locked
            sounds.play(.lock)
        } catch {
            message = error.localizedDescription
        }
    }

    /// The flicker prompt's "Use …": switches the shutter mode (this session, the next rooms and Settings) and
    /// re-locks from the same metered exposure.
    func acceptFlicker() {
        guard let mains = flickerPrompt, let previous = lockedCamera, canMeterOrLock else {
            flickerPrompt = nil
            return
        }
        flickerPrompt = nil
        let mode = mains.shutterMode
        request.settings.shutterMode = mode
        UserDefaults.standard.set(mode.rawValue, forKey: SettingKey.shutterMode)
        do {
            var locked = try camera.lock(mode: mode, metered: previous.metered)
            locked.flickerSuspected = mains
            locked.flickerPrompt = .accepted
            applyLock(locked)
        } catch {
            message = error.localizedDescription
        }
    }

    /// The flicker prompt's "Keep 1/240 s".
    func declineFlicker() {
        flickerPrompt = nil
        guard var locked = lockedCamera, locked.flickerPrompt == .shown else { return }
        locked.flickerPrompt = .declined
        lockedCamera = locked
        bundle.camera = locked
        writeSessionJSON()
    }

    private func applyLock(_ locked: LockedCamera) {
        lockedCamera = locked
        bundle.camera = locked
        cameraState = .locked
        meterProgress = 0
        writeSessionJSON()
        sounds.play(.lock)
    }

    /// After a configuration re-run (next room, lap): ARKit resets the lock a moment later, so re-apply it and
    /// check for ~2 s that it holds (re-applying while it doesn't). Still lost after that: the orange Relock.
    /// Replaces 0.4.0's fixed 500 ms sleep + one relock: polls every 100 ms, re-applying whenever the lock is gone,
    /// and is done once it has held for 0.3 s at least 0.6 s after the run (ARKit's reset lands a moment after
    /// `run`), at most 2 s. Typically ~0.7 s.
    private func relockAfterConfiguration() async {
        guard let locked = lockedCamera else { return }
        let start = ProcessInfo.processInfo.systemUptime
        var held = 0
        while ProcessInfo.processInfo.systemUptime - start < 2 {
            if camera.holds(locked) {
                held += 1
                if held >= 3 && ProcessInfo.processInfo.systemUptime - start >= 0.6 { break }
            } else {
                held = 0
                try? camera.relock(locked)
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        cameraState = camera.holds(locked) ? .locked : .lockLost
    }

    // MARK: Recording and stills

    func startRecording() {
        guard canRecord, let configuration else { return }
        flickerPrompt = nil  // unanswered: stays "shown" in session.json
        if cameraState == .metering { lock() }
        if cameraState == .lockLost { relock() }
        sink.timings.reset()
        lockResets = 0
        let rec = StreamRecorder(
            dir: request.dir, depthEvery: request.settings.depthEvery,
            nominalFPS: Double(configuration.videoFormat.framesPerSecond),
            smoothedDepth: request.settings.smoothedDepth, timings: sink.timings)
        recorder = rec
        sink.setRecorder(rec, smoothedDepth: request.settings.smoothedDepth)
        if bundle.camera == nil { bundle.camera = cameraState == .locked ? lockedCamera : camera.unlockedSnapshot }
        recordingStartedAt = .now
        phase = .recording
        anchorLog.begin(sessionId: request.sessionId, dir: request.dir)
        if sweepPending {
            startSweep()
        } else if autoShoots {
            sink.setAutoStills(.init(trigger: trigger, stills: stills))
        }
    }

    // MARK: Doorway sweep

    /// Both faces of the wall around the doorway within seconds, in one recording: the pipeline measures the
    /// wall's thickness there with almost no drift. Recording continues; auto stills pause.
    private func startSweep() {
        sweepPending = false
        sweepT0 = sink.snapshot.lastT
        sweepStartedAt = .now
        sweepProgress = SweepProgress()
        feedSweep()
    }

    /// The sweep ends by itself once the camera turned round and stepped through (then 2 s to look), or at 20 s.
    private func feedSweep() {
        guard var p = sweepProgress else { return }
        let pose = sink.snapshot.pose
        let t = ProcessInfo.processInfo.systemUptime
        let state: SweepProgress.State
        if let m = pose {
            let forward = -SIMD2(Double(m.columns.2.x), Double(m.columns.2.z))
            state = p.add(t: t, x: Double(m.columns.3.x), z: Double(m.columns.3.z), yaw: atan2(forward.y, forward.x))
        } else {
            state = p.add(t: t, x: nil, z: nil, yaw: nil)
        }
        update(\.sweepProgress, p)
        if state != .running { endSweep() }
    }

    /// Ends the sweep (Done, countdown over, or the session closing) and records it in session.json `sweeps`.
    func endSweep() {
        guard let t0 = sweepT0 else { return }
        bundle.sweeps.append(Sweep(t0: t0, t1: sink.snapshot.lastT))
        sweepT0 = nil
        sweepStartedAt = nil
        sweepProgress = nil
        writeSessionJSON()
        if phase == .recording && autoShoots { sink.setAutoStills(.init(trigger: trigger, stills: stills)) }
    }

    /// Texture sessions: a tap shoots at the first steady moment within 0.5 s, or not at all (low tone + hint):
    /// never seconds later while the camera is moving.
    func shootStill() {
        guard canStill else { return }
        sink.requestTap(.init(
            stills: stills, camera: camera,
            onMissed: { [weak self] miss in Task { @MainActor in self?.tapNotTaken(busy: miss == .busy) } }))
    }

    private func tapNotTaken(busy: Bool) {
        tapMissed = TapMissed(at: .now, busy: busy)
        sounds.play(.tapMissed)
    }

    // MARK: Closing a session (Next room, closing lap, Finish)

    /// Next room (scan sessions, while recording): stops this room's recording at once and saves it in the
    /// background while ARKit keeps running (the next room shares this frame, recorded as its `world_map_in`).
    /// The name is asked meanwhile; the next room is prepared (RoomPlan, configuration, camera relock) once this
    /// one is saved, and records only when the user taps Record, with the doorway sweep.
    func nextRoom() {
        guard phase == .recording, request.type == .scan else { return }
        beginClose(after: .room)
    }

    /// The closing lap (type `lap`, after the last room): like Next room, without a name and without RoomPlan.
    /// Walk back through every room to the start, so the end of the recording sees what its start saw (loop
    /// closures for the pose graph).
    func startLap() {
        guard phase == .recording, request.type == .scan else { return }
        beginClose(after: .lap)
    }

    /// Stops recording, saves the session and stops ARKit. Also from the closing panel ("Finish instead"): then
    /// the saving under way carries on and ARKit stops at its end (a next session already prepared is dropped;
    /// no folder was made for it yet).
    func finish() {
        switch phase {
        case .ready, .recording:
            beginClose(after: .finish)
        case .closing:
            after = .finish
            phase = .finishing
            switch closeStatus {
            case .preparing, .ready, .setupFailed: stopAll()
            default: break  // still saving, or failed (the panel offers Retry / Continue anyway): ends after it
            }
        default:
            break
        }
    }

    /// The name for the next room (can be given while the previous one is still saving).
    func submitNextRoom(name: String) {
        guard phase == .closing, after == .room else { return }
        nextRoomName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        openNextIfReady()
    }

    /// From the closing panel: the closing lap instead of another room.
    func switchToLap() {
        guard phase == .closing, after == .room else { return }
        after = .lap
        nextRoomName = nil
        if prepareStarted {
            // A RoomPlan session may already run for the room: re-prepare for the lap.
            nextPrepared = false
            Task { await prepareNext() }
        }
    }

    /// Failed retryable steps: try them again (walk around a little first for the world map).
    func retryClose() {
        if case .setupFailed = closeStatus {
            openNextIfReady()
            return
        }
        guard case .failed = closeStatus, closing != nil else { return }
        setStatus("Retrying…")  // synchronously: a second tap finds no .failed and does nothing
        closeTask = Task { await runRetryable() }
    }

    /// Failed retryable steps: keep going without them (each stays a warning in the session's notes).
    func continueAfterFailure() {
        guard case .failed = closeStatus, closing != nil else { return }
        setStatus("Saving session.json…")  // synchronously: a second tap finds no .failed and does nothing
        closeTask = Task { await completeClose(giveUp: true) }
    }

    /// What `closing` keeps of the session being saved. The engine's `request`, `bundle`, `stills` and
    /// `roomSession` still belong to it until the next session opens.
    private struct Closing {
        enum Step: CaseIterable { case worldMap, mesh, room }
        var notes: [String] = []  // warnings that no retry can fix
        var pending: [Step] = []  // retryable steps not done yet
        var errors: [Step: String] = [:]
        var meshAnchors: [ARMeshAnchor] = []
        var roomData: CapturedRoomData?
        var finalAnchors: [AnchorLog.Snapshot] = []
        var stats = ""
        var timing = ""
        var anchorsClosed = false
        var neverRelocalised = false  // at the tap (prepareNext resets `relocalisation` for the next session)
    }

    private func beginClose(after next: AfterClose) {
        guard phase == .ready || phase == .recording else { return }
        meterTask?.cancel()
        after = next
        nextRoomName = nil
        nextPrepared = false
        prepareStarted = false
        closingLabel = request.roomName.isEmpty ? (request.type == .lap ? "the lap" : request.sessionId) : request.roomName
        phase = next == .finish ? .finishing : .closing

        // Stop everything that records, now: nothing of the walk to the next room ends up in this session.
        endSweep()
        sweepPending = false
        anchorLog.stopTicks()
        sink.cancelTap()
        sink.setAutoStills(nil)
        sink.setRecorder(nil, smoothedDepth: false)
        roomSession?.stop(pauseARSession: false)  // RoomPlan stops adding the next room's walls; didEnd brings the data
        var c = Closing()
        c.neverRelocalised = relocalisation == .relocalising
        c.meshAnchors = arSession.currentFrame?.anchors.compactMap { $0 as? ARMeshAnchor } ?? []
        closing = c
        closeTask = Task { await runClose() }
    }

    private func setStatus(_ text: String) {
        closeStatus = .saving(text)
        progress = text
    }

    /// The steps that run once (video, stills, RoomPlan's data), then the retryable ones.
    private func runClose() async {
        guard var c = closing else { return }
        var frames = framesRecorded
        if let recorder {
            setStatus("Finishing video…")
            let result = await recorder.finish()
            bundle.video = result.video
            bundle.depth = result.depth
            frames = result.frames
            if let failure = result.failure { c.notes.append("stream: \(failure)") }
            if result.frames == 0 { c.notes.append("no stream frames recorded") }
        }
        // After the queued frames drained (up to 8 were still pending at the tap: scan 5's counts were 4-12 low).
        let dropped = recorder?.snapshot.dropped ?? 0
        recorder = nil
        setStatus("Saving stills…")
        await stills.finish()
        c.stats = captureStats(frames: frames, dropped: dropped)
        c.timing = sink.timings.note(ourDrops: dropped)
        if roomSession != nil {
            setStatus("Waiting for RoomPlan…")
            if let data = await awaitRoomEnd() {
                c.roomData = data
            } else {
                c.notes.append("RoomPlan returned no data")
            }
        }
        c.pending = Closing.Step.allCases.filter { step in
            switch step {
            case .worldMap: true
            case .mesh: !c.meshAnchors.isEmpty
            case .room: c.roomData != nil
            }
        }
        closing = c
        await runRetryable()
    }

    /// World map (+ the anchors in it), mesh.ply, room.json/usdz. Each failure is kept for Retry.
    private func runRetryable() async {
        guard var c = closing else { return }
        c.errors = [:]
        for step in c.pending {
            do {
                switch step {
                case .worldMap:
                    setStatus("Saving world map…")
                    c.finalAnchors = try await saveWorldMap()
                case .mesh:
                    setStatus("Saving mesh…")
                    if c.meshAnchors.isEmpty {  // a retry: the mesh as it is now
                        c.meshAnchors = arSession.currentFrame?.anchors.compactMap { $0 as? ARMeshAnchor } ?? []
                    }
                    try await saveMesh(c.meshAnchors)
                case .room:
                    setStatus("Building room (RoomPlan)…")
                    if let data = c.roomData { try await exportRoom(data) }
                }
                c.pending.removeAll { $0 == step }
                startPreparingIfARKitDone(c)
            } catch {
                if step == .mesh { c.meshAnchors = [] }  // don't hold ARKit's buffers while waiting for an answer
                c.errors[step] = Self.describe(step, error)
            }
        }
        closing = c
        if c.errors.isEmpty {
            await completeClose(giveUp: false)
        } else {
            closeStatus = .failed(c.pending.compactMap { c.errors[$0] })
            progress = ""
        }
    }

    /// The next room / lap can be prepared (new RoomPlan run, configuration, relock) as soon as the steps that read
    /// the ARSession (world map, mesh) are done: RoomPlan's export only processes the data it already has (Apple's
    /// multi-room flow builds rooms while the next one is captured). Its recording still waits for the whole save.
    private func startPreparingIfARKitDone(_ c: Closing) {
        guard !prepareStarted, phase == .closing, after == .room || after == .lap,
            !c.pending.contains(.worldMap), !c.pending.contains(.mesh)
        else { return }
        prepareStarted = true
        Task { await prepareNext() }
    }

    private static func describe(_ step: Closing.Step, _ error: any Error) -> String {
        switch step {
        case .worldMap: "world map not saved: \(error.localizedDescription)"
        case .mesh: "mesh.ply not saved: \(error.localizedDescription)"
        case .room: "RoomPlan export failed: \(error.localizedDescription)"
        }
    }

    /// anchors.jsonl's `final` lines, then session.json with `ended_at` and the notes. Then what comes next.
    private func completeClose(giveUp: Bool) async {
        guard var c = closing else { return }
        if giveUp { c.notes += c.pending.compactMap { c.errors[$0] } }
        if !c.anchorsClosed {
            await anchorLog.end(finals: c.finalAnchors, t: sink.snapshot.lastT)
            c.anchorsClosed = true
        }
        // TODO: calibration.json (lens distortion). ARKit doesn't expose AVCameraCalibrationData for its
        // camera; needs a separate AVFoundation capture (phase 5, with the ultra-wide session). Not written.
        bundle.endedAt = .now
        if bundle.camera == nil { bundle.camera = camera.unlockedSnapshot }
        if c.neverRelocalised { c.notes.append("never relocalised against \(bundle.worldMapIn ?? "the world map")") }
        bundle.notes = ([bundle.notes, c.stats, c.timing, startup.note() ?? ""].filter { !$0.isEmpty } + c.notes)
            .joined(separator: "; ")
        closing = nil
        if !writeSessionJSON() {
            // Nothing else to retry: the message banner says so. The folder has everything else.
            c.notes.append("session.json not updated")
        }
        let warnings = c.notes

        switch after {
        case .finish, nil:
            progress = warnings.isEmpty ? "Saved." : "Saved with warnings: " + warnings.joined(separator: "; ")
            stopAll()
        case .room, .lap:
            if !warnings.isEmpty { message = "\(closingLabel): " + warnings.joined(separator: "; ") }
            if prepareStarted {
                closeStatus = nextPrepared ? .ready : .preparing(after == .lap ? "Preparing the closing lap…" : "Preparing the next room…")
                progress = ""
                openNextIfReady()
            } else {
                prepareStarted = true
                await prepareNext()
            }
        }
    }

    /// Stops ARKit (and a RoomPlan session prepared for a next room) and ends the capture.
    private func stopAll() {
        closeTask = nil
        roomSession?.stop(pauseARSession: true)
        roomSession = nil
        arSession.pause()
        stopHUD()
        if !progress.hasPrefix("Saved") { progress = "Saved." }
        closeStatus = nil
        phase = .finished
    }

    /// Next room / lap, once the previous session is saved: a new RoomPlan session (rooms), our configuration
    /// again (RoomPlan's drops sceneDepth; it resets the camera lock), the lock re-applied and checked. The
    /// session folder is made when the name is there too (`openNextIfReady`).
    private func prepareNext() async {
        guard phase == .closing, let next = after, next != .finish else { return }
        let type: SessionType = next == .lap ? .lap : .scan
        if closing == nil {  // else the save is still running and its panel shows that
            closeStatus = .preparing(type == .lap ? "Preparing the closing lap…" : "Preparing the next room…")
            progress = ""
        }
        trigger.reset()
        relocalisation = .notRequested
        roomSummary = nil
        roomPlanInstruction = nil
        configuration?.initialWorldMap = nil  // already in the map's frame: don't relocalise again

        if usesRoomPlan(type) {
            runRoomPlan()
            let deadline = Date.now.addingTimeInterval(8)
            while !roomPlanActive && Date.now < deadline { try? await Task.sleep(for: .milliseconds(50)) }
        } else {
            if let room = roomSession {  // prepared for a room, then switched to the lap: drop it
                roomSession = nil
                room.stop(pauseARSession: false)
            }
            sink.setForward(nil)  // the previous room's RoomPlan has stopped
            configuration?.planeDetection = []  // planes are only for RoomPlan; less heat on the lap
        }
        guard phase == .closing, after == next else { return }  // finished or switched meanwhile
        applyConfiguration()
        await relockAfterConfiguration()
        guard phase == .closing, after == next else { return }
        nextPrepared = true
        if closing == nil { closeStatus = .ready }
        openNextIfReady()
    }

    /// Opens the next session (folder, session.json) once it is prepared and, for a room, named. Recording then
    /// waits for Record (with the doorway sweep's steps shown).
    private func openNextIfReady() {
        // Only once the previous session is fully saved (`closing` is cleared at the end of `completeClose`).
        guard phase == .closing, closing == nil, nextPrepared, let next = after, next != .finish else { return }
        if next == .room && nextRoomName == nil { return }
        let type: SessionType = next == .lap ? .lap : .scan
        let room = next == .lap ? "" : (nextRoomName ?? "")
        let previous = request
        do {
            let (id, dir) = try ProjectStore.makeSessionDir(
                projectDir: previous.dir.deletingLastPathComponent(), roomName: next == .lap ? "lap" : room)
            request = SessionRequest(
                project: previous.project, sessionId: id, dir: dir, type: type, roomName: room,
                worldMapFrom: nil, settings: previous.settings)
        } catch {
            closeStatus = .setupFailed("Could not create the folder for \(type == .lap ? "the closing lap" : "the next room"): \(error.localizedDescription)")
            return
        }
        bundle = BundleKit.Session(
            projectId: request.project.projectId, sessionId: request.sessionId, type: request.type,
            device: DeviceInfo.bundleDevice, worldMapIn: previous.sessionId, continuedFrom: previous.sessionId, startedAt: .now,
            roomName: room.isEmpty ? nil : room)
        if cameraState == .locked, var carried = lockedCamera {
            if carried.flickerPrompt == .shown { carried.flickerPrompt = nil }  // asked (and unanswered) in an earlier room
            bundle.camera = carried
        }
        stills = StillRecorder(dir: request.dir, anchorLog: anchorLog, motion: sink.motion, timings: sink.timings)
        stills.setOnShot(onShot)
        recordingStartedAt = nil
        recordingElapsed = 0
        framesRecorded = 0
        framesDropped = 0
        depthFrames = 0
        depthWindow = []
        maxThermal = thermal
        writeSessionJSON()
        sweepPending = type == .scan
        after = nil
        closeStatus = nil
        nextRoomName = nil
        nextPrepared = false
        prepareStarted = false
        phase = .ready
    }

    /// One line for session.json `notes`, answering the device-spike questions (plan, phase 0) from real scans:
    /// recording length, dropped frames, still latency and whether high-res frames carry depth, peak heat.
    private func captureStats(frames: Int, dropped: Int) -> String {
        let st = stills.snapshot
        let rec = Int(recordingElapsed)
        var parts = [String(format: "rec %d:%02d", rec / 60, rec % 60), "\(frames) frames (\(dropped) dropped)", "\(st.count) stills"]
        if !bundle.sweeps.isEmpty { parts.append("\(bundle.sweeps.count) doorway sweep\(bundle.sweeps.count == 1 ? "" : "s")") }
        parts.append("\(anchorLog.count) anchors")
        if lockResets > 0 { parts.append("camera lock reset by ARKit \(lockResets)x (re-applied)") }
        let ms = st.latencies.map { Int($0 * 1000) }.sorted()
        if !ms.isEmpty {
            parts.append("still latency median \(ms[ms.count / 2]) ms, max \(ms.last ?? 0) ms")
            parts.append("still depth \(st.withDepth)/\(ms.count)")
        }
        let heat = ["nominal", "fair", "serious", "critical"]
        if let r = camera.isoRange { parts.append("format ISO \(Int(r.lowerBound))-\(Int(r.upperBound))") }
        parts.append("thermal max \(heat[min(maxThermal.rawValue, heat.count - 1)])")
        maxThermal = ProcessInfo.processInfo.thermalState
        return parts.joined(separator: ", ")
    }

    /// Stops everything and deletes the session folder (before recording: the first room, or a next room / lap
    /// that was opened but not recorded; the rooms before it stay saved).
    func discard() async {
        guard phase == .starting || phase == .ready || phase == .recording else { return }
        meterTask?.cancel()
        phase = .finishing
        progress = "Discarding…"
        sink.cancelTap()
        sink.setAutoStills(nil)
        sink.setRecorder(nil, smoothedDepth: false)
        if let recorder { _ = await recorder.finish() }
        await stills.finish()
        await anchorLog.end(finals: [], t: 0)
        roomSession?.stop(pauseARSession: true)
        roomSession = nil
        arSession.pause()
        stopHUD()
        let dir = request.dir
        await Task.detached(priority: .userInitiated) { try? FileManager.default.removeItem(at: dir) }.value
        phase = .finished
    }

    /// Fetches and archives ARKit's world map, writes it (off the main thread) and returns our anchors in it.
    private func saveWorldMap() async throws -> [AnchorLog.Snapshot] {
        let session = arSession
        // ARKit has no timeout of its own: give up after 30 s (Retry / Continue anyway then appear).
        let saved: WorldMapIO.Saved = try await withCheckedThrowingContinuation { cont in
            let once = OnceResume(cont)
            WorldMapIO.archiveCurrent(of: session, completion: once.resume)
            Task {
                try? await Task.sleep(for: .seconds(30))
                once.resume(.failure(WorldMapError.timedOut))
            }
        }
        let url = request.dir.appending(path: BundleLayout.worldMapFile)
        try await Task.detached(priority: .userInitiated) { try saved.data.write(to: url, options: .atomic) }.value
        bundle.worldMapOut = request.sessionId
        return saved.anchors
    }

    private func saveMesh(_ anchors: [ARMeshAnchor]) async throws {
        let url = request.dir.appending(path: BundleLayout.meshFile)
        let mesh = Handoff(value: anchors)
        let counts = try await Task.detached(priority: .userInitiated) {
            try MeshExporter.write(anchors: mesh.value, to: url)
        }.value
        log.info("mesh.ply: \(counts.vertices) vertices, \(counts.faces) faces")
    }

    /// RoomPlan was stopped at the tap (`beginClose`); waits for its didEnd (at most 20 s).
    private func awaitRoomEnd() async -> CapturedRoomData? {
        if roomEnded { return roomEndedData }
        return await withCheckedContinuation { cont in
            roomEndWaiter = cont
            roomEndToken += 1
            let token = roomEndToken
            Task { [weak self] in
                try? await Task.sleep(for: .seconds(20))
                if self?.roomEndToken == token { self?.resumeRoomEnd(nil) }
            }
        }
    }

    private func resumeRoomEnd(_ data: CapturedRoomData?) {
        roomEndWaiter?.resume(returning: data)
        roomEndWaiter = nil
    }

    private func exportRoom(_ data: CapturedRoomData) async throws {
        let builder = RoomBuilder(options: [.beautifyObjects])
        let room = try await builder.capturedRoom(from: data)
        let dir = request.dir
        try await Task.detached(priority: .userInitiated) {
            let json = try JSONEncoder().encode(room)
            try json.write(to: dir.appending(path: BundleLayout.roomJSONFile), options: .atomic)
            try room.export(to: dir.appending(path: BundleLayout.roomUSDZFile))
        }.value
    }

    // MARK: RoomPlan callbacks (hopped to the main actor)

    private func isCurrent(_ id: ObjectIdentifier) -> Bool { roomSession.map(ObjectIdentifier.init) == id }

    fileprivate func roomPlanDidStart(_ id: ObjectIdentifier) {
        guard isCurrent(id) else { return }
        roomPlanActive = true
        startup.mark("roomplan")
        if phase == .starting && configuration != nil { applyConfiguration() }
    }

    fileprivate func roomPlanDidEnd(_ id: ObjectIdentifier, _ data: CapturedRoomData?, error: String?) {
        guard isCurrent(id) else { return }
        roomPlanActive = false
        roomEnded = true
        roomEndedData = data
        if let error, phase != .finishing, phase != .closing { message = "RoomPlan stopped: \(error)" }
        resumeRoomEnd(data)
    }

    fileprivate func roomPlanUpdated(_ id: ObjectIdentifier, _ room: CapturedRoom) {
        guard isCurrent(id), phase != .closing else { return }
        roomSummary = "walls \(room.walls.count) · doors \(room.doors.count) · windows \(room.windows.count) · openings \(room.openings.count) · objects \(room.objects.count)"
    }

    fileprivate func roomPlanInstruction(_ id: ObjectIdentifier, _ text: String?) {
        guard isCurrent(id), roomPlanInstruction != text else { return }
        roomPlanInstruction = text
    }

    nonisolated fileprivate static func text(for instruction: RoomCaptureSession.Instruction) -> String? {
        switch instruction {
        case .moveCloseToWall: "Move closer to the wall"
        case .moveAwayFromWall: "Move away from the wall"
        case .slowDown: "Slow down"
        case .turnOnLight: "Turn on more light"
        case .lowTexture: "Low texture"
        case .normal: nil
        @unknown default: nil
        }
    }

    // MARK: HUD

    private func startHUD() {
        hudTask = Task { [weak self] in
            var tick = 0
            while !Task.isCancelled {
                guard let engine = self else { return }
                engine.refreshHUD(slow: tick % 8 == 0)
                if tick % 120 == 0 { engine.refreshFreeDisk() }  // every 30 s
                tick += 1
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
    }

    private func stopHUD() {
        hudTask?.cancel()
        hudTask = nil
        meterTask?.cancel()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    /// Assigns only real changes: an `@Observable` property re-renders its views on every assignment, equal or
    /// not, and this runs 4 times a second (the whole capture screen used to re-render each time).
    private func update<T: Equatable>(_ keyPath: ReferenceWritableKeyPath<CaptureEngine, T>, _ value: T) {
        if self[keyPath: keyPath] != value { self[keyPath: keyPath] = value }
    }

    private func refreshHUD(slow: Bool) {
        let s = sink.snapshot
        update(\.tracking, s.tracking)
        update(\.mapping, s.mapping)
        update(\.hasDepth, s.hasDepth)
        update(\.interrupted, s.interrupted)
        if s.lastT > 0 { startup.mark("first frame") }
        if s.tracking == .normal { startup.mark("tracking") }
        if let error = s.error, message == nil { message = "ARKit: \(error)" }
        // Only after our configuration (with initialWorldMap) ran and ARKit reported relocalising: before
        // that, tracking can be normal against RoomPlan's own fresh map.
        if relocalisation == .relocalising && s.sawRelocalizing && s.tracking == .normal {
            relocalisation = .relocalised
            sounds.play(.found)
        }
        ensureDelegate()

        let now = Date.now
        if sweepStartedAt != nil { feedSweep() }
        // Texture sessions have no recording: their anchors start once the position is found.
        if request.type == .texture && phase == .ready && positioned && !anchorLog.isLogging {
            anchorLog.begin(sessionId: request.sessionId, dir: request.dir)
        }
        update(\.anchorCount, anchorLog.count)
        update(\.elapsed, now.timeIntervalSince(bundle.startedAt).rounded(.down))
        if let r = recorder?.snapshot {
            update(\.framesRecorded, r.frames)
            update(\.framesDropped, r.dropped)
            update(\.depthFrames, r.depthFrames)
            if let failure = r.failure, message == nil { message = "Recording: \(failure)" }
            depthWindow.append((now, r.depthFrames))
            depthWindow.removeAll { now.timeIntervalSince($0.time) > 2 }
            if let first = depthWindow.first, now.timeIntervalSince(first.time) > 0.5 {
                update(\.depthHz, (Double(r.depthFrames - first.count) / now.timeIntervalSince(first.time) * 10).rounded() / 10)
            }
        }
        if let started = recordingStartedAt, phase == .recording {
            update(\.recordingElapsed, now.timeIntervalSince(started).rounded(.down))
        }
        // "Turn slower" also as a sound (the hint is easy to miss while looking at the room): on each onset, at
        // most every 8 s, not during the doorway sweep (moving fast is part of it).
        let tooFast = phase == .recording && sweepStartedAt == nil && sink.motion.current.tooFast
        if tooFast && !wasTooFast && now.timeIntervalSince(lastSlowDownCue) >= Self.slowDownCueEvery {
            sounds.play(.slowDown)
            lastSlowDownCue = now
        }
        wasTooFast = tooFast
        let st = stills.snapshot
        update(\.stillCount, st.count)
        update(\.stillBusy, st.busy)
        if let latency = st.lastLatency {
            update(\.lastStillInfo, "\(Int(latency * 1000)) ms, depth \(st.lastHadDepth == true ? "yes" : "no")")
        }
        if let e = st.lastError, message == nil { message = e }

        // ARKit reset the lock (a configuration re-run): put the same values back at once; only if that fails does
        // the orange Relock appear.
        if cameraState == .locked && !camera.isStillLocked, let locked = lockedCamera {
            lockResets += 1
            if (try? camera.relock(locked)) == nil { cameraState = .lockLost }
        }
        if slow {
            update(\.thermal, ProcessInfo.processInfo.thermalState)
            if thermal.rawValue > maxThermal.rawValue { maxThermal = thermal }
        }
    }

    /// The free-disk query (`volumeAvailableCapacityForImportantUsage`) can take a while: off the main thread.
    private func refreshFreeDisk() {
        guard !diskQueryRunning else { return }
        diskQueryRunning = true
        Task { [weak self] in
            let bytes = await Task.detached(priority: .utility) { DeviceInfo.freeDiskBytes() }.value
            self?.freeBytes = bytes
            self?.diskQueryRunning = false
        }
    }

    @discardableResult
    private func writeSessionJSON() -> Bool {
        do {
            try BundleJSON.writeDocument(bundle, to: request.dir.appending(path: BundleLayout.sessionFile))
            return true
        } catch {
            message = "Could not write session.json: \(error.localizedDescription)"
            return false
        }
    }
}

// RoomPlan calls its delegate on its own queue; hop to the main actor.
extension CaptureEngine: RoomCaptureSessionDelegate {
    nonisolated func captureSession(_ session: RoomCaptureSession, didStartWith configuration: RoomCaptureSession.Configuration) {
        let id = ObjectIdentifier(session)
        Task { @MainActor in self.roomPlanDidStart(id) }
    }

    nonisolated func captureSession(_ session: RoomCaptureSession, didEndWith data: CapturedRoomData, error: (any Error)?) {
        let message = error?.localizedDescription
        let id = ObjectIdentifier(session)
        Task { @MainActor in self.roomPlanDidEnd(id, data, error: message) }
    }

    nonisolated func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
        let id = ObjectIdentifier(session)
        Task { @MainActor in self.roomPlanUpdated(id, room) }
    }

    nonisolated func captureSession(_ session: RoomCaptureSession, didProvide instruction: RoomCaptureSession.Instruction) {
        let text = Self.text(for: instruction)
        let id = ObjectIdentifier(session)
        Task { @MainActor in self.roomPlanInstruction(id, text) }
    }
}
