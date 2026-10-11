import ARKit
import BundleKit
import Foundation

/// The ARSession delegate. Runs on the session's delegate queue (never the main thread unless RoomPlan
/// changes that), copies what it needs out of each ARFrame and hands it to the `StreamRecorder`, then
/// returns so ARKit can recycle the frame's buffers.
///
/// RoomPlan, when given our ARSession, may install its own session delegate (Apple forum thread 763400).
/// After RoomPlan starts, `CaptureEngine` re-installs this sink and passes RoomPlan's delegate as `forward`;
/// every callback is forwarded to it, so both RoomPlan and the recorder see all frames.
///
/// It also drives `AnchorLog`: a tick anchor per second (tracking normal) and ARKit's updates to our anchors,
/// and times itself and the forward to RoomPlan (`CaptureTimings`).
final class FrameSink: NSObject, ARSessionDelegate, @unchecked Sendable {
    struct Status: Sendable {
        var tracking: BundleKit.TrackingState = .notAvailable
        var mapping: String = "none"
        var hasDepth = false
        var pose: simd_float4x4?  // the latest camera pose while tracking is normal (doorway sweep guidance)
        var lastT: TimeInterval = 0
        var interrupted = false
        var sawRelocalizing = false  // since the last resetRelocalisation()
        var error: String?
    }

    private let lock = NSLock()
    private var status = Status()
    private var recorder: StreamRecorder?
    private var smoothedDepth = false
    private var autoStills: AutoStills?
    private var tap: TapRequest?
    private var tapCount = 0
    private weak var forward: (any ARSessionDelegate)?
    private let copier = PixelBufferCopier()  // delegate queue only

    let camera: CameraControl
    let anchorLog: AnchorLog
    /// Camera motion (predicted blur): steadiness cues, the auto-still gate, the tap's steady wait, still `motion`.
    let motion = MotionMeter()
    /// Stage timings for the session notes and Instruments (reset at each recording start).
    let timings: CaptureTimings
    /// Capture coverage for the overlay: fed with the recorded frames (it throttles itself to 5 Hz).
    let coverage: CoverageMap

    init(camera: CameraControl, anchorLog: AnchorLog) {
        self.camera = camera
        self.anchorLog = anchorLog
        let timings = CaptureTimings()
        self.timings = timings
        coverage = CoverageMap(timings: timings)
    }

    var snapshot: Status { lock.withLock { status } }

    func setRecorder(_ recorder: StreamRecorder?, smoothedDepth: Bool) {
        lock.withLock {
            self.recorder = recorder
            self.smoothedDepth = smoothedDepth
        }
    }

    /// Auto-shoot while recording: the trigger decides, the recorder shoots (`nil` turns it off).
    struct AutoStills {
        let trigger: StillTrigger
        let stills: StillRecorder
    }

    func setAutoStills(_ auto: AutoStills?) {
        lock.withLock { autoStills = auto }
    }

    /// A tapped still (texture sessions only: scans and laps have no shutter button since 0.4.0) waiting for a
    /// steady moment. It shoots at the first frame within `tapWindow` where the camera is steady enough
    /// (predicted blur < `MotionMeter.tapSteadyBlurPx` = 1.5 px at 960 px: tapping jolts the iPad, so this
    /// mostly waits out the jolt) and no other still is in flight. Never later: after the window it is dropped
    /// and `onMissed` says why. A new tap replaces a waiting one.
    struct TapRequest: Sendable {
        enum Miss: Sendable { case moving, busy }
        let stills: StillRecorder
        let camera: CameraControl
        let onMissed: @Sendable (Miss) -> Void
        fileprivate var id = 0
        fileprivate var deadline: TimeInterval?
    }

    static let tapWindow: TimeInterval = 0.5

    func requestTap(_ request: TapRequest) {
        lock.withLock {
            tapCount += 1
            var r = request
            r.id = tapCount
            tap = r
        }
    }

    /// Drops a waiting tap without a cue (the session is closing).
    func cancelTap() {
        lock.withLock { tap = nil }
    }

    func resetRelocalisation() {
        lock.withLock { status.sawRelocalizing = false }
    }

    func setForward(_ delegate: (any ARSessionDelegate)?) {
        lock.withLock { forward = delegate === self ? nil : delegate }
    }

    private var forwardTarget: (any ARSessionDelegate)? { lock.withLock { forward } }

    // MARK: Frames

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        timings.measure(timings.delegate, "delegate") {
            handle(frame, session: session)
            if let forward = forwardTarget {
                timings.measure(timings.roomPlanForward, "roomplan forward") { forward.session?(session, didUpdate: frame) }
            }
        }
    }

    private func handle(_ frame: ARFrame, session: ARSession) {
        let cam = frame.camera
        let tracking = BundleKit.TrackingState(cam.trackingState)
        let (recorder, smoothed, auto) = lock.withLock { () -> (StreamRecorder?, Bool, AutoStills?) in
            status.tracking = tracking
            status.mapping = frame.worldMappingStatus.label
            status.sawRelocalizing = status.sawRelocalizing || tracking == .limitedRelocalizing
            status.pose = tracking == .normal ? cam.transform : nil
            status.hasDepth = frame.sceneDepth != nil
            status.lastT = frame.timestamp
            return (self.recorder, smoothedDepth, autoStills)
        }

        if tracking == .normal {
            anchorLog.tick(t: frame.timestamp, transform: cam.transform)
            motion.update(
                t: frame.timestamp, pose: cam.transform, fx: Double(cam.intrinsics.columns.0.x),
                width: Int(cam.imageResolution.width), height: Int(cam.imageResolution.height),
                exposureS: cam.exposureDuration)
        } else {
            motion.invalidate()
        }

        handleTap(frame: frame, tracking: tracking, session: session)

        // `consider` first: it must see every frame. Then the motion gate: steady at this instant (re-checked each
        // frame while the trigger holds, so the shot lands on the first steady frame).
        if let auto, recorder != nil, tracking == .normal, auto.trigger.consider(t: frame.timestamp, pose: cam.transform),
            motion.current.steady
        {
            auto.stills.capture(
                session: session, source: .auto, camera: camera, stream: recorder, trigger: auto.trigger,
                pose: cam.transform, t: frame.timestamp)
        }
        guard let recorder else { return }
        timings.noteFrame(t: frame.timestamp, fps: recorder.nominalFPS)
        coverage.offer(frame)

        guard let wantsDepth = recorder.reserve() else { return }
        guard let pixels = copier.copy(frame.capturedImage) else {
            recorder.cancel()
            return
        }
        var depth: DepthCopy?
        if wantsDepth, let data = smoothed ? frame.smoothedSceneDepth : frame.sceneDepth {
            depth = DepthCopy(data)
        }
        recorder.submit(
            StreamRecorder.FrameJob(
                t: frame.timestamp, cameraToWorld: Mat4(cam.transform), intrinsics: cam.bundleIntrinsics,
                tracking: tracking, exposure: cam.bundleExposure(iso: camera.iso), pixels: pixels, depth: depth))
    }

    /// Shoots or drops the waiting tap (see `TapRequest`).
    private func handleTap(frame: ARFrame, tracking: BundleKit.TrackingState, session: ARSession) {
        let t = frame.timestamp
        guard let request = lock.withLock({ () -> TapRequest? in
            if tap != nil && tap?.deadline == nil { tap?.deadline = t + Self.tapWindow }
            return tap
        }) else { return }
        let reading = motion.current
        let steady = tracking == .normal && reading.t > 0 && reading.blurPx < MotionMeter.tapSteadyBlurPx
        let shot = steady && request.stills.capture(
            session: session, source: .texture, camera: request.camera, stream: nil)
        let expired = !shot && t >= (request.deadline ?? t)
        guard shot || expired else { return }
        let current = lock.withLock { () -> Bool in
            guard tap?.id == request.id else { return false }  // replaced by a newer tap meanwhile
            tap = nil
            return true
        }
        if expired && current { request.onMissed(steady ? .busy : .moving) }
    }

    // MARK: Session events (recorded for the HUD, then forwarded)

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let tracking = BundleKit.TrackingState(camera.trackingState)
        lock.withLock {
            status.tracking = tracking
            status.sawRelocalizing = status.sawRelocalizing || tracking == .limitedRelocalizing
        }
        forwardTarget?.session?(session, cameraDidChangeTrackingState: camera)
    }

    func session(_ session: ARSession, didFailWithError error: any Error) {
        lock.withLock { status.error = error.localizedDescription }
        forwardTarget?.session?(session, didFailWithError: error)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        lock.withLock { status.interrupted = true }
        forwardTarget?.sessionWasInterrupted?(session)
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        lock.withLock { status.interrupted = false }
        forwardTarget?.sessionInterruptionEnded?(session)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        forwardTarget?.session?(session, didAdd: anchors)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        anchorLog.updated(anchors, t: lock.withLock { status.lastT })
        forwardTarget?.session?(session, didUpdate: anchors)
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        forwardTarget?.session?(session, didRemove: anchors)
    }

    // Anything else (relocalisation prompt, collaboration data, ...) goes to the forward target if it
    // implements it.
    override func responds(to aSelector: Selector!) -> Bool {
        if super.responds(to: aSelector) { return true }
        return forwardTarget?.responds(to: aSelector) ?? false
    }

    override func forwardingTarget(for aSelector: Selector!) -> Any? {
        if let target = forwardTarget, target.responds(to: aSelector) { return target }
        return super.forwardingTarget(for: aSelector)
    }
}
