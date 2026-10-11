import ARKit
import BundleKit
import CoreImage
import CoreVideo
import Foundation
import os

/// Auto-shot (`StillTrigger`, scan and lap sessions) and tapped (texture sessions) stills through
/// `ARSession.captureHighResolutionFrame` (~12 MP on the M2 iPad Pro).
///
/// Each still is written straight to `stills/NNNNNN.jpg` (JPEG q0.95) on a private queue, with its
/// `stills.jsonl` line; only one still is in flight at a time, so at most one high-res buffer is held.
/// Each still also gets an `ARAnchor` at its camera pose (`AnchorLog`).
///
/// `MotionMeter` describes the motion at capture time (`motion` in stills.jsonl); `onShot` drives the UI cues.
/// A request while a still is in flight is refused: FrameSink's tap request retries until its 0.5 s window ends.
final class StillRecorder: @unchecked Sendable {
    struct Stats: Sendable {
        var count = 0
        var busy = false
        var lastError: String?
        // Phase-0 spike aids: request-to-frame latency, and whether the high-res frame carried sceneDepth.
        var lastLatency: TimeInterval?
        var lastHadDepth: Bool?
        var latencies: [TimeInterval] = []
        var withDepth = 0
    }

    private struct Meta: Sendable {
        let t: TimeInterval
        let cameraToWorld: Mat4
        let intrinsics: Intrinsics
        let tracking: BundleKit.TrackingState
        let exposure: Exposure
        let whiteBalance: WhiteBalanceGains?
        let gravity: Vec3?
    }

    static let jpegQuality = 0.95

    let dir: URL
    private let anchorLog: AnchorLog
    private let motion: MotionMeter?
    private let timings: CaptureTimings?
    private let queue = DispatchQueue(label: "flatblender.stills", qos: .userInitiated)
    /// One for the app: a CIContext is costly to create (it was made per session, i.e. per room).
    private static let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var ciContext: CIContext { Self.ciContext }
    private let log = Logger(subsystem: "FlatBlender", category: "stills")

    private let lock = NSLock()
    private var stats = Stats()
    private var nextId: Int
    private var onShot: (@Sendable (StillSource) -> Void)?

    // Confined to `queue`.
    private var lines: JSONLWriter?

    init(dir: URL, anchorLog: AnchorLog, motion: MotionMeter? = nil, timings: CaptureTimings? = nil) {
        self.dir = dir
        self.anchorLog = anchorLog
        self.motion = motion
        self.timings = timings
        // Continue numbering if the session already has stills (ids start at 1).
        let existing = (try? FileManager.default.contentsOfDirectory(
            atPath: dir.appending(path: BundleLayout.stillsDir).path(percentEncoded: false))) ?? []
        nextId = existing.compactMap { Int($0.split(separator: ".").first ?? "") }.max().map { $0 + 1 } ?? 1
        stats.count = existing.count
    }

    var snapshot: Stats { lock.withLock { stats } }

    /// Called (on any queue) as soon as the high-res frame arrives, with the still's source. For the sound and the
    /// steadiness ring's flash.
    func setOnShot(_ handler: (@Sendable (StillSource) -> Void)?) {
        lock.withLock { onShot = handler }
    }

    /// Requests one high-resolution frame; false (nothing happens) while another still is in flight (only one
    /// high-res buffer is held at a time). With `trigger`, the shot is noted there at once (`pose`, the frame's
    /// at the request; `t` for auto stills' minimum interval).
    ///
    /// Nonisolated on purpose: the completion closure is then not main-actor isolated, so ARKit may call it
    /// on any queue without tripping Swift's isolation checks.
    @discardableResult
    func capture(
        session: ARSession, source: StillSource, camera: CameraControl, stream: StreamRecorder?,
        trigger: StillTrigger? = nil, pose: simd_float4x4? = nil, t: TimeInterval? = nil
    ) -> Bool {
        let id: Int? = lock.withLock {
            guard !stats.busy else { return nil }
            stats.busy = true
            defer { nextId += 1 }
            return nextId
        }
        guard let id else { return false }
        if let pose { trigger?.note(t: t, pose: pose) }

        let requested = Date.now
        session.captureHighResolutionFrame { [self] frame, error in
            guard let frame else {
                finishOne(error: "high-res capture failed: \(error?.localizedDescription ?? "no frame")")
                return
            }
            let latency = Date.now.timeIntervalSince(requested)
            let hadDepth = frame.sceneDepth != nil
            lock.withLock {
                stats.lastLatency = latency
                stats.lastHadDepth = hadDepth
                stats.latencies.append(latency)
                if hadDepth { stats.withDepth += 1 }
            }
            // Copy what we need and let go of the ARFrame; keep only its pixel buffer for the JPEG.
            let cam = frame.camera
            let meta = Meta(
                t: frame.timestamp, cameraToWorld: Mat4(cam.transform), intrinsics: cam.bundleIntrinsics,
                tracking: BundleKit.TrackingState(cam.trackingState), exposure: cam.bundleExposure(iso: camera.iso),
                whiteBalance: camera.whiteBalance, gravity: Vec3.gravityInCamera(cameraToWorld: cam.transform))
            anchorLog.addStill(id: id, t: frame.timestamp, transform: cam.transform)
            let stillMotion = motion?.assessStill(intrinsics: meta.intrinsics, exposureS: meta.exposure.durationS)
            lock.withLock { onShot }?(source)
            let pixels = Handoff(value: frame.capturedImage)
            queue.async { [self] in
                write(id: id, source: source, meta: meta, motion: stillMotion, pixels: pixels.value, stream: stream)
            }
        }
        return true
    }

    private func write(
        id: Int, source: StillSource, meta: Meta, motion: StillMotion?, pixels: CVPixelBuffer, stream: StreamRecorder?
    ) {
        do {
            if lines == nil { lines = try JSONLWriter(url: dir.appending(path: BundleLayout.stillsFile)) }
            let file = BundleLayout.stillPath(id: id)
            func encode() throws {
                let image = CIImage(cvPixelBuffer: pixels)
                let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
                let options = [
                    CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): Self.jpegQuality
                ]
                guard let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: options) else {
                    throw StillError.jpeg
                }
                try jpeg.write(to: dir.appending(path: file))
            }
            if let timings { try timings.measure(timings.stillJPEG, "still jpeg", encode) } else { try encode() }
            // Intrinsics are at the still's resolution; the JPEG is the unrotated sensor image, same size.
            let still = Still(
                id: id, file: file, t: meta.t, source: source, intrinsics: meta.intrinsics,
                cameraToWorld: meta.cameraToWorld.isFinite ? meta.cameraToWorld : nil, tracking: meta.tracking,
                exposure: meta.exposure, whiteBalance: meta.whiteBalance, nearestFrame: stream?.nearestFrame(to: meta.t),
                gravity: meta.gravity, motion: motion)
            try lines?.append(still)
            release { $0.count += 1 }
        } catch {
            finishOne(error: "still \(id): \(error.localizedDescription)")
        }
    }

    private func finishOne(error: String) {
        log.error("\(error)")
        release { $0.lastError = error }
    }

    /// Ends the still in flight (updating its stats).
    private func release(_ update: (inout Stats) -> Void) {
        lock.withLock {
            update(&stats)
            stats.busy = false
        }
    }

    /// Waits for the still in flight (if any) and closes stills.jsonl.
    func finish() async {
        // ARKit normally answers within a few hundred ms; don't hang the finish if it never does.
        let deadline = Date.now.addingTimeInterval(5)
        while snapshot.busy && Date.now < deadline { try? await Task.sleep(for: .milliseconds(50)) }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [self] in
                try? lines?.close()
                lines = nil
                cont.resume()
            }
        }
    }
}

enum StillError: LocalizedError {
    case jpeg

    var errorDescription: String? { "JPEG encoding failed." }
}
