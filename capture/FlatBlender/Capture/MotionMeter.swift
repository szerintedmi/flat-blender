import BundleKit
import Foundation
import simd

/// Live camera motion from ARKit poses, for the steadiness cues, the auto-still gate (with `StillTrigger`), the
/// tapped still's steady wait (texture sessions) and the `motion` of each still: angular and linear speed between
/// consecutive frames, smoothed (~60 ms), turned into the motion blur predicted at the training resolution
/// (`MotionBlur`, 960 px long side) with the frame's exposure. Steady = < 1 px. "Too fast" = > 2 px (≈ 4 px at
/// 1920 px) for 0.3 s; it clears after 1 s below 1.5 px.
///
/// Until 0.3.0 (7) this also scored the stream's Laplacian sharpness (every 6th frame, 1.5 s median) to gate auto
/// stills and flag blurry taps. Dropped in 0.4.0: pose-predicted blur is the cheaper, view-independent signal,
/// and the pipeline measures sharpness itself when it selects images.
///
/// `update` runs on the ARKit delegate queue, `current` on the main actor, `assessStill` in ARKit's high-res
/// callback: shared state is behind `lock`.
final class MotionMeter: @unchecked Sendable {
    struct Reading: Sendable {
        var t: TimeInterval = 0  // last frame with a pose (0: none yet)
        var angularSpeed = 0.0  // rad/s, smoothed
        var speed = 0.0  // m/s, smoothed
        var exposureS = 0.0
        var blurPx = 0.0  // predicted at the training resolution
        var tooFast = false

        var steady: Bool { t > 0 && blurPx < MotionMeter.steadyBlurPx }
    }

    static let steadyBlurPx = 1.0
    static let fastBlurPx = 2.0
    static let calmBlurPx = 1.5
    static let fastAfter: TimeInterval = 0.3
    static let calmAfter: TimeInterval = 1.0
    /// A tapped still (texture sessions) waits for predicted blur below this (training resolution; a little above
    /// "steady", since tapping the screen itself jolts the iPad).
    static let tapSteadyBlurPx = 1.5
    static let smoothing: TimeInterval = 0.06

    private let lock = NSLock()
    private var reading = Reading()
    private var last: (t: TimeInterval, pose: simd_float4x4)?
    private var fastSince: TimeInterval?
    private var calmSince: TimeInterval?

    var current: Reading { lock.withLock { reading } }

    /// One frame with tracking normal. `fx` and `width`/`height` are the frame's (stream) intrinsics.
    func update(t: TimeInterval, pose: simd_float4x4, fx: Double, width: Int, height: Int, exposureS: Double) {
        lock.withLock {
            defer { last = (t, pose) }
            guard let last, t > last.t, t - last.t < 0.2 else { return }
            let dt = t - last.t
            let w = Self.rotationAngle(last.pose, pose) / dt
            let v = Double(simd_distance(Self.position(pose), Self.position(last.pose))) / dt
            let a = 1 - exp(-dt / Self.smoothing)
            reading.angularSpeed += a * (w - reading.angularSpeed)
            reading.speed += a * (v - reading.speed)
            reading.t = t
            reading.exposureS = exposureS
            let px = MotionBlur.pixels(
                fx: fx, angularSpeedRadS: reading.angularSpeed, speedMS: reading.speed, exposureS: exposureS)
            reading.blurPx = MotionBlur.atTrainingResolution(px, width: width, height: height)

            if reading.blurPx > Self.fastBlurPx {
                calmSince = nil
                let since = fastSince ?? t
                fastSince = since
                if t - since >= Self.fastAfter { reading.tooFast = true }
            } else {
                fastSince = nil
                if reading.blurPx < Self.calmBlurPx {
                    let since = calmSince ?? t
                    calmSince = since
                    if t - since >= Self.calmAfter { reading.tooFast = false }
                } else {
                    calmSince = nil
                }
            }
        }
    }

    /// Tracking isn't normal: poses may jump, so start the speed estimate over at the next good frame.
    func invalidate() {
        lock.withLock { last = nil }
    }

    /// The `motion` of a still captured with `intrinsics` (the still's) and `exposureS`: the latest smoothed
    /// speeds, and the blur they predict at the still's resolution. Nil before the first tracked frame.
    func assessStill(intrinsics: Intrinsics, exposureS: Double?) -> StillMotion? {
        let r = current
        guard r.t > 0 else { return nil }
        let px = MotionBlur.pixels(
            fx: intrinsics.fx, angularSpeedRadS: r.angularSpeed, speedMS: r.speed, exposureS: exposureS ?? r.exposureS)
        func round(_ x: Double, _ step: Double) -> Double { (x / step).rounded() * step }
        return StillMotion(
            angularSpeedDegS: round(r.angularSpeed * 180 / .pi, 0.1), speedMS: round(r.speed, 0.001),
            blurPx: round(px, 0.01))
    }

    private static func position(_ m: simd_float4x4) -> SIMD3<Float> { SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z) }

    /// Rotation angle (radians) between two camera orientations, all axes (roll blurs the image edges too).
    private static func rotationAngle(_ a: simd_float4x4, _ b: simd_float4x4) -> Double {
        func rot(_ m: simd_float4x4) -> simd_double3x3 {
            simd_double3x3(
                SIMD3<Double>(Double(m.columns.0.x), Double(m.columns.0.y), Double(m.columns.0.z)),
                SIMD3<Double>(Double(m.columns.1.x), Double(m.columns.1.y), Double(m.columns.1.z)),
                SIMD3<Double>(Double(m.columns.2.x), Double(m.columns.2.y), Double(m.columns.2.z)))
        }
        let q = simd_quatd(rot(a).transpose * rot(b))
        // 2·atan2(|v|, |w|) stays accurate for the tiny per-frame angles (acos near 1 doesn't).
        return 2 * atan2(simd_length(q.imag), abs(q.real))
    }
}
