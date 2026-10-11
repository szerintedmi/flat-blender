import Foundation
import simd

/// Decides when to auto-shoot a still (scan and lap sessions): the camera is nearly still (< 0.3 rad/s,
/// < 0.15 m/s over the last ~0.25 s) and the view is new (> 0.3 m or > 15 deg from every still so far). FrameSink
/// adds `MotionMeter`'s steadiness on top: predicted blur < 1 px at 960 px at this instant.
///
/// `consider` runs on the ARKit delegate queue; `note` and `reset` may come from the main actor, hence the lock.
final class StillTrigger: @unchecked Sendable {
    static let maxTurnRate = 0.3  // rad/s
    static let maxSpeed = 0.15  // m/s
    static let minMove: Float = 0.3  // m
    static let minTurn = 15.0 * Double.pi / 180
    static let window: TimeInterval = 0.25
    static let minInterval: TimeInterval = 0.8  // between auto stills, so the stream recorder keeps up

    private let lock = NSLock()
    private var recent: [(t: TimeInterval, pose: simd_float4x4)] = []
    private var shots: [simd_float4x4] = []
    private var lastShot: TimeInterval = -.infinity

    /// Returns true if a still should be shot now. The caller reports an accepted shot with `note`.
    func consider(t: TimeInterval, pose: simd_float4x4) -> Bool {
        lock.withLock {
            recent.append((t, pose))
            recent.removeAll { t - $0.t > Self.window * 2 }
            guard t - lastShot >= Self.minInterval,
                let old = recent.first(where: { t - $0.t >= Self.window }), t > old.t
            else { return false }
            let dt = t - old.t
            let speed = Double(simd_distance(Self.position(pose), Self.position(old.pose))) / dt
            let turn = Self.angle(pose, old.pose) / dt
            guard speed < Self.maxSpeed, turn < Self.maxTurnRate else { return false }
            return shots.allSatisfy {
                simd_distance(Self.position(pose), Self.position($0)) > Self.minMove || Self.angle(pose, $0) > Self.minTurn
            }
        }
    }

    /// Records a still taken at `pose`.
    func note(t: TimeInterval?, pose: simd_float4x4) {
        lock.withLock {
            shots.append(pose)
            if let t { lastShot = t }
        }
    }

    func reset() {
        lock.withLock {
            recent = []
            shots = []
            lastShot = -.infinity
        }
    }

    private static func position(_ m: simd_float4x4) -> SIMD3<Float> { SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z) }

    /// Angle between the viewing directions (camera -z) of two poses, in radians.
    private static func angle(_ a: simd_float4x4, _ b: simd_float4x4) -> Double {
        let fa = -SIMD3(a.columns.2.x, a.columns.2.y, a.columns.2.z)
        let fb = -SIMD3(b.columns.2.x, b.columns.2.y, b.columns.2.z)
        return Double(acos(min(1, max(-1, simd_dot(simd_normalize(fa), simd_normalize(fb))))))
    }
}
