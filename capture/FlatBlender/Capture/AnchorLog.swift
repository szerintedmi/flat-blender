import ARKit
import BundleKit
import Foundation
import os

/// Adds an `ARAnchor` at every still and every 1 s, and logs what ARKit later does with them to
/// `anchors.jsonl`, to test whether ARKit's loop closures move anchors (free drift correction if they do).
///
/// Lines: `add` when we add one; `update` from the session delegate's `didUpdate anchors` for any session's
/// anchors (they persist on the ARSession across Next room), when it moved > 1 mm or > 0.05° since its last
/// line; `final` for every one of ours in the world map saved when a session closes. Only names made by
/// `BundleLayout.anchorName`, so RoomPlan's and ARKit's own anchors (planes, meshes) are ignored.
///
/// Threading: `tick` and `updated` run on the ARKit delegate queue, `addStill` wherever ARKit delivers the
/// high-res frame, `begin` / `end` on the main actor. Shared state is behind `lock`; the file and the
/// per-anchor last-logged transforms are confined to `queue`.
final class AnchorLog: @unchecked Sendable {
    /// One of our anchors, copied out of ARKit.
    struct Snapshot: Sendable {
        let id: UUID
        let name: String
        let transform: simd_float4x4

        init?(_ anchor: ARAnchor) {
            guard BundleLayout.isAnchorName(anchor.name), let name = anchor.name else { return nil }
            id = anchor.identifier
            self.name = name
            transform = anchor.transform
        }
    }

    static let tickInterval: TimeInterval = 1
    static let minMove = 0.001  // m
    static let minTurn = 0.05 * Double.pi / 180  // rad

    private let session: ARSession
    private let queue = DispatchQueue(label: "flatblender.anchors", qos: .utility)
    private let log = Logger(subsystem: "FlatBlender", category: "anchors")

    // Shared, behind `lock`.
    private let lock = NSLock()
    private var sessionId: String?  // set while logging
    private var ticking = false
    private var nextTick = 1
    private var lastTick: TimeInterval = -.infinity
    private var added = 0

    // Confined to `queue`.
    private var lines: JSONLWriter?
    private var lastLogged: [UUID: simd_float4x4] = [:]  // one entry per anchor (1/s): small

    init(session: ARSession) {
        self.session = session
    }

    /// Anchors added in the current session (for the HUD and notes).
    var count: Int { lock.withLock { added } }

    /// Starts logging into `dir/anchors.jsonl` with names prefixed by `sessionId`, and the 1 s ticks.
    func begin(sessionId: String, dir: URL) {
        let started = lock.withLock { () -> Bool in
            guard self.sessionId == nil else { return false }
            self.sessionId = sessionId
            ticking = true
            nextTick = 1
            lastTick = -.infinity
            added = 0
            return true
        }
        guard started else { return }
        queue.async { [self] in
            do {
                lines = try JSONLWriter(url: dir.appending(path: BundleLayout.anchorsFile))
            } catch {
                log.error("anchors.jsonl: \(error.localizedDescription)")
            }
        }
    }

    var isLogging: Bool { lock.withLock { sessionId != nil } }

    /// Stops the 1 s ticks (still anchors keep coming until `end`), before the session's world map is saved.
    func stopTicks() {
        lock.withLock { ticking = false }
    }

    /// Per frame (tracking normal): adds a tick anchor at the camera when one is due.
    func tick(t: TimeInterval, transform: simd_float4x4) {
        let name: String? = lock.withLock {
            guard ticking, let sessionId, t - lastTick >= Self.tickInterval else { return nil }
            lastTick = t
            defer { nextTick += 1 }
            return BundleLayout.anchorName(sessionId: sessionId, kind: .tick, number: nextTick)
        }
        if let name { add(name: name, t: t, transform: transform) }
    }

    /// At a still's high-res frame: an anchor at its camera pose.
    func addStill(id: Int, t: TimeInterval, transform: simd_float4x4) {
        guard let sessionId = lock.withLock({ sessionId }) else { return }
        add(name: BundleLayout.anchorName(sessionId: sessionId, kind: .still, number: id), t: t, transform: transform)
    }

    private func add(name: String, t: TimeInterval, transform: simd_float4x4) {
        let anchor = ARAnchor(name: name, transform: transform)
        session.add(anchor: anchor)
        lock.withLock { added += 1 }
        let id = anchor.identifier
        queue.async { [self] in write(.add, id: id, name: name, t: t, transform: transform) }
    }

    /// From `session(_:didUpdate:)`. `t` is the latest ARFrame timestamp.
    func updated(_ anchors: [ARAnchor], t: TimeInterval) {
        let ours = anchors.compactMap(Snapshot.init)
        guard !ours.isEmpty, isLogging else { return }
        queue.async { [self] in
            for a in ours {
                if let last = lastLogged[a.id] {
                    let d = poseDifference(last, a.transform)
                    guard d.metres > Self.minMove || d.radians > Self.minTurn else { continue }
                }
                write(.update, id: a.id, name: a.name, t: t, transform: a.transform)
            }
        }
    }

    /// Writes `final` lines for our anchors in the session's saved world map (empty if it wasn't saved), and
    /// closes anchors.jsonl. Updates until the next `begin` are not logged; the next session's file picks them
    /// up, since the throttle compares against the last line written.
    func end(finals: [Snapshot], t: TimeInterval) async {
        lock.withLock {
            sessionId = nil
            ticking = false
        }
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [self] in
                for a in finals { write(.final, id: a.id, name: a.name, t: t, transform: a.transform) }
                try? lines?.close()
                lines = nil
                cont.resume()
            }
        }
    }

    // MARK: Queue

    private func write(_ event: AnchorEventKind, id: UUID, name: String, t: TimeInterval, transform: simd_float4x4) {
        guard let lines else { return }
        let m = Mat4(transform)
        guard m.isFinite else { return }
        do {
            try lines.append(AnchorEvent(t: t, id: id.uuidString, name: name, event: event, transform: m))
            lastLogged[id] = transform
        } catch {
            log.error("anchors.jsonl line: \(error.localizedDescription)")
        }
    }
}
