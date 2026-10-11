import BundleKit
import Foundation
import os

/// Per-stage timing of our own capture work, for the session notes ("timing …") and, as `OSSignposter`
/// intervals (subsystem "FlatBlender", category Points of Interest), for Instruments. One instance lives in
/// `FrameSink` for the whole ARSession; `reset()` at each recording start, so each session's notes describe
/// that session.
///
/// ARKit gaps: a timestamp step between two delegate callbacks of > 1.5 frame intervals is a frame ARKit never
/// delivered (camera or ARKit itself dropped it, e.g. because the delegate held its buffers too long). Our own
/// drops (`StreamRecorder.reserve` refusing a frame) are counted separately by the recorder: they leave gaps in
/// frames.jsonl, not in the delegate's timestamps.
final class CaptureTimings: @unchecked Sendable {
    static let signposter = OSSignposter(subsystem: "FlatBlender", category: .pointsOfInterest)

    /// Delegate callbacks over this count as slow (at 60 fps a frame lasts 16.7 ms; ARKit has few buffers).
    static let slowDelegateS = 0.008

    let delegate = StageTimer(overS: slowDelegateS)  // FrameSink.session(_:didUpdate:), forward included
    let roomPlanForward = StageTimer()  // the forward to RoomPlan's delegate inside it
    let stream = StageTimer()  // StreamRecorder: video append + job bookkeeping, per frame
    let depthPNG = StageTimer()  // depth + confidence PNG encode and write, per depth frame
    let coverage = StageTimer()  // CoverageMap: one depth frame's integration + carving + ray casts
    let stillJPEG = StageTimer()  // StillRecorder: JPEG encode + write, per still

    private let lock = NSLock()
    private var lastT: TimeInterval?
    private var gaps = 0  // timestamp steps > 1.5 frame intervals
    private var missing = 0  // frames those gaps skipped

    /// One delegate frame while recording. `fps`: the video format's nominal rate.
    func noteFrame(t: TimeInterval, fps: Double) {
        lock.withLock {
            defer { lastT = t }
            guard let last = lastT, fps > 0, t > last else { return }
            let steps = (t - last) * fps
            if steps > 1.5 {
                gaps += 1
                missing += Int(steps.rounded()) - 1
            }
        }
    }

    func reset() {
        for t in [delegate, roomPlanForward, stream, depthPNG, coverage, stillJPEG] { t.reset() }
        lock.withLock {
            lastT = nil
            gaps = 0
            missing = 0
        }
    }

    /// Runs `body`, recording its duration in `timer` and as a signpost interval named `name`.
    @inline(__always)
    func measure<T>(_ timer: StageTimer, _ name: StaticString, _ body: () throws -> T) rethrows -> T {
        let state = Self.signposter.beginInterval(name)
        let start = DispatchTime.now().uptimeNanoseconds
        defer {
            timer.record(seconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            Self.signposter.endInterval(name, state)
        }
        return try body()
    }

    /// The compact "timing …" note, e.g. `timing delegate p50 0.9 p99 3.1 max 12.4 ms (n 18476, >8 ms 3),
    /// roomplan fwd p50 0.3 max 6.0 ms, stream max 9.8 ms, depth png max 7.1 ms, coverage max 31 ms, still jpeg
    /// max 140 ms, arkit gaps 4 (5 frames), our drops 0`.
    func note(ourDrops: Int) -> String {
        func ms(_ s: Double?) -> String {
            guard let s else { return "-" }
            let v = s * 1000
            return v >= 20 ? String(format: "%.0f", v) : String(format: "%.1f", v)
        }
        let d = delegate.summary
        var parts = [
            "delegate p50 \(ms(d.p50S)) p99 \(ms(d.p99S)) max \(ms(d.maxS)) ms (n \(d.count), >8 ms \(d.over))"
        ]
        let f = roomPlanForward.summary
        if f.count > 0 { parts.append("roomplan fwd p50 \(ms(f.p50S)) max \(ms(f.maxS)) ms") }
        for (label, timer) in [
            ("stream", stream), ("depth png", depthPNG), ("coverage", coverage), ("still jpeg", stillJPEG),
        ] {
            let s = timer.summary
            if s.count > 0 { parts.append("\(label) max \(ms(s.maxS)) ms") }
        }
        let (g, m) = lock.withLock { (gaps, missing) }
        parts.append("arkit gaps \(g) (\(m) frame\(m == 1 ? "" : "s"))")
        parts.append("our drops \(ourDrops)")
        return "timing " + parts.joined(separator: ", ")
    }
}
