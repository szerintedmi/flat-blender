import Foundation

/// Durations of one stage of the capture work (ARKit delegate, video append, depth PNGs, …): count, total, max,
/// how many exceeded `overS`, and p50 / p99 from a 0.1 ms histogram up to 100 ms (longer ones count in the
/// last bin; the max is exact). Thread-safe (one lock; `record` is a few adds), so any queue may record.
public final class StageTimer: @unchecked Sendable {
    public struct Summary: Sendable, Equatable {
        public var count = 0
        public var totalS = 0.0
        public var maxS = 0.0
        public var over = 0  // samples > overS
        public var p50S: Double?
        public var p99S: Double?

        public var meanS: Double { count > 0 ? totalS / Double(count) : 0 }
    }

    public static let binS = 0.0001
    public static let bins = 1000

    public let overS: Double?
    private let lock = NSLock()
    private var count = 0
    private var total = 0.0
    private var max = 0.0
    private var over = 0
    private var histogram: [Int]

    public init(overS: Double? = nil) {
        self.overS = overS
        histogram = Array(repeating: 0, count: Self.bins)
    }

    public func record(seconds s: Double) {
        guard s.isFinite, s >= 0 else { return }
        let bin = Swift.min(Self.bins - 1, Int(s / Self.binS))
        lock.withLock {
            count += 1
            total += s
            if s > max { max = s }
            if let overS, s > overS { over += 1 }
            histogram[bin] += 1
        }
    }

    public func reset() {
        lock.withLock {
            count = 0
            total = 0
            max = 0
            over = 0
            for i in histogram.indices { histogram[i] = 0 }
        }
    }

    public var summary: Summary {
        lock.withLock {
            Summary(
                count: count, totalS: total, maxS: max, over: over, p50S: percentile(0.5), p99S: percentile(0.99))
        }
    }

    /// Upper edge of the bin holding the q-quantile (capped at the max); call with the lock held.
    private func percentile(_ q: Double) -> Double? {
        guard count > 0 else { return nil }
        let rank = Swift.max(1, Int((q * Double(count)).rounded(.up)))
        var seen = 0
        for (i, n) in histogram.enumerated() {
            seen += n
            if seen >= rank { return Swift.min(max, Double(i + 1) * Self.binS) }
        }
        return max
    }
}
