import AVFoundation
import BundleKit
import CoreVideo
import Foundation
import os

/// Records the ARKit stream: `video.mov` (HEVC), one `frames.jsonl` line per video frame, and depth +
/// confidence PNGs for every Nth frame.
///
/// Threading: `reserve` / `submit` are called on the ARKit delegate queue with data already copied out of
/// the ARFrame. The video append runs on `queue` (serial); everything after it (depth + confidence PNGs and the
/// frames.jsonl line) on `linesQueue` (serial), in the order `queue` hands the frames over, so frames.jsonl is
/// in `idx` order and each line names its PNGs only if they were written. The video append (which frees the
/// frame's ~4 MB YUV copy and its `maxPending` slot) never waits for PNG encoding or file creation: on scan 4
/// (0.3.0 (6), PNGs on the video queue) all 19 one-frame gaps in frames.jsonl came 3 frames after a depth frame,
/// refused by `reserve` while a depth job stalled the queue. Counters shared between the queues live behind
/// `lock`. Queue-confined state is only touched on its queue.
final class StreamRecorder: @unchecked Sendable {
    struct FrameJob: @unchecked Sendable {  // CVPixelBuffer is not Sendable; ownership moves to `queue`
        let t: TimeInterval
        let cameraToWorld: Mat4
        let intrinsics: Intrinsics
        let tracking: BundleKit.TrackingState
        let exposure: Exposure
        let pixels: CVPixelBuffer  // our own copy, not an ARKit pool buffer
        let depth: DepthCopy?
    }

    struct Stats: Sendable {
        var frames = 0  // lines in frames.jsonl == frames in video.mov
        var depthFrames = 0
        var dropped = 0
        var pending = 0
        var failure: String?
    }

    struct Result: Sendable {
        var frames: Int
        var video: VideoFormat?
        var depth: DepthFormat?
        var failure: String?
    }

    /// Frames copied but not yet appended to the video. Each holds one ~4.1 MB YUV copy (1920×1440 4:2:0),
    /// so 8 is ≤ 33 MB; beyond this we drop frames (no frames.jsonl line, so `idx` stays equal to the frame's
    /// index in video.mov). Was 4 until 0.3.0 (7).
    static let maxPending = 8

    let dir: URL
    let depthEvery: Int
    let nominalFPS: Double
    let smoothedDepth: Bool
    private let timings: CaptureTimings?

    private let queue = DispatchQueue(label: "flatblender.stream", qos: .userInitiated)
    private let linesQueue = DispatchQueue(label: "flatblender.stream.lines", qos: .userInitiated)
    private let log = Logger(subsystem: "FlatBlender", category: "stream")

    // Shared, behind `lock`.
    private let lock = NSLock()
    private var stats = Stats()
    private var acceptingFrames = true
    private var framesSinceDepth: Int
    private var recent: [(t: TimeInterval, idx: Int)] = []  // last ~10 s, for still -> nearest_frame

    // Confined to `queue`.
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var size: (width: Int, height: Int)?
    private var nextIdx = 0
    private var lastT = -Double.infinity
    private var finalizing = false

    // Confined to `linesQueue`.
    private var frameLines: JSONLWriter?
    private var depthSize: (width: Int, height: Int)?

    init(dir: URL, depthEvery: Int, nominalFPS: Double, smoothedDepth: Bool, timings: CaptureTimings? = nil) {
        self.dir = dir
        self.depthEvery = max(1, depthEvery)
        self.nominalFPS = nominalFPS
        self.smoothedDepth = smoothedDepth
        self.timings = timings
        framesSinceDepth = max(1, depthEvery)  // the first frame carries depth
    }

    // MARK: Delegate-queue side

    /// Reserves a slot for one frame. Returns nil if the frame must be dropped; otherwise whether this frame
    /// should carry depth. Must be balanced by `submit` or `cancel`.
    func reserve() -> Bool? {
        lock.withLock {
            guard acceptingFrames, stats.failure == nil else { return nil }
            guard stats.pending < Self.maxPending else {
                stats.dropped += 1
                return nil
            }
            stats.pending += 1
            return framesSinceDepth >= depthEvery
        }
    }

    func cancel() {
        lock.withLock {
            stats.pending -= 1
            stats.dropped += 1
        }
    }

    func submit(_ job: FrameJob) {
        lock.withLock { framesSinceDepth = job.depth != nil ? 1 : framesSinceDepth + 1 }
        queue.async { [self] in
            if let timings {
                timings.measure(timings.stream, "stream") { process(job) }
            } else {
                process(job)
            }
            lock.withLock { stats.pending -= 1 }
        }
    }

    var snapshot: Stats { lock.withLock { stats } }

    /// Index of the recorded frame closest in time to `t` (within 1 s), for `Still.nearest_frame`.
    func nearestFrame(to t: TimeInterval) -> Int? {
        lock.withLock {
            guard let best = recent.min(by: { abs($0.t - t) < abs($1.t - t) }), abs(best.t - t) < 1 else { return nil }
            return best.idx
        }
    }

    // MARK: Encoding queue

    /// Appends the frame to the video, then hands its line (and depth) to `linesQueue`.
    private func process(_ job: FrameJob) {
        guard !finalizing else { return dropOne() }
        let w = CVPixelBufferGetWidth(job.pixels)
        let h = CVPixelBufferGetHeight(job.pixels)
        if writer == nil {
            do {
                try startWriter(width: w, height: h, firstT: job.t)
            } catch {
                return fail("could not start video: \(error.localizedDescription)")
            }
        }
        guard let writer, let input, let adaptor, let size else { return dropOne() }
        // A configuration re-run could change the video format mid-recording; keep the file consistent.
        guard size == (w, h), job.t > lastT else { return dropOne() }
        guard writer.status == .writing else {
            return fail("video writer: \(writer.error?.localizedDescription ?? "status \(writer.status.rawValue)")")
        }
        guard input.isReadyForMoreMediaData else { return dropOne() }
        let pts = CMTime(seconds: job.t, preferredTimescale: 1_000_000)
        guard adaptor.append(job.pixels, withPresentationTime: pts) else {
            return fail("video append: \(writer.error?.localizedDescription ?? "unknown error")")
        }
        let idx = nextIdx
        nextIdx += 1
        lastT = job.t
        let line = Line(
            idx: idx, t: job.t, cameraToWorld: job.cameraToWorld, intrinsics: job.intrinsics, tracking: job.tracking,
            exposure: job.exposure, depth: job.depth)
        linesQueue.async { [self] in writeLine(line) }
    }

    /// What `linesQueue` needs of a frame (the pixels stay behind: released once appended).
    private struct Line: Sendable {
        let idx: Int
        let t: TimeInterval
        let cameraToWorld: Mat4
        let intrinsics: Intrinsics
        let tracking: BundleKit.TrackingState
        let exposure: Exposure
        let depth: DepthCopy?
    }

    // MARK: Lines queue

    private func writeLine(_ job: Line) {
        let idx = job.idx
        var depthPath: String?
        var confPath: String?
        if let d = job.depth {
            func encode() {
                do {
                    let png = try DepthPNG.encodeDepth(millimetres: d.millimetres, width: d.width, height: d.height)
                    try DepthPNG.write(png, to: dir.appending(path: BundleLayout.depthPath(frame: idx)))
                    depthPath = BundleLayout.depthPath(frame: idx)
                    if let c = d.confidence {
                        let cpng = try DepthPNG.encodeConfidence(c, width: d.width, height: d.height)
                        try DepthPNG.write(cpng, to: dir.appending(path: BundleLayout.confidencePath(frame: idx)))
                        confPath = BundleLayout.confidencePath(frame: idx)
                    }
                    if depthSize == nil { depthSize = (d.width, d.height) }
                } catch {
                    log.error("depth \(idx): \(error.localizedDescription)")
                }
            }
            if let timings { timings.measure(timings.depthPNG, "depth png", encode) } else { encode() }
        }

        let frame = Frame(
            idx: idx, t: job.t, cameraToWorld: job.cameraToWorld, intrinsics: job.intrinsics,
            tracking: job.tracking, exposure: job.exposure, depth: depthPath, confidence: confPath)
        do {
            if frameLines == nil { frameLines = try JSONLWriter(url: dir.appending(path: BundleLayout.framesFile)) }
            try frameLines?.append(frame)
        } catch {
            // Only possible for a non-finite pose (or an unwritable folder). The frame stays in video.mov; ingest
            // tolerates a gap in idx.
            log.error("frames.jsonl line \(idx): \(error.localizedDescription)")
        }

        lock.withLock {
            stats.frames = idx + 1
            if depthPath != nil { stats.depthFrames += 1 }
            recent.append((job.t, idx))
            if recent.count > 900 { recent.removeFirst(recent.count - 600) }
        }
    }

    private func startWriter(width: Int, height: Int, firstT: TimeInterval) throws {
        let url = dir.appending(path: BundleLayout.videoFile)
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        // Fragments every 2 s: a crash leaves a playable movie up to the last fragment.
        writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)
        // 0.3 bit/px ≈ 50 Mbit/s at 1920x1440@60 (≈ 6.2 MB/s): ~3.7 GB of video per 10 min room, plus the depth
        // PNGs. 0.1 bit/px (16.7 Mbit/s) left the frames ~2.5x softer than the stills, partly from compression.
        let bitsPerPixel = 0.3
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Int(Double(width * height) * nominalFPS * bitsPerPixel),
                AVVideoExpectedSourceFrameRateKey: nominalFPS,
                AVVideoMaxKeyFrameIntervalKey: Int(nominalFPS),
                AVVideoAllowFrameReorderingKey: false,  // decode order == presentation order == idx
            ] as [String: Any],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else { throw RecorderError.cannotAddInput }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? RecorderError.cannotStart }
        writer.startSession(atSourceTime: CMTime(seconds: firstT, preferredTimescale: 1_000_000))
        self.writer = writer
        self.input = input
        self.adaptor = adaptor
        size = (width, height)
    }

    private func dropOne() {
        lock.withLock { stats.dropped += 1 }
    }

    private func fail(_ message: String) {
        log.error("\(message)")
        lock.withLock {
            if stats.failure == nil { stats.failure = message }
        }
    }

    // MARK: Finish

    /// Stops accepting frames, encodes what is queued, finalises video.mov, then writes the remaining lines
    /// and PNGs and closes frames.jsonl.
    func finish() async -> Result {
        lock.withLock { acceptingFrames = false }
        let video = await withCheckedContinuation { (cont: CheckedContinuation<VideoFormat?, Never>) in
            queue.async { [self] in
                finalizing = true
                let format = size.map { VideoFormat(width: $0.width, height: $0.height, fps: nominalFPS, codec: .hevc) }
                guard let writer, let input else {
                    cont.resume(returning: format)
                    return
                }
                input.markAsFinished()
                writer.finishWriting { [self] in
                    queue.async { [self] in
                        if self.writer?.status != .completed {
                            fail("finishing video: \(self.writer?.error?.localizedDescription ?? "unknown error")")
                        }
                        cont.resume(returning: format)
                    }
                }
            }
        }
        // Every line was queued on `linesQueue` before `finalizing` was set, so this runs after all of them.
        return await withCheckedContinuation { (cont: CheckedContinuation<Result, Never>) in
            linesQueue.async { [self] in
                try? frameLines?.close()
                frameLines = nil
                let s = snapshot
                cont.resume(returning: Result(
                    frames: s.frames, video: video,
                    depth: depthSize.map { DepthFormat(width: $0.width, height: $0.height, smoothed: smoothedDepth) },
                    failure: s.failure))
            }
        }
    }
}

enum RecorderError: LocalizedError {
    case cannotAddInput
    case cannotStart

    var errorDescription: String? {
        switch self {
        case .cannotAddInput: "The video writer rejected the HEVC input."
        case .cannotStart: "The video writer could not start."
        }
    }
}
