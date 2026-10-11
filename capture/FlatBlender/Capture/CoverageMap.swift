import ARKit
import Foundation
import simd

/// Live capture coverage: which surfaces still lack views from distinct directions that the user could have
/// taken from where they have already been (plan P6 b1; prototyped offline on scan 3).
///
/// - **Seen** (per 8 cm world voxel, a 32-bit mask over 32 Fibonacci-sphere direction bins): at ~5 Hz while
///   recording, the frame's LiDAR depth map (256x192, every 2nd pixel) is unprojected; a pixel counts when it is
///   0.3-4 m away, confidence medium or high, not at a grazing angle (> 15 deg from the surface, normal from the
///   depth neighbours), and the frame is sharp (rotational blur = angular rate x exposure x focal < 6 px). It sets
///   the bit of the direction surface -> camera in its voxel (2+ pixels of a frame per voxel and bin).
/// - **Needed**: every 0.25 m the camera moves, that position becomes a viewpoint; 2 s and 30 s later ~5.4k rays
///   are cast from it through the occupied voxels (3 m, 6 cm steps). The first voxel a ray hits gets the bit of
///   the direction voxel -> viewpoint, if that direction is not grazing for the voxel's surface.
/// - **Deficit** of a voxel (its own masks OR its 6 neighbours', restricted to directions within 75 deg of its
///   normal): `min(needed-but-not-seen bins, 3 - seen bins)`, 0 for ceilings. 0 = fine; 1-3 = how many more
///   directions it wants, and could get from where the user has walked.
/// - **Free space** (0.3.0 (7), scan 4): a voxel that later depth frames see *through* (the measured surface is
///   > max(12 cm, 5 %) behind it, twice, with no new points in it) is dropped: no dot, rays pass through it, no
///   mini-map cell. Without it, depth integrated with a bad pose (ARKit reports `.normal` a few frames before
///   the relocalisation snap lands: 31 cm on scan 4), long-range LiDAR bias and things that moved left surfaces
///   floating in front of the real ones; they could never be seen again, so their dots never cleared (half the
///   dots in view on scan 4, a third of them asking for the direction the camera was in). Seeing points in it
///   again revives it.
/// - **Settling**: nothing is taken (depth or viewpoints) until tracking has been normal for 1 s without a pose
///   jump (> 10 cm between consecutive frames).
///
/// The overlay shows the deficit as small world-anchored dots, one per voxel; a dot that clears (deficit 0) turns
/// green for 1.5 s, so the user sees that looking from the new angle worked. The mini-map shows the deficit per
/// 20 cm floor cell for the walls around the user.
///
/// Threading: `offer` runs on the ARSession delegate queue (cheap: a throttle check, and every 0.2 s a copy of the
/// depth + confidence maps); everything else runs on the private serial `queue`. All voxel state is confined to
/// that queue; the small shared state for `offer` is behind `lock`.
final class CoverageMap: @unchecked Sendable {
    // MARK: Parameters (same as the offline prototype)

    static let voxel: Float = 0.08
    static let binCount = 32
    static let minDistance: Float = 0.3
    static let maxDistance: Float = 4.0
    static let cosGrazing: Float = cos(75 * .pi / 180)
    static let maxBlurPixels = 6.0
    static let interval: TimeInterval = 0.2  // depth frames at 5 Hz
    static let pixelStride = 2
    static let minPixels: UInt8 = 2
    static let targetBins = 3
    static let viewpointStep: Float = 0.25
    static let needRange: Float = 3.0
    static let rayStep: Float = 0.06
    static let castDelays: [TimeInterval] = [2, 30]
    /// Viewpoints cast per depth frame at most (each ~5.4k rays); the rest wait for the next frames. Scan 5's one
    /// 137 ms coverage job was several casts coming due together (both passes, after a fast walk).
    static let maxCastsPerJob = 2
    static let maxVoxels = 400_000  // ~25 MB worst case; a flat is ~100k
    static let dotRange: Float = 3.5
    static let maxDots = 8192
    static let settle: TimeInterval = 1.0
    static let jump: Float = 0.1  // m between consecutive frames: a relocalisation snap, not motion
    static let carveFrames: UInt8 = 2
    static let clearedFor: TimeInterval = 1.5

    static let bins: [SIMD3<Float>] = (0..<binCount).map { k in
        let i = Float(k) + 0.5
        let phi = acos(1 - 2 * i / Float(binCount))
        let theta = Float.pi * (1 + sqrt(5)) * i
        return SIMD3(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi))  // y up (ARKit world)
    }

    /// Roughly even directions from -70 to +55 deg elevation (people look down at furniture more than up).
    static let rays: [SIMD3<Float>] = {
        var out: [SIMD3<Float>] = []
        let rings = 56
        for r in 0..<rings {
            let el = (-70 + 125 * Float(r) / Float(rings - 1)) * .pi / 180
            let n = max(1, Int((120 * cos(el)).rounded()))
            let offset = Float(r % 2) * 0.5
            for j in 0..<n {
                let az = (Float(j) + offset) * 2 * .pi / Float(n)
                out.append(SIMD3(cos(el) * cos(az), sin(el), cos(el) * sin(az)))
            }
        }
        return out
    }()

    // MARK: Shared with the delegate queue (lock)

    private let lock = NSLock()
    private var busy = false
    private var lastTake: TimeInterval = -.infinity
    private var recent: [(t: TimeInterval, q: simd_quatf)] = []
    private var latestPose: simd_float4x4?
    private var lastViewpoint: SIMD3<Float>?
    private var lastOffer: (t: TimeInterval, position: SIMD3<Float>)?
    private var settledFrom: TimeInterval?
    private var version = 0

    // MARK: Queue-confined state

    private let queue = DispatchQueue(label: "flatblender.coverage", qos: .utility)
    private var index: [Int64: Int32] = [:]
    private var seen: [UInt32] = []
    private var need: [UInt32] = []
    private var normalSum: [SIMD3<Float>] = []
    private var pointSum: [SIMD3<Float>] = []
    private var pointCount: [UInt32] = []
    /// Frames in a row that saw through the voxel (free space); dropped at `carveFrames`.
    private var through: [UInt8] = []
    private struct Viewpoint {
        var due: TimeInterval
        var position: SIMD3<Float>
        var pass: Int
    }
    private var pending: [Viewpoint] = []
    private var path: [SIMD3<Float>] = []
    private var lastDotsVersion = -1
    private var lastDotsPose: simd_float4x4?
    private var shownLevels: [Int64: Int] = [:]  // voxel key -> level drawn at the last refresh
    private var cleared: [Int64: (p: SIMD3<Float>, n: SIMD3<Float>, until: TimeInterval)] = [:]
    /// Scratch for `ingest`, kept between depth frames (no 12k-entry dictionaries allocated 5 times a second).
    private var scratchCodes: [Int64: UInt8] = [:]
    private var scratchSums: [Int64: PointSum] = [:]

    private struct PointSum {
        var p = SIMD3<Float>.zero
        var n = SIMD3<Float>.zero
        var c: UInt32 = 0
        mutating func add(_ p: SIMD3<Float>, _ n: SIMD3<Float>) {
            self.p += p
            self.n += n
            c += 1
        }
    }

    /// Times each depth frame's job (`CaptureTimings.coverage`); nothing else.
    private let timings: CaptureTimings?

    init(timings: CaptureTimings? = nil) {
        self.timings = timings
        index.reserveCapacity(65_536)
    }

    // MARK: Input (ARSession delegate queue)

    /// Offers a recorded frame. Takes its depth at most every 0.2 s, and never while the previous one is still
    /// being processed (so the work is bounded whatever the frame rate). Nothing is taken until tracking has
    /// settled (`settle` s of normal tracking without a pose jump).
    func offer(_ frame: ARFrame) {
        let t = frame.timestamp
        let pose = frame.camera.transform
        let position = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
        guard case .normal = frame.camera.trackingState else {
            lock.withLock { settledFrom = nil }
            return
        }
        let q = simd_quatf(pose)
        let (take, rate, newViewpoint): (Bool, Double, Bool) = lock.withLock {
            latestPose = pose
            if let last = lastOffer, t - last.t > 0.5 || simd_distance(last.position, position) > Self.jump { settledFrom = nil }
            lastOffer = (t, position)
            if settledFrom == nil { settledFrom = t }
            guard let since = settledFrom, t - since >= Self.settle else { return (false, 0, false) }
            recent.append((t, q))
            if recent.count > 8 { recent.removeFirst(recent.count - 8) }
            var moved = false
            if lastViewpoint.map({ simd_distance($0, position) >= Self.viewpointStep }) ?? true {
                lastViewpoint = position
                moved = true
            }
            guard !busy, t - lastTake >= Self.interval else { return (false, 0, moved) }
            busy = true
            lastTake = t
            // Angular rate over the last ~4 frames (like the prototype's +-2 frames).
            let old = recent.first { t - $0.t <= 0.075 } ?? recent[0]
            let dt = t - old.t
            let rate = dt > 0.001 ? Double(Self.angle(old.q, q)) / dt : 0
            return (true, rate, moved)
        }
        if newViewpoint {
            queue.async { self.pending.append(Viewpoint(due: t + Self.castDelays[0], position: position, pass: 0)) }
        }
        guard take else { return }
        guard let job = DepthJob(frame: frame, angularRate: rate) else {
            lock.withLock { busy = false }
            return
        }
        queue.async {
            let work = {
                self.ingest(job)
                self.castDue(now: t)
            }
            if let timings = self.timings { timings.measure(timings.coverage, "coverage", work) } else { work() }
            self.lock.withLock {
                self.busy = false
                self.version += 1
            }
        }
    }

    /// Forgets everything (new ARSession world).
    func reset() {
        lock.withLock {
            latestPose = nil
            lastViewpoint = nil
            lastOffer = nil
            settledFrom = nil
            recent = []
            version += 1
        }
        queue.async {
            self.index.removeAll(keepingCapacity: true)
            self.seen = []
            self.need = []
            self.normalSum = []
            self.pointSum = []
            self.pointCount = []
            self.through = []
            self.pending = []
            self.path = []
            self.shownLevels = [:]
            self.cleared = [:]
        }
    }

    // MARK: Seen

    private struct DepthJob {
        let depth: [Float32]
        let confidence: [UInt8]?
        let width: Int
        let height: Int
        let fx: Float, fy: Float, cx: Float, cy: Float  // at depth-map resolution
        let pose: simd_float4x4
        let blurPixels: Double

        init?(frame: ARFrame, angularRate: Double) {
            guard let data = frame.sceneDepth ?? frame.smoothedSceneDepth,
                let depth = Self.copy(data.depthMap, as: Float32.self)
            else { return nil }
            width = CVPixelBufferGetWidth(data.depthMap)
            height = CVPixelBufferGetHeight(data.depthMap)
            self.depth = depth
            confidence = data.confidenceMap.flatMap { Self.copy($0, as: UInt8.self) }
            let cam = frame.camera
            let scale = Float(width) / Float(cam.imageResolution.width)
            let k = cam.intrinsics
            fx = k[0][0] * scale
            fy = k[1][1] * scale
            cx = k[2][0] * scale
            cy = k[2][1] * scale
            pose = cam.transform
            blurPixels = angularRate * cam.exposureDuration * Double(k[0][0])
        }

        static func copy<T>(_ buffer: CVPixelBuffer, as _: T.Type) -> [T]? {
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
            let w = CVPixelBufferGetWidth(buffer)
            let h = CVPixelBufferGetHeight(buffer)
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            guard rowBytes >= w * MemoryLayout<T>.size else { return nil }
            return [T](unsafeUninitializedCapacity: w * h) { out, count in
                for y in 0..<h {
                    let row = (base + y * rowBytes).assumingMemoryBound(to: T.self)
                    for x in 0..<w { out[y * w + x] = row[x] }
                }
                count = w * h
            }
        }
    }

    private func ingest(_ job: DepthJob) {
        guard job.blurPixels <= Self.maxBlurPixels else { return }
        let w = job.width, h = job.height
        let rot = simd_float3x3(
            SIMD3(job.pose.columns.0.x, job.pose.columns.0.y, job.pose.columns.0.z),
            SIMD3(job.pose.columns.1.x, job.pose.columns.1.y, job.pose.columns.1.z),
            SIMD3(job.pose.columns.2.x, job.pose.columns.2.y, job.pose.columns.2.z))
        let origin = SIMD3(job.pose.columns.3.x, job.pose.columns.3.y, job.pose.columns.3.z)

        @inline(__always) func point(_ x: Int, _ y: Int) -> SIMD3<Float> {
            let d = job.depth[y * w + x]
            return SIMD3((Float(x) - job.cx) / job.fx * d, -(Float(y) - job.cy) / job.fy * d, -d)
        }

        carve(job, rot: rot, origin: origin)

        var codes = scratchCodes
        var sums = scratchSums
        scratchCodes = [:]  // unique storage: removeAll(keepingCapacity:) then clears in place, no copy
        scratchSums = [:]
        codes.removeAll(keepingCapacity: true)
        sums.removeAll(keepingCapacity: true)
        if codes.capacity < 8192 { codes.reserveCapacity(8192) }
        if sums.capacity < 4096 { sums.reserveCapacity(4096) }
        defer {
            scratchCodes = codes
            scratchSums = sums
        }
        var y = 1
        while y < h - 1 {
            var x = 1
            while x < w - 1 {
                defer { x += Self.pixelStride }
                let d = job.depth[y * w + x]
                guard d > Self.minDistance, d < Self.maxDistance, d.isFinite else { continue }
                if let conf = job.confidence, conf[y * w + x] < 1 { continue }
                // Normal from the 4 neighbours: all of them need depth.
                guard job.depth[y * w + x - 1] > 0, job.depth[y * w + x + 1] > 0, job.depth[(y - 1) * w + x] > 0,
                    job.depth[(y + 1) * w + x] > 0
                else { continue }
                let p = point(x, y)
                var n = simd_cross(point(x + 1, y) - point(x - 1, y), point(x, y + 1) - point(x, y - 1))
                let len = simd_length(n)
                guard len > 1e-9, len.isFinite else { continue }
                n /= len
                let view = -p / simd_length(p)  // point -> camera
                var cosv = simd_dot(n, view)
                if cosv < 0 {
                    n = -n
                    cosv = -cosv
                }
                guard cosv > Self.cosGrazing else { continue }
                let pw = rot * p + origin
                let vw = rot * view
                let nw = rot * n
                let key = Self.key(pw)
                let code = key << 5 | Int64(Self.bin(vw))
                codes[code, default: 0] &+= 1
                sums[key, default: PointSum()].add(pw, nw)
            }
            y += Self.pixelStride
        }
        for (code, count) in codes where count >= Self.minPixels {
            guard let i = voxel(code >> 5, create: true) else { continue }
            seen[i] |= 1 << UInt32(code & 31)
            through[i] = 0  // points in it again: a real surface
        }
        for (key, s) in sums {
            guard let i = voxel(key, create: false), pointCount[i] < 4096 else { continue }
            pointSum[i] += s.p
            normalSum[i] += s.n
            pointCount[i] += s.c
        }
    }

    /// Free space: voxels in this depth frame's view whose surface the frame measures well *behind* them (it sees
    /// through them) count one more strike; voxels it measures on the surface are cleared. One pass over all
    /// voxels (~100k for a flat: a few ms at 5 Hz).
    private func carve(_ job: DepthJob, rot: simd_float3x3, origin: SIMD3<Float>) {
        let w = job.width, h = job.height
        let toCamera = rot.transpose
        let maxSq = Self.maxDistance * Self.maxDistance
        for i in 0..<pointCount.count where pointCount[i] > 0 {
            let rel = pointSum[i] / Float(pointCount[i]) - origin
            let distSq = simd_length_squared(rel)
            guard distSq < maxSq, distSq > Self.minDistance * Self.minDistance else { continue }
            let c = toCamera * rel
            let z = -c.z
            guard z > Self.minDistance else { continue }
            let u = job.cx + job.fx * c.x / z
            let v = job.cy - job.fy * c.y / z
            guard u >= 0, v >= 0, u < Float(w), v < Float(h) else { continue }
            let px = Int(v) * w + Int(u)
            let d = job.depth[px]
            guard d > 0, d.isFinite else { continue }
            if let conf = job.confidence, conf[px] < 1 { continue }
            let margin = max(0.12, 0.05 * z)
            if d > z + margin {
                if through[i] < .max { through[i] += 1 }
            } else if abs(d - z) <= margin {
                through[i] = 0
            }
        }
    }

    @inline(__always) private func alive(_ i: Int) -> Bool {
        pointCount[i] > 0 && through[i] < Self.carveFrames
    }

    private func voxel(_ key: Int64, create: Bool) -> Int? {
        if let i = index[key] { return Int(i) }
        guard create, seen.count < Self.maxVoxels else { return nil }
        let i = seen.count
        index[key] = Int32(i)
        seen.append(0)
        need.append(0)
        normalSum.append(.zero)
        pointSum.append(.zero)
        pointCount.append(0)
        through.append(0)
        return i
    }

    // MARK: Needed

    private func castDue(now: TimeInterval) {
        var keep: [Viewpoint] = []
        var casts = 0
        for item in pending {
            guard item.due <= now, casts < Self.maxCastsPerJob else {
                keep.append(item)
                continue
            }
            casts += 1
            cast(from: item.position)
            if item.pass == 0 {
                path.append(item.position)
                if path.count > 8000 { path.removeFirst(2000) }
            }
            // Cast again later, once the surroundings have also been seen from elsewhere.
            let next = item.pass + 1
            if next < Self.castDelays.count {
                keep.append(Viewpoint(due: item.due - Self.castDelays[item.pass] + Self.castDelays[next], position: item.position, pass: next))
            }
        }
        pending = keep
    }

    private func cast(from position: SIMD3<Float>) {
        let steps = Int(Self.needRange / Self.rayStep)
        for dir in Self.rays {
            var lastKey = Int64.min
            for s in 1...steps {
                let q = position + dir * (Float(s) * Self.rayStep)
                let key = Self.key(q)
                if key == lastKey { continue }
                lastKey = key
                guard let i32 = index[key] else { continue }
                let i = Int(i32)
                guard through[i] < Self.carveFrames else { continue }  // free space: the ray goes on
                guard pointCount[i] > 0 else { break }
                let n = simd_normalize(normalSum[i])
                var toCam = position - pointSum[i] / Float(pointCount[i])
                let len = simd_length(toCam)
                guard len > 1e-3 else { break }
                toCam /= len
                if simd_dot(n, toCam) > Self.cosGrazing {
                    need[i] |= 1 << UInt32(Self.bin(toCam))
                }
                break
            }
        }
    }

    // MARK: Output

    /// Vertices of the dot overlay (7 per dot: centre + hexagon), grouped by deficit level 1, 2, 3, then the dots
    /// that just cleared (shown green for `clearedFor` s).
    struct DotMesh: Sendable {
        var positions: [SIMD3<Float>]
        var levelCounts: [Int]  // dots per level 1...3, then cleared, in this order in `positions`
        var boundsMin: SIMD3<Float>
        var boundsMax: SIMD3<Float>
        var dotCount: Int { levelCounts.reduce(0, +) }
    }

    /// Dots near the latest camera pose; nil when nothing changed since the last call (no new data, camera moved
    /// < 10 cm, no cleared dot fading out), so the caller can skip the mesh update.
    func dotMesh() async -> DotMesh? {
        let (pose, version) = lock.withLock { (latestPose, self.version) }
        guard let pose else { return nil }
        return await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: self.makeDots(pose: pose, version: version)) }
        }
    }

    private func makeDots(pose: simd_float4x4, version: Int) -> DotMesh? {
        let cam = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
        let now = ProcessInfo.processInfo.systemUptime
        cleared = cleared.filter { $0.value.until > now }
        if version == lastDotsVersion, cleared.isEmpty, let last = lastDotsPose,
            simd_distance(SIMD3(last.columns.3.x, last.columns.3.y, last.columns.3.z), cam) < 0.1
        {
            return nil
        }
        lastDotsVersion = version
        lastDotsPose = pose

        struct Candidate {
            var key: Int64
            var p: SIMD3<Float>
            var n: SIMD3<Float>
            var d: Float
            var level: Int  // 1...3; 4 = just cleared (green)
        }
        var candidates: [Candidate] = []
        for (key, i32) in index {
            let i = Int(i32)
            guard need[i] != 0, alive(i) else { continue }
            let p = pointSum[i] / Float(pointCount[i])
            let d = simd_distance(p, cam)
            guard d < Self.dotRange, d > Self.minDistance else { continue }
            let n = simd_normalize(normalSum[i])
            let level = deficit(key: key, normal: n)
            if level > 0 { candidates.append(Candidate(key: key, p: p, n: n, d: d, level: level)) }
        }
        if candidates.count > Self.maxDots {
            candidates.sort { $0.d < $1.d }
            candidates.removeLast(candidates.count - Self.maxDots)
        }

        // Dots drawn last time that are now covered (deficit 0; not merely out of range or dropped as free space)
        // turn green for a moment: the user sees that the new angle worked.
        let current = Set(candidates.map(\.key))
        for (key, _) in shownLevels where !current.contains(key) {
            guard let i32 = index[key], alive(Int(i32)) else { continue }
            let i = Int(i32)
            let p = pointSum[i] / Float(pointCount[i])
            let d = simd_distance(p, cam)
            guard d < Self.dotRange, d > Self.minDistance else { continue }
            let n = simd_normalize(normalSum[i])
            if deficit(key: key, normal: n) == 0 { cleared[key] = (p, n, now + Self.clearedFor) }
        }
        for key in current { cleared[key] = nil }
        for (key, c) in cleared where candidates.count < Self.maxDots {
            candidates.append(Candidate(key: key, p: c.p, n: c.n, d: simd_distance(c.p, cam), level: 4))
        }
        candidates.sort { $0.level < $1.level }

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(candidates.count * 7)
        var counts = [0, 0, 0, 0]
        var shown: [Int64: Int] = [:]
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        let baseRadius: [Float] = [0.012, 0.017, 0.022, 0.017]
        for c in candidates {
            // Size encodes the deficit; shrink close to the lens (no giant dots) and fade out at the range limit.
            var r = baseRadius[c.level - 1] * min(1, max(0.4, c.d / 1.2))
            r *= Self.smoothstep(Self.dotRange, Self.dotRange - 0.7, c.d) * Self.smoothstep(0.3, 0.6, c.d)
            guard r > 0.002 else { continue }
            if c.level <= 3 { shown[c.key] = c.level }
            let n = simd_dot(cam - c.p, c.n) < 0 ? -c.n : c.n
            let a = simd_normalize(abs(n.y) < 0.9 ? simd_cross(n, SIMD3(0, 1, 0)) : simd_cross(n, SIMD3(1, 0, 0)))
            let b = simd_cross(n, a)
            let centre = c.p + n * 0.015  // just in front of the reconstructed surface
            positions.append(centre)
            for k in 0..<6 {
                let ang = Float(k) * .pi / 3
                positions.append(centre + r * (cos(ang) * a + sin(ang) * b))
            }
            lo = simd_min(lo, centre - r)
            hi = simd_max(hi, centre + r)
            counts[c.level - 1] += 1
        }
        shownLevels = shown
        if positions.isEmpty { lo = .zero; hi = .zero }
        return DotMesh(positions: positions, levelCounts: counts, boundsMin: lo, boundsMax: hi)
    }

    /// 0...3: how many more distinct directions this voxel wants and could get from where the user has been.
    private func deficit(key: Int64, normal n: SIMD3<Float>) -> Int {
        guard n.y > -0.7 else { return 0 }  // ceilings are never asked for
        var s: UInt32 = 0, m: UInt32 = 0
        for off in Self.neighbours {
            if let j = index[key &+ off], alive(Int(j)) {
                s |= seen[Int(j)]
                m |= need[Int(j)]
            }
        }
        let hemi = Self.hemisphere(n)
        let seenBins = (s & hemi).nonzeroBitCount
        let missing = (m & ~s & hemi).nonzeroBitCount
        return min(missing, max(0, Self.targetBins - seenBins))
    }

    /// Top-down state around the camera for the mini-map, in camera-relative metres (x = right, y = forward).
    struct MapSnapshot: Sendable {
        struct Cell: Sendable {
            var x: Float
            var y: Float
            var flagged: Float  // 0 = covered; else share of the cell's wall voxels that want 2+ more directions
        }
        var cells: [Cell]
        var path: [SIMD2<Float>]
        var cellSize: Float
        var extent: Float
    }

    func mapSnapshot(extent: Float = 8, cellSize: Float = 0.2) async -> MapSnapshot? {
        guard let pose = lock.withLock({ latestPose }) else { return nil }
        return await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: self.makeMap(pose: pose, extent: extent, cellSize: cellSize)) }
        }
    }

    private func makeMap(pose: simd_float4x4, extent: Float, cellSize: Float) -> MapSnapshot {
        let cam = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
        var fwd = -SIMD2(pose.columns.2.x, pose.columns.2.z)
        fwd = simd_length(fwd) > 1e-3 ? simd_normalize(fwd) : SIMD2(0, -1)
        let right = SIMD2(-fwd.y, fwd.x)  // f x up, in (x, z)
        let reach = extent * 0.75
        var cells: [Int64: (count: Int, bad: Int)] = [:]
        for (key, i32) in index {
            let i = Int(i32)
            guard alive(i) else { continue }
            let p = pointSum[i] / Float(pointCount[i])
            guard abs(p.x - cam.x) < reach, abs(p.z - cam.z) < reach, p.y > cam.y - 1.4, p.y < cam.y + 0.9 else { continue }
            let n = simd_normalize(normalSum[i])
            guard abs(n.y) < 0.6 else { continue }  // walls and furniture sides
            let bad = need[i] != 0 && deficit(key: key, normal: n) >= 2
            let cx = Int64(floor(p.x / cellSize)), cz = Int64(floor(p.z / cellSize))
            let ck = (cx + (1 << 20)) << 21 | (cz + (1 << 20))
            cells[ck, default: (0, 0)].count += 1
            if bad { cells[ck, default: (0, 0)].bad += 1 }
        }
        func local(_ x: Float, _ z: Float) -> SIMD2<Float> {
            let d = SIMD2(x - cam.x, z - cam.z)
            return SIMD2(simd_dot(d, right), simd_dot(d, fwd))
        }
        var out: [MapSnapshot.Cell] = []
        for (ck, v) in cells where v.count >= 2 {
            let cx = Float((ck >> 21) - (1 << 20)), cz = Float((ck & ((1 << 21) - 1)) - (1 << 20))
            let l = local((cx + 0.5) * cellSize, (cz + 0.5) * cellSize)
            guard abs(l.x) < extent / 2, abs(l.y) < extent / 2 else { continue }
            let share = Float(v.bad) / Float(v.count)
            out.append(.init(x: l.x, y: l.y, flagged: share >= 0.25 ? share : 0))
        }
        let trail = path.suffix(2000).map { local($0.x, $0.z) }.filter { abs($0.x) < extent / 2 && abs($0.y) < extent / 2 }
        return MapSnapshot(cells: out, path: trail, cellSize: cellSize, extent: extent)
    }

    // MARK: Helpers

    private static let neighbours: [Int64] = [
        0, 1 << 32, -(1 << 32), 1 << 16, -(1 << 16), 1, -1,
    ]

    /// Voxel key: 16 bits per axis (offset 2^15, i.e. +-2.6 km at 8 cm).
    @inline(__always) static func key(_ p: SIMD3<Float>) -> Int64 {
        let q = SIMD3<Int64>(Int64(floor(p.x / voxel)), Int64(floor(p.y / voxel)), Int64(floor(p.z / voxel))) &+ (1 << 15)
        return (q.x & 0xFFFF) << 32 | (q.y & 0xFFFF) << 16 | (q.z & 0xFFFF)
    }

    @inline(__always) static func bin(_ dir: SIMD3<Float>) -> Int {
        var best = 0
        var bestDot = -Float.infinity
        for (k, b) in bins.enumerated() {
            let d = simd_dot(b, dir)
            if d > bestDot {
                bestDot = d
                best = k
            }
        }
        return best
    }

    /// Bins within 75 deg of the normal.
    static func hemisphere(_ n: SIMD3<Float>) -> UInt32 {
        var m: UInt32 = 0
        for (k, b) in bins.enumerated() where simd_dot(b, n) > cosGrazing { m |= 1 << UInt32(k) }
        return m
    }

    static func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
        let t = min(1, max(0, (x - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }

    static func angle(_ a: simd_quatf, _ b: simd_quatf) -> Float {
        2 * acos(min(1, abs(simd_dot(a.vector, b.vector))))
    }
}
