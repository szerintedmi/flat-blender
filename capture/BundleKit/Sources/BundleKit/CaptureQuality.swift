import Foundation

// Pure capture-quality math used by the app (camera lock, flicker hint, steadiness cues, auto-still gate).
// Lives here, not in the app, so `swift test` covers it on macOS.

/// How Lock turns the metered auto exposure into a locked one: shorter shutter first, ISO up to a noise-safe
/// ceiling, then a longer shutter again, and only then underexposure.
///
/// Why: on scan 3 the lock was 1/120 s at ISO 284. Hallway turns of 21 / 47 deg/s (p50 / p90) with
/// fx ≈ 1524 px (1920 px wide) smear f·ω·t ≈ 4.6 / 10 px at full resolution, 2.3 / 5 px even at the 960 px
/// the splat trains on. 1/240 s halves that; the same brightness then needs ISO 568.
///
/// ISO ceiling 800 (or the format's max ISO if lower): the iPad Pro's wide camera is a small phone-class
/// sensor (~1 µm photosites; the 1920×1440 stream is binned/downscaled from 12 MP). Its video formats report a
/// max ISO in the low thousands, but above ~800 luma noise and chroma blotches become visible in the stream,
/// HEVC spends its bits on that noise (smearing real texture), and noise is view-inconsistent detail the splat
/// turns into fuzz. 800 is ~1.5 stops above scan 3's
/// lock, enough for 1/240 s in every scan-3 room, and well below the format's max ISO (the session notes
/// record the active format's ISO range, to tune this on device).
///
/// If even ISO 800 at 1/120 s is too dark, we keep 1/120 s and ISO 800 and accept underexposure (recorded as
/// `underexposure_ev`): a darker but sharp, clean image normalises well (the pipeline already matches
/// brightness per image), while blur and noise can't be undone.
///
/// Flicker: lights on 50 Hz mains flicker at 100 Hz (60 Hz mains: 120 Hz), and only shutters that are
/// multiples of 1/100 s (1/120 s) are free of rolling bands. `ShutterMode.flicker50Hz` / `.flicker60Hz` fix the
/// shutter at 1/100 s / 1/120 s instead (same ISO ceiling, then underexposure); only in a room too bright for
/// that at the format's min ISO is the shutter shortened (daylight then dominates, so flicker doesn't matter).
public enum ShutterMode: String, Codable, Sendable, CaseIterable {
    case sharp  // 1/240 s preferred (default)
    case flicker50Hz = "flicker_50hz"  // fixed 1/100 s
    case flicker60Hz = "flicker_60hz"  // fixed 1/120 s

    /// The fixed shutter of a flicker-safe mode (nil: `sharp`).
    public var fixedDurationS: Double? {
        switch self {
        case .sharp: nil
        case .flicker50Hz: 1.0 / 100
        case .flicker60Hz: 1.0 / 120
        }
    }
}

/// Mains frequency whose lighting flicker the metered exposure points to.
public enum FlickerMains: String, Codable, Sendable, CaseIterable {
    case hz50 = "50hz"  // lights flicker at 100 Hz
    case hz60 = "60hz"  // lights flicker at 120 Hz

    /// The flicker-safe shutter mode for these mains.
    public var shutterMode: ShutterMode { self == .hz50 ? .flicker50Hz : .flicker60Hz }
}

/// What happened to the "LED flicker likely" prompt at Lock (`session.json` camera `flicker_prompt`).
public enum FlickerPromptOutcome: String, Codable, Sendable, CaseIterable {
    case shown  // shown, never answered (e.g. recording started)
    case accepted  // switched to the flicker-safe mode and re-locked
    case declined  // kept the sharp shutter
}

public enum ExposurePolicy {
    /// Shutter we aim for (s). Halves the blur of the old 1/120 s cap.
    public static let preferredDurationS = 1.0 / 240
    /// Longest shutter we accept before underexposing (s): the old cap.
    public static let longestDurationS = 1.0 / 120
    /// Highest ISO we raise to (the format's max ISO if that is lower).
    public static let isoCeiling = 800.0

    public struct Choice: Equatable, Sendable {
        public var durationS: Double
        public var iso: Double
        /// Stops darker than the metered exposure (0 when brightness is kept).
        public var underexposureEv: Double
    }

    /// Relative tolerance of `suspectedFlicker`.
    public static let flickerTolerance = 0.01

    /// Auto exposure under flickering light settles on a whole number of flicker periods: k/100 s on 50 Hz
    /// mains, k/120 s on 60 Hz. A metered shutter within ~1 % of k/100 s (k ≥ 1) suggests 50 Hz flicker, of
    /// k/120 s 60 Hz. Nil when neither matches, or both do (k/20 s: 1/20 s, 1/10 s, …; can't tell). Scan 4
    /// metered exactly 0.010002 s (1/100 s) at ISO 258 under LED light on UK mains.
    public static func suspectedFlicker(meteredDurationS d: Double, tolerance: Double = flickerTolerance) -> FlickerMains? {
        guard d.isFinite, d > 0 else { return nil }
        func matches(_ hz: Double) -> Bool {
            let k = (d * hz).rounded()
            return k >= 1 && abs(d - k / hz) <= tolerance * k / hz
        }
        switch (matches(100), matches(120)) {
        case (true, false): return .hz50
        case (false, true): return .hz60
        default: return nil
        }
    }

    /// - Parameters: the metered (auto) exposure, and the active format's ISO and duration limits.
    public static func choose(
        meteredDurationS: Double, meteredISO: Double, minISO: Double, maxISO: Double, minDurationS: Double,
        maxDurationS: Double, mode: ShutterMode = .sharp
    ) -> Choice {
        func clamp(_ x: Double, _ lo: Double, _ hi: Double) -> Double { min(max(x, lo), max(lo, hi)) }
        // Brightness ∝ duration × ISO.
        let light = max(0, meteredDurationS) * max(0, meteredISO)
        let ceiling = clamp(isoCeiling, minISO, maxISO)
        if let fixed = mode.fixedDurationS {
            var duration = clamp(fixed, minDurationS, maxDurationS)
            var iso = min(light / duration, ceiling)
            if iso < minISO {  // too bright even at min ISO: shorten rather than overexpose
                iso = minISO
                duration = clamp(light / minISO, minDurationS, maxDurationS)
            }
            let under = light > 0 ? max(0, log2(light / (duration * iso))) : 0
            return Choice(durationS: duration, iso: iso, underexposureEv: under < 1e-6 ? 0 : under)
        }
        let longest = clamp(longestDurationS, minDurationS, maxDurationS)
        // Never lengthen a shutter auto exposure already made shorter than 1/240 s (bright rooms).
        var duration = clamp(min(meteredDurationS, preferredDurationS), minDurationS, maxDurationS)
        var iso = duration > 0 ? light / duration : meteredISO
        if iso > ceiling {
            // Too dark for 1/240 s within the ceiling: lengthen the shutter back towards 1/120 s.
            iso = ceiling
            duration = max(duration, min(longest, light / ceiling))
        }
        iso = clamp(iso, minISO, maxISO)
        let got = duration * iso
        let under = light > 0 && got > 0 ? max(0, log2(light / got)) : 0
        return Choice(durationS: duration, iso: iso, underexposureEv: under < 1e-6 ? 0 : under)
    }
}

/// Predicted motion blur from camera motion and exposure.
public enum MotionBlur {
    /// Long side of the images the splat trains on (`--max-side 960`); the steadiness cues are judged there.
    public static let trainingLongSide = 960.0
    /// Typical distance to what is in view indoors, to turn translation speed into image motion (m).
    public static let assumedDepthM = 1.5

    /// Blur length in pixels for an image with focal length `fx` (pixels): fx · (ω + v / depth) · exposure.
    /// Rotation dominates indoors: 30 deg/s at 1/240 s and fx 762 (960 px wide) is ~1.7 px.
    public static func pixels(
        fx: Double, angularSpeedRadS: Double, speedMS: Double, exposureS: Double, depthM: Double = assumedDepthM
    ) -> Double {
        guard fx.isFinite, exposureS.isFinite, exposureS > 0 else { return 0 }
        return fx * (abs(angularSpeedRadS) + abs(speedMS) / max(depthM, 0.1)) * exposureS
    }

    /// `px` (measured on an image of `width`×`height`) at the training resolution.
    public static func atTrainingResolution(_ px: Double, width: Int, height: Int) -> Double {
        px * trainingLongSide / Double(max(1, max(width, height)))
    }
}

/// Decides when auto exposure has settled after "Lock exposure" was tapped (0.5.0), so the lock can happen by
/// itself instead of a second button. Fed ~15 times a second with the camera's exposure duration, ISO, white
/// balance gains and whether AVFoundation still reports adjusting (exposure, white balance or focus).
///
/// Steady: brightness `log2(duration × ISO)` within `maxEvChange`, the duration within `maxDurationChange` and
/// each white balance gain within `maxGainChange` of the first sample of the steady run (held tighter on the
/// duration because the flicker hint reads it, `ExposurePolicy.suspectedFlicker`). Settled: steady for
/// `steadyFor` with nothing adjusting, or for `steadyForAnyway` whatever the flags say (under ARKit
/// `isAdjustingExposure` can stay on when auto exposure is pinned at a limit in a dark room, and white balance
/// lingers in mixed light), and never before `minMetering` (switching from a lock back to auto takes a moment to
/// show). After `timeout` it gives up: the caller locks anyway with what auto exposure has then.
public struct ExposureSettle: Sendable {
    public enum State: Equatable, Sendable {
        /// Still metering; `progress` 0...1 towards settled (drops back when the exposure moves again).
        case metering(progress: Double)
        case settled
        case timedOut
    }

    public static let minMetering: Double = 0.8
    public static let steadyFor: Double = 0.6
    public static let steadyForAnyway: Double = 1.2
    public static let timeout: Double = 5
    public static let maxEvChange = 0.05
    public static let maxDurationChange = 0.02
    public static let maxGainChange = 0.02

    private struct Run {
        var t: Double
        var ev: Double
        var durationS: Double
        var gains: [Double]
    }

    private var start: Double?
    private var run: Run?

    public init() {}

    /// One sample at time `t` (s, any monotonic clock). `gains`: white balance (red, green, blue), if known.
    /// Non-finite or non-positive exposure values restart the steady run.
    public mutating func add(
        t: Double, durationS: Double, iso: Double, gains: [Double] = [], adjusting: Bool
    ) -> State {
        let start = self.start ?? t
        self.start = start
        if t - start >= Self.timeout { return .timedOut }
        guard durationS.isFinite, durationS > 0, iso.isFinite, iso > 0 else {
            run = nil
            return .metering(progress: 0)
        }
        let ev = log2(durationS * iso)
        if let r = run, abs(ev - r.ev) <= Self.maxEvChange,
            abs(durationS - r.durationS) <= Self.maxDurationChange * r.durationS,
            r.gains.count == gains.count,
            zip(r.gains, gains).allSatisfy({ abs($1 - $0) <= Self.maxGainChange * abs($0) })
        {
            // still steady
        } else {
            run = Run(t: t, ev: ev, durationS: durationS, gains: gains)
        }
        let runStart = run?.t ?? t
        let due = max(start + Self.minMetering, runStart + (adjusting ? Self.steadyForAnyway : Self.steadyFor))
        if t >= due { return .settled }
        let span = due - start
        return .metering(progress: span > 0 ? min(1, max(0, (t - start) / span)) : 0)
    }
}

/// Guides the doorway sweep by the camera's pose (0.5.0). On scan 5 no sweep was really done (0.19–0.45 m of
/// movement, 7–119° of turn, never the ~180° the wall's far face needs) while a 10 s countdown ran. Now the panel
/// shows what is still missing and ends by itself `holdAfter` seconds after the camera has turned `turnNeededDeg`
/// (either way, yaw about gravity) and moved `moveNeededM` (horizontally) from where the sweep started: time to
/// look at the wall from this side. At `maxDuration` it ends anyway. Fed a few times a second; position in metres
/// (x, z: the horizontal plane of ARKit's gravity-aligned world), yaw in radians.
public struct SweepProgress: Sendable, Equatable {
    public static let turnNeededDeg = 150.0
    public static let moveNeededM = 0.5
    public static let holdAfter = 2.0
    public static let maxDuration = 20.0

    public enum State: Equatable, Sendable {
        case running
        case done  // turned and moved, and looked for `holdAfter`
        case timedOut
    }

    private var t0: Double?
    private var origin: (x: Double, z: Double)?
    private var yaw0 = 0.0
    private var lastYaw = 0.0  // unwrapped
    private var metAt: Double?
    /// Largest turn (degrees) and horizontal distance (m) from the start so far.
    public private(set) var turnedDeg = 0.0
    public private(set) var movedM = 0.0

    public var turned: Bool { turnedDeg >= Self.turnNeededDeg }
    public var moved: Bool { movedM >= Self.moveNeededM }

    public init() {}

    public static func == (a: SweepProgress, b: SweepProgress) -> Bool {
        a.turnedDeg == b.turnedDeg && a.movedM == b.movedM && a.metAt == b.metAt && a.t0 == b.t0
    }

    /// One pose at time `t`; nil `x`/`z`/`yaw` (tracking not normal) only advances the clock.
    public mutating func add(t: Double, x: Double?, z: Double?, yaw: Double?) -> State {
        let t0 = self.t0 ?? t
        self.t0 = t0
        if let x, let z, let yaw, x.isFinite, z.isFinite, yaw.isFinite {
            if let o = origin {
                var d = yaw - lastYaw
                while d > .pi { d -= 2 * .pi }
                while d < -.pi { d += 2 * .pi }
                lastYaw += d
                turnedDeg = max(turnedDeg, abs(lastYaw - yaw0) * 180 / .pi)
                movedM = max(movedM, ((x - o.x) * (x - o.x) + (z - o.z) * (z - o.z)).squareRoot())
            } else {
                origin = (x, z)
                yaw0 = yaw
                lastYaw = yaw
            }
        }
        if turned && moved && metAt == nil { metAt = t }
        if let metAt, t - metAt >= Self.holdAfter { return .done }
        return t - t0 >= Self.maxDuration ? .timedOut : .running
    }
}
