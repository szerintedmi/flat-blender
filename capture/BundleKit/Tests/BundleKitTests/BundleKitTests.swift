import CoreGraphics
import Foundation
import ImageIO
import Testing
import simd

@testable import BundleKit

private func tempDir() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "bundlekit-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func jsonObject(_ data: Data) throws -> [String: Any] {
    try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
}

/// Decodes a PNG and returns its raw samples without any colour conversion.
private func decodeGray(_ png: Data) throws -> (width: Int, height: Int, bits: Int, samples: [Int]) {
    let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
    let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    let raw = try #require(image.dataProvider?.data) as Data
    let bits = image.bitsPerComponent
    #expect(image.bitsPerPixel == bits, "expected a single grey channel")
    let little = image.bitmapInfo.contains(.byteOrder16Little)
    var samples: [Int] = []
    samples.reserveCapacity(image.width * image.height)
    raw.withUnsafeBytes { buf in
        for y in 0..<image.height {
            for x in 0..<image.width {
                let o = y * image.bytesPerRow + x * bits / 8
                if bits == 16 {
                    let a = Int(buf[o]), b = Int(buf[o + 1])
                    samples.append(little ? a | (b << 8) : (a << 8) | b)
                } else {
                    samples.append(Int(buf[o]))
                }
            }
        }
    }
    return (image.width, image.height, bits, samples)
}

@Suite struct MatrixTests {
    @Test func rowMajorFromSimd() {
        // simd is column-major: columns.3 holds the translation.
        var m = matrix_identity_float4x4
        m.columns.0 = [1, 2, 3, 0]  // first column
        m.columns.3 = [10, 20, 30, 1]
        let mat = Mat4(m)
        #expect(mat.rows[0] == [1, 0, 0, 10])
        #expect(mat.rows[1] == [2, 1, 0, 20])
        #expect(mat.rows[2] == [3, 0, 1, 30])
        #expect(mat.rows[3] == [0, 0, 0, 1])
        // m[row][col] == columns.col[row] for every element
        for r in 0..<4 {
            for c in 0..<4 { #expect(mat.rows[r][c] == Double(m[c][r])) }
        }
    }

    @Test func roundTripsThroughSimdAndJSON() throws {
        let q = simd_quatf(angle: 0.3, axis: simd_normalize(SIMD3<Float>(1, 2, 3)))
        var m = simd_float4x4(q)
        m.columns.3 = [0.5, -1.25, 2, 1]
        let mat = Mat4(m)
        let back = mat.simd
        for c in 0..<4 {
            for r in 0..<4 { #expect(Float(back[c][r]) == m[c][r]) }
        }
        let json = try JSONEncoder().encode(mat)
        let decoded = try JSONDecoder().decode(Mat4.self, from: json)
        #expect(decoded == mat)
        let nested = try #require(try JSONSerialization.jsonObject(with: json) as? [[Double]])
        #expect(nested.count == 4 && nested.allSatisfy { $0.count == 4 })
        #expect(nested[0][3] == 0.5 && nested[1][3] == -1.25)
    }

    @Test func shortestDoubleKeepsFloatValue() {
        #expect(shortestDouble(0.1) == 0.1)
        for f: Float in [1452.2531, -0.000123, 3.4e38, 1e-30] {
            #expect(Float(shortestDouble(f)) == f)
        }
    }

    @Test func poseDifferenceResolvesMillimetresAndHundredthsOfDegrees() {
        var a = simd_float4x4(simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3<Float>(0.2, 1, 0.1))))
        a.columns.3 = [3.25, 1.4, -7.5, 1]
        let same = poseDifference(a, a)
        #expect(same.metres == 0 && same.radians < 1e-6)
        var b = a * simd_float4x4(simd_quatf(angle: Float(0.06 * Double.pi / 180), axis: [0, 0, 1]))
        b.columns.3 = [3.2512, 1.4, -7.5, 1]
        let d = poseDifference(a, b)
        #expect(abs(d.metres - 0.0012) < 1e-5)
        #expect(abs(d.radians * 180 / .pi - 0.06) < 0.005)
    }

    @Test func gravityInCamera() throws {
        // Identity: camera axes == world axes, gravity is -y.
        let g = try #require(Vec3.gravityInCamera(cameraToWorld: matrix_identity_float4x4))
        #expect(g == Vec3(0, -1, 0))
        // Camera rolled 90 degrees about its viewing axis (-z): world down shows up along camera x.
        let roll = simd_float4x4(simd_quatf(angle: .pi / 2, axis: [0, 0, 1]))
        let g2 = try #require(Vec3.gravityInCamera(cameraToWorld: roll))
        #expect(abs(g2.x - -1) < 1e-6 && abs(g2.y) < 1e-6 && abs(g2.z) < 1e-6)
    }
}

@Suite struct ModelEncodingTests {
    @Test func intrinsicsFromARKitMatrix() {
        let k = simd_float3x3(columns: ([1450, 0, 0], [0, 1451, 0], [960, 720, 1]))
        let i = Intrinsics(k, width: 1920, height: 1440)
        #expect(i == Intrinsics(fx: 1450, fy: 1451, cx: 960, cy: 720, width: 1920, height: 1440))
        let d = i.scaled(width: 256, height: 192)
        #expect(abs(d.fx - 1450 * 256 / 1920) < 1e-9 && abs(d.cy - 720 * 192 / 1440) < 1e-9)
    }

    @Test func frameUsesSnakeCaseAndOmitsNil() throws {
        let frame = Frame(
            idx: 3, t: 1.5, cameraToWorld: .identity,
            intrinsics: Intrinsics(fx: 1, fy: 1, cx: 0.5, cy: 0.5, width: 2, height: 2),
            tracking: .limitedExcessiveMotion, exposure: Exposure(durationS: 0.008, offsetEv: .nan))
        let obj = try jsonObject(BundleJSON.lineEncoder().encode(frame))
        #expect(Set(obj.keys) == ["idx", "t", "camera_to_world", "intrinsics", "tracking", "exposure"])
        #expect(obj["tracking"] as? String == "limited_excessive_motion")
        let exposure = try #require(obj["exposure"] as? [String: Any])
        #expect(Set(exposure.keys) == ["duration_s"])  // NaN offset dropped, nil iso omitted
    }

    @Test func lineEncoderDoesNotEscapeSlashes() throws {
        let frame = Frame(
            idx: 0, t: 0, cameraToWorld: .identity,
            intrinsics: Intrinsics(fx: 1, fy: 1, cx: 0, cy: 0, width: 1, height: 1), tracking: .normal,
            depth: BundleLayout.depthPath(frame: 12), confidence: BundleLayout.confidencePath(frame: 12))
        let s = try #require(String(data: BundleJSON.lineEncoder().encode(frame), encoding: .utf8))
        #expect(s.contains("\"depth\":\"depth/000012.png\""))
        #expect(s.contains("\"confidence\":\"confidence/000012.png\""))
        #expect(!s.contains("\n"))
    }

    @Test func sessionRoundTripAndKeys() throws {
        let s = Session(
            projectId: "p", sessionId: "s1", type: .texture,
            device: Device(model: "iPad14,3", osVersion: "26.0", appVersion: "0.1.0"),
            camera: LockedCamera(locked: false), startedAt: Date(timeIntervalSince1970: 1_800_000_000.25),
            roomName: "hall")
        let data = try BundleJSON.documentEncoder().encode(s)
        let obj = try jsonObject(data)
        #expect(obj["format"] as? String == "flat-bundle")
        #expect(obj["version"] as? Int == 1)
        #expect(obj["type"] as? String == "texture")
        #expect(obj["started_at"] as? String == "2027-01-15T08:00:00.250Z")
        #expect(obj["ended_at"] == nil && obj["video"] == nil && obj["world_map_in"] == nil && obj["continued_from"] == nil)
        #expect(obj["notes"] as? String == "")
        let device = try #require(obj["device"] as? [String: Any])
        #expect(Set(device.keys) == ["model", "os_version", "app_name", "app_version"])
        #expect(obj["sweeps"] as? [Any] != nil && (obj["sweeps"] as? [Any])?.isEmpty == true)
        let back = try BundleJSON.decoder().decode(Session.self, from: data)
        #expect(back == s)
    }

    @Test func sessionSweepsAndOlderFilesWithout() throws {
        var s = Session(
            projectId: "p", sessionId: "s2", type: .lap,
            device: Device(model: "iPad14,3", osVersion: "26.0", appVersion: "0.2.0 (5)"),
            worldMapIn: "s1", continuedFrom: "s1", startedAt: Date(timeIntervalSince1970: 1_800_000_000))
        s.sweeps = [Sweep(t0: 2400.5, t1: 2410.25)]
        let data = try BundleJSON.documentEncoder().encode(s)
        let obj = try jsonObject(data)
        #expect(obj["type"] as? String == "lap")
        #expect(obj["continued_from"] as? String == "s1" && obj["world_map_in"] as? String == "s1")
        #expect(obj["room_name"] == nil)
        let sweeps = try #require(obj["sweeps"] as? [[String: Double]])
        #expect(sweeps == [["t0": 2400.5, "t1": 2410.25]])
        #expect(try BundleJSON.decoder().decode(Session.self, from: data) == s)

        // session.json from before build 5 has no "sweeps".
        var old = obj
        old["sweeps"] = nil
        old["continued_from"] = nil  // nor "continued_from" (app < 0.5)
        let oldData = try JSONSerialization.data(withJSONObject: old)
        let decoded = try BundleJSON.decoder().decode(Session.self, from: oldData)
        #expect(decoded.sweeps == [] && decoded.continuedFrom == nil)
    }

    @Test func anchorEventKeys() throws {
        let e = AnchorEvent(
            t: 2401.125, id: "6F9619FF-8B86-D011-B42D-00CF4FC964FF",
            name: BundleLayout.anchorName(sessionId: "20261006-120000-hall", kind: .still, number: 3), event: .final,
            transform: .identity)
        let line = try BundleJSON.lineEncoder().encode(e)
        let obj = try jsonObject(line)
        #expect(Set(obj.keys) == ["t", "id", "name", "event", "transform"])
        #expect(obj["name"] as? String == "20261006-120000-hall/still/3")
        #expect(obj["event"] as? String == "final")
        #expect(obj["transform"] as? [[Double]] == Mat4.identity.rows)
        #expect(try BundleJSON.decoder().decode(AnchorEvent.self, from: line) == e)
    }
    @Test func enumValuesMatchPython() {
        #expect(SessionType.allCases.map(\.rawValue) == ["scan", "lap", "texture", "ultrawide"])
        #expect(AnchorEventKind.allCases.map(\.rawValue) == ["add", "update", "final"])
        #expect(
            TrackingState.allCases.map(\.rawValue) == [
                "normal", "limited_initializing", "limited_excessive_motion", "limited_insufficient_features",
                "limited_relocalizing", "not_available",
            ])
        #expect(StillSource.allCases.map(\.rawValue) == ["tap", "auto", "texture", "ultrawide", "video"])
    }

    @Test func stillKeys() throws {
        let still = Still(
            id: 1, file: BundleLayout.stillPath(id: 1), t: 2, source: .tap, cameraToWorld: .identity,
            whiteBalance: WhiteBalanceGains(red: 2, green: 1, blue: 1.5), sharpness: 12.5, nearestFrame: 4,
            gravity: Vec3(0, -1, 0))
        let obj = try jsonObject(BundleJSON.lineEncoder().encode(still))
        #expect(
            Set(obj.keys) == [
                "id", "file", "t", "source", "camera_to_world", "white_balance", "sharpness", "nearest_frame",
                "gravity",
            ])
        #expect(obj["file"] as? String == "stills/000001.jpg")
        #expect(obj["gravity"] as? [Double] == [0, -1, 0])
    }

    @Test func stillMotion() throws {
        // What app 0.4+ writes: motion only (no sharpness, no blur flag).
        let still = Still(
            id: 2, file: BundleLayout.stillPath(id: 2), t: 3, source: .auto,
            motion: StillMotion(angularSpeedDegS: 31.5, speedMS: 0.25, blurPx: 4.5))
        let line = try BundleJSON.lineEncoder().encode(still)
        let obj = try jsonObject(line)
        #expect(Set(obj.keys) == ["id", "file", "t", "source", "motion"])
        let motion = try #require(obj["motion"] as? [String: Double])
        #expect(motion == ["angular_speed_deg_s": 31.5, "speed_m_s": 0.25, "blur_px": 4.5])
        #expect(try BundleJSON.decoder().decode(Still.self, from: line) == still)

        let bare = try jsonObject(
            BundleJSON.lineEncoder().encode(StillMotion(angularSpeedDegS: 1, speedMS: 0, blurPx: .nan)))
        #expect(bare["blur_px"] as? Double == 0)
    }

    @Test func stillLinesFromAppBefore04StillDecode() throws {
        // A stills.jsonl line as 0.3.0 (6) wrote it on scan 4 (sharpness, stream sharpness, blur flag).
        let line = Data(
            """
            {"id":7,"file":"stills/000007.jpg","t":1234.5,"source":"tap","sharpness":61.2,"nearest_frame":40,\
            "motion":{"angular_speed_deg_s":21.3,"speed_m_s":0.12,"blur_px":5.1,"frame_sharpness":40.5,\
            "median_sharpness":70.1},"flagged_blurry":true}
            """.utf8)
        let still = try BundleJSON.decoder().decode(Still.self, from: line)
        #expect(still.sharpness == 61.2 && still.flaggedBlurry == true)
        #expect(still.motion?.frameSharpness == 40.5 && still.motion?.medianSharpness == 70.1)
    }

    @Test func lockedCameraMeteredAndUnderexposure() throws {
        let c = LockedCamera(
            locked: true, exposureDurationS: 1.0 / 120, iso: 800, metered: Exposure(durationS: 1.0 / 30, iso: 640),
            underexposureEv: 1.678, shutterMode: .flicker50Hz)
        let data = try BundleJSON.lineEncoder().encode(c)
        let obj = try jsonObject(data)
        #expect(Set(obj.keys) == ["locked", "exposure_duration_s", "iso", "metered", "underexposure_ev", "shutter_mode"])
        #expect(obj["shutter_mode"] as? String == "flicker_50hz")
        #expect(ShutterMode.allCases.map(\.rawValue) == ["sharp", "flicker_50hz", "flicker_60hz"])
        let metered = try #require(obj["metered"] as? [String: Double])
        #expect(Set(metered.keys) == ["duration_s", "iso"])
        #expect(try BundleJSON.decoder().decode(LockedCamera.self, from: data) == c)
        // Older files: neither key.
        #expect(Set(try jsonObject(BundleJSON.lineEncoder().encode(LockedCamera(locked: false))).keys) == ["locked"])
    }

    @Test func lockedCameraFlickerFields() throws {
        let c = LockedCamera(locked: true, shutterMode: .flicker50Hz, flickerSuspected: .hz50, flickerPrompt: .accepted)
        let data = try BundleJSON.lineEncoder().encode(c)
        let obj = try jsonObject(data)
        #expect(obj["flicker_suspected"] as? String == "50hz")
        #expect(obj["flicker_prompt"] as? String == "accepted")
        #expect(FlickerMains.allCases.map(\.rawValue) == ["50hz", "60hz"])
        #expect(FlickerPromptOutcome.allCases.map(\.rawValue) == ["shown", "accepted", "declined"])
        #expect(try BundleJSON.decoder().decode(LockedCamera.self, from: data) == c)
    }

    @Test func scan4CameraDecodesAndSuggests50Hz() throws {
        // session.json `camera` of scan 4 (data/fourth/raw/20261008-201609-living, app 0.3.0 (6)).
        let json = Data(
            """
            {"exposure_duration_s":0.004166,"iso":619.8161,"lens_position":0.7058823,"locked":true,\
            "metered":{"duration_s":0.010002,"iso":258.20508},"shutter_mode":"sharp","underexposure_ev":0,\
            "white_balance":{"blue":2.3164062,"green":1,"red":1.6132812}}
            """.utf8)
        let c = try BundleJSON.decoder().decode(LockedCamera.self, from: json)
        #expect(c.flickerSuspected == nil && c.flickerPrompt == nil)
        let metered = try #require(c.metered?.durationS)
        #expect(ExposurePolicy.suspectedFlicker(meteredDurationS: metered) == .hz50)
    }
}

@Suite struct CaptureQualityTests {
    private func choose(
        _ d: Double, _ iso: Double, maxISO: Double = 2304, mode: ShutterMode = .sharp
    ) -> ExposurePolicy.Choice {
        ExposurePolicy.choose(
            meteredDurationS: d, meteredISO: iso, minISO: 34, maxISO: maxISO, minDurationS: 1.0 / 20000,
            maxDurationS: 1.0 / 3, mode: mode)
    }

    @Test func flickerSafeModesFixTheShutter() {
        // Scan 3's light (1/120 s ISO 284) at a fixed 1/100 s: ISO 236.67, no underexposure.
        let c50 = choose(1.0 / 120, 284, mode: .flicker50Hz)
        #expect(close(c50.durationS, 1.0 / 100) && close(c50.iso, 284 * 100.0 / 120) && c50.underexposureEv == 0)
        // Dark (1/30 s ISO 1000) at a fixed 1/120 s: ISO 800 and log2(5) stops under.
        let c60 = choose(1.0 / 30, 1000, mode: .flicker60Hz)
        #expect(close(c60.durationS, 1.0 / 120) && close(c60.iso, 800) && close(c60.underexposureEv, log2(5)))
        // Daylight-bright (1/2000 s ISO 34): shorter than 1/100 s at min ISO instead of overexposing.
        let bright = choose(1.0 / 2000, 34, mode: .flicker50Hz)
        #expect(close(bright.durationS, 1.0 / 2000) && close(bright.iso, 34) && bright.underexposureEv == 0)
    }

    private func close(_ a: Double, _ b: Double, _ tol: Double = 1e-9) -> Bool { abs(a - b) <= tol * max(1, abs(b)) }

    @Test func scan3HallwayGoesTo240WithIsoRaised() {
        // Scan 3's lock was 1/120 s at ISO 284 (after the old cap). Same light at 1/240 s needs ISO 568.
        let c = choose(1.0 / 120, 284)
        #expect(close(c.durationS, 1.0 / 240))
        #expect(close(c.iso, 568))
        #expect(c.underexposureEv == 0)
    }

    @Test func brightRoomKeepsTheMeteredShortShutter() {
        let c = choose(1.0 / 500, 50)
        #expect(close(c.durationS, 1.0 / 500))
        #expect(close(c.iso, 50))
        #expect(c.underexposureEv == 0)
    }

    @Test func dimRoomLengthensTheShutterAtTheCeiling() {
        // 1/60 s ISO 300: 1/240 s would need ISO 1200 > 800, so ISO 800 and 1/160 s (between 1/240 and 1/120).
        let c = choose(1.0 / 60, 300)
        #expect(close(c.iso, 800))
        #expect(close(c.durationS, 1.0 / 160))
        #expect(c.underexposureEv == 0)
    }

    @Test func darkRoomUnderexposesInsteadOfBlurOrNoise() {
        // 1/30 s ISO 1000: even 1/120 s at ISO 800 is log2(4 × 1000 / 800) ≈ 2.32 stops short.
        let c = choose(1.0 / 30, 1000)
        #expect(close(c.durationS, 1.0 / 120))
        #expect(close(c.iso, 800))
        #expect(close(c.underexposureEv, log2(5), 1e-9))
    }

    @Test func formatMaxIsoBelowTheCeilingWins() {
        let c = choose(1.0 / 120, 284, maxISO: 400)
        #expect(close(c.iso, 400))
        #expect(close(c.durationS, 284.0 / 120 / 400))
        #expect(c.underexposureEv == 0)
    }

    @Test func blurPrediction() {
        // Scan 3 hallway, 47 deg/s at 1/120 s with fx 1524 px: ~10 px at 1920 wide, ~5 px at 960.
        let px = MotionBlur.pixels(fx: 1524, angularSpeedRadS: 47 * .pi / 180, speedMS: 0, exposureS: 1.0 / 120)
        #expect(abs(px - 10.42) < 0.01)
        #expect(abs(MotionBlur.atTrainingResolution(px, width: 1920, height: 1440) - 5.21) < 0.01)
        // Translation: 0.3 m/s at 1.5 m is 0.2 rad/s of image motion.
        #expect(close(MotionBlur.pixels(fx: 1000, angularSpeedRadS: 0, speedMS: 0.3, exposureS: 0.01), 2))
        #expect(MotionBlur.pixels(fx: 1000, angularSpeedRadS: 1, speedMS: 0, exposureS: .nan) == 0)
    }

    @Test func flickerFromTheMeteredShutter() {
        func f(_ d: Double) -> FlickerMains? { ExposurePolicy.suspectedFlicker(meteredDurationS: d) }
        // Scan 4: auto exposure settled on exactly 1/100 s under LED light (UK mains).
        #expect(f(0.010002) == .hz50)
        #expect(f(1.0 / 100 * 1.009) == .hz50 && f(1.0 / 100 * 0.991) == .hz50)  // within 1 %
        #expect(f(1.0 / 100 * 1.02) == nil)  // 2 % off: no
        #expect(f(2.0 / 100) == .hz50 && f(3.0 / 100) == .hz50)
        #expect(f(1.0 / 120) == .hz60 && f(0.0083) == .hz60 && f(2.0 / 120) == .hz60)
        // Not a whole number of flicker periods.
        #expect(f(1.0 / 240) == nil && f(1.0 / 60) == .hz60 && f(1.0 / 50) == .hz50 && f(0.0125) == nil)
        #expect(f(1.0 / 1000) == nil && f(1.0 / 45) == nil)
        #expect(f(1.0 / 30) == .hz60 && f(1.0 / 25) == .hz50)  // 4/120 s, 4/100 s
        // k/20 s is both k'/100 and k''/120: can't tell.
        #expect(f(1.0 / 20) == nil && f(1.0 / 10) == nil)
        #expect(f(0) == nil && f(.nan) == nil && f(-0.01) == nil)
    }
}

@Suite struct StageTimerTests {
    @Test func countsMaxOverAndPercentiles() {
        let t = StageTimer(overS: 0.008)
        #expect(t.summary.count == 0 && t.summary.p50S == nil)
        for _ in 0..<98 { t.record(seconds: 0.00105) }
        t.record(seconds: 0.00955)  // over 8 ms
        t.record(seconds: 0.25)  // beyond the histogram: last bin, exact max
        t.record(seconds: .nan)  // ignored
        let s = t.summary
        #expect(s.count == 100 && s.over == 2)
        #expect(abs(s.totalS - (98 * 0.00105 + 0.00955 + 0.25)) < 1e-12)
        #expect(s.maxS == 0.25)
        #expect(abs((s.p50S ?? 0) - 0.0011) < 1e-9)  // upper edge of the 1.0–1.1 ms bin
        #expect(abs((s.p99S ?? 0) - 0.0096) < 1e-9)  // the 99th of 100 samples: the 9.55 ms one
        t.reset()
        #expect(t.summary == StageTimer.Summary())
    }
}

@Suite struct JSONLWriterTests {
    @Test func appendsOneLinePerValueAndReopens() throws {
        let dir = try tempDir()
        let url = dir.appending(path: "x.jsonl")
        let w = try JSONLWriter(url: url)
        try w.append(["a": 1])
        // Visible on disk before close (no user-space buffering).
        #expect(try String(contentsOf: url, encoding: .utf8) == "{\"a\":1}\n")
        try w.append(["a": 2])
        try w.close()
        let w2 = try JSONLWriter(url: url)
        try w2.append(["a": 3])
        try w2.close()
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        #expect(lines == ["{\"a\":1}", "{\"a\":2}", "{\"a\":3}"])
    }
}

@Suite struct PNGTests {
    @Test func depthIs16BitMillimetres() throws {
        let w = 256, h = 192
        var metres = [Float](repeating: 0, count: w * h)
        for i in metres.indices { metres[i] = Float(i % 5000) / 1000 }  // 0 ... 4.999 m
        metres[1] = .nan
        metres[2] = -1
        metres[3] = 70  // beyond 65.535 m: clamped
        metres[4] = 1.2344  // rounds to 1234 mm
        metres[5] = 1.2346  // rounds to 1235 mm
        let png = try DepthPNG.encodeDepth(metres: metres, width: w, height: h)
        #expect(png.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        let img = try decodeGray(png)
        #expect(img.width == w && img.height == h && img.bits == 16)
        #expect(img.samples[0] == 0 && img.samples[1] == 0 && img.samples[2] == 0)
        #expect(img.samples[3] == 65535 && img.samples[4] == 1234 && img.samples[5] == 1235)
        for i in stride(from: 6, to: metres.count, by: 997) {
            #expect(img.samples[i] == Int((metres[i] * 1000).rounded()))
        }
    }

    @Test func depthFromStridedBuffer() {
        // 3x2 Float32 map with rows padded to 16 bytes.
        let bytesPerRow = 16
        var raw = [UInt8](repeating: 0, count: bytesPerRow * 2)
        let values: [[Float]] = [[1, 2, 3], [0.5, .infinity, 0.0015]]
        raw.withUnsafeMutableBytes { buf in
            for y in 0..<2 {
                for x in 0..<3 {
                    buf.storeBytes(of: values[y][x], toByteOffset: y * bytesPerRow + x * 4, as: Float32.self)
                }
            }
        }
        let mm = raw.withUnsafeBytes {
            DepthPNG.millimetres(fromMetres: $0.baseAddress!, width: 3, height: 2, bytesPerRow: bytesPerRow)
        }
        #expect(mm == [1000, 2000, 3000, 500, 0, 2])
    }

    @Test func confidenceIs8Bit() throws {
        let w = 256, h = 192
        let levels = (0..<(w * h)).map { UInt8($0 % 3) }
        let img = try decodeGray(try DepthPNG.encodeConfidence(levels, width: w, height: h))
        #expect(img.width == w && img.height == h && img.bits == 8)
        #expect(img.samples == levels.map(Int.init))
    }

    @Test func rejectsWrongSize() {
        #expect(throws: ImageEncodingError.self) {
            try DepthPNG.encodeConfidence([0, 1], width: 2, height: 2)
        }
    }
}

@Suite struct PLYTests {
    @Test func binaryLayout() throws {
        let a = MeshChunk(vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 2]])
        let b = MeshChunk(vertices: [[0, 0, 1], [1, 0, 1], [0, 1, 1]], faces: [[0, 2, 1]])
        let data = try PLYWriter.encode([a, b], comment: "test")
        let marker = Data("end_header\n".utf8)
        let range = try #require(data.range(of: marker))
        let header = try #require(String(data: data[..<range.upperBound], encoding: .ascii))
        #expect(header.hasPrefix("ply\nformat binary_little_endian 1.0\ncomment test\n"))
        #expect(header.contains("element vertex 6\n"))
        #expect(header.contains("element face 2\nproperty list uchar int vertex_indices\n"))
        #expect(!header.contains("nx"))
        let body = data[range.upperBound...]
        #expect(body.count == 6 * 12 + 2 * 13)
        let bytes = [UInt8](body)
        func f32(_ o: Int) -> Float { Float(bitPattern: UInt32(bytes[o]) | UInt32(bytes[o + 1]) << 8 | UInt32(bytes[o + 2]) << 16 | UInt32(bytes[o + 3]) << 24) }
        func i32(_ o: Int) -> Int32 { Int32(bitPattern: UInt32(bytes[o]) | UInt32(bytes[o + 1]) << 8 | UInt32(bytes[o + 2]) << 16 | UInt32(bytes[o + 3]) << 24) }
        #expect(f32(3 * 12) == 0 && f32(3 * 12 + 8) == 1)  // 4th vertex z
        let faces = 6 * 12
        #expect(bytes[faces] == 3 && i32(faces + 1) == 0 && i32(faces + 5) == 1 && i32(faces + 9) == 2)
        // Second chunk's indices are re-based by 3.
        #expect(bytes[faces + 13] == 3 && i32(faces + 14) == 3 && i32(faces + 18) == 5 && i32(faces + 22) == 4)
    }

    @Test func normalsWhenEveryChunkHasThem() throws {
        let a = MeshChunk(vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], normals: Array(repeating: [0, 0, 1], count: 3), faces: [[0, 1, 2]])
        let data = try PLYWriter.encode([a])
        let range = try #require(data.range(of: Data("end_header\n".utf8)))
        let header = try #require(String(data: data[..<range.upperBound], encoding: .ascii))
        #expect(header.contains("property float nx\nproperty float ny\nproperty float nz\n"))
        #expect(data[range.upperBound...].count == 3 * 24 + 13)
    }

    @Test func rejectsBadIndex() {
        #expect(throws: PLYError.self) {
            try PLYWriter.encode([MeshChunk(vertices: [[0, 0, 0]], faces: [[0, 0, 1]])])
        }
    }
}

@Suite struct LayoutTests {
    @Test func paths() throws {
        #expect(BundleLayout.depthPath(frame: 123) == "depth/000123.png")
        #expect(BundleLayout.confidencePath(frame: 0) == "confidence/000000.png")
        #expect(BundleLayout.stillPath(id: 1) == "stills/000001.jpg")
        #expect(BundleLayout.padded(1_234_567) == "1234567")
        #expect(BundleLayout.anchorName(sessionId: "20261006-120000-hall", kind: .tick, number: 12) == "20261006-120000-hall/tick/12")
        #expect(BundleLayout.isAnchorName("20261006-120000-hall/tick/12"))
        #expect(BundleLayout.isAnchorName("20261006-130000/still/1"))
        for other in [nil, "", "plane", "x/tick/", "x/mesh/1", "/tick/1", "a/b/still/1"] {
            #expect(!BundleLayout.isAnchorName(other))
        }
        let dir = try tempDir().appending(path: "session")
        try BundleLayout.createDirectories(at: dir)
        for sub in ["depth", "confidence", "stills"] {
            var isDir: ObjCBool = false
            #expect(FileManager.default.fileExists(atPath: dir.appending(path: sub).path, isDirectory: &isDir) && isDir.boolValue)
        }
    }
}

@Suite struct ExposureSettleTests {
    typealias Sample = (durationS: Double, iso: Double, gains: [Double], adjusting: Bool)

    /// Feeds samples at 15 Hz from t0 until the detector leaves `.metering` (or `until`).
    private func run(
        _ s: inout ExposureSettle, from t0: Double, until: Double, sample: (Double) -> Sample
    ) -> (state: ExposureSettle.State, t: Double) {
        var t = t0
        var state = ExposureSettle.State.metering(progress: 0)
        while t <= until {
            let x = sample(t)
            state = s.add(t: t, durationS: x.durationS, iso: x.iso, gains: x.gains, adjusting: x.adjusting)
            if case .metering = state {} else { return (state, t) }
            t += 1.0 / 15
        }
        return (state, t)
    }

    private let wb = [2.0, 1.0, 1.8]

    @Test func steadyExposureSettlesAfterTheMinimumTime() {
        var s = ExposureSettle()
        let (state, t) = run(&s, from: 100, until: 110) { _ in (1.0 / 60, 300, wb, false) }
        #expect(state == .settled)
        #expect(t >= 100 + ExposureSettle.minMetering && t < 100 + ExposureSettle.minMetering + 0.1)
    }

    @Test func waitsUntilTheExposureStopsMoving() {
        var s = ExposureSettle()
        // Auto exposure ramps for 1.5 s (adjusting), then holds 1/100 s at ISO 258 (scan 4's LED room).
        let (state, t) = run(&s, from: 0, until: 10) { t in
            t < 1.5 ? (1.0 / 30 - t * 0.01, 600 - t * 200, wb, true) : (0.010002, 258, wb, false)
        }
        #expect(state == .settled)
        #expect(t >= 1.5 + ExposureSettle.steadyFor && t < 1.5 + ExposureSettle.steadyFor + 0.15)
    }

    @Test func aStuckAdjustingFlagOnlyDelaysTheLock() {
        var s = ExposureSettle()
        // Dark room: auto exposure pinned at its limit keeps reporting adjusting, but the values don't move.
        let (state, t) = run(&s, from: 0, until: 10) { _ in (1.0 / 30, 800, wb, true) }
        #expect(state == .settled)
        #expect(t >= ExposureSettle.steadyForAnyway && t < ExposureSettle.steadyForAnyway + 0.1)
    }

    @Test func whiteBalanceStillMovingKeepsMetering() {
        var s = ExposureSettle()
        // Exposure steady from the start, the red gain converging until 2 s (5 % per 0.2 s).
        let (state, t) = run(&s, from: 0, until: 10) { t in
            (1.0 / 240, 450, [t < 2 ? 1.5 + 0.25 * t : 2.0, 1.0, 1.8], false)
        }
        #expect(state == .settled)
        #expect(t > 2 && t < 2 + ExposureSettle.steadyFor + 0.15)  // not at 0.8 s, when the exposure alone was steady
    }

    @Test func driftTimesOut() {
        var s = ExposureSettle()
        // Brightness creeping by 0.1 EV per sample: never steady, so it times out.
        let (state, t) = run(&s, from: 0, until: 10) { t in (0.01 * pow(2, t * 1.5), 200, wb, false) }
        #expect(state == .timedOut)
        #expect(abs(t - ExposureSettle.timeout) < 0.1)
    }

    @Test func smallJitterIsSteady() {
        var s = ExposureSettle()
        // ±1 % ISO, ±0.5 % duration and ±0.5 % gain jitter: inside 0.05 EV / 2 % / 2 %.
        let (state, _) = run(&s, from: 0, until: 10) { t in
            let k = Int(t * 15) % 2 == 0 ? 1.0 : -1.0
            return (1.0 / 240 * (1 + 0.005 * k), 450 * (1 + 0.01 * k), wb.map { $0 * (1 + 0.005 * k) }, false)
        }
        #expect(state == .settled)
    }

    @Test func durationSwapIsNotSteadyEvenAtTheSameBrightness() {
        var s = ExposureSettle()
        // Same brightness, but the shutter alternates 1/100 s and 1/120 s: the flicker hint needs a settled duration.
        let (state, _) = run(&s, from: 0, until: 10) { t in
            Int(t * 15) % 2 == 0 ? (1.0 / 100, 240, wb, false) : (1.0 / 120, 288, wb, false)
        }
        #expect(state == .timedOut)
    }

    @Test func progressRisesAndInvalidSamplesRestart() {
        var s = ExposureSettle()
        guard case .metering(let p0) = s.add(t: 0, durationS: 0.01, iso: 200, adjusting: false),
            case .metering(let p1) = s.add(t: 0.4, durationS: 0.01, iso: 200, adjusting: false),
            case .metering(let p2) = s.add(t: 0.5, durationS: .nan, iso: 200, adjusting: false)
        else {
            Issue.record("expected metering")
            return
        }
        #expect(p0 == 0 && abs(p1 - 0.5) < 1e-9 && p2 == 0)
    }
}

@Suite struct SweepProgressTests {
    @Test func turningRoundAndSteppingThroughEndsIt() {
        var s = SweepProgress()
        var state = SweepProgress.State.running
        var t = 0.0
        // 4 Hz: step 1 m through the door over 3 s while turning 180° (through the ±180° wrap), then look.
        while t <= 10 {
            let f = min(1, t / 3)
            state = s.add(t: t, x: f, z: 0, yaw: 2.5 + f * .pi)  // 2.5 → 5.64 rad, wraps past π
            if state != .running { break }
            t += 0.25
        }
        #expect(state == .done)
        #expect(s.turned && s.moved && abs(s.turnedDeg - 180) < 1e-6 && abs(s.movedM - 1) < 1e-9)
        #expect(abs(t - (2.5 + SweepProgress.holdAfter)) < 0.01)  // both met at ~2.5 s (150°), done 2 s later
    }

    @Test func scan5StyleSweepNeverEndsEarly() {
        var s = SweepProgress()
        var state = SweepProgress.State.running
        var t = 0.0
        // Scan 5: 0.3 m and a 90° look around, back and forth.
        while t <= 30 {
            state = s.add(t: t, x: 0.3 * sin(t), z: 0, yaw: .pi / 4 * sin(t * 2))
            if state != .running { break }
            t += 0.25
        }
        #expect(state == .timedOut && !s.turned && !s.moved)
        #expect(abs(t - SweepProgress.maxDuration) < 0.3)
    }

    @Test func lostTrackingOnlyAdvancesTheClock() {
        var s = SweepProgress()
        #expect(s.add(t: 0, x: nil, z: nil, yaw: nil) == .running)
        #expect(s.add(t: 1, x: 5, z: 5, yaw: 0) == .running)  // the first pose is the origin
        #expect(s.movedM == 0 && s.turnedDeg == 0)
        _ = s.add(t: 2, x: 5, z: 5.6, yaw: -2.7)  // -155°
        #expect(s.moved && s.turned)
    }
}
