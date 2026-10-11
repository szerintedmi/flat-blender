// bundlekit-sample <output-dir>
//
// Writes a small synthetic capture bundle with BundleKit's encoders, so the pipeline can check the Swift
// encoding against its Pydantic models (contract test). Contents:
//   session.json, frames.jsonl (6 frames), depth/ + confidence/ PNGs (256x192, every 2nd frame),
//   stills/000001.jpg + 000002.jpg (tiny generated JPEGs), stills.jsonl, mesh.ply (a unit cube),
//   anchors.jsonl (add / update / final lines for a still anchor and a tick anchor), one doorway sweep.
// There is no video.mov: frames.jsonl refers to frame indices of a video that doesn't exist.

import BundleKit
import CoreGraphics
import Foundation
import simd

let args = CommandLine.arguments
guard args.count == 2 else {
    FileHandle.standardError.write(Data("usage: bundlekit-sample <output-dir>\n".utf8))
    exit(2)
}
let dir = URL(filePath: args[1], directoryHint: .isDirectory)

do {
    try? FileManager.default.removeItem(at: dir)
    try BundleLayout.createDirectories(at: dir)

    let videoW = 1920, videoH = 1440
    let depthW = 256, depthH = 192
    let streamK = Intrinsics(fx: 1452.25, fy: 1452.25, cx: 959.5, cy: 719.25, width: videoW, height: videoH)
    let start = Date(timeIntervalSince1970: 1_791_300_000.125)  // fixed, so output is reproducible

    // session.json
    var session = Session(
        projectId: "sample-flat", sessionId: "20261006-120000-hall", type: .scan,
        device: Device(model: "iPad14,3", osVersion: "26.0", appVersion: "0.1.0 (1)"),
        video: VideoFormat(width: videoW, height: videoH, fps: 60),
        depth: DepthFormat(width: depthW, height: depthH, smoothed: false),
        camera: LockedCamera(
            locked: true, exposureDurationS: 1.0 / 120.0, iso: 320, lensPosition: 0.8125,
            whiteBalance: WhiteBalanceGains(red: 1.875, green: 1.0, blue: 2.25),
            metered: Exposure(durationS: 0.010002, iso: 258.2), shutterMode: .sharp, flickerSuspected: .hz50,
            flickerPrompt: .shown),
        worldMapIn: nil, worldMapOut: "20261006-120000-hall", startedAt: start, roomName: "hall",
        notes: "synthetic sample from bundlekit-sample")

    // frames.jsonl + depth/confidence PNGs
    let frames = try JSONLWriter(url: dir.appending(path: BundleLayout.framesFile))
    let t0 = 12345.678
    var poses: [simd_float4x4] = []
    for idx in 0..<6 {
        let t = t0 + Double(idx) / 60
        let angle = Float(idx) * 0.05
        var pose = simd_float4x4(simd_quatf(angle: angle, axis: SIMD3<Float>(0, 1, 0)))
        pose.columns.3 = SIMD4<Float>(0.1 * Float(idx), 1.2, -0.05 * Float(idx), 1)
        poses.append(pose)

        var depthPath: String?
        var confPath: String?
        if idx % 2 == 0 {
            var metres = [Float](repeating: 0, count: depthW * depthH)
            var conf = [UInt8](repeating: 0, count: depthW * depthH)
            for y in 0..<depthH {
                for x in 0..<depthW {
                    let i = y * depthW + x
                    metres[i] = 1.0 + 2.0 * Float(x) / Float(depthW - 1) + 0.01 * Float(idx)
                    conf[i] = UInt8((x / 32 + y / 32) % 3)
                }
            }
            for y in 0..<16 { for x in 0..<16 { metres[y * depthW + x] = .nan } }  // a hole: no data
            depthPath = BundleLayout.depthPath(frame: idx)
            confPath = BundleLayout.confidencePath(frame: idx)
            try DepthPNG.write(
                DepthPNG.encodeDepth(metres: metres, width: depthW, height: depthH),
                to: dir.appending(path: depthPath!))
            try DepthPNG.write(
                DepthPNG.encodeConfidence(conf, width: depthW, height: depthH), to: dir.appending(path: confPath!))
        }

        try frames.append(
            Frame(
                idx: idx, t: t, cameraToWorld: Mat4(pose), intrinsics: streamK,
                tracking: idx == 0 ? .limitedInitializing : .normal,
                exposure: Exposure(durationS: 1.0 / 120.0, offsetEv: 0, iso: 320),
                depth: depthPath, confidence: confPath))
    }
    try frames.close()

    // stills: tiny generated JPEGs + stills.jsonl
    let stills = try JSONLWriter(url: dir.appending(path: BundleLayout.stillsFile))
    let stillW = 160, stillH = 120
    var rgb = [UInt8](repeating: 0, count: stillW * stillH * 4)
    for y in 0..<stillH {
        for x in 0..<stillW {
            let checker: UInt8 = ((x / 10 + y / 10) % 2 == 0) ? 200 : 40
            let i = (y * stillW + x) * 4
            rgb[i] = checker
            rgb[i + 1] = UInt8(x * 255 / (stillW - 1))
            rgb[i + 2] = UInt8(y * 255 / (stillH - 1))
            rgb[i + 3] = 255
        }
    }
    let rgbData = Data(rgb)
    guard let provider = CGDataProvider(data: rgbData as CFData),
        let image = CGImage(
            width: stillW, height: stillH, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: stillW * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { throw ImageEncodingError.cgImage }
    let jpeg = try ImageWriter.jpeg(image, quality: 0.95)
    // A real still has its own (higher) resolution; here the intrinsics are the stream's scaled to the JPEG.
    let stillK = streamK.scaled(width: stillW, height: stillH)

    for (id, frameIdx, source) in [(1, 2, StillSource.auto), (2, 5, StillSource.auto)] {
        let file = BundleLayout.stillPath(id: id)
        try jpeg.write(to: dir.appending(path: file))
        try stills.append(
            Still(
                id: id, file: file, t: t0 + Double(frameIdx) / 60 + 0.004, source: source, intrinsics: stillK,
                cameraToWorld: Mat4(poses[frameIdx]), tracking: .normal,
                exposure: Exposure(durationS: 1.0 / 120.0, offsetEv: 0, iso: 320),
                whiteBalance: WhiteBalanceGains(red: 1.875, green: 1.0, blue: 2.25), nearestFrame: frameIdx,
                gravity: Vec3.gravityInCamera(cameraToWorld: poses[frameIdx]),
                motion: StillMotion(angularSpeedDegS: 2.5, speedMS: 0.02, blurPx: 0.4)))
    }
    try stills.close()

    // anchors.jsonl: a still anchor and a tick anchor, added, moved 2 mm by ARKit, and the world map's final.
    let anchors = try JSONLWriter(url: dir.appending(path: BundleLayout.anchorsFile))
    let anchorIds = ["0F6E1A52-5D2B-4C3E-9A7B-1C2D3E4F5A6B", "7A8B9C0D-1E2F-4A3B-8C5D-6E7F8A9B0C1D"]
    let anchorNames = [
        BundleLayout.anchorName(sessionId: session.sessionId, kind: .still, number: 1),
        BundleLayout.anchorName(sessionId: session.sessionId, kind: .tick, number: 1),
    ]
    for i in 0..<2 {
        let pose = poses[i == 0 ? 2 : 5]
        var moved = pose
        moved.columns.3.x += 0.002
        let tAdd = t0 + Double(i == 0 ? 2 : 5) / 60
        for (t, event, m) in [(tAdd, AnchorEventKind.add, pose), (tAdd + 0.5, .update, moved), (t0 + 2, .final, moved)] {
            try anchors.append(AnchorEvent(t: t, id: anchorIds[i], name: anchorNames[i], event: event, transform: Mat4(m)))
        }
    }
    try anchors.close()
    session.sweeps = [Sweep(t0: t0 + 1.0 / 60, t1: t0 + 4.0 / 60)]

    // mesh.ply: unit cube
    let v: [SIMD3<Float>] = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    let f: [SIMD3<UInt32>] = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ]
    try PLYWriter.write(
        [MeshChunk(vertices: v, normals: v.map { simd_normalize($0 - 0.5) }, faces: f)],
        to: dir.appending(path: BundleLayout.meshFile), comment: "bundlekit-sample unit cube")

    session.endedAt = start.addingTimeInterval(42.5)
    try BundleJSON.writeDocument(session, to: dir.appending(path: BundleLayout.sessionFile))

    print("wrote sample bundle to \(dir.path(percentEncoded: false))")
} catch {
    FileHandle.standardError.write(Data("bundlekit-sample: \(error)\n".utf8))
    exit(1)
}
