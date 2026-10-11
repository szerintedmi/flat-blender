import ARKit
import BundleKit
import CoreVideo
import Foundation

/// Wraps a value that is not `Sendable` (e.g. a `CVPixelBuffer`) for a hand-off to exactly one other queue.
/// The sender must not touch the value after handing it off.
struct Handoff<T>: @unchecked Sendable {
    let value: T
}

extension BundleKit.TrackingState {
    init(_ state: ARCamera.TrackingState) {
        switch state {
        case .normal: self = .normal
        case .notAvailable: self = .notAvailable
        case .limited(let reason):
            switch reason {
            case .initializing: self = .limitedInitializing
            case .excessiveMotion: self = .limitedExcessiveMotion
            case .insufficientFeatures: self = .limitedInsufficientFeatures
            case .relocalizing: self = .limitedRelocalizing
            @unknown default: self = .notAvailable
            }
        }
    }

    var label: String {
        switch self {
        case .normal: "Normal"
        case .limitedInitializing: "Initialising"
        case .limitedExcessiveMotion: "Too fast"
        case .limitedInsufficientFeatures: "Low detail"
        case .limitedRelocalizing: "Relocalising"
        case .notAvailable: "Not available"
        }
    }
}

extension ARCamera {
    /// Intrinsics at `imageResolution` (the video resolution for stream frames, the still resolution for
    /// high-resolution frames).
    var bundleIntrinsics: Intrinsics {
        Intrinsics(intrinsics, width: Int(imageResolution.width), height: Int(imageResolution.height))
    }

    func bundleExposure(iso: Float?) -> Exposure {
        Exposure(
            durationS: exposureDuration, offsetEv: Double(exposureOffset),
            iso: iso.map { shortestDouble($0) })
    }
}

extension ARFrame.WorldMappingStatus {
    var label: String {
        switch self {
        case .notAvailable: "none"
        case .limited: "limited"
        case .extending: "extending"
        case .mapped: "mapped"
        @unknown default: "?"
        }
    }
}

/// Copies ARKit's camera images into buffers we own, so ARKit's small capture buffer pool is released as
/// soon as the delegate callback returns (holding ARKit buffers stalls the camera).
final class PixelBufferCopier {
    private var pool: CVPixelBufferPool?
    private var poolKey: (OSType, Int, Int)?

    func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let format = CVPixelBufferGetPixelFormatType(src)
        let w = CVPixelBufferGetWidth(src)
        let h = CVPixelBufferGetHeight(src)
        if pool == nil || poolKey.map({ $0 != (format, w, h) }) == true {
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: format,
                kCVPixelBufferWidthKey: w,
                kCVPixelBufferHeightKey: h,
                kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any](),
            ]
            var newPool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &newPool)
            pool = newPool
            poolKey = (format, w, h)
        }
        guard let pool else { return nil }
        var dstOut: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dstOut) == kCVReturnSuccess, let dst = dstOut else {
            return nil
        }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        if CVPixelBufferIsPlanar(src) {
            for plane in 0..<CVPixelBufferGetPlaneCount(src) {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                    let d = CVPixelBufferGetBaseAddressOfPlane(dst, plane)
                else { return nil }
                copyRows(
                    s, CVPixelBufferGetBytesPerRowOfPlane(src, plane), d, CVPixelBufferGetBytesPerRowOfPlane(dst, plane),
                    rows: CVPixelBufferGetHeightOfPlane(src, plane))
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            copyRows(s, CVPixelBufferGetBytesPerRow(src), d, CVPixelBufferGetBytesPerRow(dst), rows: h)
        }
        // Colour attachments (YCbCr matrix, transfer function) so the encoder tags the video correctly.
        CVBufferPropagateAttachments(src, dst)
        return dst
    }

    private func copyRows(_ s: UnsafeRawPointer, _ sStride: Int, _ d: UnsafeMutableRawPointer, _ dStride: Int, rows: Int) {
        if sStride == dStride {
            memcpy(d, s, sStride * rows)
        } else {
            let n = min(sStride, dStride)
            for r in 0..<rows { memcpy(d + r * dStride, s + r * sStride, n) }
        }
    }
}

/// A depth + confidence map copied out of an `ARDepthData` (256x192 on the M2 iPad Pro).
struct DepthCopy: Sendable {
    let width: Int
    let height: Int
    let millimetres: [UInt16]
    let confidence: [UInt8]?

    init?(_ data: ARDepthData) {
        let depth = data.depthMap
        guard CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32 else { return nil }
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let w = CVPixelBufferGetWidth(depth)
        let h = CVPixelBufferGetHeight(depth)
        width = w
        height = h
        millimetres = DepthPNG.millimetres(
            fromMetres: base, width: w, height: h, bytesPerRow: CVPixelBufferGetBytesPerRow(depth))
        confidence = data.confidenceMap.flatMap { Self.copyConfidence($0, width: w, height: h) }
    }

    private static func copyConfidence(_ conf: CVPixelBuffer, width: Int, height: Int) -> [UInt8]? {
        guard CVPixelBufferGetPixelFormatType(conf) == kCVPixelFormatType_OneComponent8,
            CVPixelBufferGetWidth(conf) == width, CVPixelBufferGetHeight(conf) == height
        else { return nil }
        CVPixelBufferLockBaseAddress(conf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(conf) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(conf)
        var out = [UInt8](repeating: 0, count: width * height)
        out.withUnsafeMutableBytes { dst in
            for y in 0..<height {
                memcpy(dst.baseAddress! + y * width, base + y * stride, width)
            }
        }
        return out
    }
}
