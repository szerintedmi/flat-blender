import CoreGraphics
import Foundation
import ImageIO

public enum ImageEncodingError: Error, CustomStringConvertible {
    case badSize(width: Int, height: Int, count: Int)
    case cgImage
    case destination
    case finalize

    public var description: String {
        switch self {
        case let .badSize(w, h, n): "buffer has \(n) samples, expected \(w)x\(h)"
        case .cgImage: "could not create CGImage"
        case .destination: "could not create image destination"
        case .finalize: "could not finalize image"
        }
    }
}

/// Depth and confidence maps as PNG, in the bundle's conventions:
/// - depth: 16-bit greyscale, millimetres, 0 = no data (ARKit Float32 metres, rounded to 1 mm);
/// - confidence: 8-bit greyscale, 0/1/2 (`ARConfidenceLevel` low/medium/high).
///
/// Uses ImageIO with an uncalibrated DeviceGray colour space, so sample values are stored as-is (no gamma or
/// colour conversion); readers like `cv2.imread(path, cv2.IMREAD_UNCHANGED)` get the raw values back.
public enum DepthPNG {
    /// Largest depth that fits: 65.535 m.
    public static let maxMillimetres = Int(UInt16.max)

    /// Metres to millimetres. Non-finite, zero and negative values become 0 (no data); values beyond
    /// 65.535 m are clamped.
    @inlinable
    public static func millimetres(fromMetres m: Float) -> UInt16 {
        guard m.isFinite, m > 0 else { return 0 }
        let mm = (m * 1000).rounded()
        return mm >= Float(UInt16.max) ? UInt16.max : UInt16(mm)
    }

    /// Converts a tightly packed or strided Float32 depth map (metres) to millimetres.
    /// - Parameter bytesPerRow: row stride of `base` in bytes (CVPixelBuffer rows may be padded).
    public static func millimetres(
        fromMetres base: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int
    ) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: width * height)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<height {
                let row = (base + y * bytesPerRow).assumingMemoryBound(to: Float32.self)
                for x in 0..<width {
                    dst[y * width + x] = millimetres(fromMetres: row[x])
                }
            }
        }
        return out
    }

    public static func millimetres(fromMetres metres: [Float]) -> [UInt16] {
        metres.map { millimetres(fromMetres: $0) }
    }

    /// 16-bit greyscale PNG from millimetres.
    public static func encodeDepth(millimetres mm: [UInt16], width: Int, height: Int) throws -> Data {
        guard mm.count == width * height else { throw ImageEncodingError.badSize(width: width, height: height, count: mm.count) }
        // Host is little-endian on every Apple platform we target; say so explicitly.
        let bytes = mm.withUnsafeBytes { Data($0) }
        return try encodeGray(bytes, width: width, height: height, bitsPerComponent: 16)
    }

    /// 16-bit greyscale PNG from Float32 metres.
    public static func encodeDepth(metres: [Float], width: Int, height: Int) throws -> Data {
        try encodeDepth(millimetres: millimetres(fromMetres: metres), width: width, height: height)
    }

    /// 8-bit greyscale PNG of ARKit confidence levels (0/1/2).
    public static func encodeConfidence(_ levels: [UInt8], width: Int, height: Int) throws -> Data {
        guard levels.count == width * height else {
            throw ImageEncodingError.badSize(width: width, height: height, count: levels.count)
        }
        return try encodeGray(Data(levels), width: width, height: height, bitsPerComponent: 8)
    }

    /// Writes PNG data to `url` (no atomic rename: depth files are written once and never replaced).
    public static func write(_ png: Data, to url: URL) throws {
        try png.write(to: url)
    }

    static func encodeGray(_ bytes: Data, width: Int, height: Int, bitsPerComponent: Int) throws -> Data {
        let bytesPerRow = width * bitsPerComponent / 8
        var info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        if bitsPerComponent == 16 { info.insert(.byteOrder16Little) }
        guard let provider = CGDataProvider(data: bytes as CFData),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: bitsPerComponent, bitsPerPixel: bitsPerComponent,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: info,
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw ImageEncodingError.cgImage }
        return try ImageWriter.encode(image, type: "public.png", properties: [:])
    }
}

/// Small ImageIO wrapper used for PNGs (and JPEGs in the sample generator / tests).
public enum ImageWriter {
    /// Encodes a CGImage. `type` is a UTI string such as `public.png` or `public.jpeg`.
    public static func encode(_ image: CGImage, type: String, properties: [CFString: Any]) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data as CFMutableData, type as CFString, 1, nil) else {
            throw ImageEncodingError.destination
        }
        CGImageDestinationAddImage(dest, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw ImageEncodingError.finalize }
        return data as Data
    }

    /// JPEG with the given quality (0...1).
    public static func jpeg(_ image: CGImage, quality: Double) throws -> Data {
        try encode(image, type: "public.jpeg", properties: [kCGImageDestinationLossyCompressionQuality: quality])
    }
}
