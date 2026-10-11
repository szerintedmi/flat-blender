import Foundation

/// JSON encoders/decoders configured for the bundle format.
/// Dates are ISO 8601 UTC with milliseconds (`2026-10-06T17:45:12.123Z`), which pydantic parses as `datetime`.
public enum BundleJSON {
    private static let dateStyle = Date.ISO8601FormatStyle(includingFractionalSeconds: true)
    private static let dateStyleNoFraction = Date.ISO8601FormatStyle()

    /// Compact single-line encoder for JSONL lines.
    public static func lineEncoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
        e.dateEncodingStrategy = .custom { date, encoder in
            var c = encoder.singleValueContainer()
            try c.encode(date.formatted(dateStyle))
        }
        return e
    }

    /// Pretty encoder for `session.json` and other small documents.
    public static func documentEncoder() -> JSONEncoder {
        let e = lineEncoder()
        e.outputFormatting = [.withoutEscapingSlashes, .prettyPrinted, .sortedKeys]
        return e
    }

    public static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .custom { decoder in
            let c = try decoder.singleValueContainer()
            let s = try c.decode(String.self)
            if let date = try? dateStyle.parse(s) { return date }
            if let date = try? dateStyleNoFraction.parse(s) { return date }
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "Not an ISO 8601 date: \(s)")
        }
        return d
    }

    /// Writes a document atomically (temp file + rename), so a crash never leaves a half-written session.json.
    public static func writeDocument<T: Encodable>(_ value: T, to url: URL) throws {
        var data = try documentEncoder().encode(value)
        data.append(0x0A)
        try data.write(to: url, options: .atomic)
    }

    public static func readDocument<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try decoder().decode(type, from: Data(contentsOf: url))
    }
}

/// Appends one JSON object per line to a `.jsonl` file.
///
/// Each line goes to the file in a single `write(2)` with no user-space buffering, so every completed line is
/// in the OS page cache immediately and survives an app crash; a crash can at most lose the line in flight.
/// Call `sync()` for an `fsync` (e.g. at the end of a session) if you also need power-loss durability.
///
/// Not thread-safe: confine an instance to one serial queue.
public final class JSONLWriter {
    public let url: URL
    private let handle: FileHandle
    private let encoder = BundleJSON.lineEncoder()
    public private(set) var lineCount = 0
    private var closed = false

    /// Opens (creating if needed) `url` for appending.
    public init(url: URL) throws {
        self.url = url
        if !FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            guard FileManager.default.createFile(atPath: url.path(percentEncoded: false), contents: nil) else {
                throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path(percentEncoded: false)])
            }
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    deinit {
        if !closed { try? handle.close() }
    }

    public func append<T: Encodable>(_ value: T) throws {
        var line = try encoder.encode(value)
        line.append(0x0A)
        try handle.write(contentsOf: line)
        lineCount += 1
    }

    public func sync() throws {
        try handle.synchronize()
    }

    public func close() throws {
        guard !closed else { return }
        closed = true
        try handle.synchronize()
        try handle.close()
    }
}
