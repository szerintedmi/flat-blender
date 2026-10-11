import Foundation

/// File and directory names inside a `<session>/` bundle directory.
/// Mirrors the `*_FILE` / `*_DIR` constants in `pipeline/scanmodel/models/bundle.py`.
public enum BundleLayout {
    public static let format = "flat-bundle"
    public static let version = 1

    public static let sessionFile = "session.json"
    public static let framesFile = "frames.jsonl"
    public static let stillsFile = "stills.jsonl"
    public static let videoFile = "video.mov"
    public static let depthDir = "depth"
    public static let confidenceDir = "confidence"
    public static let stillsDir = "stills"
    public static let calibrationFile = "calibration.json"
    public static let meshFile = "mesh.ply"
    public static let roomJSONFile = "room.json"
    public static let roomUSDZFile = "room.usdz"
    public static let worldMapFile = "worldmap.arworldmap"
    public static let anchorsFile = "anchors.jsonl"

    /// Zero-padded six-digit number, e.g. `000123`.
    public static func padded(_ n: Int) -> String {
        let s = String(n)
        return s.count >= 6 ? s : String(repeating: "0", count: 6 - s.count) + s
    }

    /// `depth/000123.png`, relative to the session dir.
    public static func depthPath(frame idx: Int) -> String { "\(depthDir)/\(padded(idx)).png" }

    /// `confidence/000123.png`, relative to the session dir.
    public static func confidencePath(frame idx: Int) -> String { "\(confidenceDir)/\(padded(idx)).png" }

    /// `stills/000001.jpg`, relative to the session dir.
    public static func stillPath(id: Int) -> String { "\(stillsDir)/\(padded(id)).jpg" }

    /// What an anchor FlatBlender added marks: a still, or the 1 s tick.
    public enum AnchorKind: String, Sendable {
        case still
        case tick
    }

    /// `ARAnchor.name` of an anchor FlatBlender added: `<session_id>/still/<still id>` or `<session_id>/tick/<n>`.
    public static func anchorName(sessionId: String, kind: AnchorKind, number: Int) -> String {
        "\(sessionId)/\(kind.rawValue)/\(number)"
    }

    /// True for names made by `anchorName` (any session's); false for RoomPlan's and ARKit's own anchors.
    public static func isAnchorName(_ name: String?) -> Bool {
        guard let name else { return false }
        let parts = name.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 3 && !parts[0].isEmpty && AnchorKind(rawValue: String(parts[1])) != nil
            && Int(parts[2]) != nil
    }

    /// Creates the session dir and its `depth/`, `confidence/` and `stills/` subdirectories.
    public static func createDirectories(at sessionDir: URL) throws {
        let fm = FileManager.default
        for sub in ["", depthDir, confidenceDir, stillsDir] {
            let url = sub.isEmpty ? sessionDir : sessionDir.appending(path: sub, directoryHint: .isDirectory)
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
    }
}
