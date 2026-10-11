import ARKit
import Foundation

/// Saving and loading `worldmap.arworldmap` (NSKeyedArchiver, secure coding).
///
/// Nonisolated on purpose: the ARKit completion closure formed here is not main-actor isolated, so ARKit
/// may call it on any queue.
enum WorldMapIO {
    /// The archived map, and our anchors in it (for anchors.jsonl `final` lines: one fetch for both).
    struct Saved: Sendable {
        let data: Data
        let anchors: [AnchorLog.Snapshot]
    }

    static func archiveCurrent(
        of session: ARSession, completion: @escaping @Sendable (Result<Saved, any Error>) -> Void
    ) {
        session.getCurrentWorldMap { map, error in
            guard let map else {
                completion(.failure(error ?? WorldMapError.unavailable))
                return
            }
            do {
                let data = try NSKeyedArchiver.archivedData(withRootObject: map, requiringSecureCoding: true)
                completion(.success(Saved(data: data, anchors: map.anchors.compactMap(AnchorLog.Snapshot.init))))
            } catch {
                completion(.failure(error))
            }
        }
    }

    static func load(from url: URL) throws -> ARWorldMap {
        let data = try Data(contentsOf: url)
        guard let map = try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data) else {
            throw WorldMapError.unreadable
        }
        return map
    }
}

enum WorldMapError: LocalizedError {
    case unavailable
    case unreadable
    case timedOut

    var errorDescription: String? {
        switch self {
        case .unavailable: "ARKit has no world map yet (walk around a little more)."
        case .unreadable: "The world map file could not be read."
        case .timedOut: "ARKit did not deliver the world map within 30 s."
        }
    }
}

/// Resumes a continuation once; later calls are ignored (whichever of ARKit's answer and the timeout comes second).
final class OnceResume<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var cont: CheckedContinuation<T, any Error>?

    init(_ cont: CheckedContinuation<T, any Error>) { self.cont = cont }

    func resume(_ result: Result<T, any Error>) {
        let c = lock.withLock { () -> CheckedContinuation<T, any Error>? in
            defer { cont = nil }
            return cont
        }
        c?.resume(with: result)
    }
}
