import Foundation

/// Zips a session or project folder for AirDrop / Files. `NSFileCoordinator` with `.forUploading` produces a zip of a
/// directory; we move it out of the coordinator's temporary location before it is deleted.
enum SessionExporter {
    /// Returns `tmp/<project>_<session>.zip`. Runs off the main thread (bundles are 1-3 GB).
    static func zip(sessionDir: URL, projectId: String) async throws -> URL {
        try await zip(dir: sessionDir, name: "\(projectId)_\(sessionDir.lastPathComponent).zip")
    }

    /// Returns `tmp/<project>.zip` with project.json and every session; it unzips to
    /// `<project>/<session>/`, which the pipeline reads from `data/<project>/raw/`.
    static func zip(projectDir: URL) async throws -> URL {
        try await zip(dir: projectDir, name: "\(projectDir.lastPathComponent).zip")
    }

    private static func zip(dir sessionDir: URL, name: String) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            let dest = FileManager.default.temporaryDirectory.appending(path: name)
            try? FileManager.default.removeItem(at: dest)

            var coordinatorError: NSError?
            var copyError: (any Error)?
            NSFileCoordinator().coordinate(readingItemAt: sessionDir, options: .forUploading, error: &coordinatorError) { zipURL in
                do {
                    try FileManager.default.moveItem(at: zipURL, to: dest)
                } catch {
                    do {
                        try FileManager.default.copyItem(at: zipURL, to: dest)
                    } catch {
                        copyError = error
                    }
                }
            }
            if let coordinatorError { throw coordinatorError }
            if let copyError { throw copyError }
            return dest
        }.value
    }

    /// Deletes zips left in tmp by earlier exports.
    static func cleanUp() {
        let tmp = FileManager.default.temporaryDirectory
        let items = (try? FileManager.default.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil)) ?? []
        for url in items where url.pathExtension == "zip" { try? FileManager.default.removeItem(at: url) }
    }
}
