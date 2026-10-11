import BundleKit
import Foundation
import Observation

/// `project.json`: one project = one flat. App-owned, not part of the bundle contract (the pipeline reads
/// only the session bundles inside the project folder).
struct ProjectInfo: Codable, Sendable, Identifiable, Hashable {
    var format = "flatblender-project"
    var version = 1
    var projectId: String
    var name: String
    var createdAt: Date

    var id: String { projectId }

    private enum CodingKeys: String, CodingKey {
        case format, version
        case projectId = "project_id"
        case name
        case createdAt = "created_at"
    }
}

/// A session folder as listed in the project screen.
struct SessionSummary: Identifiable, Sendable, Hashable {
    let id: String  // session id == folder name
    let dir: URL
    let session: BundleKit.Session?  // nil if session.json is missing or unreadable
    let hasWorldMap: Bool
    let stillCount: Int

    var startedAt: Date { session?.startedAt ?? .distantPast }

    static func == (a: SessionSummary, b: SessionSummary) -> Bool { a.id == b.id && a.dir == b.dir }
    func hash(into h: inout Hasher) { h.combine(id) }
}

/// Plain-folder persistence: `Documents/Projects/<project_id>/project.json` and
/// `Documents/Projects/<project_id>/<session_id>/` bundle directories. No database: the Files app and
/// AirDrop see exactly what the pipeline will read.
@MainActor
@Observable
final class ProjectStore {
    private(set) var projects: [ProjectInfo] = []
    private(set) var sessions: [String: [SessionSummary]] = [:]
    var lastError: String?

    @ObservationIgnored let root: URL

    init(root: URL? = nil) {
        self.root = root ?? URL.documentsDirectory.appending(path: "Projects", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: self.root, withIntermediateDirectories: true)
        reload()
    }

    func projectDir(_ p: ProjectInfo) -> URL { root.appending(path: p.projectId, directoryHint: .isDirectory) }

    func reload() {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        projects = dirs.compactMap { dir in
            try? BundleJSON.readDocument(ProjectInfo.self, from: dir.appending(path: "project.json"))
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    @discardableResult
    func createProject(name: String) throws -> ProjectInfo {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = Self.slug(trimmed.isEmpty ? "flat" : trimmed)
        var id = base
        var n = 2
        while FileManager.default.fileExists(atPath: root.appending(path: id).path(percentEncoded: false)) {
            id = "\(base)-\(n)"
            n += 1
        }
        let info = ProjectInfo(projectId: id, name: trimmed.isEmpty ? id : trimmed, createdAt: .now)
        let dir = projectDir(info)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try BundleJSON.writeDocument(info, to: dir.appending(path: "project.json"))
        reload()
        return info
    }

    /// Deleting a scan's folder (thousands of depth PNGs, GBs of video) takes seconds: off the main thread. The row
    /// goes at once.
    func deleteProject(_ p: ProjectInfo) {
        let dir = projectDir(p)
        projects.removeAll { $0.projectId == p.projectId }
        sessions[p.projectId] = nil
        Task {
            if let error = await Self.remove(dir) { lastError = "Could not delete project: \(error)" }
            reload()
        }
    }

    nonisolated private static func remove(_ dir: URL) async -> String? {
        await Task.detached(priority: .userInitiated) { () -> String? in
            do {
                try FileManager.default.removeItem(at: dir)
                return nil
            } catch {
                return error.localizedDescription
            }
        }.value
    }

    /// Re-reads a project's session folders off the main thread (every session.json and stills folder: scan 5's
    /// project has 6 sessions and 310 stills) and publishes the list when done.
    func reloadSessions(_ p: ProjectInfo) {
        let dir = projectDir(p)
        let id = p.projectId
        Task {
            let list = await Task.detached(priority: .userInitiated) { Self.scanSessions(in: dir) }.value
            sessions[id] = list
        }
    }

    nonisolated static func scanSessions(in dir: URL) -> [SessionSummary] {
        let fm = FileManager.default
        let children = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        return children.compactMap { url -> SessionSummary? in
            guard (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return nil }
            let session = try? BundleJSON.readDocument(
                BundleKit.Session.self, from: url.appending(path: BundleLayout.sessionFile))
            let stills = (try? fm.contentsOfDirectory(atPath: url.appending(path: BundleLayout.stillsDir).path(percentEncoded: false)))?.count ?? 0
            return SessionSummary(
                id: url.lastPathComponent, dir: url, session: session,
                hasWorldMap: fm.fileExists(atPath: url.appending(path: BundleLayout.worldMapFile).path(percentEncoded: false)),
                stillCount: stills)
        }
        .sorted { $0.startedAt > $1.startedAt }
    }

    func deleteSession(_ s: SessionSummary, in p: ProjectInfo) {
        sessions[p.projectId]?.removeAll { $0.id == s.id }
        Task {
            if let error = await Self.remove(s.dir) { lastError = "Could not delete session: \(error)" }
            reloadSessions(p)
        }
    }

    /// The world map to continue from: the most recent finished *scan* or *lap* session that saved one (after
    /// Next room and the closing lap, the last session's map covers every room of that walk). Texture
    /// sessions' maps cover only what they looked at, so they are a fallback only.
    func latestWorldMapSession(in p: ProjectInfo) -> SessionSummary? {
        if sessions[p.projectId] == nil { sessions[p.projectId] = Self.scanSessions(in: projectDir(p)) }
        let finished = (sessions[p.projectId] ?? []).filter { $0.hasWorldMap && $0.session?.endedAt != nil }
        return finished.first { $0.session?.type == .scan || $0.session?.type == .lap } ?? finished.first
    }

    /// Creates `<project>/<session_id>/` with its subfolders. Session ids are `yyyyMMdd-HHmmss-<room>`.
    func makeSessionDir(in p: ProjectInfo, roomName: String, date: Date = .now) throws -> (id: String, dir: URL) {
        try Self.makeSessionDir(projectDir: projectDir(p), roomName: roomName, date: date)
    }

    nonisolated static func makeSessionDir(projectDir: URL, roomName: String, date: Date = .now) throws -> (id: String, dir: URL) {
        let c = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute, .second], from: date)
        func two(_ v: Int?) -> String { String(format: "%02d", v ?? 0) }
        var id = "\(c.year ?? 0)\(two(c.month))\(two(c.day))-\(two(c.hour))\(two(c.minute))\(two(c.second))"
        let room = Self.slug(roomName)
        if !room.isEmpty { id += "-\(room)" }
        let dir = projectDir.appending(path: id, directoryHint: .isDirectory)
        guard !FileManager.default.fileExists(atPath: dir.path(percentEncoded: false)) else {
            throw CocoaError(.fileWriteFileExists, userInfo: [NSFilePathErrorKey: dir.path(percentEncoded: false)])
        }
        try BundleLayout.createDirectories(at: dir)
        return (id, dir)
    }

    /// Lowercase ASCII letters, digits and single dashes, at most 32 characters.
    nonisolated static func slug(_ s: String) -> String {
        var out = ""
        var dash = false
        for ch in s.lowercased().unicodeScalars {
            if ("a"..."z").contains(ch) || ("0"..."9").contains(ch) {
                out.unicodeScalars.append(ch)
                dash = false
            } else if !dash && !out.isEmpty {
                out.append("-")
                dash = true
            }
        }
        while out.hasSuffix("-") { out.removeLast() }
        return String(out.prefix(32))
    }
}
