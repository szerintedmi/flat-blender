import BundleKit
import SwiftUI

/// One project: its sessions (bundle folders), new session, export, delete.
struct ProjectDetailView: View {
    let project: ProjectInfo

    @Environment(ProjectStore.self) private var store
    @State private var showingNewSession = false
    @State private var capture: ActiveCapture?
    @State private var exporting: String?
    @State private var shareItem: ShareItem?
    @State private var pendingDelete: SessionSummary?

    private var sessions: [SessionSummary] { store.sessions[project.projectId] ?? [] }

    var body: some View {
        List {
            Section {
                if sessions.isEmpty {
                    Text("No sessions yet. Start with a scan session in the first room.")
                        .foregroundStyle(.secondary)
                }
                ForEach(sessions) { s in
                    SessionRow(summary: s, exporting: exporting == s.id)
                        .swipeActions {
                            Button("Delete", role: .destructive) { pendingDelete = s }
                            Button("Export") { export(s) }.tint(.blue)
                        }
                        .contextMenu {
                            Button("Export (AirDrop / Files)", systemImage: "square.and.arrow.up") { export(s) }
                            Button("Delete", systemImage: "trash", role: .destructive) { pendingDelete = s }
                        }
                }
            } header: {
                HStack {
                    Text("Sessions")
                    if exporting == project.projectId {
                        ProgressView().controlSize(.small)
                        Text("Zipping all…")
                    }
                }
            } footer: {
                Text("Folder: Files › On My iPad › FlatBlender › Projects › \(project.projectId)")
            }
        }
        .navigationTitle(project.name)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Export all", systemImage: "square.and.arrow.up.on.square") { exportAll() }
                    .disabled(sessions.isEmpty || exporting != nil)
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button("New session", systemImage: "plus.viewfinder") { showingNewSession = true }
            }
        }
        .task { store.reloadSessions(project) }
        .refreshable { store.reloadSessions(project) }
        .sheet(isPresented: $showingNewSession) {
            NewSessionView(project: project, latestMap: store.latestWorldMapSession(in: project)) { request in
                showingNewSession = false
                capture = ActiveCapture(engine: CaptureEngine(request: request))
            }
        }
        .fullScreenCover(item: $capture, onDismiss: { store.reloadSessions(project) }) { active in
            CaptureView(engine: active.engine)
        }
        .sheet(item: $shareItem) { item in
            ShareSheet(items: [item.url])
        }
        .confirmationDialog(
            "Delete session \(pendingDelete?.id ?? "")?", isPresented: .constant(pendingDelete != nil),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let s = pendingDelete { store.deleteSession(s, in: project) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        }
    }

    /// One zip of the whole project folder (every session), for `data/<project>/raw/` on the Mac.
    private func exportAll() {
        guard exporting == nil else { return }
        exporting = project.projectId
        Task {
            SessionExporter.cleanUp()
            do {
                shareItem = ShareItem(url: try await SessionExporter.zip(projectDir: store.projectDir(project)))
            } catch {
                store.lastError = "Export failed: \(error.localizedDescription)"
            }
            exporting = nil
        }
    }

    private func export(_ s: SessionSummary) {
        guard exporting == nil else { return }
        exporting = s.id
        Task {
            SessionExporter.cleanUp()
            do {
                let url = try await SessionExporter.zip(sessionDir: s.dir, projectId: project.projectId)
                shareItem = ShareItem(url: url)
            } catch {
                store.lastError = "Export failed: \(error.localizedDescription)"
            }
            exporting = nil
        }
    }
}

struct ActiveCapture: Identifiable {
    let id = UUID()
    let engine: CaptureEngine
}

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

private struct SessionRow: View {
    let summary: SessionSummary
    let exporting: Bool

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(summary.session?.roomName ?? summary.id).font(.headline)
                    if let type = summary.session?.type {
                        Text(type.rawValue)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Self.color(type).opacity(0.2), in: Capsule())
                    }
                }
                Text(details).font(.caption).foregroundStyle(.secondary)
                if let notes = summary.session?.notes, !notes.isEmpty {
                    Text(notes).font(.caption2).foregroundStyle(.secondary).lineLimit(3)
                }
            }
            Spacer()
            if exporting {
                ProgressView().controlSize(.small)
                Text("Zipping…").font(.caption)
            }
            if summary.hasWorldMap {
                Image(systemName: "map").foregroundStyle(.secondary).help("Has a world map")
            }
        }
    }

    private static func color(_ type: SessionType) -> Color {
        switch type {
        case .scan: .blue
        case .lap: .green
        case .texture: .orange
        case .ultrawide: .purple
        }
    }

    private var details: String {
        guard let s = summary.session else { return "\(summary.id) · session.json missing or unreadable" }
        var parts = [summary.id, s.startedAt.formatted(date: .abbreviated, time: .shortened)]
        if let end = s.endedAt {
            let d = Int(end.timeIntervalSince(s.startedAt))
            parts.append(String(format: "%d:%02d min", d / 60, d % 60))
        } else {
            parts.append("not finished")
        }
        parts.append("\(summary.stillCount) stills")
        if let from = s.worldMapIn { parts.append("map from \(from)") }
        return parts.joined(separator: " · ")
    }
}
