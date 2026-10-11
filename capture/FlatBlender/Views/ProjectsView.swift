import SwiftUI

/// Home screen: the list of projects (one per flat).
struct ProjectsView: View {
    @Environment(ProjectStore.self) private var store
    @State private var showingNew = false
    @State private var newName = ""
    @State private var showingSettings = false
    @State private var pendingDelete: ProjectInfo?

    var body: some View {
        NavigationStack {
            List {
                if store.projects.isEmpty {
                    ContentUnavailableView(
                        "No projects", systemImage: "house",
                        description: Text("A project is one flat. Create one, then add a session per room."))
                }
                ForEach(store.projects) { project in
                    NavigationLink(value: project) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(project.name).font(.headline)
                            Text("\(project.projectId) · created \(project.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        Button("Delete", role: .destructive) { pendingDelete = project }
                    }
                }
            }
            .navigationTitle("FlatBlender")
            .navigationDestination(for: ProjectInfo.self) { ProjectDetailView(project: $0) }
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Settings", systemImage: "gearshape") { showingSettings = true }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("New project", systemImage: "plus") {
                        newName = ""
                        showingNew = true
                    }
                }
            }
            .refreshable { store.reload() }
            .alert("New project", isPresented: $showingNew) {
                TextField("Name, e.g. Flat 3B", text: $newName)
                Button("Create") {
                    do {
                        try store.createProject(name: newName)
                    } catch {
                        store.lastError = error.localizedDescription
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            .confirmationDialog(
                "Delete \(pendingDelete?.name ?? "")?", isPresented: .constant(pendingDelete != nil),
                titleVisibility: .visible
            ) {
                Button("Delete project and all its sessions", role: .destructive) {
                    if let p = pendingDelete { store.deleteProject(p) }
                    pendingDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingDelete = nil }
            }
            .alert("Error", isPresented: .constant(store.lastError != nil)) {
                Button("OK") { store.lastError = nil }
            } message: {
                Text(store.lastError ?? "")
            }
            .sheet(isPresented: $showingSettings) { SettingsView() }
        }
    }
}
