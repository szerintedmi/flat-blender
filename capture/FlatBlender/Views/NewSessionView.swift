import BundleKit
import SwiftUI

/// Sheet to start a session: room name, session type, and whether to continue an earlier world map.
struct NewSessionView: View {
    let project: ProjectInfo
    let latestMap: SessionSummary?
    let onStart: (SessionRequest) -> Void

    @Environment(ProjectStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var roomName = ""
    @State private var type: SessionType = .scan
    @State private var continueMap = true
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Room name, e.g. hall", text: $roomName)
                        .textInputAutocapitalization(.never)
                } header: {
                    Text("Room")
                } footer: {
                    Text("One room per session: tap Next room in each doorway while recording.")
                }
                Section {
                    Picker("Type", selection: $type) {
                        Text("Scan: stream + depth + RoomPlan + auto stills").tag(SessionType.scan)
                        Text("Texture: find the position, then close-up stills (no video)").tag(SessionType.texture)
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    Label("Ultra-wide (AVFoundation, no poses): TODO, phase 5", systemImage: "camera.aperture")
                        .foregroundStyle(.secondary)
                } header: {
                    Text("Session type")
                }
                Section {
                    if let latestMap {
                        Toggle(isOn: type == .texture ? .constant(true) : $continueMap) {
                            VStack(alignment: .leading) {
                                Text("Continue world map")
                                Text("From \(latestMap.session?.roomName ?? latestMap.id) (\(latestMap.id)). Start in a doorway that overlaps that room.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        Text(type == .texture
                            ? "Texture sessions need a world map: scan the room first."
                            : "No earlier world map in this project: this session starts a new one.")
                            .foregroundStyle(type == .texture ? .red : .secondary)
                    }
                } header: {
                    Text("World map")
                }
                if let error {
                    Text(error).foregroundStyle(.red)
                }
            }
            .navigationTitle("New session")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Start") { start() }.disabled(type == .texture && latestMap == nil)
                }
            }
            .onAppear {
                // Nothing to continue from in a fresh project.
                if latestMap == nil { continueMap = false }
            }
        }
    }

    private func start() {
        do {
            let room = roomName.trimmingCharacters(in: .whitespacesAndNewlines)
            let (id, dir) = try store.makeSessionDir(in: project, roomName: room)
            onStart(
                SessionRequest(
                    project: project, sessionId: id, dir: dir, type: type, roomName: room,
                    worldMapFrom: continueMap || type == .texture ? latestMap : nil, settings: .current()))
        } catch {
            self.error = error.localizedDescription
        }
    }
}
