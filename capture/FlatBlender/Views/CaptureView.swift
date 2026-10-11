import ARKit
import BundleKit
import RealityKit
import SwiftUI

/// Full-screen capture: live camera (+ optional live mesh), HUD, and the session controls
/// Lock exposure → Record → (Next room: saved in the background while naming the next, Record + doorway sweep …)
/// → Finish (→ closing lap → Record → Finish).
struct CaptureView: View {
    let engine: CaptureEngine

    @Environment(\.dismiss) private var dismiss
    @AppStorage(SettingKey.showMesh) private var showMesh = true
    @AppStorage(SettingKey.showCoverage) private var showCoverage = true
    @AppStorage(SettingKey.showCoverageMap) private var showCoverageMap = true
    @State private var confirmDiscard = false
    @State private var askLap = false

    var body: some View {
        ZStack {
            ARViewContainer(session: engine.arSession, showMesh: showMesh, coverage: engine.coverage, showCoverage: showCoverage)
                .ignoresSafeArea()

            // Coverage mini-map and the dots' key: top right, left of the controls column (the dots themselves are
            // in the AR view).
            if showCoverage && engine.phase == .recording {
                VStack(alignment: .leading, spacing: 6) {
                    if showCoverageMap { CoverageMiniMap(overlay: engine.coverage) }
                    CoverageLegend()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .padding(.top, 16)
                .padding(.trailing, 162)
            }

            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    HUDView(engine: engine)
                    banners
                    Spacer()
                    MotionHint(engine: engine)
                    if let started = engine.sweepStartedAt {
                        SweepPanel(started: started, progress: engine.sweepProgress) { engine.endSweep() }
                    } else if engine.phase == .ready && engine.sweepPending {
                        NextRoomReadyPanel(engine: engine)
                    } else if engine.phase == .ready && engine.request.type == .lap {
                        LapReadyPanel(engine: engine)
                    } else if let hint {
                        Text(hint).font(.callout).padding(8).background(.ultraThinMaterial, in: Capsule())
                    }
                }
                Spacer()
                controls
            }
            .padding()

            if engine.phase == .closing {
                ClosingPanel(engine: engine)
            } else if engine.phase == .finishing || engine.phase == .finished {
                finishingOverlay
            }
        }
        .statusBarHidden()
        .persistentSystemOverlays(.hidden)
        .task { await engine.start() }
        .confirmationDialog("Discard this session?", isPresented: $confirmDiscard, titleVisibility: .visible) {
            Button("Discard and delete the folder", role: .destructive) {
                Task {
                    await engine.discard()
                    dismiss()
                }
            }
            Button("Cancel", role: .cancel) {}
        }
        .confirmationDialog("Closing lap (recommended)", isPresented: $askLap, titleVisibility: .visible) {
            Button("Start lap") { engine.startLap() }
            Button("Finish without lap") { engine.finish() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Saves this room, then you tap Record and walk back through every room to where you started. Slow, chest height, look at the walls. About 1–2 min.")
        }
    }

    /// One line of guidance for the current step.
    private var hint: String? {
        switch engine.phase {
        case .ready where !engine.positioned: nil  // PositionStatus explains
        case .ready where engine.request.type == .texture: "Tap the shutter for close-ups"
        case .ready where engine.request.type == .lap: "Record, then walk back through every room to the start"
        case .ready where engine.cameraState == .metering: nil  // the metering panel explains
        case .ready: engine.cameraState == .locked ? "Start in a doorway, then Record" : "Point at a mid-bright wall, then Lock exposure"
        case .recording where engine.request.type == .scan: "Slow loops at chest height · pause to auto-shoot · Next room at each doorway"
        case .recording where engine.request.type == .lap: "Closing lap · back through every room to the start · then Finish"
        default: nil
        }
    }

    // MARK: Banners

    @ViewBuilder private var banners: some View {
        PositionStatus(engine: engine)
        #if DEBUG
        // Scan 4 (0.3.0 (6)) ran a Debug build: per-pixel code ~300x slower, stills took 2-6 s to save.
        Banner(text: "Debug build: per-pixel work runs ~300x slower. Scan with a Release build (the scheme's Run default).", color: .orange)
        #endif
        if engine.phase == .starting {
            Banner(text: !engine.progress.isEmpty ? engine.progress : engine.roomPlanActive ? "Configuring ARKit…" : "Starting ARKit\(engine.request.type == .scan && engine.request.settings.roomPlanEnabled ? " + RoomPlan" : "")…", color: .blue)
        }
        if engine.cameraState == .metering && engine.phase == .ready {
            MeteringPanel(progress: engine.meterProgress)
        }
        if engine.cameraState == .lockLost {
            Banner(text: "Camera lock was reset (configuration re-run). Tap Relock: it puts the same exposure back.", color: .red)
        }
        if let mains = engine.flickerPrompt, engine.phase == .ready, engine.cameraState == .locked {
            FlickerPromptPanel(mains: mains, accept: engine.acceptFlicker, decline: engine.declineFlicker)
        }
        if engine.cameraState == .locked, engine.phase == .ready, let under = engine.lockedCamera?.underexposureEv, under >= 1 {
            Banner(text: "Dim room: locked \(String(format: "%.1f", under)) stops dark to keep the video sharp. More light helps (then tap Locked to meter again).", color: .orange)
        }
        if engine.thermal == .serious || engine.thermal == .critical {
            Banner(text: "Device is \(engine.thermal == .critical ? "critically" : "very") hot: finish this room soon.", color: .red)
        }
        if let instruction = engine.roomPlanInstruction {
            Banner(text: "RoomPlan: \(instruction)", color: .indigo)
        }
        if let message = engine.message {
            Banner(text: message, color: .red) { engine.message = nil }
        }
    }

    // MARK: Controls

    private var controls: some View {
        VStack(spacing: 14) {
            if engine.phase == .ready || engine.phase == .starting {
                Button { confirmDiscard = true } label: { Label("Discard", systemImage: "xmark") }
                    .buttonStyle(.bordered)
            }
            exposureButton
                .disabled(!engine.canMeterOrLock)

            Spacer()

            // Texture sessions: the shutter. Scan and lap sessions shoot auto stills only (0.4.0): the ring shows
            // steadiness and flashes on each still, with the count inside.
            ZStack {
                SteadinessRing(engine: engine)
                if engine.hasShutter {
                    Button { engine.shootStill() } label: {
                        ZStack {
                            Circle().fill(.white).frame(width: 76, height: 76)
                            Circle().stroke(.black.opacity(0.3), lineWidth: 3).frame(width: 64, height: 64)
                            if engine.stillBusy { ProgressView() }
                        }
                    }
                    .disabled(!engine.canStill)
                    .opacity(engine.canStill ? 1 : 0.4)
                    .accessibilityLabel("Take still")
                } else if engine.phase == .recording {
                    VStack(spacing: 0) {
                        Text("\(engine.stillCount)").font(.title3.bold().monospacedDigit())
                        Text("auto").font(.caption2).foregroundStyle(.secondary)
                    }
                    .accessibilityLabel("\(engine.stillCount) auto stills")
                }
            }
            .frame(width: 90, height: 90)

            Spacer()

            switch engine.phase {
            case .ready where engine.request.type == .texture:
                Button { engine.finish() } label: { Label("Finish", systemImage: "stop.circle") }
                    .buttonStyle(.borderedProminent).tint(.red)
            case .ready:
                Button { engine.startRecording() } label: { Label("Record", systemImage: "record.circle") }
                    .buttonStyle(.borderedProminent).tint(.red)
                    .disabled(!engine.canRecord)
                if !engine.positioned {
                    Text("waits for position").font(.caption).foregroundStyle(.secondary)
                }
            case .recording:
                if engine.request.type == .scan {
                    // Stops this room at once (it saves while you name the next one).
                    Button { engine.nextRoom() } label: { Label("Next room", systemImage: "door.left.hand.open") }
                        .buttonStyle(.bordered)
                }
                Button {
                    // After the last room: offer the closing lap (once; Finish on the lap just finishes).
                    if engine.request.type == .scan { askLap = true } else { engine.finish() }
                } label: { Label("Finish", systemImage: "stop.circle") }
                    .buttonStyle(.borderedProminent).tint(.red)
            case .closing:
                // Recording waits for the previous room's save (and the next room's setup).
                Button {} label: { Label(closingRecordLabel, systemImage: "record.circle") }
                    .buttonStyle(.borderedProminent).tint(.red)
                    .disabled(true)
            default:
                EmptyView()
            }
        }
        .frame(width: 130)
        .controlSize(.large)
    }

    /// One exposure control (0.5.0; it was Meter, then Lock): "Lock exposure" meters and locks by itself once
    /// auto exposure settles (tap again to lock at once); green "Locked" with the values (tap to meter again);
    /// orange "Relock" when ARKit reset the lock (puts the same values back).
    @ViewBuilder private var exposureButton: some View {
        switch engine.cameraState {
        case .auto:
            Button { engine.exposureTapped() } label: { Label("Lock exposure", systemImage: "lock.open") }
                .buttonStyle(.bordered)
        case .metering:
            Button { engine.exposureTapped() } label: {
                VStack(spacing: 4) {
                    Label("Metering…", systemImage: "camera.metering.center.weighted")
                    ProgressView(value: engine.meterProgress).tint(.black)
                    Text("tap to lock now").font(.caption2)
                }
            }
            .buttonStyle(.borderedProminent).tint(.yellow).foregroundStyle(.black)
        case .locked:
            Button { engine.exposureTapped() } label: {
                VStack(spacing: 2) {
                    Label("Locked", systemImage: "lock.fill")
                    if let values = Self.lockValues(engine.lockedCamera) { Text(values).font(.caption2.monospacedDigit()) }
                }
            }
            .buttonStyle(.borderedProminent).tint(.green)
        case .lockLost:
            Button { engine.exposureTapped() } label: { Label("Relock", systemImage: "lock.trianglebadge.exclamationmark") }
                .buttonStyle(.borderedProminent).tint(.orange)
        }
    }

    private var closingRecordLabel: String {
        switch engine.closeStatus {
        case .saving, .failed, nil: "Saving previous room…"
        case .preparing, .setupFailed: "Preparing…"
        case .ready: "Name the room"
        }
    }

    static func lockValues(_ c: LockedCamera?) -> String? {
        guard let c, let d = c.exposureDurationS, let iso = c.iso else { return nil }
        return "1/\(Int((1 / d).rounded())) s · ISO \(Int(iso))"
    }

    private var finishingOverlay: some View {
        VStack(spacing: 16) {
            if case .failed(let errors) = engine.closeStatus {
                CloseFailure(errors: errors)
                HStack {
                    Button("Retry") { engine.retryClose() }.buttonStyle(.borderedProminent)
                    Button("Finish anyway") { engine.continueAfterFailure() }.buttonStyle(.bordered)
                }
            } else {
                if engine.phase != .finished {
                    ProgressView()
                }
                Text(engine.progress).multilineTextAlignment(.center)
            }
            if engine.phase == .finished {
                Button("Done") { dismiss() }.buttonStyle(.borderedProminent)
            }
        }
        .padding(28)
        .frame(maxWidth: 520)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
    }
}

// MARK: - HUD

private struct HUDView: View {
    let engine: CaptureEngine

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
            row("Session", "\(engine.request.type.rawValue) · \(engine.request.roomName.isEmpty ? engine.request.sessionId : engine.request.roomName)")
            row("Tracking", engine.tracking.label + (engine.interrupted ? " (interrupted)" : ""), color: trackingColor)
            row("World map", engine.mapping)
            row("Elapsed", Self.clock(engine.elapsed) + (engine.phase == .recording || engine.recordingElapsed > 0 ? "  (rec \(Self.clock(engine.recordingElapsed)))" : ""),
                color: engine.elapsed > 600 ? .orange : .primary)
            row("Frames", "\(engine.framesRecorded)" + (engine.framesDropped > 0 ? "  (\(engine.framesDropped) dropped)" : ""))
            row("Depth", engine.hasDepth ? "\(engine.depthFrames) · \(String(format: "%.1f", engine.depthHz)) Hz" : "no sceneDepth!",
                color: engine.hasDepth ? .primary : .red)
            row("Stills", "\(engine.stillCount)")
            if let info = engine.lastStillInfo { row("Last still", info) }
            row("Anchors", "\(engine.anchorCount)")
            row("Camera", cameraText, color: engine.cameraState == .locked ? .green : (engine.cameraState == .lockLost ? .red : .primary))
            if let room = engine.roomSummary { row("RoomPlan", room) }
            row("Thermal", thermalText, color: thermalColor)
            row("Free disk", engine.freeBytes.map { $0.formatted(.byteCount(style: .file)) } ?? "?",
                color: (engine.freeBytes ?? .max) < 8_000_000_000 ? .orange : .primary)  // ~20 min of video at 50 Mbit/s
        }
        .font(.system(.callout, design: .monospaced))
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func row(_ label: String, _ value: String, color: Color = .primary) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value).foregroundStyle(color)
        }
    }

    private var trackingColor: Color {
        switch engine.tracking {
        case .normal: .green
        case .notAvailable: .red
        default: .orange
        }
    }

    private var cameraText: String {
        switch engine.cameraState {
        case .auto: return "auto (Lock exposure)"
        case .metering: return "metering…"
        case .lockLost: return "lock lost"
        case .locked:
            guard let c = engine.lockedCamera else { return "locked" }
            let shutter = c.exposureDurationS.map { "1/\(Int((1 / $0).rounded())) s" } ?? "?"
            let iso = c.iso.map { "ISO \(Int($0))" } ?? ""
            let under = (c.underexposureEv ?? 0) >= 0.1 ? String(format: " -%.1f EV", c.underexposureEv ?? 0) : ""
            return "locked \(shutter) \(iso)\(under)"
        }
    }

    private var thermalText: String {
        switch engine.thermal {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "?"
        }
    }

    private var thermalColor: Color {
        switch engine.thermal {
        case .nominal: .primary
        case .fair: .yellow
        default: .red
        }
    }

    static func clock(_ t: TimeInterval) -> String {
        let s = max(0, Int(t))
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

/// "LED flicker likely" at Lock: auto exposure metered a whole number of flicker periods (k/100 s or k/120 s)
/// while the sharp 1/240 s shutter would band under such light. Non-modal; the user chooses.
private struct FlickerPromptPanel: View {
    let mains: FlickerMains
    let accept: () -> Void
    let decline: () -> Void

    private var hz: String { mains == .hz50 ? "50 Hz" : "60 Hz" }
    private var shutter: String { mains == .hz50 ? "1/100 s" : "1/120 s" }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("LED flicker likely: use Flicker-safe \(hz)?").font(.headline)
            Text("Auto exposure settled on a multiple of \(shutter), as it does under LED or fluorescent light on \(hz) mains. 1/240 s may show horizontal bands.")
                .font(.callout)
            HStack {
                Button("Use \(hz) (\(shutter))", action: accept).buttonStyle(.borderedProminent).tint(.white).foregroundStyle(.black)
                Button("Keep 1/240 s", action: decline).buttonStyle(.bordered).tint(.white)
            }
        }
        .pill(.orange)
    }
}

/// The doorway sweep at the start of each Next room's recording, guided by the pose (0.5.0; scan 5's sweeps were
/// never really done under a 10 s countdown): turn round (150°) and step through (0.5 m), then it ends by itself
/// 2 s later; at most 20 s. Done is enabled after 5 s.
private struct SweepPanel: View {
    let started: Date
    let progress: SweepProgress?
    let onDone: () -> Void

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.25)) { context in
            let elapsed = context.date.timeIntervalSince(started)
            VStack(alignment: .leading, spacing: 6) {
                Text("Doorway sweep").font(.headline)
                Text("1. Look at the wall around the door, in the room you're leaving")
                step(done: progress?.moved == true, "2. Step through the door", detail: progress.map { String(format: "%.1f / 0.5 m", $0.movedM) })
                step(done: progress?.turned == true, "3. Turn round and look at the same wall from this side",
                     detail: progress.map { "\(Int($0.turnedDeg))° / 150°" })
                HStack {
                    Button("Done", action: onDone)
                        .buttonStyle(.bordered).tint(.white)
                        .disabled(elapsed < CaptureEngine.sweepMinDuration)
                    Text(progress.map { $0.moved && $0.turned } == true ? "Hold on the wall…" : "Ends by itself when both are done")
                        .font(.caption)
                }
                .padding(.top, 4)
            }
            .font(.callout)
        }
        .pill(.blue)
    }

    private func step(done: Bool, _ text: String, detail: String?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: done ? "checkmark.circle.fill" : "circle")
            Text(text)
            if let detail, !done { Text(detail).font(.caption.monospacedDigit()).opacity(0.8) }
        }
    }
}

/// A next room is open and waits for Record: the doorway sweep's steps (it starts with the recording) and the
/// exposure carried over, with the one tap to meter this room instead (scan 5 carried one lock outdoors).
private struct NextRoomReadyPanel: View {
    let engine: CaptureEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("\(engine.request.roomName.isEmpty ? "Next room" : engine.request.roomName): stand in the doorway, then Record").font(.headline)
            Text("Record starts the doorway sweep: look at the wall around the door, step through, turn round and look at it from this side.")
            if engine.cameraState == .locked, let values = CaptureView.lockValues(engine.lockedCamera) {
                Text("Exposure carried over (\(values)). Very different light here (outdoors, much darker)? Tap Locked to meter this room first.")
                    .font(.caption)
            }
            if !engine.positioned { Text("Waiting for tracking…").font(.caption) }
        }
        .font(.callout)
        .pill(.blue)
    }
}

/// The closing lap is open and waits for Record.
private struct LapReadyPanel: View {
    let engine: CaptureEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Closing lap: tap Record, then walk back through every room to the start").font(.headline)
            Text("Slow, chest height, look at the walls. Finish when you are back where you started.")
            if !engine.positioned { Text("Waiting for tracking…").font(.caption) }
        }
        .font(.callout)
        .pill(.green)
    }
}

/// While "Lock exposure" meters: what to do, and how close the automatic lock is.
private struct MeteringPanel: View {
    let progress: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Label("Point at a mid-bright wall and hold still", systemImage: "camera.metering.center.weighted").font(.headline)
            ProgressView(value: progress).tint(.black)
            Text("Locks by itself when the exposure settles (or tap Metering to lock now).").font(.caption)
        }
        .foregroundStyle(.black)
        .padding(12)
        .frame(maxWidth: 460, alignment: .leading)
        .background(Color.yellow.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
    }
}

/// After Next room / Start lap: the previous session saving (progress, or what failed with Retry / Continue
/// anyway), the next room's name (can be typed meanwhile), and the way out ("Not another room…": the closing lap
/// or Finish). Non-modal: the camera view and tracking stay visible.
private struct ClosingPanel: View {
    let engine: CaptureEngine
    @State private var name = ""
    @FocusState private var nameFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            status
            Divider()
            next
        }
        .padding(24)
        .frame(maxWidth: 520, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18))
        .onAppear { nameFocused = engine.after == .room }
    }

    @ViewBuilder private var status: some View {
        switch engine.closeStatus {
        case .saving(let step):
            savingRow(step)
        case nil:
            savingRow(nil)
        case .failed(let errors):
            CloseFailure(errors: errors)
            HStack {
                Button("Retry") { engine.retryClose() }.buttonStyle(.borderedProminent)
                Button("Continue anyway") { engine.continueAfterFailure() }.buttonStyle(.bordered)
            }
            Text("Continue anyway keeps everything else; what failed stays a warning in that session's notes.").font(.caption)
        case .preparing(let text):
            saved
            HStack(spacing: 10) {
                ProgressView()
                Text(text)
            }
        case .ready:
            saved
        case .setupFailed(let error):
            saved
            Text(error).foregroundStyle(.red)
            Button("Retry") { engine.retryClose() }.buttonStyle(.borderedProminent)
        }
    }

    private func savingRow(_ step: String?) -> some View {
        HStack(spacing: 10) {
            ProgressView()
            Text("Saving \(engine.closingLabel)… \(stepText(step))").font(.headline)
        }
    }

    private func stepText(_ step: String?) -> String {
        guard let step else { return "" }
        return step.prefix(1).lowercased() + step.dropFirst()
    }

    private var saved: some View {
        Label("\(engine.closingLabel) saved", systemImage: "checkmark.circle.fill").font(.headline).foregroundStyle(.green)
    }

    @ViewBuilder private var next: some View {
        if engine.after == .room {
            if let submitted = engine.nextRoomName {
                Text("Next: \(submitted.isEmpty ? "the next room" : submitted). Record is ready once it is set up.")
            } else {
                Text("Next room").font(.headline)
                HStack {
                    TextField("Room name, e.g. kitchen", text: $name)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)
                        .focused($nameFocused)
                        .onSubmit(submit)
                    Button("Continue", action: submit).buttonStyle(.borderedProminent)
                }
                Text("You can type while it saves. Then stand in the doorway and tap Record: the doorway sweep starts with the recording.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } else {
            Text("Next: the closing lap. Tap Record when it is ready.")
        }
        Menu(engine.after == .room ? "Not another room…" : "Not the lap…") {
            if engine.after == .room {
                Button("Closing lap instead", systemImage: "arrow.uturn.backward") { engine.switchToLap() }
            }
            Button("Finish", systemImage: "stop.circle") { engine.finish() }
        }
    }

    private func submit() {
        nameFocused = false
        engine.submitNextRoom(name: name)
    }
}

private struct CloseFailure: View {
    let errors: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Not everything was saved", systemImage: "exclamationmark.triangle.fill").font(.headline).foregroundStyle(.red)
            ForEach(errors, id: \.self) { Text($0).font(.callout) }
        }
    }
}

/// Big, unmissable position state: orange "Finding position" with a timer (and help after 30 s) until ARKit
/// relocalises against the loaded world map, green "Position found" for a few seconds after, red when
/// tracking is lost later. Recording and stills are disabled while it is not green.
private struct PositionStatus: View {
    let engine: CaptureEngine
    @State private var foundAt: Date?

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            content(now: context.date)
        }
        .onChange(of: engine.relocalisation) { _, new in
            if new == .relocalised { foundAt = .now }
        }
    }

    @ViewBuilder private func content(now: Date) -> some View {
        if engine.relocalisation == .relocalising {
            let waited = Int(now.timeIntervalSince(engine.relocalisingSince ?? now))
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 10) {
                    ProgressView().tint(.white)
                    Text("Finding position… \(waited / 60):\(String(format: "%02d", waited % 60))").font(.title3.bold())
                }
                Text("Stand where \(engine.request.worldMapFrom?.session?.roomName ?? "the earlier scan") was scanned and look around slowly.")
                if waited >= 30 {
                    Text("Still nothing? Aim at well-lit, detailed areas (not blank walls). Or Discard and start a session without the world map.")
                        .font(.callout)
                }
            }
            .pill(.orange)
        } else if let foundAt, now.timeIntervalSince(foundAt) < 4 {
            Label("Position found", systemImage: "checkmark.circle.fill").font(.title3.bold()).pill(.green)
        } else if engine.phase == .ready || engine.phase == .recording, engine.tracking != .normal {
            let starting = engine.tracking == .limitedInitializing || engine.tracking == .notAvailable
            Label(starting ? "Starting tracking… move the iPad slowly" : "Tracking: \(engine.tracking.label). Move slowly, aim at detailed areas.",
                  systemImage: starting ? "hourglass" : "exclamationmark.triangle.fill")
                .font(.headline).pill(starting ? .orange : .red)
        }
    }
}

private extension View {
    func pill(_ color: Color) -> some View {
        padding(12)
            .frame(maxWidth: 460, alignment: .leading)
            .background(color.opacity(0.85), in: RoundedRectangle(cornerRadius: 12))
            .foregroundStyle(.white)
    }
}

private struct Banner: View {
    let text: String
    let color: Color
    var onClose: (() -> Void)?

    var body: some View {
        HStack(alignment: .top) {
            Text(text).font(.callout)
            if let onClose {
                Button(action: onClose) { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.plain)
            }
        }
        .padding(10)
        .frame(maxWidth: 460, alignment: .leading)
        .background(color.opacity(0.75), in: RoundedRectangle(cornerRadius: 10))
        .foregroundStyle(.white)
    }
}

// MARK: - AR preview

/// RealityKit ARView showing our ARSession (no automatic configuration: CaptureEngine runs the session).
struct ARViewContainer: UIViewRepresentable {
    let session: ARSession
    let showMesh: Bool
    var coverage: CoverageOverlay?
    var showCoverage = false

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.session = session
        // Keep rendering cheap: the preview is only for framing, and heat limits session length.
        view.renderOptions = [
            .disableMotionBlur, .disableDepthOfField, .disableHDR, .disableCameraGrain, .disablePersonOcclusion,
            .disableGroundingShadows, .disableAREnvironmentLighting, .disableFaceMesh,
        ]
        view.debugOptions = showMesh ? [.showSceneUnderstanding] : []
        coverage?.attach(to: view)
        coverage?.setVisible(showCoverage)
        return view
    }

    func updateUIView(_ view: ARView, context: Context) {
        let options: ARView.DebugOptions = showMesh ? [.showSceneUnderstanding] : []
        if view.debugOptions != options { view.debugOptions = options }
        coverage?.setVisible(showCoverage)
    }

    func makeCoordinator() -> Coordinator { Coordinator(coverage: coverage) }

    /// Takes the coverage dots out of the scene before the view (and its scene) goes.
    static func dismantleUIView(_ view: ARView, coordinator: Coordinator) {
        coordinator.coverage?.detach()
    }

    final class Coordinator {
        let coverage: CoverageOverlay?
        init(coverage: CoverageOverlay?) { self.coverage = coverage }
    }
}
