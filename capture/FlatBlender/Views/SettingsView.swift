import BundleKit
import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(SettingKey.roomPlanEnabled) private var roomPlan = true
    @AppStorage(SettingKey.depthEvery) private var depthEvery = 6
    @AppStorage(SettingKey.smoothedDepth) private var smoothed = false
    @AppStorage(SettingKey.showMesh) private var showMesh = true
    @AppStorage(SettingKey.showCoverage) private var showCoverage = true
    @AppStorage(SettingKey.showCoverageMap) private var showCoverageMap = true
    @AppStorage(SettingKey.shutterMode) private var shutterMode = ShutterMode.sharp
    @AppStorage(SettingKey.captureSounds) private var captureSounds = true

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("RoomPlan in scan sessions", isOn: $roomPlan)
                } footer: {
                    Text("Runs RoomCaptureSession on the same ARSession and writes room.json + room.usdz on finish.")
                }
                Section {
                    Stepper(value: $depthEvery, in: 1...30) {
                        Text("Depth every \(depthEvery) frame\(depthEvery == 1 ? "" : "s") (\(String(format: "%.1f", 60.0 / Double(depthEvery))) Hz at 60 fps)")
                    }
                    Toggle("Use smoothed depth", isOn: $smoothed)
                } header: {
                    Text("Depth")
                } footer: {
                    Text("Depth + confidence PNGs (256x192, 16-bit mm) are written for every Nth recorded video frame. Smoothed depth is temporally filtered by ARKit (only computed when this is on); raw depth is the default.")
                }
                Section {
                    Picker("Shutter at Lock", selection: $shutterMode) {
                        Text("Sharp (1/240 s)").tag(ShutterMode.sharp)
                        Text("Flicker-safe 50 Hz (1/100 s)").tag(ShutterMode.flicker50Hz)
                        Text("Flicker-safe 60 Hz (1/120 s)").tag(ShutterMode.flicker60Hz)
                    }
                } header: {
                    Text("Camera")
                } footer: {
                    Text("ISO is raised up to 800 to keep the brightness, then the image is left darker. If stills or video show horizontal bands under LED or fluorescent light, use Flicker-safe 50 Hz (UK/EU mains) or 60 Hz (US). Applies to sessions started afterwards. When the metering points to flicker, Lock offers the switch.")
                }
                Section {
                    Toggle("Show live mesh", isOn: $showMesh)
                    Toggle("Show coverage", isOn: $showCoverage)
                    Toggle("Coverage mini-map", isOn: $showCoverageMap)
                        .disabled(!showCoverage)
                } header: {
                    Text("Preview")
                } footer: {
                    Text("Coverage: while recording, magenta dots mark surfaces seen from too few angles so far (bigger = more missing), counting only angles you could have had from where you have walked. Step sideways and look at them again: a dot turns green when it is covered, then goes. The mini-map shows the same for the walls around you.")
                }
                Section {
                    Toggle("Capture sounds", isOn: $captureSounds)
                } header: {
                    Text("Feedback")
                } footer: {
                    Text("Short tones: a faint tick for each auto still, two low notes when you turn too fast for sharp video, metering, lock and Position found; in texture sessions a click for each tapped still and a falling pair when a tap was not taken (camera not steady within 0.5 s). They play at media volume, also in Silent Mode; the video has no sound track. iPads have no vibration motor, so there is no haptic feedback.")
                }
                Section("About") {
                    LabeledContent("App", value: DeviceInfo.appVersion)
                    LabeledContent("Device", value: DeviceInfo.machine)
                    LabeledContent("iPadOS", value: DeviceInfo.osVersion)
                    LabeledContent("Bundle format", value: "flat-bundle v1")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .onAppear { CaptureSettings.registerDefaults() }
    }
}
