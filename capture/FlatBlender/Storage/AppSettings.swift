import BundleKit
import Foundation

/// User settings (UserDefaults keys shared by `@AppStorage` in SettingsView and the capture engine).
enum SettingKey {
    static let roomPlanEnabled = "roomPlanEnabled"
    static let depthEvery = "depthEveryNthFrame"
    static let smoothedDepth = "useSmoothedDepth"
    static let showMesh = "showMeshOverlay"
    static let showCoverage = "showCoverageOverlay"
    static let showCoverageMap = "showCoverageMiniMap"
    static let shutterMode = "shutterMode"
    static let captureSounds = "captureSounds"
}

/// Snapshot of the settings a capture session uses, taken when the session starts.
struct CaptureSettings: Sendable {
    var roomPlanEnabled = true
    /// Write depth + confidence PNGs for every Nth recorded video frame (6 = 10 Hz at 60 fps).
    var depthEvery = 6
    /// Use `smoothedSceneDepth` instead of `sceneDepth` for the PNGs (both are enabled in the configuration).
    var smoothedDepth = false
    /// Shutter policy at Lock: sharp 1/240 s, or flicker-safe 1/100 s / 1/120 s (`ExposurePolicy`).
    var shutterMode: ShutterMode = .sharp

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            SettingKey.roomPlanEnabled: true,
            SettingKey.depthEvery: 6,
            SettingKey.smoothedDepth: false,
            SettingKey.showMesh: true,
            SettingKey.showCoverage: true,
            SettingKey.showCoverageMap: true,
            SettingKey.shutterMode: ShutterMode.sharp.rawValue,
            SettingKey.captureSounds: true,
        ])
    }

    static func current() -> CaptureSettings {
        registerDefaults()
        let d = UserDefaults.standard
        return CaptureSettings(
            roomPlanEnabled: d.bool(forKey: SettingKey.roomPlanEnabled),
            depthEvery: max(1, d.integer(forKey: SettingKey.depthEvery)),
            smoothedDepth: d.bool(forKey: SettingKey.smoothedDepth),
            shutterMode: d.string(forKey: SettingKey.shutterMode).flatMap(ShutterMode.init(rawValue:)) ?? .sharp)
    }
}
