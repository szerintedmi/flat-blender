import ARKit
import AVFoundation
import BundleKit
import Foundation

/// Meter / lock of exposure, white balance and focus on ARKit's primary camera, through
/// `ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera`.
///
/// `AVCaptureDevice` isn't Sendable. Configuration happens on the main actor; the per-frame reads of `iso`
/// and white balance gains (KVO-observable, atomic properties) happen from the ARKit delegate and the
/// stills queue, which is why this wrapper is `@unchecked Sendable`.
final class CameraControl: @unchecked Sendable {
    let device: AVCaptureDevice?

    init() {
        device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
    }

    var isAvailable: Bool { device != nil }

    var iso: Float? { device?.iso }

    var whiteBalance: WhiteBalanceGains? {
        guard let g = device?.deviceWhiteBalanceGains else { return nil }
        return WhiteBalanceGains(red: shortestDouble(g.redGain), green: shortestDouble(g.greenGain), blue: shortestDouble(g.blueGain))
    }

    /// Back to continuous auto exposure, white balance and focus, so the camera meters the scene.
    @MainActor
    func meter() throws {
        guard let device else { throw CameraControlError.noDevice }
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            device.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
    }

    /// Locks the metered values, re-balanced by `ExposurePolicy` (fine-tuning the metering, once, at Lock):
    /// shutter 1/240 s with ISO raised to keep the brightness, ISO capped at 800; if that is too dark the
    /// shutter goes back up to 1/120 s, and beyond that the image is left darker (`underexposure_ev`).
    /// Returns what was locked, for `session.json`.
    ///
    /// `metered`: re-lock in another shutter mode from an earlier lock's metered exposure (the device is locked
    /// by then, so its current values are no longer what auto exposure chose). White balance and focus are
    /// read from the device either way (still the locked values then).
    @MainActor
    func lock(mode: ShutterMode = .sharp, metered: Exposure? = nil) throws -> LockedCamera {
        guard let device else { throw CameraControlError.noDevice }
        let format = device.activeFormat
        let meteredDuration = metered?.durationS ?? device.exposureDuration.seconds
        let meteredISO = metered?.iso.map { Float($0) } ?? device.iso
        let choice = ExposurePolicy.choose(
            meteredDurationS: meteredDuration, meteredISO: Double(meteredISO), minISO: Double(format.minISO),
            maxISO: Double(format.maxISO), minDurationS: format.minExposureDuration.seconds,
            maxDurationS: format.maxExposureDuration.seconds, mode: mode)
        var duration = CMTime(seconds: choice.durationS, preferredTimescale: 1_000_000)
        if duration < format.minExposureDuration { duration = format.minExposureDuration }
        if duration > format.maxExposureDuration { duration = format.maxExposureDuration }
        let iso = min(max(Float(choice.iso), format.minISO), format.maxISO)

        let maxGain = device.maxWhiteBalanceGain
        var gains = device.deviceWhiteBalanceGains
        gains.redGain = min(max(gains.redGain, 1), maxGain)
        gains.greenGain = min(max(gains.greenGain, 1), maxGain)
        gains.blueGain = min(max(gains.blueGain, 1), maxGain)

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        if device.isExposureModeSupported(.custom) {
            device.setExposureModeCustom(duration: duration, iso: iso, completionHandler: nil)
        } else if device.isExposureModeSupported(.locked) {
            device.exposureMode = .locked
        }
        if device.isWhiteBalanceModeSupported(.locked) {
            device.setWhiteBalanceModeLocked(with: gains, completionHandler: nil)
        }
        let lens = device.lensPosition
        if device.isLockingFocusWithCustomLensPositionSupported {
            device.setFocusModeLocked(lensPosition: lens, completionHandler: nil)
        } else if device.isFocusModeSupported(.locked) {
            device.focusMode = .locked
        }

        return LockedCamera(
            locked: true, exposureDurationS: duration.seconds, iso: shortestDouble(iso),
            lensPosition: shortestDouble(lens),
            whiteBalance: WhiteBalanceGains(
                red: shortestDouble(gains.redGain), green: shortestDouble(gains.greenGain),
                blue: shortestDouble(gains.blueGain)),
            metered: Exposure(durationS: meteredDuration, iso: shortestDouble(meteredISO)),
            underexposureEv: (choice.underexposureEv * 100).rounded() / 100, shutterMode: mode)
    }

    /// The active format's ISO range, for the session notes (to tune `ExposurePolicy.isoCeiling` on device).
    var isoRange: ClosedRange<Float>? {
        guard let format = device?.activeFormat, format.minISO <= format.maxISO else { return nil }
        return format.minISO...format.maxISO
    }

    /// Re-applies values from an earlier `lock()` (ARKit re-running the configuration resets them), so the
    /// next room's session keeps the same exposure, white balance and focus.
    @MainActor
    func relock(_ c: LockedCamera) throws {
        guard let device else { throw CameraControlError.noDevice }
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        if let d = c.exposureDurationS, let iso = c.iso, device.isExposureModeSupported(.custom) {
            device.setExposureModeCustom(
                duration: CMTime(seconds: d, preferredTimescale: 1_000_000), iso: Float(iso), completionHandler: nil)
        }
        if let wb = c.whiteBalance, device.isWhiteBalanceModeSupported(.locked) {
            let gains = AVCaptureDevice.WhiteBalanceGains(redGain: Float(wb.red), greenGain: Float(wb.green), blueGain: Float(wb.blue))
            device.setWhiteBalanceModeLocked(with: gains, completionHandler: nil)
        }
        if let lens = c.lensPosition, device.isLockingFocusWithCustomLensPositionSupported {
            device.setFocusModeLocked(lensPosition: Float(lens), completionHandler: nil)
        }
    }

    /// What auto exposure is doing right now, for `ExposureSettle` while metering: duration, ISO, white balance
    /// gains (red, green, blue), and whether AVFoundation reports exposure, white balance or focus still adjusting
    /// (advisory only: under ARKit the flags can stick).
    @MainActor
    var meteringSample: (durationS: Double, iso: Double, gains: [Double], adjusting: Bool)? {
        guard let device else { return nil }
        let g = device.deviceWhiteBalanceGains
        return (
            device.exposureDuration.seconds, Double(device.iso), [Double(g.redGain), Double(g.greenGain), Double(g.blueGain)],
            device.isAdjustingExposure || device.isAdjustingWhiteBalance || device.isAdjustingFocus
        )
    }

    /// True while the device still holds `c` (locked modes, and exposure within 3 % of the locked values): ARKit
    /// re-running the configuration can reset the lock a frame or two after `run`.
    @MainActor
    func holds(_ c: LockedCamera) -> Bool {
        guard let device, isStillLocked else { return false }
        func near(_ a: Double, _ b: Double?) -> Bool { b.map { abs(a - $0) <= 0.03 * $0 } ?? true }
        return near(device.exposureDuration.seconds, c.exposureDurationS) && near(Double(device.iso), c.iso)
    }

    /// Current (unlocked) values, for `session.json` when the user never locked.
    var unlockedSnapshot: LockedCamera {
        guard let device else { return LockedCamera(locked: false) }
        return LockedCamera(
            locked: false, exposureDurationS: device.exposureDuration.seconds, iso: shortestDouble(device.iso),
            lensPosition: shortestDouble(device.lensPosition), whiteBalance: whiteBalance)
    }

    /// True while every locked setting is still in a locked/custom mode (ARKit re-running the configuration
    /// can reset them; the HUD shows it).
    var isStillLocked: Bool {
        guard let device else { return false }
        let exposureLocked = device.exposureMode == .custom || device.exposureMode == .locked
        return exposureLocked && device.whiteBalanceMode == .locked && device.focusMode == .locked
    }
}

enum CameraControlError: LocalizedError {
    case noDevice

    var errorDescription: String? { "ARKit doesn't expose a configurable camera device." }
}
