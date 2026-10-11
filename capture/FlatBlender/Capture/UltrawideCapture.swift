import Foundation

// TODO(phase 5): ultra-wide sessions (SessionType.ultrawide).
//
// Plan: an AVFoundation-only capture (no ARKit, so no poses) with the 10 MP ultra-wide camera:
// - AVCaptureSession + AVCapturePhotoOutput on .builtInUltraWideCamera, locked exposure/WB like CameraControl.
// - stills/NNNNNN.jpg + stills.jsonl lines with source "ultrawide", no intrinsics / camera_to_world
//   (`flat register` solves them on the desktop), session.json with video = nil and depth = nil.
// - Optionally calibration.json from AVCameraCalibrationData (photo output delivers it for this camera).
//
// The session type is listed in the new-session sheet but disabled until this exists.
enum UltrawideCapture {
    static let isImplemented = false
}
