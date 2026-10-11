import Foundation
import simd

// Codable mirrors of pipeline/scanmodel/models/bundle.py (bundle format v1).
//
// Rules that keep the two sides in sync:
// - Keys are snake_case via explicit CodingKeys (no key strategy on the encoder).
// - Enum raw values are the Python StrEnum values.
// - Optional fields are omitted when nil (synthesized `encodeIfPresent`), which pydantic reads as None.
// - Values are written as ARKit reports them; `flat ingest` does the axis conversion.
//
// The Python models are the source of truth. If you change one side, change the other and re-run the
// contract test (README: "Contract test").

public enum SessionType: String, Codable, Sendable, CaseIterable {
    case scan  // stream + auto stills + RoomPlan
    case lap  // closing lap: stream + auto stills, no RoomPlan; continues the last room's ARSession
    case texture  // relocalise, then tap-shoot close-ups only (the only session type with a shutter button, 0.4+)
    case ultrawide  // AVFoundation 10 MP, no ARKit, no poses
}

public enum TrackingState: String, Codable, Sendable, CaseIterable {
    case normal
    case limitedInitializing = "limited_initializing"
    case limitedExcessiveMotion = "limited_excessive_motion"
    case limitedInsufficientFeatures = "limited_insufficient_features"
    case limitedRelocalizing = "limited_relocalizing"
    case notAvailable = "not_available"
}

public enum StillSource: String, Codable, Sendable, CaseIterable {
    case tap
    case auto
    case texture
    case ultrawide
    case video  // extracted from video.mov (importers only)
}

/// Pinhole intrinsics in pixels, for the image they are stored with.
public struct Intrinsics: Codable, Sendable, Equatable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var width: Int
    public var height: Int

    public init(fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    }

    /// From `ARCamera.intrinsics` (column-major 3x3: fx = m[0][0], fy = m[1][1], cx = m[2][0], cy = m[2][1])
    /// and `ARCamera.imageResolution`.
    public init(_ k: simd_float3x3, width: Int, height: Int) {
        self.init(
            fx: shortestDouble(k.columns.0.x), fy: shortestDouble(k.columns.1.y),
            cx: shortestDouble(k.columns.2.x), cy: shortestDouble(k.columns.2.y),
            width: width, height: height)
    }

    /// Same field of view at another resolution (scale about the image corner, as ARKit does).
    public func scaled(width: Int, height: Int) -> Intrinsics {
        let sx = Double(width) / Double(self.width)
        let sy = Double(height) / Double(self.height)
        return Intrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy, width: width, height: height)
    }

    private enum CodingKeys: String, CodingKey {
        case fx, fy, cx, cy, width, height
    }
}

public struct Exposure: Codable, Sendable, Equatable {
    public var durationS: Double?  // ARCamera.exposureDuration
    public var offsetEv: Double?  // ARCamera.exposureOffset
    public var iso: Double?  // AVCaptureDevice.iso at capture time

    /// Non-finite values become nil (JSON has no NaN).
    public init(durationS: Double? = nil, offsetEv: Double? = nil, iso: Double? = nil) {
        self.durationS = durationS.flatMap { $0.isFinite ? $0 : nil }
        self.offsetEv = offsetEv.flatMap { $0.isFinite ? $0 : nil }
        self.iso = iso.flatMap { $0.isFinite ? $0 : nil }
    }

    private enum CodingKeys: String, CodingKey {
        case durationS = "duration_s"
        case offsetEv = "offset_ev"
        case iso
    }
}

public struct WhiteBalanceGains: Codable, Sendable, Equatable {
    public var red: Double
    public var green: Double
    public var blue: Double

    public init(red: Double, green: Double, blue: Double) {
        self.red = red
        self.green = green
        self.blue = blue
    }

    private enum CodingKeys: String, CodingKey {
        case red, green, blue
    }
}

public struct Device: Codable, Sendable, Equatable {
    public var model: String  // hw.machine, e.g. "iPad14,3"
    public var osVersion: String
    public var appName: String
    public var appVersion: String

    public init(model: String, osVersion: String, appName: String = "FlatBlender", appVersion: String) {
        self.model = model
        self.osVersion = osVersion
        self.appName = appName
        self.appVersion = appVersion
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case osVersion = "os_version"
        case appName = "app_name"
        case appVersion = "app_version"
    }
}

public enum VideoCodec: String, Codable, Sendable {
    case hevc
    case h264
}

public struct VideoFormat: Codable, Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var fps: Double
    public var codec: VideoCodec

    public init(width: Int, height: Int, fps: Double, codec: VideoCodec = .hevc) {
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
    }

    private enum CodingKeys: String, CodingKey {
        case width, height, fps, codec
    }
}

public struct DepthFormat: Codable, Sendable, Equatable {
    public var width: Int
    public var height: Int
    public var unit: String  // always "mm"
    public var smoothed: Bool  // true if from smoothedSceneDepth instead of sceneDepth

    public init(width: Int = 256, height: Int = 192, smoothed: Bool = false) {
        self.width = width
        self.height = height
        self.unit = "mm"
        self.smoothed = smoothed
    }

    private enum CodingKeys: String, CodingKey {
        case width, height, unit, smoothed
    }
}

/// Camera settings locked after metering.
public struct LockedCamera: Codable, Sendable, Equatable {
    public var locked: Bool
    public var exposureDurationS: Double?
    public var iso: Double?
    public var lensPosition: Double?  // AVCaptureDevice.lensPosition (focus)
    public var whiteBalance: WhiteBalanceGains?
    /// What auto exposure had chosen when Lock was pressed (duration_s, iso); the lock re-balances it (`ExposurePolicy`).
    public var metered: Exposure?
    /// Stops darker than metered, when even the ISO ceiling at the longest shutter could not match it (0 = none).
    public var underexposureEv: Double?
    /// The Settings shutter policy used at Lock (`sharp`, `flicker_50hz`, `flicker_60hz`).
    public var shutterMode: ShutterMode?
    /// Mains flicker the metered shutter points to (`ExposurePolicy.suspectedFlicker`: k/100 s → `50hz`,
    /// k/120 s → `60hz`), in any shutter mode. App 0.4+.
    public var flickerSuspected: FlickerMains?
    /// The "LED flicker likely" prompt at Lock (sharp mode with flicker suspected): `shown`, `accepted`
    /// (switched to the flicker-safe mode and re-locked) or `declined`. Nil: not shown. App 0.4+.
    public var flickerPrompt: FlickerPromptOutcome?

    public init(
        locked: Bool, exposureDurationS: Double? = nil, iso: Double? = nil, lensPosition: Double? = nil,
        whiteBalance: WhiteBalanceGains? = nil, metered: Exposure? = nil, underexposureEv: Double? = nil,
        shutterMode: ShutterMode? = nil, flickerSuspected: FlickerMains? = nil,
        flickerPrompt: FlickerPromptOutcome? = nil
    ) {
        self.locked = locked
        self.exposureDurationS = exposureDurationS
        self.iso = iso
        self.lensPosition = lensPosition
        self.whiteBalance = whiteBalance
        self.metered = metered
        self.underexposureEv = underexposureEv.flatMap { $0.isFinite ? $0 : nil }
        self.shutterMode = shutterMode
        self.flickerSuspected = flickerSuspected
        self.flickerPrompt = flickerPrompt
    }

    private enum CodingKeys: String, CodingKey {
        case locked
        case exposureDurationS = "exposure_duration_s"
        case iso
        case lensPosition = "lens_position"
        case whiteBalance = "white_balance"
        case metered
        case underexposureEv = "underexposure_ev"
        case shutterMode = "shutter_mode"
        case flickerSuspected = "flicker_suspected"
        case flickerPrompt = "flicker_prompt"
    }
}

/// A guided doorway sweep: both faces of the wall around a door within a few seconds.
public struct Sweep: Codable, Sendable, Equatable {
    public var t0: Double  // ARFrame.timestamp
    public var t1: Double

    public init(t0: Double, t1: Double) {
        self.t0 = t0
        self.t1 = t1
    }

    private enum CodingKeys: String, CodingKey {
        case t0, t1
    }
}

/// `session.json`.
public struct Session: Codable, Sendable, Equatable {
    public var format: String
    public var version: Int
    public var projectId: String
    public var sessionId: String
    public var type: SessionType
    public var device: Device
    public var video: VideoFormat?
    public var depth: DepthFormat?
    public var camera: LockedCamera?
    public var worldMapIn: String?  // session id whose ARWorldMap was loaded, or whose ARSession this one continued (Next room)
    public var worldMapOut: String?  // id of the world map saved by this session (worldmap.arworldmap)
    /// Session id whose ARSession this one continued without a restart (Next room, closing lap; app 0.5+): same
    /// world frame, one trajectory however long the pause between the two recordings. Nil otherwise.
    public var continuedFrom: String?
    public var startedAt: Date
    public var endedAt: Date?
    public var roomName: String?
    public var importer: String?
    public var sweeps: [Sweep]  // guided doorway sweeps (Next room); optional in the file, default []
    public var notes: String

    public init(
        projectId: String, sessionId: String, type: SessionType, device: Device, video: VideoFormat? = nil,
        depth: DepthFormat? = nil, camera: LockedCamera? = nil, worldMapIn: String? = nil,
        worldMapOut: String? = nil, continuedFrom: String? = nil, startedAt: Date, endedAt: Date? = nil, roomName: String? = nil,
        importer: String? = nil, sweeps: [Sweep] = [], notes: String = ""
    ) {
        self.format = BundleLayout.format
        self.version = BundleLayout.version
        self.projectId = projectId
        self.sessionId = sessionId
        self.type = type
        self.device = device
        self.video = video
        self.depth = depth
        self.camera = camera
        self.worldMapIn = worldMapIn
        self.worldMapOut = worldMapOut
        self.continuedFrom = continuedFrom
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.roomName = roomName
        self.importer = importer
        self.sweeps = sweeps
        self.notes = notes
    }

    private enum CodingKeys: String, CodingKey {
        case format, version
        case projectId = "project_id"
        case sessionId = "session_id"
        case type, device, video, depth, camera
        case worldMapIn = "world_map_in"
        case worldMapOut = "world_map_out"
        case continuedFrom = "continued_from"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case roomName = "room_name"
        case importer, sweeps, notes
    }

    /// Synthesized, except that `sweeps` may be missing (session.json written before build 5).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        format = try c.decode(String.self, forKey: .format)
        version = try c.decode(Int.self, forKey: .version)
        projectId = try c.decode(String.self, forKey: .projectId)
        sessionId = try c.decode(String.self, forKey: .sessionId)
        type = try c.decode(SessionType.self, forKey: .type)
        device = try c.decode(Device.self, forKey: .device)
        video = try c.decodeIfPresent(VideoFormat.self, forKey: .video)
        depth = try c.decodeIfPresent(DepthFormat.self, forKey: .depth)
        camera = try c.decodeIfPresent(LockedCamera.self, forKey: .camera)
        worldMapIn = try c.decodeIfPresent(String.self, forKey: .worldMapIn)
        worldMapOut = try c.decodeIfPresent(String.self, forKey: .worldMapOut)
        continuedFrom = try c.decodeIfPresent(String.self, forKey: .continuedFrom)
        startedAt = try c.decode(Date.self, forKey: .startedAt)
        endedAt = try c.decodeIfPresent(Date.self, forKey: .endedAt)
        roomName = try c.decodeIfPresent(String.self, forKey: .roomName)
        importer = try c.decodeIfPresent(String.self, forKey: .importer)
        sweeps = try c.decodeIfPresent([Sweep].self, forKey: .sweeps) ?? []
        notes = try c.decode(String.self, forKey: .notes)
    }
}

/// One line of `frames.jsonl`. `idx` is the frame's index in video.mov.
public struct Frame: Codable, Sendable, Equatable {
    public var idx: Int
    public var t: Double
    public var cameraToWorld: Mat4
    public var intrinsics: Intrinsics
    public var tracking: TrackingState
    public var exposure: Exposure?
    public var depth: String?  // "depth/000123.png"
    public var confidence: String?  // "confidence/000123.png"

    public init(
        idx: Int, t: Double, cameraToWorld: Mat4, intrinsics: Intrinsics, tracking: TrackingState,
        exposure: Exposure? = nil, depth: String? = nil, confidence: String? = nil
    ) {
        self.idx = idx
        self.t = t
        self.cameraToWorld = cameraToWorld
        self.intrinsics = intrinsics
        self.tracking = tracking
        self.exposure = exposure
        self.depth = depth
        self.confidence = confidence
    }

    private enum CodingKeys: String, CodingKey {
        case idx, t
        case cameraToWorld = "camera_to_world"
        case intrinsics, tracking, exposure, depth, confidence
    }
}

/// How the camera moved when a still was taken (ARKit poses, smoothed over ~60 ms). Lets the pipeline weigh or
/// drop shaky stills.
public struct StillMotion: Codable, Sendable, Equatable {
    public var angularSpeedDegS: Double  // rotation rate of the camera, deg/s
    public var speedMS: Double  // translation speed of the camera, m/s
    /// Predicted motion blur at this still's resolution: `MotionBlur.pixels` (fx · (ω + v / 1.5 m) · exposure).
    public var blurPx: Double
    // Written by app < 0.4 only (decoded so old bundles still read): the stream's Laplacian sharpness (long side
    // 640 px) near the still, and its median over the last ~1.5 s (10 s before 0.3.0 (7)).
    public var frameSharpness: Double?
    public var medianSharpness: Double?

    public init(
        angularSpeedDegS: Double, speedMS: Double, blurPx: Double, frameSharpness: Double? = nil,
        medianSharpness: Double? = nil
    ) {
        self.angularSpeedDegS = angularSpeedDegS.isFinite ? angularSpeedDegS : 0
        self.speedMS = speedMS.isFinite ? speedMS : 0
        self.blurPx = blurPx.isFinite ? blurPx : 0
        self.frameSharpness = frameSharpness.flatMap { $0.isFinite ? $0 : nil }
        self.medianSharpness = medianSharpness.flatMap { $0.isFinite ? $0 : nil }
    }

    private enum CodingKeys: String, CodingKey {
        case angularSpeedDegS = "angular_speed_deg_s"
        case speedMS = "speed_m_s"
        case blurPx = "blur_px"
        case frameSharpness = "frame_sharpness"
        case medianSharpness = "median_sharpness"
    }
}

/// One line of `stills.jsonl`.
public struct Still: Codable, Sendable, Equatable {
    public var id: Int
    public var file: String  // "stills/000001.jpg"
    public var t: Double?
    public var source: StillSource
    public var intrinsics: Intrinsics?
    public var cameraToWorld: Mat4?
    public var tracking: TrackingState?
    public var exposure: Exposure?
    public var whiteBalance: WhiteBalanceGains?
    public var sharpness: Double?  // written by app < 0.4 only: variance of Laplacian on a downscaled luma image
    public var nearestFrame: Int?  // idx of the closest stream frame in time
    public var gravity: Vec3?  // unit gravity vector in camera axes
    public var motion: StillMotion?
    /// Written by app < 0.4 only: the app's guess that a tapped still was blurry. The pipeline measures
    /// sharpness itself.
    public var flaggedBlurry: Bool?

    public init(
        id: Int, file: String, t: Double? = nil, source: StillSource, intrinsics: Intrinsics? = nil,
        cameraToWorld: Mat4? = nil, tracking: TrackingState? = nil, exposure: Exposure? = nil,
        whiteBalance: WhiteBalanceGains? = nil, sharpness: Double? = nil, nearestFrame: Int? = nil,
        gravity: Vec3? = nil, motion: StillMotion? = nil, flaggedBlurry: Bool? = nil
    ) {
        self.id = id
        self.file = file
        self.t = t
        self.source = source
        self.intrinsics = intrinsics
        self.cameraToWorld = cameraToWorld
        self.tracking = tracking
        self.exposure = exposure
        self.whiteBalance = whiteBalance
        self.sharpness = sharpness.flatMap { $0.isFinite ? $0 : nil }
        self.nearestFrame = nearestFrame
        self.gravity = gravity
        self.motion = motion
        self.flaggedBlurry = flaggedBlurry
    }

    private enum CodingKeys: String, CodingKey {
        case id, file, t, source, intrinsics
        case cameraToWorld = "camera_to_world"
        case tracking, exposure
        case whiteBalance = "white_balance"
        case sharpness
        case nearestFrame = "nearest_frame"
        case gravity, motion
        case flaggedBlurry = "flagged_blurry"
    }
}

public enum AnchorEventKind: String, Codable, Sendable, CaseIterable {
    case add  // we added the anchor
    case update  // ARKit moved it (ARSessionDelegate didUpdate), logged when it moved > 1 mm or > 0.05 deg
    case final  // its transform in the world map saved when a session closed
}

/// One line of `anchors.jsonl`: an ARAnchor FlatBlender added (at each still and every 1 s) or ARKit's later
/// change to it, so ARKit's own loop-closure corrections can be read back. Names: `BundleLayout.anchorName`.
public struct AnchorEvent: Codable, Sendable, Equatable {
    public var t: Double  // latest ARFrame.timestamp when observed
    public var id: String  // ARAnchor.identifier (UUID string)
    public var name: String
    public var event: AnchorEventKind
    public var transform: Mat4  // ARKit world, row-major like camera_to_world

    public init(t: Double, id: String, name: String, event: AnchorEventKind, transform: Mat4) {
        self.t = t
        self.id = id
        self.name = name
        self.event = event
        self.transform = transform
    }

    private enum CodingKeys: String, CodingKey {
        case t, id, name, event, transform
    }
}

/// `calibration.json`. Not written by FlatBlender yet: ARKit doesn't expose `AVCameraCalibrationData` for the
/// streaming camera. Kept here so the contract is complete.
public struct Calibration: Codable, Sendable, Equatable {
    public var referenceWidth: Int
    public var referenceHeight: Int
    public var intrinsicMatrix: [[Double]]  // row-major 3x3, at reference size
    public var lensDistortionCenter: [Double]  // [x, y] pixels, at reference size
    public var lensDistortionLookupTable: [Double]
    public var inverseLensDistortionLookupTable: [Double]
    public var pixelSizeMm: Double?

    private enum CodingKeys: String, CodingKey {
        case referenceWidth = "reference_width"
        case referenceHeight = "reference_height"
        case intrinsicMatrix = "intrinsic_matrix"
        case lensDistortionCenter = "lens_distortion_center"
        case lensDistortionLookupTable = "lens_distortion_lookup_table"
        case inverseLensDistortionLookupTable = "inverse_lens_distortion_lookup_table"
        case pixelSizeMm = "pixel_size_mm"
    }
}
