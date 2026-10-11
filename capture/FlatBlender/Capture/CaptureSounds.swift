import AVFoundation
import Foundation

/// Short synthesized tones for capture feedback, played through our own `AVAudioEngine` at media volume.
///
/// Why not `AudioServicesPlaySystemSound` (0.3.0 (6) and before): system sounds are silent in Silent Mode, and
/// sound 1104 (the keyboard click) also needs Settings › Sounds › Keyboard Clicks; scan 4 heard nothing. iPads
/// have no Taptic Engine, so there is no haptic feedback at all; sound is the only non-visual channel.
///
/// The audio session is `.playback` + `.mixWithOthers`: plays in Silent Mode (the in-app setting turns the cues
/// off) and doesn't stop the user's music. Nothing records audio: `StreamRecorder` writes a video-only movie and
/// ARKit's `providesAudioData` is off, so the tones can't end up in a recording.
///
/// The tones are rendered once (sine + a little 2nd harmonic, 3 ms attack, exponential decay), one player per cue.
///
/// Everything runs on a private serial queue (0.5.0): activating the audio session and starting the engine took the
/// main thread for a moment at the first cue (the first Meter tap). `prewarm()` does that when a capture starts.
final class CaptureSounds: @unchecked Sendable {
    enum Cue: CaseIterable {
        case meter, lock, found
        case tapStill, autoStill, tapMissed
        case slowDown
    }

    static let shared = CaptureSounds()

    private let queue = DispatchQueue(label: "flatblender.sounds", qos: .userInitiated)
    // Confined to `queue`.
    private let engine = AVAudioEngine()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)
    private var players: [Cue: AVAudioPlayerNode] = [:]
    private var buffers: [Cue: AVAudioPCMBuffer] = [:]
    private var prepared = false

    private init() {
        CaptureSettings.registerDefaults()
    }

    var enabled: Bool { UserDefaults.standard.bool(forKey: SettingKey.captureSounds) }

    /// Sets up the audio session, nodes and tones ahead of the first cue (no sound).
    func prewarm() {
        guard enabled else { return }
        queue.async { [self] in _ = start() }
    }

    func play(_ cue: Cue) {
        guard enabled else { return }
        queue.async { [self] in
            guard start(), let player = players[cue], let buffer = buffers[cue] else { return }
            player.scheduleBuffer(buffer, at: nil, options: .interrupts)
            if !player.isPlaying { player.play() }
        }
    }

    /// Sets up the session, nodes and buffers once; (re)starts the engine when it stopped (route or
    /// configuration change, interruption). False when audio is unavailable: the cue is skipped. On `queue`.
    private func start() -> Bool {
        guard let format else { return false }
        if !prepared {
            do {
                try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [.mixWithOthers])
                try AVAudioSession.sharedInstance().setActive(true)
            } catch {
                return false
            }
            for cue in Cue.allCases {
                let player = AVAudioPlayerNode()
                engine.attach(player)
                engine.connect(player, to: engine.mainMixerNode, format: format)
                players[cue] = player
                buffers[cue] = Self.render(Self.tones(cue), format: format)
            }
            prepared = true
        }
        if !engine.isRunning {
            do { try engine.start() } catch { return false }
        }
        return true
    }

    private struct Tone {
        var at: Double  // s from the start of the cue
        var hz: Double
        var length: Double
        var gain: Float
    }

    /// Subtle and distinct: a fainter higher tick for an auto still; texture sessions: a click for a tapped still,
    /// a falling low pair for a tap that was not taken; two low notes for "slow down".
    private static func tones(_ cue: Cue) -> [Tone] {
        switch cue {
        case .meter: [Tone(at: 0, hz: 1319, length: 0.07, gain: 0.18)]
        case .lock: [Tone(at: 0, hz: 1047, length: 0.06, gain: 0.2), Tone(at: 0.07, hz: 1568, length: 0.08, gain: 0.2)]
        case .found:
            [Tone(at: 0, hz: 880, length: 0.12, gain: 0.2), Tone(at: 0.1, hz: 1109, length: 0.12, gain: 0.2),
             Tone(at: 0.2, hz: 1319, length: 0.18, gain: 0.2)]
        case .tapStill: [Tone(at: 0, hz: 2093, length: 0.05, gain: 0.3)]
        case .autoStill: [Tone(at: 0, hz: 2637, length: 0.03, gain: 0.12)]
        case .tapMissed: [Tone(at: 0, hz: 523, length: 0.1, gain: 0.25), Tone(at: 0.11, hz: 392, length: 0.16, gain: 0.25)]
        case .slowDown: [Tone(at: 0, hz: 440, length: 0.1, gain: 0.22), Tone(at: 0.16, hz: 440, length: 0.1, gain: 0.22)]
        }
    }

    private static func render(_ tones: [Tone], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let rate = format.sampleRate
        let total = (tones.map { $0.at + $0.length }.max() ?? 0) + 0.01
        let frames = AVAudioFrameCount(total * rate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
            let out = buffer.floatChannelData?[0]
        else { return nil }
        buffer.frameLength = frames
        for i in 0..<Int(frames) { out[i] = 0 }
        for tone in tones {
            let start = Int(tone.at * rate)
            let count = Int(tone.length * rate)
            for k in 0..<count where start + k < Int(frames) {
                let t = Double(k) / rate
                let envelope = min(1, t / 0.003) * exp(-t / (tone.length / 4))
                let phase = 2 * Double.pi * tone.hz * t
                out[start + k] += tone.gain * Float(envelope * (sin(phase) + 0.25 * sin(2 * phase)))
            }
        }
        return buffer
    }
}
