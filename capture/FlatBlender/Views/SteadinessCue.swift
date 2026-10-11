import SwiftUI

/// Soft ring around the shutter button (texture sessions) or the auto-still count (scan and lap sessions): green
/// when the camera is steady (predicted motion blur < 1 px at the training resolution), amber while it moves; a
/// short white flash when a still is captured. Only shown while stills can be taken. Never blocks the shutter:
/// it sits behind the button and doesn't take touches.
struct SteadinessRing: View {
    let engine: CaptureEngine

    private static let amber = Color(red: 1, green: 0.72, blue: 0.2)

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.1)) { context in
            let reading = engine.motion
            let shot = engine.stillCue.map { context.date.timeIntervalSince($0.at) }.flatMap { $0 < 0.4 ? $0 : nil }
            let color: Color = shot != nil ? .white : (reading.steady ? .green : Self.amber)
            Circle()
                .stroke(color.opacity(shot != nil ? 1 : 0.85), lineWidth: shot != nil ? 6 : 4)
                .frame(width: 90, height: 90)
                .scaleEffect(shot.map { 1 + 0.12 * (1 - $0 / 0.4) } ?? 1)
                .opacity(active ? 1 : 0)
                .animation(.easeOut(duration: 0.2), value: reading.steady)
                .accessibilityLabel(reading.steady ? "Steady" : "Moving")
        }
        .allowsHitTesting(false)
    }

    /// Texture: same condition as the shutter (`CaptureEngine.canStill`), minus "a still is in flight". Scan and
    /// lap: while recording.
    private var active: Bool {
        engine.positioned && (engine.hasShutter ? engine.phase == .ready : engine.phase == .recording)
    }
}

/// Brief, non-modal hints in a small capsule: "not taken" after a tap (texture sessions) that found no steady
/// moment within 0.5 s (2.5 s), otherwise "turn slower" while recording and the camera moves too fast for sharp
/// video (predicted blur > 2 px at the training resolution for 0.3 s; clears after 1 s of calmer motion).
struct MotionHint: View {
    let engine: CaptureEngine

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.2)) { context in
            let text = hint(now: context.date)
            Group {
                if let text {
                    Label(text, systemImage: "hand.raised")
                        .font(.callout)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                        .transition(.opacity)
                }
            }
            .animation(.easeInOut(duration: 0.25), value: text)
        }
        .allowsHitTesting(false)
    }

    private func hint(now: Date) -> String? {
        if let miss = engine.tapMissed, now.timeIntervalSince(miss.at) < 2.5,
            engine.stillCue.map({ $0.at < miss.at }) ?? true
        {
            return miss.busy ? "Not taken: the last still is still saving. Tap again" : "Not taken: hold still, then tap"
        }
        if engine.phase == .recording, engine.motion.tooFast {
            return "Turn a little slower for sharp video"
        }
        return nil
    }
}
