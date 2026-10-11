import SwiftUI

/// Small heading-up map of the walls around the user (8 m across): grey where they are covered, magenta where at
/// least a quarter of a 20 cm cell still wants 2+ more viewing directions; the walked path in faint cyan; the user
/// as a dot with a view wedge pointing up. Answers "where should I walk / look again", without covering the view.
struct CoverageMiniMap: View {
    let overlay: CoverageOverlay
    var size: CGFloat = 150

    var body: some View {
        Canvas { ctx, canvasSize in
            guard let snap = overlay.miniMap else { return }
            let scale = canvasSize.width / CGFloat(snap.extent)
            let c = CGPoint(x: canvasSize.width / 2, y: canvasSize.height / 2)
            func pt(_ x: Float, _ y: Float) -> CGPoint { CGPoint(x: c.x + CGFloat(x) * scale, y: c.y - CGFloat(y) * scale) }
            let cs = CGFloat(snap.cellSize) * scale

            for cell in snap.cells {
                let p = pt(cell.x, cell.y)
                let rect = CGRect(x: p.x - cs / 2, y: p.y - cs / 2, width: cs, height: cs)
                if cell.flagged > 0 {
                    let a = 0.5 + 0.5 * min(1, Double(cell.flagged) / 0.6)
                    ctx.fill(Path(rect), with: .color(Color(CoverageOverlay.color).opacity(a)))
                } else {
                    ctx.fill(Path(rect), with: .color(.white.opacity(0.4)))
                }
            }
            if snap.path.count > 1 {
                var path = Path()
                path.move(to: pt(snap.path[0].x, snap.path[0].y))
                for (a, b) in zip(snap.path, snap.path.dropFirst()) {
                    // Gaps (relocalisation jumps) are not drawn as lines.
                    if simdDistance(a, b) > 1 { path.move(to: pt(b.x, b.y)) } else { path.addLine(to: pt(b.x, b.y)) }
                }
                ctx.stroke(path, with: .color(.cyan.opacity(0.45)), lineWidth: 1.5)
            }
            var wedge = Path()
            wedge.move(to: c)
            wedge.addLine(to: CGPoint(x: c.x - 13, y: c.y - 22))
            wedge.addLine(to: CGPoint(x: c.x + 13, y: c.y - 22))
            wedge.closeSubpath()
            ctx.fill(wedge, with: .color(.white.opacity(0.3)))
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 4, y: c.y - 4, width: 8, height: 8)), with: .color(.white))
        }
        .frame(width: size, height: size)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .opacity(overlay.miniMap == nil ? 0 : 1)
        .allowsHitTesting(false)
        .accessibilityLabel("Coverage map: magenta walls still need views from more directions")
    }

    private func simdDistance(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).squareRoot()
    }
}

/// Two-line key under the mini-map, so the dots need no explanation: magenta = this surface still needs a view
/// from another angle, green = it just got one.
struct CoverageLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            row(Color(CoverageOverlay.color), "Look from another angle")
            row(Color(CoverageOverlay.clearedColor), "Covered")
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
        .foregroundStyle(.white)
        .allowsHitTesting(false)
    }

    private func row(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 9, height: 9)
            Text(text)
        }
    }
}
