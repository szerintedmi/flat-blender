import ARKit
import Observation
import RealityKit
import UIKit

/// Draws the coverage deficit (`CoverageMap`) into the AR preview: small magenta hexagon dots, one per 8 cm
/// voxel that still wants views from more directions; bigger and more opaque the more it wants (1-3). A dot that
/// clears turns green for a moment (the new angle worked), then goes. They sit
/// 1.5 cm in front of the surface, so ARKit's scene-understanding wireframe stays readable between them, and are
/// occluded by the reconstructed mesh (scene-understanding occlusion), so dots behind walls don't show.
///
/// One `ModelEntity` backed by a fixed-capacity `LowLevelMesh` (8192 dots = 57k vertices, 147k indices,
/// ~1.5 MB), refilled in place at most twice a second; the indices never change (one fan per dot), only the
/// vertex positions and the four parts (one per deficit level, each with its own opacity, and the cleared dots).
/// Also refreshes the mini-map snapshot once a second.
@MainActor
@Observable
final class CoverageOverlay {
    let map: CoverageMap
    private(set) var miniMap: CoverageMap.MapSnapshot?

    @ObservationIgnored private weak var arView: ARView?
    @ObservationIgnored private var anchor: AnchorEntity?
    @ObservationIgnored private var entity: ModelEntity?
    @ObservationIgnored private var mesh: LowLevelMesh?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var visible = true
    @ObservationIgnored private var configured = false

    static let color = UIColor(red: 1.0, green: 0.24, blue: 0.75, alpha: 1)  // pink-magenta: apart from cyan wire and warm rooms
    static let clearedColor = UIColor(red: 0.3, green: 0.95, blue: 0.45, alpha: 1)
    static let opacities: [Float] = [0.55, 0.75, 0.9]
    static let clearedOpacity: Float = 0.85
    static let refresh: Duration = .milliseconds(500)

    init(map: CoverageMap) {
        self.map = map
    }

    /// Adds the overlay to `view` (once) and starts the refresh loop. Call `detach()` before the view is released:
    /// RealityKit entities freed after their scene is gone crash in CoreRE.
    func attach(to view: ARView) {
        guard arView !== view else { return }
        arView = view
        let anchor = AnchorEntity(world: .zero)
        view.scene.addAnchor(anchor)
        self.anchor = anchor
        task?.cancel()
        task = Task { [weak self] in
            var tick = 0
            while !Task.isCancelled {
                guard let self, self.arView != nil else { return }
                if self.visible {
                    if let dots = await self.map.dotMesh() { self.update(dots) }
                    if tick % 2 == 0 { self.miniMap = await self.map.mapSnapshot() }
                }
                tick += 1
                try? await Task.sleep(for: Self.refresh)
            }
        }
    }

    /// Stops the refresh loop and takes the dots out of the scene while the view is still alive (`attach` can
    /// add them again later).
    func detach() {
        task?.cancel()
        task = nil
        anchor?.removeFromParent()
        anchor = nil
        entity = nil
        mesh = nil
        arView = nil
        miniMap = nil
    }

    /// Shows or hides the dots (accumulation continues either way). Scene-understanding occlusion is on only
    /// while they are shown.
    func setVisible(_ on: Bool) {
        guard on != visible || !configured else { return }  // updateUIView calls this on every SwiftUI update
        configured = true
        visible = on
        entity?.isEnabled = on && (entity?.model != nil)
        guard let view = arView else { return }
        if on {
            view.environment.sceneUnderstanding.options.insert(.occlusion)
        } else {
            view.environment.sceneUnderstanding.options.remove(.occlusion)
            miniMap = nil
        }
    }

    private func update(_ dots: CoverageMap.DotMesh) {
        guard let anchor else { return }
        if dots.dotCount == 0 {
            entity?.isEnabled = false
            return
        }
        do {
            let mesh = try self.mesh ?? Self.makeMesh()
            self.mesh = mesh
            let n = min(dots.dotCount, CoverageMap.maxDots)
            mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
                let dst = raw.bindMemory(to: SIMD3<Float>.self)
                dots.positions.withUnsafeBufferPointer { src in
                    _ = dst.initialize(from: src.prefix(n * 7))
                }
            }
            let bounds = BoundingBox(min: dots.boundsMin, max: dots.boundsMax)
            var parts: [LowLevelMesh.Part] = []
            var first = 0
            for (level, count) in dots.levelCounts.enumerated() where count > 0 {
                let c = min(count, n - first)
                guard c > 0 else { break }
                parts.append(LowLevelMesh.Part(
                    indexOffset: first * 18 * MemoryLayout<UInt32>.size, indexCount: c * 18, topology: .triangle,
                    materialIndex: level, bounds: bounds))
                first += c
            }
            mesh.parts.replaceAll(parts)
            if entity == nil {
                let resource = try MeshResource(from: mesh)
                let model = ModelEntity(mesh: resource, materials: Self.materials())
                anchor.addChild(model)
                entity = model
            }
            entity?.isEnabled = visible
        } catch {
            // LowLevelMesh unavailable or out of memory: no overlay, capture unaffected.
            entity?.isEnabled = false
        }
    }

    private static func makeMesh() throws -> LowLevelMesh {
        let stride = MemoryLayout<SIMD3<Float>>.stride
        let desc = LowLevelMesh.Descriptor(
            vertexCapacity: CoverageMap.maxDots * 7,
            vertexAttributes: [.init(semantic: .position, format: .float3, offset: 0)],
            vertexLayouts: [.init(bufferIndex: 0, bufferStride: stride)],
            indexCapacity: CoverageMap.maxDots * 18,
            indexType: .uint32)
        let mesh = try LowLevelMesh(descriptor: desc)
        // Fixed fans: dot k = vertices 7k (centre) and 7k+1...7k+6 (hexagon).
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            var o = 0
            for k in 0..<CoverageMap.maxDots {
                let base = UInt32(k * 7)
                for j in 0..<6 {
                    idx[o] = base
                    idx[o + 1] = base + 1 + UInt32(j)
                    idx[o + 2] = base + 1 + UInt32((j + 1) % 6)
                    o += 3
                }
            }
        }
        return mesh
    }

    /// One per `DotMesh.levelCounts` entry: deficit 1, 2, 3 (magenta), cleared (green).
    private static func materials() -> [UnlitMaterial] {
        (opacities.map { (color, $0) } + [(clearedColor, clearedOpacity)]).map { color, opacity in
            var m = UnlitMaterial(color: color)
            m.blending = .transparent(opacity: .init(floatLiteral: opacity))
            m.faceCulling = .none
            m.writesDepth = false
            return m
        }
    }
}
