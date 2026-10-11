import ARKit
import BundleKit
import Foundation
import Metal
import simd

/// `mesh.ply`: ARKit's scene-reconstruction mesh (all ARMeshAnchors) in world coordinates.
enum MeshExporter {
    /// Converts mesh anchors to world-space chunks (vertex positions transformed by the anchor's transform,
    /// normals rotated by it).
    static func chunks(from anchors: [ARMeshAnchor]) -> [MeshChunk] {
        anchors.compactMap { anchor in
            let g = anchor.geometry
            let m = anchor.transform
            let rot = simd_float3x3(
                SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            guard let local = float3s(g.vertices) else { return nil }
            let vertices = local.map { v -> SIMD3<Float> in
                let w = m * SIMD4(v, 1)
                return SIMD3(w.x, w.y, w.z)
            }
            let normals = float3s(g.normals).map { $0.map { simd_normalize(rot * $0) } }
            guard let faces = triangles(g.faces) else { return nil }
            return MeshChunk(vertices: vertices, normals: normals?.count == vertices.count ? normals : nil, faces: faces)
        }
    }

    static func write(anchors: [ARMeshAnchor], to url: URL) throws -> (vertices: Int, faces: Int) {
        let chunks = chunks(from: anchors)
        try PLYWriter.write(
            chunks, to: url,
            comment: "FlatBlender mesh.ply: ARMeshAnchors in ARKit world coordinates (metres, +y up, gravity aligned)")
        return (chunks.reduce(0) { $0 + $1.vertices.count }, chunks.reduce(0) { $0 + $1.faces.count })
    }

    private static func float3s(_ source: ARGeometrySource) -> [SIMD3<Float>]? {
        guard source.format == .float3, source.count > 0 else { return nil }
        let base = source.buffer.contents() + source.offset
        return (0..<source.count).map { i in
            let p = (base + i * source.stride).assumingMemoryBound(to: Float.self)
            return SIMD3(p[0], p[1], p[2])
        }
    }

    private static func triangles(_ element: ARGeometryElement) -> [SIMD3<UInt32>]? {
        guard element.primitiveType == .triangle, element.indexCountPerPrimitive == 3 else { return nil }
        let base = element.buffer.contents()
        let n = element.count
        switch element.bytesPerIndex {
        case 4:
            let p = base.assumingMemoryBound(to: UInt32.self)
            return (0..<n).map { SIMD3(p[$0 * 3], p[$0 * 3 + 1], p[$0 * 3 + 2]) }
        case 2:
            let p = base.assumingMemoryBound(to: UInt16.self)
            return (0..<n).map { SIMD3(UInt32(p[$0 * 3]), UInt32(p[$0 * 3 + 1]), UInt32(p[$0 * 3 + 2])) }
        default:
            return nil
        }
    }
}
