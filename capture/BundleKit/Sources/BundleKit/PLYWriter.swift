import Foundation
import simd

/// A triangle mesh in world coordinates (one ARMeshAnchor after applying its transform).
public struct MeshChunk: Sendable {
    public var vertices: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]?  // per vertex, same count as `vertices`, or nil
    public var faces: [SIMD3<UInt32>]  // indices into this chunk's `vertices`

    public init(vertices: [SIMD3<Float>], normals: [SIMD3<Float>]? = nil, faces: [SIMD3<UInt32>]) {
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
    }
}

public enum PLYError: Error {
    case normalCountMismatch
    case indexOutOfRange
}

/// Binary little-endian PLY writer for `mesh.ply`.
///
/// Layout: `element vertex` with float x/y/z (+ float nx/ny/nz when every chunk has normals), then
/// `element face` with `list uchar int vertex_indices` (triangles). Readable by Open3D, trimesh and Blender.
public enum PLYWriter {
    /// Merges `chunks` (re-basing face indices) and writes one PLY to `url`.
    public static func write(_ chunks: [MeshChunk], to url: URL, comment: String? = nil) throws {
        try encode(chunks, comment: comment).write(to: url, options: .atomic)
    }

    public static func encode(_ chunks: [MeshChunk], comment: String? = nil) throws -> Data {
        let withNormals = !chunks.isEmpty && chunks.allSatisfy { $0.normals != nil }
        var vertexCount = 0
        var faceCount = 0
        for c in chunks {
            if withNormals, c.normals?.count != c.vertices.count { throw PLYError.normalCountMismatch }
            vertexCount += c.vertices.count
            faceCount += c.faces.count
        }

        var header = "ply\nformat binary_little_endian 1.0\n"
        if let comment {
            for line in comment.split(separator: "\n") { header += "comment \(line)\n" }
        }
        header += "element vertex \(vertexCount)\nproperty float x\nproperty float y\nproperty float z\n"
        if withNormals { header += "property float nx\nproperty float ny\nproperty float nz\n" }
        header += "element face \(faceCount)\nproperty list uchar int vertex_indices\nend_header\n"

        let vertexStride = withNormals ? 24 : 12
        var data = Data()
        data.reserveCapacity(header.utf8.count + vertexCount * vertexStride + faceCount * 13)
        data.append(contentsOf: Array(header.utf8))

        for c in chunks {
            for (i, v) in c.vertices.enumerated() {
                appendFloats(&data, v.x, v.y, v.z)
                if withNormals, let n = c.normals?[i] { appendFloats(&data, n.x, n.y, n.z) }
            }
        }
        var base: UInt32 = 0
        for c in chunks {
            let n = UInt32(c.vertices.count)
            for f in c.faces {
                guard f.x < n, f.y < n, f.z < n else { throw PLYError.indexOutOfRange }
                data.append(3)
                appendInt32(&data, Int32(bitPattern: base + f.x))
                appendInt32(&data, Int32(bitPattern: base + f.y))
                appendInt32(&data, Int32(bitPattern: base + f.z))
            }
            base += n
        }
        return data
    }

    private static func appendFloats(_ data: inout Data, _ a: Float, _ b: Float, _ c: Float) {
        let packed = SIMD3<UInt32>(a.bitPattern.littleEndian, b.bitPattern.littleEndian, c.bitPattern.littleEndian)
        withUnsafeBytes(of: packed) { data.append(contentsOf: $0.prefix(12)) }
    }

    private static func appendInt32(_ data: inout Data, _ v: Int32) {
        withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
    }
}
