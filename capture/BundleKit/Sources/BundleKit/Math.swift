import Foundation
import simd

/// Converts a `Float` to the `Double` with the shortest decimal representation that round-trips to the same
/// `Float` (e.g. `0.1f` becomes `0.1`, not `0.10000000149011612`). Keeps the JSONL files small and readable
/// without losing any precision ARKit had.
@inlinable
public func shortestDouble(_ value: Float) -> Double {
    guard value.isFinite else { return Double(value) }
    return Double(value.description) ?? Double(value)
}

/// A 4x4 matrix stored as a row-major nested list, `m[row][col]`, translation in `m[0...2][3]`.
/// Encodes as `[[Double]]` (4 rows of 4), matching `Mat4` in `pipeline/scanmodel/models/common.py`.
public struct Mat4: Codable, Sendable, Equatable {
    public var rows: [[Double]]

    public init(rows: [[Double]]) {
        precondition(rows.count == 4 && rows.allSatisfy { $0.count == 4 }, "Mat4 needs 4 rows of 4")
        self.rows = rows
    }

    /// From a simd (column-major) matrix such as `ARCamera.transform`: `m[row][col] = columns.col[row]`.
    public init(_ m: simd_float4x4) {
        let c = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        rows = (0..<4).map { r in (0..<4).map { col in shortestDouble(c[col][r]) } }
    }

    public init(_ m: simd_double4x4) {
        let c = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
        rows = (0..<4).map { r in (0..<4).map { col in c[col][r] } }
    }

    public static let identity = Mat4(matrix_identity_double4x4)

    /// Back to a simd (column-major) matrix.
    public var simd: simd_double4x4 {
        simd_double4x4(columns: (
            SIMD4(rows[0][0], rows[1][0], rows[2][0], rows[3][0]),
            SIMD4(rows[0][1], rows[1][1], rows[2][1], rows[3][1]),
            SIMD4(rows[0][2], rows[1][2], rows[2][2], rows[3][2]),
            SIMD4(rows[0][3], rows[1][3], rows[2][3], rows[3][3])
        ))
    }

    public var isFinite: Bool { rows.allSatisfy { $0.allSatisfy(\.isFinite) } }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rows = try container.decode([[Double]].self)
        guard rows.count == 4, rows.allSatisfy({ $0.count == 4 }) else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Mat4 needs 4 rows of 4")
        }
        self.rows = rows
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rows)
    }
}

/// Three doubles encoded as a JSON array `[x, y, z]` (`Vec3` in the pipeline).
public struct Vec3: Codable, Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    public init(_ v: SIMD3<Float>) {
        self.init(shortestDouble(v.x), shortestDouble(v.y), shortestDouble(v.z))
    }

    /// Unit gravity direction in camera axes, derived from a camera-to-world pose in a gravity-aligned world
    /// (ARKit `.gravity` alignment: world +y is up, so gravity is world -y): `g_cam = R^T * (0, -1, 0)`.
    public static func gravityInCamera(cameraToWorld m: simd_float4x4) -> Vec3? {
        // R^T * (0,-1,0) = -(row 1 of R) = -(c0.y, c1.y, c2.y)
        let g = -SIMD3<Float>(m.columns.0.y, m.columns.1.y, m.columns.2.y)
        let n = simd_length(g)
        guard n.isFinite, n > 1e-6 else { return nil }
        return Vec3(g / n)
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let v = try container.decode([Double].self)
        guard v.count == 3 else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Vec3 needs 3 values")
        }
        self.init(v[0], v[1], v[2])
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode([x, y, z])
    }
}

/// Translation (metres) and rotation (radians) between two rigid poses, e.g. an anchor's transform before and
/// after ARKit corrected its map. Computed in double precision: the thresholds that matter are ~1 mm and 0.05°.
public func poseDifference(_ a: simd_float4x4, _ b: simd_float4x4) -> (metres: Double, radians: Double) {
    func translation(_ m: simd_float4x4) -> SIMD3<Double> { SIMD3<Double>(SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)) }
    func rotation(_ m: simd_float4x4) -> simd_double3x3 {
        let c = [m.columns.0, m.columns.1, m.columns.2].map { SIMD3<Double>(SIMD3($0.x, $0.y, $0.z)) }
        return simd_double3x3(columns: (c[0], c[1], c[2]))
    }
    let q = simd_quatd(rotation(a).transpose * rotation(b))
    let radians = 2 * atan2(simd_length(q.imag), abs(q.real))
    return (simd_distance(translation(a), translation(b)), radians)
}
