// swift-tools-version: 6.2
// BundleKit: the FlatBlender side of capture bundle format v1 (see pipeline/scanmodel/models/bundle.py).
// Foundation + simd + ImageIO/CoreGraphics only, so it builds and tests on macOS without ARKit.

import PackageDescription

let package = Package(
    name: "BundleKit",
    platforms: [.iOS(.v26), .macOS(.v26)],
    products: [
        .library(name: "BundleKit", targets: ["BundleKit"]),
        .executable(name: "bundlekit-sample", targets: ["bundlekit-sample"]),
    ],
    targets: [
        .target(name: "BundleKit"),
        .executableTarget(name: "bundlekit-sample", dependencies: ["BundleKit"]),
        .testTarget(name: "BundleKitTests", dependencies: ["BundleKit"]),
    ],
    swiftLanguageModes: [.v6]
)
