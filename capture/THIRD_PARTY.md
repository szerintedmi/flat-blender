# Third-party notices

## Stray Scanner (reference only)

- Project: https://github.com/strayrobots/scanner (MIT licence)
- Use: **reference only.** Its recorder design (ARKit session → HEVC video writer, PNG depth encoder,
  per-frame odometry) was looked at as an example of what a LiDAR capture app has to do.
- **No Stray Scanner code was copied** into FlatBlender or BundleKit. All encoders here (HEVC writer,
  16-bit depth PNG via ImageIO, JSONL, PLY) were written from scratch against Apple's public APIs.
- FlatBlender writes its own bundle format (`pipeline/scanmodel/models/bundle.py`), not Stray's. The pipeline's
  `scanmodel ingest --from stray` importer reads stock Stray captures; that's a format adapter, not code reuse.

If Stray code is ever copied in, add its MIT copyright notice and licence text here, next to the files that
use it.

## Other dependencies

None. FlatBlender uses only Apple frameworks (SwiftUI, ARKit, RealityKit, RoomPlan, AVFoundation, CoreImage,
ImageIO). BundleKit uses Foundation, simd, ImageIO and CoreGraphics. No Swift packages or CocoaPods.
