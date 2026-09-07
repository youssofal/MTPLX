import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - WordmarkView
//
// Jet Chrome wordmark. Single source of truth across the top strip, the
// menubar popover, the About sheet, the chat empty state, and the
// onboarding hero. Renders the bundled PNG (`Resources/Wordmark/Wordmark.png`)
// at any height >= 14pt so the polished steel glyphs ship as the
// designer drew them; below that (and for any caller that explicitly
// opts in via `fallbackTracking`) it falls back to live `Text("MTPLX")`
// painted with the `MTPLXChromeText` modifier so the wordmark stays
// vector-crisp at menu-bar-mini sizes where a raster would blur.
//
// On the cream look the same PNG is re-mapped into gunmetal (see
// `WordmarkInk`) so the wordmark reads dark on paper instead of washing
// out; the fallback text picks up the graphite chrome gradient by itself.
//
// The PNG is loaded via the cached `wordmarkNSImage()` lookup below
// rather than `Image("Wordmark", bundle: .module)` so the release app can
// ship a normal signed bundle with assets in `Contents/Resources`.
// The cached lookup probes `Bundle.main`, then scans every loaded
// `Bundle.allBundles` for any `Wordmark.png`, so the raster ships as long
// as the file is bundled anywhere reachable.
// If every probe fails, we fall back to the chrome-text rendering
// rather than rendering nothing.

struct WordmarkView: View {
    var height: CGFloat
    var fallbackTracking: Bool

    @Environment(\.colorScheme) private var colorScheme

    init(height: CGFloat, fallbackTracking: Bool = false) {
        self.height = height
        self.fallbackTracking = fallbackTracking
    }

    var body: some View {
        Group {
            if useRaster, let nsImage = wordmarkNSImage(for: colorScheme) {
                Image(nsImage: nsImage)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(height: height)
            } else {
                Text("MTPLX")
                    .font(.system(size: height * 0.9, weight: .heavy, design: .rounded))
                    .tracking(height * 0.05)
                    .chromeText()
                    .fixedSize()
            }
        }
        .accessibilityLabel("MTPLX")
        .accessibilityAddTraits(.isHeader)
    }

    private var useRaster: Bool {
        !fallbackTracking && height >= 14
    }
}

// MARK: - Wordmark NSImage lookup
//
// Resolves Wordmark.png from any reachable bundle. The result is
// cached so we don't re-scan every paint; once located, subsequent
// calls return the same NSImage instance.

@MainActor
private let cachedWordmarkImage: NSImage? = {
    let resourceName = "Wordmark"
    let resourceExt  = "png"

    // 1) The main bundle (used when the release app bundles the PNG flat
    //    into `Contents/Resources` rather than into a sub-bundle).
    if let url = Bundle.main.url(forResource: resourceName, withExtension: resourceExt),
       let img = NSImage(contentsOf: url) {
        return img
    }

    // 2) Every loaded bundle. Catches development layouts and any other
    //    plausible location before falling back to text.
    for bundle in Bundle.allBundles {
        if let url = bundle.url(forResource: resourceName, withExtension: resourceExt),
           let img = NSImage(contentsOf: url) {
            return img
        }
    }

    // 3) Sibling/resource scan. Walks the executable directory, app
    //    resources, app root, and parent looking for any `Wordmark.png`
    //    before we drop to the text fallback.
    let probeRoots: [URL] = {
        var roots: [URL] = []
        let exec = Bundle.main.executableURL?.deletingLastPathComponent()
        if let exec { roots.append(exec) }
        if let resources = Bundle.main.resourceURL { roots.append(resources) }
        roots.append(Bundle.main.bundleURL)
        roots.append(Bundle.main.bundleURL.deletingLastPathComponent())
        return roots
    }()
    let fm = FileManager.default
    for root in probeRoots {
        guard let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else {
            continue
        }
        for entry in entries where entry.pathExtension == "bundle" {
            let candidate = entry.appendingPathComponent("\(resourceName).\(resourceExt)")
            if let img = NSImage(contentsOf: candidate) {
                return img
            }
        }
        let direct = root.appendingPathComponent("\(resourceName).\(resourceExt)")
        if let img = NSImage(contentsOf: direct) {
            return img
        }
    }

    return nil
}()

/// The cream-look raster, derived once from the shipped PNG.
@MainActor
private let cachedWordmarkInkImage: NSImage? = {
    guard let source = cachedWordmarkImage else { return nil }
    return WordmarkInk.derive(from: source)
}()

@MainActor
func wordmarkNSImage() -> NSImage? { cachedWordmarkImage }

/// The wordmark raster for a color scheme: the shipped chrome PNG on
/// jet, the gunmetal derivation on cream (falling back to the shipped
/// PNG if the derivation is unavailable).
@MainActor
func wordmarkNSImage(for scheme: ColorScheme) -> NSImage? {
    scheme == .light ? (cachedWordmarkInkImage ?? cachedWordmarkImage) : cachedWordmarkImage
}

// MARK: - WordmarkInk
//
// The wordmark PNG is a polished-steel raster (about 2,100 distinct
// tones: bright top edges, dimmer mid-glyph, a faint glow fringe), not a
// flat alpha glyph, so template rendering would throw the bevel away and
// leaving it as-is would wash out on cream. Instead the cream look keeps
// the designer's shading and re-maps its luminance into a gunmetal
// range. The map is order-preserving, so highlights stay the lightest
// pixels and the glyph still reads as light catching a curved bevel,
// now dark on paper. The low-alpha glow fringe is tapered so it settles
// into a soft shadow instead of a smudge. Derived lazily from the
// shipped asset, so there is no second PNG for the build script to
// package.

enum WordmarkInk {
    /// Source luminance at or below this maps to the darkest gunmetal.
    private static let sourceFloor = 40.0
    /// Source luminance span (above the floor) mapped onto the ink range.
    private static let sourceSpan = 215.0
    /// Darkest gunmetal luminance and the span up to the lightest (0...255).
    private static let inkFloor = 20.0
    private static let inkSpan = 118.0

    static func derive(from source: NSImage) -> NSImage? {
        guard let cgImage = source.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0,
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
              )
        else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { return nil }

        let pixels = data.assumingMemoryBound(to: UInt8.self)
        let byteCount = width * height * 4
        var offset = 0
        while offset < byteCount {
            let alpha = Double(pixels[offset + 3])
            if alpha > 0 {
                // Un-premultiply, re-map luminance, taper the glow
                // fringe, then premultiply again.
                let unscale = 255.0 / alpha
                let red = min(255.0, Double(pixels[offset]) * unscale)
                let green = min(255.0, Double(pixels[offset + 1]) * unscale)
                let blue = min(255.0, Double(pixels[offset + 2]) * unscale)
                let luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                let position = min(1.0, max(0.0, (luminance - sourceFloor) / sourceSpan))
                let target = inkFloor + position * inkSpan
                let factor = target / max(luminance, 1.0)
                let coverage = alpha / 255.0
                let taperedAlpha = alpha * (0.55 + 0.45 * coverage)
                let premultiply = taperedAlpha / 255.0
                pixels[offset] = UInt8(min(255.0, red * factor * premultiply + 0.5))
                pixels[offset + 1] = UInt8(min(255.0, green * factor * premultiply + 0.5))
                pixels[offset + 2] = UInt8(min(255.0, blue * factor * premultiply + 0.5))
                pixels[offset + 3] = UInt8(min(255.0, taperedAlpha + 0.5))
            }
            offset += 4
        }

        guard let inked = context.makeImage() else { return nil }
        return NSImage(cgImage: inked, size: source.size)
    }
}

// MARK: - WordmarkSubtitle

struct WordmarkSubtitle: View {
    var dividerWidth: CGFloat = 240

    var body: some View {
        VStack(spacing: 12) {
            Rectangle()
                .fill(Brand.separator)
                .frame(width: dividerWidth, height: 1)
            Text(tr("native MTP · Apple Silicon"))
                .font(BrandFont.subtitle())
                .foregroundStyle(Brand.typeSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}
