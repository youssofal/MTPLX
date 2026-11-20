import AppKit
import MTPLXAppCore
import SwiftUI

struct ClientMark: View {
    let target: LaunchTarget
    var size: CGFloat = 18
    var tint: Color = Brand.typeBody

    var body: some View {
        Image(nsImage: ClientMarkImages.templateImage(for: target, size: size))
            .renderingMode(.template)
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .accessibilityHidden(true)
    }
}

/// A plain SF Symbol at the marks' size, weight and tint: terminals in
/// Settings' Where clients open, and any row that is not a client.
struct ClientSymbol: View {
    let systemImage: String
    var size: CGFloat = 16
    var tint: Color = Brand.typeBody

    var body: some View {
        Image(systemName: systemImage)
            .font(.system(size: size * 0.8, weight: .regular))
            .symbolRenderingMode(.monochrome)
            .foregroundStyle(tint)
            .frame(width: size, height: size)
            .accessibilityHidden(true)
    }
}

/// SwiftUI rows and AppKit menus share the same monochrome artwork.
@MainActor
enum ClientMarkImages {
    private static var templates: [LaunchTarget: [CGFloat: NSImage]] = [:]

    static func artworkURL(for target: LaunchTarget) -> URL? {
        let name: String
        switch target {
        case .openCode: name = "OpenCodeMark"
        case .hermes: name = "HermesLogo"
        case .claudeCode: name = "ClaudeMark"
        case .codex: name = "ChatGPTMark"
        default: return nil
        }
        return Bundle.main.url(forResource: name, withExtension: "png")
            ?? Bundle.module.url(forResource: name, withExtension: "png")
    }

    static func templateImage(for target: LaunchTarget, size: CGFloat = 16) -> NSImage {
        if let cached = templates[target]?[size] { return cached }
        let frame = NSSize(width: size, height: size)
        let image: NSImage
        if let url = artworkURL(for: target), let artwork = NSImage(contentsOf: url) {
            let height = target == .openCode ? size * 15 / 16 : size
            let scale = height / max(artwork.size.width, artwork.size.height)
            let fitted = NSSize(width: artwork.size.width * scale, height: artwork.size.height * scale)
            image = NSImage(size: frame, flipped: false) { bounds in
                let rect = NSRect(x: bounds.midX - fitted.width / 2, y: bounds.midY - fitted.height / 2,
                                  width: fitted.width, height: fitted.height)
                artwork.draw(in: rect)
                return true
            }
        } else if target == .deepseekHarness {
            image = NSImage(size: frame, flipped: true) { rect in
                guard let context = NSGraphicsContext.current?.cgContext else { return false }
                let unit = UnitCanvas(origin: rect.origin, side: rect.width)
                context.addPath(whale(unit).cgPath)
                context.setLineWidth(max(1, (size * 7.5).rounded() / 100))
                context.setLineCap(.round)
                context.setLineJoin(.round)
                context.setStrokeColor(NSColor.black.cgColor)
                context.strokePath()
                return true
            }
        } else {
            let symbol = NSImage(systemSymbolName: target.systemImage, accessibilityDescription: nil)
            image = symbol?.withSymbolConfiguration(.init(pointSize: size * 0.8, weight: .regular))
                ?? symbol ?? NSImage(size: frame)
        }
        image.isTemplate = true
        templates[target, default: [:]][size] = image
        return image
    }

    private static func whale(_ u: UnitCanvas) -> Path {
        var path = Path()
        // Back: from the snout over the head to the root of the tail.
        path.move(to: u.point(0.06, 0.62))
        path.addCurve(
            to: u.point(0.66, 0.52),
            control1: u.point(0.08, 0.30),
            control2: u.point(0.48, 0.26)
        )
        // The tail stalk curling up over the back...
        path.addCurve(
            to: u.point(0.80, 0.18),
            control1: u.point(0.74, 0.46),
            control2: u.point(0.80, 0.30)
        )
        // ...into two broad flukes.
        path.move(to: u.point(0.80, 0.18))
        path.addQuadCurve(to: u.point(0.66, 0.08), control: u.point(0.70, 0.18))
        path.move(to: u.point(0.80, 0.18))
        path.addQuadCurve(to: u.point(0.95, 0.12), control: u.point(0.90, 0.20))
        // Belly: from the tail root, round under the body, to the snout.
        path.move(to: u.point(0.66, 0.52))
        path.addCurve(
            to: u.point(0.06, 0.62),
            control1: u.point(0.62, 0.88),
            control2: u.point(0.16, 0.90)
        )
        // Mouth and eye.
        path.move(to: u.point(0.07, 0.66))
        path.addQuadCurve(to: u.point(0.30, 0.68), control: u.point(0.18, 0.72))
        path.addEllipse(in: u.rect(CGRect(x: 0.25, y: 0.48, width: 0.06, height: 0.06)))
        return path
    }

}

/// Maps the marks' unit square onto the frame.
private struct UnitCanvas {
    let origin: CGPoint
    let side: CGFloat

    func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
        CGPoint(x: origin.x + x * side, y: origin.y + y * side)
    }

    func rect(_ unit: CGRect) -> CGRect {
        CGRect(
            x: origin.x + unit.minX * side,
            y: origin.y + unit.minY * side,
            width: unit.width * side,
            height: unit.height * side
        )
    }
}
