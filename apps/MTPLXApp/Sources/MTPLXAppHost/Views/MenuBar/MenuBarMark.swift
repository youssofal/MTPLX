import AppKit
import MTPLXAppCore

// MARK: - MenuBarMark
//
// MTPLX's M for the menu bar: the app icon's monogram (two interlocking
// strokes) as a one-colour template image, so macOS draws it white on a
// dark or tinted bar and black on a light one. Drawn in code: no resource
// to bundle, sharp at every scale. The state reads from the shape, never
// a colour: stopped is the M dimmed, starting or stopping adds a dot, a
// degraded or crashed engine is the M dimmed with the dot.

enum MenuBarMark {
    enum Badge: Equatable {
        case none
        case dot
    }

    /// The image for an engine state; built once per state change.
    static func image(for state: DaemonStateKind, height: CGFloat = 15) -> NSImage {
        switch state {
        case .running:
            return image(height: height, dimmed: false, badge: .none)
        case .starting, .warming, .stopping:
            return image(height: height, dimmed: false, badge: .dot)
        case .degraded, .crashed:
            return image(height: height, dimmed: true, badge: .dot)
        case .stopped:
            return image(height: height, dimmed: true, badge: .none)
        }
    }

    static func image(height: CGFloat, dimmed: Bool, badge: Badge) -> NSImage {
        let width = (height * 1.18).rounded()
        let size = NSSize(width: width, height: height)
        let image = NSImage(size: size, flipped: false) { rect in
            let alpha: CGFloat = dimmed ? 0.45 : 1
            NSColor.black.withAlphaComponent(alpha).setStroke()
            let line = max(1.5, (rect.height * 0.15).rounded(.toNearestOrAwayFromZero))
            let inset = line / 2 + 0.5
            let left = rect.minX + inset
            let right = rect.maxX - inset
            let bottom = rect.minY + inset
            let top = rect.maxY - inset
            let midY = rect.minY + rect.height * 0.32
            let span = right - left
            // Left stroke: up the left side and down past the middle; the
            // right stroke mirrors it, and the two cross in the centre.
            for points in [
                [NSPoint(x: left, y: bottom), NSPoint(x: left, y: top), NSPoint(x: left + span * 0.60, y: midY)],
                [NSPoint(x: right - span * 0.60, y: midY), NSPoint(x: right, y: top), NSPoint(x: right, y: bottom)],
            ] {
                let path = NSBezierPath()
                path.lineWidth = line
                path.lineCapStyle = .round
                path.lineJoinStyle = .round
                path.move(to: points[0])
                path.line(to: points[1])
                path.line(to: points[2])
                path.stroke()
            }
            if badge == .dot {
                let d = rect.height * 0.30
                NSColor.black.setFill()
                NSBezierPath(ovalIn: NSRect(x: rect.maxX - d, y: rect.maxY - d, width: d, height: d)).fill()
            }
            return true
        }
        image.isTemplate = true
        return image
    }
}
