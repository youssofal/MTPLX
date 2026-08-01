import AppKit
import SwiftUI

// MARK: - WindowSizingTuner
//
// Kills the per-display-cycle window min-size storm (2026-07-31 perf
// hunt, deferred item 1; receipts in outputs/app-frontend-hunt-*.md).
//
// The mechanism being removed: with default `sizingOptions`
// (.standardBounds), AppKit re-derives the window's content-size
// extrema from the SwiftUI graph on constraint invalidation —
// `NSHostingView.updateConstraints →
// updateWindowContentSizeExtremaIfNecessary → minSize()` — which is a
// FULL `sizeThatFits` walk of every realized view. Streaming
// invalidates constraints continuously, so that walk ran every display
// cycle and was the length-independent ~50 ms per-flush floor (and
// ~2.7 s of a 14 s scroll sample on a 20k-token transcript).
//
// Our root view's size is determined by the window (it fills it), so
// deriving window extrema from content buys nothing: set
// `sizingOptions = []` on the window's hosting view and pin an
// explicit `contentMinSize` equal to the floor ContentView used to
// express via `.frame(minWidth: 420, minHeight: 540)`. Sheets and
// overlays own separate hosting views and keep default behavior.
//
// `MTPLX_APP_SIZING_TUNER=0` disables (diagnostic escape hatch).

/// Unconstrained protocol conformance so the generic
/// `NSHostingView<Content>` can be recognized and configured without
/// knowing `Content` at the call site.
@MainActor
private protocol MTPLXHostingSizingConfigurable: AnyObject {
    var mtplxSizingOptions: NSHostingSizingOptions { get set }
}

extension NSHostingView: MTPLXHostingSizingConfigurable {
    var mtplxSizingOptions: NSHostingSizingOptions {
        get { sizingOptions }
        set { sizingOptions = newValue }
    }
}

struct WindowSizingTuner: NSViewRepresentable {
    static let contentMinSize = NSSize(width: 420, height: 540)

    static var isEnabled: Bool {
        switch ProcessInfo.processInfo.environment["MTPLX_APP_SIZING_TUNER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "off", "no": return false
        default: return true
        }
    }

    func makeNSView(context: Context) -> TunerView {
        TunerView()
    }

    func updateNSView(_ nsView: TunerView, context: Context) {
        nsView.applyIfNeeded()
    }

    final class TunerView: NSView {
        private weak var tunedWindow: NSWindow?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                self?.applyIfNeeded()
            }
        }

        func applyIfNeeded() {
            guard WindowSizingTuner.isEnabled,
                  let window,
                  window !== tunedWindow,
                  let hosting = window.contentView as? MTPLXHostingSizingConfigurable
            else { return }
            hosting.mtplxSizingOptions = []
            window.contentMinSize = WindowSizingTuner.contentMinSize
            tunedWindow = window
        }
    }
}
