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

    // Cached: this gate is consulted on every constraints pass, and
    // ProcessInfo.environment rebuilds its whole dictionary per access
    // (uncached, it alone was ~26% of the idle main thread on
    // 2026-08-17 — the same bug class as the AIMEDiagnostics gate).
    static let isEnabled: Bool = {
        switch ProcessInfo.processInfo.environment["MTPLX_APP_SIZING_TUNER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0", "false", "off", "no": return false
        default: return true
        }
    }()

    func makeNSView(context: Context) -> TunerView {
        TunerView()
    }

    func updateNSView(_ nsView: TunerView, context: Context) {
        nsView.applyIfNeeded()
    }

    final class TunerView: NSView {
        // Participate in EVERY constraints pass. Constraint updates run
        // bottom-up, and this view is a descendant of the hosting view,
        // so `updateConstraints` below neutralizes `sizingOptions`
        // BEFORE NSHostingView.updateConstraints derives window extrema
        // in the same pass — a SwiftUI scene update re-arming the
        // options between cycles can never buy another transcript walk.
        // (2026-08-17: the one-shot apply verifiably cleared the
        // options, yet the A/B profile still showed ~2,300 walk samples
        // — something re-arms; per-pass neutralization is race-free.)
        override class var requiresConstraintBasedLayout: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeRunLoopObserver()
            guard window != nil else { return }
            DispatchQueue.main.async { [weak self] in
                _ = self?.applyIfNeeded()
            }
            installRunLoopObserver()
        }

        // The one ordering that actually wins (2026-08-18): SwiftUI
        // re-arms `sizingOptions` during the RENDER phase of a display
        // cycle — i.e. AFTER that cycle's constraints flush. Clearing
        // inside `updateConstraints` therefore always ran one phase too
        // early: the next flush saw re-armed options and walked the
        // whole transcript again (~60x/s post-stream, ~28% of the main
        // thread on an 11k-token chat; two chained-re-arm designs lost
        // the same race from different sides). A runloop observer that
        // fires after render leaves the options empty for the NEXT
        // flush, so `NSHostingView.updateConstraints` skips the extrema
        // derivation outright.
        //
        // The mask MUST include `.beforeTimers`, not just
        // `.beforeWaiting` (founder stutter, 2026-08-18 round three):
        // CFRunLoop skips the beforeWaiting phase on any iteration that
        // handled a source and keeps polling — which is exactly what a
        // human interacting with the app produces (mouse-moved / wheel
        // event storms). Under interaction the waiting-only observer
        // starved, the transcript walk returned full-size, and Core
        // Animation's own beforeWaiting commit coalesced behind it:
        // freeze-then-burst that ONLY reproduced with a hand on the
        // mouse, never under hands-off automation. `.beforeTimers`
        // fires on every iteration, polling included, and iteration
        // N+1's beforeTimers still sits after iteration N's render —
        // same ordering guarantee, no starvation window. beforeWaiting
        // stays in the mask as the idle-edge belt. Cost: one property
        // read per iteration while active; zero when idle.
        private var runLoopObserver: CFRunLoopObserver?

        private func installRunLoopObserver() {
            guard runLoopObserver == nil else { return }
            let observer = CFRunLoopObserverCreateWithHandler(
                kCFAllocatorDefault,
                CFRunLoopActivity.beforeTimers.rawValue
                    | CFRunLoopActivity.beforeWaiting.rawValue,
                true,
                0
            ) { [weak self] _, _ in
                MainActor.assumeIsolated {
                    _ = self?.applyIfNeeded()
                }
            }
            runLoopObserver = observer
            CFRunLoopAddObserver(
                CFRunLoopGetMain(), observer, .commonModes
            )
        }

        private func removeRunLoopObserver() {
            if let runLoopObserver {
                CFRunLoopRemoveObserver(
                    CFRunLoopGetMain(), runLoopObserver, .commonModes
                )
            }
            runLoopObserver = nil
        }

        override func updateConstraints() {
            applyIfNeeded()
            super.updateConstraints()
        }

        // layout() runs whenever our superview lays out (every streaming
        // flush): clear a scene re-arm here too, and ask for another
        // constraints pass only when there was one. Asking every time
        // (the 2026-08-17 design, kept beside the run-loop observer that
        // superseded it) fed itself: the constraints pass laid the view
        // out again, which asked again, so the window ran its display
        // cycle at the screen's refresh rate for as long as the app was
        // open, minimized and idle included.
        override func layout() {
            super.layout()
            if applyIfNeeded(), !needsUpdateConstraints {
                needsUpdateConstraints = true
            }
        }

        // Cached: applyIfNeeded now runs per constraints pass, and
        // ProcessInfo.environment rebuilds its dictionary per access.
        static let debugEnabled: Bool =
            ProcessInfo.processInfo.environment["MTPLX_SIZING_TUNER_DEBUG"] == "1"

        private var rearmObservations = 0

        /// Neutralizes the hosting view's sizing options and pins the
        /// window minimum. Returns true when it OBSERVED non-empty
        /// options (i.e. something re-armed since the last clear) —
        /// the signal the caller uses to decide whether to chain
        /// another constraints pass.
        @discardableResult
        func applyIfNeeded() -> Bool {
            // 2026-08-17 field regression: the original cast
            // `window.contentView as? MTPLXHostingSizingConfigurable`
            // silently failed — the hosting view is a DESCENDANT of the
            // content view on this window shape — so the tuner never
            // applied and the min-size storm it documents shipped in
            // 2.8.x (34% of the idle main thread; worse while
            // streaming). Search the subtree, and re-assert on every
            // update instead of latching: SwiftUI scene updates can
            // re-arm `sizingOptions`, and both writes are idempotent.
            guard WindowSizingTuner.isEnabled, let window else { return false }
            guard let hosting = Self.hostingView(in: window.contentView, depth: 4)
            else { return false }
            let observedRearm = !hosting.mtplxSizingOptions.isEmpty
            if observedRearm {
                if Self.debugEnabled {
                    rearmObservations += 1
                    if rearmObservations <= 8 || rearmObservations % 100 == 0 {
                        FileHandle.standardError.write(Data(
                            "[sizing-tuner] non-empty options seen (n=\(rearmObservations)) value=\(hosting.mtplxSizingOptions.rawValue)\n".utf8
                        ))
                    }
                }
                hosting.mtplxSizingOptions = []
            }
            if window.contentMinSize != WindowSizingTuner.contentMinSize {
                window.contentMinSize = WindowSizingTuner.contentMinSize
            }
            return observedRearm
        }

        private static func hostingView(
            in view: NSView?,
            depth: Int
        ) -> MTPLXHostingSizingConfigurable? {
            guard let view else { return nil }
            if let hosting = view as? MTPLXHostingSizingConfigurable {
                return hosting
            }
            guard depth > 0 else { return nil }
            for subview in view.subviews {
                if let found = hostingView(in: subview, depth: depth - 1) {
                    return found
                }
            }
            return nil
        }
    }
}
