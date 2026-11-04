import SwiftUI

// MARK: - Motion
//
// Single source of truth for app-wide animation timings, easings, and
// scales. The launch / inference / model overlays used to triplicate
// these numbers; the gauge headline used to hardcode its own spring;
// row hover effects had their own private constants. Anything that
// reads "this overlay feels different from that overlay" usually means
// one of these knobs drifted. Keep all overlay choreography pinned to
// these constants so visual parity is structural, not a coincidence.

enum Motion {
    // MARK: - Overlay choreography (LaunchOverlay, InferenceParamsOverlay, ModelPickerOverlay)

    /// The toolbar popovers' reveal: border draw-in, header and every row in
    /// one transaction, complete 0.2 s after the click. The old reveal
    /// staggered ten beats over about a second (header at 0.22 s, the last
    /// row at 0.72 s plus its spring), and each beat re-ran the popover's
    /// body; on the Settings popover that read as slow and mushy.
    static let overlayEnter: Animation = .smooth(duration: 0.20)

    /// Border-stroke draw-in for the benchmark overlay. Trims the
    /// rounded-rect outline from 0 to 1.
    static let overlayBorder: Animation = .smooth(duration: 0.40)

    /// Benchmark overlay header pop. Drives the title fade + 6pt slide.
    static let overlayHeaderSpring: Animation = .spring(response: 0.32, dampingFraction: 0.85)

    /// Delay after the benchmark overlay's entry begins before its header
    /// animates in.
    static let overlayHeaderDelay: TimeInterval = 0.22

    /// Exit animation for every overlay state reset.
    static let overlayExit: Animation = .easeIn(duration: 0.18)

    // MARK: - Row hover (launch / inference / model rows)

    /// Outer popover corner radius for the chrome-strip overlays.
    /// Shared so concentric-corner math (inner radius = outer radius
    /// − inset) is exact across `LaunchOverlay`, `ModelPickerOverlay`,
    /// and `InferenceParamsOverlay`.
    static let overlayCornerRadius: CGFloat = 12

    /// Inset between popover edge and a row's hover/selection card.
    /// Tuned with `rowHoverCornerRadius` so corners are concentric
    /// (12 − 4 = 8). Bumping one without the other will visually
    /// misalign the inner card against the popover edge.
    static let rowHoverInset: CGFloat = 4

    /// Inner-row card corner radius. Must equal
    /// `overlayCornerRadius − rowHoverInset` to stay concentric with
    /// the popover. The previous 8 + 6-inset combination was off by
    /// 2pt and read as "doesn't fit".
    static let rowHoverCornerRadius: CGFloat = overlayCornerRadius - rowHoverInset

    /// How much a row magnifies on hover. Reads as a magnify, not a
    /// colour change.
    static let rowHoverScale: CGFloat = 1.02

    /// Spring used for the row magnify + shadow drop.
    static let rowHoverSpring: Animation = .spring(response: 0.28, dampingFraction: 0.80)

    /// Background fill applied to a hovered row card.
    static let rowHoverFill: Color = Brand.wash.opacity(0.05)

    /// Shadow opacity applied to a hovered row card.
    static let rowHoverShadowOpacity: Double = 0.55

    /// Shadow radius applied to a hovered row card.
    static let rowHoverShadowRadius: CGFloat = 10

    /// Shadow y-offset applied to a hovered row card.
    static let rowHoverShadowYOffset: CGFloat = 4

    // MARK: - Control hover (interactive widget lift inside a popover)
    //
    // The inference overlay used to lift either the whole section card
    // (loud) or the whole label+slider stack inside an opaque highlight
    // (still loud). Both read as "this rectangle is selected".
    //
    // The control-level treatment is the opposite: no card, no fill, no
    // scale. The widget itself (the slider track + thumb, the segmented
    // pill, the toggle switch) floats up ~1.5pt with a soft shadow
    // underneath. Label text and value text don't move at all, so the
    // user reads "the dial I'm reaching for lifted off the surface"
    // instead of "a panel is highlighted".
    //
    // Pure offset + shadow, no scale. Scaling a slider track makes it
    // feel magnified rather than lifted; the offset is the entire
    // signal.

    /// Spring for control lift. Slightly snappier than the row hover so
    /// each widget feels responsive without bouncing.
    static let controlHoverSpring: Animation = .spring(response: 0.22, dampingFraction: 0.85)

    /// Vertical translation applied to a hovered control.
    static let controlHoverOffsetY: CGFloat = -1.5

    /// Soft drop shadow underneath a hovered control. Tuned to read
    /// against `Brand.raisedSurface` without competing with the slider
    /// thumb's built-in shadow.
    static let controlHoverShadowOpacity: Double = 0.40
    static let controlHoverShadowRadius: CGFloat = 5
    static let controlHoverShadowYOffset: CGFloat = 2

    // MARK: - Gauge value smoothing
    //
    // Mirrors the web dashboard's Motion spring
    // (`useSpring(motionValue, { stiffness: 140, damping: 22, mass: 0.6 })`
    // in `dashboard/src/components/TPSGauge.tsx`). The combination of
    // smoothing the underlying value with this spring and applying
    // `.contentTransition(.numericText())` on the headline label gives the
    // speedtest "tick up / tick down" feel without rapid integer flicker.

    static let gaugeValueSpring: Animation = .interpolatingSpring(
        mass: 0.6, stiffness: 140, damping: 22
    )

    /// Animation used when an arc tick label first appears in a mode.
    static let gaugeTickSpawn: Animation = .spring(response: 0.32, dampingFraction: 0.85)

    /// Per-tick stagger when the arc tick row spawns in.
    static let gaugeTickStaggerStep: TimeInterval = 0.04

    // MARK: - Acceptance bars / metric tiles

    /// Easing applied to acceptance bar widths so changes glide rather
    /// than snap. Numeric labels use `.contentTransition(.numericText())`.
    static let metricBar: Animation = .easeOut(duration: 0.30)

    // MARK: - Surface lift (LiveTab daemon-state elevation reveal)
    //
    // When the daemon transitions to `.running` (i.e. the model has
    // finished loading), the LiveTab tiles + panels crossfade from
    // their flat 2D treatment to the canonical chrome panel treatment
    // — and settle back down when the daemon stops. The reveal is
    // staggered left-to-right + top-to-bottom so the dashboard reads
    // as "powering on" rather than snapping all at once.
    //
    // Reverse direction (running → stopped) reuses the same spring and
    // stagger so a stop feels symmetric to a start.

    /// Spring for the per-surface flat ↔ chrome crossfade. Tuned so a
    /// single surface settles in ~0.55s with a soft tail; combined
    /// with the per-index stagger below the full row reveal lands in
    /// just under a second.
    static let surfaceLiftSpring: Animation = .spring(response: 0.55, dampingFraction: 0.86)

    /// Per-index stagger between surface lifts. Five tiles × 0.045s
    /// puts the row reveal at ~0.18s of stagger; the two panels below
    /// pick up the chain after the row finishes.
    static let surfaceLiftStaggerStep: TimeInterval = 0.045
}
