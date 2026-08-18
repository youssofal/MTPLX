import SwiftUI
import MTPLXAppCore

// MARK: - GaugeView (flat)
//
// Minimalist hero gauge. One white arc, one hairline track, one big
// number, one subtitle. Prefill mode shifts the arc to the single
// brand accent (soft blue). No AngularGradient chrome, no warm-chrome
// shoulder, no halo glow, no inner raised well, no extra border ring.
//
// Rendering is Core Animation `Circle().trim().stroke(...)` so idle
// GPU is effectively zero and ProMotion adapts the refresh rate
// automatically.

struct GaugeView: View {
    let mode: GaugeMode
    let stickyMax: Double
    let ceiling: Double
    let motionEnabled: Bool
    var borderTint: Color? = nil
    var diameter: CGFloat = 280
    /// Optional handler invoked when the user taps the hero while the
    /// gauge is in `.dim` mode (power-button look). Pass `nil` to make
    /// the gauge non-interactive.
    var onPowerTap: (() -> Void)? = nil

    @State private var newMaxPulse: Int = 0
    @State private var powerHovering: Bool = false
    @State private var powerPressing: Bool = false
    /// Lagged version of `mode.scaledValue` used to drive the fill
    /// arc. Reset to 0 on entry into running mode, then sprung up to
    /// the live value — gives the speedometer-needle swing instead of
    /// snapping to whatever the first decode reading was.
    @State private var displayedScaledValue: Double = 0
    /// Headline value as committed to the display. Updated on a fixed
    /// cadence (every `speedDisplayInterval` seconds) so the digits
    /// roll discretely with `.contentTransition(.numericText())`
    /// instead of blurring through a continuous spring interpolation.
    @State private var displayedSpeed: Double = 0
    /// Latest live target value, updated on every mode change. Read by
    /// the throttled commit task so when the cadence fires we display
    /// the freshest reading rather than the value at scheduling time.
    @State private var pendingTargetSpeed: Double = 0
    /// Latest live target arc fraction (0...1). Committed alongside
    /// `pendingTargetSpeed` so the dial position and the headline
    /// always reflect the same reading.
    @State private var pendingTargetFraction: Double = 0
    /// Wall-clock time of the most recent commit. Used to enforce the
    /// `speedDisplayInterval` floor between visible digit transitions.
    @State private var lastSpeedCommitAt: Date = .distantPast
    /// In-flight deferred commit task; nil when nothing is scheduled.
    @State private var pendingSpeedTask: Task<Void, Never>? = nil
    /// Pulse trigger fired on each commit so the headline emphasises
    /// the digit change rather than just morphing silently.
    @State private var headlinePulse: Int = 0
    /// Cached identity for tick spawn animation — bumping it triggers
    /// the spawn-in stagger again.
    @State private var tickModeKey: String = ""

    /// Minimum interval between visible headline digit transitions.
    /// Drives both the counter AND the arc fill so the dial and the
    /// number always show the same value — they tick together on the
    /// same 0.5 s cadence rather than the arc continuously tracking
    /// the live value while the counter sits at a stale commit.
    private static let speedDisplayInterval: TimeInterval = 0.5
    /// Continuous arc rotation. Holds the gauge's resting angle (135°)
    /// while not loading; while loading, a `Task` increments it ~4° per
    /// frame at ~60fps. Cancelling the task is a hard stop — unlike
    /// `repeatForever` which SwiftUI can't reliably cancel.
    @State private var arcRotation: Double = 135
    @State private var spinTask: Task<Void, Never>? = nil

    private enum ModeFamily: Equatable {
        case dim
        case loading
        case tps
        case prefill
        case degraded

        init(_ mode: GaugeMode) {
            switch mode {
            case .dim: self = .dim
            case .loading: self = .loading
            case .tps: self = .tps
            case .prefill: self = .prefill
            case .degraded: self = .degraded
            }
        }
    }

    var body: some View {
        let lineWidth: CGFloat = max(8, diameter * 0.040)

        ZStack {
            arcShape(lineWidth: lineWidth)
            arcTickMarks(lineWidth: lineWidth)
            arcFill(progress: displayedScaledValue, lineWidth: lineWidth)
            stickyMaxTick(lineWidth: lineWidth)
            centerStack
                .phaseAnimator(
                    [0, 1, 0],
                    trigger: newMaxPulse
                ) { content, phase in
                    content
                        .shadow(
                            color: Brand.accentChrome.opacity(0.45 * phase),
                            radius: 14 * phase,
                            x: 0,
                            y: 0
                        )
                } animation: { _ in
                    motionEnabled ? .smooth(duration: 0.4) : nil
                }
                // Opt the center number out of the gauge-wide
                // `.animation(value: mode)` below. That blanket drives
                // the arc's trim morph between modes, but it also used
                // to animate the number's *position* whenever `mode`
                // changed mid-window-resize — so the digits slid out of
                // the ring while the arc snapped. The digit-roll and
                // pulse are keyed on `displayedSpeed`/`headlinePulse`,
                // not `mode`, so nulling the mode animation here keeps
                // them intact while pinning the number's position.
                .animation(nil, value: mode)
        }
        .frame(width: diameter, height: diameter)
        .padding(8)
        .contentShape(Circle())
        .onHover { hovering in
            guard mode.isDim, onPowerTap != nil else { return }
            powerHovering = hovering
        }
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    if mode.isDim, onPowerTap != nil { powerPressing = true }
                }
                .onEnded { _ in
                    let wasPressing = powerPressing
                    powerPressing = false
                    if wasPressing, mode.isDim { onPowerTap?() }
                }
        )
        .animation(motionEnabled ? .smooth(duration: 0.55) : nil, value: mode)
        .onChange(of: stickyMaxKey) { _, _ in
            guard motionEnabled else { return }
            newMaxPulse &+= 1
        }
        .onChange(of: mode) { old, new in
            // Both the headline number AND the arc fill ride the same
            // throttled commit. They tick together on the
            // `speedDisplayInterval` cadence so the dial position
            // always corresponds to the displayed number. The arc's
            // SPRING animation runs from commit to commit (smooth
            // glide); between commits, the arc stays still.
            let oldFamily = ModeFamily(old)
            let newFamily = ModeFamily(new)
            let newIsFill = newFamily == .tps || newFamily == .prefill
            let familyChanged = oldFamily != newFamily
            let newTickKey = tickIdentity(for: new)
            if newTickKey != tickModeKey {
                tickModeKey = newTickKey
            }

            // Non-fill modes (dim/loading/degraded) reset the arc
            // immediately; the commit path handles fill modes.
            if !newIsFill {
                displayedScaledValue = 0
            }

            scheduleHeadlineCommit(
                target: new.speedValue ?? 0,
                targetFraction: new.scaledValue,
                isFillMode: newIsFill,
                familyChanged: familyChanged
            )
        }
        .onChange(of: mode.isLoading) { _, isLoading in
            updateSpinState(isLoading: isLoading)
        }
        .onAppear {
            updateSpinState(isLoading: mode.isLoading)
            // Seed both the throttled headline and the arc fill on
            // first render so they don't sit at zero waiting for the
            // first commit window after the view mounts.
            if let initialSpeed = mode.speedValue {
                pendingTargetSpeed = initialSpeed
                displayedSpeed = initialSpeed
                pendingTargetFraction = mode.scaledValue
                displayedScaledValue = mode.scaledValue
                lastSpeedCommitAt = Date()
            }
        }
        .onDisappear {
            spinTask?.cancel()
            spinTask = nil
            pendingSpeedTask?.cancel()
            pendingSpeedTask = nil
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(mode.isDim ? "Start MTPLX" : "MTPLX hero gauge")
        .accessibilityValue(accessibilityValue)
        .accessibilityAddTraits(mode.isDim && onPowerTap != nil ? .isButton : [])
    }

    // MARK: - Arc shape (single morphing element)
    //
    // The hero outline is ONE arc. Trim morphs:
    //   .dim     → 1.00  (full circle, power-button ring)
    //   .loading → 0.18  (snake — short arc that spins around)
    //   .tps     → 0.75  (270° speedometer)
    // And rotation morphs:
    //   .loading → continuously incremented +360° per 1.4s
    //   else     → 135° (the gauge's resting bottom-left start point)
    //
    // SwiftUI animates the trim change via the parent's
    // `.animation(.smooth, value: mode)`. The rotation is driven by
    // local @State so the spin can be cancelled cleanly when leaving
    // loading mode without leaving the repeatForever animation alive.

    private var arcCoverage: CGFloat {
        switch mode {
        case .dim: return 1.0
        case .loading: return 0.18
        case .tps, .prefill, .degraded: return 0.75
        }
    }

    private var arcStrokeColor: Color {
        if mode.isDegraded { return Brand.danger }
        if mode.isDim     { return Color.white.opacity(0.14) }
        // Track stays a quiet dim grey so the bright white fill arc
        // has unambiguous contrast against it. Previously the track
        // was `Brand.typeBody` (off-white) which was visually
        // indistinguishable from the fill `Brand.typeHi` (also
        // off-white) — the gauge looked like a single white arc with
        // no obvious fill level.
        return Color.white.opacity(0.10)
    }

    private func arcShape(lineWidth: CGFloat) -> some View {
        Circle()
            .trim(from: 0, to: arcCoverage)
            .stroke(
                arcStrokeColor,
                style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
            )
            .rotationEffect(.degrees(arcRotation))
            .padding(lineWidth / 2)
            .shadow(
                color: mode.isLoading ? Brand.typeBody.opacity(0.30) : .clear,
                radius: mode.isLoading ? lineWidth * 0.4 : 0
            )
    }

    private func arcFill(progress: Double, lineWidth: CGFloat) -> some View {
        Circle()
            .trim(from: 0, to: 0.75 * max(0, min(1, progress)))
            .stroke(
                arcColor,
                style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
            )
            .rotationEffect(.degrees(135))
            .padding(lineWidth / 2)
            .opacity(mode.isDim || mode.isLoading ? 0 : 1)
    }

    private var arcColor: Color {
        if mode.isDegraded { return Brand.danger.opacity(0.75) }
        // Prefill mode (warm) reads as the warm-steel sister tone;
        // decode (cool) reads as the polished chrome accent. Both are
        // chrome — distinguished only by warmth, never by hue.
        if mode.isWarm     { return Brand.accentWarm }
        return Brand.accentChrome
    }

    // MARK: - Sticky max tick

    private func stickyMaxTick(lineWidth: CGFloat) -> some View {
        GeometryReader { proxy in
            let r = min(proxy.size.width, proxy.size.height) / 2 - lineWidth / 2
            let center = CGPoint(x: proxy.size.width / 2, y: proxy.size.height / 2)
            let fraction = (stickyMax > 0 && ceiling > 0) ? min(1, stickyMax / ceiling) : -1
            let degrees = 135.0 + 270.0 * max(0, fraction)
            let angle = degrees * .pi / 180.0
            let cosA = CGFloat(cos(angle))
            let sinA = CGFloat(sin(angle))
            let pos = CGPoint(x: center.x + r * cosA, y: center.y + r * sinA)
            Circle()
                .fill(Brand.typeHi)
                .frame(width: lineWidth * 0.45, height: lineWidth * 0.45)
                .position(pos)
                .opacity(fraction >= 0 && !mode.isDim ? 0.9 : 0)
        }
    }

    // MARK: - Center label

    /// Center stack — the three big visuals (power glyph during
    /// `.dim`, nothing during `.loading`, TPS number once running)
    /// share the same vertical slot so the layout doesn't jump
    /// between modes. Mode swaps are instant (no `.transition`
    /// modifiers) so the previous mode's content doesn't ghost
    /// behind the new one. Subtitle + caption appear ONLY when they
    /// carry information.
    @ViewBuilder
    private var centerStack: some View {
        VStack(spacing: 6) {
            ZStack {
                if mode.isDim {
                    powerGlyph
                } else if mode.isLoading {
                    // Intentionally empty — the outer ring carries
                    // the loading animation.
                    Color.clear.frame(width: 1, height: 1)
                } else {
                    tpsNumber
                }
            }
            .frame(height: 76)

            if !mode.subtitle.isEmpty {
                Text(mode.subtitle.lowercased())
                    .font(.system(size: 11, weight: .medium, design: .default))
                    .foregroundStyle(Brand.typeSecondary)
            }

            if let caption = mode.caption {
                Text(mode.preservesCaptionCase ? caption : caption.lowercased())
                    .font(.system(size: 10, weight: .regular, design: .default))
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity)
    }

    /// Power button (`.dim`). Granular hover — only the glyph reacts.
    @ViewBuilder
    private var powerGlyph: some View {
        Image(systemName: "power")
            .font(.system(size: 56, weight: .light))
            .symbolRenderingMode(.monochrome)
            .foregroundStyle(powerGlyphColor)
            .scaleEffect(powerPressing ? 0.95 : (powerHovering ? 1.08 : 1.0))
            .shadow(
                color: powerHovering ? Brand.typeBody.opacity(0.35) : .clear,
                radius: powerHovering ? 12 : 0
            )
            .animation(motionEnabled ? .spring(response: 0.30, dampingFraction: 0.74) : nil,
                       value: powerHovering)
            .animation(motionEnabled ? .spring(response: 0.18, dampingFraction: 0.62) : nil,
                       value: powerPressing)
    }

    /// Running TPS number (`.tps`) or live prefill TPS (`.prefill`).
    /// The displayed value is committed on a 2-second cadence
    /// (`speedDisplayInterval`) so each visible change is a clean,
    /// discrete digit-roll via `.contentTransition(.numericText())` —
    /// the rolling-digits effect SwiftUI was designed for. Between
    /// commits the number is rock-stable, which is why this reads as
    /// a counter and not a blur. A spring scale pulse on each commit
    /// adds the "flick" the user asked for without disturbing the
    /// digit roll itself.
    @ViewBuilder
    private var tpsNumber: some View {
        Text(headlineLabel)
            .font(.system(size: 64, weight: .heavy, design: .rounded))
            .monospacedDigit()
            .foregroundStyle(centerForeground)
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .frame(width: heroNumberWidth, alignment: .center)
            .contentTransition(.numericText())
            .shadow(
                color: chromeHaloActive ? Color.white.opacity(0.06) : .clear,
                radius: chromeHaloActive ? 8 : 0
            )
            .phaseAnimator(
                [1.0, 1.06, 1.0],
                trigger: headlinePulse
            ) { content, scale in
                content.scaleEffect(scale)
            } animation: { _ in
                motionEnabled ? .spring(response: 0.28, dampingFraction: 0.72) : nil
            }
    }

    /// Whether the hero TPS should carry the polished-steel ambient
    /// halo. Off in degraded / dim states so the semantic foreground
    /// (danger red, 30% white power-glyph) reads cleanly without a
    /// neutral white halo competing for attention.
    private var chromeHaloActive: Bool {
        !mode.isDegraded && !mode.isDim
    }

    /// Width reserved for the hero number. Sized for the widest
    /// expected reading ("1500" prefill TPS at the 64pt heavy-rounded
    /// monospaced-digit weight), but never wider than the ring so the
    /// digits stay inside the circle at any diameter (the responsive
    /// dashboard shrinks `diameter` on narrow windows). Anything smaller
    /// centres within the reserved box; anything wider scales down via
    /// `minimumScaleFactor`.
    private var heroNumberWidth: CGFloat { min(200, diameter * 0.72) }

    /// Headline string driven by the throttled `displayedSpeed`. For
    /// prefill modes where TPS is unknown but progress is known, fall
    /// back to the mode's own label so the user sees the percent
    /// instead of an always-zero number.
    private var headlineLabel: String {
        if mode.speedValue != nil {
            return Format.tps(displayedSpeed)
        }
        return mode.centerLabel
    }

    /// Schedule the next visible commit of BOTH the headline number
    /// and the arc fill, enforcing `speedDisplayInterval` between
    /// visible ticks. Updates that arrive inside the window are
    /// absorbed into `pendingTargetSpeed` / `pendingTargetFraction`
    /// so the deferred commit always uses the freshest value rather
    /// than a stale snapshot.
    private func scheduleHeadlineCommit(
        target: Double,
        targetFraction: Double,
        isFillMode: Bool,
        familyChanged: Bool
    ) {
        // Mode is dim/loading/degraded — no headline value to throttle.
        guard isFillMode else {
            pendingSpeedTask?.cancel()
            pendingSpeedTask = nil
            pendingTargetSpeed = 0
            pendingTargetFraction = 0
            displayedSpeed = 0
            displayedScaledValue = 0
            lastSpeedCommitAt = .distantPast
            return
        }

        pendingTargetSpeed = target
        pendingTargetFraction = targetFraction

        let interval = Self.speedDisplayInterval
        let now = Date()
        let elapsed = now.timeIntervalSince(lastSpeedCommitAt)
        // First-non-zero bypass: when the gauge is at 0 (idle, waiting
        // for first reading) and a non-zero target arrives, commit
        // immediately rather than waiting up to 0.5 s for the next
        // throttle window. This is what gives the dial its
        // "wakes up instantly when the request starts" feel instead of
        // the previous slow ramp-from-zero look.
        let firstNonZeroReading = displayedSpeed == 0 && target > 0
        let shouldCommitNow = familyChanged
            || lastSpeedCommitAt == .distantPast
            || firstNonZeroReading
            || elapsed >= interval

        if shouldCommitNow {
            commitHeadlineDisplay(familyChanged: familyChanged)
            return
        }

        // Inside the throttle window: only schedule one deferred task.
        // Subsequent updates inside the window simply refresh
        // `pendingTargetSpeed` / `pendingTargetFraction`; the
        // scheduled task will pick those up when it fires.
        if pendingSpeedTask != nil { return }
        let delay = max(0, interval - elapsed)
        pendingSpeedTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            commitHeadlineDisplay(familyChanged: false)
        }
    }

    /// Commit `pendingTargetSpeed` to the visible `displayedSpeed`
    /// AND `pendingTargetFraction` to the visible
    /// `displayedScaledValue` in the same animation transaction. The
    /// digit roll and the arc glide therefore happen together.
    ///
    /// First commit (whether on a fresh family change or the very
    /// first reading of a session) snaps without animation so the
    /// gauge appears immediately at the live value — no slow ramp
    /// from zero. Subsequent commits ride the fast
    /// `Motion.gaugeValueSpring` so dial position glides between
    /// 0.5 s ticks without overshooting.
    private func commitHeadlineDisplay(familyChanged: Bool) {
        let isFirstReading = lastSpeedCommitAt == .distantPast
        let arcAnimation: Animation? = (motionEnabled && !isFirstReading)
            ? Motion.gaugeValueSpring
            : nil
        if let arcAnimation {
            withAnimation(arcAnimation) {
                displayedScaledValue = pendingTargetFraction
            }
        } else {
            displayedScaledValue = pendingTargetFraction
        }

        // Single state-write to `displayedSpeed`. SwiftUI applies
        // `.contentTransition(.numericText())` against this change
        // for the digit roll and `.phaseAnimator(trigger: headlinePulse)`
        // for the emphasis pulse. On the first reading we still
        // bump the pulse so the gauge gives a small "I'm alive" cue.
        displayedSpeed = pendingTargetSpeed
        headlinePulse &+= 1
        lastSpeedCommitAt = Date()
        pendingSpeedTask = nil
    }

    private var powerGlyphColor: Color {
        if powerPressing || powerHovering { return Brand.typeBody }
        return Brand.typeSecondary
    }

    /// Given the current rotation (which may be hundreds of degrees
    /// past 360° from accumulated spin cycles), find the nearest
    /// equivalent that lands on `target` so the settle spring rotates
    /// the shorter direction.
    private func nearestRotation(to target: Double, from current: Double) -> Double {
        let mod = current.truncatingRemainder(dividingBy: 360)
        let positive = mod >= 0 ? mod : mod + 360
        var delta = target - positive
        if delta > 180  { delta -= 360 }
        if delta < -180 { delta += 360 }
        return current + delta
    }

    /// Start or stop the loading spin. Driving rotation from a Task
    /// (rather than `.repeatForever`) gives us a hard kill-switch:
    /// cancelling the task stops new state updates dead, then a
    /// finite spring tweens cleanly to the gauge resting angle 135°.
    private func updateSpinState(isLoading: Bool) {
        spinTask?.cancel()
        spinTask = nil

        guard motionEnabled else {
            arcRotation = isLoading ? arcRotation : 135
            return
        }

        if isLoading {
            spinTask = Task { @MainActor in
                let degreesPerFrame: Double = 4.3  // ~258°/sec at 60fps
                while !Task.isCancelled {
                    arcRotation += degreesPerFrame
                    if arcRotation > 720 { arcRotation -= 360 }
                    try? await Task.sleep(for: .microseconds(16_667)) // 60fps
                }
            }
        } else {
            let target = nearestRotation(to: 135, from: arcRotation)
            withAnimation(.spring(response: 0.55, dampingFraction: 0.85)) {
                arcRotation = target
            }
        }
    }

    private var centerForeground: AnyShapeStyle {
        if mode.isDegraded { return AnyShapeStyle(Brand.danger) }
        if mode.isDim      { return AnyShapeStyle(Color.white.opacity(0.30)) }
        // Polished chrome gradient on the hero TPS digits so they read
        // as the same polished steel as the wordmark + ALL-TIME MAX
        // badge. The neighbouring shadow halo (see `chromeHaloActive`)
        // adds the ambient sheen.
        return AnyShapeStyle(Brand.chromeAccent)
    }

    // MARK: - Arc tick marks (speedtest-style numeric labels)
    //
    // Drawn on the inside of the arc at the positions corresponding to
    // the mode's `tickStops`. Both the labels and the fill needle use
    // `GaugeMode.arcFraction` so they always land exactly on the same
    // angle for the same value. On entry into a mode, the labels
    // spawn-in one by one with a small stagger so the dial feels
    // assembled rather than dumped.

    @ViewBuilder
    private func arcTickMarks(lineWidth: CGFloat) -> some View {
        if mode.showsTicks {
            GeometryReader { proxy in
                let r = min(proxy.size.width, proxy.size.height) / 2 - lineWidth / 2
                let center = CGPoint(x: proxy.size.width / 2, y: proxy.size.height / 2)
                let labelInset: CGFloat = lineWidth * 0.95 + 12
                let labelRadius = max(8, r - labelInset)
                ForEach(Array(mode.tickStops.enumerated()), id: \.offset) { idx, stop in
                    // Tick label positions are evenly spaced around
                    // the arc; the needle (arcFill) does the
                    // piecewise-linear interpolation between them.
                    let fraction = GaugeMode.tickFraction(index: idx, tickStops: mode.tickStops)
                    let degrees = 135.0 + 270.0 * fraction
                    let angle = degrees * .pi / 180.0
                    let cosA = CGFloat(cos(angle))
                    let sinA = CGFloat(sin(angle))
                    let pos = CGPoint(
                        x: center.x + labelRadius * cosA,
                        y: center.y + labelRadius * sinA
                    )
                    Text(tickLabel(stop))
                        .font(.system(size: 10, weight: .semibold, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(Brand.typeTertiary)
                        .position(pos)
                        .transition(.opacity.combined(with: .scale(scale: 0.6)))
                        .animation(
                            motionEnabled
                                ? Motion.gaugeTickSpawn.delay(Double(idx) * Motion.gaugeTickStaggerStep)
                                : nil,
                            value: tickModeKey
                        )
                }
            }
            .id(tickModeKey)
        }
    }

    private func tickLabel(_ value: Double) -> String {
        if value >= 1000 { return String(format: "%.0f", value / 1000) + "k" }
        if value == floor(value) { return String(format: "%.0f", value) }
        return String(format: "%.1f", value)
    }

    /// String identity used to drive `ArcTickMarks` spawn animations.
    /// Tied to the mode family rather than the live value so the ticks
    /// only re-stagger when the dial visibly switches scale.
    private func tickIdentity(for mode: GaugeMode) -> String {
        switch mode {
        case .tps: return "tps-100"
        case .prefill: return "prefill-1500"
        case .dim: return "dim"
        case .loading: return "loading"
        case .degraded: return "degraded"
        }
    }

    // MARK: - Helpers

    private var stickyMaxKey: Int { Int((stickyMax * 100).rounded()) }

    private var accessibilityValue: String {
        switch mode {
        case let .tps(decode, _):
            return "\(Format.tps(decode)) TPS decoding"
        case let .prefill(progress, livePrefillTPS, eta, _):
            var parts = ["Prefilling \(Int((progress * 100).rounded())) percent"]
            if let live = livePrefillTPS { parts.append("\(Format.tps(live)) TPS") }
            if let eta { parts.append("ETA \(Format.duration(eta))") }
            return parts.joined(separator: ", ")
        case .dim:                  return "MTPLX stopped, tap to start".mtplxLocalized
        case .loading(let phase):   return phase.label.lowercased()
        case .degraded:             return "MTPLX degraded".mtplxLocalized
        }
    }
}
