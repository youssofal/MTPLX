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
    /// The fill arc's value (0...1 of the sweep): sprung from commit to
    /// commit with the gauge's spring, drawn on the gauge's own timeline
    /// and at rest between moves.
    @State private var arc = AdditiveSpringValue(spring: Self.arcSpring, pointsPerUnit: 600)
    /// The hero number's flick on each new reading.
    @State private var pulse = GaugePulse(pointsPerUnit: 100)
    /// The hero number's digits, rolling from the last reading to this one.
    @State private var roll = GaugeRoll(text: "0.00", travel: Self.fullRollTravel)
    /// Headline value as committed to the display. Updated on a fixed
    /// cadence (every `speedDisplayInterval` seconds) so the digits
    /// roll discretely instead of blurring through a continuous spring
    /// interpolation.
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
    /// Headline text waiting for the next commit (the rate, or the
    /// prefill percent while no sane rate is known).
    @State private var pendingLabel: String = "0.00"
    /// Whether the arc, the flick or the digits are moving: the gauge's
    /// timeline runs only while this is true.
    @State private var moving = false
    /// Ends `moving` when the last move comes to rest.
    @State private var restTask: Task<Void, Never>? = nil
    /// Whether the window that shows this gauge can be seen (AppKit's
    /// occlusion state). A gauge nobody can see draws nothing: readings
    /// wait, and the latest lands without animation once it shows again.
    @State private var windowVisible = true
    /// Cached identity for tick spawn animation — bumping it triggers
    /// the spawn-in stagger again.
    @State private var tickModeKey: String = ""

    /// Minimum interval between visible headline digit transitions.
    /// Drives both the counter AND the arc fill so the dial and the
    /// number always show the same value — they tick together on the
    /// same 0.5 s cadence rather than the arc continuously tracking
    /// the live value while the counter sits at a stale commit.
    private static let speedDisplayInterval: TimeInterval = 0.5
    /// The arc's spring, the one the needle has always used (mass 0.6,
    /// stiffness 140, damping 22, as on the web dashboard's gauge): an
    /// interpolating spring, so overlapping moves add up.
    static let arcSpring = DampedSpring(mass: 0.6, stiffness: 140, damping: 22)
    /// How far a full-size digit travels as it leaves or arrives: the
    /// height of the band the digits fill.
    static let fullRollTravel: Double = RollingDigits.fullSizeTravel
    /// The shortest time between two frames of the gauge's motion: every
    /// other refresh on a 120 Hz display, every refresh at 60 Hz. Under
    /// 1/60 s so a refresh that lands a hair early is not skipped (which
    /// would show some frames twice as long as others).
    private static let frameInterval: TimeInterval = 1.0 / 80
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

        /// Modes whose arc fills to the reading.
        var isFill: Bool { self == .tps || self == .prefill }
    }

    var body: some View {
        let lineWidth: CGFloat = max(8, diameter * 0.040)

        ZStack {
            arcShape(lineWidth: lineWidth)
            arcTickMarks(lineWidth: lineWidth)
            // The needle, the flick and the digit roll are drawn on this
            // timeline: about 60 frames a second while one of them moves,
            // the resting frame, then nothing until the next reading. Left
            // to SwiftUI's springs they kept the whole window redrawing at
            // 120 Hz for as long as a model decoded (the springs took longer
            // to settle than the 0.5 s between readings).
            TimelineView(.animation(minimumInterval: Self.frameInterval, paused: !moving)) { context in
                // At rest, draw the resting frame whatever date the paused
                // timeline hands over.
                let time = moving ? context.date.timeIntervalSinceReferenceDate : motionEnd
                ZStack {
                    arcFill(progress: arc.value(at: time), lineWidth: lineWidth)
                    centerStack(at: time)
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
                        // Opt the center number out of the gauge-wide mode
                        // animation below. That blanket drives the arc's trim
                        // morph between modes, but it also used
                        // to animate the number's *position* whenever `mode`
                        // changed mid-window-resize — so the digits slid out of
                        // the ring while the arc snapped. The digit roll and
                        // flick run on the timeline, not on `mode`.
                        .animation(nil, value: mode)
                }
            }
            // Out of the timeline (it does not move with it); it sits on
            // the ring, clear of the number, so drawing it last changes
            // nothing on screen. It glides to a new all-time max.
            stickyMaxTick(lineWidth: lineWidth)
                .animation(motionEnabled ? .smooth(duration: 0.55) : nil, value: stickyMax)
        }
        .frame(width: diameter, height: diameter)
        .background(WindowVisibilityReporter { visible in
            // Reported from AppKit while views move between windows; land
            // the change after this update.
            Task { @MainActor in windowVisible = visible }
        })
        // The dial is an instrument: the arc, ticks and sticky-max dot
        // are placed by angle, and speed always grows clockwise, so the
        // ring keeps its LTR geometry even when the app runs RTL.
        .environment(\.layoutDirection, .leftToRight)
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
        // The morph between modes (power ring, loading snake, speedometer;
        // their colours and the fill's fade). Keyed on the kind of mode,
        // not on every reading: a reading arrives several times a second,
        // and an animation on each would put the needle, which the
        // timeline draws, back under SwiftUI's animation at full rate.
        .animation(motionEnabled ? .smooth(duration: 0.55) : nil, value: ModeFamily(mode))
        .onChange(of: stickyMaxKey) { _, _ in
            guard motionEnabled else { return }
            newMaxPulse &+= 1
        }
        .onChange(of: windowVisible) { _, visible in
            if visible {
                // Show the latest reading at once.
                pendingSpeedTask?.cancel()
                pendingSpeedTask = nil
                if ModeFamily(mode).isFill { commitHeadlineDisplay(familyChanged: false, snap: true) }
            } else {
                // The reading waits for the window to show again.
                pendingSpeedTask?.cancel()
                pendingSpeedTask = nil
                settleMotion()
            }
        }
        .onChange(of: motionEnabled) { _, enabled in
            if !enabled { settleMotion() }
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
                arc.move(to: 0, at: Self.now, animated: false)
            }

            scheduleHeadlineCommit(
                target: new.speedValue ?? 0,
                targetFraction: new.scaledValue,
                label: headlineLabel(for: new),
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
            pendingLabel = headlineLabel(for: mode)
            roll = GaugeRoll(text: pendingLabel, travel: rollTravel)
            arc.pointsPerUnit = arcPointsPerUnit
            pulse.pointsPerUnit = Double(heroNumberWidth) / 2
            if let initialSpeed = mode.speedValue {
                pendingTargetSpeed = initialSpeed
                displayedSpeed = initialSpeed
                pendingTargetFraction = mode.scaledValue
                arc.move(to: mode.scaledValue, at: Self.now, animated: false)
                lastSpeedCommitAt = Date()
            }
        }
        .onDisappear {
            spinTask?.cancel()
            spinTask = nil
            pendingSpeedTask?.cancel()
            pendingSpeedTask = nil
            restTask?.cancel()
            restTask = nil
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(mode.isDim ? tr("Start MTPLX") : tr("MTPLX hero gauge"))
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
        if mode.isDim     { return Brand.wash.opacity(0.14) }
        // Track stays a quiet dim grey so the bright white fill arc
        // has unambiguous contrast against it. Previously the track
        // was `Brand.typeBody` (off-white) which was visually
        // indistinguishable from the fill `Brand.typeHi` (also
        // off-white) — the gauge looked like a single white arc with
        // no obvious fill level.
        return Brand.wash.opacity(0.10)
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
            // A layer of its own: a frame of the needle redraws the needle,
            // not the ring, the tick labels and the text around it.
            .drawingGroup()
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
    private func centerStack(at time: Double) -> some View {
        VStack(spacing: 6) {
            ZStack {
                if mode.isDim {
                    powerGlyph
                } else if mode.isLoading {
                    // Intentionally empty — the outer ring carries
                    // the loading animation.
                    Color.clear.frame(width: 1, height: 1)
                } else {
                    heroNumber(at: time)
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
    /// The displayed value is committed on a 0.5 s cadence
    /// (`speedDisplayInterval`) so each visible change is a clean,
    /// discrete digit roll: the digits that changed slide out and in,
    /// the others stay put. Between commits the number is rock-stable,
    /// which is why this reads as a counter and not a blur. A spring
    /// scale flick on each new reading adds the "flick" the user asked
    /// for. Both run on the gauge's timeline (`roll`, `pulse`).
    @ViewBuilder
    private func heroNumber(at time: Double) -> some View {
        RollingDigits(roll: roll, time: time)
            .font(.system(size: 64, weight: .heavy, design: .rounded))
            .monospacedDigit()
            .foregroundStyle(centerForeground)
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .frame(width: heroNumberWidth, alignment: .center)
            .shadow(
                color: chromeHaloActive ? Brand.chromeHalo : .clear,
                radius: chromeHaloActive ? 8 : 0
            )
            // Draw the number and its halo on the GPU as one layer. Left to
            // itself the halo's blur was redrawn on the CPU (CoreGraphics,
            // then vImage fanned out over worker threads) on every frame of
            // every digit roll and flick: about one core of app CPU for the
            // whole decode. The inset keeps the halo, the 6% flick and the
            // rolling digits inside the layer; the negative inset leaves
            // layout unchanged.
            .padding(Self.heroLayerInset)
            .drawingGroup()
            .padding(Self.heroLayerOutset)
            // The flick scales the finished layer: its frames move the
            // layer, they do not draw the digits again at each new size.
            .scaleEffect(pulse.scale(at: time))
    }

    /// How far a rolling digit travels: `fullRollTravel` for the hero's
    /// full-size digits, less where a narrow gauge shrinks them to fit.
    private var rollTravel: Double {
        Self.fullRollTravel * min(1, Double(heroNumberWidth) / 140)
    }

    /// Room around the hero number for its 8 pt halo, its 1.06 pulse and
    /// digits rolling in and out vertically.
    private static let heroLayerInset = EdgeInsets(top: 40, leading: 20, bottom: 40, trailing: 20)
    private static let heroLayerOutset = EdgeInsets(top: -40, leading: -20, bottom: -40, trailing: -20)

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

    /// Headline string for a reading. For prefill modes where TPS is
    /// unknown but progress is known, the mode's own label, so the user
    /// sees the percent instead of an always-zero number.
    private func headlineLabel(for mode: GaugeMode) -> String {
        if let speed = mode.speedValue {
            return Format.tps(speed)
        }
        return mode.centerLabel
    }

    /// Screen points per unit of the arc's value (the 270° sweep).
    private var arcPointsPerUnit: Double {
        let lineWidth = max(8, diameter * 0.040)
        return 0.75 * Double.pi * Double(diameter - lineWidth)
    }

    /// When everything in motion is at rest.
    private var motionEnd: Double {
        max(arc.restTime ?? 0, pulse.end, roll.end)
    }

    /// Motion runs only where someone can see it, and only when the user
    /// has not turned it off (Performance mode, Reduce Motion).
    private var animates: Bool { motionEnabled && windowVisible }

    private static var now: Double { Date().timeIntervalSinceReferenceDate }

    /// State changes the gauge draws itself: never through an implicit
    /// animation, which would run at the display's full rate again.
    private func withoutAnimation(_ body: () -> Void) {
        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction, body)
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
        label: String,
        isFillMode: Bool,
        familyChanged: Bool
    ) {
        // Mode is dim/loading/degraded — no headline value to throttle.
        guard isFillMode else {
            pendingSpeedTask?.cancel()
            pendingSpeedTask = nil
            pendingTargetSpeed = 0
            pendingTargetFraction = 0
            pendingLabel = label
            displayedSpeed = 0
            lastSpeedCommitAt = .distantPast
            withoutAnimation {
                roll.roll(to: label, at: Self.now, animated: false)
                pulse.stop()
            }
            scheduleRest()
            return
        }

        pendingTargetSpeed = target
        pendingTargetFraction = targetFraction
        pendingLabel = label

        // Nobody can see this gauge: keep the reading for when they can.
        guard windowVisible else { return }

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

    /// Commit the pending reading to the arc and the number together: the
    /// arc glides to it on the gauge's spring while the digits that
    /// changed roll and the number flicks.
    ///
    /// The first commit (a fresh family change or the very first reading
    /// of a session) puts the arc at the live value without a glide, so
    /// there is no slow ramp from zero; the number still rolls and flicks,
    /// a small "I'm alive" cue. A reading whose text did not change neither
    /// rolls nor flicks, and an arc move under half a pixel lands at once:
    /// a gauge whose reading holds draws nothing. `snap` lands everything
    /// at once (a window that shows again).
    private func commitHeadlineDisplay(familyChanged: Bool, snap: Bool = false) {
        let now = Self.now
        let isFirstReading = lastSpeedCommitAt == .distantPast
        let animated = animates && !snap
        withoutAnimation {
            // The window may have been resized since the last reading.
            arc.pointsPerUnit = arcPointsPerUnit
            pulse.pointsPerUnit = Double(heroNumberWidth) / 2
            roll.travel = rollTravel
            arc.move(to: pendingTargetFraction, at: now, animated: animated && !isFirstReading)
            if pendingLabel != roll.to {
                roll.roll(to: pendingLabel, at: now, animated: animated)
                if animated { pulse.fire(at: now) }
            }
            if !animated {
                roll.settle()
                pulse.stop()
            }
            displayedSpeed = pendingTargetSpeed
            lastSpeedCommitAt = Date()
            pendingSpeedTask = nil
        }
        scheduleRest()
    }

    /// Runs the timeline until the last move comes to rest, then stops it.
    private func scheduleRest() {
        restTask?.cancel()
        restTask = nil
        let remaining = motionEnd - Self.now
        guard remaining > 0 else {
            if moving { withoutAnimation { moving = false } }
            return
        }
        if !moving { withoutAnimation { moving = true } }
        restTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(remaining))
            guard !Task.isCancelled else { return }
            withoutAnimation { moving = false }
            restTask = nil
        }
    }

    /// Lands everything in flight at its resting place and stops the
    /// timeline: the window can no longer be seen, or motion was turned off.
    private func settleMotion() {
        withoutAnimation {
            arc.move(to: arc.target, at: Self.now, animated: false)
            roll.settle()
            pulse.stop()
        }
        scheduleRest()
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
        if mode.isDim      { return AnyShapeStyle(Brand.wash.opacity(0.30)) }
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
            return tr("%@ TPS decoding", Format.tps(decode))
        case let .prefill(progress, livePrefillTPS, eta, _):
            var parts = [tr("Prefilling %lld percent", Int((progress * 100).rounded()))]
            if let live = livePrefillTPS { parts.append(tr("%@ TPS", Format.tps(live))) }
            if let eta { parts.append(tr("ETA %@", Format.duration(eta))) }
            return parts.joined(separator: ", ")
        case .dim:                  return tr("MTPLX stopped, tap to start")
        case .loading(let phase):   return phase.label.lowercased()
        case .degraded:             return tr("MTPLX degraded")
        }
    }
}
