import SwiftUI
import MTPLXAppCore

// MARK: - LiveTile
//
// Small piano-on-piano tile used in `TileRow`. Same proportions for
// every metric: a label cap, a big chrome number, an always-reserved
// unit slot, and an always-reserved caption slot. Live metric updates
// should be visually stable — no digit-roll animation, no width
// changes, no blur while decode is running.
//
// Jet Chrome pass: tile fills the height proposed by the parent
// layout (so `EqualColumnsLayout`'s max-height propagation makes
// every tile in the row exactly the same size, not just the same
// width). The label / value / caption slots are always rendered —
// when a tile has no unit or caption, the slot still occupies its
// natural line height via a space placeholder so the value glyph
// sits at the same vertical baseline across every tile in the row.
//
// The background is delegated to `LiftedSurface`, which renders the
// flat 2D card when `lifted == false` and crossfades into the
// canonical chrome panel treatment (gradient fill + 3-stop chrome
// stroke + mid elevation shadow) when `lifted == true`. `liftDelay`
// staggers each tile in the row so the dashboard lights up
// left-to-right as the daemon reaches `.running`.

struct LiveTile: View {
    let label: String
    let value: String
    var unit: String? = nil
    var systemImage: String? = nil
    var tint: Color = Brand.typeHi
    var caption: String? = nil
    var lifted: Bool = false
    var liftDelay: TimeInterval = 0
    var liftAnimation: Animation? = Motion.surfaceLiftSpring

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Brand.typeTertiary)
                }
                Text(label.uppercased())
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(2)
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(size: 24, weight: .black, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.55)
                    .contentTransition(.numericText())
                Text(unit ?? " ")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .tracking(1)
                    .foregroundStyle(Brand.typeSecondary)
                    .opacity(unit == nil ? 0 : 1)
            }

            // Caption slot is always rendered — with a non-empty
            // placeholder when there's no caption — so the value
            // glyph sits at the same vertical baseline across every
            // tile in a TileRow. Without this, caption-less tiles
            // were 18pt shorter than tiles with captions and the row
            // read as a jagged stair-step.
            Text(caption ?? " ")
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .tracking(1)
                .foregroundStyle(Brand.typeTertiary)
                .lineLimit(1)
                .opacity(caption == nil ? 0 : 1)
                .contentTransition(.numericText())
        }
        .padding(14)
        .frame(
            maxWidth: .infinity,
            maxHeight: .infinity,
            alignment: .topLeading
        )
        .background {
            LiftedSurface(
                lifted: lifted,
                cornerRadius: Brand.Radii.m,
                animation: liftAnimation,
                delay: liftDelay
            )
        }
    }
}

// MARK: - TileRow
//
// 5-tile horizontal row beneath the hero gauge. Lifetime / Cached /
// Memory / MinMax / Depth. Each tile gets `LiveTile` styling for
// consistency. KeyframeAnimator-style entrance stagger is achieved via
// per-index `.transition` delay on first paint.

struct TileRow: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    /// Content width handed down from `LiveTab` so the row can choose a
    /// column count without its own `GeometryReader` (which collapses to
    /// zero height inside the scroll view's `VStack`).
    var availableWidth: CGFloat = 0

    /// Uniform descriptor for each tile so the five metrics can be laid
    /// out by a single responsive grid (and wrap cleanly on narrow
    /// windows) instead of five hand-placed columns.
    private struct TileSpec: Identifiable {
        let label: String
        let value: String
        var unit: String? = nil
        let systemImage: String
        var caption: String? = nil
        let liftIndex: Int
        var id: String { label }
    }

    var body: some View {
        let snapshot = backend.snapshot
        let lifetime = snapshot?.lifetime
        let mem = backend.mem
        let machine = backend.health
        let rolling = backend.rolling
        let latest = backend.latest
        let smoothed = backend.smoothedMetrics
        let sessionBank = backend.sessionBank

        // Same warm-up gate as the gauge — until we've actually
        // observed a `.completed` SSE event, treat lifetime / rolling
        // tiles as zero so the warm-up doesn't masquerade as
        // "22 tok · 1 req". Memory tile is fine because it's a true
        // system metric, not request-derived.
        let hasRealRequest = backend.observedCompletionCount > 0
        let liveLifetime = hasRealRequest ? lifetime : nil
        let liveRolling = hasRealRequest ? rolling : nil
        let liveLatest = hasRealRequest ? latest : nil
        let liveSmoothed = hasRealRequest ? smoothed : SmoothedMetrics()

        // `lifted` is true only once the daemon has actually reached
        // `.running` — `.starting` and `.warming` keep the row flat
        // so the chrome catch reads as a reward for a completed load,
        // not a teaser during it. The per-index stagger lights the
        // row up left-to-right as the model comes online.
        let isRunning = backend.daemonState.kind == .running
        let liftAnimation: Animation? = themeStore.reduceMotionPreference
            ? nil
            : Motion.surfaceLiftSpring

        let specs: [TileSpec] = [
            TileSpec(
                label: tr("Context"),
                value: contextValue(machine: machine),
                systemImage: "text.alignleft",
                caption: contextCaption(lifetime: liveLifetime),
                liftIndex: 0
            ),
            TileSpec(
                label: tr("Cached"),
                value: cacheValue(
                    smoothed: liveSmoothed,
                    latest: liveLatest,
                    lifetime: liveLifetime,
                    sessionBank: sessionBank
                ),
                unit: "tok",
                systemImage: "tray.full",
                caption: cacheHitCaption(
                    latest: liveLatest,
                    lifetime: liveLifetime,
                    sessionBank: sessionBank
                ),
                liftIndex: 1
            ),
            TileSpec(
                label: tr("Memory"),
                value: memoryValue(mem: mem),
                unit: memoryUnit(machine: machine),
                systemImage: "memorychip",
                caption: memoryCaption(
                    mem: mem,
                    machine: machine,
                    pressureLevel: backend.memoryPressureLevel,
                    recentShed: backend.memoryGuardRecentShed
                ),
                liftIndex: 2
            ),
            TileSpec(
                label: tr("5-min Min/Max"),
                value: minMaxValue(rolling: liveRolling),
                unit: "TPS",
                systemImage: "chart.bar.xaxis",
                caption: liveRolling?.mean.map { tr("avg %@", Format.tps($0)) },
                liftIndex: 3
            ),
            TileSpec(
                label: tr("Avg Prefill"),
                value: Format.tps(snapshot?.prefillRates?.averageTokS),
                unit: "TPS",
                systemImage: "gauge.with.dots.needle.bottom.50percent",
                caption: snapshot?.prefillRates?.peakTokS.map { tr("peak %@", Format.tps($0)) },
                liftIndex: 4
            ),
        ]

        // Equal-width flexible columns: five across on a normal/wide
        // dashboard, wrapping to three then two as the window narrows so
        // each tile stays readable instead of crushing to a sliver.
        let columns = Array(
            repeating: GridItem(.flexible(minimum: 0), spacing: Brand.Spacing.s3, alignment: .top),
            count: columnCount(for: availableWidth)
        )

        LazyVGrid(columns: columns, alignment: .leading, spacing: Brand.Spacing.s3) {
            ForEach(specs) { spec in
                LiveTile(
                    label: spec.label,
                    value: spec.value,
                    unit: spec.unit,
                    systemImage: spec.systemImage,
                    caption: spec.caption,
                    lifted: isRunning,
                    liftDelay: liftDelay(forIndex: spec.liftIndex),
                    liftAnimation: liftAnimation
                )
            }
        }
    }

    /// Column count for the metric row. Five tiles fit comfortably above
    /// ~540pt of content width; below that they wrap.
    private func columnCount(for width: CGFloat) -> Int {
        if width >= 540 { return 5 }
        if width >= 360 { return 3 }
        return 2
    }

    /// Per-index stagger used when the row lifts on daemon `.running`.
    /// Reduce Motion users get a flat 0 so all tiles flip together
    /// (paired with `liftAnimation == nil` so there's no spring
    /// either).
    private func liftDelay(forIndex index: Int) -> TimeInterval {
        guard !themeStore.reduceMotionPreference else { return 0 }
        return Double(index) * Motion.surfaceLiftStaggerStep
    }

    private func minMaxValue(rolling: RollingMetrics?) -> String {
        guard let rolling, let min = rolling.min, let max = rolling.max else { return "—" }
        return "\(Format.tps(min)) / \(Format.tps(max))"
    }

    // MARK: Context tile

    /// Current context use vs the model's max window. The headline
    /// only reads "current / max" once the live runtime has actually
    /// reported its own context window — the configured value isn't
    /// the source of truth (the loaded model / harness is, and the
    /// configured value may be wider than what the model can really
    /// accept). When no model is loaded, or the daemon hasn't yet
    /// published a health payload, we read "—" instead of inventing
    /// a misleading max.
    ///
    /// The "current" signal cascades through the freshest source:
    ///   1. Any in-flight request's prompt plus generated token count.
    ///   2. The largest prefix length across active engine sessions
    ///      (covers the read-only "between requests, last chat is
    ///      still loaded" case).
    ///   3. Zero. Once the model is loaded but nothing has been chat
    ///      yet, the headline reads `0 / max`.
    private func contextValue(machine: HealthPayload?) -> String {
        guard let maxTokens = machine?.contextWindow, maxTokens > 0 else {
            return "—"
        }
        let current = currentContextTokens()
        return "\(Format.compactTokens(current)) / \(Format.compactTokens(maxTokens))"
    }

    private func currentContextTokens() -> Int {
        if let inFlightContext = backend.inFlight.lazy.compactMap(\.contextTokens).max() {
            return inFlightContext
        }
        if let sessions = backend.sessions?.sessions, !sessions.isEmpty {
            return sessions.map(\.prefixLen).max() ?? 0
        }
        return 0
    }

    private func contextCaption(lifetime: LifetimeSnapshot?) -> String? {
        guard let lifetime else { return nil }
        return tr("lifetime %@ tok", Format.compactTokens(lifetime.tokensTotal))
    }

    // MARK: Computed helpers

    private func cacheValue(
        smoothed: SmoothedMetrics,
        latest: MetricsLatest?,
        lifetime: LifetimeSnapshot?,
        sessionBank: SessionBank?
    ) -> String {
        // Prefer the smoothed live cached-token count when available so
        // the headline doesn't strobe between two close integers per
        // progress frame. Falls through to SessionBank restore evidence
        // because AR/OpenCode requests can reuse a restored prefix even
        // when the request envelope reports a cold prefill.
        if let smoothedCached = smoothed.cachedTokens, smoothedCached >= 0.5 {
            return Format.integer(Int(smoothedCached.rounded()))
        }
        if let cached = latest?.cachedTokens, cached > 0 {
            return Format.integer(cached)
        }
        if let total = lifetime?.cachedTokensTotal, total > 0 {
            return Format.integer(total)
        }
        return Format.integer(sessionBank?.lastEffectiveCachedTokens)
    }

    private func cacheHitCaption(
        latest: MetricsLatest?,
        lifetime: LifetimeSnapshot?,
        sessionBank: SessionBank?
    ) -> String? {
        if latest?.sessionCacheHit == true {
            if let total = lifetime?.cachedTokensTotal, total > 0 {
                return tr("hit · total %@ tok", Format.integer(total))
            }
            return "hit"
        }
        if sessionBank?.hasEffectiveCacheHit == true {
            let source = sessionBank?.lastEffectiveCacheSource ?? "cache"
            let hits = sessionBank?.restoreHitCount ?? 0
            if hits > 0 {
                return tr("%@ hit", source)
            }
            return tr("%@ hit", source)
        }
        if let total = lifetime?.tokensTotal, let cached = lifetime?.cachedTokensTotal, total > 0 {
            let rate = Double(cached) / Double(total)
            return tr("rate %@", Format.percent(rate))
        }
        return "miss"
    }

    private func memoryValue(mem: MemSnapshot?) -> String {
        let used = (mem?.activeMemoryBytes ?? 0) + (mem?.cacheMemoryBytes ?? 0)
        if used == 0 { return "—" }
        return Format.gigabytes(used)
    }

    private func memoryUnit(machine: HealthPayload?) -> String? {
        guard let total = machine?.unifiedMemoryBytes, total > 0 else { return nil }
        return "/ \(Format.gigabytes(total))"
    }

    private func memoryCaption(
        mem: MemSnapshot?, machine: HealthPayload?, pressureLevel: Int = 0,
        recentShed: Bool = false
    ) -> String? {
        guard let used = mem.flatMap({ ($0.activeMemoryBytes ?? 0) + ($0.cacheMemoryBytes ?? 0) }),
              let total = machine?.unifiedMemoryBytes,
              total > 0
        else { return nil }
        let pct = Double(used) / Double(total)
        // The raw (unclamped) percentage stays — 129% is a receipt, not a
        // rendering bug — but past 100% or under pressure the caption says
        // what it means instead of leaving the user to guess (#305).
        let base = tr("%@ used", Format.percent(pct, fractionDigits: 0))
        if pressureLevel >= 4 { return base + tr(" · critical pressure") }
        // Same contract as the banner (605a1006): a shed claim requires an
        // actual shed in the guard ring, never pressure level alone.
        if pressureLevel >= 2 {
            return base + (recentShed ? tr(" · pressure, shedding cache") : tr(" · memory pressure"))
        }
        if pct > 1.0 { return base + tr(" · over budget") }
        return base
    }

}
