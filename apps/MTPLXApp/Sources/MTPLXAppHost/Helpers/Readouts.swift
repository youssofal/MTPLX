import CoreGraphics
import Foundation
import MTPLXAppCore

// MARK: - Readouts the dashboards share
//
// Numbers the dashboards used to show could mislead: a prefill "rate" of
// 153 tok/s for a cached turn that read 19 new tokens, and 33.8% memory
// headroom while the memory guard was trimming. Each is decided here once,
// so the Live tab, Activity and System cannot disagree about it.

// MARK: Prefill rate

extension MetricsLatest {
    /// New prompt tokens from which a turn's prefill rate is a rate. A
    /// forward pass costs about as much for 19 tokens as for a few hundred,
    /// so below this the figure is one pass's fixed cost divided by a small
    /// count. At 2,048 that fixed cost is under a tenth of the read on the
    /// fastest Mac and smaller on every slower one. The server counts its
    /// session figure from the same number (`READABLE_PREFILL_TOKENS`).
    static let readablePrefillTokens = 2_048

    /// This turn's prefill rate when it read enough new tokens for the rate
    /// to mean something; nil for cached and short turns.
    var readablePrefillTokS: Double? {
        guard let new = newPrefillTokens, new >= Self.readablePrefillTokens,
              let rate = prefillTokS, rate.isFinite, rate > 0
        else { return nil }
        return rate
    }

    /// The worst thermal pressure while this turn's prompt was read, from its
    /// arrival to its first token. Nil from older servers.
    var prefillThermalPressure: String? { values["thermal_pressure_prefill_max"]?.stringValue }

    /// The prompt was read while macOS was slowing the GPU, so its rate is a
    /// throttled one (on 12 November two such reads ran at 845 and 868 tok/s
    /// where a cool Mac read the same kind of turn at 1,684).
    var prefillThrottled: Bool { ThermalPressureNotice.isThrottling(prefillThermalPressure) }
}

/// Prefill speed over turns that read a real amount of new tokens: all their
/// new tokens over all the time spent reading them, so each turn weighs what
/// it cost. Cached and short turns are left out, not averaged in, and the
/// reads taken on a throttled chip are counted so a low figure says why.
struct PrefillSpeedSummary: Equatable {
    var averageTokS: Double
    var peakTokS: Double?
    var turns: Int
    var throttledTurns: Int

    /// Since the server started, from its own totals: the session's figure,
    /// which a window of recent rows would lose its first long read from.
    init?(lifetime: LifetimeSnapshot?) {
        guard let lifetime,
              let turns = lifetime.prefillReadsTotal, turns > 0,
              let tokens = lifetime.prefillReadTokensTotal,
              let seconds = lifetime.prefillReadSecondsTotal, seconds > 0
        else { return nil }
        averageTokS = Double(tokens) / seconds
        peakTokS = lifetime.prefillReadPeakTokS
        self.turns = turns
        throttledTurns = lifetime.prefillThrottledReadsTotal ?? 0
    }

    /// Over the recent rows an older server reports.
    init?(recent: [MetricsLatest]) {
        var tokens = 0.0
        var seconds = 0.0
        var peak = 0.0
        var turns = 0
        var throttled = 0
        for row in recent {
            guard let rate = row.readablePrefillTokS,
                  let new = row.newPrefillTokens
            else { continue }
            tokens += Double(new)
            seconds += Double(new) / rate
            peak = max(peak, rate)
            turns += 1
            throttled += row.prefillThrottled ? 1 : 0
        }
        guard turns > 0, seconds > 0 else { return nil }
        averageTokS = tokens / seconds
        peakTokS = peak
        self.turns = turns
        throttledTurns = throttled
    }

    /// Compact tile caption: "1/4 throttled" when heat slowed any prefill
    /// reads, else the peak read speed, else the count of qualifying reads.
    var caption: String? {
        if throttledTurns > 0 {
            return tr("%@/%@ throttled", Format.integer(throttledTurns), Format.integer(turns))
        }
        if let peakTokS {
            return tr("peak %@", Format.tps(peakTokS))
        }
        return tr("%@ reads", Format.integer(turns))
    }

    /// Hover explanation naming qualifying prefill reads (2K+ new tokens) vs
    /// total turns so compact tile captions stay unambiguous.
    var helpText: String {
        if throttledTurns > 0 {
            return tr(
                "%@ of %@ prefill reads (2K+ new tokens) ran while macOS slowed the GPU.",
                Format.integer(throttledTurns),
                Format.integer(turns)
            )
        }
        return tr(
            "Token-weighted average across %@ prefill reads (2K+ new tokens).",
            Format.integer(turns)
        )
    }
}

// MARK: Engine memory

/// Memory as the memory guard sees it: the engine's line (MLX's account
/// plus the host memory it charges past the seat's allowance) against the
/// Metal memory limit the engine plans within. The guard trims the warm
/// cache and sheds work on this line, so this is what "headroom" means for
/// MTPLX. The Mac's whole RAM is the wrong denominator: on 12 November the
/// app showed a third of the Mac free while the guard was trimming.
struct EngineMemoryReading: Equatable {
    /// The Metal memory limit: the line the guard compares with.
    var limitBytes: Int
    /// MLX's own account: active memory plus the reusable buffer pool.
    var mlxBytes: Int
    /// The guard's last reading (`allocator_fraction` × limit), when the
    /// daemon reported one.
    var guardBytes: Int?

    /// What counts against the limit now: the guard's reading, or MLX's own
    /// account when that is higher (memory grew since the guard's tick) or
    /// the guard has not read yet.
    var usedBytes: Int { max(guardBytes ?? 0, mlxBytes) }
    var fraction: Double { Double(usedBytes) / Double(limitBytes) }
    /// Under the limit (positive) or over it (negative).
    var roomBytes: Int { limitBytes - usedBytes }
    /// Process memory outside MLX's account that the guard counts.
    var chargedOutsideMLXBytes: Int { max(0, usedBytes - mlxBytes) }

    init?(mem: MemSnapshot?, allocatorFraction: Double?, limitBytes: Int?) {
        guard let mem, mem.ok,
              let limitBytes, limitBytes > 0
        else { return nil }
        let mlx = max(0, mem.activeMemoryBytes ?? 0) + max(0, mem.cacheMemoryBytes ?? 0)
        guard mlx > 0 else { return nil }
        self.limitBytes = limitBytes
        mlxBytes = mlx
        if let allocatorFraction, allocatorFraction.isFinite, allocatorFraction > 0 {
            guardBytes = Int((allocatorFraction * Double(limitBytes)).rounded())
        } else {
            guardBytes = nil
        }
    }

    init(limitBytes: Int, mlxBytes: Int, guardBytes: Int?) {
        self.limitBytes = limitBytes
        self.mlxBytes = mlxBytes
        self.guardBytes = guardBytes
    }
}

extension MTPLXBackendStore {
    /// The Metal limit the engine configured, else the plan's usable bytes
    /// (the server plans with that same limit).
    var engineMemoryLimitBytes: Int? {
        health?.metalMemoryCaps?.memoryLimitBytes ?? memoryPlan?.usableBytes
    }

    var engineMemoryReading: EngineMemoryReading? {
        EngineMemoryReading(
            mem: mem,
            allocatorFraction: snapshot?.allocatorFraction,
            limitBytes: engineMemoryLimitBytes
        )
    }
}

// MARK: Thinking before the answer

/// How long a streamed turn thought before it answered, from its request
/// record: the answer to "why did this turn take so long" when the cache hit
/// and the model was reasoning (at xhigh in a long Pi session the first
/// answer text comes 5 to 35 s after the first token). Nil for a turn that
/// did not think and for records from an engine that does not report it.
struct TurnThinking: Equatable {
    /// First token to the last reasoning delta: how long the model reasoned.
    /// The prompt read before the first token is the TTFT column's, so a
    /// cold 100K prefill is not counted as thinking.
    var seconds: Double
    /// Reasoning tokens the stream counted, when reported.
    var tokens: Int?
    /// Request arrival to the first answer text or tool call; nil when the
    /// turn sent none.
    var firstAnswerSeconds: Double?
    /// Kind of first visible output (`"tool_call"`, `"text"`, or nil).
    var firstVisibleKind: String?

    init?(_ row: MetricsLatest) {
        guard let reasoningEnd = row.reasoningS, reasoningEnd.isFinite, reasoningEnd >= 0 else { return nil }
        if let tokens = row.reasoningTokens, tokens <= 0 { return nil }
        let firstToken = row.ttftS.flatMap { $0.isFinite && $0 >= 0 && $0 <= reasoningEnd ? $0 : nil } ?? 0
        seconds = reasoningEnd - firstToken
        tokens = row.reasoningTokens
        firstAnswerSeconds = row.firstVisibleS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        firstVisibleKind = row.firstVisibleKind?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    /// "thought for 34 s, 1,850 tokens · first answer at 35 s" (or
    /// "first tool call at ..." when `first_visible_kind == "tool_call"`).
    var summary: String {
        var text = tokens.map {
            tr("thought for %@, %@ tokens", Self.seconds(seconds), Format.integer($0))
        } ?? tr("thought for %@", Self.seconds(seconds))
        if let first = firstAnswerSeconds {
            if firstVisibleKind == "tool_call" {
                text += " · " + tr("first tool call at %@", Self.seconds(first))
            } else {
                text += " · " + tr("first answer at %@", Self.seconds(first))
            }
        }
        return text
    }

    /// Tenths below 10 s, whole seconds to a minute, then minutes.
    static func seconds(_ value: Double) -> String {
        if value < 10 {
            return tr("%@ s", value.formatted(.number.precision(.fractionLength(1))))
        }
        if value < 60 {
            return tr("%@ s", value.formatted(.number.precision(.fractionLength(0))))
        }
        return Format.duration(value)
    }
}

// MARK: - Dashboard layout

/// Shared clearance for the floating `^ Expand chat` handle when the daemon is
/// running in dashboard mode, so bottom cards in every tab scroll completely
/// clear of the pill.
enum DashboardLayout {
    static let expandChatBottomClearance: CGFloat = 72

    static func scrollBottomPadding(daemonRunning: Bool) -> CGFloat {
        daemonRunning ? 76 : 20
    }
}

// MARK: - Decode chart & hero truthfulness

/// Single completed-turn series powering `DecodeChart`, the `5-min Min/Max`
/// tile, and `ALL-TIME MAX` on the hero gauge. Keeping all three on the same
/// completed-turn series prevents mid-turn instantaneous spikes from plotting
/// above `ALL-TIME MAX` or contradicting the 5-minute window statistics.
struct DecodeChartSummary: Equatable {
    static let defaultMaxRenderedPoints = 240

    let points: [Double]
    let sampleCount: Int
    let min: Double?
    let max: Double?
    let mean: Double?
    let allTimeMax: Double

    init(rolling: RollingMetrics?, maxRenderedPoints: Int = Self.defaultMaxRenderedPoints) {
        guard let rolling else {
            self.points = []
            self.sampleCount = 0
            self.min = nil
            self.max = nil
            self.mean = nil
            self.allTimeMax = 0
            return
        }
        let completed = rolling.history
            .map(\.tokS)
            .filter { $0.isFinite && $0 >= 0 }
        let rendered = Self.downsamplePreservingPeak(completed, maxCount: maxRenderedPoints)
        let seriesPeak = completed.max()
        let seriesMin = completed.min()
        let resolvedMax: Double? = {
            switch (rolling.max, seriesPeak) {
            case let (a?, b?): return Swift.max(a, b)
            case let (a?, nil): return a
            case let (nil, b?): return b
            case (nil, nil): return nil
            }
        }()
        let resolvedMin: Double? = {
            switch (rolling.min, seriesMin) {
            case let (a?, b?): return Swift.min(a, b)
            case let (a?, nil): return a
            case let (nil, b?): return b
            case (nil, nil): return nil
            }
        }()
        let resolvedMean: Double? = {
            if let mean = rolling.mean, mean.isFinite { return mean }
            guard !completed.isEmpty else { return nil }
            return completed.reduce(0, +) / Double(completed.count)
        }()
        let sticky = rolling.stickyAllTimeMax.isFinite ? Swift.max(0, rolling.stickyAllTimeMax) : 0
        self.points = rendered
        self.sampleCount = completed.count
        self.min = resolvedMin
        self.max = resolvedMax
        self.mean = resolvedMean
        self.allTimeMax = Swift.max(sticky, resolvedMax ?? 0)
    }

    private static func downsamplePreservingPeak(_ points: [Double], maxCount: Int) -> [Double] {
        guard maxCount > 1, points.count > maxCount else { return points }
        let stride = Double(points.count - 1) / Double(maxCount - 1)
        var result = (0..<maxCount).map { idx in
            points[Swift.min(points.count - 1, Int((Double(idx) * stride).rounded()))]
        }
        if let peakIdx = points.indices.max(by: { points[$0] < points[$1] }) {
            let mapped = Swift.min(
                maxCount - 1,
                Swift.max(0, Int((Double(peakIdx) / stride).rounded()))
            )
            result[mapped] = points[peakIdx]
        }
        return result
    }
}

/// Clarifies what the hero gauge is displaying (live decode vs finished last
/// turn held at rest alongside the 5-minute average) and exposes the
/// consistent `ALL-TIME MAX` value.
struct HeroDecodeCaption: Equatable {
    let statusSubtitle: String?
    let allTimeMax: Double?

    init(
        headlineDecode: HeadlineDecodeReading,
        isLiveDecode: Bool,
        hasRealRequest: Bool,
        rolling: RollingMetrics?
    ) {
        let summary = DecodeChartSummary(rolling: hasRealRequest ? rolling : nil)
        let avg = summary.mean.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }

        if isLiveDecode {
            if let avg {
                self.statusSubtitle = tr("live · 5-min avg %@ TPS", Format.tps(avg))
            } else {
                self.statusSubtitle = tr("live decode")
            }
        } else if hasRealRequest, headlineDecode.value != nil {
            if let avg {
                self.statusSubtitle = tr("last turn · 5-min avg %@ TPS", Format.tps(avg))
            } else {
                self.statusSubtitle = tr("last turn")
            }
        } else {
            self.statusSubtitle = nil
        }

        if hasRealRequest, summary.allTimeMax > 0 {
            self.allTimeMax = summary.allTimeMax
        } else {
            self.allTimeMax = nil
        }
    }
}

// MARK: - Cached tile readout

/// Compact, truthful readout for the `CACHED` tile on LiveTab. Keeps current
/// prompt cached tokens strictly distinct from lifetime cached tokens and
/// formats large totals compactly (`2.57M total`) so the caption never
/// truncates with an ellipsis.
struct CachedTileReadout: Equatable {
    let value: String
    let unit: String?
    let caption: String?
    let isLifetimeFallback: Bool

    static func formatCompactTokens(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "—" }
        let locale = L10n.language.locale
        if value < 1_000 { return String(value) }
        if value < 9_950 {
            return String(format: "%.1fK", locale: locale, Double(value) / 1_000.0)
        }
        if value < 999_500 {
            return String(format: "%.0fK", locale: locale, Double(value) / 1_000.0)
        }
        if value < 9_995_000 {
            return String(format: "%.2fM", locale: locale, Double(value) / 1_000_000.0)
        }
        if value < 99_950_000 {
            return String(format: "%.1fM", locale: locale, Double(value) / 1_000_000.0)
        }
        if value < 999_500_000 {
            return String(format: "%.0fM", locale: locale, Double(value) / 1_000_000.0)
        }
        return String(format: "%.2fB", locale: locale, Double(value) / 1_000_000_000.0)
    }

    init(
        smoothed: SmoothedMetrics,
        latest: MetricsLatest?,
        lifetime: LifetimeSnapshot?,
        sessionBank: SessionBank?
    ) {
        let lifetimeTotal = lifetime.flatMap { $0.cachedTokensTotal > 0 ? $0.cachedTokensTotal : nil }
        let currentCached: Int? = {
            if let s = smoothed.cachedTokens, s >= 0.5 {
                return Int(s.rounded())
            }
            if let c = latest?.cachedTokens, c > 0 {
                return c
            }
            if latest?.sessionCacheHit != false,
               let b = sessionBank?.lastEffectiveCachedTokens,
               b > 0 {
                return b
            }
            return nil
        }()

        if let current = currentCached {
            self.value = Format.integer(current)
            self.unit = "tok"
            self.isLifetimeFallback = false
            if latest?.sessionCacheHit == true {
                if let total = lifetimeTotal {
                    self.caption = tr("hit · %@ total", Self.formatCompactTokens(total))
                } else {
                    self.caption = tr("hit")
                }
            } else if sessionBank?.hasEffectiveCacheHit == true {
                let source = sessionBank?.lastEffectiveCacheSource ?? "cache"
                if let total = lifetimeTotal {
                    self.caption = tr("%@ · %@ total", source, Self.formatCompactTokens(total))
                } else {
                    self.caption = tr("%@ hit", source)
                }
            } else if let total = lifetimeTotal {
                self.caption = tr("%@ total", Self.formatCompactTokens(total))
            } else {
                self.caption = nil
            }
        } else if latest?.sessionCacheHit == false || latest?.cachedTokens == 0 {
            self.value = "0"
            self.unit = "tok"
            self.isLifetimeFallback = false
            if let total = lifetimeTotal {
                self.caption = tr("miss · %@ total", Self.formatCompactTokens(total))
            } else {
                self.caption = tr("miss")
            }
        } else if let total = lifetimeTotal {
            self.value = Self.formatCompactTokens(total)
            self.unit = "tok"
            self.isLifetimeFallback = true
            self.caption = tr("lifetime total")
        } else {
            self.value = "—"
            self.unit = "tok"
            self.isLifetimeFallback = false
            self.caption = nil
        }
    }
}

// MARK: - Verify waterfall summary

/// Breakdown of speculative draft + verify wall time for `VerifyWaterfallExpander`.
/// Supports Qwen MTP sub-phase receipts, Gemma 4 assistant receipts, and
/// Inkling K3 receipts (`draft_time_s`, `verify_forward_time_s`, `verify_time_s`).
struct VerifyWaterfallSummary: Equatable {
    struct Segment: Equatable {
        let key: String
        let label: String
        let seconds: Double
    }

    let segments: [Segment]
    let totalSeconds: Double
    let draftSeconds: Double?
    let verifySeconds: Double?
    let verifyForwardSeconds: Double?
    let verifyCalls: Int?

    var msPerCallSeconds: Double? {
        guard let calls = verifyCalls, calls > 0, let verify = verifySeconds, verify > 0 else {
            return nil
        }
        return verify / Double(calls)
    }

    init(latest: MetricsLatest?) {
        func positive(_ value: Double?) -> Double? {
            guard let value, value.isFinite, value > 0 else { return nil }
            return value
        }

        let draft = positive(latest?.draftTimeS)
        let verify = positive(latest?.verifyTimeS)
        let verifyForward = positive(latest?.verifyForwardTimeS)
        let targetDist = positive(latest?.verifyTargetDistTimeS ?? latest?.verifyLogitsEvalTimeS)
        let verifyHidden = positive(latest?.verifyHiddenEvalTimeS)
        let verifyOther = positive(latest?.verifyEvalUnattributedS)

        self.draftSeconds = draft
        self.verifySeconds = verify
        self.verifyForwardSeconds = verifyForward
        self.verifyCalls = latest?.verifyCalls.flatMap { $0 > 0 ? $0 : nil }

        var built: [Segment] = []
        if let draft {
            built.append(Segment(key: "draft", label: "draft", seconds: draft))
        }

        let hasDetailedSubphases = targetDist != nil || verifyHidden != nil || verifyOther != nil
        if hasDetailedSubphases {
            if let vf = verifyForward {
                built.append(Segment(key: "verify_forward", label: "verify_forward", seconds: vf))
            }
            if let td = targetDist {
                built.append(Segment(key: "target_dist", label: "target_dist", seconds: td))
            }
            if let vh = verifyHidden {
                built.append(Segment(key: "verify_hidden", label: "verify_hidden", seconds: vh))
            }
            if let vo = verifyOther {
                built.append(Segment(key: "verify_other", label: "verify_other", seconds: vo))
            }
            let subSum = (verifyForward ?? 0) + (targetDist ?? 0) + (verifyHidden ?? 0) + (verifyOther ?? 0)
            if let verify, verifyOther == nil {
                let overhead = verify - subSum
                if overhead >= 0.0005 {
                    built.append(Segment(key: "verify_overhead", label: "verify_overhead", seconds: overhead))
                }
            }
        } else if let vf = verifyForward {
            if let verify {
                let forwardPart = min(vf, verify)
                built.append(Segment(key: "verify_forward", label: "verify_forward", seconds: forwardPart))
                let overhead = verify - forwardPart
                if overhead >= 0.0005 {
                    built.append(Segment(key: "verify_overhead", label: "verify_overhead", seconds: overhead))
                }
            } else {
                built.append(Segment(key: "verify_forward", label: "verify_forward", seconds: vf))
            }
        } else if let verify {
            built.append(Segment(key: "verify", label: "verify", seconds: verify))
        }

        self.segments = built
        self.totalSeconds = built.reduce(0) { $0 + $1.seconds }
    }

    var footerSummary: String? {
        switch (draftSeconds, verifySeconds) {
        case let (d?, v?):
            return tr("Draft %@ · Verify %@", Format.duration(d), Format.duration(v))
        case let (nil, v?):
            return tr("Verify: %@", Format.duration(v))
        case let (d?, nil):
            return tr("Draft: %@", Format.duration(d))
        case (nil, nil):
            return nil
        }
    }

    static func emptyStateMessage(daemonState: DaemonStateKind) -> String {
        if daemonState == .running {
            return tr("Collecting timing on next turn.")
        }
        return tr("Detailed timing will show here once a model is running.")
    }
}

// MARK: - System cooling readout

/// Active fan mode, live RPMs, and macOS SoC thermal state for `SystemTab`.
/// Surfaces real active fan mode and live fan RPMs whenever thermal status is
/// available, even while background Thermal Polling is switched off.
struct SystemCoolingReadout: Equatable {
    let activeFanMode: MTPLXFanMode
    let snapshot: ThermalSnapshot?
    let pollingEnabled: Bool
    let helperUnavailable: Bool
    let thermalPressureLevel: String?

    var hasLiveFans: Bool {
        snapshot?.ok == true && !(snapshot?.fans.isEmpty ?? true)
    }

    var isThrottling: Bool {
        ThermalPressureNotice.isThrottling(thermalPressureLevel)
    }

    var thermalStateLabel: String? {
        switch thermalPressureLevel?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "nominal": return tr("Nominal")
        case "moderate": return tr("Moderate")
        case "heavy": return tr("Heavy (throttling)")
        case "trapping": return tr("Trapping (throttling)")
        case "sleeping": return tr("Critical (throttling)")
        default: return nil
        }
    }

    var coolingBadgeText: String {
        tr("Fans: %@", activeFanMode.title)
    }

    var pollingBadgeText: String {
        pollingEnabled ? tr("Background polling on") : tr("Live while System is open")
    }

    init(
        pollingEnabled: Bool,
        thermal: ThermalSnapshot?,
        thermalStatus: DynamicObject?,
        health: HealthPayload?,
        currentFanMode: String?,
        configuredFanMode: String
    ) {
        self.pollingEnabled = pollingEnabled

        let rawMode = [
            thermalStatus?.string("current_mode"),
            currentFanMode,
            health?.fanMode,
            configuredFanMode,
        ]
        .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
        .first { !$0.isEmpty }
        self.activeFanMode = MTPLXFanMode.normalized(rawMode)

        let statusSummary = ThermalSnapshot.from(fanSummary: thermalStatus?.object("fan_summary"))
        let healthSummary = ThermalSnapshot.from(fanSummary: health?.thermal?.fanSummary?.values)
        let candidates: [ThermalSnapshot?] = pollingEnabled
            ? [thermal, statusSummary, healthSummary]
            : [statusSummary, thermal, healthSummary]

        let live = candidates.compactMap { $0 }.first { $0.ok && !$0.fans.isEmpty }
        let resolved = live ?? candidates.compactMap { $0 }.first
        self.snapshot = resolved

        let statusExplicitlyFailed = thermalStatus?.bool("ok") == false
            && (thermalStatus?.object("detection")?["available"]?.boolValue != true)
        self.helperUnavailable = live == nil && (resolved?.ok == false || statusExplicitlyFailed)
        self.thermalPressureLevel = health?.thermalPressure?.level
    }
}

// MARK: - Activity truthfulness readouts

enum AgentLaneReadout {
    static func resolvedEffort(
        latest: MetricsLatest?,
        snapshotResolvedEffort: String? = nil,
        snapshotSettingsEffort: String? = nil
    ) -> String? {
        if let perRequest = latest?.resolvedReasoningEffort, !perRequest.isEmpty {
            return perRequest
        }
        guard latest?.requestEnableThinking != false,
              latest?.requestReasoningMode?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "off"
        else { return nil }
        for candidate in [snapshotResolvedEffort, snapshotSettingsEffort] {
            if let trimmed = candidate?.trimmingCharacters(in: .whitespacesAndNewlines),
               !trimmed.isEmpty {
                return trimmed
            }
        }
        return nil
    }

    static func text(
        latest: MetricsLatest?,
        snapshotResolvedEffort: String? = nil,
        snapshotSettingsEffort: String? = nil
    ) -> String {
        guard let latest else { return tr("No agent lane data yet.") }
        var parts: [String] = []
        if let client = latest.requestClientHint, !client.isEmpty {
            parts.append(client)
        }
        if let mode = latest.requestGenerationMode, !mode.isEmpty {
            parts.append(mode.uppercased())
        }
        if let depth = latest.requestEffectiveMtpDepth
            ?? latest.requestDepth
            ?? latest.mtpDepth
            ?? latest.speculativeDepth {
            parts.append(tr("depth %lld", depth))
        }
        if let reasoning = latest.requestReasoningMode, !reasoning.isEmpty {
            parts.append(tr("reasoning %@", reasoning))
        } else if let thinking = latest.requestEnableThinking {
            parts.append(thinking ? tr("thinking on") : tr("thinking off"))
        }
        if let effort = resolvedEffort(
            latest: latest,
            snapshotResolvedEffort: snapshotResolvedEffort,
            snapshotSettingsEffort: snapshotSettingsEffort
        ) {
            parts.append(tr("effort %@", effort))
        }
        if let filteredCount = latest.requestFilteredToolCount {
            parts.append(tr("%@ active tools", Format.integer(filteredCount)))
        } else if let filtered = latest.requestFilteredToolNames {
            parts.append(tr("%@ active tools", Format.integer(filtered.count)))
        } else if let tools = latest.requestToolNames {
            parts.append(tr("%lld tools", tools.count))
        } else if let toolCount = latest.requestToolCount {
            parts.append(tr("%@ tools", Format.integer(toolCount)))
        }
        if let hidden = latest.requestHiddenToolNames, !hidden.isEmpty {
            parts.append(tr("%lld hidden", hidden.count))
        }
        return parts.isEmpty ? tr("No agent lane data yet.") : parts.joined(separator: " · ")
    }
}

/// Resolves real RAM bytes and cumulative hit counts across `SessionRow` and
/// `SessionBank` so live-reference leases (`held_nbytes`) never show as 0 KB.
enum SessionBytesReadout {
    static func bytes(for session: SessionRow, sessionBank: SessionBank?) -> Int {
        let bankBytes = sessionBank?.prefixes?
            .filter { $0.sessionId == session.sessionId }
            .reduce(0) { $0 + max(0, $1.effectiveNbytes) } ?? 0
        return max(session.effectiveBytes, bankBytes)
    }

    static func hits(for session: SessionRow, sessionBank: SessionBank?) -> Int {
        let bankHits = sessionBank?.prefixes?
            .filter { $0.sessionId == session.sessionId }
            .reduce(0) { $0 + max(0, $1.effectiveHits) } ?? 0
        return max(session.effectiveHits ?? 0, bankHits)
    }
}
