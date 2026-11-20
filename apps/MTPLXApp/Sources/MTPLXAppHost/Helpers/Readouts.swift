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

    /// "2 of 9 reads throttled" when heat slowed any of them, else the peak.
    var caption: String? {
        if throttledTurns > 0 {
            return tr("%@ of %@ reads throttled", Format.integer(throttledTurns), Format.integer(turns))
        }
        return peakTokS.map { tr("peak %@", Format.tps($0)) }
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

    init?(_ row: MetricsLatest) {
        guard let reasoningEnd = row.reasoningS, reasoningEnd.isFinite, reasoningEnd >= 0 else { return nil }
        if let tokens = row.reasoningTokens, tokens <= 0 { return nil }
        let firstToken = row.ttftS.flatMap { $0.isFinite && $0 >= 0 && $0 <= reasoningEnd ? $0 : nil } ?? 0
        seconds = reasoningEnd - firstToken
        tokens = row.reasoningTokens
        firstAnswerSeconds = row.firstVisibleS.flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
    }

    /// "thought for 34 s, 1,850 tokens · first answer at 35 s"
    var summary: String {
        var text = tokens.map {
            tr("thought for %@, %@ tokens", Self.seconds(seconds), Format.integer($0))
        } ?? tr("thought for %@", Self.seconds(seconds))
        if let first = firstAnswerSeconds {
            text += " · " + tr("first answer at %@", Self.seconds(first))
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
