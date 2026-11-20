import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// The three readouts that misled on 12 November, pinned on the numbers
/// from that session's request log and snapshot.
final class ReadoutsTests: XCTestCase {
    private func row(_ values: [String: JSONValue]) -> MetricsLatest {
        MetricsLatest(values: values)
    }

    private func prefillRow(new: Int, rate: Double, cached: Int = 0) -> MetricsLatest {
        row([
            "new_prefill_tokens": .number(Double(new)),
            "prefill_tok_s": .number(rate),
            "cached_tokens": .number(Double(cached)),
        ])
    }

    // MARK: Prefill

    func testACachedTurnThatReadNineteenTokensHasNoPrefillRate() {
        // The founder's last Pi turn: 19 new tokens, 155,893 cached, "153 tok/s".
        let cached = prefillRow(new: 19, rate: 153, cached: 155_893)
        XCTAssertNil(cached.readablePrefillTokS)
        let long = prefillRow(new: 4_096, rate: 2_500)
        XCTAssertEqual(long.readablePrefillTokS, 2_500)
        XCTAssertNil(prefillRow(new: 2_047, rate: 1_800).readablePrefillTokS)
        XCTAssertEqual(prefillRow(new: 2_048, rate: 1_800).readablePrefillTokS, 1_800)
    }

    func testAveragePrefillCountsOnlyTurnsThatReadARealAmount() throws {
        let rows = [
            prefillRow(new: 19, rate: 153, cached: 155_893),
            prefillRow(new: 4_096, rate: 2_500),
            prefillRow(new: 8_192, rate: 2_400),
            prefillRow(new: 600, rate: 900, cached: 120_000),
        ]
        let summary = try XCTUnwrap(PrefillSpeedSummary(recent: rows))
        XCTAssertEqual(summary.turns, 2)
        XCTAssertEqual(summary.peakTokS, 2_500)
        // All new tokens over all reading time: weighted by what each cost.
        let expected = (4_096.0 + 8_192.0) / (4_096.0 / 2_500 + 8_192.0 / 2_400)
        XCTAssertEqual(summary.averageTokS, expected, accuracy: 0.001)
        XCTAssertGreaterThan(summary.averageTokS, 2_400)
    }

    func testNoAverageWhenEveryTurnWasCachedOrShort() {
        XCTAssertNil(PrefillSpeedSummary(recent: [
            prefillRow(new: 19, rate: 153, cached: 155_893),
            prefillRow(new: 512, rate: 700),
        ]))
        XCTAssertNil(PrefillSpeedSummary(recent: []))
    }

    private func throttledRow(new: Int, rate: Double, pressure: String) -> MetricsLatest {
        var values = prefillRow(new: new, rate: rate).values
        values["thermal_pressure_prefill_max"] = .string(pressure)
        return row(values)
    }

    func testAReadOnAThrottledChipSaysSo() {
        // 12 November: 6,843 new tokens at 91K read at 868 tok/s, heavy pressure.
        XCTAssertTrue(throttledRow(new: 6_843, rate: 868, pressure: "heavy").prefillThrottled)
        XCTAssertTrue(throttledRow(new: 6_843, rate: 868, pressure: "trapping").prefillThrottled)
        XCTAssertFalse(throttledRow(new: 6_843, rate: 1_684, pressure: "moderate").prefillThrottled)
        XCTAssertFalse(throttledRow(new: 6_843, rate: 1_684, pressure: "nominal").prefillThrottled)
        // An older server reports no prefill level: no claim either way.
        XCTAssertFalse(prefillRow(new: 6_843, rate: 868).prefillThrottled)
    }

    func testRecentRowsCountTheReadsHeatSlowed() throws {
        let summary = try XCTUnwrap(PrefillSpeedSummary(recent: [
            prefillRow(new: 50_000, rate: 2_200),
            throttledRow(new: 6_843, rate: 868, pressure: "heavy"),
            throttledRow(new: 19, rate: 153, pressure: "heavy"),  // cached, not a read
        ]))
        XCTAssertEqual(summary.turns, 2)
        XCTAssertEqual(summary.throttledTurns, 1)
        XCTAssertEqual(summary.caption, "1 of 2 reads throttled")
    }

    func testTheSessionFigureComesFromTheServersOwnTotals() throws {
        let json = """
        {"started_at_s": 1, "uptime_s": 3600, "prompt_tokens_total": 900000,
         "completion_tokens_total": 59420, "cached_tokens_total": 840000, "tokens_total": 959420,
         "requests_total": 43, "cancelled_total": 0,
         "prefill_reads_total": 2, "prefill_read_tokens_total": 56843,
         "prefill_read_seconds_total": 30.611, "prefill_read_peak_tok_s": 2200,
         "prefill_throttled_reads_total": 0}
        """
        let lifetime = try JSONDecoder().decode(LifetimeSnapshot.self, from: Data(json.utf8))
        let summary = try XCTUnwrap(PrefillSpeedSummary(lifetime: lifetime))
        XCTAssertEqual(summary.averageTokS, 56_843 / 30.611, accuracy: 0.001)
        XCTAssertEqual(summary.turns, 2)
        XCTAssertEqual(summary.caption, "peak \(Format.integer(2200))")
        // An older server sends no totals: the tile falls back to recent rows.
        let old = try JSONDecoder().decode(LifetimeSnapshot.self, from: Data("""
        {"started_at_s": 1, "uptime_s": 1, "prompt_tokens_total": 0, "completion_tokens_total": 0,
         "cached_tokens_total": 0, "tokens_total": 0, "requests_total": 0, "cancelled_total": 0}
        """.utf8))
        XCTAssertNil(PrefillSpeedSummary(lifetime: old))
    }

    // MARK: Engine memory

    private func mem(active: Int, cache: Int) throws -> MemSnapshot {
        let json = """
        {"ok": true, "active_memory_bytes": \(active), "cache_memory_bytes": \(cache),
         "phys_footprint_bytes": 107374182400, "host_overhang_bytes": 16384}
        """
        return try JSONDecoder().decode(MemSnapshot.self, from: Data(json.utf8))
    }

    func testMemoryReadsAgainstTheGuardLimitNotTheMacsRAM() throws {
        // System tab at 01:59: MLX active 84.72 GB, pool 16 KB, limit
        // 96.64 GB, while the guard was trimming. 128 GB of RAM showed 33.8%
        // headroom; against the limit the guard's line was over it.
        let limit = 96_636_764_160
        let snapshot = try mem(active: 84_720_000_000, cache: 16_384)
        XCTAssertEqual(snapshot.physFootprintBytes, 107_374_182_400)
        let reading = try XCTUnwrap(EngineMemoryReading(
            mem: snapshot,
            allocatorFraction: 1.034,
            limitBytes: limit
        ))
        XCTAssertEqual(reading.usedBytes, Int((1.034 * Double(limit)).rounded()))
        XCTAssertLessThan(reading.roomBytes, 0, "over the limit, not 43 GB free")
        XCTAssertGreaterThan(reading.fraction, 1.0)
        XCTAssertEqual(reading.chargedOutsideMLXBytes, reading.usedBytes - reading.mlxBytes)
    }

    func testWithoutAGuardReadingMLXsOwnAccountCounts() throws {
        let snapshot = try mem(active: 40_000_000_000, cache: 2_000_000_000)
        let reading = try XCTUnwrap(EngineMemoryReading(
            mem: snapshot, allocatorFraction: 0, limitBytes: 96_636_764_160
        ))
        XCTAssertNil(reading.guardBytes)
        XCTAssertEqual(reading.usedBytes, 42_000_000_000)
        XCTAssertEqual(reading.chargedOutsideMLXBytes, 0)
        // A guard tick older than the last growth never reads below MLX's account.
        let stale = try XCTUnwrap(EngineMemoryReading(
            mem: snapshot, allocatorFraction: 0.30, limitBytes: 96_636_764_160
        ))
        XCTAssertEqual(stale.usedBytes, 42_000_000_000)
        XCTAssertNil(EngineMemoryReading(mem: snapshot, allocatorFraction: 0.5, limitBytes: nil))
    }

    // MARK: Thinking before the answer

    private func thinkingRow(tokens: Int?, seconds: Double?, firstAnswer: Double?) -> MetricsLatest {
        var values: [String: JSONValue] = ["decode_tok_s": .number(70)]
        if let tokens { values["reasoning_tokens"] = .number(Double(tokens)) }
        if let seconds { values["reasoning_s"] = .number(seconds) }
        values["first_visible_s"] = firstAnswer.map { .number($0) } ?? .null
        return row(values)
    }

    private func tenths(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(1)))
    }

    func testATurnThatThoughtSaysForHowLongAndWhenItAnswered() throws {
        L10n.activate(.english)
        let thinking = try XCTUnwrap(TurnThinking(thinkingRow(tokens: 1850, seconds: 34.2, firstAnswer: 35.0)))
        XCTAssertEqual(
            thinking.summary,
            "thought for 34 s, \(Format.integer(1850)) tokens · first answer at 35 s"
        )
    }

    func testATurnThatDidNotThinkShowsNothing() {
        XCTAssertNil(TurnThinking(thinkingRow(tokens: nil, seconds: nil, firstAnswer: 0.7)))
        XCTAssertNil(TurnThinking(thinkingRow(tokens: 0, seconds: nil, firstAnswer: 0.7)))
        XCTAssertNil(TurnThinking(thinkingRow(tokens: 0, seconds: 0.4, firstAnswer: 0.7)))
        XCTAssertNil(TurnThinking(row(["decode_tok_s": .number(70)])), "an engine that does not report it")
    }

    func testShortThinkingKeepsATenthAndLongThinkingReadsInMinutes() throws {
        L10n.activate(.english)
        let short = try XCTUnwrap(TurnThinking(thinkingRow(tokens: 300, seconds: 5.46, firstAnswer: 5.52)))
        XCTAssertEqual(short.summary, "thought for \(tenths(5.5)) s, 300 tokens · first answer at \(tenths(5.5)) s")
        let long = try XCTUnwrap(TurnThinking(thinkingRow(tokens: 9000, seconds: 125, firstAnswer: nil)))
        XCTAssertEqual(long.summary, "thought for \(Format.duration(125)), \(Format.integer(9000)) tokens")
    }

    /// A cold prompt read before the first token is TTFT, not thinking.
    func testThinkingStartsAtTheFirstTokenNotAtArrival() throws {
        L10n.activate(.english)
        var values = thinkingRow(tokens: 1850, seconds: 95.0, firstAnswer: 95.1).values
        values["ttft_s"] = .number(60.0)
        let thinking = try XCTUnwrap(TurnThinking(row(values)))
        XCTAssertEqual(thinking.summary, "thought for 35 s, \(Format.integer(1850)) tokens · first answer at \(Format.duration(95.1))")
    }

    func testAThinkingTurnWithoutATokenCountStillSaysHowLong() throws {
        L10n.activate(.english)
        let thinking = try XCTUnwrap(TurnThinking(thinkingRow(tokens: nil, seconds: 12.4, firstAnswer: 12.9)))
        XCTAssertEqual(thinking.summary, "thought for 12 s · first answer at 13 s")
    }
}
