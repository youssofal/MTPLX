import XCTest
import Foundation
@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// Verifies every dashboard truthfulness, fan-state, metadata, waterfall, and
/// layout clearance invariant across Live, Activity, and System tabs.
final class DashboardTruthfulnessTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    // MARK: 1. SystemTab active fan mode & live RPMs without Thermal Polling

    func testSystemCoolingShowsMaxModeAndLiveRpmsWhenBackgroundPollingIsOff() throws {
        let thermalStatusJSON = """
        {
          "ok": true,
          "current_mode": "max",
          "detection": {"available": true},
          "fan_summary": {
            "ok": true,
            "min_rpm": 7826,
            "max_rpm": 7830,
            "capacity_max_rpm": 7830,
            "fans": [
              {"rpm": 7830, "actual_rpm": 7830, "target_rpm": 7830, "max_capacity_rpm": 7830, "mode": "forced", "raw": {"min_rpm": 2317, "max_rpm": 7830}},
              {"rpm": 7826, "actual_rpm": 7826, "target_rpm": 7830, "max_capacity_rpm": 7830, "mode": "forced", "raw": {"min_rpm": 2317, "max_rpm": 7830}}
            ]
          }
        }
        """
        let thermalStatus = try JSONDecoder().decode(DynamicObject.self, from: Data(thermalStatusJSON.utf8))
        let healthJSON = """
        {
          "ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
          "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
          "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3",
          "fan_mode": "max",
          "thermal_pressure": {"level": "moderate"}
        }
        """
        let health = try JSONDecoder().decode(HealthPayload.self, from: Data(healthJSON.utf8))

        let readout = SystemCoolingReadout(
            pollingEnabled: false,
            thermal: nil,
            thermalStatus: thermalStatus,
            health: health,
            currentFanMode: "max",
            configuredFanMode: "smart"
        )

        XCTAssertEqual(readout.activeFanMode, .max)
        XCTAssertEqual(readout.coolingBadgeText, "Fans: Max")
        XCTAssertEqual(readout.pollingBadgeText, "Live while System is open")
        XCTAssertEqual(readout.thermalStateLabel, "Moderate")
        XCTAssertFalse(readout.isThrottling)
        XCTAssertTrue(readout.hasLiveFans)
        XCTAssertFalse(readout.helperUnavailable)
        XCTAssertEqual(readout.snapshot?.fans.count, 2)
        XCTAssertEqual(readout.snapshot?.fans[0].effectiveRpm, 7830)
        XCTAssertEqual(readout.snapshot?.fans[0].minCapacityRpm, 2317)
        XCTAssertEqual(readout.snapshot?.fans[1].effectiveRpm, 7826)
    }

    func testSystemCoolingFallsBackToHealthFanSummaryBeforeFirstStatusPoll() throws {
        let healthJSON = """
        {
          "ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
          "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
          "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3",
          "fan_mode": "smart",
          "thermal": {
            "fan_summary": {
              "ok": true,
              "fans": [{"actual_rpm": 3200, "target_rpm": 3200, "max_capacity_rpm": 7830, "mode": "auto"}]
            }
          },
          "thermal_pressure": {"level": "heavy"}
        }
        """
        let health = try JSONDecoder().decode(HealthPayload.self, from: Data(healthJSON.utf8))

        let readout = SystemCoolingReadout(
            pollingEnabled: false,
            thermal: nil,
            thermalStatus: nil,
            health: health,
            currentFanMode: nil,
            configuredFanMode: "default"
        )

        XCTAssertEqual(readout.activeFanMode, .smart)
        XCTAssertEqual(readout.coolingBadgeText, "Fans: Smart")
        XCTAssertEqual(readout.thermalStateLabel, "Heavy (throttling)")
        XCTAssertTrue(readout.isThrottling)
        XCTAssertTrue(readout.hasLiveFans)
        XCTAssertEqual(readout.snapshot?.fans.first?.effectiveRpm, 3200)
    }

    // MARK: 2. Hero finish TPS subtitle & DecodeChart vs ALL-TIME MAX consistency

    func testDecodeChartNeverExceedsAllTimeMaxDespiteSpikyLiveHistory() throws {
        let rollingJSON = """
        {
          "window_s": 300,
          "count": 4,
          "min": 33.4,
          "max": 97.7,
          "mean": 63.8,
          "history": [
            {"t": 1, "tok_s": 97.7},
            {"t": 2, "tok_s": 74.1},
            {"t": 3, "tok_s": 50.0},
            {"t": 4, "tok_s": 33.4}
          ],
          "live_history": [
            {"t": 1, "tok_s": 148.2},
            {"t": 2, "tok_s": 139.5},
            {"t": 3, "tok_s": 88.0},
            {"t": 4, "tok_s": 33.4}
          ],
          "sticky_all_time_max": 97.7
        }
        """
        let rolling = try JSONDecoder().decode(RollingMetrics.self, from: Data(rollingJSON.utf8))
        let summary = DecodeChartSummary(rolling: rolling)

        XCTAssertEqual(summary.points, [97.7, 74.1, 50.0, 33.4])
        XCTAssertEqual(summary.sampleCount, 4)
        XCTAssertEqual(summary.allTimeMax, 97.7, accuracy: 0.001)
        XCTAssertLessThanOrEqual(summary.points.max() ?? 0, summary.allTimeMax)
        XCTAssertEqual(summary.min ?? 0, 33.4, accuracy: 0.001)
        XCTAssertEqual(summary.max ?? 0, 97.7, accuracy: 0.001)
        XCTAssertEqual(summary.mean ?? 0, 63.8, accuracy: 0.001)

        // Idle hero clearly labels the held tail speed vs 5-min average.
        let idleCaption = HeroDecodeCaption(
            headlineDecode: .held(value: 33.4, completedAt: Date()),
            isLiveDecode: false,
            hasRealRequest: true,
            rolling: rolling
        )
        XCTAssertEqual(idleCaption.statusSubtitle, "last turn · 5-min avg 63.8 TPS")
        XCTAssertEqual(idleCaption.allTimeMax ?? 0, 97.7, accuracy: 0.001)

        // Active decode clearly distinguishes live decode from idle last-turn hold.
        let liveCaption = HeroDecodeCaption(
            headlineDecode: .live(71.2),
            isLiveDecode: true,
            hasRealRequest: true,
            rolling: rolling
        )
        XCTAssertEqual(liveCaption.statusSubtitle, "live · 5-min avg 63.8 TPS")
    }

    func testAllTimeMaxPromotesIfCompletedHistoryExceedsStickyMax() throws {
        let rollingJSON = """
        {
          "window_s": 300,
          "count": 2,
          "min": 60.0,
          "max": 112.5,
          "mean": 86.25,
          "history": [
            {"t": 1, "tok_s": 60.0},
            {"t": 2, "tok_s": 112.5}
          ],
          "live_history": [],
          "sticky_all_time_max": 97.7
        }
        """
        let rolling = try JSONDecoder().decode(RollingMetrics.self, from: Data(rollingJSON.utf8))
        let summary = DecodeChartSummary(rolling: rolling)
        XCTAssertEqual(summary.allTimeMax, 112.5, accuracy: 0.001)
        XCTAssertLessThanOrEqual(summary.points.max() ?? 0, summary.allTimeMax)
    }

    // MARK: 3. Cached tile compact formatting & current vs lifetime distinction

    func testCachedTileFormatsLargeLifetimeCompactlyAndDistinguishesFallback() throws {
        let lifetimeJSON = """
        {
          "started_at_s": 1, "uptime_s": 600,
          "prompt_tokens_total": 3000000, "completion_tokens_total": 40000,
          "cached_tokens_total": 2572184, "tokens_total": 3040000,
          "requests_total": 18, "cancelled_total": 0
        }
        """
        let lifetime = try JSONDecoder().decode(LifetimeSnapshot.self, from: Data(lifetimeJSON.utf8))
        let latestHit = MetricsLatest(values: [
            "cached_tokens": .number(115_186),
            "session_cache_hit": .bool(true),
        ])

        let hitReadout = CachedTileReadout(
            smoothed: SmoothedMetrics(cachedTokens: 115_186),
            latest: latestHit,
            lifetime: lifetime,
            sessionBank: nil
        )
        XCTAssertEqual(hitReadout.value, Format.integer(115_186))
        XCTAssertEqual(hitReadout.caption, "hit · 2.57M total")
        XCTAssertFalse(hitReadout.isLifetimeFallback)

        // Explicit cache miss keeps current prompt at 0 instead of masquerading lifetime total.
        let latestMiss = MetricsLatest(values: [
            "cached_tokens": .number(0),
            "session_cache_hit": .bool(false),
        ])
        let missReadout = CachedTileReadout(
            smoothed: SmoothedMetrics(),
            latest: latestMiss,
            lifetime: lifetime,
            sessionBank: nil
        )
        XCTAssertEqual(missReadout.value, "0")
        XCTAssertEqual(missReadout.caption, "miss · 2.57M total")
        XCTAssertFalse(missReadout.isLifetimeFallback)

        // Idle snapshot with no per-turn cache field explicitly labels lifetime fallback.
        let idleFallback = CachedTileReadout(
            smoothed: SmoothedMetrics(),
            latest: nil,
            lifetime: lifetime,
            sessionBank: nil
        )
        XCTAssertEqual(idleFallback.value, "2.57M")
        XCTAssertEqual(idleFallback.caption, "lifetime total")
        XCTAssertTrue(idleFallback.isLifetimeFallback)
    }

    // MARK: 4. Inkling K3 VerifyWaterfall breakdown & running empty state

    func testInklingK3VerifyWaterfallShowsDraftForwardAndOverheadSegments() {
        // Inkling K3 with draft_time_s + verify_forward_time_s + verify_time_s.
        let k3Full = MetricsLatest(values: [
            "verify_calls": .number(100),
            "draft_time_s": .number(0.120),
            "verify_forward_time_s": .number(0.280),
            "verify_time_s": .number(0.340),
        ])
        let summaryFull = VerifyWaterfallSummary(latest: k3Full)
        XCTAssertEqual(summaryFull.segments.map(\.key), ["draft", "verify_forward", "verify_overhead"])
        XCTAssertEqual(summaryFull.segments[0].seconds, 0.120, accuracy: 0.0001)
        XCTAssertEqual(summaryFull.segments[1].seconds, 0.280, accuracy: 0.0001)
        XCTAssertEqual(summaryFull.segments[2].seconds, 0.060, accuracy: 0.0001)
        XCTAssertEqual(summaryFull.totalSeconds, 0.460, accuracy: 0.0001)
        XCTAssertEqual(summaryFull.msPerCallSeconds ?? 0, 0.0034, accuracy: 0.00001)
        XCTAssertEqual(summaryFull.footerSummary, "Draft 120 ms · Verify 340 ms")

        // Inkling K3 when only draft_time_s + verify_time_s are reported (no verify_forward_time_s).
        let k3Basic = MetricsLatest(values: [
            "verify_calls": .number(100),
            "draft_time_s": .number(0.110),
            "verify_time_s": .number(0.340),
        ])
        let summaryBasic = VerifyWaterfallSummary(latest: k3Basic)
        XCTAssertEqual(summaryBasic.segments.map(\.key), ["draft", "verify"])
        XCTAssertEqual(summaryBasic.totalSeconds, 0.450, accuracy: 0.0001)

        // Running daemon without completed timing must never say "once a model is running".
        let runningMsg = VerifyWaterfallSummary.emptyStateMessage(daemonState: .running)
        XCTAssertEqual(runningMsg, "Collecting timing on next turn.")
        XCTAssertFalse(runningMsg.lowercased().contains("once a model is running"))

        let stoppedMsg = VerifyWaterfallSummary.emptyStateMessage(daemonState: .stopped)
        XCTAssertEqual(stoppedMsg, "Detailed timing will show here once a model is running.")
    }

    // MARK: 5. ActivityTab metadata truthfulness (effort, held_nbytes, hits, clearance)

    func testAgentLaneDisplaysResolvedReasoningEffortFromRequestOrSnapshot() {
        let withRequestEffort = MetricsLatest(values: [
            "request_client_hint": .string("pi"),
            "request_generation_mode": .string("mtp"),
            "request_effective_mtp_depth": .number(3),
            "request_reasoning_mode": .string("auto"),
            "resolved_reasoning_effort": .string("xhigh"),
            "request_filtered_tool_count": .number(8),
        ])
        let lane = AgentLaneReadout.text(latest: withRequestEffort)
        XCTAssertTrue(lane.contains("effort xhigh"), lane)
        XCTAssertTrue(lane.contains("reasoning auto"), lane)

        // Falls back to snapshot resolved effort when thinking is not disabled.
        let withoutRequestEffort = MetricsLatest(values: [
            "request_client_hint": .string("pi"),
            "request_reasoning_mode": .string("on"),
        ])
        let laneFromSnapshot = AgentLaneReadout.text(
            latest: withoutRequestEffort,
            snapshotResolvedEffort: "high"
        )
        XCTAssertTrue(laneFromSnapshot.contains("effort high"), laneFromSnapshot)

        // Thinking explicitly disabled must not fabricate a snapshot effort.
        let thinkingOff = MetricsLatest(values: [
            "request_client_hint": .string("pi"),
            "request_enable_thinking": .bool(false),
        ])
        let laneOff = AgentLaneReadout.text(
            latest: thinkingOff,
            snapshotResolvedEffort: "xhigh"
        )
        XCTAssertFalse(laneOff.contains("effort"), laneOff)
    }

    func testSessionBankAndEngineSessionsPreserveHeldNbytesAndCumulativeHits() throws {
        let bankJSON = """
        {
          "max_entries": 8,
          "entries": 1,
          "total_nbytes": 0,
          "lease_nbytes": 428000000,
          "hits": 15,
          "ram_hits": 12,
          "ssd_hits": 3,
          "prefixes": [
            {
              "session_id": "sess-alpha",
              "token_hash": "abc",
              "prefix_len": 115186,
              "hits": 2,
              "cumulative_hits": 11,
              "nbytes": 0,
              "held_nbytes": 428000000,
              "created_at_s": 100,
              "last_access_s": 200,
              "has_live_ref": true
            }
          ]
        }
        """
        let bank = try JSONDecoder().decode(SessionBank.self, from: Data(bankJSON.utf8))
        XCTAssertEqual(bank.effectiveTotalNbytes, 428_000_000)
        XCTAssertEqual(bank.prefixes?.first?.effectiveNbytes, 428_000_000)
        XCTAssertEqual(bank.prefixes?.first?.effectiveHits, 11)
        XCTAssertEqual(bank.restoreHitCount, 15)

        // Engine session reporting bytes=0 still resolves non-zero RAM footprint & hits from bank.
        let sessionJSON = """
        {
          "session_id": "sess-alpha",
          "prefix_len": 115186,
          "bytes": 0,
          "last_access_s": 200
        }
        """
        let session = try JSONDecoder().decode(SessionRow.self, from: Data(sessionJSON.utf8))
        XCTAssertEqual(SessionBytesReadout.bytes(for: session, sessionBank: bank), 428_000_000)
        XCTAssertEqual(SessionBytesReadout.hits(for: session, sessionBank: bank), 11)

        // Genuinely zero bytes stays 0.
        let emptySessionJSON = """
        {"session_id": "sess-empty", "prefix_len": 0, "bytes": 0, "last_access_s": 200}
        """
        let emptySession = try JSONDecoder().decode(SessionRow.self, from: Data(emptySessionJSON.utf8))
        XCTAssertEqual(SessionBytesReadout.bytes(for: emptySession, sessionBank: bank), 0)
        XCTAssertEqual(SessionBytesReadout.hits(for: emptySession, sessionBank: bank), 0)
    }

    func testBottomClearancePreventsExpandChatOverlap() {
        XCTAssertGreaterThanOrEqual(DashboardLayout.expandChatBottomClearance, 72)
        XCTAssertGreaterThanOrEqual(DashboardLayout.scrollBottomPadding(daemonRunning: true), 72)
        XCTAssertEqual(DashboardLayout.scrollBottomPadding(daemonRunning: false), 20)
    }
}
