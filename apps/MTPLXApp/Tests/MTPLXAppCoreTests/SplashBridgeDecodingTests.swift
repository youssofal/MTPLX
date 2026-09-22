import XCTest
@testable import MTPLXAppCore

/// The app must decode a Splash-backed daemon with the same DTOs it uses for
/// the MLX one. A missing non-optional field fails the whole decode and blanks
/// the dashboard, so these fixtures are real responses captured from a live
/// `mtplx serve --engine splash`, not hand-written approximations.
final class SplashBridgeDecodingTests: XCTestCase {

    private func fixture(_ name: String) throws -> Data {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: name, withExtension: "json"),
            "missing fixture \(name).json"
        )
        return try Data(contentsOf: url)
    }

    private var decoder: JSONDecoder { MTPLXAPIClient.makeDefaultDecoder() }

    func testHealthDecodes() throws {
        let health = try decoder.decode(HealthPayload.self, from: fixture("splash_health"))
        XCTAssertTrue(health.ok)
        XCTAssertEqual(health.model, "incoai/Qwen3.8-27B-Splash")
        XCTAssertTrue(health.modelPath.hasSuffix("incoai/Qwen3.8-27B-Splash"))
        XCTAssertEqual(health.contextWindow, 131072)
        // What the supervisor verifies a launch against.
        XCTAssertEqual(health.startup?.launchId, "verify-launch-1")
        // Splash drafts with DFlash 2, which is not MTPLX's MTP.
        XCTAssertFalse(health.mtpEnabled)
        XCTAssertEqual(health.startup?.modelControls?.backendID, "splash")
    }

    func testSnapshotDecodesSoTheDashboardRenders() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        XCTAssertEqual(snapshot.modelId, "incoai/Qwen3.8-27B-Splash")
        XCTAssertEqual(snapshot.contextWindow, 131072)
        XCTAssertTrue(snapshot.mem.ok)
        XCTAssertGreaterThan(snapshot.uptimeS, 0)
    }

    /// Captured after two real streamed generations, so the rows the Live tab
    /// draws from are populated rather than structurally valid and empty.
    func testSnapshotCarriesTheSpeedsAndTheMinMaxRow() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        let rolling = snapshot.rolling
        XCTAssertGreaterThanOrEqual(rolling.count, 2)
        let low = try XCTUnwrap(rolling.min), high = try XCTUnwrap(rolling.max)
        XCTAssertGreaterThan(low, 0)
        XCTAssertGreaterThanOrEqual(high, low)
        XCTAssertNotNil(rolling.mean)
        XCTAssertNotNil(rolling.p95)
        XCTAssertGreaterThan(rolling.stickyAllTimeMax, 0)
        XCTAssertFalse(rolling.history.isEmpty)

        // The newest request is last; its envelope is what `latest` becomes.
        let newest = try XCTUnwrap(snapshot.recent.last)
        XCTAssertGreaterThan(try XCTUnwrap(newest.decodeTokS), 0)
        XCTAssertGreaterThan(try XCTUnwrap(newest.prefillTokS), 0)
        XCTAssertNotNil(newest.ttftS)

        // The memory tile shows the engine, not just its KV pages.
        XCTAssertGreaterThan(try XCTUnwrap(snapshot.mem.activeMemoryBytes), 10_000_000_000)
        XCTAssertGreaterThan(snapshot.lifetime.completionTokensTotal, 0)
    }

    /// Avg Prefill, Cached and Context read optional snapshot blocks, which
    /// decode as nil — and render as a dash — when the server omits them.
    func testOptionalBlocksBehindThePrefillCachedAndContextTiles() throws {
        let snapshot = try decoder.decode(
            DashboardSnapshot.self, from: fixture("splash_mtplx_snapshot")
        )
        let rates = try XCTUnwrap(snapshot.prefillRates, "Avg Prefill reads prefill_rates")
        XCTAssertGreaterThan(rates.tokens, 0)
        XCTAssertGreaterThan(try XCTUnwrap(rates.averageTokS), 0)
        XCTAssertNotNil(rates.peakTokS)

        // Second turn of a conversation: most of the prompt was reused.
        let newest = try XCTUnwrap(snapshot.recent.last)
        XCTAssertGreaterThan(newest.values["cached_tokens"]?.intValue ?? 0, 0)
        XCTAssertEqual(newest.values["session_cache_hit"]?.boolValue, true)
        XCTAssertGreaterThan(snapshot.lifetime.cachedTokensTotal, 0)

        // Both turns belong to one conversation, so one session row.
        XCTAssertEqual(snapshot.sessions.count, 1)
        let row = try XCTUnwrap(snapshot.sessions.sessions.first)
        XCTAssertGreaterThan(row.prefixLen, 0)
        XCTAssertGreaterThan(row.bytes, 0)
        XCTAssertNotNil(snapshot.latest)
    }

    func testCapabilitiesDecodeAndDeclareWhatSplashLacks() throws {
        let capabilities = try decoder.decode(
            AppCapabilities.self, from: fixture("splash_mtplx_app_capabilities")
        )
        XCTAssertTrue(capabilities.ok)
        XCTAssertEqual(capabilities.features["mtp"], false)
        XCTAssertEqual(capabilities.features["kv_quantization"], false)
        XCTAssertEqual(capabilities.features["chat"], true)
        XCTAssertGreaterThan(capabilities.snapshotInterval.defaultMs, 0)
    }

    func testSettingsFreezeKVQuantWithAReason() throws {
        let settings = try decoder.decode(
            MutableSettings.self, from: fixture("splash_mtplx_settings")
        )
        let policy = try XCTUnwrap(settings.kvQuantPolicy)
        XCTAssertFalse(policy.supported)
        XCTAssertEqual(policy.modes, ["q8"])
        XCTAssertNotNil(policy.disabledReason, "a locked control must say why")
    }

    /// The params panel writes through /v1/mtplx/settings and adopts the
    /// reply. Splash's reply carries the values it will actually run, so a
    /// top_k past its 32 comes back as 32 and the panel follows.
    func testSettingsReplyCarriesTheValuesSplashRuns() throws {
        let current = try decoder.decode(
            MutableSettings.self, from: fixture("splash_mtplx_settings")
        )
        XCTAssertNotNil(current.temperature)
        XCTAssertNotNil(current.topK)
        XCTAssertEqual(current.samplingDefaults?.topK, 20)
        let updated = try decoder.decode(
            MutableSettings.self, from: fixture("splash_mtplx_settings_update")
        )
        XCTAssertEqual(updated.temperature, 0.7)
        XCTAssertEqual(updated.topK, 32)
        XCTAssertEqual(updated.presencePenalty, 0)
        XCTAssertEqual(updated.reasoning, "on")
        XCTAssertEqual(updated.generationMode, "splash")
        XCTAssertEqual(updated.reasoningEffort, "low")
    }

    /// The params panel shows its effort picker when the reasoning policy
    /// lists levels; Splash's come from the package's own chat template.
    func testSplashAdvertisesItsThinkingLevels() throws {
        let settings = try decoder.decode(
            MutableSettings.self, from: fixture("splash_mtplx_settings")
        )
        let policy = try XCTUnwrap(settings.reasoningPolicy)
        XCTAssertTrue(policy.supported)
        XCTAssertEqual(policy.modes, ["auto", "on", "off"])
        XCTAssertEqual(policy.effortLevels, ["xhigh", "medium", "low"])
        XCTAssertEqual(policy.defaultEffort, "xhigh")
        let health = try decoder.decode(HealthPayload.self, from: fixture("splash_health"))
        XCTAssertEqual(health.startup?.modelControls?.reasoning, policy)
    }

    func testRemainingContractEndpointsDecode() throws {
        let sessions = try decoder.decode(
            SessionsPayload.self, from: fixture("splash_admin_sessions")
        )
        XCTAssertEqual(sessions.count, sessions.sessions.count)
        let prefill = try decoder.decode(
            PrefillHistoryPayload.self, from: fixture("splash_mtplx_prefill_history")
        )
        XCTAssertFalse(prefill.history.isEmpty, "one row per completed request")
        let models = try decoder.decode(ModelsResponse.self, from: fixture("splash_models"))
        XCTAssertEqual(models.data.first?.id, "incoai/Qwen3.8-27B-Splash")
    }

    // MARK: Chat stream

    private var chatClient: MTPLXChatClient {
        MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:1")!))
    }

    private func frame(_ name: String) throws -> String {
        String(decoding: try fixture(name), as: UTF8.self)
    }

    /// The chat's per-reply footer (tok/s, out, in, TTFT) is built from the
    /// finish frame's usage and mtplx_stats. A Splash reply that lacked them
    /// rendered with no footer at all.
    func testSplashFinishFrameCarriesTheChatFooterStats() throws {
        let events = try XCTUnwrap(
            chatClient.streamEvents(fromDataPayload: frame("splash_chat_finish_frame"))
        )
        guard case .finished(let reason, let usage, let stats)? = events.last else {
            return XCTFail("expected .finished, got \(events)")
        }
        XCTAssertEqual(reason, "stop")
        XCTAssertEqual(usage?.promptTokens, 61)
        XCTAssertEqual(usage?.completionTokens, 92)
        // The footer's "cached" item reads the OpenAI-standard usage detail.
        XCTAssertEqual(usage?.cachedTokens, 0)
        let tokS = try XCTUnwrap(stats?.rawDecodeTokS)
        XCTAssertGreaterThan(tokS, 0)
        XCTAssertGreaterThan(try XCTUnwrap(stats?.ttftS), 0)
        // The held header reading divides these two.
        XCTAssertNotNil(stats?.raw.values["completion_tokens"])
        XCTAssertNotNil(stats?.raw.values["decode_elapsed_s"])
    }

    /// The live tok/s chip in the chat header moves on mtplx_progress frames.
    func testSplashProgressFrameDrivesTheLiveChip() throws {
        let events = try XCTUnwrap(
            chatClient.streamEvents(fromDataPayload: frame("splash_chat_progress_frame"))
        )
        guard case .progress(let progress)? = events.first else {
            return XCTFail("expected .progress, got \(events)")
        }
        XCTAssertGreaterThan(try XCTUnwrap(progress.completionTokens), 0)
        XCTAssertNotNil(progress.raw.values["decode_elapsed_s"])
    }
}
