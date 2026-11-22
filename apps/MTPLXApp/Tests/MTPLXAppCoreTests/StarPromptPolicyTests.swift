import XCTest
import Foundation
@testable import MTPLXAppCore

/// The one-time GitHub star prompt: it is owed after the third successful
/// model load, the count and the answer survive the settings file, and
/// nothing winds them back.
final class StarPromptPolicyTests: XCTestCase {
    private let stamp = Date(timeIntervalSinceReferenceDate: 800_000_000)

    private func settingsURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-star-prompt-\(UUID().uuidString)")
            .appendingPathComponent("settings.json")
    }

    func testOwedOnlyAfterTheThirdLoadOnceOnboarded() {
        var config = MTPLXAppConfiguration(onboardingCompletedAt: stamp)
        XCTAssertEqual(MTPLXAppConfiguration.starPromptLoadThreshold, 3)
        for loads in 0..<3 {
            config.successfulModelLoads = loads
            XCTAssertFalse(config.shouldOfferStarPrompt, "not owed after \(loads) loads")
        }
        config.successfulModelLoads = 3
        XCTAssertTrue(config.shouldOfferStarPrompt)
        config.starPromptCompletedAt = stamp
        XCTAssertFalse(config.shouldOfferStarPrompt, "answered once is answered")
    }

    func testNeverOwedDuringOnboarding() {
        let config = MTPLXAppConfiguration(successfulModelLoads: 5)
        XCTAssertNil(config.onboardingCompletedAt)
        XCTAssertFalse(config.shouldOfferStarPrompt)
    }

    func testRecordRoundTripsThroughTheSettingsCodec() throws {
        var config = MTPLXAppConfiguration(onboardingCompletedAt: stamp)
        config.successfulModelLoads = 3
        config.starPromptCompletedAt = stamp.addingTimeInterval(60)
        let data = try JSONEncoder().encode(config)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["successful_model_loads"] as? Int, 3)
        XCTAssertNotNil(object["star_prompt_completed_at"])

        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)
        XCTAssertEqual(decoded.successfulModelLoads, 3)
        XCTAssertEqual(decoded.starPromptCompletedAt, stamp.addingTimeInterval(60))
        XCTAssertFalse(decoded.shouldOfferStarPrompt)
    }

    func testExistingSettingsFileWithoutTheKeysStartsCountingFromZero() throws {
        let json = """
        {"onboarding_completed_at": 800000000.0}
        """
        let config = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.successfulModelLoads, 0)
        XCTAssertNil(config.starPromptCompletedAt)
        XCTAssertFalse(config.shouldOfferStarPrompt)
    }

    func testWrongTypedFieldsDegradeInsteadOfLosingTheSettingsFile() throws {
        let json = """
        {"onboarding_completed_at": 800000000.0, "successful_model_loads": "three", "star_prompt_completed_at": "yesterday"}
        """
        let config = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.successfulModelLoads, 0)
        XCTAssertNil(config.starPromptCompletedAt)
    }

    @MainActor
    func testCountsLoadsUpToTheThresholdAndPersistsThem() throws {
        let url = settingsURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(onboardingCompletedAt: stamp),
            settingsStore: MTPLXSettingsStore(settingsURL: url)
        )
        store.recordSuccessfulModelLoad()
        store.recordSuccessfulModelLoad()
        XCTAssertEqual(store.configuration.successfulModelLoads, 2)
        XCTAssertFalse(store.configuration.shouldOfferStarPrompt)
        store.recordSuccessfulModelLoad()
        XCTAssertTrue(store.configuration.shouldOfferStarPrompt)
        store.recordSuccessfulModelLoad()
        XCTAssertEqual(store.configuration.successfulModelLoads, 3, "counting stops once the prompt is owed")

        let reloaded = try MTPLXSettingsStore(settingsURL: url).load()
        XCTAssertEqual(reloaded.successfulModelLoads, 3, "the count survives a relaunch")
        XCTAssertTrue(reloaded.shouldOfferStarPrompt)
    }

    @MainActor
    func testAnswerStopsCountingAndSurvivesARelaunch() throws {
        let url = settingsURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(onboardingCompletedAt: stamp, successfulModelLoads: 3),
            settingsStore: MTPLXSettingsStore(settingsURL: url)
        )
        store.markStarPromptCompleted(at: stamp)
        store.markStarPromptCompleted(at: stamp.addingTimeInterval(500))
        XCTAssertEqual(store.configuration.starPromptCompletedAt, stamp, "the first answer is kept")
        store.recordSuccessfulModelLoad()
        XCTAssertEqual(store.configuration.successfulModelLoads, 3)

        let reloaded = try MTPLXSettingsStore(settingsURL: url).load()
        XCTAssertEqual(reloaded.starPromptCompletedAt, stamp)
        XCTAssertFalse(reloaded.shouldOfferStarPrompt, "it does not come back after a relaunch")
    }

    @MainActor
    func testAStaleSettingsCopyCannotWindTheRecordBack() async throws {
        let url = settingsURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(onboardingCompletedAt: stamp),
            settingsStore: MTPLXSettingsStore(settingsURL: url)
        )
        // A Settings draft taken before the loads and the answer.
        var staleDraft = store.configuration
        staleDraft.port = 8123
        for _ in 0..<3 { store.recordSuccessfulModelLoad() }
        store.markStarPromptCompleted(at: stamp)

        try await store.applyConfiguration(staleDraft, restartIfRunning: false)
        XCTAssertEqual(store.configuration.port, 8123, "the draft's own edit lands")
        XCTAssertEqual(store.configuration.successfulModelLoads, 3)
        XCTAssertEqual(store.configuration.starPromptCompletedAt, stamp)

        try store.saveSettings(staleDraft)
        XCTAssertEqual(store.configuration.successfulModelLoads, 3)
        XCTAssertEqual(store.configuration.starPromptCompletedAt, stamp)
        let reloaded = try MTPLXSettingsStore(settingsURL: url).load()
        XCTAssertEqual(reloaded.starPromptCompletedAt, stamp)
    }

    @MainActor
    func testALaunchThatReachesReadyCountsAsALoad() async throws {
        let url = settingsURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let supervisor = DaemonSupervisor()
        let store = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(onboardingCompletedAt: stamp),
            settingsStore: MTPLXSettingsStore(settingsURL: url),
            supervisor: supervisor
        )
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let running = supervisor.supervisionSnapshot()
        await store.finishReadyDaemon(
            target: nil,
            configuration: store.configuration,
            replaceExistingClient: false,
            launchID: nil,
            lifecycleEpoch: running.lifecycleEpoch
        )
        XCTAssertEqual(store.configuration.successfulModelLoads, 1)
        await supervisor.stop(graceSeconds: 0)
        await store.awaitDaemonTeardown()
    }
}
