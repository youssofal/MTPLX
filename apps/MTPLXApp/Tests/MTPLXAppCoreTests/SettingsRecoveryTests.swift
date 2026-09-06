import XCTest

@testable import MTPLXAppCore

/// A `settings.json` that does not decode cleanly must never silently
/// become defaults: one bad field degrades only that field, a bad tuned
/// record or custom model is skipped while its siblings load, and a file
/// that cannot be read at all is kept beside itself (dated, never deleted)
/// before the app falls back and says so.
final class SettingsRecoveryTests: XCTestCase {
    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-settings-recovery-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeStore() -> MTPLXSettingsStore {
        MTPLXSettingsStore(settingsURL: temporaryDirectory().appendingPathComponent("settings.json"))
    }

    private func write(_ json: String, to store: MTPLXSettingsStore) throws {
        try Data(json.utf8).write(to: store.settingsURL)
    }

    /// The reference date is 2001-01-01; 800_000_000 s later is a fixed,
    /// non-nil onboarding stamp that survives the default Date strategy.
    private let onboardingStamp = 800_000_000.0

    private func goodRecord(model: String, value: Int) -> String {
        """
        {"schema_version": 1, "model_id": "\(model)", "model_family": "qwen", "backend_id": "qwen3_next",
         "control_field": "depth", "control_value": \(value), "candidates": ["1", "2", "3"], "tuned_at": 700000000}
        """
    }

    private let customModelJSON = """
    {"id": "custom-a", "displayName": "Custom A", "shortName": "A", "detail": "mine",
     "hfModelID": "someone/custom-a", "localCandidates": []}
    """

    // MARK: - Lenient fields

    func testWrongTypedFieldDegradesOnlyThatFieldAndKeepsOnboarding() throws {
        let store = makeStore()
        try write("""
        {
          "port": "eight thousand",
          "host": "0.0.0.0",
          "api_key": "sk-live-secret-123",
          "hugging_face_handle": "someone",
          "hf_endpoint": "https://hf-mirror.com",
          "onboarding_completed_at": \(onboardingStamp),
          "custom_models": [\(customModelJSON)],
          "tuned_control_records_by_model": {"/models/a": \(goodRecord(model: "/models/a", value: 3))}
        }
        """, to: store)

        let result = store.loadWithRecovery()
        let configuration = result.configuration

        XCTAssertNil(result.recovery, "a file with one bad field is not an unreadable file")
        XCTAssertEqual(configuration.port, MTPLXAppConfiguration().port, "the bad field falls back to its default")
        XCTAssertEqual(configuration.host, "0.0.0.0")
        XCTAssertEqual(configuration.apiKey, "sk-live-secret-123")
        XCTAssertEqual(configuration.huggingFaceHandle, "someone")
        XCTAssertEqual(configuration.hfEndpoint, "https://hf-mirror.com")
        XCTAssertEqual(configuration.onboardingCompletedAt, Date(timeIntervalSinceReferenceDate: onboardingStamp))
        XCTAssertEqual(configuration.customModels.map(\.id), ["custom-a"])
        XCTAssertEqual(configuration.tunedControlRecordsByModel["/models/a"]?.controlValue, 3)
        XCTAssertEqual(result.degradedFields.map(\.path), ["port"])
        XCTAssertTrue(result.degradedFields[0].reason.contains("Int"), result.degradedFields[0].reason)
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.settingsURL.path), "a readable file stays where it is")
    }

    func testUndecodableTunedRecordIsSkippedAndTheOthersLoad() throws {
        let store = makeStore()
        try write("""
        {
          "onboarding_completed_at": \(onboardingStamp),
          "tuned_control_record": {"schema_version": 1, "model_id": "legacy", "control_value": "three"},
          "tuned_control_records_by_model": {
            "/models/good": \(goodRecord(model: "/models/good", value: 2)),
            "/models/bad": {"schema_version": 1, "model_id": "/models/bad", "model_family": "qwen",
                            "backend_id": "qwen3_next", "control_field": "depth", "control_value": "three",
                            "candidates": [], "tuned_at": 700000000},
            "/models/also-good": \(goodRecord(model: "/models/also-good", value: 1))
          }
        }
        """, to: store)

        let result = store.loadWithRecovery()

        XCTAssertNil(result.recovery)
        XCTAssertNil(result.configuration.tunedControlRecord, "a legacy record with a bad field is dropped, not defaulted")
        XCTAssertEqual(
            result.configuration.tunedControlRecordsByModel.keys.sorted(),
            ["/models/also-good", "/models/good"]
        )
        XCTAssertEqual(result.configuration.tunedControlRecordsByModel["/models/good"]?.controlValue, 2)
        XCTAssertNotNil(result.configuration.onboardingCompletedAt)
        XCTAssertEqual(
            result.degradedFields.map(\.path).sorted(),
            ["tuned_control_record", "tuned_control_records_by_model./models/bad"]
        )
    }

    func testMalformedCustomModelIsSkippedAndTheOthersLoad() throws {
        let store = makeStore()
        try write("""
        {
          "custom_models": [
            \(customModelJSON),
            {"id": "broken", "displayName": "Broken"},
            42,
            {"id": "custom-b", "displayName": "Custom B", "shortName": "B", "detail": "",
             "hfModelID": "someone/custom-b", "localCandidates": ["/models/b"]}
          ]
        }
        """, to: store)

        let result = store.loadWithRecovery()

        XCTAssertNil(result.recovery)
        XCTAssertEqual(result.configuration.customModels.map(\.id), ["custom-a", "custom-b"])
        XCTAssertEqual(result.degradedFields.map(\.path), ["custom_models[1]", "custom_models[2]"])
    }

    func testWrongTypedCollectionFallsBackToDefaultOnly() throws {
        let store = makeStore()
        try write(#"{"custom_models": "not a list", "embedding_models": 7, "api_key": "kept"}"#, to: store)

        let result = store.loadWithRecovery()

        XCTAssertNil(result.recovery)
        XCTAssertEqual(result.configuration.customModels, [])
        XCTAssertEqual(result.configuration.embeddingModels, [])
        XCTAssertEqual(result.configuration.apiKey, "kept")
        XCTAssertEqual(result.degradedFields.map(\.path).sorted(), ["custom_models", "embedding_models"])
    }

    func testRemovedCustomModelStaysRemovedAfterSaveAndLoad() throws {
        let store = makeStore()
        var configuration = MTPLXAppConfiguration()
        configuration.rememberCustomModel(repoID: "Foo/Bar")
        configuration.rememberCustomModel(repoID: "Foo/Baz")
        let removedID = try XCTUnwrap(configuration.customModels.first?.id)

        XCTAssertTrue(configuration.removeCustomModel(id: removedID))
        try store.save(configuration)

        let loaded = try store.load()
        XCTAssertEqual(loaded.customModels.map(\.hfModelID), ["Foo/Baz"])
    }

    // MARK: - Unreadable file

    func testNonJSONFileIsKeptBesideItselfWithDatedSuffixAndNoticeSet() throws {
        let store = makeStore()
        let original = "this is not json {"
        try write(original, to: store)
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 3
        components.hour = 15; components.minute = 4; components.second = 5
        let now = try XCTUnwrap(Calendar.current.date(from: components))

        let result = store.loadWithRecovery(now: now)

        let expected = URL(fileURLWithPath: store.settingsURL.path + ".unreadable-20260903-150405")
        guard case .unreadableFileKept(let preservedAt, let reason)? = result.recovery else {
            return XCTFail("expected an unreadable-file notice, got \(String(describing: result.recovery))")
        }
        XCTAssertEqual(preservedAt, expected)
        XCTAssertFalse(reason.isEmpty)
        XCTAssertEqual(result.configuration, MTPLXAppConfiguration())
        XCTAssertTrue(result.degradedFields.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.settingsURL.path), "the unreadable file is moved, not left to be overwritten")
        XCTAssertEqual(try String(contentsOf: expected, encoding: .utf8), original, "the original bytes are preserved untouched")
        XCTAssertEqual(result.recovery?.fileURL, expected)
    }

    func testJSONThatIsNotAnObjectIsUnreadableAndASecondFileInTheSameSecondIsNotOverwritten() throws {
        let store = makeStore()
        let now = Date(timeIntervalSinceReferenceDate: 800_000_000)
        try write("[1, 2, 3]", to: store)
        let first = store.loadWithRecovery(now: now)
        guard case .unreadableFileKept(let firstPreservedAt, _)? = first.recovery else {
            return XCTFail("expected an unreadable-file notice, got \(String(describing: first.recovery))")
        }

        try write("\"just a string\"", to: store)
        let second = store.loadWithRecovery(now: now)
        guard case .unreadableFileKept(let secondPreservedAt, _)? = second.recovery else {
            return XCTFail("expected an unreadable-file notice, got \(String(describing: second.recovery))")
        }

        XCTAssertNotEqual(firstPreservedAt, secondPreservedAt)
        XCTAssertEqual(secondPreservedAt.path, firstPreservedAt.path + "-2")
        XCTAssertEqual(try String(contentsOf: firstPreservedAt, encoding: .utf8), "[1, 2, 3]")
        XCTAssertEqual(try String(contentsOf: secondPreservedAt, encoding: .utf8), "\"just a string\"")
    }

    func testAbsentFileIsDefaultsWithoutANotice() {
        let store = makeStore()

        let result = store.loadWithRecovery()

        XCTAssertNil(result.recovery)
        XCTAssertTrue(result.degradedFields.isEmpty)
        XCTAssertEqual(result.configuration, MTPLXAppConfiguration())
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.settingsURL.path))
    }

    // MARK: - Store

    @MainActor
    func testBackendStorePublishesTheNoticeAndTheNextSaveNeverTouchesThePreservedFile() throws {
        let settingsStore = makeStore()
        try write("{ broken", to: settingsStore)
        let backend = MTPLXBackendStore(settingsStore: settingsStore)

        backend.loadPersistedSettings()

        guard case .unreadableFileKept(let preservedAt, _)? = backend.settingsRecoveryNotice else {
            return XCTFail("expected an unreadable-file notice, got \(String(describing: backend.settingsRecoveryNotice))")
        }
        XCTAssertEqual(backend.configuration, MTPLXAppConfiguration())

        var next = backend.configuration
        next.port = 9001
        try backend.saveSettings(next)

        XCTAssertEqual(try String(contentsOf: preservedAt, encoding: .utf8), "{ broken")
        XCTAssertEqual(try settingsStore.load().port, 9001)

        backend.dismissSettingsRecoveryNotice()
        XCTAssertNil(backend.settingsRecoveryNotice)
    }

    @MainActor
    func testBackendStoreLoadsAFileWithOneBadFieldWithoutResettingOnboarding() async throws {
        let settingsStore = makeStore()
        try write("""
        {"stream_snapshot_interval_ms": "fast", "onboarding_completed_at": \(onboardingStamp), "api_key": "kept"}
        """, to: settingsStore)
        let backend = MTPLXBackendStore(settingsStore: settingsStore)

        backend.loadPersistedSettings()

        XCTAssertNil(backend.settingsRecoveryNotice)
        XCTAssertNotNil(backend.configuration.onboardingCompletedAt)
        XCTAssertEqual(backend.configuration.apiKey, "kept")
        XCTAssertEqual(backend.configuration.streamSnapshotIntervalMs, MTPLXAppConfiguration().streamSnapshotIntervalMs)
        // The degraded field is logged with its coding path (the Logs pane
        // reads the same store through refreshLogs).
        var logged: [String] = []
        for _ in 0..<50 where logged.isEmpty {
            await Task.yield()
            await backend.refreshLogs()
            logged = backend.logs.map(\.message)
        }
        XCTAssertTrue(
            logged.contains { $0.hasPrefix("settings: stream_snapshot_interval_ms could not be read") },
            logged.joined(separator: "\n")
        )
    }
}
