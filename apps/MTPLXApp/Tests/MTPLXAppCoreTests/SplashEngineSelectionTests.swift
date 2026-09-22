import XCTest
@testable import MTPLXAppCore

/// Selecting a non-MLX engine must change argv, not just a stored string.
///
/// The MLX runtime flags the builder emits (scheduler mode, batching preset,
/// MTP depth, adaptive policy) describe machinery the Splash bridge does not
/// have. Passing them anyway would make the launch record claim settings that
/// never took effect, which is the exact dishonesty `flagValue` exists to
/// prevent.
final class SplashEngineSelectionTests: XCTestCase {

    func testDefaultEngineIsMLX() {
        XCTAssertEqual(MTPLXAppConfiguration.defaultEngine, "mlx")
        XCTAssertEqual(MTPLXAppConfiguration().engine, "mlx")
    }

    func testEngineVocabularyRejectsAnythingArgparseWouldRefuse() {
        XCTAssertEqual(MTPLXAppConfiguration.normalizedEngine("splash"), "splash")
        XCTAssertEqual(MTPLXAppConfiguration.normalizedEngine("  SPLASH "), "splash")
        XCTAssertEqual(MTPLXAppConfiguration.normalizedEngine("mlx"), "mlx")
        // A persisted value from a newer build, or a hand-edited settings
        // file, must fall back rather than fail the daemon launch.
        XCTAssertEqual(MTPLXAppConfiguration.normalizedEngine("vllm"), "mlx")
        XCTAssertEqual(MTPLXAppConfiguration.normalizedEngine(""), "mlx")
    }

    func testSplashArgvKeepsTransportFlagsAndDropsMLXRuntimeFlags() {
        let built = [
            "serve",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--model", "/models/qwen3-next-mlx",
            "--profile", "turbo",
            "--generation-mode", "mtp",
            "--scheduler-mode", "serial",
            "--batching-preset", "balanced",
            "--depth", "3",
            "--context-window", "65536",
            "--no-stats-footer",
        ]
        let argv = MTPLXCommandBuilder.engineServeArguments(
            built,
            engine: "splash",
            model: "incoai/Qwen3.8-27B-Splash"
        )

        XCTAssertEqual(argv.first, "serve")
        XCTAssertEqual(MTPLXCommandBuilder.flagValue("--engine", in: argv), "splash")
        // The engine's own package, not the MLX checkpoint.
        XCTAssertEqual(
            MTPLXCommandBuilder.flagValue("--model", in: argv),
            "incoai/Qwen3.8-27B-Splash"
        )
        XCTAssertEqual(MTPLXCommandBuilder.flagValue("--host", in: argv), "127.0.0.1")
        XCTAssertEqual(MTPLXCommandBuilder.flagValue("--port", in: argv), "8000")
        XCTAssertEqual(MTPLXCommandBuilder.flagValue("--context-window", in: argv), "65536")

        for dropped in [
            "--profile", "--generation-mode", "--scheduler-mode",
            "--batching-preset", "--depth", "--no-stats-footer",
        ] {
            XCTAssertFalse(
                argv.contains(dropped),
                "\(dropped) describes the MLX runtime and must not reach the bridge"
            )
        }
    }

    func testSplashModelIsSeparateFromTheMLXModel() {
        // Switching engines must not overwrite either side's model choice.
        var configuration = MTPLXAppConfiguration()
        configuration.model = "/models/qwen3-next-mlx"
        configuration.engine = "splash"
        XCTAssertEqual(configuration.splashModel, "incoai/Qwen3.8-27B-Splash")
        XCTAssertEqual(configuration.model, "/models/qwen3-next-mlx")

        let argv = MTPLXCommandBuilder.engineServeArguments(
            ["serve", "--model", configuration.model],
            engine: configuration.engine,
            model: configuration.splashModel
        )
        XCTAssertEqual(
            MTPLXCommandBuilder.flagValue("--model", in: argv),
            "incoai/Qwen3.8-27B-Splash"
        )
    }

    func testEngineSurvivesASettingsRoundTrip() throws {
        var configuration = MTPLXAppConfiguration()
        configuration.engine = "splash"
        configuration.splashModel = "incoai/Qwen3.6-35B-A3B-Splash"

        let data = try JSONEncoder().encode(configuration)
        let restored = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)

        XCTAssertEqual(restored.engine, "splash")
        XCTAssertEqual(restored.splashModel, "incoai/Qwen3.6-35B-A3B-Splash")
    }
}

/// The Settings package list is driven by `mtplx splash list --json`, whose
/// stdout also carries Splash's human-readable installer output.
final class SplashPackageStoreTests: XCTestCase {

    func testPayloadIsReadPastTheInstallersOwnOutput() throws {
        // Splash's installer prints progress on the same stream, so the JSON
        // is the last line, not the whole buffer.
        let mixed = """
        Installing incoai/Qwen3.8-27B-Splash; missing artifacts will be downloaded.
        Fetching 79 files:  42%|####      | 33/79
        {"ok": true, "engine": "splash", "packages": []}
        """
        let data = try XCTUnwrap(SplashPackageStore.lastJSONObject(in: mixed))
        let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(decoded?["engine"] as? String, "splash")
    }

    func testNoPayloadIsReportedRatherThanGuessed() {
        XCTAssertNil(SplashPackageStore.lastJSONObject(in: "error: Splash is not installed"))
        XCTAssertNil(SplashPackageStore.lastJSONObject(in: ""))
    }

    func testPackageDecodesTheSnakeCaseSizeAndFormatsIt() throws {
        let json = """
        {"id":"incoai/Qwen3.8-27B-Splash","installed":false,
         "path":"/tmp/x","download_bytes":17400000000}
        """
        let package = try JSONDecoder().decode(
            SplashPackageStore.Package.self, from: Data(json.utf8)
        )
        XCTAssertFalse(package.installed)
        XCTAssertEqual(package.downloadSizeLabel, "17.4 GB")
    }

    func testMissingSizeDoesNotFabricateOne() throws {
        let json = #"{"id":"a/b","installed":true,"path":"/tmp/x"}"#
        let package = try JSONDecoder().decode(
            SplashPackageStore.Package.self, from: Data(json.utf8)
        )
        XCTAssertNil(package.downloadSizeLabel)
    }
}

/// One catalog feeds the top-bar picker, the Settings rows, and the argv
/// vocabulary; these keep the three from drifting apart.
final class SplashPackageCatalogTests: XCTestCase {

    func testEveryChoiceTheBuilderAcceptsHasAPickerEntry() {
        XCTAssertEqual(
            MTPLXAppConfiguration.splashPackageChoices,
            SplashPackageOption.catalog.map(\.id)
        )
        XCTAssertTrue(
            MTPLXAppConfiguration.splashPackageChoices
                .contains(MTPLXAppConfiguration.defaultSplashModel),
            "the default package must be one the picker can show as selected"
        )
    }

    func testEveryEntryHasANameADetailLineAndASize() {
        for option in SplashPackageOption.catalog {
            XCTAssertFalse(option.displayName.isEmpty)
            XCTAssertFalse(option.detail.isEmpty, "\(option.id) needs its one-line detail")
            XCTAssertGreaterThan(option.downloadBytes, 1_000_000_000)
            XCTAssertTrue(option.id.hasPrefix("incoai/"), "--model takes the repo id")
        }
    }

    func testInstallStateIsReadFromSplashsOwnModelDirectory() {
        let option = SplashPackageOption.catalog[0]
        XCTAssertTrue(
            option.installURL.path.hasSuffix(
                "Library/Application Support/Splash/models/incoai/Qwen3.8-27B-Splash"
            )
        )
    }

    func testActiveModelReferenceFollowsTheEngine() {
        var configuration = MTPLXAppConfiguration()
        configuration.model = "/models/qwen-mlx"
        configuration.splashModel = "incoai/Qwen3.6-35B-A3B-Splash"
        configuration.engine = "mlx"
        XCTAssertEqual(configuration.activeModelReference, "/models/qwen-mlx")
        configuration.engine = "splash"
        XCTAssertEqual(configuration.activeModelReference, "incoai/Qwen3.6-35B-A3B-Splash")
    }
}

/// Splash's KV width is fixed in its kernels, and the app has more than one
/// KV control: the Settings card and the top-bar inference-params panel. Both
/// resolve a `KVQuantPolicy`, so the lock lives in one shared value.
final class SplashKVQuantLockTests: XCTestCase {

    func testTheSharedPolicyLocksToQ8WithAReason() {
        let policy = MTPLXAppConfiguration.splashKVQuantPolicy
        XCTAssertFalse(policy.supported, "the control must not be offered")
        XCTAssertEqual(policy.modes, ["q8"])
        let reason = policy.disabledReason ?? ""
        XCTAssertTrue(reason.contains("only supports q8"),
                      "a locked control has to say what it is locked to")
        XCTAssertTrue(reason.contains("MLX"),
                      "and where the other widths are available")
    }

    func testSelectingSplashPinsTheStoredWidthToQ8() {
        // Whatever the MLX side left behind, the saved config must describe
        // what Splash will actually run.
        var configuration = MTPLXAppConfiguration()
        configuration.pagedKVQuantization = "q4"
        configuration.engine = "splash"
        let argv = MTPLXCommandBuilder.engineServeArguments(
            ["serve", "--kv-quant", configuration.pagedKVQuantization],
            engine: "splash",
            model: MTPLXAppConfiguration.defaultSplashModel
        )
        XCTAssertFalse(argv.contains("--kv-quant"),
                       "Splash has no KV flag; it must never reach the bridge")
    }

    /// A new KV control added later must resolve the same policy, not invent
    /// its own fallback the way both existing panels originally did.
    func testEveryKVControlResolvesPolicyRatherThanHardcodingModes() throws {
        let views = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // MTPLXAppCoreTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // MTPLXApp
            .appendingPathComponent("Sources/MTPLXAppHost/Views")
        let files = FileManager.default.enumerator(at: views, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" } ?? []
        var offenders: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            // Only surfaces that *offer* a narrower width need gating. A file
            // that merely pins the value to q8 while switching engines (the
            // top bar, the model picker) is doing the right thing already.
            guard text.contains("\"q4\"") else { continue }
            let consultsPolicy = text.contains("kvQuantPolicy")
                || text.contains("splashKVQuantPolicy")
                || text.contains("splashEngineSelected")
            if !consultsPolicy { offenders.append(file.lastPathComponent) }
        }
        XCTAssertTrue(
            offenders.isEmpty,
            "these offer a KV width without asking whether the engine allows it: \(offenders)"
        )
    }
}
