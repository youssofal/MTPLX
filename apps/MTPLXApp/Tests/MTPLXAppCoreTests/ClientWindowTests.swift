import XCTest

@testable import MTPLXAppCore

/// Pi and OpenCode are configured from the window the daemon executes, and
/// that whole window is Pi's answer ceiling. On 2026-09-29 the app wrote
/// 262,144 into Pi's contextWindow and maxTokens from Settings while the
/// engine could not serve a conversation that long. Builds from 59288061 then
/// wrote half the window as Pi's maxTokens, which Pi sends as every answer's
/// ceiling while the prompt is short (131,072 of 262,144 on the founder's
/// Mac); the server already caps each answer to the memory actually free.
final class ClientWindowTests: XCTestCase {
    private let flashNext = "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    private let qwen36 = "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"

    // MARK: Health

    func testHealthDecodesTheExecutionWindowAndToleratesItsAbsence() throws {
        // A daemon built from 59288061 also sends answer_tokens, half the
        // window; clients are configured from `tokens` alone.
        let with = try JSONDecoder().decode(HealthPayload.self, from: Data(Self.health(window: """
        ,"execution_window": {"tokens": 98304, "answer_tokens": 49152, "basis": "machine_fit"}
        """).utf8))
        XCTAssertEqual(with.executionWindow, ServedExecutionWindow(tokens: 98_304, basis: "machine_fit"))
        let without = try JSONDecoder().decode(HealthPayload.self, from: Data(Self.health(window: "").utf8))
        XCTAssertNil(without.executionWindow)
    }

    // MARK: Pi

    func testPiWithoutAServedWindowGetsTheWholeSetting() throws {
        // Before the daemon answers, or when it publishes no window.
        for served in [nil, ServedExecutionWindow(tokens: 0)] {
            let url = temporaryDirectory().appendingPathComponent("models.json")
            _ = try PiIntegration(configURL: url).sync(
                configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 65_536),
                servedWindow: served
            )
            let model = try piModel(at: url)
            XCTAssertEqual(model["contextWindow"]?.intValue, 65_536)
            XCTAssertEqual(model["maxTokens"]?.intValue, 65_536)
        }
    }

    func testPiIsConfiguredFromTheServedWindowAtBoundaryValues() throws {
        for tokens in [4_096, 32_768, 98_304, 262_144] {
            let url = temporaryDirectory().appendingPathComponent("models.json")
            _ = try PiIntegration(configURL: url).sync(
                configuration: MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144),
                servedWindow: ServedExecutionWindow(tokens: tokens)
            )
            let model = try piModel(at: url)
            XCTAssertEqual(model["contextWindow"]?.intValue, tokens)
            XCTAssertEqual(model["maxTokens"]?.intValue, tokens)
        }
    }

    func testPiPairMTPLXWroteFollowsTheServedWindow() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)
        let configuration = MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 131_072)

        // The write before the daemon starts: the setting, whole.
        _ = try integration.sync(configuration: configuration)
        var model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 131_072)
        XCTAssertEqual(model["maxTokens"]?.intValue, 131_072)

        // Once the daemon publishes its window, MTPLX's pair follows it whole.
        let served = ServedExecutionWindow(tokens: 98_304)
        XCTAssertTrue(try integration.sync(configuration: configuration, servedWindow: served).didChange)
        model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 98_304)
        XCTAssertEqual(model["maxTokens"]?.intValue, 98_304)
        XCTAssertFalse(try integration.sync(configuration: configuration, servedWindow: served).didChange)
        // The next launch's first write keeps it instead of flipping back.
        XCTAssertFalse(try integration.sync(configuration: configuration).didChange)
        // A later daemon with a larger window moves MTPLX's own pair again.
        _ = try integration.sync(
            configuration: configuration,
            servedWindow: ServedExecutionWindow(tokens: 262_144)
        )
        model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 262_144)
        XCTAssertEqual(model["maxTokens"]?.intValue, 262_144)
    }

    func testPiKeepsAWindowPairTheUserChose() throws {
        // #282: only maxTokens equal to the window is MTPLX's pair. Any other
        // pair is the user's before and after the daemon answers, half the
        // window included (only internal builds from 59288061 wrote that, so
        // anywhere else it is a user edit), and so is a maxTokens set on an
        // entry without its own window (the sync fills in the 65,536 setting
        // beside it).
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)
        let configuration = MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 65_536)
        let pairs: [(window: Int?, maxTokens: Int)] = [
            (65_536, 20_000), (262_144, 65_536), (32_768, 16_385),
            (262_144, 131_072), (98_305, 49_152), (nil, 20_000), (nil, 32_768),
        ]
        for pair in pairs {
            try writePiModel(at: url, model: qwen36, contextWindow: pair.window, maxTokens: pair.maxTokens)
            for served in [nil, ServedExecutionWindow(tokens: 98_304), nil] {
                _ = try integration.sync(configuration: configuration, servedWindow: served)
                let model = try piModel(at: url)
                XCTAssertEqual(model["contextWindow"]?.intValue, pair.window ?? 65_536)
                XCTAssertEqual(model["maxTokens"]?.intValue, pair.maxTokens)
            }
        }
    }

    func testPiKeepsAMaxTokensTheUserSetWithoutAWindow() throws {
        // The review's case: the user set only maxTokens 65,536; the first
        // sync fills in the 131,072 setting beside it, and the pair it makes
        // (half the window) stays theirs through repeated syncs, before and
        // after the daemon publishes its window.
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)
        let configuration = MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: 131_072)
        try writePiModel(at: url, model: qwen36, contextWindow: nil, maxTokens: 65_536)
        for served in [nil, nil, ServedExecutionWindow(tokens: 98_304), nil] {
            _ = try integration.sync(configuration: configuration, servedWindow: served)
            let model = try piModel(at: url)
            XCTAssertEqual(model["contextWindow"]?.intValue, 131_072)
            XCTAssertEqual(model["maxTokens"]?.intValue, 65_536)
        }
    }

    func testOwnershipSignatureRecognisesOnlyMTPLXPairs() {
        func entry(_ window: Int, _ maxTokens: Int) -> [String: JSONValue] {
            ["contextWindow": .number(Double(window)), "maxTokens": .number(Double(maxTokens))]
        }
        // Every MTPLX writer advertises the whole window.
        XCTAssertTrue(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(262_144, 262_144)))
        XCTAssertTrue(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(98_304, 98_304)))
        // Half the window is a user pair: 2.12.0 only ever wrote the whole one.
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(262_144, 131_072)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(98_305, 49_152)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(131_072, 65_536)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(131_072, 20_000)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(262_144, 65_536)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(["contextWindow": .number(8_192)]))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(["maxTokens": .number(8_192)]))
    }

    // MARK: OpenCode

    func testOpenCodeLimitsFollowTheServedWindowAtBoundaryValues() throws {
        // limit.output keeps the rule 2.12.0 shipped (#480): OpenCode holds
        // min(limit.output, 32,000) of the window back for the reply, so the
        // limit is half the window, capped at the 32,000 OpenCode injects.
        for (tokens, output) in [(8_192, 4_096), (32_768, 16_384), (40_000, 20_000), (262_144, 32_000)] {
            let url = temporaryDirectory().appendingPathComponent("opencode.json")
            _ = try openCode(url).sync(
                configuration: MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144),
                servedWindow: ServedExecutionWindow(tokens: tokens)
            )
            let limit = try openCodeLimit(at: url)
            XCTAssertEqual(limit["context"]?.intValue, tokens)
            XCTAssertEqual(limit["output"]?.intValue, output)
        }
    }

    func testOpenCodeKeepsItsLimitsUntilTheDaemonPublishesAWindow() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let integration = openCode(url)
        let configuration = MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144)
        // First launch, nothing written yet: the setting.
        _ = try integration.sync(configuration: configuration)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 262_144)
        XCTAssertEqual(try openCodeLimit(at: url)["output"]?.intValue, 32_000)
        // The daemon executes less: the limits follow it.
        let served = ServedExecutionWindow(tokens: 40_000)
        XCTAssertTrue(try integration.sync(configuration: configuration, servedWindow: served).didChange)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 40_000)
        XCTAssertEqual(try openCodeLimit(at: url)["output"]?.intValue, 20_000)
        // The next launch's first write keeps them instead of flipping back.
        XCTAssertFalse(try integration.sync(configuration: configuration).didChange)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 40_000)
    }

    func testOpenCodePluginStripsTheReplyReserveMTPLXConfigured() throws {
        // OpenCode sends maxOutputTokens = min(limit.output, 32,000) with every
        // request and hands the plugin the model with its limit (CLI 1.18.29,
        // Desktop 1.18.31). Below a 64K window limit.output is half the window
        // (#480), and a plugin that stripped only 32,000 left every answer
        // capped there: 16,384 on 32,768. The generated plugin runs on the
        // request OpenCode builds from the config the app wrote: the injected
        // value goes at every window; a user cap and a limit.output the user
        // chose reach the server.
        for tokens in [20_480, 32_768, 65_536, 262_144] {
            let url = temporaryDirectory().appendingPathComponent("opencode.json")
            let result = try openCode(url).sync(
                configuration: MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144),
                servedWindow: ServedExecutionWindow(tokens: tokens)
            )
            let limit = try openCodeLimit(at: url)
            let context = try XCTUnwrap(limit["context"]?.intValue)
            let output = try XCTUnwrap(limit["output"]?.intValue)
            XCTAssertEqual(context, tokens)
            XCTAssertEqual(output, min(32_000, tokens / 2))
            let model: [String: Any] = [
                "providerID": "mtplx",
                "id": OpenCodeIntegration.modelID(for: flashNext),
                "limit": ["context": context, "output": output],
            ]
            var chosen = model
            chosen["limit"] = ["context": context, "output": 9_000]
            // What the CLI writes for --max-response-tokens: the cap as
            // limit.output and as the model's header, here equal to the
            // reserve and above OpenCode's 32,000 ceiling.
            var atReserve = model
            atReserve["headers"] = ["x-mtplx-max-response-tokens": String(output)]
            var aboveCeiling = model
            aboveCeiling["limit"] = ["context": context, "output": 50_000]
            aboveCeiling["headers"] = ["x-mtplx-max-response-tokens": "50000"]
            let results = try chatParamsAfterHook(
                plugin: URL(fileURLWithPath: result.sessionHeadersPluginPath)
                    .appendingPathComponent("index.js"),
                cases: [
                    "injected": (["model": model], ["maxOutputTokens": min(output, 32_000)]),
                    "userCap": (["model": model], ["maxOutputTokens": 9_000]),
                    "absent": (["model": model], [:]),
                    "chosenLimit": (["model": chosen], ["maxOutputTokens": 9_000]),
                    "atReserve": (["model": atReserve], ["maxOutputTokens": min(output, 32_000)]),
                    "aboveCeiling": (["model": aboveCeiling], ["maxOutputTokens": 32_000]),
                ]
            )
            // JSON.stringify drops undefined-valued keys.
            XCTAssertNil(results["injected"]?["maxOutputTokens"], "window \(tokens)")
            XCTAssertEqual(results["userCap"]?["maxOutputTokens"] as? Int, 9_000, "window \(tokens)")
            XCTAssertNil(results["absent"]?["maxOutputTokens"], "window \(tokens)")
            XCTAssertEqual(results["chosenLimit"]?["maxOutputTokens"] as? Int, 9_000, "window \(tokens)")
            XCTAssertEqual(results["atReserve"]?["maxOutputTokens"] as? Int, output, "window \(tokens)")
            XCTAssertEqual(results["aboveCeiling"]?["maxOutputTokens"] as? Int, 50_000, "window \(tokens)")
        }
    }

    // MARK: Helpers

    /// Runs the generated plugin's chat.params hook under node on each
    /// (input, output) pair and returns the outputs; skips without node.
    private func chatParamsAfterHook(
        plugin: URL,
        cases: [String: (input: [String: Any], output: [String: Any])]
    ) throws -> [String: [String: Any]] {
        let searchPath = ProcessInfo.processInfo.environment["PATH"] ?? ""
        guard let node = searchPath.split(separator: ":")
            .map({ URL(fileURLWithPath: String($0)).appendingPathComponent("node") })
            .first(where: { FileManager.default.isExecutableFile(atPath: $0.path) })
        else { throw XCTSkip("node is not installed") }
        let directory = temporaryDirectory()
        // A .mjs copy loads as a module on every node version.
        try FileManager.default.copyItem(at: plugin, to: directory.appendingPathComponent("plugin.mjs"))
        let payload = try JSONSerialization.data(
            withJSONObject: cases.mapValues { [$0.input, $0.output] }
        )
        let harness = directory.appendingPathComponent("harness.mjs")
        try """
        import plugin from "./plugin.mjs";
        const hooks = await plugin();
        const cases = \(String(decoding: payload, as: UTF8.self));
        const results = {};
        for (const [name, [input, output]] of Object.entries(cases)) {
          await hooks["chat.params"](input, output);
          results[name] = output;
        }
        console.log(JSON.stringify(results));
        """.write(to: harness, atomically: true, encoding: .utf8)
        let process = Process()
        process.executableURL = node
        process.arguments = [harness.path]
        let stdout = Pipe()
        process.standardOutput = stdout
        try process.run()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: [String: Any]])
    }


    private func openCode(_ url: URL) -> OpenCodeIntegration {
        OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: url.deletingLastPathComponent().appendingPathComponent("default.dat")
        )
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-client-window-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func piModel(at url: URL) throws -> [String: JSONValue] {
        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let models = try XCTUnwrap(
            root["providers"]?.objectValue?["mtplx"]?.objectValue?["models"]?.arrayValue
        )
        return try XCTUnwrap(models.first?.objectValue)
    }

    private func writePiModel(
        at url: URL,
        model: String? = nil,
        contextWindow: Int?,
        maxTokens: Int
    ) throws {
        var entry: [String: JSONValue] = [
            "id": .string(PiIntegration.modelID(for: model ?? flashNext)),
            "maxTokens": .number(Double(maxTokens)),
        ]
        if let contextWindow {
            entry["contextWindow"] = .number(Double(contextWindow))
        }
        let root: [String: JSONValue] = [
            "providers": .object([
                "mtplx": .object([
                    "baseUrl": .string("http://127.0.0.1:8000/v1"),
                    "models": .array([.object(entry)]),
                ]),
            ]),
        ]
        try JSONEncoder().encode(root).write(to: url)
    }

    private func openCodeLimit(at url: URL) throws -> [String: JSONValue] {
        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let models = try XCTUnwrap(root["provider"]?.objectValue?["mtplx"]?.objectValue?["models"]?.objectValue)
        let model = try XCTUnwrap(models.values.first?.objectValue)
        return try XCTUnwrap(model["limit"]?.objectValue)
    }

    private static func health(window: String) -> String {
        """
        {"ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
         "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
         "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3"\(window)}
        """
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
}
