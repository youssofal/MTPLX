import Darwin
import XCTest

@testable import MTPLXAppCore

/// Claude Code and Codex in the Start menu (night 4, D4).
///
/// The app opens both clients from the same plan as `mtplx start
/// claude-code|codex` and `mtplx connect`: `mtplx/agent_clients.py` writes
/// `Tests/Fixtures/agent_client_plans.json` (scripts/write_agent_client_plans.py)
/// and the Swift twin must build the identical argv, environment, removed
/// variables and file bytes for every row. Nothing here starts Terminal or a
/// real client.
final class AgentCLIIntegrationTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Inputs: Decodable {
            let client: String
            let baseURL: String
            let modelID: String
            let contextWindow: Int?
            let apiKey: String?

            enum CodingKeys: String, CodingKey {
                case client
                case baseURL = "base_url"
                case modelID = "model_id"
                case contextWindow = "context_window"
                case apiKey = "api_key"
            }
        }

        struct Plan: Decodable {
            let client: String
            let argv: [String]
            let env: [String: String]
            let unsetEnv: [String]
            let files: [String: String]

            enum CodingKeys: String, CodingKey {
                case client, argv, env, files
                case unsetEnv = "unset_env"
            }
        }

        struct Row: Decodable {
            let inputs: Inputs
            let plan: Plan
        }

        struct DesktopInputs: Decodable {
            let baseURL: String
            let modelID: String
            let contextWindow: Int?
            let apiKey: String?
            let existingConfig: String?

            enum CodingKeys: String, CodingKey {
                case baseURL = "base_url"
                case modelID = "model_id"
                case contextWindow = "context_window"
                case apiKey = "api_key"
                case existingConfig = "existing_config"
            }
        }

        struct DesktopPlan: Decodable {
            let argv: [String]
            let files: [String: String]
            let folders: [String]
        }

        struct DesktopRow: Decodable {
            let inputs: DesktopInputs
            let plan: DesktopPlan
        }

        let version: Int
        let localAPIKey: String
        let codexDesktopBundleID: String
        let codexDesktopMinVersion: [Int]
        let rows: [Row]
        let codexDesktopRows: [DesktopRow]

        enum CodingKeys: String, CodingKey {
            case version, rows
            case localAPIKey = "local_api_key"
            case codexDesktopBundleID = "codex_desktop_bundle_id"
            case codexDesktopMinVersion = "codex_desktop_min_version"
            case codexDesktopRows = "codex_desktop_rows"
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/agent_client_plans.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    /// The Codex desktop instance: the same `open -n` line and the same
    /// merged config.toml bytes as `mtplx connect codex --desktop`, for the
    /// file the app wrote in the 12 Nov test, a user's own settings, tricky
    /// multi-line values and an old provider written as subtables.
    func testCodexDesktopPlanMatchesTheCLIByteForByte() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.codexDesktopBundleID, CodexDesktopIntegration.bundleIdentifier)
        XCTAssertEqual(fixture.codexDesktopMinVersion, CodexDesktopIntegration.minimumVersion)
        XCTAssertGreaterThanOrEqual(fixture.codexDesktopRows.count, 15)
        for (index, row) in fixture.codexDesktopRows.enumerated() {
            let plan = CodexDesktopIntegration.plan(
                appPath: "/Applications/ChatGPT.app",
                baseURL: row.inputs.baseURL,
                modelID: row.inputs.modelID,
                contextWindow: row.inputs.contextWindow,
                apiKey: row.inputs.apiKey,
                existingConfig: row.inputs.existingConfig,
                home: "<home>",
                userData: "<user-data>"
            )
            XCTAssertEqual(plan.argv, row.plan.argv, "row \(index)")
            XCTAssertEqual(plan.folders, row.plan.folders, "row \(index)")
            XCTAssertEqual(plan.files, row.plan.files, "config bytes for row \(index)")
            // A second launch on the file the first one wrote changes nothing.
            let written = try XCTUnwrap(plan.files["<home>/config.toml"])
            XCTAssertEqual(
                CodexDesktopIntegration.configText(
                    existing: written,
                    baseURL: row.inputs.baseURL,
                    modelID: row.inputs.modelID,
                    contextWindow: row.inputs.contextWindow,
                    apiKey: row.inputs.apiKey
                ),
                written,
                "row \(index) is not stable across launches"
            )
            if let key = row.inputs.apiKey {
                XCTAssertFalse(plan.argv.contains { $0.contains(key) }, "row \(index): key on argv")
            }
        }
    }

    func testCodexDesktopIsFoundByBundleIDAndFencedToTheProvenVersion() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-codex-app-\(UUID().uuidString.lowercased())")
        defer { try? FileManager.default.removeItem(at: root) }
        func fakeApp(_ folder: String, _ name: String, _ bundleID: String, _ version: String) throws -> URL {
            let directory = root.appendingPathComponent(folder, isDirectory: true)
            let contents = directory.appendingPathComponent("\(name)/Contents", isDirectory: true)
            try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
            let data = try PropertyListSerialization.data(
                fromPropertyList: [
                    "CFBundleIdentifier": bundleID,
                    "CFBundleShortVersionString": version,
                    "CFBundleExecutable": (name as NSString).deletingPathExtension,
                ],
                format: .xml,
                options: 0
            )
            try data.write(to: contents.appendingPathComponent("Info.plist"))
            return directory
        }
        let chat = try fakeApp("a", "ChatGPT.app", "com.openai.chat", "1.0")
        XCTAssertNil(CodexDesktopIntegration.findApp(searchDirectories: [chat]))
        let proven = try fakeApp("b", "Codex.app", CodexDesktopIntegration.bundleIdentifier, "26.1106.51428")
        XCTAssertEqual(
            CodexDesktopIntegration.findApp(searchDirectories: [chat, proven]),
            CodexDesktopApp(
                path: proven.appendingPathComponent("Codex.app").path,
                version: "26.1106.51428",
                supported: true,
                executablePath: proven.appendingPathComponent("Codex.app/Contents/MacOS/Codex").path
            )
        )
        let old = try fakeApp("c", "ChatGPT.app", CodexDesktopIntegration.bundleIdentifier, "26.1005.1")
        XCTAssertEqual(CodexDesktopIntegration.findApp(searchDirectories: [old])?.supported, false)
        XCTAssertTrue(CodexDesktopIntegration.isAtLeast([26, 1106, 51428], [26, 1106]))
        XCTAssertTrue(CodexDesktopIntegration.isAtLeast([27], [26, 1106]))
        XCTAssertFalse(CodexDesktopIntegration.isAtLeast([26], [26, 1106]))
        XCTAssertFalse(CodexDesktopIntegration.isAtLeast([], [26, 1106]))
    }

    /// The copy is the app's main process with MTPLX's own --user-data-dir
    /// in its argv: never the user's own Codex app, a helper process, a
    /// look-alike folder or the marker hidden in the environment.
    func testTheCodexCopyIsTheMainProcessWithMTPLXsFolderOnly() {
        let main = "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"
        let folder = "/Users/someone/.mtplx/clients/codex-desktop/user-data"
        let instance = CodexDesktopInstance(processID: 4242, executablePath: main, userDataPath: folder)
        func check(_ executable: String, _ argv: [String]) -> Bool {
            CodexDesktopIntegration.isRunning(instance) { pid in
                pid == 4242 ? (executable, argv) : nil
            }
        }
        XCTAssertTrue(check(main, [main, "--user-data-dir=\(folder)"]))
        XCTAssertFalse(check(main, [main]), "the user's own Codex app has no MTPLX folder")
        XCTAssertFalse(check(main, [main, "--user-data-dir=\(folder)-other"]))
        XCTAssertFalse(check(main, ["--user-data-dir=\(folder)"]), "argv[0] is not an argument")
        XCTAssertFalse(check(
            "/Applications/ChatGPT.app/Contents/Frameworks/ChatGPT Helper (Renderer).app/Contents/MacOS/ChatGPT Helper (Renderer)",
            ["helper", "--type=renderer", "--user-data-dir=\(folder)"]
        ), "a helper process is not the copy")
        XCTAssertFalse(CodexDesktopIntegration.isRunning(instance) { _ in nil }, "a gone process is not running")

        // A KERN_PROCARGS2 image: argc, exec path, padding, argv, environment.
        var image = withUnsafeBytes(of: Int32(2).littleEndian) { Array($0) }
        for text in [main, "", "", main, "--user-data-dir=\(folder)", "MARK=--user-data-dir=/x", ""] {
            image += Array(text.utf8) + [0]
        }
        let parsed = CodexDesktopIntegration.parseProcessArguments(image)
        XCTAssertEqual(parsed?.executable, main)
        XCTAssertEqual(parsed?.argv, [main, "--user-data-dir=\(folder)"])
        XCTAssertNil(CodexDesktopIntegration.parseProcessArguments([1, 0]))
        // The real kernel image of this test process.
        let own = CodexDesktopIntegration.processArguments(getpid())
        XCTAssertEqual(own?.argv.first.map { URL(fileURLWithPath: $0).lastPathComponent }?.isEmpty, false)
    }

    /// Stop ends exactly the Codex copy MTPLX opened (by its process), and
    /// never a process that only looks like it.
    @MainActor
    func testStopClosesTheCodexCopyItOpenedAndNothingElse() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-codex-copy-\(UUID().uuidString.lowercased())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let folder = directory.appendingPathComponent("codex-desktop/user-data").path
        func launchCopy(_ marker: String) throws -> Process {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", "sleep 30; exit 0", marker]
            try process.run()
            return process
        }
        let copy = try launchCopy("--user-data-dir=\(folder)")
        let lookalike = try launchCopy("--user-data-dir=\(folder)-not-mtplx")
        defer {
            for process in [copy, lookalike] where process.isRunning {
                process.terminate()
                process.waitUntilExit()
            }
        }
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 9),
            agentCLIIntegration: AgentCLIIntegration(
                clientsDirectory: directory.appendingPathComponent("clients"),
                handoffDirectory: directory.appendingPathComponent("handoffs")
            ),
            localFanRestorer: { true }
        )
        let instance = CodexDesktopInstance(
            processID: copy.processIdentifier,
            executablePath: "/bin/bash",
            userDataPath: folder
        )
        XCTAssertTrue(CodexDesktopIntegration.isRunning(instance))
        XCTAssertFalse(CodexDesktopIntegration.isRunning(CodexDesktopInstance(
            processID: lookalike.processIdentifier, executablePath: "/bin/bash", userDataPath: folder
        )))
        backend.recordLaunchedCodexDesktopInstance(instance)
        XCTAssertEqual(backend.codexDesktopInstanceProcessIDsForTesting, [copy.processIdentifier])

        await backend.stopDaemon()

        var exited = !copy.isRunning
        for _ in 0..<200 where !exited {
            try? await Task.sleep(for: .milliseconds(10))
            exited = !copy.isRunning
        }
        XCTAssertTrue(exited, "Stop must close the Codex copy MTPLX opened")
        XCTAssertTrue(lookalike.isRunning, "Stop must never signal a process that is not the copy")
        XCTAssertTrue(backend.codexDesktopInstanceProcessIDsForTesting.isEmpty)
        // Its pid is gone: ending it again signals nothing.
        XCTAssertFalse(CodexDesktopIntegration.end(instance, graceSeconds: 0.1))
    }

    func testCodexDesktopFoldersAreMTPLXsOwnNeverTheUsersCodexHome() {
        let integration = AgentCLIIntegration(
            clientsDirectory: URL(fileURLWithPath: "/Users/someone/.mtplx/clients"),
            handoffDirectory: URL(fileURLWithPath: "/Users/someone/.mtplx/handoffs")
        )
        XCTAssertEqual(integration.codexDesktopHome.path, "/Users/someone/.mtplx/clients/codex-desktop/home")
        XCTAssertEqual(integration.codexDesktopUserData.path, "/Users/someone/.mtplx/clients/codex-desktop/user-data")
        XCTAssertFalse(integration.codexDesktopHome.pathComponents.contains(".codex"))
    }

    func testPlansMatchTheCLIPlanByteForByte() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.version, 2)
        XCTAssertEqual(fixture.localAPIKey, AgentCLIIntegration.localAPIKey)
        XCTAssertEqual(
            Set(fixture.rows.map(\.inputs.client)),
            Set(AgentCLIClient.allCases.map(\.rawValue))
        )
        XCTAssertGreaterThanOrEqual(fixture.rows.count, 6)
        for row in fixture.rows {
            let client = try XCTUnwrap(AgentCLIClient(rawValue: row.inputs.client))
            let plan = AgentCLIIntegration.plan(
                client: client,
                baseURL: row.inputs.baseURL,
                modelID: row.inputs.modelID,
                contextWindow: row.inputs.contextWindow,
                apiKey: row.inputs.apiKey,
                settingsPath: "<settings>"
            )
            let cell = "\(row.inputs.client) \(row.inputs.modelID) window \(String(describing: row.inputs.contextWindow))"
            XCTAssertEqual(plan.client.rawValue, row.plan.client, cell)
            XCTAssertEqual(plan.argv, row.plan.argv, cell)
            XCTAssertEqual(plan.env, row.plan.env, cell)
            XCTAssertEqual(plan.unsetEnv, row.plan.unsetEnv, cell)
            XCTAssertEqual(plan.files, row.plan.files, "settings file bytes for \(cell)")
        }
    }

    func testStartMenuListsBothClientsAndMapsThemOneToOne() {
        XCTAssertTrue(LaunchTarget.allCases.contains(.claudeCode))
        XCTAssertTrue(LaunchTarget.allCases.contains(.codex))
        for client in AgentCLIClient.allCases {
            XCTAssertEqual(AgentCLIClient(target: client.target), client)
        }
        for target in LaunchTarget.allCases where target != .claudeCode && target != .codex {
            XCTAssertNil(AgentCLIClient(target: target), "\(target) is not a terminal agent client")
        }
        XCTAssertNil(AgentCLIClient(target: nil))
        XCTAssertEqual(LaunchTarget.claudeCode.title, "Claude Code")
        XCTAssertEqual(LaunchTarget.codex.title, "Codex")
        XCTAssertFalse(LaunchTarget.claudeCode.hasInlineForm)
        XCTAssertFalse(LaunchTarget.codex.hasInlineForm)
    }

    func testCredentialsNeverRideOnArgvAndLogsMaskThem() {
        for client in AgentCLIClient.allCases {
            let plan = AgentCLIIntegration.plan(
                client: client,
                baseURL: "http://127.0.0.1:8000",
                modelID: "m",
                contextWindow: 65_536,
                apiKey: "sk-secret",
                settingsPath: "/tmp/s.json"
            )
            XCTAssertFalse(plan.argv.contains { $0.contains("sk-secret") }, "\(client)")
            let shown = AgentCLIIntegration.displayCommand(plan)
            XCTAssertFalse(shown.contains("sk-secret"), "\(client)")
            XCTAssertTrue(shown.contains("<api key>"), "\(client)")
            // The open-server placeholder is not a secret and stays readable.
            let open = AgentCLIIntegration.plan(
                client: client,
                baseURL: "http://127.0.0.1:8000",
                modelID: "m",
                contextWindow: nil,
                apiKey: nil,
                settingsPath: "/tmp/s.json"
            )
            XCTAssertTrue(AgentCLIIntegration.displayCommand(open).contains(AgentCLIIntegration.localAPIKey))
        }
    }

    func testTerminalScriptCarriesTheLeaseProtocolAndRunsTheUsersCopy() throws {
        let plan = AgentCLIIntegration.plan(
            client: .claudeCode,
            baseURL: "http://127.0.0.1:8000",
            modelID: "mtplx-qwen38-27b-optimized-speed",
            contextWindow: 131_072,
            apiKey: nil,
            settingsPath: "/Users/someone/.mtplx/clients/claude-code/settings.json"
        )
        let handoffID = UUID()
        let script = AgentCLIIntegration.terminalScript(
            plan: plan,
            workspace: "/Users/someone/My Projects",
            fallbackExecutable: "/Users/someone/.local/bin/claude",
            handoffID: handoffID,
            receiptURL: URL(fileURLWithPath: "/h/open-claude-code-x.pid"),
            cancellationMarkerURL: URL(fileURLWithPath: "/h/open-claude-code-x.cancelled")
        )
        let lines = script.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.first, "#!/bin/zsh")
        XCTAssertTrue(lines.contains("export MTPLX_APP_HANDOFF_ID=\(handoffID.uuidString.lowercased())"))
        XCTAssertTrue(lines.contains("cd '/Users/someone/My Projects' || exit 1"))
        XCTAssertTrue(lines.contains("unset ANTHROPIC_API_KEY"))
        XCTAssertTrue(lines.contains("export ANTHROPIC_BASE_URL=http://127.0.0.1:8000"))
        XCTAssertTrue(lines.contains("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072"))
        // The user's PATH decides which copy runs; a known install path is
        // the fallback, and a missing client prints how to install it.
        let lookup = try XCTUnwrap(lines.firstIndex(of: "_mtplx_bin=\"$(command -v claude 2>/dev/null)\""))
        XCTAssertTrue(lines[lookup + 1].contains("/Users/someone/.local/bin/claude"))
        XCTAssertTrue(lines[lookup + 2].contains("code.claude.com"))
        XCTAssertTrue(lines[lookup + 2].contains("exit 127"))
        // Receipt after the lookup, a marker check on both sides, then exec.
        let receipt = try XCTUnwrap(lines.firstIndex { $0.hasPrefix("print -r -- \"$$\"") })
        XCTAssertGreaterThan(receipt, lookup)
        XCTAssertEqual(lines[receipt - 4], "if [[ -e \"$_mtplx_handoff_cancel\" ]]; then")
        XCTAssertEqual(lines[receipt + 2], "if [[ -e \"$_mtplx_handoff_cancel\" ]]; then")
        XCTAssertEqual(
            lines.last,
            "exec \"$_mtplx_bin\" --settings /Users/someone/.mtplx/clients/claude-code/settings.json"
                + " --append-system-prompt "
                + AgentCLIIntegration.shellQuote(AgentCLIIntegration.claudeCodeTerminalNote)
        )
        // Exports are sorted, so two launches of one plan write one script.
        let exports = lines.filter { $0.hasPrefix("export ") && !$0.contains("MTPLX_APP_HANDOFF_ID") }
        XCTAssertEqual(exports, exports.sorted())
    }

    func testCodexScriptQuotesEveryTOMLOverride() {
        let plan = AgentCLIIntegration.plan(
            client: .codex,
            baseURL: "http://127.0.0.1:8000",
            modelID: "mtplx-qwen38-flash-next-optimized-speed",
            contextWindow: 262_144,
            apiKey: nil,
            settingsPath: "<unused>"
        )
        XCTAssertTrue(plan.files.isEmpty, "Codex config.toml is never written")
        let script = AgentCLIIntegration.terminalScript(
            plan: plan,
            workspace: "/w",
            fallbackExecutable: nil,
            handoffID: UUID(),
            receiptURL: URL(fileURLWithPath: "/h/r.pid"),
            cancellationMarkerURL: URL(fileURLWithPath: "/h/m.cancelled")
        )
        let exec = script.split(separator: "\n").map(String.init).last ?? ""
        XCTAssertTrue(exec.contains("-c 'model_provider=\"mtplx\"'"), exec)
        XCTAssertTrue(exec.contains("-c 'model_providers.mtplx.base_url=\"http://127.0.0.1:8000/v1\"'"), exec)
        XCTAssertTrue(exec.contains("-c 'web_search=\"disabled\"'"), exec)
        XCTAssertTrue(exec.contains("-c model_context_window=262144"), exec)
        XCTAssertTrue(exec.hasSuffix("-m mtplx-qwen38-flash-next-optimized-speed"), exec)
        XCTAssertTrue(script.contains("export MTPLX_API_KEY=mtplx-local"))
        XCTAssertTrue(script.contains("developers.openai.com/codex/cli"))
    }

    func testPlanFilesAreWrittenOwnerOnlyInAFolderMTPLXOwns() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-agent-cli-\(UUID().uuidString.lowercased())")
        defer { try? FileManager.default.removeItem(at: directory) }
        let integration = AgentCLIIntegration(
            clientsDirectory: directory.appendingPathComponent("clients"),
            handoffDirectory: directory.appendingPathComponent("handoffs")
        )
        let plan = integration.plan(
            client: .claudeCode,
            configuration: MTPLXAppConfiguration(model: "/models/Qwen3.8-27B-MTPLX-Optimized-Speed"),
            baseURL: URL(string: "http://127.0.0.1:8123")!,
            apiKey: "sk-local",
            servedWindow: ServedExecutionWindow(tokens: 65_536, basis: "fit")
        )
        try AgentCLIIntegration.writePlanFiles(plan)
        let settings = integration.claudeCodeSettingsURL
        XCTAssertEqual(Array(plan.files.keys), [settings.path])
        XCTAssertFalse(settings.pathComponents.contains(".claude"))
        let attributes = try FileManager.default.attributesOfItem(atPath: settings.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        let folder = try FileManager.default.attributesOfItem(atPath: settings.deletingLastPathComponent().path)
        XCTAssertEqual((folder[.posixPermissions] as? NSNumber)?.intValue, 0o700)
        let parsed = try JSONSerialization.jsonObject(with: Data(contentsOf: settings)) as? [String: [String: String]]
        XCTAssertEqual(parsed?["env"]?["ANTHROPIC_AUTH_TOKEN"], "sk-local")
        XCTAssertEqual(parsed?["env"]?["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8123")
        XCTAssertEqual(parsed?["env"]?["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "65536")
        XCTAssertEqual(parsed?["env"]?["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "32000")
    }

    func testAnUnknownServedWindowLeavesTheClientsOwnDefault() {
        let integration = AgentCLIIntegration(
            clientsDirectory: URL(fileURLWithPath: "/tmp/c"),
            handoffDirectory: URL(fileURLWithPath: "/tmp/h")
        )
        for window in [nil, ServedExecutionWindow(tokens: 0)] {
            let claude = integration.plan(
                client: .claudeCode,
                configuration: MTPLXAppConfiguration(),
                baseURL: URL(string: "http://127.0.0.1:8000")!,
                apiKey: nil,
                servedWindow: window
            )
            XCTAssertNil(claude.env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
            let codex = integration.plan(
                client: .codex,
                configuration: MTPLXAppConfiguration(),
                baseURL: URL(string: "http://127.0.0.1:8000")!,
                apiKey: nil,
                servedWindow: window
            )
            XCTAssertFalse(codex.argv.contains { $0.hasPrefix("model_context_window=") })
        }
    }

    /// The Qwen 3.8 sampler law: never 0.6. The agent lanes leave the
    /// sampler to the model family, unlike the older OpenCode and Hermes
    /// presets.
    func testAgentClientLanesNeverPinTheQwen38SamplerTo0Point6() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-agent-lane-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let fake = directory.appendingPathComponent("mtplx")
        try "#!/bin/sh\nexit 0\n".write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": directory.path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": String(UInt64(128) * 1024 * 1024 * 1024),
        ])
        for model in [
            "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
            "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
        ] {
            for target in [LaunchTarget.claudeCode, .codex] {
                let command = try builder.buildServeCommand(
                    configuration: MTPLXAppConfiguration(executablePath: fake.path, model: model),
                    target: target,
                    launchID: "agent-lane"
                )
                let argv = command.arguments
                let cell = "\(target) \(model)"
                XCTAssertNotEqual(MTPLXCommandBuilder.flagValue("--temperature", in: argv), "0.6", cell)
                XCTAssertEqual(MTPLXCommandBuilder.flagValue("--scheduler-mode", in: argv), "serial", cell)
                XCTAssertEqual(MTPLXCommandBuilder.flagValue("--ssd-session-cache", in: argv), "on", cell)
                XCTAssertEqual(command.environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"], "1", cell)
            }
        }
    }

    /// Stop reaps exactly the process the handoff opened: the PID must carry
    /// this handoff's UUID in its environment.
    @MainActor
    func testStopReapsAnAgentHandoffByItsExactReceiptOnly() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-agent-stop-\(UUID().uuidString.lowercased())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let handoffID = UUID()
        let owned = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        let unrelated = try launchSleep(environment: [:])
        defer {
            for process in [owned, unrelated] where process.isRunning {
                process.terminate()
                process.waitUntilExit()
            }
        }
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 9),
            agentCLIIntegration: AgentCLIIntegration(
                clientsDirectory: directory.appendingPathComponent("clients"),
                handoffDirectory: directory.appendingPathComponent("handoffs")
            ),
            localFanRestorer: { true }
        )
        backend.recordLaunchedAgentCLITerminalHandoffLease(
            MTPLXTerminalHandoffLease(
                handoffID: handoffID,
                processID: Int(owned.processIdentifier),
                cancellationMarkerURL: directory.appendingPathComponent("claude.cancelled")
            ),
            client: .claudeCode
        )
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, [handoffID])

        await backend.stopDaemon()

        var exited = !owned.isRunning
        for _ in 0..<100 where !exited {
            try? await Task.sleep(for: .milliseconds(10))
            exited = !owned.isRunning
        }
        XCTAssertTrue(exited, "Stop must reap the Claude Code handoff it opened")
        XCTAssertTrue(unrelated.isRunning, "Stop must never signal a process without the handoff UUID")
        XCTAssertTrue(backend.terminalHandoffLeaseIDsForTesting.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("claude.cancelled").path
        ))
    }

    func testNoticeOnlyWhenTheClientDidNotStart() {
        let launched = AgentCLILaunchResult(
            client: .codex,
            action: .launched,
            command: "env codex",
            detail: "opened Codex in Terminal",
            terminalHandoffLease: nil
        )
        XCTAssertNil(ClientHandoffNotice.agentCLI(result: launched))
        let failed = AgentCLILaunchResult(
            client: .claudeCode,
            action: .unavailable,
            command: "env claude",
            detail: "no receipt",
            terminalHandoffLease: nil
        )
        let notice = ClientHandoffNotice.agentCLI(result: failed)
        XCTAssertEqual(notice?.target, .claudeCode)
        XCTAssertEqual(notice?.status, "Claude Code did not start")
        XCTAssertEqual(notice?.isWarning, true)
    }

    private func launchSleep(environment: [String: String]) throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", "import time; time.sleep(30)"]
        process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
        try process.run()
        return process
    }
}
