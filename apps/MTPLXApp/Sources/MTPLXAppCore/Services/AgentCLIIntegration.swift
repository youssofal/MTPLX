import Darwin
import Foundation

/// Claude Code and Codex in the Start menu: each opens in Terminal, talking
/// to the local MTPLX server on the picked model.
///
/// Swift twin of `mtplx/agent_clients.py`, which `mtplx start claude-code`,
/// `mtplx start codex` and `mtplx connect` use. `AgentCLIIntegrationTests`
/// compares the plans built here with `Tests/Fixtures/agent_client_plans.json`,
/// written from the Python module, so the app and the CLI open the same
/// client the same way.
///
/// Nothing in the user's own Claude Code or Codex setup is edited. Claude Code
/// gets a per-launch environment plus `--settings` naming a file MTPLX owns
/// (it ranks above the user's and the project's settings, so a gateway set
/// there cannot take this session elsewhere). Codex gets the `mtplx` provider
/// through `-c` overrides on its command line, for that one run.
public enum AgentCLIClient: String, CaseIterable, Sendable {
    case claudeCode = "claude-code"
    case codex

    public init?(target: LaunchTarget?) {
        switch target {
        case .claudeCode: self = .claudeCode
        case .codex: self = .codex
        default: return nil
        }
    }

    public var target: LaunchTarget {
        switch self {
        case .claudeCode: return .claudeCode
        case .codex: return .codex
        }
    }

    public var displayName: String {
        switch self {
        case .claudeCode: return "Claude Code"
        case .codex: return "Codex"
        }
    }

    var executableName: String {
        switch self {
        case .claudeCode: return "claude"
        case .codex: return "codex"
        }
    }

    /// Printed in Terminal when the client is not installed.
    var installHint: String {
        switch self {
        case .claudeCode: return "Install Claude Code from https://code.claude.com/docs/en/setup"
        case .codex: return "Install the Codex CLI from https://developers.openai.com/codex/cli"
        }
    }
}

public struct AgentCLIPlan: Equatable, Sendable {
    public let client: AgentCLIClient
    public let argv: [String]
    public let env: [String: String]
    public let unsetEnv: [String]
    /// Path to text of each MTPLX-owned file the launch writes (0600).
    public let files: [String: String]
}

public enum AgentCLILaunchAction: String, Equatable, Sendable {
    case launched
    case unavailable
}

public struct AgentCLILaunchResult: Equatable, Sendable {
    public let client: AgentCLIClient
    public let action: AgentCLILaunchAction
    /// The command with credentials masked, for the Logs pane.
    public let command: String
    public let detail: String
    public let terminalHandoffLease: MTPLXTerminalHandoffLease?
    /// The Codex app copy MTPLX opened, when Codex opened as the desktop app.
    public var codexDesktopInstance: CodexDesktopInstance? = nil
}

public struct AgentCLIIntegration: Sendable {
    public static let localAPIKey = "mtplx-local"
    static let claudeCodeAPITimeoutMS = "3000000"
    static let claudeCodeDefaultOutputTokens = 32_000
    static let claudeCodeMaxOutputTokensCap = 128_000
    /// Claude Code sends every MCP tool definition in full to a host that is
    /// not Anthropic's unless tool search is on (820 tools made a 254,636-token
    /// first prompt); the server keeps deferred tools out of the prompt.
    static let claudeCodeToolSearch = "true"
    /// Appended to Claude Code's system prompt (`--append-system-prompt`), the
    /// same text as `mtplx/agent_clients.py`: its own prompt says only that
    /// replies are markdown in a terminal, and a local model wrote raw LaTeX
    /// there (16 Nov), then searched the web after "without searching".
    static let claudeCodeTerminalNote =
        "Your replies are shown in a terminal that draws Markdown but not LaTeX: write math "
        + "as plain text with Unicode symbols (x², √2, ∑, ≤, →, ∂f/∂x), never with $...$, "
        + "\\(...\\) or other LaTeX commands. When the user asks you not to search or not to "
        + "use a tool, keep to that for the rest of the topic unless they say otherwise."
    static let claudeCodeUnsetEnv = [
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    ]
    static let codexProviderID = "mtplx"
    static let codexAPIKeyEnv = "MTPLX_API_KEY"
    static let codexWebSearch = "disabled"

    /// `~/.mtplx/clients`: Claude Code's MTPLX settings file lives here.
    public let clientsDirectory: URL
    /// `~/.mtplx/handoffs`: one-use Terminal scripts, receipts, markers.
    public let handoffDirectory: URL

    public init(
        clientsDirectory: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("clients", isDirectory: true),
        handoffDirectory: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("handoffs", isDirectory: true)
    ) {
        self.clientsDirectory = clientsDirectory
        self.handoffDirectory = handoffDirectory
    }

    public var claudeCodeSettingsURL: URL {
        clientsDirectory
            .appendingPathComponent("claude-code", isDirectory: true)
            .appendingPathComponent("settings.json")
    }

    // MARK: - The plan (mirrors mtplx.agent_clients.build_plan)

    static func claudeCodeOutputTokens(contextWindow: Int?) -> Int? {
        guard let contextWindow, contextWindow > 0 else { return nil }
        return max(
            claudeCodeDefaultOutputTokens,
            min(claudeCodeMaxOutputTokensCap, contextWindow / 4)
        )
    }

    /// `apiKey` is the key the server requires (nil for an open localhost
    /// server). It travels in the environment or in the 0600 settings file,
    /// never on the command line.
    public static func plan(
        client: AgentCLIClient,
        baseURL: String,
        modelID: String,
        contextWindow: Int?,
        apiKey: String?,
        settingsPath: String,
        executable: String? = nil
    ) -> AgentCLIPlan {
        let root = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        let key = apiKey.flatMap { $0.isEmpty ? nil : $0 } ?? localAPIKey
        switch client {
        case .claudeCode:
            // A base URL without a credential keeps the claude.ai login
            // active, and Claude Code would send that login to this server.
            var env = [
                "ANTHROPIC_BASE_URL": root,
                "ANTHROPIC_AUTH_TOKEN": key,
                "ANTHROPIC_MODEL": modelID,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": modelID,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": modelID,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": modelID,
                "CLAUDE_CODE_SUBAGENT_MODEL": modelID,
                "API_TIMEOUT_MS": claudeCodeAPITimeoutMS,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "ENABLE_TOOL_SEARCH": claudeCodeToolSearch,
            ]
            if let contextWindow,
               let outputTokens = claudeCodeOutputTokens(contextWindow: contextWindow) {
                env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = String(contextWindow)
                env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = String(outputTokens)
            }
            return AgentCLIPlan(
                client: client,
                argv: [
                    executable ?? client.executableName,
                    "--settings", settingsPath,
                    "--append-system-prompt", claudeCodeTerminalNote,
                ],
                env: env,
                unsetEnv: claudeCodeUnsetEnv,
                files: [settingsPath: settingsJSON(env: env)]
            )
        case .codex:
            var argv = [executable ?? client.executableName]
            for override in codexConfigOverrides(baseURL: root, contextWindow: contextWindow) {
                argv.append(contentsOf: ["-c", override])
            }
            argv.append(contentsOf: ["-m", modelID])
            return AgentCLIPlan(
                client: client,
                argv: argv,
                env: [codexAPIKeyEnv: key],
                unsetEnv: [],
                files: [:]
            )
        }
    }

    /// The `-c key=value` overrides, as TOML values, in a fixed order.
    static func codexConfigOverrides(baseURL: String, contextWindow: Int?) -> [String] {
        let provider = "model_providers.\(codexProviderID)"
        var overrides = [
            "model_provider=\"\(codexProviderID)\"",
            "\(provider).name=\"MTPLX\"",
            "\(provider).base_url=\(jsonString(baseURL + "/v1"))",
            "\(provider).wire_api=\"responses\"",
            "\(provider).env_key=\"\(codexAPIKeyEnv)\"",
            "\(provider).requires_openai_auth=false",
            "\(provider).http_headers={ \"x-mtplx-client\" = \"codex\" }",
            // Codex adds a server-hosted web_search tool by default; a local
            // server cannot run it and refuses the request.
            "web_search=\"\(codexWebSearch)\"",
        ]
        if let contextWindow, contextWindow > 0 {
            overrides.append("model_context_window=\(contextWindow)")
        }
        return overrides
    }

    /// `json.dumps({"env": env}, indent=2, sort_keys=True) + "\n"`, byte for byte.
    static func settingsJSON(env: [String: String]) -> String {
        let rows = env.keys.sorted().map { key in
            "    \(jsonString(key)): \(jsonString(env[key] ?? ""))"
        }
        return "{\n  \"env\": {\n" + rows.joined(separator: ",\n") + "\n  }\n}\n"
    }

    /// A JSON string literal as Python's `json.dumps` writes it (ASCII only).
    static func jsonString(_ value: String) -> String {
        var out = "\""
        for unit in value.utf16 {
            switch unit {
            case 0x22: out += "\\\""
            case 0x5C: out += "\\\\"
            case 0x0A: out += "\\n"
            case 0x0D: out += "\\r"
            case 0x09: out += "\\t"
            case 0x08: out += "\\b"
            case 0x0C: out += "\\f"
            case 0x20..<0x7F: out.unicodeScalars.append(Unicode.Scalar(UInt8(unit)))
            default: out += String(format: "\\u%04x", unit)
            }
        }
        return out + "\""
    }

    /// The command with credentials masked, for logs (`display_command`).
    public static func displayCommand(_ plan: AgentCLIPlan) -> String {
        var parts = ["env"]
        for name in plan.unsetEnv { parts.append(contentsOf: ["-u", name]) }
        for key in plan.env.keys.sorted() {
            var value = plan.env[key] ?? ""
            if key == "ANTHROPIC_AUTH_TOKEN" || key == codexAPIKeyEnv, value != localAPIKey {
                value = "<api key>"
            }
            parts.append("\(key)=\(shellQuote(value))")
        }
        parts.append(plan.argv.map(shellQuote).joined(separator: " "))
        return parts.joined(separator: " ")
    }

    /// Python's `shlex.quote`.
    static func shellQuote(_ value: String) -> String {
        let safe = CharacterSet(
            charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
        )
        if !value.isEmpty, value.unicodeScalars.allSatisfy({ safe.contains($0) }) {
            return value
        }
        return "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    // MARK: - Terminal handoff

    /// A known install path, used when the user's own PATH does not name the
    /// client (the script asks `command -v` first, as their terminal would).
    static func knownExecutable(
        for client: AgentCLIClient,
        homeDirectory: String = NSHomeDirectory(),
        fileManager: FileManager = .default
    ) -> String? {
        for path in [
            "\(homeDirectory)/.local/bin/\(client.executableName)",
            "/opt/homebrew/bin/\(client.executableName)",
            "/usr/local/bin/\(client.executableName)",
            "\(homeDirectory)/.npm-global/bin/\(client.executableName)",
        ] where fileManager.isExecutableFile(atPath: path) {
            return path
        }
        return nil
    }

    /// The zsh script Terminal runs. It carries the lease protocol of the Pi
    /// and Hermes handoffs: the UUID in the environment, a PID receipt
    /// written just before `exec`, and a cancellation marker checked on both
    /// sides of the receipt, so Stop reaps exactly the process it opened.
    static func terminalScript(
        plan: AgentCLIPlan,
        workspace: String,
        fallbackExecutable: String?,
        handoffID: UUID,
        receiptURL: URL,
        cancellationMarkerURL: URL
    ) -> String {
        var lines = [
            "#!/bin/zsh",
            "_mtplx_handoff_cancel=\(shellQuote(cancellationMarkerURL.path))",
            "_mtplx_handoff_receipt=\(shellQuote(receiptURL.path))",
            "export \(MTPLXTerminalHandoffLease.environmentVariable)=\(shellQuote(handoffID.uuidString.lowercased()))",
            "if [[ -e \"$_mtplx_handoff_cancel\" ]]; then",
            "  exit 0",
            "fi",
            "cd \(shellQuote(workspace)) || exit 1",
        ]
        for name in plan.unsetEnv { lines.append("unset \(name)") }
        for key in plan.env.keys.sorted() {
            lines.append("export \(key)=\(shellQuote(plan.env[key] ?? ""))")
        }
        let name = plan.argv.first ?? plan.client.executableName
        lines.append("_mtplx_bin=\"$(command -v \(shellQuote(name)) 2>/dev/null)\"")
        if let fallbackExecutable {
            lines.append(
                "[[ -z \"$_mtplx_bin\" && -x \(shellQuote(fallbackExecutable)) ]] && _mtplx_bin=\(shellQuote(fallbackExecutable))"
            )
        }
        lines.append(
            "if [[ -z \"$_mtplx_bin\" ]]; then print -r -- \(shellQuote(plan.client.installHint)); exit 127; fi"
        )
        lines.append(contentsOf: [
            "if [[ -e \"$_mtplx_handoff_cancel\" ]]; then",
            "  exit 0",
            "fi",
            "umask 077",
            "print -r -- \"$$\" > \"${_mtplx_handoff_receipt}.$$.tmp\"",
            "mv -f \"${_mtplx_handoff_receipt}.$$.tmp\" \"$_mtplx_handoff_receipt\"",
            "if [[ -e \"$_mtplx_handoff_cancel\" ]]; then",
            "  exit 0",
            "fi",
            (["exec \"$_mtplx_bin\""] + plan.argv.dropFirst().map(shellQuote)).joined(separator: " "),
        ])
        return lines.joined(separator: "\n") + "\n"
    }

    /// Writes the plan's MTPLX-owned files, owner-only.
    static func writePlanFiles(_ plan: AgentCLIPlan) throws {
        let fileManager = FileManager.default
        for (path, text) in plan.files {
            let url = URL(fileURLWithPath: path)
            let directory = url.deletingLastPathComponent()
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
            try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            try Data(text.utf8).write(to: url, options: [.atomic])
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }

    /// The plan for the running daemon: its address, its key, the model it
    /// serves and the window it executes (`/health` `execution_window`).
    public func plan(
        client: AgentCLIClient,
        configuration: MTPLXAppConfiguration,
        baseURL: URL,
        apiKey: String?,
        servedWindow: ServedExecutionWindow?
    ) -> AgentCLIPlan {
        Self.plan(
            client: client,
            baseURL: baseURL.absoluteString,
            modelID: OpenCodeIntegration.modelID(for: configuration.model),
            contextWindow: servedWindow.flatMap { $0.tokens > 0 ? $0.tokens : nil },
            apiKey: apiKey,
            settingsPath: claudeCodeSettingsURL.path
        )
    }

    @MainActor
    public func launchInTerminal(
        client: AgentCLIClient,
        configuration: MTPLXAppConfiguration,
        baseURL: URL,
        apiKey: String?,
        servedWindow: ServedExecutionWindow?,
        isCurrent: (() -> Bool)? = nil
    ) async -> AgentCLILaunchResult {
        let plan = plan(
            client: client,
            configuration: configuration,
            baseURL: baseURL,
            apiKey: apiKey,
            servedWindow: servedWindow
        )
        let command = Self.displayCommand(plan)
        func result(
            _ action: AgentCLILaunchAction,
            _ detail: String,
            lease: MTPLXTerminalHandoffLease? = nil
        ) -> AgentCLILaunchResult {
            AgentCLILaunchResult(
                client: client,
                action: action,
                command: command,
                detail: detail,
                terminalHandoffLease: lease
            )
        }
        // Never shown: a stale handoff only reaches the log.
        let stale = "\(client.displayName) handoff cancelled because the daemon lifecycle changed."
        guard isCurrent?() ?? true else { return result(.unavailable, stale) }
        #if os(macOS)
        let handoffID = UUID()
        let suffix = "\(client.rawValue)-\(handoffID.uuidString.lowercased())"
        let commandURL = handoffDirectory.appendingPathComponent("open-\(suffix).command")
        let receiptURL = handoffDirectory.appendingPathComponent("open-\(suffix).pid")
        let markerURL = handoffDirectory.appendingPathComponent("open-\(suffix).cancelled")
        let couldNotOpen = { (reason: String) in
            result(.unavailable, tr("MTPLX could not open Terminal for %@: %@", client.displayName, reason))
        }
        do {
            try Self.writePlanFiles(plan)
            let script = Self.terminalScript(
                plan: plan,
                workspace: PiIntegration.resolvedWorkspacePath(configuration: configuration),
                fallbackExecutable: Self.knownExecutable(for: client),
                handoffID: handoffID,
                receiptURL: receiptURL,
                cancellationMarkerURL: markerURL
            )
            try MTPLXTerminalHandoffLease.writeSecureCommandScript(script, to: commandURL)
        } catch {
            return couldNotOpen(String(describing: error))
        }
        guard isCurrent?() ?? true else {
            await cancelPendingTerminalHandoff(
                handoffID: handoffID,
                commandURL: commandURL,
                receiptURL: receiptURL,
                cancellationMarkerURL: markerURL
            )
            return result(.unavailable, stale)
        }

        // The terminal the user picked for this client (Settings, or
        // `claude_code_opens_in` / `codex_opens_in`).
        let opening = ClientOpenings.currentTerminal(client.target)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", opening.openApplication, commandURL.path]
        let stderr = Pipe()
        process.standardError = stderr
        // Called from the main actor: a wedged LaunchServices `open` costs one
        // bounded window, never a beachball (#158 pattern).
        let stderrTail = SubprocessTailBuffer(capacity: 4096)
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { stderrTail.append(chunk) }
        }
        defer { stderr.fileHandleForReading.readabilityHandler = nil }
        let watchdog = SubprocessWatchdog(process)
        let opened: Bool
        do {
            try process.run()
            opened = watchdog.wait(for: process, timeout: 30) && process.terminationStatus == 0
        } catch {
            await cancelPendingTerminalHandoff(
                handoffID: handoffID,
                commandURL: commandURL,
                receiptURL: receiptURL,
                cancellationMarkerURL: markerURL
            )
            return couldNotOpen(String(describing: error))
        }
        guard opened else {
            await cancelPendingTerminalHandoff(
                handoffID: handoffID,
                commandURL: commandURL,
                receiptURL: receiptURL,
                cancellationMarkerURL: markerURL
            )
            let message = stderrTail.snapshot().trimmingCharacters(in: .whitespacesAndNewlines)
            return couldNotOpen(message.isEmpty ? "open exited \(process.terminationStatus)" : message)
        }
        let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoffID,
            receiptURL: receiptURL,
            cancellationMarkerURL: markerURL,
            commandURL: commandURL,
            isCurrent: isCurrent
        )
        let didNotStart = tr(
            "MTPLX opened Terminal, but %@ did not start. If Terminal says it is not installed, install it and pick it again.",
            client.displayName
        )
        if receipt.cancellationRequested {
            if let lease = receipt.lease { _ = cancelTerminalHandoff(lease) }
            return result(.unavailable, (isCurrent?() ?? true) ? didNotStart : stale)
        }
        guard let lease = receipt.lease else { return result(.unavailable, didNotStart) }
        guard isCurrent?() ?? true else {
            _ = cancelTerminalHandoff(lease)
            return result(.unavailable, stale)
        }
        return result(.launched, "opened \(client.displayName) in \(opening.terminalName)", lease: lease)
        #else
        return result(.unavailable, "opening \(client.displayName) needs macOS Terminal")
        #endif
    }

    /// Covers every failure after the script is durable: a late Terminal
    /// launch sees the marker; a receipt from the final-check race is reaped
    /// only after proving the exact UUID in that process's environment.
    @MainActor
    private func cancelPendingTerminalHandoff(
        handoffID: UUID,
        commandURL: URL,
        receiptURL: URL,
        cancellationMarkerURL: URL
    ) async {
        let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoffID,
            receiptURL: receiptURL,
            cancellationMarkerURL: cancellationMarkerURL,
            commandURL: commandURL,
            isCurrent: { false },
            timeoutSeconds: 0,
            delayedCancellationSeconds: 1
        )
        if let lease = receipt.lease {
            _ = cancelTerminalHandoff(lease)
        }
    }

    /// Ends one exact handoff. Never process discovery: the PID must still
    /// carry this handoff's UUID in its environment.
    @MainActor
    @discardableResult
    public func cancelTerminalHandoff(_ lease: MTPLXTerminalHandoffLease) -> Bool {
        let cancellationMarked = MTPLXTerminalHandoffLease.writeCancellationMarker(
            at: lease.cancellationMarkerURL
        )
        let commandRemoved = lease.commandURL.map {
            MTPLXTerminalHandoffLease.removeDurableCommandScript(at: $0)
        } ?? true
        let receiptRemoved = lease.receiptURL.map {
            MTPLXTerminalHandoffLease.removeDurableCommandScript(at: $0)
        } ?? true
        let pid = pid_t(lease.processID)
        guard pid > 1,
              MTPLXTerminalHandoffLease.process(pid: pid, hasExactHandoffID: lease.handoffID)
        else { return false }
        if kill(pid, 0) == 0 {
            _ = kill(pid, SIGTERM)
            for _ in 0..<20 {
                if kill(pid, 0) != 0 { break }
                Thread.sleep(forTimeInterval: 0.05)
            }
            if kill(pid, 0) == 0 { _ = kill(pid, SIGKILL) }
        }
        return cancellationMarked && commandRemoved && receiptRemoved
    }
}
