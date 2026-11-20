import Darwin
import Foundation
#if canImport(AppKit)
import AppKit
#endif

/// The Codex desktop app (bundle `com.openai.codex`, shipped as ChatGPT.app)
/// opened as MTPLX's own instance.
///
/// Swift twin of the Codex desktop part of `mtplx/agent_clients.py`
/// (`mtplx start codex`, `mtplx connect codex --desktop`). The app has its
/// own isolated-launch path: started with `open -n`, `CODEX_HOME`,
/// `CODEX_ELECTRON_USER_DATA_PATH` and a matching `--user-data-dir`, it runs
/// as a separate instance on those folders beside the user's own instance and
/// leaves `~/.codex` and its normal data folder alone. MTPLX owns both folders
/// (`~/.mtplx/clients/codex-desktop`, created on first use, about 1 GB once
/// the app has started there). Proven on 26.1106.51428 on 12 Nov 2026: both
/// folders honoured, two instances side by side, no sign-in screen with the
/// mtplx provider, the user's history and settings unchanged. An older build
/// may not know the user-data variable, so it gets the Codex CLI in Terminal.
///
/// The instance is the app's main process with MTPLX's own `--user-data-dir`
/// on its command line, read from the kernel's argv image: the user's own
/// Codex app never carries that argument, so Stop can only ever end the copy
/// MTPLX opened.
public struct CodexDesktopApp: Equatable, Sendable {
    public let path: String
    public let version: String
    public let supported: Bool
    /// `Contents/MacOS/<CFBundleExecutable>`: the instance's main process.
    public let executablePath: String

    public init(path: String, version: String, supported: Bool, executablePath: String) {
        self.path = path
        self.version = version
        self.supported = supported
        self.executablePath = executablePath
    }
}

/// One running copy of the Codex app that MTPLX opened.
public struct CodexDesktopInstance: Equatable, Sendable {
    public let processID: Int32
    public let executablePath: String
    public let userDataPath: String

    public init(processID: Int32, executablePath: String, userDataPath: String) {
        self.processID = processID
        self.executablePath = executablePath
        self.userDataPath = userDataPath
    }

    /// The command-line argument that marks MTPLX's copy.
    var marker: String { "--user-data-dir=\(userDataPath)" }
}

public struct CodexDesktopPlan: Equatable, Sendable {
    public let argv: [String]
    /// Path to text of the instance's config.toml (written 0600).
    public let files: [String: String]
    public let folders: [String]
}

public enum CodexDesktopIntegration {
    public static let bundleIdentifier = "com.openai.codex"
    public static let minimumVersion = [26, 1106]
    static let appNames = ["ChatGPT.app", "Codex.app"]
    static let ownedKeys: Set<String> = ["model", "model_provider", "web_search", "model_context_window"]
    static let providerTable = "model_providers.\(AgentCLIIntegration.codexProviderID)"
    /// A new copy's composer effort (the app starts a custom model at low);
    /// MTPLX's measured coding-agent default. The user's own line wins.
    static let defaultEffort = "medium"
    static let configHeader =
        "# MTPLX sets model, model_provider, web_search, model_context_window and "
        + "[model_providers.mtplx] on each launch; everything else here is kept."

    // MARK: - Finding the app

    static func versionTuple(_ text: String) -> [Int] {
        var parts: [Int] = []
        for piece in text.split(separator: ".", omittingEmptySubsequences: false) {
            let digits = piece.filter(\.isNumber)
            guard !digits.isEmpty, let value = Int(digits) else { break }
            parts.append(value)
        }
        return parts
    }

    /// Python's tuple `>=`.
    static func isAtLeast(_ version: [Int], _ minimum: [Int]) -> Bool {
        for (left, right) in zip(version, minimum) where left != right {
            return left > right
        }
        return version.count >= minimum.count
    }

    public static func findApp(
        searchDirectories: [URL] = [
            URL(fileURLWithPath: "/Applications", isDirectory: true),
            URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Applications", isDirectory: true),
        ]
    ) -> CodexDesktopApp? {
        for root in searchDirectories {
            for name in appNames {
                let bundle = root.appendingPathComponent(name, isDirectory: true)
                let plistURL = bundle.appendingPathComponent("Contents/Info.plist")
                guard let data = try? Data(contentsOf: plistURL),
                      let info = try? PropertyListSerialization.propertyList(from: data, format: nil)
                        as? [String: Any],
                      info["CFBundleIdentifier"] as? String == bundleIdentifier
                else { continue }
                let version = info["CFBundleShortVersionString"] as? String ?? ""
                let executable = info["CFBundleExecutable"] as? String ?? ""
                return CodexDesktopApp(
                    path: bundle.path,
                    version: version,
                    supported: !executable.isEmpty
                        && isAtLeast(versionTuple(version), minimumVersion),
                    executablePath: bundle.appendingPathComponent("Contents/MacOS")
                        .appendingPathComponent(executable).path
                )
            }
        }
        return nil
    }

    // MARK: - The running copy (mirrors codex_desktop_instances)

    /// Executable path and argv from a KERN_PROCARGS2 image: argc, the exec
    /// path, padding, then argc NUL-terminated strings (the environment
    /// follows and is not read). Nil for a malformed image.
    static func parseProcessArguments(_ bytes: [UInt8]) -> (executable: String, argv: [String])? {
        guard bytes.count >= 4 else { return nil }
        let argc = bytes.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 0, as: Int32.self)) }
        guard argc >= 0 else { return nil }
        var cursor = 4
        func cString() -> String? {
            guard cursor < bytes.count, let end = bytes[cursor...].firstIndex(of: 0) else { return nil }
            defer { cursor = end + 1 }
            return String(decoding: bytes[cursor..<end], as: UTF8.self)
        }
        guard let executable = cString() else { return nil }
        while cursor < bytes.count, bytes[cursor] == 0 { cursor += 1 }
        var argv: [String] = []
        for _ in 0..<argc {
            guard let argument = cString() else { return nil }
            argv.append(argument)
        }
        return (executable, argv)
    }

    static func processArguments(_ pid: Int32) -> (executable: String, argv: [String])? {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var size = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &size, nil, 0) == 0, size > 4 else { return nil }
        var bytes = [UInt8](repeating: 0, count: size)
        guard bytes.withUnsafeMutableBytes({
            sysctl(&mib, UInt32(mib.count), $0.baseAddress, &size, nil, 0)
        }) == 0 else { return nil }
        return parseProcessArguments(Array(bytes.prefix(size)))
    }

    /// True while `instance.processID` is still the app's main process with
    /// MTPLX's `--user-data-dir`; false for any other process (a reused pid,
    /// the user's own Codex app, a helper process).
    static func isRunning(
        _ instance: CodexDesktopInstance,
        arguments: (Int32) -> (executable: String, argv: [String])? = processArguments
    ) -> Bool {
        guard instance.processID > 1, let image = arguments(instance.processID) else { return false }
        let executable = URL(fileURLWithPath: image.executable).resolvingSymlinksInPath().path
        let expected = URL(fileURLWithPath: instance.executablePath).resolvingSymlinksInPath().path
        return executable == expected && image.argv.dropFirst().contains(instance.marker)
    }

    /// MTPLX's running copies of the app on `userData` (normally one).
    public static func instances(of app: CodexDesktopApp, userData: String) -> [CodexDesktopInstance] {
        #if canImport(AppKit)
        return NSWorkspace.shared.runningApplications
            .filter { $0.bundleIdentifier == bundleIdentifier }
            .map {
                CodexDesktopInstance(
                    processID: $0.processIdentifier,
                    executablePath: app.executablePath,
                    userDataPath: userData
                )
            }
            .filter { isRunning($0) }
        #else
        return []
        #endif
    }

    /// Ends one copy MTPLX opened: SIGTERM (the app closes its windows and
    /// databases), SIGKILL only if it is still that same process after
    /// `graceSeconds`. Never signals a process that is not the copy. Blocks
    /// up to `graceSeconds`; call it off the main actor.
    @discardableResult
    public static func end(_ instance: CodexDesktopInstance, graceSeconds: TimeInterval = 3) -> Bool {
        guard isRunning(instance) else { return false }
        _ = kill(instance.processID, SIGTERM)
        let deadline = Date().addingTimeInterval(graceSeconds)
        while Date() < deadline {
            if !isRunning(instance) { return true }
            Thread.sleep(forTimeInterval: 0.1)
        }
        if isRunning(instance) { _ = kill(instance.processID, SIGKILL) }
        return true
    }

    // MARK: - The instance's config.toml (mirrors codex_desktop_config_text)

    static func isOwnedTable(_ table: String) -> Bool {
        table == providerTable || table.hasPrefix(providerTable + ".")
    }

    public static func configText(
        existing: String?,
        baseURL: String,
        modelID: String,
        contextWindow: Int?,
        apiKey: String?
    ) -> String {
        let root = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        var owned = [
            "model = \(AgentCLIIntegration.jsonString(modelID))",
            "model_provider = \"\(AgentCLIIntegration.codexProviderID)\"",
            "web_search = \"\(AgentCLIIntegration.codexWebSearch)\"",
        ]
        if let contextWindow, contextWindow > 0 {
            owned.append("model_context_window = \(contextWindow)")
        }
        var provider = [
            "[\(providerTable)]",
            "name = \"MTPLX\"",
            "base_url = \(AgentCLIIntegration.jsonString(root + "/v1"))",
            "wire_api = \"responses\"",
            "requires_openai_auth = false",
            "http_headers = { \"x-mtplx-client\" = \"codex\" }",
        ]
        if let apiKey, !apiKey.isEmpty {
            provider.append("experimental_bearer_token = \(AgentCLIIntegration.jsonString(apiKey))")
        }
        var top: [String] = []
        var tables: [String] = []
        var hasEffort = false
        for statement in TOMLLines.statements(existing ?? "") {
            if statement.table.isEmpty {
                if let key = statement.key, ownedKeys.contains(key) { continue }
                if statement.lines == [configHeader] { continue }
                hasEffort = hasEffort || statement.key == "model_reasoning_effort"
                top.append(contentsOf: statement.lines)
            } else if !isOwnedTable(statement.table) {
                tables.append(contentsOf: statement.lines)
            }
        }
        if !hasEffort {
            owned.append("model_reasoning_effort = \"\(defaultEffort)\"")
        }
        while let first = top.first, TOMLLines.pyStrip(first).isEmpty { top.removeFirst() }
        while let last = top.last, TOMLLines.pyStrip(last).isEmpty { top.removeLast() }
        while let last = tables.last, TOMLLines.pyStrip(last).isEmpty { tables.removeLast() }
        var out = [configHeader] + owned + top + [""]
        if !tables.isEmpty { out += tables + [""] }
        out += provider
        return out.joined(separator: "\n") + "\n"
    }

    public static func plan(
        appPath: String,
        baseURL: String,
        modelID: String,
        contextWindow: Int?,
        apiKey: String?,
        existingConfig: String?,
        home: String,
        userData: String
    ) -> CodexDesktopPlan {
        let configPath = (home as NSString).appendingPathComponent("config.toml")
        return CodexDesktopPlan(
            argv: [
                "/usr/bin/open",
                "-n",
                "--env",
                "CODEX_HOME=\(home)",
                "--env",
                "CODEX_ELECTRON_USER_DATA_PATH=\(userData)",
                appPath,
                "--args",
                "--user-data-dir=\(userData)",
            ],
            files: [
                configPath: configText(
                    existing: existingConfig,
                    baseURL: baseURL,
                    modelID: modelID,
                    contextWindow: contextWindow,
                    apiKey: apiKey
                ),
            ],
            folders: [home, userData]
        )
    }
}

extension AgentCLIIntegration {
    public var codexDesktopHome: URL {
        clientsDirectory
            .appendingPathComponent("codex-desktop", isDirectory: true)
            .appendingPathComponent("home", isDirectory: true)
    }

    public var codexDesktopUserData: URL {
        clientsDirectory
            .appendingPathComponent("codex-desktop", isDirectory: true)
            .appendingPathComponent("user-data", isDirectory: true)
    }

    /// Opens the Codex desktop app as MTPLX's own instance on the running
    /// daemon. Never `~/.codex`; the previous instance config is kept beside
    /// the new one as `config.toml.mtplx-previous`. A copy MTPLX opened
    /// earlier is closed first, so the new one starts on this server's
    /// address, model and window. The result carries the new copy's process,
    /// which Stop and a daemon exit end.
    @MainActor
    public func launchCodexDesktop(
        app: CodexDesktopApp,
        configuration: MTPLXAppConfiguration,
        baseURL: URL,
        apiKey: String?,
        servedWindow: ServedExecutionWindow?,
        instanceTimeoutSeconds: TimeInterval = 15
    ) async -> AgentCLILaunchResult {
        let userData = codexDesktopUserData.path
        for previous in CodexDesktopIntegration.instances(of: app, userData: userData) {
            await Task.detached(priority: .userInitiated) {
                CodexDesktopIntegration.end(previous)
            }.value
        }
        let configURL = codexDesktopHome.appendingPathComponent("config.toml")
        let existing: String?
        do {
            existing = try String(contentsOf: configURL, encoding: .utf8)
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            existing = nil
        } catch {
            return codexDesktopResult(.unavailable, String(describing: error), command: "")
        }
        let plan = CodexDesktopIntegration.plan(
            appPath: app.path,
            baseURL: baseURL.absoluteString,
            modelID: OpenCodeIntegration.modelID(for: configuration.model),
            contextWindow: servedWindow.flatMap { $0.tokens > 0 ? $0.tokens : nil },
            apiKey: apiKey,
            existingConfig: existing,
            home: codexDesktopHome.path,
            userData: userData
        )
        let command = plan.argv.map(Self.shellQuote).joined(separator: " ")
        do {
            for folder in plan.folders {
                try MTPLXTerminalHandoffLease.prepareArtifactDirectory(URL(fileURLWithPath: folder))
            }
            if let existing {
                let backup = configURL.deletingLastPathComponent()
                    .appendingPathComponent("config.toml.mtplx-previous")
                try Data(existing.utf8).write(to: backup, options: [.atomic])
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
            }
            for (path, text) in plan.files {
                try Data(text.utf8).write(to: URL(fileURLWithPath: path), options: [.atomic])
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path)
            }
        } catch {
            return codexDesktopResult(.unavailable, String(describing: error), command: command)
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: plan.argv[0])
        process.arguments = Array(plan.argv.dropFirst())
        let stderr = Pipe()
        process.standardError = stderr
        let stderrTail = SubprocessTailBuffer(capacity: 4096)
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { stderrTail.append(chunk) }
        }
        defer { stderr.fileHandleForReading.readabilityHandler = nil }
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            return codexDesktopResult(.unavailable, String(describing: error), command: command)
        }
        guard watchdog.wait(for: process, timeout: 30), process.terminationStatus == 0 else {
            let message = stderrTail.snapshot().trimmingCharacters(in: .whitespacesAndNewlines)
            return codexDesktopResult(
                .unavailable,
                message.isEmpty ? "open exited \(process.terminationStatus)" : message,
                command: command
            )
        }
        // `open` returns once LaunchServices has the request; the copy's main
        // process appears a moment later.
        let deadline = Date().addingTimeInterval(instanceTimeoutSeconds)
        var instance: CodexDesktopInstance?
        while instance == nil {
            instance = CodexDesktopIntegration.instances(of: app, userData: userData).first
            if instance != nil || Date() >= deadline { break }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
        let folder = (codexDesktopHome.deletingLastPathComponent().path as NSString).abbreviatingWithTildeInPath
        guard let instance else {
            return codexDesktopResult(
                .launched,
                "opened the Codex app \(app.version) as MTPLX's own instance (\(folder)), "
                    + "but its process did not appear within \(Int(instanceTimeoutSeconds)) s, "
                    + "so Stop cannot close it",
                command: command
            )
        }
        return codexDesktopResult(
            .launched,
            "opened the Codex app \(app.version) as MTPLX's own instance (\(folder)), pid \(instance.processID)",
            command: command,
            instance: instance
        )
    }

    private func codexDesktopResult(
        _ action: AgentCLILaunchAction,
        _ detail: String,
        command: String,
        instance: CodexDesktopInstance? = nil
    ) -> AgentCLILaunchResult {
        AgentCLILaunchResult(
            client: .codex,
            action: action,
            command: command,
            detail: detail,
            terminalHandoffLease: nil,
            codexDesktopInstance: instance
        )
    }
}
