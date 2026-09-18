import Darwin
import Foundation

public struct OMPConfigResult: Equatable, Sendable {
    public let configPath: String
    public let baseURL: String
    public let modelReference: String
    public let launchCommand: String
    public let didChange: Bool
    public let backupPath: String?
}

public enum OMPLaunchAction: String, Equatable, Sendable {
    case launched
    case unavailable
}

public struct OMPLaunchResult: Equatable, Sendable {
    public let action: OMPLaunchAction
    public let command: String
    public let detail: String
    public let launchedProcessIDs: [Int]
    public let terminalHandoffLease: MTPLXTerminalHandoffLease?

    public init(
        action: OMPLaunchAction,
        command: String,
        detail: String,
        launchedProcessIDs: [Int] = [],
        terminalHandoffLease: MTPLXTerminalHandoffLease? = nil
    ) {
        self.action = action
        self.command = command
        self.detail = detail
        self.launchedProcessIDs = launchedProcessIDs
        self.terminalHandoffLease = terminalHandoffLease
    }
}

/// Installs MTPLX as an Oh My Pi custom provider and launches one exact OMP
/// Terminal handoff. The YAML writer is deliberately narrow: it edits only
/// `providers.mtplx`, preserves all other provider text, and rejects malformed
/// mapping structure before replacing an invalid file with a backed-up copy.
public struct OMPIntegration: Sendable {
    public static let providerID = "mtplx"
    public static let localAPIKey = "mtplx-local"
    public static let agentOperatingHintsFilename = "omp-agent-operating-hints.md"
    public static let agentOperatingHints = """
    MTPLX agent operating hints:
    - Treat tool calls and long context as expensive user-visible latency. Read only the exact files and ranges needed for the next decision.
    - Prefer focused edits and the narrowest relevant build, typecheck, or smoke check.
    - Keep final answers concise and evidence-based. Mention the files changed and checks run.
    - If a command appears stuck, stop waiting, explain it, and choose a narrower verification path.
    """

    public let configURL: URL
    public let handoffDirectory: URL
    private let executableResolver: @Sendable () -> String?

    public init(
        configURL: URL = OMPIntegration.defaultConfigURL(),
        handoffDirectory: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("handoffs", isDirectory: true),
        executableResolver: @escaping @Sendable () -> String? = {
            OMPIntegration.resolvedExecutable()
        }
    ) {
        self.configURL = configURL
        self.handoffDirectory = handoffDirectory
        self.executableResolver = executableResolver
    }

    public static func defaultConfigURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".omp", isDirectory: true)
            .appendingPathComponent("agent", isDirectory: true)
            .appendingPathComponent("models.yml")
    }

    public static func modelID(for model: String) -> String {
        OpenCodeIntegration.modelID(for: model)
    }

    public static func modelReference(for model: String) -> String {
        "\(providerID)/\(modelID(for: model))"
    }

    public static func agentOperatingHintsURL(
        homeDirectory: String = NSHomeDirectory()
    ) -> URL {
        URL(fileURLWithPath: homeDirectory)
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent(agentOperatingHintsFilename)
    }

    public static func launchCommand(
        for model: String,
        workspacePath: String = NSHomeDirectory()
    ) -> String {
        "omp --model \(modelReference(for: model)) --cwd \(shellQuote(workspacePath)) "
            + "--allow-home --append-system-prompt \(shellQuote(agentOperatingHintsURL().path))"
    }

    @discardableResult
    public func sync(configuration: MTPLXAppConfiguration) throws -> OMPConfigResult {
        let modelID = Self.modelID(for: configuration.model)
        let modelReference = Self.modelReference(for: configuration.model)
        let baseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        let apiKey = configuration.apiKey?.isEmpty == false
            ? configuration.apiKey!
            : Self.localAPIKey
        let contextWindow = configuration.effectiveContextWindow(default: 131_072)
        let workspacePath = Self.resolvedWorkspacePath(configuration: configuration)
        let fileManager = FileManager.default
        let directory = configURL.deletingLastPathComponent()
        let directoryExisted = fileManager.fileExists(atPath: directory.path)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        if !directoryExisted {
            try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        }

        let existingData: Data?
        if fileManager.fileExists(atPath: configURL.path) {
            let values = try configURL.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
            guard values.isSymbolicLink != true, values.isRegularFile == true else {
                throw OMPConfigError.unsafeConfigFile(configURL.path)
            }
            existingData = try Data(contentsOf: configURL)
        } else {
            existingData = nil
        }

        let provider = Self.providerBlock(
            modelID: modelID,
            baseURL: baseURL,
            apiKey: apiKey,
            contextWindow: contextWindow,
            reasoningEnabled: OpenCodeIntegration.reasoningEnabled(forModelID: modelID)
        )
        var invalidExisting = false
        let nextText: String
        if let existingData {
            guard let existingText = String(data: existingData, encoding: .utf8) else {
                invalidExisting = true
                nextText = Self.newDocument(providerBlock: provider)
                return try writeChangedConfig(
                    existingData: existingData,
                    nextText: nextText,
                    invalidExisting: invalidExisting,
                    baseURL: baseURL,
                    modelReference: modelReference,
                    workspacePath: workspacePath,
                    model: configuration.model
                )
            }
            do {
                nextText = try Self.mergingProviderBlock(provider, into: existingText)
            } catch let error as OMPYAMLStructureError {
                if case .tabIndentation = error {
                    // Tabs cannot indent YAML mappings, so this file is
                    // demonstrably invalid and can be recovered after backup.
                    invalidExisting = true
                    nextText = Self.newDocument(providerBlock: provider)
                } else {
                    // The narrow merger intentionally supports only the
                    // conventional block mapping OMP writes. Other shapes may
                    // still be valid YAML. Fail closed instead of replacing
                    // an active config and dropping unrelated providers.
                    throw OMPConfigError.unsupportedConfigStyle(configURL.path)
                }
            }
        } else {
            nextText = Self.newDocument(providerBlock: provider)
        }

        let nextData = Data(nextText.utf8)
        if existingData == nextData {
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
            return OMPConfigResult(
                configPath: configURL.path,
                baseURL: baseURL,
                modelReference: modelReference,
                launchCommand: Self.launchCommand(
                    for: configuration.model,
                    workspacePath: workspacePath
                ),
                didChange: false,
                backupPath: nil
            )
        }
        return try writeChangedConfig(
            existingData: existingData,
            nextText: nextText,
            invalidExisting: invalidExisting,
            baseURL: baseURL,
            modelReference: modelReference,
            workspacePath: workspacePath,
            model: configuration.model
        )
    }

    private func writeChangedConfig(
        existingData: Data?,
        nextText: String,
        invalidExisting: Bool,
        baseURL: String,
        modelReference: String,
        workspacePath: String,
        model: String
    ) throws -> OMPConfigResult {
        let fileManager = FileManager.default
        var backupURL: URL?
        if let existingData {
            let backup = uniqueBackupURL(reason: invalidExisting ? "invalid" : "bak")
            try existingData.write(to: backup, options: [.atomic])
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: backup.path)
            backupURL = backup
        }
        try Data(nextText.utf8).write(to: configURL, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
        return OMPConfigResult(
            configPath: configURL.path,
            baseURL: baseURL,
            modelReference: modelReference,
            launchCommand: Self.launchCommand(for: model, workspacePath: workspacePath),
            didChange: true,
            backupPath: backupURL?.path
        )
    }

    @MainActor
    public func launchInTerminal(
        configuration: MTPLXAppConfiguration,
        isCurrent: (() -> Bool)? = nil
    ) async -> OMPLaunchResult {
        let workspacePath = Self.resolvedWorkspacePath(configuration: configuration)
        let displayCommand = Self.launchCommand(
            for: configuration.model,
            workspacePath: workspacePath
        )
        guard let executable = executableResolver() else {
            return OMPLaunchResult(
                action: .unavailable,
                command: displayCommand,
                detail: "OMP is not installed. Install it with Homebrew or Bun, then pick OMP again."
            )
        }
        let command = Self.terminalLaunchCommand(
            for: configuration.model,
            workspacePath: workspacePath,
            executable: executable
        )
        guard isCurrent?() ?? true else {
            return staleHandoffResult(command: command)
        }
        #if os(macOS)
        let handoff = makeTerminalHandoffFiles()
        do {
            guard isCurrent?() ?? true else { return staleHandoffResult(command: command) }
            try writeTerminalCommandFile(command: command, handoff: handoff)
        } catch {
            return OMPLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not prepare OMP terminal command: \(error)"
            )
        }
        guard isCurrent?() ?? true else {
            await cancelPendingTerminalHandoff(handoff)
            return staleHandoffResult(command: command)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Terminal", handoff.commandURL.path]
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
            guard isCurrent?() ?? true else {
                await cancelPendingTerminalHandoff(handoff)
                return staleHandoffResult(command: command)
            }
            try process.run()
            guard watchdog.wait(for: process, timeout: 30) else {
                await cancelPendingTerminalHandoff(handoff)
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "could not open OMP automatically: open timed out after 30s and was terminated"
                )
            }
            guard process.terminationStatus == 0 else {
                await cancelPendingTerminalHandoff(handoff)
                let message = stderrTail.snapshot().trimmingCharacters(in: .whitespacesAndNewlines)
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: message.isEmpty
                        ? "could not open OMP automatically: open exited \(process.terminationStatus)"
                        : "could not open OMP automatically: \(message)"
                )
            }
            let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
                handoffID: handoff.handoffID,
                receiptURL: handoff.receiptURL,
                cancellationMarkerURL: handoff.cancellationMarkerURL,
                commandURL: handoff.commandURL,
                isCurrent: isCurrent
            )
            if receipt.cancellationRequested {
                if let lease = receipt.lease { _ = cancelTerminalHandoff(lease) }
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: !(isCurrent?() ?? true)
                        ? "OMP handoff cancelled because the daemon lifecycle changed."
                        : "OMP Terminal did not report its launch receipt."
                )
            }
            guard let lease = receipt.lease else {
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "OMP Terminal did not report its launch receipt."
                )
            }
            // The shell writes its receipt immediately before `exec omp`.
            // Give exec and OMP's startup validation one bounded window, then
            // require the exact handoff token to still belong to that PID.
            // A missing binary or rejected config must surface as unavailable,
            // not a permanent false "OMP running" state.
            try? await Task.sleep(nanoseconds: 350_000_000)
            guard MTPLXTerminalHandoffLease.process(
                pid: pid_t(lease.processID),
                hasExactHandoffID: lease.handoffID
            ) else {
                _ = cancelTerminalHandoff(lease)
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "OMP exited during startup. Check the OMP install and MTPLX provider config."
                )
            }
            guard isCurrent?() ?? true else {
                _ = cancelTerminalHandoff(lease)
                return OMPLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "OMP handoff cancelled because the daemon lifecycle changed.",
                    launchedProcessIDs: [lease.processID],
                    terminalHandoffLease: lease
                )
            }
            return OMPLaunchResult(
                action: .launched,
                command: command,
                detail: "opened OMP in Terminal",
                launchedProcessIDs: [lease.processID],
                terminalHandoffLease: lease
            )
        } catch {
            await cancelPendingTerminalHandoff(handoff)
            return OMPLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not open OMP automatically: \(error)"
            )
        }
        #else
        return OMPLaunchResult(
            action: .unavailable,
            command: command,
            detail: "automatic OMP launch currently requires macOS Terminal"
        )
        #endif
    }

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
        Self.terminate(pid: pid)
        return cancellationMarked && commandRemoved && receiptRemoved
    }

    public static func resolvedWorkspacePath(
        configuration: MTPLXAppConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        PiIntegration.resolvedWorkspacePath(configuration: configuration, environment: environment)
    }

    static func isOMPAgentCommand(_ command: String) -> Bool {
        let words = commandWords(command)
        guard let first = words.first else { return false }
        let executableMatch: Bool
        if isOMPExecutableToken(first) {
            executableMatch = true
        } else if isNodeExecutableToken(first), words.count >= 2 {
            executableMatch = isOMPExecutableToken(words[1]) || isOMPAgentScriptToken(words[1])
        } else {
            executableMatch = false
        }
        guard executableMatch else { return false }
        guard let modelFlag = words.firstIndex(of: "--model"), modelFlag + 1 < words.count else {
            return false
        }
        return words[modelFlag + 1].lowercased().hasPrefix("\(providerID)/")
    }

    private func staleHandoffResult(command: String) -> OMPLaunchResult {
        OMPLaunchResult(
            action: .unavailable,
            command: command,
            detail: "OMP handoff cancelled because the daemon lifecycle changed."
        )
    }

    @MainActor
    private func cancelPendingTerminalHandoff(_ handoff: TerminalHandoffFiles) async {
        let receipt = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoff.handoffID,
            receiptURL: handoff.receiptURL,
            cancellationMarkerURL: handoff.cancellationMarkerURL,
            commandURL: handoff.commandURL,
            isCurrent: { false },
            timeoutSeconds: 0,
            delayedCancellationSeconds: 1
        )
        if let lease = receipt.lease { _ = cancelTerminalHandoff(lease) }
    }

    private struct TerminalHandoffFiles: Sendable {
        let handoffID: UUID
        let commandURL: URL
        let receiptURL: URL
        let cancellationMarkerURL: URL
    }

    private func makeTerminalHandoffFiles() -> TerminalHandoffFiles {
        let handoffID = UUID()
        let suffix = handoffID.uuidString.lowercased()
        return TerminalHandoffFiles(
            handoffID: handoffID,
            commandURL: handoffDirectory.appendingPathComponent("open-omp-\(suffix).command"),
            receiptURL: handoffDirectory.appendingPathComponent("open-omp-\(suffix).pid"),
            cancellationMarkerURL: handoffDirectory.appendingPathComponent("open-omp-\(suffix).cancelled")
        )
    }

    private func writeTerminalCommandFile(
        command: String,
        handoff: TerminalHandoffFiles
    ) throws {
        try MTPLXTerminalHandoffLease.prepareArtifactDirectory(
            handoff.commandURL.deletingLastPathComponent()
        )
        _ = try Self.writeAgentOperatingHintsFile()
        let script = """
        #!/bin/zsh
        _mtplx_handoff_cancel=\(Self.shellQuote(handoff.cancellationMarkerURL.path))
        _mtplx_handoff_receipt=\(Self.shellQuote(handoff.receiptURL.path))
        export MTPLX_APP_HANDOFF_ID=\(Self.shellQuote(handoff.handoffID.uuidString.lowercased()))
        if [[ -e "$_mtplx_handoff_cancel" ]]; then
          exit 0
        fi
        umask 077
        print -r -- "$$" > "${_mtplx_handoff_receipt}.$$.tmp"
        mv -f "${_mtplx_handoff_receipt}.$$.tmp" "$_mtplx_handoff_receipt"
        if [[ -e "$_mtplx_handoff_cancel" ]]; then
          exit 0
        fi
        exec \(command)
        """
        try MTPLXTerminalHandoffLease.writeSecureCommandScript(script, to: handoff.commandURL)
    }

    @discardableResult
    private static func writeAgentOperatingHintsFile(
        homeDirectory: String = NSHomeDirectory()
    ) throws -> URL {
        let url = agentOperatingHintsURL(homeDirectory: homeDirectory)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = Data(agentOperatingHints.utf8)
        if (try? Data(contentsOf: url)) != data {
            try data.write(to: url, options: [.atomic])
        }
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
        return url
    }

    private static func terminalLaunchCommand(
        for model: String,
        workspacePath: String,
        executable: String
    ) -> String {
        "\(shellQuote(executable)) --model \(shellQuote(modelReference(for: model))) "
            + "--cwd \(shellQuote(workspacePath)) --allow-home --append-system-prompt "
            + shellQuote(agentOperatingHintsURL().path)
    }

    public static func resolvedExecutable(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        let home = NSHomeDirectory()
        var candidates = [
            "/opt/homebrew/bin/omp",
            "/usr/local/bin/omp",
            "\(home)/.local/bin/omp",
            "\(home)/.bun/bin/omp",
            "\(home)/.npm-global/bin/omp",
        ]
        candidates.append(contentsOf: (environment["PATH"] ?? "")
            .split(separator: ":")
            .map { URL(fileURLWithPath: String($0)).appendingPathComponent("omp").path })
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return nil
    }

    private static func providerBlock(
        modelID: String,
        baseURL: String,
        apiKey: String,
        contextWindow: Int,
        reasoningEnabled: Bool
    ) -> [String] {
        [
            "  mtplx:",
            "    baseUrl: \(yamlQuote(baseURL))",
            "    api: openai-completions",
            "    apiKey: \(yamlQuote(apiKey))",
            "    authHeader: true",
            "    headers:",
            "      x-mtplx-client: omp",
            "    compat:",
            "      supportsDeveloperRole: false",
            "      supportsReasoningEffort: \(reasoningEnabled ? "true" : "false")",
            "      maxTokensField: max_tokens",
            "    models:",
            "      - id: \(yamlQuote(modelID))",
            "        name: \(yamlQuote("MTPLX \(modelID)"))",
            "        reasoning: \(reasoningEnabled ? "true" : "false")",
            "        supportsTools: true",
            "        input: [text]",
            "        contextWindow: \(contextWindow)",
            "        maxTokens: 65536",
            "        omitMaxOutputTokens: true",
            "        cost:",
            "          input: 0",
            "          output: 0",
            "          cacheRead: 0",
            "          cacheWrite: 0",
        ]
    }

    private static func newDocument(providerBlock: [String]) -> String {
        (["providers:"] + providerBlock).joined(separator: "\n") + "\n"
    }

    /// Validate a conventional block-style YAML mapping and replace only the
    /// MTPLX provider. Unknown root keys and every other provider remain byte
    /// for byte identical. MTPLX provider unknown fields are retained too.
    private static func mergingProviderBlock(
        _ freshProvider: [String],
        into source: String
    ) throws -> String {
        var lines = source.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if source.hasSuffix("\n"), lines.last == "" { lines.removeLast() }
        try validateIndentation(lines)

        var rootKeys: Set<String> = []
        var providersIndex: Int?
        var index = 0
        while index < lines.count {
            let line = lines[index]
            if isTrivia(line) || line.trimmingCharacters(in: .whitespaces) == "---"
                || line.trimmingCharacters(in: .whitespaces) == "..." {
                index += 1
                continue
            }
            let info = try mappingLine(line)
            guard info.indent == 0 else { throw OMPYAMLStructureError.invalidRoot }
            guard rootKeys.insert(info.key).inserted else { throw OMPYAMLStructureError.duplicateKey }
            if info.key == "providers" {
                guard info.value.isEmpty else { throw OMPYAMLStructureError.invalidProviders }
                providersIndex = index
            }
            index = nextLine(atOrAfter: index + 1, in: lines) {
                !isTrivia($0) && indentation($0) == 0
            } ?? lines.count
        }

        guard let providersIndex else {
            if lines.isEmpty || lines.allSatisfy({ $0.isEmpty }) {
                return newDocument(providerBlock: freshProvider)
            }
            if !lines.isEmpty, lines.last != "" { lines.append("") }
            lines.append("providers:")
            lines.append(contentsOf: freshProvider)
            return lines.joined(separator: "\n") + "\n"
        }

        let providersEnd = nextLine(atOrAfter: providersIndex + 1, in: lines) {
            !isTrivia($0) && indentation($0) == 0
        } ?? lines.count
        var providerKeys: Set<String> = []
        var mtplxRange: Range<Int>?
        index = providersIndex + 1
        while index < providersEnd {
            if isTrivia(lines[index]) {
                index += 1
                continue
            }
            let info = try mappingLine(lines[index])
            guard info.indent == 2, info.value.isEmpty else {
                throw OMPYAMLStructureError.invalidProviders
            }
            guard providerKeys.insert(info.key).inserted else {
                throw OMPYAMLStructureError.duplicateKey
            }
            let end = nextLine(atOrAfter: index + 1, in: lines) {
                !isTrivia($0) && indentation($0) <= 2
            } ?? providersEnd
            if info.key == providerID { mtplxRange = index..<end }
            index = end
        }

        if let mtplxRange {
            let existing = Array(lines[mtplxRange])
            let merged = try mergeExistingMTPLXProvider(existing, fresh: freshProvider)
            lines.replaceSubrange(mtplxRange, with: merged)
        } else {
            lines.insert(contentsOf: freshProvider, at: providersEnd)
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Connection identity is MTPLX-owned. Unknown provider fields and
    /// user-added header/compat fields survive, while the model list is merged
    /// by id so custom models and custom fields on the current model survive.
    private static func mergeExistingMTPLXProvider(
        _ existing: [String],
        fresh: [String]
    ) throws -> [String] {
        var lines = existing
        _ = try directFields(in: lines, indent: 4, start: 1, end: lines.count)
        for (key, value) in [
            ("baseUrl", valueForField("baseUrl", in: fresh, indent: 4)),
            ("api", valueForField("api", in: fresh, indent: 4)),
            ("apiKey", valueForField("apiKey", in: fresh, indent: 4)),
            ("authHeader", valueForField("authHeader", in: fresh, indent: 4)),
        ] {
            try upsertScalar(key, value: value, in: &lines, indent: 4, start: 1)
        }
        try mergeMap(
            "headers",
            values: ["x-mtplx-client": "omp"],
            in: &lines,
            parentIndent: 4,
            start: 1
        )
        let reasoning = valueForNestedField(
            "supportsReasoningEffort", parent: "compat", in: fresh
        )
        try mergeMap(
            "compat",
            values: [
                "supportsDeveloperRole": "false",
                "supportsReasoningEffort": reasoning,
                "maxTokensField": "max_tokens",
            ],
            in: &lines,
            parentIndent: 4,
            start: 1
        )
        try mergeModels(fresh: fresh, into: &lines)
        return lines
    }

    private static func mergeMap(
        _ key: String,
        values: [String: String],
        in lines: inout [String],
        parentIndent: Int,
        start: Int
    ) throws {
        let fields = try directFields(in: lines, indent: parentIndent, start: start, end: lines.count)
        if let field = fields[key] {
            guard field.value.isEmpty else { throw OMPYAMLStructureError.invalidProviders }
            for nestedKey in values.keys.sorted() {
                try upsertScalar(
                    nestedKey,
                    value: values[nestedKey]!,
                    in: &lines,
                    indent: parentIndent + 2,
                    start: field.start + 1,
                    boundaryIndent: parentIndent
                )
            }
        } else {
            lines.append(String(repeating: " ", count: parentIndent) + "\(key):")
            for nestedKey in values.keys.sorted() {
                lines.append(
                    String(repeating: " ", count: parentIndent + 2)
                        + "\(nestedKey): \(values[nestedKey]!)"
                )
            }
        }
    }

    private static func mergeModels(fresh: [String], into lines: inout [String]) throws {
        let fields = try directFields(in: lines, indent: 4, start: 1, end: lines.count)
        guard let models = fields["models"] else {
            lines.append(contentsOf: freshModelLines(from: fresh))
            return
        }
        guard models.value.isEmpty else { throw OMPYAMLStructureError.invalidProviders }
        let modelID = scalarValue(valueForModelField("id", in: fresh))
        var index = models.start + 1
        var matchingRange: Range<Int>?
        while index < models.end {
            if isTrivia(lines[index]) {
                index += 1
                continue
            }
            let indent = indentation(lines[index])
            guard indent == 6,
                  stripComment(String(lines[index].dropFirst(6))).hasPrefix("-")
            else { throw OMPYAMLStructureError.invalidModels }
            let end = nextLine(atOrAfter: index + 1, in: lines) {
                !isTrivia($0) && indentation($0) <= 6
            } ?? models.end
            if modelIDInItem(Array(lines[index..<end])) == modelID {
                matchingRange = index..<end
                break
            }
            index = end
        }
        guard let matchingRange else {
            lines.insert(contentsOf: Array(freshModelLines(from: fresh).dropFirst()), at: models.end)
            return
        }
        var item = Array(lines[matchingRange])
        // These fields describe the currently loaded MTPLX model, so a
        // restart must refresh stale values. Unknown fields remain untouched.
        for key in [
            "name", "reasoning", "supportsTools", "input", "contextWindow", "maxTokens",
            "omitMaxOutputTokens",
        ] {
            try upsertScalar(
                key,
                value: key == "omitMaxOutputTokens" ? "true" : valueForModelField(key, in: fresh),
                in: &item,
                indent: 8,
                start: 1
            )
        }
        try mergeMap(
            "cost",
            values: [
                "input": "0",
                "output": "0",
                "cacheRead": "0",
                "cacheWrite": "0",
            ],
            in: &item,
            parentIndent: 8,
            start: 1
        )
        lines.replaceSubrange(matchingRange, with: item)
    }

    private struct YAMLField {
        let start: Int
        let end: Int
        let value: String
    }

    private static func directFields(
        in lines: [String],
        indent: Int,
        start: Int,
        end: Int
    ) throws -> [String: YAMLField] {
        var result: [String: YAMLField] = [:]
        var index = start
        while index < end {
            if isTrivia(lines[index]) {
                index += 1
                continue
            }
            let lineIndent = indentation(lines[index])
            if lineIndent < indent { break }
            guard lineIndent == indent else {
                throw OMPYAMLStructureError.invalidProviders
            }
            let info = try mappingLine(lines[index])
            let fieldEnd = nextLine(atOrAfter: index + 1, in: lines) {
                !isTrivia($0) && indentation($0) <= indent
            } ?? end
            guard result[info.key] == nil else { throw OMPYAMLStructureError.duplicateKey }
            result[info.key] = YAMLField(start: index, end: fieldEnd, value: info.value)
            index = fieldEnd
        }
        return result
    }

    private static func upsertScalar(
        _ key: String,
        value: String,
        in lines: inout [String],
        indent: Int,
        start: Int,
        boundaryIndent: Int? = nil
    ) throws {
        let boundary = nextLine(atOrAfter: start, in: lines) {
            !isTrivia($0) && indentation($0) <= (boundaryIndent ?? (indent - 2))
        } ?? lines.count
        let fields = try directFields(in: lines, indent: indent, start: start, end: boundary)
        let replacement = String(repeating: " ", count: indent) + "\(key): \(value)"
        if let field = fields[key] {
            guard !field.value.isEmpty, field.end == field.start + 1 else {
                throw OMPYAMLStructureError.invalidProviders
            }
            lines[field.start] = replacement
        } else {
            lines.insert(replacement, at: boundary)
        }
    }

    private static func freshModelLines(from fresh: [String]) -> [String] {
        guard let modelsIndex = fresh.firstIndex(of: "    models:") else { return [] }
        return Array(fresh[modelsIndex...])
    }

    private static func modelIDInItem(_ item: [String]) -> String? {
        guard let first = item.first else { return nil }
        let firstContent = stripComment(String(first.dropFirst(6)))
            .trimmingCharacters(in: .whitespaces)
        if firstContent.hasPrefix("- id:") {
            return scalarValue(String(firstContent.dropFirst(5)))
        }
        for line in item.dropFirst() {
            guard indentation(line) == 8,
                  let info = try? mappingLine(line), info.key == "id"
            else { continue }
            return scalarValue(info.value)
        }
        return nil
    }

    private static func valueForField(_ key: String, in lines: [String], indent: Int) -> String {
        for line in lines where indentation(line) == indent {
            guard let info = try? mappingLine(line), info.key == key else { continue }
            return info.value
        }
        preconditionFailure("missing generated OMP field \(key)")
    }

    private static func valueForNestedField(_ key: String, parent: String, in lines: [String]) -> String {
        guard let parentIndex = lines.firstIndex(of: "    \(parent):") else {
            preconditionFailure("missing generated OMP map \(parent)")
        }
        for line in lines[(parentIndex + 1)...] {
            let indent = indentation(line)
            if indent <= 4 { break }
            guard indent == 6, let info = try? mappingLine(line), info.key == key else { continue }
            return info.value
        }
        preconditionFailure("missing generated OMP field \(parent).\(key)")
    }

    private static func valueForModelField(_ key: String, in lines: [String]) -> String {
        for line in lines {
            guard indentation(line) == 8, let info = try? mappingLine(line), info.key == key else {
                continue
            }
            return info.value
        }
        if key == "id", let line = lines.first(where: { $0.hasPrefix("      - id:") }) {
            return String(line.dropFirst("      - id:".count)).trimmingCharacters(in: .whitespaces)
        }
        preconditionFailure("missing generated OMP model field \(key)")
    }

    private struct MappingInfo {
        let indent: Int
        let key: String
        let value: String
    }

    private static func mappingLine(_ line: String) throws -> MappingInfo {
        let indent = indentation(line)
        let content = stripComment(String(line.dropFirst(indent)))
            .trimmingCharacters(in: .whitespaces)
        guard let colon = yamlColon(in: content) else { throw OMPYAMLStructureError.invalidMapping }
        let rawKey = String(content[..<colon]).trimmingCharacters(in: .whitespaces)
        let key = scalarValue(rawKey)
        guard !key.isEmpty,
              key.unicodeScalars.allSatisfy({
                  CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-.")).contains($0)
              })
        else { throw OMPYAMLStructureError.invalidMapping }
        let value = String(content[content.index(after: colon)...])
            .trimmingCharacters(in: .whitespaces)
        return MappingInfo(indent: indent, key: key, value: value)
    }

    private static func yamlColon(in text: String) -> String.Index? {
        var singleQuoted = false
        var doubleQuoted = false
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character == "\\", doubleQuoted {
                let next = text.index(after: index)
                index = next < text.endIndex ? text.index(after: next) : next
                continue
            }
            if character == "'", !doubleQuoted {
                let next = text.index(after: index)
                if singleQuoted, next < text.endIndex, text[next] == "'" {
                    index = text.index(after: next)
                    continue
                }
                singleQuoted.toggle()
            } else if character == "\"", !singleQuoted {
                doubleQuoted.toggle()
            } else if character == ":", !singleQuoted, !doubleQuoted {
                return index
            }
            index = text.index(after: index)
        }
        return nil
    }

    private static func stripComment(_ text: String) -> String {
        var singleQuoted = false
        var doubleQuoted = false
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if character == "\\", doubleQuoted {
                let next = text.index(after: index)
                index = next < text.endIndex ? text.index(after: next) : next
                continue
            }
            if character == "'", !doubleQuoted {
                let next = text.index(after: index)
                if singleQuoted, next < text.endIndex, text[next] == "'" {
                    index = text.index(after: next)
                    continue
                }
                singleQuoted.toggle()
            } else if character == "\"", !singleQuoted {
                doubleQuoted.toggle()
            } else if character == "#", !singleQuoted, !doubleQuoted {
                return String(text[..<index])
            }
            index = text.index(after: index)
        }
        return text
    }

    private static func scalarValue(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        guard trimmed.count >= 2, let first = trimmed.first, let last = trimmed.last,
              first == last, first == "'" || first == "\""
        else { return trimmed }
        var result = String(trimmed.dropFirst().dropLast())
        if first == "'" { result = result.replacingOccurrences(of: "''", with: "'") }
        return result
    }

    private static func validateIndentation(_ lines: [String]) throws {
        for line in lines {
            let prefix = line.prefix { $0 == " " || $0 == "\t" }
            if prefix.contains("\t") { throw OMPYAMLStructureError.tabIndentation }
        }
    }

    private static func indentation(_ line: String) -> Int {
        line.prefix { $0 == " " }.count
    }

    private static func isTrivia(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty || trimmed.hasPrefix("#")
    }

    private static func nextLine(
        atOrAfter start: Int,
        in lines: [String],
        matching predicate: (String) -> Bool
    ) -> Int? {
        guard start < lines.count else { return nil }
        return (start..<lines.count).first { predicate(lines[$0]) }
    }

    private static func yamlQuote(_ value: String) -> String {
        var quoted = "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x22:
                quoted += "\\\""
            case 0x5c:
                quoted += "\\\\"
            case 0x08:
                quoted += "\\b"
            case 0x09:
                quoted += "\\t"
            case 0x0a:
                quoted += "\\n"
            case 0x0c:
                quoted += "\\f"
            case 0x0d:
                quoted += "\\r"
            case 0x00...0x1f, 0x7f:
                quoted += String(format: "\\u%04X", scalar.value)
            default:
                quoted.unicodeScalars.append(scalar)
            }
        }
        return quoted + "\""
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    private static func commandWords(_ command: String) -> [String] {
        command.split(whereSeparator: { $0.isWhitespace })
            .map { stripShellTokenQuotes(String($0)) }
            .filter { !$0.isEmpty }
    }

    private static func stripShellTokenQuotes(_ token: String) -> String {
        var value = token.trimmingCharacters(in: .whitespacesAndNewlines)
        while value.count >= 2, let first = value.first, let last = value.last,
              (first == "'" || first == "\""), first == last {
            value.removeFirst()
            value.removeLast()
        }
        return value
    }

    private static func isOMPExecutableToken(_ token: String) -> Bool {
        URL(fileURLWithPath: token).lastPathComponent == "omp"
    }

    private static func isNodeExecutableToken(_ token: String) -> Bool {
        URL(fileURLWithPath: token).lastPathComponent == "node"
    }

    private static func isOMPAgentScriptToken(_ token: String) -> Bool {
        let normalized = token.replacingOccurrences(of: "\\", with: "/")
        return normalized.contains("@oh-my-pi/pi-coding-agent/")
            && URL(fileURLWithPath: normalized).lastPathComponent == "cli.js"
    }

    private static func terminate(pid: pid_t) {
        guard kill(pid, 0) == 0 else { return }
        _ = kill(pid, SIGTERM)
        for _ in 0..<20 {
            if kill(pid, 0) != 0 { return }
            Thread.sleep(forTimeInterval: 0.05)
        }
        _ = kill(pid, SIGKILL)
    }

    private func uniqueBackupURL(reason: String) -> URL {
        let directory = configURL.deletingLastPathComponent()
        let timestamp = Self.timestamp()
        let basename = configURL.lastPathComponent
        var candidate = directory.appendingPathComponent("\(basename).\(reason)-\(timestamp).bak")
        var index = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = directory.appendingPathComponent(
                "\(basename).\(reason)-\(timestamp)-\(index).bak"
            )
            index += 1
        }
        return candidate
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }
}

private enum OMPYAMLStructureError: Error {
    case tabIndentation
    case invalidRoot
    case invalidProviders
    case invalidModels
    case invalidMapping
    case duplicateKey
}

private enum OMPConfigError: LocalizedError {
    case unsafeConfigFile(String)
    case unsupportedConfigStyle(String)

    var errorDescription: String? {
        switch self {
        case .unsafeConfigFile(let path):
            return "refusing to replace non-regular or symbolic OMP config at \(path)"
        case .unsupportedConfigStyle(let path):
            return "refusing to rewrite unsupported OMP YAML style at \(path)"
        }
    }
}
