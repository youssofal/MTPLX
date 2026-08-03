import Foundation
#if canImport(AppKit)
import AppKit
#endif

public enum OpenCodeIntegrationError: Error, Equatable {
    case invalidTopLevelConfig(String)
    case desktopAppNotFound(String)
    case desktopRelaunchFailed(String)
}

public struct OpenCodeConfigResult: Equatable, Sendable {
    public let configPath: String
    public let baseURL: String
    public let modelReference: String
    public let legacySessionHeadersPluginPath: String
    public let didChange: Bool
    public let backupPath: String?
    public let reasoningVisibilityPath: String
    public let reasoningVisibilityDidChange: Bool
    public let reasoningVisibilityBackupPath: String?
}

public struct OpenCodeDesktopStateRepairResult: Equatable, Sendable {
    public let path: String
    public let didChange: Bool
    public let backupPath: String?
    public let removedEntries: Int
    public let missingPaths: [String]
    public let status: String
}

public enum OpenCodeDesktopAction: String, Equatable, Sendable {
    case unavailable
    case opened
    case relaunched
    case focused
}

public struct OpenCodeDesktopResult: Equatable, Sendable {
    public let action: OpenCodeDesktopAction
    public let wasRunning: Bool
    public let didTerminateExistingInstance: Bool
    public let didOpen: Bool
    public let detail: String
}

private struct OpenCodeReasoningVisibilityResult: Equatable, Sendable {
    let path: String
    let didChange: Bool
    let backupPath: String?
}

public struct OpenCodeIntegration: Sendable {
    private static let desktopGlobalStoreName = "opencode.global.dat"
    private static let sessionHeadersPluginName = "mtplx-session-headers.js"

    public let configURL: URL
    public let desktopSettingsStoreURL: URL
    public let desktopBundleIdentifier: String
    public let desktopApplicationURL: URL

    public init(
        configURL: URL = OpenCodeIntegration.defaultConfigURL(),
        desktopSettingsStoreURL: URL = OpenCodeIntegration.defaultDesktopSettingsStoreURL(),
        desktopBundleIdentifier: String = "ai.opencode.desktop",
        desktopApplicationURL: URL = URL(fileURLWithPath: "/Applications/OpenCode.app")
    ) {
        self.configURL = configURL
        self.desktopSettingsStoreURL = desktopSettingsStoreURL
        self.desktopBundleIdentifier = desktopBundleIdentifier
        self.desktopApplicationURL = desktopApplicationURL
    }

    public static func defaultConfigURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".config")
            .appendingPathComponent("opencode")
            .appendingPathComponent("opencode.json")
    }

    public static func defaultDesktopSettingsStoreURL() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library")
            .appendingPathComponent("Application Support")
            .appendingPathComponent("ai.opencode.desktop")
            .appendingPathComponent("default.dat")
    }

    @discardableResult
    public func sync(configuration: MTPLXAppConfiguration) throws -> OpenCodeConfigResult {
        let modelID = Self.modelID(for: configuration.model)
        let modelReference = "mtplx/\(modelID)"
        let baseURL = Self.baseURLString(host: configuration.host, port: configuration.port)
        let contextLimit = configuration.effectiveContextWindow(default: 262_144)
        let sessionHeadersPluginURL = configURL.deletingLastPathComponent()
            .appendingPathComponent(Self.sessionHeadersPluginName)
        var backupURL: URL?

        var root = try loadRoot()
        var providers = root["provider"]?.objectValue ?? [:]
        providers["mtplx"] = .object(
            Self.providerConfig(
                modelID: modelID,
                baseURL: baseURL,
                apiKey: configuration.apiKey,
                contextLimit: contextLimit
            )
        )
        root["provider"] = .object(providers)
        root["model"] = .string(modelReference)
        root["small_model"] = .string(modelReference)
        _ = Self.removeManagedSessionHeadersPlugin(from: &root)
        if root["$schema"] == nil {
            root["$schema"] = .string("https://opencode.ai/config.json")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let nextData = try encoder.encode(root)

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: configURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let legacyPluginFileDidChange = try Self.removeLegacySessionHeadersPluginFileIfPresent(
            at: sessionHeadersPluginURL
        )

        let existingData = try? Data(contentsOf: configURL)
        if existingData == nextData {
            let visibility = try ensureReasoningSummariesVisible()
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
            return OpenCodeConfigResult(
                configPath: configURL.path,
                baseURL: baseURL,
                modelReference: modelReference,
                legacySessionHeadersPluginPath: sessionHeadersPluginURL.path,
                didChange: legacyPluginFileDidChange || visibility.didChange,
                backupPath: nil,
                reasoningVisibilityPath: visibility.path,
                reasoningVisibilityDidChange: visibility.didChange,
                reasoningVisibilityBackupPath: visibility.backupPath
            )
        }

        if existingData != nil {
            let backup = uniqueBackupURL(reason: "bak")
            try fileManager.copyItem(at: configURL, to: backup)
            backupURL = backup
        }
        try nextData.write(to: configURL, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)

        let visibility = try ensureReasoningSummariesVisible()
        return OpenCodeConfigResult(
            configPath: configURL.path,
            baseURL: baseURL,
            modelReference: modelReference,
            legacySessionHeadersPluginPath: sessionHeadersPluginURL.path,
            didChange: true,
            backupPath: backupURL?.path,
            reasoningVisibilityPath: visibility.path,
            reasoningVisibilityDidChange: visibility.didChange,
            reasoningVisibilityBackupPath: visibility.backupPath
        )
    }

    /// OpenCode Desktop keeps provider configuration inside its sidecar
    /// process. Updating `~/.config/opencode/opencode.json` is necessary,
    /// but an already-running Desktop instance can keep using a stale
    /// baseURL until its sidecar restarts. The MTPLX app's OpenCode launch
    /// target therefore owns this handoff: after the daemon is ready, reload
    /// Desktop so users do not have to discover the "restart OpenCode" fix.
    @MainActor
    public func reloadDesktopAfterDaemonReady() async -> OpenCodeDesktopResult {
        #if canImport(AppKit)
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: desktopApplicationURL.path) else {
            return OpenCodeDesktopResult(
                action: .unavailable,
                wasRunning: false,
                didTerminateExistingInstance: false,
                didOpen: false,
                detail: "OpenCode.app not found at \(desktopApplicationURL.path)"
            )
        }
        let stateRepair = repairDesktopStateBeforeLaunch()

        let running = NSRunningApplication
            .runningApplications(withBundleIdentifier: desktopBundleIdentifier)
            .filter { !$0.isTerminated }
        let wasRunning = !running.isEmpty
        var terminatedExisting = false

        if wasRunning {
            for app in running {
                app.terminate()
            }
            terminatedExisting = await waitUntilApplicationsExit(running, timeoutSeconds: 5)
            if !terminatedExisting {
                for app in running where !app.isTerminated {
                    app.forceTerminate()
                }
                terminatedExisting = await waitUntilApplicationsExit(running, timeoutSeconds: 2)
            }
        }

        let didOpen = await openDesktopApplication()
        let action: OpenCodeDesktopAction
        if wasRunning {
            action = didOpen ? .relaunched : .unavailable
        } else {
            action = didOpen ? .opened : .unavailable
        }

        return OpenCodeDesktopResult(
            action: action,
            wasRunning: wasRunning,
            didTerminateExistingInstance: wasRunning ? terminatedExisting : false,
            didOpen: didOpen,
            detail: (wasRunning
                ? "reloaded OpenCode Desktop so its sidecar re-reads MTPLX provider config"
                : "opened OpenCode Desktop")
                + (stateRepair.didChange
                   ? "; repaired \(stateRepair.removedEntries) stale OpenCode workspace state entr\(stateRepair.removedEntries == 1 ? "y" : "ies")"
                   : "")
        )
        #else
        return OpenCodeDesktopResult(
            action: .unavailable,
            wasRunning: false,
            didTerminateExistingInstance: false,
            didOpen: false,
            detail: "AppKit is unavailable"
        )
        #endif
    }

    public static func modelID(for model: String) -> String {
        let lower = model.lowercased()
        if lower.contains("gemma4") || lower.contains("gemma-4") {
            return "gemma4-mtplx-optimized-speed"
        }
        if lower.contains("stepfun")
            || lower.contains("step3p5")
            || lower.contains("step3.7")
            || lower.contains("step-3.7")
            || lower.contains("step-3-7")
        {
            return "step-3.7-flash-mtplx-step3p5"
        }
        if lower.contains("qwen3.6-35b-a3b")
            || lower.contains("qwen36-35b-a3b")
            || lower.contains("qwen3-6-35b-a3b")
        {
            return "mtplx-qwen36-35b-a3b-optimized-speed"
        }
        if lower.contains("qwen3.5-4b")
            || lower.contains("qwen35-4b")
            || lower.contains("qwen3-5-4b")
        {
            return "qwen3.5-4b-mtplx-optimized-speed"
        }
        if lower.contains("qwen") && lower.contains("optimized-speed-v2") {
            return "mtplx-qwen36-27b-optimized-speed-v2"
        }
        if lower.contains("qwen") && lower.contains("optimized-speed-fp16") {
            return "mtplx-qwen36-27b-optimized-speed-fp16"
        }
        if lower.contains("qwen") && lower.contains("optimized-speed") {
            return "mtplx-qwen36-27b-optimized-speed"
        }
        if lower.contains("qwen") && lower.contains("optimized-quality-fp16") {
            return "mtplx-qwen36-27b-optimized-quality-fp16"
        }
        if lower.contains("qwen") && lower.contains("optimized-quality") {
            return "mtplx-qwen36-27b-optimized-quality"
        }
        if lower.contains("gdn8-speed4") {
            return "mtplx-qwen36-27b-gdn8-speed4"
        }

        let lastComponent = URL(fileURLWithPath: model).lastPathComponent
        let seed = lastComponent.isEmpty ? model : lastComponent
        let sanitized = seed
            .lowercased()
            .map { character -> Character in
                if character.isLetter || character.isNumber || character == "-" || character == "_" {
                    return character
                }
                return "-"
            }
        let collapsed = String(sanitized)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        return collapsed.isEmpty ? "mtplx-local-model" : collapsed
    }

    public static func baseURLString(host: String, port: Int) -> String {
        // Shared bind->connect resolution (MTPLXServerURLs is the single
        // Swift twin of mtplx/server_urls.py).
        MTPLXServerURLs.baseURL(bindHost: host, port: port)
            .absoluteString + "/v1"
    }

    public static func samplerTopK(forModelID modelID: String) -> Int {
        let lower = modelID.lowercased()
        if lower.contains("gemma4") || lower.contains("gemma-4") {
            return 64
        }
        return 20
    }

    public static func samplerTemperature(forModelID modelID: String) -> Double {
        let lower = modelID.lowercased()
        if lower.contains("gemma4") || lower.contains("gemma-4") {
            return 1.0
        }
        return 0.6
    }

    public static func samplerTopP(forModelID modelID: String) -> Double {
        let lower = modelID.lowercased()
        if lower.contains("gemma4") || lower.contains("gemma-4") {
            return 0.95
        }
        if lower.contains("step") {
            return 0.95
        }
        return 1.0
    }

    public static func reasoningEnabled(forModelID modelID: String) -> Bool {
        _ = modelID
        return true
    }

    public static func reasoningEffort(forModelID modelID: String) -> String? {
        modelID.lowercased().contains("step") ? "low" : nil
    }

    public func repairDesktopStateBeforeLaunch() -> OpenCodeDesktopStateRepairResult {
        let globalStoreURL = desktopSettingsStoreURL
            .deletingLastPathComponent()
            .appendingPathComponent(Self.desktopGlobalStoreName)
        return Self.repairDeadWorkspaceState(globalStoreURL: globalStoreURL)
    }

    private func loadRoot() throws -> [String: JSONValue] {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: configURL.path) else {
            return [:]
        }
        let data = try Data(contentsOf: configURL)
        guard !data.isEmpty else {
            return [:]
        }
        do {
            return try JSONDecoder().decode([String: JSONValue].self, from: data)
        } catch {
            let backup = uniqueBackupURL(reason: "invalid")
            try fileManager.moveItem(at: configURL, to: backup)
            return [:]
        }
    }

    private static func removeManagedSessionHeadersPlugin(from root: inout [String: JSONValue]) -> Bool {
        guard let current = root["plugin"] else { return false }
        if let plugins = current.arrayValue {
            let remaining = plugins.filter { plugin in
                guard let path = plugin.stringValue else { return true }
                return URL(fileURLWithPath: path).lastPathComponent != sessionHeadersPluginName
            }
            let didChange = remaining.count != plugins.count
            if remaining.isEmpty {
                root["plugin"] = nil
            } else {
                root["plugin"] = .array(remaining)
            }
            return didChange
        }
        if let path = current.stringValue,
           URL(fileURLWithPath: path).lastPathComponent == sessionHeadersPluginName {
            root["plugin"] = nil
            return true
        }
        return false
    }

    private static func removeLegacySessionHeadersPluginFileIfPresent(at url: URL) throws -> Bool {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return false
        }
        try FileManager.default.removeItem(at: url)
        return true
    }

    public static func repairDeadWorkspaceState(
        globalStoreURL: URL,
        fileManager: FileManager = .default
    ) -> OpenCodeDesktopStateRepairResult {
        guard fileManager.fileExists(atPath: globalStoreURL.path) else {
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: false,
                backupPath: nil,
                removedEntries: 0,
                missingPaths: [],
                status: "missing_store"
            )
        }
        guard
            let data = try? Data(contentsOf: globalStoreURL),
            let rootObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: false,
                backupPath: nil,
                removedEntries: 0,
                missingPaths: [],
                status: "unreadable_store"
            )
        }

        var root = rootObject
        let missingPaths = missingWorkspacePaths(in: root, fileManager: fileManager)
        guard !missingPaths.isEmpty else {
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: false,
                backupPath: nil,
                removedEntries: 0,
                missingPaths: [],
                status: "clean"
            )
        }

        var removedEntries = 0
        if let layoutText = root["layout"] as? String,
           var layout = jsonObject(from: layoutText) {
            for sectionName in ["sessionTabs", "sessionView"] {
                guard let section = layout[sectionName] as? [String: Any] else { continue }
                var next: [String: Any] = [:]
                for (key, value) in section {
                    if let decodedPath = projectPath(fromOpenCodeKey: key),
                       missingPaths.contains(decodedPath) {
                        removedEntries += 1
                        continue
                    }
                    next[key] = value
                }
                layout[sectionName] = next
            }
            root["layout"] = jsonString(from: layout)
        }

        if let pageText = root["layout.page"] as? String,
           var page = jsonObject(from: pageText) {
            if let sessions = page["lastProjectSession"] as? [String: Any] {
                let next = sessions.filter { !missingPaths.contains($0.key) }
                removedEntries += sessions.count - next.count
                page["lastProjectSession"] = next
            }
            for mapName in ["workspaceOrder", "workspaceName", "workspaceBranchName", "workspaceExpanded"] {
                guard let map = page[mapName] as? [String: Any] else { continue }
                let next = map.filter { !missingPaths.contains($0.key) }
                removedEntries += map.count - next.count
                page[mapName] = next
            }
            root["layout.page"] = jsonString(from: page)
        }

        if let serverText = root["server"] as? String,
           var server = jsonObject(from: serverText),
           var projects = server["projects"] as? [String: Any] {
            for (group, entries) in projects {
                guard let rows = entries as? [[String: Any]] else { continue }
                let next = rows.filter { row in
                    guard let worktree = row["worktree"] as? String else { return true }
                    return !missingPaths.contains(worktree)
                }
                removedEntries += rows.count - next.count
                projects[group] = next
            }
            server["projects"] = projects
            root["server"] = jsonString(from: server)
        }

        guard removedEntries > 0,
              let nextData = try? JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted])
        else {
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: false,
                backupPath: nil,
                removedEntries: 0,
                missingPaths: Array(missingPaths).sorted(),
                status: "no_matching_entries"
            )
        }

        do {
            let backupURL = uniqueBackupURL(for: globalStoreURL, reason: "dead-workspaces")
            try fileManager.copyItem(at: globalStoreURL, to: backupURL)
            try nextData.write(to: globalStoreURL, options: [.atomic])
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: globalStoreURL.path)
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: true,
                backupPath: backupURL.path,
                removedEntries: removedEntries,
                missingPaths: Array(missingPaths).sorted(),
                status: "repaired"
            )
        } catch {
            return OpenCodeDesktopStateRepairResult(
                path: globalStoreURL.path,
                didChange: false,
                backupPath: nil,
                removedEntries: 0,
                missingPaths: Array(missingPaths).sorted(),
                status: "write_failed"
            )
        }
    }

    private static func missingWorkspacePaths(
        in root: [String: Any],
        fileManager: FileManager
    ) -> Set<String> {
        var candidates = Set<String>()
        if let layoutText = root["layout"] as? String,
           let layout = jsonObject(from: layoutText) {
            for sectionName in ["sessionTabs", "sessionView"] {
                guard let section = layout[sectionName] as? [String: Any] else { continue }
                for key in section.keys {
                    if let decodedPath = projectPath(fromOpenCodeKey: key) {
                        candidates.insert(decodedPath)
                    }
                }
            }
        }
        if let pageText = root["layout.page"] as? String,
           let page = jsonObject(from: pageText) {
            if let sessions = page["lastProjectSession"] as? [String: Any] {
                candidates.formUnion(sessions.keys)
            }
            for mapName in ["workspaceOrder", "workspaceName", "workspaceBranchName", "workspaceExpanded"] {
                if let map = page[mapName] as? [String: Any] {
                    candidates.formUnion(map.keys)
                }
            }
        }
        if let serverText = root["server"] as? String,
           let server = jsonObject(from: serverText),
           let projects = server["projects"] as? [String: Any] {
            for entries in projects.values {
                guard let rows = entries as? [[String: Any]] else { continue }
                for row in rows {
                    if let worktree = row["worktree"] as? String {
                        candidates.insert(worktree)
                    }
                }
            }
        }
        return Set(candidates.filter { path in
            path.hasPrefix("/") && !fileManager.fileExists(atPath: path)
        })
    }

    private static func projectPath(fromOpenCodeKey key: String) -> String? {
        let projectKey = String(key.split(separator: "/", maxSplits: 1).first ?? "")
        guard !projectKey.isEmpty else { return nil }
        let base64 = projectKey
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let remainder = projectKey.count % 4
        let padded = remainder == 0
            ? base64
            : base64 + String(repeating: "=", count: 4 - remainder)
        guard let data = Data(base64Encoded: padded),
              let decoded = String(data: data, encoding: .utf8),
              decoded.hasPrefix("/")
        else {
            return nil
        }
        return decoded
    }

    private static func jsonObject(from text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func jsonString(from object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: []),
              let text = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return text
    }

    private static func providerConfig(
        modelID: String,
        baseURL: String,
        apiKey: String?,
        contextLimit: Int
    ) -> [String: JSONValue] {
        var options: [String: JSONValue] = [
            "baseURL": .string(baseURL),
            "timeout": .bool(false),
            "chunkTimeout": .number(900_000),
            "headers": .object([
                "x-mtplx-client": .string("opencode")
            ]),
        ]
        if let apiKey, !apiKey.isEmpty {
            options["apiKey"] = .string(apiKey)
        }

        return [
            "npm": .string("@ai-sdk/openai-compatible"),
            "name": .string("MTPLX (local)"),
            "options": .object(options),
            "models": .object([
                modelID: .object([
                    "name": .string("MTPLX \(modelID)"),
                    "reasoning": .bool(false),
                    "tool_call": .bool(true),
                    "temperature": .bool(false),
                    "limit": .object([
                        "context": .number(Double(contextLimit)),
                        "output": .number(Double(contextLimit)),
                    ]),
                    "modalities": .object([
                        "input": .array([.string("text")]),
                        "output": .array([.string("text")]),
                    ]),
                ]),
            ]),
        ]
    }

    private func uniqueBackupURL(reason: String) -> URL {
        uniqueBackupURL(for: configURL, reason: reason)
    }

    private func uniqueBackupURL(for url: URL, reason: String) -> URL {
        Self.uniqueBackupURL(for: url, reason: reason)
    }

    private static func uniqueBackupURL(for url: URL, reason: String) -> URL {
        let directory = url.deletingLastPathComponent()
        let timestamp = Self.timestamp()
        let basename = url.lastPathComponent
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

    private func ensureReasoningSummariesVisible() throws -> OpenCodeReasoningVisibilityResult {
        let fileManager = FileManager.default
        var root: [String: JSONValue] = [:]
        var backupURL: URL?
        let existingData = try? Data(contentsOf: desktopSettingsStoreURL)

        if let existingData, !existingData.isEmpty {
            do {
                root = try JSONDecoder().decode([String: JSONValue].self, from: existingData)
            } catch {
                let backup = uniqueBackupURL(for: desktopSettingsStoreURL, reason: "invalid")
                try fileManager.createDirectory(
                    at: desktopSettingsStoreURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try fileManager.copyItem(at: desktopSettingsStoreURL, to: backup)
                backupURL = backup
                root = [:]
            }
        }

        var settings: [String: JSONValue] = [:]
        if let raw = root["settings.v3"]?.stringValue,
           let data = raw.data(using: .utf8),
           let parsed = try? JSONDecoder().decode([String: JSONValue].self, from: data) {
            settings = parsed
        } else if let object = root["settings.v3"]?.objectValue {
            settings = object
        }

        var general = settings["general"]?.objectValue ?? [:]
        if general["showReasoningSummaries"]?.boolValue == true {
            return OpenCodeReasoningVisibilityResult(
                path: desktopSettingsStoreURL.path,
                didChange: false,
                backupPath: nil
            )
        }

        general["showReasoningSummaries"] = .bool(true)
        settings["general"] = .object(general)

        let settingsEncoder = JSONEncoder()
        settingsEncoder.outputFormatting = [.sortedKeys]
        let settingsData = try settingsEncoder.encode(settings)
        root["settings.v3"] = .string(String(decoding: settingsData, as: UTF8.self))

        let rootEncoder = JSONEncoder()
        rootEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let nextData = try rootEncoder.encode(root)

        try fileManager.createDirectory(
            at: desktopSettingsStoreURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if existingData != nil, backupURL == nil {
            let backup = uniqueBackupURL(for: desktopSettingsStoreURL, reason: "reasoning-visible")
            try fileManager.copyItem(at: desktopSettingsStoreURL, to: backup)
            backupURL = backup
        }
        try nextData.write(to: desktopSettingsStoreURL, options: [.atomic])
        try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: desktopSettingsStoreURL.path)

        return OpenCodeReasoningVisibilityResult(
            path: desktopSettingsStoreURL.path,
            didChange: true,
            backupPath: backupURL?.path
        )
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    #if canImport(AppKit)
    @MainActor
    private func waitUntilApplicationsExit(
        _ applications: [NSRunningApplication],
        timeoutSeconds: TimeInterval
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if applications.allSatisfy(\.isTerminated) {
                return true
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        return applications.allSatisfy(\.isTerminated)
    }

    @MainActor
    private func openDesktopApplication() async -> Bool {
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true

        return await withCheckedContinuation { continuation in
            NSWorkspace.shared.openApplication(
                at: desktopApplicationURL,
                configuration: configuration
            ) { _, error in
                continuation.resume(returning: error == nil)
            }
        }
    }
    #endif
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { value } else { nil }
    }
}
