import Foundation

/// Where each agent client opens: a terminal app the user picks, or an app.
///
/// Swift twin of `mtplx/client_opening.py`. One setting per client, shared
/// with the CLI through `~/.mtplx/config.toml` (`pi_opens_in = "Ghostty"`,
/// `codex_opens_in = "mtplx-app"`). A terminal is any installed app that runs
/// a `.command` script, found from what each app declares in its Info.plist,
/// never from a list of names. With no choice saved, a client opens where it
/// always has. `Tests/Fixtures/client_opening_cases.json`, written from the
/// Python module, keeps the two in step.
public enum ClientOpenings {
    public static let terminalChoice = "terminal"
    public static let mtplxApp = "mtplx-app"
    public static let ownApp = "app"
    static let appleTerminalBundleID = "com.apple.Terminal"
    /// A `.command` file's content type. An app that declares it with the
    /// Shell role runs the script in a new window, as Terminal does.
    static let shellScriptType = "com.apple.terminal.shell-script"

    public struct Client: Equatable, Sendable {
        public let id: String
        public let name: String
        public let configKey: String
        /// What it can open as besides a terminal.
        public let appSurfaces: [String]
        /// Where it opens with no choice saved.
        public let defaultChoice: String
        /// Its desktop app ("Claude" for Claude Code).
        public let appName: String?
    }

    public static let clients: [Client] = [
        Client(id: "pi", name: "Pi", configKey: "pi_opens_in",
               appSurfaces: [], defaultChoice: terminalChoice, appName: nil),
        Client(id: "opencode", name: "OpenCode", configKey: "opencode_opens_in",
               appSurfaces: [ownApp], defaultChoice: ownApp, appName: "OpenCode"),
        Client(id: "hermes", name: "Hermes", configKey: "hermes_opens_in",
               appSurfaces: [ownApp], defaultChoice: ownApp, appName: "Hermes"),
        Client(id: "claude-code", name: "Claude Code", configKey: "claude_code_opens_in",
               appSurfaces: [], defaultChoice: terminalChoice, appName: nil),
        Client(id: "codex", name: "Codex", configKey: "codex_opens_in",
               appSurfaces: [mtplxApp], defaultChoice: mtplxApp, appName: "Codex"),
        // Its app is its browser app (`dsh web`), which MTPLX starts and stops.
        Client(id: "deepseek-harness", name: "DeepSeek Harness", configKey: "deepseek_harness_opens_in",
               appSurfaces: [ownApp], defaultChoice: ownApp, appName: "DeepSeek Harness web"),
    ]

    public static func client(_ id: String) -> Client? {
        clients.first { $0.id == id }
    }

    public static func client(for target: LaunchTarget) -> Client? {
        switch target {
        case .pi: return client("pi")
        case .openCode: return client("opencode")
        case .hermes: return client("hermes")
        case .claudeCode: return client("claude-code")
        case .codex: return client("codex")
        case .deepseekHarness: return client("deepseek-harness")
        default: return nil
        }
    }

    public struct TerminalApp: Equatable, Hashable, Sendable, Identifiable {
        public let bundleID: String
        public let name: String
        public let path: String
        public var id: String { bundleID }

        public init(bundleID: String, name: String, path: String) {
            self.bundleID = bundleID
            self.name = name
            self.path = path
        }

        /// A choice names a terminal by bundle id or app name, any case.
        func matches(_ choice: String) -> Bool {
            let wanted = choice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            return wanted == bundleID.lowercased() || wanted == name.lowercased()
        }
    }

    public struct Opening: Equatable, Sendable {
        public let client: String
        /// `terminalChoice`, `mtplxApp` or `ownApp`.
        public let surface: String
        /// nil: Terminal by name.
        public let terminal: TerminalApp?
        public let note: String?

        public var opensInTerminal: Bool { surface == ClientOpenings.terminalChoice }
        public var terminalName: String { terminal?.name ?? "Terminal" }
        /// The application `open -a` is given for a `.command` script.
        public var openApplication: String { terminal?.path ?? "Terminal" }
    }

    // MARK: - Finding terminals

    public static let searchDirectories: [URL] = [
        URL(fileURLWithPath: "/System/Applications/Utilities", isDirectory: true),
        URL(fileURLWithPath: "/Applications", isDirectory: true),
        URL(fileURLWithPath: "/Applications/Utilities", isDirectory: true),
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications"),
    ]

    /// The terminal an app's Info.plist describes, or nil when the app does
    /// not run shell scripts.
    public static func terminal(fromInfo info: [String: Any], path: String) -> TerminalApp? {
        guard let bundleID = info["CFBundleIdentifier"] as? String, !bundleID.isEmpty else {
            return nil
        }
        let kinds = info["CFBundleDocumentTypes"] as? [[String: Any]] ?? []
        let runsScripts = kinds.contains { kind in
            (kind["CFBundleTypeRole"] as? String) == "Shell"
                && ((kind["LSItemContentTypes"] as? [String]) ?? []).contains(shellScriptType)
        }
        guard runsScripts else { return nil }
        let stem = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        let name = [info["CFBundleDisplayName"], info["CFBundleName"]]
            .compactMap { $0 as? String }
            .first { !$0.isEmpty } ?? stem
        return TerminalApp(bundleID: bundleID, name: name, path: path)
    }

    static func terminal(atBundle bundle: URL) -> TerminalApp? {
        let plist = bundle.appendingPathComponent("Contents/Info.plist")
        guard let data = try? Data(contentsOf: plist),
              let info = try? PropertyListSerialization.propertyList(from: data, format: nil)
                as? [String: Any]
        else { return nil }
        return terminal(fromInfo: info, path: bundle.path)
    }

    /// Terminal first, the rest by name; one entry per bundle id (the first
    /// one seen wins).
    public static func sorted(_ terminals: [TerminalApp]) -> [TerminalApp] {
        var seen = Set<String>()
        let unique = terminals.filter { seen.insert($0.bundleID).inserted }
        return unique.sorted { lhs, rhs in
            let lhsApple = lhs.bundleID == appleTerminalBundleID
            let rhsApple = rhs.bundleID == appleTerminalBundleID
            if lhsApple != rhsApple { return lhsApple }
            return lhs.name.lowercased() < rhs.name.lowercased()
        }
    }

    /// Every installed app that runs a `.command` script.
    public static func installedTerminals(
        in directories: [URL] = searchDirectories,
        fileManager: FileManager = .default
    ) -> [TerminalApp] {
        var found: [TerminalApp] = []
        for directory in directories {
            guard let children = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ) else { continue }
            let bundles = children
                .filter { $0.pathExtension == "app" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            found.append(contentsOf: bundles.compactMap(terminal(atBundle:)))
        }
        return sorted(found)
    }

    // MARK: - Resolving a choice

    /// The opening for `clientID` from its saved choice. No choice is the
    /// client's default; a terminal that is no longer installed opens
    /// Terminal and says so, so a launch never fails over a setting.
    public static func resolve(
        client clientID: String,
        choice: String?,
        terminals: [TerminalApp]
    ) -> Opening {
        guard let client = client(clientID) else {
            return Opening(client: clientID, surface: terminalChoice, terminal: nil, note: nil)
        }
        let trimmed = (choice ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let value = trimmed.isEmpty ? client.defaultChoice : trimmed
        if client.appSurfaces.contains(value) {
            return Opening(client: client.id, surface: value, terminal: nil, note: nil)
        }
        let fallback = terminals.first { $0.bundleID == appleTerminalBundleID }
        if value == terminalChoice {
            return Opening(client: client.id, surface: terminalChoice, terminal: fallback, note: nil)
        }
        if value == mtplxApp || value == ownApp {
            return Opening(
                client: client.id, surface: terminalChoice, terminal: fallback,
                note: "\(client.name) has no app to open; opening Terminal"
            )
        }
        if let chosen = terminals.first(where: { $0.matches(value) }) {
            return Opening(client: client.id, surface: terminalChoice, terminal: chosen, note: nil)
        }
        return Opening(
            client: client.id, surface: terminalChoice, terminal: fallback,
            note: "\(value) is not installed; opening Terminal"
        )
    }

    /// The terminal a client's command runs in: the one the user chose, else
    /// Terminal (no choice, or an app this caller is not opening).
    public static func terminalFor(
        client clientID: String,
        choice: String?,
        terminals: [TerminalApp]
    ) -> Opening {
        let opening = resolve(client: clientID, choice: choice, terminals: terminals)
        if opening.opensInTerminal { return opening }
        return Opening(
            client: clientID,
            surface: terminalChoice,
            terminal: terminals.first { $0.bundleID == appleTerminalBundleID },
            note: nil
        )
    }

    // MARK: - The shared setting

    /// Every client's saved choice from the shared config (nil: not set).
    public static func savedChoices(
        configURL: URL = SharedModelFolders.configURL()
    ) -> [String: String] {
        guard let text = try? String(contentsOf: configURL, encoding: .utf8) else { return [:] }
        var choices: [String: String] = [:]
        for client in clients {
            if case .some(.some(.string(let value))) = TOMLLines.topLevelValue(text, key: client.configKey) {
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { choices[client.id] = trimmed }
            }
        }
        return choices
    }

    /// Saves one client's choice, keeping every other line of the file; nil
    /// removes it (the client opens where it always has).
    public static func save(
        choice: String?,
        for clientID: String,
        configURL: URL = SharedModelFolders.configURL()
    ) throws {
        guard let client = client(clientID) else { return }
        let fileManager = FileManager.default
        // A file that exists but cannot be read is never replaced: it holds
        // the user's other CLI settings.
        let existing: String? = fileManager.fileExists(atPath: configURL.path)
            ? try String(contentsOf: configURL, encoding: .utf8)
            : nil
        let text = TOMLLines.setTopLevelValues(
            existing ?? "# MTPLX user configuration\n",
            [(client.configKey, choice.map(TOMLLines.Value.string))]
        )
        guard text != existing else { return }
        try fileManager.createDirectory(
            at: configURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(text.utf8).write(to: configURL, options: [.atomic])
    }

    /// What one client opens as from the shared setting (`mtplxApp`,
    /// `ownApp` or `terminalChoice`), without looking for terminals.
    public static func currentSurface(
        _ target: LaunchTarget,
        configURL: URL = SharedModelFolders.configURL()
    ) -> String? {
        guard let client = client(for: target) else { return nil }
        return resolve(
            client: client.id,
            choice: savedChoices(configURL: configURL)[client.id],
            terminals: []
        ).surface
    }

    /// The terminal one client's command runs in, from the shared setting.
    public static func currentTerminal(
        _ target: LaunchTarget,
        configURL: URL = SharedModelFolders.configURL(),
        terminals: [TerminalApp]? = nil
    ) -> Opening {
        let id = client(for: target)?.id ?? target.rawValue
        return terminalFor(
            client: id,
            choice: savedChoices(configURL: configURL)[id],
            terminals: terminals ?? installedTerminals()
        )
    }
}
