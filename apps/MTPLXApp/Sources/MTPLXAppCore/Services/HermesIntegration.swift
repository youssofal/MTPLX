import Darwin
import Foundation
#if os(macOS)
import AppKit
#endif

public struct HermesProfile: Identifiable, Equatable, Sendable {
    public let name: String
    public let path: String
    public let isDefault: Bool

    public var id: String { name }

    public init(name: String, path: String, isDefault: Bool) {
        self.name = name
        self.path = path
        self.isDefault = isDefault
    }
}

public struct HermesInstallStatus: Equatable, Sendable {
    public enum Kind: String, Equatable, Sendable {
        case ready
        case missing
        case incompatible
    }

    public enum GatewayHealth: String, Equatable, Sendable {
        case healthy
        case warning
        case unavailable
    }

    public let kind: Kind
    public let executablePath: String?
    public let versionSummary: String?
    public let updateSummary: String?
    public let gatewaySummary: String?
    public let gatewayHealth: GatewayHealth?
    public let enabledToolsets: [String]
    public let capabilitySummary: String
    public let integrationSummaries: [String]
    public let warnings: [String]
    public let detail: String
    public let updateCommand: String?

    public var gatewayNeedsRepair: Bool {
        gatewayHealth == .warning || gatewayHealth == .unavailable
    }

    public static func ready(
        executablePath: String,
        versionSummary: String?,
        updateSummary: String? = nil,
        gatewaySummary: String? = nil,
        gatewayHealth: GatewayHealth? = nil,
        integrationSummaries: [String] = [],
        warnings: [String] = []
    ) -> HermesInstallStatus {
        HermesInstallStatus(
            kind: .ready,
            executablePath: executablePath,
            versionSummary: versionSummary,
            updateSummary: updateSummary,
            gatewaySummary: gatewaySummary,
            gatewayHealth: gatewayHealth,
            enabledToolsets: HermesIntegration.codingToolsetNames,
            capabilitySummary: HermesIntegration.capabilitySummary,
            integrationSummaries: integrationSummaries,
            warnings: warnings,
            detail: versionSummary ?? "Hermes is ready.",
            updateCommand: updateSummary == nil ? nil : "hermes update"
        )
    }

    public static func missing() -> HermesInstallStatus {
        HermesInstallStatus(
            kind: .missing,
            executablePath: nil,
            versionSummary: nil,
            updateSummary: nil,
            gatewaySummary: nil,
            gatewayHealth: nil,
            enabledToolsets: HermesIntegration.codingToolsetNames,
            capabilitySummary: HermesIntegration.capabilitySummary,
            integrationSummaries: [],
            warnings: [],
            detail: "Hermes is not on PATH.",
            updateCommand: "pip install -U hermes-agent[web,pty]"
        )
    }

    public static func incompatible(
        executablePath: String,
        versionSummary: String?,
        detail: String
    ) -> HermesInstallStatus {
        HermesInstallStatus(
            kind: .incompatible,
            executablePath: executablePath,
            versionSummary: versionSummary,
            updateSummary: nil,
            gatewaySummary: nil,
            gatewayHealth: nil,
            enabledToolsets: HermesIntegration.codingToolsetNames,
            capabilitySummary: HermesIntegration.capabilitySummary,
            integrationSummaries: [],
            warnings: [],
            detail: detail,
            updateCommand: "hermes update"
        )
    }
}

public struct HermesConfigResult: Equatable, Sendable {
    public let profileName: String
    public let profilePath: String
    public let configPath: String
    public let envPath: String
    public let baseURL: String
    public let modelReference: String
    public let workspacePath: String
    public let launchCommand: String
    public let didChange: Bool
    public let configBackupPath: String?
    public let envBackupPath: String?
}

public struct HermesGatewayRepairResult: Equatable, Sendable {
    public let startSummary: String
    public let statusSummary: String?
    public let statusHealth: HermesInstallStatus.GatewayHealth?
}

public enum HermesLaunchAction: String, Equatable, Sendable {
    case launched
    case unavailable
}

public struct HermesLaunchResult: Equatable, Sendable {
    public let action: HermesLaunchAction
    public let command: String
    public let detail: String
}

public enum HermesIntegrationError: Error, Equatable, LocalizedError {
    case executableNotFound
    case incompatible(String)
    case launchFailed(String)
    case dashboardTokenTimeout
    case profileCreateFailed(String)

    public var errorDescription: String? {
        switch self {
        case .executableNotFound:
            return "Hermes is not installed or not on PATH."
        case .incompatible(let detail):
            return detail
        case .launchFailed(let detail):
            return "Hermes could not start: \(detail)"
        case .dashboardTokenTimeout:
            return "Hermes dashboard started, but the session token never appeared."
        case .profileCreateFailed(let detail):
            return "Hermes profile could not be created: \(detail)"
        }
    }
}

public final class HermesSidecar: @unchecked Sendable {
    public let process: Process
    public let port: Int
    public let token: String
    public let dashboardURL: URL
    public let webSocketURL: URL

    init(process: Process, port: Int, token: String) {
        self.process = process
        self.port = port
        self.token = token
        self.dashboardURL = URL(string: "http://127.0.0.1:\(port)/")!
        self.webSocketURL = URL(string: "ws://127.0.0.1:\(port)/api/ws?token=\(token)")!
    }

    public func stop() {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(2)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
    }
}

public struct HermesSessionReference: Equatable, Sendable {
    public let profileName: String
    public let sessionID: String
    public let title: String?

    public init(profileName: String, sessionID: String, title: String?) {
        self.profileName = profileName
        self.sessionID = sessionID
        self.title = title
    }
}

public struct HermesIntegration: Sendable {
    public static let profileName = "mtplx"
    public static let localAPIKey = PiIntegration.localAPIKey
    public static let nativeDashboardSupported = false
    public static let codingToolsetNames = ["terminal", "file", "web", "browser", "messaging"]
    public static let codingToolsets = codingToolsetNames.joined(separator: ",")
    public static let capabilitySummary = "Terminal, file, web, browser, and messaging tools."
    public static let gatewayStatusCommand = "env -u HERMES_HOME hermes gateway status"
    public static let gatewayTruthHint = "MTPLX uses a profile-scoped HERMES_HOME for model routing, while Hermes Gateway runs from the root ~/.hermes LaunchAgent. Plain `hermes status` can therefore report profile-local Gateway state as not loaded even when Telegram is connected. MTPLX mirrors the root Gateway channel directory into the profile before launch, so send_message(action='list') should show the same discovered targets. For live messaging truth, use `env -u HERMES_HOME hermes gateway status` and then send_message(action='list')."
    public static let messagingSetupHint = "Messaging uses Hermes Gateway. Telegram setup is the interactive `hermes gateway setup` flow: choose Telegram, provide TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, and optionally TELEGRAM_HOME_CHANNEL. \(gatewayTruthHint) If the service definition is stale run `hermes gateway start`. Use send_message(action='list') before sending when the destination is unclear. If send_message lists no targets, distinguish configured credentials from connected or discovered targets before saying Telegram is unconfigured. On MTPLXApp for macOS, do not recommend sudo/systemd/system-service setup, and prefer `hermes gateway start` over `hermes gateway install` for this app-owned repair path. Never print token, user id, or channel id values from `.env`; say only configured, missing, connected, not connected, or needs repair unless the user explicitly asks for a redacted diagnostic."
    public static let systemPrompt = "You are Hermes inside MTPLXApp on macOS. You have terminal, file, web, browser, and messaging tools, including the messaging send_message tool when Hermes Gateway is running. Telegram setup uses the interactive `hermes gateway setup` command where the user chooses Telegram; Telegram configuration lives in TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, and TELEGRAM_HOME_CHANNEL. \(gatewayTruthHint) Treat `hermes status` under the MTPLX profile as a profile-local diagnostic only; never use it by itself to conclude Telegram cannot connect. Never print token, user id, channel id, webhook URL, API key, or other secret values from `.env`; report only configured, missing, connected, not connected, or needs repair unless the user explicitly asks for a redacted diagnostic. If global LaunchAgent status says the service definition is stale, tell the user to run `hermes gateway start`; if the gateway is loaded and send_message(action='list') runs, still say messaging is available, with LaunchAgent repair recommended. If send_message(action='list') returns no targets, that means no connected or discovered destinations yet; do not conclude Telegram credentials are missing until you perform a sanitized presence check for TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, and TELEGRAM_HOME_CHANNEL in the MTPLX profile .env or root Hermes .env. If setup or status fails because Hermes root .env is not valid UTF-8, tell the user to repair that .env before retrying setup. For MTPLXApp/macOS setup, do not recommend sudo, systemd, system services, `hermes gateway install`, or `hermes gateway install --system`; prefer `hermes gateway start` for the app-owned LaunchAgent repair path. Do not invent HERMES_GATEWAY_* env vars, do not require Cloudflare for normal polling setup, and do not claim Telegram is unsupported when the messaging toolset is enabled. When explaining how messaging works after setup, mention send_message and tell the user to use send_message(action='list') before sending if the destination is unclear. When asked what tools are available, explicitly name browser and messaging if enabled. Do not send external messages unless the user explicitly gives the destination and content."

    public let hermesHome: URL
    public let executablePath: String?
    public let environment: [String: String]
    public let terminalCommandURL: URL
    /// Where the Hermes Desktop app reads its startup profile pin
    /// (`{"profile": "<name>"}`, validated by their renderer's
    /// PROFILE_NAME_RE; the desktop persists its own selection thereafter).
    public let activeProfileURL: URL
    /// Test seam: bypass bootstrap-layout + LaunchServices discovery.
    public let desktopApplicationOverride: URL?

    public init(
        hermesHome: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".hermes", isDirectory: true),
        executablePath: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        terminalCommandURL: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("open-hermes.command"),
        activeProfileURL: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Application Support/Hermes")
            .appendingPathComponent("active-profile.json"),
        desktopApplicationOverride: URL? = nil
    ) {
        self.hermesHome = hermesHome
        self.executablePath = executablePath
        self.environment = environment
        self.terminalCommandURL = terminalCommandURL
        self.activeProfileURL = activeProfileURL
        self.desktopApplicationOverride = desktopApplicationOverride
    }

    public func discoverProfiles() -> [HermesProfile] {
        let fileManager = FileManager.default
        var profiles: [HermesProfile] = [
            HermesProfile(
                name: "default",
                path: hermesHome.path,
                isDefault: true
            )
        ]

        let profilesRoot = hermesHome.appendingPathComponent("profiles", isDirectory: true)
        guard let children = try? fileManager.contentsOfDirectory(
            at: profilesRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return profiles
        }

        let named = children
            .filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
                    && Self.isValidProfileName(url.lastPathComponent)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map {
                HermesProfile(
                    name: $0.lastPathComponent,
                    path: $0.path,
                    isDefault: false
                )
            }
        profiles.append(contentsOf: named)
        return profiles
    }

    public func installStatus() async -> HermesInstallStatus {
        guard let executable = resolveExecutable() else {
            return .missing()
        }

        let output = (try? await runAndCapture(executableURL: executable, arguments: ["--version"])) ?? ""
        let versionSummary = output
            .split(separator: "\n")
            .first
            .map(String.init)
        let gatewayOutput = (try? await runAndCapture(
            executableURL: executable,
            arguments: ["gateway", "status"],
            environment: gatewayCommandEnvironment()
        )) ?? ""
        let localStatus = localMessagingStatus()
        let chatHelp = (try? await runAndCapture(
            executableURL: executable,
            arguments: ["chat", "--help"]
        )) ?? ""
        if chatHelp.contains("--query") && chatHelp.contains("--source") {
            return .ready(
                executablePath: executable.path,
                versionSummary: versionSummary,
                updateSummary: Self.updateSummary(fromVersionOutput: output),
                gatewaySummary: Self.gatewaySummary(fromStatusOutput: gatewayOutput),
                gatewayHealth: Self.gatewayHealth(fromStatusOutput: gatewayOutput),
                integrationSummaries: localStatus.summaries,
                warnings: localStatus.warnings + Self.gatewayWarnings(fromStatusOutput: gatewayOutput)
            )
        }

        return .incompatible(
            executablePath: executable.path,
            versionSummary: versionSummary,
            detail: "Hermes must expose the chat command before MTPLX can launch it."
        )
    }

    public func repairGateway() async throws -> HermesGatewayRepairResult {
        guard let executable = resolveExecutable() else {
            throw HermesIntegrationError.executableNotFound
        }
        let startOutput = try await runAndCapture(
            executableURL: executable,
            arguments: ["gateway", "start"],
            environment: gatewayCommandEnvironment()
        )
        let statusOutput = (try? await runAndCapture(
            executableURL: executable,
            arguments: ["gateway", "status"],
            environment: gatewayCommandEnvironment()
        )) ?? ""
        return HermesGatewayRepairResult(
            startSummary: Self.gatewayRepairSummary(fromStartOutput: startOutput),
            statusSummary: Self.gatewaySummary(fromStatusOutput: statusOutput),
            statusHealth: Self.gatewayHealth(fromStatusOutput: statusOutput)
        )
    }

    public func launchEnvironment(configuration: MTPLXAppConfiguration) -> [String: String] {
        let modelID = OpenCodeIntegration.modelID(for: configuration.model)
        let apiKey = configuration.apiKey?.isEmpty == false
            ? configuration.apiKey!
            : PiIntegration.localAPIKey
        let reasoning = Self.reasoningMode(for: configuration)
        let reasoningEffort = Self.reasoningEffort(for: configuration)
        var env = environment
        let baseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        env["OPENAI_BASE_URL"] = baseURL
        env["CUSTOM_BASE_URL"] = baseURL
        env["OPENAI_API_KEY"] = apiKey
        env["HERMES_MODEL"] = modelID
        env["HERMES_INFERENCE_MODEL"] = modelID
        env["HERMES_INFERENCE_PROVIDER"] = "custom"
        env["HERMES_MTPLX_REASONING"] = reasoning
        env["HERMES_MTPLX_SHOW_REASONING"] = reasoning == "off" ? "0" : "1"
        if let reasoningEffort {
            env["HERMES_MTPLX_REASONING_EFFORT"] = reasoningEffort
        } else {
            env.removeValue(forKey: "HERMES_MTPLX_REASONING_EFFORT")
        }
        env["HERMES_YOLO_MODE"] = configuration.hermesAutoApprove ? "1" : "0"
        env["HERMES_MTPLX_TOOLSETS"] = Self.codingToolsets
        env["HERMES_MTPLX_CAPABILITIES"] = Self.capabilitySummary
        env["HERMES_MTPLX_MESSAGING_NOTE"] = Self.messagingSetupHint
        env["HERMES_MTPLX_GATEWAY_STATUS_COMMAND"] = Self.gatewayStatusCommand
        env["HERMES_MTPLX_GATEWAY_TRUTH_NOTE"] = Self.gatewayTruthHint
        if !localMessagingBridgeEnvironment().isEmpty {
            env["HERMES_SESSION_PLATFORM"] = "mtplx-app"
        } else {
            env.removeValue(forKey: "HERMES_SESSION_PLATFORM")
        }
        let localStatus = localMessagingStatus()
        env["HERMES_MTPLX_MESSAGING_SUMMARY"] = localStatus.summaries.joined(separator: " ")
        if localStatus.warnings.isEmpty {
            env.removeValue(forKey: "HERMES_MTPLX_MESSAGING_WARNINGS")
        } else {
            env["HERMES_MTPLX_MESSAGING_WARNINGS"] = localStatus.warnings.joined(separator: " ")
        }
        env["HERMES_WORKSPACE"] = Self.resolvedWorkspacePath(
            configuration: configuration,
            environment: environment
        )
        env["TERMINAL_CWD"] = env["HERMES_WORKSPACE"]
        return env
    }

    /// Builds the environment and argument vector for a dashboard server that
    /// belongs only to this app process. It does not launch the process.
    public func serveLaunchSpec(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration,
        token: String,
        launchID: String,
        parentPID: Int32
    ) throws -> HermesServeLaunchSpec {
        guard Self.isValidEmbeddedLaunchID(launchID) else {
            throw HermesIntegrationError.launchFailed("invalid embedded launch identifier")
        }
        guard let executableURL = resolveExecutable() else {
            throw HermesIntegrationError.executableNotFound
        }

        let modelID = OpenCodeIntegration.modelID(for: configuration.model)
        let baseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        let apiKey = configuration.apiKey?.isEmpty == false
            ? configuration.apiKey!
            : Self.localAPIKey
        var processEnvironment = launchEnvironment(configuration: configuration)
        processEnvironment.removeValue(forKey: "HERMES_HOME")
        processEnvironment["CUSTOM_BASE_URL"] = baseURL
        processEnvironment["OPENAI_BASE_URL"] = baseURL
        processEnvironment["OPENAI_API_KEY"] = apiKey
        processEnvironment["HERMES_MODEL"] = modelID
        processEnvironment["HERMES_INFERENCE_MODEL"] = modelID
        processEnvironment["HERMES_INFERENCE_PROVIDER"] = "custom"
        processEnvironment["HERMES_DASHBOARD_SESSION_TOKEN"] = token
        processEnvironment["HERMES_SESSION_PLATFORM"] = "mtplx-app"
        processEnvironment["MTPLX_HERMES_LAUNCH_ID"] = launchID
        processEnvironment["MTPLX_HERMES_PARENT_PID"] = String(parentPID)

        var arguments: [String] = []
        if !profile.isDefault {
            arguments += ["-p", profile.name]
        }
        arguments += [
            "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", launchID,
        ]
        return HermesServeLaunchSpec(
            executableURL: executableURL,
            arguments: arguments,
            environment: processEnvironment,
            token: token,
            launchID: launchID,
            parentPID: parentPID
        )
    }

    /// Routes profiles without writing to them. An unavailable profile is
    /// intentionally kept distinct from a readable, user-managed profile.
    public func routingState(
        for profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) -> HermesProfileRoutingState {
        guard let effective = Self.effectiveProfileConfiguration(at: URL(fileURLWithPath: profile.path)) else {
            return .unavailable("Profile configuration is unavailable.")
        }
        let expectedBaseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        let expectedModelID = OpenCodeIntegration.modelID(for: configuration.model)
        if effective.provider == "custom"
            && effective.baseURL == expectedBaseURL
            && effective.modelReference == expectedModelID {
            return .mtplx
        }
        return .external
    }

    @discardableResult
    public func sync(configuration: MTPLXAppConfiguration) throws -> HermesConfigResult {
        let modelID = OpenCodeIntegration.modelID(for: configuration.model)
        let baseURL = OpenCodeIntegration.baseURLString(
            host: configuration.host,
            port: configuration.port
        )
        let apiKey = configuration.apiKey?.isEmpty == false
            ? configuration.apiKey!
            : Self.localAPIKey
        let reasoning = Self.reasoningMode(for: configuration)
        let reasoningEffort = Self.reasoningEffort(for: configuration)
        let workspacePath = Self.resolvedWorkspacePath(
            configuration: configuration,
            environment: environment
        )
        let messagingEnvironment = localMessagingBridgeEnvironment()
        let profileURL = hermesHome
            .appendingPathComponent("profiles", isDirectory: true)
            .appendingPathComponent(Self.profileName, isDirectory: true)
        let configURL = profileURL.appendingPathComponent("config.yaml")
        let envURL = profileURL.appendingPathComponent(".env")

        try ensureProfileDirectory(profileURL)

        // Issue #131: the profile config is user-editable. Regenerating it
        // from the template alone silently wiped every section the app does
        // not own (memory/providers/delegation/…), so the template is merged
        // over the existing file instead: app-owned keys are rewritten, all
        // other content is preserved byte-for-byte.
        let existingConfigText = try? String(contentsOf: configURL, encoding: .utf8)
        let configText = Self.mergedConfigYAML(
            existing: existingConfigText,
            template: Self.configYAML(
                modelID: modelID,
                baseURL: baseURL,
                apiKey: apiKey,
                workspacePath: workspacePath,
                showReasoning: reasoning != "off",
                reasoningEffort: reasoningEffort
            )
        )
        let envText = Self.dotenv(
            modelID: modelID,
            baseURL: baseURL,
            apiKey: apiKey,
            workspacePath: workspacePath,
            autoApprove: configuration.hermesAutoApprove,
            reasoning: reasoning,
            reasoningEffort: reasoningEffort,
            messagingEnvironment: messagingEnvironment
        )
        let configWrite = try writeIfChanged(
            configText,
            to: configURL,
            permissions: 0o600,
            backupReason: "mtplx"
        )
        let envWrite = try writeIfChanged(
            envText,
            to: envURL,
            permissions: 0o600,
            backupReason: "mtplx"
        )
        let didMirrorChannels = try mirrorRootChannelDirectory(into: profileURL)

        return HermesConfigResult(
            profileName: Self.profileName,
            profilePath: profileURL.path,
            configPath: configURL.path,
            envPath: envURL.path,
            baseURL: baseURL,
            modelReference: modelID,
            workspacePath: workspacePath,
            launchCommand: Self.launchCommand(for: configuration.model),
            didChange: configWrite.didChange || envWrite.didChange || didMirrorChannels,
            configBackupPath: configWrite.backupPath,
            envBackupPath: envWrite.backupPath
        )
    }

    public static func launchCommand(for model: String) -> String {
        terminalLaunchCommand(
            hermesExecutable: "hermes",
            profileName: Self.profileName,
            modelID: OpenCodeIntegration.modelID(for: model),
            autoApprove: true
        )
    }

    @discardableResult
    public func stopLaunchedTerminalAgents() -> Int {
        let pids = Self.appLaunchedTerminalAgentPIDs()
        for pid in pids {
            Self.terminate(pid: pid)
        }
        return pids.count
    }

    public func hasLaunchedTerminalAgent() -> Bool {
        !Self.appLaunchedTerminalAgentPIDs().isEmpty
    }

    public static let desktopBundleIdentifier = "com.nousresearch.hermes"

    /// The built Hermes Desktop bundle to launch, or nil for CLI-only
    /// installs. Bootstrap layout first (`hermes desktop` builds inside the
    /// agent checkout under ~/.hermes), LaunchServices second.
    /// `/Applications/Hermes.app` is only their Tauri *setup stub*
    /// (`com.nousresearch.hermes.setup`) — a different bundle id, so the
    /// LaunchServices lookup cannot match it.
    public func desktopApplicationURL(fileManager: FileManager = .default) -> URL? {
        if let override = desktopApplicationOverride {
            return fileManager.fileExists(atPath: override.path) ? override : nil
        }
        let desktopRelease = hermesHome
            .appendingPathComponent("hermes-agent", isDirectory: true)
            .appendingPathComponent("apps", isDirectory: true)
            .appendingPathComponent("desktop", isDirectory: true)
            .appendingPathComponent("release", isDirectory: true)
        for arch in ["mac-arm64", "mac"] {
            let candidate = desktopRelease
                .appendingPathComponent(arch, isDirectory: true)
                .appendingPathComponent("Hermes.app", isDirectory: true)
            if fileManager.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        #if os(macOS)
        if let resolved = NSWorkspace.shared.urlForApplication(
            withBundleIdentifier: Self.desktopBundleIdentifier
        ) {
            return resolved
        }
        #endif
        return nil
    }

    /// Pin the Desktop app's startup profile to the MTPLX profile. Returns
    /// the previously pinned profile name (if any) so callers can surface
    /// the switch to the user.
    @discardableResult
    public func writeActiveDesktopProfile(fileManager: FileManager = .default) throws -> String? {
        try fileManager.createDirectory(
            at: activeProfileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        var previous: String?
        if let data = try? Data(contentsOf: activeProfileURL),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            previous = object["profile"] as? String
        }
        let payload = try JSONSerialization.data(
            withJSONObject: ["profile": Self.profileName]
        )
        try payload.write(to: activeProfileURL, options: .atomic)
        return previous
    }

    #if os(macOS)
    /// Launch the built Hermes Desktop app pinned to the MTPLX profile —
    /// the OpenCode-style flow. Returns nil when no built Desktop bundle
    /// exists so the caller can fall back to the Terminal handoff.
    @MainActor
    public func launchDesktopApplication(
        configuration: MTPLXAppConfiguration
    ) async -> HermesLaunchResult? {
        guard let appURL = desktopApplicationURL() else { return nil }
        let command = "open \(appURL.path)"
        do {
            _ = try sync(configuration: configuration)
        } catch {
            return HermesLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not sync Hermes profile: \(error)"
            )
        }
        let previous: String?
        do {
            previous = try writeActiveDesktopProfile()
        } catch {
            return HermesLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not pin Hermes Desktop to the MTPLX profile: \(error)"
            )
        }
        let openConfiguration = NSWorkspace.OpenConfiguration()
        openConfiguration.activates = true
        let opened = await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            NSWorkspace.shared.openApplication(
                at: appURL,
                configuration: openConfiguration
            ) { _, error in
                continuation.resume(returning: error == nil)
            }
        }
        guard opened else {
            return HermesLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not open Hermes Desktop at \(appURL.path)"
            )
        }
        let previousNote: String
        if let previous, previous != Self.profileName {
            previousNote = " (was \(previous); the in-app profile picker switches back)"
        } else {
            previousNote = ""
        }
        return HermesLaunchResult(
            action: .launched,
            command: command,
            detail: "opened Hermes Desktop pinned to profile \(Self.profileName)\(previousNote)"
        )
    }

    /// Desktop-first launch: the built Desktop app when present (same flow
    /// as the OpenCode card), the Terminal handoff otherwise. A present-but-
    /// broken Desktop bundle falls back to Terminal rather than stranding
    /// the user, carrying both details.
    @MainActor
    public func launch(configuration: MTPLXAppConfiguration) async -> HermesLaunchResult {
        guard let desktop = await launchDesktopApplication(configuration: configuration) else {
            return launchInTerminal(configuration: configuration)
        }
        if desktop.action == .launched {
            return desktop
        }
        let terminal = launchInTerminal(configuration: configuration)
        return HermesLaunchResult(
            action: terminal.action,
            command: terminal.command,
            detail: "\(desktop.detail); fell back to Terminal: \(terminal.detail)"
        )
    }
    #endif

    public func launchInTerminal(configuration: MTPLXAppConfiguration) -> HermesLaunchResult {
        do {
            _ = try sync(configuration: configuration)
        } catch {
            return HermesLaunchResult(
                action: .unavailable,
                command: Self.launchCommand(for: configuration.model),
                detail: "could not sync Hermes profile: \(error)"
            )
        }

        guard let executable = resolveExecutable() else {
            return HermesLaunchResult(
                action: .unavailable,
                command: Self.launchCommand(for: configuration.model),
                detail: "Hermes is not installed or not on PATH."
            )
        }

        let command = Self.terminalLaunchCommand(
            hermesExecutable: executable.path,
            profileName: Self.profileName,
            modelID: OpenCodeIntegration.modelID(for: configuration.model),
            autoApprove: configuration.hermesAutoApprove
        )
        #if os(macOS)
        let scriptURL: URL
        do {
            scriptURL = try writeTerminalCommandFile(
                command: command,
                hermesExecutablePath: executable.path,
                configuration: configuration
            )
        } catch {
            return HermesLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not prepare Hermes terminal command: \(error)"
            )
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Terminal", scriptURL.path]
        let stderr = Pipe()
        process.standardError = stderr
        // The backend store calls this from the main actor, so a wedged
        // LaunchServices `open` must cost one bounded window, never
        // beachball the app (#158 pattern).
        let stderrTail = SubprocessTailBuffer(capacity: 4096)
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { stderrTail.append(chunk) }
        }
        defer { stderr.fileHandleForReading.readabilityHandler = nil }
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
            guard watchdog.wait(for: process, timeout: 30) else {
                return HermesLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: "could not open Hermes automatically: open timed out after 30s and was terminated"
                )
            }
            guard process.terminationStatus == 0 else {
                let message = stderrTail.snapshot()
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return HermesLaunchResult(
                    action: .unavailable,
                    command: command,
                    detail: message.isEmpty
                        ? "could not open Hermes automatically: open exited \(process.terminationStatus)"
                        : "could not open Hermes automatically: \(message)"
                )
            }
            return HermesLaunchResult(
                action: .launched,
                command: command,
                detail: "opened Hermes in Terminal"
            )
        } catch {
            return HermesLaunchResult(
                action: .unavailable,
                command: command,
                detail: "could not open Hermes automatically: \(error)"
            )
        }
        #else
        return HermesLaunchResult(
            action: .unavailable,
            command: command,
            detail: "automatic Hermes launch currently requires macOS Terminal"
        )
        #endif
    }

    public func startDashboard(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> HermesSidecar {
        _ = profile
        _ = configuration
        throw HermesIntegrationError.incompatible(
            "This Hermes build exposes CLI chat and ACP, not the dashboard WebSocket surface."
        )
    }

    public func createProfile(named rawName: String) async throws -> HermesProfile {
        guard let executable = resolveExecutable() else {
            throw HermesIntegrationError.executableNotFound
        }
        let name = rawName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard Self.isValidProfileName(name) else {
            throw HermesIntegrationError.profileCreateFailed("Use lowercase letters, numbers, dashes, or underscores.")
        }
        let output = try await runAndCapture(
            executableURL: executable,
            arguments: ["profile", "create", name]
        )
        guard discoverProfiles().contains(where: { $0.name == name }) else {
            throw HermesIntegrationError.profileCreateFailed(output)
        }
        return HermesProfile(
            name: name,
            path: hermesHome.appendingPathComponent("profiles").appendingPathComponent(name).path,
            isDefault: false
        )
    }

    private struct WriteResult: Equatable {
        let didChange: Bool
        let backupPath: String?
    }

    private func ensureProfileDirectory(_ profileURL: URL) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: profileURL, withIntermediateDirectories: true)
        for subdirectory in [
            "memories",
            "sessions",
            "skills",
            "skins",
            "logs",
            "plans",
            "workspace",
            "cron",
        ] {
            try fileManager.createDirectory(
                at: profileURL.appendingPathComponent(subdirectory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: profileURL.path)
    }

    private func writeIfChanged(
        _ text: String,
        to url: URL,
        permissions: Int,
        backupReason: String
    ) throws -> WriteResult {
        let fileManager = FileManager.default
        let data = Data(text.utf8)
        let existingData = try? Data(contentsOf: url)
        if existingData == data {
            try? fileManager.setAttributes([.posixPermissions: permissions], ofItemAtPath: url.path)
            return WriteResult(didChange: false, backupPath: nil)
        }

        var backupURL: URL?
        if existingData != nil {
            let backup = uniqueBackupURL(for: url, reason: backupReason)
            try fileManager.copyItem(at: url, to: backup)
            backupURL = backup
        }
        try data.write(to: url, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: permissions], ofItemAtPath: url.path)
        return WriteResult(didChange: true, backupPath: backupURL?.path)
    }

    private func mirrorRootChannelDirectory(into profileURL: URL) throws -> Bool {
        let rootDirectoryURL = hermesHome.appendingPathComponent("channel_directory.json")
        let profileDirectoryURL = profileURL.appendingPathComponent("channel_directory.json")
        let fileManager = FileManager.default

        guard fileManager.fileExists(atPath: rootDirectoryURL.path) else {
            if fileManager.fileExists(atPath: profileDirectoryURL.path) {
                try fileManager.removeItem(at: profileDirectoryURL)
                return true
            }
            return false
        }

        let rootData = try Data(contentsOf: rootDirectoryURL)
        let existingData = try? Data(contentsOf: profileDirectoryURL)
        if existingData == rootData {
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: profileDirectoryURL.path)
            return false
        }
        try rootData.write(to: profileDirectoryURL, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: profileDirectoryURL.path)
        return true
    }

    private func uniqueBackupURL(for url: URL, reason: String) -> URL {
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

    private struct HermesEffectiveProfileConfiguration {
        let provider: String
        let baseURL: String
        let modelReference: String
    }

    /// Reads only the routing fields from a profile. This deliberately does
    /// not try to repair malformed user configuration: callers need a stable
    /// unavailable state instead of silently treating a broken profile as an
    /// external one.
    private static func effectiveProfileConfiguration(
        at profileURL: URL
    ) -> HermesEffectiveProfileConfiguration? {
        let configURL = profileURL.appendingPathComponent("config.yaml")
        guard let configText = try? String(contentsOf: configURL, encoding: .utf8) else {
            return nil
        }
        let document = parseTopLevelBlocks(configText)
        guard let modelBlock = document.blocks.first(where: { $0.keyName == "model" }) else {
            return nil
        }
        var values: [String: String] = [:]
        for child in directChildBlocks(of: modelBlock) {
            guard
                values[child.key] == nil,
                let value = yamlScalarValue(in: child.lines.first ?? "")
            else {
                return nil
            }
            values[child.key] = value
        }
        guard
            let provider = values["provider"], !provider.isEmpty,
            let modelReference = values["default"], !modelReference.isEmpty
        else {
            return nil
        }

        let envURL = profileURL.appendingPathComponent(".env")
        let envText = try? String(contentsOf: envURL, encoding: .utf8)
        let baseURL = values["base_url"]
            ?? envText.flatMap { dotenvValue("CUSTOM_BASE_URL", in: $0) }
        guard let baseURL, !baseURL.isEmpty else { return nil }
        return HermesEffectiveProfileConfiguration(
            provider: provider,
            baseURL: baseURL,
            modelReference: modelReference
        )
    }

    private static func yamlScalarValue(in line: String) -> String? {
        guard let colon = line.firstIndex(of: ":") else { return nil }
        var value = String(line[line.index(after: colon)...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return nil }
        if value.hasPrefix("\"") || value.hasPrefix("'") {
            guard value.count >= 2, value.first == value.last else { return nil }
            value = String(value.dropFirst().dropLast())
        } else if let commentStart = value.firstIndex(of: "#") {
            value = String(value[..<commentStart]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !value.isEmpty, !value.hasPrefix("["), !value.hasPrefix("{") else { return nil }
        return value
    }

    private static func isValidEmbeddedLaunchID(_ value: String) -> Bool {
        value.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil
    }

    private static func configYAML(
        modelID: String,
        baseURL: String,
        apiKey: String,
        workspacePath: String,
        showReasoning: Bool,
        reasoningEffort: String?
    ) -> String {
        let effortLine = reasoningEffort.map { "  reasoning_effort: \(yamlQuote($0))\n" } ?? ""
        let showReasoningText = showReasoning ? "true" : "false"
        return """
        model:
          default: \(yamlQuote(modelID))
          provider: custom
          base_url: \(yamlQuote(baseURL))
          api_key: \(yamlQuote(apiKey))
          api_mode: chat_completions
        """ + "\n" + effortLine + """
        toolsets:
          - terminal
          - file
          - web
          - browser
          - messaging
        agent:
          system_prompt: \(yamlQuote(systemPrompt))
          max_turns: 200
          tool_use_enforcement: auto
        terminal:
          backend: local
          cwd: \(yamlQuote(workspacePath))
          timeout: 180
          persistent_shell: true
        display:
          streaming: true
          show_reasoning: \(showReasoningText)
          tool_progress: all
        """ + "\n"
    }

    // MARK: - Profile config merge (issue #131)

    /// One top-level block of a YAML document, kept as raw text: the key
    /// line plus every line that belongs under it, and the comment/blank
    /// lines that directly precede it.
    struct YAMLTopLevelBlock {
        var leadingLines: [String]
        var keyName: String
        var lines: [String]
    }

    struct YAMLTopLevelDocument {
        var preamble: [String]
        var blocks: [YAMLTopLevelBlock]
        var trailing: [String]
    }

    /// Children the app owns under a template section even when the current
    /// template does not emit them — conditional lines must be able to
    /// disappear instead of being resurrected as "user content". Today that
    /// is only `model.reasoning_effort` (emitted only while an effort is
    /// configured).
    static let conditionallyOwnedChildKeys: [String: Set<String>] = [
        "model": ["reasoning_effort"]
    ]

    /// Merge the generated template over the existing profile config.
    ///
    /// The app owns the template's top-level sections and their child keys;
    /// those are rewritten every sync. Everything else — unknown top-level
    /// sections (`memory:`, `providers:`, …), unknown child keys under owned
    /// sections (`model.max_tokens`), comments, blank lines — is preserved
    /// byte-for-byte. Sequence-shaped owned sections (`toolsets:`) are
    /// rewritten wholly. The merge is idempotent, so an unchanged
    /// configuration produces an unchanged file and `writeIfChanged` skips
    /// the rewrite (and its backup) entirely.
    ///
    /// The child-key scan assumes the template's own two-space indentation,
    /// which is what the app has always written; user files started from our
    /// template keep that shape.
    static func mergedConfigYAML(existing: String?, template: String) -> String {
        guard
            let existing,
            !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return template
        }
        let templateDoc = parseTopLevelBlocks(template)
        let existingDoc = parseTopLevelBlocks(existing)
        // Nothing recognizable to preserve: the template wins (the caller's
        // writeIfChanged still snapshots a backup of the old file).
        guard !existingDoc.blocks.isEmpty else { return template }

        let templateKeys = Set(templateDoc.blocks.map(\.keyName))
        var out: [String] = existingDoc.preamble
        for templateBlock in templateDoc.blocks {
            let existingBlock = existingDoc.blocks.first {
                $0.keyName == templateBlock.keyName
            }
            out += existingBlock?.leadingLines ?? []
            out += templateBlock.lines
            guard let existingBlock else { continue }
            let ownedChildKeys = Set(directChildBlocks(of: templateBlock).map(\.key))
                .union(Self.conditionallyOwnedChildKeys[templateBlock.keyName] ?? [])
            // A template section without mapping children is sequence- or
            // scalar-shaped; the app owns its full body.
            guard !ownedChildKeys.isEmpty else { continue }
            for child in directChildBlocks(of: existingBlock)
            where !ownedChildKeys.contains(child.key) {
                out += child.lines
            }
        }
        for existingBlock in existingDoc.blocks
        where !templateKeys.contains(existingBlock.keyName) {
            out += existingBlock.leadingLines
            out += existingBlock.lines
        }
        out += existingDoc.trailing
        return out.joined(separator: "\n") + "\n"
    }

    static func parseTopLevelBlocks(_ text: String) -> YAMLTopLevelDocument {
        var preamble: [String] = []
        var blocks: [YAMLTopLevelBlock] = []
        var pending: [String] = []
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        for line in lines {
            if let key = topLevelKeyName(of: line) {
                blocks.append(
                    YAMLTopLevelBlock(leadingLines: pending, keyName: key, lines: [line])
                )
                pending = []
            } else if line.isEmpty || line.hasPrefix("#") {
                // Could belong to the current block or introduce the next
                // one; decided when the following line arrives (or at EOF).
                if blocks.isEmpty { preamble.append(line) } else { pending.append(line) }
            } else if var last = blocks.popLast() {
                // Indented content or a column-zero continuation (sequence
                // items, flow scalars): body of the current block, together
                // with any buffered comment/blank lines above it.
                last.lines += pending
                pending = []
                last.lines.append(line)
                blocks.append(last)
            } else {
                preamble += pending
                pending = []
                preamble.append(line)
            }
        }
        return YAMLTopLevelDocument(preamble: preamble, blocks: blocks, trailing: pending)
    }

    static func topLevelKeyName(of line: String) -> String? {
        guard
            let first = line.first,
            first != " ", first != "\t", first != "#", first != "-", first != "%"
        else {
            return nil
        }
        guard let colon = line.firstIndex(of: ":") else { return nil }
        let key = String(line[line.startIndex..<colon])
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        return key.isEmpty ? nil : key
    }

    static func directChildBlocks(
        of block: YAMLTopLevelBlock
    ) -> [(key: String, lines: [String])] {
        var children: [(key: String, lines: [String])] = []
        for line in block.lines.dropFirst() {
            if let key = directChildKeyName(of: line) {
                children.append((key, [line]))
            } else if !children.isEmpty {
                children[children.count - 1].lines.append(line)
            }
        }
        return children
    }

    static func directChildKeyName(of line: String) -> String? {
        guard line.hasPrefix("  "), !line.hasPrefix("   ") else { return nil }
        let rest = String(line.dropFirst(2))
        guard
            let first = rest.first,
            first != " ", first != "\t", first != "#", first != "-"
        else {
            return nil
        }
        guard let colon = rest.firstIndex(of: ":") else { return nil }
        let key = String(rest[rest.startIndex..<colon])
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        return key.isEmpty ? nil : key
    }

    private static func dotenv(
        modelID: String,
        baseURL: String,
        apiKey: String,
        workspacePath: String,
        autoApprove: Bool,
        reasoning: String,
        reasoningEffort: String?,
        messagingEnvironment: [String: String] = [:]
    ) -> String {
        let bridgeText = messagingEnvironment
            .sorted { $0.key < $1.key }
            .map { key, value in "\(key)=\(dotenvQuote(value))" }
            .joined(separator: "\n")
        var text = """
        OPENAI_BASE_URL=\(dotenvQuote(baseURL))
        CUSTOM_BASE_URL=\(dotenvQuote(baseURL))
        OPENAI_API_KEY=\(dotenvQuote(apiKey))
        HERMES_MODEL=\(dotenvQuote(modelID))
        HERMES_INFERENCE_MODEL=\(dotenvQuote(modelID))
        HERMES_INFERENCE_PROVIDER=custom
        HERMES_MTPLX_REASONING=\(dotenvQuote(reasoning))
        HERMES_MTPLX_SHOW_REASONING=\(reasoning == "off" ? "0" : "1")
        HERMES_YOLO_MODE=\(autoApprove ? "1" : "0")
        HERMES_MTPLX_TOOLSETS=\(dotenvQuote(codingToolsets))
        HERMES_MTPLX_CAPABILITIES=\(dotenvQuote(capabilitySummary))
        HERMES_MTPLX_MESSAGING_NOTE=\(dotenvQuote(messagingSetupHint))
        HERMES_MTPLX_GATEWAY_STATUS_COMMAND=\(dotenvQuote(gatewayStatusCommand))
        HERMES_MTPLX_GATEWAY_TRUTH_NOTE=\(dotenvQuote(gatewayTruthHint))
        HERMES_WORKSPACE=\(dotenvQuote(workspacePath))
        TERMINAL_CWD=\(dotenvQuote(workspacePath))
        """
        if let reasoningEffort {
            text += "HERMES_MTPLX_REASONING_EFFORT=\(dotenvQuote(reasoningEffort))\n"
        }
        if !bridgeText.isEmpty {
            text += "\n" + bridgeText
        }
        return text + "\n"
    }

    private static func reasoningMode(for configuration: MTPLXAppConfiguration) -> String {
        guard let raw = configuration.reasoning?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        else {
            return "auto"
        }
        switch raw {
        case "on", "off":
            return raw
        default:
            return "auto"
        }
    }

    private static func reasoningEffort(for configuration: MTPLXAppConfiguration) -> String? {
        let raw = configuration.reasoningEffort?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch raw {
        case "low", "medium", "high":
            return raw
        default:
            return nil
        }
    }

    private static func terminalLaunchCommand(
        hermesExecutable: String,
        profileName: String,
        modelID: String,
        autoApprove: Bool
    ) -> String {
        var parts = [
            shellQuote(hermesExecutable),
            "-p",
            shellQuote(profileName),
            "chat",
            "--model",
            shellQuote(modelID),
            "--toolsets",
            shellQuote(codingToolsets),
        ]
        if autoApprove {
            parts.append("--yolo")
        }
        parts.append(contentsOf: [
            "--source",
            shellQuote("mtplx-app"),
        ])
        return parts.joined(separator: " ")
    }

    private func writeTerminalCommandFile(
        command: String,
        hermesExecutablePath: String,
        configuration: MTPLXAppConfiguration
    ) throws -> URL {
        let directory = terminalCommandURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let profileURL = hermesHome
            .appendingPathComponent("profiles", isDirectory: true)
            .appendingPathComponent(Self.profileName, isDirectory: true)
        let env = launchEnvironment(configuration: configuration)
        let workspacePath = env["HERMES_WORKSPACE"] ?? Self.resolvedWorkspacePath(
            configuration: configuration,
            environment: environment
        )
        let script = """
        #!/bin/zsh
        cd \(Self.shellQuote(workspacePath))
        print -r -- \(Self.shellQuote("MTPLX Hermes tools: \(Self.codingToolsets)"))
        print -r -- \(Self.shellQuote(Self.messagingSetupHint))
        print -r -- \(Self.shellQuote("MTPLX Hermes messaging: \(env["HERMES_MTPLX_MESSAGING_SUMMARY"] ?? "")"))
        if [[ -n \(Self.shellQuote(env["HERMES_MTPLX_MESSAGING_WARNINGS"] ?? "")) ]]; then
          print -r -- \(Self.shellQuote("MTPLX Hermes warning: \(env["HERMES_MTPLX_MESSAGING_WARNINGS"] ?? "")"))
        fi
        _mtplx_hermes_exe=\(Self.shellQuote(hermesExecutablePath))
        _mtplx_messaging_summary=\(Self.shellQuote(env["HERMES_MTPLX_MESSAGING_SUMMARY"] ?? ""))
        if [[ "$_mtplx_messaging_summary" != *"No root Hermes messaging platform is configured."* && "$_mtplx_messaging_summary" != *"No Hermes root .env found for messaging setup."* ]]; then
          _mtplx_gateway_status="$(env -u HERMES_HOME "$_mtplx_hermes_exe" gateway status 2>&1)"
          if [[ "$_mtplx_gateway_status" == *"not loaded"* || "$_mtplx_gateway_status" == *"stale relative"* || -z "$_mtplx_gateway_status" ]]; then
            print -r -- "MTPLX Hermes gateway: repairing via hermes gateway start"
            env -u HERMES_HOME "$_mtplx_hermes_exe" gateway start >/dev/null 2>&1
            _mtplx_gateway_status="$(env -u HERMES_HOME "$_mtplx_hermes_exe" gateway status 2>&1)"
          fi
          if [[ "$_mtplx_gateway_status" == *"Gateway service is loaded"* ]]; then
            print -r -- "MTPLX Hermes gateway: loaded for messaging"
          else
            print -r -- "MTPLX Hermes gateway warning: ${_mtplx_gateway_status//$'\\n'/; }"
          fi
        fi
        _mtplx_root_channel_directory=\(Self.shellQuote(hermesHome.appendingPathComponent("channel_directory.json").path))
        _mtplx_profile_channel_directory=\(Self.shellQuote(profileURL.appendingPathComponent("channel_directory.json").path))
        if [[ -f "$_mtplx_root_channel_directory" ]]; then
          cp "$_mtplx_root_channel_directory" "$_mtplx_profile_channel_directory"
          chmod 600 "$_mtplx_profile_channel_directory" 2>/dev/null || true
        fi
        export HERMES_HOME=\(Self.shellQuote(profileURL.path))
        export OPENAI_BASE_URL=\(Self.shellQuote(env["OPENAI_BASE_URL"] ?? ""))
        export CUSTOM_BASE_URL=\(Self.shellQuote(env["CUSTOM_BASE_URL"] ?? ""))
        export OPENAI_API_KEY=\(Self.shellQuote(env["OPENAI_API_KEY"] ?? ""))
        export HERMES_MODEL=\(Self.shellQuote(env["HERMES_MODEL"] ?? ""))
        export HERMES_INFERENCE_MODEL=\(Self.shellQuote(env["HERMES_INFERENCE_MODEL"] ?? ""))
        export HERMES_INFERENCE_PROVIDER=custom
        export HERMES_YOLO_MODE=\(Self.shellQuote(env["HERMES_YOLO_MODE"] ?? "1"))
        export HERMES_MTPLX_TOOLSETS=\(Self.shellQuote(env["HERMES_MTPLX_TOOLSETS"] ?? Self.codingToolsets))
        export HERMES_MTPLX_CAPABILITIES=\(Self.shellQuote(env["HERMES_MTPLX_CAPABILITIES"] ?? Self.capabilitySummary))
        export HERMES_MTPLX_MESSAGING_NOTE=\(Self.shellQuote(env["HERMES_MTPLX_MESSAGING_NOTE"] ?? Self.messagingSetupHint))
        export HERMES_MTPLX_GATEWAY_STATUS_COMMAND=\(Self.shellQuote(env["HERMES_MTPLX_GATEWAY_STATUS_COMMAND"] ?? Self.gatewayStatusCommand))
        export HERMES_MTPLX_GATEWAY_TRUTH_NOTE=\(Self.shellQuote(env["HERMES_MTPLX_GATEWAY_TRUTH_NOTE"] ?? Self.gatewayTruthHint))
        export HERMES_MTPLX_MESSAGING_SUMMARY=\(Self.shellQuote(env["HERMES_MTPLX_MESSAGING_SUMMARY"] ?? ""))
        export HERMES_MTPLX_MESSAGING_WARNINGS=\(Self.shellQuote(env["HERMES_MTPLX_MESSAGING_WARNINGS"] ?? ""))
        export HERMES_SESSION_PLATFORM=\(Self.shellQuote(env["HERMES_SESSION_PLATFORM"] ?? ""))
        export HERMES_WORKSPACE=\(Self.shellQuote(workspacePath))
        export TERMINAL_CWD=\(Self.shellQuote(workspacePath))
        exec \(command)
        """ + "\n"
        try script.write(to: terminalCommandURL, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: terminalCommandURL.path
        )
        return terminalCommandURL
    }

    private struct LocalMessagingStatus {
        var summaries: [String]
        var warnings: [String]
    }

    private static let messagingBridgeKeys = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "TELEGRAM_REPLY_TO_MODE",
        "TELEGRAM_PROXY",
        "TELEGRAM_FALLBACK_IPS",
        "DISCORD_BOT_TOKEN",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_BOT_TOKEN",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "WHATSAPP_ENABLED",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
    ]

    private func localMessagingStatus() -> LocalMessagingStatus {
        let envURL = hermesHome.appendingPathComponent(".env")
        guard let data = try? Data(contentsOf: envURL) else {
            return LocalMessagingStatus(
                summaries: ["No Hermes root .env found for messaging setup."],
                warnings: []
            )
        }

        var warnings: [String] = []
        if String(data: data, encoding: .utf8) == nil {
            warnings.append("Hermes root .env has invalid UTF-8; some Hermes status/tools commands may fail.")
        }
        let text = String(decoding: data, as: UTF8.self)
        var configured: [String] = []
        if Self.dotenvHasValue("TELEGRAM_BOT_TOKEN", in: text) {
            configured.append(
                Self.dotenvHasValue("TELEGRAM_HOME_CHANNEL", in: text)
                    ? "Telegram configured with a home channel."
                    : "Telegram configured; no home channel set."
            )
        }
        if Self.dotenvHasValue("DISCORD_BOT_TOKEN", in: text) {
            configured.append("Discord configured.")
        }
        if Self.dotenvHasValue("SLACK_BOT_TOKEN", in: text) {
            configured.append("Slack configured.")
        }
        if Self.dotenvValue("WHATSAPP_ENABLED", in: text)?.lowercased() == "true" {
            configured.append("WhatsApp enabled.")
        }

        return LocalMessagingStatus(
            summaries: configured.isEmpty
                ? ["No root Hermes messaging platform is configured."]
                : configured,
            warnings: warnings
        )
    }

    private func localMessagingBridgeEnvironment() -> [String: String] {
        let envURL = hermesHome.appendingPathComponent(".env")
        guard let data = try? Data(contentsOf: envURL) else { return [:] }
        let text = String(decoding: data, as: UTF8.self)

        var bridged: [String: String] = [:]
        for key in Self.messagingBridgeKeys {
            guard let value = Self.dotenvValue(key, in: text),
                  !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { continue }
            bridged[key] = value
        }
        if !bridged.isEmpty {
            bridged["HERMES_SESSION_PLATFORM"] = "mtplx-app"
        }
        return bridged
    }

    private static func updateSummary(fromVersionOutput output: String) -> String? {
        output
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { $0.lowercased().contains("update available") }
    }

    private static func gatewaySummary(fromStatusOutput output: String) -> String? {
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        var parts: [String] = []
        if text.localizedCaseInsensitiveContains("Gateway service is loaded") {
            parts.append("Gateway service loaded")
        }
        if let pid = launchctlValue("PID", in: text) {
            parts.append("PID \(pid)")
        }
        if text.localizedCaseInsensitiveContains("stale relative") {
            parts.append("service definition stale")
        }
        if text.localizedCaseInsensitiveContains("not loaded") {
            parts.append("Gateway service not loaded")
        }
        if parts.isEmpty {
            return text
                .split(separator: "\n")
                .prefix(2)
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: "; ")
        }
        return parts.joined(separator: "; ")
    }

    static func gatewayHealth(fromStatusOutput output: String) -> HermesInstallStatus.GatewayHealth? {
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if text.localizedCaseInsensitiveContains("not loaded") {
            return .unavailable
        }
        if text.localizedCaseInsensitiveContains("stale relative") {
            return .warning
        }
        if text.localizedCaseInsensitiveContains("Gateway service is loaded") {
            return .healthy
        }
        return .warning
    }

    static func gatewayWarnings(fromStatusOutput output: String) -> [String] {
        let text = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        var warnings: [String] = []
        if text.localizedCaseInsensitiveContains("stale relative") {
            warnings.append("Hermes Gateway LaunchAgent is stale; run `hermes gateway start` before relying on messaging.")
        }
        if text.localizedCaseInsensitiveContains("not loaded") {
            warnings.append("Hermes Gateway is not loaded; run `hermes gateway start` before using messaging.")
        }
        return warnings
    }

    private static func gatewayRepairSummary(fromStartOutput output: String) -> String {
        let lines = output
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else {
            return "Hermes Gateway start command completed."
        }
        if let loaded = lines.first(where: { $0.localizedCaseInsensitiveContains("loaded") }) {
            return loaded
        }
        if let started = lines.first(where: { $0.localizedCaseInsensitiveContains("start") }) {
            return started
        }
        return lines.prefix(2).joined(separator: "; ")
    }

    private static func launchctlValue(_ key: String, in text: String) -> String? {
        let marker = "\"\(key)\" = "
        guard let markerRange = text.range(of: marker) else { return nil }
        let tail = text[markerRange.upperBound...]
        let end = tail.firstIndex(where: { $0 == ";" || $0.isNewline }) ?? tail.endIndex
        return tail[..<end]
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"").union(.whitespacesAndNewlines))
    }

    private static func dotenvHasValue(_ key: String, in text: String) -> Bool {
        guard let value = dotenvValue(key, in: text) else { return false }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func dotenvValue(_ key: String, in text: String) -> String? {
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = String(rawLine).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            if line.hasPrefix("export ") {
                line = String(line.dropFirst("export ".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let equals = line.firstIndex(of: "="),
                  String(line[..<equals]) == key
            else { continue }
            var value = String(line[line.index(after: equals)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            } else if value.hasPrefix("'"), value.hasSuffix("'"), value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value
        }
        return nil
    }

    private func resolveExecutable() -> URL? {
        if let executablePath, FileManager.default.isExecutableFile(atPath: executablePath) {
            return URL(fileURLWithPath: executablePath)
        }
        for path in searchPaths() {
            let candidate = URL(fileURLWithPath: path).appendingPathComponent("hermes")
            if FileManager.default.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }

    private func searchPaths() -> [String] {
        var paths = (environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
        let home = environment["HOME"] ?? NSHomeDirectory()
        for extra in [
            "\(home)/.local/bin",
            "\(home)/.cargo/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ] where !paths.contains(extra) {
            paths.append(extra)
        }
        return paths
    }

    public static func resolvedWorkspacePath(
        configuration: MTPLXAppConfiguration,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let configured = MTPLXAppConfiguration.normalizedHermesWorkspacePath(
            configuration.hermesWorkspacePath
        )
        if isDirectory(configured) {
            return configured
        }

        let fallback = MTPLXAppConfiguration.defaultHermesWorkspacePath()
        if isDirectory(fallback) {
            return fallback
        }

        return environment["HOME"] ?? NSHomeDirectory()
    }

    private static func isDirectory(_ path: String) -> Bool {
        var isDirectory = ObjCBool(false)
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    private func waitForDashboardToken(port: Int) async throws -> String {
        let url = URL(string: "http://127.0.0.1:\(port)/")!
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            if let (data, _) = try? await URLSession.shared.data(from: url),
               let html = String(data: data, encoding: .utf8),
               let token = Self.extractDashboardToken(from: html) {
                return token
            }
            try? await Task.sleep(for: .milliseconds(150))
        }
        throw HermesIntegrationError.dashboardTokenTimeout
    }

    private func runAndCapture(
        executableURL: URL,
        arguments: [String],
        environment overrideEnvironment: [String: String]? = nil,
        timeout: TimeInterval = 60
    ) async throws -> String {
        let processEnvironment = overrideEnvironment ?? environment
        return try await Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = executableURL
            process.arguments = arguments
            process.environment = processEnvironment
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            // Drain as data arrives and bound the wait (#158): the old
            // read-after-exit deadlocked once `hermes` output crossed
            // the 64KB pipe buffer, and a wedged gateway command hung
            // installStatus()/repairGateway() forever. Callers parse
            // this output, so the drain joins on EOF rather than
            // racing a readabilityHandler against termination.
            let watchdog = SubprocessWatchdog(process)
            try process.run()
            let output = SubprocessPipeDrain(pipe)
            guard watchdog.wait(for: process, timeout: timeout) else {
                output.join(timeout: 1)
                throw HermesIntegrationError.launchFailed(
                    output.snapshot()
                        + "\n[hermes \(arguments.joined(separator: " ")) timed out after \(Int(timeout))s and was terminated]"
                )
            }
            output.join()
            let text = output.snapshot()
            guard process.terminationStatus == 0 else {
                throw HermesIntegrationError.launchFailed(text)
            }
            return text
        }.value
    }

    private func gatewayCommandEnvironment() -> [String: String] {
        var env = environment
        env.removeValue(forKey: "HERMES_HOME")
        return env
    }

    static func isAppLaunchedTerminalAgentCommand(_ command: String) -> Bool {
        let text = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return false }
        let parts = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let hermesIndex: Int?
        if parts.first.map({ URL(fileURLWithPath: $0).lastPathComponent }) == "hermes" {
            hermesIndex = 0
        } else if parts.count > 1,
                  parts.first.map({ URL(fileURLWithPath: $0).lastPathComponent.hasPrefix("python") }) == true,
                  URL(fileURLWithPath: parts[1]).lastPathComponent == "hermes" {
            hermesIndex = 1
        } else {
            hermesIndex = nil
        }
        guard let hermesIndex else {
            return false
        }
        let arguments = " \(parts.dropFirst(hermesIndex + 1).joined(separator: " ")) "
        return arguments.contains(" chat ")
            && arguments.contains(" -p mtplx ")
            && arguments.contains(" --source mtplx-app ")
    }

    private static func appLaunchedTerminalAgentPIDs() -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-axo", "pid=,command="]
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-hermes-ps-\(UUID().uuidString).txt")
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        guard let outputHandle = try? FileHandle(forWritingTo: outputURL) else { return [] }
        defer {
            try? outputHandle.close()
            try? FileManager.default.removeItem(at: outputURL)
        }
        process.standardOutput = outputHandle
        let exitDone = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in
            exitDone.signal()
        }
        do {
            try process.run()
        } catch {
            return []
        }
        if exitDone.wait(timeout: .now() + 1.0) == .timedOut {
            process.terminate()
            if exitDone.wait(timeout: .now() + 0.5) == .timedOut {
                return []
            }
        }
        try? outputHandle.synchronize()
        let outputData = (try? Data(contentsOf: outputURL)) ?? Data()
        guard let output = String(data: outputData, encoding: .utf8) else { return [] }
        let currentPID = getpid()
        return output.split(separator: "\n").compactMap { row -> pid_t? in
            let text = String(row).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let firstSpace = text.firstIndex(where: { $0.isWhitespace }) else {
                return nil
            }
            guard let pid = pid_t(text[..<firstSpace]),
                  pid > 1,
                  pid != currentPID
            else {
                return nil
            }
            let command = String(text[firstSpace...])
            return isAppLaunchedTerminalAgentCommand(command) ? pid : nil
        }
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

    private static func projectURL(fromVersionOutput output: String) -> URL? {
        for line in output.split(separator: "\n") {
            let text = String(line)
            guard text.hasPrefix("Project:") else { continue }
            let path = text
                .dropFirst("Project:".count)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else { return nil }
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return nil
    }

    private static func projectSupportsGatewayWebSocket(_ projectURL: URL) -> Bool {
        let candidates = [
            projectURL.appendingPathComponent("tui_gateway/ws.py"),
            projectURL.appendingPathComponent("hermes_cli/web_server.py"),
        ]
        return candidates.contains { url in
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                return false
            }
            return text.contains("/api/ws")
                && text.contains("handle_ws")
        }
    }

    private static func dashboardHelpSupportsGateway(_ help: String) -> Bool {
        help.contains("--tui")
            && help.contains("--no-open")
            && help.contains("--port")
    }

    static func extractDashboardToken(from html: String) -> String? {
        let marker = "window.__HERMES_SESSION_TOKEN__=\""
        guard let start = html.range(of: marker)?.upperBound else { return nil }
        guard let end = html[start...].firstIndex(of: "\"") else { return nil }
        let token = String(html[start..<end])
        return token.isEmpty ? nil : token
    }

    private static func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }

    private static func yamlQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "''"))'"
    }

    private static func dotenvQuote(_ value: String) -> String {
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        return "\"\(escaped)\""
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }

    private static func isValidProfileName(_ name: String) -> Bool {
        guard !name.isEmpty else { return false }
        return name.allSatisfy { character in
            character.isLowercase
                || character.isNumber
                || character == "-"
                || character == "_"
        }
    }

    private static func freeLoopbackPort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw HermesIntegrationError.launchFailed("could not allocate a socket") }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(0).bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bindResult = withUnsafePointer(to: &addr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw HermesIntegrationError.launchFailed("could not bind a local dashboard port")
        }

        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &addr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &len)
            }
        }
        guard nameResult == 0 else {
            throw HermesIntegrationError.launchFailed("could not resolve a local dashboard port")
        }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}
