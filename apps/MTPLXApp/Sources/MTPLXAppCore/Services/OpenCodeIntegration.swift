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
    public let sessionHeadersPluginPath: String
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
    /// A PID is exposed only when LaunchServices created a process that was
    /// not already present before this invocation. It is diagnostic only;
    /// stale cleanup needs the PID-plus-launch-date identity below.
    public let launchedProcessID: Int?
    public let launchedDesktopIdentity: MTPLXDesktopHandoffIdentity?

    public init(
        action: OpenCodeDesktopAction,
        wasRunning: Bool,
        didTerminateExistingInstance: Bool,
        didOpen: Bool,
        detail: String,
        launchedProcessID: Int? = nil,
        launchedDesktopIdentity: MTPLXDesktopHandoffIdentity? = nil
    ) {
        self.action = action
        self.wasRunning = wasRunning
        self.didTerminateExistingInstance = didTerminateExistingInstance
        self.didOpen = didOpen
        self.detail = detail
        self.launchedProcessID = launchedProcessID
        self.launchedDesktopIdentity = launchedDesktopIdentity
    }
}

private struct OpenCodeReasoningVisibilityResult: Equatable, Sendable {
    let path: String
    let didChange: Bool
    let backupPath: String?
}

public struct OpenCodeIntegration: Sendable {
    private static let desktopGlobalStoreName = "opencode.global.dat"
    /// The managed plugin is the package the CLI installs
    /// (`mtplx.opencode.write_opencode_session_headers_plugin`), byte for
    /// byte, at the same place: `plugins/mtplx-session-headers/` beside
    /// opencode.json, with `package.json`, the V1 entrypoint `index.js` and
    /// the V2 entrypoint `server.js`. OpenCode 1.x imports the package main
    /// and calls its `chat.headers`/`chat.params` hooks; OpenCode 2.x
    /// resolves `./server` and calls only `model.request`, so a single-file
    /// V1 plugin did nothing there. Both writers compare content before
    /// rewriting, so the app and the CLI never fight over the files.
    /// `OpenCodePluginParityTests` reads the CLI's sources at test time and
    /// fails when the two drift.
    static let sessionHeadersPackageName = "mtplx-session-headers"
    /// The single-file plugin earlier app versions wrote next to
    /// opencode.json. Its registration is replaced by the package so the
    /// V1 hooks never run twice; the file itself is inert once unregistered.
    static let legacySessionHeadersPluginName = "mtplx-session-headers.js"

    /// `index.js` (`mtplx.opencode.OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE`).
    /// It carries the session headers and strips exactly the values
    /// OpenCode 1.x injects: its output ceiling (provider/transform.ts
    /// maxOutputTokens, min(limit.output, 32,000) on every request, which on
    /// a small window is the reply reserve `outputLimit(forContextWindow:)`
    /// writes) and the qwen-keyed sampler OpenCode <= 1.18.20 sends. MTPLX
    /// then owns the uncapped generation contract while every explicit
    /// client choice passes through untouched, and an answer cap the user
    /// gave the CLI (the model's `x-mtplx-max-response-tokens` header) is
    /// sent whole.
    static let sessionHeadersPluginSource = """
    const mtplxProviderID = (input) =>
      input?.model?.providerID || input?.provider?.id;

    const mtplxInjectedOutputCap = 32000;
    const mtplxRequestedOutputHeader = "x-mtplx-max-response-tokens";
    const mtplxInjectedQwenTemperature = 0.55;
    const mtplxInjectedQwenTopP = 1;

    export const MTPLXSessionHeaders = async () => ({
      "chat.headers": async (input, output) => {
        output.headers ||= {};
        const providerID = mtplxProviderID(input);
        if (providerID && providerID !== "mtplx") return;
        output.headers["x-mtplx-client"] = "opencode";
        if (input?.sessionID) {
          output.headers["x-mtplx-session-id"] = String(input.sessionID);
        }
        if (input?.message?.id) {
          output.headers["x-mtplx-client-turn-id"] = String(input.message.id);
        }
      },
      "chat.params": async (input, output) => {
        const providerID = mtplxProviderID(input);
        if (providerID && providerID !== "mtplx") return;
        // OpenCode sends maxOutputTokens = min(limit.output, 32000) with every
        // request and hands this hook the model with its limit and headers.
        // MTPLX writes limit.output as OpenCode's reply reserve, half the window
        // and at most 32,000 (#480), so by default that value is OpenCode's own
        // and is stripped: the server's own limits apply. An answer cap the user
        // asked MTPLX for (--max-response-tokens) is the model's
        // x-mtplx-max-response-tokens header and is sent whole, past OpenCode's
        // 32,000 ceiling; a limit.output that is not MTPLX's reserve is sent the
        // same way. Any other value is a cap set in OpenCode itself and passes
        // through untouched.
        const positive = (value) => (Number.isInteger(value) && value > 0 ? value : null);
        const limit = input?.model?.limit;
        const configuredOutput = positive(limit?.output);
        const opencodeDefault = configuredOutput === null
          ? mtplxInjectedOutputCap
          : Math.min(configuredOutput, mtplxInjectedOutputCap);
        if (output.maxOutputTokens === opencodeDefault) {
          const context = positive(limit?.context);
          const mtplxReserve = context === null
            ? null
            : Math.min(mtplxInjectedOutputCap, Math.max(1, Math.floor(context / 2)));
          const requested = positive(Number(input?.model?.headers?.[mtplxRequestedOutputHeader]));
          output.maxOutputTokens = requested
            ?? (configuredOutput !== null && configuredOutput !== mtplxReserve
              ? configuredOutput
              : undefined);
        }
        // OpenCode <= 1.18.20 (Desktop 1.18.18 included) injects a qwen-keyed
        // sampler (temperature 0.55, topP 1) for any model id containing
        // "qwen"; 1.18.21 removed the rule. Strip exactly that injected pair so
        // the MTPLX server's family-native sampler applies; any other value is
        // a deliberate client choice and passes through untouched.
        const modelID = String(input?.model?.id ?? input?.model?.modelID ?? "").toLowerCase();
        if (modelID.includes("qwen")) {
          if (output.temperature === mtplxInjectedQwenTemperature) {
            output.temperature = undefined;
          }
          if (output.topP === mtplxInjectedQwenTopP) {
            output.topP = undefined;
          }
        }
      }
    });
    export default MTPLXSessionHeaders;

    """

    /// `server.js` (`mtplx.opencode.OPENCODE_SESSION_HEADERS_V2_SOURCE`): the
    /// OpenCode 2.x entrypoint. Its `model.request` hook sends the session
    /// headers and OpenCode's request kind (`x-mtplx-request-kind`), which
    /// lets the server tell a title request from the conversation it names.
    static let sessionHeadersServerSource = """
    // Older V1 imports index.js; modern V1 and V2 resolve this entrypoint.
    // No prompt, tool-schema or generation-option rewriting belongs here.
    import { MTPLXSessionHeaders } from "./index.js";
    export default {
      id: "mtplx.session-headers",
      server: MTPLXSessionHeaders,
      async setup(ctx) {
        await ctx.session.hook("model.request", (event) => {
          event.headers["x-mtplx-client"] = "opencode";
          event.headers["x-mtplx-session-id"] = String(event.sessionID);
          if (event.kind) event.headers["x-mtplx-request-kind"] = String(event.kind);
        }, { providerID: "mtplx" });
      }
    };

    """

    /// `package.json`, as the CLI's `json.dumps(..., indent=2)` writes it.
    static let sessionHeadersPackageJSON = """
    {
      "name": "mtplx-session-headers",
      "private": true,
      "type": "module",
      "main": "./index.js",
      "exports": {
        ".": "./index.js",
        "./server": "./server.js"
      }
    }

    """

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

    /// `servedWindow` is the window the running daemon published
    /// (`/health` `execution_window`); the limits follow it. Without it (the
    /// write before the daemon starts) the limits already written for this
    /// model are kept, so a launch never rewrites them twice; a model with no
    /// limits yet gets the window setting.
    @discardableResult
    public func sync(
        configuration: MTPLXAppConfiguration,
        servedWindow: ServedExecutionWindow? = nil
    ) throws -> OpenCodeConfigResult {
        let modelID = Self.modelID(for: configuration.model)
        let modelReference = "mtplx/\(modelID)"
        let baseURL = Self.baseURLString(host: configuration.host, port: configuration.port)
        let sessionHeadersPluginURL = Self.sessionHeadersPluginURL(configURL: configURL)
        var backupURL: URL?

        // A file OpenCode accepts (comments, trailing commas) is merged; one
        // that does not parse throws before anything is written.
        let existingRoot = try loadRoot()
        var root = existingRoot ?? [:]
        var providers = root["provider"]?.objectValue ?? [:]
        let limits = Self.modelLimits(
            configuration: configuration,
            servedWindow: servedWindow,
            existingProvider: providers["mtplx"],
            modelID: modelID
        )
        providers["mtplx"] = .object(
            Self.providerConfig(
                modelID: modelID,
                baseURL: baseURL,
                apiKey: configuration.apiKey,
                limits: limits,
                vision: MTPLXModelOption.supportsVision(model: configuration.model),
                reasoningEffort: Self.resolvedReasoningEffort(
                    forModelID: modelID,
                    configuredEffort: configuration.reasoningEffort
                )
            )
        )
        root["provider"] = .object(providers)
        root["model"] = .string(modelReference)
        root["small_model"] = .string(modelReference)
        _ = Self.ensureManagedSessionHeadersPlugin(
            in: &root,
            path: sessionHeadersPluginURL.path
        )
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
        let pluginFileDidChange = try Self.installSessionHeadersPlugin(
            at: sessionHeadersPluginURL
        )

        // A file that already says this is left exactly as the user wrote
        // it, comments and formatting included.
        if let existingRoot, existingRoot == root {
            let visibility = try ensureReasoningSummariesVisible()
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: configURL.path)
            return OpenCodeConfigResult(
                configPath: configURL.path,
                baseURL: baseURL,
                modelReference: modelReference,
                sessionHeadersPluginPath: sessionHeadersPluginURL.path,
                didChange: pluginFileDidChange || visibility.didChange,
                backupPath: nil,
                reasoningVisibilityPath: visibility.path,
                reasoningVisibilityDidChange: visibility.didChange,
                reasoningVisibilityBackupPath: visibility.backupPath
            )
        }

        if fileManager.fileExists(atPath: configURL.path) {
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
            sessionHeadersPluginPath: sessionHeadersPluginURL.path,
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
    public func reloadDesktopAfterDaemonReady(
        isCurrent: (() -> Bool)? = nil
    ) async -> OpenCodeDesktopResult {
        #if canImport(AppKit)
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: desktopApplicationURL.path) else {
            return OpenCodeDesktopResult(
                action: .unavailable,
                wasRunning: false,
                didTerminateExistingInstance: false,
                didOpen: false,
                detail: tr("OpenCode.app not found at %@", desktopApplicationURL.path)
            )
        }
        guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
        let preexistingProcessIDs = Set(
            NSRunningApplication
                .runningApplications(withBundleIdentifier: desktopBundleIdentifier)
                .filter { !$0.isTerminated }
                .map { Int($0.processIdentifier) }
        )
        let stateRepair = repairDesktopStateBeforeLaunch()
        guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }

        let running = NSRunningApplication
            .runningApplications(withBundleIdentifier: desktopBundleIdentifier)
            .filter { !$0.isTerminated }
        let wasRunning = !running.isEmpty
        var terminatedExisting = false

        if wasRunning {
            guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
            for app in running {
                app.terminate()
            }
            terminatedExisting = await waitUntilApplicationsExit(running, timeoutSeconds: 5)
            guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
            if !terminatedExisting {
                guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
                for app in running where !app.isTerminated {
                    app.forceTerminate()
                }
                terminatedExisting = await waitUntilApplicationsExit(running, timeoutSeconds: 2)
                guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
            }
        }

        guard isCurrent?() ?? true else { return staleDesktopHandoffResult() }
        let opened = await openDesktopApplication()
        let launchedDesktopIdentity: MTPLXDesktopHandoffIdentity?
        if opened.didOpen,
           let processID = opened.processID,
           let launchDate = opened.launchDate,
           !preexistingProcessIDs.contains(processID) {
            launchedDesktopIdentity = MTPLXDesktopHandoffIdentity(
                processID: processID,
                launchDate: launchDate
            )
        } else {
            launchedDesktopIdentity = nil
        }
        guard isCurrent?() ?? true else {
            return staleDesktopHandoffResult(launchedDesktopIdentity: launchedDesktopIdentity)
        }
        let action: OpenCodeDesktopAction
        if wasRunning {
            action = opened.didOpen ? .relaunched : .unavailable
        } else {
            action = opened.didOpen ? .opened : .unavailable
        }

        return OpenCodeDesktopResult(
            action: action,
            wasRunning: wasRunning,
            didTerminateExistingInstance: wasRunning ? terminatedExisting : false,
            didOpen: opened.didOpen,
            detail: (wasRunning
                ? "reloaded OpenCode Desktop so its sidecar re-reads MTPLX provider config"
                : "opened OpenCode Desktop")
                + (stateRepair.didChange
                   ? "; repaired \(stateRepair.removedEntries) stale OpenCode workspace state entr\(stateRepair.removedEntries == 1 ? "y" : "ies")"
                   : ""),
            launchedProcessID: launchedDesktopIdentity?.processID,
            launchedDesktopIdentity: launchedDesktopIdentity
        )
        #else
        return OpenCodeDesktopResult(
            action: .unavailable,
            wasRunning: false,
            didTerminateExistingInstance: false,
            didOpen: false,
            detail: tr("AppKit is unavailable")
        )
        #endif
    }

    private func staleDesktopHandoffResult(
        launchedDesktopIdentity: MTPLXDesktopHandoffIdentity? = nil
    ) -> OpenCodeDesktopResult {
        OpenCodeDesktopResult(
            action: .unavailable,
            wasRunning: false,
            didTerminateExistingInstance: false,
            didOpen: false,
            detail: tr("OpenCode handoff cancelled because the daemon lifecycle changed."),
            launchedProcessID: launchedDesktopIdentity?.processID,
            launchedDesktopIdentity: launchedDesktopIdentity
        )
    }

    /// Reap only the process opened by this invocation. The Store calls this
    /// when the lifecycle changes after LaunchServices returns a new PID; an
    /// already-running OpenCode instance is deliberately never targeted.
    @MainActor
    @discardableResult
    public func cancelLaunchedDesktop(_ identity: MTPLXDesktopHandoffIdentity) -> Bool {
        #if canImport(AppKit)
        guard identity.processID > 1,
              let application = NSRunningApplication
                .runningApplications(withBundleIdentifier: desktopBundleIdentifier)
                .first(where: {
                    !$0.isTerminated
                        && identity.matches(
                            processID: Int($0.processIdentifier),
                            launchDate: $0.launchDate
                        )
                })
        else { return false }
        application.terminate()
        return true
        #else
        _ = identity
        return false
        #endif
    }

    public static func modelID(for model: String) -> String {
        // Exact released identities, including legacy Bonsai folder/served
        // aliases. MiMo must match here: its names also contain "qwen" and
        // "optimized-speed", which the generic branch below claims for the
        // Qwen 3.6 27B.
        if let option = MTPLXModelOption.option(matching: model),
           [
               "flash-next-optimized-quality",
               "bonsai-2-27b-optimized-speed",
               "mimo-v26-qwen-9b-optimized-speed",
           ].contains(option.id) {
            return "mtplx-\(option.id)"
        }
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
        // Flash-Next (qwen4_exp) before the 3.8 branch: the pack names
        // carry "Qwen3.8-Flash-Next"+"bare-speed"/"optimized-speed" and
        // would otherwise be claimed by the 27B ids below (engine twin:
        // default_models._public_model_id_from_name resolves flash-next
        // first). Derivative names fall through to the sanitized id.
        if lower.contains("flash-next") || lower.contains("flash_next") {
            if lower.contains("bare-speed") {
                return "mtplx-flash-next-bare-speed"
            }
            if lower.contains("optimized-speed") {
                return "mtplx-flash-next-optimized-speed"
            }
        }
        if (lower.contains("flash-next") || lower.contains("flash_next"))
            && lower.contains("optimized-quality") {
            return sanitizedModelID(from: model)
        }
        // Qwen 3.8 family before the generic qwen branches: a 3.8 name
        // also contains "qwen"+"optimized-speed"/"optimized-quality" and
        // would otherwise be claimed by the 3.6 ids below.
        if lower.contains("qwen3.8") || lower.contains("qwen38") || lower.contains("qwen3-8") {
            // The FP16 precision siblings are served under the parent id
            // plus "-fp16" (mirrors default_models._public_model_id_from_name),
            // so the OpenCode config names the id the server advertises.
            let precision = lower.contains("-fp16") ? "-fp16" : ""
            if lower.contains("bare-speed") {
                return "mtplx-qwen38-27b-bare-speed" + precision
            }
            if lower.contains("optimized-quality") {
                return "mtplx-qwen38-27b-optimized-quality" + precision
            }
            if lower.contains("optimized-speed") {
                return "mtplx-qwen38-27b-optimized-speed" + precision
            }
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

        return sanitizedModelID(from: model)
    }

    private static func sanitizedModelID(from model: String) -> String {
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

    /// Swift twin of `descriptors.reasoning_policy_for_model` for the
    /// OpenCode config surface: nil = no verified reasoning codec, [] =
    /// reasoning without an effort dial (Qwen3.5/3.6 trunk), a list = the
    /// family effort dial.
    public static func reasoningEffortLevels(forModelID modelID: String) -> [String]? {
        let lower = modelID.lowercased()
        if MTPLXModelOption.isBonsai2Model(modelID) {
            return ["xhigh", "medium"]
        }
        // Flash-Next (qwen4_exp) before the 3.8 markers: the pack names
        // carry "Qwen3.8-Flash-Next" and would otherwise be claimed by the
        // 27B codec below (engine twin: descriptors._QWEN4_PREVIEW_MARKER
        // routes flash-next away first).
        if lower.contains("flash-next") || lower.contains("flash_next") || lower.contains("qwen4") {
            // QWEN4_EXP_REASONING_CODEC: same official effort triple.
            return ["xhigh", "medium", "low"]
        }
        if MTPLXModelOption.modelFamily(for: modelID) == "qwen3_8" {
            // QWEN3_8_REASONING_CODEC: official reasoning_effort levels.
            return ["xhigh", "medium", "low"]
        }
        if lower.contains("step") {
            return ["low", "medium", "high"]
        }
        if lower.contains("qwen") {
            return []
        }
        return nil
    }

    public static func reasoningEffort(forModelID modelID: String) -> String? {
        let lower = modelID.lowercased()
        if lower.contains("flash-next") || lower.contains("flash_next") || lower.contains("qwen4") {
            // Agent-lane default is medium (engine codec default_agent_effort):
            // the 2026-08-28 wall-clock A/B on the identical multifile coding
            // task measured xhigh 150.2s vs medium 44.2s with the same correct
            // output. Chat surfaces keep the family chat default (xhigh); the
            // OpenCode effort picker still offers xhigh per request. Checked
            // before the 3.8 markers, which the pack names also contain.
            return "medium"
        }
        if MTPLXModelOption.modelFamily(for: modelID) == "qwen3_8" {
            // QWEN3_8_REASONING_CODEC default: medium (strict max-fan A/B,
            // 2026-08-14 — same correct uncapped result 51.52s vs 314.91s
            // at xhigh).
            return "medium"
        }
        return lower.contains("step") ? "low" : nil
    }

    /// The effort OpenCode's model entry carries: the app dial when the user
    /// set one, otherwise the family default. Explicit effort choices made
    /// inside OpenCode merge after model options and win per request.
    public static func resolvedReasoningEffort(
        forModelID modelID: String,
        configuredEffort: String?
    ) -> String? {
        guard reasoningEffortLevels(forModelID: modelID) != nil else { return nil }
        if let configured = configuredEffort?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            !configured.isEmpty,
            configured != "auto"
        {
            return configured
        }
        return reasoningEffort(forModelID: modelID)
    }

    public func repairDesktopStateBeforeLaunch() -> OpenCodeDesktopStateRepairResult {
        let globalStoreURL = desktopSettingsStoreURL
            .deletingLastPathComponent()
            .appendingPathComponent(Self.desktopGlobalStoreName)
        return Self.repairDeadWorkspaceState(globalStoreURL: globalStoreURL)
    }

    /// The current `opencode.json`, read as OpenCode reads it (JSON with
    /// comments and trailing commas), or nil when there is none. A file that
    /// does not parse is left in place and reported (`ClientConfigFileError`);
    /// it used to be moved aside and replaced by MTPLX's provider alone.
    private func loadRoot() throws -> [String: JSONValue]? {
        try ClientConfigFile.readObject(at: configURL)
    }

    /// The managed plugin package: `plugins/mtplx-session-headers` beside
    /// opencode.json (`mtplx.opencode.opencode_session_headers_plugin_path`).
    static func sessionHeadersPluginURL(configURL: URL) -> URL {
        configURL.deletingLastPathComponent()
            .appendingPathComponent("plugins", isDirectory: true)
            .appendingPathComponent(sessionHeadersPackageName, isDirectory: true)
    }

    /// Register the managed plugin in the config's `plugin` list, replacing
    /// any stale registration of it under another path (the CLI's merge
    /// rule): the package or the single file earlier versions installed,
    /// since a second copy would run the V1 hooks twice.
    private static func ensureManagedSessionHeadersPlugin(
        in root: inout [String: JSONValue],
        path: String
    ) -> Bool {
        let existing: [JSONValue]
        if let current = root["plugin"] {
            if let plugins = current.arrayValue {
                existing = plugins
            } else {
                existing = [current]
            }
        } else {
            existing = []
        }
        var next = existing.filter { plugin in
            guard let pluginPath = plugin.stringValue else { return true }
            if pluginPath == path { return true }
            let name = URL(fileURLWithPath: pluginPath).lastPathComponent
            return name != sessionHeadersPackageName && name != legacySessionHeadersPluginName
        }
        if !next.contains(where: { $0.stringValue == path }) {
            next.append(.string(path))
        }
        let didChange = next != existing
        root["plugin"] = .array(next)
        return didChange
    }

    /// Write the managed plugin package. Each file is content-compared
    /// before writing, so repeat launches (and the Python `mtplx start
    /// opencode` writer, which installs the identical bytes) never churn
    /// it, and is private to the user like the CLI's copy.
    private static func installSessionHeadersPlugin(at directory: URL) throws -> Bool {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var changed = false
        for (name, source) in [
            ("package.json", sessionHeadersPackageJSON),
            ("index.js", sessionHeadersPluginSource),
            ("server.js", sessionHeadersServerSource),
        ] {
            let url = directory.appendingPathComponent(name)
            let data = Data(source.utf8)
            if let existing = try? Data(contentsOf: url), existing == data {
                continue
            }
            try data.write(to: url, options: [.atomic])
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
            changed = true
        }
        return changed
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

    /// OpenCode's built-in effort tiers for reasoning-capable
    /// openai-compatible models (sst/opencode provider/transform.ts
    /// OPENAI_EFFORTS, identical at 1.18.18 and 1.18.21). The generated
    /// config disables the tiers a family contract does not define so
    /// OpenCode's effort picker mirrors the MTPLX dial.
    private static let openCodeDefaultEffortTiers = [
        "none", "minimal", "low", "medium", "high", "xhigh",
    ]

    private static func effortVariants(familyLevels: [String]) -> [String: JSONValue] {
        var variants: [String: JSONValue] = [:]
        for tier in openCodeDefaultEffortTiers where !familyLevels.contains(tier) {
            variants[tier] = .object(["disabled": .bool(true)])
        }
        // Every family tier is declared as an explicit variant, not only the
        // ones outside OPENAI_EFFORTS: Desktop 1.18.21's picker does not
        // surface its full built-in list for a custom openai-compatible
        // provider (xhigh was missing live for the Flash-Next dial while
        // low/medium rendered). An explicit variant renders on every version
        // and merges over a same-named built-in.
        for level in familyLevels {
            variants[level] = .object(["reasoningEffort": .string(level)])
        }
        return variants
    }

    /// OpenCode plans `min(limit.output, 32_000)` of the window for the reply
    /// and compacts once a turn's total tokens reach the rest (overflow.ts,
    /// 1.18.29). Mirroring the context into the output limit left a zero-token
    /// conversation window on small seats, so every reply was summarised
    /// (issue #480). Half the window, capped at the 32,000 OpenCode injects.
    /// OpenCode also sends this reserve as every request's max_tokens; the
    /// managed plugin strips it there (`mtplxReserve`), so the server's own
    /// limits apply at every window. SYNC: mtplx/opencode.py
    /// `opencode_output_limit`.
    static func outputLimit(forContextWindow context: Int) -> Int {
        min(32_000, max(1, context / 2))
    }

    /// `limit.context` and `limit.output` for the MTPLX model: from the
    /// served window when known, else the limits this model already has,
    /// else the window setting. The output limit is OpenCode's reply reserve
    /// for that window (`outputLimit(forContextWindow:)`).
    static func modelLimits(
        configuration: MTPLXAppConfiguration,
        servedWindow: ServedExecutionWindow?,
        existingProvider: JSONValue?,
        modelID: String
    ) -> (context: Int, output: Int) {
        if servedWindow == nil,
           let limit = existingProvider?.objectValue?["models"]?.objectValue?[modelID]?
               .objectValue?["limit"]?.objectValue,
           let context = limit["context"]?.intValue, context > 0,
           let output = limit["output"]?.intValue, output > 0 {
            return (context, output)
        }
        let context = ClientContextBudget.window(
            configuration: configuration,
            served: servedWindow,
            defaultWindow: 262_144
        )
        return (context, outputLimit(forContextWindow: context))
    }

    private static func providerConfig(
        modelID: String,
        baseURL: String,
        apiKey: String?,
        limits: (context: Int, output: Int),
        vision: Bool,
        reasoningEffort: String?
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

        // Reasoning + temperature are declared capable so OpenCode
        // round-trips assistant reasoning_content (preserve_thinking) and
        // transmits explicit client-side choices; with nothing chosen,
        // OpenCode 1.18.21 sends no sampler for MTPLX model ids and the
        // server's family defaults (the app's source of truth) apply. The
        // family sampler is deliberately not written into model options:
        // @ai-sdk/openai-compatible 2.0.41 has no per-model sampler
        // transport, only reasoningEffort rides options.
        let effortLevels = Self.reasoningEffortLevels(forModelID: modelID)
        let reasoningSupported = effortLevels != nil
        var model: [String: JSONValue] = [
            "name": .string("MTPLX \(modelID)"),
            "reasoning": .bool(reasoningSupported),
            "tool_call": .bool(true),
            "temperature": .bool(true),
            "limit": .object([
                "context": .number(Double(limits.context)),
                "output": .number(Double(limits.output)),
            ]),
            "modalities": .object([
                "input": .array(vision ? [.string("text"), .string("image")] : [.string("text")]),
                "output": .array([.string("text")]),
            ]),
        ]
        if let effortLevels {
            if let reasoningEffort, !reasoningEffort.isEmpty {
                model["options"] = .object([
                    "reasoningEffort": .string(reasoningEffort)
                ])
            }
            let variants = Self.effortVariants(familyLevels: effortLevels)
            if !variants.isEmpty {
                model["variants"] = .object(variants)
            }
        }

        return [
            "npm": .string("@ai-sdk/openai-compatible"),
            "name": .string("MTPLX (local)"),
            "options": .object(options),
            "models": .object([
                modelID: .object(model),
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
    private func openDesktopApplication() async -> (
        didOpen: Bool,
        processID: Int?,
        launchDate: Date?
    ) {
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = true

        return await withCheckedContinuation { continuation in
            NSWorkspace.shared.openApplication(
                at: desktopApplicationURL,
                configuration: configuration
            ) { application, error in
                continuation.resume(
                    returning: (
                        error == nil,
                        application.map { Int($0.processIdentifier) },
                        application?.launchDate
                    )
                )
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
