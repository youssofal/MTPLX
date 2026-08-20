import Foundation

public enum MTPLXCommandBuilderError: Error, Equatable {
    case executableNotFound(String)
}

public struct MTPLXCommandBuilder: Sendable {
    public var environment: [String: String]

    public static let homebrewInstallCommand = "brew install youssofal/mtplx/mtplx"
    private static let localRuntimeInfoKey = "MTPLXLocalRuntimeWrapperPath"
    private static let allowLocalRuntimeInfoKey = "MTPLXAllowLocalRuntimeWrapper"
    private static let bundledRuntimeWheelEnvKey = "MTPLX_BUNDLED_RUNTIME_WHEEL"
    private static let bundledThermalForgeEnvKey = "MTPLX_BUNDLED_THERMALFORGE"
    private static let bundledPythonEnvKey = "MTPLX_APP_BUNDLED_PYTHON"
    private static let blockedAppSubprocessEnvironmentKeys: Set<String> = [
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "MTPLX_APP_ALLOW_SOURCE_WRAPPER",
        "MTPLX_APP_DISABLE_STANDARD_PATHS",
        "MTPLX_APP_HOMEBREW_PATH",
        "MTPLX_APP_SOURCE_WRAPPER_PATH",
        "MTPLX_FAST_MLX_SOURCE_PATH_ACTIVE",
        "PYENV_VERSION",
        "PYTHONBREAKPOINT",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_PYTHON_INSTALL_DIR",
        "VIRTUAL_ENV",
    ]
    private static let blockedAppSubprocessEnvironmentPrefixes = [
        "DYLD_",
        // PIP_* vars silently rewrite pip behavior (--user, index,
        // target dirs) for the app-owned venv; the bootstrapper sets
        // the ones it actually wants after this filter.
        "PIP_",
    ]

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.environment = environment
    }

    public static func missingRuntimeMessage() -> String {
        "MTPLX command-line runtime was not found. Install it with Homebrew: \(homebrewInstallCommand). Then relaunch MTPLX."
    }

    public static func expandedPATH(environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        searchPaths(environment: environment).joined(separator: ":")
    }

    public static func appSubprocessEnvironment(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var env = environment.filter { key, _ in
            guard !blockedAppSubprocessEnvironmentKeys.contains(key) else {
                return false
            }
            return !blockedAppSubprocessEnvironmentPrefixes.contains { key.hasPrefix($0) }
        }
        env["PATH"] = expandedPATH(environment: environment)
        env["MTPLX_DISABLE_FAST_MLX_AUTODISCOVERY"] = "1"
        if let bundledRuntimeWheel = Self.bundledRuntimeWheelPath(environment: environment) {
            env[bundledRuntimeWheelEnvKey] = bundledRuntimeWheel
        }
        if let bundledThermalForge = Self.bundledThermalForgePath() {
            env[bundledThermalForgeEnvKey] = bundledThermalForge
        }
        return pythonBytecodeSafeEnvironment(environment: env)
    }

    /// Preserve a subprocess's caller-owned environment while ensuring an
    /// app-owned Python interpreter cannot mutate the signed application.
    /// Call sites that intentionally need broader developer, Forge, or
    /// publishing variables use this narrower helper instead of the full app
    /// subprocess sanitizer above.
    static func pythonBytecodeSafeEnvironment(
        environment: [String: String]
    ) -> [String: String] {
        var env = environment
        // The bundled interpreter lives inside the signed app. CPython would
        // otherwise create or refresh stdlib __pycache__ files there during
        // first-run venv setup, invalidating the app's code signature. Keep
        // bytecode caching enabled for fast launches, but route every app-
        // spawned Python process to the user's disposable cache directory.
        let home = environment["HOME"].flatMap { $0.isEmpty ? nil : $0 }
            ?? NSHomeDirectory()
        env["PYTHONPYCACHEPREFIX"] = URL(fileURLWithPath: home)
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Caches", isDirectory: true)
            .appendingPathComponent("MTPLX", isDirectory: true)
            .appendingPathComponent("PythonBytecode", isDirectory: true)
            .path
        return env
    }

    /// The concrete profile a fresh user should measure and then run for a
    /// model. Keeping onboarding tune on this resolver prevents its benchmark
    /// from selecting a depth under the legacy Burst lane and then launching
    /// the finished daemon under a different profile. Families without a
    /// measured preset fall back to "sustained" — the same default the
    /// engine resolves for non-flagship models when no --profile is given.
    public static func recommendedProfile(
        for model: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let preset = TargetPreset.preset(
            for: nil,
            processEnvironment: environment
        ).applyingModelDefaults(
            for: model,
            processEnvironment: environment
        )
        return preset.profile.flatMap(MTPLXAppConfiguration.launchableProfile) ?? "sustained"
    }

    public static func resolveHomebrewExecutable(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        if let explicit = environment["MTPLX_APP_HOMEBREW_PATH"] {
            let trimmed = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            return FileManager.default.isExecutableFile(atPath: trimmed)
                ? URL(fileURLWithPath: trimmed)
                : nil
        }
        for path in searchPaths(environment: environment) {
            let candidate = URL(fileURLWithPath: path).appendingPathComponent("brew").path
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return URL(fileURLWithPath: candidate)
            }
        }
        return nil
    }

    public static func resolveInstalledExecutable(
        explicitPath: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        allowDevelopmentWrapper: Bool = false
    ) throws -> URL {
        let sourceWrapperAllowed = Self.developmentWrapperAllowed(
            environment: environment,
            explicitAllow: allowDevelopmentWrapper
        )
        if let explicitPath, !explicitPath.isEmpty {
            if sourceWrapperAllowed || !isDevelopmentWrapper(explicitPath) {
                return URL(fileURLWithPath: explicitPath)
            }
        }
        for name in ["mtplx", "MTPLX"] {
            if let path = findOnPath(
                name,
                environment: environment,
                allowDevelopmentWrapper: sourceWrapperAllowed
            ) {
                return URL(fileURLWithPath: path)
            }
        }
        if sourceWrapperAllowed,
           let devWrapper = Self.developmentWrapper(environment: environment),
           FileManager.default.isExecutableFile(atPath: devWrapper.path)
        {
            return devWrapper
        }
        throw MTPLXCommandBuilderError.executableNotFound("mtplx")
    }

    /// Build the `mtplx serve …` arg list. When `target` is provided,
    /// its preset overrides the target-owned knobs for scheduler,
    /// batching, and SSD enablement so the picker semantically matches
    /// what `mtplx start <target>` would have done. Concrete cache
    /// limits that the user sees in Settings are always passed through
    /// as-is, so the launched daemon matches the visible cap.
    public func buildServeCommand(
        configuration: MTPLXAppConfiguration,
        target: LaunchTarget? = nil,
        launchID: String? = nil
    ) throws -> DaemonCommand {
        let executableURL = try resolveExecutable(configuration.executablePath)
        let resolved = ResolvedDaemonArgs(
            configuration: configuration,
            target: target,
            processEnvironment: environment
        )

        // Final allowlist before argv: no config string, however it got
        // persisted, may produce a serve command argparse refuses.
        var arguments = [
            "serve",
            "--host", configuration.host,
            "--port", String(configuration.port),
            "--model", configuration.model,
        ]
        // Only an engine-launchable profile is emitted. "auto" (and any
        // unknown string) emits NO --profile: the engine is the single
        // owner of default-profile resolution — it resolves identity from
        // the artifact (#268), promotes the flagships to turbo itself,
        // and reports the resolved profile on /health. Writing
        // "--profile sustained" here read as an explicit user choice and
        // blocked that promotion for legacy hybrids and renamed model
        // dirs (F20).
        if let profile = MTPLXAppConfiguration.launchableProfile(resolved.profile) {
            arguments.append(contentsOf: ["--profile", profile])
        }
        let launchGenerationMode = MTPLXAppConfiguration.launchableGenerationMode(
            configuration.generationMode
        )
        if launchGenerationMode != "mtp" {
            arguments.append(contentsOf: ["--generation-mode", launchGenerationMode])
        }
        if !configuration.loadMTP {
            arguments.append("--no-load-mtp")
        }
        // Target-only AR models (Laguna): the engine hard-rejects MTP loads,
        // and a persisted/tuned depth would fail the daemon launch. One seam
        // here covers every app launch path.
        let arOnlyModel = MTPLXModelOption.isAROnlyReference(configuration.model)
        if arOnlyModel {
            arguments.append("--no-mtp")
        }
        arguments.append(contentsOf: ["--scheduler-mode", resolved.schedulerMode])
        arguments.append(contentsOf: ["--batching-preset", resolved.batchingPreset])
        if let maxActiveRequests = resolved.maxActiveRequests, maxActiveRequests > 0 {
            arguments.append(contentsOf: ["--max-active-requests", String(maxActiveRequests)])
        }
        if let decodeBatchMax = resolved.decodeBatchMax, decodeBatchMax > 0 {
            arguments.append(contentsOf: ["--decode-batch-max", String(decodeBatchMax)])
        }
        if let batchWaitMs = resolved.batchWaitMs, batchWaitMs >= 0 {
            arguments.append(contentsOf: ["--batch-wait-ms", String(batchWaitMs)])
        }
        if let prefillChunkTokens = resolved.prefillChunkTokens, prefillChunkTokens > 0 {
            arguments.append(contentsOf: ["--prefill-chunk-tokens", String(prefillChunkTokens)])
        }
        // First-launch auto-tune writes a measured depth into
        // `lastTunedDepth`; backend launch defaults may override it
        // when the public control is not a Qwen-style depth knob. Gemma
        // assistant bundles expose the measured block size through the
        // same daemon flag, so the app must not clamp them to Qwen D3.
        if let depth = resolved.depth, !arOnlyModel {
            arguments.append(contentsOf: ["--depth", String(depth)])
        }
        if let verifyStrategy = resolved.verifyStrategy {
            arguments.append(contentsOf: ["--verify-strategy", verifyStrategy])
        }
        if let verifyCore = resolved.verifyCore {
            arguments.append(contentsOf: ["--verify-core", verifyCore])
        }
        if let mtpAdapterPath = resolved.mtpAdapterPath {
            arguments.append(contentsOf: ["--mtp-adapter", mtpAdapterPath])
        }
        if resolved.mergeMTPAdapter {
            arguments.append("--merge-mtp-adapter")
        }
        if let mtpQuantBits = resolved.mtpQuantBits {
            arguments.append(contentsOf: ["--mtp-quant-bits", String(mtpQuantBits)])
            arguments.append(contentsOf: ["--mtp-quant-group-size", String(resolved.mtpQuantGroupSize)])
            arguments.append(contentsOf: ["--mtp-quant-mode", resolved.mtpQuantMode])
        }
        if configuration.experimentalMTPCohorts {
            arguments.append("--experimental-mtp-cohorts")
        }
        // Retrieval models are additive: no flag means the endpoints answer 404
        // and the chat runtime behaves exactly as before.
        for reference in configuration.embeddingModels where !reference.trimmingCharacters(in: .whitespaces).isEmpty {
            arguments.append(contentsOf: ["--embedding-model", reference])
        }
        for reference in configuration.rerankerModels where !reference.trimmingCharacters(in: .whitespaces).isEmpty {
            arguments.append(contentsOf: ["--reranker-model", reference])
        }
        if !configuration.embeddingModels.isEmpty || !configuration.rerankerModels.isEmpty {
            arguments.append(contentsOf: [
                "--retrieval-max-resident", String(max(1, configuration.retrievalMaxResident)),
            ])
            if configuration.retrievalIdleTimeout > 0 {
                arguments.append(contentsOf: [
                    "--retrieval-idle-timeout", String(configuration.retrievalIdleTimeout),
                ])
            }
        }
        // Always pass the resolved SSD mode, including "off". Omitting
        // the flag delegates the decision to the serve CLI default,
        // which flipped from off to on in 2.0.0 (kvcache-v2) — turning
        // a user's explicit Settings "Off" into a silently-on cache
        // with default limits (issue #140). The app-daemon contract
        // must be explicit so it can never drift with CLI defaults.
        arguments.append(contentsOf: ["--ssd-session-cache", resolved.ssdSessionCache])
        if resolved.ssdSessionCache != "off" {
            if let ssdSessionCacheDir = configuration.ssdSessionCacheDir, !ssdSessionCacheDir.isEmpty {
                arguments.append(contentsOf: ["--ssd-session-cache-dir", ssdSessionCacheDir])
            }
            arguments.append(contentsOf: ["--ssd-session-cache-max-size", resolved.ssdSessionCacheMaxSize])
            arguments.append(contentsOf: [
                "--ssd-session-cache-min-prefix-tokens",
                String(resolved.ssdSessionCacheMinPrefixTokens),
            ])
        }
        if let contextWindow = resolved.contextWindow, contextWindow > 0 {
            arguments.append(contentsOf: ["--context-window", String(contextWindow)])
        }
        if let apiKey = configuration.apiKey, !apiKey.isEmpty {
            arguments.append(contentsOf: ["--api-key", apiKey])
        }
        if configuration.enableThermalPolling {
            arguments.append("--enable-thermal-poll")
        }
        let fanMode = MTPLXFanMode.normalized(configuration.fanMode)
        arguments.append(contentsOf: ["--fan-mode", fanMode.rawValue])
        if fanMode == .max {
            arguments.append("--require-max-fans")
        }
        if let launchID, !launchID.isEmpty {
            arguments.append(contentsOf: ["--app-launch-id", launchID])
        }
        // App launches are explicit user actions. If the backend marks a
        // model as architecture-compatible but not release-verified, still
        // attempt the real start and surface the actual loader result.
        arguments.append("--unsafe-force-unverified")
        arguments.append("--yes")
        if let reasoning = resolved.reasoning {
            arguments.append(contentsOf: ["--reasoning", reasoning])
        }
        if let preserveThinking = resolved.preserveThinking {
            arguments.append(contentsOf: ["--preserve-thinking", preserveThinking])
        }
        if let temperature = resolved.temperature {
            arguments.append(contentsOf: ["--temperature", String(temperature)])
        }
        if let topP = resolved.topP {
            arguments.append(contentsOf: ["--top-p", String(topP)])
        }
        if let topK = resolved.topK {
            arguments.append(contentsOf: ["--top-k", String(topK)])
        }
        if let draftTemperature = resolved.draftTemperature {
            arguments.append(contentsOf: ["--draft-temperature", String(draftTemperature)])
        }
        if let draftTopP = resolved.draftTopP {
            arguments.append(contentsOf: ["--draft-top-p", String(draftTopP)])
        }
        if let draftTopK = resolved.draftTopK {
            arguments.append(contentsOf: ["--draft-top-k", String(draftTopK)])
        }
        if let toolPromptMode = resolved.toolPromptMode {
            arguments.append(contentsOf: ["--tool-prompt-mode", toolPromptMode])
        }
        if let chatTemplateProfile = resolved.chatTemplateProfile {
            arguments.append(contentsOf: ["--chat-template-profile", chatTemplateProfile])
        }
        if let reasoningParser = resolved.reasoningParser {
            arguments.append(contentsOf: ["--reasoning-parser", reasoningParser])
        }
        if let reasoningEffort = resolved.reasoningEffort {
            arguments.append(contentsOf: ["--reasoning-effort", reasoningEffort])
        }
        if let adaptivePolicy = resolved.adaptivePolicy, adaptivePolicy != "none" {
            arguments.append(contentsOf: ["--adaptive-policy", adaptivePolicy])
            if let adaptiveMinDepth = resolved.adaptiveMinDepth {
                arguments.append(contentsOf: ["--adaptive-min-depth", String(adaptiveMinDepth)])
            }
            if adaptivePolicy == "expected_value" {
                if let baseDepth = resolved.adaptiveEVBaseDepth {
                    arguments.append(contentsOf: ["--adaptive-ev-base-depth", String(baseDepth)])
                }
                if let warmup = resolved.adaptiveEVWarmupFullDepthCycles {
                    arguments.append(contentsOf: ["--adaptive-ev-warmup-full-depth-cycles", String(warmup)])
                }
                if let interval = resolved.adaptiveEVExplorationInterval {
                    arguments.append(contentsOf: ["--adaptive-ev-exploration-interval", String(interval)])
                }
            }
        }
        // Native-app-owned daemons serve real clients: OpenCode, Open WebUI,
        // the future app chat, and custom OpenAI-compatible tools. Visible
        // TPS footers belong in the CLI/dashboard surfaces, not inside model
        // text returned to coding agents.
        arguments.append("--no-stats-footer")
        var environment = resolved.environment
        environment.merge(resolved.ramSessionCacheEnvironment) { _, new in new }
        environment = Self.appSubprocessEnvironment(environment: environment)
        environment["MTPLX_APP_PARENT_PID"] = String(ProcessInfo.processInfo.processIdentifier)
        if resolved.pagedKVQuantization != "off" {
            environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] = resolved.pagedKVQuantization
        }
        if let launchID, !launchID.isEmpty {
            environment["MTPLX_APP_LAUNCH_ID"] = launchID
        }
        if let mirror = MTPLXAppConfiguration.hfMirrorEnvironment(configuration.hfEndpoint) {
            environment.merge(mirror) { _, new in new }
        }
        return DaemonCommand(
            executableURL: executableURL,
            arguments: arguments,
            environment: environment
        )
    }

    private func resolveExecutable(_ explicitPath: String?) throws -> URL {
        try Self.resolveInstalledExecutable(
            explicitPath: explicitPath,
            environment: environment
        )
    }

    private static func developmentWrapper(environment: [String: String]) -> URL? {
        if let explicit = environment["MTPLX_APP_SOURCE_WRAPPER_PATH"],
           !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            let candidate = URL(fileURLWithPath: explicit)
            return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
        }
        if let explicit = Bundle.main.object(forInfoDictionaryKey: localRuntimeInfoKey) as? String,
           !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            let candidate = URL(fileURLWithPath: explicit)
            return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
        }
        #if DEBUG
        var cursor = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            cursor.deleteLastPathComponent()
        }
        let candidate = cursor.appendingPathComponent("bin").appendingPathComponent("mtplx")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
        #else
        return nil
        #endif
    }

    private static func developmentWrapperAllowed(
        environment: [String: String],
        explicitAllow: Bool
    ) -> Bool {
        guard !explicitAllow else { return true }
        let raw = environment["MTPLX_APP_ALLOW_SOURCE_WRAPPER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "1" || raw == "true" || raw == "yes" {
            return true
        }
        return (Bundle.main.object(forInfoDictionaryKey: allowLocalRuntimeInfoKey) as? Bool) == true
    }

    /// Build the search path. When the app is launched from `dist/` via
    /// `open`, macOS hands us the launchd-derived PATH (`/usr/bin:/bin:
    /// /usr/sbin:/sbin`) which doesn't include `~/.local/bin`,
    /// `/opt/homebrew/bin`, or `/usr/local/bin` — where `mtplx` actually
    /// lives. Augment the PATH with the well-known macOS bin
    /// directories before searching, so the daemon launches from a
    /// Homebrew install without the user having to set anything.
    private static func searchPaths(environment: [String: String]) -> [String] {
        let disableStandardPaths = environment["MTPLX_APP_DISABLE_STANDARD_PATHS"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let standardPathsDisabled =
            disableStandardPaths == "1"
            || disableStandardPaths == "true"
            || disableStandardPaths == "yes"
        var paths: [String] = []
        if !standardPathsDisabled {
            // The app-managed venv is a standard location too: tests pin
            // PATH to fakes and must not leak the machine's real runtime.
            paths.append(appRuntimeBinDirectory(environment: environment))
        }
        if let envPath = environment["PATH"], !envPath.isEmpty {
            for path in envPath.split(separator: ":").map(String.init) where !paths.contains(path) {
                paths.append(path)
            }
        }
        if standardPathsDisabled {
            return paths
        }
        let home = environment["HOME"] ?? NSHomeDirectory()
        let extras = [
            "\(home)/.local/bin",
            "\(home)/.cargo/bin",
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            "/usr/local/sbin",
            "/usr/bin",
            "/bin",
        ]
        for extra in extras where !paths.contains(extra) {
            paths.append(extra)
        }
        return paths
    }

    private static func findOnPath(
        _ name: String,
        environment: [String: String],
        allowDevelopmentWrapper: Bool
    ) -> String? {
        for path in searchPaths(environment: environment) {
            let candidate = URL(fileURLWithPath: path).appendingPathComponent(name).path
            if FileManager.default.isExecutableFile(atPath: candidate) {
                if !allowDevelopmentWrapper, isDevelopmentWrapper(candidate) {
                    continue
                }
                return candidate
            }
        }
        return nil
    }

    /// Locate a user-managed `mtplx` on the search path, ignoring the
    /// app-owned runtime venv and the repo development wrapper. Used
    /// by onboarding's runtime-setup step to report (and, for
    /// Homebrew installs, upgrade) a pre-existing global CLI without
    /// ever confusing it with the app's own runtime.
    public static func detectGlobalCLIExecutable(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        let appRuntimeBin = appRuntimeBinDirectory(environment: environment)
        let appRuntimeBinResolved = URL(fileURLWithPath: appRuntimeBin)
            .resolvingSymlinksInPath()
            .path
        let excludedDirectories = Set([appRuntimeBin, appRuntimeBinResolved])
        for path in searchPaths(environment: environment) {
            if excludedDirectories.contains(path) { continue }
            for name in ["mtplx", "MTPLX"] {
                let candidate = URL(fileURLWithPath: path).appendingPathComponent(name)
                guard FileManager.default.isExecutableFile(atPath: candidate.path) else { continue }
                if isDevelopmentWrapper(candidate.path) { continue }
                let resolved = candidate.resolvingSymlinksInPath().path
                if resolved.hasPrefix(appRuntimeBin + "/")
                    || resolved.hasPrefix(appRuntimeBinResolved + "/")
                {
                    continue
                }
                return candidate
            }
        }
        return nil
    }

    public static func isDevelopmentWrapperPath(_ candidatePath: String) -> Bool {
        let candidate = URL(fileURLWithPath: candidatePath).resolvingSymlinksInPath()
        guard candidate.lastPathComponent.lowercased() == "mtplx" else { return false }
        let binDir = candidate.deletingLastPathComponent()
        guard binDir.lastPathComponent == "bin" else { return false }
        let root = binDir.deletingLastPathComponent()
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("pyproject.toml").path)
            && FileManager.default.fileExists(
                atPath: root.appendingPathComponent("mtplx").appendingPathComponent("cli.py").path
            )
    }

    private static func isDevelopmentWrapper(_ candidatePath: String) -> Bool {
        isDevelopmentWrapperPath(candidatePath)
    }

    public static func appRuntimeDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let home = environment["HOME"] ?? NSHomeDirectory()
        return URL(fileURLWithPath: home)
            .appendingPathComponent("Library")
            .appendingPathComponent("Application Support")
            .appendingPathComponent("MTPLX")
            .appendingPathComponent("runtime-venv")
            .path
    }

    public static func appRuntimeBinDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        URL(fileURLWithPath: appRuntimeDirectory(environment: environment))
            .appendingPathComponent("bin")
            .path
    }

    public static func bundledRuntimeWheelPath(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let override = environment[bundledRuntimeWheelEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty,
           FileManager.default.fileExists(atPath: override)
        {
            return override
        }

        guard let resources = Bundle.main.resourceURL else { return nil }
        let runtimeDir = resources.appendingPathComponent("Runtime", isDirectory: true)
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: runtimeDir,
            includingPropertiesForKeys: nil
        ) else { return nil }
        return contents
            .filter { $0.pathExtension == "whl" && $0.lastPathComponent.hasPrefix("mtplx-") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .first?
            .path
    }

    public static func bundledThermalForgePath() -> String? {
        guard let url = Bundle.main.url(
            forResource: "thermalforge",
            withExtension: nil,
            subdirectory: "ThermalForge"
        ) else {
            return nil
        }
        return FileManager.default.isExecutableFile(atPath: url.path) ? url.path : nil
    }

    /// The python-build-standalone interpreter shipped inside the bundle
    /// (`Contents/Resources/PythonRuntime`). This is what kills the
    /// "Install Homebrew" wall: the app-owned venv can always be built from
    /// this interpreter on a pristine Mac. `MTPLX_APP_BUNDLED_PYTHON`
    /// overrides for tests and development builds.
    public static func bundledPythonExecutablePath(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let override = environment[bundledPythonEnvKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty
        {
            return FileManager.default.isExecutableFile(atPath: override)
                ? override
                : nil
        }
        guard let resources = Bundle.main.resourceURL else { return nil }
        let python = resources
            .appendingPathComponent("PythonRuntime", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("python3")
        return FileManager.default.isExecutableFile(atPath: python.path)
            ? python.path
            : nil
    }

    public static func defaultReasoningMode(for target: LaunchTarget?) -> String? {
        let normalized = TargetPreset
            .preset(for: target)
            .reasoning?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch normalized {
        case "auto", "on", "off":
            return normalized
        default:
            return nil
        }
    }
}

// MARK: - ResolvedDaemonArgs
//
// Combines the user's persisted `MTPLXAppConfiguration` knobs with the
// LaunchTarget preset (chat / openCode / dashboard / ...). The rule is
// simple: if the user has explicitly tuned a knob in Settings (the
// value differs from the framework default), keep their value. Else
// take the target's preset. This means a Sustained user who clicked
// OpenCode gets the D3 MTP path the launch card advertises plus SSD cache, but
// they can still force batching modes from Settings when they want a throughput
// experiment.

struct ResolvedDaemonArgs {
    let schedulerMode: String
    let batchingPreset: String
    let profile: String
    let maxActiveRequests: Int?
    let decodeBatchMax: Int?
    let batchWaitMs: Double?
    let prefillChunkTokens: Int?
    let depth: Int?
    let verifyStrategy: String?
    let verifyCore: String?
    let mtpAdapterPath: String?
    let mergeMTPAdapter: Bool
    let mtpQuantBits: Int?
    let mtpQuantGroupSize: Int
    let mtpQuantMode: String
    let ssdSessionCache: String
    let ssdSessionCacheMaxSize: String
    let ssdSessionCacheMinPrefixTokens: Int
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let draftTemperature: Double?
    let draftTopP: Double?
    let draftTopK: Int?
    let toolPromptMode: String?
    let chatTemplateProfile: String?
    let adaptivePolicy: String?
    let adaptiveMinDepth: Int?
    let adaptiveEVBaseDepth: Int?
    let adaptiveEVWarmupFullDepthCycles: Int?
    let adaptiveEVExplorationInterval: Int?
    let pagedKVQuantization: String
    let contextWindow: Int?
    /// App-native launch targets can carry durable Settings sampler choices.
    /// External coding-agent targets keep their measured sampler presets, but
    /// reasoning remains app-owned for every app-launched daemon.
    let reasoning: String?
    let preserveThinking: String?
    let reasoningParser: String?
    let reasoningEffort: String?
    let environment: [String: String]
    let ramSessionCacheEnvironment: [String: String]

    init(
        configuration: MTPLXAppConfiguration,
        target: LaunchTarget?,
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let preset = TargetPreset.preset(
            for: target,
            processEnvironment: processEnvironment
        ).applyingModelDefaults(
            for: configuration.model,
            processEnvironment: processEnvironment
        )

        let targetOwnsScheduling =
            target == .chat
            || target == .openWebUI
            || target == .hermes
            || target == .benchmark
        let scheduling = targetOwnsScheduling
            ? .targetDefault
            : SchedulingOverridePreset(configuration.schedulingPreset)
        let schedulingDefaults = scheduling.daemonDefaults

        schedulerMode = scheduling == .targetDefault
            ? preset.schedulerMode
            : schedulingDefaults.schedulerMode

        batchingPreset = scheduling == .targetDefault
            ? preset.batchingPreset
            : schedulingDefaults.batchingPreset

        // "auto" flows through unresolved: buildServeCommand emits no
        // --profile for it, handing default-profile resolution to the
        // engine (per-artifact; turbo for the flagships). The old
        // app-side resolution — preset.profile with a "sustained"
        // fallback — was the Swift twin of the engine's #268 path bug:
        // any model the path-substring table missed (legacy hybrids,
        // renamed dirs) launched with an explicit sustained pin. An
        // explicit user pick still wins — the Settings picker must
        // never lie (2026-07-03 turbo release).
        profile = configuration.profile

        maxActiveRequests = targetOwnsScheduling
            ? preset.maxActiveRequests
            : configuration.maxActiveRequests
                ?? (scheduling == .targetDefault ? preset.maxActiveRequests : schedulingDefaults.maxActiveRequests)

        decodeBatchMax = targetOwnsScheduling
            ? preset.decodeBatchMax
            : configuration.decodeBatchMax
                ?? (scheduling == .targetDefault ? preset.decodeBatchMax : schedulingDefaults.decodeBatchMax)

        batchWaitMs = targetOwnsScheduling
            ? preset.batchWaitMs
            : configuration.batchWaitMs
                ?? (scheduling == .targetDefault ? preset.batchWaitMs : schedulingDefaults.batchWaitMs)

        prefillChunkTokens = configuration.prefillChunkTokens
            ?? (scheduling == .targetDefault ? preset.prefillChunkTokens : schedulingDefaults.prefillChunkTokens)

        depth = Self.resolvedDraftControlValue(
            configuration,
            presetDepth: preset.depth,
            target: target
        )

        verifyStrategy = preset.verifyStrategy
        verifyCore = preset.verifyCore
        mtpAdapterPath = preset.mtpAdapterPath
        mergeMTPAdapter = preset.mergeMTPAdapter
        mtpQuantBits = preset.mtpQuantBits
        mtpQuantGroupSize = preset.mtpQuantGroupSize
        mtpQuantMode = preset.mtpQuantMode

        ssdSessionCache = Self.resolvedSSDSessionCache(
            configuration.ssdSessionCache,
            preset: preset.ssdSessionCache
        )

        ssdSessionCacheMaxSize = configuration.ssdSessionCacheMaxSize
        ssdSessionCacheMinPrefixTokens = configuration.ssdSessionCacheMinPrefixTokens

        let carriesSettingsSampler = Self.targetCarriesSettingsSampler(target)
            && Self.configurationSamplerCompatible(configuration)
        let carriesCompatibleLiveSettings = Self.configurationLiveSettingsCompatible(configuration)
        let carriesSettingsReasoning = preset.acceptsSettingsReasoning
            && Self.targetCarriesSettingsReasoning(
                target,
                configuration: configuration
            )
            && carriesCompatibleLiveSettings
        temperature = carriesSettingsSampler
            ? (configuration.temperature ?? preset.temperature)
            : preset.temperature
        topP = carriesSettingsSampler
            ? (configuration.topP ?? preset.topP)
            : preset.topP
        topK = carriesSettingsSampler
            ? (configuration.topK ?? preset.topK)
            : preset.topK

        draftTemperature = carriesSettingsSampler
            ? (configuration.temperature ?? preset.draftTemperature)
            : preset.draftTemperature
        draftTopP = carriesSettingsSampler
            ? (configuration.topP ?? preset.draftTopP)
            : preset.draftTopP
        draftTopK = carriesSettingsSampler
            ? (configuration.topK ?? preset.draftTopK)
            : preset.draftTopK
        toolPromptMode = preset.toolPromptMode
        chatTemplateProfile = preset.chatTemplateProfile
        adaptivePolicy = preset.adaptivePolicy
        adaptiveMinDepth = preset.adaptiveMinDepth
        adaptiveEVBaseDepth = preset.adaptiveEVBaseDepth
        adaptiveEVWarmupFullDepthCycles = preset.adaptiveEVWarmupFullDepthCycles
        adaptiveEVExplorationInterval = preset.adaptiveEVExplorationInterval
        let requestedPagedKVQuantization = Self.normalizedPagedKVQuantization(
            configuration.pagedKVQuantization
        )
        pagedKVQuantization = Self.modelAllowsPagedKVQuantization(configuration.model)
            ? requestedPagedKVQuantization
            : "off"
        contextWindow = configuration.compatibleContextWindowOverride()

        let resolvedReasoning = carriesSettingsReasoning
            ? (Self.normalizedReasoning(configuration.reasoning) ?? preset.reasoning)
            : preset.reasoning
        let resolvedReasoningEffort = carriesSettingsReasoning
            ? (Self.normalizedReasoningEffort(configuration.reasoningEffort) ?? preset.reasoningEffort)
            : preset.reasoningEffort
        reasoning = resolvedReasoning
        preserveThinking = resolvedReasoning == "off"
            ? (preset.preserveThinking == nil ? nil : "off")
            : preset.preserveThinking
        reasoningParser = preset.reasoningParser
        reasoningEffort = resolvedReasoning == "off" ? nil : resolvedReasoningEffort

        environment = preset.environment
        ramSessionCacheEnvironment = Self.ramSessionCacheEnvironment(
            from: configuration
        )
    }

    private static func compatibleTunedDraftControlValue(_ configuration: MTPLXAppConfiguration) -> Int? {
        let family = MTPLXModelOption.modelFamily(for: configuration.model)
        if MTPLXModelOption.supportsTune(family: family) {
            return configuration.compatibleTunedDepth()
        }
        if family == "gemma4" {
            return configuration.compatibleTunedControlValue(controlField: "draft_block_size")
        }
        return nil
    }

    private static func resolvedDraftControlValue(
        _ configuration: MTPLXAppConfiguration,
        presetDepth: Int?,
        target: LaunchTarget?
    ) -> Int? {
        let family = MTPLXModelOption.modelFamily(for: configuration.model)
        let tuned = compatibleTunedDraftControlValue(configuration)
        if family == "gemma4", let tuned {
            return tuned
        }
        // Lane split mirrors targetCarriesSettingsSampler: user-facing
        // lanes honor a compatible per-model tune over the model's
        // catalog launch default (the tune measured this exact model on
        // this exact Mac); coding-agent lanes keep their literal preset
        // depth (OpenCode stays D3 by decision).
        if targetCarriesSettingsSampler(target), let tuned {
            return tuned
        }
        return presetDepth ?? tuned
    }

    private static func modelAllowsPagedKVQuantization(_ model: String) -> Bool {
        // Same qwen3_next attention layout for 3.5 / 3.6 / 3.8; the server's
        // kv_quant_policy declares q8/q4 supported for all three.
        let family = MTPLXModelOption.modelFamily(for: model)
        return family == "qwen3_5" || family == "qwen3_6" || family == "qwen3_8"
    }

    private static func normalizedReasoning(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "on", "off":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private static func normalizedReasoningEffort(_ raw: String?) -> String? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "low", "medium", "high", "xhigh":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            return nil
        }
    }

    private static func targetCarriesSettingsSampler(_ target: LaunchTarget?) -> Bool {
        switch target {
        case nil, .chat, .benchmark, .openWebUI:
            return true
        case .pi, .openCode, .hermes, .other:
            return false
        }
    }

    private static func targetCarriesSettingsReasoning(
        _ target: LaunchTarget?,
        configuration: MTPLXAppConfiguration
    ) -> Bool {
        _ = target
        _ = configuration
        return true
    }

    private static func configurationSamplerCompatible(_ configuration: MTPLXAppConfiguration) -> Bool {
        configurationLiveSettingsCompatible(configuration)
    }

    private static func configurationLiveSettingsCompatible(_ configuration: MTPLXAppConfiguration) -> Bool {
        let family = MTPLXModelOption.modelFamily(for: configuration.model)
        if let storedFamily = configuration.liveSettingsModelFamily,
           !storedFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return MTPLXModelOption.settingsFamiliesCompatible(
                stored: storedFamily,
                current: family
            )
        }
        return MTPLXModelOption.supportsTune(family: family)
    }

    private static func normalizedPagedKVQuantization(_ raw: String) -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().replacingOccurrences(of: "-", with: "_") {
        case "q8", "q8_0", "int8":
            return "q8"
        case "q4", "q4_0", "int4":
            return "q4"
        default:
            return "off"
        }
    }

    private static func resolvedSSDSessionCache(_ raw: String, preset: String) -> String {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "target-default", "default", "":
            return preset
        case "off", "on", "write-only":
            return normalized
        default:
            return preset
        }
    }

    private static func ramSessionCacheEnvironment(
        from configuration: MTPLXAppConfiguration
    ) -> [String: String] {
        guard configuration.ramSessionCachePolicy != "target-default" else {
            // #229: "target-default" means "let the preset/engine decide the
            // policy" — but a user-entered explicit size in Settings was
            // silently dropped here, which reads as the setting working
            // while the bank runs at auto size. Pass explicit sizes through;
            // "auto"/empty still defer entirely.
            var explicit: [String: String] = [:]
            let maxSize = configuration.ramSessionCacheMaxSize
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !maxSize.isEmpty, maxSize.lowercased() != "auto" {
                explicit["MTPLX_SESSION_BANK_MAX_BYTES"] = maxSize
            }
            let perSession = configuration.ramSessionCachePerSessionMaxSize
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !perSession.isEmpty, perSession.lowercased() != "auto" {
                explicit["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] = perSession
            }
            return explicit
        }
        let entries = max(1, configuration.ramSessionCacheMaxEntries)
        var environment = [
            "MTPLX_SESSION_BLOCK_PREFIX_RESTORE": configuration.ramSessionBlockPrefixRestore ? "1" : "0",
            "MTPLX_SESSION_BANK_MAX_ENTRIES": String(entries),
            "MTPLX_SESSION_BANK_MAX_BYTES": configuration.ramSessionCacheMaxSize,
            "MTPLX_SESSION_BANK_PER_SESSION_BYTES": configuration.ramSessionCachePerSessionMaxSize,
        ]
        if configuration.ramSessionCachePolicy == "minimal" {
            environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"] = "0"
            environment["MTPLX_SESSION_BANK_MAX_ENTRIES"] = "1"
            environment["MTPLX_SESSION_BANK_MAX_BYTES"] = "1G"
            environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] = "1G"
        }
        return environment
    }
}

private enum SchedulingOverridePreset: String {
    case targetDefault = "target-default"
    case latency
    case throughput
    case agent

    init(_ raw: String) {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        {
        case "latency", "serial-latency":
            self = .latency
        case "throughput", "ar-batch-throughput":
            self = .throughput
        case "agent", "ar-batch-agent":
            self = .agent
        default:
            self = .targetDefault
        }
    }

    var daemonDefaults: TargetPreset {
        switch self {
        case .targetDefault:
            return TargetPreset()
        case .latency:
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "latency"
            )
        case .throughput:
            return TargetPreset(
                schedulerMode: "ar_batch",
                batchingPreset: "throughput",
                maxActiveRequests: 8,
                decodeBatchMax: 8,
                batchWaitMs: 20,
                prefillChunkTokens: 2048
            )
        case .agent:
            return TargetPreset(
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                maxActiveRequests: 4,
                decodeBatchMax: 4,
                batchWaitMs: 50,
                prefillChunkTokens: 2048
            )
        }
    }

}

private enum ModelLaunchFamily {
    case qwen36_35BOptimizedSpeed
    case qwen36_27BOptimizedSpeed
    case qwen36_27BOptimizedQuality
    case qwen35_9BOptimizedSpeed
    case qwen38_27B
    case gemma4
    case step
    case hy3
    case qwenDefault

    static func detect(_ model: String) -> ModelLaunchFamily {
        let normalized = model
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        if normalized.contains("qwen3.6-35b-a3b-mtplx-official4-cyankiwi")
            || normalized.contains("qwen3.6-35b-a3b-mtplx-optimized-speed")
            || normalized.contains("qwen3.6-35b-a3b-mtplx-flat4-cyankiwimtp")
            || normalized.contains("qwen36-35b-a3b-optimized-speed")
            || normalized.contains("mtplx-qwen36-35b-a3b-optimized-speed")
        {
            return .qwen36_35BOptimizedSpeed
        }
        // 9B (6-bit) family, incl. the -FP16 sibling. Promoted to turbo
        // 2026-07-07 with the 6-bit hexpack split-K kernels (live ABBA:
        // MTP D3 110/102 vs sustained 90/69 tok/s, AR flat).
        if normalized.contains("qwen3.5-9b-mtplx-optimized-speed")
            || normalized.contains("qwen35-9b-optimized-speed")
        {
            return .qwen35_9BOptimizedSpeed
        }
        // Qwen3.8 27B MTPLX family (Bare Speed / Optimized Speed /
        // Optimized Quality). Trunk geometry is identical to the Qwen3.6
        // 27B flagships, so the verify kernels and their quant-bits gates
        // carry over; the launch contract (turbo + the official 1.0
        // thinking sampler) is the model card's, not the 3.6 coding one.
        if normalized.contains("qwen3.8-27b-mtplx")
            || normalized.contains("qwen38-27b")
            || MTPLXModelOption.modelFamily(for: model) == "qwen3_8"
        {
            return .qwen38_27B
        }
        // 27B Speed family (4-bit affine, incl. the -FP16 sibling).
        if normalized.contains("qwen3.6-27b-mtplx-optimized-speed")
            || normalized.contains("qwen36-27b-optimized-speed")
        {
            return .qwen36_27BOptimizedSpeed
        }
        // 27B Quality family (8-bit affine). The substring also matches the
        // -FP16 sibling (M1/M2 quality routing target, added 2026-07-07),
        // which shares the q8 packs the ULP-exact vk kernels cover.
        if normalized.contains("qwen3.6-27b-mtplx-optimized-quality")
            || normalized.contains("qwen36-27b-optimized-quality")
        {
            return .qwen36_27BOptimizedQuality
        }
        // Tencent Hy3 295B MoE (hy_v3): dynamic 2-bit MTPLX artifact.
        if normalized.contains("hy3-295b") || normalized.contains("hunyuan-3") {
            return .hy3
        }
        if normalized.contains("step3.7")
            || normalized.contains("step-3.7")
            || normalized.contains("step3p5")
            || normalized.contains("step-3.5")
            || normalized.contains("stepfun")
        {
            return .step
        }
        if normalized.contains("gemma4")
            || normalized.contains("gemma-4")
            || normalized.contains("gemma-4-")
        {
            return .gemma4
        }
        let marker = URL(fileURLWithPath: NSString(string: model).expandingTildeInPath)
            .appendingPathComponent("mtplx_pair.json")
            .path
        if FileManager.default.fileExists(atPath: marker) {
            return .gemma4
        }
        return .qwenDefault
    }

    /// A "-FP16" precision sibling of a base artifact — the variant the
    /// legacy (M1/M2) tier routes to. Same INT4 packs; BF16 floats
    /// downcast to FP16.
    static func isFP16PrecisionVariant(_ model: String) -> Bool {
        let normalized = model
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        return normalized.contains("-fp16")
    }
}

private struct TargetPreset {
    private static let step37AdapterResourceName = "c4-mtp-adapter-20260603-134243-r4"

    var schedulerMode: String = "serial"
    var batchingPreset: String = "latency"
    var profile: String? = nil
    var maxActiveRequests: Int? = nil
    var decodeBatchMax: Int? = nil
    var batchWaitMs: Double? = nil
    var prefillChunkTokens: Int? = nil
    var depth: Int? = nil
    var verifyStrategy: String? = nil
    var verifyCore: String? = nil
    var mtpAdapterPath: String? = nil
    var mergeMTPAdapter: Bool = false
    var mtpQuantBits: Int? = nil
    var mtpQuantGroupSize: Int = 64
    var mtpQuantMode: String = "affine"
    var ssdSessionCache: String = "off"
    var temperature: Double? = nil
    var topP: Double? = nil
    var topK: Int? = nil
    var draftTemperature: Double? = nil
    var draftTopP: Double? = nil
    var draftTopK: Int? = nil
    var toolPromptMode: String? = nil
    var chatTemplateProfile: String? = nil
    var adaptivePolicy: String? = nil
    var adaptiveMinDepth: Int? = nil
    var adaptiveEVBaseDepth: Int? = nil
    var adaptiveEVWarmupFullDepthCycles: Int? = nil
    var adaptiveEVExplorationInterval: Int? = nil
    var reasoning: String? = nil
    var preserveThinking: String? = nil
    var reasoningParser: String? = nil
    var reasoningEffort: String? = nil
    var acceptsSettingsReasoning: Bool = true
    var environment: [String: String] = [:]

    private static let highMemoryThresholdBytes: UInt64 = 96 * 1024 * 1024 * 1024
    private static let defaultOpenCodeSessionBankMaxEntries = "6"
    private static let highMemoryOpenCodeSessionBankMaxEntries = "32"

    private static func physicalMemoryBytes(
        processEnvironment: [String: String]
    ) -> UInt64 {
        if let raw = processEnvironment["MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES"],
           let value = UInt64(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
           value > 0 {
            return value
        }
        return ProcessInfo.processInfo.physicalMemory
    }

    private static func codingAgentRuntimeEnvironment(
        processEnvironment: [String: String]
    ) -> [String: String] {
        let highMemory = physicalMemoryBytes(
            processEnvironment: processEnvironment
        ) >= highMemoryThresholdBytes
        var environment = [
            // Route only — the MIN_CONTEXT/MIN_Q/MAX_Q overrides (32768/3/5,
            // unmeasured 1.0.0 launch values) are gone so the engine defaults
            // (65536/4/5) govern. Issue #228: async_per_head below 64k
            // measured 4-7x SLOWER decode at 43k ctx; restoring the engine
            // threshold recovered 4-6.8x on the reporter's table.
            "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE": "async_per_head",
            "MTPLX_SESSION_BLOCK_PREFIX_RESTORE": "1",
            "MTPLX_SESSION_BANK_MAX_ENTRIES": highMemory
                ? highMemoryOpenCodeSessionBankMaxEntries
                : defaultOpenCodeSessionBankMaxEntries,
            "MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S": "30.0",
            "MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS": "4096",
            // Mirrors the CLI coding-agent lane (_opencode_memory_env_defaults);
            // this key was CLI-only drift until the 2026-08-03 parity audit.
            "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "1",
            "MTPLX_LAZY_BONUS_VERIFY": "1",
            "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER": "1",
            "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE": "1",
            "MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES": "72",
            "MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE": "8",
            "MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS": "120",
            "MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS": "12",
            "MTPLX_TOOL_PROMPT_MODE": "hybrid",
            "MTPLX_CHAT_TEMPLATE_PROFILE": "local_qwen36",
        ]
        // "auto": the engine budgets half of the RAM left after the model
        // weights (floor 1 GiB, cap 48 GiB). Replaces the flat 24G/8G tier
        // that ignored the loaded model's size — safe on 32 GB Macs, roomy
        // on 128 GB ones. Explicit user limits from Settings override this
        // after the base environment merge.
        environment["MTPLX_SESSION_BANK_MAX_BYTES"] = "auto"
        environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] = "auto"
        return environment
    }

    func applyingModelDefaults(
        for model: String,
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) -> TargetPreset {
        switch ModelLaunchFamily.detect(model) {
        case .qwen36_35BOptimizedSpeed:
            return applyingQwen36_35BOptimizedSpeedDefaults()
        case .qwen36_27BOptimizedSpeed:
            return applyingQwen36_27BOptimizedSpeedDefaults()
        case .qwen36_27BOptimizedQuality:
            return applyingQwen36_27BOptimizedQualityDefaults()
        case .qwen35_9BOptimizedSpeed:
            return applyingQwen35_9BOptimizedSpeedDefaults()
        case .qwen38_27B:
            return applyingQwen38_27BDefaults()
        case .qwenDefault:
            return self
        case .gemma4:
            return applyingGemma4Defaults()
        case .step:
            return applyingStepDefaults(processEnvironment: processEnvironment)
        case .hy3:
            return applyingHy3Defaults()
        }
    }

    private func applyingQwen36_27BOptimizedSpeedDefaults() -> TargetPreset {
        var preset = self
        // Turbo = sustained plus the clean-room NAX verify kernels
        // (engine TURBO_PROFILE carries the env). Chat lane measured
        // 2026-07-02: 44.7 -> 58-60 tok/s on the app launch flags.
        // The FP16 sibling (what the M1/M2 tier routes to) was promoted
        // per-artifact on 2026-07-07 after its first e2e measurement:
        // same INT4/g64 weight packs the vk kernels cover, turbo measured
        // 1.98-2.08x over true AR (55-60 tok/s D3 vs 27-29 AR, acceptance
        // 86/66/51, peak 18.6 GB) vs 1.34x on sustained, whose fp16
        // verify-hidden-eval tax eats ~84% of decode wall. Same
        // measurement-first discipline that admitted Quality q8.
        preset.profile = "turbo"
        preset.applyQwen36ThinkingSampler()
        return preset
    }

    private func applyingQwen36_27BOptimizedQualityDefaults() -> TargetPreset {
        var preset = self
        // 8-bit Quality also gets turbo: the q8 verify_kernels branch is
        // ULP-exact and measured +22-40% on the chat lane 2026-07-03
        // (31-36 -> 43-44 tok/s; verify 81-93 -> 61-64 ms/call). The old
        // "Quality stays sustained" ruling was about compiled verify
        // (-15/-18%), which turbo does not use. The Quality-FP16 sibling
        // (M1/M2 tier, built 2026-07-07) measured 2.5x over true AR under
        // turbo on its own artifact (43.8/42.1 vs 17.4 tok/s D3).
        preset.profile = "turbo"
        preset.applyQwen36ThinkingSampler()
        return preset
    }

    private func applyingQwen35_9BOptimizedSpeedDefaults() -> TargetPreset {
        var preset = self
        // 6-bit 9B earned turbo on 2026-07-07 when the 6-bit hexpack
        // split-K verify kernels landed: live ABBA on the artifact
        // measured MTP D3 110/102 tok/s under turbo vs 90/69 sustained
        // (true AR 62-65 flat both profiles). Exactness gated vs stock
        // across {bf16, fp16} x gs{32, 64, 128}; the load-time selfcheck
        // re-proves it on every user's silicon.
        preset.profile = "turbo"
        preset.applyQwen36ThinkingSampler()
        return preset
    }

    private func applyingQwen38_27BDefaults() -> TargetPreset {
        var preset = self
        // Qwen3.8 27B rides the same 4/8-bit affine packs the vk/NAX verify
        // kernels cover (trunk geometry identical to the 3.6 27B flagships),
        // so it launches turbo like them; the day-one A/B on the real
        // artifacts owns the final ruling before release. Sampler is the
        // model card's official thinking-mode triple (1.0/0.95/20), NOT the
        // 3.6-era 0.6 coding sampler. reasoning_effort and preserve_thinking
        // are deliberately not pinned here: the server's qwen3_8 family
        // policy resolves them (measured coding default medium, preserve) and
        // stays the single owner.
        preset.profile = "turbo"
        preset.temperature = 1.0
        preset.topP = 0.95
        preset.topK = 20
        // The draft sampler is deliberately NOT pinned for this family. Each
        // 3.8 artifact stamps its measured `recommended_draft_sampler` in
        // mtplx_runtime.json and the server resolves it from the stamp — the
        // same path `mtplx serve` takes with zero flags. (Which number is
        // right belongs to the artifact: the drop-day strict max-fan A/B
        // measured draft 1.0 over the earlier uncontrolled 0.6 for Bare
        // Speed — the stamp is the restamp surface, never this file.) One
        // owner, so the app and the CLI serve the
        // same draft sampler for the same artifact (incl. the FP16 siblings)
        // and a stamp change never needs an app release. A user-set sampler
        // in Settings still carries to the draft, as for every family.
        return preset
    }

    // Qwen3.6 thinking-mode recommended sampling (0.6/0.95/20) — the
    // same triple the 35B and Step presets already pin. The 27B models
    // had no launch family until 2026-07-02 and silently fell through
    // to the CLI default of 1.0.
    private mutating func applyQwen36ThinkingSampler() {
        temperature = 0.6
        topP = 0.95
        topK = 20
        draftTemperature = 0.6
        draftTopP = 0.95
        draftTopK = 20
    }

    private func applyingQwen36_35BOptimizedSpeedDefaults() -> TargetPreset {
        var preset = self
        preset.depth = 1
        preset.verifyStrategy = "target_prefix"
        preset.temperature = 0.6
        preset.topP = 0.95
        preset.topK = 20
        preset.draftTemperature = 0.6
        preset.draftTopP = 0.95
        preset.draftTopK = 20
        preset.chatTemplateProfile = "local_qwen36"
        preset.reasoningParser = "qwen3"
        return preset
    }

    private func applyingHy3Defaults() -> TargetPreset {
        var preset = self
        // Tencent Hy3 official inference settings (generation_config.json):
        // temperature 0.9, top_p 1.0, top_k off — NOT the Qwen 0.6/0.95/20
        // coding triple. Depth 1: the model ships a single NextN head and
        // deeper reuse drafts carry ~20% positional acceptance (pure tax).
        // Profile stays nil so the runtime contract's recommended profile
        // (sustained) and the hy_v3 descriptor own the launch defaults.
        preset.profile = nil
        preset.depth = 1
        preset.temperature = 0.9
        preset.topP = 1.0
        preset.topK = 0
        preset.draftTemperature = 0.9
        preset.draftTopP = 1.0
        preset.draftTopK = 0
        preset.chatTemplateProfile = "tokenizer"
        preset.environment["MTPLX_CHAT_TEMPLATE_PROFILE"] = "tokenizer"
        return preset
    }

    private func applyingGemma4Defaults() -> TargetPreset {
        var preset = self
        // Gemma assistant bundles have their own runtime contract. Benchmark's
        // Qwen burst profile must not leak into that path.
        preset.profile = nil
        // Assistant-pair acceptance collapses by draft position
        // ([68.5, 45, 31, 21, 16]% measured 2026-07-03), making blocks
        // past 2 EV-negative: depth 6 decoded 23-24.5 tok/s in-app vs
        // 30.9 at depth 2 (+27%). Matches the engine's default cap; a
        // measured per-model tune still overrides via
        // resolvedDraftControlValue.
        preset.depth = 2
        preset.temperature = 1.0
        preset.topP = 0.95
        preset.topK = 64
        preset.draftTemperature = 1.0
        preset.draftTopP = 0.95
        preset.draftTopK = 64
        preset.chatTemplateProfile = "tokenizer"
        preset.reasoningParser = "gemma4"
        preset.adaptivePolicy = nil
        preset.adaptiveMinDepth = nil
        preset.adaptiveEVBaseDepth = nil
        preset.adaptiveEVWarmupFullDepthCycles = nil
        preset.adaptiveEVExplorationInterval = nil
        if preset.reasoning == nil {
            preset.reasoning = "auto"
        }
        preset.environment["MTPLX_CHAT_TEMPLATE_PROFILE"] = "tokenizer"
        return preset
    }

    private func applyingStepDefaults(
        processEnvironment: [String: String]
    ) -> TargetPreset {
        var preset = self
        // The Step product lane is the measured q4-aware D1 path from
        // the Step 3.7 work: D2/D3, adapter merge, and target-prefix
        // verifier all lost the broad gate and stay out of app defaults.
        preset.profile = nil
        preset.depth = 1
        preset.verifyStrategy = "trim_commit"
        preset.verifyCore = "stock"
        preset.mtpAdapterPath = Self.step37AdapterPath(
            processEnvironment: processEnvironment
        )
        preset.mergeMTPAdapter = false
        preset.mtpQuantBits = 4
        preset.mtpQuantGroupSize = 64
        preset.mtpQuantMode = "affine"
        preset.temperature = 0.6
        preset.topP = 0.95
        preset.topK = 20
        preset.draftTemperature = 0.6
        preset.draftTopP = 0.95
        preset.draftTopK = 20
        preset.chatTemplateProfile = "tokenizer"
        preset.reasoning = "auto"
        preset.preserveThinking = "auto"
        preset.reasoningParser = "step3p5"
        preset.reasoningEffort = "low"
        preset.acceptsSettingsReasoning = true
        preset.adaptivePolicy = nil
        preset.adaptiveMinDepth = nil
        preset.adaptiveEVBaseDepth = nil
        preset.adaptiveEVWarmupFullDepthCycles = nil
        preset.adaptiveEVExplorationInterval = nil
        preset.environment["MTPLX_CHAT_TEMPLATE_PROFILE"] = "tokenizer"
        preset.environment.removeValue(forKey: "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE")
        preset.environment.removeValue(forKey: "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT")
        preset.environment.removeValue(forKey: "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q")
        preset.environment.removeValue(forKey: "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q")
        return preset
    }

    private static func step37AdapterPath(
        processEnvironment: [String: String]
    ) -> String? {
        if let override = processEnvironment["MTPLX_STEP_MTP_ADAPTER"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty
        {
            return override
        }

        if let bundled = bundledStep37AdapterPath() {
            return bundled
        }

        var packageRoot = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 {
            packageRoot.deleteLastPathComponent()
        }
        let sourceResourceCandidate = packageRoot
            .appendingPathComponent("Sources")
            .appendingPathComponent("MTPLXAppCore")
            .appendingPathComponent("Resources")
            .appendingPathComponent("StepAdapters")
            .appendingPathComponent("\(step37AdapterResourceName).npz")
        if FileManager.default.fileExists(atPath: sourceResourceCandidate.path) {
            return sourceResourceCandidate.path
        }

        var cursor = URL(fileURLWithPath: #filePath)
        for _ in 0..<6 {
            cursor.deleteLastPathComponent()
        }
        let candidate = cursor
            .appendingPathComponent("outputs")
            .appendingPathComponent("adapters")
            .appendingPathComponent("\(step37AdapterResourceName).npz")
        return FileManager.default.fileExists(atPath: candidate.path)
            ? candidate.path
            : nil
    }

    private static func bundledStep37AdapterPath() -> String? {
        guard let url = Bundle.main.url(
            forResource: step37AdapterResourceName,
            withExtension: "npz",
            subdirectory: "StepAdapters"
        ) else {
            return nil
        }
        return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
    }

    /// Per-target defaults, mirroring `mtplx start <target>` from the
    /// release CLI. Verified against `mtplx start opencode --dry-run
    /// --json` in the V1 release LOG.
    static func preset(
        for target: LaunchTarget?,
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment
    ) -> TargetPreset {
        guard let target else { return TargetPreset(reasoning: "auto") }
        switch target {
        case .chat:
            // In-app chat is one foreground stream. Keep its daemon launch
            // aligned with the old browser WebUI path; coding-agent runtime
            // extras belong to Pi/OpenCode/custom-client targets, not plain
            // chat.
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "solo",
                reasoning: "auto"
            )
        case .pi:
            // Pi is still a coding-agent surface: its own prompt is
            // shorter than OpenCode's, but tool-history turns can still
            // cross the long-context verify-cost cliff. Give it the
            // coding-agent batch lane without OpenCode's four-slot
            // sidecar posture, keep SSD off by default, use the same
            // measured sampler as OpenCode. Reasoning is app-owned; the
            // preset must not silently enable thinking behind the UI.
            var piEnv = codingAgentRuntimeEnvironment(
                processEnvironment: processEnvironment
            )
            // Leave long-context depth policy to the sustained runtime profile.
            // The launch-readiness Pi runs showed D2 is the current failing lane
            // above 20k, so the app must not silently cap Pi below its configured
            // depth before the runtime can measure the actual request.
            piEnv["MTPLX_TOOL_RESULT_COMPACT_THRESHOLD_CHARS"] = "1200"
            piEnv["MTPLX_ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES"] = "32"
            piEnv["MTPLX_ACTIVE_READ_INSPECTION_LINE_MAX_CHARS"] = "180"
            piEnv["MTPLX_ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES"] = "32"
            piEnv["MTPLX_ACTIVE_TOOL_RESULT_LINE_MAX_CHARS"] = "220"
            return TargetPreset(
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                maxActiveRequests: 2,
                decodeBatchMax: 2,
                batchWaitMs: 50,
                prefillChunkTokens: 2048,
                topP: 0.95,
                topK: 20,
                draftTemperature: 0.6,
                draftTopP: 0.95,
                draftTopK: 20,
                toolPromptMode: "hybrid",
                chatTemplateProfile: "local_qwen36",
                adaptivePolicy: "expected_value",
                adaptiveMinDepth: 1,
                adaptiveEVBaseDepth: 2,
                adaptiveEVWarmupFullDepthCycles: 4,
                adaptiveEVExplorationInterval: 32,
                reasoning: "auto",
                preserveThinking: "auto",
                environment: piEnv
            )
        case .openWebUI:
            // Solo serving; batching off so a single chat request keeps
            // full MTP throughput. Response length is owned by the
            // request/client or explicit live settings, never hidden
            // launch presets.
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "solo",
                reasoning: "auto"
            )
        case .openCode:
            // OpenCode's launch card promises D3 MTP. Keep that as the default
            // product path: the fair AR batch lane is useful for explicit
            // throughput experiments, but it is slower than solo MTP on current
            // coding-agent turns and should not silently replace the path users
            // chose. Keep depth 3 literal here: adaptive EV measured well on
            // some long contexts, but it starves short OpenCode turns of real
            // depth-3 drafts and drops the Desktop greeting path back into the
            // 30 tok/s band.
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "latency",
                prefillChunkTokens: 2048,
                depth: 3,
                ssdSessionCache: "on",
                temperature: 0.6,
                topP: 0.95,
                topK: 20,
                draftTemperature: 0.7,
                draftTopP: 0.95,
                draftTopK: 20,
                toolPromptMode: "hybrid",
                chatTemplateProfile: "local_qwen36",
                reasoning: "auto",
                environment: codingAgentRuntimeEnvironment(
                    processEnvironment: processEnvironment
                )
            )
        case .hermes:
            // Hermes is a foreground coding agent, not a generic batch client.
            // Keep it on the measured OpenCode latency lane so Settings'
            // throughput/agent batching experiments cannot silently slow the
            // agent chat path.
            var env = codingAgentRuntimeEnvironment(
                processEnvironment: processEnvironment
            )
            env["MTPLX_CLIENT"] = "hermes"
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "latency",
                prefillChunkTokens: 2048,
                ssdSessionCache: "on",
                temperature: 0.6,
                topP: 1.0,
                topK: 20,
                draftTemperature: 0.6,
                draftTopP: 1.0,
                draftTopK: 20,
                toolPromptMode: "hybrid",
                chatTemplateProfile: "local_qwen36",
                adaptivePolicy: "expected_value",
                adaptiveMinDepth: 1,
                adaptiveEVBaseDepth: 2,
                adaptiveEVWarmupFullDepthCycles: 4,
                adaptiveEVExplorationInterval: 32,
                reasoning: "auto",
                environment: env
            )
        case .other:
            // Custom OpenAI/Anthropic-compatible client. The user picks
            // their own port + API key in the LaunchOverlay inline
            // form; pick a balanced preset that works for both single
            // and 2-3 client agents (Cursor, Codex CLI, Claude Code).
            return TargetPreset(
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                maxActiveRequests: 4,
                decodeBatchMax: 4,
                batchWaitMs: 50,
                prefillChunkTokens: 2048,
                ssdSessionCache: "on"
            )
        case .benchmark:
            // Native AIME runner: one long math stream at a time.
            // AIME is a sustained 30-question benchmark. Do not force the
            // Qwen cold-burst profile here; the configured runtime profile
            // must remain the source of truth so Settings and first-run
            // defaults actually apply.
            return TargetPreset(
                schedulerMode: "serial",
                batchingPreset: "latency",
                prefillChunkTokens: 2048,
                ssdSessionCache: "off",
                topP: 0.95,
                topK: 20,
                draftTemperature: 0.6,
                draftTopP: 0.95,
                draftTopK: 20,
                reasoning: "auto"
            )
        }
    }
}
