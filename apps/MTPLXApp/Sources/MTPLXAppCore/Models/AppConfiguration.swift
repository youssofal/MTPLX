import Foundation

// MARK: - Lenient settings decoding

/// One stored settings value that could not be decoded and fell back to
/// its default: `path` is the JSON location (`port`,
/// `custom_models[1]`, `tuned_control_records_by_model.<model>`), `reason`
/// the decoder's complaint.
public struct SettingsDecodeIssue: Equatable, Sendable, CustomStringConvertible {
    public let path: String
    public let reason: String

    public init(path: String, reason: String) {
        self.path = path
        self.reason = reason
    }

    public var description: String { "\(path): \(reason)" }

    static func path(_ codingPath: [any CodingKey]) -> String {
        var text = ""
        for key in codingPath {
            if let index = key.intValue {
                text += "[\(index)]"
            } else {
                text += text.isEmpty ? key.stringValue : ".\(key.stringValue)"
            }
        }
        return text
    }

    /// A short, log-friendly rendering of a decoding failure.
    static func describe(_ error: Error) -> String {
        switch error as? DecodingError {
        case .typeMismatch(let type, let context)?:
            return "expected \(type)" + (context.debugDescription.isEmpty ? "" : " (\(context.debugDescription))")
        case .valueNotFound(let type, _)?:
            return "expected \(type), found null"
        case .keyNotFound(let key, _)?:
            return "missing \(key.stringValue)"
        case .dataCorrupted(let context)?:
            return context.debugDescription.isEmpty ? "malformed value" : context.debugDescription
        default:
            return String(describing: error)
        }
    }
}

/// Collects the fields a settings decode had to degrade. Installed in
/// `Decoder.userInfo` by the settings store so the configuration decoder
/// and the per-element wrappers can report without changing the
/// `Codable` surface; a decode without a collector still degrades
/// leniently and simply has nowhere to report.
public final class SettingsDecodeIssues: @unchecked Sendable {
    public static let userInfoKey = CodingUserInfoKey(rawValue: "com.mtplx.app.settingsDecodeIssues")!

    private let lock = NSLock()
    private var issues: [SettingsDecodeIssue] = []

    public init() {}

    public func record(path: [any CodingKey], error: Error) {
        let issue = SettingsDecodeIssue(
            path: SettingsDecodeIssue.path(path),
            reason: SettingsDecodeIssue.describe(error)
        )
        lock.withLock { issues.append(issue) }
    }

    public var all: [SettingsDecodeIssue] {
        lock.withLock { issues }
    }

    static func installed(in decoder: any Decoder) -> SettingsDecodeIssues? {
        decoder.userInfo[userInfoKey] as? SettingsDecodeIssues
    }
}

/// Decodes one element of a collection, or records the failure and yields
/// nil, so one malformed custom model or tuned record never sinks the rest
/// of its collection.
struct LenientElement<Value: Decodable>: Decodable {
    let value: Value?

    init(from decoder: any Decoder) {
        do {
            value = try Value(from: decoder)
        } catch {
            value = nil
            SettingsDecodeIssues.installed(in: decoder)?.record(path: decoder.codingPath, error: error)
        }
    }
}

extension KeyedDecodingContainer {
    /// `decodeIfPresent` that treats a wrong-typed or malformed value like
    /// a missing one: the field falls back to its default and the failure
    /// is recorded with its coding path instead of failing the whole file.
    func lenientDecodeIfPresent<Value: Decodable>(
        _ type: Value.Type,
        forKey key: Key,
        issues: SettingsDecodeIssues?
    ) -> Value? {
        do {
            return try decodeIfPresent(type, forKey: key)
        } catch {
            issues?.record(path: codingPath + [key], error: error)
            return nil
        }
    }

    /// Lenient array decode: a wrong-typed array falls back to nil, and a
    /// malformed element is skipped while its siblings load.
    func lenientDecodeArrayIfPresent<Element: Decodable>(
        of type: Element.Type,
        forKey key: Key,
        issues: SettingsDecodeIssues?
    ) -> [Element]? {
        lenientDecodeIfPresent([LenientElement<Element>].self, forKey: key, issues: issues)?
            .compactMap(\.value)
    }

    /// Lenient string-keyed dictionary decode with the same per-entry
    /// tolerance as the array form.
    func lenientDecodeDictionaryIfPresent<Value: Decodable>(
        of type: Value.Type,
        forKey key: Key,
        issues: SettingsDecodeIssues?
    ) -> [String: Value]? {
        lenientDecodeIfPresent([String: LenientElement<Value>].self, forKey: key, issues: issues)?
            .compactMapValues(\.value)
    }
}

public struct TunedControlRecord: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var modelID: String
    public var modelFamily: String
    public var backendID: String
    public var controlField: String
    public var controlValue: Int
    public var candidates: [String]
    public var tunedAt: Date

    public init(
        schemaVersion: Int = 1,
        modelID: String,
        modelFamily: String,
        backendID: String,
        controlField: String,
        controlValue: Int,
        candidates: [String],
        tunedAt: Date
    ) {
        self.schemaVersion = schemaVersion
        self.modelID = modelID
        self.modelFamily = modelFamily
        self.backendID = backendID
        self.controlField = controlField
        self.controlValue = controlValue
        self.candidates = candidates
        self.tunedAt = tunedAt
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelID = "model_id"
        case modelFamily = "model_family"
        case backendID = "backend_id"
        case controlField = "control_field"
        case controlValue = "control_value"
        case candidates
        case tunedAt = "tuned_at"
    }
}

public struct MTPLXAppConfiguration: Codable, Equatable, Sendable {
    /// The daemon's own default (`mtplx serve --stream-stall-deadline-s`);
    /// the flag is only passed when the user changed it.
    public static let defaultStreamStallDeadlineSeconds: Double = 300

    public var executablePath: String?
    public var model: String
    public var profile: String
    public var host: String
    public var port: Int
    public var generationMode: String
    public var loadMTP: Bool
    public var schedulerMode: String
    public var batchingPreset: String
    /// Explicit scheduling picker state. `target-default` means the
    /// selected launch target owns scheduler/batching defaults; concrete
    /// values like `latency`, `throughput`, and `agent` force that mode
    /// even for Pi/OpenCode.
    public var schedulingPreset: String
    public var maxActiveRequests: Int?
    public var decodeBatchMax: Int?
    public var batchWaitMs: Double?
    public var prefillChunkTokens: Int?
    public var experimentalMTPCohorts: Bool
    /// Models served on /v1/embeddings, as `REF` or `REF=SERVED_ID`.
    public var embeddingModels: [String]
    /// Models served on /v1/rerank. The same REF in both roles loads once.
    public var rerankerModels: [String]
    /// How many retrieval models stay resident before the oldest is unloaded.
    public var retrievalMaxResident: Int
    /// Seconds of inactivity before retrieval weights are released; 0 = never.
    public var retrievalIdleTimeout: Double
    /// Seconds a stream may wait on a model owner that makes no progress
    /// before it is failed; 0 turns the watchdog off (issue #448). Passed to
    /// the daemon as `--stream-stall-deadline-s`, because a LaunchServices
    /// app never sees a shell `export`.
    public var streamStallDeadlineSeconds: Double
    public var ramSessionCachePolicy: String
    public var ramSessionBlockPrefixRestore: Bool
    public var ramSessionCacheMaxEntries: Int
    public var ramSessionCacheMaxSize: String
    public var ramSessionCachePerSessionMaxSize: String
    public var pagedKVQuantization: String
    public var ssdSessionCache: String
    public var ssdSessionCacheDir: String?
    public var ssdSessionCacheMaxSize: String
    public var ssdSessionCacheMinPrefixTokens: Int
    public var contextWindow: Int?
    /// Family that owns the explicit context-window override. Missing
    /// means legacy Qwen-era settings; those do not apply to Gemma,
    /// Step, GLM, DeepSeek, or custom models.
    public var contextWindowModelFamily: String?
    /// Durable sampler controls from the Settings popover. These are
    /// sent to a live daemon and carried into app-native runs such as
    /// Chat and AIME. External client targets keep their own measured
    /// launch presets so one experiment cannot silently slow them.
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    /// One-shot guard for the 2026-07-02 legacy-sampler migration: once
    /// true, sanitize never touches the sampler again, so a deliberate
    /// 1.0/0.95/20 (or anything else) persists. Fresh configs start
    /// migrated-false, run the check once, and flip it.
    public var samplerLegacyTripleMigrated: Bool
    /// One-shot guards for the 2026-07-03 turbo-release migrations
    /// (legacy default profile -> "auto"; 250 ms stream cadence -> 100).
    public var profileLegacyDefaultMigrated: Bool
    public var streamCadenceMigrated: Bool
    /// OpenAI-style presence penalty (0 = exact no-op; Qwen recommends 0
    /// for coding). Round-trips through the daemon's live settings like
    /// temperature/topP/topK.
    public var presencePenalty: Double?
    public var reasoning: String?
    public var reasoningEffort: String?
    /// Family that owns the persisted sampler/reasoning values above.
    /// Missing means legacy Qwen-era settings; those are back-compatible
    /// only with Qwen families.
    public var liveSettingsModelFamily: String?
    public var apiKey: String?
    public var enableThermalPolling: Bool
    public var streamSnapshotIntervalMs: Int
    public var performanceLock: Bool
    public var launchDaemonOnOpen: Bool
    /// Opt-in app-managed recovery after an abnormal daemon exit. It applies
    /// to launches created in this app session; the launch command and API key
    /// remain process-memory-only in DaemonSupervisor.
    public var automaticDaemonRestart: Bool
    /// When on, the app launches Hermes in auto-approve ("YOLO") mode so
    /// the agent runs tools without prompting. Off makes Hermes ask for
    /// approval. Applies the next time Hermes is started.
    public var hermesAutoApprove: Bool
    /// Fan policy for daemon launches and live mode changes.
    /// `smart` is the V1 default: fans boost only during visible generation.
    public var fanMode: String
    /// Legacy compatibility for configs saved before `fan_mode`.
    /// New code should use `fanMode`; this mirrors `fanMode == "max"`.
    public var pinFansAtMaxOnStart: Bool
    /// Most recently picked `mtplx start <target>` surface. Menu
    /// commands and startup defaults can reuse it, but the stopped-state
    /// Play button still opens the picker so users are never trapped in
    /// yesterday's client surface.
    public var lastLaunchTarget: String
    /// Project root handed to terminal coding agents launched by MTPLX.
    /// Pi uses this as its shell cwd; Hermes uses it for both terminal
    /// commands and file tools, so relative paths stay anchored to the
    /// same workspace.
    public var hermesWorkspacePath: String
    /// Last Hermes profile chosen from the app-owned Hermes agent
    /// picker. The app uses this only to resume the last agent when
    /// the user presses Play again; it never mutates Hermes profile
    /// config on disk.
    public var lastHermesProfile: String?
    /// Durable Hermes session id (`session_key` / saved session id)
    /// for the last app-owned Hermes agent. Ephemeral dashboard ports
    /// and auth tokens are intentionally not persisted.
    public var lastHermesSessionID: String?
    /// User-facing title for the last Hermes session, if Hermes had
    /// one. This is display-only and may be stale if the session was
    /// renamed outside MTPLXApp.
    public var lastHermesSessionTitle: String?
    /// Timestamp when the user finished the first-launch onboarding
    /// flow. `nil` means onboarding has not yet been completed — the
    /// app gates the entire shell on this and renders the onboarding
    /// experience instead. Set once at the end of `FinishStep` and
    /// never cleared by the runtime.
    public var onboardingCompletedAt: Date?
    /// When the user was asked to pick the app language. Onboarding's
    /// Language step stamps it for new installs; every install that
    /// finished onboarding before that step existed (pre-2.11) is asked
    /// once, on the first launch after the update, by the language
    /// prompt sheet — and stamped no matter how the sheet was closed.
    /// `nil` means "never asked". Never cleared by the runtime.
    public var languagePromptCompletedAt: Date?
    /// Depth picked by onboarding or the most recent `mtplx tune` run
    /// on this Mac. Threaded into `mtplx serve --depth N` so every
    /// daemon launch honours the selected value. `nil` means the app
    /// should use the daemon default.
    public var lastTunedDepth: Int?
    /// When the depth above was measured by a real tune run. Safe
    /// defaults chosen during onboarding intentionally leave this nil.
    public var lastTunedAt: Date?
    /// Versioned tuning state for the selected model control. The
    /// legacy integer above is still decoded for Qwen back-compat, but
    /// non-Qwen families must never inherit it.
    public var tunedControlRecord: TunedControlRecord?
    /// Model-scoped tune records keyed by local install paths and HF
    /// repo ids. This prevents a tune result from one downloaded model
    /// from leaking into another model that happens to share a family.
    public var tunedControlRecordsByModel: [String: TunedControlRecord]
    /// User-added Hugging Face models shown in the top-left model
    /// picker. Official models stay in `MTPLXModelOption.officialCatalog`;
    /// this array is only the user's personal additions.
    public var customModels: [MTPLXModelOption]
    /// User's Hugging Face handle, captured the first time they
    /// publish a forged model so subsequent Publish flows can pre-fill
    /// the `<handle>/<branded-name>` repo field. Persisted only when
    /// the user explicitly publishes; we never sniff this from the
    /// Keychain token or filesystem.
    public var huggingFaceHandle: String?
    /// Optional Hugging Face endpoint override for model downloads
    /// (issue #96: huggingface.co is blocked in mainland China). Applied
    /// to daemon and pull subprocesses as HF_ENDPOINT; the stored HF
    /// token never travels to a non-official endpoint.
    public var hfEndpoint: String?
    /// Optional hard memory cap for the daemon, in whole GB (issue #431).
    /// `nil` leaves the engine's own plan in charge (75% of RAM, bounded);
    /// a value is passed down as MTPLX_MEMORY_LIMIT_BYTES so the launched
    /// daemon sizes its allocator caps and its context fit against the
    /// number the user can see in Settings.
    public var memoryLimitGB: Int?
    /// "I know what I'm doing" swap opt-in (issue #427). On, the daemon
    /// launches with MTPLX_ALLOW_SWAP=1, which drops the machine-fit clamp
    /// on the context window and accepts paging instead of refusing the
    /// request. Off is the shipped default: the fit clamp stays.
    public var allowSwap: Bool

    public init(
        executablePath: String? = nil,
        model: String = MTPLXAppConfiguration.defaultLocalModelPath(),
        profile: String = "auto",
        host: String = "127.0.0.1",
        port: Int = 8000,
        generationMode: String = "mtp",
        loadMTP: Bool = true,
        schedulerMode: String = "serial",
        batchingPreset: String = "latency",
        schedulingPreset: String = "target-default",
        maxActiveRequests: Int? = nil,
        decodeBatchMax: Int? = nil,
        batchWaitMs: Double? = nil,
        prefillChunkTokens: Int? = nil,
        experimentalMTPCohorts: Bool = false,
        embeddingModels: [String] = [],
        rerankerModels: [String] = [],
        retrievalMaxResident: Int = 2,
        retrievalIdleTimeout: Double = 0,
        streamStallDeadlineSeconds: Double = MTPLXAppConfiguration.defaultStreamStallDeadlineSeconds,
        ramSessionCachePolicy: String = "target-default",
        ramSessionBlockPrefixRestore: Bool = true,
        ramSessionCacheMaxEntries: Int = 8,
        ramSessionCacheMaxSize: String = "auto",
        ramSessionCachePerSessionMaxSize: String = "auto",
        pagedKVQuantization: String = "off",
        ssdSessionCache: String = "target-default",
        ssdSessionCacheDir: String? = nil,
        ssdSessionCacheMaxSize: String = "auto",
        ssdSessionCacheMinPrefixTokens: Int = 512,
        contextWindow: Int? = nil,
        contextWindowModelFamily: String? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        samplerLegacyTripleMigrated: Bool = false,
        profileLegacyDefaultMigrated: Bool = false,
        streamCadenceMigrated: Bool = false,
        presencePenalty: Double? = nil,
        reasoning: String? = nil,
        reasoningEffort: String? = nil,
        liveSettingsModelFamily: String? = nil,
        apiKey: String? = nil,
        enableThermalPolling: Bool = false,
        streamSnapshotIntervalMs: Int = 100,
        performanceLock: Bool = false,
        launchDaemonOnOpen: Bool = false,
        automaticDaemonRestart: Bool = false,
        hermesAutoApprove: Bool = true,
        fanMode: String? = nil,
        pinFansAtMaxOnStart: Bool = false,
        lastLaunchTarget: String = LaunchTarget.chat.rawValue,
        hermesWorkspacePath: String = MTPLXAppConfiguration.defaultHermesWorkspacePath(),
        lastHermesProfile: String? = nil,
        lastHermesSessionID: String? = nil,
        lastHermesSessionTitle: String? = nil,
        onboardingCompletedAt: Date? = nil,
        languagePromptCompletedAt: Date? = nil,
        lastTunedDepth: Int? = nil,
        lastTunedAt: Date? = nil,
        tunedControlRecord: TunedControlRecord? = nil,
        tunedControlRecordsByModel: [String: TunedControlRecord] = [:],
        customModels: [MTPLXModelOption] = [],
        huggingFaceHandle: String? = nil,
        hfEndpoint: String? = nil,
        memoryLimitGB: Int? = nil,
        allowSwap: Bool = false
    ) {
        self.executablePath = executablePath
        self.model = model
        self.profile = profile
        self.host = host
        self.port = port
        self.generationMode = generationMode
        self.loadMTP = loadMTP
        self.schedulerMode = schedulerMode
        self.batchingPreset = batchingPreset
        self.schedulingPreset = Self.normalizedSchedulingPreset(
            schedulingPreset,
            schedulerMode: schedulerMode,
            batchingPreset: batchingPreset
        )
        self.maxActiveRequests = maxActiveRequests
        self.decodeBatchMax = decodeBatchMax
        self.batchWaitMs = batchWaitMs
        self.prefillChunkTokens = prefillChunkTokens
        self.experimentalMTPCohorts = experimentalMTPCohorts
        self.embeddingModels = embeddingModels
        self.rerankerModels = rerankerModels
        self.retrievalMaxResident = retrievalMaxResident
        self.retrievalIdleTimeout = retrievalIdleTimeout
        self.streamStallDeadlineSeconds = max(0, streamStallDeadlineSeconds)
        self.ramSessionCachePolicy = ramSessionCachePolicy
        self.ramSessionBlockPrefixRestore = ramSessionBlockPrefixRestore
        self.ramSessionCacheMaxEntries = ramSessionCacheMaxEntries
        self.ramSessionCacheMaxSize = ramSessionCacheMaxSize
        self.ramSessionCachePerSessionMaxSize = ramSessionCachePerSessionMaxSize
        self.pagedKVQuantization = pagedKVQuantization
        self.ssdSessionCache = ssdSessionCache
        self.ssdSessionCacheDir = ssdSessionCacheDir
        self.ssdSessionCacheMaxSize = ssdSessionCacheMaxSize
        self.ssdSessionCacheMinPrefixTokens = ssdSessionCacheMinPrefixTokens
        self.contextWindow = contextWindow
        self.contextWindowModelFamily = contextWindowModelFamily
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.samplerLegacyTripleMigrated = samplerLegacyTripleMigrated
        self.profileLegacyDefaultMigrated = profileLegacyDefaultMigrated
        self.streamCadenceMigrated = streamCadenceMigrated
        self.presencePenalty = presencePenalty
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.liveSettingsModelFamily = liveSettingsModelFamily
        self.apiKey = apiKey
        self.enableThermalPolling = enableThermalPolling
        self.streamSnapshotIntervalMs = streamSnapshotIntervalMs
        self.performanceLock = performanceLock
        self.launchDaemonOnOpen = launchDaemonOnOpen
        self.automaticDaemonRestart = automaticDaemonRestart
        self.hermesAutoApprove = hermesAutoApprove
        let resolvedFanMode = MTPLXFanMode.normalized(
            fanMode ?? (pinFansAtMaxOnStart ? MTPLXFanMode.max.rawValue : MTPLXFanMode.smart.rawValue)
        )
        self.fanMode = resolvedFanMode.rawValue
        self.pinFansAtMaxOnStart = resolvedFanMode == .max
        self.lastLaunchTarget = lastLaunchTarget
        self.hermesWorkspacePath = Self.normalizedHermesWorkspacePath(hermesWorkspacePath)
        self.lastHermesProfile = lastHermesProfile
        self.lastHermesSessionID = lastHermesSessionID
        self.lastHermesSessionTitle = lastHermesSessionTitle
        self.onboardingCompletedAt = onboardingCompletedAt
        self.languagePromptCompletedAt = languagePromptCompletedAt
        self.lastTunedDepth = lastTunedDepth
        self.lastTunedAt = lastTunedAt
        self.tunedControlRecord = tunedControlRecord
        self.tunedControlRecordsByModel = tunedControlRecordsByModel
        self.customModels = customModels
        self.huggingFaceHandle = huggingFaceHandle
        self.hfEndpoint = hfEndpoint
        self.memoryLimitGB = Self.normalizedMemoryLimitGB(memoryLimitGB)
        self.allowSwap = allowSwap
    }

    /// Whether this launch owes the user the one-time language prompt:
    /// onboarding is done (a fresh install is still inside onboarding,
    /// whose Language step is the prompt) and nobody has asked yet.
    public var shouldOfferLanguagePrompt: Bool {
        onboardingCompletedAt != nil && languagePromptCompletedAt == nil
    }

    /// Fresh installs must be portable. Installed local copies are discovered
    /// by the model catalog; the default configuration should never point at
    /// a developer machine path.
    public static func defaultLocalModelPath() -> String {
        // Qwen 3.8 Optimized Speed is the recommended pick and fresh-install
        // default (2026-08-15 release); mirrors DEFAULT_HF_MODEL_ID in
        // mtplx/profiles.py.
        return "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    }

    public static func defaultHermesWorkspacePath() -> String {
        let fileManager = FileManager.default
        if let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first,
           fileManager.fileExists(atPath: documents.path) {
            return documents.path
        }
        return NSHomeDirectory()
    }

    public static func normalizedHermesWorkspacePath(_ raw: String?) -> String {
        let trimmed = (raw ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return defaultHermesWorkspacePath()
        }
        return (trimmed as NSString).expandingTildeInPath
    }

    public mutating func rememberCustomModel(repoID: String) {
        guard let option = MTPLXModelOption.customHuggingFaceModel(repoID: repoID) else {
            return
        }
        guard MTPLXModelOption.option(matching: option.hfModelID) == nil else {
            return
        }
        customModels.removeAll { existing in
            existing.matches(option.hfModelID) || option.matches(existing.hfModelID)
        }
        customModels.append(option)
    }

    /// Persist a locally-forged model into the picker. Called by the
    /// Forge wizard's Registered stage when the build completes;
    /// dedup is by id (same branded name re-forged) AND by local
    /// path (an existing entry pointing at the same directory wins).
    public mutating func rememberForgedModel(
        brandedName: String,
        localPath: String,
        sizeBytes: Int64 = 0,
        peakMemoryGiB: Double = 0
    ) {
        guard let option = MTPLXModelOption.forgedModel(
            brandedName: brandedName,
            localPath: localPath,
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakMemoryGiB
        ) else { return }
        customModels.removeAll { existing in
            existing.id == option.id || existing.localCandidates.contains(localPath)
        }
        customModels.append(option)
    }

    /// Persist a model folder the user chose into the picker, so switching
    /// back to it later is a click instead of the path again. Dedup is by
    /// path (an entry that already points at this folder — forged, Hugging
    /// Face, or local — is already its row and stays) and by id (a newly
    /// chosen folder wins over an older folder of the same name). A
    /// catalog model's own install directory is never recorded: the
    /// catalog row already launches it.
    public mutating func rememberLocalFolderModel(path: String) {
        guard let option = MTPLXModelOption.localFolderModel(path: path) else { return }
        let folder = option.hfModelID
        guard !MTPLXModelOption.officialCatalog.contains(where: { $0.hasLocalCandidate(at: folder) }),
              !customModels.contains(where: { $0.hasLocalCandidate(at: folder) })
        else { return }
        customModels.removeAll { $0.id == option.id }
        customModels.append(option)
    }

    public mutating func applyForgeRuntimeDefaults(
        modelPath: String,
        verification: ForgeVerification,
        sourceRepo: String? = nil,
        tunedAt: Date = Date()
    ) {
        guard verification.bestDepth > 0 else { return }
        let family = Self.forgeRuntimeFamily(modelPath: modelPath, sourceRepo: sourceRepo)
        model = modelPath
        generationMode = "mtp"
        loadMTP = true
        liveSettingsModelFamily = family
        if MTPLXModelOption.supportsTune(family: family) {
            lastTunedDepth = verification.bestDepth
        }
        lastTunedAt = tunedAt
        let record = TunedControlRecord(
            modelID: modelPath,
            modelFamily: family,
            backendID: Self.forgeBackendID(for: family),
            controlField: Self.forgeTuneControlField(for: family),
            controlValue: verification.bestDepth,
            candidates: Self.forgeTuneCandidates(for: family),
            tunedAt: tunedAt
        )
        tunedControlRecord = record
        for key in Self.tuneRecordKeys(modelPath: modelPath, repoID: sourceRepo) {
            tunedControlRecordsByModel[key] = record
        }
    }

    private static func forgeRuntimeFamily(modelPath: String, sourceRepo: String?) -> String {
        let pathFamily = MTPLXModelOption.modelFamily(for: modelPath)
        if pathFamily != "unknown" { return pathFamily }
        if let sourceRepo {
            let sourceFamily = MTPLXModelOption.modelFamily(for: sourceRepo)
            if sourceFamily != "unknown" { return sourceFamily }
        }
        return pathFamily
    }

    private static func forgeTuneControlField(for family: String) -> String {
        family == "gemma4" ? "draft_block_size" : "depth"
    }

    private static func forgeBackendID(for family: String) -> String {
        switch family {
        case "gemma4": return "gemma4_assistant"
        case "step": return "step3p5_mtp"
        case "deepseek": return "deepseek_v3_mtp"
        case "glm": return "glm4_moe_mtp"
        default: return "qwen3_next"
        }
    }

    private static func forgeTuneCandidates(for family: String) -> [String] {
        family == "gemma4"
            ? ["2", "3", "4", "5", "6", "7", "8"]
            : ["1", "2", "3"]
    }

    enum CodingKeys: String, CodingKey {
        case executablePath = "executable_path"
        case model
        case profile
        case host
        case port
        case generationMode = "generation_mode"
        case loadMTP = "load_mtp"
        case schedulerMode = "scheduler_mode"
        case batchingPreset = "batching_preset"
        case schedulingPreset = "scheduling_preset"
        case maxActiveRequests = "max_active_requests"
        case decodeBatchMax = "decode_batch_max"
        case batchWaitMs = "batch_wait_ms"
        case prefillChunkTokens = "prefill_chunk_tokens"
        case experimentalMTPCohorts = "experimental_mtp_cohorts"
        case embeddingModels = "embedding_models"
        case rerankerModels = "reranker_models"
        case retrievalMaxResident = "retrieval_max_resident"
        case retrievalIdleTimeout = "retrieval_idle_timeout"
        case streamStallDeadlineSeconds = "stream_stall_deadline_s"
        case ramSessionCachePolicy = "ram_session_cache_policy"
        case ramSessionBlockPrefixRestore = "ram_session_block_prefix_restore"
        case ramSessionCacheMaxEntries = "ram_session_cache_max_entries"
        case ramSessionCacheMaxSize = "ram_session_cache_max_size"
        case ramSessionCachePerSessionMaxSize = "ram_session_cache_per_session_max_size"
        case pagedKVQuantization = "paged_kv_quantization"
        case ssdSessionCache = "ssd_session_cache"
        case ssdSessionCacheDir = "ssd_session_cache_dir"
        case ssdSessionCacheMaxSize = "ssd_session_cache_max_size"
        case ssdSessionCacheMinPrefixTokens = "ssd_session_cache_min_prefix_tokens"
        case contextWindow = "context_window"
        case contextWindowModelFamily = "context_window_model_family"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case samplerLegacyTripleMigrated = "sampler_legacy_triple_migrated"
        case profileLegacyDefaultMigrated = "profile_legacy_default_migrated"
        case streamCadenceMigrated = "stream_cadence_migrated"
        case presencePenalty = "presence_penalty"
        case reasoning
        case reasoningEffort = "reasoning_effort"
        case liveSettingsModelFamily = "live_settings_model_family"
        case apiKey = "api_key"
        case enableThermalPolling = "enable_thermal_polling"
        case streamSnapshotIntervalMs = "stream_snapshot_interval_ms"
        case performanceLock = "performance_lock"
        case launchDaemonOnOpen = "launch_daemon_on_open"
        case automaticDaemonRestart = "automatic_daemon_restart"
        case hermesAutoApprove = "hermes_auto_approve"
        case fanMode = "fan_mode"
        case pinFansAtMaxOnStart = "pin_fans_at_max_on_start"
        case lastLaunchTarget = "last_launch_target"
        case hermesWorkspacePath = "hermes_workspace_path"
        case lastHermesProfile = "last_hermes_profile"
        case lastHermesSessionID = "last_hermes_session_id"
        case lastHermesSessionTitle = "last_hermes_session_title"
        case onboardingCompletedAt = "onboarding_completed_at"
        case languagePromptCompletedAt = "language_prompt_completed_at"
        case lastTunedDepth = "last_tuned_depth"
        case lastTunedAt = "last_tuned_at"
        case tunedControlRecord = "tuned_control_record"
        case tunedControlRecordsByModel = "tuned_control_records_by_model"
        case customModels = "custom_models"
        case huggingFaceHandle = "hugging_face_handle"
        case hfEndpoint = "hf_endpoint"
        case memoryLimitGB = "memory_limit_gb"
        case allowSwap = "allow_swap"
    }

    /// Field-lenient decoder. A missing key falls back to the default, and
    /// so does a wrong-typed or malformed value: one bad field (a hand edit,
    /// a downgrade after a newer build changed a type, a future record
    /// schema bump) must never make the whole file unreadable, because an
    /// unreadable file used to send the user back through onboarding and
    /// then overwrite their custom models, API key, tuned records and
    /// mirror settings with defaults. Each degraded field is reported to the
    /// `SettingsDecodeIssues` collector when the decoder carries one. The
    /// only failures that still throw are structural — the document is not
    /// a JSON object at all — and the settings store treats those as an
    /// unreadable file to be set aside, not decoded.
    public init(from decoder: Decoder) throws {
        let defaults = MTPLXAppConfiguration()
        let issues = SettingsDecodeIssues.installed(in: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func field<Value: Decodable>(_ type: Value.Type, _ key: CodingKeys) -> Value? {
            container.lenientDecodeIfPresent(type, forKey: key, issues: issues)
        }
        executablePath = field(String.self, .executablePath)
        model = field(String.self, .model) ?? defaults.model
        profile = field(String.self, .profile) ?? defaults.profile
        host = field(String.self, .host) ?? defaults.host
        port = field(Int.self, .port) ?? defaults.port
        generationMode = field(String.self, .generationMode) ?? defaults.generationMode
        loadMTP = field(Bool.self, .loadMTP) ?? defaults.loadMTP
        schedulerMode = field(String.self, .schedulerMode) ?? defaults.schedulerMode
        batchingPreset = field(String.self, .batchingPreset) ?? defaults.batchingPreset
        let decodedSchedulingPreset = field(String.self, .schedulingPreset)
        schedulingPreset = Self.normalizedSchedulingPreset(
            decodedSchedulingPreset ?? defaults.schedulingPreset,
            schedulerMode: schedulerMode,
            batchingPreset: batchingPreset,
            inferLegacyModePair: false
        )
        maxActiveRequests = field(Int.self, .maxActiveRequests)
        decodeBatchMax = field(Int.self, .decodeBatchMax)
        batchWaitMs = field(Double.self, .batchWaitMs)
        prefillChunkTokens = field(Int.self, .prefillChunkTokens)
        experimentalMTPCohorts = field(Bool.self, .experimentalMTPCohorts) ?? defaults.experimentalMTPCohorts
        embeddingModels = container.lenientDecodeArrayIfPresent(of: String.self, forKey: .embeddingModels, issues: issues)
            ?? defaults.embeddingModels
        rerankerModels = container.lenientDecodeArrayIfPresent(of: String.self, forKey: .rerankerModels, issues: issues)
            ?? defaults.rerankerModels
        retrievalMaxResident = field(Int.self, .retrievalMaxResident) ?? defaults.retrievalMaxResident
        retrievalIdleTimeout = field(Double.self, .retrievalIdleTimeout) ?? defaults.retrievalIdleTimeout
        streamStallDeadlineSeconds = max(
            0, field(Double.self, .streamStallDeadlineSeconds) ?? defaults.streamStallDeadlineSeconds
        )
        ramSessionCachePolicy = field(String.self, .ramSessionCachePolicy) ?? defaults.ramSessionCachePolicy
        ramSessionBlockPrefixRestore = field(Bool.self, .ramSessionBlockPrefixRestore) ?? defaults.ramSessionBlockPrefixRestore
        ramSessionCacheMaxEntries = field(Int.self, .ramSessionCacheMaxEntries) ?? defaults.ramSessionCacheMaxEntries
        ramSessionCacheMaxSize = field(String.self, .ramSessionCacheMaxSize) ?? defaults.ramSessionCacheMaxSize
        ramSessionCachePerSessionMaxSize = field(String.self, .ramSessionCachePerSessionMaxSize) ?? defaults.ramSessionCachePerSessionMaxSize
        pagedKVQuantization = field(String.self, .pagedKVQuantization) ?? defaults.pagedKVQuantization
        ssdSessionCache = field(String.self, .ssdSessionCache) ?? defaults.ssdSessionCache
        ssdSessionCacheDir = field(String.self, .ssdSessionCacheDir)
        ssdSessionCacheMaxSize = field(String.self, .ssdSessionCacheMaxSize) ?? defaults.ssdSessionCacheMaxSize
        ssdSessionCacheMinPrefixTokens = field(Int.self, .ssdSessionCacheMinPrefixTokens) ?? defaults.ssdSessionCacheMinPrefixTokens
        contextWindow = field(Int.self, .contextWindow)
        contextWindowModelFamily = field(String.self, .contextWindowModelFamily)
        temperature = field(Double.self, .temperature)
        topP = field(Double.self, .topP)
        topK = field(Int.self, .topK)
        samplerLegacyTripleMigrated = field(Bool.self, .samplerLegacyTripleMigrated) ?? false
        profileLegacyDefaultMigrated = field(Bool.self, .profileLegacyDefaultMigrated) ?? false
        streamCadenceMigrated = field(Bool.self, .streamCadenceMigrated) ?? false
        presencePenalty = field(Double.self, .presencePenalty)
        reasoning = field(String.self, .reasoning)
        reasoningEffort = field(String.self, .reasoningEffort)
        liveSettingsModelFamily = field(String.self, .liveSettingsModelFamily)
        apiKey = field(String.self, .apiKey)
        enableThermalPolling = field(Bool.self, .enableThermalPolling) ?? defaults.enableThermalPolling
        streamSnapshotIntervalMs = field(Int.self, .streamSnapshotIntervalMs) ?? defaults.streamSnapshotIntervalMs
        performanceLock = field(Bool.self, .performanceLock) ?? defaults.performanceLock
        launchDaemonOnOpen = field(Bool.self, .launchDaemonOnOpen) ?? defaults.launchDaemonOnOpen
        automaticDaemonRestart = field(Bool.self, .automaticDaemonRestart) ?? defaults.automaticDaemonRestart
        hermesAutoApprove = field(Bool.self, .hermesAutoApprove) ?? defaults.hermesAutoApprove
        let decodedFanMode = field(String.self, .fanMode)
        let legacyPin = field(Bool.self, .pinFansAtMaxOnStart)
        let legacyFallback: String
        if let legacyPin {
            legacyFallback = legacyPin ? MTPLXFanMode.max.rawValue : MTPLXFanMode.default.rawValue
        } else {
            legacyFallback = defaults.fanMode
        }
        let resolvedFanMode = MTPLXFanMode.normalized(decodedFanMode ?? legacyFallback)
        fanMode = resolvedFanMode.rawValue
        pinFansAtMaxOnStart = resolvedFanMode == .max
        lastLaunchTarget = field(String.self, .lastLaunchTarget) ?? defaults.lastLaunchTarget
        hermesWorkspacePath = Self.normalizedHermesWorkspacePath(
            field(String.self, .hermesWorkspacePath) ?? defaults.hermesWorkspacePath
        )
        lastHermesProfile = field(String.self, .lastHermesProfile)
        lastHermesSessionID = field(String.self, .lastHermesSessionID)
        lastHermesSessionTitle = field(String.self, .lastHermesSessionTitle)
        onboardingCompletedAt = field(Date.self, .onboardingCompletedAt)
        languagePromptCompletedAt = field(Date.self, .languagePromptCompletedAt)
        lastTunedDepth = field(Int.self, .lastTunedDepth)
        lastTunedAt = field(Date.self, .lastTunedAt)
        // A tune record with a missing or mistyped field carries nothing
        // usable, so the record is skipped as a whole and its siblings
        // load; the legacy single record follows the same rule.
        tunedControlRecord = field(TunedControlRecord.self, .tunedControlRecord)
        tunedControlRecordsByModel = container.lenientDecodeDictionaryIfPresent(
            of: TunedControlRecord.self,
            forKey: .tunedControlRecordsByModel,
            issues: issues
        ) ?? defaults.tunedControlRecordsByModel
        customModels = container.lenientDecodeArrayIfPresent(of: MTPLXModelOption.self, forKey: .customModels, issues: issues)
            ?? defaults.customModels
        huggingFaceHandle = field(String.self, .huggingFaceHandle)
        hfEndpoint = field(String.self, .hfEndpoint)
        memoryLimitGB = Self.normalizedMemoryLimitGB(field(Int.self, .memoryLimitGB))
        allowSwap = field(Bool.self, .allowSwap) ?? defaults.allowSwap
        sanitizeLaunchCriticalFields()
    }

    /// Engine-launchable allowlists. SYNC PAIR: mtplx/profiles.py
    /// PROFILE_CHOICES and mtplx GENERATION_MODES. A value outside these
    /// kills `mtplx serve` at argument parsing, which the user experiences
    /// as a daemon that is degraded on every start, so any persisted
    /// config must decode back to something launchable — or to "auto",
    /// which launches with no --profile flag at all.
    static let engineProfiles: Set<String> = [
        "stable", "performance-cold", "sustained", "turbo", "exact",
        "max-diagnostic",
    ]

    /// "auto" is persistable but never launchable: it means "use the
    /// recommended profile for the selected model" and is resolved to a
    /// concrete engine profile by the command builder before argv.
    static let persistedProfiles: Set<String> = engineProfiles.union(["auto"])
    static let engineGenerationModes: Set<String> = ["mtp", "ar"]

    /// The normalized value when it is engine-launchable, else nil.
    /// "auto", typos, and every other string return nil: the caller
    /// omits `--profile` entirely and the engine — the single owner of
    /// default-profile resolution — picks the per-artifact profile
    /// (turbo for the flagships) and reports it on /health. The old
    /// coercion to "sustained" here was the app-side half of the
    /// historic resolve-to-sustained bug class: it turned "no explicit
    /// choice" into an explicit sustained pick the engine had to obey.
    public static func launchableProfile(_ raw: String) -> String? {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return engineProfiles.contains(value) ? value : nil
    }

    public static func launchableGenerationMode(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return engineGenerationModes.contains(value) ? value : "mtp"
    }

    /// Environment additions for Hugging Face downloads when the user
    /// configured a mirror. huggingface_hub sends the stored token to
    /// whatever HF_ENDPOINT points at, so both token variables are
    /// overridden to empty alongside any non-official endpoint. Returns
    /// nil when no valid mirror is configured (including the official
    /// host, where nothing should change).
    /// Largest memory cap the picker accepts, in GB. Well past the 512 GB
    /// M3 Ultra so no real Mac is clamped, small enough that a typo cannot
    /// overflow the byte conversion.
    public static let maximumMemoryLimitGB = 2048

    /// Clamp a user-entered memory cap. Zero, negatives, and nonsense fall
    /// back to `nil` (engine default) rather than launching a daemon with a
    /// cap it cannot honor. 1 GB is the floor: below that no model loads.
    public static func normalizedMemoryLimitGB(_ raw: Int?) -> Int? {
        guard let raw, raw > 0 else { return nil }
        return min(max(raw, 1), maximumMemoryLimitGB)
    }

    /// Daemon environment for the Settings memory card (issues #431, #427).
    ///
    /// The engine reads both of these from its own process environment
    /// (`MTPLX_MEMORY_LIMIT_BYTES` sizes the Metal allocator caps, and
    /// `MTPLX_ALLOW_SWAP` drops the machine-fit context clamp), which is the
    /// only override channel a GUI launcher has: `mtplx serve` exposes no
    /// flag for either. An unset limit and a swap toggle left off contribute
    /// nothing, so an untouched card launches a byte-identical daemon.
    public static func memoryOverrideEnvironment(
        memoryLimitGB: Int?,
        allowSwap: Bool
    ) -> [String: String] {
        var environment: [String: String] = [:]
        if let gigabytes = normalizedMemoryLimitGB(memoryLimitGB) {
            environment["MTPLX_MEMORY_LIMIT_BYTES"] = String(Int64(gigabytes) * 1024 * 1024 * 1024)
        }
        if allowSwap {
            environment["MTPLX_ALLOW_SWAP"] = "1"
        }
        return environment
    }

    /// Picker-level normalization of the Performance mode selection. The
    /// Settings picker, the persisted `scheduling_preset`, and the launch
    /// resolver all read the choice through this one function so a tag can
    /// never drift away from what gets saved (2026-09-03, issue #398).
    public static func schedulingPresetSelection(_ raw: String) -> String {
        switch raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        {
        case "latency", "serial-latency":
            return "latency"
        case "throughput", "ar-batch-throughput":
            return "throughput"
        case "agent", "ar-batch-agent":
            return "agent"
        default:
            return "target-default"
        }
    }

    public static func hfMirrorEnvironment(_ rawEndpoint: String?) -> [String: String]? {
        guard let rawEndpoint else { return nil }
        let trimmed = rawEndpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let url = URL(string: trimmed),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty
        else {
            return nil
        }
        if host.lowercased() == "huggingface.co" {
            return nil
        }
        return [
            "HF_ENDPOINT": trimmed,
            "HF_TOKEN": "",
            "HUGGING_FACE_HUB_TOKEN": "",
        ]
    }

    /// Early V1 builds wrote picker values the engine never accepted
    /// ("auto", "sustained-max"). "sustained-max" meant sustained plus
    /// pinned fans, so the fan intent survives the profile rewrite.
    public mutating func sanitizeLaunchCriticalFields() {
        let profileValue = profile.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if profileValue == "sustained-max" || profileValue == "sustained_max" {
            fanMode = MTPLXFanMode.max.rawValue
            pinFansAtMaxOnStart = true
        }
        // "auto" persists as-is (the launch omits --profile so the engine
        // resolves per model); unknown strings normalize to "auto" too —
        // never to a concrete profile the user did not pick.
        profile = Self.persistedProfiles.contains(profileValue)
            ? profileValue
            : "auto"
        generationMode = Self.launchableGenerationMode(generationMode)
        // One-shot migration (2026-07-03, turbo release): a persisted
        // "sustained" predating the Auto option was never a choice — it
        // was the field's default. Migrate it to "auto" once so the
        // recommended per-model profile (turbo for the 27Bs) applies;
        // any profile picked after this keeps winning forever.
        if !profileLegacyDefaultMigrated {
            if profile == "sustained" {
                profile = "auto"
            }
            profileLegacyDefaultMigrated = true
        }
        // One-shot migration (2026-07-03): 250 ms stream snapshots alias
        // against 8-bit verify steps (200-300 ms late in generation) —
        // the founder's freeze-then-vomit stutter. 100 ms is the new
        // default; a cadence chosen after this migration sticks.
        if !streamCadenceMigrated {
            if streamSnapshotIntervalMs == 250 {
                streamSnapshotIntervalMs = 100
            }
            streamCadenceMigrated = true
        }
        // Legacy sampler migration (2026-07-02): before the 27B launch
        // family existed, its sampler fell through to the CLI default
        // (1.0/0.95/20) and the app persisted that effective triple back
        // to settings, which then overrode any later per-model preset.
        // The exact untouched triple is a fingerprint of "never chosen",
        // so it yields back to preset authority (Qwen3.6 thinking spec
        // is 0.6/0.95/20 — founder-confirmed). Any other combination is
        // treated as a deliberate user choice and preserved.
        if !samplerLegacyTripleMigrated {
            if temperature == 1.0, topP == 0.95, topK == 20 {
                temperature = nil
                topP = nil
                topK = nil
            }
            samplerLegacyTripleMigrated = true
        }
    }

    public mutating func applySchedulingPreset(_ raw: String) {
        let normalized = Self.normalizedSchedulingPreset(
            raw,
            schedulerMode: schedulerMode,
            batchingPreset: batchingPreset,
            inferLegacyModePair: false
        )
        schedulingPreset = normalized
        switch normalized {
        case "latency":
            schedulerMode = "serial"
            batchingPreset = "latency"
        case "throughput":
            schedulerMode = "ar_batch"
            batchingPreset = "throughput"
        case "agent":
            schedulerMode = "ar_batch"
            batchingPreset = "agent"
        default:
            schedulerMode = "serial"
            batchingPreset = "latency"
        }
        maxActiveRequests = nil
        decodeBatchMax = nil
        batchWaitMs = nil
    }

    public func compatibleContextWindowOverride() -> Int? {
        guard let raw = contextWindow, raw > 0 else { return nil }
        let family = MTPLXModelOption.modelFamily(for: model)
        if let storedFamily = contextWindowModelFamily {
            guard MTPLXModelOption.settingsFamiliesCompatible(
                stored: storedFamily,
                current: family
            ) else { return nil }
        } else if !MTPLXModelOption.supportsTune(family: family) {
            return nil
        }
        let maximum = MTPLXModelOption.maxContextWindow(forFamily: family)
        let snapped = Int((Double(raw) / 1024.0).rounded()) * 1024
        return max(4_096, min(maximum, snapped))
    }

    public func compatibleTunedDepth() -> Int? {
        guard MTPLXModelOption.supportsTune(family: MTPLXModelOption.modelFamily(for: model)) else {
            return nil
        }
        return compatibleTunedControlValue(controlField: "depth")
    }

    public func compatibleTunedControlValue(controlField: String) -> Int? {
        let requestedField = controlField.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requestedField.isEmpty else { return nil }
        let family = MTPLXModelOption.modelFamily(for: model)
        if let record = tunedControlRecordForCurrentModel(),
           record.schemaVersion == 1,
           record.controlField == requestedField,
           record.modelFamily == family,
           (scopedTuneRecordMatchesCurrentModel(record) || MTPLXModelOption.modelsMatch(record.modelID, model)),
           Self.tunedControlValueIsValid(record.controlValue, family: family, controlField: requestedField)
        {
            return record.controlValue
        }
        // A tuned record that belongs to a DIFFERENT model means the
        // legacy lastTunedDepth is that model's residue too — a 35B
        // onboarding tune must not leak into a 27B launch and pull the
        // release model off its contract depth (QA-107).
        if let record = tunedControlRecordForCurrentModel(),
           !scopedTuneRecordMatchesCurrentModel(record),
           !MTPLXModelOption.modelsMatch(record.modelID, model)
        {
            return nil
        }
        guard requestedField == "depth",
              MTPLXModelOption.supportsTune(family: family),
              let raw = lastTunedDepth,
              (1...3).contains(raw)
        else {
            return nil
        }
        return raw
    }

    public mutating func saveTuneResult(
        modelPath: String,
        repoID: String?,
        family: String,
        result: TuneResult,
        tunedAt: Date = Date()
    ) {
        saveTunedControl(
            modelPath: modelPath,
            repoID: repoID,
            family: family,
            controlValue: Self.preferredMTPControlValue(family: family, result: result),
            tunedAt: tunedAt
        )
    }

    public mutating func saveSafeTunedDefault(
        modelPath: String,
        repoID: String?,
        family: String,
        tunedAt: Date = Date()
    ) {
        saveTunedControl(
            modelPath: modelPath,
            repoID: repoID,
            family: family,
            controlValue: Self.safeMTPControlValue(for: family),
            tunedAt: tunedAt
        )
    }

    private mutating func saveTunedControl(
        modelPath: String,
        repoID: String?,
        family: String,
        controlValue: Int,
        tunedAt: Date
    ) {
        let controlField = Self.forgeTuneControlField(for: family)
        guard Self.tunedControlValueIsValid(controlValue, family: family, controlField: controlField) else {
            return
        }
        let candidates = TuneCandidate.candidates(forFamily: family).map(\.displayLabel)
        let record = TunedControlRecord(
            modelID: modelPath,
            modelFamily: family,
            backendID: Self.forgeBackendID(for: family),
            controlField: controlField,
            controlValue: controlValue,
            candidates: candidates.isEmpty ? Self.forgeTuneCandidates(for: family) : candidates,
            tunedAt: tunedAt
        )
        generationMode = "mtp"
        loadMTP = true
        liveSettingsModelFamily = family
        if controlField == "depth" {
            lastTunedDepth = controlValue
        } else {
            lastTunedDepth = nil
        }
        lastTunedAt = tunedAt
        tunedControlRecord = record
        for key in Self.tuneRecordKeys(modelPath: modelPath, repoID: repoID) {
            tunedControlRecordsByModel[key] = record
        }
    }

    private func tunedControlRecordForCurrentModel() -> TunedControlRecord? {
        for key in Self.tuneRecordKeys(modelPath: model, repoID: nil) {
            if let record = tunedControlRecordsByModel[key],
               MTPLXModelOption.modelsMatch(record.modelID, model) || key == Self.tuneRecordKey(model)
            {
                return record
            }
        }
        if let scoped = tunedControlRecordsByModel.values.first(where: {
            MTPLXModelOption.modelsMatch($0.modelID, model)
        }) {
            return scoped
        }
        return tunedControlRecord
    }

    private func scopedTuneRecordMatchesCurrentModel(_ record: TunedControlRecord) -> Bool {
        for key in Self.tuneRecordKeys(modelPath: model, repoID: nil) {
            if tunedControlRecordsByModel[key] == record {
                return true
            }
        }
        return false
    }

    private static func preferredMTPControlValue(family: String, result: TuneResult) -> Int {
        let controlField = forgeTuneControlField(for: family)
        if result.bestDepth > 0,
           tunedControlValueIsValid(result.bestDepth, family: family, controlField: controlField)
        {
            return result.bestDepth
        }
        if let bestMTP = result.allCandidates
            .filter({ $0.candidate != .ar })
            .filter({ tunedControlValueIsValid($0.candidate.controlValue, family: family, controlField: controlField) })
            .max(by: { $0.tokS < $1.tokS })
        {
            return bestMTP.candidate.controlValue
        }
        return safeMTPControlValue(for: family)
    }

    private static func safeMTPControlValue(for family: String) -> Int {
        family == "gemma4" ? 6 : 2
    }

    private static func tuneRecordKeys(modelPath: String, repoID: String?) -> [String] {
        var keys: [String] = []
        func append(_ raw: String?) {
            guard let raw else { return }
            let key = tuneRecordKey(raw)
            guard !key.isEmpty, !keys.contains(key) else { return }
            keys.append(key)
        }
        append(modelPath)
        append(NSString(string: modelPath).expandingTildeInPath)
        append(URL(fileURLWithPath: modelPath).lastPathComponent)
        append(repoID)
        if let repoID {
            append(repoID.replacingOccurrences(of: "/", with: "--"))
        }
        return keys
    }

    private static func tuneRecordKey(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
    }

    private static func tunedControlValueIsValid(
        _ value: Int,
        family: String,
        controlField: String
    ) -> Bool {
        switch (family, controlField) {
        case ("qwen3_5", "depth"), ("qwen3_6", "depth"):
            return (1...3).contains(value)
        case ("qwen3_8", "depth"):
            // The artifact carries deeper training depths, but live serving
            // is safety-capped at D3 until the D4 daemon death is root-caused.
            return (1...3).contains(value)
        case ("gemma4", "draft_block_size"):
            return (2...8).contains(value)
        default:
            return false
        }
    }

    public func effectiveContextWindow(default defaultValue: Int) -> Int {
        compatibleContextWindowOverride()
            ?? min(MTPLXModelOption.maxContextWindow(for: model), max(4_096, defaultValue))
    }

    private static func normalizedSchedulingPreset(
        _ raw: String,
        schedulerMode: String,
        batchingPreset: String,
        inferLegacyModePair: Bool = true
    ) -> String {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        switch normalized {
        case "target-default", "default", "auto", "":
            guard inferLegacyModePair else { return "target-default" }
            if schedulerMode == "ar_batch" && batchingPreset == "throughput" {
                return "throughput"
            }
            if schedulerMode == "ar_batch" && batchingPreset == "agent" {
                return "agent"
            }
            // Legacy settings files stored serial/latency even when the
            // user had never chosen a scheduling override. Treat that
            // pair as target-default unless the new explicit preset says
            // otherwise.
            return "target-default"
        default:
            // One table for the concrete tags, shared with the Settings
            // picker so a menu label can never drift from what is saved.
            return schedulingPresetSelection(normalized)
        }
    }
}

public struct DaemonCommand: Equatable, Sendable {
    public var executableURL: URL
    public var arguments: [String]
    public var environment: [String: String]

    public init(
        executableURL: URL,
        arguments: [String],
        environment: [String: String] = [:]
    ) {
        self.executableURL = executableURL
        self.arguments = arguments
        self.environment = environment
    }
}

// MARK: - Secret redaction for logged or exported argv

extension DaemonCommand {
    /// Stand-in for a secret value in any logged or exported command line.
    public static let redactedValue = "<redacted>"

    /// Flag-name suffixes whose following value is a secret. `--api-key`
    /// is the one the daemon takes today; the suffix rule covers every
    /// future `*-key`, `*-token`, `*-secret` and `*-password` flag without
    /// another edit here. `--api-key-file` ends in `-file` and is a path,
    /// so it is deliberately not matched.
    private static let secretFlagSuffixes = ["-key", "-token", "-secret", "-password"]

    /// True when `argument` is a flag whose value must never be logged.
    /// Accepts both `--flag` and `--flag=value` spellings.
    public static func isSecretFlag(_ argument: String) -> Bool {
        guard argument.hasPrefix("-") else { return false }
        var name = argument.drop { $0 == "-" }.lowercased()
        if let equals = name.firstIndex(of: "=") {
            name = String(name[..<equals])
        }
        guard !name.isEmpty else { return false }
        return secretFlagSuffixes.contains { name.hasSuffix($0) }
    }

    /// `arguments` with every secret flag value replaced by
    /// `redactedValue`. Every other argument is returned unchanged, in
    /// order, so the result still reads as the command that ran.
    public static func redactingSecrets(in arguments: [String]) -> [String] {
        var redacted: [String] = []
        redacted.reserveCapacity(arguments.count)
        var maskNext = false
        for argument in arguments {
            if maskNext {
                redacted.append(redactedValue)
                maskNext = false
                continue
            }
            guard isSecretFlag(argument) else {
                redacted.append(argument)
                continue
            }
            if let equals = argument.firstIndex(of: "=") {
                redacted.append(String(argument[...equals]) + redactedValue)
            } else {
                redacted.append(argument)
                maskNext = true
            }
        }
        return redacted
    }

    /// The arguments as they may appear in logs, diagnostics, or bug
    /// reports.
    public var redactedArguments: [String] {
        Self.redactingSecrets(in: arguments)
    }

    /// The full command line with secrets masked. This is the only form
    /// of the command that may be written to a log store or exported.
    public var redactedCommandLine: String {
        ([executableURL.path] + redactedArguments).joined(separator: " ")
    }
}
