import Foundation

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
        hermesAutoApprove: Bool = true,
        fanMode: String? = nil,
        pinFansAtMaxOnStart: Bool = false,
        lastLaunchTarget: String = LaunchTarget.chat.rawValue,
        hermesWorkspacePath: String = MTPLXAppConfiguration.defaultHermesWorkspacePath(),
        lastHermesProfile: String? = nil,
        lastHermesSessionID: String? = nil,
        lastHermesSessionTitle: String? = nil,
        onboardingCompletedAt: Date? = nil,
        lastTunedDepth: Int? = nil,
        lastTunedAt: Date? = nil,
        tunedControlRecord: TunedControlRecord? = nil,
        tunedControlRecordsByModel: [String: TunedControlRecord] = [:],
        customModels: [MTPLXModelOption] = [],
        huggingFaceHandle: String? = nil,
        hfEndpoint: String? = nil
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
        self.lastTunedDepth = lastTunedDepth
        self.lastTunedAt = lastTunedAt
        self.tunedControlRecord = tunedControlRecord
        self.tunedControlRecordsByModel = tunedControlRecordsByModel
        self.customModels = customModels
        self.huggingFaceHandle = huggingFaceHandle
        self.hfEndpoint = hfEndpoint
    }

    /// Fresh installs must be portable. Installed local copies are discovered
    /// by the model catalog; the default configuration should never point at
    /// a developer machine path.
    public static func defaultLocalModelPath() -> String {
        return "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
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
        case hermesAutoApprove = "hermes_auto_approve"
        case fanMode = "fan_mode"
        case pinFansAtMaxOnStart = "pin_fans_at_max_on_start"
        case lastLaunchTarget = "last_launch_target"
        case hermesWorkspacePath = "hermes_workspace_path"
        case lastHermesProfile = "last_hermes_profile"
        case lastHermesSessionID = "last_hermes_session_id"
        case lastHermesSessionTitle = "last_hermes_session_title"
        case onboardingCompletedAt = "onboarding_completed_at"
        case lastTunedDepth = "last_tuned_depth"
        case lastTunedAt = "last_tuned_at"
        case tunedControlRecord = "tuned_control_record"
        case tunedControlRecordsByModel = "tuned_control_records_by_model"
        case customModels = "custom_models"
        case huggingFaceHandle = "hugging_face_handle"
        case hfEndpoint = "hf_endpoint"
    }

    public init(from decoder: Decoder) throws {
        let defaults = MTPLXAppConfiguration()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        executablePath = try container.decodeIfPresent(String.self, forKey: .executablePath)
        model = try container.decodeIfPresent(String.self, forKey: .model) ?? defaults.model
        profile = try container.decodeIfPresent(String.self, forKey: .profile) ?? defaults.profile
        host = try container.decodeIfPresent(String.self, forKey: .host) ?? defaults.host
        port = try container.decodeIfPresent(Int.self, forKey: .port) ?? defaults.port
        generationMode = try container.decodeIfPresent(String.self, forKey: .generationMode) ?? defaults.generationMode
        loadMTP = try container.decodeIfPresent(Bool.self, forKey: .loadMTP) ?? defaults.loadMTP
        schedulerMode = try container.decodeIfPresent(String.self, forKey: .schedulerMode) ?? defaults.schedulerMode
        batchingPreset = try container.decodeIfPresent(String.self, forKey: .batchingPreset) ?? defaults.batchingPreset
        let decodedSchedulingPreset = try container.decodeIfPresent(String.self, forKey: .schedulingPreset)
        schedulingPreset = Self.normalizedSchedulingPreset(
            decodedSchedulingPreset ?? defaults.schedulingPreset,
            schedulerMode: schedulerMode,
            batchingPreset: batchingPreset,
            inferLegacyModePair: false
        )
        maxActiveRequests = try container.decodeIfPresent(Int.self, forKey: .maxActiveRequests)
        decodeBatchMax = try container.decodeIfPresent(Int.self, forKey: .decodeBatchMax)
        batchWaitMs = try container.decodeIfPresent(Double.self, forKey: .batchWaitMs)
        prefillChunkTokens = try container.decodeIfPresent(Int.self, forKey: .prefillChunkTokens)
        experimentalMTPCohorts = try container.decodeIfPresent(Bool.self, forKey: .experimentalMTPCohorts) ?? defaults.experimentalMTPCohorts
        embeddingModels = try container.decodeIfPresent([String].self, forKey: .embeddingModels) ?? defaults.embeddingModels
        rerankerModels = try container.decodeIfPresent([String].self, forKey: .rerankerModels) ?? defaults.rerankerModels
        retrievalMaxResident = try container.decodeIfPresent(Int.self, forKey: .retrievalMaxResident) ?? defaults.retrievalMaxResident
        ramSessionCachePolicy = try container.decodeIfPresent(String.self, forKey: .ramSessionCachePolicy) ?? defaults.ramSessionCachePolicy
        ramSessionBlockPrefixRestore = try container.decodeIfPresent(Bool.self, forKey: .ramSessionBlockPrefixRestore) ?? defaults.ramSessionBlockPrefixRestore
        ramSessionCacheMaxEntries = try container.decodeIfPresent(Int.self, forKey: .ramSessionCacheMaxEntries) ?? defaults.ramSessionCacheMaxEntries
        ramSessionCacheMaxSize = try container.decodeIfPresent(String.self, forKey: .ramSessionCacheMaxSize) ?? defaults.ramSessionCacheMaxSize
        ramSessionCachePerSessionMaxSize = try container.decodeIfPresent(String.self, forKey: .ramSessionCachePerSessionMaxSize) ?? defaults.ramSessionCachePerSessionMaxSize
        pagedKVQuantization = try container.decodeIfPresent(String.self, forKey: .pagedKVQuantization) ?? defaults.pagedKVQuantization
        ssdSessionCache = try container.decodeIfPresent(String.self, forKey: .ssdSessionCache) ?? defaults.ssdSessionCache
        ssdSessionCacheDir = try container.decodeIfPresent(String.self, forKey: .ssdSessionCacheDir)
        ssdSessionCacheMaxSize = try container.decodeIfPresent(String.self, forKey: .ssdSessionCacheMaxSize) ?? defaults.ssdSessionCacheMaxSize
        ssdSessionCacheMinPrefixTokens = try container.decodeIfPresent(Int.self, forKey: .ssdSessionCacheMinPrefixTokens) ?? defaults.ssdSessionCacheMinPrefixTokens
        contextWindow = try container.decodeIfPresent(Int.self, forKey: .contextWindow)
        contextWindowModelFamily = try container.decodeIfPresent(String.self, forKey: .contextWindowModelFamily)
        temperature = try container.decodeIfPresent(Double.self, forKey: .temperature)
        topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        samplerLegacyTripleMigrated = try container.decodeIfPresent(
            Bool.self, forKey: .samplerLegacyTripleMigrated
        ) ?? false
        profileLegacyDefaultMigrated = try container.decodeIfPresent(
            Bool.self, forKey: .profileLegacyDefaultMigrated
        ) ?? false
        streamCadenceMigrated = try container.decodeIfPresent(
            Bool.self, forKey: .streamCadenceMigrated
        ) ?? false
        presencePenalty = try container.decodeIfPresent(Double.self, forKey: .presencePenalty)
        reasoning = try container.decodeIfPresent(String.self, forKey: .reasoning)
        reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        liveSettingsModelFamily = try container.decodeIfPresent(String.self, forKey: .liveSettingsModelFamily)
        apiKey = try container.decodeIfPresent(String.self, forKey: .apiKey)
        enableThermalPolling = try container.decodeIfPresent(Bool.self, forKey: .enableThermalPolling) ?? defaults.enableThermalPolling
        streamSnapshotIntervalMs = try container.decodeIfPresent(Int.self, forKey: .streamSnapshotIntervalMs) ?? defaults.streamSnapshotIntervalMs
        performanceLock = try container.decodeIfPresent(Bool.self, forKey: .performanceLock) ?? defaults.performanceLock
        launchDaemonOnOpen = try container.decodeIfPresent(Bool.self, forKey: .launchDaemonOnOpen) ?? defaults.launchDaemonOnOpen
        hermesAutoApprove = try container.decodeIfPresent(Bool.self, forKey: .hermesAutoApprove) ?? defaults.hermesAutoApprove
        let decodedFanMode = try container.decodeIfPresent(String.self, forKey: .fanMode)
        let legacyPin = try container.decodeIfPresent(Bool.self, forKey: .pinFansAtMaxOnStart)
        let legacyFallback: String
        if let legacyPin {
            legacyFallback = legacyPin ? MTPLXFanMode.max.rawValue : MTPLXFanMode.default.rawValue
        } else {
            legacyFallback = defaults.fanMode
        }
        let resolvedFanMode = MTPLXFanMode.normalized(decodedFanMode ?? legacyFallback)
        fanMode = resolvedFanMode.rawValue
        pinFansAtMaxOnStart = resolvedFanMode == .max
        lastLaunchTarget = try container.decodeIfPresent(String.self, forKey: .lastLaunchTarget) ?? defaults.lastLaunchTarget
        hermesWorkspacePath = Self.normalizedHermesWorkspacePath(
            try container.decodeIfPresent(String.self, forKey: .hermesWorkspacePath)
                ?? defaults.hermesWorkspacePath
        )
        lastHermesProfile = try container.decodeIfPresent(String.self, forKey: .lastHermesProfile)
        lastHermesSessionID = try container.decodeIfPresent(String.self, forKey: .lastHermesSessionID)
        lastHermesSessionTitle = try container.decodeIfPresent(String.self, forKey: .lastHermesSessionTitle)
        onboardingCompletedAt = try container.decodeIfPresent(Date.self, forKey: .onboardingCompletedAt)
        lastTunedDepth = try container.decodeIfPresent(Int.self, forKey: .lastTunedDepth)
        lastTunedAt = try container.decodeIfPresent(Date.self, forKey: .lastTunedAt)
        tunedControlRecord = try container.decodeIfPresent(TunedControlRecord.self, forKey: .tunedControlRecord)
        tunedControlRecordsByModel = try container.decodeIfPresent(
            [String: TunedControlRecord].self,
            forKey: .tunedControlRecordsByModel
        ) ?? defaults.tunedControlRecordsByModel
        customModels = try container.decodeIfPresent([MTPLXModelOption].self, forKey: .customModels) ?? defaults.customModels
        huggingFaceHandle = try container.decodeIfPresent(String.self, forKey: .huggingFaceHandle)
        hfEndpoint = try container.decodeIfPresent(String.self, forKey: .hfEndpoint)
        sanitizeLaunchCriticalFields()
    }

    /// Engine-launchable allowlists. SYNC PAIR: mtplx/profiles.py
    /// PROFILE_CHOICES and mtplx GENERATION_MODES. A value outside these
    /// kills `mtplx serve` at argument parsing, which the user experiences
    /// as a daemon that is degraded on every start, so any persisted
    /// config must decode back to something launchable.
    static let engineProfiles: Set<String> = [
        "stable", "performance-cold", "sustained", "turbo", "exact",
        "max-diagnostic",
    ]

    /// "auto" is persistable but never launchable: it means "use the
    /// recommended profile for the selected model" and is resolved to a
    /// concrete engine profile by the command builder before argv.
    static let persistedProfiles: Set<String> = engineProfiles.union(["auto"])
    static let engineGenerationModes: Set<String> = ["mtp", "ar"]

    public static func launchableProfile(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return engineProfiles.contains(value) ? value : "sustained"
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
        // "auto" persists as-is (per-model resolution happens at launch);
        // everything else must be engine-launchable.
        profile = Self.persistedProfiles.contains(profileValue)
            ? profileValue
            : Self.launchableProfile(profile)
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
        case "serial-latency", "latency":
            return "latency"
        case "ar-batch-throughput", "throughput":
            return "throughput"
        case "ar-batch-agent", "agent":
            return "agent"
        default:
            return "target-default"
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
