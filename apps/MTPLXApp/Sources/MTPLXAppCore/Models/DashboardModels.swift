import Foundation

public struct AppCapabilities: Codable, Equatable, Sendable {
    public struct SnapshotInterval: Codable, Equatable, Sendable {
        public var defaultMs: Int
        public var minMs: Int
        public var maxMs: Int
        public var nativeDefaultMs: Int
        public var performanceLockMs: Int

        enum CodingKeys: String, CodingKey {
            case defaultMs = "default_ms"
            case minMs = "min_ms"
            case maxMs = "max_ms"
            case nativeDefaultMs = "native_default_ms"
            case performanceLockMs = "performance_lock_ms"
        }
    }

    public var ok: Bool
    public var name: String
    public var apiVersion: Int
    public var endpoints: [String: String]
    public var mutableSettings: [String]
    public var restartRequiredSettings: [String]
    public var snapshotInterval: SnapshotInterval
    public var features: [String: Bool]
    public var scheduler: DynamicObject?

    enum CodingKeys: String, CodingKey {
        case ok
        case name
        case apiVersion = "api_version"
        case endpoints
        case mutableSettings = "mutable_settings"
        case restartRequiredSettings = "restart_required_settings"
        case snapshotInterval = "snapshot_interval"
        case features
        case scheduler
    }
}

/// Describes how the loaded backend exposes its speculative-decode depth
/// control, so the UI can render the right range/label/unit per model
/// (Qwen = "Draft depth" 1-3, Gemma = "Draft block" 2-8) instead of a
/// hardcoded D1-D3. Mirrors `DraftSemantics.to_dict()` on the server.
public struct DraftControl: Codable, Equatable, Sendable {
    public var supported: Bool?
    public var requestField: String?
    public var displayLabel: String?
    public var defaultValue: Int?
    public var minimum: Int?
    public var maximum: Int?
    public var unit: String?
    public var valueLabels: [String]?

    public init(
        supported: Bool? = nil,
        requestField: String? = nil,
        displayLabel: String? = nil,
        defaultValue: Int? = nil,
        minimum: Int? = nil,
        maximum: Int? = nil,
        unit: String? = nil,
        valueLabels: [String]? = nil
    ) {
        self.supported = supported
        self.requestField = requestField
        self.displayLabel = displayLabel
        self.defaultValue = defaultValue
        self.minimum = minimum
        self.maximum = maximum
        self.unit = unit
        self.valueLabels = valueLabels
    }

    enum CodingKeys: String, CodingKey {
        case supported
        case requestField = "request_field"
        case displayLabel = "display_label"
        case defaultValue = "default"
        case minimum
        case maximum
        case unit
        case valueLabels = "value_labels"
    }
}

public struct SamplingDefaults: Codable, Equatable, Sendable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var familyDefaultReason: String?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        familyDefaultReason: String? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.familyDefaultReason = familyDefaultReason
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case familyDefaultReason = "family_default_reason"
    }
}

public struct ReasoningPolicy: Codable, Equatable, Sendable {
    public var supported: Bool
    public var parser: String?
    public var displayName: String?
    public var modes: [String]
    public var defaultMode: String?
    public var historyPolicy: String?
    public var effortLevels: [String]?
    public var defaultEffort: String?

    public init(
        supported: Bool,
        parser: String? = nil,
        displayName: String? = nil,
        modes: [String] = ["auto", "on", "off"],
        defaultMode: String? = nil,
        historyPolicy: String? = nil,
        effortLevels: [String] = [],
        defaultEffort: String? = nil
    ) {
        self.supported = supported
        self.parser = parser
        self.displayName = displayName
        self.modes = modes
        self.defaultMode = defaultMode
        self.historyPolicy = historyPolicy
        self.effortLevels = effortLevels
        self.defaultEffort = defaultEffort
    }

    enum CodingKeys: String, CodingKey {
        case supported
        case parser
        case displayName = "display_name"
        case modes
        case defaultMode = "default"
        case historyPolicy = "history_policy"
        case effortLevels = "effort_levels"
        case defaultEffort = "default_effort"
    }
}

public struct TunePolicy: Codable, Equatable, Sendable {
    public var supported: Bool
    public var supportedFamilies: [String]
    public var controlField: String?
    public var candidates: [String]
    public var unsupportedReason: String?

    enum CodingKeys: String, CodingKey {
        case supported
        case supportedFamilies = "supported_families"
        case controlField = "control_field"
        case candidates
        case unsupportedReason = "unsupported_reason"
    }
}

public struct KVQuantPolicy: Codable, Equatable, Sendable {
    public var supported: Bool
    public var modes: [String]
    public var restartRequired: Bool?
    public var proofLevel: String?
    public var disabledReason: String?

    public init(
        supported: Bool,
        modes: [String],
        restartRequired: Bool? = nil,
        proofLevel: String? = nil,
        disabledReason: String? = nil
    ) {
        self.supported = supported
        self.modes = modes
        self.restartRequired = restartRequired
        self.proofLevel = proofLevel
        self.disabledReason = disabledReason
    }

    enum CodingKeys: String, CodingKey {
        case supported
        case modes
        case restartRequired = "restart_required"
        case proofLevel = "proof_level"
        case disabledReason = "disabled_reason"
    }
}

public struct ContextWindowPolicy: Codable, Equatable, Sendable {
    public var supported: Bool
    public var minimum: Int?
    public var maximum: Int?
    public var defaultValue: Int?
    public var step: Int?
    public var source: String?
    public var unit: String?

    public init(
        supported: Bool = true,
        minimum: Int? = nil,
        maximum: Int? = nil,
        defaultValue: Int? = nil,
        step: Int? = nil,
        source: String? = nil,
        unit: String? = nil
    ) {
        self.supported = supported
        self.minimum = minimum
        self.maximum = maximum
        self.defaultValue = defaultValue
        self.step = step
        self.source = source
        self.unit = unit
    }

    enum CodingKeys: String, CodingKey {
        case supported
        case minimum
        case maximum
        case defaultValue = "default"
        case step
        case source
        case unit
    }
}

public struct ModelControls: Codable, Equatable, Sendable {
    public var schemaVersion: Int?
    public var modelRef: String?
    public var modelFamily: String?
    public var backendID: String?
    public var architectureID: String?
    public var supportLevel: String?
    public var displayName: String?
    public var draftControl: DraftControl?
    public var sampling: SamplingDefaults?
    public var reasoning: ReasoningPolicy?
    public var tune: TunePolicy?
    public var kvQuant: KVQuantPolicy?
    public var contextWindow: ContextWindowPolicy?

    public init(
        schemaVersion: Int? = nil,
        modelRef: String? = nil,
        modelFamily: String? = nil,
        backendID: String? = nil,
        architectureID: String? = nil,
        supportLevel: String? = nil,
        displayName: String? = nil,
        draftControl: DraftControl? = nil,
        sampling: SamplingDefaults? = nil,
        reasoning: ReasoningPolicy? = nil,
        tune: TunePolicy? = nil,
        kvQuant: KVQuantPolicy? = nil,
        contextWindow: ContextWindowPolicy? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.modelRef = modelRef
        self.modelFamily = modelFamily
        self.backendID = backendID
        self.architectureID = architectureID
        self.supportLevel = supportLevel
        self.displayName = displayName
        self.draftControl = draftControl
        self.sampling = sampling
        self.reasoning = reasoning
        self.tune = tune
        self.kvQuant = kvQuant
        self.contextWindow = contextWindow
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelRef = "model_ref"
        case modelFamily = "model_family"
        case backendID = "backend_id"
        case architectureID = "architecture_id"
        case supportLevel = "support_level"
        case displayName = "display_name"
        case draftControl = "draft_control"
        case sampling
        case reasoning
        case tune
        case kvQuant = "kv_quant"
        case contextWindow = "context_window"
    }
}

public struct MutableSettings: Codable, Equatable, Sendable {
    public var generationMode: String?
    public var depth: Int?
    /// Maximum speculative depth/block the loaded backend supports
    /// (server `depth_max`). Drives the depth slider's upper bound so it
    /// adapts to the model instead of a hardcoded 3.
    public var depthMax: Int?
    /// Full per-model draft-control descriptor (label/unit/min/max).
    public var draftControl: DraftControl?
    public var modelControls: ModelControls?
    public var modelFamily: String?
    public var architectureID: String?
    public var supportLevel: String?
    public var reasoningPolicy: ReasoningPolicy?
    public var kvQuantPolicy: KVQuantPolicy?
    public var tunePolicy: TunePolicy?
    public var contextWindowPolicy: ContextWindowPolicy?
    public var samplingDefaults: SamplingDefaults?
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    /// OpenAI-style presence penalty (0 = exact no-op). Live-mutable via
    /// `/v1/mtplx/settings`; the daemon maps it onto its
    /// `--default-presence-penalty` server default.
    public var presencePenalty: Double?
    public var maxResponseTokens: Int?
    public var streamInterval: Int?
    public var enableThinking: Bool?
    public var reasoningParser: String?
    public var reasoning: String?
    public var reasoningEffort: String?
    public var prefillChunkTokens: Int?

    public init(
        generationMode: String? = nil,
        depth: Int? = nil,
        depthMax: Int? = nil,
        draftControl: DraftControl? = nil,
        modelControls: ModelControls? = nil,
        modelFamily: String? = nil,
        architectureID: String? = nil,
        supportLevel: String? = nil,
        reasoningPolicy: ReasoningPolicy? = nil,
        kvQuantPolicy: KVQuantPolicy? = nil,
        tunePolicy: TunePolicy? = nil,
        contextWindowPolicy: ContextWindowPolicy? = nil,
        samplingDefaults: SamplingDefaults? = nil,
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        presencePenalty: Double? = nil,
        maxResponseTokens: Int? = nil,
        streamInterval: Int? = nil,
        enableThinking: Bool? = nil,
        reasoningParser: String? = nil,
        reasoning: String? = nil,
        reasoningEffort: String? = nil,
        prefillChunkTokens: Int? = nil
    ) {
        self.generationMode = generationMode
        self.depth = depth
        self.depthMax = depthMax
        self.draftControl = draftControl
        self.modelControls = modelControls
        self.modelFamily = modelFamily
        self.architectureID = architectureID
        self.supportLevel = supportLevel
        self.reasoningPolicy = reasoningPolicy
        self.kvQuantPolicy = kvQuantPolicy
        self.tunePolicy = tunePolicy
        self.contextWindowPolicy = contextWindowPolicy
        self.samplingDefaults = samplingDefaults
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.presencePenalty = presencePenalty
        self.maxResponseTokens = maxResponseTokens
        self.streamInterval = streamInterval
        self.enableThinking = enableThinking
        self.reasoningParser = reasoningParser
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.prefillChunkTokens = prefillChunkTokens
    }

    enum CodingKeys: String, CodingKey {
        case generationMode = "generation_mode"
        case depth
        case depthMax = "depth_max"
        case draftControl = "draft_control"
        case modelControls = "model_controls"
        case modelFamily = "model_family"
        case architectureID = "architecture_id"
        case supportLevel = "support_level"
        case reasoningPolicy = "reasoning_policy"
        case kvQuantPolicy = "kv_quant_policy"
        case tunePolicy = "tune_policy"
        case contextWindowPolicy = "context_window_policy"
        case samplingDefaults = "sampling_defaults"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case presencePenalty = "presence_penalty"
        case maxResponseTokens = "max_response_tokens"
        case streamInterval = "stream_interval"
        case enableThinking = "enable_thinking"
        case reasoningParser = "reasoning_parser"
        case reasoning
        case reasoningEffort = "reasoning_effort"
        case prefillChunkTokens = "prefill_chunk_tokens"
    }
}

public struct PrefillState: Codable, Equatable, Sendable {
    public var phase: String
    public var tokensDone: Int?
    public var tokensTotal: Int
    public var cachedTokens: Int?
    public var newPrefillTokens: Int?
    public var elapsedS: Double?
    public var promptEvalTimeS: Double?
    public var prefillTokS: Double?
    public var prefillComputeTokS: Double?
    public var prefillWallTokS: Double?
    public var cumulativePrefillTokS: Double?
    public var livePrefillTokS: Double?
    public var chunkSize: Int?
    public var chunkElapsedS: Double?
    public var chunkPrefillTokS: Double?
    public var cacheHit: Bool?
    public var cacheSource: String?
    public var ssdCacheHit: Bool?
    public var ssdCachedTokens: Int?
    public var ssdRestoreS: Double?
    public var ssdSuffixTokens: Int?
    public var startedS: Double?

    public init(
        phase: String,
        tokensDone: Int? = nil,
        tokensTotal: Int = 0,
        cachedTokens: Int? = nil,
        newPrefillTokens: Int? = nil,
        elapsedS: Double? = nil,
        promptEvalTimeS: Double? = nil,
        prefillTokS: Double? = nil,
        prefillComputeTokS: Double? = nil,
        prefillWallTokS: Double? = nil,
        cumulativePrefillTokS: Double? = nil,
        livePrefillTokS: Double? = nil,
        chunkSize: Int? = nil,
        chunkElapsedS: Double? = nil,
        chunkPrefillTokS: Double? = nil,
        cacheHit: Bool? = nil,
        cacheSource: String? = nil,
        ssdCacheHit: Bool? = nil,
        ssdCachedTokens: Int? = nil,
        ssdRestoreS: Double? = nil,
        ssdSuffixTokens: Int? = nil,
        startedS: Double? = nil
    ) {
        self.phase = phase
        self.tokensDone = tokensDone
        self.tokensTotal = tokensTotal
        self.cachedTokens = cachedTokens
        self.newPrefillTokens = newPrefillTokens
        self.elapsedS = elapsedS
        self.promptEvalTimeS = promptEvalTimeS
        self.prefillTokS = prefillTokS
        self.prefillComputeTokS = prefillComputeTokS
        self.prefillWallTokS = prefillWallTokS
        self.cumulativePrefillTokS = cumulativePrefillTokS
        self.livePrefillTokS = livePrefillTokS
        self.chunkSize = chunkSize
        self.chunkElapsedS = chunkElapsedS
        self.chunkPrefillTokS = chunkPrefillTokS
        self.cacheHit = cacheHit
        self.cacheSource = cacheSource
        self.ssdCacheHit = ssdCacheHit
        self.ssdCachedTokens = ssdCachedTokens
        self.ssdRestoreS = ssdRestoreS
        self.ssdSuffixTokens = ssdSuffixTokens
        self.startedS = startedS
    }

    enum CodingKeys: String, CodingKey {
        case phase
        case tokensDone = "tokens_done"
        case tokensTotal = "tokens_total"
        case cachedTokens = "cached_tokens"
        case newPrefillTokens = "new_prefill_tokens"
        case elapsedS = "elapsed_s"
        case promptEvalTimeS = "prompt_eval_time_s"
        case prefillTokS = "prefill_tok_s"
        case prefillComputeTokS = "prefill_compute_tok_s"
        case prefillWallTokS = "prefill_wall_tok_s"
        case cumulativePrefillTokS = "cumulative_prefill_tok_s"
        case livePrefillTokS = "live_prefill_tok_s"
        case chunkSize = "chunk_size"
        case chunkElapsedS = "chunk_elapsed_s"
        case chunkPrefillTokS = "chunk_prefill_tok_s"
        case cacheHit = "cache_hit"
        case cacheSource = "cache_source"
        case ssdCacheHit = "ssd_cache_hit"
        case ssdCachedTokens = "ssd_cached_tokens"
        case ssdRestoreS = "ssd_restore_s"
        case ssdSuffixTokens = "ssd_suffix_tokens"
        case startedS = "started_s"
    }
}

public struct InFlightRequest: Codable, Equatable, Sendable, Identifiable {
    public var id: String { requestId }
    public var requestId: String
    public var startedS: Double
    public var ageS: Double
    public var sessionId: String?
    public var model: String?
    public var promptPreview: String
    public var promptTokens: Int?
    public var lastProgress: DynamicObject
    public var prefillState: PrefillState?
    public var cancelled: Bool

    enum CodingKeys: String, CodingKey {
        case requestId = "request_id"
        case startedS = "started_s"
        case ageS = "age_s"
        case sessionId = "session_id"
        case model
        case promptPreview = "prompt_preview"
        case promptTokens = "prompt_tokens"
        case lastProgress = "last_progress"
        case prefillState = "prefill_state"
        case cancelled
    }
}

/// Lifecycle-aware headline decode reading for the gauge. Models the
/// three states the user actually cares about, with explicit semantics
/// instead of relying on whether `latest.decode_tok_s` happens to be
/// non-nil at any given instant:
///
/// - `.absent` — no decode reading observed this daemon run (idle
///   between launch and the first request).
/// - `.live(value)` — there is an in-flight request and decode tokens
///   are actively streaming.
/// - `.held(value, completedAt:)` — the most recent request has
///   completed; the gauge holds at the request's final average until
///   another request starts or the daemon stops.
public enum HeadlineDecodeReading: Equatable, Sendable {
    case absent
    case live(Double)
    case held(value: Double, completedAt: Date)

    public var value: Double? {
        switch self {
        case .absent: return nil
        case .live(let v): return v
        case .held(let v, _): return v
        }
    }

    public var isLive: Bool {
        if case .live = self { return true }
        return false
    }

    public var isHeld: Bool {
        if case .held = self { return true }
        return false
    }
}

/// Spring-friendly mirrors of the noisiest live metrics. Populated by
/// `MTPLXBackendStore` via a short EMA over each progress frame so the
/// UI can read a stable value without flickering between two integers
/// per second.
public struct SmoothedMetrics: Equatable, Sendable {
    public var verifyCalls: Double?
    public var cachedTokens: Double?
    /// `accepted / drafted` per depth, smoothed to bar width.
    public var acceptanceRateByDepth: [Double]
    /// Server-reported mean acceptance probability per depth, smoothed.
    public var meanAcceptByDepth: [Double]

    public init(
        verifyCalls: Double? = nil,
        cachedTokens: Double? = nil,
        acceptanceRateByDepth: [Double] = [],
        meanAcceptByDepth: [Double] = []
    ) {
        self.verifyCalls = verifyCalls
        self.cachedTokens = cachedTokens
        self.acceptanceRateByDepth = acceptanceRateByDepth
        self.meanAcceptByDepth = meanAcceptByDepth
    }

    /// Average of `meanAcceptByDepth`, or `nil` when no data has been
    /// observed yet.
    public var meanAcceptance: Double? {
        guard !meanAcceptByDepth.isEmpty else {
            guard !acceptanceRateByDepth.isEmpty else { return nil }
            return acceptanceRateByDepth.reduce(0.0, +) / Double(acceptanceRateByDepth.count)
        }
        return meanAcceptByDepth.reduce(0.0, +) / Double(meanAcceptByDepth.count)
    }
}

public struct AcceptanceCounterRow: Equatable, Sendable {
    public var label: String
    public var accepted: Int
    public var drafted: Int

    public init(label: String, accepted: Int, drafted: Int) {
        self.label = label
        self.accepted = accepted
        self.drafted = drafted
    }

    public var rate: Double {
        drafted > 0 ? Double(accepted) / Double(drafted) : 0
    }
}

public enum RequestCacheVerdict: Equatable, Sendable {
    case hit
    case miss
    case unknown
}

public struct MetricsLatest: Codable, Equatable, Sendable {
    public var values: [String: JSONValue]

    public init(values: [String: JSONValue] = [:]) {
        self.values = values
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.values = try container.decode([String: JSONValue].self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    public var decodeTokS: Double? { values["decode_tok_s"]?.doubleValue }
    public var prefillTokS: Double? { values["prefill_tok_s"]?.doubleValue }
    public var ttftS: Double? { values["ttft_s"]?.doubleValue }
    public var sessionId: String? { values["session_id"]?.stringValue }
    public var cacheSource: String? { values["cache_source"]?.stringValue }
    public var ssdCacheHit: Bool? { values["ssd_cache_hit"]?.boolValue }
    public var ssdCachedTokens: Int? { values["ssd_cached_tokens"]?.intValue }
    public var ssdRestoreS: Double? { values["ssd_restore_s"]?.doubleValue }
    public var ssdSuffixTokens: Int? { values["ssd_suffix_tokens"]?.intValue }
    public var requestCacheVerdict: RequestCacheVerdict {
        switch values["session_cache_hit"]?.boolValue {
        case true: return .hit
        case false: return .miss
        case nil: return .unknown
        }
    }

    public func acceptanceCounterRows() -> [AcceptanceCounterRow] {
        let acceptedByDepth = intArrayValue(for: "accepted_by_depth")
        if !acceptedByDepth.isEmpty {
            let draftedByDepth = intArrayValue(for: "drafted_by_depth")
            let verifyCallsFallback = values["verify_calls"]?.intValue ?? 0
            return acceptedByDepth.enumerated().compactMap { idx, acceptedCount in
                let drafted = draftedByDepth.indices.contains(idx)
                    ? draftedByDepth[idx]
                    : verifyCallsFallback
                guard drafted > 0 else { return nil }
                return AcceptanceCounterRow(
                    label: "D\(idx + 1)",
                    accepted: acceptedCount,
                    drafted: drafted
                )
            }
        }

        guard let accepted = values["accepted_drafts"]?.intValue,
              let drafted = values["drafted_tokens"]?.intValue,
              drafted > 0
        else { return [] }

        return [
            AcceptanceCounterRow(
                label: "ALL",
                accepted: accepted,
                drafted: drafted
            ),
        ]
    }

    private func intArrayValue(for key: String) -> [Int] {
        guard case let .array(items)? = values[key] else { return [] }
        return items.compactMap(\.intValue)
    }
}

public struct RollingTPSPoint: Codable, Equatable, Sendable {
    public var t: Double
    public var tokS: Double
    public var sessionId: String?

    enum CodingKeys: String, CodingKey {
        case t
        case tokS = "tok_s"
        case sessionId = "session_id"
    }
}

public struct RollingMetrics: Codable, Equatable, Sendable {
    public var windowS: Double
    public var count: Int
    public var min: Double?
    public var max: Double?
    public var mean: Double?
    public var p50: Double?
    public var p95: Double?
    public var history: [RollingTPSPoint]
    public var liveHistory: [RollingTPSPoint]
    public var stickyAllTimeMax: Double

    enum CodingKeys: String, CodingKey {
        case windowS = "window_s"
        case count
        case min
        case max
        case mean
        case p50
        case p95
        case history
        case liveHistory = "live_history"
        case stickyAllTimeMax = "sticky_all_time_max"
    }
}

public struct LifetimeSnapshot: Codable, Equatable, Sendable {
    public var startedAtS: Double
    public var uptimeS: Double
    public var promptTokensTotal: Int
    public var completionTokensTotal: Int
    public var cachedTokensTotal: Int
    public var tokensTotal: Int
    public var requestsTotal: Int
    public var cancelledTotal: Int

    enum CodingKeys: String, CodingKey {
        case startedAtS = "started_at_s"
        case uptimeS = "uptime_s"
        case promptTokensTotal = "prompt_tokens_total"
        case completionTokensTotal = "completion_tokens_total"
        case cachedTokensTotal = "cached_tokens_total"
        case tokensTotal = "tokens_total"
        case requestsTotal = "requests_total"
        case cancelledTotal = "cancelled_total"
    }
}

public struct SessionBankPrefix: Codable, Equatable, Sendable {
    public var sessionId: String
    public var prefixLen: Int
    public var hits: Int
    public var nbytes: Int
    public var createdAtS: Double
    public var lastAccessS: Double
    public var policyFingerprint: String?
    public var hasLiveRef: Bool?

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case prefixLen = "prefix_len"
        case hits
        case nbytes
        case createdAtS = "created_at_s"
        case lastAccessS = "last_access_s"
        case policyFingerprint = "policy_fingerprint"
        case hasLiveRef = "has_live_ref"
    }
}

public struct SessionBankColdTier: Codable, Equatable, Sendable {
    public var enabled: Bool?
    public var restorable: Bool?
    public var mode: String?
    public var dir: String?
    public var manifestPath: String?
    public var formatVersion: Int?
    public var entries: Int?
    public var bytes: Int?
    public var logicalBytes: Int?
    public var physicalBytes: Int?
    public var livePhysicalBytes: Int?
    public var managedFileBytes: Int?
    public var managedDiskBytes: Int?
    public var databaseFileBytes: Int?
    public var databaseDiskBytes: Int?
    public var untrackedFileBytes: Int?
    public var untrackedDiskBytes: Int?
    public var managedFileCount: Int?
    public var managedDirCount: Int?
    public var diskUsageScanS: Double?
    public var diskUsageLastScanS: Double?
    public var diskUsageScanPending: Bool?
    public var diskUsageStale: Bool?
    public var dedupedBytes: Int?
    public var dedupeRatio: Double?
    public var maxBytes: Int?
    public var minPrefixTokens: Int?
    public var blockSize: Int?
    public var writerQueueDepth: Int?
    public var writesEnqueued: Int?
    public var writesCompleted: Int?
    public var writeFailures: Int?
    public var dedupedBlobHits: Int?
    public var entriesEvicted: Int?
    public var restoreHits: Int?
    public var restoreMisses: Int?
    public var restoreFailures: Int?
    public var corruptEntries: Int?
    public var lastMissReason: String?
    public var lastArchivePath: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case restorable
        case mode
        case dir
        case manifestPath = "manifest_path"
        case formatVersion = "format_version"
        case entries
        case bytes
        case logicalBytes = "logical_bytes"
        case physicalBytes = "physical_bytes"
        case livePhysicalBytes = "live_physical_bytes"
        case managedFileBytes = "managed_file_bytes"
        case managedDiskBytes = "managed_disk_bytes"
        case databaseFileBytes = "database_file_bytes"
        case databaseDiskBytes = "database_disk_bytes"
        case untrackedFileBytes = "untracked_file_bytes"
        case untrackedDiskBytes = "untracked_disk_bytes"
        case managedFileCount = "managed_file_count"
        case managedDirCount = "managed_dir_count"
        case diskUsageScanS = "disk_usage_scan_s"
        case diskUsageLastScanS = "disk_usage_last_scan_s"
        case diskUsageScanPending = "disk_usage_scan_pending"
        case diskUsageStale = "disk_usage_stale"
        case dedupedBytes = "deduped_bytes"
        case dedupeRatio = "dedupe_ratio"
        case maxBytes = "max_bytes"
        case minPrefixTokens = "min_prefix_tokens"
        case blockSize = "block_size"
        case writerQueueDepth = "writer_queue_depth"
        case writesEnqueued = "writes_enqueued"
        case writesCompleted = "writes_completed"
        case writeFailures = "write_failures"
        case dedupedBlobHits = "deduped_blob_hits"
        case entriesEvicted = "entries_evicted"
        case restoreHits = "restore_hits"
        case restoreMisses = "restore_misses"
        case restoreFailures = "restore_failures"
        case corruptEntries = "corrupt_entries"
        case lastMissReason = "last_miss_reason"
        case lastArchivePath = "last_archive_path"
    }
}

public struct SessionBank: Codable, Equatable, Sendable {
    public var maxEntries: Int?
    public var maxBytes: Int?
    public var perSessionMaxBytes: Int?
    public var entries: Int?
    public var totalNbytes: Int?
    public var lastMissReason: String?
    public var lastRestoreSource: String?
    public var lastSsdRestoreS: Double?
    public var lastPrefixDiagnostic: DynamicObject?
    public var coldTier: SessionBankColdTier?
    public var prefixes: [SessionBankPrefix]?
    public var evictionLog: [DynamicObject]?

    enum CodingKeys: String, CodingKey {
        case maxEntries = "max_entries"
        case maxBytes = "max_bytes"
        case perSessionMaxBytes = "per_session_max_bytes"
        case entries
        case totalNbytes = "total_nbytes"
        case lastMissReason = "last_miss_reason"
        case lastRestoreSource = "last_restore_source"
        case lastSsdRestoreS = "last_ssd_restore_s"
        case lastPrefixDiagnostic = "last_prefix_diagnostic"
        case coldTier = "cold_tier"
        case prefixes
        case evictionLog = "eviction_log"
    }

    public var restoreHitCount: Int {
        let coldHits = coldTier?.restoreHits ?? 0
        let prefixHits = prefixes?.reduce(0) { $0 + max(0, $1.hits) } ?? 0
        return max(coldHits, prefixHits)
    }

    public var lastEffectiveCachedTokens: Int? {
        guard let values = lastPrefixDiagnostic?.values else { return nil }
        if let miss = values["miss_reason"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
           !miss.isEmpty {
            return nil
        }
        if let nearest = positiveInt(values["nearest_boundary_tokens"]) {
            return nearest
        }
        if let common = positiveInt(values["common_prefix_tokens"]) {
            return common
        }
        return positiveInt(values["stored_prefix_len"])
    }

    public var lastEffectiveCacheSource: String? {
        if let source = lastPrefixDiagnostic?.values["cache_source"]?.stringValue,
           let normalized = normalizedCacheSource(source) {
            return normalized
        }
        if let source = lastRestoreSource,
           let normalized = normalizedCacheSource(source) {
            return normalized
        }
        return nil
    }

    public var hasEffectiveCacheHit: Bool {
        lastEffectiveCachedTokens != nil || restoreHitCount > 0
    }

    private func positiveInt(_ value: JSONValue?) -> Int? {
        guard let intValue = value?.intValue, intValue > 0 else { return nil }
        return intValue
    }

    private func normalizedCacheSource(_ raw: String) -> String? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty, normalized != "none", normalized != "cold" else { return nil }
        return normalized
    }
}

public struct SessionRow: Codable, Equatable, Sendable, Identifiable {
    public var id: String { sessionId }
    public var sessionId: String
    public var prefixLen: Int
    public var bytes: Int
    public var inFlight: Bool?
    public var lastAccessS: Double

    enum CodingKeys: String, CodingKey {
        case sessionId = "session_id"
        case prefixLen = "prefix_len"
        case bytes
        case inFlight = "in_flight"
        case lastAccessS = "last_access_s"
    }
}

public struct SessionsPayload: Codable, Equatable, Sendable {
    public var sessions: [SessionRow]
    public var count: Int
    public var sessionBank: SessionBank?

    enum CodingKeys: String, CodingKey {
        case sessions
        case count
        case sessionBank = "session_bank"
    }
}

public struct MemSnapshot: Codable, Equatable, Sendable {
    public var ok: Bool
    public var activeMemoryBytes: Int?
    public var cacheMemoryBytes: Int?
    public var peakMemoryBytes: Int?
    /// Attribution buckets: `active` split into what users actually ask
    /// about. Weights come from shard file sizes, the bank figure is exact,
    /// working set is the derived remainder (live KV + activations).
    public var modelWeightsBytes: Int?
    public var sessionBankBytes: Int?
    public var generationWorkingBytes: Int?
    public var error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case activeMemoryBytes = "active_memory_bytes"
        case cacheMemoryBytes = "cache_memory_bytes"
        case peakMemoryBytes = "peak_memory_bytes"
        case modelWeightsBytes = "model_weights_bytes"
        case sessionBankBytes = "session_bank_bytes"
        case generationWorkingBytes = "generation_working_bytes"
        case error
    }
}

public struct ThermalFan: Codable, Equatable, Sendable {
    public var rpm: Int?
    public var targetRpm: Int?
    public var actualRpm: Int?
    public var maxCapacityRpm: Int?
    public var mode: String?

    enum CodingKeys: String, CodingKey {
        case rpm
        case targetRpm = "target_rpm"
        case actualRpm = "actual_rpm"
        case maxCapacityRpm = "max_capacity_rpm"
        case mode
    }
}

public struct ThermalSnapshot: Codable, Equatable, Sendable {
    public var ok: Bool
    public var minRpm: Int?
    public var maxRpm: Int?
    public var fans: [ThermalFan]

    enum CodingKeys: String, CodingKey {
        case ok
        case minRpm = "min_rpm"
        case maxRpm = "max_rpm"
        case fans
    }
}

public struct MachineInfo: Codable, Equatable, Sendable {
    public var chipName: String?
    public var machineModel: String?
    public var unifiedMemoryBytes: Int?

    enum CodingKeys: String, CodingKey {
        case chipName = "chip"
        case machineModel = "machine_model"
        case unifiedMemoryBytes = "unified_memory_bytes"
    }
}

public struct HealthPayload: Codable, Equatable, Sendable {
    public struct Startup: Codable, Equatable, Sendable {
        public var launchId: String?
        public var pid: Int?
        public var startedAt: Double?
        public var modelId: String?
        public var modelControls: ModelControls?
        public var warmup: DynamicObject?

        enum CodingKeys: String, CodingKey {
            case launchId = "launch_id"
            case pid
            case startedAt = "started_at"
            case modelId = "model_id"
            case modelControls = "model_controls"
            case warmup
        }
    }

    public struct Thermal: Codable, Equatable, Sendable {
        public var maxRequested: Bool?
        public var maxVerified: Bool?
        public var actualRampVerified: Bool?
        public var fanSummary: DynamicObject?
        public var smart: DynamicObject?
        public var verifiedAt: String?
        public var verified: DynamicObject?

        enum CodingKeys: String, CodingKey {
            case maxRequested = "max_requested"
            case maxVerified = "max_verified"
            case actualRampVerified = "actual_ramp_verified"
            case fanSummary = "fan_summary"
            case smart
            case verifiedAt = "verified_at"
            case verified
        }
    }

    public var ok: Bool
    public var model: String
    public var modelPath: String
    public var generationMode: String
    public var loadMtp: Bool
    public var mtpEnabled: Bool
    public var depth: Int
    public var profile: DynamicObject
    public var contextWindow: Int
    public var maxResponseTokens: Int?
    public var activeRequests: Int
    public var fanMode: String?
    public var fanBoostActive: Bool?
    public var smartFanActiveCount: Int?
    public var smartFanLastTransitionAt: Double?
    public var smartFanLastError: String?
    public var reasoningParser: String
    public var chipName: String?
    public var machineModel: String?
    public var unifiedMemoryBytes: Int?
    public var scheduler: DynamicObject?
    public var sessionBank: SessionBank?
    public var ssdSessionCache: SessionBankColdTier?
    public var startup: Startup?
    public var thermal: Thermal?
    public var vision: VisionCapability?

    public struct VisionCapability: Codable, Equatable, Sendable {
        public var enabled: Bool
        public var formats: [String]?
    }

    enum CodingKeys: String, CodingKey {
        case ok
        case model
        case modelPath = "model_path"
        case generationMode = "generation_mode"
        case loadMtp = "load_mtp"
        case mtpEnabled = "mtp_enabled"
        case depth
        case profile
        case contextWindow = "context_window"
        case maxResponseTokens = "max_response_tokens"
        case activeRequests = "active_requests"
        case fanMode = "fan_mode"
        case fanBoostActive = "fan_boost_active"
        case smartFanActiveCount = "smart_fan_active_count"
        case smartFanLastTransitionAt = "smart_fan_last_transition_at"
        case smartFanLastError = "smart_fan_last_error"
        case reasoningParser = "reasoning_parser"
        case chipName = "chip"
        case machineModel = "machine_model"
        case unifiedMemoryBytes = "unified_memory_bytes"
        case scheduler
        case sessionBank = "session_bank"
        case ssdSessionCache = "ssd_session_cache"
        case startup
        case thermal
        case vision
    }
}

public struct RetrievalModelStatus: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var role: String
    public var modelRef: String
    public var loaded: Bool
    public var resident: Bool
    public var maxTokens: Int
    public var batchSize: Int
    public var requests: Int
    public var items: Int
    public var tokens: Int
    public var errors: Int
    public var computeSeconds: Double
    public var loadSeconds: Double
    public var lastUsedS: Double?
    public var lastError: String?
    public var itemsPerSecond: Double?
    public var avgLatencyMs: Double?

    /// Retrieval models load on first request, so "configured" and "ready" are
    /// different states a user needs to tell apart at a glance.
    public var isEmbedding: Bool { role == "embedding" }
    public var hasBeenUsed: Bool { requests > 0 }
    /// A model that only ever failed must not read as merely idle.
    public var hasFailed: Bool { errors > 0 }

    enum CodingKeys: String, CodingKey {
        case id
        case role
        case modelRef = "model_ref"
        case loaded
        case resident
        case maxTokens = "max_tokens"
        case batchSize = "batch_size"
        case requests
        case items
        case tokens
        case errors
        case computeSeconds = "computeSeconds"
        case loadSeconds = "loadSeconds"
        case lastUsedS = "lastUsedS"
        case lastError = "lastError"
        case itemsPerSecond = "itemsPerSecond"
        case avgLatencyMs = "avgLatencyMs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        role = try container.decodeIfPresent(String.self, forKey: .role) ?? "embedding"
        modelRef = try container.decodeIfPresent(String.self, forKey: .modelRef) ?? id
        loaded = try container.decodeIfPresent(Bool.self, forKey: .loaded) ?? false
        resident = try container.decodeIfPresent(Bool.self, forKey: .resident) ?? false
        maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 0
        batchSize = try container.decodeIfPresent(Int.self, forKey: .batchSize) ?? 0
        requests = try container.decodeIfPresent(Int.self, forKey: .requests) ?? 0
        items = try container.decodeIfPresent(Int.self, forKey: .items) ?? 0
        tokens = try container.decodeIfPresent(Int.self, forKey: .tokens) ?? 0
        errors = try container.decodeIfPresent(Int.self, forKey: .errors) ?? 0
        computeSeconds = try container.decodeIfPresent(Double.self, forKey: .computeSeconds) ?? 0
        loadSeconds = try container.decodeIfPresent(Double.self, forKey: .loadSeconds) ?? 0
        lastUsedS = try container.decodeIfPresent(Double.self, forKey: .lastUsedS)
        lastError = try container.decodeIfPresent(String.self, forKey: .lastError)
        itemsPerSecond = try container.decodeIfPresent(Double.self, forKey: .itemsPerSecond)
        avgLatencyMs = try container.decodeIfPresent(Double.self, forKey: .avgLatencyMs)
    }

    public init(
        id: String,
        role: String,
        modelRef: String = "",
        loaded: Bool = false,
        resident: Bool = false,
        maxTokens: Int = 0,
        batchSize: Int = 0,
        requests: Int = 0,
        items: Int = 0,
        tokens: Int = 0,
        errors: Int = 0,
        computeSeconds: Double = 0,
        loadSeconds: Double = 0,
        lastUsedS: Double? = nil,
        lastError: String? = nil,
        itemsPerSecond: Double? = nil,
        avgLatencyMs: Double? = nil
    ) {
        self.id = id
        self.role = role
        self.modelRef = modelRef
        self.loaded = loaded
        self.resident = resident
        self.maxTokens = maxTokens
        self.batchSize = batchSize
        self.requests = requests
        self.items = items
        self.tokens = tokens
        self.errors = errors
        self.computeSeconds = computeSeconds
        self.loadSeconds = loadSeconds
        self.lastUsedS = lastUsedS
        self.lastError = lastError
        self.itemsPerSecond = itemsPerSecond
        self.avgLatencyMs = avgLatencyMs
    }
}

public struct RetrievalStatus: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var maxResident: Int
    public var resident: [String]
    public var models: [RetrievalModelStatus]

    public var embedders: [RetrievalModelStatus] { models.filter { $0.role == "embedding" } }
    public var rerankers: [RetrievalModelStatus] { models.filter { $0.role == "rerank" } }

    enum CodingKeys: String, CodingKey {
        case enabled
        case maxResident = "max_resident"
        case resident
        case models
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        maxResident = try container.decodeIfPresent(Int.self, forKey: .maxResident) ?? 0
        resident = try container.decodeIfPresent([String].self, forKey: .resident) ?? []
        models = try container.decodeIfPresent([RetrievalModelStatus].self, forKey: .models) ?? []
    }

    public init(enabled: Bool = false, maxResident: Int = 0, resident: [String] = [], models: [RetrievalModelStatus] = []) {
        self.enabled = enabled
        self.maxResident = maxResident
        self.resident = resident
        self.models = models
    }
}

public struct DashboardSnapshot: Codable, Equatable, Sendable {
    public var ts: Double
    public var modelId: String
    public var profile: DynamicObject
    public var contextWindow: Int
    public var activeRequests: Int
    public var inFlight: [InFlightRequest]
    public var latest: MetricsLatest?
    public var recent: [MetricsLatest]
    public var rolling: RollingMetrics
    public var lifetime: LifetimeSnapshot
    public var sessions: SessionsPayload
    public var sessionBank: SessionBank
    public var mem: MemSnapshot
    public var thermal: ThermalSnapshot?
    public var thermalWhenS: Double?
    public var settings: MutableSettings
    public var scheduler: DynamicObject?
    public var machine: MachineInfo
    public var uptimeS: Double
    /// Absent on daemons built before retrieval shipped, so it decodes to a
    /// disabled status rather than failing the whole snapshot.
    public var retrieval: RetrievalStatus?

    enum CodingKeys: String, CodingKey {
        case ts
        case modelId = "model_id"
        case profile
        case contextWindow = "context_window"
        case activeRequests = "active_requests"
        case inFlight = "in_flight"
        case latest
        case recent
        case rolling
        case lifetime
        case sessions
        case sessionBank = "session_bank"
        case mem
        case thermal
        case thermalWhenS = "thermal_when_s"
        case settings
        case scheduler
        case machine
        case uptimeS = "uptime_s"
        case retrieval
    }
}

public struct PrefillHistoryPayload: Codable, Equatable, Sendable {
    public var capacity: Int
    public var history: [DynamicObject]
}

// MARK: - Fan mode (V1 thermal control)

public struct FanModeRequest: Codable, Equatable, Sendable {
    public var mode: String
    public var requireActualRamp: Bool
    public var timeoutS: Double?

    public init(mode: String, requireActualRamp: Bool = false, timeoutS: Double? = nil) {
        self.mode = mode
        self.requireActualRamp = requireActualRamp
        self.timeoutS = timeoutS
    }

    enum CodingKeys: String, CodingKey {
        case mode
        case requireActualRamp = "require_actual_ramp"
        case timeoutS = "timeout_s"
    }
}

public struct FanModeResponse: Codable, Equatable, Sendable {
    public var verified: Bool
    public var currentMode: String?
    public var fanSummary: DynamicObject?
    public var result: DynamicObject?

    enum CodingKeys: String, CodingKey {
        case verified
        case currentMode = "current_mode"
        case fanSummary = "fan_summary"
        case result
    }

    public init(
        verified: Bool,
        currentMode: String? = nil,
        fanSummary: DynamicObject? = nil,
        result: DynamicObject? = nil
    ) {
        self.verified = verified
        self.currentMode = currentMode
        self.fanSummary = fanSummary
        self.result = result
    }
}

// MARK: - Model registry (/v1/models)

public struct ModelsResponse: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable, Identifiable {
        public var id: String
        public var object: String?
        public var ownedBy: String?

        enum CodingKeys: String, CodingKey {
            case id
            case object
            case ownedBy = "owned_by"
        }
    }

    public var object: String
    public var data: [Model]
}
