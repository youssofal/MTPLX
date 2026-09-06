import Foundation
import MTPLXAppCore

// MARK: - MetricsLatest typed accessors
//
// The backend `MetricsLatest` is intentionally untyped (mirrors the full
// `_mtplx_dashboard_snapshot.latest` shape over time). Views shouldn't reach
// into `values["key"]` directly — go through these accessors instead.

extension MetricsLatest {
    // --- volume ---
    var promptTokens: Int? { intValue("prompt_tokens") }
    var cachedTokens: Int? { intValue("cached_tokens") }
    var newPrefillTokens: Int? { intValue("new_prefill_tokens") }
    var completionTokens: Int? { intValue("completion_tokens") }
    var generatedTokens: Int? { intValue("generated_tokens") }

    // --- timing ---
    var elapsedS: Double? { doubleValue("elapsed_s") }
    var requestElapsedS: Double? { doubleValue("request_elapsed_s") }
    var promptEvalTimeS: Double? { doubleValue("prompt_eval_time_s") }
    var decodeElapsedS: Double? { doubleValue("decode_elapsed_s") }

    // --- rates ---
    var serverTokS: Double? { doubleValue("server_tok_s") }
    var requestTokS: Double? { doubleValue("request_tok_s") }
    var displayDecodeTokS: Double? {
        firstPositive(
            doubleValue("display_decode_tok_s"),
            slidingLast64,
            slidingLast32,
            slidingFirst64,
            slidingFirst32,
            decodeTokS
        )
    }

    /// Headline decode source for the gauge. TPS means raw generation rate:
    /// live progress uses `decode_tok_s`, and completed turns hold the final
    /// average `decode_tok_s`. Window/display rates are diagnostics only.
    var liveDecodeTokS: Double? {
        firstPositive(
            decodeTokS,
            doubleValue("display_decode_tok_s"),
            slidingLast64,
            slidingLast32
        )
    }
    var slidingFirst32: Double? { doubleValue("sliding_decode_tok_s_first_32") }
    var slidingFirst64: Double? { doubleValue("sliding_decode_tok_s_first_64") }
    var slidingLast32: Double? { doubleValue("sliding_decode_tok_s_last_32") }
    var slidingLast64: Double? { doubleValue("sliding_decode_tok_s_last_64") }

    // --- speculative-decoding ---
    var mtpDepth: Int? { intValue("mtp_depth") }
    var speculativeDepth: Int? { intValue("speculative_depth") }
    var verifyCalls: Int? { intValue("verify_calls") }
    var acceptedDrafts: Int? { intValue("accepted_drafts") }
    var rejectedDrafts: Int? { intValue("rejected_drafts") }
    var draftedTokens: Int? { intValue("drafted_tokens") }
    var correctionTokens: Int? { intValue("correction_tokens") }
    var bonusTokens: Int? { intValue("bonus_tokens") }
    var acceptedByDepth: [Int]? { intArray("accepted_by_depth") }
    var draftedByDepth: [Int]? { intArray("drafted_by_depth") }
    var meanAcceptByDepth: [Double]? { doubleArray("mean_accept_probability_by_depth") }

    // --- verify-decomposition ---
    var verifyTimeS: Double? { doubleValue("verify_time_s") }
    var draftTimeS: Double? { doubleValue("draft_time_s") }
    var acceptTimeS: Double? { doubleValue("accept_time_s") }
    var repairTimeS: Double? { doubleValue("repair_time_s") }
    var targetForwardTimeS: Double? { doubleValue("target_forward_time_s") }
    var verifyForwardTimeS: Double? { doubleValue("verify_forward_time_s") }
    var verifyEvalTimeS: Double? { doubleValue("verify_eval_time_s") }
    var verifyLogitsEvalTimeS: Double? { doubleValue("verify_logits_eval_time_s") }
    var verifyHiddenEvalTimeS: Double? { doubleValue("verify_hidden_eval_time_s") }
    var verifyTargetDistTimeS: Double? { doubleValue("verify_target_distribution_time_s") }
    var verifyEvalUnattributedS: Double? { doubleValue("verify_eval_unattributed_time_s") }
    var snapshotTimeS: Double? { doubleValue("snapshot_time_s") }
    var commitTimeS: Double? { doubleValue("commit_time_s") }
    var captureCommitTimeS: Double? { doubleValue("capture_commit_time_s") }
    var rollbackTimeS: Double? { doubleValue("rollback_time_s") }

    // --- cache ---
    var sessionCacheHit: Bool? { boolValue("session_cache_hit") }
    var cacheMissReason: String? { stringValue("cache_miss_reason") }
    var sessionRestoreMode: String? { stringValue("session_restore_mode") }
    var opencodeToolHistoryLiveFrontierRestore: Bool? {
        boolValue("opencode_tool_history_live_frontier_restore")
    }
    var liveFrontierPolicy: String? { stringValue("live_frontier_policy") }
    var liveFrontierResultTurn: Bool? { boolValue("live_frontier_result_turn") }
    var liveFrontierHit: Bool? { boolValue("live_frontier_hit") }
    var liveFrontierRestoreMode: String? { stringValue("live_frontier_restore_mode") }
    var liveFrontierMissReason: String? { stringValue("live_frontier_miss_reason") }
    var requestSessionKeepLiveRef: Bool? { boolValue("request_session_keep_live_ref") }
    var requestSessionKeepLiveRefReason: String? { stringValue("request_session_keep_live_ref_reason") }
    var requestGenerationMode: String? { stringValue("request_generation_mode") }
    var requestDepth: Int? { intValue("request_depth") }
    var requestEffectiveMtpDepth: Int? { intValue("request_effective_mtp_depth") }
    var requestReasoningMode: String? { stringValue("request_reasoning_mode") }
    var requestEnableThinking: Bool? { boolValue("request_enable_thinking") }
    var requestEnableThinkingOverride: Bool? { boolValue("request_enable_thinking_override") }
    var requestReasoningParser: String? { stringValue("request_reasoning_parser") }
    var preserveThinking: String? { stringValue("preserve_thinking") }
    var preserveThinkingEffective: Bool? { boolValue("preserve_thinking_effective") }
    var stripAssistantReasoningHistory: Bool? { boolValue("strip_assistant_reasoning_history") }
    var requestClientHint: String? { stringValue("request_client_hint") }
    var requestToolCount: Int? { intValue("request_tool_count") }
    var requestToolNames: [String]? { stringArray("request_tool_names") }
    var requestFilteredToolCount: Int? { intValue("request_filtered_tool_count") }
    var requestFilteredToolNames: [String]? { stringArray("request_filtered_tool_names") }
    var requestHiddenToolNames: [String]? { stringArray("request_hidden_tool_names") }
    var requestToolsHiddenByBridge: Bool? { boolValue("request_tools_hidden_by_bridge") }
    var contextLen: Int? { intValue("context_len") }
    var lockWaitTimeS: Double? { doubleValue("lock_wait_time_s") }
    var transcriptRawMessageChars: Int? { intValue("transcript_raw_message_chars") }
    var transcriptCanonicalMessageChars: Int? { intValue("transcript_canonical_message_chars") }
    var transcriptCanonicalized: Bool? { boolValue("transcript_canonicalized") }
    var transcriptCompactedToolResultChars: Int? { intValue("transcript_compacted_tool_result_chars") }
    var transcriptCompactedActiveToolResultChars: Int? { intValue("transcript_compacted_active_tool_result_chars") }
    var transcriptCompactedActiveToolResultReadHints: Int? { intValue("transcript_compacted_active_tool_result_read_hints") }
    var transcriptCompactedActiveReadChars: Int? { intValue("transcript_compacted_active_read_chars") }
    var transcriptInspectionReadBudgetCandidateMessages: Int? { intValue("transcript_inspection_read_budget_candidate_messages") }
    var transcriptInspectionReadBudgetMaxLinesPerFile: Int? { intValue("transcript_inspection_read_budget_max_lines_per_file") }

    // --- memory ---
    var peakMemoryBytes: Int? { intValue("peak_memory_bytes") }

    // --- reasoning ---
    var reasoningTokens: Int? { intValue("reasoning_tokens") }
    var answerTokens: Int? { intValue("answer_tokens") }
    var reasoningReentries: Int? { intValue("reasoning_reentries") }

    // --- server caps ---
    var requestMaxTokens: Int? { intValue("request_max_tokens") }
    var serverMaxResponseTokens: Int? { intValue("server_max_response_tokens") }
    var effectiveMaxTokens: Int? { intValue("effective_max_tokens") }
    var decodeLeaseTokens: Int? { intValue("decode_lease_tokens") }
    var uncappedResponseRequested: Bool? { boolValue("uncapped_response_requested") }
    var uncappedResponseLeaseApplied: Bool? { boolValue("uncapped_response_lease_applied") }
    var remainingContextTokens: Int? { intValue("remaining_context_tokens") }
    var serverCapApplied: Bool? { boolValue("server_cap_applied") }
    var contextCapApplied: Bool? { boolValue("context_cap_applied") }

    // --- helpers ---

    private func doubleValue(_ key: String) -> Double? { values[key]?.doubleValue }
    private func firstPositive(_ values: Double?...) -> Double? {
        for value in values {
            if let value, value.isFinite, value > 0 {
                return value
            }
        }
        return nil
    }
    private func intValue(_ key: String) -> Int? {
        if let n = values[key]?.doubleValue { return Int(n) }
        return nil
    }
    private func boolValue(_ key: String) -> Bool? { values[key]?.boolValue }
    private func stringValue(_ key: String) -> String? { values[key]?.stringValue }
    private func intArray(_ key: String) -> [Int]? {
        guard case .array(let items)? = values[key] else { return nil }
        return items.compactMap { $0.doubleValue.map(Int.init) }
    }
    private func doubleArray(_ key: String) -> [Double]? {
        guard case .array(let items)? = values[key] else { return nil }
        return items.compactMap { $0.doubleValue }
    }
    private func stringArray(_ key: String) -> [String]? {
        guard case .array(let items)? = values[key] else { return nil }
        let strings = items.compactMap { $0.stringValue }
        return strings.isEmpty ? nil : strings
    }
}

// MARK: - DynamicObject convenience

extension DynamicObject {
    func string(_ key: String) -> String? { values[key]?.stringValue }
    func double(_ key: String) -> Double? { values[key]?.doubleValue }
    func int(_ key: String) -> Int? {
        if let n = values[key]?.doubleValue { return Int(n) }
        return nil
    }
    func bool(_ key: String) -> Bool? { values[key]?.boolValue }

    func object(_ key: String) -> [String: JSONValue]? {
        if case .object(let value) = values[key] { return value }
        return nil
    }

    func array(_ key: String) -> [JSONValue]? {
        if case .array(let value) = values[key] { return value }
        return nil
    }
}

// MARK: - JSONValue convenience

extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
}

// MARK: - Profile accessors

/// The backend exposes `profile` as a `DynamicObject` rather than a typed
/// struct because the underlying shape changes per release (sustained,
/// performance-cold, max, custom). These read the common-case fields.
extension DynamicObject {
    var profileName: String? {
        string("name") ?? string("display_name") ?? string("id")
    }

    var profileFamily: String? {
        string("family") ?? string("profile_family")
    }
}

// MARK: - PrefillState convenience

extension PrefillState {
    var progress: Double {
        guard tokensTotal > 0, let done = tokensDone else { return 0 }
        return max(0, min(1, Double(done) / Double(tokensTotal)))
    }

    var displayPrefillTokS: Double? {
        livePrefillTokS
            ?? chunkPrefillTokS
            ?? cumulativePrefillTokS
            ?? prefillTokS
    }

    /// Sanity-filtered prefill TPS. The server constructs several rate
    /// fields from `tokens / elapsed` arithmetic, and on early or
    /// cached chunks those values blow up into the 10k–100k tok/s
    /// range — they're token counts in disguise, not real throughput.
    /// Each source is gated individually so a leaked chunk spike can't
    /// dribble through a fallback. When all sources are out of range,
    /// returns `nil` and the gauge falls back to the percent + tokens
    /// caption.
    ///
    /// Measured chunk work is the live speed, independent of earlier setup
    /// and cache restore. The dial's scale must not reject a genuine rate
    /// above its last tick. Retain the legacy filter only for unmeasured rates.
    func sanePrefillTokS(maxAxis: Double = 1500) -> Double? {
        if let count = chunkSize, count > 0,
           let elapsed = chunkElapsedS, elapsed.isFinite, elapsed > 0 {
            let rate = Double(count) / elapsed
            if rate.isFinite { return rate }
        }
        // 1.2× the dial leaves a little headroom for legitimate
        // over-shoots without admitting the tokens-as-rate artefacts.
        let ceiling = maxAxis * 1.2

        func sane(_ value: Double?) -> Double? {
            guard let value, value.isFinite, value > 0, value <= ceiling else {
                return nil
            }
            return value
        }

        return sane(livePrefillTokS)
            ?? sane(chunkPrefillTokS)
            ?? sane(cumulativePrefillTokS)
            ?? sane(prefillTokS)
    }

    var etaSeconds: Double? {
        guard let done = tokensDone,
              tokensTotal > 0,
              done > 0,
              done < tokensTotal
        else { return nil }
        let remaining = max(0, tokensTotal - done)
        var candidates: [Double] = []
        if let tokS = prefillTokS, tokS > 0 {
            candidates.append(Double(remaining) / tokS)
        }
        if let chunkTokS = chunkPrefillTokS, chunkTokS > 0 {
            candidates.append(Double(remaining) / chunkTokS)
        }
        if let elapsed = elapsedS,
           elapsed > 0,
           let longContextETA = longContextAdjustedETA(
                elapsed: elapsed,
                done: done,
                remaining: remaining
           ) {
            candidates.append(longContextETA)
        }
        return candidates
            .filter { $0.isFinite && $0 >= 0 }
            .max()
    }

    var isActive: Bool { phase == "started" || phase == "chunk" }

    private func longContextAdjustedETA(
        elapsed: Double,
        done: Int,
        remaining: Int
    ) -> Double? {
        // Long-context prefill is not token-linear: later chunks usually
        // cost more because the attention/cache state is larger. The raw
        // cumulative average underestimates 100k+ prompts, so use a bounded
        // work-progress curve only for long prompts.
        guard tokensTotal >= 32_768 else { return nil }
        let tokenProgress = max(0.0001, min(0.9999, Double(done) / Double(tokensTotal)))
        let exponent: Double
        if tokensTotal >= 98_304 {
            exponent = 1.55
        } else if tokensTotal >= 65_536 {
            exponent = 1.45
        } else {
            exponent = 1.35
        }
        let workProgress = pow(tokenProgress, exponent)
        guard workProgress > 0, workProgress < 1 else { return nil }
        let curved = elapsed * (1 - workProgress) / workProgress
        let linear = Double(remaining) / max(Double(done) / elapsed, 0.0001)
        return min(max(curved, linear), linear * 2.25)
    }
}

// MARK: - SessionRow + InFlightRequest helpers

extension SessionRow {
    var ageSeconds: Double {
        max(0, Date().timeIntervalSince1970 - lastAccessS)
    }
}

extension InFlightRequest {
    var shortId: String { Format.shortId(requestId) }

    var promptDigest: String {
        let trimmed = promptPreview.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return tr("(empty prompt)") }
        if trimmed.count <= 64 { return trimmed }
        return String(trimmed.prefix(60)) + "…"
    }

    var hasDecodeProgress: Bool {
        if let completion = lastProgress.values["completion_tokens"]?.doubleValue, completion > 0 {
            return true
        }
        if let generated = lastProgress.values["generated_tokens"]?.doubleValue, generated > 0 {
            return true
        }
        if let decodeElapsed = lastProgress.values["decode_elapsed_s"]?.doubleValue, decodeElapsed > 0 {
            return true
        }
        if let decodeTPS = lastProgress.values["decode_tok_s"]?.doubleValue, decodeTPS > 0 {
            return true
        }
        return false
    }
}

// MARK: - DaemonState semantic

enum DaemonStateKind: String {
    case stopped, starting, warming, running, degraded, stopping, crashed

    var label: String {
        switch self {
        case .stopped: tr("Stopped")
        case .starting: tr("Starting")
        case .warming: tr("Warming")
        case .running: tr("Running")
        case .degraded: tr("Degraded")
        case .stopping: tr("Stopping")
        case .crashed: tr("Crashed")
        }
    }

    var symbol: String {
        switch self {
        case .stopped: "power"
        case .starting: "circle.dotted"
        case .warming: "thermometer.medium"
        case .running: "circle.fill"
        case .degraded: "exclamationmark.triangle.fill"
        case .stopping: "stop.fill"
        case .crashed: "xmark.octagon.fill"
        }
    }
}

extension DaemonState {
    var kind: DaemonStateKind {
        switch self {
        case .stopped: .stopped
        case .starting: .starting
        case .warming: .warming
        case .running: .running
        case .degraded: .degraded
        case .stopping: .stopping
        case .crashed: .crashed
        }
    }

    var detail: String? {
        switch self {
        case .degraded(let message): message
        case .crashed(let status?): tr("exit %@", String(status))
        default: nil
        }
    }
}
