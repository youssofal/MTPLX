import Foundation

/// Plain sentences for why the cache did or did not cover a prompt.
///
/// The server reports causes as codes: the prefill event's `reread`
/// (mtplx/prefill_plan.py), the session bank's miss reasons, the SSD tier's
/// lookup misses and the agent frontier's miss reasons. The app shows a
/// sentence in the user's language for each and never the code itself
/// (2026-09-29: "miss · ssd_prefix_miss" was all a user saw after a 138 s
/// re-read).
public enum CacheExplanation {
    // MARK: Re-read, while the wait happens

    /// One sentence per fact, in reading order: the cause, the SSD tier's
    /// state when it could not keep a copy, where the state resumes, an SSD
    /// restore, and what is read now with the server's estimate.
    public static func rereadSentences(
        _ reread: PrefillReread,
        ssd: SSDLowDiskNotice? = nil
    ) -> [String] {
        var sentences: [String] = []
        if let cause = causeSentence(reread) {
            sentences.append(cause)
        }
        if let ssd, reread.ssdChecked == true, reread.source != "ssd",
           let cause = reread.cause, leftMemory.contains(cause) {
            // The state left memory and the SSD tier had no copy; say what
            // the tier is doing now rather than guess why it had none.
            sentences.append(
                ssd.severity == .full
                    ? tr("The SSD cache is not saving right now: the disk is almost full")
                    : tr("The SSD cache may skip saves right now: free disk space is low")
            )
        }
        let limitAt = reread.resumeLimitAtToken ?? 0
        switch reread.resumeLimit {
        case "screenshot":
            sentences.append(tr("Resuming before the screenshot at token %@", tokens(limitAt)))
        case "saved_state" where limitAt > 0:
            sentences.append(tr("Resuming from the saved state at token %@", tokens(limitAt)))
        default:
            break
        }
        let restored = reread.restorePointTokens ?? 0
        if reread.source == "ssd", restored > 0 {
            sentences.append(tr("Restored %@ tokens from the SSD cache", tokens(restored)))
        }
        if let recompute = reread.recomputeTokens, recompute > 0 {
            let count = tokens(recompute)
            let rereading = reread.explainsAReread
            if let eta = reread.etaS, eta.isFinite, eta >= 0 {
                sentences.append(
                    rereading
                        ? tr("Re-reading %@ tokens, about %@", count, duration(eta))
                        : tr("Reading %@ new tokens, about %@", count, duration(eta))
                )
            } else {
                sentences.append(
                    rereading ? tr("Re-reading %@ tokens", count) : tr("Reading %@ new tokens", count)
                )
            }
        }
        return sentences
    }

    /// The sentences on one line, for captions and in-flight rows.
    public static func rereadLine(_ reread: PrefillReread, ssd: SSDLowDiskNotice? = nil) -> String {
        rereadSentences(reread, ssd: ssd).joined(separator: " · ")
    }

    /// Causes under which the conversation's state left memory.
    private static let leftMemory: Set<String> = [
        "switched_conversation", "freed_for_memory", "too_large", "evicted", "not_cached",
    ]

    private static func causeSentence(_ reread: PrefillReread) -> String? {
        let at = reread.causeAtToken ?? 0
        switch reread.cause {
        case "history_changed":
            return at > 0
                ? tr("History changed at token %@", tokens(at))
                : tr("History changed from the start")
        case "short_shared_prefix":
            // Only the opening matches: Pi's compaction summaries and the turn
            // after them share one session id, and an early edit looks the
            // same, so the sentence states the overlap, never the intent.
            return tr("Only the first %@ tokens match this conversation's saved state", tokens(at))
        case "screenshot_changed":
            return tr("The screenshot at token %@ changed", tokens(at))
        case "new_conversation":
            return tr("New conversation or first turn since MTPLX started")
        case "not_cached":
            return tr("No saved state for this conversation")
        case "switched_conversation":
            return tr("Switched conversations: this one's saved state was replaced")
        case "freed_for_memory":
            return tr("Saved state was freed to relieve memory pressure")
        case "too_large":
            return tr("The conversation was too large to keep in memory")
        case "evicted":
            return tr("Saved state was removed from memory")
        case "settings_changed":
            return tr("Settings changed since the last turn")
        case "cache_off":
            return tr("The prompt cache is off for this request")
        default:
            // A cause this app does not know yet: the numbers still show.
            return nil
        }
    }

    // MARK: Miss reasons, after the fact

    /// A plain sentence for a cache or frontier miss code; nil for no code.
    /// Unknown codes get a general sentence, never the code.
    public static func missReason(_ code: String?) -> String? {
        guard let raw = code?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        let code = raw.lowercased()
        if let sentence = frontierSentence(code) ?? bankSentence(code) {
            return sentence
        }
        // The frontier prefixes the bank's own code when it has no reason
        // of its own ("miss_" + code).
        if code.hasPrefix("miss_"), let sentence = bankSentence(String(code.dropFirst(5))) {
            return sentence
        }
        return tr("The cache could not be used for this request")
    }

    private static func frontierSentence(_ code: String) -> String? {
        switch code {
        case "miss_no_tool_result":
            return tr("This turn did not carry a tool result")
        case "miss_no_assistant_tool_frontier":
            return tr("No earlier tool call to resume from")
        case "miss_unknown_tool_id":
            return tr("A tool result did not match any earlier tool call")
        case "miss_live_frontier_not_armed":
            return tr("The previous turn did not keep its live state for a tool result")
        case "miss_wrong_session_or_no_prior_frontier":
            return tr("No earlier turn of this conversation to resume from")
        case "miss_template_changed":
            return bankSentence("template_mismatch")
        case "miss_policy_changed":
            return bankSentence("policy_mismatch")
        case "miss_model_changed":
            return bankSentence("model_mismatch")
        case "miss_cache_evicted":
            return bankSentence("evicted")
        case "miss_snapshot_desync":
            return bankSentence("snapshot_desync")
        case "miss_live_frontier_consumed_or_missing":
            return bankSentence("no_snapshot_coverage")
        case "miss_prompt_prefix_changed":
            return bankSentence("prefix_divergence_at_token")
        default:
            return nil
        }
    }

    private static func bankSentence(_ code: String) -> String? {
        switch code {
        // Session bank (CacheMissReason).
        case "new_session":
            return tr("New conversation: nothing saved for it yet")
        case "prefix_divergence_at_token":
            return tr("The conversation history changed, so the saved state no longer matched")
        case "model_mismatch":
            return tr("The model changed since this conversation was saved")
        case "template_mismatch":
            return tr("The chat template changed since this conversation was saved")
        case "policy_mismatch":
            return tr("Generation settings changed since this conversation was saved")
        case "evicted":
            return tr("Saved state was removed from memory")
        case "background_bypass":
            return tr("Background requests do not use the cache")
        case "session_busy":
            return tr("The conversation was busy with another request")
        case "snapshot_desync":
            return tr("The saved state was out of step with the conversation and was not used")
        case "no_snapshot_coverage", "no_gdn_boundaries":
            return tr("No saved state covers this point of the conversation")
        case "oversized_snapshot_skipped":
            return tr("The conversation was too large to keep in memory")
        // The RAM lane's partial-restore refusals.
        case "block_prefix_disabled":
            return tr("Restoring part of a conversation is turned off")
        case _ where code.hasPrefix("below_block_min_match"):
            return tr("Too little of the conversation matched to restore")
        // The SSD tier's lookups.
        case "ssd_cache_off":
            return tr("The SSD cache is off")
        case "ssd_cache_write_only":
            return tr("The SSD cache is only saving, not restoring")
        case "ssd_empty_lookup":
            return tr("Nothing is saved in the SSD cache yet")
        case _ where code.hasPrefix("ssd_restore_error"):
            return tr("Reading the saved state from the SSD failed")
        case "ssd_prefix_miss":
            return tr("No saved state for this conversation in memory or on the SSD")
        case "ssd_prefix_not_better_than_ram", "ssd_prefix_shadowed_by_resident_duplicate":
            return tr("The copy in memory was as good as the one on the SSD")
        case "ssd_prefix_boundary_below_floor":
            // A conversation on the SSD shares the start, but its checkpoint
            // lies before the point the saved state already reaches.
            return tr("The copy in memory was as good as the one on the SSD")
        case "ssd_prefix_no_recurrent_boundary":
            // An older conversation on the SSD shares the start, but has no
            // saved checkpoint inside the shared part to resume from.
            return bankSentence("no_snapshot_coverage")
        case "ssd_format_mismatch", "ssd_mtp_epoch_mismatch", "legacy_ssd_cache_archived":
            return tr("The SSD copy was saved by a different MTPLX version or model setup")
        case "ssd_payload_missing", "ssd_missing_mtp_generation_state", "ssd_missing_mtp_history":
            return tr("The SSD copy was incomplete")
        // Requests the server keeps off the cache.
        case "vision_request_cache_bypass":
            return tr("This image request could not use the cache")
        case "opencode_tool_history_cache_bypass":
            return tr("This tool history could not use the cache")
        case "request_cache_bypass":
            return tr("The cache was skipped for this request")
        case "mtp_batch_cold_prefill", _ where code.hasPrefix("ar_batch_"):
            return tr("Batched requests could not reuse the saved state")
        default:
            return nil
        }
    }

    // MARK: Formatting

    static func tokens(_ count: Int) -> String {
        count.formatted(.number.grouping(.automatic))
    }

    /// The app's duration style (the same localized units as the gauge).
    static func duration(_ seconds: Double) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", locale: L10n.language.locale, max(1, seconds.rounded()))
        }
        let total = Int(seconds.rounded())
        if total < 3600 {
            return tr("%dm %02ds", total / 60, total % 60)
        }
        return tr("%dh %02dm", total / 3600, (total % 3600) / 60)
    }
}
