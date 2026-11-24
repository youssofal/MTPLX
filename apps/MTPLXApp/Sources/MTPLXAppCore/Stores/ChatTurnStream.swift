import Foundation

// MARK: - ChatTurnStream
//
// The complete accumulation state of ONE in-flight assistant turn,
// owned by the conversation that asked for it (issue #324).
//
// Before this type existed, `ChatViewModel` kept a single app-wide set
// of streaming fields and `select(_:)` reset them on every conversation
// switch. Switching away mid-stream therefore (a) destroyed the visible
// partial answer, and (b) orphaned the still-running request: its
// events kept folding into state that no longer belonged to any
// conversation, a follow-up send elsewhere superseded the generation
// token, and the finished turn was never persisted — the server did all
// the work and the transcript kept the user prompt with no reply.
//
// Keying this state by conversation makes switching a pure view change:
// the stream keeps accumulating HERE regardless of what is visible, the
// view model's published surface mirrors whichever conversation is
// selected, and persistence reads this object — never the visible
// conversation's state. It also means a turn in conversation A and a
// turn in conversation B can be in flight at the same time without
// sharing a byte (each holds its own server session id via the
// conversation, so SessionBank warm-prefix reuse is unaffected).
//
// @MainActor like the view model that owns it; SSE events reach it only
// through `ChatViewModel.handleEvent`, which resolves the stream by
// `(conversationID, turnID)` — Sendable identity that can cross the
// stream callback boundary, unlike this class.

@MainActor
final class ChatTurnStream {
    /// The conversation this turn answers. Held strongly for the turn's
    /// lifetime (the tool-loop task already retained it before this
    /// refactor); `delete(_:)` cancels the turn before deleting the
    /// model, so the reference cannot outlive the store row.
    let conversation: ChatConversation
    let conversationID: UUID
    /// Identity shared by every assistant/tool message this turn's tool
    /// loop persists, AND the token SSE events are routed by: a
    /// replaced or cancelled turn's late events resolve against the
    /// registry with a stale `turnID` and are dropped — the per-turn
    /// successor of the old view-model-wide `streamGeneration` counter.
    let turnID = UUID()
    let startedAt = Date()

    // Live documents. Fresh per turn; the view model exposes the
    // CURRENT conversation's pair, so the live wells re-bind on switch.
    let reasoningDocument = StreamingDocumentStore(mode: .plainLines)
    let contentDocument = StreamingDocumentStore(mode: .plainLines)

    // Published-surface state (mirrored by ChatViewModel's computed
    // properties for whichever conversation is visible).
    var phase: StreamingPhase
    var hasReasoning = false
    var hasContent = false
    var pendingToolTraces: [PendingToolTrace] = []
    var liveTurnSources: [SourceRecord] = []
    var decodeReading: HeadlineDecodeReading = .absent
    var handoffAssistantMessageID: UUID?

    // Coalescing buffers ahead of the documents (small, CoW-shared).
    var reasoningBuffer = ""
    var contentBuffer = ""

    // Request plumbing.
    var requestId: String?
    var task: Task<Void, Never>?

    // Per-round accumulation for the tool loop.
    var roundToolCalls: [Int: AccumulatingToolCall] = [:]
    /// The terminal chunk's finish reason, set only when one arrives.
    /// Nil after the byte stream ends means the daemon never finished
    /// the reply (process death, a dropped connection): the round is
    /// lost, not complete. It used to default to "stop", which turned
    /// every such half answer into a persisted, complete-looking reply.
    var roundFinishReason: String?
    var roundUsage: ChatUsage?
    var roundStats: ChatStreamStats?
    /// The daemon's failure message when the round ended with its
    /// `finish_reason: "error"` frame. A round that carries one is a
    /// failed reply, whatever else arrived.
    var roundServerError: String?

    // Think-span accounting (multi-round "Thought · Ns" chip).
    var reasoningStartedAt: Date?
    var completedThinkingMs = 0
    /// Character offset into the accumulated live reasoning document
    /// where the CURRENT round's reasoning begins. The document only
    /// ever appends, so a count-based offset stays valid.
    var roundReasoningStartOffset = 0

    // Sources gathered across the turn's tool calls (raw + deduped).
    var turnSourceAccumulator: [SourceRecord] = []

    var leakedThinkingSplitter = ChatThinkingTagSplitter()

    // Live decode chip: sliding-window samples + throttle.
    var decodeWindowSamples: [(t: Double, tokens: Double)] = []
    var lastLiveDecodeUpdateAt: Date = .distantPast

    // Typewriter pacing state, per turn so concurrent turns pace
    // independently.
    var typewriterPacer = StreamTypewriterPacer()

    init(conversation: ChatConversation, phase: StreamingPhase) {
        self.conversation = conversation
        self.conversationID = conversation.id
        self.phase = phase
    }

    /// Full accumulated reasoning (document + unflushed buffer).
    /// O(reasoning) per access — turn-boundary use only.
    var reasoningText: String { reasoningDocument.rawText + reasoningBuffer }
    /// Full accumulated answer (document + unflushed buffer).
    /// O(answer) per access — turn-boundary use only.
    var contentText: String { contentDocument.rawText + contentBuffer }
    /// System uptime when the oldest currently unrevealed character in
    /// `contentBuffer` arrived from the stream, or `nil` when empty.
    var oldestBufferedArrivalUptime: Double? { typewriterPacer.oldestBufferedArrivalUptime }
}

// MARK: - StreamTypewriterPacer
//
// Lag-bounded adaptive per-tick reveal budget for the answer typewriter.
// Adapts seamlessly to real frame `dt` (120 Hz ProMotion 8.33 ms and
// 60 Hz 16.7 ms alike):
//
// 1. Short-window smoothed arrival rate (`rateWindowSeconds = 0.35s`) with
//    a fast upward-attack sub-window (`fastRateWindowSeconds = 0.15s`) so
//    sudden jumps from 80 tok/s to 300+ tok/s ramp in <150 ms instead of
//    lagging behind for a full second.
// 2. Intra-round coalescing (`burstCoalesceSeconds = 0.75 / 60.0`) keeps
//    sub-frame per-token SSE frames (3–7 ms apart) from shrinking the
//    inter-round gap estimate, so verify/copy rounds type evenly across
//    their gap rather than pasting whole.
// 3. Explicit queue-lag bound (`targetLagSeconds = 0.12s`, `hardLagSeconds
//    = 0.22s`, relaxed naturally to the inter-round spread horizon when
//    rounds themselves arrive slower than 120 ms): whenever backlog or
//    oldest-unrevealed character age exceeds target lag, a smooth
//    proportional catch-up rate drains excess over multiple frames so
//    visible output stays tight to the live TPS gauge without ever
//    "freezing then pasting two lines".
struct StreamTypewriterPacer {
    /// Short wall-clock span the primary arrival-rate window looks back over.
    static let rateWindowSeconds = 0.35
    /// Fast upward-attack sub-window so sudden 80 -> 300+ tok/s bursts ramp
    /// within ~150 ms without reacting to isolated sub-frame bursts.
    static let fastRateWindowSeconds = 0.15
    /// Minimum span required before the fast sub-window can override the
    /// primary window (strictly above intra-round per-token frame spacing).
    static let fastRateMinSpanSeconds = 0.08
    /// Longest gap a drain is planned around; a longer stall drains the
    /// next burst over this span rather than crawling for seconds.
    static let maxPlannedGapSeconds = 1.0
    static let interArrivalSmoothing = 0.3
    /// Arrivals closer than this belong to one round (per-token frames of
    /// a single commit); only wider gaps update the inter-round estimate.
    /// Three quarters of a 60 Hz frame: well above the few-millisecond
    /// intra-round spacing, well below real multi-token round gaps.
    static let burstCoalesceSeconds = 0.75 / 60.0
    /// Drain horizon as a multiple of the expected gap. Slightly past the
    /// next expected arrival, so a block's tail overlaps the next block
    /// instead of leaving idle frames when a round runs a little long;
    /// the carried remainder is a few characters, never a paste.
    static let drainHorizon = 1.25
    /// Soft queue-lag target for continuous / fast streams (seconds).
    static let targetLagSeconds = 0.12
    /// Hard queue-lag ceiling where smooth catch-up accelerates so live
    /// output stays tight behind the TPS gauge (seconds).
    static let hardLagSeconds = 0.22

    private struct PendingBatch {
        let uptime: Double
        var remainingChars: Int
    }

    private(set) var arrivedCharsTotal = 0
    private var arrivalSamples: [(uptime: Double, chars: Int)] = []
    private var pendingBatches: [PendingBatch] = []
    private var pendingHead = 0
    private var trackedPendingChars = 0
    private(set) var lastArrivalUptime: Double = 0
    private var arrivals = 0
    /// Smoothed seconds between content chunks; 0 until two arrivals.
    private(set) var expectedGapSeconds: Double = 0
    /// Uptime of the previous tick; nil before the first one. An uptime
    /// clock may legitimately read 0, so presence is tracked, not the
    /// value.
    private var lastTickUptime: Double?

    /// Arrival uptime of the oldest currently unrevealed character, or `nil`
    /// when no characters are pending.
    var oldestBufferedArrivalUptime: Double? {
        guard pendingHead < pendingBatches.count else { return nil }
        return pendingBatches[pendingHead].uptime
    }

    mutating func recordArrival(chars: Int, now: Double) {
        guard chars > 0 else { return }
        arrivedCharsTotal += chars
        if arrivals > 0 {
            let sinceLast = now - lastArrivalUptime
            // A frame within the same round: the burst grows, the
            // inter-round estimate is untouched. The deadline anchor still
            // moves to the newest frame so a round is drained from its end.
            if sinceLast >= Self.burstCoalesceSeconds {
                let gap = min(Self.maxPlannedGapSeconds, sinceLast)
                expectedGapSeconds = expectedGapSeconds <= 0
                    ? gap
                    : expectedGapSeconds * (1 - Self.interArrivalSmoothing)
                        + gap * Self.interArrivalSmoothing
            }
        }
        lastArrivalUptime = now
        arrivals += 1
        arrivalSamples.append((uptime: now, chars: arrivedCharsTotal))
        pendingBatches.append(PendingBatch(uptime: now, remainingChars: chars))
        trackedPendingChars += chars
    }

    /// Record that `chars` characters were revealed from the front of the
    /// buffer, advancing the oldest-unrevealed arrival timestamp in O(1)
    /// amortized time.
    mutating func noteRevealed(chars: Int) {
        guard chars > 0 else { return }
        var remaining = chars
        while remaining > 0, pendingHead < pendingBatches.count {
            let headChars = pendingBatches[pendingHead].remainingChars
            if remaining >= headChars {
                remaining -= headChars
                trackedPendingChars -= headChars
                pendingHead += 1
            } else {
                pendingBatches[pendingHead].remainingChars = headChars - remaining
                trackedPendingChars -= remaining
                remaining = 0
            }
        }
        compactPendingBatchesIfNeeded()
    }

    /// Clear all pending arrival timestamps (full drain or round reset).
    mutating func noteDrainedAll() {
        pendingBatches.removeAll(keepingCapacity: true)
        pendingHead = 0
        trackedPendingChars = 0
    }

    /// Reconcile tracked pending batches with an externally passed `backlog`
    /// count (covers callers/tests that reveal via `pacedCut` directly
    /// without invoking `noteRevealed`).
    private mutating func syncPendingBacklog(to backlog: Int) {
        guard backlog > 0 else {
            noteDrainedAll()
            return
        }
        if trackedPendingChars > backlog {
            noteRevealed(chars: trackedPendingChars - backlog)
        }
    }

    private mutating func compactPendingBatchesIfNeeded() {
        if pendingHead >= pendingBatches.count {
            pendingBatches.removeAll(keepingCapacity: true)
            pendingHead = 0
            trackedPendingChars = 0
        } else if pendingHead >= 32, pendingHead * 2 >= pendingBatches.count {
            pendingBatches.removeFirst(pendingHead)
            pendingHead = 0
        }
    }

    /// A display tick with nothing to reveal. Keeps the tick clock
    /// current so the first frame after a pause is budgeted as one
    /// frame, not as the whole pause.
    mutating func noteIdleTick(now: Double) {
        lastTickUptime = now
        noteDrainedAll()
    }

    /// Characters to reveal on this tick given `backlog` unrevealed
    /// characters. Unclamped: the caller applies the per-tick floor and
    /// ceiling.
    mutating func tickBudget(backlog: Int, now: Double) -> Int {
        // Clamp the tick span so a main-thread stall does not grant one
        // giant budget. A same-instant second call (the mid-event
        // backstop right after a tick) gets no budget and leaves the
        // caller's floor to keep typing.
        let elapsed = lastTickUptime.map { now - $0 } ?? 1.0 / 60.0
        lastTickUptime = now
        syncPendingBacklog(to: backlog)
        guard backlog > 0, elapsed > 0 else { return 0 }
        let dt = min(elapsed, 0.1)

        // Retain samples within the short rate window (keeping at least the
        // newest two so slow inter-round gaps still yield a valid rate).
        while arrivalSamples.count > 2,
              let first = arrivalSamples.first,
              now - first.uptime > Self.rateWindowSeconds {
            arrivalSamples.removeFirst()
        }

        // Primary short-window rate + fast upward-attack sub-window rate.
        var effectiveRate = 0.0
        if let first = arrivalSamples.first,
           let last = arrivalSamples.last,
           last.uptime > first.uptime {
            let span = last.uptime - first.uptime
            let windowRate = Double(last.chars - first.chars) / max(span, 0.08)
            effectiveRate = windowRate

            // Walk back to the anchor ~fastRateWindowSeconds before `last`
            // so sudden jumps from 80 -> 300+ tok/s attack in <150 ms.
            if arrivalSamples.count >= 3, span > Self.fastRateMinSpanSeconds {
                var idx = arrivalSamples.count - 2
                while idx > 0,
                      last.uptime - arrivalSamples[idx].uptime < Self.fastRateWindowSeconds {
                    idx -= 1
                }
                let anchor = arrivalSamples[idx]
                let fastSpan = last.uptime - anchor.uptime
                if fastSpan >= Self.fastRateMinSpanSeconds {
                    let fastRate = Double(last.chars - anchor.chars) / fastSpan
                    effectiveRate = max(effectiveRate, fastRate)
                }
            }
        }
        let rateBudget = effectiveRate * dt

        // Inter-round spread deadline: type each verify/copy block smoothly
        // across the expected gap to the next round.
        var deadlineBudget = 0.0
        if expectedGapSeconds > 0 {
            let timeLeft = max(
                dt, lastArrivalUptime + expectedGapSeconds * Self.drainHorizon - now
            )
            deadlineBudget = Double(backlog) * dt / timeLeft
        }

        // Explicit lag-bounded queue controller:
        // Relax target lag to the natural inter-round horizon when rounds
        // arrive slower than `targetLagSeconds` (e.g. 250 ms copy-lane gap),
        // while tightly bounding high-TPS streams to ~120–220 ms.
        let effectiveTargetLag = max(
            Self.targetLagSeconds,
            expectedGapSeconds * Self.drainHorizon
        )
        let lagCatchUpBand = Self.hardLagSeconds - Self.targetLagSeconds
        let effectiveHardLag = max(
            Self.hardLagSeconds,
            effectiveTargetLag + lagCatchUpBand
        )

        var lagBudget = 0.0
        if effectiveRate > 0 {
            let targetBacklog = effectiveRate * effectiveTargetLag
            let excess = Double(backlog) - targetBacklog
            if excess > 0 {
                let catchUpSpan = max(dt * 2, effectiveHardLag - effectiveTargetLag)
                lagBudget = (effectiveRate + excess / catchUpSpan) * dt
            }
        }

        // Age-based lag bound: if the oldest queued characters have crossed
        // `effectiveTargetLag`, smoothly accelerate drainage of only the
        // overdue prefix so lag stays bounded even while rate is ramping.
        if let oldest = oldestBufferedArrivalUptime {
            let oldestAge = max(0, now - oldest)
            if oldestAge >= effectiveTargetLag {
                var overdueChars = 0
                var i = pendingHead
                while i < pendingBatches.count {
                    let batch = pendingBatches[i]
                    if now - batch.uptime >= effectiveTargetLag {
                        overdueChars += batch.remainingChars
                        i += 1
                    } else {
                        break
                    }
                }
                if overdueChars > 0 {
                    let remainingTime = max(dt * 2, effectiveHardLag - oldestAge)
                    let ageBudget = rateBudget + Double(overdueChars) * dt / remainingTime
                    lagBudget = max(lagBudget, ageBudget)
                }
            }
        }

        return Int(max(rateBudget, max(deadlineBudget, lagBudget)).rounded(.up))
    }
}
