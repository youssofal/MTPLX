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
    var agentRunID: String?
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
}

// MARK: - StreamTypewriterPacer
//
// Per-tick reveal budget for the answer typewriter. The engine hands the
// app one content delta per committed verify round, and a context-copy
// round commits a whole block (about 110 characters every 110-250 ms).
// The previous estimator sampled the arrival rate only on ticks that
// received bytes and divided the burst by one frame, so a copy block
// measured as thousands of characters per second and was pasted whole in
// the next frame, then nothing showed until the next round. This one
// measures the rate over a wall-clock window that includes the idle
// frames, tracks the gap between chunks, and sizes each tick so the
// backlog is gone right when the next chunk is expected: a burst is
// typed across the gap instead of pasted, and a steady token-by-token
// stream still reveals every tick with no added lag. The caller owns
// the clock, so the schedule can be replayed tick by tick in a test.
//
// The gap that matters is the one between ROUNDS. The engine writes one
// SSE frame per committed token, so a round lands as several frames a few
// milliseconds apart (measured 2026-09-03 on the copy lane: p50 frame gap
// 7 ms, round gap 40-60 ms). Feeding those sub-frame gaps into the
// estimate dragged the expected gap to ~15 ms, the drain deadline then
// fell inside the very next display tick, and every round was pasted
// whole — the raw 16-20 Hz round cadence on screen instead of typing.
// Arrivals closer together than a display frame are the same burst and
// leave the estimate alone.
struct StreamTypewriterPacer {
    /// Wall-clock span the arrival-rate window looks back over.
    static let rateWindowSeconds = 1.2
    /// Longest gap a drain is planned around; a longer stall drains the
    /// next burst over this span rather than crawling for seconds.
    static let maxPlannedGapSeconds = 1.0
    static let interArrivalSmoothing = 0.3
    /// Arrivals closer than this belong to one round (per-token frames of
    /// a single commit); only wider gaps update the inter-round estimate.
    /// Three quarters of a 60 Hz frame: well above the few-millisecond
    /// intra-round spacing, well below any real round gap.
    static let burstCoalesceSeconds = 0.75 / 60.0
    /// Drain horizon as a multiple of the expected gap. Slightly past the
    /// next expected arrival, so a block's tail overlaps the next block
    /// instead of leaving idle frames when a round runs a little long;
    /// the carried remainder is a few characters, never a paste.
    static let drainHorizon = 1.25

    private(set) var arrivedCharsTotal = 0
    private var arrivalSamples: [(uptime: Double, chars: Int)] = []
    private(set) var lastArrivalUptime: Double = 0
    private var arrivals = 0
    /// Smoothed seconds between content chunks; 0 until two arrivals.
    private(set) var expectedGapSeconds: Double = 0
    /// Uptime of the previous tick; nil before the first one. An uptime
    /// clock may legitimately read 0, so presence is tracked, not the
    /// value.
    private var lastTickUptime: Double?

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
    }

    /// A display tick with nothing to reveal. Keeps the tick clock
    /// current so the first frame after a pause is budgeted as one
    /// frame, not as the whole pause.
    mutating func noteIdleTick(now: Double) {
        lastTickUptime = now
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
        guard backlog > 0, elapsed > 0 else { return 0 }
        let dt = min(elapsed, 0.1)
        while arrivalSamples.count > 2,
              let first = arrivalSamples.first,
              now - first.uptime > Self.rateWindowSeconds {
            arrivalSamples.removeFirst()
        }
        var rateBudget = 0.0
        if let first = arrivalSamples.first,
           let last = arrivalSamples.last,
           last.uptime > first.uptime {
            let rate = Double(last.chars - first.chars) / max(last.uptime - first.uptime, 0.1)
            rateBudget = rate * dt
        }
        var deadlineBudget = 0.0
        if expectedGapSeconds > 0 {
            let timeLeft = max(
                dt, lastArrivalUptime + expectedGapSeconds * Self.drainHorizon - now
            )
            deadlineBudget = Double(backlog) * dt / timeLeft
        }
        return Int(max(rateBudget, deadlineBudget).rounded(.up))
    }
}
