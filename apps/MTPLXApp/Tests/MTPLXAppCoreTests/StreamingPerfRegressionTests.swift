import XCTest
@testable import MTPLXAppCore

// MARK: - StreamingPerfRegressionTests
//
// Pins the 2026-08-17 streaming-freeze fixes: per-frame render cost
// must stay O(blocks), the typewriter can never paste an unbounded
// backlog in one frame, and the metrics SSE framing must keep exact
// message boundaries. Each test guards a specific mechanism that let
// "freeze → vomit" ship while every engine-side number looked healthy.

final class StreamingPerfRegressionTests: XCTestCase {

    // MARK: fenceCount (utf8 rewrite) parity

    /// Reference implementation: the original Character walk.
    private func referenceFenceCount(_ text: String) -> Int {
        var count = 0
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("```") {
                count += 1
                index = text.index(index, offsetBy: 3)
                continue
            }
            index = text.index(after: index)
        }
        return count
    }

    func testFenceCountMatchesReferenceWalk() {
        let samples = [
            "",
            "```",
            "``",
            "````",       // 4 backticks: one fence, one leftover
            "``````",     // 6 backticks: two fences
            "`````",      // 5 backticks: one fence
            "no fences at all",
            "prefix ```swift\ncode\n``` suffix",
            "emoji 🐦🎮 before ``` and after",
            "backtick ` single ` pairs `` still ``",
            "日本語テキスト```コード```終わり",
            String(repeating: "`", count: 31),
            "```one``` middle ```two``` ```three```",
        ]
        for sample in samples {
            XCTAssertEqual(
                StreamingMarkdownBlockSafety.fenceCount(in: sample),
                referenceFenceCount(sample),
                "fenceCount diverged for: \(sample.debugDescription)"
            )
        }
    }

    func testBlockCarriesCachedRenderMetrics() {
        let block = StreamingDocumentBlock(
            id: 7,
            text: "a ```\nb ``` c",
            kind: .plain,
            finalized: true
        )
        XCTAssertEqual(block.fenceMarkerCount, 2)
        XCTAssertEqual(block.lineCount, 2)
    }

    func testBlockClassificationMatchesTextClassification() {
        let texts = [
            "prose line",
            "```swift",
            "let x = 1",
            "print(x)",
            "```",
            "after the fence",
            "inline ``` odd fence prose",
            "tail line",
        ]
        let blocks = texts.enumerated().map { index, text in
            StreamingDocumentBlock(
                id: index,
                text: text,
                kind: .plain,
                finalized: index < texts.count - 1
            )
        }
        let fromTexts = StreamingMarkdownBlockSafety.classifyRoles(texts)
        let fromBlocks = StreamingMarkdownBlockSafety.classifyRoles(blocks)
        XCTAssertEqual(fromTexts, fromBlocks)
    }

    // MARK: line-segment coalescing still fires with the counter gate

    @MainActor
    func testLineCoalescingStillMergesWithCandidateGate() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 8
        defer { StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil }
        let store = StreamingDocumentStore(mode: .plainLines)
        for lineNumber in 0..<40 {
            store.append("line number \(lineNumber)\n")
        }
        XCTAssertGreaterThan(store.liveSegmentMergeCount, 0,
            "candidate gate must not starve the coalescer")
        XCTAssertLessThan(store.blocks.count, 40,
            "merges must keep realized block count sublinear in lines")
        // The document text survives merging byte for byte.
        XCTAssertEqual(
            store.blocks.map(\.text).joined(separator: "\n"),
            (0..<40).map { "line number \($0)" }.joined(separator: "\n")
        )
        XCTAssertEqual(
            store.rawText,
            (0..<40).map { "line number \($0)\n" }.joined()
        )
    }

    @MainActor
    func testFenceLinesAreNeverMerged() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        defer { StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil }
        let store = StreamingDocumentStore(mode: .plainLines)
        store.append("```swift\n")
        for lineNumber in 0..<30 {
            store.append("code \(lineNumber)\n")
        }
        store.append("```\n")
        for block in store.blocks where block.text.contains("\n") {
            XCTAssertFalse(block.text.contains("```"),
                "merged segment may never contain a fence line")
        }
    }

    // MARK: typewriter reveal ceiling (the anti-vomit bound)

    @MainActor
    func testPacedCutBoundsSingleFrameReveal() {
        // Even a runaway budget must respect the frame ceiling — the
        // unbounded whole-drain WAS the "vomit" paste.
        let backlog = String(repeating: "x", count: 10_000)
        let (reveal, rest) = ChatViewModel.pacedCut(backlog, budget: 10_000)
        XCTAssertLessThanOrEqual(
            reveal.count,
            ChatViewModel.typewriterMaxRevealCharacters,
            "a stalled-then-recovered stream must catch up as fast typing, not one paste"
        )
        XCTAssertLessThan(reveal.count, backlog.count / 10)
        XCTAssertEqual(reveal + rest, backlog, "no bytes may be lost or reordered")
    }

    @MainActor
    func testPacedCutKeepsTypingAliveOnZeroBudget() {
        // While the arrival-rate EMA warms up the budget can be 0; the
        // floor keeps characters flowing instead of freezing the reveal.
        let backlog = String(repeating: "y", count: 100)
        let (reveal, rest) = ChatViewModel.pacedCut(backlog, budget: 0)
        XCTAssertEqual(reveal.count, 3)
        XCTAssertEqual(reveal + rest, backlog)
    }

    @MainActor
    func testPacedCutDrainsSmallBuffersWhole() {
        let small = "ab"
        let (reveal, rest) = ChatViewModel.pacedCut(small, budget: 0)
        XCTAssertEqual(reveal, small)
        XCTAssertEqual(rest, "")
    }

    // MARK: typewriter pacer (a copy-lane burst types across the gap)

    private struct PacerTick {
        let tick: Int
        /// Unrevealed characters when the tick began.
        let pending: Int
        let revealed: Int
        let backlogAfter: Int
        /// The pacer had seen two arrivals (an inter-arrival estimate).
        let estimatorWarm: Bool
    }

    private struct PacerRun {
        var ticks: [PacerTick] = []
        /// Unrevealed characters just before each arrival after the first.
        var backlogBeforeArrival: [Int] = []
        var fed = ""
        var revealed = ""
    }

    /// Replays the display tick exactly as `flushStreamingBuffers` does
    /// (pacer budget, then `pacedCut`) on a simulated uptime clock:
    /// `chunk` characters land every `gap` seconds and the reveal ticks
    /// at 60 Hz. The stream end drains whatever is left, as finalize,
    /// cancel and the stream return do.
    ///
    /// `frames` > 1 delivers each chunk the way the engine's wire does —
    /// one SSE frame per committed token, `frameGap` seconds apart — so a
    /// round is a burst of small arrivals, not one big one.
    /// `bigEvery`/`bigChunk` make every `bigEvery`-th round a context-copy
    /// block of `bigChunk` characters.
    @MainActor
    private func replayPacer(
        chunk: Int, gap: Double, chunks: Int, frames: Int = 1, frameGap: Double = 0,
        bigEvery: Int = 0, bigChunk: Int = 0
    ) -> PacerRun {
        var pacer = StreamTypewriterPacer()
        var run = PacerRun()
        var buffer = ""
        let tickSeconds = 1.0 / 60.0
        let totalTicks = Int((Double(chunks) * gap / tickSeconds).rounded(.up))
        var delivered = 0
        var framesDelivered = 0
        for tick in 0..<totalTicks {
            let now = Double(tick) * tickSeconds
            // Deliver every frame due by this tick; the SSE task lands
            // ahead of the frame that reveals it.
            while delivered < chunks {
                let frameIndex = framesDelivered % frames
                let arrival = Double(delivered) * gap + Double(frameIndex) * frameGap
                guard arrival <= now + 1e-9 else { break }
                if delivered > 0, frameIndex == 0 { run.backlogBeforeArrival.append(buffer.count) }
                let letter = Character(UnicodeScalar(UInt8(97 + delivered % 26)))
                let roundChars = (bigEvery > 0 && delivered % bigEvery == bigEvery - 1) ? bigChunk : chunk
                let frameChars = roundChars / frames
                let count = frameIndex == frames - 1 ? roundChars - frameChars * (frames - 1) : frameChars
                let text = String(repeating: letter, count: count)
                buffer += text
                run.fed += text
                pacer.recordArrival(chars: count, now: arrival)
                framesDelivered += 1
                if frameIndex == frames - 1 { delivered += 1 }
            }
            guard delivered > 0 else { continue }
            let pending = buffer.count
            var revealedNow = 0
            if buffer.isEmpty {
                pacer.noteIdleTick(now: now)
            } else {
                let budget = pacer.tickBudget(backlog: pending, now: now)
                let cut = ChatViewModel.pacedCut(buffer, budget: budget)
                run.revealed += cut.reveal
                buffer = cut.rest
                revealedNow = cut.reveal.count
            }
            run.ticks.append(PacerTick(
                tick: tick,
                pending: pending,
                revealed: revealedNow,
                backlogAfter: buffer.count,
                estimatorWarm: pacer.expectedGapSeconds > 0
            ))
        }
        run.revealed += buffer
        return run
    }

    @MainActor
    func testPacerSpreadsCopyLaneBurstAcrossTheRoundGap() {
        // Context-copy lane: one ~110-character block per engine round,
        // every 250 ms. The old estimator pasted each block in one frame.
        let chunk = 110
        let run = replayPacer(chunk: chunk, gap: 0.25, chunks: 8)
        let warm = run.ticks.filter(\.estimatorWarm)
        XCTAssertFalse(warm.isEmpty)
        // (a) No frame pastes more than about a quarter of a block once
        //     the estimator has two samples.
        for sample in warm {
            XCTAssertLessThanOrEqual(sample.revealed, chunk / 4,
                "tick \(sample.tick) revealed \(sample.revealed) characters: a paste, not typing")
        }
        // The block is typed across the whole gap: nearly every frame
        // between two rounds reveals something.
        let steady = run.ticks.filter { $0.estimatorWarm && $0.tick >= 30 }
        let flowing = steady.filter { $0.revealed > 0 }.count
        XCTAssertGreaterThanOrEqual(flowing, steady.count * 9 / 10,
            "the reveal must flow through the gap rather than idle after a paste")
        // (b) The backlog is gone by the time the next block lands
        //     (within one frame's reveal).
        // The drain horizon deliberately reaches a little past the next
        // expected arrival, so a small remainder (well under a third of
        // a block) may still be typing when the next block lands; it
        // must never grow round over round.
        XCTAssertGreaterThanOrEqual(run.backlogBeforeArrival.count, 7)
        for backlog in run.backlogBeforeArrival.dropFirst() {
            XCTAssertLessThanOrEqual(backlog, chunk / 3,
                "a block must be nearly typed out as the next one arrives, not pile up")
        }
        if let first = run.backlogBeforeArrival.dropFirst().first,
           let last = run.backlogBeforeArrival.last {
            XCTAssertLessThanOrEqual(last, first + 3, "the carried remainder must not grow")
        }
        // (d) The end-of-stream drain reveals everything, in order.
        XCTAssertEqual(run.revealed, run.fed)
    }

    @MainActor
    func testPacerTypesPerTokenFrameRoundsAcrossTheRoundGap() {
        // The wire shape the engine actually sends (measured 2026-09-03 on a
        // Flash-Next follow-up turn, 2,613 frames): a round every ~42 ms,
        // landing as two per-token frames ~3 ms apart, ~14 characters. Fed
        // to the old estimator, the 3 ms intra-round gaps pulled the
        // expected gap to ~19 ms, so the drain deadline fell inside the
        // next tick and each round revealed as 12, 2, 0 — a paste, an
        // afterthought, an idle frame — at the raw round cadence. With
        // rounds coalesced the estimate is ~39 ms and the same round types
        // 7, 7, 7 (well, 7, 6, 6, 6 ...) on every display tick.
        let chunk = 14
        let run = replayPacer(chunk: chunk, gap: 0.042, chunks: 60, frames: 2, frameGap: 0.003)
        let warm = run.ticks.filter { $0.estimatorWarm && $0.tick >= 12 }
        XCTAssertFalse(warm.isEmpty)
        // A 14-character round spans two and a half display ticks: no tick
        // reveals more than ~60% of one. (Old estimator: 14 in one tick.)
        for sample in warm {
            XCTAssertLessThanOrEqual(sample.revealed, chunk * 6 / 10,
                "tick \(sample.tick) revealed \(sample.revealed) of a \(chunk)-character round: a paste")
        }
        // The reveal flows on nearly every tick. (Old estimator: 61%.)
        let flowing = warm.filter { $0.revealed > 0 }.count
        XCTAssertGreaterThanOrEqual(flowing, warm.count * 9 / 10,
            "the reveal must type through the round gap, not paste and idle")
        XCTAssertEqual(run.revealed, run.fed)
    }

    @MainActor
    func testPacerTypesACopyLaneBlockInsteadOfPastingIt() {
        // Same wire shape, with every sixth round a context-copy block of
        // ~100 characters (measured p99 88, max 114). The old estimator
        // pasted 86 of the 100 in a single frame; coalesced, the block is
        // typed across the following ticks (37, 37, 26).
        let run = replayPacer(
            chunk: 14, gap: 0.042, chunks: 60, frames: 2, frameGap: 0.003,
            bigEvery: 6, bigChunk: 100
        )
        let warm = run.ticks.filter { $0.estimatorWarm && $0.tick >= 12 }
        XCTAssertFalse(warm.isEmpty)
        let biggest = warm.map(\.revealed).max() ?? 0
        XCTAssertLessThan(biggest, 60,
            "a copy block must be typed across several frames, not pasted (\(biggest) in one)")
        // Bursts drain before they pile up: the block is gone within the
        // next few rounds, so no arrival ever finds more than one block
        // still unrevealed.
        for backlog in run.backlogBeforeArrival.dropFirst(6) {
            XCTAssertLessThanOrEqual(backlog, 100)
        }
        XCTAssertEqual(run.revealed, run.fed)
    }

    func testPacerIgnoresSubFrameArrivalsInTheGapEstimate() {
        var pacer = StreamTypewriterPacer()
        // Round 1: four frames 7 ms apart; round 2 lands 50 ms later.
        pacer.recordArrival(chars: 13, now: 1.000)
        pacer.recordArrival(chars: 13, now: 1.007)
        pacer.recordArrival(chars: 13, now: 1.014)
        pacer.recordArrival(chars: 13, now: 1.021)
        XCTAssertEqual(pacer.expectedGapSeconds, 0, "frames of one round are not a gap sample")
        pacer.recordArrival(chars: 13, now: 1.071)
        XCTAssertEqual(pacer.expectedGapSeconds, 0.05, accuracy: 1e-9)
        pacer.recordArrival(chars: 13, now: 1.078)
        XCTAssertEqual(pacer.expectedGapSeconds, 0.05, accuracy: 1e-9,
            "a 7 ms frame inside round 2 must not pull the estimate down")
    }

    @MainActor
    func testPacerKeepsTokenStreamRevealingEveryTickWithoutLag() {
        // (c) Plain token-by-token arrival: ~5 characters every 42 ms.
        let chunk = 5
        let run = replayPacer(chunk: chunk, gap: 0.042, chunks: 60)
        XCTAssertFalse(run.ticks.isEmpty)
        XCTAssertGreaterThan(run.ticks.filter { $0.pending > 0 }.count, 60)
        for sample in run.ticks {
            if sample.pending > 0 {
                XCTAssertGreaterThan(sample.revealed, 0,
                    "tick \(sample.tick) had text pending and revealed nothing")
            }
            XCTAssertLessThanOrEqual(sample.backlogAfter, chunk,
                "tick \(sample.tick) let the backlog grow past one chunk")
        }
        for backlog in run.backlogBeforeArrival {
            XCTAssertLessThanOrEqual(backlog, chunk,
                "steady arrival must not accumulate lag")
        }
        XCTAssertEqual(run.revealed, run.fed)
    }

    @MainActor
    func testPacerBudgetIsZeroWithoutBacklogAndColdEstimatorFloorsAtThree() {
        var pacer = StreamTypewriterPacer()
        XCTAssertEqual(pacer.tickBudget(backlog: 0, now: 1.0), 0)
        pacer.recordArrival(chars: 110, now: 1.0)
        // One sample: no rate, no gap estimate; the caller's floor of
        // three characters keeps the first block typing.
        let budget = pacer.tickBudget(backlog: 110, now: 1.0 + 1.0 / 60.0)
        XCTAssertEqual(budget, 0)
        let (reveal, rest) = ChatViewModel.pacedCut(String(repeating: "z", count: 110), budget: budget)
        XCTAssertEqual(reveal.count, 3)
        XCTAssertEqual(rest.count, 107)
    }

    @MainActor
    func testDisplayLinkFrameRateRangeUsesNativeProMotionRefreshRate() {
        let proMotion = ChatViewModel.preferredDisplayLinkFrameRateRange(maximumFramesPerSecond: 120)
        XCTAssertEqual(proMotion.minimum, 60)
        XCTAssertEqual(proMotion.maximum, 120)
        XCTAssertEqual(proMotion.preferred, 120)

        let standard = ChatViewModel.preferredDisplayLinkFrameRateRange(maximumFramesPerSecond: 60)
        XCTAssertEqual(standard.minimum, 60)
        XCTAssertEqual(standard.maximum, 60)
        XCTAssertEqual(standard.preferred, 60)
    }

    @MainActor
    func testPacer120HzHighTPSRampStaysLagBoundedAndSmooth() {
        // Simulate a 120 Hz ProMotion display where generation starts at
        // ~80 tok/s (~320 chars/s) for 0.6 s, then suddenly accelerates to
        // ~350 tok/s (~1,400 chars/s, 14 chars every 10 ms) for 2.0 s.
        // The old 1.2 s rate window + unbounded queue fell >500-1,500 ms
        // behind the live TPS gauge on this ramp.
        var pacer = StreamTypewriterPacer()
        var buffer = ""
        var fed = ""
        var revealed = ""
        let tickSeconds = 1.0 / 120.0

        // Build arrival schedule: 0..0.6s at 320 chars/s, 0.6..2.6s at 1,400 chars/s.
        var schedule: [(t: Double, count: Int)] = []
        var t = 0.0
        while t < 0.6 {
            schedule.append((t, 8))
            t += 0.025 // 320 chars/s
        }
        while t < 2.6 {
            schedule.append((t, 14))
            t += 0.010 // 1,400 chars/s (~350 tok/s)
        }

        var nextArrival = 0
        var lagSamplesMs: [Double] = []
        var maxSingleFrameReveal = 0
        let totalTicks = Int((2.7 / tickSeconds).rounded(.up))
        for tick in 0..<totalTicks {
            let now = Double(tick) * tickSeconds
            while nextArrival < schedule.count, schedule[nextArrival].t <= now + 1e-9 {
                let item = schedule[nextArrival]
                let chunk = String(repeating: "a", count: item.count)
                buffer += chunk
                fed += chunk
                pacer.recordArrival(chars: item.count, now: item.t)
                nextArrival += 1
            }
            if buffer.isEmpty {
                pacer.noteIdleTick(now: now)
                continue
            }
            let oldestBeforeCut = pacer.oldestBufferedArrivalUptime
            let budget = pacer.tickBudget(backlog: buffer.count, now: now)
            let cut = ChatViewModel.pacedCut(buffer, budget: budget)
            revealed += cut.reveal
            buffer = cut.rest
            if buffer.isEmpty {
                pacer.noteDrainedAll()
            } else {
                pacer.noteRevealed(chars: cut.reveal.count)
            }
            maxSingleFrameReveal = max(maxSingleFrameReveal, cut.reveal.count)
            if let oldestBeforeCut, now >= 0.65, now <= 2.55 {
                lagSamplesMs.append(max(0, (now - oldestBeforeCut) * 1000))
            }
        }
        revealed += buffer

        XCTAssertEqual(revealed, fed, "all bytes must be revealed in order")
        XCTAssertFalse(lagSamplesMs.isEmpty)
        let sortedLags = lagSamplesMs.sorted()
        let p95LagMs = sortedLags[min(sortedLags.count - 1, Int(Double(sortedLags.count) * 0.95))]
        let maxLagMs = sortedLags.last ?? 0
        XCTAssertLessThanOrEqual(
            p95LagMs, 180.0,
            "120 Hz 350 tok/s p95 arrival-to-paint lag drifted to \(p95LagMs) ms"
        )
        XCTAssertLessThanOrEqual(
            maxLagMs, 240.0,
            "120 Hz 350 tok/s max lag (\(maxLagMs) ms) exceeded lag bound"
        )
        XCTAssertLessThan(
            maxSingleFrameReveal, 60,
            "high-TPS ramp must reveal smoothly frame-by-frame, not paste (\(maxSingleFrameReveal) chars)"
        )
    }

    @MainActor
    func testPacer120HzCopyLane350TokSBurstsStaySmoothAndLagBounded() {
        // Context-copy lane at ~350 tok/s on 120 Hz: 140-char bursts landing
        // every 100 ms (two 70-char SSE frames 3 ms apart).
        var pacer = StreamTypewriterPacer()
        var buffer = ""
        var fed = ""
        var revealed = ""
        let tickSeconds = 1.0 / 120.0
        var schedule: [(t: Double, count: Int)] = []
        for round in 0..<25 {
            let base = Double(round) * 0.10
            schedule.append((base, 70))
            schedule.append((base + 0.003, 70))
        }

        var nextArrival = 0
        var warmReveals: [Int] = []
        var warmLagsMs: [Double] = []
        let totalTicks = Int((2.6 / tickSeconds).rounded(.up))
        for tick in 0..<totalTicks {
            let now = Double(tick) * tickSeconds
            while nextArrival < schedule.count, schedule[nextArrival].t <= now + 1e-9 {
                let item = schedule[nextArrival]
                let chunk = String(repeating: "c", count: item.count)
                buffer += chunk
                fed += chunk
                pacer.recordArrival(chars: item.count, now: item.t)
                nextArrival += 1
            }
            if buffer.isEmpty {
                pacer.noteIdleTick(now: now)
                continue
            }
            let oldestBefore = pacer.oldestBufferedArrivalUptime
            let budget = pacer.tickBudget(backlog: buffer.count, now: now)
            let cut = ChatViewModel.pacedCut(buffer, budget: budget)
            revealed += cut.reveal
            buffer = cut.rest
            if buffer.isEmpty {
                pacer.noteDrainedAll()
            } else {
                pacer.noteRevealed(chars: cut.reveal.count)
            }
            if now >= 0.25, now <= 2.45 {
                warmReveals.append(cut.reveal.count)
                if let oldestBefore {
                    warmLagsMs.append(max(0, (now - oldestBefore) * 1000))
                }
            }
        }
        revealed += buffer

        XCTAssertEqual(revealed, fed)
        XCTAssertFalse(warmReveals.isEmpty)
        let maxReveal = warmReveals.max() ?? 0
        XCTAssertLessThan(
            maxReveal, 35,
            "140-char copy block at 120 Hz must spread across ~12 frames, not paste (\(maxReveal) chars)"
        )
        let maxLag = warmLagsMs.max() ?? 0
        XCTAssertLessThanOrEqual(
            maxLag, 220.0,
            "120 Hz copy-lane lag (\(maxLag) ms) exceeded bound"
        )
    }

    func testPacerOldestBufferedArrivalUptimeTracksFifoOrder() {
        var pacer = StreamTypewriterPacer()
        XCTAssertNil(pacer.oldestBufferedArrivalUptime)
        pacer.recordArrival(chars: 10, now: 1.0)
        pacer.recordArrival(chars: 20, now: 1.05)
        XCTAssertEqual(pacer.oldestBufferedArrivalUptime ?? -1, 1.0, accuracy: 1e-9)

        pacer.noteRevealed(chars: 6)
        XCTAssertEqual(pacer.oldestBufferedArrivalUptime ?? -1, 1.0, accuracy: 1e-9)

        pacer.noteRevealed(chars: 4)
        XCTAssertEqual(pacer.oldestBufferedArrivalUptime ?? -1, 1.05, accuracy: 1e-9)

        pacer.noteDrainedAll()
        XCTAssertNil(pacer.oldestBufferedArrivalUptime)
    }

    @MainActor
    func testUIStreamPerfProbeAggregatesLagBacklogAndSwiftUIPublishes() {
        let probe = UIStreamPerfProbe(environment: ["MTPLX_UI_PERF": "1"])
        XCTAssertTrue(probe.enabled)
        probe.turnStarted()
        probe.flushApplied(
            drainedBytes: 12, applyMs: 0.4, blocksAfter: 1,
            backlogChars: 40, lagMs: 18.2
        )
        probe.flushApplied(
            drainedBytes: 16, applyMs: 0.5, blocksAfter: 2,
            backlogChars: 125, lagMs: 64.7
        )
        probe.flushApplied(
            drainedBytes: 14, applyMs: 0.3, blocksAfter: 2,
            backlogChars: 20, lagMs: 31.4
        )
        probe.swiftUIItemsPublished()
        probe.swiftUIItemsPublished()
        probe.turnEnded(requestId: "test-req-lag")

        let summary = probe.lastTurnSummary
        XCTAssertEqual(summary?.flushes, 3)
        XCTAssertEqual(summary?.backlogCharsMax, 125)
        XCTAssertEqual(summary?.swiftUIItemsPublishes, 2)
        XCTAssertEqual(summary?.lagMsP50 ?? -1, 31.4, accuracy: 0.1)
        XCTAssertEqual(summary?.lagMsMax ?? -1, 64.7, accuracy: 0.1)
    }

    // MARK: SSE line accumulator framing

    private func messages(from payload: String) -> [SSEMessage] {
        var accumulator = SSELineAccumulator()
        var out: [SSEMessage] = []
        for byte in payload.utf8 {
            if let message = accumulator.consume(byte) {
                out.append(message)
            }
        }
        return out
    }

    func testAccumulatorFramesCRLFAndLFMessages() {
        let payload = "event: snapshot\r\ndata: {\"a\":1}\r\n\r\n"
            + ": heartbeat comment\n"
            + "event: progress\ndata: {\"b\":2}\n\n"
            + "data: first\ndata: second\n\n"
        let parsed = messages(from: payload)
        XCTAssertEqual(parsed, [
            SSEMessage(event: "snapshot", data: "{\"a\":1}"),
            SSEMessage(event: "progress", data: "{\"b\":2}"),
            SSEMessage(event: "message", data: "first\nsecond"),
        ])
    }

    func testAccumulatorMatchesLegacyParser() {
        let payload = "event: thermal\ndata: {\"t\":61.5}\n\n"
            + "event: new_max_tps\r\ndata: {\"tps\":81.2}\r\n\r\n"
        let legacy = SSEParser().parse(payload)
        XCTAssertEqual(messages(from: payload), legacy)
    }

    func testAccumulatorHoldsIncompleteMessage() {
        // No trailing blank line: nothing may be emitted early.
        XCTAssertTrue(messages(from: "event: x\ndata: 1\n").isEmpty)
    }
}
