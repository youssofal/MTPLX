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
        XCTAssertLessThanOrEqual(reveal.count, 256,
            "a stalled-then-recovered stream must catch up as fast typing, not one paste")
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
