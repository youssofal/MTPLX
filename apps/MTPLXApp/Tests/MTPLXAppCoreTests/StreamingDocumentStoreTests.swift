import XCTest
@testable import MTPLXAppCore

@MainActor
final class StreamingDocumentStoreTests: XCTestCase {
    func testSingleLineDeltasKeepStableTailBlock() {
        let store = StreamingDocumentStore(mode: .mathLines)

        store.append("Let $x")
        let firstID = store.blocks.first?.id

        store.append(" be")

        XCTAssertEqual(store.blocks.count, 1)
        XCTAssertEqual(store.blocks.first?.id, firstID)
        XCTAssertEqual(store.blocks.first?.text, "Let $x be")
        XCTAssertEqual(store.rawText, "Let $x be")
        XCTAssertFalse(store.blocks.first?.finalized ?? true)
    }

    func testMultiLineDeltasFinalizeOnlyCompletedLines() {
        let store = StreamingDocumentStore(mode: .mathLines)

        store.append("Let $x=1$\nThen\nTail")

        XCTAssertEqual(store.blocks.map(\.text), ["Let $x=1$", "Then", "Tail"])
        XCTAssertEqual(store.blocks.map(\.finalized), [true, true, false])
        XCTAssertEqual(store.rawText, "Let $x=1$\nThen\nTail")
        guard case .mathRuns(let runs) = store.blocks[0].kind else {
            return XCTFail("Expected first line to keep parsed math runs")
        }
        XCTAssertTrue(runs.contains { $0.kind == .inlineMath && $0.text == "x=1" })
    }

    func testLaterDeltasDoNotReparseFinalizedMathLines() {
        let store = StreamingDocumentStore(mode: .mathLines)

        store.append("Let $x=1$\n")
        let finalizedID = store.blocks.first?.id
        #if DEBUG
        let mathParseCount = store.diagnostics.mathParseCount
        #endif

        store.append("plain tail")

        XCTAssertEqual(store.blocks.first?.id, finalizedID)
        XCTAssertEqual(store.blocks.first?.text, "Let $x=1$")
        #if DEBUG
        XCTAssertEqual(store.diagnostics.mathParseCount, mathParseCount)
        #endif
    }

    func testMarkdownParagraphsFinalizeAtStableBoundaries() {
        let store = StreamingDocumentStore(mode: .markdown)

        store.append("First paragraph")
        XCTAssertEqual(store.blocks.count, 1)
        XCTAssertFalse(store.blocks[0].finalized)

        store.append("\n\nSecond paragraph")

        XCTAssertEqual(store.blocks.map(\.text), ["First paragraph\n\n", "Second paragraph"])
        XCTAssertEqual(store.blocks.map(\.finalized), [true, false])
        XCTAssertEqual(store.rawText, "First paragraph\n\nSecond paragraph")
    }

    func testMarkdownCodeFenceStreamsAsStableCodeBlock() {
        let store = StreamingDocumentStore(mode: .markdown)

        store.append("Intro\n```swift\nlet x = 1")

        XCTAssertEqual(store.blocks.count, 2)
        XCTAssertEqual(store.blocks[0].text, "Intro\n")
        XCTAssertTrue(store.blocks[0].finalized)
        guard case .codeFence(let language, let code, let closed) = store.blocks[1].kind else {
            return XCTFail("Expected live code fence block")
        }
        XCTAssertEqual(language, "swift")
        XCTAssertEqual(code, "let x = 1")
        XCTAssertFalse(closed)

        store.append("\n```\n\nAfter")

        guard case .codeFence(let closedLanguage, let closedCode, let isClosed) = store.blocks[1].kind else {
            return XCTFail("Expected finalized code fence block")
        }
        XCTAssertEqual(closedLanguage, "swift")
        XCTAssertEqual(closedCode, "let x = 1\n")
        XCTAssertTrue(isClosed)
        XCTAssertTrue(store.blocks[1].finalized)
        XCTAssertEqual(store.rawText, "Intro\n```swift\nlet x = 1\n```\n\nAfter")
    }

    func testEscapedDollarsAndIncompleteMathRemainReadable() {
        let runs = StreamingDocumentStore.mathRuns(in: #"cost \$5 and $x+1$ then $unfinished"#)

        XCTAssertEqual(runs.map(\.kind), [.text, .inlineMath, .text])
        XCTAssertEqual(runs[0].text, #"cost \$5 and "#)
        XCTAssertEqual(runs[1].text, "x+1")
        XCTAssertEqual(runs[2].text, " then $unfinished")
    }

    func testTickerAndCurrencyDollarsRemainText() {
        let text = "launched $YCO at 15 and claimed over $1 billion in volume"
        let runs = StreamingDocumentStore.mathRuns(in: text)

        XCTAssertEqual(runs.count, 1)
        XCTAssertEqual(runs[0].kind, .text)
        XCTAssertEqual(runs[0].text, text)
    }

    func testBareLatexComparisonStreamsAsInlineMath() {
        let runs = StreamingDocumentStore.mathRuns(in: #"So d_i\le6<9."#)

        XCTAssertEqual(runs.map(\.kind), [.text, .inlineMath, .text])
        XCTAssertEqual(runs[0].text, "So ")
        XCTAssertEqual(runs[1].text, #"d_i\le6<9"#)
        XCTAssertEqual(runs[2].text, ".")
    }

    func testBareLatexSetMembershipKeepsEscapedBracesTogether() {
        let runs = StreamingDocumentStore.mathRuns(in: #"So d_i\in \{1,\dots,9\}."#)

        XCTAssertEqual(runs.map(\.kind), [.text, .inlineMath, .text])
        XCTAssertEqual(runs[0].text, "So ")
        XCTAssertEqual(runs[1].text, #"d_i\in \{1,\dots,9\}"#)
        XCTAssertEqual(runs[2].text, ".")
    }

    func testBackslashPathsRemainPlainText() {
        let text = #"Use C:\Users\name if needed"#
        let runs = StreamingDocumentStore.mathRuns(in: text)

        XCTAssertEqual(runs.count, 1)
        XCTAssertEqual(runs[0].kind, .text)
        XCTAssertEqual(runs[0].text, text)
    }

    func testMathLineModeParsesBareLatexTail() {
        let store = StreamingDocumentStore(mode: .mathLines)

        store.append(#"We assumed d_i\ge1."#)

        XCTAssertEqual(store.blocks.count, 1)
        guard case .mathRuns(let runs) = store.blocks[0].kind else {
            return XCTFail("Expected bare LaTeX in the live tail to parse as math runs")
        }
        XCTAssertEqual(runs.map(\.kind), [.text, .inlineMath, .text])
        XCTAssertEqual(runs[1].text, #"d_i\ge1"#)
    }

    func testReadableLatexFormatterCoversAIMEOperators() {
        // 2026-07-31: ^/_ now render as real Unicode scripts wherever
        // the glyphs exist (founder: literal carets are not math).
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"d_i\le6<9"#),
            "dᵢ≤6<9"
        )
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"d_i\in \{1,\dots,9\}"#),
            "dᵢ∈ {1,…,9}"
        )
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\frac{a+b}{2}"#),
            "(a+b)/2"
        )
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\int_{-\infty}^{\infty} e^{-x^2}"#),
            "∫_(-∞)^∞ e^-x²"
        )
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\sum_{p=0}^6 \binom{6}{p}"#),
            "Σₚ₌₀⁶ C(6, p)"
        )
    }

    func testIncrementalWordCountAndRawReconstruction() {
        let store = StreamingDocumentStore(mode: .plainLines)

        store.append("one ")
        store.append("two\nthree")

        XCTAssertEqual(store.wordCount, 3)
        XCTAssertEqual(store.rawText, "one two\nthree")
        XCTAssertEqual(store.blocks.map(\.text), ["one two", "three"])
    }

    func testPlainTextModeKeepsCodeOutputAsSingleLiveBlock() {
        let store = StreamingDocumentStore(mode: .plainText)

        store.append("```python\n")
        store.append("print('one')\nprint('two')\n")
        store.append("```")

        XCTAssertEqual(store.rawText, "```python\nprint('one')\nprint('two')\n```")
        XCTAssertEqual(store.blocks.count, 1)
        XCTAssertEqual(store.blocks.first?.text, store.rawText)
        XCTAssertFalse(store.blocks.first?.finalized ?? true)
        XCTAssertEqual(store.wordCount, 4)
    }

    func testPlainLinesModeStreamsCodeAsPlainRowsNotMarkdownBlocks() {
        let store = StreamingDocumentStore(mode: .plainLines)

        store.append("```python\nprint('one')\nprint('two')\n```")

        XCTAssertEqual(store.rawText, "```python\nprint('one')\nprint('two')\n```")
        XCTAssertEqual(store.blocks.map(\.text), ["```python", "print('one')", "print('two')", "```"])
        XCTAssertFalse(store.blocks.contains { block in
            if case .codeFence = block.kind {
                return true
            }
            return false
        })
    }

    func testLongUnfinalizedLineIsSoftSegmented() {
        let store = StreamingDocumentStore(mode: .mathLines)
        let longLine = String(repeating: "x", count: 5_000)

        store.append(longLine)

        XCTAssertEqual(store.rawText, longLine)
        XCTAssertGreaterThan(store.blocks.count, 1)
        XCTAssertLessThanOrEqual(store.blocks.map { $0.text.count }.max() ?? 0, 2_048)
        XCTAssertFalse(store.blocks.last?.finalized ?? true)
    }
}

// MARK: - Line-segment coalescing (2026-07-31 streaming-jank fix)

@MainActor
final class StreamingDocumentSegmentCoalescingTests: XCTestCase {
    override func tearDown() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil
        super.tearDown()
    }

    private func feedLines(_ store: StreamingDocumentStore, count: Int) {
        for index in 0..<count {
            store.append("line-\(index)\n")
        }
    }

    func testOldFinalizedLinesMergeIntoOneSegmentBlock() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        let store = StreamingDocumentStore(mode: .plainLines)

        // 4 (segment) + 8 (fresh window) finalized lines trip one merge.
        feedLines(store, count: 12)
        store.append("open tail")

        let merged = store.blocks.filter { $0.text.contains("\n") }
        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(
            merged.first?.text,
            "line-0\nline-1\nline-2\nline-3"
        )
        XCTAssertTrue(merged.first?.finalized ?? false)
        // 1 segment + 8 fresh finalized lines + growing tail.
        XCTAssertEqual(store.blocks.count, 10)
        XCTAssertEqual(store.blocks.last?.text, "open tail")
        XCTAssertFalse(store.blocks.last?.finalized ?? true)
    }

    func testRawTextAndRecentTextSurviveMerging() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        let store = StreamingDocumentStore(mode: .plainLines)
        let reference = StreamingDocumentStore(mode: .plainLines)
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 0
        // reference store built with merging disabled
        feedLines(reference, count: 20)
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        feedLines(store, count: 20)

        XCTAssertEqual(store.rawText, reference.rawText)
        // recentText's contract is "roughly the last N characters for a
        // live preview" — the exact cut position of a partial first
        // line may shift by one separator because a merged segment's
        // internal newlines count against the character budget while
        // inter-block separators do not. The trailing complete lines
        // must be identical.
        let mergedRecent = store.recentText(characterLimit: 64)
        let referenceRecent = reference.recentText(characterLimit: 64)
        let mergedLines = mergedRecent.split(separator: "\n").map(String.init)
        let referenceLines = referenceRecent.split(separator: "\n").map(String.init)
        XCTAssertEqual(
            mergedLines.suffix(8),
            referenceLines.suffix(8)
        )
        XCTAssertTrue(mergedRecent.hasSuffix("line-19"))
        XCTAssertLessThan(store.blocks.count, reference.blocks.count)
    }

    func testMergeDisabledWhenSegmentSizeZero() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 0
        let store = StreamingDocumentStore(mode: .plainLines)

        feedLines(store, count: 40)

        XCTAssertFalse(store.blocks.contains { $0.text.contains("\n") })
        // Every append ended in a newline, so there is no growing tail.
        XCTAssertEqual(store.blocks.count, 40)
    }

    func testMathLinesModeNeverMerges() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        let store = StreamingDocumentStore(mode: .mathLines)

        feedLines(store, count: 30)

        XCTAssertFalse(store.blocks.contains { $0.text.contains("\n") })
    }

    func testRepeatedMergesKeepSegmentingAsLinesAccumulate() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        let store = StreamingDocumentStore(mode: .plainLines)

        feedLines(store, count: 40)

        let segments = store.blocks.filter { $0.text.contains("\n") }
        XCTAssertGreaterThanOrEqual(segments.count, 2)
        // Reassembling segments + lines must reproduce the exact input
        // (all appends ended in newline, so no tail block exists).
        let reassembled = store.blocks.map(\.text).joined(separator: "\n")
        let expected = (0..<40).map { "line-\($0)" }.joined(separator: "\n")
        XCTAssertEqual(reassembled, expected)
    }
}
