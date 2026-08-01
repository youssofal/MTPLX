import AppKit
import XCTest
@testable import MTPLXAppCore

// MARK: - Math readable-text coverage (2026-07-31 founder repro:
// jacobian answer was wall-to-wall \mathbb/\det/\partial soup)

final class MathReadableTextTests: XCTestCase {
    func testBlackboardAndArrow() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"F: \mathbb{C}^3 \to \mathbb{C}^3"#),
            "F: ℂ³ → ℂ³"
        )
    }

    func testSingleLetterBlackboardShorthand() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\C^3\to \C^3"#),
            "ℂ³→ ℂ³"
        )
    }

    func testUnicodeScripts() {
        XCTAssertEqual(StreamingMathTextFormatter.readableText(from: "x^2 + y^{13}"), "x² + y¹³")
        XCTAssertEqual(StreamingMathTextFormatter.readableText(from: "f_1 + a_{n}"), "f₁ + aₙ")
        XCTAssertEqual(StreamingMathTextFormatter.readableText(from: "x^25"), "x²⁵")
        XCTAssertEqual(StreamingMathTextFormatter.readableText(from: "e^{x+y}"), "eˣ⁺ʸ")
        // Unmappable script bodies (no superscript q glyph) keep the caret form.
        XCTAssertEqual(StreamingMathTextFormatter.readableText(from: "e^{q+1}"), "e^q+1")
    }

    func testInlineTupleDollarsConvert() {
        let runs = StreamingDocumentStore.mathRuns(in: "at the origin $(0,0,0)$ provides")
        XCTAssertTrue(runs.contains { $0.kind == .inlineMath && $0.text == "(0,0,0)" }, "\(runs)")
    }

    func testCurrencyDollarsStayText() {
        let runs = StreamingDocumentStore.mathRuns(in: "he paid $5, then $6.")
        XCTAssertFalse(runs.contains { $0.kind != .text }, "\(runs)")
    }

    func testDisplayMatrixStructure() {
        let content = StreamingMathTextFormatter.displayContent(
            from: #"J_F(0,0,0) = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ 2 & 0 & 0 \end{pmatrix}"#
        )
        guard case .matrix(let prefix, let open, let close, let rows, _) = content else {
            return XCTFail("expected matrix, got \(content)")
        }
        XCTAssertEqual(open, "(")
        XCTAssertEqual(close, ")")
        XCTAssertEqual(rows.count, 3)
        XCTAssertEqual(rows[0], ["0", "0", "1"])
        XCTAssertEqual(rows[2], ["2", "0", "0"])
        XCTAssertTrue(prefix.contains("J"), prefix)
    }

    func testDisplayLoneFractionStacks() {
        let content = StreamingMathTextFormatter.displayContent(
            from: #"\frac{x+1}{y-2}"#
        )
        guard case .fraction(_, let numerator, let denominator, _) = content else {
            return XCTFail("expected fraction, got \(content)")
        }
        XCTAssertEqual(numerator, "x+1")
        XCTAssertEqual(denominator, "y-2")
    }

    func testOperatorNames() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\det J = -2"#),
            "det J = -2"
        )
    }

    func testGreekCapitalsAreCaseSensitive() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\Sigma \neq \sigma"#),
            "Σ ≠ σ"
        )
    }

    func testSqrtBraced() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"\sqrt{x+1}"#),
            "√(x+1)"
        )
    }

    func testPmatrixRowsReadable() {
        let out = StreamingMathTextFormatter.readableText(
            from: #"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"#
        )
        XCTAssertTrue(out.contains("("), out)
        XCTAssertTrue(out.contains(")"), out)
        XCTAssertTrue(out.contains(";"), out)
        XCTAssertFalse(out.contains("begin"), out)
        XCTAssertFalse(out.contains("&"), out)
    }

    func testTextWrapperPassesThrough() {
        XCTAssertEqual(
            StreamingMathTextFormatter.readableText(from: #"x \text{ if } y"#),
            "x  if  y"
        )
    }

    func testPartialFractionStillWorks() {
        let out = StreamingMathTextFormatter.readableText(
            from: #"\frac{\partial f}{\partial x}"#
        )
        XCTAssertTrue(out.contains("∂"), out)
        XCTAssertTrue(out.contains("/"), out)
        XCTAssertFalse(out.contains("\\partial"), out)
    }

    func testNoBackslashSoupOnRepro() {
        let out = StreamingMathTextFormatter.readableText(
            from: #"J = \det(\frac{\partial(f_1, f_2, f_3)}{\partial(x, y, z)})"#
        )
        XCTAssertFalse(out.contains("\\det"), out)
        XCTAssertFalse(out.contains("\\frac"), out)
        XCTAssertFalse(out.contains("\\partial"), out)
    }
}

// MARK: - Syntax highlighter

final class CodeHighlighterTests: XCTestCase {
    private func colors(in attributed: NSAttributedString) -> Set<NSColor> {
        var found: Set<NSColor> = []
        attributed.enumerateAttribute(
            .foregroundColor,
            in: NSRange(location: 0, length: attributed.length)
        ) { value, _, _ in
            if let color = value as? NSColor { found.insert(color) }
        }
        return found
    }

    func testPythonKeywordStringCommentNumber() {
        let (line, endState) = MTPLXCodeHighlighter.highlightLine(
            "def go(x=3):  # start \"quoted\"",
            language: .python,
            state: .none
        )
        XCTAssertEqual(line.string, "def go(x=3):  # start \"quoted\"")
        XCTAssertEqual(endState, .none)
        let palette = MTPLXCodeHighlighter.Palette.dark
        let used = colors(in: line)
        XCTAssertTrue(used.contains(palette.keyword))
        XCTAssertTrue(used.contains(palette.function))
        XCTAssertTrue(used.contains(palette.number))
        XCTAssertTrue(used.contains(palette.comment))
    }

    func testTripleStringCarriesAcrossLines() {
        let open = MTPLXCodeHighlighter.highlightLine(
            "doc = \"\"\"start",
            language: .python,
            state: .none
        )
        XCTAssertEqual(open.endState, .tripleString("\""))
        let middle = MTPLXCodeHighlighter.highlightLine(
            "still inside",
            language: .python,
            state: open.endState
        )
        XCTAssertEqual(middle.endState, .tripleString("\""))
        let palette = MTPLXCodeHighlighter.Palette.dark
        XCTAssertEqual(colors(in: middle.line), [palette.string])
        let close = MTPLXCodeHighlighter.highlightLine(
            "end\"\"\" + 1",
            language: .python,
            state: middle.endState
        )
        XCTAssertEqual(close.endState, .none)
    }

    func testHighlightPreservesExactText() {
        let code = "class Bird:\n    def __init__(self):\n        self.y = 0.5  # center\n"
        let attributed = MTPLXCodeHighlighter.highlightCode(code, language: .python)
        XCTAssertEqual(attributed.string, code)
    }

    func testSegmentEndStateThreading() {
        let seg = "a = '''x\nstill\nend''' + 1"
        let end = MTPLXCodeHighlighter.highlightSegmentEndState(
            seg, language: .python, state: .none
        )
        XCTAssertEqual(end, .none)
        let openEnd = MTPLXCodeHighlighter.highlightSegmentEndState(
            "a = '''x\nstill", language: .python, state: .none
        )
        XCTAssertEqual(openEnd, .tripleString("'"))
    }
}

// MARK: - Fence roles + fence-aware coalescing

final class FenceRoleTests: XCTestCase {
    func testRolesForFenceRun() {
        let texts = ["intro", "```python", "x = 1", "y = 2", "```", "outro"]
        let classification = StreamingMarkdownBlockSafety.classifyRoles(texts)
        XCTAssertEqual(classification.fenceRoles[0], .none)
        XCTAssertEqual(classification.fenceRoles[1], .open(language: "python"))
        XCTAssertEqual(classification.fenceRoles[2], .interior)
        XCTAssertEqual(classification.fenceRoles[3], .interior)
        XCTAssertEqual(classification.fenceRoles[4], .close)
        XCTAssertEqual(classification.fenceRoles[5], .none)
        // Safety flags unchanged relative to classify()
        XCTAssertEqual(
            classification.settledSafe,
            StreamingMarkdownBlockSafety.classify(texts)
        )
    }

    func testMergedInteriorSegmentKeepsInteriorRole() {
        let texts = ["```py", "a = 1\nb = 2\nc = 3", "d = 4", "```"]
        let classification = StreamingMarkdownBlockSafety.classifyRoles(texts)
        XCTAssertEqual(classification.fenceRoles[1], .interior)
        XCTAssertEqual(classification.fenceRoles[2], .interior)
        XCTAssertEqual(classification.fenceRoles[3], .close)
    }

    @MainActor
    func testCoalescingNeverMergesAcrossFenceLines() {
        StreamingDocumentStore.lineSegmentSizeOverrideForTesting = 4
        defer { StreamingDocumentStore.lineSegmentSizeOverrideForTesting = nil }
        let store = StreamingDocumentStore(mode: .plainLines)
        // 3 prose lines, a fence open, then plenty of code lines.
        store.append("p1\np2\np3\n```python\n")
        for i in 0..<20 {
            store.append("code\(i)\n")
        }
        // Any merged (multi-line) block must not contain a fence line,
        // and the ```python line must survive as its own block.
        let merged = store.blocks.filter { $0.text.contains("\n") }
        for block in merged {
            XCTAssertFalse(block.text.contains("```"), block.text)
        }
        XCTAssertTrue(store.blocks.contains { $0.text == "```python" })
        // Interior lines did coalesce (the contiguous-run path works).
        XCTAssertTrue(merged.contains { $0.text.hasPrefix("code") })
    }
}
