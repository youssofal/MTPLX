import Combine
import Foundation
import os

// MARK: - StreamingDocumentStore

/// Incremental text model for live UI streams.
///
/// The store keeps the full raw text for persistence, but the published UI
/// surface is stable blocks. Appending text only parses newly finalized blocks
/// and the active tail block; existing blocks keep their ids and parsed payloads.
@MainActor
public final class StreamingDocumentStore: ObservableObject {
    public enum Mode: Equatable, Sendable {
        case plainText
        case plainLines
        case mathLines
        case markdown
    }

    @Published public private(set) var blocks: [StreamingDocumentBlock] = []
    public private(set) var revision: Int = 0
    public private(set) var wordCount: Int = 0
    /// Running count of line/block finalizations and segment merges,
    /// available in RELEASE builds so UIStreamPerfProbe can emit a
    /// record for every finalized line (the "poll every single line"
    /// requirement) and correlate stalls with merge events.
    public private(set) var liveFinalizedCount: Int = 0
    public private(set) var liveSegmentMergeCount: Int = 0

    #if DEBUG
    public private(set) var diagnostics = StreamingDocumentDiagnostics()
    #endif

    public let mode: Mode

    private let revisionSubject = PassthroughSubject<Int, Never>()
    private var rawTextStorage: String = ""
    private var tailText: String = ""
    private var nextBlockID = 0
    private var tailBlockID = 0
    private var inCodeFence = false
    private var codeFenceLanguage: String?
    private var lastScalarWasWord = false
    private static let maxLiveLineCharacters = 2_048

    #if DEBUG
    private static let signpostLog = OSLog(
        subsystem: "com.mtplx.app",
        category: "StreamingDocument"
    )
    #endif

    public init(mode: Mode) {
        self.mode = mode
        self.tailBlockID = nextBlockID
        self.nextBlockID += 1
    }

    public var rawText: String { rawTextStorage }
    public var isEmpty: Bool { rawTextStorage.isEmpty }
    public var bottomID: Int? { blocks.last?.id }
    public var revisionPublisher: AnyPublisher<Int, Never> {
        revisionSubject.eraseToAnyPublisher()
    }

    public func recentText(characterLimit: Int) -> String {
        guard characterLimit > 0, !blocks.isEmpty else { return "" }
        var remaining = characterLimit
        var chunks: [String] = []
        for block in blocks.reversed() {
            guard remaining > 0 else { break }
            let suffix = block.text.suffix(remaining)
            chunks.append(String(suffix))
            remaining -= suffix.count
            if suffix.startIndex != block.text.startIndex {
                break
            }
        }
        let separator = mode == .plainLines || mode == .mathLines ? "\n" : ""
        return chunks.reversed().joined(separator: separator)
    }

    public func reset() {
        rawTextStorage = ""
        tailText = ""
        blocks = []
        revision = 0
        wordCount = 0
        liveFinalizedCount = 0
        liveSegmentMergeCount = 0
        nextBlockID = 0
        tailBlockID = nextBlockID
        nextBlockID += 1
        inCodeFence = false
        codeFenceLanguage = nil
        lastScalarWasWord = false
        #if DEBUG
        diagnostics = StreamingDocumentDiagnostics()
        #endif
        revisionSubject.send(revision)
    }

    public func append(_ delta: String) {
        guard !delta.isEmpty else { return }
        let appendStarted = ProcessInfo.processInfo.systemUptime
        if AIMEDiagnostics.isEnabled {
            AIMEDiagnostics.signpost(.documentAppend)
        }
        #if DEBUG
        let signpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(
            .begin,
            log: Self.signpostLog,
            name: "Append",
            signpostID: signpostID,
            "mode=%{public}@ bytes=%{public}d",
            String(describing: mode),
            delta.utf8.count
        )
        diagnostics.appendCount += 1
        defer {
            os_signpost(
                .end,
                log: Self.signpostLog,
                name: "Append",
                signpostID: signpostID
            )
        }
        #endif
        rawTextStorage.append(delta)
        let counter = Self.countWordStarts(in: delta, previousEndedInWord: lastScalarWasWord)
        wordCount += counter.count
        lastScalarWasWord = counter.endedInWord

        switch mode {
        case .plainText:
            appendPlainTextDelta(delta)
        case .plainLines, .mathLines:
            appendLineDelta(delta)
        case .markdown:
            appendMarkdownDelta(delta)
        }
        #if DEBUG
        diagnostics.renderPublicationCount += 1
        os_signpost(
            .event,
            log: Self.signpostLog,
            name: "Publish",
            "mode=%{public}@ revision=%{public}d blocks=%{public}d",
            String(describing: mode),
            revision + 1,
            blocks.count
        )
        #endif
        if AIMEDiagnostics.isEnabled {
            let appendMs = (ProcessInfo.processInfo.systemUptime - appendStarted) * 1000
            let shouldRecord = appendMs >= 4 || AIMEDiagnostics.shouldRecordCadenced(
                "document_append_finished",
                intervalS: 1,
                identity: String(describing: mode)
            )
            guard shouldRecord else {
                advanceRevision()
                return
            }
            AIMEDiagnostics.record(
                "document_append_finished",
                fields: [
                    "mode": .string(String(describing: mode)),
                    "delta_bytes": .int(delta.utf8.count),
                    "blocks_after": .int(blocks.count),
                    "revision_after": .int(revision + 1),
                    "word_count": .int(wordCount),
                    "append_ms": .double(appendMs)
                ]
            )
        }
        advanceRevision()
    }

    private func advanceRevision() {
        revision += 1
        revisionSubject.send(revision)
    }

    // MARK: - Line modes

    private func appendPlainTextDelta(_ delta: String) {
        tailText.append(delta)
        upsertTail(text: tailText, kind: .unfinished, finalized: false)
    }

    private func appendLineDelta(_ delta: String) {
        tailText.append(delta)
        while let newline = tailText.firstIndex(of: "\n") {
            var line = String(tailText[..<newline])
            if line.last == "\r" {
                line.removeLast()
            }
            upsertTail(text: line, finalized: true)
            allocateNewTail()
            tailText = String(tailText[tailText.index(after: newline)...])
        }
        while tailText.count > Self.maxLiveLineCharacters {
            let split = tailText.index(
                tailText.startIndex,
                offsetBy: Self.maxLiveLineCharacters
            )
            let segment = String(tailText[..<split])
            upsertTail(text: segment, finalized: true)
            allocateNewTail()
            tailText = String(tailText[split...])
        }
        coalesceFinalizedLineBlocksIfNeeded()
        upsertTail(text: tailText, finalized: false)
    }

    // MARK: - Line-segment coalescing (2026-07-31 streaming-jank fix)
    //
    // In line modes every finalized line is its own block, so a long
    // code answer accumulates thousands of blocks — and thousands of
    // realized SwiftUI views. Layout cost per flush then grows with
    // answer length, which is exactly the founder-reported "starts
    // smooth, then freezes and vomits words" curve (receipts:
    // outputs/app-frontend-hunt-20260731.md — UI flushes sank from
    // ~10.6/s to ~6/s with 250–800 ms stalls while the engine held a
    // flat 54 tok/s).
    //
    // Once enough OLD finalized lines pile up, fold them into one
    // multi-line segment block: the transcript's realized-view count
    // stays O(answer/segment) instead of O(lines). A fresh window of
    // recent lines is left unmerged so the just-frozen line's promotion
    // to settled markdown never visibly re-wraps. The raw text, word
    // count, and tail behavior are unchanged; `recentText` still walks
    // blocks in order. Fence-safety classification happens at the view
    // layer over block TEXTS, so a merged segment that opens a fence
    // without closing it simply stays on the plain-text path — same
    // rendering as unmerged, thirty-odd times fewer views.
    //
    // `MTPLX_STREAM_SEGMENT_LINES` tunes the segment size (default 32;
    // 0 disables, which is also the A/B baseline arm).
    static let lineSegmentSizeDefault = 32
    nonisolated(unsafe) static var lineSegmentSizeOverrideForTesting: Int?
    private static let lineSegmentSize: Int = {
        guard let raw = ProcessInfo.processInfo.environment["MTPLX_STREAM_SEGMENT_LINES"],
              let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return lineSegmentSizeDefault }
        return min(max(value, 0), 512)
    }()
    private static let lineSegmentFreshWindow = 8

    private var effectiveLineSegmentSize: Int {
        Self.lineSegmentSizeOverrideForTesting ?? Self.lineSegmentSize
    }

    private func coalesceFinalizedLineBlocksIfNeeded() {
        // plainLines only: mathLines blocks carry parsed math runs that
        // must stay per-line, and the other modes never finalize lines.
        guard mode == .plainLines else { return }
        let segmentSize = effectiveLineSegmentSize
        guard segmentSize > 0 else { return }

        // Single-line blocks never contain "\n"; merged segments always
        // do. That distinction is the "already merged" marker, so no
        // block-struct change is needed. Fence lines (```lang / ```)
        // are excluded so a merged segment is always entirely inside or
        // entirely outside a fence — the classifier's fence roles (and
        // therefore live-card grouping + syntax coloring) stay exact
        // for merged content.
        var lineIndexes: [Int] = []
        for (index, block) in blocks.enumerated()
        where block.finalized
            && !block.text.contains("\n")
            && !StreamingMarkdownBlockSafety.isFenceLine(block.text) {
            lineIndexes.append(index)
        }
        guard lineIndexes.count >= segmentSize + Self.lineSegmentFreshWindow else {
            return
        }

        // Merge the FIRST contiguous run of candidates long enough to
        // fold. (A fence line between candidates leaves an index gap;
        // simply taking the first `segmentSize` candidates would fail
        // the contiguity requirement forever once a fence scrolled by,
        // and block count would grow unbounded again.)
        var runStart = 0
        var runLength = 1
        var chosenStart: Int?
        for k in 1..<lineIndexes.count {
            if lineIndexes[k] == lineIndexes[k - 1] + 1 {
                runLength += 1
                if runLength >= segmentSize {
                    chosenStart = runStart
                    break
                }
            } else {
                runStart = k
                runLength = 1
            }
        }
        guard let chosenStart else { return }
        // Leave the newest candidates unmerged (fresh window) so the
        // just-frozen lines never visibly reflow.
        guard chosenStart + segmentSize <= lineIndexes.count - Self.lineSegmentFreshWindow else {
            return
        }
        let first = lineIndexes[chosenStart]
        let last = lineIndexes[chosenStart + segmentSize - 1]

        let merged = blocks[first...last].map(\.text).joined(separator: "\n")
        let mergedBlock = StreamingDocumentBlock(
            id: blocks[first].id,
            text: merged,
            kind: .plain,
            finalized: true
        )
        blocks.replaceSubrange(first...last, with: [mergedBlock])
        liveSegmentMergeCount += 1
        #if DEBUG
        diagnostics.segmentMergeCount += 1
        diagnostics.visibleBlockCount = blocks.count
        #endif
    }

    // MARK: - Markdown mode

    private func appendMarkdownDelta(_ delta: String) {
        tailText.append(delta)

        while true {
            if inCodeFence {
                let searchStart: String.Index
                if tailText.hasPrefix("```"),
                   let newline = tailText.firstIndex(of: "\n") {
                    searchStart = tailText.index(after: newline)
                } else {
                    searchStart = tailText.startIndex
                }
                guard searchStart < tailText.endIndex,
                      let fence = tailText.range(of: "```", range: searchStart..<tailText.endIndex) else {
                    upsertTail(
                        text: tailText,
                        kind: .codeFence(language: codeFenceLanguage, code: codeBody(from: tailText), closed: false),
                        finalized: false
                    )
                    return
                }
                let blockText = String(tailText[..<fence.upperBound])
                upsertTail(
                    text: blockText,
                    kind: .codeFence(language: codeFenceLanguage, code: codeBody(from: blockText), closed: true),
                    finalized: true
                )
                allocateNewTail()
                tailText = String(tailText[fence.upperBound...])
                inCodeFence = false
                codeFenceLanguage = nil
                continue
            }

            if tailText.hasPrefix("```") {
                inCodeFence = true
                codeFenceLanguage = Self.codeFenceLanguage(in: tailText)
                continue
            }

            if let openingFence = Self.openingCodeFenceRange(in: tailText) {
                let prefix = String(tailText[..<openingFence.lowerBound])
                if !prefix.isEmpty {
                    let kind: StreamingDocumentBlockKind = prefix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        ? .plain
                        : .markdown
                    upsertTail(text: prefix, kind: kind, finalized: true)
                    allocateNewTail()
                    tailText = String(tailText[openingFence.lowerBound...])
                    continue
                }
            }

            guard let boundary = tailText.range(of: "\n\n") else {
                upsertTail(text: tailText, kind: .unfinished, finalized: false)
                return
            }

            let blockText = String(tailText[..<boundary.upperBound])
            let kind: StreamingDocumentBlockKind = blockText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? .plain
                : .markdown
            upsertTail(text: blockText, kind: kind, finalized: true)
            allocateNewTail()
            tailText = String(tailText[boundary.upperBound...])
        }
    }

    private func codeBody(from block: String) -> String {
        guard block.hasPrefix("```") else { return block }
        let afterFence = block.index(block.startIndex, offsetBy: 3)
        guard let newline = block[afterFence...].firstIndex(of: "\n") else { return "" }
        var body = String(block[block.index(after: newline)...])
        if body.hasSuffix("```") {
            body.removeLast(3)
        }
        return body
    }

    // MARK: - Block mutation

    private func allocateNewTail() {
        tailBlockID = nextBlockID
        nextBlockID += 1
    }

    private func upsertTail(text: String, finalized: Bool) {
        upsertTail(text: text, kind: kindForLine(text: text, finalized: finalized), finalized: finalized)
    }

    private func upsertTail(
        text: String,
        kind: StreamingDocumentBlockKind,
        finalized: Bool
    ) {
        if text.isEmpty, !finalized {
            blocks.removeAll { $0.id == tailBlockID && !$0.finalized }
            return
        }

        let block = StreamingDocumentBlock(
            id: tailBlockID,
            text: text,
            kind: kind,
            finalized: finalized
        )

        if let index = blocks.firstIndex(where: { $0.id == tailBlockID }) {
            blocks[index] = block
        } else {
            blocks.append(block)
        }
        if finalized {
            liveFinalizedCount += 1
        }
        #if DEBUG
        os_signpost(
            .event,
            log: Self.signpostLog,
            name: "ParseBlock",
            "mode=%{public}@ finalized=%{public}d bytes=%{public}d",
            String(describing: mode),
            finalized ? 1 : 0,
            text.utf8.count
        )
        diagnostics.visibleBlockCount = blocks.count
        if finalized {
            diagnostics.finalizedBlockCount += 1
        } else {
            diagnostics.tailParseCount += 1
        }
        switch kind {
        case .mathRuns:
            diagnostics.mathParseCount += 1
        case .markdown, .codeFence:
            diagnostics.markdownParseCount += 1
        default:
            break
        }
        #endif
    }

    private func kindForLine(text: String, finalized: Bool) -> StreamingDocumentBlockKind {
        switch mode {
        case .plainText:
            return finalized ? .plain : .unfinished
        case .plainLines:
            return finalized ? .plain : .unfinished
        case .mathLines:
            let parseStarted = ProcessInfo.processInfo.systemUptime
            let runs = Self.mathRuns(in: text)
            let parseMs = (ProcessInfo.processInfo.systemUptime - parseStarted) * 1000
            if AIMEDiagnostics.isEnabled {
                let hasMath = runs.contains { $0.kind != .text }
                let shouldRecord = parseMs >= 1 || AIMEDiagnostics.shouldRecordCadenced(
                    "math_parse",
                    intervalS: 2,
                    identity: String(describing: mode)
                )
                if shouldRecord {
                    AIMEDiagnostics.signpost(.mathParse)
                    AIMEDiagnostics.record(
                        "math_parse",
                        fields: [
                            "finalized": .bool(finalized),
                            "line_bytes": .int(text.utf8.count),
                            "run_count": .int(runs.count),
                            "has_math": .bool(hasMath),
                            "parse_ms": .double(parseMs)
                        ]
                    )
                }
            }
            if runs.count == 1, runs[0].kind == .text {
                return finalized ? .plain : .unfinished
            }
            return .mathRuns(runs)
        case .markdown:
            return finalized ? .markdown : .unfinished
        }
    }

    // MARK: - Math parsing

    public nonisolated static func mathRuns(in source: String) -> [StreamingMathRun] {
        guard !source.isEmpty else { return [] }
        var output: [StreamingMathRun] = []
        var scan = source.startIndex
        var textStart = source.startIndex

        func appendRun(_ kind: StreamingMathRun.Kind, _ text: String) {
            guard !text.isEmpty else { return }
            if kind == .text, let last = output.last, last.kind == .text {
                output[output.count - 1].text.append(text)
            } else {
                output.append(StreamingMathRun(id: output.count, kind: kind, text: text))
            }
        }

        func emitText(upTo end: String.Index) {
            guard textStart < end else { return }
            appendRun(.text, String(source[textStart..<end]))
            textStart = end
        }

        while scan < source.endIndex {
            if source[scan] == "\\" {
                let next = source.index(after: scan)
                if next < source.endIndex, source[next] == "(" {
                    emitText(upTo: scan)
                    let bodyStart = source.index(after: next)
                    if let close = source.range(of: "\\)", range: bodyStart..<source.endIndex) {
                        appendRun(.inlineMath, String(source[bodyStart..<close.lowerBound]))
                        scan = close.upperBound
                        textStart = scan
                        continue
                    }
                    appendRun(.text, String(source[scan...]))
                    scan = source.endIndex
                    textStart = scan
                    break
                }
                if next < source.endIndex, source[next] == "[" {
                    emitText(upTo: scan)
                    let bodyStart = source.index(after: next)
                    if let close = source.range(of: "\\]", range: bodyStart..<source.endIndex) {
                        appendRun(.displayMath, String(source[bodyStart..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines))
                        scan = close.upperBound
                        textStart = scan
                        continue
                    }
                    appendRun(.text, String(source[scan...]))
                    scan = source.endIndex
                    textStart = scan
                    break
                }
            }

            guard source[scan] == "$", !Self.isEscapedDollar(at: scan, in: source) else {
                scan = source.index(after: scan)
                continue
            }

            emitText(upTo: scan)
            if source[scan...].hasPrefix("$$") {
                let bodyStart = source.index(scan, offsetBy: 2)
                if let close = Self.nextUnescapedDollarPair(in: source, from: bodyStart) {
                    appendRun(.displayMath, String(source[bodyStart..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines))
                    scan = close.upperBound
                    textStart = scan
                    continue
                }
                appendRun(.text, String(source[scan...]))
                scan = source.endIndex
                textStart = scan
                break
            } else {
                let bodyStart = source.index(after: scan)
                if let close = Self.nextUnescapedDollar(in: source, from: bodyStart) {
                    let body = String(source[bodyStart..<close])
                    if Self.isLikelyInlineMath(body) {
                        appendRun(.inlineMath, body)
                        scan = source.index(after: close)
                        textStart = scan
                        continue
                    }
                    scan = source.index(after: scan)
                    continue
                }
                appendRun(.text, String(source[scan...]))
                scan = source.endIndex
                textStart = scan
                break
            }
        }

        if textStart < source.endIndex {
            appendRun(.text, String(source[textStart...]))
        }

        let parsed = output.isEmpty ? [StreamingMathRun(id: 0, kind: .text, text: source)] : output
        if parsed.count == 1, parsed[0].kind == .text,
           let bareLatex = Self.bareLatexRuns(in: source) {
            return bareLatex
        }
        return parsed
    }

    private nonisolated static func isLikelyInlineMath(_ body: String) -> Bool {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 240, !trimmed.contains("\n") else {
            return false
        }
        if trimmed.range(of: #"\\[A-Za-z]+"#, options: .regularExpression) != nil {
            return true
        }
        let mathSymbols = CharacterSet(charactersIn: "=+-*/^_{}<>≤≥±×÷√∑∫≈≠")
        if trimmed.unicodeScalars.contains(where: { mathSymbols.contains($0) }) {
            return true
        }
        if trimmed.range(
            of: #"^[A-Za-z][A-Za-z0-9]*\([^\)]*\)$"#,
            options: .regularExpression
        ) != nil {
            return true
        }
        // Tuples, coordinates, intervals, bare numbers and short symbol
        // runs: $(0,0,0)$, $13/2$, $P_1$, $x$, $(1, -3/2, 13/2)$. The
        // 2026-07-31 jacobian answer leaked literal $(0,0,0)$ on screen
        // because none of the operator checks above fire for a plain
        // tuple. Guard rails against currency ("paid $5, then $6"):
        // the body must not have edge whitespace and must be composed
        // entirely of math-ish characters.
        guard trimmed == body else { return false }
        let mathish = CharacterSet(charactersIn: "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.()[]|!'")
        let allMathish = trimmed.unicodeScalars.allSatisfy { mathish.contains($0) }
        guard allMathish else { return false }
        let hasDigit = trimmed.contains { $0.isNumber }
        if hasDigit && (trimmed.contains("(") || trimmed.contains(",") || trimmed.count <= 8) {
            return true
        }
        // Single short identifiers ($x$, $xy$, $J$) — but not words.
        if !hasDigit, trimmed.count <= 2, !trimmed.contains(" ") {
            return true
        }
        return false
    }

    private nonisolated static func bareLatexRuns(in source: String) -> [StreamingMathRun]? {
        guard source.contains("\\") else { return nil }
        var spans: [Range<String.Index>] = []
        var index = source.startIndex

        while index < source.endIndex {
            guard source[index] == "\\" else {
                index = source.index(after: index)
                continue
            }

            let commandStart = source.index(after: index)
            guard commandStart < source.endIndex, source[commandStart].isLetter else {
                index = source.index(after: index)
                continue
            }

            var commandEnd = commandStart
            while commandEnd < source.endIndex, source[commandEnd].isLetter {
                commandEnd = source.index(after: commandEnd)
            }

            let command = String(source[commandStart..<commandEnd])
            guard isBareLatexCommand(command) else {
                index = commandEnd
                continue
            }

            let span = bareLatexSpan(around: index, commandEnd: commandEnd, in: source)
            if !span.isEmpty {
                spans.append(span)
                index = span.upperBound
            } else {
                index = commandEnd
            }
        }

        guard !spans.isEmpty else { return nil }
        return mathRuns(fromBareLatexSpans: mergeBareLatexSpans(spans, in: source), source: source)
    }

    private nonisolated static func bareLatexSpan(
        around commandStart: String.Index,
        commandEnd: String.Index,
        in source: String
    ) -> Range<String.Index> {
        var lower = commandStart
        while lower > source.startIndex {
            let previous = source.index(before: lower)
            guard isBareLatexToken(source[previous]) else { break }
            lower = previous
        }

        let prefix = String(source[lower..<commandStart])
        if !prefix.isEmpty, !isLikelyBareLatexPrefix(prefix) {
            lower = commandStart
        }

        var upper = commandEnd
        while upper < source.endIndex {
            if isBareLatexToken(source[upper]) {
                upper = source.index(after: upper)
                continue
            }

            if source[upper].isWhitespace {
                var probe = upper
                while probe < source.endIndex, source[probe].isWhitespace {
                    probe = source.index(after: probe)
                }
                if probe < source.endIndex, isBareLatexContinuationAfterSpace(source[probe]) {
                    upper = probe
                    continue
                }
            }

            break
        }

        while upper > lower {
            let previous = source.index(before: upper)
            if isBareLatexTrailingPunctuation(source[previous]) {
                upper = previous
            } else {
                break
            }
        }

        return lower..<upper
    }

    private nonisolated static func mergeBareLatexSpans(
        _ spans: [Range<String.Index>],
        in source: String
    ) -> [Range<String.Index>] {
        var merged: [Range<String.Index>] = []
        for span in spans.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            guard let last = merged.last else {
                merged.append(span)
                continue
            }
            if span.lowerBound <= last.upperBound || shouldMergeBareLatexGap(source[last.upperBound..<span.lowerBound]) {
                merged[merged.count - 1] = last.lowerBound..<max(last.upperBound, span.upperBound)
            } else {
                merged.append(span)
            }
        }
        return merged
    }

    private nonisolated static func mathRuns(
        fromBareLatexSpans spans: [Range<String.Index>],
        source: String
    ) -> [StreamingMathRun] {
        var output: [StreamingMathRun] = []
        var cursor = source.startIndex

        func append(_ kind: StreamingMathRun.Kind, _ range: Range<String.Index>) {
            guard !range.isEmpty else { return }
            output.append(StreamingMathRun(id: output.count, kind: kind, text: String(source[range])))
        }

        for span in spans {
            append(.text, cursor..<span.lowerBound)
            append(.inlineMath, span)
            cursor = span.upperBound
        }
        append(.text, cursor..<source.endIndex)
        return output
    }

    private nonisolated static func isBareLatexCommand(_ command: String) -> Bool {
        switch command.lowercased() {
        case "le", "leq", "ge", "geq", "lt", "gt",
             "ne", "neq", "equiv", "approx", "sim", "simeq",
             "in", "notin", "subset", "subseteq", "supset", "supseteq",
             "to", "rightarrow", "leftarrow", "implies", "iff",
             "times", "cdot", "div", "pm", "mp",
             "sqrt", "frac", "dfrac", "tfrac", "binom", "choose",
             "sum", "prod", "int", "lim", "max", "min",
             "log", "ln", "sin", "cos", "tan",
             "dots", "ldots", "cdots",
             "alpha", "beta", "gamma", "delta", "epsilon", "theta",
             "lambda", "mu", "pi", "sigma", "phi", "omega",
             "overline", "underline", "vec", "hat", "bar",
             "lfloor", "rfloor", "lceil", "rceil", "left", "right", "mid":
            return true
        default:
            return false
        }
    }

    private nonisolated static func isBareLatexToken(_ character: Character) -> Bool {
        if character.isLetter || character.isNumber {
            return true
        }
        switch character {
        case "\\", "_", "^", "{", "}", "[", "]", "(", ")", "<", ">", "=",
             "+", "-", "*", "/", "|", "!", "&", "%", ",":
            return true
        default:
            return false
        }
    }

    private nonisolated static func isBareLatexContinuationAfterSpace(_ character: Character) -> Bool {
        character == "\\" || character == "{" || character == "[" || character == "(" || character.isNumber
    }

    private nonisolated static func isBareLatexTrailingPunctuation(_ character: Character) -> Bool {
        character == "." || character == "," || character == ";" || character == ":"
    }

    private nonisolated static func shouldMergeBareLatexGap(_ gap: Substring) -> Bool {
        !gap.isEmpty && gap.count <= 2 && gap.allSatisfy(\.isWhitespace)
    }

    private nonisolated static func isLikelyBareLatexPrefix(_ prefix: String) -> Bool {
        if prefix.count <= 3 {
            return true
        }
        return prefix.contains { character in
            character.isNumber || character == "_" || character == "^" || character == "{" ||
                character == "}" || character == "(" || character == ")" || character == "=" ||
                character == "+" || character == "-" || character == "*" || character == "/"
        }
    }

    private nonisolated static func nextUnescapedDollar(in source: String, from start: String.Index) -> String.Index? {
        var index = start
        while index < source.endIndex {
            if source[index] == "$", !isEscapedDollar(at: index, in: source) {
                return index
            }
            index = source.index(after: index)
        }
        return nil
    }

    private nonisolated static func nextUnescapedDollarPair(in source: String, from start: String.Index) -> Range<String.Index>? {
        var index = start
        while index < source.endIndex {
            if source[index...].hasPrefix("$$"), !isEscapedDollar(at: index, in: source) {
                let upper = source.index(index, offsetBy: 2)
                return index..<upper
            }
            index = source.index(after: index)
        }
        return nil
    }

    private nonisolated static func isEscapedDollar(at index: String.Index, in source: String) -> Bool {
        guard index > source.startIndex else { return false }
        var slashCount = 0
        var cursor = source.index(before: index)
        while true {
            guard source[cursor] == "\\" else { break }
            slashCount += 1
            if cursor == source.startIndex { break }
            cursor = source.index(before: cursor)
        }
        return slashCount % 2 == 1
    }

    private static func codeFenceLanguage(in text: String) -> String? {
        guard text.hasPrefix("```") else { return nil }
        let afterFence = text.index(text.startIndex, offsetBy: 3)
        let end = text[afterFence...].firstIndex(of: "\n") ?? text.endIndex
        let raw = text[afterFence..<end].trimmingCharacters(in: .whitespacesAndNewlines)
        return raw.isEmpty ? nil : raw
    }

    private static func openingCodeFenceRange(in text: String) -> Range<String.Index>? {
        var lineStart = text.startIndex
        while lineStart < text.endIndex {
            var probe = lineStart
            while probe < text.endIndex, text[probe] == " " || text[probe] == "\t" {
                probe = text.index(after: probe)
            }
            if text[probe...].hasPrefix("```") {
                return probe..<text.index(probe, offsetBy: 3)
            }
            guard let newline = text[probe...].firstIndex(of: "\n") else {
                break
            }
            lineStart = text.index(after: newline)
        }
        return nil
    }

    // MARK: - Counters

    private static func countWordStarts(
        in delta: String,
        previousEndedInWord: Bool
    ) -> (count: Int, endedInWord: Bool) {
        var count = 0
        var inWord = previousEndedInWord
        for scalar in delta.unicodeScalars {
            if isWordScalar(scalar) {
                if !inWord {
                    count += 1
                    inWord = true
                }
            } else {
                inWord = false
            }
        }
        return (count, inWord)
    }

    private static func isWordScalar(_ scalar: UnicodeScalar) -> Bool {
        !(scalar == " " || scalar == "\n" || scalar == "\t" || scalar == "\r")
    }
}

public struct StreamingDocumentBlock: Identifiable, Equatable, Sendable {
    public let id: Int
    public var text: String
    public var kind: StreamingDocumentBlockKind
    public var finalized: Bool

    public init(
        id: Int,
        text: String,
        kind: StreamingDocumentBlockKind,
        finalized: Bool
    ) {
        self.id = id
        self.text = text
        self.kind = kind
        self.finalized = finalized
    }
}

public enum StreamingDocumentBlockKind: Equatable, Sendable {
    case plain
    case mathRuns([StreamingMathRun])
    case markdown
    case codeFence(language: String?, code: String, closed: Bool)
    case unfinished
}

public struct StreamingMathRun: Identifiable, Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case text
        case inlineMath
        case displayMath
    }

    public let id: Int
    public var kind: Kind
    public var text: String

    public init(id: Int, kind: Kind, text: String) {
        self.id = id
        self.kind = kind
        self.text = text
    }
}

/// Structured content for a DISPLAY math line, so matrices render as
/// real stacked grids with tall delimiters and lone fractions render
/// stacked — instead of the one-line "( a  b ; c  d )" flattening
/// (2026-07-31 founder: "does this look like mathematical notation?").
public enum MathDisplayContent: Equatable, Sendable {
    case plain(String)
    case matrix(prefix: String, open: String, close: String, rows: [[String]], suffix: String)
    case fraction(prefix: String, numerator: String, denominator: String, suffix: String)
}

public enum StreamingMathTextFormatter {

    /// Parse a display-math latex body into structured content. Falls
    /// back to the readable one-liner for anything unrecognized.
    public static func displayContent(from latex: String) -> MathDisplayContent {
        let matrixEnvs: [String: (String, String)] = [
            "pmatrix": ("(", ")"), "bmatrix": ("[", "]"),
            "Bmatrix": ("{", "}"), "vmatrix": ("|", "|"),
            "Vmatrix": ("‖", "‖"), "matrix": ("", ""),
            "smallmatrix": ("(", ")"), "cases": ("{", "")
        ]
        for (env, delims) in matrixEnvs {
            guard let beginRange = latex.range(of: "\\begin{\(env)}"),
                  let endRange = latex.range(
                    of: "\\end{\(env)}",
                    range: beginRange.upperBound..<latex.endIndex
                  )
            else { continue }
            let body = String(latex[beginRange.upperBound..<endRange.lowerBound])
            let rows = body
                .components(separatedBy: "\\\\")
                .map { row in
                    row.components(separatedBy: "&").map {
                        readableText(from: $0).trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                }
                .filter { row in row.contains { !$0.isEmpty } }
            guard !rows.isEmpty else { continue }
            let prefix = readableText(from: String(latex[..<beginRange.lowerBound]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let suffix = readableText(from: String(latex[endRange.upperBound...]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return .matrix(
                prefix: prefix,
                open: delims.0,
                close: delims.1,
                rows: rows,
                suffix: suffix
            )
        }

        // Lone top-level fraction (optionally with an "lhs =" prefix
        // and nothing after): render stacked.
        if let fracRange = latex.range(of: "\\frac"),
           let numerator = bracedGroup(in: latex, from: fracRange.upperBound),
           let denominator = bracedGroup(in: latex, from: numerator.upperBound),
           latex[denominator.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let prefix = readableText(from: String(latex[..<fracRange.lowerBound]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return .fraction(
                prefix: prefix,
                numerator: readableText(from: numerator.body)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                denominator: readableText(from: denominator.body)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                suffix: ""
            )
        }

        return .plain(readableText(from: latex))
    }

    public static func readableText(from latex: String) -> String {
        // No commands: still normalize ^/_ scripts (x^2 → x², f_1 → f₁).
        guard latex.contains("\\") else { return normalizeScripts(in: latex) }
        var output = ""
        var index = latex.startIndex

        while index < latex.endIndex {
            guard latex[index] == "\\" else {
                output.append(latex[index])
                index = latex.index(after: index)
                continue
            }

            let next = latex.index(after: index)
            guard next < latex.endIndex else {
                output.append(latex[index])
                index = next
                continue
            }

            if latex[next] == "{" || latex[next] == "}" {
                output.append(latex[next])
                index = latex.index(after: next)
                continue
            }

            if latex[next] == "," || latex[next] == ";" || latex[next] == ":" || latex[next] == " " {
                output.append(" ")
                index = latex.index(after: next)
                continue
            }

            // \\ is LaTeX's row separator (matrices, cases, aligned):
            // read it as "; " so multi-row structures stay one readable
            // line. \! is negative thin space; \| is the norm bars.
            if latex[next] == "\\" {
                output.append("; ")
                index = latex.index(after: next)
                continue
            }
            if latex[next] == "!" {
                index = latex.index(after: next)
                continue
            }
            if latex[next] == "|" {
                output.append("‖")
                index = latex.index(after: next)
                continue
            }

            guard latex[next].isLetter else {
                output.append(latex[index])
                index = next
                continue
            }

            var commandEnd = next
            while commandEnd < latex.endIndex, latex[commandEnd].isLetter {
                commandEnd = latex.index(after: commandEnd)
            }

            let command = String(latex[next..<commandEnd])
            if isFractionCommand(command),
               let numerator = bracedGroup(in: latex, from: commandEnd),
               let denominator = bracedGroup(in: latex, from: numerator.upperBound) {
                let top = readableText(from: numerator.body).trimmingCharacters(in: .whitespacesAndNewlines)
                let bottom = readableText(from: denominator.body).trimmingCharacters(in: .whitespacesAndNewlines)
                output.append(wrappedFractionPart(top))
                output.append("/")
                output.append(wrappedFractionPart(bottom))
                index = denominator.upperBound
                continue
            }

            if isBinomialCommand(command),
               let topGroup = bracedGroup(in: latex, from: commandEnd),
               let bottomGroup = bracedGroup(in: latex, from: topGroup.upperBound) {
                let top = readableText(from: topGroup.body).trimmingCharacters(in: .whitespacesAndNewlines)
                let bottom = readableText(from: bottomGroup.body).trimmingCharacters(in: .whitespacesAndNewlines)
                output.append("C(")
                output.append(top)
                output.append(", ")
                output.append(bottom)
                output.append(")")
                index = bottomGroup.upperBound
                continue
            }

            // Font/wrapper commands take one braced argument and read
            // as their (recursively readable) body: \mathbb{C} -> ℂ,
            // \text{ if }, \operatorname{Jac}, \vec{v} -> v⃗ …
            // (2026-07-31 founder math repro: the jacobian answer is
            // wall-to-wall \mathbb/\det/\partial and rendered as raw
            // backslash soup before this.)
            if let styled = styledGroupReplacement(
                command: command,
                in: latex,
                argumentStart: commandEnd
            ) {
                output.append(styled.text)
                index = styled.upperBound
                continue
            }

            // \sqrt{...} and \sqrt[n]{...}
            if command == "sqrt" {
                var argStart = commandEnd
                var indexPrefix = ""
                if argStart < latex.endIndex, latex[argStart] == "[" {
                    if let closeBracket = latex[argStart...].firstIndex(of: "]") {
                        indexPrefix = String(latex[latex.index(after: argStart)..<closeBracket])
                        argStart = latex.index(after: closeBracket)
                    }
                }
                if let group = bracedGroup(in: latex, from: argStart) {
                    let body = readableText(from: group.body)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    output.append(indexPrefix)
                    output.append("√")
                    if body.count > 1 {
                        output.append("(\(body))")
                    } else {
                        output.append(body)
                    }
                    index = group.upperBound
                    continue
                }
            }

            // \begin{env} / \end{env}: emit the visual bracket for the
            // environment and let the interior flow through (rows are
            // "; " via \\, columns "  " via &).
            if command == "begin" || command == "end",
               let envGroup = bracedGroup(in: latex, from: commandEnd) {
                output.append(environmentDelimiter(
                    env: envGroup.body,
                    opening: command == "begin"
                ))
                index = envGroup.upperBound
                continue
            }

            if let replacement = replacement(for: command) {
                output.append(replacement)
            } else {
                output.append(String(latex[index..<commandEnd]))
            }
            index = commandEnd
        }

        return normalizeScripts(in: output.replacingOccurrences(of: "&", with: "  "))
    }

    /// One-braced-argument styling/wrapper commands.
    private static func styledGroupReplacement(
        command: String,
        in latex: String,
        argumentStart: String.Index
    ) -> (text: String, upperBound: String.Index)? {
        let accents: [String: Character] = [
            "vec": "\u{20D7}", "hat": "\u{0302}", "bar": "\u{0304}",
            "tilde": "\u{0303}", "dot": "\u{0307}", "ddot": "\u{0308}",
            "overline": "\u{0304}", "widehat": "\u{0302}", "widetilde": "\u{0303}"
        ]
        let passthrough: Set<String> = [
            "mathbf", "mathrm", "mathit", "mathsf", "mathcal", "mathfrak",
            "boldsymbol", "bm", "text", "textbf", "textit", "textrm",
            "texttt", "operatorname", "mbox", "emph"
        ]
        if command == "mathbb" {
            guard let group = bracedGroup(in: latex, from: argumentStart) else { return nil }
            return (doubleStruck(group.body), group.upperBound)
        }
        if passthrough.contains(command) {
            guard let group = bracedGroup(in: latex, from: argumentStart) else { return nil }
            return (readableText(from: group.body), group.upperBound)
        }
        if let combining = accents[command] {
            guard let group = bracedGroup(in: latex, from: argumentStart) else { return nil }
            let body = readableText(from: group.body)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if body.count == 1 {
                return (body + String(combining), group.upperBound)
            }
            return (body, group.upperBound)
        }
        return nil
    }

    /// Double-struck (blackboard bold) letters for \mathbb and the
    /// single-letter shorthand macros (\C, \R …) chat models use.
    private static func doubleStruck(_ body: String) -> String {
        let map: [Character: String] = [
            "C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ",
            "R": "ℝ", "Z": "ℤ", "F": "𝔽", "A": "𝔸", "B": "𝔹",
            "D": "𝔻", "E": "𝔼", "G": "𝔾", "K": "𝕂", "1": "𝟙"
        ]
        return body.map { map[$0] ?? String($0) }.joined()
    }

    private static func environmentDelimiter(env: String, opening: Bool) -> String {
        let name = env.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "*", with: "")
        switch name {
        case "pmatrix": return opening ? "(" : ")"
        case "bmatrix": return opening ? "[" : "]"
        case "Bmatrix": return opening ? "{" : "}"
        case "vmatrix", "Vmatrix": return "|"
        case "cases": return opening ? "{ " : ""
        default: return ""
        }
    }

    private static func replacement(for command: String) -> String? {
        // Case-sensitive entries first: greek capitals are distinct
        // commands (\Sigma ≠ \sigma — the lowercased switch below used
        // to fold them together), and the single-letter blackboard
        // shorthands (\C, \R …) that chat models emit for number sets.
        switch command {
        case "C", "R", "N", "Z", "Q", "H", "F":
            return doubleStruck(command)
        case "Gamma": return "Γ"
        case "Delta": return "Δ"
        case "Theta": return "Θ"
        case "Lambda": return "Λ"
        case "Xi": return "Ξ"
        case "Pi": return "Π"
        case "Sigma": return "Σ"
        case "Upsilon": return "Υ"
        case "Phi": return "Φ"
        case "Psi": return "Ψ"
        case "Omega": return "Ω"
        case "Rightarrow", "Longrightarrow": return "⇒"
        case "Leftarrow", "Longleftarrow": return "⇐"
        case "Leftrightarrow", "Longleftrightarrow": return "⇔"
        case "Re": return "ℜ"
        case "Im": return "ℑ"
        default: break
        }
        switch command.lowercased() {
        case "le", "leq": return "≤"
        case "ge", "geq": return "≥"
        case "ne", "neq": return "≠"
        case "approx": return "≈"
        case "equiv": return "≡"
        case "in": return "∈"
        case "notin": return "∉"
        case "subset": return "⊂"
        case "subseteq": return "⊆"
        case "supset": return "⊃"
        case "supseteq": return "⊇"
        case "times": return "×"
        case "cdot": return "·"
        case "div": return "÷"
        case "pm": return "±"
        case "mp": return "∓"
        case "to", "rightarrow": return "→"
        case "leftarrow": return "←"
        case "implies": return "⇒"
        case "iff": return "↔"
        case "dots", "ldots", "cdots": return "…"
        case "sqrt": return "√"
        case "sum": return "Σ"
        case "prod": return "Π"
        case "int": return "∫"
        case "lim": return "lim"
        case "left", "right": return ""
        case "lfloor": return "⌊"
        case "rfloor": return "⌋"
        case "lceil": return "⌈"
        case "rceil": return "⌉"
        case "mid": return "|"
        case "max": return "max"
        case "min": return "min"
        case "log": return "log"
        case "ln": return "ln"
        case "sin": return "sin"
        case "cos": return "cos"
        case "tan": return "tan"
        case "alpha": return "α"
        case "beta": return "β"
        case "gamma": return "γ"
        case "delta": return "δ"
        case "epsilon": return "ε"
        case "theta": return "θ"
        case "lambda": return "λ"
        case "mu": return "μ"
        case "pi": return "π"
        case "sigma": return "σ"
        case "phi": return "φ"
        case "omega": return "ω"
        case "infty", "infin": return "∞"
        case "partial": return "∂"
        case "nabla": return "∇"
        case "cup": return "∪"
        case "cap": return "∩"
        case "forall": return "∀"
        case "exists": return "∃"
        case "det": return "det"
        case "dim": return "dim"
        case "ker": return "ker"
        case "deg": return "deg"
        case "gcd": return "gcd"
        case "arg": return "arg"
        case "exp": return "exp"
        case "mod", "bmod", "pmod": return "mod "
        case "mapsto", "longmapsto": return "↦"
        case "longrightarrow": return "→"
        case "longleftarrow": return "←"
        case "langle": return "⟨"
        case "rangle": return "⟩"
        case "circ": return "∘"
        case "bullet": return "•"
        case "star": return "⋆"
        case "oplus": return "⊕"
        case "otimes": return "⊗"
        case "emptyset", "varnothing": return "∅"
        case "setminus": return "∖"
        case "prime": return "′"
        case "degree": return "°"
        case "angle": return "∠"
        case "perp": return "⊥"
        case "parallel": return "∥"
        case "sim": return "∼"
        case "simeq": return "≃"
        case "cong": return "≅"
        case "propto": return "∝"
        case "because": return "∵"
        case "therefore": return "∴"
        case "neg", "lnot": return "¬"
        case "land", "wedge": return "∧"
        case "lor", "vee": return "∨"
        case "oint": return "∮"
        case "iint": return "∬"
        case "nmid": return "∤"
        case "vert": return "|"
        case "colon": return ":"
        case "quad": return "  "
        case "qquad": return "    "
        case "ell": return "ℓ"
        case "hbar": return "ℏ"
        case "aleph": return "ℵ"
        case "eta": return "η"
        case "zeta": return "ζ"
        case "kappa": return "κ"
        case "rho", "varrho": return "ρ"
        case "tau": return "τ"
        case "xi": return "ξ"
        case "chi": return "χ"
        case "psi": return "ψ"
        case "nu": return "ν"
        case "iota": return "ι"
        case "upsilon": return "υ"
        case "varepsilon": return "ε"
        case "varphi": return "φ"
        case "vartheta": return "ϑ"
        case "varsigma": return "ς"
        case "cot": return "cot"
        case "sec": return "sec"
        case "csc": return "csc"
        case "sinh": return "sinh"
        case "cosh": return "cosh"
        case "tanh": return "tanh"
        case "arcsin": return "arcsin"
        case "arccos": return "arccos"
        case "arctan": return "arctan"
        case "sup": return "sup"
        case "inf": return "inf"
        default: return nil
        }
    }

    private static func isBinomialCommand(_ command: String) -> Bool {
        switch command.lowercased() {
        case "binom":
            return true
        default:
            return false
        }
    }

    private static func isFractionCommand(_ command: String) -> Bool {
        switch command.lowercased() {
        case "frac", "dfrac", "tfrac":
            return true
        default:
            return false
        }
    }

    private static func bracedGroup(
        in source: String,
        from start: String.Index
    ) -> (body: String, upperBound: String.Index)? {
        guard start < source.endIndex, source[start] == "{" else { return nil }
        var depth = 0
        var index = start
        let bodyStart = source.index(after: start)

        while index < source.endIndex {
            if source[index] == "{" {
                depth += 1
            } else if source[index] == "}" {
                depth -= 1
                if depth == 0 {
                    return (String(source[bodyStart..<index]), source.index(after: index))
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func wrappedFractionPart(_ value: String) -> String {
        guard value.count > 1 else { return value }
        if value.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" || $0 == "^" }) {
            return value
        }
        return "(\(value))"
    }

    /// Converts `^…` / `_…` into REAL Unicode super/subscripts wherever
    /// the glyphs exist (x^2 → x², f_1 → f₁, ℂ^3 → ℂ³, x^{13} → x¹³),
    /// falling back to the caret/underscore form only for unmappable
    /// bodies. 2026-07-31 founder: "does this look like mathematical
    /// notation?" — literal ^ and _ do not.
    private static let superscriptMap: [Character: Character] = [
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
        "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
        "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ",
        "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
        "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
        "z": "ᶻ", "T": "ᵀ"
    ]

    private static let subscriptMap: [Character: Character] = [
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
        "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎",
        "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
        "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
        "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ"
    ]

    private static func mapped(_ body: String, via table: [Character: Character]) -> String? {
        guard !body.isEmpty else { return nil }
        var out = ""
        for ch in body {
            guard let m = table[ch] else { return nil }
            out.append(m)
        }
        return out
    }

    private static func normalizeScripts(in source: String) -> String {
        var output = ""
        var index = source.startIndex

        while index < source.endIndex {
            let character = source[index]
            guard character == "_" || character == "^" else {
                output.append(character)
                index = source.index(after: index)
                continue
            }
            let table = character == "^" ? superscriptMap : subscriptMap

            let next = source.index(after: index)
            // Braced script: ^{13}, _{n+1}
            if next < source.endIndex, source[next] == "{",
               let group = bracedGroup(in: source, from: next) {
                let body = readableText(from: group.body)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if let converted = mapped(body, via: table) {
                    output.append(converted)
                } else if character == "_", body.count > 1 {
                    output.append("_(")
                    output.append(body)
                    output.append(")")
                } else {
                    output.append(character)
                    output.append(body)
                }
                index = group.upperBound
                continue
            }

            // Unbraced script: consume a digit run (x^25 means x²⁵ in
            // chat-model output) or a single letter (f_i).
            if next < source.endIndex {
                var end = next
                if source[next].isNumber || source[next] == "-" || source[next] == "−" {
                    end = source.index(after: next)
                    while end < source.endIndex, source[end].isNumber {
                        end = source.index(after: end)
                    }
                } else if source[next].isLetter {
                    end = source.index(after: next)
                }
                if end > next {
                    let body = String(source[next..<end])
                    if let converted = mapped(body, via: table) {
                        output.append(converted)
                        index = end
                        continue
                    }
                }
            }

            output.append(character)
            index = next
        }

        return output
    }
}

#if DEBUG
public struct StreamingDocumentDiagnostics: Equatable, Sendable {
    public var appendCount: Int = 0
    public var finalizedBlockCount: Int = 0
    public var tailParseCount: Int = 0
    public var markdownParseCount: Int = 0
    public var mathParseCount: Int = 0
    public var visibleBlockCount: Int = 0
    public var renderPublicationCount: Int = 0
    public var segmentMergeCount: Int = 0

    public init() {}
}
#endif
