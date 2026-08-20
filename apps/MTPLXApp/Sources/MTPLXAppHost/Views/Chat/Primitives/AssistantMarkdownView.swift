import AppKit
import Combine
import SwiftUI
import MarkdownUI
import MTPLXAppCore

// MARK: - AssistantMarkdownView
//
// Markdown renderer for assistant bubbles. Settled bubbles render fully;
// the live streaming path promotes frozen fence-safe blocks to the same
// settled pipeline (rendered once, cached by Equatable text) while the
// growing tail stays plain text — markdown during streaming with no
// per-token parsing cost.

struct AssistantMarkdownView: View {
    let content: String
    let isStreaming: Bool
    /// Performance mode (founder contract 2026-07-31): when the
    /// performance lock is on, markdown and syntax coloring are OFF —
    /// everything renders as plain text.
    let plainTextOnly: Bool

    init(_ content: String, isStreaming: Bool = false, plainTextOnly: Bool = false) {
        self.content = content
        self.isStreaming = isStreaming
        self.plainTextOnly = plainTextOnly
    }

    var body: some View {
        if isStreaming || plainTextOnly {
            StreamingPlainTextView(text: content)
        } else {
            SettledAssistantMarkdownView(content: content)
        }
    }
}

private struct SettledAssistantMarkdownView: View {
    let content: String

    private var blocks: [SettledMarkdownBlock] {
        CachedSettledMarkdownBlocks.blocks(for: content)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(blocks) { block in
                switch block.kind {
                case .prose(let text):
                    if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        AssistantProseMarkdownView(text: text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                case .code(let language, let code, let lineCount):
                    AssistantCodeBlockView(language: language, code: code, lineCount: lineCount)
                        .padding(.vertical, 4)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct AssistantProseMarkdownView: View {
    let text: String

    private var lines: [AssistantProseLine] {
        AssistantProseLine.parse(text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(lines) { line in
                lineView(line)
            }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func lineView(_ line: AssistantProseLine) -> some View {
        switch line.kind {
        case .blank:
            Color.clear
                .frame(height: 2)
                .accessibilityHidden(true)
        case .heading(let level, let text):
            Text(Self.inlineAttributed(text))
                .font(Self.headingFont(level: level))
                .foregroundStyle(Brand.typeHi)
                .padding(.top, level <= 2 ? 8 : 5)
                .padding(.bottom, 1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .bullet(let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("•")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 12, alignment: .trailing)
                Text(Self.inlineAttributed(text))
                    .font(.system(size: 14))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .ordered(let marker, let text):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(marker)
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 28, alignment: .trailing)
                Text(Self.inlineAttributed(text))
                    .font(.system(size: 14))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case .quote(let text):
            Text(Self.inlineAttributed(text))
                .font(.system(size: 14))
                .foregroundStyle(Brand.typeSecondary)
                .padding(.leading, 12)
                .overlay(alignment: .leading) {
                    Rectangle()
                        .fill(Brand.separatorStrong)
                        .frame(width: 2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .paragraph(let text):
            Text(Self.inlineAttributed(text))
                .font(.system(size: 14))
                .foregroundStyle(Brand.typeHi)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        case .math(let latex):
            // Display equations render structured (matrices as stacked
            // grids with tall delimiters, lone fractions stacked) with
            // readable-math fallback. The Jun-6 perf rewrite dropped
            // the math path and chat regressed to raw $$...$$ (QA-108);
            // the 2026-07-31 rework killed the one-line flattening.
            MathDisplayLineView(latex: latex)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical, 4)
        case .table(let rows, let hasHeader):
            AssistantTableView(rows: rows, hasHeader: hasHeader)
                .padding(.vertical, 4)
        }
    }

    private static func headingFont(level: Int) -> Font {
        switch level {
        case 1: .system(size: 21, weight: .heavy)
        case 2: .system(size: 17, weight: .heavy)
        case 3: .system(size: 15, weight: .bold)
        default: .system(size: 14, weight: .bold)
        }
    }

    private static func inlineAttributed(_ text: String) -> AttributedString {
        let readable = Self.withReadableInlineMath(text)
        return (try? AttributedString(
            markdown: readable,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .inlineOnlyPreservingWhitespace
            )
        )) ?? AttributedString(readable)
    }

    /// Replaces inline LaTeX spans (`$...$`, `\(...\)`, and any `$$...$$`
    /// embedded mid-line) with the readable math text the AIME surface
    /// uses, before inline-markdown attribution. Pure text-in/text-out so
    /// the settled-prose perf path stays line-based (QA-108).
    private static func withReadableInlineMath(_ text: String) -> String {
        guard text.contains("$") || text.contains("\\(") else { return text }
        let runs = StreamingDocumentStore.mathRuns(in: text)
        guard runs.contains(where: { $0.kind != .text }) else { return text }
        return runs.map { run in
            run.kind == .text
                ? run.text
                : StreamingMathTextFormatter.readableText(from: run.text)
        }.joined()
    }
}

private struct AssistantProseLine: Identifiable, Equatable {
    enum Kind: Equatable {
        case blank
        case heading(level: Int, text: String)
        case bullet(String)
        case ordered(marker: String, text: String)
        case quote(String)
        case paragraph(String)
        case math(String)
        case table(rows: [[String]], hasHeader: Bool)
    }

    let id: Int
    let kind: Kind

    static func parse(_ source: String) -> [AssistantProseLine] {
        let rawLines = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var lines: [AssistantProseLine] = []
        var index = 0
        var cursor = 0
        while cursor < rawLines.count {
            let trimmed = rawLines[cursor].trimmingCharacters(in: .whitespaces)
            // Pipe tables: a run of |-prefixed lines. ("markdown doesn't
            // even work" — the settled flappy answer showed raw
            // | Feature | Details | pipes, 2026-07-31.)
            if trimmed.hasPrefix("|"), trimmed.hasSuffix("|"), trimmed.count > 2 {
                var tableLines: [String] = []
                var lookahead = cursor
                while lookahead < rawLines.count {
                    let candidate = rawLines[lookahead].trimmingCharacters(in: .whitespaces)
                    guard candidate.hasPrefix("|") else { break }
                    tableLines.append(candidate)
                    lookahead += 1
                }
                if tableLines.count >= 2,
                   let table = parseTable(tableLines) {
                    lines.append(AssistantProseLine(id: index, kind: table))
                    index += 1
                    cursor = lookahead
                    continue
                }
            }
            // Block-form display math: a bare $$ (or \[) line opens a
            // block that runs to the matching closer; the interior is
            // one centered math row (QA-108).
            if trimmed == "$$" || trimmed == "\\[" {
                let closer = trimmed == "$$" ? "$$" : "\\]"
                var body: [String] = []
                var lookahead = cursor + 1
                while lookahead < rawLines.count,
                      rawLines[lookahead].trimmingCharacters(in: .whitespaces) != closer
                {
                    body.append(rawLines[lookahead])
                    lookahead += 1
                }
                if lookahead < rawLines.count, !body.isEmpty {
                    lines.append(AssistantProseLine(
                        id: index,
                        kind: .math(body.joined(separator: " ").trimmingCharacters(in: .whitespaces))
                    ))
                    index += 1
                    cursor = lookahead + 1
                    continue
                }
            }
            lines.append(AssistantProseLine(id: index, kind: Self.kind(for: rawLines[cursor])))
            index += 1
            cursor += 1
        }
        return lines
    }

    private static func kind(for rawLine: String) -> Kind {
        let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return .blank }
        if let math = displayMath(in: trimmed) {
            return .math(math)
        }
        if let heading = heading(in: trimmed) {
            return heading
        }
        if let bullet = bullet(in: trimmed) {
            return .bullet(bullet)
        }
        if let ordered = ordered(in: trimmed) {
            return ordered
        }
        if trimmed.hasPrefix(">") {
            let quote = trimmed
                .dropFirst()
                .trimmingCharacters(in: .whitespaces)
            return .quote(quote)
        }
        return .paragraph(trimmed)
    }

    /// Single-line display math: `$$...$$` or `\[...\]` filling the
    /// whole line (the shape chat models emit most).
    private static func displayMath(in line: String) -> String? {
        for (open, close) in [("$$", "$$"), ("\\[", "\\]")] {
            if line.hasPrefix(open), line.hasSuffix(close),
               line.count > open.count + close.count
            {
                let body = line
                    .dropFirst(open.count)
                    .dropLast(close.count)
                    .trimmingCharacters(in: .whitespaces)
                if !body.isEmpty { return body }
            }
        }
        return nil
    }

    private static func heading(in line: String) -> Kind? {
        var cursor = line.startIndex
        var level = 0
        while cursor < line.endIndex, line[cursor] == "#", level < 6 {
            level += 1
            cursor = line.index(after: cursor)
        }
        guard level > 0, cursor < line.endIndex, line[cursor].isWhitespace else {
            return nil
        }
        let text = line[cursor...]
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "#").union(.whitespaces))
        return text.isEmpty ? nil : .heading(level: level, text: text)
    }

    private static func bullet(in line: String) -> String? {
        guard let marker = line.first,
              marker == "*" || marker == "-" || marker == "+",
              line.count >= 2
        else { return nil }
        let bodyStart = line.index(after: line.startIndex)
        guard line[bodyStart].isWhitespace else { return nil }
        let text = line[bodyStart...].trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : text
    }

    private static func ordered(in line: String) -> Kind? {
        var cursor = line.startIndex
        while cursor < line.endIndex, line[cursor].isNumber {
            cursor = line.index(after: cursor)
        }
        guard cursor > line.startIndex,
              cursor < line.endIndex,
              line[cursor] == "." || line[cursor] == ")"
        else { return nil }
        let markerEnd = line.index(after: cursor)
        guard markerEnd < line.endIndex, line[markerEnd].isWhitespace else { return nil }
        let marker = String(line[line.startIndex...cursor])
        let text = line[markerEnd...].trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : .ordered(marker: marker, text: text)
    }

    /// `| a | b |` lines → rows of trimmed cells. The `|---|---|`
    /// separator row is dropped and marks the first row as a header.
    private static func parseTable(_ tableLines: [String]) -> Kind? {
        func cells(of line: String) -> [String] {
            var body = line
            if body.hasPrefix("|") { body.removeFirst() }
            if body.hasSuffix("|") { body.removeLast() }
            return body
                .split(separator: "|", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces) }
        }
        func isSeparatorRow(_ row: [String]) -> Bool {
            !row.isEmpty && row.allSatisfy { cell in
                !cell.isEmpty && cell.allSatisfy { $0 == "-" || $0 == ":" }
            }
        }

        var rows = tableLines.map(cells)
        var hasHeader = false
        if rows.count >= 2, isSeparatorRow(rows[1]) {
            hasHeader = true
            rows.remove(at: 1)
        }
        rows.removeAll(where: isSeparatorRow)
        guard !rows.isEmpty, rows.contains(where: { $0.count >= 2 }) else {
            return nil
        }
        let columns = rows.map(\.count).max() ?? 0
        let padded = rows.map { row -> [String] in
            row.count < columns
                ? row + Array(repeating: "", count: columns - row.count)
                : row
        }
        return .table(rows: padded, hasHeader: hasHeader)
    }
}

// MARK: - MathDisplayLineView

/// One display-math line. Matrices lay out as a Grid between tall thin
/// delimiters; a lone \frac stacks numerator over denominator; anything
/// else falls back to the readable one-liner. Parsed once per settled
/// line (the settled pipeline caches by block text).
private struct MathDisplayLineView: View {
    let latex: String

    var body: some View {
        switch StreamingMathTextFormatter.displayContent(from: latex) {
        case .plain(let text):
            Text(text)
                .font(.system(size: 15, weight: .medium, design: .serif))
                .foregroundStyle(Brand.typeHi)
        case .matrix(let prefix, let open, let close, let rows, let suffix):
            HStack(alignment: .center, spacing: 6) {
                if !prefix.isEmpty {
                    Text(prefix)
                        .font(.system(size: 15, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                }
                if !open.isEmpty {
                    Text(open)
                        .font(.system(size: delimiterSize(for: rows.count), weight: .ultraLight))
                        .foregroundStyle(Brand.typeHi)
                }
                Grid(horizontalSpacing: 14, verticalSpacing: 4) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                Text(cell)
                                    .font(.system(size: 15, weight: .medium, design: .serif))
                                    .foregroundStyle(Brand.typeHi)
                            }
                        }
                    }
                }
                if !close.isEmpty {
                    Text(close)
                        .font(.system(size: delimiterSize(for: rows.count), weight: .ultraLight))
                        .foregroundStyle(Brand.typeHi)
                }
                if !suffix.isEmpty {
                    Text(suffix)
                        .font(.system(size: 15, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                }
            }
        case .fraction(let prefix, let numerator, let denominator, let suffix):
            HStack(alignment: .center, spacing: 8) {
                if !prefix.isEmpty {
                    Text(prefix)
                        .font(.system(size: 15, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                }
                VStack(spacing: 3) {
                    Text(numerator)
                        .font(.system(size: 14, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                    Rectangle()
                        .fill(Brand.typeHi)
                        .frame(height: 1)
                    Text(denominator)
                        .font(.system(size: 14, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                }
                .fixedSize()
                if !suffix.isEmpty {
                    Text(suffix)
                        .font(.system(size: 15, weight: .medium, design: .serif))
                        .foregroundStyle(Brand.typeHi)
                }
            }
        }
    }

    private func delimiterSize(for rowCount: Int) -> CGFloat {
        min(72, CGFloat(max(1, rowCount)) * 22)
    }
}

// MARK: - AssistantTableView

/// Settled-transcript pipe table. Parsed once per prose block (cached
/// with the block), laid out with Grid — no live-streaming cost, since
/// tables only render through the settled pipeline.
private struct AssistantTableView: View {
    let rows: [[String]]
    let hasHeader: Bool

    var body: some View {
        // NOT Grid, NOT a horizontal ScrollView. Both shipped smushed
        // rows (2026-08-17 field bug, verified live twice): Grid sizes
        // rows before flexible wrapping cells resolve their heights —
        // cells then draw their full wrapped height over a single-line
        // row pitch. A VStack of top-aligned HStack rows with equal
        // flexible columns cannot overlap by construction: row height
        // IS the tallest wrapped cell, and equal fractions keep the
        // columns aligned across rows. Wide tables compress columns
        // instead of scrolling; correct beats scrollable.
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                HStack(alignment: .top, spacing: 0) {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                        Text(Self.inline(cell))
                            .font(.system(size: 12.5, weight: rowIndex == 0 && hasHeader ? .semibold : .regular))
                            .foregroundStyle(rowIndex == 0 && hasHeader ? Brand.typeHi : Brand.typeBody)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                }
                .fixedSize(horizontal: false, vertical: true)
                .background(
                    rowIndex == 0 && hasHeader
                        ? Color.white.opacity(0.05)
                        : (rowIndex % 2 == 0 ? Color.clear : Color.white.opacity(0.02))
                )
                if rowIndex == 0 && hasHeader {
                    Divider()
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.bgInner.opacity(0.5))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Brand.separator, lineWidth: 0.5)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .inlineOnlyPreservingWhitespace
            )
        )) ?? AttributedString(text)
    }
}

private struct SettledMarkdownBlock: Identifiable, Equatable {
    enum Kind: Equatable {
        case prose(String)
        case code(language: String?, code: String, lineCount: Int)
    }

    let id: Int
    let kind: Kind
}

private final class SettledMarkdownBlocksBox: NSObject {
    let blocks: [SettledMarkdownBlock]

    init(blocks: [SettledMarkdownBlock]) {
        self.blocks = blocks
    }
}

private enum CachedSettledMarkdownBlocks {
    nonisolated(unsafe) private static let cache: NSCache<NSString, SettledMarkdownBlocksBox> = {
        let cache = NSCache<NSString, SettledMarkdownBlocksBox>()
        cache.countLimit = 256
        cache.totalCostLimit = 16_000_000
        return cache
    }()

    static func blocks(for source: String) -> [SettledMarkdownBlock] {
        let key = source as NSString
        if let cached = cache.object(forKey: key) {
            return cached.blocks
        }

        let parsed = parse(source)
        cache.setObject(
            SettledMarkdownBlocksBox(blocks: parsed),
            forKey: key,
            cost: source.utf8.count
        )
        return parsed
    }

    static func removeAllObjects() {
        cache.removeAllObjects()
    }

    private static func parse(_ source: String) -> [SettledMarkdownBlock] {
        guard !source.isEmpty else { return [] }
        var blocks: [SettledMarkdownBlock] = []
        var cursor = source.startIndex

        func appendProse(upTo fenceStart: String.Index) {
            guard cursor < fenceStart else { return }
            let text = String(source[cursor..<fenceStart])
            blocks.append(SettledMarkdownBlock(id: blocks.count, kind: .prose(text)))
        }

        while cursor < source.endIndex {
            guard let fence = source.range(of: "```", range: cursor..<source.endIndex) else {
                appendProse(upTo: source.endIndex)
                break
            }

            appendProse(upTo: fence.lowerBound)

            let languageAndBodyStart = fence.upperBound
            let bodyStart: String.Index
            let language: String?
            if let newline = source[languageAndBodyStart...].firstIndex(of: "\n") {
                let rawLanguage = source[languageAndBodyStart..<newline]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                language = rawLanguage.isEmpty ? nil : rawLanguage
                bodyStart = source.index(after: newline)
            } else {
                let rawLanguage = source[languageAndBodyStart...]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: rawLanguage.isEmpty ? nil : rawLanguage,
                            code: "",
                            lineCount: 1
                        )
                    )
                )
                cursor = source.endIndex
                break
            }

            if let closingFence = source.range(of: "```", range: bodyStart..<source.endIndex) {
                let code = String(source[bodyStart..<closingFence.lowerBound])
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: language,
                            code: code,
                            lineCount: AssistantCodeMetrics.lineCount(in: code)
                        )
                    )
                )
                cursor = closingFence.upperBound
            } else {
                let code = String(source[bodyStart...])
                blocks.append(
                    SettledMarkdownBlock(
                        id: blocks.count,
                        kind: .code(
                            language: language,
                            code: code,
                            lineCount: AssistantCodeMetrics.lineCount(in: code)
                        )
                    )
                )
                cursor = source.endIndex
            }
        }

        return blocks
    }
}

enum ChatRenderCaches {}

extension ChatRenderCaches {
    static func clearMemoryPressureSensitiveCaches() {
        clearSettledMarkdownBlocks()
        clearMarkdownDocuments()
    }

    static func clearSettledMarkdownBlocks() {
        CachedSettledMarkdownBlocks.removeAllObjects()
    }
}

private enum AssistantCodeMetrics {
    static func lineCount(in code: String) -> Int {
        max(1, code.reduce(1) { count, character in
            character == "\n" ? count + 1 : count
        })
    }
}

// MARK: - StreamingAssistantMarkdownView

/// Append-only lex-state chain for the currently OPEN fence card. The
/// interior only grows at its tail while a fence streams, but the view
/// used to re-lex the WHOLE interior per body evaluation — O(fence
/// lines) of cache-key interpolation per frame, a top term of the
/// 2026-08-17 streaming-freeze field regression. Prefix identity is
/// (block id, utf8 length): frozen blocks never change text, a segment
/// merge changes both, and a document reset restarts ids — all diverge
/// the prefix and force a re-lex from the divergence point only.
@MainActor
final class StreamingFenceLexChain {
    struct Entry {
        let blockID: Int
        let textUTF8Count: Int
        let endState: MTPLXCodeHighlighter.LexState
    }

    var language: MTPLXCodeHighlighter.Language?
    var entries: [Entry] = []

    func reset(language: MTPLXCodeHighlighter.Language?) {
        self.language = language
        entries.removeAll(keepingCapacity: true)
    }
}

struct StreamingAssistantMarkdownView: View {
    let document: StreamingDocumentStore
    var fallbackText: String = ""
    /// Performance mode: no markdown promotion, no code card, no
    /// syntax coloring — the pure plain-line stream.
    var plainTextOnly: Bool = false

    var body: some View {
        Group {
            if plainTextOnly {
                StreamingPlainDocumentView(
                    document: document,
                    fallbackText: fallbackText
                )
            } else {
                StreamingRichDocumentView(
                    document: document,
                    fallbackText: fallbackText
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .transaction { tx in
            tx.animation = nil
        }
    }

    /// Groups blocks into render items using the classifier's fence
    /// roles. One linear pass per body evaluation; per-line highlight
    /// states are threaded through the cached lexer (dictionary hits
    /// for every already-frozen line, one real lex for a new line).
    @MainActor
    static func renderItems(
        for blocks: [StreamingDocumentBlock],
        lexChain: StreamingFenceLexChain? = nil
    ) -> [StreamingRenderItem] {
        UIStreamPerfProbe.renderTimed(.renderItems, size: blocks.count) {
            renderItemsBody(for: blocks, lexChain: lexChain)
        }
    }

    private static func renderItemsBody(
        for blocks: [StreamingDocumentBlock],
        lexChain: StreamingFenceLexChain? = nil
    ) -> [StreamingRenderItem] {
        // Consumes the fence counts stamped on the blocks at
        // construction — recounting text here was O(document) per frame.
        let classification = StreamingMarkdownBlockSafety.classifyRoles(blocks)
        var items: [StreamingRenderItem] = []
        items.reserveCapacity(blocks.count + 4)
        var index = 0
        while index < blocks.count {
            switch classification.fenceRoles[index] {
            case .open(let label):
                let language = MTPLXCodeHighlighter.Language.detect(fromFenceLabel: label)
                var interior: [StreamingDocumentBlock] = []
                var closed = false
                var next = index + 1
                scan: while next < blocks.count {
                    switch classification.fenceRoles[next] {
                    case .interior:
                        interior.append(blocks[next])
                        next += 1
                    case .close:
                        closed = true
                        next += 1
                        break scan
                    default:
                        break scan
                    }
                }
                if closed {
                    items.append(.closedCode(
                        id: blocks[index].id,
                        language: language,
                        code: interior.map(\.text).joined(separator: "\n")
                    ))
                } else {
                    var fragments: [StreamingCodeFragment] = []
                    fragments.reserveCapacity(max(1, interior.count))
                    // Thread the lex state through the interior,
                    // resuming from the append-only chain cache: frozen
                    // prefix rows cost two Int compares each; only rows
                    // past the divergence point (in practice: none, or
                    // the just-frozen line) actually re-lex. The LAST
                    // interior row is the growing tail — its end state
                    // feeds nothing this frame, so it is never lexed
                    // here; it lexes once on the frame after it freezes.
                    if let lexChain, lexChain.language != language {
                        lexChain.reset(language: language)
                    }
                    var state = MTPLXCodeHighlighter.LexState.none
                    var reusable = lexChain?.entries.count ?? 0
                    for (offset, block) in interior.enumerated() {
                        fragments.append(StreamingCodeFragment(
                            block: block,
                            entryTag: state.cacheTag
                        ))
                        if offset == interior.count - 1 { break }
                        let bytes = block.text.utf8.count
                        if let lexChain, offset < reusable {
                            let cached = lexChain.entries[offset]
                            if cached.blockID == block.id, cached.textUTF8Count == bytes {
                                state = cached.endState
                                continue
                            }
                            lexChain.entries.removeSubrange(offset...)
                            reusable = offset
                        }
                        if block.text.contains("\n") {
                            state = MTPLXCodeHighlighter
                                .highlightSegmentEndState(block.text, language: language, state: state)
                        } else {
                            state = MTPLXCodeHighlighter
                                .highlightLine(block.text, language: language, state: state)
                                .endState
                        }
                        lexChain?.entries.append(.init(
                            blockID: block.id,
                            textUTF8Count: bytes,
                            endState: state
                        ))
                    }
                    if interior.isEmpty {
                        // Header-only card so an empty just-opened fence
                        // still shows its chrome.
                        fragments.append(StreamingCodeFragment(
                            block: StreamingDocumentBlock(
                                id: blocks[index].id &+ 1_000_000,
                                text: "",
                                kind: .unfinished,
                                finalized: false
                            ),
                            entryTag: MTPLXCodeHighlighter.LexState.none.cacheTag
                        ))
                    }
                    items.append(.openCode(
                        id: blocks[index].id,
                        language: language,
                        fragments: fragments
                    ))
                }
                index = next
            case .none, .interior, .close, .mixed:
                // Keep prose visually immutable while it streams. Promoting
                // each completed line from plain Text to MarkdownUI changed
                // its color, spacing, and height when the following line
                // arrived — the visible "whole answer flickers every few
                // lines" regression. The persisted bubble performs the one
                // Markdown promotion after the response is complete; open
                // fences still use the incremental highlighted card above.
                items.append(.plain(blocks[index]))
                index += 1
            }
        }
        return items
    }

    @MainActor
    static func openCodeFragments(
        in document: StreamingDocumentStore,
        fenceID: Int,
        lexChain: StreamingFenceLexChain? = nil
    ) -> [StreamingCodeFragment]? {
        for item in renderItems(for: document.blocks, lexChain: lexChain) {
            if case .openCode(let id, _, let fragments) = item, id == fenceID {
                return fragments
            }
        }
        return nil
    }
}

/// Performance Lock deliberately retains the simple observed SwiftUI path.
/// It has no syntax/TextKit work, so direct block publication is cheap and
/// its semantics stay exactly as before.
private struct StreamingPlainDocumentView: View {
    @ObservedObject var document: StreamingDocumentStore
    let fallbackText: String

    var body: some View {
        if document.blocks.isEmpty {
            StreamingPlainTextView(text: fallbackText)
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(document.blocks) { block in
                    StreamingPlainBlockView(block: block)
                        .equatable()
                }
            }
        }
    }
}

/// Publishes SwiftUI structure changes only. While an open code fence grows,
/// its TextKit bridge listens to the document directly; characters no longer
/// invalidate the entire SwiftUI transcript 30-60 times per second. We still
/// publish a bounded height ramp for the open fence (six 4-line steps to the
/// full slot), the one fence-close handoff, and normal prose changes.
// Internal (not private): the Host flatness tests drive this model and the
// TextKit viewport directly — they are the O(n^2) tripwires.
@MainActor
final class StreamingRichRenderModel: ObservableObject {
    @Published private(set) var items: [StreamingRenderItem] = []

    /// Revealed-line bucket for the trailing open fence, in 4-line steps
    /// capped at the count that fills the card's 420 pt slot. This is what
    /// lets the live code card grow with its content instead of reserving
    /// the whole empty slot the moment a fence opens (2026-08-19 field
    /// report: "huge blank space it fills up"). Publishing the BUCKET —
    /// not the line count — bounds the SwiftUI republish cost to at most
    /// six transcript relayouts per fence, after which the slot is fixed
    /// and the transcript sleeps again.
    @Published private(set) var openFenceLineBucket = 0

    /// Every revision's derivation, unfiltered. The open code card's TextKit
    /// coordinator pumps its text from this; `items` above only publishes
    /// structural changes to SwiftUI. One derivation serves both sinks —
    /// renderItems used to run twice per revision, each pass O(blocks)
    /// (streamwar 2026-08-19).
    private(set) var latestItems: [StreamingRenderItem] = []
    let perRevision = PassthroughSubject<[StreamingRenderItem], Never>()

    private let document: StreamingDocumentStore
    private let lexChain = StreamingFenceLexChain()
    private var revisionCancellable: AnyCancellable?

    init(document: StreamingDocumentStore) {
        self.document = document
        refresh(force: true)
        revisionCancellable = document.revisionPublisher.sink { [weak self] _ in
            self?.refresh()
        }
    }

    private func refresh(force: Bool = false) {
        let next = StreamingAssistantMarkdownView.renderItems(
            for: document.blocks,
            lexChain: lexChain
        )
        latestItems = next
        perRevision.send(next)
        let bucket = Self.openFenceLineBucket(for: next)
        if bucket != openFenceLineBucket {
            openFenceLineBucket = bucket
        }
        if force || !Self.samePresentation(items, next) {
            items = next
        }
    }

    /// Full slot = 420 pt; the card's height formula is lines * 17 + 22, so
    /// 24 lines saturate it. Rounding UP to the next 4-line step keeps the
    /// box at least as tall as its revealed content, so the surface stays
    /// top-anchored through the whole ramp (no tail-follow flicker between
    /// steps).
    static func openFenceLineBucket(for items: [StreamingRenderItem]) -> Int {
        guard case let .openCode(_, _, fragments)? = items.last else { return 0 }
        let lines = fragments.reduce(0) { $0 + $1.lineCount }
        return min(24, ((max(lines, 1) + 3) / 4) * 4)
    }

    private static func samePresentation(
        _ lhs: [StreamingRenderItem],
        _ rhs: [StreamingRenderItem]
    ) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for (old, new) in zip(lhs, rhs) {
            switch (old, new) {
            case (.settled(let a), .settled(let b)),
                 (.plain(let a), .plain(let b)):
                guard a == b else { return false }
            case let (.openCode(aID, aLanguage, _),
                      .openCode(bID, bLanguage, _)):
                // Fragment growth is deliberately NOT compared: the live
                // card's text flows through the TextKit coordinator, so
                // SwiftUI has nothing to re-evaluate while a fence streams
                // (streamwar 2026-08-19; the old per-line height bucket
                // forced a full-transcript relayout on each of a fence's
                // first 24 line boundaries). The card's height ramp flows
                // through `openFenceLineBucket` instead — a separate
                // published value that changes at most six times per fence.
                guard aID == bID, aLanguage == bLanguage else { return false }
            case let (.closedCode(aID, aLanguage, aCode),
                      .closedCode(bID, bLanguage, bCode)):
                guard aID == bID, aLanguage == bLanguage, aCode == bCode else {
                    return false
                }
            default:
                return false
            }
        }
        return true
    }

}

private struct StreamingRichDocumentView: View {
    let document: StreamingDocumentStore
    let fallbackText: String
    @StateObject private var renderModel: StreamingRichRenderModel

    init(document: StreamingDocumentStore, fallbackText: String) {
        self.document = document
        self.fallbackText = fallbackText
        _renderModel = StateObject(
            wrappedValue: StreamingRichRenderModel(document: document)
        )
    }

    var body: some View {
        if renderModel.items.isEmpty {
            StreamingPlainTextView(text: fallbackText)
        } else {
            // This MUST remain a plain VStack. The outer transcript is moved
            // by an AppKit scroll driver; LazyVStack can cull visible rows
            // against stale SwiftUI scroll bookkeeping.
            VStack(alignment: .leading, spacing: 0) {
                ForEach(renderModel.items) { item in
                    itemView(item)
                }
            }
        }
    }

    @ViewBuilder
    private func itemView(_ item: StreamingRenderItem) -> some View {
        switch item {
        case .settled(let block):
            StreamingSettledBlockView(text: block.text)
                .equatable()
        case .plain(let block):
            StreamingPlainBlockView(block: block)
                .equatable()
        case .openCode(let id, let language, _):
            StreamingOpenCodeCardView(
                id: id,
                language: language,
                lineBucket: renderModel.openFenceLineBucket,
                document: document,
                renderModel: renderModel
            )
            .equatable()
        case .closedCode(let id, let language, let code):
            StreamingClosedCodeCardView(id: id, language: language, code: code)
                .equatable()
        }
    }
}

enum StreamingRenderItem: Identifiable {
    case settled(StreamingDocumentBlock)
    case plain(StreamingDocumentBlock)
    case openCode(id: Int, language: MTPLXCodeHighlighter.Language, fragments: [StreamingCodeFragment])
    case closedCode(id: Int, language: MTPLXCodeHighlighter.Language, code: String)

    var id: Int {
        switch self {
        case .settled(let block): return block.id
        case .plain(let block): return block.id
        case .openCode(let id, _, _): return id
        case .closedCode(let id, _, _): return id
        }
    }
}

struct StreamingCodeFragment: Equatable {
    let id: Int
    let text: String
    let entryTag: String
    let lineCount: Int

    init(block: StreamingDocumentBlock, entryTag: String) {
        id = block.id
        text = block.text
        self.entryTag = entryTag
        lineCount = block.lineCount
    }
}

// MARK: Incremental live code card

/// One bounded TextKit surface for an OPEN code fence. SwiftUI owns the
/// card chrome and height; NSTextStorage owns the growing code. This
/// keeps the live view hierarchy constant whether the model writes 20
/// lines or 2,000.
private struct StreamingOpenCodeCardView: View, Equatable {
    let id: Int
    let language: MTPLXCodeHighlighter.Language
    let lineBucket: Int
    let document: StreamingDocumentStore
    let renderModel: StreamingRichRenderModel

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id
            && lhs.language == rhs.language
            && lhs.lineBucket == rhs.lineBucket
            && lhs.document === rhs.document
            && lhs.renderModel === rhs.renderModel
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Text(language == .generic ? "CODE" : language.rawValue.uppercased())
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.typeTertiary)
                Spacer(minLength: 12)
                Text("STREAMING")
                    .font(.system(size: 8, weight: .heavy, design: .monospaced))
                    .tracking(1.2)
                    .foregroundStyle(Brand.typeTertiary.opacity(0.7))
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        currentCode,
                        forType: .string
                    )
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                        .font(.system(size: 10, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Brand.typeTertiary)
                .help("Copy code")
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 5)
            .background(Color.white.opacity(0.035))

            StreamingCodeTextViewport(
                renderModel: renderModel,
                fenceID: id,
                language: language
            )
            .frame(height: viewportHeight)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(
                "\(language.rawValue) code block, streaming"
            )
            .accessibilityHint("Use the Copy button to copy the full code.")
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.bgInner)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .padding(.top, 4)
    }

    private var viewportHeight: CGFloat {
        // Grow with the revealed code in 4-line steps, capped at the full
        // 420 pt slot. The bucket is published by StreamingRichRenderModel
        // at most six times per fence, so the transcript relayout cost
        // stays bounded (the old per-line ramp re-laid the transcript on
        // each of the first 24 lines; the interim fixed-420 slot showed a
        // huge blank box for the fence's first seconds — 2026-08-19 field
        // report). Same formula as the settled card's codeViewportHeight,
        // so the open -> closed handoff doesn't jump. Content top-anchors
        // inside the slot (LiveTailTextSurface.anchorsTopWhenShort), and
        // past the cap the slot is fixed again — zero per-line transcript
        // relayout for long fences.
        min(420, max(78, CGFloat(lineBucket) * 17 + 22))
    }

    @MainActor
    private var currentCode: String {
        StreamingAssistantMarkdownView
            .openCodeFragments(in: document, fenceID: id)?
            .map(\.text)
            .joined(separator: "\n") ?? ""
    }
}

struct StreamingCodeTextViewport: NSViewRepresentable {
    let renderModel: StreamingRichRenderModel
    let fenceID: Int
    let language: MTPLXCodeHighlighter.Language

    @MainActor
    final class Coordinator {
        struct AppliedFragment {
            let id: Int
            let text: String
            let entryTag: String
            let renderedUTF16Length: Int
        }

        weak var surface: LiveTailTextSurface?
        var renderModel: StreamingRichRenderModel?
        var fenceID: Int?
        var requestedLanguage: MTPLXCodeHighlighter.Language?
        var appliedLanguage: MTPLXCodeHighlighter.Language?
        var applied: [AppliedFragment] = []
        var itemsCancellable: AnyCancellable?

        func attach(
            renderModel: StreamingRichRenderModel,
            fenceID: Int,
            language: MTPLXCodeHighlighter.Language,
            surface: LiveTailTextSurface
        ) {
            let surfaceChanged = self.surface !== surface
            self.surface = surface
            let sameSource = self.renderModel === renderModel
                && self.fenceID == fenceID
                && requestedLanguage == language
            if !sameSource || surfaceChanged {
                itemsCancellable?.cancel()
                self.renderModel = renderModel
                self.fenceID = fenceID
                requestedLanguage = language
                appliedLanguage = nil
                applied.removeAll(keepingCapacity: true)
                // One derivation per revision: the render model already
                // derives every revision's items; this sink consumes them
                // instead of re-running renderItems over all blocks.
                itemsCancellable = renderModel.perRevision.sink { [weak self] items in
                    self?.refresh(items: items)
                }
            }
            refresh(items: renderModel.latestItems)
        }

        private func refresh(items: [StreamingRenderItem]) {
            guard let surface,
                  let fenceID,
                  let language = requestedLanguage else { return }
            for item in items {
                guard case .openCode(let id, _, let fragments) = item,
                      id == fenceID else { continue }
                UIStreamPerfProbe.renderTimed(.applyRender, size: fragments.count) {
                    StreamingCodeTextViewport.apply(
                        fragments: fragments,
                        language: language,
                        to: surface,
                        coordinator: self
                    )
                }
                return
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> LiveTailTextSurface {
        let surface = LiveTailTextSurface(frame: .zero)
        surface.contentInsets = NSSize(width: 12, height: 10)
        surface.anchorsTopWhenShort = true
        surface.setAccessibilityElement(false)
        context.coordinator.attach(
            renderModel: renderModel,
            fenceID: fenceID,
            language: language,
            surface: surface
        )
        return surface
    }

    func updateNSView(_ surface: LiveTailTextSurface, context: Context) {
        context.coordinator.attach(
            renderModel: renderModel,
            fenceID: fenceID,
            language: language,
            surface: surface
        )
    }

    static func apply(
        fragments allFragments: [StreamingCodeFragment],
        language: MTPLXCodeHighlighter.Language,
        to surface: LiveTailTextSurface,
        coordinator: Coordinator
    ) {
        let storage = surface.textStorage

        // Bound TextKit work to a window of the fence tail (streamwar
        // 2026-08-19, restoring the ebd51228 bound that 37dbd81d deleted).
        // Without it the storage holds the ENTIRE growing fence behind the
        // fixed 420 pt viewport and every draw's tail-layout query costs
        // O(fence) — the measured line-boundary freeze. The window is
        // anchored on the first already-rendered fragment id, so it only
        // ever extends or front-trims at existing boundaries; it is never
        // recomputed from scratch mid-stream (recomputing each frame slid
        // the window start every line and forced full-window repaints —
        // the flicker 37dbd81d chased when it removed the bound).
        let fragments: [StreamingCodeFragment]
        if coordinator.appliedLanguage == language,
           let anchorID = coordinator.applied.first?.id,
           let anchorIndex = allFragments.firstIndex(where: { $0.id == anchorID }) {
            fragments = Array(allFragments[anchorIndex...])
        } else {
            // First attach, language change, or the anchor left the block
            // list (document reset): render the bounded tail fresh.
            fragments = Array(Self.visibleTail(in: allFragments))
            coordinator.applied.removeAll(keepingCapacity: true)
        }

        // StreamingDocumentStore periodically folds frozen line blocks into
        // one multiline segment. That changes block structure but not one
        // byte of rendered code. Reconcile that metadata-only merge before
        // finding the changed suffix so TextKit keeps its existing glyphs and
        // colors instead of repainting the whole visible card.
        if coordinator.appliedLanguage == language {
            reconcileFrozenMerges(fragments: fragments, coordinator: coordinator)
        }

        var common = 0
        if coordinator.appliedLanguage == language {
            let count = min(coordinator.applied.count, fragments.count)
            while common < count {
                let old = coordinator.applied[common]
                let new = fragments[common]
                guard old.id == new.id,
                      old.text == new.text,
                      old.entryTag == new.entryTag else { break }
                common += 1
            }
        }

        if common == coordinator.applied.count,
           common == fragments.count,
           coordinator.appliedLanguage == language {
            return
        }

        let unchangedUTF16 = coordinator.applied
            .prefix(common)
            .reduce(0) { $0 + $1.renderedUTF16Length }
        let suffix = NSMutableAttributedString()
        var nextApplied = Array(coordinator.applied.prefix(common))

        for index in common..<fragments.count {
            let fragment = fragments[index]
            let hasSeparator = index > 0
            if hasSeparator {
                suffix.append(NSAttributedString(
                    string: "\n",
                    attributes: [
                        .font: MTPLXCodeHighlighter.codeFont,
                        .foregroundColor: NSColor(calibratedWhite: 0.88, alpha: 1.0),
                    ]
                ))
            }
            suffix.append(MTPLXCodeHighlighter.highlightedFragment(
                fragment.text.isEmpty ? " " : fragment.text,
                language: language,
                entryTag: fragment.entryTag
            ))
            nextApplied.append(Coordinator.AppliedFragment(
                id: fragment.id,
                text: fragment.text,
                entryTag: fragment.entryTag,
                renderedUTF16Length: (hasSeparator ? 1 : 0)
                    + (fragment.text.isEmpty ? 1 : fragment.text.utf16.count)
            ))
        }

        storage.beginEditing()
        storage.replaceCharacters(
            in: NSRange(
                location: unchangedUTF16,
                length: max(0, storage.length - unchangedUTF16)
            ),
            with: suffix
        )
        storage.endEditing()
        coordinator.appliedLanguage = language
        coordinator.applied = nextApplied
        trimRenderedHead(coordinator: coordinator, storage: storage)
        surface.textDidChange()
    }

    /// Keep TextKit work bounded even after a 60k-token answer. Two
    /// viewport-heights of logical lines preserve lexer continuity and make
    /// wrapped long lines safe, while the full code remains in the document
    /// store and Copy action.
    static func visibleTail(
        in fragments: [StreamingCodeFragment],
        minimumLines: Int = 48
    ) -> ArraySlice<StreamingCodeFragment> {
        var start = fragments.endIndex
        var lines = 0
        while start > fragments.startIndex, lines < minimumLines {
            start = fragments.index(before: start)
            lines += fragments[start].lineCount
        }
        return fragments[start...]
    }

    /// Slide the rendered window forward by dropping whole leading merged
    /// segments (and fence lines) once enough lines remain. Only those are
    /// safe anchors: the store never re-merges a multiline segment and never
    /// merges a fence line, so the new head's id stays findable in every
    /// future block list. A recent single line is NOT trimmed — a later
    /// coalesce could absorb it mid-segment, orphan the anchor, and force a
    /// full-window repaint (the flicker this design exists to avoid). Head
    /// deletion never changes the pixels of surviving lines; the surface
    /// draws bottom-anchored.
    private static func trimRenderedHead(
        coordinator: Coordinator,
        storage: NSTextStorage,
        minimumLines: Int = 48
    ) {
        func lineCount(_ text: String) -> Int {
            text.utf8.reduce(into: 1) { count, byte in
                if byte == 0x0A { count += 1 }
            }
        }
        var totalLines = coordinator.applied.reduce(0) { $0 + lineCount($1.text) }
        var deleteUTF16 = 0
        while coordinator.applied.count > 1 {
            let head = coordinator.applied[0]
            let headIsPermanentBoundary = head.text.contains("\n")
                || StreamingMarkdownBlockSafety.isFenceLine(head.text)
            guard headIsPermanentBoundary else { break }
            let headLines = lineCount(head.text)
            guard totalLines - headLines >= minimumLines else { break }
            // The head's stored length excludes a separator; the separator
            // between it and the next fragment is stored in the NEXT
            // fragment's length. Delete head + that separator and re-tag
            // the new head as separator-free.
            deleteUTF16 += head.renderedUTF16Length + 1
            coordinator.applied.removeFirst()
            let newHead = coordinator.applied[0]
            coordinator.applied[0] = Coordinator.AppliedFragment(
                id: newHead.id,
                text: newHead.text,
                entryTag: newHead.entryTag,
                renderedUTF16Length: newHead.renderedUTF16Length - 1
            )
            totalLines -= headLines
        }
        guard deleteUTF16 > 0 else { return }
        storage.beginEditing()
        storage.deleteCharacters(
            in: NSRange(location: 0, length: min(deleteUTF16, storage.length))
        )
        storage.endEditing()
    }

    /// Collapse coordinator metadata when the document store combines a run
    /// of frozen lines. The attributed storage is already byte-for-byte
    /// correct, so touching it would only create a flash and needless layout.
    private static func reconcileFrozenMerges(
        fragments: [StreamingCodeFragment],
        coordinator: Coordinator
    ) {
        guard !coordinator.applied.isEmpty, !fragments.isEmpty else { return }

        let old = coordinator.applied
        var normalized: [Coordinator.AppliedFragment] = []
        normalized.reserveCapacity(fragments.count)
        var oldIndex = 0
        var newIndex = 0

        while oldIndex < old.count, newIndex < fragments.count {
            let previous = old[oldIndex]
            let current = fragments[newIndex]

            if previous.id == current.id,
               previous.text == current.text,
               previous.entryTag == current.entryTag {
                normalized.append(previous)
                oldIndex += 1
                newIndex += 1
                continue
            }

            guard previous.id == current.id,
                  previous.entryTag == current.entryTag,
                  current.text.contains("\n") else { break }

            var mergedText = ""
            var mergedUTF16Length = 0
            var scan = oldIndex
            var matched = false
            while scan < old.count {
                if scan > oldIndex {
                    mergedText.append("\n")
                }
                mergedText.append(old[scan].text)
                mergedUTF16Length += old[scan].renderedUTF16Length

                if mergedText == current.text {
                    normalized.append(Coordinator.AppliedFragment(
                        id: current.id,
                        text: current.text,
                        entryTag: current.entryTag,
                        renderedUTF16Length: mergedUTF16Length
                    ))
                    oldIndex = scan + 1
                    newIndex += 1
                    matched = true
                    break
                }
                guard current.text.hasPrefix(mergedText) else { break }
                scan += 1
            }
            guard matched else { break }
        }

        guard oldIndex > 0 else { return }
        normalized.append(contentsOf: old[oldIndex...])
        coordinator.applied = normalized
    }
}

/// A CLOSED fence during streaming: flips once to the exact settled
/// code card, so the end-of-turn handoff to the persisted transcript
/// doesn't jump.
private struct StreamingClosedCodeCardView: View, Equatable {
    let id: Int
    let language: MTPLXCodeHighlighter.Language
    let code: String

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.language == rhs.language && lhs.code == rhs.code
    }

    var body: some View {
        AssistantCodeBlockView(
            language: language == .generic ? nil : language.rawValue,
            code: code
        )
        .padding(.vertical, 4)
    }
}

/// A frozen streaming block promoted to the settled markdown pipeline.
/// Equatable on its text: SwiftUI evaluates the body once when the
/// block freezes and never again during the rest of the stream.
private struct StreamingSettledBlockView: View, Equatable {
    let text: String

    nonisolated static func == (lhs: StreamingSettledBlockView, rhs: StreamingSettledBlockView) -> Bool {
        lhs.text == rhs.text
    }

    var body: some View {
        SettledAssistantMarkdownView(content: text)
            .padding(.bottom, 6)
    }
}

private struct StreamingPlainTextView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 14))
            .foregroundStyle(Brand.typeHi)
            .textSelection(.disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct StreamingPlainBlockView: View, Equatable {
    let block: StreamingDocumentBlock

    nonisolated static func == (lhs: StreamingPlainBlockView, rhs: StreamingPlainBlockView) -> Bool {
        lhs.block == rhs.block
    }

    var body: some View {
        Text(block.text.isEmpty ? " " : block.text)
            .font(.system(size: 14))
            .foregroundStyle(Brand.typeHi)
            .textSelection(.disabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - Brand-themed MarkdownUI Theme

extension Theme {
    /// Brand-themed MarkdownUI theme for the in-app chat. Mirrors the
    /// shape of Aphanes' theme but rewires every color/font to MTPLX
    /// tokens. MainActor-scoped because the `markdownMargin` /
    /// `markdownTextStyle` SwiftUI modifiers used inside the block
    /// closures are main-actor isolated under Swift 6.
    @MainActor
    static var mtplxChat: Theme {
        Theme()
            .text {
                ForegroundColor(Brand.typeHi)
                FontSize(14)
                FontFamilyVariant(.normal)
            }
            .code {
                FontFamilyVariant(.monospaced)
                FontSize(13)
                ForegroundColor(Brand.typeHi)
                BackgroundColor(Color.white.opacity(0.06))
            }
            .link {
                ForegroundColor(Brand.accentChrome)
                UnderlineStyle(.single)
            }
            .strong { FontWeight(.semibold) }
            .emphasis { FontStyle(.italic) }
            .strikethrough { StrikethroughStyle(.single) }
            .heading1 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.heavy)
                        FontSize(22)
                    }
                    .markdownMargin(top: 18, bottom: 8)
            }
            .heading2 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.heavy)
                        FontSize(18)
                    }
                    .markdownMargin(top: 16, bottom: 6)
            }
            .heading3 { configuration in
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Brand.typeHi)
                        FontWeight(.bold)
                        FontSize(15)
                    }
                    .markdownMargin(top: 14, bottom: 4)
            }
            .paragraph { configuration in
                configuration.label
                    .markdownMargin(top: 0, bottom: 8)
            }
            .blockquote { configuration in
                configuration.label
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) {
                        Rectangle()
                            .fill(Brand.separatorStrong)
                            .frame(width: 2)
                    }
                    .foregroundStyle(Brand.typeSecondary)
            }
            .codeBlock { configuration in
                AssistantCodeBlockView(
                    language: configuration.language,
                    code: configuration.content
                )
                    .markdownMargin(top: 8, bottom: 8)
            }
            .listItem { configuration in
                configuration.label
                    .markdownMargin(top: 4)
            }
            .table { configuration in
                configuration.label
                    .markdownMargin(top: 8, bottom: 8)
            }
            .tableCell { configuration in
                configuration.label
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .foregroundStyle(Brand.typeHi)
            }
    }
}

private struct AssistantCodeBlockView: View {
    let language: String?
    let code: String
    let lineCount: Int

    @State private var showCopied = false

    init(language: String?, code: String, lineCount: Int? = nil) {
        self.language = language
        self.code = code
        self.lineCount = lineCount ?? AssistantCodeMetrics.lineCount(in: code)
    }

    private var languageLabel: String? {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed?.isEmpty == false ? trimmed : nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                if let languageLabel {
                    Text(languageLabel.uppercased())
                        .font(.system(size: 9, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(1)
                }

                Spacer(minLength: 12)

                copyButton
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 5)
            .background(Color.white.opacity(0.035))

            CodeTextViewport(
                code: code,
                language: language,
                highlighted: !ChatRenderPreferences.plainTextOnly
            )
                .frame(height: codeViewportHeight)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(codeAccessibilitySummary)
            .accessibilityHint("Use the Copy button to copy the full code.")
        }
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.bgInner)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var codeAccessibilitySummary: String {
        let label = languageLabel ?? "Plain text"
        return "\(label) code block, \(code.count) characters"
    }

    private var codeViewportHeight: CGFloat {
        let unclamped = CGFloat(lineCount) * 17 + 22
        return min(420, max(78, unclamped))
    }

    private var copyButton: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(code, forType: .string)
            showCopied = true

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1.4))
                showCopied = false
            }
        } label: {
            Label(showCopied ? "Copied" : "Copy", systemImage: showCopied ? "checkmark" : "doc.on.doc")
                .font(.system(size: 11, weight: .semibold))
                .labelStyle(.titleAndIcon)
                .foregroundStyle(showCopied ? Brand.success : Brand.typeTertiary)
        }
        .buttonStyle(.plain)
        .help(showCopied ? "Copied" : "Copy code")
    }
}

private struct CodeTextViewport: NSViewRepresentable {
    let code: String
    var language: String? = nil
    var highlighted: Bool = true

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.horizontalScrollElasticity = .none

        let textView = NSTextView()
        textView.drawsBackground = false
        textView.isEditable = false
        textView.isSelectable = false
        textView.isRichText = false
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textColor = NSColor(calibratedWhite: 0.88, alpha: 1.0)
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.heightTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: max(1, scrollView.contentSize.width),
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.autoresizingMask = [.width]
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.setAccessibilityElement(false)
        apply(to: textView)
        context.coordinator.appliedCode = code
        context.coordinator.appliedHighlighted = highlighted

        scrollView.documentView = textView
        textView.frame = NSRect(origin: .zero, size: scrollView.contentSize)
        return scrollView
    }

    final class Coordinator {
        var appliedCode: String?
        var appliedHighlighted: Bool?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        // Compare against the retained applied String, NOT
        // `textView.string`: that getter materializes the whole
        // NSTextStorage into a fresh Swift String on every update
        // (O(code) per frame on a giant block). The retained String
        // shares storage with `code` when unchanged, so `==` is a
        // pointer check. Tracking `highlighted` also fixes a latent
        // bug: toggling performance mode used to leave stale coloring.
        if context.coordinator.appliedCode != code
            || context.coordinator.appliedHighlighted != highlighted {
            apply(to: textView)
            context.coordinator.appliedCode = code
            context.coordinator.appliedHighlighted = highlighted
        }
    }

    /// Syntax-colored code via the freeze-time lexer (cached by
    /// content, so a settled block pays the lex exactly once);
    /// plain white when highlighting is off (performance mode).
    private func apply(to textView: NSTextView) {
        if highlighted {
            let attributed = MTPLXCodeHighlighter.highlightCode(
                code,
                language: MTPLXCodeHighlighter.Language.detect(fromFenceLabel: language)
            )
            textView.textStorage?.setAttributedString(attributed)
        } else {
            textView.string = code
        }
    }
}
