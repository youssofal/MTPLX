import AppKit
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
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Text(Self.inline(cell))
                                .font(.system(size: 12.5, weight: rowIndex == 0 && hasHeader ? .semibold : .regular))
                                .foregroundStyle(rowIndex == 0 && hasHeader ? Brand.typeHi : Brand.typeBody)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .frame(maxWidth: 260, alignment: .leading)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .background(
                        rowIndex == 0 && hasHeader
                            ? Color.white.opacity(0.05)
                            : (rowIndex % 2 == 0 ? Color.clear : Color.white.opacity(0.02))
                    )
                    if rowIndex == 0 && hasHeader {
                        Divider().gridCellUnsizedAxes(.horizontal)
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

struct StreamingAssistantMarkdownView: View {
    @ObservedObject var document: StreamingDocumentStore
    var fallbackText: String = ""
    /// Performance mode: no markdown promotion, no code card, no
    /// syntax coloring — the pure plain-line stream.
    var plainTextOnly: Bool = false

    var body: some View {
        Group {
            if document.blocks.isEmpty {
                StreamingPlainTextView(text: fallbackText)
            } else if plainTextOnly {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(document.blocks) { block in
                        StreamingPlainBlockView(block: block)
                            .equatable()
                    }
                }
            } else {
                // Frozen fence-safe blocks render as full markdown ONCE
                // (Equatable on text, so they never repaint as later
                // tokens arrive). Fence regions render as a LIVE code
                // card: the ```lang line becomes the card header, each
                // frozen interior line is lexed exactly once
                // (freeze-time highlighting, cached), and the growing
                // tail line re-lexes only itself. While the fence is
                // OPEN the card is per-row views — no O(fence) work per
                // flush; the moment it closes, the region flips once to
                // the exact settled code card. Per-token cost stays one
                // linear classify pass + the tail repaint (2026-07-03
                // contract, extended 2026-07-31).
                let items = Self.renderItems(for: document.blocks)
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(items) { item in
                        itemView(item)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .transaction { tx in
            tx.animation = nil
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
        case .fenceHeader(let id, let language, _):
            StreamingCodeCardHeaderView(id: id, language: language)
                .equatable()
        case .fenceLine(let block, let language, let entryTag, let isLast):
            StreamingCodeCardLineView(
                text: block.text,
                language: language,
                entryTag: entryTag,
                isLast: isLast,
                blockID: block.id
            )
            .equatable()
        case .closedCode(let id, let language, let code):
            StreamingClosedCodeCardView(id: id, language: language, code: code)
                .equatable()
        }
    }

    /// Groups blocks into render items using the classifier's fence
    /// roles. One linear pass per body evaluation; per-line highlight
    /// states are threaded through the cached lexer (dictionary hits
    /// for every already-frozen line, one real lex for a new line).
    static func renderItems(for blocks: [StreamingDocumentBlock]) -> [StreamingRenderItem] {
        let classification = StreamingMarkdownBlockSafety.classifyRoles(blocks.map(\.text))
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
                    items.append(.fenceHeader(
                        id: blocks[index].id,
                        language: language,
                        label: label
                    ))
                    var state = MTPLXCodeHighlighter.LexState.none
                    for (offset, block) in interior.enumerated() {
                        items.append(.fenceLine(
                            block: block,
                            language: language,
                            entryTag: state.cacheTag,
                            isLast: offset == interior.count - 1
                        ))
                        if block.text.contains("\n") {
                            state = MTPLXCodeHighlighter
                                .highlightSegmentEndState(block.text, language: language, state: state)
                        } else {
                            state = MTPLXCodeHighlighter
                                .highlightLine(block.text, language: language, state: state)
                                .endState
                        }
                    }
                    if interior.isEmpty {
                        // Header-only card so an empty just-opened fence
                        // still shows its chrome.
                        items.append(.fenceLine(
                            block: StreamingDocumentBlock(
                                id: blocks[index].id &+ 1_000_000,
                                text: "",
                                kind: .unfinished,
                                finalized: false
                            ),
                            language: language,
                            entryTag: MTPLXCodeHighlighter.LexState.none.cacheTag,
                            isLast: true
                        ))
                    }
                }
                index = next
            case .none, .interior, .close, .mixed:
                if index < classification.settledSafe.count, classification.settledSafe[index] {
                    items.append(.settled(blocks[index]))
                } else {
                    items.append(.plain(blocks[index]))
                }
                index += 1
            }
        }
        return items
    }
}

enum StreamingRenderItem: Identifiable {
    case settled(StreamingDocumentBlock)
    case plain(StreamingDocumentBlock)
    case fenceHeader(id: Int, language: MTPLXCodeHighlighter.Language, label: String?)
    case fenceLine(block: StreamingDocumentBlock, language: MTPLXCodeHighlighter.Language, entryTag: String, isLast: Bool)
    case closedCode(id: Int, language: MTPLXCodeHighlighter.Language, code: String)

    var id: Int {
        switch self {
        case .settled(let block): return block.id
        case .plain(let block): return block.id
        case .fenceHeader(let id, _, _): return id
        case .fenceLine(let block, _, _, _): return block.id
        case .closedCode(let id, _, _): return id
        }
    }
}

// MARK: Live code card rows

/// Header row of an OPEN streaming fence: language chip + card top
/// chrome. Equatable on identity+language — renders once per fence.
private struct StreamingCodeCardHeaderView: View, Equatable {
    let id: Int
    let language: MTPLXCodeHighlighter.Language

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.language == rhs.language
    }

    var body: some View {
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
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 5)
        .background(Color.white.opacity(0.035))
        .background(Brand.bgInner)
        .clipShape(UnevenRoundedRectangle(
            topLeadingRadius: 10, bottomLeadingRadius: 0,
            bottomTrailingRadius: 0, topTrailingRadius: 10,
            style: .continuous
        ))
        .padding(.top, 4)
    }
}

/// One code line inside an OPEN streaming fence, syntax-colored via
/// the freeze-time lexer cache. Equatable on (text, language, entry
/// state): frozen lines never re-evaluate; only the growing tail line
/// repaints, re-lexing just itself.
private struct StreamingCodeCardLineView: View, Equatable {
    let text: String
    let language: MTPLXCodeHighlighter.Language
    let entryTag: String
    let isLast: Bool
    let blockID: Int

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.text == rhs.text
            && lhs.language == rhs.language
            && lhs.entryTag == rhs.entryTag
            && lhs.isLast == rhs.isLast
            && lhs.blockID == rhs.blockID
    }

    var body: some View {
        Text(AttributedString(MTPLXCodeHighlighter.highlightedFragment(
            text.isEmpty ? " " : text,
            language: language,
            entryTag: entryTag
        )))
        .textSelection(.disabled)
        .frame(maxWidth: .infinity, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, 12)
        .padding(.bottom, isLast ? 10 : 0)
        .background(Brand.bgInner)
        .clipShape(UnevenRoundedRectangle(
            topLeadingRadius: 0, bottomLeadingRadius: isLast ? 10 : 0,
            bottomTrailingRadius: isLast ? 10 : 0, topTrailingRadius: 0,
            style: .continuous
        ))
        .padding(.bottom, isLast ? 4 : 0)
    }
}

/// A CLOSED fence during streaming: flips once to the exact settled
/// code card (highlighted NSTextView with horizontal scroll), so the
/// end-of-turn handoff to the persisted transcript doesn't jump.
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
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = true

        let textView = NSTextView()
        textView.drawsBackground = false
        textView.isEditable = false
        textView.isSelectable = false
        textView.isRichText = false
        textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.textColor = NSColor(calibratedWhite: 0.88, alpha: 1.0)
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.heightTracksTextView = false
        textView.textContainer?.containerSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.isHorizontallyResizable = true
        textView.isVerticallyResizable = true
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.setAccessibilityElement(false)
        apply(to: textView)

        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        if textView.string != code {
            apply(to: textView)
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
