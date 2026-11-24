import AppKit
import Combine
import SwiftUI
import MTPLXAppCore

// MARK: - StreamingProseTextSurface
//
// Single-surface AppKit TextKit renderer for live streaming Markdown prose.
// Mirrors `StreamingCodeTextViewport` so contiguous prose lines render inside
// one lightweight layer-backed NSView fed directly by `StreamingRichRenderModel
// .perRevision`, rather than publishing dozens of `.plain` SwiftUI `Text` views
// on every 8.3 ms display tick.
//
// Key invariants:
//   - Incremental prefix diffing + frozen-merge reconciliation: only the newly
//     changed tail block formats and mutates `NSTextStorage`.
//   - Height stability: `invalidateIntrinsicContentSize()` fires ONLY when the
//     laid-out wrapped height actually changes (line wrap or newline), never on
//     mid-line character appends — zero SwiftUI transcript relayouts mid-line.
//   - Bounded chunking (handled by `StreamingRichRenderModel`): each surface
//     caps at ~64–100 logical lines so CALayer backing stores stay compact and
//     TextKit layout/draw cost stays flat O(1) even on 5,000-line answers.

private extension NSAttributedString.Key {
    static let mtplxBlockquoteDepth = NSAttributedString.Key("com.mtplx.streaming.blockquoteDepth")
    static let mtplxHorizontalRule = NSAttributedString.Key("com.mtplx.streaming.horizontalRule")
}

// MARK: - StreamingProseMarkdownFormatter

/// Fast, single-pass, regex-free Markdown line formatter for live streaming
/// prose. Only styles matched inline delimiters (`**bold**`, `*italic*`,
/// `` `code` ``, `~~strike~~`, `[label](url)`) so unclosed streaming prefixes
/// never flicker while characters arrive at 120 Hz.
@MainActor
enum StreamingProseMarkdownFormatter {
    private static let typeHiColor = BrandPalette.nsColor(\.typeHi)
    private static let typeSecondaryColor = BrandPalette.nsColor(\.typeSecondary)
    private static let accentChromeColor = BrandPalette.nsColor(\.accentChrome)
    static let separatorStrongColor = BrandPalette.nsColor(\.separatorStrong)
    private static let inlineCodeBackgroundColor = NSColor(name: nil) { appearance in
        BrandPalette.palette(for: appearance).wash.nsColor.withAlphaComponent(0.06)
    }

    private static let bodyFont = NSFont.systemFont(ofSize: 14, weight: .regular)
    private static let bodyBoldFont = NSFont.systemFont(ofSize: 14, weight: .semibold)
    private static let bodyItalicFont = NSFontManager.shared.convert(bodyFont, toHaveTrait: .italicFontMask)
    private static let bodyBoldItalicFont = NSFontManager.shared.convert(bodyBoldFont, toHaveTrait: .italicFontMask)
    private static let inlineCodeFont = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
    private static let bulletMarkerFont = NSFont.systemFont(ofSize: 14, weight: .bold)
    private static let orderedMarkerFont = NSFont.monospacedSystemFont(ofSize: 14, weight: .semibold)
    private static let blankFont = NSFont.systemFont(ofSize: 6, weight: .regular)
    private static let ruleFont = NSFont.systemFont(ofSize: 10, weight: .regular)

    private static let headingFonts: [NSFont] = [
        NSFont.systemFont(ofSize: 21, weight: .heavy),
        NSFont.systemFont(ofSize: 17, weight: .heavy),
        NSFont.systemFont(ofSize: 15, weight: .bold),
        NSFont.systemFont(ofSize: 14, weight: .bold),
    ]

    static let separatorNewline = NSAttributedString(
        string: "\n",
        attributes: [
            .font: NSFont.systemFont(ofSize: 6, weight: .regular),
            .foregroundColor: typeHiColor,
        ]
    )

    private static let bodyParagraphStyle: NSParagraphStyle = {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 3
        return style
    }()

    private static let blankParagraphStyle: NSParagraphStyle = {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 0
        style.paragraphSpacing = 2
        return style
    }()

    private static let ruleParagraphStyle: NSParagraphStyle = {
        let style = NSMutableParagraphStyle()
        style.paragraphSpacingBefore = 6
        style.paragraphSpacing = 6
        return style
    }()

    private static let headingParagraphStyles: [NSParagraphStyle] = [
        makeHeadingStyle(before: 10, after: 3),
        makeHeadingStyle(before: 8, after: 2),
        makeHeadingStyle(before: 6, after: 2),
        makeHeadingStyle(before: 5, after: 2),
    ]

    nonisolated(unsafe) private static let finalizedLineCache: NSCache<NSString, NSAttributedString> = {
        let cache = NSCache<NSString, NSAttributedString>()
        cache.countLimit = 1_024
        return cache
    }()

    nonisolated static func clearCache() {
        finalizedLineCache.removeAllObjects()
    }

    private static func makeHeadingStyle(before: CGFloat, after: CGFloat) -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 2
        style.paragraphSpacingBefore = before
        style.paragraphSpacing = after
        return style
    }

    private static func makeListStyle(indentDepth: Int, markerWidth: CGFloat) -> NSParagraphStyle {
        let baseIndent = CGFloat(indentDepth) * 14
        let head = baseIndent + markerWidth
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 3
        style.firstLineHeadIndent = baseIndent
        style.headIndent = head
        style.tabStops = [NSTextTab(textAlignment: .left, location: head)]
        return style
    }

    private static func makeQuoteStyle(depth: Int) -> NSParagraphStyle {
        let indent = CGFloat(depth) * 12
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 3
        style.firstLineHeadIndent = indent
        style.headIndent = indent
        return style
    }

    /// Formats a document block (single logical line or `\n`-joined merged
    /// segment). Every logical line formats identically whether single or
    /// merged, so `reconcileFrozenMerges` stays byte-exact without touching
    /// `NSTextStorage`.
    static func formatBlock(_ text: String, finalized: Bool) -> NSAttributedString {
        guard text.contains("\n") else {
            return formatLine(text, finalized: finalized)
        }
        let result = NSMutableAttributedString()
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        for (index, rawLine) in lines.enumerated() {
            if index > 0 {
                result.append(separatorNewline)
            }
            result.append(formatLine(String(rawLine), finalized: finalized))
        }
        return result
    }

    static func formatLine(_ rawLine: String, finalized: Bool) -> NSAttributedString {
        if finalized, let cached = finalizedLineCache.object(forKey: rawLine as NSString) {
            return cached
        }
        let formatted = buildLine(rawLine)
        if finalized {
            finalizedLineCache.setObject(formatted, forKey: rawLine as NSString)
        }
        return formatted
    }

    private static func buildLine(_ rawLine: String) -> NSAttributedString {
        let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            return NSAttributedString(
                string: " ",
                attributes: [
                    .font: blankFont,
                    .foregroundColor: typeHiColor,
                    .paragraphStyle: blankParagraphStyle,
                ]
            )
        }

        if isHorizontalRule(trimmed) {
            return NSAttributedString(
                string: " ",
                attributes: [
                    .font: ruleFont,
                    .foregroundColor: separatorStrongColor,
                    .paragraphStyle: ruleParagraphStyle,
                    .mtplxHorizontalRule: true,
                ]
            )
        }

        if let heading = parseHeading(trimmed) {
            let levelIdx = min(max(heading.level - 1, 0), headingFonts.count - 1)
            let font = headingFonts[levelIdx]
            let style = headingParagraphStyles[levelIdx]
            return formatInline(
                heading.text,
                baseFont: font,
                boldFont: font,
                italicFont: NSFontManager.shared.convert(font, toHaveTrait: .italicFontMask),
                boldItalicFont: NSFontManager.shared.convert(font, toHaveTrait: .italicFontMask),
                foreground: typeHiColor,
                paragraphStyle: style,
                extraAttributes: [:]
            )
        }

        if let quote = parseBlockquote(trimmed) {
            let style = makeQuoteStyle(depth: quote.depth)
            let extra: [NSAttributedString.Key: Any] = [.mtplxBlockquoteDepth: quote.depth]
            if quote.text.isEmpty {
                var attrs: [NSAttributedString.Key: Any] = [
                    .font: bodyFont,
                    .foregroundColor: typeSecondaryColor,
                    .paragraphStyle: style,
                ]
                attrs[.mtplxBlockquoteDepth] = quote.depth
                return NSAttributedString(string: " ", attributes: attrs)
            }
            return formatInline(
                quote.text,
                baseFont: bodyFont,
                boldFont: bodyBoldFont,
                italicFont: bodyItalicFont,
                boldItalicFont: bodyBoldItalicFont,
                foreground: typeSecondaryColor,
                paragraphStyle: style,
                extraAttributes: extra
            )
        }

        let leadingSpaces = countLeadingSpaces(in: rawLine)
        let indentDepth = min(leadingSpaces / 2, 4)

        if let bulletText = parseBullet(trimmed) {
            let style = makeListStyle(indentDepth: indentDepth, markerWidth: 16)
            let output = NSMutableAttributedString(
                string: "•\t",
                attributes: [
                    .font: bulletMarkerFont,
                    .foregroundColor: typeSecondaryColor,
                    .paragraphStyle: style,
                ]
            )
            output.append(formatInline(
                bulletText,
                baseFont: bodyFont,
                boldFont: bodyBoldFont,
                italicFont: bodyItalicFont,
                boldItalicFont: bodyBoldItalicFont,
                foreground: typeHiColor,
                paragraphStyle: style,
                extraAttributes: [:]
            ))
            return output
        }

        if let ordered = parseOrdered(trimmed) {
            let markerWidth: CGFloat = ordered.marker.count <= 2 ? 26 : CGFloat(ordered.marker.count * 9 + 10)
            let style = makeListStyle(indentDepth: indentDepth, markerWidth: markerWidth)
            let output = NSMutableAttributedString(
                string: "\(ordered.marker)\t",
                attributes: [
                    .font: orderedMarkerFont,
                    .foregroundColor: typeSecondaryColor,
                    .paragraphStyle: style,
                ]
            )
            output.append(formatInline(
                ordered.text,
                baseFont: bodyFont,
                boldFont: bodyBoldFont,
                italicFont: bodyItalicFont,
                boldItalicFont: bodyBoldItalicFont,
                foreground: typeHiColor,
                paragraphStyle: style,
                extraAttributes: [:]
            ))
            return output
        }

        return formatInline(
            rawLine,
            baseFont: bodyFont,
            boldFont: bodyBoldFont,
            italicFont: bodyItalicFont,
            boldItalicFont: bodyBoldItalicFont,
            foreground: typeHiColor,
            paragraphStyle: bodyParagraphStyle,
            extraAttributes: [:]
        )
    }

    // MARK: Block-level line parsers

    private static func countLeadingSpaces(in line: String) -> Int {
        var count = 0
        for ch in line {
            if ch == " " { count += 1 }
            else if ch == "\t" { count += 2 }
            else { break }
        }
        return count
    }

    private static func isHorizontalRule(_ trimmed: String) -> Bool {
        guard trimmed.count >= 3 else { return false }
        var marker: Character?
        var markerCount = 0
        for ch in trimmed {
            if ch == " " || ch == "\t" { continue }
            guard ch == "-" || ch == "*" || ch == "_" else { return false }
            if let existing = marker {
                guard existing == ch else { return false }
            } else {
                marker = ch
            }
            markerCount += 1
        }
        return markerCount >= 3
    }

    private static func parseHeading(_ trimmed: String) -> (level: Int, text: String)? {
        var cursor = trimmed.startIndex
        var level = 0
        while cursor < trimmed.endIndex, trimmed[cursor] == "#", level < 6 {
            level += 1
            cursor = trimmed.index(after: cursor)
        }
        guard level > 0, cursor < trimmed.endIndex, trimmed[cursor].isWhitespace else {
            return nil
        }
        let body = trimmed[cursor...]
            .trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "#").union(.whitespaces))
        return body.isEmpty ? nil : (level, body)
    }

    private static func parseBlockquote(_ trimmed: String) -> (depth: Int, text: String)? {
        guard trimmed.hasPrefix(">") else { return nil }
        var cursor = trimmed.startIndex
        var depth = 0
        while cursor < trimmed.endIndex, trimmed[cursor] == ">", depth < 3 {
            depth += 1
            cursor = trimmed.index(after: cursor)
            while cursor < trimmed.endIndex, trimmed[cursor] == " " {
                cursor = trimmed.index(after: cursor)
            }
        }
        let rest = String(trimmed[cursor...]).trimmingCharacters(in: .whitespaces)
        return (depth, rest)
    }

    private static func parseBullet(_ trimmed: String) -> String? {
        guard let first = trimmed.first,
              first == "-" || first == "*" || first == "+",
              trimmed.count >= 2 else { return nil }
        let after = trimmed.index(after: trimmed.startIndex)
        guard trimmed[after].isWhitespace else { return nil }
        let body = trimmed[after...].trimmingCharacters(in: .whitespaces)
        return body.isEmpty ? nil : body
    }

    private static func parseOrdered(_ trimmed: String) -> (marker: String, text: String)? {
        var cursor = trimmed.startIndex
        while cursor < trimmed.endIndex, trimmed[cursor].isNumber {
            cursor = trimmed.index(after: cursor)
        }
        guard cursor > trimmed.startIndex,
              cursor < trimmed.endIndex,
              trimmed[cursor] == "." || trimmed[cursor] == ")" else { return nil }
        let after = trimmed.index(after: cursor)
        guard after < trimmed.endIndex, trimmed[after].isWhitespace else { return nil }
        let marker = String(trimmed[trimmed.startIndex...cursor])
        let body = trimmed[after...].trimmingCharacters(in: .whitespaces)
        return body.isEmpty ? nil : (marker, body)
    }

    // MARK: Inline Markdown scanner

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

    private static func formatInline(
        _ rawText: String,
        baseFont: NSFont,
        boldFont: NSFont,
        italicFont: NSFont,
        boldItalicFont: NSFont,
        foreground: NSColor,
        paragraphStyle: NSParagraphStyle,
        extraAttributes: [NSAttributedString.Key: Any]
    ) -> NSAttributedString {
        let text = withReadableInlineMath(rawText)
        var baseAttrs: [NSAttributedString.Key: Any] = [
            .font: baseFont,
            .foregroundColor: foreground,
            .paragraphStyle: paragraphStyle,
        ]
        for (k, v) in extraAttributes { baseAttrs[k] = v }

        // Fast path: pure ASCII/plain line with no inline markdown triggers.
        var hasSpecial = false
        for byte in text.utf8 {
            if byte == 0x2A || byte == 0x5F || byte == 0x60 || byte == 0x7E
                || byte == 0x5B || byte == 0x5C {
                hasSpecial = true
                break
            }
        }
        if !hasSpecial {
            return NSAttributedString(string: text, attributes: baseAttrs)
        }

        let output = NSMutableAttributedString()
        var plainBuffer = ""
        let chars = Array(text)
        let n = chars.count
        var i = 0

        func flushPlain() {
            guard !plainBuffer.isEmpty else { return }
            output.append(NSAttributedString(string: plainBuffer, attributes: baseAttrs))
            plainBuffer.removeAll(keepingCapacity: true)
        }

        while i < n {
            let ch = chars[i]

            // Backslash escape for markdown punctuation.
            if ch == "\\", i + 1 < n {
                let next = chars[i + 1]
                if next == "*" || next == "_" || next == "`" || next == "~"
                    || next == "[" || next == "]" || next == "\\" {
                    plainBuffer.append(next)
                    i += 2
                    continue
                }
            }

            // Inline code: `...`
            if ch == "`" {
                if let close = findClosingBacktick(in: chars, from: i + 1) {
                    if close > i + 1 {
                        flushPlain()
                        let codeText = String(chars[(i + 1)..<close])
                        var codeAttrs = baseAttrs
                        codeAttrs[.font] = inlineCodeFont
                        codeAttrs[.backgroundColor] = inlineCodeBackgroundColor
                        output.append(NSAttributedString(string: codeText, attributes: codeAttrs))
                        i = close + 1
                        continue
                    }
                }
                plainBuffer.append(ch)
                i += 1
                continue
            }

            // Markdown link: [label](url)
            if ch == "[" {
                if let link = parseInlineLink(in: chars, from: i) {
                    flushPlain()
                    var linkAttrs = baseAttrs
                    linkAttrs[.foregroundColor] = accentChromeColor
                    linkAttrs[.underlineStyle] = NSUnderlineStyle.single.rawValue
                    output.append(NSAttributedString(string: link.label, attributes: linkAttrs))
                    i = link.nextIndex
                    continue
                }
                plainBuffer.append(ch)
                i += 1
                continue
            }

            // Strikethrough: ~~...~~
            if ch == "~", i + 1 < n, chars[i + 1] == "~" {
                if let close = findClosingDouble(chars, marker: "~", from: i + 2) {
                    flushPlain()
                    let inner = String(chars[(i + 2)..<close])
                    var strikeAttrs = baseAttrs
                    strikeAttrs[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
                    output.append(NSAttributedString(string: inner, attributes: strikeAttrs))
                    i = close + 2
                    continue
                }
                plainBuffer.append(ch)
                i += 1
                continue
            }

            // Bold-italic / Bold / Italic: ***, **, *, ___, __, _
            if ch == "*" || ch == "_" {
                let isUnderscore = ch == "_"
                let prevIsWord = i > 0 && isWordChar(chars[i - 1])
                if !isUnderscore || !prevIsWord {
                    // ***...*** or ___...___
                    if i + 2 < n, chars[i + 1] == ch, chars[i + 2] == ch,
                       let close = findClosingTriple(chars, marker: ch, from: i + 3, requireWordBoundaryAfter: isUnderscore) {
                        flushPlain()
                        let inner = String(chars[(i + 3)..<close])
                        var biAttrs = baseAttrs
                        biAttrs[.font] = boldItalicFont
                        output.append(NSAttributedString(string: inner, attributes: biAttrs))
                        i = close + 3
                        continue
                    }
                    // **...** or __...__
                    if i + 1 < n, chars[i + 1] == ch,
                       let close = findClosingDouble(chars, marker: ch, from: i + 2, requireWordBoundaryAfter: isUnderscore) {
                        flushPlain()
                        let inner = String(chars[(i + 2)..<close])
                        var boldAttrs = baseAttrs
                        boldAttrs[.font] = boldFont
                        output.append(NSAttributedString(string: inner, attributes: boldAttrs))
                        i = close + 2
                        continue
                    }
                    // *...* or _..._
                    if let close = findClosingSingle(chars, marker: ch, from: i + 1, requireWordBoundaryAfter: isUnderscore) {
                        flushPlain()
                        let inner = String(chars[(i + 1)..<close])
                        var italicAttrs = baseAttrs
                        italicAttrs[.font] = italicFont
                        output.append(NSAttributedString(string: inner, attributes: italicAttrs))
                        i = close + 1
                        continue
                    }
                }
                plainBuffer.append(ch)
                i += 1
                continue
            }

            plainBuffer.append(ch)
            i += 1
        }

        flushPlain()
        return output.length > 0 ? output : NSAttributedString(string: " ", attributes: baseAttrs)
    }

    private static func isWordChar(_ ch: Character) -> Bool {
        ch.isLetter || ch.isNumber
    }

    private static func findClosingBacktick(in chars: [Character], from start: Int) -> Int? {
        var j = start
        while j < chars.count {
            if chars[j] == "`" { return j }
            j += 1
        }
        return nil
    }

    private static func findClosingSingle(
        _ chars: [Character],
        marker: Character,
        from start: Int,
        requireWordBoundaryAfter: Bool
    ) -> Int? {
        guard start < chars.count, !chars[start].isWhitespace, chars[start] != marker else { return nil }
        var j = start
        while j < chars.count {
            if chars[j] == "\\", j + 1 < chars.count {
                j += 2
                continue
            }
            if chars[j] == marker, j > start, !chars[j - 1].isWhitespace {
                let nextIsSame = j + 1 < chars.count && chars[j + 1] == marker
                let nextIsWord = j + 1 < chars.count && isWordChar(chars[j + 1])
                if !nextIsSame && (!requireWordBoundaryAfter || !nextIsWord) {
                    return j
                }
            }
            j += 1
        }
        return nil
    }

    private static func findClosingDouble(
        _ chars: [Character],
        marker: Character,
        from start: Int,
        requireWordBoundaryAfter: Bool = false
    ) -> Int? {
        guard start < chars.count, !chars[start].isWhitespace else { return nil }
        var j = start
        while j + 1 < chars.count {
            if chars[j] == "\\", j + 1 < chars.count {
                j += 2
                continue
            }
            if chars[j] == marker, chars[j + 1] == marker, j > start, !chars[j - 1].isWhitespace {
                let nextIsWord = j + 2 < chars.count && isWordChar(chars[j + 2])
                if !requireWordBoundaryAfter || !nextIsWord {
                    return j
                }
            }
            j += 1
        }
        return nil
    }

    private static func findClosingTriple(
        _ chars: [Character],
        marker: Character,
        from start: Int,
        requireWordBoundaryAfter: Bool
    ) -> Int? {
        guard start < chars.count, !chars[start].isWhitespace else { return nil }
        var j = start
        while j + 2 < chars.count {
            if chars[j] == marker, chars[j + 1] == marker, chars[j + 2] == marker,
               j > start, !chars[j - 1].isWhitespace {
                let nextIsWord = j + 3 < chars.count && isWordChar(chars[j + 3])
                if !requireWordBoundaryAfter || !nextIsWord {
                    return j
                }
            }
            j += 1
        }
        return nil
    }

    private static func parseInlineLink(
        in chars: [Character],
        from openBracket: Int
    ) -> (label: String, nextIndex: Int)? {
        let n = chars.count
        var j = openBracket + 1
        while j < n, chars[j] != "]" {
            if chars[j] == "[" { return nil }
            j += 1
        }
        guard j > openBracket + 1, j + 2 < n, chars[j + 1] == "(" else { return nil }
        var k = j + 2
        while k < n, chars[k] != ")" {
            if chars[k].isWhitespace { return nil }
            k += 1
        }
        guard k > j + 2, k < n else { return nil }
        let label = String(chars[(openBracket + 1)..<j])
        return (label, k + 1)
    }
}

// MARK: - StreamingProseNSView

/// Auto-sizing AppKit TextKit surface for one bounded prose chunk.
/// Invalidates intrinsic content size ONLY when wrapped line height changes,
/// keeping mid-line character appends completely invisible to SwiftUI layout.
@MainActor
final class StreamingProseNSView: NSView {
    let textStorage = NSTextStorage()
    private let layoutManager = NSLayoutManager()
    private let textContainer = NSTextContainer(size: NSSize(width: 1, height: CGFloat.greatestFiniteMagnitude))

    private(set) var measuredHeight: CGFloat = 1
    /// Test-visible counter of `invalidateIntrinsicContentSize()` calls so
    /// regression tests can prove zero SwiftUI relayouts mid-line.
    private(set) var intrinsicSizeInvalidationCount = 0

    override var isFlipped: Bool { true }

    override var intrinsicContentSize: NSSize {
        NSSize(width: NSView.noIntrinsicMetric, height: max(1, measuredHeight))
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.masksToBounds = false
        layerContentsRedrawPolicy = .onSetNeedsDisplay
        textContainer.lineFragmentPadding = 0
        textContainer.lineBreakMode = .byWordWrapping
        textContainer.widthTracksTextView = false
        textContainer.heightTracksTextView = false
        layoutManager.allowsNonContiguousLayout = false
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)
        if frameRect.width > 1 {
            textContainer.containerSize = NSSize(width: frameRect.width, height: .greatestFiniteMagnitude)
        }
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidChangeEffectiveAppearance() {
        super.viewDidChangeEffectiveAppearance()
        needsDisplay = true
    }

    override func layout() {
        super.layout()
        if bounds.width > 1 {
            updateWidthAndMeasureIfNeeded(bounds.width, notifyIntrinsicSize: true)
        }
    }

    func fittingHeight(forWidth width: CGFloat) -> CGFloat {
        let clamped = max(1, width)
        updateWidthAndMeasureIfNeeded(clamped, notifyIntrinsicSize: false)
        return max(1, measuredHeight)
    }

    func textDidChange(editedCharacterLocation: Int) {
        guard textContainer.containerSize.width > 1 else {
            needsDisplay = true
            return
        }
        let nextHeight = computeLaidOutHeight()
        if abs(nextHeight - measuredHeight) >= 0.5 {
            measuredHeight = nextHeight
            intrinsicSizeInvalidationCount += 1
            invalidateIntrinsicContentSize()
            needsDisplay = true
            return
        }
        // Height unchanged (mid-line append): zero SwiftUI relayout; only
        // dirty the vertical band from the edited line to the bottom.
        if let dirty = dirtyBand(fromCharacterLocation: editedCharacterLocation) {
            setNeedsDisplay(dirty)
        } else {
            needsDisplay = true
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard textStorage.length > 0 else { return }

        UIStreamPerfProbe.renderTimed(.draw, size: textStorage.length) {
            if layoutManager.firstUnlaidCharacterIndex() < textStorage.length {
                layoutManager.ensureLayout(for: textContainer)
            }
            let glyphRange = layoutManager.glyphRange(
                forBoundingRect: dirtyRect,
                in: textContainer
            )
            guard glyphRange.length > 0 else { return }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current?.cgContext.clip(to: bounds)
            layoutManager.drawBackground(forGlyphRange: glyphRange, at: .zero)
            layoutManager.drawGlyphs(forGlyphRange: glyphRange, at: .zero)
            drawDecorations(forGlyphRange: glyphRange)
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    private func drawDecorations(forGlyphRange glyphRange: NSRange) {
        let charRange = layoutManager.characterRange(forGlyphRange: glyphRange, actualGlyphRange: nil)
        guard charRange.length > 0 else { return }

        // Blockquote vertical accent bars.
        textStorage.enumerateAttribute(.mtplxBlockquoteDepth, in: charRange, options: []) { value, range, _ in
            guard let depth = value as? Int, depth > 0 else { return }
            let attrGlyphs = layoutManager.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            StreamingProseMarkdownFormatter.separatorStrongColor.setFill()
            layoutManager.enumerateLineFragments(forGlyphRange: attrGlyphs) { lineRect, _, _, _, _ in
                for level in 1...depth {
                    let barRect = NSRect(
                        x: CGFloat(level - 1) * 12 + 2,
                        y: lineRect.minY + 1,
                        width: 2,
                        height: max(2, lineRect.height - 2)
                    )
                    barRect.fill()
                }
            }
        }

        // Horizontal rules.
        let viewWidth = bounds.width
        textStorage.enumerateAttribute(.mtplxHorizontalRule, in: charRange, options: []) { value, range, _ in
            guard (value as? Bool) == true else { return }
            let attrGlyphs = layoutManager.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            StreamingProseMarkdownFormatter.separatorStrongColor.setFill()
            layoutManager.enumerateLineFragments(forGlyphRange: attrGlyphs) { lineRect, _, _, _, _ in
                let ruleRect = NSRect(
                    x: 0,
                    y: floor(lineRect.midY),
                    width: viewWidth,
                    height: 1
                )
                ruleRect.fill()
            }
        }
    }

    private func updateWidthAndMeasureIfNeeded(_ width: CGFloat, notifyIntrinsicSize: Bool) {
        let targetSize = NSSize(width: width, height: .greatestFiniteMagnitude)
        let widthChanged = abs(textContainer.containerSize.width - targetSize.width) >= 0.5
        if widthChanged {
            textContainer.containerSize = targetSize
            if textStorage.length > 0 {
                layoutManager.invalidateLayout(
                    forCharacterRange: NSRange(location: 0, length: textStorage.length),
                    actualCharacterRange: nil
                )
            }
        }
        let nextHeight = computeLaidOutHeight()
        if abs(nextHeight - measuredHeight) >= 0.5 {
            measuredHeight = nextHeight
            if notifyIntrinsicSize {
                intrinsicSizeInvalidationCount += 1
                invalidateIntrinsicContentSize()
            }
            needsDisplay = true
        } else if widthChanged {
            needsDisplay = true
        }
    }

    private func computeLaidOutHeight() -> CGFloat {
        guard textStorage.length > 0 else { return 1 }
        let lastRange = NSRange(location: textStorage.length - 1, length: 1)
        layoutManager.ensureLayout(forCharacterRange: lastRange)
        let glyphRange = layoutManager.glyphRange(forCharacterRange: lastRange, actualCharacterRange: nil)
        var height: CGFloat = 0
        if glyphRange.length > 0 {
            let usedRect = layoutManager.lineFragmentRect(
                forGlyphAt: NSMaxRange(glyphRange) - 1,
                effectiveRange: nil
            )
            height = usedRect.maxY
        }
        if layoutManager.extraLineFragmentTextContainer === textContainer {
            height = max(height, layoutManager.extraLineFragmentRect.maxY)
        }
        return max(1, ceil(height))
    }

    private func dirtyBand(fromCharacterLocation location: Int) -> NSRect? {
        guard textStorage.length > 0, location > 0, location < textStorage.length,
              bounds.width > 0, bounds.height > 0 else { return nil }
        let safeLoc = min(location, textStorage.length - 1)
        let charRange = NSRange(location: safeLoc, length: 1)
        layoutManager.ensureLayout(forCharacterRange: charRange)
        let glyphRange = layoutManager.glyphRange(forCharacterRange: charRange, actualCharacterRange: nil)
        guard glyphRange.length > 0 else { return nil }
        let lineRect = layoutManager.lineFragmentRect(forGlyphAt: glyphRange.location, effectiveRange: nil)
        let topY = max(0, floor(lineRect.minY) - 4)
        return NSRect(
            x: 0,
            y: topY,
            width: bounds.width,
            height: max(1, bounds.height - topY)
        )
    }
}

// MARK: - StreamingProseTextSurface (NSViewRepresentable)

struct StreamingProseTextSurface: NSViewRepresentable {
    let renderModel: StreamingRichRenderModel
    let proseID: Int

    @MainActor
    final class Coordinator {
        struct AppliedBlock {
            let id: Int
            let text: String
            let finalized: Bool
            let renderedUTF16Length: Int
        }

        weak var surface: StreamingProseNSView?
        var renderModel: StreamingRichRenderModel?
        var proseID: Int?
        var applied: [AppliedBlock] = []
        var itemsCancellable: AnyCancellable?

        func attach(
            renderModel: StreamingRichRenderModel,
            proseID: Int,
            surface: StreamingProseNSView
        ) {
            let surfaceChanged = self.surface !== surface
            self.surface = surface
            let sameSource = self.renderModel === renderModel && self.proseID == proseID
            if !sameSource || surfaceChanged {
                itemsCancellable?.cancel()
                self.renderModel = renderModel
                self.proseID = proseID
                applied.removeAll(keepingCapacity: true)
                itemsCancellable = renderModel.perRevision.sink { [weak self] items in
                    self?.refresh(items: items)
                }
            }
            refresh(items: renderModel.latestItems)
        }

        private func refresh(items: [StreamingRenderItem]) {
            guard let surface, let proseID else { return }
            for item in items {
                guard case .prose(let id, let blocks) = item, id == proseID else { continue }
                UIStreamPerfProbe.renderTimed(.applyRender, size: blocks.count) {
                    StreamingProseTextSurface.apply(
                        blocks: blocks,
                        to: surface,
                        coordinator: self
                    )
                }
                return
            }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> StreamingProseNSView {
        let surface = StreamingProseNSView(frame: .zero)
        surface.setAccessibilityElement(false)
        context.coordinator.attach(
            renderModel: renderModel,
            proseID: proseID,
            surface: surface
        )
        return surface
    }

    func updateNSView(_ surface: StreamingProseNSView, context: Context) {
        context.coordinator.attach(
            renderModel: renderModel,
            proseID: proseID,
            surface: surface
        )
    }

    func sizeThatFits(
        _ proposal: ProposedViewSize,
        nsView: StreamingProseNSView,
        context: Context
    ) -> CGSize? {
        let width = proposal.width ?? nsView.bounds.width
        guard width.isFinite, width > 1 else {
            return CGSize(width: max(1, proposal.width ?? 1), height: max(1, nsView.measuredHeight))
        }
        let height = nsView.fittingHeight(forWidth: width)
        return CGSize(width: width, height: height)
    }

    /// Incrementally applies `blocks` into `surface.textStorage`. Frozen
    /// prefix blocks and coalesced line-segment merges skip formatting and
    /// TextKit mutation entirely; only the changed tail formats and updates.
    @MainActor
    static func apply(
        blocks: [StreamingDocumentBlock],
        to surface: StreamingProseNSView,
        coordinator: Coordinator
    ) {
        reconcileFrozenMerges(blocks: blocks, coordinator: coordinator)

        var common = 0
        let pairCount = min(coordinator.applied.count, blocks.count)
        while common < pairCount {
            let old = coordinator.applied[common]
            let new = blocks[common]
            guard old.id == new.id, old.text == new.text else { break }
            common += 1
        }

        if common == coordinator.applied.count, common == blocks.count {
            return
        }

        let storage = surface.textStorage
        let unchangedUTF16 = coordinator.applied
            .prefix(common)
            .reduce(0) { $0 + $1.renderedUTF16Length }

        let suffix = NSMutableAttributedString()
        var nextApplied = Array(coordinator.applied.prefix(common))
        nextApplied.reserveCapacity(blocks.count)

        for index in common..<blocks.count {
            let block = blocks[index]
            let hasSeparator = index > 0
            if hasSeparator {
                suffix.append(StreamingProseMarkdownFormatter.separatorNewline)
            }
            let formatted = StreamingProseMarkdownFormatter.formatBlock(
                block.text,
                finalized: block.finalized
            )
            suffix.append(formatted)
            nextApplied.append(Coordinator.AppliedBlock(
                id: block.id,
                text: block.text,
                finalized: block.finalized,
                renderedUTF16Length: (hasSeparator ? 1 : 0) + formatted.length
            ))
        }

        let replaceLocation = min(unchangedUTF16, storage.length)
        let replaceLength = max(0, storage.length - replaceLocation)
        storage.beginEditing()
        storage.replaceCharacters(
            in: NSRange(location: replaceLocation, length: replaceLength),
            with: suffix
        )
        storage.endEditing()
        coordinator.applied = nextApplied
        surface.textDidChange(editedCharacterLocation: replaceLocation)
    }

    /// Collapses coordinator metadata when `StreamingDocumentStore` folds a
    /// contiguous run of frozen lines into one multiline segment block.
    /// Because each line formats identically whether single or merged, the
    /// storage is already exact and needs zero mutation or redraw.
    @MainActor
    private static func reconcileFrozenMerges(
        blocks: [StreamingDocumentBlock],
        coordinator: Coordinator
    ) {
        guard !coordinator.applied.isEmpty, !blocks.isEmpty else { return }

        let old = coordinator.applied
        var normalized: [Coordinator.AppliedBlock] = []
        normalized.reserveCapacity(blocks.count)
        var oldIndex = 0
        var newIndex = 0

        while oldIndex < old.count, newIndex < blocks.count {
            let previous = old[oldIndex]
            let current = blocks[newIndex]

            if previous.id == current.id, previous.text == current.text {
                normalized.append(previous)
                oldIndex += 1
                newIndex += 1
                continue
            }

            guard previous.id == current.id, current.text.contains("\n") else { break }

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
                    normalized.append(Coordinator.AppliedBlock(
                        id: current.id,
                        text: current.text,
                        finalized: current.finalized,
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
