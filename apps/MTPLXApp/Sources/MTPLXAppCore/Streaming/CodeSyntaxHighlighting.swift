import AppKit
import Foundation

// MARK: - CodeSyntaxHighlighting
//
// Freeze-time syntax coloring for code (2026-07-31 founder ask:
// "syntax coloring without decreasing performance").
//
// The perf contract that makes coloring ≈free rides the streaming
// architecture that already exists: lines freeze exactly once, so each
// line is lexed exactly once, at the moment it stops changing — cost
// is O(new line) per flush (single-digit microseconds for a code
// line), never O(document). Everything is cached by (language, entry
// state, text), so re-evaluated SwiftUI bodies hit the cache and the
// settled transcript reuses the exact same runs.
//
// This is a deliberate line-state lexer (strings / comments / keywords
// / numbers / calls), NOT a grammar engine: TextMate/regex grammars
// are 10-100x slower and are how other apps end up with laggy
// highlighted streams. Fidelity target is "Xcode-adjacent", not
// perfect parsing; unknown languages fall back to a generic C-like
// ruleset, and anything unhandled renders in the base color exactly as
// before.
//
// Rendering colored text costs the same as plain text at draw time —
// glyph layout dominates; color runs are nearly free. The A/B gate
// (ui_turn_render_summary flush-gap thirds + stall census) is the
// enforcement that this stays true.

/// Render-layer switchboard for performance mode (founder contract
/// 2026-07-31: performance lock ⇒ syntax coloring off AND markdown
/// off). Views that observe MTPLXBackendStore pass the flag down as a
/// parameter; this mirror covers deep leaves (MarkdownUI theme
/// closures, NSView viewports) that can't take a parameter. Kept in
/// sync by ChatView on configuration changes.
@MainActor
public enum ChatRenderPreferences {
    public static var plainTextOnly = false
}

public enum MTPLXCodeHighlighter {

    // MARK: Language

    public enum Language: String, Hashable, CaseIterable, Sendable {
        case python, swift, javascript, typescript, json, bash, c, cpp
        case rust, go, html, css, generic

        /// Maps a fence label ("python", "py", "c++", "shell"…) to a
        /// lexer language. Unknown labels get the generic C-like rules.
        public static func detect(fromFenceLabel label: String?) -> Language {
            guard let label = label?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(),
                !label.isEmpty
            else { return .generic }
            switch label {
            case "python", "py", "python3", "pygame": return .python
            case "swift": return .swift
            case "javascript", "js", "jsx", "node": return .javascript
            case "typescript", "ts", "tsx": return .typescript
            case "json", "jsonc": return .json
            case "bash", "sh", "zsh", "shell", "console": return .bash
            case "c", "h", "objc", "objective-c", "m": return .c
            case "cpp", "c++", "cc", "hpp", "cxx": return .cpp
            case "rust", "rs": return .rust
            case "go", "golang": return .go
            case "html", "xml", "svg": return .html
            case "css", "scss", "less": return .css
            default: return .generic
            }
        }
    }

    // MARK: Cross-line lexer state

    /// The only state that survives a line boundary. Tiny by design —
    /// it keys the per-line cache together with the text.
    public enum LexState: Hashable, Sendable {
        case none
        case blockComment          // /* ... */  (c-family, swift, js, css)
        case tripleString(Character) // ''' or """ (python)

        public var cacheTag: String {
            switch self {
            case .none: return "n"
            case .blockComment: return "b"
            case .tripleString(let q): return q == "'" ? "s" : "d"
            }
        }

        public init(cacheTag: String) {
            switch cacheTag {
            case "b": self = .blockComment
            case "s": self = .tripleString("'")
            case "d": self = .tripleString("\"")
            default: self = .none
            }
        }
    }

    // MARK: Palette

    /// Fixed dark-theme palette (matches the app's code viewport
    /// background). Kept here rather than on Brand so Core stays
    /// self-contained and unit-testable.
    public struct Palette: Sendable {
        public let base: NSColor
        public let keyword: NSColor
        public let type: NSColor
        public let string: NSColor
        public let comment: NSColor
        public let number: NSColor
        public let function: NSColor
        public let decorator: NSColor

        public static let dark = Palette(
            base: NSColor(calibratedWhite: 0.88, alpha: 1.0),
            keyword: NSColor(srgbRed: 1.00, green: 0.48, blue: 0.70, alpha: 1.0),
            type: NSColor(srgbRed: 0.42, green: 0.87, blue: 1.00, alpha: 1.0),
            string: NSColor(srgbRed: 0.99, green: 0.64, blue: 0.41, alpha: 1.0),
            comment: NSColor(srgbRed: 0.50, green: 0.55, blue: 0.60, alpha: 1.0),
            number: NSColor(srgbRed: 0.82, green: 0.66, blue: 1.00, alpha: 1.0),
            function: NSColor(srgbRed: 0.31, green: 0.69, blue: 1.00, alpha: 1.0),
            decorator: NSColor(srgbRed: 0.85, green: 0.80, blue: 0.47, alpha: 1.0)
        )
    }

    // NSFont isn't Sendable; fonts are cheap lookups, so derive per use.
    public static var codeFont: NSFont {
        NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
    }

    // MARK: Public API

    /// Highlight ONE line (no trailing newline). Returns the attributed
    /// line and the lexer state at end-of-line. Cached — repeated calls
    /// with the same (language, state, text) are a dictionary hit.
    public static func highlightLine(
        _ text: String,
        language: Language,
        state: LexState
    ) -> (line: NSAttributedString, endState: LexState) {
        let key = "\(language.rawValue)|\(state.cacheTag)|\(text)" as NSString
        if let hit = lineCache.object(forKey: key) {
            return (hit.attributed, hit.endState)
        }
        let result = lex(text, language: language, entryState: state)
        lineCache.setObject(
            CachedLine(attributed: result.line, endState: result.endState),
            forKey: key,
            cost: text.utf8.count
        )
        return result
    }

    /// Highlight a whole multi-line body (settled code blocks, merged
    /// segments). Lines are lexed with carried state and joined; the
    /// full result is cached by (language, entry state, content).
    public static func highlightCode(
        _ code: String,
        language: Language,
        entryState: LexState = .none
    ) -> NSAttributedString {
        let key = "\(language.rawValue)|#\(entryState.cacheTag)|\(code)" as NSString
        if let hit = blockCache.object(forKey: key) {
            return hit
        }
        let joined = NSMutableAttributedString()
        var state = entryState
        var first = true
        for line in code.split(separator: "\n", omittingEmptySubsequences: false) {
            if !first {
                joined.append(NSAttributedString(
                    string: "\n",
                    attributes: [.font: codeFont]
                ))
            }
            first = false
            let out = highlightLine(String(line), language: language, state: state)
            joined.append(out.line)
            state = out.endState
        }
        blockCache.setObject(joined, forKey: key, cost: code.utf8.count)
        return joined
    }

    /// Attributed fragment for a streaming block: a single line or a
    /// merged multi-line segment, lexed from the entry state carried
    /// across the fence. Cache-backed both ways.
    public static func highlightedFragment(
        _ text: String,
        language: Language,
        entryTag: String
    ) -> NSAttributedString {
        let state = LexState(cacheTag: entryTag)
        if text.contains("\n") {
            return highlightCode(text, language: language, entryState: state)
        }
        return highlightLine(text, language: language, state: state).line
    }

    /// Lexer state after a multi-line segment (for threading state
    /// across merged blocks without materializing their runs).
    public static func highlightSegmentEndState(
        _ text: String,
        language: Language,
        state: LexState
    ) -> LexState {
        var current = state
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            current = highlightLine(String(line), language: language, state: current).endState
        }
        return current
    }

    public static func clearCaches() {
        lineCache.removeAllObjects()
        blockCache.removeAllObjects()
    }

    // MARK: Caches

    private final class CachedLine: NSObject {
        let attributed: NSAttributedString
        let endState: LexState
        init(attributed: NSAttributedString, endState: LexState) {
            self.attributed = attributed
            self.endState = endState
        }
    }

    nonisolated(unsafe) private static let lineCache: NSCache<NSString, CachedLine> = {
        let cache = NSCache<NSString, CachedLine>()
        cache.countLimit = 8_192
        cache.totalCostLimit = 8_000_000
        return cache
    }()

    nonisolated(unsafe) private static let blockCache: NSCache<NSString, NSAttributedString> = {
        let cache = NSCache<NSString, NSAttributedString>()
        cache.countLimit = 128
        cache.totalCostLimit = 16_000_000
        return cache
    }()

    // MARK: Lexer

    private struct Rules {
        let lineComment: String?
        let hasBlockComment: Bool      // /* */
        let hasTripleString: Bool      // python
        let stringQuotes: Set<Character>
        let keywords: Set<String>
        let secondaryKeywords: Set<String>  // builtins/constants -> type color
    }

    private static func rules(for language: Language) -> Rules {
        switch language {
        case .python:
            return Rules(
                lineComment: "#", hasBlockComment: false, hasTripleString: true,
                stringQuotes: ["\"", "'"],
                keywords: [
                    "def", "class", "return", "if", "elif", "else", "for", "while",
                    "break", "continue", "pass", "import", "from", "as", "with",
                    "try", "except", "finally", "raise", "lambda", "yield", "global",
                    "nonlocal", "assert", "del", "in", "not", "and", "or", "is",
                    "async", "await", "match", "case"
                ],
                secondaryKeywords: [
                    "True", "False", "None", "self", "cls", "print", "len", "range",
                    "int", "float", "str", "list", "dict", "set", "tuple", "bool",
                    "super", "isinstance", "enumerate", "zip", "map", "filter",
                    "min", "max", "abs", "sum", "round", "open", "type"
                ]
            )
        case .swift:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\""],
                keywords: [
                    "func", "let", "var", "if", "else", "guard", "for", "while",
                    "repeat", "switch", "case", "default", "break", "continue",
                    "return", "struct", "class", "enum", "protocol", "extension",
                    "import", "public", "private", "internal", "fileprivate", "open",
                    "static", "final", "override", "init", "deinit", "throws",
                    "throw", "try", "catch", "async", "await", "actor", "in", "where",
                    "some", "any", "nil", "true", "false", "self", "Self", "weak",
                    "lazy", "mutating", "nonisolated", "defer", "typealias"
                ],
                secondaryKeywords: [
                    "String", "Int", "Double", "Bool", "Array", "Dictionary", "Set",
                    "Optional", "Void", "Character", "Float", "CGFloat", "Data", "URL"
                ]
            )
        case .javascript, .typescript:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\"", "'", "`"],
                keywords: [
                    "function", "const", "let", "var", "if", "else", "for", "while",
                    "do", "switch", "case", "default", "break", "continue", "return",
                    "class", "extends", "new", "delete", "typeof", "instanceof",
                    "in", "of", "import", "export", "from", "as", "async", "await",
                    "yield", "try", "catch", "finally", "throw", "this", "super",
                    "null", "undefined", "true", "false", "static", "get", "set",
                    "interface", "type", "enum", "implements", "readonly", "public",
                    "private", "protected", "namespace", "declare", "void"
                ],
                secondaryKeywords: [
                    "console", "window", "document", "Math", "JSON", "Object",
                    "Array", "String", "Number", "Boolean", "Promise", "Map", "Set"
                ]
            )
        case .json:
            return Rules(
                lineComment: nil, hasBlockComment: false, hasTripleString: false,
                stringQuotes: ["\""],
                keywords: ["true", "false", "null"],
                secondaryKeywords: []
            )
        case .bash:
            return Rules(
                lineComment: "#", hasBlockComment: false, hasTripleString: false,
                stringQuotes: ["\"", "'"],
                keywords: [
                    "if", "then", "else", "elif", "fi", "for", "in", "do", "done",
                    "while", "until", "case", "esac", "function", "return", "local",
                    "export", "source", "set", "unset", "readonly", "shift", "exit",
                    "echo", "cd", "true", "false"
                ],
                secondaryKeywords: []
            )
        case .c, .cpp:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\"", "'"],
                keywords: [
                    "int", "char", "float", "double", "void", "long", "short",
                    "signed", "unsigned", "struct", "union", "enum", "typedef",
                    "const", "static", "extern", "register", "volatile", "inline",
                    "if", "else", "for", "while", "do", "switch", "case", "default",
                    "break", "continue", "return", "goto", "sizeof", "class",
                    "public", "private", "protected", "virtual", "override", "new",
                    "delete", "namespace", "using", "template", "typename", "auto",
                    "nullptr", "true", "false", "this", "constexpr", "noexcept"
                ],
                secondaryKeywords: ["std", "size_t", "uint32_t", "int32_t", "uint64_t", "int64_t", "bool"]
            )
        case .rust:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\""],
                keywords: [
                    "fn", "let", "mut", "const", "static", "if", "else", "match",
                    "for", "while", "loop", "break", "continue", "return", "struct",
                    "enum", "trait", "impl", "pub", "use", "mod", "crate", "self",
                    "Self", "super", "where", "async", "await", "move", "ref",
                    "true", "false", "unsafe", "dyn", "as", "in", "type"
                ],
                secondaryKeywords: [
                    "String", "Vec", "Option", "Result", "Some", "None", "Ok",
                    "Err", "Box", "i32", "i64", "u32", "u64", "f32", "f64", "usize", "bool", "str"
                ]
            )
        case .go:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\"", "'", "`"],
                keywords: [
                    "func", "var", "const", "type", "struct", "interface", "map",
                    "chan", "if", "else", "for", "range", "switch", "case",
                    "default", "break", "continue", "return", "go", "defer",
                    "select", "package", "import", "nil", "true", "false", "make",
                    "new", "len", "cap", "append"
                ],
                secondaryKeywords: ["string", "int", "int64", "float64", "bool", "byte", "error", "rune"]
            )
        case .html:
            return Rules(
                lineComment: nil, hasBlockComment: false, hasTripleString: false,
                stringQuotes: ["\"", "'"],
                keywords: [],
                secondaryKeywords: []
            )
        case .css:
            return Rules(
                lineComment: nil, hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\"", "'"],
                keywords: [],
                secondaryKeywords: []
            )
        case .generic:
            return Rules(
                lineComment: "//", hasBlockComment: true, hasTripleString: false,
                stringQuotes: ["\"", "'"],
                keywords: [
                    "if", "else", "for", "while", "return", "function", "func",
                    "def", "class", "struct", "let", "var", "const", "import",
                    "true", "false", "null", "nil", "new", "break", "continue",
                    "switch", "case", "try", "catch", "throw", "public", "private"
                ],
                secondaryKeywords: []
            )
        }
    }

    private static func lex(
        _ text: String,
        language: Language,
        entryState: LexState
    ) -> (line: NSAttributedString, endState: LexState) {
        let palette = Palette.dark
        let rules = rules(for: language)
        let out = NSMutableAttributedString()
        let chars = Array(text)
        var i = 0
        var state = entryState

        func emit(_ range: Range<Int>, _ color: NSColor) {
            guard !range.isEmpty else { return }
            out.append(NSAttributedString(
                string: String(chars[range]),
                attributes: [.font: codeFont, .foregroundColor: color]
            ))
        }

        // Resume an open container from the previous line.
        switch state {
        case .blockComment:
            var end = chars.count
            var closed = false
            var j = 0
            while j + 1 < chars.count {
                if chars[j] == "*" && chars[j + 1] == "/" {
                    end = j + 2
                    closed = true
                    break
                }
                j += 1
            }
            emit(0..<end, palette.comment)
            if !closed {
                return (out, .blockComment)
            }
            i = end
            state = .none
        case .tripleString(let quote):
            var end = chars.count
            var closed = false
            var j = 0
            while j + 2 < chars.count + 1 {
                if j + 2 < chars.count + 1, j + 3 <= chars.count,
                   chars[j] == quote, chars[j + 1] == quote, chars[j + 2] == quote {
                    end = j + 3
                    closed = true
                    break
                }
                j += 1
            }
            emit(0..<end, palette.string)
            if !closed {
                return (out, .tripleString(quote))
            }
            i = end
            state = .none
        case .none:
            break
        }

        var plainStart = i

        func flushPlain(upTo end: Int) {
            emit(plainStart..<end, palette.base)
        }

        while i < chars.count {
            let c = chars[i]

            // Line comment
            if let lc = rules.lineComment, matches(chars, at: i, lc) {
                flushPlain(upTo: i)
                emit(i..<chars.count, palette.comment)
                return (out, .none)
            }

            // Block comment open
            if rules.hasBlockComment, matches(chars, at: i, "/*") {
                flushPlain(upTo: i)
                var j = i + 2
                while j + 1 < chars.count {
                    if chars[j] == "*" && chars[j + 1] == "/" {
                        emit(i..<(j + 2), palette.comment)
                        i = j + 2
                        plainStart = i
                        break
                    }
                    j += 1
                }
                if j + 1 >= chars.count {
                    emit(i..<chars.count, palette.comment)
                    return (out, .blockComment)
                }
                continue
            }

            // Python triple-quoted string
            if rules.hasTripleString, c == "\"" || c == "'",
               matches(chars, at: i, String(repeating: String(c), count: 3)) {
                flushPlain(upTo: i)
                var j = i + 3
                var closed = false
                while j + 2 < chars.count + 1, j + 3 <= chars.count {
                    if chars[j] == c, chars[j + 1] == c, chars[j + 2] == c {
                        emit(i..<(j + 3), palette.string)
                        i = j + 3
                        plainStart = i
                        closed = true
                        break
                    }
                    j += 1
                }
                if !closed {
                    emit(i..<chars.count, palette.string)
                    return (out, .tripleString(c))
                }
                continue
            }

            // Single-line string
            if rules.stringQuotes.contains(c) {
                flushPlain(upTo: i)
                var j = i + 1
                var closed = false
                while j < chars.count {
                    if chars[j] == "\\" { j += 2; continue }
                    if chars[j] == c { closed = true; break }
                    j += 1
                }
                let end = closed ? j + 1 : chars.count
                emit(i..<end, palette.string)
                i = end
                plainStart = i
                continue
            }

            // Number
            if c.isNumber, i == 0 || !isWordChar(chars[i - 1]) {
                flushPlain(upTo: i)
                var j = i + 1
                while j < chars.count,
                      chars[j].isHexDigit || chars[j] == "." || chars[j] == "_"
                        || chars[j] == "x" || chars[j] == "X" || chars[j] == "o"
                        || chars[j] == "b" || chars[j] == "e" || chars[j] == "E" {
                    j += 1
                }
                emit(i..<j, palette.number)
                i = j
                plainStart = i
                continue
            }

            // Decorator / attribute (@something)
            if c == "@", i + 1 < chars.count, isWordStart(chars[i + 1]) {
                flushPlain(upTo: i)
                var j = i + 1
                while j < chars.count, isWordChar(chars[j]) || chars[j] == "." {
                    j += 1
                }
                emit(i..<j, palette.decorator)
                i = j
                plainStart = i
                continue
            }

            // Identifier / keyword / call / Type
            if isWordStart(c) {
                var j = i + 1
                while j < chars.count, isWordChar(chars[j]) {
                    j += 1
                }
                let word = String(chars[i..<j])
                let color: NSColor?
                if rules.keywords.contains(word) {
                    color = palette.keyword
                } else if rules.secondaryKeywords.contains(word) {
                    color = palette.type
                } else if let first = word.first, first.isUppercase {
                    color = palette.type
                } else {
                    // function call: next non-space char is '('
                    var k = j
                    while k < chars.count, chars[k] == " " { k += 1 }
                    color = (k < chars.count && chars[k] == "(") ? palette.function : nil
                }
                if let color {
                    flushPlain(upTo: i)
                    emit(i..<j, color)
                    plainStart = j
                }
                i = j
                continue
            }

            i += 1
        }

        flushPlain(upTo: chars.count)
        return (out, .none)
    }

    private static func matches(_ chars: [Character], at index: Int, _ needle: String) -> Bool {
        let n = Array(needle)
        guard index + n.count <= chars.count else { return false }
        for (offset, ch) in n.enumerated() where chars[index + offset] != ch {
            return false
        }
        return true
    }

    private static func isWordStart(_ c: Character) -> Bool {
        c.isLetter || c == "_"
    }

    private static func isWordChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_"
    }
}
