import Foundation

/// Line-level TOML editing that keeps everything it does not own.
///
/// Swift twin of `mtplx/toml_lines.py`. MTPLX edits a few keys in files that
/// people and other programs also write: the model folders in
/// `~/.mtplx/config.toml` (shared by the CLI and the app) and MTPLX's
/// provider in the Codex desktop instance's `config.toml`. A file is split
/// into statements (a table header, a key whose value may span lines, or a
/// comment/blank line) so only the caller's own statements are replaced.
/// `Tests/Fixtures/toml_lines_cases.json` and `agent_client_plans.json`,
/// written from the Python module, keep the two byte-identical.
public enum TOMLLines {
    public enum Value: Equatable, Sendable {
        case string(String)
        case strings([String])
    }

    struct Statement: Equatable {
        let table: String
        let key: String?
        let lines: [String]
    }

    static func pyStrip(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Bracket depth after `text`, outside strings and comments; `quote` is a
    /// multi-line string delimiter carried over from the line before.
    static func valueDepth(_ text: String, depth start: Int, quote open: String?) -> (Int, String?) {
        let chars = Array(text.unicodeScalars)
        var depth = start
        var quote = open
        var i = 0
        func startsWith(_ token: String, at index: Int) -> Bool {
            let pattern = Array(token.unicodeScalars)
            guard index + pattern.count <= chars.count else { return false }
            return Array(chars[index..<(index + pattern.count)]) == pattern
        }
        while i < chars.count {
            if let open = quote {
                var found: Int?
                var j = i
                while j < chars.count {
                    if startsWith(open, at: j) { found = j; break }
                    j += 1
                }
                guard let end = found else { return (depth, quote) }
                i = end + open.unicodeScalars.count
                quote = nil
                continue
            }
            let ch = chars[i]
            if startsWith("\"\"\"", at: i) || startsWith("'''", at: i) {
                quote = String(String.UnicodeScalarView(chars[i..<(i + 3)]))
                i += 3
                continue
            }
            if ch == "\"" {
                i += 1
                while i < chars.count, chars[i] != "\"" {
                    i += chars[i] == "\\" ? 2 : 1
                }
                i += 1
                continue
            }
            if ch == "'" {
                var j = i + 1
                while j < chars.count, chars[j] != "'" { j += 1 }
                i = j < chars.count ? j + 1 : chars.count
                continue
            }
            if ch == "#" { break }
            if ch == "[" || ch == "{" { depth += 1 }
            if ch == "]" || ch == "}" { depth -= 1 }
            i += 1
        }
        return (depth, quote)
    }

    static func statements(_ text: String) -> [Statement] {
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        var result: [Statement] = []
        var table = ""
        var i = 0
        while i < lines.count {
            let line = lines[i]
            let stripped = pyStrip(line)
            if stripped.hasPrefix("[") {
                var header = pyStrip(String(
                    stripped.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""
                ))
                if header.hasPrefix("[[") {
                    header = String(header.dropFirst(2).dropLast(2))
                } else {
                    header = String(header.dropFirst().dropLast())
                }
                table = pyStrip(header)
                result.append(Statement(table: table, key: nil, lines: [line]))
                i += 1
                continue
            }
            if stripped.isEmpty || stripped.hasPrefix("#") || !stripped.contains("=") {
                result.append(Statement(table: table, key: nil, lines: [line]))
                i += 1
                continue
            }
            let halves = stripped.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = pyStrip(String(halves[0]))
                .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                .trimmingCharacters(in: CharacterSet(charactersIn: "'"))
            var block = [line]
            var (depth, quote) = valueDepth(String(halves[1]), depth: 0, quote: nil)
            while depth > 0 || quote != nil, i + 1 < lines.count {
                i += 1
                block.append(lines[i])
                (depth, quote) = valueDepth(lines[i], depth: depth, quote: quote)
            }
            result.append(Statement(table: table, key: key, lines: block))
            i += 1
        }
        return result
    }

    /// `render_value`: JSON strings and arrays of them, as `mtplx config set`
    /// writes them.
    static func render(_ value: Value) -> String {
        switch value {
        case .string(let text):
            return AgentCLIIntegration.jsonString(text)
        case .strings(let items):
            return "[" + items.map(AgentCLIIntegration.jsonString).joined(separator: ", ") + "]"
        }
    }

    /// `set_top_level_values`: set top-level keys in order (nil removes the
    /// key) and keep every other line. An existing key is replaced where it
    /// stands; a new key goes after the last top-level line.
    public static func setTopLevelValues(_ text: String?, _ updates: [(String, Value?)]) -> String {
        var pending = updates
        var out: [String] = []
        var firstTable: Int?
        for statement in statements(text ?? "") {
            if statement.table.isEmpty,
               let key = statement.key,
               let index = pending.firstIndex(where: { $0.0 == key }) {
                let value = pending.remove(at: index).1
                if let value { out.append("\(key) = \(render(value))") }
                continue
            }
            if firstTable == nil, statement.key == nil, pyStrip(statement.lines[0]).hasPrefix("[") {
                firstTable = out.count
            }
            out.append(contentsOf: statement.lines)
        }
        let added = pending.compactMap { key, value in value.map { "\(key) = \(render($0))" } }
        if !added.isEmpty {
            if let firstTable {
                var insertAt = firstTable
                while insertAt > 0, pyStrip(out[insertAt - 1]).isEmpty { insertAt -= 1 }
                out.insert(contentsOf: added, at: insertAt)
            } else {
                out.append(contentsOf: added)
            }
        }
        return out.joined(separator: "\n") + (out.isEmpty ? "" : "\n")
    }

    // MARK: - Reading values

    /// The top-level value of `key`, if it is a string or an array of strings.
    public static func topLevelValue(_ text: String, key: String) -> Value?? {
        for statement in statements(text) where statement.table.isEmpty && statement.key == key {
            let joined = statement.lines.joined(separator: "\n")
            guard let equals = joined.firstIndex(of: "=") else { return .some(nil) }
            var parser = ValueParser(Array(joined[joined.index(after: equals)...].unicodeScalars))
            return .some(parser.parseValue())
        }
        return nil
    }

    private struct ValueParser {
        let chars: [Unicode.Scalar]
        var i = 0

        init(_ chars: [Unicode.Scalar]) { self.chars = chars }

        mutating func skipSpace() {
            while i < chars.count {
                let ch = chars[i]
                if ch == " " || ch == "\t" || ch == "\n" || ch == "\r" || ch == "," {
                    i += 1
                } else if ch == "#" {
                    while i < chars.count, chars[i] != "\n" { i += 1 }
                } else {
                    return
                }
            }
        }

        mutating func parseValue() -> Value? {
            skipSpace()
            guard i < chars.count else { return nil }
            if chars[i] == "[" {
                i += 1
                var items: [String] = []
                while true {
                    skipSpace()
                    guard i < chars.count else { return nil }
                    if chars[i] == "]" { return .strings(items) }
                    guard let item = parseString() else { return nil }
                    items.append(item)
                }
            }
            return parseString().map(Value.string)
        }

        mutating func parseString() -> String? {
            guard i < chars.count else { return nil }
            if chars[i] == "'" {
                i += 1
                var out = String.UnicodeScalarView()
                while i < chars.count, chars[i] != "'" { out.append(chars[i]); i += 1 }
                guard i < chars.count else { return nil }
                i += 1
                return String(out)
            }
            guard chars[i] == "\"" else { return nil }
            i += 1
            var out = String.UnicodeScalarView()
            while i < chars.count, chars[i] != "\"" {
                if chars[i] == "\\", i + 1 < chars.count {
                    let escape = chars[i + 1]
                    i += 2
                    switch escape {
                    case "n": out.append("\n")
                    case "t": out.append("\t")
                    case "r": out.append("\r")
                    case "b": out.append("\u{08}")
                    case "f": out.append("\u{0C}")
                    case "u", "U":
                        let width = escape == "u" ? 4 : 8
                        guard i + width <= chars.count,
                              let value = UInt32(String(String.UnicodeScalarView(chars[i..<(i + width)])), radix: 16)
                        else { return nil }
                        i += width
                        if let scalar = Unicode.Scalar(value) {
                            out.append(scalar)
                        } else if (0xD800...0xDBFF).contains(value), i + 6 <= chars.count,
                                  chars[i] == "\\", chars[i + 1] == "u",
                                  let low = UInt32(String(String.UnicodeScalarView(chars[(i + 2)..<(i + 6)])), radix: 16),
                                  let scalar = Unicode.Scalar(0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00)) {
                            // A surrogate pair, as json.dumps writes astral characters.
                            out.append(scalar)
                            i += 6
                        } else {
                            return nil
                        }
                    default: out.append(escape)
                    }
                    continue
                }
                out.append(chars[i])
                i += 1
            }
            guard i < chars.count else { return nil }
            i += 1
            return String(out)
        }
    }
}
