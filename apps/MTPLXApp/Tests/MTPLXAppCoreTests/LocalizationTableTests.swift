import XCTest
import Foundation
@testable import MTPLXAppCore

/// Guards the shipped string tables: every language carries every English
/// key with the same format placeholders, the code only uses keys the
/// English table knows, the onboarding and settings screens are really
/// translated, and the live-switch lookup returns per-language text.
final class LocalizationTableTests: XCTestCase {
    private static let languages = AppLanguage.allCases

    /// The source-tree Localization directory (the DEBUG root of L10n).
    private static var localizationDirectory: URL {
        let candidates = L10n.localizationRoots()
        return candidates.last(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("en.lproj/Localizable.strings").path)
        }) ?? candidates.last!
    }

    private static var packageRoot: URL {
        localizationDirectory  // .../Sources/MTPLXAppCore/Resources/Localization
            .deletingLastPathComponent() // Resources
            .deletingLastPathComponent() // MTPLXAppCore
            .deletingLastPathComponent() // Sources
            .deletingLastPathComponent() // package root
    }

    private static func table(_ language: AppLanguage) throws -> [String: String] {
        let url = localizationDirectory.appendingPathComponent("\(language.code).lproj/Localizable.strings")
        let data = try Data(contentsOf: url)
        let plist = try PropertyListSerialization.propertyList(from: data, options: [], format: nil)
        return try XCTUnwrap(plist as? [String: String], "\(language.code) table must parse as key/value strings")
    }

    private struct FormatArgument: Equatable {
        let position: Int
        let type: String
    }

    private static let specifier = try! NSRegularExpression(
        pattern: #"%(?:(\d+)\$)?[-+ #0]*\d*(?:\.\d+)?(lld|ld|d|@|lf|f|s|u|x|X|%)"#
    )

    /// Normalizes both implicit and positional specifiers to source-argument
    /// identity. Sorting types alone lets `%@, %.0f` and `%.0f, %@` compare
    /// equal even though the latter reads each C vararg as the wrong type.
    private static func placeholders(_ s: String) -> [FormatArgument] {
        let ns = s as NSString
        var nextImplicitPosition = 1
        return specifier.matches(in: s, range: NSRange(location: 0, length: ns.length))
            .compactMap { match in
                let type = ns.substring(with: match.range(at: 2))
                guard type != "%" else { return nil }
                let position: Int
                if match.range(at: 1).location != NSNotFound {
                    position = Int(ns.substring(with: match.range(at: 1)))!
                } else {
                    position = nextImplicitPosition
                    nextImplicitPosition += 1
                }
                return FormatArgument(position: position, type: type)
            }
            .sorted {
                if $0.position != $1.position { return $0.position < $1.position }
                return $0.type < $1.type
            }
    }

    override func setUp() {
        super.setUp()
        L10n.resetBundleCache()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    // MARK: Completeness

    func testEveryLanguageShipsATable() throws {
        for language in Self.languages {
            let table = try Self.table(language)
            XCTAssertGreaterThan(table.count, 1000, "\(language.code) table is suspiciously small")
        }
    }

    func testEveryLanguageCarriesExactlyTheEnglishKeys() throws {
        let english = Set(try Self.table(.english).keys)
        for language in Self.languages where language != .english {
            let keys = Set(try Self.table(language).keys)
            let missing = english.subtracting(keys).sorted()
            let extra = keys.subtracting(english).sorted()
            XCTAssertTrue(missing.isEmpty, "\(language.code) is missing \(missing.count) keys, e.g. \(missing.prefix(5))")
            XCTAssertTrue(extra.isEmpty, "\(language.code) has \(extra.count) keys unknown to English, e.g. \(extra.prefix(5))")
        }
    }

    func testEnglishTableMapsEveryKeyToItself() throws {
        for (key, value) in try Self.table(.english) {
            XCTAssertEqual(key, value)
        }
    }

    // MARK: Placeholders

    func testTranslationsPreserveFormatPlaceholders() throws {
        let english = try Self.table(.english)
        for language in Self.languages where language != .english {
            let table = try Self.table(language)
            for (key, value) in table {
                guard english[key] != nil else { continue }
                XCTAssertEqual(
                    Self.placeholders(value), Self.placeholders(key),
                    "\(language.code): format-argument contract differs for \(key.debugDescription) -> \(value.debugDescription)"
                )
                XCTAssertFalse(value.trimmingCharacters(in: .whitespaces).isEmpty, "\(language.code): empty value for \(key.debugDescription)")
                XCTAssertEqual(key.hasPrefix(" "), value.hasPrefix(" "), "\(language.code): leading space lost for \(key.debugDescription)")
                XCTAssertEqual(key.hasSuffix(" "), value.hasSuffix(" "), "\(language.code): trailing space lost for \(key.debugDescription)")
            }
        }
    }

    func testPositionalPlaceholdersAreAllOrNothing() throws {
        let positional = try NSRegularExpression(pattern: #"%(\d+)\$"#)
        for language in Self.languages where language != .english {
            for (key, value) in try Self.table(language) {
                let ns = value as NSString
                let positions = positional.matches(in: value, range: NSRange(location: 0, length: ns.length)).count
                guard positions > 0 else { continue }
                XCTAssertEqual(positions, Self.placeholders(key).count, "\(language.code): mixed positional and plain placeholders in \(value.debugDescription)")
            }
        }
    }

    func testPlaceholderIdentityDoesNotIgnoreImplicitReordering() {
        let source = Self.placeholders("%@ %.0f")
        XCTAssertEqual(source, Self.placeholders("%1$@ %2$.0f"))
        XCTAssertEqual(source, Self.placeholders("%2$.0f %1$@"))
        XCTAssertNotEqual(source, Self.placeholders("%.0f %@"))
    }

    func testChineseReorderedFormatArgumentsRenderCorrectly() {
        XCTAssertEqual(
            L10n.string(
                "Chosen for %@ with %.0f GB unified memory.",
                language: .simplifiedChinese,
                arguments: ["M5 Max", 128.0]
            ),
            "已为配备 128 GB 统一内存的 M5 Max 选用。"
        )
        XCTAssertEqual(
            L10n.string(
                "Restart %lld scheduled in %.1fs.",
                language: .simplifiedChinese,
                arguments: [Int64(3), 1.5]
            ),
            "已安排在 1.5 秒后进行第 3 次重启。"
        )
    }

    // MARK: Source keys

    func testEverySourceKeyExistsInTheEnglishTable() throws {
        let english = try Self.table(.english)
        let sources = Self.packageRoot.appendingPathComponent("Sources")
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        let call = try NSRegularExpression(
            pattern: #"(?:\b(?:tr|L10n\.string|status\??)\(\s*|\bupdateLocalized\([^,]+,[^,]+,\s*|\b(?:localizedDetailKey|detailLocalizationKey):\s*)"((?:[^"\\]|\\.)*)""#
        )
        var scanned = 0
        var unknown: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            let searchable = text
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { line in
                    let line = String(line)
                    return line.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : line
                }
                .joined(separator: "\n")
            let ns = searchable as NSString
            for match in call.matches(in: searchable, range: NSRange(location: 0, length: ns.length)) {
                let key = Self.unescapeSwiftLiteral(ns.substring(with: match.range(at: 1)))
                scanned += 1
                if english[key] == nil { unknown.append(key) }
            }
        }
        XCTAssertGreaterThan(scanned, 1500, "expected localization call sites to be scanned")
        XCTAssertTrue(unknown.isEmpty, "keys used in code but missing from en.lproj: \(unknown.prefix(10))")
    }

    private static func unescapeSwiftLiteral(_ s: String) -> String {
        var out = ""
        var iterator = s.makeIterator()
        while let c = iterator.next() {
            guard c == "\\" else { out.append(c); continue }
            guard let n = iterator.next() else { break }
            switch n {
            case "n": out.append("\n")
            case "t": out.append("\t")
            case "r": out.append("\r")
            case "0": out.append("\0")
            case "u":
                var hex = ""
                _ = iterator.next() // {
                while let h = iterator.next(), h != "}" { hex.append(h) }
                if let scalar = UInt32(hex, radix: 16).flatMap(Unicode.Scalar.init) { out.unicodeScalars.append(scalar) }
            default: out.append(n)
            }
        }
        return out
    }

    // MARK: No English leftovers on the key screens

    private static let spotKeys = [
        "Choose your language", "Search languages", "Get Started", "Your Mac", "Recommended models",
        "Setting up MTPLX", "Settings", "Performance", "Language", "Cancel", "Continue", "Back",
        "Start MTPLX", "Stop MTPLX", "Done", "Next", "Retry", "Restart", "Clear All",
        "2–3× faster", "Auto-tuned", "On-device",
        "Checking MTPLX runtime", "Checking fan control", "Checking for an existing mtplx command",
        "Installing MTPLX runtime", "Installing fan control", "Repairing MTPLX runtime",
        "Add a model", "Choose a model folder", "Use Folder", "Local model folder on this Mac.",
        "The fastest way to run local AI.", "How MTPLX handles several requests at once. Changes apply after a restart.",
        "The language MTPLX uses across the app. Changes apply immediately.",
        "MTPLX speaks %lld languages. Your pick applies right away, and you can change it anytime in Settings.",
    ]

    func testOnboardingAndSettingsKeysAreTranslated() throws {
        let english = try Self.table(.english)
        for key in Self.spotKeys {
            XCTAssertNotNil(english[key], "spot key missing from English: \(key)")
        }
        for language in [AppLanguage.simplifiedChinese, .japanese, .arabic, .russian, .hindi, .korean] {
            let table = try Self.table(language)
            for key in Self.spotKeys {
                let value = table[key] ?? ""
                XCTAssertFalse(value.isEmpty, "\(language.code): \(key) untranslated (missing)")
                // Brand tokens may survive inside the value but the value must not be the English text.
                XCTAssertNotEqual(value, key, "\(language.code): \(key) is still English")
            }
        }
    }

    func testGlossaryCancelIsConsistentAcrossLanguages() throws {
        let expected: [AppLanguage: String] = [
            .english: "Cancel", .simplifiedChinese: "取消", .spanish: "Cancelar", .hindi: "रद्द करें",
            .arabic: "إلغاء", .brazilianPortuguese: "Cancelar", .french: "Annuler", .russian: "Отмена",
            .japanese: "キャンセル", .german: "Abbrechen", .korean: "취소", .indonesian: "Batal",
            .turkish: "İptal",
        ]
        for (language, word) in expected {
            XCTAssertEqual(try Self.table(language)["Cancel"], word, language.code)
        }
    }

    // MARK: Live switch

    func testSameKeyResolvesDifferentlyPerLanguage() throws {
        let english = L10n.string("Settings", language: .english)
        XCTAssertEqual(english, "Settings")
        var seen: Set<String> = [english]
        for language in Self.languages where language != .english {
            let value = L10n.string("Settings", language: language)
            XCTAssertNotEqual(value, "Settings", language.code)
            seen.insert(value)
        }
        XCTAssertGreaterThan(seen.count, 6, "translations must not collapse into one string")
    }

    func testActivatingALanguageSwitchesTrImmediately() throws {
        XCTAssertEqual(tr("Cancel"), "Cancel")
        L10n.activate(.japanese)
        XCTAssertEqual(tr("Cancel"), "キャンセル")
        L10n.activate(.german)
        XCTAssertEqual(tr("Cancel"), "Abbrechen")
        L10n.activate(.english)
        XCTAssertEqual(tr("Cancel"), "Cancel")
    }

    @MainActor
    func testLanguageStoreDrivesTr() throws {
        let suite = "mtplx.tests.language.switch.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        addTeardownBlock { UserDefaults(suiteName: suite)?.removePersistentDomain(forName: suite) }
        let store = LanguageStore(defaults: defaults, preferredLanguages: ["en"])
        XCTAssertEqual(tr("Cancel"), "Cancel")
        store.language = .french
        XCTAssertEqual(tr("Cancel"), "Annuler", "tr answers in the new language before any view re-renders")
        store.language = .english
        XCTAssertEqual(tr("Cancel"), "Cancel")
    }

    func testModelDetailsResolveWhenLanguageChangesAfterCatalogInitialization() throws {
        let key = "4-bit dynamic quant. Great coding speeds and good quality."
        let model = try XCTUnwrap(
            MTPLXModelOption.officialCatalog.first { $0.id == "qwen38-27b-optimized-speed" }
        )
        XCTAssertEqual(model.detail, key)
        XCTAssertEqual(model.localizedDetail, key)

        L10n.activate(.simplifiedChinese)
        XCTAssertEqual(model.localizedDetail, "4 位动态量化。编程速度极快，质量良好。")

        let legacy = MTPLXModelOption(
            id: "custom-example--model",
            displayName: "Model",
            shortName: "Model",
            detail: "旧语言中已缓存的说明",
            hfModelID: "Example/Model",
            localCandidates: []
        )
        let decoded = try JSONDecoder().decode(
            MTPLXModelOption.self,
            from: JSONEncoder().encode(legacy)
        )
        XCTAssertEqual(
            decoded.localizedDetail,
            "自定义 Hugging Face 模型。当模型仓库包含边车文件时，MTPLX 将使用 MTP。"
        )

        let literal = MTPLXModelOption(
            id: "third-party",
            displayName: "Third Party",
            shortName: "Third Party",
            detail: "Publisher-authored detail",
            hfModelID: "Example/ThirdParty",
            localCandidates: []
        )
        XCTAssertEqual(literal.localizedDetail, "Publisher-authored detail")
    }

    func testDiscoveredLibraryDescriptionSurvivesLanguageSwitchAndLegacyDecode() throws {
        let row = MTPLXModelOption(
            id: "local:/library/example",
            displayName: "Example", shortName: "Example", detail: "Previously translated text",
            hfModelID: "Example/Model", localCandidates: []
        )
        let decoded = try JSONDecoder().decode(MTPLXModelOption.self, from: JSONEncoder().encode(row))
        let key = "Local MTPLX model in a configured library."
        XCTAssertEqual(decoded.localizedDetail, key)
        L10n.activate(.simplifiedChinese)
        XCTAssertEqual(decoded.localizedDetail, L10n.string(key, language: .simplifiedChinese))
        L10n.activate(.english)
        XCTAssertEqual(decoded.localizedDetail, key)
    }

    func testNewDescriptionsExistInEveryLanguageAndResolveLive() throws {
        for id in ["flash-next-optimized-quality", "bonsai-2-27b-optimized-speed", "mimo-v26-qwen-9b-optimized-speed"] {
            let model = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == id })
            let key = model.detail
            for language in Self.languages {
                let value = try XCTUnwrap(Self.table(language)[key])
                if language != .english { XCTAssertNotEqual(value, key) }
                L10n.activate(language)
                XCTAssertEqual(model.localizedDetail, value)
                if id.hasPrefix("bonsai") { XCTAssertTrue(value.contains("Prism ML")) }
                if id.hasPrefix("mimo") {
                    // Brand and base-model names stay untranslated.
                    XCTAssertTrue(value.contains("Xiaomi"), language.code)
                    XCTAssertTrue(value.contains("Qwen 3.5 9B"), language.code)
                }
            }
        }
        L10n.activate(.english)
    }

    func testFormatArgumentsFlowThroughTranslatedTemplates() throws {
        let key = "Step %lld of %lld"
        XCTAssertNotNil(try Self.table(.english)[key])
        for language in Self.languages {
            let text = L10n.string(key, language: language, arguments: [2, 7])
            XCTAssertTrue(text.contains("2") && text.contains("7"), "\(language.code): \(text)")
            XCTAssertFalse(text.contains("%lld"), "\(language.code): unformatted placeholder in \(text)")
        }
    }

    // MARK: Bundle script parity

    func testBundleScriptShipsEveryLanguage() throws {
        let script = Self.packageRoot.appendingPathComponent("script/build_and_run.sh")
        let text = try String(contentsOf: script, encoding: .utf8)
        let regex = try NSRegularExpression(pattern: #"LOCALIZATION_CODES=\(([^)]*)\)"#)
        let ns = text as NSString
        let match = try XCTUnwrap(regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)), "build_and_run.sh must list LOCALIZATION_CODES")
        let codes = ns.substring(with: match.range(at: 1)).split(separator: " ").map(String.init)
        XCTAssertEqual(codes, Self.languages.map(\.code))
    }
}
