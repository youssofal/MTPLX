import XCTest

@testable import MTPLXAppCore

/// Where each client opens: the app resolves the setting exactly as
/// `mtplx start` does. `mtplx/client_opening.py` writes
/// `Tests/Fixtures/client_opening_cases.json`
/// (scripts/write_client_opening_cases.py); every row must match.
final class ClientOpeningTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Terminal: Decodable, Equatable {
            let bundleID: String
            let name: String
            let path: String

            enum CodingKeys: String, CodingKey {
                case name, path
                case bundleID = "bundle_id"
            }

            var app: ClientOpenings.TerminalApp {
                ClientOpenings.TerminalApp(bundleID: bundleID, name: name, path: path)
            }
        }

        struct Opening: Decodable {
            let client: String
            let surface: String
            let terminal: Terminal?
            let note: String?
        }

        struct Client: Decodable {
            let id: String
            let name: String
            let configKey: String
            let appSurfaces: [String]
            let defaultChoice: String
            let appName: String?

            enum CodingKeys: String, CodingKey {
                case id, name
                case configKey = "config_key"
                case appSurfaces = "app_surfaces"
                case defaultChoice = "default"
                case appName = "app_name"
            }
        }

        struct Case: Decodable {
            let client: String
            let choice: String?
            let opening: Opening
            let terminalFor: Opening

            enum CodingKeys: String, CodingKey {
                case client, choice, opening
                case terminalFor = "terminal_for"
            }
        }

        struct InfoCase: Decodable {
            let info: [String: PlistValue]
            let terminal: Terminal?
        }

        let clients: [Client]
        let terminals: [Terminal]
        let resolve: [Case]
        let infos: [InfoCase]
    }

    /// The Info.plist subset the fixture carries: strings, and arrays of
    /// document-type dictionaries.
    private enum PlistValue: Decodable {
        case string(String)
        case documentTypes([[String: PlistValue]])
        case strings([String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .string(text)
            } else if let list = try? container.decode([String].self) {
                self = .strings(list)
            } else {
                self = .documentTypes(try container.decode([[String: PlistValue]].self))
            }
        }

        var plist: Any {
            switch self {
            case .string(let text): return text
            case .strings(let list): return list
            case .documentTypes(let kinds): return kinds.map { $0.mapValues(\.plist) }
            }
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/client_opening_cases.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func assertSame(
        _ swift: ClientOpenings.Opening,
        _ python: Fixture.Opening,
        _ label: String
    ) {
        XCTAssertEqual(swift.client, python.client, label)
        XCTAssertEqual(swift.surface, python.surface, label)
        XCTAssertEqual(swift.terminal, python.terminal?.app, label)
        XCTAssertEqual(swift.note, python.note, label)
    }

    func testTheClientsMatchTheCLI() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.clients.map(\.id), ClientOpenings.clients.map(\.id))
        for (python, swift) in zip(fixture.clients, ClientOpenings.clients) {
            XCTAssertEqual(python.name, swift.name)
            XCTAssertEqual(python.configKey, swift.configKey)
            XCTAssertEqual(python.appSurfaces, swift.appSurfaces)
            XCTAssertEqual(python.defaultChoice, swift.defaultChoice)
            XCTAssertEqual(python.appName, swift.appName)
        }
    }

    func testEveryChoiceResolvesAsTheCLIResolvesIt() throws {
        let fixture = try loadFixture()
        let terminals = fixture.terminals.map(\.app)
        XCTAssertGreaterThanOrEqual(fixture.resolve.count, 50)
        for row in fixture.resolve {
            let label = "\(row.client) \(row.choice ?? "nil")"
            assertSame(
                ClientOpenings.resolve(client: row.client, choice: row.choice, terminals: terminals),
                row.opening,
                label
            )
            assertSame(
                ClientOpenings.terminalFor(client: row.client, choice: row.choice, terminals: terminals),
                row.terminalFor,
                "terminal for \(label)"
            )
        }
    }

    func testTerminalsAreReadFromWhatEachAppDeclares() throws {
        let fixture = try loadFixture()
        for row in fixture.infos {
            let info = row.info.mapValues(\.plist)
            XCTAssertEqual(
                ClientOpenings.terminal(fromInfo: info, path: "/Applications/X.app"),
                row.terminal?.app,
                "\(info["CFBundleIdentifier"] ?? "no id")"
            )
        }
    }

    func testInstalledTerminalsListTerminalFirstAndOncePerApp() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("client-opening-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func bundle(_ folder: String, _ name: String, _ bundleID: String, role: String = "Shell") throws {
            let contents = root.appendingPathComponent("\(folder)/\(name).app/Contents", isDirectory: true)
            try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
            let info: [String: Any] = [
                "CFBundleIdentifier": bundleID,
                "CFBundleName": name,
                "CFBundleDocumentTypes": [
                    ["CFBundleTypeRole": role, "LSItemContentTypes": [ClientOpenings.shellScriptType]],
                ],
            ]
            let data = try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            try data.write(to: contents.appendingPathComponent("Info.plist"))
        }
        try bundle("a", "Warp", "dev.warp.Warp-Stable")
        try bundle("a", "Terminal", "com.apple.Terminal")
        try bundle("a", "Editor", "com.example.editor", role: "Editor")
        try bundle("b", "Warp", "dev.warp.Warp-Stable")
        let found = ClientOpenings.installedTerminals(
            in: [root.appendingPathComponent("a"), root.appendingPathComponent("b"), root.appendingPathComponent("none")]
        )
        XCTAssertEqual(found.map(\.name), ["Terminal", "Warp"])
        // The first folder's copy wins (the temporary folder is reached
        // through /var -> /private/var, so compare resolved paths).
        XCTAssertEqual(
            found.last.map { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path },
            root.appendingPathComponent("a/Warp.app").resolvingSymlinksInPath().path
        )
    }

    func testAChoiceIsSavedBesideTheUsersOtherSettings() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("client-opening-\(UUID().uuidString).toml")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("# mine\ncontext_window = 65536\n".utf8).write(to: url)
        try ClientOpenings.save(choice: "Ghostty", for: "pi", configURL: url)
        try ClientOpenings.save(choice: ClientOpenings.terminalChoice, for: "codex", configURL: url)
        XCTAssertEqual(ClientOpenings.savedChoices(configURL: url), ["pi": "Ghostty", "codex": "terminal"])
        XCTAssertEqual(ClientOpenings.currentSurface(.codex, configURL: url), ClientOpenings.terminalChoice)
        XCTAssertEqual(ClientOpenings.currentSurface(.openCode, configURL: url), ClientOpenings.ownApp)
        try ClientOpenings.save(choice: nil, for: "pi", configURL: url)
        XCTAssertEqual(
            try String(contentsOf: url, encoding: .utf8),
            "# mine\ncontext_window = 65536\ncodex_opens_in = \"terminal\"\n"
        )
    }
}
