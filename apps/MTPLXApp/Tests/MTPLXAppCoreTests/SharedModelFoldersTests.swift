import XCTest

@testable import MTPLXAppCore

/// Model folders shared by the app and `mtplx folders` (night 4, D5): one
/// setting, `model_dir` and `model_dirs` in the CLI's config.toml. Every
/// test uses scratch files; the real ~/.mtplx/config.toml is never touched
/// (an XCTest run turns the automatic sharing off).
final class SharedModelFoldersTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-folders-\(UUID().uuidString.lowercased())", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    private struct Cases: Decodable {
        struct Case: Decodable {
            let text: String?
            let updates: [String: [String]?]
            let output: String
        }
        let version: Int
        let cases: [Case]
    }

    private enum UpdateValue: Decodable {
        case string(String)
        case strings([String])
    }

    /// Byte-for-byte the same file as `mtplx folders` writes, so the app and
    /// the CLI never rewrite each other's lines.
    func testLineEditsMatchTheCLIByteForByte() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/toml_lines_cases.json")
        let data = try Data(contentsOf: url)
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let cases = try XCTUnwrap(root["cases"] as? [[String: Any]])
        XCTAssertEqual(cases.count, 32)
        for (index, item) in cases.enumerated() {
            let text = item["text"] as? String
            let updatesObject = try XCTUnwrap(item["updates"] as? [String: Any])
            // The Python cases insert keys alphabetically (model_dir, model_dirs).
            let updates: [(String, TOMLLines.Value?)] = updatesObject.keys.sorted().map { key in
                switch updatesObject[key] {
                case let value as String: return (key, .string(value))
                case let value as [String]: return (key, .strings(value))
                default: return (key, nil)
                }
            }
            XCTAssertEqual(
                TOMLLines.setTopLevelValues(text, updates),
                item["output"] as? String,
                "case \(index)"
            )
        }
    }

    func testReadsWhatTheCLIWritesIncludingEscapesAndMultiLineArrays() throws {
        let config = scratch.appendingPathComponent("config.toml")
        try """
        # MTPLX user configuration
        model = "Youssofal/x"
        model_dir = "/Volumes/SSD/mod\\u00e9ls"
        model_dirs = [
          "/a \\"b\\"",  # note
          '/literal',
        ]

        [extra]
        model_dir = "inner"
        """.write(to: config, atomically: true, encoding: .utf8)
        let setting = try XCTUnwrap(SharedModelFolders.read(from: config))
        XCTAssertEqual(setting.modelDir, "/Volumes/SSD/modéls")
        XCTAssertEqual(setting.modelDirs, ["/a \"b\"", "/literal"])
        XCTAssertNil(try SharedModelFolders.read(from: scratch.appendingPathComponent("absent.toml")))
    }

    func testWriteKeepsTheUsersOtherSettingsAndSkipsAnUnchangedFile() throws {
        let config = scratch.appendingPathComponent("config.toml")
        let original = "# mine\nprofile = \"sustained\"\n\n[extra]\nx = 1\n"
        try original.write(to: config, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: config.path)
        let environment = ["HOME": scratch.path]
        try SharedModelFolders.write(
            primary: "/Volumes/SSD/models",
            additional: ["/Users/me/Models"],
            to: config,
            environment: environment
        )
        let text = try String(contentsOf: config, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("# mine\nprofile = \"sustained\"\nmodel_dir = \"/Volumes/SSD/models\"\nmodel_dirs = [\"/Users/me/Models\"]\n"))
        XCTAssertTrue(text.hasSuffix("[extra]\nx = 1\n"))
        let mode = (try FileManager.default.attributesOfItem(atPath: config.path)[.posixPermissions] as? NSNumber)?.intValue
        XCTAssertEqual(mode, 0o600)

        // The default download folder is no line; an empty list stays a key.
        try SharedModelFolders.write(
            primary: scratch.appendingPathComponent(".mtplx/models").path,
            additional: [],
            to: config,
            environment: environment
        )
        let reset = try XCTUnwrap(SharedModelFolders.read(from: config))
        XCTAssertNil(reset.modelDir)
        XCTAssertEqual(reset.modelDirs, [])
        XCTAssertTrue(reset.isSet)

        // An unreadable file holds the user's other CLI settings: never replaced.
        let broken = scratch.appendingPathComponent("broken.toml")
        try Data([0xff, 0xfe, 0x00]).write(to: broken)
        XCTAssertThrowsError(try SharedModelFolders.write(
            primary: "/x", additional: [], to: broken, environment: environment
        ))
        XCTAssertEqual(try Data(contentsOf: broken), Data([0xff, 0xfe, 0x00]))
    }

    func testAFolderOnAnUnpluggedDriveIsNotConnectedNotMissing() throws {
        let unplugged = "/Volumes/mtplx-test-unplugged-\(UUID().uuidString)/models"
        XCTAssertEqual(
            SharedModelFolders.availability(of: unplugged),
            .notConnected(volume: (unplugged as NSString).deletingLastPathComponent)
        )
        XCTAssertEqual(SharedModelFolders.availability(of: scratch.path), .available)
        XCTAssertEqual(SharedModelFolders.availability(of: scratch.appendingPathComponent("nope").path), .missing)
        let file = scratch.appendingPathComponent("file")
        try Data("x".utf8).write(to: file)
        XCTAssertEqual(SharedModelFolders.availability(of: file.path), .notAFolder)
        // The #466 layout: a link onto an external drive that is unplugged.
        let link = scratch.appendingPathComponent("models-link")
        try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: unplugged)
        if case .notConnected = SharedModelFolders.availability(of: link.path) {} else {
            XCTFail("a dangling link onto /Volumes must read as not connected")
        }
        let pack = scratch.appendingPathComponent("Youssofal--Pack/config.json")
        try FileManager.default.createDirectory(at: pack.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: pack)
        XCTAssertEqual(SharedModelFolders.packCount(in: scratch.path), 1)
    }

    /// The settings store reads the shared file and writes it on save, once
    /// either side has named folders; before that, the app's own folders are
    /// written there once so the CLI sees them.
    func testSettingsStoreSharesFoldersWithTheCLIFile() throws {
        let settings = scratch.appendingPathComponent("settings.json")
        let shared = scratch.appendingPathComponent("config.toml")
        let store = MTPLXSettingsStore(settingsURL: settings, modelFolderSharing: .at(shared))
        let ssd = scratch.appendingPathComponent("ssd", isDirectory: true)
        try FileManager.default.createDirectory(at: ssd, withIntermediateDirectories: true)

        // The app had an extra folder before sharing existed: the CLI learns it.
        var configuration = MTPLXAppConfiguration()
        configuration.addModelDirectories([ssd.path])
        try MTPLXSettingsStore(settingsURL: settings, modelFolderSharing: .off).save(configuration)
        _ = try store.load()
        XCTAssertEqual(try SharedModelFolders.read(from: shared)?.modelDirs, [ssd.path])

        // `mtplx folders add` on the CLI side shows in the app on next load.
        let other = "/Volumes/mtplx-test-unplugged-\(UUID().uuidString)/llm"
        let text = try String(contentsOf: shared, encoding: .utf8)
        try TOMLLines.setTopLevelValues(text, [("model_dirs", .strings([ssd.path, other]))])
            .write(to: shared, atomically: true, encoding: .utf8)
        let loaded = try store.load()
        XCTAssertEqual(loaded.additionalModelDirectories, [ssd.path, other])

        // Removing it in the app writes the shared file for the CLI.
        var edited = loaded
        edited.additionalModelDirectories = [ssd.path]
        try store.save(edited)
        XCTAssertEqual(try SharedModelFolders.read(from: shared)?.modelDirs, [ssd.path])
    }

    func testAutomaticSharingIsOffUnderTests() {
        XCTAssertNil(MTPLXSettingsStore().sharedModelFoldersURL)
        XCTAssertNil(MTPLXSettingsStore(settingsURL: scratch.appendingPathComponent("s.json")).sharedModelFoldersURL)
    }
}
