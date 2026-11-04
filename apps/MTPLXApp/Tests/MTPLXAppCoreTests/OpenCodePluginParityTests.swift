import XCTest
import Foundation
@testable import MTPLXAppCore

/// The app installs the OpenCode plugin the CLI installs (mtplx/opencode.py),
/// byte for byte and at the same place. These tests run the CLI's own writer
/// and merge rule from this checkout at test time and compare what the two
/// produce, so the copies cannot drift apart unnoticed: a V1-only app plugin
/// once did nothing under OpenCode 2.x, which calls only `model.request`.
final class OpenCodePluginParityTests: XCTestCase {
    /// apps/MTPLXApp/Tests/MTPLXAppCoreTests/<this file> -> the repo root.
    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MTPLXAppCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // MTPLXApp
            .deletingLastPathComponent()  // apps
            .deletingLastPathComponent()  // repo root
    }

    private static let sessionIDLine =
        "      event.headers[\"x-mtplx-session-id\"] = String(event.sessionID);\n"
    /// The V2 line that passes OpenCode's request kind. The CLI branch for
    /// 2.12.3 adds it in the same release; until both branches meet, a CLI
    /// copy without it must equal the app's copy minus exactly this line.
    private static let requestKindLine =
        "      if (event.kind) event.headers[\"x-mtplx-request-kind\"] = String(event.kind);\n"

    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-opencode-parity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let scratch {
            try? FileManager.default.removeItem(at: scratch)
        }
    }

    // MARK: Files

    func testAppWritesTheCLIPluginPackageByteForByte() throws {
        let cliConfig = scratch.appendingPathComponent("cli/opencode.json")
        let cliPackage = try runCLI("""
        import mtplx.opencode as o
        print(o.write_opencode_session_headers_plugin(sys.argv[2]))
        """, cliConfig.path).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(
            URL(fileURLWithPath: cliPackage).standardizedFileURL.path,
            OpenCodeIntegration.sessionHeadersPluginURL(configURL: cliConfig).standardizedFileURL.path,
            "the app and the CLI install the package at the same place"
        )

        let appConfig = scratch.appendingPathComponent("app/opencode.json")
        let result = try OpenCodeIntegration(
            configURL: appConfig,
            desktopSettingsStoreURL: scratch.appendingPathComponent("app/default.dat")
        ).sync(configuration: MTPLXAppConfiguration(
            model: "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
            port: 8000
        ))
        let appPackage = URL(fileURLWithPath: result.sessionHeadersPluginPath)
        XCTAssertEqual(
            appPackage.standardizedFileURL.path,
            OpenCodeIntegration.sessionHeadersPluginURL(configURL: appConfig).standardizedFileURL.path
        )

        let fileManager = FileManager.default
        let cliFiles = try fileManager.contentsOfDirectory(atPath: cliPackage).sorted()
        let appFiles = try fileManager.contentsOfDirectory(atPath: appPackage.path).sorted()
        XCTAssertEqual(appFiles, cliFiles)
        XCTAssertEqual(appFiles, ["index.js", "package.json", "server.js"])

        for name in cliFiles {
            let cliText = try String(
                contentsOf: URL(fileURLWithPath: cliPackage).appendingPathComponent(name),
                encoding: .utf8
            )
            let appText = try String(contentsOf: appPackage.appendingPathComponent(name), encoding: .utf8)
            if name == "server.js", !cliText.contains("x-mtplx-request-kind") {
                XCTAssertTrue(cliText.contains(Self.sessionIDLine), "CLI server.js lost its session id line")
                XCTAssertEqual(
                    appText,
                    cliText.replacingOccurrences(
                        of: Self.sessionIDLine,
                        with: Self.sessionIDLine + Self.requestKindLine
                    ),
                    "app server.js must be the CLI's plus the request-kind line"
                )
            } else {
                XCTAssertEqual(appText, cliText, "\(name) differs from the CLI's copy")
            }
            let mode = try XCTUnwrap(
                fileManager.attributesOfItem(atPath: appPackage.appendingPathComponent(name).path)[.posixPermissions]
                    as? NSNumber
            )
            XCTAssertEqual(mode.intValue & 0o777, 0o600, "\(name) must be private to the user like the CLI's")
        }
        XCTAssertTrue(
            try String(contentsOf: appPackage.appendingPathComponent("server.js"), encoding: .utf8)
                .contains(Self.requestKindLine)
        )
    }

    // MARK: Registration

    func testAppRegistersThePluginWithTheCLIMergeRule() throws {
        let appConfig = scratch.appendingPathComponent("app/opencode.json")
        let managed = OpenCodeIntegration.sessionHeadersPluginURL(configURL: appConfig).path
        let legacy = appConfig.deletingLastPathComponent().appendingPathComponent("mtplx-session-headers.js").path
        let existing = [
            "/tmp/keep-plugin.js",
            "/elsewhere/mtplx-session-headers.js",
            legacy,
            "/elsewhere/plugins/mtplx-session-headers",
            "opencode-wakatime",
        ]

        let fragment = #"{"provider": {"mtplx": {}}, "model": "mtplx/m", "small_model": "mtplx/m"}"#
        let cliPlugins = try runCLI("""
        import json
        import mtplx.opencode as o
        merged = o.merge_opencode_config(
            {"plugin": json.loads(sys.argv[3])},
            config_fragment=json.loads(sys.argv[4]),
            session_headers_plugin_path=sys.argv[2],
        )
        print(json.dumps(merged["plugin"]))
        """, managed, String(decoding: try JSONEncoder().encode(existing), as: UTF8.self), fragment)

        try FileManager.default.createDirectory(
            at: appConfig.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try JSONEncoder().encode(["plugin": existing]).write(to: appConfig)
        _ = try OpenCodeIntegration(
            configURL: appConfig,
            desktopSettingsStoreURL: scratch.appendingPathComponent("app/default.dat")
        ).sync(configuration: MTPLXAppConfiguration(model: "/models/Qwen3.8-27B-MTPLX-Optimized-Speed"))
        struct Registered: Decodable { let plugin: [String] }
        let appPlugins = try JSONDecoder().decode(Registered.self, from: Data(contentsOf: appConfig)).plugin

        XCTAssertEqual(appPlugins, try JSONDecoder().decode([String].self, from: Data(cliPlugins.utf8)))
        // Only the managed package is left of MTPLX's own copies, so the V1
        // hooks run once; everything else the user registered stays.
        XCTAssertEqual(appPlugins, ["/tmp/keep-plugin.js", "opencode-wakatime", managed])
    }

    // MARK: Behavior under OpenCode 2.x

    func testServerEntrypointSendsSessionHeadersAndRequestKind() throws {
        let appConfig = scratch.appendingPathComponent("app/opencode.json")
        let result = try OpenCodeIntegration(
            configURL: appConfig,
            desktopSettingsStoreURL: scratch.appendingPathComponent("app/default.dat")
        ).sync(configuration: MTPLXAppConfiguration(model: "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"))
        let server = URL(fileURLWithPath: result.sessionHeadersPluginPath).appendingPathComponent("server.js")

        let harness = scratch.appendingPathComponent("harness.mjs")
        try """
        import plugin from \(String(decoding: try JSONEncoder().encode(server.absoluteString), as: UTF8.self));
        const hooks = [];
        await plugin.setup({ session: { hook: async (name, fn, options) => { hooks.push({ name, fn, options }); } } });
        const run = async (event) => { for (const hook of hooks) await hook.fn(event); return event.headers; };
        const v1 = await plugin.server();
        const v1Headers = { headers: {} };
        await v1["chat.headers"]({ sessionID: "ses_9", message: { id: "msg_1" }, model: { providerID: "mtplx" } }, v1Headers);
        console.log(JSON.stringify({
          id: plugin.id,
          hooks: hooks.map((hook) => ({ name: hook.name, options: hook.options })),
          title: await run({ sessionID: "ses_9", kind: "title", headers: {} }),
          primary: await run({ sessionID: "ses_9", headers: {} }),
          v1: v1Headers.headers,
        }));
        """.write(to: harness, atomically: true, encoding: .utf8)

        let output = try runNode(harness)
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any])
        XCTAssertEqual(payload["id"] as? String, "mtplx.session-headers")
        let hooks = try XCTUnwrap(payload["hooks"] as? [[String: Any]])
        XCTAssertEqual(hooks.count, 1)
        XCTAssertEqual(hooks.first?["name"] as? String, "model.request")
        XCTAssertEqual((hooks.first?["options"] as? [String: Any])?["providerID"] as? String, "mtplx")
        XCTAssertEqual(payload["title"] as? [String: String], [
            "x-mtplx-client": "opencode",
            "x-mtplx-session-id": "ses_9",
            "x-mtplx-request-kind": "title",
        ])
        XCTAssertEqual(payload["primary"] as? [String: String], [
            "x-mtplx-client": "opencode",
            "x-mtplx-session-id": "ses_9",
        ])
        XCTAssertEqual(payload["v1"] as? [String: String], [
            "x-mtplx-client": "opencode",
            "x-mtplx-session-id": "ses_9",
            "x-mtplx-client-turn-id": "msg_1",
        ])
    }

    // MARK: Helpers

    /// Runs `body` with the checkout's `mtplx` package importable and
    /// returns stdout. The CLI module only reads and writes the scratch paths
    /// it is given here.
    private func runCLI(_ body: String, _ arguments: String...) throws -> String {
        let python = URL(fileURLWithPath: "/usr/bin/python3")
        guard FileManager.default.isExecutableFile(atPath: python.path) else {
            XCTFail("python3 is required to read the CLI's plugin sources")
            return ""
        }
        let script = "import sys\nsys.path.insert(0, sys.argv[1])\n" + body
        return try run(python, ["-I", "-c", script, Self.repoRoot.path] + arguments)
    }

    private func runNode(_ script: URL) throws -> String {
        let searchPath = ProcessInfo.processInfo.environment["PATH"] ?? ""
        let candidates = searchPath.split(separator: ":").map(String.init) + ["/opt/homebrew/bin", "/usr/local/bin"]
        guard let node = candidates
            .map({ URL(fileURLWithPath: $0).appendingPathComponent("node") })
            .first(where: { FileManager.default.isExecutableFile(atPath: $0.path) })
        else { throw XCTSkip("node is not installed") }
        return try run(node, [script.path])
    }

    private func run(_ executable: URL, _ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.currentDirectoryURL = scratch
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        process.standardInput = FileHandle.nullDevice
        try process.run()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        XCTAssertEqual(
            process.terminationStatus, 0,
            "\(executable.lastPathComponent) failed: \(String(decoding: err, as: UTF8.self))"
        )
        return String(decoding: out, as: UTF8.self)
    }
}
