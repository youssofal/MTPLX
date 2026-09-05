import Foundation
import XCTest
@testable import MTPLXAppCore

/// The profile .env is one KEY=value statement per line. A configured
/// reasoning effort used to be appended straight onto the template's last
/// line, fusing TERMINAL_CWD and HERMES_MTPLX_REASONING_EFFORT into one
/// statement that Hermes' dotenv parser rejected ("could not parse
/// statement"), silently dropping both the working directory and the effort.
final class HermesDotenvReasoningEffortTests: XCTestCase {
    func testReasoningEffortLandsOnItsOwnDotenvLine() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("HermesDotenv-\(UUID().uuidString)", isDirectory: true)
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        let workspace = root.appendingPathComponent("Workspace", isDirectory: true)
        try FileManager.default.createDirectory(
            at: hermesHome.appendingPathComponent("profiles", isDirectory: true),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }

        let integration = HermesIntegration(
            hermesHome: hermesHome,
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            terminalCommandURL: root.appendingPathComponent(".mtplx").appendingPathComponent("open-hermes.command"),
            activeProfileURL: root.appendingPathComponent("active-profile.json")
        )
        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 8123,
                reasoningEffort: "xhigh",
                apiKey: "",
                hermesWorkspacePath: workspace.path
            )
        )

        let envText = try String(contentsOfFile: result.envPath, encoding: .utf8)
        let lines = envText.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        for line in lines {
            let key = line.prefix { $0 != "=" }
            XCTAssertTrue(line.contains("="), "not a KEY=value statement: \(line)")
            XCTAssertFalse(key.isEmpty, "empty key: \(line)")
            XCTAssertTrue(
                key.allSatisfy { $0.isUppercase || $0.isNumber || $0 == "_" },
                "key is not a dotenv identifier, so a value ran into it: \(line)"
            )
        }
        XCTAssertTrue(lines.contains("TERMINAL_CWD=\"\(workspace.path)\""), envText)
        XCTAssertTrue(lines.contains("HERMES_MTPLX_REASONING_EFFORT=\"xhigh\""), envText)
    }
}
