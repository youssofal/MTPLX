import Foundation
import XCTest
@testable import MTPLXAppCore

final class MTPLXWorkspaceToolServiceTests: XCTestCase {
    private func makeWorkspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-workspace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }

    private func object(_ result: String) throws -> [String: Any] {
        let data = try XCTUnwrap(result.data(using: .utf8))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func run(_ arguments: [String], in directory: URL) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = arguments
        process.currentDirectoryURL = directory
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? ""
    }

    func testPublicChatDispatchCannotBypassDaemonWorkspaceBoundary() async throws {
        let result = try await object(
            MTPLXChatToolFactory().dispatch(
                name: "write_file",
                argumentsJSON: #"{"path":"bypass.txt","content":"blocked"}"#
            )
        )

        XCTAssertEqual(result["error"] as? String, "unknown_tool")
    }

    func testReadSearchAndListStayInsideWorkspace() async throws {
        let root = try makeWorkspace()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("Sources"),
            withIntermediateDirectories: true
        )
        try Data("MTPLX workspace marker\n".utf8)
            .write(to: root.appendingPathComponent("Sources/Example.swift"), options: .atomic)
        let service = MTPLXWorkspaceToolService()

        let listed = try await object(
            service.dispatch(
                name: "list_files",
                argumentsJSON: #"{"path":"Sources","depth":1}"#,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertTrue(
            (listed["files"] as? [String])?.contains("Sources/Example.swift") == true,
            "\(listed)"
        )

        let read = try await object(
            service.dispatch(
                name: "read_file",
                argumentsJSON: #"{"path":"Sources/Example.swift"}"#,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(read["content"] as? String, "MTPLX workspace marker\n")

        let searched = try await object(
            service.dispatch(
                name: "search_files",
                argumentsJSON: #"{"query":"workspace marker"}"#,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual((searched["matches"] as? [[String: Any]])?.count, 1)

        let rejected = try await object(
            service.dispatch(
                name: "read_file",
                argumentsJSON: #"{"path":"../outside.txt"}"#,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(rejected["error"] as? String, "path_outside_workspace")
    }

    func testMutatingToolsRequireApproval() async throws {
        let root = try makeWorkspace()
        let service = MTPLXWorkspaceToolService()
        let args = #"{"path":"created.txt","content":"approved write"}"#

        let pending = try await object(
            service.dispatch(
                name: "write_file",
                argumentsJSON: args,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(pending["error"] as? String, "approval_required")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("created.txt").path))

        let written = try await object(
            service.dispatch(
                name: "write_file",
                argumentsJSON: args,
                workspaceRoot: root.path,
                approved: true
            )
        )
        XCTAssertEqual(written["written"] as? Bool, true)
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("created.txt")),
            "approved write"
        )
    }

    func testApprovedCommandRunsInWorkspaceAndCapturesOutput() async throws {
        let root = try makeWorkspace()
        let service = MTPLXWorkspaceToolService()
        let args = #"{"command":"printf workspace-tool","timeout_seconds":5}"#

        let pending = try await object(
            service.dispatch(
                name: "run_command",
                argumentsJSON: args,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(pending["error"] as? String, "approval_required")

        let result = try await object(
            service.dispatch(
                name: "run_command",
                argumentsJSON: args,
                workspaceRoot: root.path,
                approved: true
            )
        )
        XCTAssertEqual(result["exit_code"] as? Int, 0)
        XCTAssertEqual(result["stdout"] as? String, "workspace-tool")
        XCTAssertEqual(result["timed_out"] as? Bool, false)

        let timed = try await object(
            service.dispatch(
                name: "run_command",
                argumentsJSON: #"{"command":"sleep 2","timeout_seconds":1}"#,
                workspaceRoot: root.path,
                approved: true
            )
        )
        XCTAssertEqual(timed["timed_out"] as? Bool, true)
    }

    func testRepositoryToolsAndPatchWorkflow() async throws {
        let root = try makeWorkspace()
        try Data("# original\n".utf8)
            .write(to: root.appendingPathComponent("README.md"), options: .atomic)
        XCTAssertTrue(try run(["git", "init", "-q"], in: root).isEmpty)
        _ = try run(["git", "add", "README.md"], in: root)
        let service = MTPLXWorkspaceToolService()

        let names = service.toolDefinitions().map { $0.function.name }
        XCTAssertEqual(
            names,
            [
                "list_files", "read_file", "search_files", "inspect_repo",
                "git_status", "git_diff", "write_file", "apply_patch",
                "run_tests", "run_command"
            ]
        )

        let repo = try await object(
            service.dispatch(
                name: "inspect_repo",
                argumentsJSON: "{}",
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(repo["is_git_repository"] as? Bool, true)
        XCTAssertTrue((repo["project_markers"] as? [String])?.contains("Package.swift") == false)

        let status = try await object(
            service.dispatch(
                name: "git_status",
                argumentsJSON: "{}",
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(status["exit_code"] as? Int, 0)
        XCTAssertTrue((status["stdout"] as? String)?.contains("## No commits yet on") == true)

        let stagedDiff = try await object(
            service.dispatch(
                name: "git_diff",
                argumentsJSON: #"{"scope":"staged"}"#,
                workspaceRoot: root.path,
                approved: false
            )
        )
        let stagedScopes = try XCTUnwrap(stagedDiff["scopes"] as? [[String: Any]])
        XCTAssertTrue((stagedScopes.first?["diff"] as? String)?.contains("+# original") == true)

        let patch = [
            "diff --git a/README.md b/README.md",
            "--- a/README.md",
            "+++ b/README.md",
            "@@ -1 +1 @@",
            "-# original",
            "+# patched",
            ""
        ].joined(separator: "\n")
        let pending = try await object(
            service.dispatch(
                name: "apply_patch",
                argumentsJSON: try String(
                    data: JSONSerialization.data(withJSONObject: ["patch": patch]),
                    encoding: .utf8
                ).unwrap()
                ,
                workspaceRoot: root.path,
                approved: false
            )
        )
        XCTAssertEqual(pending["error"] as? String, "approval_required")

        let applied = try await object(
            service.dispatch(
                name: "apply_patch",
                argumentsJSON: try String(
                    data: JSONSerialization.data(withJSONObject: ["patch": patch]),
                    encoding: .utf8
                ).unwrap(),
                workspaceRoot: root.path,
                approved: true
            )
        )
        XCTAssertEqual(applied["validated"] as? Bool, true, "\(applied)")
        XCTAssertEqual(applied["applied"] as? Bool, true, "\(applied)")
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("README.md")),
            "# patched\n"
        )
    }
}

private extension Optional where Wrapped == String {
    func unwrap() throws -> String {
        try XCTUnwrap(self)
    }
}
