import Foundation
import XCTest
@testable import MTPLXAppCore

final class HermesExecutionPolicyTests: XCTestCase {
    func testSyncInheritsRootExecutionPolicyAndPreservesProfileChoice() throws {
        for backend in [nil, "local", "ssh"] as [String?] {
            for indent in ["  ", "    "] {
                let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
                let root = "terminal:\n" + ["backend: docker", "docker_image: example/coding:test",
                    "docker_volumes: ['/data:/data:ro']"].map { indent + $0 }.joined(separator: "\n")
                    + "\nproviders:\n  private: untouched\n"
                try root.write(to: home.appendingPathComponent("config.yaml"), atomically: true, encoding: .utf8)
                let profile = home.appendingPathComponent("profiles/mtplx/config.yaml")
                if let backend {
                    try FileManager.default.createDirectory(at: profile.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try "terminal:\n  backend: \(backend)\n  ssh_host: chosen-host\n"
                        .write(to: profile, atomically: true, encoding: .utf8)
                }
                let integration = HermesIntegration(hermesHome: home,
                    environment: ["HOME": home.path, "PATH": "/usr/bin:/bin"],
                    terminalCommandURL: home.appendingPathComponent("open-hermes.command"))
                let configuration = MTPLXAppConfiguration(model: "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
                    host: "127.0.0.1", port: 8123, apiKey: "", hermesWorkspacePath: home.path)
                _ = try integration.sync(configuration: configuration)
                let text = try String(contentsOf: profile, encoding: .utf8)
                XCTAssertTrue(text.contains("  backend: \(backend ?? "docker")\n"))
                XCTAssertFalse(text.contains("providers:"))
                if backend == nil {
                    XCTAssertTrue(text.contains("  docker_image: example/coding:test\n"))
                    XCTAssertTrue(text.contains("  docker_volumes: ['/data:/data:ro']\n"))
                } else {
                    XCTAssertFalse(text.contains("docker_image:"))
                    XCTAssertTrue(text.contains("  ssh_host: chosen-host\n"))
                }
                XCTAssertFalse(try integration.sync(configuration: configuration).didChange)
            }
        }
    }
}
