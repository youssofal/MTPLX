import Foundation
import XCTest
@testable import MTPLXAppCore

final class HermesExecutionPolicyTests: XCTestCase {
    func testVisionMetadataAndWeightedPrefill() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try "{\"vision_config\":{}}".write(to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        XCTAssertFalse(MTPLXModelOption.supportsVision(model: directory.path))
        try "{\"weight_map\":{\"model.visual.patch_embed.proj.weight\":\"model-1.safetensors\"}}".write(
            to: directory.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        XCTAssertTrue(MTPLXModelOption.supportsVision(model: directory.path))
        let rows = [MetricsLatest(values: ["new_prefill_tokens": .number(30000), "prefill_compute_tok_s": .number(600)]),
                    MetricsLatest(values: ["new_prefill_tokens": .number(100), "prefill_compute_tok_s": .number(200)])]
        XCTAssertEqual(try XCTUnwrap(MetricsLatest.aggregatePrefillRate(rows)), 30100 / 50.5, accuracy: 0.0001)
    }
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
                XCTAssertTrue(text.contains("compression:\n  tool_image_retention: until_compaction\n"))
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

    func testSyncReadsRootConfigOnlyWhenInheriting() throws {
        try XCTSkipIf(getuid() == 0, "root reads through mode 000")
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let root = home.appendingPathComponent("config.yaml")
        try "terminal:\n  backend: docker\n".write(to: root, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: root.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: root.path) }
        let profile = home.appendingPathComponent("profiles/mtplx/config.yaml")
        try FileManager.default.createDirectory(at: profile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "terminal:\n  backend: ssh\n  ssh_host: chosen-host\n".write(to: profile, atomically: true, encoding: .utf8)
        let integration = HermesIntegration(hermesHome: home,
            environment: ["HOME": home.path, "PATH": "/usr/bin:/bin"],
            terminalCommandURL: home.appendingPathComponent("open-hermes.command"))
        let configuration = MTPLXAppConfiguration(model: "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
            host: "127.0.0.1", port: 8123, apiKey: "", hermesWorkspacePath: home.path)
        // An explicit profile backend never depends on the root config, so the
        // unreadable root must not fail this profile's sync.
        _ = try integration.sync(configuration: configuration)
        XCTAssertTrue(try String(contentsOf: profile, encoding: .utf8).contains("  backend: ssh\n"))
        // Without a backend the root has to be read; unreadable is a thrown
        // error, not a silently dropped sandbox choice.
        try "terminal:\n  cwd: /workspace\n".write(to: profile, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try integration.sync(configuration: configuration))
    }
}
