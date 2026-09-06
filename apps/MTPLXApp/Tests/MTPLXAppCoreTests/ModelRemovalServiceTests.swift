import Foundation
import XCTest

@testable import MTPLXAppCore

final class ModelRemovalServiceTests: XCTestCase {
    func testRemovalResultDecodesCLIJSON() throws {
        let data = Data(
            #"{"repo_id":"owner/pack","path":"/cache/owner--pack","removed":true,"size_bytes_removed":42000000000}"#.utf8
        )

        let result = try JSONDecoder().decode(CachedModelRemovalResult.self, from: data)

        XCTAssertEqual(result.repoID, "owner/pack")
        XCTAssertEqual(result.path, "/cache/owner--pack")
        XCTAssertTrue(result.removed)
        XCTAssertEqual(result.sizeBytesRemoved, 42_000_000_000)
    }

    func testRemovalArgumentsUsePublicCLIAndExplicitCacheRoot() {
        XCTAssertEqual(
            ModelDownloader.removalArguments(
                repo: "owner/pack",
                cacheRoot: URL(fileURLWithPath: "/models")
            ),
            [
                "remove", "owner/pack", "--missing-ok", "--json",
                "--cache-dir", "/models",
            ]
        )
    }

    func testCachedModelReferenceOnlyAcceptsDirectManagedChildren() {
        let downloader = ModelDownloader(
            processEnvironment: ["HOME": "/Users/test"],
            modelCacheRoot: URL(fileURLWithPath: "/models", isDirectory: true)
        )

        XCTAssertEqual(
            downloader.cachedModelReference(forInstalledPath: "/models/owner--pack"),
            "owner/pack"
        )
        XCTAssertNil(
            downloader.cachedModelReference(forInstalledPath: "/models/nested/owner--pack")
        )
        XCTAssertNil(
            downloader.cachedModelReference(forInstalledPath: "/Users/test/models/owner--pack")
        )
    }

    func testCachedModelReferenceRefusesSymlinkedUserManagedModel() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-link-\(UUID().uuidString)", isDirectory: true)
        let cache = root.appendingPathComponent("cache", isDirectory: true)
        let external = root.appendingPathComponent("external", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = cache.appendingPathComponent("owner--pack", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: external)
        let downloader = ModelDownloader(modelCacheRoot: cache)

        XCTAssertNil(downloader.cachedModelReference(forInstalledPath: link.path))
    }

    func testRemovalRunsCLIAndDecodesFreedBytes() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = root.appendingPathComponent("mtplx")
        let argumentsLog = root.appendingPathComponent("arguments.log")
        let script = """
        #!/bin/sh
        printf '%s' "$*" > "$MTPLX_ARGUMENTS_LOG"
        printf '%s\n' '{"repo_id":"owner/pack","path":"/cache/owner--pack","removed":true,"size_bytes_removed":1234}'
        """
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: executable.path
        )
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "MTPLX_ARGUMENTS_LOG": argumentsLog.path,
            ],
            modelCacheRoot: cacheRoot,
            executableOverride: executable
        )

        let result = try await downloader.removeCachedModel(repo: "owner/pack")

        XCTAssertTrue(result.removed)
        XCTAssertEqual(result.sizeBytesRemoved, 1_234)
        XCTAssertEqual(
            try String(contentsOf: argumentsLog, encoding: .utf8),
            "remove owner/pack --missing-ok --json --cache-dir \(cacheRoot.path)"
        )
    }

    @MainActor
    func testBackendRefusesToRemoveSelectedModel() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-selected-remove-\(UUID().uuidString)", isDirectory: true)
        let installed = root.appendingPathComponent("owner--pack", isDirectory: true)
        var configuration = MTPLXAppConfiguration()
        configuration.model = installed.path
        let store = MTPLXBackendStore(
            configuration: configuration,
            modelDownloader: ModelDownloader(modelCacheRoot: root)
        )

        do {
            _ = try await store.removeCachedModel(
                repoID: "owner/pack",
                installedPath: installed.path
            )
            XCTFail("expected selected-model refusal")
        } catch let error as CachedModelRemovalError {
            XCTAssertEqual(error, .selectedModel)
        }
    }
}
