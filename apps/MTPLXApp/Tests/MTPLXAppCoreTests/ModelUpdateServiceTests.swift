import XCTest

@testable import MTPLXAppCore

/// Sparkle-for-models (2.9.0): the app surfaces `mtplx models --check`
/// verdicts and drives one-click delta updates through the ordinary pull
/// pipeline. These tests pin the CLI JSON contract and the store plumbing.
final class ModelUpdateServiceTests: XCTestCase {
    private actor Counter {
        private(set) var value = 0
        func increment() { value += 1 }
    }

    func testModelUpdatePayloadDecodesCLIJSON() throws {
        let json = #"""
        {"cache_dir": "/x", "engine_version": "2.9.0", "updates_available": 1,
         "models": [{"repo_id": "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
           "path": "/x/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
           "state": "update-available", "local_revision": "aaa",
           "remote_revision": "bbb", "source": "manifest",
           "note": "Quantized MTP head: smaller and faster",
           "min_engine_version": "2.7.0", "update_bytes": 451270880,
           "changed_files": ["mtp.safetensors", "config.json"]}]}
        """#
        let payload = try JSONDecoder().decode(
            ModelUpdateCheckPayload.self,
            from: Data(json.utf8)
        )
        XCTAssertEqual(payload.updatesAvailable, 1)
        let row = try XCTUnwrap(payload.models.first)
        XCTAssertTrue(row.isUpdateAvailable)
        XCTAssertFalse(row.requiresEngineUpdate)
        XCTAssertEqual(row.shortName, "Qwen3.8-27B-MTPLX-Optimized-Speed")
        XCTAssertEqual(row.updateBytes, 451_270_880)
        XCTAssertEqual(row.changedFiles, ["mtp.safetensors", "config.json"])
    }

    func testEngineGateStateDecodes() throws {
        let json = #"""
        {"models": [{"repo_id": "a/b", "path": null,
          "state": "engine-update-required", "local_revision": null,
          "remote_revision": "ccc", "source": "manifest", "note": null,
          "min_engine_version": "9.9.9", "update_bytes": null,
          "changed_files": []}]}
        """#
        let payload = try JSONDecoder().decode(
            ModelUpdateCheckPayload.self,
            from: Data(json.utf8)
        )
        let row = try XCTUnwrap(payload.models.first)
        XCTAssertTrue(row.requiresEngineUpdate)
        XCTAssertFalse(row.isUpdateAvailable)
        XCTAssertEqual(row.minEngineVersion, "9.9.9")
    }

    @MainActor
    func testRefreshModelUpdatesPublishesAndThrottles() async {
        let calls = Counter()
        let row = ModelUpdateInfo(
            repoID: "Youssofal/Pack",
            state: "update-available",
            updateBytes: 42
        )
        let store = MTPLXBackendStore(
            modelUpdateChecker: {
                await calls.increment()
                return [row]
            }
        )
        await store.refreshModelUpdates()
        XCTAssertEqual(store.modelUpdates, [row])
        XCTAssertEqual(store.availableModelPackUpdates, [row])

        // Within the 6 h window a plain refresh is a no-op.
        await store.refreshModelUpdates()
        let afterThrottle = await calls.value
        XCTAssertEqual(afterThrottle, 1)

        await store.refreshModelUpdates(force: true)
        let afterForce = await calls.value
        XCTAssertEqual(afterForce, 2)
    }

    @MainActor
    func testRefreshFailureKeepsPriorRowsAndStaysQuiet() async {
        struct Boom: Error {}
        let store = MTPLXBackendStore(
            modelUpdateChecker: { throw Boom() }
        )
        await store.refreshModelUpdates()
        XCTAssertEqual(store.modelUpdates, [])
        XCTAssertNil(store.modelPackUpdatingRepoID)
    }

    func testUpdateStreamInvokesModelsUpdateNotPull() {
        // The 2.9.0 Update button shelled plain `pull`, which reuses
        // fresh-looking legacy caches and silently no-ops on exactly the
        // packs an update targets. Updates must go through models --update.
        XCTAssertEqual(
            ModelDownloader.streamArguments(repo: "owner/pack", update: true),
            ["models", "--update", "owner/pack", "--progress-json"]
        )
        XCTAssertEqual(
            ModelDownloader.streamArguments(repo: "owner/pack", update: false),
            ["pull", "owner/pack", "--progress-json"]
        )
        XCTAssertEqual(
            ModelDownloader.streamArguments(
                repo: "owner/pack",
                update: true,
                destinationPath: "/models/legacy-pack"
            ),
            [
                "models", "--update", "owner/pack", "--progress-json",
                "--installed-path", "/models/legacy-pack",
            ]
        )
    }
}
