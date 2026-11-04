import XCTest
import Foundation
@testable import MTPLXAppCore

/// What the app asks the engine for while it serves. Every dashboard
/// snapshot is built inside the serving process while it decodes, so the
/// app asks for one every 500 ms at most (the engine builds them only while
/// a request runs) and every second while its window cannot be seen. The
/// live gauge reads progress frames, which keep the configured 100 ms.
@MainActor
final class MetricsCadenceTests: XCTestCase {
    private func store(_ configuration: MTPLXAppConfiguration = MTPLXAppConfiguration()) -> MTPLXBackendStore {
        let settings = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-cadence-\(UUID().uuidString)")
            .appendingPathComponent("settings.json")
        return MTPLXBackendStore(
            configuration: configuration,
            settingsStore: MTPLXSettingsStore(settingsURL: settings)
        )
    }

    func testDefaultCadenceAsksForSnapshotsEvery500msAndKeepsTheGaugeAt100ms() {
        let backend = store()
        XCTAssertEqual(backend.configuration.streamSnapshotIntervalMs, 100)
        // 10 snapshots a second used to be built inside the engine during
        // every Pi or OpenCode decode; the floor makes it 2.
        XCTAssertEqual(backend.metricsSnapshotIntervalMs, 500)
        XCTAssertEqual(backend.progressFrameIntervalMs, 100)
    }

    func testASlowerChosenCadenceIsKept() {
        var configuration = MTPLXAppConfiguration()
        configuration.streamSnapshotIntervalMs = 2000
        let backend = store(configuration)
        XCTAssertEqual(backend.metricsSnapshotIntervalMs, 2000)
        XCTAssertEqual(backend.progressFrameIntervalMs, 2000)
    }

    func testPerformanceModeStaysAtOneSecond() {
        var configuration = MTPLXAppConfiguration()
        configuration.performanceLock = true
        let backend = store(configuration)
        XCTAssertEqual(backend.metricsSnapshotIntervalMs, 1000)
        XCTAssertEqual(backend.progressFrameIntervalMs, 1000)
    }

    func testAHiddenWindowDropsSnapshotsAndProgressToOneSecondUntilItShowsAgain() {
        let backend = store()
        backend.setDashboardVisible(false)
        XCTAssertEqual(backend.metricsSnapshotIntervalMs, 1000)
        XCTAssertEqual(backend.progressFrameIntervalMs, 1000)
        backend.setDashboardVisible(true)
        XCTAssertEqual(backend.metricsSnapshotIntervalMs, 500)
        XCTAssertEqual(backend.progressFrameIntervalMs, 100)
    }
}
