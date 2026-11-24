import Foundation
import XCTest

@testable import MTPLXAppCore

@MainActor
final class SettingsPersistenceRegressionTests: XCTestCase {
    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-settings-persistence-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String, in directory: URL, script: String = "#!/bin/sh\nexit 0\n") throws -> URL {
        let url = directory.appendingPathComponent(name)
        try script.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    func testInklingServeCommandPreservesSavedMaxFanModeAndReasoningSettings() throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let fake = try makeExecutable(named: "mtplx", in: root)
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": root.path,
            "HOME": root.path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": String(128 * 1_073_741_824),
        ])

        var config = MTPLXAppConfiguration(
            executablePath: fake.path,
            model: "inkling2-mini-optimized-speed",
            profile: "auto"
        )
        config.fanMode = "max"
        config.pinFansAtMaxOnStart = true
        config.reasoning = "on"
        config.reasoningEffort = "high"
        config.liveSettingsModelFamily = "inkling2"

        let command = try builder.buildServeCommand(configuration: config, target: .chat, launchID: "inkling-max-fans")
        XCTAssertTrue(command.arguments.contains("--fan-mode"), "expected --fan-mode in \(command.arguments)")
        if let idx = command.arguments.firstIndex(of: "--fan-mode") {
            XCTAssertEqual(command.arguments[idx + 1], "max")
        }
        XCTAssertTrue(command.arguments.contains("--require-max-fans"))
        if let idx = command.arguments.firstIndex(of: "--reasoning") {
            XCTAssertEqual(command.arguments[idx + 1], "on")
        } else {
            XCTFail("expected --reasoning on in \(command.arguments)")
        }
        if let idx = command.arguments.firstIndex(of: "--reasoning-effort") {
            XCTAssertEqual(command.arguments[idx + 1], "high")
        } else {
            XCTFail("expected --reasoning-effort high in \(command.arguments)")
        }

        config.reasoning = "off"
        let offCommand = try builder.buildServeCommand(configuration: config, target: .chat, launchID: "inkling-reasoning-off")
        if let idx = offCommand.arguments.firstIndex(of: "--reasoning") {
            XCTAssertEqual(offCommand.arguments[idx + 1], "off")
        } else {
            XCTFail("expected --reasoning off in \(offCommand.arguments)")
        }
        XCTAssertFalse(offCommand.arguments.contains("--reasoning-effort"))
    }

    func testEffectiveFanModeStaysMaxAfterStopRestoresHardwareFans() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)

        let initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: 19_912,
            fanMode: "max",
            pinFansAtMaxOnStart: true
        )
        try settingsStore.save(initial)

        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore,
            localFanRestorer: { true }
        )
        backend.ownServerRanSinceFanRestore = true

        await backend.stopDaemon()
        await backend.awaitDaemonTeardown()

        // Hardware fans were restored to Apple's curve, so currentFanMode records "default"...
        XCTAssertEqual(backend.currentFanMode, "default")
        // ...but the user's saved launch preference and Settings effectiveFanMode stay Max.
        XCTAssertEqual(backend.configuration.fanMode, "max")
        XCTAssertTrue(backend.configuration.pinFansAtMaxOnStart)
        XCTAssertEqual(backend.effectiveFanMode, .max)
        let persisted = try settingsStore.load()
        XCTAssertEqual(persisted.fanMode, "max")
        XCTAssertTrue(persisted.pinFansAtMaxOnStart)
    }

    func testUnverifiedDaemonFanFallbackNeverOverwritesSavedMaxFanMode() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)
        let sleeper = try makeExecutable(named: "sleep-daemon", in: root, script: "#!/bin/sh\nsleep 30\n")
        let supervisor = DaemonSupervisor()
        defer { Task { await supervisor.stop(graceSeconds: 0) } }

        let command = DaemonCommand(executableURL: sleeper, arguments: [])
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )

        var initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: 19_913,
            fanMode: "max",
            pinFansAtMaxOnStart: true
        )
        initial.reasoning = "on"
        initial.reasoningEffort = "high"
        try settingsStore.save(initial)

        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore,
            supervisor: supervisor,
            fanModeSetter: { _, _, _, _ in
                FanModeResponse(verified: false, currentMode: "default")
            }
        )
        let snap = supervisor.supervisionSnapshot()
        backend.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: snap.revision,
                state: .running,
                restartStatus: .runningAfterRestart(attempt: 1),
                restartCount: 0,
                lifecycleEpoch: snap.lifecycleEpoch,
                recoveryGeneration: 0
            )
        )

        try await backend.setFanMode(MTPLXFanMode.max.rawValue)

        XCTAssertEqual(backend.configuration.fanMode, "max")
        XCTAssertTrue(backend.configuration.pinFansAtMaxOnStart)
        XCTAssertEqual(try settingsStore.load().fanMode, "max")
        XCTAssertEqual(backend.currentFanMode, "default")
        await supervisor.stop(graceSeconds: 0)
    }

    func testSetFanModeFailureDoesNotRevertConcurrentSettingsEdits() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)
        let sleeper = try makeExecutable(named: "sleep-daemon", in: root, script: "#!/bin/sh\nsleep 30\n")
        let supervisor = DaemonSupervisor()
        defer { Task { await supervisor.stop(graceSeconds: 0) } }

        let command = DaemonCommand(executableURL: sleeper, arguments: [])
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )

        enum FakeFanError: Error { case timedOut }
        let entered = expectation(description: "entered fanModeSetter")
        let releaseGate = AsyncGate()

        var initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: 19_914
        )
        initial.fanMode = "smart"
        initial.reasoning = "on"
        initial.reasoningEffort = "medium"
        initial.liveSettingsModelFamily = "inkling2"
        try settingsStore.save(initial)

        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore,
            supervisor: supervisor,
            fanModeSetter: { _, _, _, _ in
                entered.fulfill()
                await releaseGate.wait()
                throw FakeFanError.timedOut
            }
        )
        let snap = supervisor.supervisionSnapshot()
        backend.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: snap.revision,
                state: .running,
                restartStatus: .runningAfterRestart(attempt: 1),
                restartCount: 0,
                lifecycleEpoch: snap.lifecycleEpoch,
                recoveryGeneration: 0
            )
        )

        let fanTask = Task { @MainActor in
            try await backend.setFanMode(MTPLXFanMode.max.rawValue)
        }
        await fulfillment(of: [entered], timeout: 2.0)

        // Meanwhile, the user edits another setting and saves.
        var concurrent = backend.configuration
        concurrent.reasoning = "off"
        concurrent.reasoningEffort = "low"
        concurrent.memoryLimitGB = 80
        try backend.saveSettings(concurrent)

        await releaseGate.open()
        do {
            try await fanTask.value
            XCTFail("expected setFanMode to throw")
        } catch {
            // Expected timeout failure
        }

        XCTAssertEqual(backend.configuration.reasoning, "off")
        XCTAssertEqual(backend.configuration.reasoningEffort, "low")
        XCTAssertEqual(backend.configuration.memoryLimitGB, 80)
        await supervisor.stop(graceSeconds: 0)
    }

    func testPartialLiveSettingsPatchWhileStoppedDoesNotBakeStaleDaemonSamplerIntoConfig() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)

        var initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: 19_915
        )
        initial.fanMode = "max"
        initial.pinFansAtMaxOnStart = true
        initial.liveSettingsModelFamily = "inkling2"
        try settingsStore.save(initial)

        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore
        )

        // Update only reasoning while stopped: should succeed locally without
        // attempting HTTP and without writing nil/default sampler overrides.
        try await backend.updateLiveSettings(
            MutableSettings(reasoning: "off", reasoningEffort: "low")
        )

        XCTAssertEqual(backend.configuration.reasoning, "off")
        XCTAssertEqual(backend.configuration.reasoningEffort, "low")
        XCTAssertEqual(backend.configuration.liveSettingsModelFamily, "inkling2")
        XCTAssertNil(backend.configuration.temperature)
        XCTAssertNil(backend.configuration.topP)
        XCTAssertNil(backend.configuration.topK)
        XCTAssertEqual(backend.configuration.fanMode, "max")

        let persisted = try settingsStore.load()
        XCTAssertEqual(persisted.reasoning, "off")
        XCTAssertEqual(persisted.reasoningEffort, "low")
        XCTAssertEqual(persisted.liveSettingsModelFamily, "inkling2")
        XCTAssertNil(persisted.temperature)
        XCTAssertEqual(persisted.fanMode, "max")
    }

    func testSaveSettingsWhileRunningKeepsLiveModelControlsIntact() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)
        let sleeper = try makeExecutable(named: "sleep-daemon", in: root, script: "#!/bin/sh\nsleep 30\n")
        let supervisor = DaemonSupervisor()
        defer { Task { await supervisor.stop(graceSeconds: 0) } }

        let command = DaemonCommand(executableURL: sleeper, arguments: [])
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )

        var initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: 19_916
        )
        initial.fanMode = "default"
        initial.liveSettingsModelFamily = "inkling2"
        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore,
            supervisor: supervisor
        )
        let snap = supervisor.supervisionSnapshot()
        backend.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: snap.revision,
                state: .running,
                restartStatus: .runningAfterRestart(attempt: 1),
                restartCount: 0,
                lifecycleEpoch: snap.lifecycleEpoch,
                recoveryGeneration: 0
            )
        )

        // Seed live settings via a stopped-safe update before marking running, or via snapshot JSON.
        var next = backend.configuration
        next.fanMode = "max"
        try backend.saveSettings(next)

        XCTAssertEqual(backend.configuration.fanMode, "max")
        XCTAssertTrue(backend.configuration.pinFansAtMaxOnStart)
        await supervisor.stop(graceSeconds: 0)
    }

    func testPassiveSnapshotNeverOverwritesSavedUserConfiguration() async throws {
        let root = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let settingsURL = root.appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: settingsURL)

        // Start a tiny local HTTP server serving `/v1/mtplx/settings` with
        // daemon defaults that intentionally differ from the user's saved config.
        let python = try makeExecutable(
            named: "mock_settings_server.py",
            in: root,
            script: """
            #!/usr/bin/env python3
            import http.server, json, socketserver, sys
            port = int(sys.argv[1])
            payload = {
                "generation_mode": "ar",
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "reasoning": "on",
                "reasoning_effort": "low",
                "model_family": "inkling2",
            }
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                def log_message(self, *args):
                    pass
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
                httpd.serve_forever()
            """
        )
        let port = try freeLocalPort()
        let supervisor = DaemonSupervisor()
        defer { Task { await supervisor.stop(graceSeconds: 0) } }

        _ = try await supervisor.start(
            command: DaemonCommand(executableURL: python, arguments: [String(port)]),
            healthBaseURL: URL(string: "http://127.0.0.1:\(port)")!,
            probeHealth: false
        )
        // Wait briefly for the mock HTTP server socket to bind.
        let probeURL = URL(string: "http://127.0.0.1:\(port)/v1/mtplx/settings")!
        for _ in 0..<40 {
            if let (_, resp) = try? await URLSession.shared.data(from: probeURL),
               (resp as? HTTPURLResponse)?.statusCode == 200 {
                break
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }

        var initial = MTPLXAppConfiguration(
            model: "inkling2-mini-optimized-speed",
            port: port
        )
        initial.generationMode = "mtp"
        initial.fanMode = "max"
        initial.pinFansAtMaxOnStart = true
        initial.temperature = nil
        initial.topP = nil
        initial.topK = nil
        initial.reasoning = "off"
        initial.reasoningEffort = "high"
        initial.liveSettingsModelFamily = "inkling2"
        try settingsStore.save(initial)

        let backend = MTPLXBackendStore(
            configuration: initial,
            settingsStore: settingsStore,
            supervisor: supervisor
        )
        let snap = supervisor.supervisionSnapshot()
        backend.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: snap.revision,
                state: .running,
                restartStatus: .runningAfterRestart(attempt: 1),
                restartCount: 0,
                lifecycleEpoch: snap.lifecycleEpoch,
                recoveryGeneration: 0
            )
        )

        // Even with persist: true, passive daemon reads must only update live
        // in-memory `backend.settings`, NEVER clobber `configuration` or disk.
        try await backend.refreshLiveSettingsFromDaemon(persist: true)

        XCTAssertEqual(backend.settings?.generationMode, "ar")
        XCTAssertEqual(backend.settings?.temperature, 0.6)
        XCTAssertEqual(backend.configuration.generationMode, "mtp")
        XCTAssertNil(backend.configuration.temperature)
        XCTAssertNil(backend.configuration.topP)
        XCTAssertNil(backend.configuration.topK)
        XCTAssertEqual(backend.configuration.reasoning, "off")
        XCTAssertEqual(backend.configuration.reasoningEffort, "high")
        XCTAssertEqual(backend.configuration.fanMode, "max")
        XCTAssertTrue(backend.configuration.pinFansAtMaxOnStart)

        // Saving an unrelated setting while running must not wipe the live
        // daemon `backend.settings` snapshot (e.g. modelControls/reasoningPolicy).
        var updated = backend.configuration
        updated.performanceLock = true
        try backend.saveSettings(updated)
        XCTAssertEqual(backend.settings?.generationMode, "ar")
        XCTAssertEqual(backend.settings?.temperature, 0.6)

        let persisted = try settingsStore.load()
        XCTAssertEqual(persisted.generationMode, "mtp")
        XCTAssertNil(persisted.temperature)
        XCTAssertEqual(persisted.reasoning, "off")
        XCTAssertEqual(persisted.reasoningEffort, "high")
        XCTAssertEqual(persisted.fanMode, "max")
        XCTAssertTrue(persisted.performanceLock)
        await supervisor.stop(graceSeconds: 0)
    }

    private func freeLocalPort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw NSError(domain: "Socket", code: Int(errno)) }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else { throw NSError(domain: "Socket", code: Int(errno)) }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        guard nameResult == 0 else { throw NSError(domain: "Socket", code: Int(errno)) }
        return Int(UInt16(bigEndian: addr.sin_port))
    }
}

private actor AsyncGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func open() {
        isOpen = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending {
            waiter.resume()
        }
    }
}
