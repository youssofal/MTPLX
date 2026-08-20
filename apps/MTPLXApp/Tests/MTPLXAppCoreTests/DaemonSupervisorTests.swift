import Foundation
import Darwin
import XCTest
@testable import MTPLXAppCore

private actor SupervisionSignal {
    private enum WaitKind: Sendable {
        case scheduled
        case exhausted
        case stopped
        case crashed
        case warming
        case stopping
    }

    private var latest = DaemonSupervisionSnapshot(
        state: .stopped,
        restartStatus: .idle,
        restartCount: 0,
        recoveryGeneration: 0
    )
    private var sawNonStoppedState = false
    private var scheduledWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var exhaustedWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var stoppedWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var crashedWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var warmingWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var stoppingWaiters: [UUID: CheckedContinuation<Bool, Never>] = [:]
    private var recoveryWaiters: [UUID: (target: Int, continuation: CheckedContinuation<Bool, Never>)] = [:]

    func record(_ snapshot: DaemonSupervisionSnapshot) {
        latest = snapshot
        if snapshot.state != .stopped {
            sawNonStoppedState = true
        }
        if case .scheduled = snapshot.restartStatus {
            scheduledWaiters.values.forEach { $0.resume(returning: true) }
            scheduledWaiters.removeAll()
        }
        if case .exhausted = snapshot.restartStatus {
            exhaustedWaiters.values.forEach { $0.resume(returning: true) }
            exhaustedWaiters.removeAll()
        }
        if sawNonStoppedState, snapshot.state == .stopped {
            stoppedWaiters.values.forEach { $0.resume(returning: true) }
            stoppedWaiters.removeAll()
        }
        if case .crashed = snapshot.state {
            crashedWaiters.values.forEach { $0.resume(returning: true) }
            crashedWaiters.removeAll()
        }
        if snapshot.state == .warming {
            warmingWaiters.values.forEach { $0.resume(returning: true) }
            warmingWaiters.removeAll()
        }
        if snapshot.state == .stopping {
            stoppingWaiters.values.forEach { $0.resume(returning: true) }
            stoppingWaiters.removeAll()
        }
        let reachedRecovery = recoveryWaiters.filter {
            snapshot.recoveryGeneration >= $0.value.target
        }
        reachedRecovery.values.forEach { $0.continuation.resume(returning: true) }
        reachedRecovery.keys.forEach { recoveryWaiters.removeValue(forKey: $0) }
    }

    func waitForScheduled() async -> Bool {
        if case .scheduled = latest.restartStatus { return true }
        return await wait(kind: .scheduled)
    }

    func waitForExhausted() async -> Bool {
        if case .exhausted = latest.restartStatus { return true }
        return await wait(kind: .exhausted)
    }

    func waitForStoppedAfterLaunch() async -> Bool {
        if sawNonStoppedState, latest.state == .stopped { return true }
        return await wait(kind: .stopped)
    }

    func waitForCrash() async -> Bool {
        if case .crashed = latest.state { return true }
        return await wait(kind: .crashed)
    }

    func waitForWarming() async -> Bool {
        if latest.state == .warming { return true }
        return await wait(kind: .warming)
    }

    func waitForStopping() async -> Bool {
        if latest.state == .stopping { return true }
        return await wait(kind: .stopping)
    }

    func waitForRecoveryGeneration(_ target: Int) async -> Bool {
        if latest.recoveryGeneration >= target { return true }
        let id = UUID()
        return await withCheckedContinuation { continuation in
            recoveryWaiters[id] = (target, continuation)
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await self?.timeoutRecovery(id)
            }
        }
    }

    private func wait(kind: WaitKind) async -> Bool {
        let id = UUID()
        return await withCheckedContinuation { continuation in
            switch kind {
            case .scheduled: scheduledWaiters[id] = continuation
            case .exhausted: exhaustedWaiters[id] = continuation
            case .stopped: stoppedWaiters[id] = continuation
            case .crashed: crashedWaiters[id] = continuation
            case .warming: warmingWaiters[id] = continuation
            case .stopping: stoppingWaiters[id] = continuation
            }
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                await self?.timeout(id, kind: kind)
            }
        }
    }

    private func timeout(_ id: UUID, kind: WaitKind) {
        let continuation: CheckedContinuation<Bool, Never>?
        switch kind {
        case .scheduled: continuation = scheduledWaiters.removeValue(forKey: id)
        case .exhausted: continuation = exhaustedWaiters.removeValue(forKey: id)
        case .stopped: continuation = stoppedWaiters.removeValue(forKey: id)
        case .crashed: continuation = crashedWaiters.removeValue(forKey: id)
        case .warming: continuation = warmingWaiters.removeValue(forKey: id)
        case .stopping: continuation = stoppingWaiters.removeValue(forKey: id)
        }
        continuation?.resume(returning: false)
    }

    private func timeoutRecovery(_ id: UUID) {
        recoveryWaiters.removeValue(forKey: id)?.continuation.resume(returning: false)
    }
}

private actor RestartGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func release() {
        continuation?.resume()
        continuation = nil
    }
}

private actor BeforeRunGate {
    private var armed = false
    private var didEnter = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func arm() {
        armed = true
        didEnter = false
    }

    func waitIfArmed() async {
        guard armed else { return }
        armed = false
        didEnter = true
        let entered = enteredWaiters
        enteredWaiters.removeAll()
        entered.forEach { $0.resume() }
        await withCheckedContinuation { releaseContinuation = $0 }
    }

    func waitUntilEntered() async {
        if didEnter { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private actor RestartDelayRecorder {
    private var values: [TimeInterval] = []

    func record(_ delay: TimeInterval) {
        values.append(delay)
    }

    func snapshot() -> [TimeInterval] {
        values
    }
}

private final class TerminationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var armed = false
    private let entered = DispatchSemaphore(value: 0)
    private let release = DispatchSemaphore(value: 0)

    func arm() {
        lock.lock()
        armed = true
        lock.unlock()
    }

    func blockIfArmed() {
        lock.lock()
        let shouldBlock = armed
        lock.unlock()
        guard shouldBlock else { return }
        entered.signal()
        _ = release.wait(timeout: .now() + 5)
    }

    func waitUntilEntered() -> Bool {
        entered.wait(timeout: .now() + 1) == .success
    }

    func unblock() {
        release.signal()
    }
}

private actor FanRestoreRecorder {
    private var calls = 0

    func restore() -> Bool {
        calls += 1
        return true
    }

    func count() -> Int { calls }
}

private actor HealthProbeGate {
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var responseContinuation: CheckedContinuation<HealthPayload?, Never>?

    func waitForResponse() async -> HealthPayload? {
        let entered = enteredWaiters
        enteredWaiters.removeAll()
        entered.forEach { $0.resume() }
        return await withCheckedContinuation { responseContinuation = $0 }
    }

    func waitUntilEntered() async {
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func release(_ health: HealthPayload?) {
        responseContinuation?.resume(returning: health)
        responseContinuation = nil
    }
}

/// Returns a ready health payload for the initial launch, then holds the
/// automatic retry inside its health wait until the test releases it.
private actor AutomaticRetryHealthGate {
    private let readyHealth: HealthPayload
    private var calls = 0
    private var secondProbeWaiters: [CheckedContinuation<Void, Never>] = []
    private var responseContinuation: CheckedContinuation<HealthPayload?, Never>?

    init(readyHealth: HealthPayload) {
        self.readyHealth = readyHealth
    }

    func probe() async -> HealthPayload? {
        calls += 1
        if calls == 1 {
            return readyHealth
        }
        let waiters = secondProbeWaiters
        secondProbeWaiters.removeAll()
        waiters.forEach { $0.resume() }
        return await withCheckedContinuation { responseContinuation = $0 }
    }

    func waitUntilAutomaticRetryProbe() async {
        guard calls >= 2 else {
            await withCheckedContinuation { secondProbeWaiters.append($0) }
            return
        }
    }

    func release(_ health: HealthPayload? = nil) {
        responseContinuation?.resume(returning: health)
        responseContinuation = nil
    }
}

final class DaemonSupervisorTests: XCTestCase {
    private func releaseControlledCommand(
        releaseFile: URL,
        exitStatus: Int32
    ) -> DaemonCommand {
        let path = shellQuoted(releaseFile.path)
        return DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "while [ -e \(path) ]; do sleep 0.01; done; exit \(exitStatus)"]
        )
    }

    private func shellQuoted(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\\"'\\\"'"))'"
    }

    private func temporaryReleaseFile() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let release = directory.appendingPathComponent("hold")
        try Data().write(to: release)
        return release
    }

    private func automaticRetryParentChildCommand(
        releaseFile: URL,
        counterFile: URL,
        parentPIDFile: URL,
        childPIDFile: URL
    ) -> DaemonCommand {
        let release = shellQuoted(releaseFile.path)
        let counter = shellQuoted(counterFile.path)
        let parent = shellQuoted(parentPIDFile.path)
        let child = shellQuoted(childPIDFile.path)
        let script = """
        count=$(cat \(counter) 2>/dev/null || echo 0)
        count=$((count + 1))
        echo "$count" > \(counter)
        if [ "$count" -eq 1 ]; then
          while [ -e \(release) ]; do sleep 0.01; done
          exit 17
        fi
        (trap 'exit 0' TERM INT; while :; do sleep 1; done) &
        child_pid=$!
        echo "$$" > \(parent)
        echo "$child_pid" > \(child)
        trap 'kill "$child_pid" 2>/dev/null; exit 0' TERM INT
        while :; do sleep 1; done
        """
        return DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", script]
        )
    }

    private func waitForPID(in file: URL) async -> pid_t? {
        for _ in 0..<100 {
            if let text = try? String(contentsOf: file),
               let value = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return value
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return nil
    }

    private func waitForExit(_ pid: pid_t) async -> Bool {
        for _ in 0..<100 {
            if kill(pid, 0) != 0 { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return kill(pid, 0) != 0
    }

    private func waitForEvidence(
        _ logs: BoundedLogStore,
        containing expected: [String]
    ) async -> String {
        for _ in 0..<100 {
            let evidence = await logs.snapshot().map(\.message).joined(separator: "\n")
            if expected.allSatisfy(evidence.contains) {
                return evidence
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return await logs.snapshot().map(\.message).joined(separator: "\n")
    }

    private func adoptedHealth(pid: Int = 12345) throws -> HealthPayload {
        let json = """
        {
          "ok": true,
          "model": "fixture",
          "model_path": "/tmp/fixture.gguf",
          "generation_mode": "mtp",
          "load_mtp": true,
          "mtp_enabled": true,
          "depth": 1,
          "profile": {},
          "context_window": 1024,
          "active_requests": 0,
          "reasoning_parser": "none",
          "startup": {"launch_id": "prior-session", "pid": \(pid)}
        }
        """
        return try JSONDecoder().decode(HealthPayload.self, from: Data(json.utf8))
    }

    func testAutomaticRestartIsOptInAndCleanExitDoesNotRestart() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 0),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
        XCTAssertEqual(snapshot.restartEligibility, .noDaemon)
    }

    func testRunFailureNeverBecomesRestartEligible() async throws {
        let supervisor = DaemonSupervisor()
        supervisor.setAutomaticRestartEnabled(true)
        do {
            _ = try await supervisor.start(
                command: DaemonCommand(
                    executableURL: URL(fileURLWithPath: "/definitely/not/a/daemon"),
                    arguments: []
                ),
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: false
            )
            XCTFail("expected launch failure")
        } catch {
            // Expected: Process.run() failed before a daemon could be owned.
        }
        XCTAssertEqual(supervisor.supervisionSnapshot().restartEligibility, .noDaemon)
    }

    func testAbnormalExitDoesNotRestartUntilEnabled() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let crashed = await signal.waitForCrash()
        XCTAssertTrue(crashed)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .crashed(17))
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
    }

    func testAdoptedPriorSessionDaemonIsExplicitlyExcludedFromAutomaticRestart() async throws {
        let health = try adoptedHealth()
        let supervisor = DaemonSupervisor(initialHealthProbe: { _, _ in health })
        supervisor.setAutomaticRestartEnabled(true)
        let adopted = try await supervisor.adoptExistingIfAppOwned(
            command: DaemonCommand(executableURL: URL(fileURLWithPath: "/bin/sh"), arguments: []),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!
        )

        XCTAssertEqual(adopted, health)
        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .running)
        XCTAssertEqual(snapshot.restartEligibility, .adoptedPriorSession)
        XCTAssertEqual(snapshot.restartStatus, .idle)
    }

    func testStopDuringAdoptionHealthProbeCannotPublishRunning() async throws {
        let health = try adoptedHealth()
        let gate = HealthProbeGate()
        let supervisor = DaemonSupervisor(initialHealthProbe: { _, _ in
            await gate.waitForResponse()
        })
        let adoptionTask = Task {
            try await supervisor.adoptExistingIfAppOwned(
                command: DaemonCommand(executableURL: URL(fileURLWithPath: "/bin/sh"), arguments: []),
                healthBaseURL: URL(string: "http://127.0.0.1:9")!
            )
        }
        await gate.waitUntilEntered()
        await supervisor.stop(graceSeconds: 0)
        await gate.release(health)
        let adopted = try await adoptionTask.value

        XCTAssertNil(adopted)
        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        XCTAssertFalse(supervisor.isRunning())
    }

    func testStopDuringHealthWaitCannotPublishRunningOrKeepRecipe() async throws {
        let health = try adoptedHealth()
        let gate = HealthProbeGate()
        let supervisor = DaemonSupervisor(
            initialHealthProbe: { _, _ in nil },
            healthWaitProbe: { _, _ in await gate.waitForResponse() }
        )
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "while :; do sleep 1; done"]
        )
        let startTask = Task {
            try await supervisor.start(
                command: command,
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: true,
                timeoutSeconds: 10
            )
        }
        await gate.waitUntilEntered()
        await supervisor.stop(graceSeconds: 0)
        await gate.release(health)
        _ = try? await startTask.value

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartEligibility, .noDaemon)
        XCTAssertFalse(supervisor.isRunning())
    }

    @MainActor
    func testPassiveCleanExitClearsActiveTransportAndConnectionState() async throws {
        let store = MTPLXBackendStore()
        let handoffID = UUID()
        let handoffDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "mtplx-passive-handoff-\(handoffID.uuidString.lowercased())",
                isDirectory: true
            )
        try FileManager.default.createDirectory(at: handoffDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: handoffDirectory) }

        // Match the real handoff's terminal child shape: the exact token must
        // live in the child environment, not merely in a PID presentation
        // list or a command-line argument.
        let pi = Process()
        pi.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        pi.arguments = ["-c", "import time; time.sleep(30)"]
        pi.environment = ProcessInfo.processInfo.environment.merging([
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ]) { _, new in new }
        try pi.run()
        guard pi.isRunning else {
            XCTFail("could not launch the token-owned Pi handoff fixture")
            return
        }

        let unrelated = Process()
        unrelated.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        unrelated.arguments = ["-c", "import time; time.sleep(30)"]
        unrelated.environment = ProcessInfo.processInfo.environment
        try unrelated.run()
        guard unrelated.isRunning else {
            XCTFail("could not launch the unrelated handoff fixture")
            return
        }

        let lookalike = Process()
        lookalike.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        lookalike.arguments = ["-c", "import time; time.sleep(30)"]
        lookalike.environment = ProcessInfo.processInfo.environment.merging([
            "MTPLX_HANDOFF_NOTE": "\(MTPLXTerminalHandoffLease.environmentVariable)=\(handoffID.uuidString.lowercased())"
        ]) { _, new in new }
        try lookalike.run()
        guard lookalike.isRunning else {
            XCTFail("could not launch the lookalike handoff fixture")
            return
        }
        defer {
            for process in [pi, unrelated, lookalike] where process.isRunning {
                process.terminate()
                process.waitUntilExit()
            }
        }

        let lease = MTPLXTerminalHandoffLease(
            handoffID: handoffID,
            processID: Int(pi.processIdentifier),
            cancellationMarkerURL: handoffDirectory.appendingPathComponent("cancelled")
        )
        XCTAssertTrue(MTPLXTerminalHandoffLease.process(
            pid: pi.processIdentifier,
            hasExactHandoffID: handoffID
        ))
        XCTAssertFalse(MTPLXTerminalHandoffLease.process(
            pid: lookalike.processIdentifier,
            hasExactHandoffID: handoffID
        ))
        store.recordLaunchedPiTerminalHandoffLease(lease)
        XCTAssertEqual(store.piTerminalAgentProcessIDs, [Int(pi.processIdentifier)])
        XCTAssertEqual(store.terminalHandoffLeaseIDsForTesting, [handoffID])
        store.startMetricsStream()
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        // The observer's initial `.stopped` snapshot represents epoch zero;
        // it must not clear a stream started before the callback is delivered.
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 0,
                state: .stopped,
                restartStatus: .idle,
                restartCount: 0,
                recoveryGeneration: 0
            )
        )
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        // Delivery order is not guaranteed once the supervisor releases its
        // lock. Apply the terminal snapshot before its older `.running`
        // snapshot and ensure the epoch still reaps this lifecycle exactly
        // once.
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 2,
                state: .stopped,
                restartStatus: .idle,
                restartCount: 0,
                lifecycleEpoch: 1,
                recoveryGeneration: 0
            )
        )
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 1,
                state: .running,
                restartStatus: .idle,
                restartCount: 0,
                lifecycleEpoch: 1,
                recoveryGeneration: 0
            )
        )
        await store.awaitDaemonTeardown()

        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertEqual(store.startupPhase, .idle)
        XCTAssertEqual(store.connectionState, .idle)
        XCTAssertFalse(store.hasActiveDaemonTransportForTesting)
        let ownedPID = pi.processIdentifier
        var didExit = kill(ownedPID, 0) != 0
        for _ in 0..<100 where !didExit {
            try? await Task.sleep(nanoseconds: 10_000_000)
            didExit = kill(ownedPID, 0) != 0
        }
        XCTAssertTrue(
            didExit,
            "passive cleanup must reap the exact token-owned Pi handoff"
        )
        XCTAssertEqual(kill(unrelated.processIdentifier, 0), 0, "passive cleanup reaped an unrelated PID")
        XCTAssertEqual(
            kill(lookalike.processIdentifier, 0),
            0,
            "a token-shaped value in another environment variable must not be reaped"
        )
        XCTAssertEqual(store.piTerminalAgentProcessIDs, [])
        XCTAssertTrue(store.terminalHandoffLeaseIDsForTesting.isEmpty)
        XCTAssertNil(store.health)
        XCTAssertNil(store.latest)
    }

    @MainActor
    func testCrashDuringPostStartHandoffCannotResurrectMetrics() async throws {
        let supervisor = DaemonSupervisor()
        let store = MTPLXBackendStore(
            supervisor: supervisor,
            beforePostStartRefresh: {
                await supervisor.stop(graceSeconds: 0)
            }
        )
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let running = supervisor.supervisionSnapshot()
        XCTAssertEqual(running.state, .running)

        await store.finishReadyDaemon(
            target: nil,
            configuration: store.configuration,
            replaceExistingClient: false,
            launchID: nil,
            lifecycleEpoch: running.lifecycleEpoch
        )
        store.applySupervisorSnapshot(supervisor.supervisionSnapshot())
        await store.awaitDaemonTeardown()

        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertFalse(store.hasActiveDaemonTransportForTesting)
        XCTAssertNil(store.health)
        XCTAssertNil(store.latest)
    }

    @MainActor
    func testStaleFanResultCannotOverwriteNewerLifecycleSettings() async throws {
        let supervisor = DaemonSupervisor()
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let firstLifecycle = supervisor.supervisionSnapshot()
        let fanGate = BeforeRunGate()
        await fanGate.arm()
        let settingsURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-daemon-fan-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: settingsURL) }
        var initialConfiguration = MTPLXAppConfiguration()
        initialConfiguration.fanMode = MTPLXFanMode.max.rawValue
        initialConfiguration.pinFansAtMaxOnStart = true
        let store = MTPLXBackendStore(
            configuration: initialConfiguration,
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL),
            supervisor: supervisor,
            fanModeSetter: { _, mode, _, _ in
                await fanGate.waitIfArmed()
                return FanModeResponse(verified: true, currentMode: mode)
            }
        )
        // Normal `.running` snapshots intentionally leave explicit-start
        // presentation state alone. This synthetic recovery snapshot puts
        // the store in the same live-fan-control state without starting a
        // separate recovery task (generation zero is already observed).
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: firstLifecycle.revision,
                state: .running,
                restartStatus: .runningAfterRestart(attempt: 1),
                restartCount: 0,
                lifecycleEpoch: firstLifecycle.lifecycleEpoch,
                recoveryGeneration: 0
            )
        )

        let staleFanApply = Task { @MainActor in
            try await store.setFanMode(MTPLXFanMode.max.rawValue, isCurrent: {
                let snapshot = supervisor.supervisionSnapshot()
                return snapshot.lifecycleEpoch == firstLifecycle.lifecycleEpoch
                    && snapshot.state == .running
            })
        }
        await fanGate.waitUntilEntered()

        await supervisor.stop(graceSeconds: 0)
        var newerSettings = store.configuration
        newerSettings.fanMode = MTPLXFanMode.default.rawValue
        newerSettings.pinFansAtMaxOnStart = false
        try store.saveSettings(newerSettings)
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let secondLifecycle = supervisor.supervisionSnapshot()
        XCTAssertGreaterThan(secondLifecycle.lifecycleEpoch, firstLifecycle.lifecycleEpoch)
        store.applySupervisorSnapshot(secondLifecycle)

        await fanGate.release()
        try await staleFanApply.value

        XCTAssertEqual(store.configuration.fanMode, MTPLXFanMode.default.rawValue)
        XCTAssertFalse(store.configuration.pinFansAtMaxOnStart)
        await supervisor.stop(graceSeconds: 0)
    }

    @MainActor
    func testStalePiHandoffAfterNewLifecycleCannotLaunchClient() async throws {
        let supervisor = DaemonSupervisor()
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let firstLifecycle = supervisor.supervisionSnapshot()
        let handoffGate = BeforeRunGate()
        await handoffGate.arm()
        let store = MTPLXBackendStore(
            supervisor: supervisor,
            beforeClientHandoffLaunch: { target in
                if target == .pi { await handoffGate.waitIfArmed() }
            }
        )
        var didCallReady = false
        store.onDaemonReady = { _ in didCallReady = true }
        let staleHandoff = Task { @MainActor in
            await store.finishReadyDaemon(
                target: .pi,
                configuration: store.configuration,
                replaceExistingClient: false,
                launchID: nil,
                lifecycleEpoch: firstLifecycle.lifecycleEpoch
            )
        }
        await handoffGate.waitUntilEntered()

        await supervisor.stop(graceSeconds: 0)
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let secondLifecycle = supervisor.supervisionSnapshot()
        XCTAssertGreaterThan(secondLifecycle.lifecycleEpoch, firstLifecycle.lifecycleEpoch)
        store.applySupervisorSnapshot(secondLifecycle)

        await handoffGate.release()
        await staleHandoff.value

        XCTAssertEqual(supervisor.supervisionSnapshot().state, .running)
        XCTAssertFalse(didCallReady)
        XCTAssertFalse(store.piTerminalAgentRunning)
        XCTAssertEqual(store.piTerminalAgentProcessIDs, [])
        XCTAssertNil(store.clientHandoffNotice)
        await supervisor.stop(graceSeconds: 0)
    }

    @MainActor
    func testStaleHermesHandoffAfterStopCannotLaunchClient() async throws {
        let supervisor = DaemonSupervisor()
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let lifecycle = supervisor.supervisionSnapshot()
        let handoffGate = BeforeRunGate()
        await handoffGate.arm()
        let store = MTPLXBackendStore(
            supervisor: supervisor,
            beforeClientHandoffLaunch: { target in
                if target == .hermes { await handoffGate.waitIfArmed() }
            }
        )
        var didCallReady = false
        store.onDaemonReady = { _ in didCallReady = true }
        let staleHandoff = Task { @MainActor in
            await store.finishReadyDaemon(
                target: .hermes,
                configuration: store.configuration,
                replaceExistingClient: false,
                launchID: nil,
                lifecycleEpoch: lifecycle.lifecycleEpoch
            )
        }
        await handoffGate.waitUntilEntered()

        await supervisor.stop(graceSeconds: 0)
        await handoffGate.release()
        await staleHandoff.value

        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        XCTAssertFalse(didCallReady)
        XCTAssertNil(store.clientHandoffNotice)
    }

    @MainActor
    func testOlderTerminalLifecycleCannotClearNewerRunningSession() async throws {
        let supervisor = DaemonSupervisor()
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let olderLifecycle = supervisor.supervisionSnapshot().lifecycleEpoch
        await supervisor.stop(graceSeconds: 0)
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let current = supervisor.supervisionSnapshot()
        XCTAssertGreaterThan(current.lifecycleEpoch, olderLifecycle)
        XCTAssertEqual(current.state, .running)

        let store = MTPLXBackendStore(supervisor: supervisor)
        let pi = Process()
        pi.executableURL = URL(fileURLWithPath: "/bin/zsh")
        pi.arguments = ["-c", "exec -a pi /bin/sleep 30"]
        try pi.run()
        defer {
            if pi.isRunning { pi.terminate() }
        }
        store.recordLaunchedPiAgentProcessIDs([Int(pi.processIdentifier)])
        store.startMetricsStream()
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 1,
                state: .running,
                restartStatus: .idle,
                restartCount: 0,
                restartEligibility: .currentSessionUnprotected,
                lifecycleEpoch: current.lifecycleEpoch,
                recoveryGeneration: current.recoveryGeneration
            )
        )
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        // Simulate delayed delivery of an older terminal callback after the
        // next lifecycle is already genuinely running in the supervisor.
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 2,
                state: .stopped,
                restartStatus: .idle,
                restartCount: 0,
                lifecycleEpoch: olderLifecycle,
                recoveryGeneration: 0
            )
        )
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)
        XCTAssertEqual(store.piTerminalAgentProcessIDs, [Int(pi.processIdentifier)])
        XCTAssertTrue(pi.isRunning)
        XCTAssertEqual(store.daemonRestartEligibility, .currentSessionUnprotected)

        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 3,
                state: .running,
                restartStatus: .idle,
                restartCount: 0,
                restartEligibility: .currentSessionUnprotected,
                lifecycleEpoch: current.lifecycleEpoch,
                recoveryGeneration: current.recoveryGeneration
            )
        )
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)
        await store.stopDaemon()
    }

    @MainActor
    func testStaleThermalCompletionCannotCancelNewerLifecycleMetricsStream() async throws {
        let supervisor = DaemonSupervisor()
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let firstLifecycle = supervisor.supervisionSnapshot()
        let thermalGate = BeforeRunGate()
        await thermalGate.arm()
        let store = MTPLXBackendStore(
            supervisor: supervisor,
            beforeThermalStatusRefresh: { await thermalGate.waitIfArmed() }
        )
        let staleRefresh = Task { @MainActor in
            await store.startMetricsAndRefreshThermal(
                lifecycleEpoch: firstLifecycle.lifecycleEpoch,
                recoveryGeneration: nil
            )
        }
        await thermalGate.waitUntilEntered()

        await supervisor.stop(graceSeconds: 0)
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let secondLifecycle = supervisor.supervisionSnapshot()
        XCTAssertGreaterThan(secondLifecycle.lifecycleEpoch, firstLifecycle.lifecycleEpoch)
        store.startMetricsStream()
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        await thermalGate.release()
        let staleRefreshResult = await staleRefresh.value
        XCTAssertFalse(staleRefreshResult)
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)
        await store.stopDaemon()
    }

    @MainActor
    func testStorePublishesAdoptedDaemonAsUnprotected() async {
        let store = MTPLXBackendStore()
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 1,
                state: .running,
                restartStatus: .idle,
                restartCount: 0,
                restartEligibility: .adoptedPriorSession,
                recoveryGeneration: 0
            )
        )

        XCTAssertEqual(store.daemonRestartEligibility, .adoptedPriorSession)
    }

    @MainActor
    func testDisabledAbnormalExitCleansTransportButPreservesCrashState() async {
        let store = MTPLXBackendStore()
        store.startMetricsStream()
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 1,
                state: .crashed(9),
                restartStatus: .idle,
                restartCount: 0,
                lifecycleEpoch: 1,
                recoveryGeneration: 0
            )
        )
        await store.awaitDaemonTeardown()

        XCTAssertEqual(store.daemonState, .crashed(9))
        XCTAssertFalse(store.hasActiveDaemonTransportForTesting)
        XCTAssertNil(store.health)
        XCTAssertNil(store.latest)
        if case .failed = store.startupPhase {} else {
            XCTFail("terminal crash should preserve a failed startup phase")
        }
        if case .failed = store.connectionState {} else {
            XCTFail("terminal crash should preserve a failed connection state")
        }
    }

    @MainActor
    func testPassiveTerminalCleanupRestoresConfiguredMaxFans() async throws {
        let supervisor = DaemonSupervisor()
        var configuration = MTPLXAppConfiguration()
        configuration.fanMode = MTPLXFanMode.max.rawValue
        configuration.pinFansAtMaxOnStart = true
        let recorder = FanRestoreRecorder()
        let store = MTPLXBackendStore(
            configuration: configuration,
            supervisor: supervisor,
            localFanRestorer: { await recorder.restore() }
        )
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let running = supervisor.supervisionSnapshot()
        await supervisor.stop(graceSeconds: 0)
        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: running.revision + 1,
                state: .stopped,
                restartStatus: .idle,
                restartCount: 0,
                lifecycleEpoch: running.lifecycleEpoch,
                recoveryGeneration: running.recoveryGeneration
            )
        )
        await store.awaitDaemonTeardown()

        let restoreCount = await recorder.count()
        XCTAssertEqual(restoreCount, 1)
        XCTAssertEqual(store.currentFanMode, MTPLXFanMode.default.rawValue)
    }

    @MainActor
    func testPassiveFanRestoreDoesNotOverwriteNewerConfigurationState() async throws {
        let supervisor = DaemonSupervisor()
        let restoreGate = BeforeRunGate()
        await restoreGate.arm()
        let settingsURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-passive-fan-race-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: settingsURL) }
        var configuration = MTPLXAppConfiguration()
        configuration.fanMode = MTPLXFanMode.max.rawValue
        configuration.pinFansAtMaxOnStart = true
        let store = MTPLXBackendStore(
            configuration: configuration,
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL),
            supervisor: supervisor,
            localFanRestorer: {
                await restoreGate.waitIfArmed()
                return true
            }
        )
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )
        _ = try await supervisor.start(
            command: command,
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        await supervisor.stop(graceSeconds: 0)
        await restoreGate.waitUntilEntered()

        var newerConfiguration = store.configuration
        newerConfiguration.fanMode = MTPLXFanMode.default.rawValue
        newerConfiguration.pinFansAtMaxOnStart = false
        try store.saveSettings(newerConfiguration)
        await restoreGate.release()
        await store.awaitDaemonTeardown()

        XCTAssertEqual(store.configuration.fanMode, MTPLXFanMode.default.rawValue)
        XCTAssertFalse(store.configuration.pinFansAtMaxOnStart)
    }

    @MainActor
    func testExhaustedAutomaticRecoveryCleansTransportButPreservesCircuitBreakerState() async {
        let store = MTPLXBackendStore()
        store.startMetricsStream()
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting)

        store.applySupervisorSnapshot(
            DaemonSupervisionSnapshot(
                revision: 1,
                state: .crashed(17),
                restartStatus: .exhausted(attempts: 3, lastExitStatus: 17),
                restartCount: 3,
                lifecycleEpoch: 2,
                recoveryGeneration: 0
            )
        )
        await store.awaitDaemonTeardown()

        XCTAssertEqual(store.daemonState, .crashed(17))
        XCTAssertFalse(store.hasActiveDaemonTransportForTesting)
        XCTAssertNil(store.health)
        XCTAssertNil(store.latest)
        XCTAssertEqual(
            store.startupPhase,
            .failed("MTPLX crashed repeatedly; automatic recovery stopped after 3 attempts.")
        )
        XCTAssertEqual(
            store.connectionState,
            .failed("Automatic restart circuit breaker is open.")
        )
    }

    func testAbnormalExitRetriesWithCircuitBreakerAndEvidence() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let logs = BoundedLogStore()
        let supervisor = DaemonSupervisor(
            logStore: logs,
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let exhausted = await signal.waitForExhausted()
        XCTAssertTrue(exhausted)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.restartCount, 2)
        XCTAssertEqual(snapshot.restartStatus, .exhausted(attempts: 2, lastExitStatus: 17))
        let evidence = await waitForEvidence(
            logs,
            containing: [
                "automatic restart scheduled: attempt 1 of 2",
                "automatic restart attempt 2 of 2 starting",
                "automatic restart circuit breaker open after 2 attempts",
            ]
        )
        XCTAssertTrue(evidence.contains("automatic restart scheduled: attempt 1 of 2"))
        XCTAssertTrue(evidence.contains("automatic restart attempt 2 of 2 starting"))
        XCTAssertTrue(evidence.contains("automatic restart circuit breaker open after 2 attempts"))
    }

    func testRestartBackoffDoublesAndCaps() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let delays = RestartDelayRecorder()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 3,
                initialDelaySeconds: 1,
                maximumDelaySeconds: 2,
                crashWindowSeconds: 60
            ),
            restartSleeper: { delay in await delays.record(delay) }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let exhausted = await signal.waitForExhausted()
        XCTAssertTrue(exhausted)
        let recordedDelays = await delays.snapshot()
        XCTAssertEqual(recordedDelays, [1, 2, 2])
    }

    func testStopWaitsForPublishedLaunchToReceiveAPID() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let pidFile = directory.appendingPathComponent("daemon.pid")
        let gate = BeforeRunGate()
        let signal = SupervisionSignal()
        await gate.arm()
        let supervisor = DaemonSupervisor(beforeProcessRun: { await gate.waitIfArmed() })
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo $$ > \(shellQuoted(pidFile.path)); while :; do sleep 1; done"]
        )
        let startTask = Task {
            try await supervisor.start(
                command: command,
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: false
            )
        }
        await gate.waitUntilEntered()
        let stopTask = Task { await supervisor.stop(graceSeconds: 0) }
        let stopping = await signal.waitForStopping()
        XCTAssertTrue(stopping)
        await gate.release()
        _ = try? await startTask.value
        await stopTask.value

        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        XCTAssertFalse(supervisor.isRunning())
        if let pidText = try? String(contentsOf: pidFile), let pid = Int32(pidText.trimmingCharacters(in: .whitespacesAndNewlines)) {
            XCTAssertNotEqual(kill(pid, 0), 0, "Stop returned while the concurrently launched daemon was still alive")
        }
    }

    func testStopBeforeProcessReservationPreventsAnyLaterLaunch() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let pidFile = directory.appendingPathComponent("daemon.pid")
        let gate = BeforeRunGate()
        await gate.arm()
        let supervisor = DaemonSupervisor(
            beforeProcessReservation: { await gate.waitIfArmed() }
        )
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "echo $$ > \(shellQuoted(pidFile.path)); while :; do sleep 1; done"]
        )
        let startTask = Task {
            try await supervisor.start(
                command: command,
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: false
            )
        }
        await gate.waitUntilEntered()
        await supervisor.stop(graceSeconds: 0)
        await gate.release()
        _ = try? await startTask.value

        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        XCTAssertFalse(supervisor.isRunning())
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: pidFile.path),
            "Start published a process after Stop had completed"
        )
    }

    func testDisablingAutomaticRestartDoesNotCancelManualLaunch() async throws {
        let gate = BeforeRunGate()
        await gate.arm()
        let supervisor = DaemonSupervisor(
            beforeProcessRun: { await gate.waitIfArmed() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        let command = DaemonCommand(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "while :; do sleep 1; done"]
        )
        let startTask = Task {
            try await supervisor.start(
                command: command,
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: false
            )
        }
        await gate.waitUntilEntered()

        supervisor.setAutomaticRestartEnabled(false)
        await gate.release()
        _ = try await startTask.value

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .running)
        XCTAssertEqual(snapshot.restartEligibility, .currentSessionUnprotected)
        XCTAssertFalse(supervisor.hasRetainedRestartRecipeForTesting)
        await supervisor.stop(graceSeconds: 0)
    }

    func testDisablingAutomaticRetryReapsParentAndChildAndEndsStopped() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-auto-disable-family-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let release = directory.appendingPathComponent("release")
        let counter = directory.appendingPathComponent("counter")
        let parentPID = directory.appendingPathComponent("parent.pid")
        let childPID = directory.appendingPathComponent("child.pid")
        try Data().write(to: release)

        let signal = SupervisionSignal()
        let healthGate = AutomaticRetryHealthGate(readyHealth: try adoptedHealth())
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            initialHealthProbe: { _, _ in nil },
            healthWaitProbe: { _, _ in await healthGate.probe() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: automaticRetryParentChildCommand(
                releaseFile: release,
                counterFile: counter,
                parentPIDFile: parentPID,
                childPIDFile: childPID
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: true,
            timeoutSeconds: 10
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }

        try FileManager.default.removeItem(at: release)
        await healthGate.waitUntilAutomaticRetryProbe()
        let parent = await waitForPID(in: parentPID)
        let child = await waitForPID(in: childPID)
        XCTAssertNotNil(parent)
        XCTAssertNotNil(child)

        supervisor.setAutomaticRestartEnabled(false)
        await healthGate.release(try adoptedHealth())
        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
        XCTAssertFalse(supervisor.hasOutstandingRestartTaskForTesting)
        if let parent {
            let parentExited = await waitForExit(parent)
            XCTAssertTrue(parentExited, "automatic disable leaked parent")
        }
        if let child {
            let childExited = await waitForExit(child)
            XCTAssertTrue(childExited, "automatic disable leaked child")
        }
    }

    func testStaleAutomaticHealthFailureCannotStopNewManualDaemon() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-stale-auto-stop-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let release = directory.appendingPathComponent("release")
        let counter = directory.appendingPathComponent("counter")
        let manualPID = directory.appendingPathComponent("manual.pid")
        try Data().write(to: release)
        let automaticScript = """
        count=$(cat \(shellQuoted(counter.path)) 2>/dev/null || echo 0)
        count=$((count + 1))
        echo "$count" > \(shellQuoted(counter.path))
        if [ "$count" -eq 1 ]; then
          while [ -e \(shellQuoted(release.path)) ]; do sleep 0.01; done
          exit 17
        fi
        while :; do sleep 1; done
        """
        let healthGate = AutomaticRetryHealthGate(readyHealth: try adoptedHealth())
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            initialHealthProbe: { _, _ in nil },
            healthWaitProbe: { _, _ in await healthGate.probe() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", automaticScript]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: true,
            timeoutSeconds: 10
        )

        try FileManager.default.removeItem(at: release)
        await healthGate.waitUntilAutomaticRetryProbe()
        await supervisor.stop(graceSeconds: 0)

        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: [
                    "-c",
                    "echo $$ > \(shellQuoted(manualPID.path)); while :; do sleep 1; done"
                ]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let manual = await waitForPID(in: manualPID)
        XCTAssertNotNil(manual)

        await healthGate.release(try adoptedHealth())
        for _ in 0..<100 where supervisor.hasOutstandingRestartTaskForTesting {
            await Task.yield()
        }

        XCTAssertEqual(supervisor.supervisionSnapshot().state, .running)
        if let manual {
            XCTAssertEqual(kill(manual, 0), 0, "stale automatic cleanup killed manual Start B")
        }
        await supervisor.stop(graceSeconds: 0)
    }

    func testDisablingInAutomaticRestartPublishGapSettlesStopped() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let gate = BeforeRunGate()
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            beforeAutomaticRestartStart: { await gate.waitIfArmed() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        await gate.arm()
        try FileManager.default.removeItem(at: release)
        await gate.waitUntilEntered()

        supervisor.setAutomaticRestartEnabled(false)
        await gate.release()
        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)
        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        XCTAssertEqual(supervisor.supervisionSnapshot().restartStatus, .idle)
        XCTAssertFalse(supervisor.hasOutstandingRestartTaskForTesting)
    }

    func testStopReapsOnlyExactTokenChildWhenRootExitsBeforeFamilyResolution() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-stop-family-gap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let parentPID = directory.appendingPathComponent("parent.pid")
        let childPID = directory.appendingPathComponent("child.pid")
        let launchID = "stop-family-\(UUID().uuidString)"
        let unrelatedLaunchID = "unrelated-\(UUID().uuidString)"
        let argumentImposterPID = directory.appendingPathComponent("argument-imposter.pid")
        let valueImposterPID = directory.appendingPathComponent("value-imposter.pid")
        let script = """
        /usr/bin/python3 -c 'import time; time.sleep(300)' &
        child_pid=$!
        echo "$$" > \(shellQuoted(parentPID.path))
        echo "$child_pid" > \(shellQuoted(childPID.path))
        trap 'exit 0' TERM INT
        while :; do sleep 1; done
        """
        let unrelated = Process()
        unrelated.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        unrelated.arguments = ["-c", "import time; time.sleep(300)"]
        unrelated.environment = ProcessInfo.processInfo.environment.merging(
            ["MTPLX_APP_LAUNCH_ID": unrelatedLaunchID]
        ) { _, new in new }
        try unrelated.run()
        let unrelatedPID = unrelated.processIdentifier
        defer {
            if unrelated.isRunning {
                unrelated.terminate()
            }
        }
        let argumentImposter = Process()
        argumentImposter.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        argumentImposter.arguments = [
            "-c",
            "import os, time; open(os.environ['PID_FILE'], 'w').write(str(os.getpid())); time.sleep(300)",
            "MTPLX_APP_LAUNCH_ID=\(launchID)"
        ]
        argumentImposter.environment = ProcessInfo.processInfo.environment.merging(
            ["PID_FILE": argumentImposterPID.path]
        ) { _, new in new }
        try argumentImposter.run()
        defer {
            if argumentImposter.isRunning {
                argumentImposter.terminate()
            }
        }
        let valueImposter = Process()
        valueImposter.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        valueImposter.arguments = [
            "-c",
            "import os, time; open(os.environ['PID_FILE'], 'w').write(str(os.getpid())); time.sleep(300)"
        ]
        valueImposter.environment = ProcessInfo.processInfo.environment.merging([
            "PID_FILE": valueImposterPID.path,
            "NOT_A_LAUNCH_ID": "MTPLX_APP_LAUNCH_ID=\(launchID)"
        ]) { _, new in new }
        try valueImposter.run()
        defer {
            if valueImposter.isRunning {
                valueImposter.terminate()
            }
        }
        let supervisor = DaemonSupervisor(
            beforeStopProcessFamilyResolution: {
                if let text = try? String(contentsOf: parentPID),
                   let pid = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                    kill(pid, SIGTERM)
                }
            }
        )
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", script],
                environment: ["MTPLX_APP_LAUNCH_ID": launchID]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let parent = await waitForPID(in: parentPID)
        let child = await waitForPID(in: childPID)
        let argumentImposterPIDValue = await waitForPID(in: argumentImposterPID)
        let valueImposterPIDValue = await waitForPID(in: valueImposterPID)
        XCTAssertNotNil(parent)
        XCTAssertNotNil(child)
        XCTAssertNotNil(argumentImposterPIDValue)
        XCTAssertNotNil(valueImposterPIDValue)

        await supervisor.stop(graceSeconds: 0)
        XCTAssertEqual(supervisor.supervisionSnapshot().state, .stopped)
        if let parent {
            let parentExited = await waitForExit(parent)
            XCTAssertTrue(parentExited)
        }
        if let child {
            let childExited = await waitForExit(child)
            XCTAssertTrue(childExited, "Stop lost token-owned child after root termination")
        }
        XCTAssertEqual(kill(unrelatedPID, 0), 0, "Stop matched an unrelated launch token")
        if let argumentImposterPIDValue {
            XCTAssertEqual(
                kill(argumentImposterPIDValue, 0),
                0,
                "Stop matched a launch token passed only as an argument"
            )
        }
        if let valueImposterPIDValue {
            XCTAssertEqual(
                kill(valueImposterPIDValue, 0),
                0,
                "Stop matched a launch token embedded in another environment value"
            )
        }
    }

    func testCompletedRestartTaskReleasesItsRecipeAfterCleanExit() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-restart-recipe-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let firstRelease = directory.appendingPathComponent("first-release")
        let secondRelease = directory.appendingPathComponent("second-release")
        let counter = directory.appendingPathComponent("counter")
        try Data().write(to: firstRelease)
        try Data().write(to: secondRelease)
        let script = """
        count=$(cat \(shellQuoted(counter.path)) 2>/dev/null || echo 0)
        count=$((count + 1))
        echo "$count" > \(shellQuoted(counter.path))
        if [ "$count" -eq 1 ]; then
          while [ -e \(shellQuoted(firstRelease.path)) ]; do sleep 0.01; done
          exit 17
        fi
        while [ -e \(shellQuoted(secondRelease.path)) ]; do sleep 0.01; done
        exit 0
        """
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 1,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", script]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            apiKey: "test-api-key",
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }

        try FileManager.default.removeItem(at: firstRelease)
        let recovered = await signal.waitForRecoveryGeneration(1)
        XCTAssertTrue(recovered)
        for _ in 0..<100 where supervisor.hasOutstandingRestartTaskForTesting {
            await Task.yield()
        }
        XCTAssertFalse(supervisor.hasOutstandingRestartTaskForTesting)
        XCTAssertTrue(supervisor.hasRetainedRestartRecipeForTesting)

        try FileManager.default.removeItem(at: secondRelease)
        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)
        for _ in 0..<100 where supervisor.hasOutstandingRestartTaskForTesting {
            await Task.yield()
        }
        XCTAssertFalse(supervisor.hasOutstandingRestartTaskForTesting)
        XCTAssertFalse(supervisor.hasRetainedRestartRecipeForTesting)
    }

    func testAutomaticRetryCleanExitBeforeTerminationHandlerDoesNotReschedule() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-auto-clean-exit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let release = directory.appendingPathComponent("release")
        let counter = directory.appendingPathComponent("counter")
        try Data().write(to: release)
        let terminationGate = TerminationGate()
        let restartGate = RestartGate()
        let postRunGate = BeforeRunGate()
        let script = """
        count=$(cat \(shellQuoted(counter.path)) 2>/dev/null || echo 0)
        count=$((count + 1))
        echo "$count" > \(shellQuoted(counter.path))
        if [ "$count" -eq 1 ]; then
          while [ -e \(shellQuoted(release.path)) ]; do sleep 0.01; done
          exit 17
        fi
        exit 0
        """
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 1,
                maximumDelaySeconds: 1,
                crashWindowSeconds: 60
            ),
            restartSleeper: { _ in await restartGate.wait() },
            beforePostRunLivenessCheck: { await postRunGate.waitIfArmed() },
            beforeTerminationHandling: { _ in
                terminationGate.blockIfArmed()
            }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", script]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }

        try FileManager.default.removeItem(at: release)
        let scheduled = await signal.waitForScheduled()
        XCTAssertTrue(scheduled)
        await postRunGate.arm()
        terminationGate.arm()
        var terminationGateUnblocked = false
        defer {
            if !terminationGateUnblocked {
                terminationGate.unblock()
            }
        }
        await restartGate.release()
        await postRunGate.waitUntilEntered()
        let handlerEntered = await Task.detached { terminationGate.waitUntilEntered() }.value
        XCTAssertTrue(handlerEntered)
        await postRunGate.release()

        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)
        // The supervisor must settle stopped before its delayed handler can
        // re-enter. Unblock only after that assertion point so the task can
        // finish its ordinary deferred cleanup.
        terminationGate.unblock()
        terminationGateUnblocked = true
        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertFalse(supervisor.hasRetainedRestartRecipeForTesting)
        for _ in 0..<100 where supervisor.hasOutstandingRestartTaskForTesting {
            await Task.yield()
        }
        XCTAssertFalse(supervisor.hasOutstandingRestartTaskForTesting)
    }

    func testDisablingDuringRestartingAbortsTheInFlightAutomaticLaunch() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let gate = BeforeRunGate()
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            beforeProcessRun: { await gate.waitIfArmed() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        await gate.arm()
        try FileManager.default.removeItem(at: release)
        await gate.waitUntilEntered()
        supervisor.setAutomaticRestartEnabled(false)
        await gate.release()
        let stopped = await signal.waitForStoppedAfterLaunch()
        XCTAssertTrue(stopped)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
        XCTAssertEqual(snapshot.recoveryGeneration, 0)
        XCTAssertFalse(supervisor.isRunning())
    }

    func testStopDuringRestartingWaitsForAndCancelsTheAutomaticLaunch() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let gate = BeforeRunGate()
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            beforeProcessRun: { await gate.waitIfArmed() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        await gate.arm()
        try FileManager.default.removeItem(at: release)
        await gate.waitUntilEntered()
        let stopTask = Task { await supervisor.stop(graceSeconds: 0) }
        let stopping = await signal.waitForStopping()
        XCTAssertTrue(stopping)
        await gate.release()
        await stopTask.value

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
        XCTAssertFalse(supervisor.isRunning())
    }

    func testExplicitStopCancelsScheduledRestartBeforeSleeperReleases() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let gate = RestartGate()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 2,
                initialDelaySeconds: 1,
                maximumDelaySeconds: 1,
                crashWindowSeconds: 60
            ),
            restartSleeper: { _ in await gate.wait() }
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let scheduled = await signal.waitForScheduled()
        XCTAssertTrue(scheduled)
        await supervisor.stop()
        await gate.release()
        for _ in 0..<8 { await Task.yield() }

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.state, .stopped)
        XCTAssertEqual(snapshot.restartStatus, .idle)
        XCTAssertEqual(snapshot.restartCount, 0)
    }

    func testReenablingWithoutAFreshLaunchDoesNotRetainThePreviousLaunchSecret() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 1,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        supervisor.setAutomaticRestartEnabled(false)
        supervisor.setAutomaticRestartEnabled(true)
        try FileManager.default.removeItem(at: release)

        let crashed = await signal.waitForCrash()
        XCTAssertTrue(crashed)
        XCTAssertEqual(supervisor.supervisionSnapshot().restartCount, 0)
        XCTAssertEqual(supervisor.supervisionSnapshot().restartStatus, .idle)
    }

    func testFreshStartAfterReenablingRestoresAutomaticRestartEligibility() async throws {
        let release = try temporaryReleaseFile()
        defer { try? FileManager.default.removeItem(at: release.deletingLastPathComponent()) }
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 1,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        supervisor.setAutomaticRestartEnabled(false)
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: releaseControlledCommand(releaseFile: release, exitStatus: 17),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }
        try FileManager.default.removeItem(at: release)
        let exhausted = await signal.waitForExhausted()
        XCTAssertTrue(exhausted)
        XCTAssertEqual(supervisor.supervisionSnapshot().restartCount, 1)
    }

    func testRecoveryGenerationRemainsMonotonicWhenCrashWindowResetsAttempts() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let counter = directory.appendingPathComponent("counter")
        let firstCrashGate = directory.appendingPathComponent("first-crash-gate")
        let secondCrashGate = directory.appendingPathComponent("second-crash-gate")
        let keepThirdRunAlive = directory.appendingPathComponent("keep-third-run-alive")
        try Data().write(to: firstCrashGate)
        try Data().write(to: secondCrashGate)
        try Data().write(to: keepThirdRunAlive)
        let script = """
        n=$(cat \(shellQuoted(counter.path)) 2>/dev/null || echo 0)
        n=$((n + 1))
        echo "$n" > \(shellQuoted(counter.path))
        if [ "$n" -eq 1 ]; then
          while [ -e \(shellQuoted(firstCrashGate.path)) ]; do sleep 0.01; done
          exit 17
        fi
        if [ "$n" -eq 2 ]; then
          while [ -e \(shellQuoted(secondCrashGate.path)) ]; do sleep 0.01; done
          exit 17
        fi
        while [ -e \(shellQuoted(keepThirdRunAlive.path)) ]; do sleep 0.01; done
        """
        let signal = SupervisionSignal()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 1,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 0
            )
        )
        supervisor.setAutomaticRestartEnabled(true)
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", script]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        supervisor.setStatusObserver { snapshot in
            Task { await signal.record(snapshot) }
        }

        try FileManager.default.removeItem(at: firstCrashGate)
        let firstRecovery = await signal.waitForRecoveryGeneration(1)
        XCTAssertTrue(firstRecovery)
        try FileManager.default.removeItem(at: secondCrashGate)
        let secondRecovery = await signal.waitForRecoveryGeneration(2)
        XCTAssertTrue(secondRecovery)

        let snapshot = supervisor.supervisionSnapshot()
        XCTAssertEqual(snapshot.restartCount, 1)
        XCTAssertEqual(snapshot.recoveryGeneration, 2)
        await supervisor.stop()
    }
}

// MARK: - Fan-ramp grace (2026-08-19 release blockers)

extension DaemonSupervisorTests {
    private static func healthyUnverifiedRampPayload() throws -> HealthPayload {
        try JSONDecoder().decode(
            HealthPayload.self,
            from: Data(
                #"""
                {"ok": true, "model": "test-model", "model_path": "/tmp/test-model",
                 "generation_mode": "mtp", "load_mtp": true, "mtp_enabled": true,
                 "depth": 3, "profile": {}, "context_window": 4096,
                 "active_requests": 0, "reasoning_parser": "qwen3",
                 "thermal": {"actual_ramp_verified": false}}
                """#.utf8
            )
        )
    }

    /// The 600 s health budget must never be inherited by the fan-ramp wait:
    /// a daemon that is already answering /health ok proceeds to ready after
    /// the bounded grace window instead of spinning out the whole budget and
    /// being reaped over a fan receipt (the model-swap "Degraded" hang).
    @MainActor
    func testHealthyDaemonProceedsAfterFanRampGraceInsteadOfReap() async throws {
        let payload = try Self.healthyUnverifiedRampPayload()
        let supervisor = DaemonSupervisor(healthWaitProbe: { _, _ in payload })
        supervisor.fanRampGraceSeconds = 0.5
        let started = Date()
        let ready = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: true,
            timeoutSeconds: 30,
            requireActualFanRamp: true
        )
        XCTAssertEqual(ready?.ok, true)
        XCTAssertLessThan(
            Date().timeIntervalSince(started),
            20,
            "ready must arrive at ramp-grace expiry, not at the health deadline"
        )
        XCTAssertTrue(supervisor.isRunning())
        await supervisor.stop(graceSeconds: 0)
    }

    /// A health budget shorter than the ramp grace still classifies as
    /// fanRampTimeout — the timeout taxonomy is unchanged.
    @MainActor
    func testFanRampTimeoutStillThrownWhenBudgetShorterThanGrace() async throws {
        let payload = try Self.healthyUnverifiedRampPayload()
        let supervisor = DaemonSupervisor(healthWaitProbe: { _, _ in payload })
        supervisor.fanRampGraceSeconds = 30
        do {
            _ = try await supervisor.start(
                command: DaemonCommand(
                    executableURL: URL(fileURLWithPath: "/bin/sh"),
                    arguments: ["-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
                ),
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: true,
                timeoutSeconds: 1.0,
                requireActualFanRamp: true
            )
            XCTFail("expected fanRampTimeout")
        } catch let error as DaemonSupervisorError {
            guard case .fanRampTimeout = error else {
                XCTFail("expected fanRampTimeout, got \(error)")
                return
            }
        }
        XCTAssertFalse(supervisor.isRunning())
    }
}
