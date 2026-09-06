import Darwin
import Foundation
import XCTest
@testable import MTPLXAppCore

private actor HandoffRecoveryGate {
    private var entered = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func wait() async {
        entered = true
        let waiters = enteredWaiters
        enteredWaiters.removeAll()
        waiters.forEach { $0.resume() }
        await withCheckedContinuation { releaseContinuation = $0 }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

final class TerminalHandoffLeaseTests: XCTestCase {
    func testExactEnvironmentTokenRejectsLookalikesAndDrainsLargePSOutput() throws {
        let handoffID = UUID()
        let token = handoffID.uuidString.lowercased()
        let process = try launchSleep(
            environment: [
                MTPLXTerminalHandoffLease.environmentVariable: token,
                // A pipe reader that waits for process exit before draining
                // hangs on this output. The lease validator must keep
                // draining while bounded by its watchdog.
                "MTPLX_HANDOFF_TEST_FILLER": String(repeating: "x", count: 65_536)
            ]
        )
        defer { stop(process) }
        XCTAssertTrue(waitUntilRunning(process))
        XCTAssertTrue(MTPLXTerminalHandoffLease.process(
            pid: process.processIdentifier,
            hasExactHandoffID: handoffID
        ))

        let wrongValue = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: token + "-suffix"
        ])
        defer { stop(wrongValue) }
        XCTAssertFalse(MTPLXTerminalHandoffLease.process(
            pid: wrongValue.processIdentifier,
            hasExactHandoffID: handoffID
        ))

        let unrelatedValue = try launchSleep(environment: [
            "MTPLX_HANDOFF_NOTE": "MTPLX_APP_HANDOFF_ID=\(token)"
        ])
        defer { stop(unrelatedValue) }
        XCTAssertFalse(MTPLXTerminalHandoffLease.process(
            pid: unrelatedValue.processIdentifier,
            hasExactHandoffID: handoffID
        ))

        let argvImpostor = try launchShell(
            "exec /usr/bin/python3 -c 'import time; time.sleep(30)' MTPLX_APP_HANDOFF_ID=\(token)",
            handoffID: nil
        )
        defer { stop(argvImpostor) }
        XCTAssertFalse(MTPLXTerminalHandoffLease.process(
            pid: argvImpostor.processIdentifier,
            hasExactHandoffID: handoffID
        ))
        XCTAssertTrue(argvImpostor.isRunning, "same-token argv must not be reaped")
    }

    @MainActor
    func testCancellationMarkerPreventsDelayedTerminalScriptFromExecuting() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appendingPathComponent("cancelled")
        let start = directory.appendingPathComponent("start")
        let executed = directory.appendingPathComponent("executed")
        let handoffID = UUID()
        let script = """
        while [[ ! -e \(shellQuote(start.path)) ]]; do sleep 0.01; done
        if [[ -e \(shellQuote(marker.path)) ]]; then exit 0; fi
        touch \(shellQuote(executed.path))
        """
        let process = try launchShell(script, handoffID: handoffID)
        defer { stop(process) }
        XCTAssertTrue(waitUntilRunning(process))

        MTPLXTerminalHandoffLease.writeCancellationMarker(at: marker)
        try Data().write(to: start)
        let didExit = await waitForExit(process)
        XCTAssertTrue(didExit)
        XCTAssertFalse(FileManager.default.fileExists(atPath: executed.path))
    }

    @MainActor
    func testInjectedMarkerFailureRemovesDurableCommandBeforeDelayedLaunch() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let command = directory.appendingPathComponent("open-pi.command")
        let receipt = directory.appendingPathComponent("open-pi.pid")
        let marker = directory.appendingPathComponent("open-pi.cancelled")
        try MTPLXTerminalHandoffLease.writeSecureCommandScript(
            "#!/bin/zsh\nexit 99\n",
            to: command
        )

        let result = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: UUID(),
            receiptURL: receipt,
            cancellationMarkerURL: marker,
            commandURL: command,
            isCurrent: { false },
            timeoutSeconds: 0,
            delayedCancellationSeconds: 0,
            markerWriter: { _ in false }
        )

        XCTAssertTrue(result.cancellationRequested)
        XCTAssertFalse(result.cancellationMarked)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: command.path),
            "a failed marker write must remove the durable script before Terminal can read it"
        )
    }

    @MainActor
    func testSuccessfulReceiptRemovesSecretBearingCommandAndReceipt() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let handoffID = UUID()
        let command = directory.appendingPathComponent("open-hermes.command")
        let receipt = directory.appendingPathComponent("open-hermes.pid")
        let marker = directory.appendingPathComponent("open-hermes.cancelled")
        let sentinel = "test-openai-key-must-not-persist"
        try MTPLXTerminalHandoffLease.writeSecureCommandScript(
            "#!/bin/zsh\nexport OPENAI_API_KEY='\(sentinel)'\nexit 0\n",
            to: command
        )
        let process = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        defer { stop(process) }
        try Data("\(process.processIdentifier)\n".utf8).write(to: receipt)

        let result = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoffID,
            receiptURL: receipt,
            cancellationMarkerURL: marker,
            commandURL: command,
            isCurrent: { true }
        )

        XCTAssertNotNil(result.lease)
        XCTAssertFalse(result.cancellationRequested)
        XCTAssertFalse(FileManager.default.fileExists(atPath: command.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: receipt.path))
    }

    @MainActor
    func testMarkerAfterFinalCheckCollectsReceiptAndReapsExactLease() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appendingPathComponent("cancelled")
        let receipt = directory.appendingPathComponent("receipt")
        let reachedFinalCheck = directory.appendingPathComponent("final-check")
        let handoffID = UUID()
        let script = """
        print -r -- "$$" > \(shellQuote(receipt.path + ".$$.tmp"))
        mv -f \(shellQuote(receipt.path + ".$$.tmp")) \(shellQuote(receipt.path))
        if [[ -e \(shellQuote(marker.path)) ]]; then exit 0; fi
        touch \(shellQuote(reachedFinalCheck.path))
        # Deliberately make the receipt-before-exec window observable. The
        # handoff validator must retry this same PID until Python inherits the
        # exact token, then reap it; a raw zsh process is not sufficient proof.
        sleep 0.1
        exec /usr/bin/python3 -c 'import time; time.sleep(30)'
        """
        let process = try launchShell(script, handoffID: handoffID)
        defer { stop(process) }
        let reachedCheck = await waitForFile(reachedFinalCheck)
        XCTAssertTrue(reachedCheck)

        let receiptResult = await MTPLXTerminalHandoffLease.awaitReceipt(
            handoffID: handoffID,
            receiptURL: receipt,
            cancellationMarkerURL: marker,
            isCurrent: { false },
            timeoutSeconds: 0,
            delayedCancellationSeconds: 1
        )
        XCTAssertTrue(receiptResult.cancellationMarked)
        let lease = try XCTUnwrap(receiptResult.lease)
        XCTAssertEqual(lease.processID, Int(process.processIdentifier))
        XCTAssertTrue(PiIntegration().cancelTerminalHandoff(lease))
        let didExit = await waitForExit(process)
        XCTAssertTrue(didExit)
    }

    @MainActor
    func testMismatchedLeaseFailsClosedAndUnrelatedProcessSurvives() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let expected = UUID()
        let process = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable:
                expected.uuidString.lowercased() + "-suffix"
        ])
        defer { stop(process) }
        XCTAssertTrue(waitUntilRunning(process))

        let lease = MTPLXTerminalHandoffLease(
            handoffID: expected,
            processID: Int(process.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("cancelled")
        )
        XCTAssertFalse(PiIntegration().cancelTerminalHandoff(lease))
        XCTAssertTrue(process.isRunning, "a lookalike token must not reap an unrelated PID")

        let noTokenShell = try launchShell("while true; do sleep 1; done", handoffID: nil)
        defer { stop(noTokenShell) }
        let noTokenLease = MTPLXTerminalHandoffLease(
            handoffID: expected,
            processID: Int(noTokenShell.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("no-token-cancelled")
        )
        XCTAssertFalse(PiIntegration().cancelTerminalHandoff(noTokenLease))
        XCTAssertTrue(noTokenShell.isRunning, "unreadable zsh ownership must fail closed")
    }

    func testArtifactsArePrivateAndPathsRemainInvocationUnique() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let firstID = UUID().uuidString.lowercased()
        let secondID = UUID().uuidString.lowercased()
        let firstCommand = directory.appendingPathComponent("open-pi-\(firstID).command")
        let secondCommand = directory.appendingPathComponent("open-pi-\(secondID).command")
        let marker = directory.appendingPathComponent("open-pi-\(firstID).cancelled")

        try MTPLXTerminalHandoffLease.prepareArtifactDirectory(directory)
        try MTPLXTerminalHandoffLease.writeSecureCommandScript("#!/bin/zsh\nexit 0\n", to: firstCommand)
        try MTPLXTerminalHandoffLease.writeSecureCommandScript("#!/bin/zsh\nexit 0\n", to: secondCommand)
        MTPLXTerminalHandoffLease.writeCancellationMarker(at: marker)

        XCTAssertNotEqual(firstCommand, secondCommand)
        XCTAssertEqual(try permissions(of: directory), 0o700)
        XCTAssertEqual(try permissions(of: firstCommand), 0o700)
        XCTAssertEqual(try permissions(of: secondCommand), 0o700)
        XCTAssertEqual(try permissions(of: marker), 0o600)
    }

    func testDesktopIdentityRequiresBothPIDAndLaunchDate() {
        let launchDate = Date(timeIntervalSinceReferenceDate: 123)
        let identity = MTPLXDesktopHandoffIdentity(processID: 1234, launchDate: launchDate)
        XCTAssertTrue(identity.matches(processID: 1234, launchDate: launchDate))
        XCTAssertFalse(identity.matches(processID: 1235, launchDate: launchDate))
        XCTAssertFalse(identity.matches(
            processID: 1234,
            launchDate: launchDate.addingTimeInterval(1)
        ))
        XCTAssertFalse(identity.matches(processID: 1234, launchDate: nil))
    }

    @MainActor
    func testHermesStoreStopReapsItsExactManualTerminalLease() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let handoffID = UUID()
        let process = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        defer { stop(process) }
        XCTAssertTrue(waitUntilRunning(process))

        let lease = MTPLXTerminalHandoffLease(
            handoffID: handoffID,
            processID: Int(process.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("manual-hermes.cancelled")
        )
        let store = HermesAgentStore()
        store.recordManualTerminalHandoffLeaseForTesting(lease)
        XCTAssertEqual(store.manualTerminalHandoffLeaseIDForTesting, handoffID)

        await store.stop()

        XCTAssertNil(store.manualTerminalHandoffLeaseIDForTesting)
        let clientExited = await waitForExit(process)
        XCTAssertTrue(
            clientExited,
            "Stop must reap the exact manual Hermes receipt rather than a discovered process list"
        )
    }

    @MainActor
    func testStaleOpenCodeGateReapsExactDesktopIdentity() {
        let identity = MTPLXDesktopHandoffIdentity(
            processID: 1234,
            launchDate: Date(timeIntervalSinceReferenceDate: 456)
        )
        let launch = OpenCodeDesktopResult(
            action: .opened,
            wasRunning: false,
            didTerminateExistingInstance: false,
            didOpen: true,
            detail: "opened",
            launchedProcessID: identity.processID,
            launchedDesktopIdentity: identity
        )
        var reaped: [MTPLXDesktopHandoffIdentity] = []
        let store = MTPLXBackendStore(openCodeDesktopCanceller: { identity in
            reaped.append(identity)
            return true
        })

        XCTAssertFalse(store.continueOpenCodeHandoff(
            launch,
            handoffID: UUID(),
            isCurrent: { false }
        ))
        XCTAssertEqual(reaped, [identity])
    }

    @MainActor
    func testToleratedRecoveryRefreshMissKeepsLeaseWhileExplicitRefreshStillDegrades() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let handoffID = UUID()
        let client = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        defer { stop(client) }
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 9)
        )
        backend.recordLaunchedPiTerminalHandoffLease(MTPLXTerminalHandoffLease(
            handoffID: handoffID,
            processID: Int(client.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("recovery.cancelled")
        ))
        backend.setDaemonStateForTesting(.running)

        do {
            try await backend.refreshStaticState(markUnreachableOnTransportFailure: false)
            XCTFail("the unavailable test endpoint must fail the refresh")
        } catch {
            // Recovery logs this one miss and lets the two-miss watchdog
            // decide whether the verified restarted daemon is truly dead.
        }
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, Set([handoffID]))
        if case .running = backend.daemonState {
            // Expected: tolerated recovery miss did not reap or degrade.
        } else {
            XCTFail("a tolerated recovery refresh miss must leave the daemon running")
        }

        do {
            try await backend.refreshStaticState()
            XCTFail("the unavailable test endpoint must fail the explicit refresh")
        } catch {
            // Default explicit refresh behavior remains fail-fast.
        }
        if case .degraded = backend.daemonState {
            // Expected: the default remains unchanged.
        } else {
            XCTFail("an explicit refresh miss must still degrade the daemon")
        }
    }

    @MainActor
    func testOMPLeaseIsPresentedAndReapedByExplicitStop() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let handoffID = UUID()
        let client = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        defer { stop(client) }
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 9)
        )
        backend.recordLaunchedOMPTerminalHandoffLease(MTPLXTerminalHandoffLease(
            handoffID: handoffID,
            processID: Int(client.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("omp.cancelled")
        ))

        XCTAssertTrue(backend.ompTerminalAgentRunning)
        XCTAssertEqual(backend.ompTerminalAgentProcessIDs, [Int(client.processIdentifier)])
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, Set([handoffID]))

        await backend.stopDaemon()

        let clientExited = await waitForExit(client)
        XCTAssertTrue(clientExited)
        XCTAssertFalse(backend.ompTerminalAgentRunning)
        XCTAssertTrue(backend.ompTerminalAgentProcessIDs.isEmpty)
        XCTAssertTrue(backend.terminalHandoffLeaseIDsForTesting.isEmpty)
    }

    @MainActor
    func testLeaseSurvivesAutomaticRecoveryStatusesThenExplicitStopReapsIt() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let release = directory.appendingPathComponent("release")
        let counter = directory.appendingPathComponent("counter")
        try Data().write(to: release)

        let scheduledGate = HandoffRecoveryGate()
        let restartingGate = HandoffRecoveryGate()
        let postStartGate = HandoffRecoveryGate()
        let supervisor = DaemonSupervisor(
            restartPolicy: DaemonRestartPolicy(
                maximumAttempts: 1,
                initialDelaySeconds: 0,
                maximumDelaySeconds: 0,
                crashWindowSeconds: 60
            ),
            restartSleeper: { _ in await scheduledGate.wait() },
            beforeAutomaticRestartStart: { await restartingGate.wait() }
        )
        let configuration = MTPLXAppConfiguration(automaticDaemonRestart: true)
        let backend = MTPLXBackendStore(
            configuration: configuration,
            supervisor: supervisor,
            localFanRestorer: { true },
            beforePostStartRefresh: { await postStartGate.wait() }
        )
        let handoffID = UUID()
        let client = try launchSleep(environment: [
            MTPLXTerminalHandoffLease.environmentVariable: handoffID.uuidString.lowercased()
        ])
        defer { stop(client) }
        let lease = MTPLXTerminalHandoffLease(
            handoffID: handoffID,
            processID: Int(client.processIdentifier),
            cancellationMarkerURL: directory.appendingPathComponent("client.cancelled")
        )
        backend.recordLaunchedPiTerminalHandoffLease(lease)

        let script = """
        count=$(cat \(shellQuote(counter.path)) 2>/dev/null || echo 0)
        count=$((count + 1))
        echo "$count" > \(shellQuote(counter.path))
        if [ "$count" -eq 1 ]; then
          while [ -e \(shellQuote(release.path)) ]; do sleep 0.01; done
          exit 17
        fi
        while :; do sleep 1; done
        """
        _ = try await supervisor.start(
            command: DaemonCommand(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", script]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        try FileManager.default.removeItem(at: release)

        let reachedScheduled = await waitForRestartStatus(supervisor) {
            if case .scheduled = $0 { return true }
            return false
        }
        XCTAssertTrue(reachedScheduled)
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, Set([handoffID]))

        await scheduledGate.waitUntilEntered()
        await scheduledGate.release()
        await restartingGate.waitUntilEntered()
        let reachedRestarting = await waitForRestartStatus(supervisor) {
            if case .restarting = $0 { return true }
            return false
        }
        XCTAssertTrue(reachedRestarting)
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, Set([handoffID]))

        await restartingGate.release()
        let reachedRunningAfterRestart = await waitForRestartStatus(supervisor) {
            if case .runningAfterRestart = $0 { return true }
            return false
        }
        XCTAssertTrue(reachedRunningAfterRestart)
        await postStartGate.waitUntilEntered()
        XCTAssertEqual(backend.terminalHandoffLeaseIDsForTesting, Set([handoffID]))

        let explicitStop = Task { @MainActor in
            await backend.stopDaemon()
        }
        await Task.yield()
        await postStartGate.release()
        await explicitStop.value
        let clientExited = await waitForExit(client)
        XCTAssertTrue(clientExited, "explicit Stop must reap the retained lease after recovery")
        XCTAssertTrue(backend.terminalHandoffLeaseIDsForTesting.isEmpty)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-handoff-tests-\(UUID().uuidString.lowercased())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func permissions(of url: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.posixPermissions] as? NSNumber)?.intValue ?? -1
    }

    private func launchSleep(environment: [String: String]) throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", "import time; time.sleep(30)"]
        process.environment = ProcessInfo.processInfo.environment.merging(environment) { _, new in new }
        try process.run()
        return process
    }

    private func launchShell(
        _ script: String,
        handoffID: UUID?,
        additionalArguments: [String] = []
    ) throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", script] + additionalArguments
        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/usr/bin:/bin"
        if let handoffID {
            environment[MTPLXTerminalHandoffLease.environmentVariable] = handoffID.uuidString.lowercased()
        }
        process.environment = environment
        try process.run()
        return process
    }

    private func waitUntilRunning(_ process: Process) -> Bool {
        for _ in 0..<50 where !process.isRunning {
            Thread.sleep(forTimeInterval: 0.01)
        }
        return process.isRunning
    }

    @MainActor
    private func waitForFile(_ url: URL) async -> Bool {
        for _ in 0..<100 {
            if FileManager.default.fileExists(atPath: url.path) { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return false
    }

    @MainActor
    private func waitForExit(_ process: Process) async -> Bool {
        for _ in 0..<100 {
            if !process.isRunning { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return !process.isRunning
    }

    @MainActor
    private func waitForRestartStatus(
        _ supervisor: DaemonSupervisor,
        matching predicate: (DaemonRestartStatus) -> Bool
    ) async -> Bool {
        for _ in 0..<200 {
            if predicate(supervisor.supervisionSnapshot().restartStatus) { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return predicate(supervisor.supervisionSnapshot().restartStatus)
    }

    private func stop(_ process: Process) {
        guard process.isRunning else { return }
        process.terminate()
        process.waitUntilExit()
    }

    private func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\\\''"))'"
    }
}
