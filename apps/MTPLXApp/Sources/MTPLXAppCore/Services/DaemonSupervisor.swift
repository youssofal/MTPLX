import Foundation

public enum DaemonState: Equatable, Sendable {
    case stopped
    case starting
    case warming
    case running
    case degraded(String)
    case stopping
    case crashed(Int32?)
}

public enum DaemonSupervisorError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case alreadyRunning
    case launchFailed(String)
    case healthTimeout
    case portOccupied(pid: Int?, launchID: String?)
    case launchIdentityMismatch(expected: String, observed: String?)
    case fanRampTimeout

    public var description: String {
        switch self {
        case .alreadyRunning:
            return "MTPLX is already running."
        case .launchFailed(let detail):
            return "MTPLX couldn't start: \(detail)"
        case .healthTimeout:
            return "MTPLX took too long to start up."
        case .portOccupied(let pid, let launchID):
            let pidText = pid.map { "pid \($0)" } ?? "unknown pid"
            if let launchID {
                return "Port is already used by MTPLX (\(pidText), launch \(launchID))."
            }
            return "Port is already used by another local app (\(pidText))."
        case .launchIdentityMismatch(let expected, let observed):
            return "MTPLX startup didn't match what we expected. Expected \(expected), got \(observed ?? "nothing")."
        case .fanRampTimeout:
            return "Couldn't get your fans to max in time."
        }
    }

    public var errorDescription: String? { description }
}

public enum DaemonStartupPhase: Equatable, Sendable {
    case idle
    case launching
    case waitingForOwnedHealth
    case rampingFans
    case warming
    case ready
    case failed(String)
}

/// Bounds automatic recovery for an app-owned daemon. The policy lives only
/// in the app process: it never writes a command line, environment, or API key
/// to disk.
public struct DaemonRestartPolicy: Equatable, Sendable {
    public var maximumAttempts: Int
    public var initialDelaySeconds: TimeInterval
    public var maximumDelaySeconds: TimeInterval
    public var crashWindowSeconds: TimeInterval

    public init(
        maximumAttempts: Int = 3,
        initialDelaySeconds: TimeInterval = 1,
        maximumDelaySeconds: TimeInterval = 30,
        crashWindowSeconds: TimeInterval = 120
    ) {
        self.maximumAttempts = max(0, maximumAttempts)
        self.initialDelaySeconds = max(0, initialDelaySeconds)
        self.maximumDelaySeconds = max(self.initialDelaySeconds, maximumDelaySeconds)
        self.crashWindowSeconds = max(0, crashWindowSeconds)
    }

    public static let `default` = DaemonRestartPolicy()
}

public enum DaemonRestartStatus: Equatable, Sendable {
    case idle
    case scheduled(attempt: Int, delaySeconds: TimeInterval)
    case restarting(attempt: Int)
    case runningAfterRestart(attempt: Int)
    case exhausted(attempts: Int, lastExitStatus: Int32?)
}

public enum DaemonRestartEligibility: Equatable, Sendable {
    case noDaemon
    case currentSessionProtected
    case currentSessionUnprotected
    case adoptedPriorSession
}

/// Small, secret-free status surface for the app chrome and logs view.
public struct DaemonSupervisionSnapshot: Equatable, Sendable {
    /// Monotonic delivery revision. Consumers that hop onto another executor
    /// must ignore an older snapshot that arrives after a newer one.
    public let revision: Int
    public let state: DaemonState
    public let restartStatus: DaemonRestartStatus
    public let restartCount: Int
    public let restartEligibility: DaemonRestartEligibility
    /// Monotonic for each app-owned or adopted daemon lifecycle attempt. It
    /// advances before the first asynchronous probe, so a terminal callback
    /// from an older daemon cannot be mistaken for a newer launch that has
    /// not yet reserved a Process.
    public let lifecycleEpoch: Int
    /// Monotonic for this supervisor instance; unlike restartCount it never
    /// resets when the circuit-breaker window rolls over.
    public let recoveryGeneration: Int

    public init(
        revision: Int = 0,
        state: DaemonState,
        restartStatus: DaemonRestartStatus,
        restartCount: Int,
        restartEligibility: DaemonRestartEligibility = .noDaemon,
        lifecycleEpoch: Int = 0,
        recoveryGeneration: Int
    ) {
        self.revision = revision
        self.state = state
        self.restartStatus = restartStatus
        self.restartCount = restartCount
        self.restartEligibility = restartEligibility
        self.lifecycleEpoch = lifecycleEpoch
        self.recoveryGeneration = recoveryGeneration
    }
}

public final class DaemonSupervisor: @unchecked Sendable {
    private struct OwnedLaunch: Sendable {
        let command: DaemonCommand
        let healthBaseURL: URL
        let apiKey: String?
        let probeHealth: Bool
        let timeoutSeconds: TimeInterval
        let expectedLaunchID: String?
        let requireActualFanRamp: Bool
        let onPhase: (@Sendable (DaemonStartupPhase) -> Void)?
    }

    private let lock = NSLock()
    private var process: Process?
    private var adoptedProcessID: pid_t?
    private let logStore: BoundedLogStore
    private let restartPolicy: DaemonRestartPolicy
    private let restartSleeper: @Sendable (TimeInterval) async -> Void
    private let initialHealthProbe: @Sendable (URL, String?) async -> HealthPayload?
    private let healthWaitProbe: @Sendable (URL, String?) async -> HealthPayload?
    private let beforeProcessReservation: @Sendable () async -> Void
    private let beforeProcessRun: @Sendable () async -> Void
    private let beforePostRunLivenessCheck: @Sendable () async -> Void
    private let beforeAutomaticRestartStart: @Sendable () async -> Void
    private let beforeStopProcessFamilyResolution: @Sendable () async -> Void
    private let beforeStopProcessFamilySignal: @Sendable () async -> Void
    private let beforeTerminationHandling: @Sendable (Process) -> Void
    private var lastOwnedLaunch: OwnedLaunch?
    private var restartTask: Task<Void, Never>?
    /// Task identity is separate from its generation: an attempt that queues
    /// its successor must not clear the successor's task in its defer block.
    private var restartTaskID: UUID?
    private var restartGeneration = 0
    private var lifecycleEpoch = 0
    private var recentCrashDates: [Date] = []
    private var automaticRestartEnabled = false
    /// Bounded wait for fan-ramp verification once /health is already ok.
    /// A healthy daemon proceeds to ready when this expires — it is never
    /// reaped over a fan receipt. Overridable for tests.
    public var fanRampGraceSeconds: TimeInterval = 30
    private var automaticRestartEligible = false
    private var automaticLaunchGeneration: Int?
    // Kept independently from the restart recipe so a Stop that begins just
    // before its root exits can still find inherited-token descendants after
    // the parent has been reaped and its PPID relationship has disappeared.
    private var ownedLaunchID: String?
    private var launchInProgress = false
    private var launchCompletionWaiters: [CheckedContinuation<Void, Never>] = []
    private var statusObserver: (@Sendable (DaemonSupervisionSnapshot) -> Void)?
    private var statusRevision = 0

    public private(set) var state: DaemonState = .stopped
    public private(set) var restartStatus: DaemonRestartStatus = .idle
    public private(set) var restartCount = 0
    public private(set) var recoveryGeneration = 0

    public init(
        logStore: BoundedLogStore = BoundedLogStore(),
        restartPolicy: DaemonRestartPolicy = .default,
        restartSleeper: @escaping @Sendable (TimeInterval) async -> Void = { delay in
            guard delay > 0 else { return }
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        },
        initialHealthProbe: @escaping @Sendable (URL, String?) async -> HealthPayload? = { baseURL, apiKey in
            try? await MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey).health()
        },
        healthWaitProbe: @escaping @Sendable (URL, String?) async -> HealthPayload? = { baseURL, apiKey in
            try? await MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey).health()
        },
        // Test seam immediately before the atomic lifecycle reservation.
        beforeProcessReservation: @escaping @Sendable () async -> Void = {},
        // Test seam for the narrow period after ownership is published but
        // before Process.run() assigns a PID. Production uses the no-op.
        beforeProcessRun: @escaping @Sendable () async -> Void = {},
        // Test seam after Process.run() but before the first liveness check.
        beforePostRunLivenessCheck: @escaping @Sendable () async -> Void = {},
        // Test seam after automatic restart state is published but before it
        // starts the next owned launch. Production uses the no-op.
        beforeAutomaticRestartStart: @escaping @Sendable () async -> Void = {},
        // Test seam immediately before Stop resolves the current process
        // family. Production uses the no-op.
        beforeStopProcessFamilyResolution: @escaping @Sendable () async -> Void = {},
        // Test seam after the full process family is snapshotted for Stop,
        // before any signal is sent. Production uses the no-op.
        beforeStopProcessFamilySignal: @escaping @Sendable () async -> Void = {},
        // Test seam immediately before a Process termination handler acquires
        // supervisor state. Production uses the no-op.
        beforeTerminationHandling: @escaping @Sendable (Process) -> Void = { _ in }
    ) {
        self.logStore = logStore
        self.restartPolicy = restartPolicy
        self.restartSleeper = restartSleeper
        self.initialHealthProbe = initialHealthProbe
        self.healthWaitProbe = healthWaitProbe
        self.beforeProcessReservation = beforeProcessReservation
        self.beforeProcessRun = beforeProcessRun
        self.beforePostRunLivenessCheck = beforePostRunLivenessCheck
        self.beforeAutomaticRestartStart = beforeAutomaticRestartStart
        self.beforeStopProcessFamilyResolution = beforeStopProcessFamilyResolution
        self.beforeStopProcessFamilySignal = beforeStopProcessFamilySignal
        self.beforeTerminationHandling = beforeTerminationHandling
    }

    public var logs: BoundedLogStore {
        logStore
    }

    public func supervisionSnapshot() -> DaemonSupervisionSnapshot {
        lock.withLock { supervisionSnapshotLocked() }
    }

    // Narrow test observability for the secret-lifetime invariant. These
    // report only booleans; no command, API key, or task details escape.
    var hasRetainedRestartRecipeForTesting: Bool {
        lock.withLock { lastOwnedLaunch != nil }
    }

    var hasOutstandingRestartTaskForTesting: Bool {
        lock.withLock { restartTask != nil || restartTaskID != nil }
    }

    /// Explicit user-controlled opt-in. Disabled is the safe default because a
    /// crash can be caused by an out-of-memory condition that should not spin.
    public func setAutomaticRestartEnabled(_ enabled: Bool) {
        let changed = lock.withLock { () -> Bool in
            guard automaticRestartEnabled != enabled else { return false }
            automaticRestartEnabled = enabled
            guard !enabled else { return true }
            // restartGeneration also guards manual launches while they are
            // between the initial probe and Process reservation. Changing an
            // unrelated Settings toggle must not make such a launch throw
            // "cancelled" and leave its process running. Invalidate it only
            // when an automatic attempt is actually pending or in progress.
            let hasAutomaticWork: Bool
            switch restartStatus {
            case .scheduled, .restarting:
                hasAutomaticWork = true
            case .idle, .runningAfterRestart, .exhausted:
                hasAutomaticWork = automaticLaunchGeneration != nil
            }
            if hasAutomaticWork {
                restartGeneration &+= 1
                cancelRestartTaskLocked()
            }
            restartStatus = .idle
            recentCrashDates.removeAll()
            restartCount = 0
            automaticRestartEligible = false
            // A disabled setting must not retain an API key merely so a later
            // toggle can restart an old daemon. Enable before a fresh launch.
            lastOwnedLaunch = nil
            // This is an expected cancellation rather than a crash. Do not
            // raw-terminate the root here: its termination handler would drop
            // `process` before stopInternal can discover/reap its children.
            // The cancelled restart task observes this state and performs the
            // normal full-family stop while it still owns the Process.
            if automaticLaunchGeneration != nil {
                state = .stopping
            }
            return true
        }
        if changed {
            notifyStatusObserver()
        }
    }

    /// The app store uses this to mirror supervised restarts into its published
    /// state. The callback deliberately contains no launch command or API key.
    public func setStatusObserver(
        _ observer: (@Sendable (DaemonSupervisionSnapshot) -> Void)?
    ) {
        let snapshot = lock.withLock { () -> DaemonSupervisionSnapshot in
            statusObserver = observer
            return supervisionSnapshotLocked()
        }
        observer?(snapshot)
    }

    public func isRunning() -> Bool {
        lock.withLock {
            process?.isRunning == true || adoptedProcessID != nil
        }
    }

    public func start(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        probeHealth: Bool = true,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        adoptExistingAppOwnedDaemon: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload? {
        try await startOwned(
            OwnedLaunch(
                command: command,
                healthBaseURL: healthBaseURL,
                apiKey: apiKey,
                probeHealth: probeHealth,
                timeoutSeconds: timeoutSeconds,
                expectedLaunchID: expectedLaunchID,
                requireActualFanRamp: requireActualFanRamp,
                onPhase: onPhase
            ),
            adoptExistingAppOwnedDaemon: adoptExistingAppOwnedDaemon,
            resetSupervision: true,
            automaticAttempt: nil
        )
    }

    private func startOwned(
        _ launch: OwnedLaunch,
        adoptExistingAppOwnedDaemon: Bool,
        resetSupervision: Bool,
        automaticAttempt: Int?,
        expectedRestartGeneration: Int? = nil
    ) async throws -> HealthPayload? {
        let command = launch.command
        let healthBaseURL = launch.healthBaseURL
        let apiKey = launch.apiKey
        let probeHealth = launch.probeHealth
        let timeoutSeconds = launch.timeoutSeconds
        let expectedLaunchID = launch.expectedLaunchID
        let requireActualFanRamp = launch.requireActualFanRamp
        let onPhase = launch.onPhase
        let launchContext = try lock.withLock { () throws -> (generation: Int, lifecycleEpoch: Int) in
            if process != nil || adoptedProcessID != nil || launchInProgress {
                throw DaemonSupervisorError.alreadyRunning
            }
            if resetSupervision {
                restartGeneration &+= 1
                cancelRestartTaskLocked()
                recentCrashDates.removeAll()
                restartCount = 0
                restartStatus = .idle
                lastOwnedLaunch = nil
                automaticRestartEligible = false
            }
            if let expectedRestartGeneration,
               (restartGeneration != expectedRestartGeneration || !automaticRestartEnabled) {
                throw DaemonSupervisorError.launchFailed("automatic restart was cancelled")
            }
            automaticLaunchGeneration = automaticAttempt == nil ? nil : restartGeneration
            // Allocate the lifecycle token before the first health probe.
            // A stopped/crashed callback for the prior daemon can otherwise
            // arrive while this launch is suspended and clear the store's
            // active launch ID before a Process exists.
            lifecycleEpoch &+= 1
            state = .starting
            return (restartGeneration, lifecycleEpoch)
        }
        let launchGeneration = launchContext.generation
        let launchLifecycleEpoch = launchContext.lifecycleEpoch
        notifyStatusObserver()
        onPhase?(.launching)

        let existingHealth = probeHealth ? await initialHealthProbe(healthBaseURL, apiKey) : nil
        if Task.isCancelled, automaticAttempt != nil {
            await stopCancelledAutomaticLaunchIfCurrent(
                generation: launchGeneration,
                lifecycleEpoch: launchLifecycleEpoch
            )
            throw DaemonSupervisorError.launchFailed("automatic restart was cancelled")
        }
        if let existing = existingHealth, existing.ok {
            if adoptExistingAppOwnedDaemon,
               canAdopt(existing, for: command, requireActualFanRamp: requireActualFanRamp) {
                guard adoptCurrentLaunch(
                    existing,
                    generation: launchGeneration,
                    lifecycleEpoch: launchLifecycleEpoch,
                    automaticAttempt: automaticAttempt
                ) else {
                    abortUnstartedLaunch(
                        generation: launchGeneration,
                        lifecycleEpoch: launchLifecycleEpoch
                    )
                    await stopCancelledAutomaticLaunchIfCurrent(
                        generation: launchGeneration,
                        lifecycleEpoch: launchLifecycleEpoch
                    )
                    throw DaemonSupervisorError.launchFailed("daemon launch was cancelled")
                }
                notifyStatusObserver()
                await logAdoption(existing)
                onPhase?(.ready)
                return existing
            }
            abortUnstartedLaunch(
                generation: launchGeneration,
                lifecycleEpoch: launchLifecycleEpoch
            )
            throw DaemonSupervisorError.portOccupied(
                pid: existing.startup?.pid,
                launchID: existing.startup?.launchId
            )
        }
        let next = Process()
        next.executableURL = command.executableURL
        next.arguments = command.arguments
        next.environment = MTPLXCommandBuilder.appSubprocessEnvironment(
            environment: ProcessInfo.processInfo.environment.merging(command.environment) { _, new in new }
        )

        let stdout = Pipe()
        let stderr = Pipe()
        next.standardOutput = stdout
        next.standardError = stderr
        attach(pipe: stdout, stream: .stdout)
        attach(pipe: stderr, stream: .stderr)
        next.terminationHandler = { [weak self] process in
            self?.beforeTerminationHandling(process)
            self?.handleTermination(of: process)
        }

        await beforeProcessReservation()

        // Validate and publish ownership under the same lock. Stop increments
        // restartGeneration under this lock, so it cannot return between a
        // successful validation and this pre-PID reservation.
        let reserved = lock.withLock { () -> Bool in
            guard launchMayProceedLocked(
                generation: launchGeneration,
                automaticAttempt: automaticAttempt
            ), lifecycleEpoch == launchLifecycleEpoch,
               process == nil, adoptedProcessID == nil, !launchInProgress
            else { return false }
            process = next
            adoptedProcessID = nil
            ownedLaunchID = launchIdentifier(from: command)
            launchInProgress = true
            // A Process has been reserved but does not have a usable PID until
            // run() returns. Keep the public phase at .starting through that
            // hand-off; Stop's launch barrier covers this interval.
            state = .starting
            return true
        }
        guard reserved else {
            abortUnstartedLaunch(
                generation: launchGeneration,
                lifecycleEpoch: launchLifecycleEpoch
            )
            await stopCancelledAutomaticLaunchIfCurrent(
                generation: launchGeneration,
                lifecycleEpoch: launchLifecycleEpoch
            )
            throw DaemonSupervisorError.launchFailed("daemon launch was cancelled")
        }
        notifyStatusObserver()

        await beforeProcessRun()
        let mayRun = lock.withLock {
            process === next &&
                lifecycleEpoch == launchLifecycleEpoch &&
                state != .stopping
        }
        guard mayRun else {
            let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
                if process === next, lifecycleEpoch == launchLifecycleEpoch {
                    process = nil
                }
                return finishLaunchLocked()
            }
            waiters.forEach { $0.resume() }
            notifyStatusObserver()
            await stopCancelledAutomaticLaunchIfCurrent(
                generation: launchGeneration,
                lifecycleEpoch: launchLifecycleEpoch
            )
            throw DaemonSupervisorError.launchFailed("daemon launch was cancelled")
        }

        do {
            try next.run()
        } catch {
            let waiters = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
                if process === next, lifecycleEpoch == launchLifecycleEpoch {
                    process = nil
                    state = .stopped
                }
                automaticLaunchGeneration = nil
                return finishLaunchLocked()
            }
            waiters.forEach { $0.resume() }
            notifyStatusObserver()
            throw DaemonSupervisorError.launchFailed(error.localizedDescription)
        }

        let wasCancelledBeforeRunCompleted = lock.withLock { () -> Bool in
            let cancelled = restartGeneration != launchGeneration ||
                (automaticAttempt != nil && !automaticRestartEnabled) ||
                lifecycleEpoch != launchLifecycleEpoch ||
                state == .stopping || process !== next
            if !cancelled {
                state = .warming
            }
            return cancelled
        }
        let launchWaiters = lock.withLock { finishLaunchLocked() }
        launchWaiters.forEach { $0.resume() }
        if wasCancelledBeforeRunCompleted {
            // The PID is valid now. An explicit Stop is already waiting on the
            // launch barrier; a disabled automatic recovery has no such caller
            // and therefore tears itself down here.
            if automaticAttempt != nil {
                await stopInternal(
                    graceSeconds: 2,
                    additionalProcessIDs: [],
                    clearSupervision: false,
                    expectedProcess: next,
                    expectedLifecycleEpoch: launchLifecycleEpoch
                )
            }
            throw DaemonSupervisorError.launchFailed("daemon launch was cancelled")
        }
        notifyStatusObserver()
        await logStore.append(
            "launched \(command.executableURL.path) \(command.arguments.joined(separator: " "))",
            stream: .system
        )

        // Do not turn a failed initial launch into a restart loop. A process
        // that has already exited before this start returns is surfaced to the
        // caller; an automatic retry records the failure through its bounded
        // retry path below.
        await beforePostRunLivenessCheck()
        guard next.isRunning else {
            let exitStatus = next.terminationStatus
            lock.withLock {
                guard process === next, lifecycleEpoch == launchLifecycleEpoch else { return }
                process = nil
                automaticLaunchGeneration = nil
                if state == .stopping {
                    state = .stopped
                } else if exitStatus == 0 {
                    // The handler may not have acquired the lock yet. Mirror
                    // its clean-exit cleanup here before dropping ownership;
                    // otherwise a retry task sees .restarting + a retained
                    // recipe and incorrectly restarts a clean exit.
                    state = .stopped
                    automaticRestartEligible = false
                    lastOwnedLaunch = nil
                    restartStatus = .idle
                    if automaticAttempt != nil {
                        // This is the task currently executing this exact
                        // Process/lifecycle. Drop its retained launch recipe
                        // now rather than waiting for its catch/defer path;
                        // a newer recovery cannot exist while ownership still
                        // matches this Process under the lock.
                        restartTask = nil
                        restartTaskID = nil
                    }
                } else {
                    state = .crashed(exitStatus)
                }
            }
            notifyStatusObserver()
            throw DaemonSupervisorError.launchFailed(
                "daemon exited during launch with status \(exitStatus)"
            )
        }

        let readyHealth: HealthPayload?
        if probeHealth {
            do {
                readyHealth = try await waitForHealth(
                    baseURL: healthBaseURL,
                    apiKey: apiKey,
                    timeoutSeconds: timeoutSeconds,
                    expectedLaunchID: expectedLaunchID,
                    requireActualFanRamp: requireActualFanRamp,
                    onPhase: onPhase
                )
            } catch {
                await stopInternal(
                    graceSeconds: 2,
                    additionalProcessIDs: [],
                    clearSupervision: resetSupervision,
                    expectedProcess: next,
                    expectedLifecycleEpoch: launchLifecycleEpoch
                )
                throw error
            }
        } else {
            readyHealth = nil
        }
        let automaticLaunchStillCurrent = lock.withLock { () -> Bool in
            guard restartGeneration == launchGeneration,
                  lifecycleEpoch == launchLifecycleEpoch,
                  process === next,
                  state != .stopping
            else { return false }
            state = .running
            if automaticRestartEnabled {
                lastOwnedLaunch = launch
            }
            automaticRestartEligible = automaticRestartEnabled && lastOwnedLaunch != nil
            automaticLaunchGeneration = nil
            if let automaticAttempt {
                restartStatus = .runningAfterRestart(attempt: automaticAttempt)
                recoveryGeneration &+= 1
            }
            return true
        }
        guard automaticLaunchStillCurrent else {
            if automaticAttempt != nil {
                await stopInternal(
                    graceSeconds: 2,
                    additionalProcessIDs: [],
                    clearSupervision: false,
                    expectedProcess: next,
                    expectedLifecycleEpoch: launchLifecycleEpoch
                )
            }
            throw DaemonSupervisorError.launchFailed("daemon launch was cancelled")
        }
        notifyStatusObserver()
        onPhase?(.ready)
        return readyHealth
    }

    private func handleTermination(of terminatedProcess: Process) {
        let exitStatus = terminatedProcess.terminationStatus
        let transition = lock.withLock { () -> (recovery: (attempt: Int, delay: TimeInterval, generation: Int)?, evidence: DaemonRestartStatus?)? in
            guard process === terminatedProcess else { return nil }
            process = nil
            automaticLaunchGeneration = nil
            guard state != .stopping else {
                state = .stopped
                return (nil, nil)
            }
            state = exitStatus == 0 ? .stopped : .crashed(exitStatus)
            guard exitStatus != 0 else {
                automaticRestartEligible = false
                lastOwnedLaunch = nil
                restartStatus = .idle
                return (nil, nil)
            }
            guard automaticRestartEligible else { return (nil, nil) }
            automaticRestartEligible = false
            let recovery = scheduleRestartLocked(lastExitStatus: exitStatus)
            return (recovery, restartStatus)
        }
        notifyStatusObserver()
        Task { [weak self] in
            guard let self else { return }
            await self.logStore.append("daemon exited with status \(exitStatus)", stream: .system)
            if let evidence = transition?.evidence {
                await self.logRestartEvidence(evidence)
            }
        }
    }

    private func scheduleRestartLocked(
        lastExitStatus: Int32?
    ) -> (attempt: Int, delay: TimeInterval, generation: Int)? {
        guard automaticRestartEnabled, let launch = lastOwnedLaunch else { return nil }
        let now = Date()
        recentCrashDates.removeAll {
            now.timeIntervalSince($0) > restartPolicy.crashWindowSeconds
        }
        recentCrashDates.append(now)
        let attempt = recentCrashDates.count
        guard attempt <= restartPolicy.maximumAttempts else {
            restartStatus = .exhausted(
                attempts: restartCount,
                lastExitStatus: lastExitStatus
            )
            lastOwnedLaunch = nil
            return nil
        }
        restartCount = attempt
        let multiplier = pow(2.0, Double(max(0, attempt - 1)))
        let delay = min(
            restartPolicy.maximumDelaySeconds,
            restartPolicy.initialDelaySeconds * multiplier
        )
        restartStatus = .scheduled(attempt: attempt, delaySeconds: delay)
        let generation = restartGeneration
        let taskID = UUID()
        restartTaskID = taskID
        restartTask = Task { [weak self] in
            await self?.runAutomaticRestart(
                launch: launch,
                attempt: attempt,
                delay: delay,
                generation: generation,
                taskID: taskID
            )
        }
        return (attempt, delay, generation)
    }

    private func runAutomaticRestart(
        launch: OwnedLaunch,
        attempt: Int,
        delay: TimeInterval,
        generation: Int,
        taskID: UUID
    ) async {
        defer { clearFinishedRestartTask(taskID) }
        await restartSleeper(delay)
        guard !Task.isCancelled else { return }
        let mayRestart = lock.withLock { () -> Bool in
            guard restartGeneration == generation else { return false }
            guard case .scheduled(let scheduledAttempt, _) = restartStatus,
                  scheduledAttempt == attempt
            else { return false }
            restartStatus = .restarting(attempt: attempt)
            // Reserve automatic ownership before the first await below. A
            // Settings toggle in the logging/start gap must settle this
            // attempt to stopped rather than leaving .starting forever.
            automaticLaunchGeneration = generation
            state = .starting
            return true
        }
        guard mayRestart else { return }
        notifyStatusObserver()
        await beforeAutomaticRestartStart()
        if Task.isCancelled {
            await stopCancelledAutomaticLaunchIfCurrent(generation: generation)
            return
        }
        await logStore.append(
            "automatic restart attempt \(attempt) of \(restartPolicy.maximumAttempts) starting",
            stream: .system
        )
        if Task.isCancelled {
            await stopCancelledAutomaticLaunchIfCurrent(generation: generation)
            return
        }
        do {
            _ = try await startOwned(
                launch,
                adoptExistingAppOwnedDaemon: false,
                resetSupervision: false,
                automaticAttempt: attempt,
                expectedRestartGeneration: generation
            )
            await logStore.append(
                "automatic restart attempt \(attempt) recovered daemon health",
                stream: .system
            )
        } catch {
            await stopCancelledAutomaticLaunchIfCurrent(generation: generation)
            let alreadyQueued = lock.withLock { () -> Bool in
                if case .scheduled(let nextAttempt, _) = restartStatus {
                    return nextAttempt > attempt
                }
                return false
            }
            if !alreadyQueued {
                let transition = lock.withLock { () -> (recovery: (attempt: Int, delay: TimeInterval, generation: Int)?, evidence: DaemonRestartStatus?)? in
                    // An explicit Stop, a manual Start, or disabling the setting can
                    // happen while the health probe above is suspended. Those actions
                    // invalidate this recovery generation, so they must never be
                    // followed by a stale retry.
                    guard restartGeneration == generation else { return nil }
                    let recovery = scheduleRestartLocked(lastExitStatus: nil)
                    return (recovery, restartStatus)
                }
                notifyStatusObserver()
                if let evidence = transition?.evidence {
                    await logRestartEvidence(evidence)
                }
            }
            await logStore.append(
                "automatic restart attempt \(attempt) failed: \(String(describing: error))",
                stream: .system
            )
        }
    }

    private func logRestartEvidence(_ status: DaemonRestartStatus) async {
        switch status {
        case .scheduled(let attempt, let delay):
            await logStore.append(
                "automatic restart scheduled: attempt \(attempt) of \(restartPolicy.maximumAttempts) in \(String(format: "%.1f", delay))s",
                stream: .system
            )
        case .exhausted(let attempts, let status):
            await logStore.append(
                "automatic restart circuit breaker open after \(attempts) attempts; last exit status \(status.map(String.init) ?? "unknown")",
                stream: .system
            )
        default:
            break
        }
    }

    private func supervisionSnapshotLocked() -> DaemonSupervisionSnapshot {
        DaemonSupervisionSnapshot(
            revision: statusRevision,
            state: state,
            restartStatus: restartStatus,
            restartCount: restartCount,
            restartEligibility: restartEligibilityLocked(),
            lifecycleEpoch: lifecycleEpoch,
            recoveryGeneration: recoveryGeneration
        )
    }

    private func notifyStatusObserver() {
        let (observer, snapshot) = lock.withLock { () -> ((@Sendable (DaemonSupervisionSnapshot) -> Void)?, DaemonSupervisionSnapshot) in
            statusRevision &+= 1
            return (statusObserver, supervisionSnapshotLocked())
        }
        observer?(snapshot)
    }

    private func cancelRestartTaskLocked() {
        restartTask?.cancel()
        restartTask = nil
        restartTaskID = nil
    }

    private func clearFinishedRestartTask(_ taskID: UUID) {
        lock.withLock {
            guard restartTaskID == taskID else { return }
            restartTask = nil
            restartTaskID = nil
        }
    }

    private func finishLaunchLocked() -> [CheckedContinuation<Void, Never>] {
        launchInProgress = false
        let waiters = launchCompletionWaiters
        launchCompletionWaiters.removeAll()
        return waiters
    }

    private func restartEligibilityLocked() -> DaemonRestartEligibility {
        if adoptedProcessID != nil {
            return .adoptedPriorSession
        }
        if process != nil {
            return automaticRestartEligible ? .currentSessionProtected : .currentSessionUnprotected
        }
        return .noDaemon
    }

    private func launchMayProceedLocked(generation: Int, automaticAttempt: Int?) -> Bool {
        restartGeneration == generation &&
            state != .stopping &&
            (automaticAttempt == nil || automaticRestartEnabled)
    }

    private func abortUnstartedLaunch(
        generation: Int,
        lifecycleEpoch: Int
    ) {
        let changed = lock.withLock { () -> Bool in
            // A second manual Start can overtake the first while both are in
            // their initial health probes. Only the owning attempt may turn
            // .starting back into .stopped or emit a terminal snapshot.
            guard restartGeneration == generation,
                  self.lifecycleEpoch == lifecycleEpoch,
                  process == nil,
                  adoptedProcessID == nil,
                  !launchInProgress
            else { return false }
            if automaticLaunchGeneration == generation {
                automaticLaunchGeneration = nil
            }
            guard state != .stopping else { return false }
            state = .stopped
            return true
        }
        if changed {
            notifyStatusObserver()
        }
    }

    private func stopCancelledAutomaticLaunchIfCurrent(
        generation: Int,
        lifecycleEpoch: Int? = nil
    ) async {
        let cancelledLaunch = lock.withLock { () -> (process: Process?, lifecycleEpoch: Int)? in
            automaticLaunchGeneration == generation &&
                (lifecycleEpoch == nil || self.lifecycleEpoch == lifecycleEpoch) &&
                state == .stopping
                ? (process, self.lifecycleEpoch)
                : nil
        }
        guard let cancelledLaunch else { return }
        await stopInternal(
            graceSeconds: 2,
            additionalProcessIDs: [],
            clearSupervision: false,
            expectedProcess: cancelledLaunch.process,
            expectedLifecycleEpoch: cancelledLaunch.lifecycleEpoch
        )
    }

    private func adoptCurrentLaunch(
        _ health: HealthPayload,
        generation: Int,
        lifecycleEpoch: Int,
        automaticAttempt: Int?
    ) -> Bool {
        let pid = health.startup?.pid.map(pid_t.init)
        return lock.withLock {
            guard launchMayProceedLocked(
                generation: generation,
                automaticAttempt: automaticAttempt
            ), self.lifecycleEpoch == lifecycleEpoch,
               process == nil, adoptedProcessID == nil, !launchInProgress
            else { return false }
            adoptedProcessID = pid
            lastOwnedLaunch = nil
            automaticRestartEligible = false
            restartStatus = .idle
            restartCount = 0
            automaticLaunchGeneration = nil
            state = .running
            return true
        }
    }

    private func waitForLaunchCompletion() async {
        let shouldWait = lock.withLock { launchInProgress }
        guard shouldWait else { return }
        await withCheckedContinuation { continuation in
            let completed = lock.withLock { () -> Bool in
                guard launchInProgress else { return true }
                launchCompletionWaiters.append(continuation)
                return false
            }
            if completed {
                continuation.resume()
            }
        }
    }

    /// Whether `start(... adoptExistingAppOwnedDaemon: true)` would adopt
    /// this health payload instead of spawning. Exposed so the store's
    /// port pre-flight can distinguish "leave it for adoption" from "move
    /// to a free port".
    public func canAdoptExisting(
        _ health: HealthPayload,
        for command: DaemonCommand,
        requireActualFanRamp: Bool
    ) -> Bool {
        canAdopt(health, for: command, requireActualFanRamp: requireActualFanRamp)
    }

    public func adoptExistingIfAppOwned(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        requireActualFanRamp: Bool = false
    ) async throws -> HealthPayload? {
        let adoption = try lock.withLock { () throws -> (generation: Int, lifecycleEpoch: Int) in
            if process != nil || adoptedProcessID != nil || launchInProgress {
                throw DaemonSupervisorError.alreadyRunning
            }
            restartGeneration &+= 1
            cancelRestartTaskLocked()
            lastOwnedLaunch = nil
            automaticRestartEligible = false
            automaticLaunchGeneration = nil
            restartStatus = .idle
            restartCount = 0
            lifecycleEpoch &+= 1
            state = .starting
            return (restartGeneration, lifecycleEpoch)
        }
        let adoptionGeneration = adoption.generation
        let adoptionLifecycleEpoch = adoption.lifecycleEpoch
        notifyStatusObserver()
        guard let existing = await initialHealthProbe(healthBaseURL, apiKey), existing.ok else {
            abortUnstartedLaunch(
                generation: adoptionGeneration,
                lifecycleEpoch: adoptionLifecycleEpoch
            )
            return nil
        }
        guard canAdopt(existing, for: command, requireActualFanRamp: requireActualFanRamp) else {
            abortUnstartedLaunch(
                generation: adoptionGeneration,
                lifecycleEpoch: adoptionLifecycleEpoch
            )
            return nil
        }
        guard adoptCurrentLaunch(
            existing,
            generation: adoptionGeneration,
            lifecycleEpoch: adoptionLifecycleEpoch,
            automaticAttempt: nil
        ) else {
            abortUnstartedLaunch(
                generation: adoptionGeneration,
                lifecycleEpoch: adoptionLifecycleEpoch
            )
            return nil
        }
        notifyStatusObserver()
        await logAdoption(existing)
        return existing
    }

    public func stop(
        graceSeconds: TimeInterval = 2,
        additionalProcessIDs: [pid_t] = []
    ) async {
        await stopInternal(
            graceSeconds: graceSeconds,
            additionalProcessIDs: additionalProcessIDs,
            clearSupervision: true
        )
    }

    private func stopInternal(
        graceSeconds: TimeInterval,
        additionalProcessIDs: [pid_t],
        clearSupervision: Bool,
        expectedProcess: Process? = nil,
        expectedLifecycleEpoch: Int? = nil
    ) async {
        let stopContext = lock.withLock { () -> (
            waitsForLaunch: Bool,
            process: Process?,
            adoptedPID: pid_t?,
            lifecycleEpoch: Int,
            launchID: String?
        )? in
            // Start A can fail its health wait after Stop A has returned and
            // Start B has already claimed this supervisor. Its cleanup must
            // never signal or clear B merely because the shared slot is live.
            guard expectedLifecycleEpoch == nil || self.lifecycleEpoch == expectedLifecycleEpoch else {
                return nil
            }
            if let expectedProcess,
               let currentProcess = process,
               currentProcess !== expectedProcess
            {
                return nil
            }
            if clearSupervision {
                restartGeneration &+= 1
                cancelRestartTaskLocked()
                lastOwnedLaunch = nil
                automaticRestartEligible = false
                automaticLaunchGeneration = nil
                restartStatus = .idle
                recentCrashDates.removeAll()
                restartCount = 0
            }
            // Keep a strong Process reference while stopping. Its termination
            // handler may clear the shared slot before this async method gets
            // to the old second lock; the retained object still gives us the
            // root PID after the launch barrier opens.
            let currentProcess = expectedProcess ?? process
            let currentAdoptedPID = adoptedProcessID
            let currentLaunchID = ownedLaunchID
            let currentLifecycleEpoch = lifecycleEpoch
            state = .stopping
            return (
                launchInProgress,
                currentProcess,
                currentAdoptedPID,
                currentLifecycleEpoch,
                currentLaunchID
            )
        }
        guard let stopContext else { return }
        notifyStatusObserver()
        // Process.run() assigns its PID synchronously. If Stop raced with the
        // published-but-not-yet-run Process, wait for that hand-off before
        // resolving the process family so Stop cannot return and leak it.
        if stopContext.waitsForLaunch {
            await waitForLaunchCompletion()
        }
        var rootPIDs: [pid_t] = []
        if let currentPID = stopContext.process?.processIdentifier {
            rootPIDs.append(currentPID)
        }
        if let adopted = stopContext.adoptedPID {
            rootPIDs.append(adopted)
        }
        rootPIDs.append(contentsOf: additionalProcessIDs)
        rootPIDs = rootPIDs.filter { $0 > 1 }
        // A daemon can exit after Stop has claimed the lifecycle but before
        // pgrep expands its descendants. Those descendants then reparent and
        // are no longer discoverable by PPID, so include the exact inherited
        // app launch token in the one-time family snapshot.
        await beforeStopProcessFamilyResolution()
        let family = Self.processFamily(
            rootPIDs: rootPIDs,
            launchID: stopContext.launchID
        )

        if !family.isEmpty {
            // The family has already been resolved, so a concurrent root
            // termination in this narrow testable gap cannot orphan a child
            // by erasing its parent relationship before expansion.
            await beforeStopProcessFamilySignal()
            Self.signal(family, SIGTERM)
            await Self.waitUntilExited(family, timeoutSeconds: graceSeconds)
            let afterTerm = family.filter(Self.pidIsAlive)
            if !afterTerm.isEmpty {
                Self.signal(afterTerm, SIGINT)
                await Self.waitUntilExited(afterTerm, timeoutSeconds: 0.5)
            }
            let afterInt = family.filter(Self.pidIsAlive)
            if !afterInt.isEmpty {
                Self.signal(afterInt, SIGKILL)
                await Self.waitUntilExited(afterInt, timeoutSeconds: 1.0)
            }
        }

        let finalized = lock.withLock { () -> Bool in
            guard lifecycleEpoch == stopContext.lifecycleEpoch else { return false }
            if let capturedProcess = stopContext.process,
               let currentProcess = process,
               currentProcess !== capturedProcess
            {
                return false
            }
            guard adoptedProcessID == nil || adoptedProcessID == stopContext.adoptedPID else {
                return false
            }
            process = nil
            adoptedProcessID = nil
            automaticLaunchGeneration = nil
            if ownedLaunchID == stopContext.launchID {
                ownedLaunchID = nil
            }
            state = .stopped
            return true
        }
        if finalized {
            notifyStatusObserver()
        }
        let pidList = family.map(String.init).joined(separator: ",")
        await logStore.append(
            pidList.isEmpty ? "daemon stopped" : "daemon process family stopped: \(pidList)",
            stream: .system
        )
    }

    /// Terminate an MTPLX daemon this supervisor does not own (e.g. a
    /// stale app-owned daemon from a previous app session holding the
    /// configured port with a different model). SIGTERM with a grace
    /// window, then SIGKILL, across the whole process family.
    public func terminateExternalDaemon(
        rootPID: pid_t,
        graceSeconds: TimeInterval = 5
    ) async {
        guard rootPID > 1 else { return }
        let family = Self.processFamily(rootPIDs: [rootPID])
        guard !family.isEmpty else { return }
        Self.signal(family, SIGTERM)
        await Self.waitUntilExited(family, timeoutSeconds: graceSeconds)
        let leftovers = family.filter(Self.pidIsAlive)
        if !leftovers.isEmpty {
            Self.signal(leftovers, SIGKILL)
            await Self.waitUntilExited(leftovers, timeoutSeconds: 1.0)
        }
        await logStore.append(
            "terminated stale MTPLX daemon pid \(rootPID)",
            stream: .system
        )
    }

    public func restart(
        command: DaemonCommand,
        healthBaseURL: URL,
        apiKey: String? = nil,
        probeHealth: Bool = true,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        adoptExistingAppOwnedDaemon: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload? {
        await stop()
        return try await start(
            command: command,
            healthBaseURL: healthBaseURL,
            apiKey: apiKey,
            probeHealth: probeHealth,
            timeoutSeconds: timeoutSeconds,
            expectedLaunchID: expectedLaunchID,
            requireActualFanRamp: requireActualFanRamp,
            adoptExistingAppOwnedDaemon: adoptExistingAppOwnedDaemon,
            onPhase: onPhase
        )
    }

    public func waitForExistingHealth(
        healthBaseURL: URL,
        apiKey: String? = nil,
        timeoutSeconds: TimeInterval = 300,
        expectedLaunchID: String? = nil,
        requireActualFanRamp: Bool = false,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)? = nil
    ) async throws -> HealthPayload {
        let health = try await waitForHealth(
            baseURL: healthBaseURL,
            apiKey: apiKey,
            timeoutSeconds: timeoutSeconds,
            expectedLaunchID: expectedLaunchID,
            requireActualFanRamp: requireActualFanRamp,
            onPhase: onPhase
        )
        lock.withLock { state = .running }
        onPhase?(.ready)
        return health
    }

    private func canAdopt(
        _ health: HealthPayload,
        for command: DaemonCommand,
        requireActualFanRamp: Bool
    ) -> Bool {
        guard health.startup?.launchId?.isEmpty == false,
              health.startup?.pid != nil
        else {
            return false
        }
        if requireActualFanRamp,
           health.thermal?.actualRampVerified != true {
            return false
        }
        if let expectedModel = expectedModelPath(from: command.arguments),
           standardizePath(health.modelPath) != standardizePath(expectedModel) {
            return false
        }
        return true
    }

    private func logAdoption(_ health: HealthPayload) async {
        await logStore.append(
            "adopted existing app-owned MTPLX daemon pid \(health.startup?.pid.map(String.init) ?? "unknown") launch \(health.startup?.launchId ?? "unknown")",
            stream: .system
        )
    }

    private func expectedModelPath(from arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: "--model"),
              arguments.indices.contains(arguments.index(after: index))
        else {
            return nil
        }
        return arguments[arguments.index(after: index)]
    }

    private func standardizePath(_ path: String) -> String {
        NSString(string: path).standardizingPath
    }

    private func launchIdentifier(from command: DaemonCommand) -> String? {
        let launchID = command.environment["MTPLX_APP_LAUNCH_ID"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return launchID?.isEmpty == false ? launchID : nil
    }

    private static func pidIsAlive(_ pid: pid_t) -> Bool {
        if kill(pid, 0) == 0 {
            return true
        }
        return errno != ESRCH
    }

    private static func processFamily(
        rootPIDs: [pid_t],
        launchID: String? = nil
    ) -> [pid_t] {
        var seen: Set<pid_t> = []
        var ordered: [pid_t] = []
        var queue = rootPIDs
        if let launchID {
            queue.append(contentsOf: processIDs(inheritingLaunchID: launchID))
        }
        while !queue.isEmpty {
            let pid = queue.removeFirst()
            guard pid > 1, !seen.contains(pid) else { continue }
            seen.insert(pid)
            ordered.append(pid)
            queue.append(contentsOf: childPIDs(of: pid))
        }
        return ordered.reversed()
    }

    private static func processIDs(inheritingLaunchID launchID: String) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        // Restrict to this user. ps only enumerates PIDs here: its textual
        // command/environment rendering has no reliable argv/env boundary.
        process.arguments = ["-x", "-o", "pid="]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            return []
        }
        let drain = SubprocessPipeDrain(output)
        guard watchdog.wait(for: process, timeout: 10) else { return [] }
        guard process.terminationStatus == 0 else { return [] }
        drain.join()
        return drain.snapshot()
            .split(whereSeparator: \.isNewline)
            .compactMap { line -> pid_t? in
                guard let pid = pid_t(
                    line.trimmingCharacters(in: .whitespacesAndNewlines)
                ),
                      processHasExactLaunchID(pid, launchID: launchID)
                else { return nil }
                return pid
            }
            .filter { $0 > 1 }
    }

    /// ps merges argv and environment into one display field, so a substring
    /// cannot establish ownership: an unrelated process could pass the token
    /// as an argument or put it inside a different environment value. Darwin's
    /// KERN_PROCARGS2 preserves the argv/environment boundary; fail closed if
    /// it cannot be read or parsed for this PID.
    private static func processHasExactLaunchID(
        _ pid: pid_t,
        launchID: String
    ) -> Bool {
        var mib: [Int32] = [CTL_KERN, KERN_PROCARGS2, pid]
        var byteCount = 0
        guard sysctl(&mib, UInt32(mib.count), nil, &byteCount, nil, 0) == 0,
              byteCount > MemoryLayout<Int32>.size
        else { return false }

        var bytes = [UInt8](repeating: 0, count: byteCount)
        guard bytes.withUnsafeMutableBytes({ buffer in
            sysctl(&mib, UInt32(mib.count), buffer.baseAddress, &byteCount, nil, 0)
        }) == 0
        else { return false }
        guard byteCount <= bytes.count else { return false }
        bytes.removeSubrange(byteCount..<bytes.count)
        guard bytes.count >= MemoryLayout<Int32>.size else { return false }

        let argc = bytes.withUnsafeBytes {
            Int($0.loadUnaligned(fromByteOffset: 0, as: Int32.self))
        }
        guard argc >= 0 else { return false }
        var cursor = MemoryLayout<Int32>.size
        guard skipCString(in: bytes, cursor: &cursor) else { return false }
        while cursor < bytes.count, bytes[cursor] == 0 {
            cursor += 1
        }
        for _ in 0..<argc {
            guard skipCString(in: bytes, cursor: &cursor) else { return false }
        }

        let expected = Array("MTPLX_APP_LAUNCH_ID=\(launchID)".utf8)
        while cursor < bytes.count {
            while cursor < bytes.count, bytes[cursor] == 0 {
                cursor += 1
            }
            guard cursor < bytes.count else { break }
            let start = cursor
            guard skipCString(in: bytes, cursor: &cursor) else { return false }
            let end = cursor - 1
            if bytes[start..<end].elementsEqual(expected) {
                return true
            }
        }
        return false
    }

    private static func skipCString(
        in bytes: [UInt8],
        cursor: inout Int
    ) -> Bool {
        guard cursor < bytes.count,
              let terminator = bytes[cursor...].firstIndex(of: 0)
        else { return false }
        cursor = terminator + 1
        return true
    }

    private static func childPIDs(of pid: pid_t) -> [pid_t] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        process.arguments = ["-P", String(pid)]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        // Bounded wait + lossless drain (#158): the PID list feeds the
        // reap path, and a wedged pgrep must degrade to "no children
        // found" (roots still get signalled) instead of hanging a
        // daemon stop/restart.
        let watchdog = SubprocessWatchdog(process)
        do {
            try process.run()
        } catch {
            return []
        }
        let drain = SubprocessPipeDrain(output)
        guard watchdog.wait(for: process, timeout: 10) else { return [] }
        guard process.terminationStatus == 0 else { return [] }
        drain.join()
        let text = drain.snapshot()
        return text
            .split(whereSeparator: \.isNewline)
            .compactMap { pid_t(String($0.trimmingCharacters(in: .whitespacesAndNewlines))) }
            .filter { $0 > 1 }
    }

    private static func signal(_ pids: [pid_t], _ signum: Int32) {
        for pid in pids where pid > 1 && pidIsAlive(pid) {
            kill(pid, signum)
        }
    }

    private static func waitUntilExited(
        _ pids: [pid_t],
        timeoutSeconds: TimeInterval
    ) async {
        let deadline = Date().addingTimeInterval(max(0, timeoutSeconds))
        while Date() < deadline {
            if !pids.contains(where: pidIsAlive) {
                return
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    private func waitForHealth(
        baseURL: URL,
        apiKey: String?,
        timeoutSeconds: TimeInterval,
        expectedLaunchID: String?,
        requireActualFanRamp: Bool,
        onPhase: (@Sendable (DaemonStartupPhase) -> Void)?
    ) async throws -> HealthPayload {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        var sawHealthyWithUnverifiedFan = false
        var healthyUnverifiedFanSince: Date?
        onPhase?(.waitingForOwnedHealth)
        while Date() < deadline {
            // A cancelled automatic-restart Task must relinquish the Process
            // through startOwned's catch/stopInternal path. Swallowing the
            // cancelled sleep below used to leave it hot-looping health probes
            // for the entire startup timeout.
            try Task.checkCancellation()
            if !isRunning() {
                let tail = await logStore.snapshot().suffix(8).map(\.message).joined(separator: " | ")
                let detail = tail.isEmpty
                    ? "daemon exited before /health became ready"
                    : "daemon exited before /health became ready: \(tail)"
                throw DaemonSupervisorError.launchFailed(detail)
            }
            if let health = await healthWaitProbe(baseURL, apiKey), health.ok {
                try Task.checkCancellation()
                if let expectedLaunchID {
                    guard health.startup?.launchId == expectedLaunchID else {
                        throw DaemonSupervisorError.launchIdentityMismatch(
                            expected: expectedLaunchID,
                            observed: health.startup?.launchId
                        )
                    }
                }
                if requireActualFanRamp,
                   health.thermal?.actualRampVerified != true {
                    // A healthy daemon is never held hostage to a fan
                    // receipt. The full health budget exists for slow model
                    // loads; inheriting it here wedged model swaps for the
                    // whole budget when ramp verification couldn't complete,
                    // then reaped a serving daemon. Give the ramp a bounded
                    // grace window and proceed — the live thermal UI shows
                    // the real fan state either way.
                    let since = healthyUnverifiedFanSince ?? Date()
                    healthyUnverifiedFanSince = since
                    if Date().timeIntervalSince(since) < fanRampGraceSeconds {
                        sawHealthyWithUnverifiedFan = true
                        onPhase?(.rampingFans)
                        try await Task.sleep(nanoseconds: 250_000_000)
                        continue
                    }
                }
                onPhase?(.warming)
                return health
            }
            try await Task.sleep(nanoseconds: 250_000_000)
        }
        if requireActualFanRamp && sawHealthyWithUnverifiedFan {
            throw DaemonSupervisorError.fanRampTimeout
        }
        throw DaemonSupervisorError.healthTimeout
    }

    private func attach(pipe: Pipe, stream: LogEntry.Stream) {
        pipe.fileHandleForReading.readabilityHandler = { [logStore] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                Task {
                    await logStore.append(String(line), stream: stream)
                }
            }
        }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
