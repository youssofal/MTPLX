import Darwin
import Foundation
import XCTest

@testable import MTPLXAppCore

// Issue #528: the header badge stuck at "Degraded — MTPLX is already
// running." over a daemon that answered /health and served requests the
// whole time, and only quitting the app cleared it. A second start request
// against the daemon the app already ran (the window's launch task running
// again after a window reopened, Start pressed while a model loaded, the
// benchmark or Hermes asking for a ready daemon) made the supervisor throw
// `alreadyRunning`; the store published that as Degraded, cancelled the
// watchdog, and waited for a launch id that never existed.
//
// These tests drive the real store and supervisor against a scripted
// `mtplx` (a small Python HTTP server, no model), the same way the app does.

/// A scripted `mtplx serve` for store lifecycle tests. It answers the
/// endpoints the store reads, reports its own pid and launch id on /health,
/// appends every spawn to `spawns.log`, and takes switches through files so a
/// test can make /health or the live-stats stream fail.
struct ReconnectFakeDaemon {
    let root: URL
    let executable: URL
    let modelDirectory: URL
    let port: Int

    struct Spawn: Equatable {
        let pid: Int
        let launchID: String?
    }

    static func make() throws -> ReconnectFakeDaemon {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-reconnect-\(UUID().uuidString)", isDirectory: true)
        let bin = root.appendingPathComponent("bin", isDirectory: true)
        let model = root.appendingPathComponent("complete-model", isDirectory: true)
        let runtime = root.appendingPathComponent("runtime", isDirectory: true)
        for directory in [bin, model, runtime] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        for file in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try Data("{}".utf8).write(to: model.appendingPathComponent(file))
        }
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))

        let server = bin.appendingPathComponent("fake_mtplx.py")
        try Data(serverSource(controlDirectory: root).utf8).write(to: server)
        let executable = bin.appendingPathComponent("mtplx")
        try writeExecutable(
            "#!/bin/sh\nexec python3 -u \(shellQuoted(server.path)) \"$@\"\n",
            to: executable
        )
        // What the launch path's runtime check resolves; it only asks for a
        // version and never starts this.
        try writeExecutable(
            "#!/bin/sh\necho 'mtplx 2.12.0 (2.12.0)'\n",
            to: runtime.appendingPathComponent("mtplx")
        )
        return ReconnectFakeDaemon(
            root: root,
            executable: executable,
            modelDirectory: model,
            port: try freeTCPPort()
        )
    }

    var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    func spawns() -> [Spawn] {
        let text = (try? String(contentsOf: root.appendingPathComponent("spawns.log"), encoding: .utf8)) ?? ""
        return text.split(separator: "\n").compactMap { line in
            let fields = line.split(separator: " ")
            guard fields.count == 2, let pid = Int(fields[0]) else { return nil }
            return Spawn(pid: pid, launchID: fields[1] == "-" ? nil : String(fields[1]))
        }
    }

    /// /health answers 503 while set: a daemon still loading its model, or
    /// one busy in a long commit (issue #487).
    func setHealthDown(_ down: Bool) throws {
        try setFlag("health-down", down)
    }

    /// The live-stats stream closes and refuses new connections while set:
    /// what the app sees across a sleep or a network blip.
    func setStreamDown(_ down: Bool) throws {
        try setFlag("stream-down", down)
    }

    /// A daemon started while set exits with status 3 before it binds: a
    /// start that genuinely fails.
    func setExitOnStart(_ exit: Bool) throws {
        try setFlag("exit-on-start", exit)
    }

    /// A daemon the app did not launch in this session: an app-owned one
    /// from an earlier session (with a launch id) or `mtplx serve` typed in
    /// a terminal (without one).
    /// With `apiKey`, the daemon requires that key on every request (a
    /// server whose key the app does not have answers 401). It serves the
    /// fixture's model unless `model` names another folder.
    func launchOutsideTheApp(launchID: String?, apiKey: String? = nil, model: URL? = nil) throws -> Process {
        let process = Process()
        process.executableURL = executable
        var arguments = ["serve", "--port", String(port), "--model", (model ?? modelDirectory).path]
        if let apiKey {
            let keyFile = root.appendingPathComponent("outside-key-\(UUID().uuidString)")
            try Data(apiKey.utf8).write(to: keyFile)
            arguments += ["--api-key-file", keyFile.path]
        }
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "MTPLX_APP_LAUNCH_ID")
        if let launchID {
            environment["MTPLX_APP_LAUNCH_ID"] = launchID
        }
        process.environment = environment
        try process.run()
        return process
    }

    /// Waits until something answers HTTP on the port, whatever it says.
    func waitUntilAnswering(timeout: TimeInterval = 10) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        let url = baseURL.appendingPathComponent("health")
        while Date() < deadline {
            if let (_, response) = try? await URLSession.shared.data(from: url),
               response is HTTPURLResponse {
                return
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw DaemonSupervisorError.healthTimeout
    }

    /// /health reports this pid instead of the daemon's own, or its own
    /// again with `nil`.
    func setReportedPID(_ pid: pid_t?) throws {
        let url = root.appendingPathComponent("report-pid")
        if let pid {
            try Data(String(pid).utf8).write(to: url)
        } else if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    /// /health answers 200 with a body that is not a health payload.
    func setHealthUndecodable(_ undecodable: Bool) throws {
        try setFlag("health-undecodable", undecodable)
    }

    func waitUntilHealthy(timeout: TimeInterval = 10, apiKey: String? = nil) async throws -> HealthPayload {
        let client = MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let health = try? await client.health(), health.ok {
                return health
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw DaemonSupervisorError.healthTimeout
    }

    /// The settings a store under test launches with. Onboarding is done
    /// and launch-on-open is on, as in the report.
    func configuration(fanMode: MTPLXFanMode) -> MTPLXAppConfiguration {
        MTPLXAppConfiguration(
            executablePath: executable.path,
            model: modelDirectory.path,
            port: port,
            launchDaemonOnOpen: true,
            fanMode: fanMode.rawValue,
            pinFansAtMaxOnStart: fanMode == .max,
            lastLaunchTarget: LaunchTarget.chat.rawValue,
            onboardingCompletedAt: Date(),
            customModels: [
                MTPLXModelOption(
                    id: "complete-model",
                    displayName: "Complete Model",
                    shortName: "Complete",
                    detail: "Fixture model",
                    hfModelID: "Example/CompleteModel",
                    localCandidates: [modelDirectory.path]
                )
            ]
        )
    }

    var settingsStore: MTPLXSettingsStore {
        MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
    }

    /// A store wired like the app's, with every side effect outside this
    /// fixture replaced: fans are recorded, never driven, the runtime
    /// check resolves the fixture's own `mtplx` without touching the
    /// network, and OpenCode's config lives under the fixture.
    @MainActor
    func makeStore(
        configuration: MTPLXAppConfiguration,
        supervisor: DaemonSupervisor = DaemonSupervisor(),
        fans: FanCallRecorder,
        openCodeConfigURL: URL? = nil,
        beforeStaticStateAnswersAreRead: (@Sendable () async -> Void)? = nil
    ) -> MTPLXBackendStore {
        // HOME inside the fixture: a failed launch writes the runtime
        // import-recheck marker under the builder's Application Support.
        let environment = [
            "PATH": root.appendingPathComponent("runtime").path + ":/usr/bin:/bin",
            "HOME": root.path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
        ]
        let store = MTPLXBackendStore(
            configuration: configuration,
            settingsStore: settingsStore,
            commandBuilder: MTPLXCommandBuilder(environment: environment),
            supervisor: supervisor,
            openCodeIntegration: OpenCodeIntegration(
                configURL: openCodeConfigURL
                    ?? root.appendingPathComponent("opencode/opencode.json"),
                desktopSettingsStoreURL: root.appendingPathComponent("opencode/desktop-settings.json")
            ),
            runtimeUpdateService: MTPLXRuntimeUpdateService(
                manifestURL: URL(string: "http://127.0.0.1:1/releases/latest.json")!,
                environment: environment
            ),
            localFanRestorer: { await fans.restore() },
            fanModeSetter: { _, mode, _, _ in
                await fans.set(mode)
                return FanModeResponse(verified: true, currentMode: mode)
            },
            beforeStaticStateAnswersAreRead: beforeStaticStateAnswersAreRead
        )
        // Before it loads a model the app looks for it on the usual MTPLX
        // ports of this Mac; no test reaches those. A test that needs another
        // port sets its own list.
        store.runningServerProbePortsOverride = []
        return store
    }

    private func setFlag(_ name: String, _ on: Bool) throws {
        let url = root.appendingPathComponent(name)
        if on {
            try Data().write(to: url)
        } else if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    private static func writeExecutable(_ text: String, to url: URL) throws {
        try Data(text.utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        defer { Darwin.close(socketFD) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var bound = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &length)
            }
        }
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: bound.sin_port))
    }

    private static func serverSource(controlDirectory: URL) -> String {
        #"""
        import json
        import os
        import signal
        import sys
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        if "--version" in sys.argv:
            print("mtplx 2.12.0 (2.12.0)")
            sys.exit(0)

        def arg(name, default=None):
            if name in sys.argv:
                index = sys.argv.index(name)
                if index + 1 < len(sys.argv):
                    return sys.argv[index + 1]
            return default

        CONTROL = r'''__CONTROL__'''
        PORT = int(arg("--port", "0"))
        MODEL = arg("--model", "/models/test")
        FAN_MODE = arg("--fan-mode", "default")
        LAUNCH_ID = arg("--app-launch-id") or os.environ.get("MTPLX_APP_LAUNCH_ID") or None
        PID = os.getpid()
        PARENT = os.getppid()
        # Like `mtplx serve`: the key is read once at startup from the file
        # the app hands over, and every request must carry it.
        API_KEY = None
        if arg("--api-key-file"):
            with open(arg("--api-key-file"), encoding="utf-8") as key_file:
                API_KEY = key_file.read().strip() or None

        def control(name):
            return os.path.join(CONTROL, name)

        def flag(name):
            return os.path.exists(control(name))

        with open(control("spawns.log"), "a", encoding="utf-8") as handle:
            handle.write("%d %s\n" % (PID, LAUNCH_ID or "-"))

        if flag("exit-on-start"):
            sys.stderr.write("fake mtplx: exiting before it binds\n")
            sys.exit(3)

        def guard():
            # Never outlive the test process, and never run for long.
            deadline = time.time() + 180
            while True:
                if os.getppid() != PARENT or time.time() > deadline:
                    os._exit(0)
                time.sleep(0.5)

        threading.Thread(target=guard, daemon=True).start()
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        signal.signal(signal.SIGINT, lambda *_: os._exit(0))

        HEALTH = {
            "ok": True,
            "model": "mtplx-test-model",
            "model_path": MODEL,
            "generation_mode": "mtp",
            "load_mtp": True,
            "mtp_enabled": True,
            "depth": 3,
            "profile": {"name": "sustained"},
            "context_window": 4096,
            "max_response_tokens": 1024,
            "active_requests": 0,
            "reasoning_parser": "qwen3",
            "chip": "Apple M5 Max",
            "machine_model": "Mac16,1",
            "unified_memory_bytes": 137438953472,
            "startup": {
                "launch_id": LAUNCH_ID,
                "pid": PID,
                "started_at": 1.0,
                "model_id": "mtplx-test-model",
                "warmup": {"ok": True},
            },
            "thermal": {
                "max_requested": True,
                "max_verified": True,
                "actual_ramp_verified": True,
                "fan_summary": {"ok": True},
                "verified_at": "1.0",
                "verified": {"ok": True},
            },
        }

        SETTINGS = {"depth": 3, "temperature": 0.6, "top_p": 0.95, "top_k": 20}

        SNAPSHOT = {
            "ts": 1.0,
            "model_id": "mtplx-test-model",
            "profile": {"name": "sustained"},
            "context_window": 4096,
            "active_requests": 0,
            "in_flight": [],
            "latest": {"decode_tok_s": 55.0, "session_id": "s1"},
            "recent": [],
            "rolling": {
                "window_s": 300.0, "count": 1, "min": 55.0, "max": 55.0,
                "mean": 55.0, "p50": 55.0, "p95": 55.0,
                "history": [{"t": 1.0, "tok_s": 55.0, "session_id": "s1"}],
                "live_history": [], "sticky_all_time_max": 55.0,
            },
            "lifetime": {
                "started_at_s": 1.0, "uptime_s": 2.0, "prompt_tokens_total": 10,
                "completion_tokens_total": 20, "cached_tokens_total": 5,
                "tokens_total": 30, "requests_total": 1, "cancelled_total": 0,
            },
            "sessions": {"sessions": [], "count": 0, "session_bank": {"prefixes": []}},
            "session_bank": {"prefixes": []},
            "mem": {"ok": True},
            "thermal": None,
            "thermal_when_s": 0.0,
            "settings": SETTINGS,
            "machine": {"chip": "Apple M5 Max", "machine_model": "Mac16,1", "unified_memory_bytes": 137438953472},
            "uptime_s": 2.0,
        }

        CAPABILITIES = {
            "ok": True,
            "name": "MTPLX test daemon",
            "api_version": 1,
            "endpoints": {},
            "mutable_settings": [],
            "restart_required_settings": [],
            "snapshot_interval": {
                "default_ms": 500, "min_ms": 250, "max_ms": 5000,
                "native_default_ms": 500, "performance_lock_ms": 1000,
            },
            "features": {},
            "scheduler": {},
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def _json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _empty(self, status):
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _authorized(self):
                return API_KEY is None or self.headers.get("Authorization") == "Bearer " + API_KEY

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if not self._authorized():
                    return self._empty(401)
                if path == "/health":
                    if flag("health-down"):
                        return self._empty(503)
                    if flag("health-undecodable"):
                        body = b"<html>not an MTPLX health payload</html>"
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    payload = dict(HEALTH)
                    if os.path.exists(control("report-pid")):
                        # The pid this daemon claims on /health: a stand-in
                        # for a pid the kernel has since given to another
                        # process.
                        with open(control("report-pid"), encoding="utf-8") as handle:
                            payload["startup"] = dict(HEALTH["startup"], pid=int(handle.read().strip()))
                    return self._json(payload)
                if path == "/v1/mtplx/metrics/stream":
                    if flag("stream-down"):
                        return self._empty(503)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    frame = ("event: snapshot\ndata: " + json.dumps(SNAPSHOT) + "\n\n").encode("utf-8")
                    try:
                        while not flag("stream-down"):
                            self.wfile.write(frame)
                            self.wfile.flush()
                            time.sleep(0.2)
                    except OSError:
                        pass
                    return
                if path == "/v1/mtplx/app/capabilities":
                    return self._json(CAPABILITIES)
                if path == "/admin/sessions":
                    return self._json({"sessions": [], "count": 0})
                if path == "/v1/mtplx/prefill_history":
                    return self._json({"capacity": 0, "history": []})
                if path == "/v1/models":
                    return self._json({
                        "object": "list",
                        "data": [{"id": "mtplx-test-model", "object": "model", "owned_by": "mtplx"}],
                    })
                if path == "/v1/mtplx/thermal/status":
                    return self._json({
                        "ok": True,
                        "current_mode": FAN_MODE,
                        "detection": {"available": True},
                        "fan_summary": {"ok": True},
                    })
                if path == "/v1/mtplx/snapshot":
                    return self._json(SNAPSHOT)
                if path == "/v1/mtplx/settings":
                    return self._json(SETTINGS)
                return self._empty(404)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                if not self._authorized():
                    return self._empty(401)
                if path == "/v1/mtplx/settings":
                    return self._json(SETTINGS)
                return self._empty(404)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        Server(("127.0.0.1", PORT), Handler).serve_forever()
        """#.replacingOccurrences(of: "__CONTROL__", with: controlDirectory.path)
    }
}

/// Records the fan side effects a store would have caused. Nothing here
/// drives real fans.
actor FanCallRecorder {
    private(set) var restores = 0
    private(set) var modes: [String] = []

    func restore() -> Bool {
        restores += 1
        return true
    }

    func set(_ mode: String) {
        modes.append(mode)
    }
}

/// A TCP listener that takes connections into its backlog and never
/// answers. To the port preflight it is another app on the port, which it
/// re-probes for the whole settle window.
final class SilentListener {
    private let descriptor: Int32

    init(port: Int) throws {
        descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port)).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, Darwin.listen(descriptor, 16) == 0 else {
            let failure = POSIXError(POSIXErrorCode(rawValue: errno) ?? .EADDRINUSE)
            Darwin.close(descriptor)
            throw failure
        }
    }

    func close() {
        Darwin.close(descriptor)
    }
}

/// Runs an action set by the test at most once, the first time `run` is
/// called after it was set; before that, `run` does nothing.
actor OneShotAction {
    private var action: (@MainActor @Sendable () async -> Void)?

    func set(_ action: @escaping @MainActor @Sendable () async -> Void) {
        self.action = action
    }

    func run() async {
        guard let action else { return }
        self.action = nil
        await action()
    }
}

/// Holds the first call through it after it is armed until the test opens
/// it; every other call passes straight through. Put in front of the
/// supervisor's first /health probe, it suspends a start at its first
/// await, where a Stop can overtake it.
actor FirstCallGate {
    private var armed: Bool
    private var used = false
    private var entered = false
    private var opened = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var release: CheckedContinuation<Void, Never>?

    init(armed: Bool = true) {
        self.armed = armed
    }

    func arm() {
        armed = true
    }

    func pass() async {
        guard armed, !used else { return }
        used = true
        entered = true
        enteredWaiters.forEach { $0.resume() }
        enteredWaiters.removeAll()
        guard !opened else { return }
        await withCheckedContinuation { release = $0 }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func open() {
        opened = true
        release?.resume()
        release = nil
    }
}

final class DaemonReconnectTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    /// The daemon half of MTPLXApp.swift's window `.task` for a user who
    /// finished onboarding. SwiftUI runs it again whenever a new main window
    /// appears, which is one way a second start reaches a running daemon.
    @MainActor
    private func runWindowLaunchTask(_ store: MTPLXBackendStore) async {
        store.loadPersistedSettings()
        if store.configuration.launchDaemonOnOpen {
            await store.startDaemon()
        } else {
            await store.attachExistingDaemonIfOwned()
        }
    }

    @MainActor
    private func badge(_ store: MTPLXBackendStore) -> DaemonStatusBadge {
        DaemonStatusBadge(daemonState: store.daemonState, connectionState: store.connectionState)
    }

    /// The Logs window is what a user pastes into a report; it must say why
    /// a start request did not launch anything.
    @MainActor
    private func logsMention(_ store: MTPLXBackendStore, _ needle: String) async -> Bool {
        for _ in 0..<50 {
            await store.refreshLogs()
            if store.logs.contains(where: { $0.message.contains(needle) }) { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return false
    }

    // MARK: A second start against the app's own daemon

    @MainActor
    func testSecondStartAgainstItsOwnDaemonStaysRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .max), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)
        let restoresBefore = await fans.restores

        // The window's launch task runs again, or Start is pressed.
        await store.startDaemon()

        XCTAssertEqual(
            store.daemonState, .running,
            "a second start against the app's own healthy daemon must not degrade"
        )
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(store.health?.startup?.pid, launched.pid)
        XCTAssertEqual(daemon.spawns().count, 1, "a second start must not launch a second daemon")
        let restoresAfter = await fans.restores
        XCTAssertEqual(restoresAfter, restoresBefore, "the running daemon's max fans must not be reset")
        try await pollUntil("live stats still open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
        XCTAssertEqual(badge(store).tone, .healthy)
        let logged = await logsMention(store, "reconnecting to it instead of launching")
        XCTAssertTrue(logged, "the Logs window records why no second launch happened")
    }

    // MARK: A second start against an adopted daemon

    @MainActor
    func testSecondLaunchTaskAgainstAnAdoptedDaemonStaysRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        // The settings file on disk is what the window task loads (report 2
        // edited it while the app was closed).
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await runWindowLaunchTask(store)
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId, "adopted, not relaunched")
        try await pollUntil("live stats open") { store.connectionState == .open }

        // A window reopens: the same task runs again.
        await runWindowLaunchTask(store)

        XCTAssertEqual(
            store.daemonState, .running,
            "a second launch task against an adopted healthy daemon must not degrade"
        )
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.pid, priorHealth.startup?.pid)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId)
        XCTAssertEqual(daemon.spawns().count, 1, "the adopted daemon is the only one")
        XCTAssertTrue(prior.isRunning)
        let restores = await fans.restores
        XCTAssertEqual(restores, 0)
        try await pollUntil("live stats still open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: A server that is not ours on the port

    /// The app's daemon is gone and a stranger answers on its port. A
    /// healthy answer from the stranger must never read as this app's engine
    /// running. The app lets its lost daemon go and stays Degraded naming
    /// the other server, rather than Stopped, which hid the cause. Stop
    /// never signals the stranger and the configured port is not moved.
    @MainActor
    func testAnotherServerOnThePortIsNamedAndNeverSignalled() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        _ = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)

        // The adopted daemon goes away and `mtplx serve`, started in a
        // terminal, takes its port.
        prior.terminate()
        prior.waitUntilExit()
        let stranger = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if stranger.isRunning { stranger.terminate() } }
        let strangerHealth = try await daemon.waitUntilHealthy()
        XCTAssertNil(strangerHealth.startup?.launchId)

        await store.refresh()
        await store.awaitDaemonTeardown()
        // The supervisor's terminal snapshot reaches the store on its own hop.
        try await pollUntil("the lost daemon is released") {
            !supervisor.isRunning() && store.daemonState != .stopping
        }

        let strangerPID = try XCTUnwrap(strangerHealth.startup?.pid)
        let expectedReason =
            "Another MTPLX server holds port \(daemon.port) (pid \(strangerPID)); this app is not connected to it."
        XCTAssertEqual(
            store.daemonState, .degraded(expectedReason),
            "another server on the port must stay visible as Degraded, naming it"
        )
        let named = badge(store)
        XCTAssertEqual(named.tone, .failed)
        XCTAssertTrue(named.label.hasPrefix("Degraded — Another MTPLX server"), named.label)
        XCTAssertEqual(named.help, "Degraded: \(expectedReason)")
        XCTAssertNotEqual(
            store.health?.startup?.pid, strangerPID,
            "a stranger's pid must never become the app's, where Stop would signal it"
        )
        XCTAssertFalse(supervisor.isRunning(), "the app no longer claims a daemon it lost")

        // Stop runs whatever the checks above found.
        await store.stopDaemon()
        XCTAssertTrue(stranger.isRunning, "Stop never signals the other server")
        let stillThere = try? await daemon.waitUntilHealthy(timeout: 3)
        XCTAssertEqual(stillThere?.startup?.pid, strangerPID, "the other server still answers after Stop")
        XCTAssertEqual(store.configuration.port, daemon.port)
    }

    /// An adopted daemon exits and another server takes its port. The live
    /// stats stream ends cleanly and reopens on the new server, and the
    /// watchdog keeps probing. Neither may let the other server's answer
    /// stand for the app's daemon (on 1de2b1c0 the watchdog published it as
    /// `health`, and Stop then signalled its pid).
    @MainActor
    private func assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: String?) async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: MTPLXAppConfiguration(),
            supervisor: supervisor,
            fans: FanCallRecorder()
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId, "adopted")
        try await pollUntil("live stats open") { store.connectionState == .open }

        prior.terminate()
        prior.waitUntilExit()
        let stranger = try daemon.launchOutsideTheApp(launchID: strangerLaunchID)
        addTeardownBlock { if stranger.isRunning { stranger.terminate() } }
        let strangerHealth = try await daemon.waitUntilHealthy()
        let strangerPID = try XCTUnwrap(strangerHealth.startup?.pid)
        XCTAssertEqual(strangerHealth.startup?.launchId, strangerLaunchID)

        try await pollUntil("the other server is named", timeout: 15) {
            if case .degraded(let reason) = store.daemonState {
                return reason.hasPrefix("Another MTPLX server holds port \(daemon.port)")
            }
            return false
        }
        // More watchdog rounds (every 3 s) before Stop: nothing may publish
        // the other server's answer as the app's daemon meanwhile.
        try await Task.sleep(nanoseconds: 4_000_000_000)
        XCTAssertEqual(
            store.daemonState,
            .degraded("Another MTPLX server holds port \(daemon.port) (pid \(strangerPID)); this app is not connected to it.")
        )
        XCTAssertNotEqual(store.health?.startup?.pid, strangerPID, "the other server's pid never becomes `health`")
        XCTAssertTrue(badge(store).label.hasPrefix("Degraded — Another MTPLX server"), badge(store).label)
        XCTAssertFalse(supervisor.isRunning(), "the app let go of the daemon it lost")

        // Stop runs whatever the checks above found.
        await store.stopDaemon()
        XCTAssertTrue(stranger.isRunning, "Stop must never signal another server")
        let stillThere = try? await daemon.waitUntilHealthy(timeout: 3)
        XCTAssertEqual(stillThere?.startup?.pid, strangerPID, "the other server still answers after Stop")
        XCTAssertEqual(daemon.spawns().count, 2, "the prior daemon and the stranger; the app launched nothing")
    }

    @MainActor
    func testAnotherServerWithoutALaunchIDAfterACleanStreamEnd() async throws {
        try await assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: nil)
    }

    @MainActor
    func testAnotherServerWithADifferentLaunchIDAfterACleanStreamEnd() async throws {
        try await assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: "other-session-\(UUID().uuidString)")
    }

    // MARK: A launch on a fallback port, then the window reopens

    /// The configured port is taken by `mtplx serve` from a terminal serving
    /// another model, so the app launches on the next free port and settings
    /// keep the configured one (#503). Reopening the window reloads settings
    /// from disk. Every later request must still reach the daemon where it
    /// listens, not the reloaded port, where the other server answers.
    @MainActor
    func testFallbackPortDaemonStaysReachableAfterTheWindowReopens() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let occupant = try daemon.launchOutsideTheApp(
            launchID: nil,
            model: daemon.root.appendingPathComponent("another-model", isDirectory: true)
        )
        addTeardownBlock { if occupant.isRunning { occupant.terminate() } }
        let occupantHealth = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        let fallbackPort = try XCTUnwrap(store.baseURL.port)
        XCTAssertNotEqual(fallbackPort, daemon.port, "the launch moved off the occupied port")
        XCTAssertNotNil(store.portFallbackNotice)
        let launched = try XCTUnwrap(store.health?.startup)
        try await pollUntil("live stats open") { store.connectionState == .open }

        // The window reopens: its launch task reloads settings and starts.
        store.loadPersistedSettings()
        XCTAssertEqual(store.configuration.port, daemon.port, "settings keep the configured port")
        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.baseURL.port, fallbackPort)
        XCTAssertEqual(
            store.apiClient.baseURL.port, fallbackPort,
            "requests go where the daemon listens, not to the reloaded port"
        )
        let answer = try await store.apiClient.health()
        XCTAssertEqual(answer.startup?.launchId, launched.launchId, "the app's daemon answers, not the occupant")
        XCTAssertNotEqual(answer.startup?.pid, occupantHealth.startup?.pid)
        XCTAssertEqual(daemon.spawns().count, 2, "the occupant and one app daemon")
        XCTAssertNotNil(store.portFallbackNotice, "the fallback banner stays while that daemon runs")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")

        await store.stopDaemon()
        XCTAssertTrue(occupant.isRunning, "Stop never signals the server on the configured port")
    }

    // MARK: A server started outside the app already serves this model

    /// `mtplx serve` from a terminal already serves the model when Play is
    /// pressed. The app used to move to the next free port and load a second
    /// copy (two Flash-Next loads do not fit in 128 GB). It now shows that
    /// server: Live works, nothing is loaded, its fans are left alone, and
    /// Stop lets go of it without signalling it and says it keeps running.
    @MainActor
    func testPlayBesideATerminalServerOnTheSameModelAttachesInsteadOfLoading() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let terminalServer = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        let terminalHealth = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .max), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(daemon.spawns().count, 1, "no second model was loaded")
        XCTAssertTrue(store.attachedToOutsideServer)
        XCTAssertEqual(store.health?.startup?.pid, terminalHealth.startup?.pid)
        XCTAssertEqual(store.baseURL.port, daemon.port)
        XCTAssertNil(store.portFallbackNotice, "the port did not move")
        let notice = try XCTUnwrap(store.outsideServerNotice)
        XCTAssertTrue(notice.contains("already serves this model"), notice)
        XCTAssertNil(store.lastLaunchScheduling, "Settings must not claim the scheduler of a server the app did not start")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
        let modesAfterStart = await fans.modes
        XCTAssertEqual(modesAfterStart, [], "the app drives no fans of a server it did not start")

        // Play again: the app reconnects to what it holds.
        await store.startDaemon()
        XCTAssertEqual(daemon.spawns().count, 1)
        XCTAssertEqual(store.daemonState, .running)

        await store.stopDaemon()
        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertTrue(terminalServer.isRunning, "Stop lets go of the server and never signals it")
        let stillServing = try await daemon.waitUntilHealthy(timeout: 3)
        XCTAssertEqual(stillServing.startup?.pid, terminalHealth.startup?.pid)
        XCTAssertFalse(store.attachedToOutsideServer)
        let afterStop = try XCTUnwrap(store.outsideServerNotice)
        XCTAssertTrue(afterStop.contains("keeps running"), afterStop)
        XCTAssertTrue(afterStop.contains("mtplx stop --port \(daemon.port)"), afterStop)
        let restores = await fans.restores
        let modes = await fans.modes
        XCTAssertEqual(restores, 0, "Stop restores no fans the app never set")
        XCTAssertEqual(modes, [])
        let logged = await logsMention(store, "letting go of the MTPLX server")
        XCTAssertTrue(logged, "the Logs window says Stop let go of it")
    }

    /// `mtplx start opencode` left its server on one of the usual MTPLX
    /// ports while the app's port is another: the app finds it there, as the
    /// CLI does, and uses it for this session; settings keep their port.
    @MainActor
    func testPlayFindsTheModelServedOnAnotherUsualPort() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let configuredPort = try ReconnectFakeDaemon.make().port
        let terminalServer = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        let terminalHealth = try await daemon.waitUntilHealthy()
        var configuration = daemon.configuration(fanMode: .default)
        configuration.port = configuredPort
        try daemon.settingsStore.save(configuration)
        let store = daemon.makeStore(configuration: configuration, fans: FanCallRecorder())
        store.runningServerProbePortsOverride = [daemon.port]
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(daemon.spawns().count, 1, "no second model was loaded")
        XCTAssertEqual(store.baseURL.port, daemon.port)
        XCTAssertEqual(store.health?.startup?.pid, terminalHealth.startup?.pid)
        XCTAssertEqual(try daemon.settingsStore.load().port, configuredPort, "settings keep their port")
        try await pollUntil("live stats open") { store.connectionState == .open }

        await store.stopDaemon()
        XCTAssertTrue(terminalServer.isRunning)
    }

    /// The app's own daemon from an earlier session holds the configured
    /// port with another model, and a terminal server already serves this
    /// model on another usual port: the stale daemon is replaced as before,
    /// and Play attaches to the terminal server instead of loading the model
    /// (it used to attach and leave the stale daemon running unseen).
    @MainActor
    func testPlayReplacesAStaleOwnDaemonAndAttachesWhereTheModelIsServed() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let other = try ReconnectFakeDaemon.make()
        let stale = try daemon.launchOutsideTheApp(
            launchID: "prior-session-\(UUID().uuidString)",
            model: daemon.root.appendingPathComponent("another-model", isDirectory: true)
        )
        addTeardownBlock { if stale.isRunning { stale.terminate() } }
        _ = try await daemon.waitUntilHealthy()
        let terminalServer = try other.launchOutsideTheApp(launchID: nil, model: daemon.modelDirectory)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        let terminalHealth = try await other.waitUntilHealthy()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        store.runningServerProbePortsOverride = [other.port]
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertTrue(store.attachedToOutsideServer)
        XCTAssertEqual(store.baseURL.port, other.port)
        XCTAssertEqual(store.health?.startup?.pid, terminalHealth.startup?.pid)
        try await pollUntil("the stale daemon is gone") { !stale.isRunning }
        XCTAssertEqual(daemon.spawns().count, 1, "only the stale daemon ever ran on the configured port")
        XCTAssertEqual(other.spawns().count, 1, "nothing was loaded beside the terminal server")
    }

    /// The server that serves the model cannot be held (it stopped answering
    /// between the port check and the attach). The app says who holds the
    /// port, with the message it always used, and loads nothing: no second
    /// copy, no port move, no fan change.
    @MainActor
    func testAnAttachThatCannotHoldTheServerRefusesAndLoadsNothing() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let terminalServer = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        _ = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        let supervisor = DaemonSupervisor(initialHealthProbe: { _, _ in nil })
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .max),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()

        let expected = "Port \(daemon.port) is held by an MTPLX server started outside the app. "
            + "Press Ctrl-C in its terminal or run `mtplx stop --port \(daemon.port)`, then press Play."
        XCTAssertEqual(store.daemonState, .degraded(expected))
        XCTAssertEqual(store.startupPhase, .failed(expected))
        XCTAssertEqual(daemon.spawns().count, 1, "nothing was loaded beside it")
        XCTAssertEqual(store.configuration.port, daemon.port, "no move to another port")
        XCTAssertNil(store.portFallbackNotice)
        XCTAssertFalse(store.attachedToOutsideServer)
        XCTAssertFalse(supervisor.isRunning())
        let restores = await fans.restores
        let modes = await fans.modes
        XCTAssertEqual(restores, 0)
        XCTAssertEqual(modes, [])
        XCTAssertTrue(terminalServer.isRunning)
    }

    /// A launch setting changed while the app shows a server it did not
    /// start: restarting would stop a server that is not the app's and load
    /// the model beside it. The change is saved and the server keeps serving.
    @MainActor
    func testAChangeThatNeedsARestartLeavesAnAttachedServerAlone() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let terminalServer = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        let terminalHealth = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertTrue(store.attachedToOutsideServer)

        var next = store.configuration
        next.automaticDaemonRestart.toggle()
        try await store.applyConfiguration(next)

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(daemon.spawns().count, 1, "nothing was loaded beside it")
        XCTAssertEqual(store.health?.startup?.pid, terminalHealth.startup?.pid)
        let failure = try XCTUnwrap(store.configurationChangeFailure)
        XCTAssertTrue(failure.contains("started outside the app"), failure)
        XCTAssertEqual(try daemon.settingsStore.load().automaticDaemonRestart, next.automaticDaemonRestart)
        XCTAssertTrue(terminalServer.isRunning)
    }

    func testTheSameModelFolderMatchesHoweverItIsSpelled() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-same-model-\(UUID().uuidString)", isDirectory: true)
        let folder = root.appendingPathComponent("pack", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let link = root.appendingPathComponent("linked-pack")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: folder)
        defer { try? FileManager.default.removeItem(at: root) }
        func health(_ path: String, ok: Bool = true) throws -> HealthPayload {
            try JSONDecoder().decode(
                HealthPayload.self,
                from: Data(#"""
                {"ok": \#(ok), "model": "m", "model_path": "\#(path)", "generation_mode": "mtp",
                 "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
                 "context_window": 4096, "active_requests": 0, "reasoning_parser": "qwen3"}
                """#.utf8)
            )
        }
        XCTAssertTrue(MTPLXBackendStore.servesSameModel(try health(folder.path), expectedModelPath: link.path))
        XCTAssertTrue(MTPLXBackendStore.servesSameModel(try health(folder.path + "/"), expectedModelPath: folder.path))
        XCTAssertFalse(MTPLXBackendStore.servesSameModel(try health(folder.path), expectedModelPath: root.path))
        XCTAssertFalse(MTPLXBackendStore.servesSameModel(try health(folder.path, ok: false), expectedModelPath: folder.path))
        XCTAssertFalse(MTPLXBackendStore.servesSameModel(try health(""), expectedModelPath: folder.path))
    }

    // MARK: Live stats drop while the daemon stays healthy

    @MainActor
    func testLiveStatsDropWithHealthyDaemonShowsLiveChannelStateNotDegraded() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        // The Mac sleeps: the stream drops while /health stays healthy.
        try daemon.setStreamDown(true)
        try await pollUntil("live stats reconnecting") {
            if case .reconnecting = store.connectionState { return true }
            return false
        }
        // On wake the window's launch task runs again.
        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running, "a dropped stats stream is not an engine failure")
        let dropped = badge(store)
        XCTAssertEqual(dropped.tone, .pending)
        XCTAssertTrue(dropped.label.hasPrefix("Running · "), "badge read \(dropped.label)")
        XCTAssertEqual(daemon.spawns().count, 1)

        // The stream comes back: Running again, same daemon, no restart.
        try daemon.setStreamDown(false)
        try await pollUntil("live stats open again", timeout: 15) { store.connectionState == .open }
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(daemon.spawns().count, 1)
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: A transient /health failure

    @MainActor
    func testTransientHealthFailureRecoversWithoutRestart() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        // /health stops answering for a few seconds (a long commit, #487)
        // just as the benchmark asks for a ready daemon.
        try daemon.setHealthDown(true)
        let answersAgain = Task {
            try? await Task.sleep(nanoseconds: 6_000_000_000)
            try? daemon.setHealthDown(false)
        }
        do {
            _ = try await store.ensureDaemonReadyForBenchmark()
            XCTFail("a daemon that did not answer /health was reported ready")
        } catch BenchmarkDaemonReadinessError.startupFailed(let reason) {
            XCTFail("a busy daemon was reported as a failed start: \(reason)")
        } catch {
            // Unreachable for now: true, it did not answer this time.
        }
        XCTAssertEqual(store.daemonState, .running, "a daemon that is slow to answer is busy, not degraded")

        // It answers again and the user presses Refresh.
        answersAgain.cancel()
        try daemon.setHealthDown(false)
        await store.refresh()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(store.health?.startup?.pid, launched.pid)
        XCTAssertEqual(daemon.spawns().count, 1, "recovered without a restart")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).tone, .healthy)
    }

    // MARK: Two starts at once

    @MainActor
    func testStartWhileAStartIsLoadingJoinsIt() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        // The model is still loading: the port is open, /health is not ready.
        try daemon.setHealthDown(true)
        let first = Task { @MainActor in await store.startDaemon() }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        // Start pressed again, or the window task ran again, mid-load.
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 1_000_000_000)
        try daemon.setHealthDown(false)
        await first.value
        await second.value

        XCTAssertEqual(store.daemonState, .running, "a start during a load must not degrade either start")
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1, "one daemon for both requests")
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid, "the loading daemon was kept, not replaced")
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let logged = await logsMention(store, "joining it")
        XCTAssertTrue(logged, "the Logs window records that the second request joined the first")
    }

    // MARK: Start after Stop

    /// A store whose first start is suspended in the supervisor's first
    /// /health probe, before it reserves a process.
    @MainActor
    private func storeWithAStartSuspendedInItsFirstProbe() async throws -> (
        daemon: ReconnectFakeDaemon,
        store: MTPLXBackendStore,
        gate: FirstCallGate,
        first: Task<Void, Never>
    ) {
        let daemon = try ReconnectFakeDaemon.make()
        let gate = FirstCallGate()
        let supervisor = DaemonSupervisor(initialHealthProbe: { url, apiKey in
            await gate.pass()
            return await DaemonSupervisor.defaultHealthProbe(url, apiKey)
        })
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            supervisor: supervisor,
            fans: FanCallRecorder()
        )
        let first = Task { @MainActor in await store.startDaemon() }
        await gate.waitUntilEntered()
        XCTAssertEqual(daemon.spawns().count, 0)
        return (daemon, store, gate, first)
    }

    /// Stop overtakes a start that is still in its first probe, then Start
    /// is pressed. The second request must launch: joining the cancelled
    /// start ended it with that start's cancellation and launched nothing.
    @MainActor
    func testStartAfterStopDuringASuspendedStartLaunches() async throws {
        let (daemon, store, gate, first) = try await storeWithAStartSuspendedInItsFirstProbe()
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.stopDaemon()
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 300_000_000)
        await gate.open()
        await first.value
        await second.value

        XCTAssertEqual(store.daemonState, .running, "Start after Stop must launch")
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    /// The same through Restart, which is Stop then Start on one task.
    @MainActor
    func testRestartDuringASuspendedStartLaunches() async throws {
        let (daemon, store, gate, first) = try await storeWithAStartSuspendedInItsFirstProbe()
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        let restart = Task { @MainActor in
            await store.stopDaemon()
            await store.startDaemon()
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        await gate.open()
        await first.value
        await restart.value

        XCTAssertEqual(store.daemonState, .running, "Restart during a start must end running")
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
    }

    /// The first caller is cancelled (its window closed) while a second
    /// request waits on the same start. Neither cancels the load.
    @MainActor
    func testCancellingTheFirstCallerWhileAnotherJoinsKeepsTheLoad() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        try daemon.setHealthDown(true)
        let first = Task { @MainActor in await store.startDaemon() }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 300_000_000)
        first.cancel()
        try await Task.sleep(nanoseconds: 300_000_000)
        try daemon.setHealthDown(false)
        await second.value
        await first.value

        XCTAssertEqual(store.daemonState, .running)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: Races that reach the typed `alreadyRunning` handling

    /// A start request checked that the supervisor held nothing, and an
    /// adoption claimed it before the request's own launch did. The launch
    /// gets `alreadyRunning`, which used to publish Degraded and restore the
    /// fans. Now the request reconnects to the adopted daemon.
    @MainActor
    func testAStartThatLosesTheRaceToAnAdoptionReconnects() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let gate = FirstCallGate()
        let supervisor = DaemonSupervisor(beforeLaunchOwnershipCheck: { await gate.pass() })
        let fans = FanCallRecorder()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .max),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        let start = Task { @MainActor in await store.startDaemon() }
        await gate.waitUntilEntered()
        // Meanwhile a daemon from an earlier session comes up on the port
        // and the app adopts it.
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        await store.attachExistingDaemonIfOwned()
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId)
        await gate.open()
        await start.value

        XCTAssertEqual(store.daemonState, .running, "losing the race to an adoption is not a failure")
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId)
        XCTAssertEqual(daemon.spawns().count, 1, "only the adopted daemon")
        let restores = await fans.restores
        XCTAssertEqual(restores, 0, "the adopted daemon's max fans are not reset")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    /// The same race inside a configuration restart: after the restart
    /// stopped the app's daemon and before it launched the new one, an
    /// adoption claimed the supervisor.
    @MainActor
    func testAConfigurationRestartThatLosesTheRaceToAnAdoptionReconnects() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let gate = FirstCallGate(armed: false)
        let supervisor = DaemonSupervisor(beforeLaunchOwnershipCheck: { await gate.pass() })
        let fans = FanCallRecorder()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        await gate.arm()

        let restart = Task { @MainActor () -> (any Error)? in
            do {
                try await store.applyConfiguration(store.configuration, restartIfRunning: true)
                return nil
            } catch {
                return error
            }
        }
        await gate.waitUntilEntered()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        await store.attachExistingDaemonIfOwned()
        await gate.open()
        let failure = await restart.value

        XCTAssertEqual(
            failure as? DaemonSupervisorError, .alreadyRunning,
            "the caller still learns that its restart did not run"
        )
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId)
        XCTAssertEqual(daemon.spawns().count, 2, "the app's first daemon and the adopted one")
        let restores = await fans.restores
        XCTAssertEqual(restores, 0)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: A stale Degraded heals when the live stats come back

    /// Whatever left the badge Degraded over a daemon that still runs (on
    /// 1de2b1c0, a second start did), the live stats reconnecting after a
    /// drop re-check the daemon and clear it, with no restart.
    @MainActor
    func testStaleDegradedClearsWhenTheLiveStatsComeBack() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        store.setDaemonStateForTesting(.degraded("MTPLX is already running."))
        try daemon.setStreamDown(true)
        try await pollUntil("live stats reconnecting") {
            if case .reconnecting = store.connectionState { return true }
            return false
        }
        XCTAssertEqual(store.daemonState, .degraded("MTPLX is already running."))
        try daemon.setStreamDown(false)

        try await pollUntil("the engine reads Running again", timeout: 15) { store.daemonState == .running }
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(daemon.spawns().count, 1, "healed without a restart")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: Stop after a reconnect, and a start that really fails

    /// Stop after a second start reconnected: the daemon is gone, the fans
    /// are restored, and nothing keeps running.
    @MainActor
    func testStopAfterAReconnectStopsTheDaemonAndRestoresFans() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .max),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        let startupPID = try XCTUnwrap(store.health?.startup?.pid)
        let pid = pid_t(startupPID)
        let restoresBefore = await fans.restores

        await store.stopDaemon()

        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertEqual(store.startupPhase, .idle)
        XCTAssertFalse(supervisor.isRunning())
        XCTAssertEqual(supervisor.currentHold(), .none)
        XCTAssertFalse(store.hasActiveDaemonTransportForTesting)
        try await pollUntil("the daemon process exited") { kill(pid, 0) != 0 }
        let restoresAfter = await fans.restores
        XCTAssertGreaterThan(restoresAfter, restoresBefore, "Stop restores the max fans")
    }

    /// A start that really fails still restores max fans and reads Degraded:
    /// the typed handling of `alreadyRunning` left other failures alone.
    @MainActor
    func testAGenuinelyFailedStartStillRestoresFansAndDegrades() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        try daemon.setExitOnStart(true)
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .max), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()

        guard case .degraded(let reason) = store.daemonState else {
            return XCTFail("a failed start reads Degraded, got \(store.daemonState)")
        }
        XCTAssertTrue(reason.contains("exited before /health"), reason)
        XCTAssertEqual(daemon.spawns().count, 1)
        let restores = await fans.restores
        XCTAssertGreaterThanOrEqual(restores, 1, "a failed start with max fans restores them")
    }

    // MARK: A configuration change that fails before its restart

    /// A change whose client config write fails before the restart reaches
    /// the running daemon. That daemon keeps serving the previous
    /// configuration: the badge stays Running and monitored, its max fans
    /// are not reset, and the failure is shown on its own. On 1de2b1c0 this
    /// published Degraded, restored the fans and stopped the watchdog; on
    /// 2b69e352 the Degraded was also kept against Refresh and reconnects.
    @MainActor
    func testAChangeThatFailsBeforeItsRestartLeavesTheEngineRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        // A directory where OpenCode's config file should be: the write fails.
        let blockedConfig = daemon.root.appendingPathComponent("opencode-blocked", isDirectory: true)
        try FileManager.default.createDirectory(at: blockedConfig, withIntermediateDirectories: true)
        let fans = FanCallRecorder()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .max),
            fans: fans,
            openCodeConfigURL: blockedConfig
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)
        let restoresBefore = await fans.restores

        var next = store.configuration
        next.lastLaunchTarget = LaunchTarget.openCode.rawValue
        do {
            try await store.applyConfiguration(next, restartIfRunning: true)
            XCTFail("the OpenCode config write was expected to fail")
        } catch {
            // Settings shows "Apply failed" with this error.
        }

        XCTAssertEqual(store.daemonState, .running, "the daemon that keeps serving is not Degraded")
        XCTAssertEqual(store.startupPhase, .ready)
        let failure = try XCTUnwrap(store.configurationChangeFailure, "the failed change is shown on its own")
        XCTAssertFalse(failure.isEmpty)
        let restoresAfter = await fans.restores
        XCTAssertEqual(restoresAfter, restoresBefore, "the serving daemon's max fans are not reset")
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(daemon.spawns().count, 1, "no restart happened")
        XCTAssertTrue(store.hasActiveDaemonTransportForTesting, "the engine is still monitored")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")

        // An explicit start clears the notice and keeps the daemon.
        await store.startDaemon()
        XCTAssertNil(store.configurationChangeFailure)
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(daemon.spawns().count, 1)
    }

    // MARK: Codex final review, finding 1: a stale answer from the replaced daemon

    /// Refresh's answers come from daemon A, and a restart replaces A with
    /// B before they are read. A's answer carries A's launch id, which is
    /// not B's: read against B it looked like another server on the port,
    /// and the app reaped its own replacement daemon. It is now dropped.
    @MainActor
    func testAStaleRefreshAnswerFromTheReplacedDaemonIsDropped() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let restartInTheWindow = OneShotAction()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            fans: FanCallRecorder(),
            beforeStaticStateAnswersAreRead: { await restartInTheWindow.run() }
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let first = try XCTUnwrap(store.health?.startup)
        await restartInTheWindow.set { @MainActor in
            try? await store.applyConfiguration(store.configuration, restartIfRunning: true)
        }

        await store.refresh()

        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 2, "A and its replacement B")
        let replacement = try XCTUnwrap(spawns.last)
        XCTAssertNotEqual(replacement.launchID, first.launchId)
        XCTAssertEqual(store.daemonState, .running, "the app's replacement daemon was not reaped")
        XCTAssertEqual(kill(pid_t(replacement.pid), 0), 0, "B is still running")
        XCTAssertEqual(store.health?.startup?.launchId, replacement.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: Codex final review, finding 2: Stop while a start still prepares

    /// Stop lands while a start is still in the port preflight, before it
    /// has launched anything. The start used to carry on after Stop
    /// returned and launch on the next free port.
    @MainActor
    func testStopDuringThePortPreflightMeansNothingLaunches() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let listener = try SilentListener(port: daemon.port)
        defer { listener.close() }
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        store.portSettleTimeoutSeconds = 3
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        let start = Task { @MainActor in await store.startDaemon() }
        // The preflight re-probes the silent port for about four seconds.
        try await Task.sleep(nanoseconds: 1_000_000_000)
        await store.stopDaemon()
        await start.value
        try await Task.sleep(nanoseconds: 500_000_000)

        XCTAssertEqual(daemon.spawns().count, 0, "nothing launches after Stop")
        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertEqual(store.startupPhase, .idle)
        XCTAssertNil(store.portFallbackNotice, "no port move for a launch that did not happen")
        XCTAssertEqual(store.configuration.port, daemon.port)
        let logged = await logsMention(store, "start abandoned")
        XCTAssertTrue(logged, "the Logs window records why nothing launched")
    }

    // MARK: Codex final review, finding 3: a pid that now belongs to another process

    /// The app adopted a daemon whose /health named pid X. The daemon then
    /// exited and the kernel gave X to an unrelated process. Here /health
    /// reports the pid of an unrelated `sleep` from the start, which is the
    /// same situation. Stop used to signal X. It now signals only a process
    /// that carries the daemon's launch id.
    @MainActor
    func testStopNeverSignalsAProcessThatNowHoldsTheAdoptedDaemonsPID() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let unrelated = Process()
        unrelated.executableURL = URL(fileURLWithPath: "/bin/sleep")
        unrelated.arguments = ["60"]
        try unrelated.run()
        addTeardownBlock { if unrelated.isRunning { unrelated.terminate() } }
        try daemon.setReportedPID(unrelated.processIdentifier)
        _ = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running, "adopted")
        XCTAssertEqual(store.health?.startup?.pid, Int(unrelated.processIdentifier))

        await store.stopDaemon()
        try await Task.sleep(nanoseconds: 300_000_000)

        XCTAssertEqual(store.daemonState, .stopped)
        XCTAssertTrue(unrelated.isRunning, "Stop never signals a process it cannot confirm as the app's daemon")
        let logged = await logsMention(store, "not signalling pid \(unrelated.processIdentifier)")
        XCTAssertTrue(logged, "the Logs window records the pid it left alone")
    }

    // MARK: Codex final review, finding 4: a server that does not identify itself

    /// An adopted daemon exits and a server that does not identify itself
    /// takes its port: /health answers 401 (a key the app does not have) or
    /// an undecodable 200. Both prove a live server and used to reset the
    /// watchdog every round, so the badge read Running for good. With the
    /// app's daemon gone, the app now lets it go and names the other server.
    @MainActor
    private func assertAnUnidentifiedServerIsNamedAfterTheAdoptedDaemonExits(
        undecodableHealth: Bool,
        strangerKey: String?
    ) async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        _ = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: MTPLXAppConfiguration(),
            supervisor: supervisor,
            fans: FanCallRecorder()
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        try await pollUntil("live stats open") { store.connectionState == .open }

        prior.terminate()
        prior.waitUntilExit()
        try daemon.setHealthUndecodable(undecodableHealth)
        let stranger = try daemon.launchOutsideTheApp(launchID: nil, apiKey: strangerKey)
        addTeardownBlock { if stranger.isRunning { stranger.terminate() } }
        try await daemon.waitUntilAnswering()

        let expected = "Another server holds port \(daemon.port); this app is not connected to it."
        try await pollUntil("the other server is named", timeout: 15) {
            store.daemonState == .degraded(expected)
        }
        XCTAssertEqual(store.daemonState, .degraded(expected))
        XCTAssertTrue(badge(store).label.hasPrefix("Degraded — Another server"), badge(store).label)
        XCTAssertFalse(supervisor.isRunning(), "the app let go of the daemon it lost")

        await store.stopDaemon()
        XCTAssertTrue(stranger.isRunning, "Stop never signals the other server")
    }

    @MainActor
    func testAServerAnswering401AfterTheAdoptedDaemonExitsIsNamed() async throws {
        try await assertAnUnidentifiedServerIsNamedAfterTheAdoptedDaemonExits(
            undecodableHealth: false,
            strangerKey: "a-key-the-app-does-not-have"
        )
    }

    @MainActor
    func testAServerAnsweringUndecodableHealthAfterTheAdoptedDaemonExitsIsNamed() async throws {
        try await assertAnUnidentifiedServerIsNamedAfterTheAdoptedDaemonExits(
            undecodableHealth: true,
            strangerKey: nil
        )
    }

    // MARK: Codex final review, finding 5: the benchmark needs the app's own daemon

    /// `mtplx serve` from a terminal holds the configured port and the app
    /// holds nothing. The benchmark used to take that server as ready, run
    /// against it and post settings to it. It now says the port is held by
    /// a server started outside the app, and starts nothing beside it.
    @MainActor
    func testTheBenchmarkRefusesAServerStartedOutsideTheApp() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let terminalServer = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if terminalServer.isRunning { terminalServer.terminate() } }
        let terminalHealth = try await daemon.waitUntilHealthy()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        do {
            let ready = try await store.ensureDaemonReadyForBenchmark()
            XCTFail("the benchmark took a server the app does not run (pid \(ready.startup?.pid ?? -1))")
        } catch BenchmarkDaemonReadinessError.startupFailed(let reason) {
            XCTAssertTrue(reason.contains("started outside the app"), reason)
        }

        XCTAssertNotEqual(store.health?.startup?.pid, terminalHealth.startup?.pid)
        XCTAssertEqual(daemon.spawns().count, 1, "no second model was started beside it")
        XCTAssertTrue(terminalServer.isRunning)
    }

    /// The app's own daemon still passes, without a second launch.
    @MainActor
    func testTheBenchmarkUsesTheAppsOwnDaemon() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        let launched = try XCTUnwrap(store.health?.startup)

        let ready = try await store.ensureDaemonReadyForBenchmark()

        XCTAssertEqual(ready.startup?.launchId, launched.launchId)
        XCTAssertEqual(daemon.spawns().count, 1)
        XCTAssertEqual(store.daemonState, .running)
    }

    // MARK: Codex final review, finding 6: the key of the daemon that keeps serving

    /// A change of the API key fails before its restart (the OpenCode
    /// config write fails), so the running daemon keeps the key it started
    /// with; the scripted daemon, like `mtplx serve`, read it once and
    /// requires it on every request. `configuration` holds the new key for
    /// the next launch. Requests to the running daemon used the new key and
    /// got 401 under a Running badge; they now carry the daemon's own key.
    @MainActor
    func testAFailedKeyChangeKeepsUsingTheKeyTheServingDaemonHas() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let blockedConfig = daemon.root.appendingPathComponent("opencode-blocked", isDirectory: true)
        try FileManager.default.createDirectory(at: blockedConfig, withIntermediateDirectories: true)
        var configuration = daemon.configuration(fanMode: .default)
        configuration.apiKey = "key-at-start"
        let store = daemon.makeStore(
            configuration: configuration,
            fans: FanCallRecorder(),
            openCodeConfigURL: blockedConfig
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        var next = store.configuration
        next.apiKey = "a-new-key"
        next.lastLaunchTarget = LaunchTarget.openCode.rawValue
        do {
            try await store.applyConfiguration(next, restartIfRunning: true)
            XCTFail("the OpenCode config write was expected to fail")
        } catch {
            // Settings shows "Apply failed" with this error.
        }

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertNotNil(store.configurationChangeFailure)
        XCTAssertEqual(store.configuration.apiKey, "a-new-key", "the saved change is kept for the next launch")
        XCTAssertEqual(store.activeAPIKey, "key-at-start")
        let answer = try await store.apiClient.health()
        XCTAssertEqual(answer.startup?.launchId, launched.launchId, "requests still reach the serving daemon")
        await store.refresh()
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
        XCTAssertEqual(daemon.spawns().count, 1)
    }

    // MARK: Closing the window during a model load

    /// Closing the main window cancels its launch task. The start runs on a
    /// task the store owns, so the load carries on and the engine comes up.
    /// Before, the cancellation reached the supervisor's health wait, which
    /// stopped the loading daemon: "Degraded — CancellationError()".
    @MainActor
    func testClosingTheWindowDuringALoadStillReachesRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        // The model is still loading: the port is open, /health is not ready.
        try daemon.setHealthDown(true)
        // The daemon half of the window's launch task.
        let window = Task { @MainActor in
            store.loadPersistedSettings()
            await store.startDaemon()
        }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        try await Task.sleep(nanoseconds: 500_000_000)
        // The user closes the window.
        window.cancel()
        try await Task.sleep(nanoseconds: 500_000_000)
        try daemon.setHealthDown(false)

        try await pollUntil("the load finished", timeout: 10) { store.daemonState == .running }
        await window.value
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1, "the loading daemon was kept, not relaunched")
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }
}
