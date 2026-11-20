import Darwin
import Foundation

/// DeepSeek Harness in the Start menu.
///
/// The app opens it through `mtplx connect deepseek-harness --open --json`:
/// `mtplx/deepseek_harness.py` owns the harness profile MTPLX writes (its
/// route to the local server, its telemetry, vendor session log and feedback
/// uploads switched off) and the check of what the installed harness
/// composes, which refuses to open a version that adds a reporting piece MTPLX
/// does not switch off. One owner, so the app and `mtplx start
/// deepseek-harness` cannot part. The harness runs on MTPLX's own home
/// (`~/.mtplx/clients/deepseek-harness`); the user's `~/.dsh` is never read
/// or written.
///
/// Its browser app is a `dsh web` server the CLI starts in its own process
/// group; the app records that pid and ends it on Stop, after proving from
/// the process's own argv that it is MTPLX's (it names MTPLX's overlay).
public struct DeepSeekHarnessIntegration: Sendable {
    public static let clientID = "deepseek-harness"
    /// The file MTPLX's harness processes name on their command line.
    public static let overlayPathSuffix = "/.mtplx/clients/deepseek-harness/mtplx.patch.yml"

    public struct LaunchResult: Equatable, Sendable {
        public var opened: Bool
        /// "web" (its browser app) or "terminal".
        public var surface: String
        /// The `dsh web` server MTPLX started, when it opened as the browser app.
        public var processID: Int32?
        public var url: String?
        public var terminal: String?
        public var detail: String
        /// The command, for the Logs pane.
        public var command: String

        public init(
            opened: Bool,
            surface: String,
            processID: Int32? = nil,
            url: String? = nil,
            terminal: String? = nil,
            detail: String,
            command: String
        ) {
            self.opened = opened
            self.surface = surface
            self.processID = processID
            self.url = url
            self.terminal = terminal
            self.detail = detail
            self.command = command
        }
    }

    private let processEnvironment: [String: String]
    private let runner: @Sendable (URL, [String], [String: String]) async throws -> (Int32, Data, Data)

    public init(
        processEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        runner: (@Sendable (URL, [String], [String: String]) async throws -> (Int32, Data, Data))? = nil
    ) {
        self.processEnvironment = processEnvironment
        self.runner = runner ?? Self.run
    }

    /// `mtplx connect deepseek-harness` for the daemon on `port`. `surface`:
    /// "web" or "terminal" for this launch, nil for the saved choice.
    public static func arguments(port: Int, workspace: String, surface: String?) -> [String] {
        var arguments = [
            "connect", clientID,
            "--open", "--json",
            "--host", "127.0.0.1",
            "--port", String(port),
            "--api-key-env", "MTPLX_API_KEY",
            "--workspace", workspace,
        ]
        switch surface {
        case "web": arguments.append("--desktop")
        case "terminal": arguments.append("--cli")
        default: break
        }
        return arguments
    }

    /// Reads the command's JSON; nil when it is not that payload.
    public static func decode(_ data: Data, command: String) -> LaunchResult? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["integration"] as? String == clientID
        else { return nil }
        let opened = object["opened"] as? Bool ?? false
        let surface = object["surface"] as? String ?? "web"
        let pid = (object["pid"] as? Int).flatMap { $0 > 1 ? Int32($0) : nil }
        let url = object["url"] as? String
        let terminal = object["terminal"] as? String
        let detail: String
        if let error = object["error"] as? String, !error.isEmpty {
            detail = error
        } else if surface == "web" {
            detail = tr("DeepSeek Harness opened in your browser at %@, with its telemetry off.", url ?? "")
        } else {
            detail = tr("DeepSeek Harness opened in %@, with its telemetry off.", terminal ?? "Terminal")
        }
        return LaunchResult(
            opened: opened,
            surface: surface,
            processID: pid,
            url: url,
            terminal: terminal,
            detail: detail,
            command: command
        )
    }

    /// Opens DeepSeek Harness on the daemon at `port`.
    public func launch(
        port: Int,
        apiKey: String?,
        workspace: String,
        surface: String? = nil
    ) async -> LaunchResult {
        let arguments = Self.arguments(port: port, workspace: workspace, surface: surface)
        let command = (["mtplx"] + arguments).joined(separator: " ")
        guard let executable = try? MTPLXCommandBuilder.resolveInstalledExecutable(
            environment: processEnvironment
        ) else {
            return LaunchResult(
                opened: false, surface: surface ?? "web",
                detail: tr("MTPLX's command-line runtime is not installed, so DeepSeek Harness could not open."),
                command: command
            )
        }
        var environment = processEnvironment
        environment["MTPLX_API_KEY"] = apiKey.flatMap { $0.isEmpty ? nil : $0 } ?? "mtplx-local"
        do {
            let (status, stdout, stderr) = try await runner(executable, arguments, environment)
            if let result = Self.decode(stdout, command: command) {
                return result
            }
            let message = String(decoding: stderr, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return LaunchResult(
                opened: false, surface: surface ?? "web",
                detail: message.isEmpty ? "mtplx connect exited \(status)" : message,
                command: command
            )
        } catch {
            return LaunchResult(
                opened: false, surface: surface ?? "web",
                detail: String(describing: error),
                command: command
            )
        }
    }

    // MARK: - The browser app's server

    /// True while `pid` is a harness MTPLX started: its own argv names
    /// MTPLX's overlay. A reused pid or the user's own harness never is.
    static func isMTPLXHarness(
        _ pid: Int32,
        arguments: (Int32) -> (executable: String, argv: [String])? = CodexDesktopIntegration.processArguments
    ) -> Bool {
        guard pid > 1, let image = arguments(pid) else { return false }
        return image.argv.contains { $0.hasSuffix(overlayPathSuffix) }
    }

    /// Ends the `dsh web` server MTPLX started and everything in its process
    /// group: SIGTERM, then SIGKILL if it is still that process after
    /// `graceSeconds`. Blocks; call it off the main actor.
    @discardableResult
    public static func end(processID pid: Int32, graceSeconds: TimeInterval = 5) -> Bool {
        guard isMTPLXHarness(pid) else { return false }
        _ = killpg(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(graceSeconds)
        while Date() < deadline {
            if !isMTPLXHarness(pid) { return true }
            Thread.sleep(forTimeInterval: 0.1)
        }
        if isMTPLXHarness(pid) { _ = killpg(pid, SIGKILL) }
        return true
    }

    // MARK: - Running the CLI

    @Sendable
    static func run(
        executable: URL, arguments: [String], environment: [String: String]
    ) async throws -> (Int32, Data, Data) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = executable
                process.arguments = arguments
                process.environment = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(
                    environment: environment
                )
                process.standardInput = FileHandle.nullDevice
                let outPipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = outPipe
                process.standardError = errPipe
                let watchdog = SubprocessWatchdog(process)
                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                let stdoutDrain = SubprocessPipeDrain(outPipe)
                let stderrDrain = SubprocessPipeDrain(errPipe)
                // First open may also download and install the private runtime.
                // Later launches return as soon as the browser server is ready.
                guard watchdog.wait(for: process, timeout: 420) else {
                    continuation.resume(returning: (-2, Data(), Data("DeepSeek Harness setup or launch timed out.".utf8)))
                    return
                }
                stdoutDrain.join()
                stderrDrain.join()
                continuation.resume(returning: (
                    process.terminationStatus,
                    stdoutDrain.snapshotData(),
                    stderrDrain.snapshotData()
                ))
            }
        }
    }
}
