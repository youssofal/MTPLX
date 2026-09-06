import CryptoKit
import Foundation
import OSLog

public enum MTPLXRuntimeBootstrapperError: Error, LocalizedError, Sendable {
    case homebrewNotFound
    case pythonNotFound
    case commandFailed(command: String, exitCode: Int32, output: String)
    case runtimeStillMissing(output: String)

    public var errorDescription: String? {
        switch self {
        case .homebrewNotFound:
            return tr("Homebrew was not found, so MTPLX could not install its command-line runtime automatically. Install Homebrew from brew.sh, then press Retry.")
        case .pythonNotFound:
            return tr("Python 3.11 or newer was not found, so MTPLX could not prepare its command-line runtime. Install Homebrew from brew.sh, then press Retry.")
        case .commandFailed(let command, let exitCode, let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.isEmpty {
                return tr("%@ failed with exit code %@.", command, String(exitCode))
            }
            return tr("%@ failed with exit code %@: %@", command, String(exitCode), detail)
        case .runtimeStillMissing(let output):
            let detail = output.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.isEmpty {
                return tr("Homebrew finished, but MTPLX still was not available on PATH.")
            }
            return tr("Homebrew finished, but MTPLX still was not available on PATH: %@", detail)
        }
    }
}

public struct MTPLXRuntimeBootstrapper: Sendable {
    public static let formula = "youssofal/mtplx/mtplx"
    private static let logger = Logger(subsystem: "com.mtplx.app", category: "RuntimeBootstrapper")

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        self.environment = environment
    }

    private let environment: [String: String]

    public func installOrUpdate(status: (@Sendable (String) -> Void)? = nil) throws -> URL {
        status?(tr("Checking MTPLX runtime"))
        let minimumVersion = minimumRuntimeVersion()
        let bundledWheel = MTPLXCommandBuilder.bundledRuntimeWheelPath(environment: environment)
        // When this bundle ships a wheel, the engine is always the
        // app-owned venv. A user-managed mtplx on PATH (pip user-site,
        // old source install) that happens to satisfy the version
        // floor must never be adopted as the engine: its Python and
        // dependency state are unknown, and on first run — before the
        // app venv exists — it would otherwise win the PATH walk and
        // every later model load runs in an environment we never
        // installed. PATH installs remain the user's; onboarding's
        // CLI row reports them separately.
        if let existing = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment),
           runtime(existing, satisfies: minimumVersion),
           bundledWheel == nil || isAppManagedRuntime(existing),
           installedRuntimeMatchesBundledWheel(installedExecutable: existing) {
            // The version floor and wheel fingerprint prove which bytes are
            // installed, not that they can run: a torn dependency upgrade or
            // a foreign pip session can leave mlx's native extension unable
            // to dlopen its own dylib ("Symbol not found: ..._scaled_dot_
            // product_attention..."), and the daemon then dies before
            // /health on every launch. `mtplx --version` never imports mlx,
            // so both checks above stay green, and reinstalling the app
            // cannot heal it because the venv lives in Application Support
            // with a still-matching fingerprint. Prove the imports once per
            // wheel (marker), re-prove after any daemon death (breadcrumb),
            // and rebuild the venv from scratch when the probe fails.
            if runtimeImportHealthAccepted(installedExecutable: existing) {
                return existing
            }
            if let wheel = bundledWheel {
                status?(tr("Repairing MTPLX runtime"))
                return try installBundledRuntime(
                    wheel: URL(fileURLWithPath: wheel),
                    rebuildFromScratch: true
                )
            }
        }
        if let wheel = bundledWheel {
            status?(tr("Installing MTPLX runtime"))
            return try installBundledRuntime(wheel: URL(fileURLWithPath: wheel))
        }
        if let existing = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment),
           minimumVersion == nil {
            return existing
        }
        status?(tr("Installing MTPLX runtime"))
        return try installHomebrewRuntime()
    }

    /// Whether `installedExecutable` can be reused as-is for this app
    /// bundle.
    ///
    /// The version floor alone cannot see same-version rebuilds:
    /// 1.0.0 build N and build N+1 ship different wheels under one
    /// semantic version, so after an auto-update the app-managed venv
    /// would silently keep serving the old code forever. For the venv
    /// the app installed itself, the install-time fingerprint marker
    /// must match the wheel this bundle ships; runtimes the app does
    /// not manage (Homebrew/system installs) keep the version-floor
    /// contract unchanged.
    func installedRuntimeMatchesBundledWheel(installedExecutable: URL) -> Bool {
        guard let wheelPath = MTPLXCommandBuilder.bundledRuntimeWheelPath(
            environment: environment
        ) else {
            return true
        }
        let runtimeDir = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment)
        )
        guard isAppManagedRuntime(installedExecutable) else {
            return true
        }
        guard let selected = try? selectedBundledRuntimeWheel(
            fallback: URL(fileURLWithPath: wheelPath),
            python: runtimeDir.appendingPathComponent("bin/python")
        ), let bundled = try? Self.wheelFingerprint(of: selected) else {
            return false
        }
        return bundled == Self.recordedWheelFingerprint(runtimeDir: runtimeDir)
    }

    /// Whether `executable` resolves into the app-owned runtime venv.
    func isAppManagedRuntime(_ executable: URL) -> Bool {
        let managed = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment)
        )
        .appendingPathComponent("bin")
        .appendingPathComponent("mtplx")
        .resolvingSymlinksInPath()
        return executable.resolvingSymlinksInPath().path == managed.path
    }

    /// SHA-256 of the wheel file, hex-encoded.
    static func wheelFingerprint(of wheel: URL) throws -> String {
        let data = try Data(contentsOf: wheel, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Marker recording which bundled wheel the app-managed venv was
    /// installed from. It lives inside the venv so it travels and dies
    /// with the install it describes.
    static func wheelFingerprintMarkerURL(runtimeDir: URL) -> URL {
        runtimeDir.appendingPathComponent("bundled-wheel.sha256")
    }

    static func recordedWheelFingerprint(runtimeDir: URL) -> String? {
        guard let raw = try? String(
            contentsOf: wheelFingerprintMarkerURL(runtimeDir: runtimeDir),
            encoding: .utf8
        ) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Best-effort: a missing marker only costs one redundant
    /// reinstall on the next launch, while failing the install over it
    /// would cost a working runtime.
    static func recordWheelFingerprint(for wheel: URL, runtimeDir: URL) {
        guard let fingerprint = try? wheelFingerprint(of: wheel) else { return }
        try? fingerprint.write(
            to: wheelFingerprintMarkerURL(runtimeDir: runtimeDir),
            atomically: true,
            encoding: .utf8
        )
    }

    // MARK: - Runtime import health (self-healing venv)

    /// The imports a daemon launch actually needs to survive. `mlx.core`
    /// is the native-extension pair (core.cpython-*.so ↔ libmlx dylib)
    /// that a torn upgrade or a foreign pip session leaves mismatched;
    /// `mtplx` catches a gutted package install.
    static let importHealthProbeSource = "import mlx.core, mtplx"

    /// Marker recording the wheel fingerprint whose venv passed the
    /// import probe. Lives inside the venv so it dies with the install
    /// it vouches for (a `--clear` rebuild wipes it).
    static func importHealthMarkerURL(runtimeDir: URL) -> URL {
        runtimeDir.appendingPathComponent("runtime-import-health.sha256")
    }

    static func recordedImportHealth(runtimeDir: URL) -> String? {
        guard let raw = try? String(
            contentsOf: importHealthMarkerURL(runtimeDir: runtimeDir),
            encoding: .utf8
        ) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func recordImportHealth(fingerprint: String, runtimeDir: URL) {
        try? fingerprint.write(
            to: importHealthMarkerURL(runtimeDir: runtimeDir),
            atomically: true,
            encoding: .utf8
        )
    }

    /// Breadcrumb requesting a full import re-probe on the next launch.
    /// Written when a daemon dies before /health became ready; lives
    /// beside the venv (not inside it) so a rebuild starts from a clean
    /// slate and clearing is always an explicit act after the probe ran.
    static func importRecheckRequestURL(environment: [String: String]) -> URL {
        URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment))
            .deletingLastPathComponent()
            .appendingPathComponent("runtime-import-recheck")
    }

    /// Called by the launch failure path: a daemon that exits before
    /// /health may be sitting on a venv that can no longer import mlx.
    /// The next `installOrUpdate` re-probes and rebuilds if broken.
    public static func requestRuntimeImportRecheck(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        let url = importRecheckRequestURL(environment: environment)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? Data("a daemon launch died before /health; verify runtime imports before the next launch\n".utf8)
            .write(to: url, options: .atomic)
    }

    static func clearRuntimeImportRecheck(environment: [String: String]) {
        try? FileManager.default.removeItem(
            at: importRecheckRequestURL(environment: environment)
        )
    }

    /// Exit-0 iff the venv's python can import the native stack. `-I`
    /// (isolated mode) ignores PYTHONPATH and user site-packages, so the
    /// probe sees exactly what the daemon's hermetic launch sees.
    func runtimeImportProbeSucceeds(runtimeDir: URL) -> Bool {
        let venvPython = runtimeDir
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
        do {
            _ = try run(
                executable: venvPython,
                arguments: ["-I", "-c", Self.importHealthProbeSource],
                displayCommand: "runtime python -c '\(Self.importHealthProbeSource)'",
                timeout: 120
            )
            return true
        } catch {
            return false
        }
    }

    /// Reuse gate for an already-installed app-managed venv: fast-path on
    /// the recorded health marker (one stat per launch), probe on first
    /// adoption of a wheel or when a daemon death requested a recheck.
    func runtimeImportHealthAccepted(installedExecutable: URL) -> Bool {
        guard let wheelPath = MTPLXCommandBuilder.bundledRuntimeWheelPath(
            environment: environment
        ) else {
            // No wheel to rebuild from: Homebrew/system runtimes keep
            // their existing contract (doctor guides those users).
            return true
        }
        guard isAppManagedRuntime(installedExecutable) else { return true }
        let runtimeDir = URL(
            fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment)
        )
        guard let selected = try? selectedBundledRuntimeWheel(
            fallback: URL(fileURLWithPath: wheelPath),
            python: runtimeDir.appendingPathComponent("bin/python")
        ) else { return false }
        let fingerprint = try? Self.wheelFingerprint(of: selected)
        let recheckRequested = FileManager.default.fileExists(
            atPath: Self.importRecheckRequestURL(environment: environment).path
        )
        if !recheckRequested,
           let fingerprint,
           Self.recordedImportHealth(runtimeDir: runtimeDir) == fingerprint {
            return true
        }
        guard runtimeImportProbeSucceeds(runtimeDir: runtimeDir) else {
            return false
        }
        if let fingerprint {
            Self.recordImportHealth(fingerprint: fingerprint, runtimeDir: runtimeDir)
        }
        Self.clearRuntimeImportRecheck(environment: environment)
        return true
    }

    public func upgradeHomebrewRuntime() throws -> URL {
        guard MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) != nil else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }
        return try runHomebrewInstallSequence(allowExistingRuntime: true)
    }

    private func installHomebrewRuntime() throws -> URL {
        guard let brew = MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }
        _ = brew
        return try runHomebrewInstallSequence(allowExistingRuntime: false)
    }

    private func runHomebrewInstallSequence(allowExistingRuntime: Bool) throws -> URL {
        guard let brew = MTPLXCommandBuilder.resolveHomebrewExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.homebrewNotFound
        }

        var lastOutput = ""
        lastOutput = try run(brew: brew, arguments: ["update"])
        if allowExistingRuntime {
            if let upgradeOutput = try? run(brew: brew, arguments: ["upgrade", Self.formula]) {
                lastOutput = upgradeOutput
            } else {
                lastOutput = try run(brew: brew, arguments: ["install", Self.formula])
            }
        } else {
            lastOutput = try run(brew: brew, arguments: ["install", Self.formula])
        }
        if let upgradeOutput = try? run(brew: brew, arguments: ["upgrade", Self.formula]) {
            lastOutput = upgradeOutput
        }
        if let linkOutput = try? run(brew: brew, arguments: ["link", "--overwrite", "mtplx"]) {
            lastOutput = linkOutput
        }

        let minimumVersion = minimumRuntimeVersion()
        if let resolved = try? resolvedInstalledRuntime(minimumVersion: minimumVersion, output: lastOutput) {
            return resolved
        }

        if let unlinkOutput = try? run(brew: brew, arguments: ["unlink", "mtplx"]) {
            lastOutput = unlinkOutput
        }
        lastOutput = try run(brew: brew, arguments: ["link", "--overwrite", "mtplx"])
        if let resolved = try? resolvedInstalledRuntime(minimumVersion: minimumVersion, output: lastOutput) {
            return resolved
        }
        throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(output: lastOutput)
    }

    private func run(brew: URL, arguments: [String]) throws -> String {
        try run(
            executable: brew,
            arguments: arguments,
            displayCommand: brewCommand(arguments),
            // Formula installs legitimately take a while on cold caches;
            // still bounded so a wedged brew cannot hold the app (#158).
            timeout: 1800
        )
    }

    private func run(
        executable: URL,
        arguments: [String],
        displayCommand: String,
        timeout: TimeInterval = 900
    ) throws -> String {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.environment = Self.hermeticSubprocessEnvironment(from: environment)

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        let output = SubprocessTailBuffer(capacity: 4096)
        stdout.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { output.append(chunk) }
        }
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty { output.append(chunk) }
        }
        defer {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
        }

        // Deadline watchdog instead of a bare waitUntilExit: a wedged child
        // (pip stuck on an unreachable index, brew waiting on a lock) held
        // the app forever with no error surface (#158). The handler is
        // installed before run() so a fast exit cannot be missed.
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }

        do {
            try process.run()
        } catch {
            throw MTPLXRuntimeBootstrapperError.commandFailed(
                command: displayCommand,
                exitCode: -1,
                output: error.localizedDescription
            )
        }
        if finished.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate()
            if finished.wait(timeout: .now() + 10) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = finished.wait(timeout: .now() + 5)
            }
            throw MTPLXRuntimeBootstrapperError.commandFailed(
                command: displayCommand,
                exitCode: -2,
                output: output.snapshot()
                    + "\n[timed out after \(Int(timeout))s and was terminated]"
            )
        }
        let tail = output.snapshot()
        guard process.terminationStatus == 0 else {
            throw MTPLXRuntimeBootstrapperError.commandFailed(
                command: displayCommand,
                exitCode: process.terminationStatus,
                output: tail
            )
        }
        return tail
    }

    private func brewCommand(_ arguments: [String]) -> String {
        ("brew " + arguments.joined(separator: " "))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Environment for every bootstrap subprocess. Beyond the standard
    /// app filtering, pip must not see the user's pip configuration at
    /// all: pip reads ~/.config/pip/pip.conf (and friends) from disk,
    /// which no env blocklist can reach, and a common `user = true`
    /// there aborts every venv install with "Can not perform a
    /// '--user' install. User site-packages are not visible in this
    /// virtualenv." Pointing PIP_CONFIG_FILE at /dev/null disables
    /// config-file loading — the same technique CPython's own
    /// ensurepip uses — and PIP_USER=0 pins user installs off at the
    /// env layer, which outranks any config file pip might still find.
    static func hermeticSubprocessEnvironment(
        from environment: [String: String]
    ) -> [String: String] {
        var env = MTPLXCommandBuilder.appSubprocessEnvironment(environment: environment)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_CONFIG_FILE"] = "/dev/null"
        env["PIP_USER"] = "0"
        return env
    }

    func selectedBundledRuntimeWheel(fallback: URL, python: URL) throws -> URL {
        var isDirectory: ObjCBool = false
        guard fallback.pathExtension == "whl",
              FileManager.default.fileExists(atPath: fallback.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled fallback runtime wheel was not found: \(fallback.path)"
            )
        }
        let resources = fallback.deletingLastPathComponent()
        let native = resources.appendingPathComponent("Native", isDirectory: true)
        guard FileManager.default.fileExists(atPath: native.path) else { return fallback }
        do {
            let contents = try FileManager.default.contentsOfDirectory(
                at: native, includingPropertiesForKeys: nil
            )
            guard contents.contains(where: { $0.pathExtension == "whl" }) else {
                return fallback
            }
            // Run against the actual venv, so pip's tags cover its Python ABI,
            // CPU and macOS version. Unsupported cells retain the pure wheel.
            let output = try run(
                executable: python,
                arguments: ["-I", "-B", resources.appendingPathComponent("select_runtime_wheel.py").path,
                            fallback.path, native.path],
                displayCommand: "Selecting compatible bundled MTPLX runtime"
            )
            let selected = URL(fileURLWithPath: output.trimmingCharacters(in: .whitespacesAndNewlines))
            guard selected.pathExtension == "whl",
                  FileManager.default.fileExists(atPath: selected.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else {
                throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                    output: "Compatible bundled runtime was not found: \(output)"
                )
            }
            return selected
        } catch {
            // Native selection is optional. Keep the known bundled runtime
            // usable while recording the failure; installation/import errors
            // still propagate, and fingerprinting uses this exact fallback.
            Self.logger.error("Native wheel selection failed; using bundled pure wheel: \(error.localizedDescription, privacy: .public)")
            return fallback
        }
    }

    private func installBundledRuntime(wheel fallback: URL, rebuildFromScratch: Bool = false) throws -> URL {
        let runtimeDir = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeDirectory(environment: environment))
        try FileManager.default.createDirectory(
            at: runtimeDir.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let python = try resolvePythonExecutable()
        try createRuntimeVenv(python: python, runtimeDir: runtimeDir, forceClear: rebuildFromScratch)
        let venvPython = runtimeDir.appendingPathComponent("bin").appendingPathComponent("python")
        // Best effort: the venv's ensurepip pip is already new enough
        // to install the bundled wheel, so a PyPI hiccup or blocked
        // network here must not fail first-run setup — the wheel
        // install below is the step that actually gates readiness.
        _ = try? run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "-U", "pip"],
            displayCommand: "runtime python -m pip install -U pip"
        )
        let wheel = try selectedBundledRuntimeWheel(fallback: fallback, python: venvPython)
        _ = try run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "-U", "\(wheel.path)[server]"],
            displayCommand: "runtime python -m pip install -U bundled MTPLX"
        )
        // pip skips a wheel whose version matches the installed one,
        // so -U alone is a no-op for same-version rebuilds — exactly
        // the case the fingerprint marker exists to catch. Force the
        // package itself back to this bundle's bytes; dependencies are
        // already satisfied by the install above, so this step needs
        // no network and unpacks one wheel.
        _ = try run(
            executable: venvPython,
            arguments: ["-m", "pip", "install", "--force-reinstall", "--no-deps", wheel.path],
            displayCommand: "runtime python -m pip install --force-reinstall bundled MTPLX"
        )

        let executable = runtimeDir.appendingPathComponent("bin").appendingPathComponent("mtplx")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled runtime install finished, but \(executable.path) was not created."
            )
        }
        let minimumVersion = minimumRuntimeVersion()
        guard runtime(executable, satisfies: minimumVersion) else {
            let observed = MTPLXRuntimeUpdateService.runtimeVersion(
                executableURL: executable,
                environment: environment
            ) ?? "unknown"
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled runtime installed \(observed), but \(minimumVersion?.description ?? "the required version") is required."
            )
        }
        // A fresh install must also prove the native stack imports before
        // being trusted: pip can resolve an mlx whose extension and dylib
        // disagree, and surfacing that here (with a retryable error) beats
        // a daemon that dies before /health with no explanation.
        guard runtimeImportProbeSucceeds(runtimeDir: runtimeDir) else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "Bundled runtime installed, but the engine failed its import check "
                    + "(mlx native libraries). Check network access to PyPI, then press Retry."
            )
        }
        Self.recordWheelFingerprint(for: wheel, runtimeDir: runtimeDir)
        if let fingerprint = try? Self.wheelFingerprint(of: wheel) {
            Self.recordImportHealth(fingerprint: fingerprint, runtimeDir: runtimeDir)
        }
        Self.clearRuntimeImportRecheck(environment: environment)
        return executable
    }

    /// Create (or repair) the app-owned runtime venv.
    ///
    /// The venv's `bin/python3` is a symlink chain into the interpreter
    /// it was built from — usually the one shipped inside the app
    /// bundle. After an app update replaces the bundle, that chain can
    /// dangle, and a plain `python -m venv` over the corpse exits 1
    /// with "[Errno 2] No such file or directory: …/bin/python3"
    /// (issue #139). Reinstalling the app cannot fix it because the
    /// venv lives in Application Support, outside the bundle. So:
    /// rebuild with `--clear` when the existing venv python is broken,
    /// and retry once with `--clear` on any other creation failure.
    /// A healthy venv keeps the plain no-clear path so same-venv
    /// updates reuse installed dependencies (fast and offline-safe).
    func createRuntimeVenv(python: URL, runtimeDir: URL, forceClear: Bool = false) throws {
        let fileManager = FileManager.default
        // Probe bin/python — the executable the install steps below
        // actually invoke. isExecutableFile resolves symlinks, so a
        // dangling chain reads as not-executable — exactly the broken
        // state. `forceClear` is the import-health repair path: the venv
        // python runs but its site-packages are poisoned, so nothing in
        // it can be reused.
        let venvPython = runtimeDir
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
        let venvBroken = fileManager.fileExists(atPath: runtimeDir.path)
            && !fileManager.isExecutableFile(atPath: venvPython.path)
        if forceClear || venvBroken {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", "--clear", runtimeDir.path],
                displayCommand: "python -m venv --clear \(runtimeDir.path)"
            )
            return
        }
        do {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", runtimeDir.path],
                displayCommand: "python -m venv \(runtimeDir.path)"
            )
        } catch {
            _ = try run(
                executable: python,
                arguments: ["-m", "venv", "--clear", runtimeDir.path],
                displayCommand: "python -m venv --clear \(runtimeDir.path)"
            )
        }
    }

    private func resolvedInstalledRuntime(
        minimumVersion: MTPLXSemanticVersion?,
        output: String
    ) throws -> URL {
        guard let resolved = try? MTPLXCommandBuilder.resolveInstalledExecutable(environment: environment) else {
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(output: output)
        }
        guard runtime(resolved, satisfies: minimumVersion) else {
            let observed = MTPLXRuntimeUpdateService.runtimeVersion(
                executableURL: resolved,
                environment: environment
            ) ?? "unknown"
            throw MTPLXRuntimeBootstrapperError.runtimeStillMissing(
                output: "\(resolved.path) is \(observed), but \(minimumVersion?.description ?? "the required version") is required.\n\(output)"
            )
        }
        return resolved
    }

    private func minimumRuntimeVersion() -> MTPLXSemanticVersion? {
        if let raw = environment["MTPLX_APP_REQUIRED_RUNTIME_VERSION"],
           let version = MTPLXSemanticVersion(raw) {
            return version
        }
        if Bundle.main.bundleURL.pathExtension == "app",
           let raw = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
           let version = MTPLXSemanticVersion(raw) {
            return version
        }
        return nil
    }

    private func runtime(_ executable: URL, satisfies minimumVersion: MTPLXSemanticVersion?) -> Bool {
        guard let minimumVersion else { return true }
        guard let raw = MTPLXRuntimeUpdateService.runtimeVersion(
            executableURL: executable,
            environment: environment
        ),
            let current = MTPLXSemanticVersion(raw)
        else { return false }
        return current >= minimumVersion
    }

    func resolvePythonExecutable() throws -> URL {
        if let explicit = environment["MTPLX_APP_PYTHON_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !explicit.isEmpty,
           FileManager.default.isExecutableFile(atPath: explicit),
           pythonVersionOK(URL(fileURLWithPath: explicit)) {
            return URL(fileURLWithPath: explicit)
        }

        // The interpreter shipped in Contents/Resources/PythonRuntime wins
        // over anything on the system: it is version-pinned, signed with
        // the app, and exists on Macs with no Homebrew or Xcode at all. A
        // venv built from it self-heals after app moves/updates via the
        // existing version-floor reinstall.
        if let bundled = MTPLXCommandBuilder.bundledPythonExecutablePath(
            environment: environment
        ) {
            let url = URL(fileURLWithPath: bundled)
            if pythonVersionOK(url) {
                return url
            }
        }

        let names = ["python3.14", "python3.13", "python3.12", "python3.11", "python3"]
        let fixedPaths = [
            "/opt/homebrew/bin/python3.14",
            "/opt/homebrew/bin/python3.13",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.14",
            "/usr/local/bin/python3.13",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3.11",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
        ]
        for path in fixedPaths {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.isExecutableFile(atPath: path), pythonVersionOK(url) {
                return url
            }
        }
        for name in names {
            for directory in MTPLXCommandBuilder.expandedPATH(environment: environment).split(separator: ":").map(String.init) {
                let url = URL(fileURLWithPath: directory).appendingPathComponent(name)
                if FileManager.default.isExecutableFile(atPath: url.path), pythonVersionOK(url) {
                    return url
                }
            }
        }
        throw MTPLXRuntimeBootstrapperError.pythonNotFound
    }

    private func pythonVersionOK(_ executable: URL) -> Bool {
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--version"]
        process.environment = MTPLXCommandBuilder.appSubprocessEnvironment(environment: environment)
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        do {
            try process.run()
        } catch {
            return false
        }
        // A version probe must never wedge the install path (#158): a
        // hung interpreter (Gatekeeper stall, dead NFS home) is treated
        // as "not usable", not waited on forever.
        if finished.wait(timeout: .now() + 15) == .timedOut {
            process.terminate()
            _ = finished.wait(timeout: .now() + 5)
            return false
        }
        var data = stdout.fileHandleForReading.readDataToEndOfFile()
        data.append(stderr.fileHandleForReading.readDataToEndOfFile())
        let output = String(data: data, encoding: .utf8) ?? ""
        guard let version = MTPLXSemanticVersion(output) else { return false }
        return version >= MTPLXSemanticVersion("3.11")!
    }
}

// SubprocessTailBuffer moved to SubprocessSupport.swift, the shared
// home of the app's watchdogged-subprocess plumbing (tail buffer,
// deadline watchdog, lossless pipe drain).
