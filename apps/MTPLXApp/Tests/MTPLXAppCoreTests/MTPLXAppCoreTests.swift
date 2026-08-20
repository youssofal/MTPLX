import XCTest
import Darwin
import AppKit
import CoreGraphics
import CoreText
@testable import MTPLXAppCore

private actor FanFallbackProbe {
    private(set) var calls = 0

    func restore() -> Bool {
        calls += 1
        return true
    }

    func count() -> Int {
        calls
    }
}

private final class StatusCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var messages: [String] = []

    func append(_ message: String) {
        lock.lock()
        messages.append(message)
        lock.unlock()
    }

    func snapshot() -> [String] {
        lock.lock()
        let copy = messages
        lock.unlock()
        return copy
    }
}

final class MTPLXAppCoreTests: XCTestCase {
    func testReleaseManifestParsesStableUpdateMetadata() throws {
        let data = """
        {
          "app_version": "1.0.0",
          "app_build": "10000",
          "minimum_cli_version": "1.0.0",
          "recommended_cli_version": "1.0.0",
          "dmg_url": "https://github.com/youssofal/mtplx/releases/download/v1.0.0/MTPLX-1.0.0.dmg",
          "dmg_sha256": "abc123",
          "pypi_version": "1.0.0",
          "homebrew_formula_version": "1.0.0",
          "release_notes_url": "https://mtplx.com/releases/notes/v1.0.0.html",
          "published_at": "2026-06-08T12:00:00Z"
        }
        """.data(using: .utf8)!

        let manifest = try MTPLXReleaseManifest.decode(data)

        XCTAssertEqual(manifest.appVersion, "1.0.0")
        XCTAssertEqual(manifest.appBuild, "10000")
        XCTAssertEqual(manifest.minimumCLIVersion, "1.0.0")
        XCTAssertEqual(manifest.recommendedCLIVersion, "1.0.0")
        XCTAssertEqual(manifest.dmgSHA256, "abc123")
        XCTAssertEqual(manifest.releaseNotesURL.absoluteString, "https://mtplx.com/releases/notes/v1.0.0.html")
    }

    func testSemanticVersionComparisonUsesNumericOrdering() throws {
        let old = try XCTUnwrap(MTPLXSemanticVersion("mtplx 0.3.8 (0.3.8)"))
        let release = try XCTUnwrap(MTPLXSemanticVersion("v1.0.0"))
        let patch = try XCTUnwrap(MTPLXSemanticVersion("1.0.1"))

        XCTAssertLessThan(old, release)
        XCTAssertLessThan(release, patch)
        XCTAssertEqual(MTPLXSemanticVersion("1.0"), MTPLXSemanticVersion("1.0.0"))
    }

    func testSessionBankEffectiveCacheHitUsesPrefixDiagnostic() throws {
        let data = """
        {
          "last_restore_source": "ram",
          "last_prefix_diagnostic": {
            "prompt_len": 8304,
            "stored_prefix_len": 5767,
            "common_prefix_tokens": 2334,
            "nearest_boundary_tokens": 2304,
            "new_prefill_tokens": 6000,
            "miss_reason": null,
            "restore_kind": "block_prefix_clone",
            "cache_source": "ram"
          },
          "cold_tier": {
            "restore_hits": 4,
            "restore_misses": 1,
            "restore_failures": 0
          }
        }
        """.data(using: .utf8)!

        let bank = try JSONDecoder().decode(SessionBank.self, from: data)

        XCTAssertEqual(bank.lastEffectiveCachedTokens, 2304)
        XCTAssertEqual(bank.restoreHitCount, 4)
        XCTAssertEqual(bank.lastEffectiveCacheSource, "ram")
        XCTAssertTrue(bank.hasEffectiveCacheHit)
    }

    func testSessionBankEffectiveCacheHitIgnoresMissDiagnostic() throws {
        let data = """
        {
          "last_restore_source": "none",
          "last_prefix_diagnostic": {
            "common_prefix_tokens": 2334,
            "nearest_boundary_tokens": 2304,
            "miss_reason": "new_session",
            "cache_source": "none"
          },
          "cold_tier": {
            "restore_hits": 0,
            "restore_misses": 2,
            "restore_failures": 0
          }
        }
        """.data(using: .utf8)!

        let bank = try JSONDecoder().decode(SessionBank.self, from: data)

        XCTAssertNil(bank.lastEffectiveCachedTokens)
        XCTAssertNil(bank.lastEffectiveCacheSource)
        XCTAssertFalse(bank.hasEffectiveCacheHit)
    }

    func testRuntimePolicyUpdatesOldHomebrewBelowMinimum() throws {
        let manifest = releaseManifest(minimumCLI: "1.0.0", recommendedCLI: "1.0.0")
        let action = MTPLXRuntimeUpdateService.action(
            version: "0.3.8",
            installKind: .homebrew,
            manifest: manifest,
            hasHomebrew: true
        )

        XCTAssertEqual(action, .updateHomebrewRequired)
    }

    func testRuntimePolicyDoesNotOverwriteOldSourceCheckout() throws {
        let manifest = releaseManifest(minimumCLI: "1.0.0", recommendedCLI: "1.0.0")
        let action = MTPLXRuntimeUpdateService.action(
            version: "0.3.8",
            installKind: .sourceCheckout,
            manifest: manifest,
            hasHomebrew: true
        )

        XCTAssertEqual(action, .manualUpdateRequired(command: "brew install youssofal/mtplx/mtplx"))
    }

    func testRuntimePolicyReinstallsAppOwnedRuntimeBelowMinimum() throws {
        let manifest = releaseManifest(minimumCLI: "1.0.0", recommendedCLI: "1.0.0")
        let action = MTPLXRuntimeUpdateService.action(
            version: "0.3.8",
            installKind: .appOwned,
            manifest: manifest,
            hasHomebrew: false
        )

        XCTAssertEqual(action, .updateBundledRequired)
    }

    func testInstallKindRecognizesAppOwnedRuntimeVenv() throws {
        let home = temporaryDirectory()
        let environment = ["HOME": home.path]
        let appBin = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(environment: environment))
        try FileManager.default.createDirectory(at: appBin, withIntermediateDirectories: true)
        let appRuntime = appBin.appendingPathComponent("mtplx")
        try "#!/bin/sh\necho 'mtplx 1.0.0 (1.0.0)'\n".data(using: .utf8)!.write(to: appRuntime)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: appRuntime.path)

        XCTAssertEqual(
            MTPLXRuntimeUpdateService.installKind(for: appRuntime, environment: environment),
            .appOwned
        )
    }

    func testDetectGlobalCLIIgnoresAppOwnedRuntime() throws {
        let home = temporaryDirectory()
        let environment = ["HOME": home.path]
        let appBin = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(environment: environment))
        try FileManager.default.createDirectory(at: appBin, withIntermediateDirectories: true)
        let appRuntime = appBin.appendingPathComponent("mtplx")
        try "#!/bin/sh\necho 'mtplx 1.0.0 (1.0.0)'\n".data(using: .utf8)!.write(to: appRuntime)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: appRuntime.path)
        let emptyPathDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyPathDir, withIntermediateDirectories: true)

        XCTAssertNil(
            MTPLXCommandBuilder.detectGlobalCLIExecutable(environment: [
                "HOME": home.path,
                "PATH": emptyPathDir.path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ]),
            "The app-owned venv must never be reported as the user's global CLI"
        )

        let globalCLI = try makeExecutable(
            named: "mtplx",
            body: "#!/bin/sh\necho 'mtplx 0.3.7 (0.3.7)'\n"
        )
        let detected = MTPLXCommandBuilder.detectGlobalCLIExecutable(environment: [
            "HOME": home.path,
            "PATH": globalCLI.deletingLastPathComponent().path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
        ])
        XCTAssertEqual(detected?.path, globalCLI.path)
    }

    /// Manifest-live + a stale pip-like CLI on PATH must install the
    /// bundled wheel instead of throwing "manual update required" —
    /// the pre-existing global CLI must never block the app.
    func testPrepareRuntimeForLaunchInstallsBundledWheelInsteadOfManualUpdate() async throws {
        let home = temporaryDirectory()
        let staleDir = home.appendingPathComponent("stale-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: staleDir, withIntermediateDirectories: true)
        let stale = staleDir.appendingPathComponent("mtplx")
        try "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'mtplx 0.3.7 (0.3.7)'; exit 0; fi\nexit 1\n"
            .data(using: .utf8)!.write(to: stale)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stale.path)

        let environment = try bundledWheelEnvironment(
            home: home,
            extraPath: staleDir.path,
            fakeInstallKind: "pipLike"
        )
        let manifestURL = try writeManifestFile(in: home, minimumCLI: "1.0.0", recommendedCLI: "1.0.0")

        let service = MTPLXRuntimeUpdateService(manifestURL: manifestURL, environment: environment)
        let executable = try await service.prepareRuntimeForLaunch()

        XCTAssertTrue(
            executable.path.hasSuffix("/Library/Application Support/MTPLX/runtime-venv/bin/mtplx"),
            executable.path
        )
        XCTAssertEqual(
            MTPLXRuntimeUpdateService.runtimeVersion(executableURL: executable, environment: environment),
            "1.0.0"
        )
    }

    /// The Mac Mini tune-failure shape: a stale user CLI on PATH that
    /// REPORTS a satisfying version (an old dev pip install claiming
    /// 1.0.0) must never become the engine while the bundle ships a
    /// wheel — adopting it would skip creating the app venv entirely
    /// and every model load would run in an environment the app never
    /// installed.
    func testPrepareRuntimeForLaunchIgnoresVersionSatisfyingPATHCLIWhenWheelBundled() async throws {
        let home = temporaryDirectory()
        let staleDir = home.appendingPathComponent("stale-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: staleDir, withIntermediateDirectories: true)
        let stale = staleDir.appendingPathComponent("mtplx")
        try "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'mtplx 1.0.0 (1.0.0)'; exit 0; fi\nexit 1\n"
            .data(using: .utf8)!.write(to: stale)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stale.path)

        let environment = try bundledWheelEnvironment(
            home: home,
            extraPath: staleDir.path,
            fakeInstallKind: "pipLike"
        )
        let manifestURL = try writeManifestFile(in: home, minimumCLI: "1.0.0", recommendedCLI: "1.0.0")

        let service = MTPLXRuntimeUpdateService(manifestURL: manifestURL, environment: environment)
        let executable = try await service.prepareRuntimeForLaunch()

        XCTAssertTrue(
            executable.path.hasSuffix("/Library/Application Support/MTPLX/runtime-venv/bin/mtplx"),
            "engine must be the app venv, got \(executable.path)"
        )
    }

    /// Transformers-style errors put the real explanation on the lines
    /// below "ImportError:" — the tune failure card must carry those
    /// lines, not a bare exception name (what the Mac Mini showed).
    func testTuneDiagnosticCapturesMultilineExceptionMessage() throws {
        let log = temporaryDirectory().appendingPathComponent("candidate.log")
        try FileManager.default.createDirectory(
            at: log.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try """
        Loading model shards...
        Traceback (most recent call last):
          File "loader.py", line 12, in load
        ImportError:
        GemmaTokenizer requires the SentencePiece library but it was not found in your environment.
        Install it with `pip install sentencepiece`.

        unrelated trailing output
        """.data(using: .utf8)!.write(to: log)

        let diagnostic = AutoTuner.diagnosticLine(fromLogAt: log.path)

        XCTAssertEqual(
            diagnostic,
            """
            ImportError:
            GemmaTokenizer requires the SentencePiece library but it was not found in your environment.
            Install it with `pip install sentencepiece`.
            """
        )
    }

    /// Manifest-live + no runtime + no Homebrew must still install the
    /// bundled wheel (previously threw "Homebrew required" without
    /// trying the wheel sitting in the app bundle).
    func testPrepareRuntimeForLaunchInstallsBundledWheelWhenNoRuntimeAndNoHomebrew() async throws {
        let home = temporaryDirectory()
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        var environment = try bundledWheelEnvironment(home: home, extraPath: emptyDir.path)
        environment["MTPLX_APP_HOMEBREW_PATH"] = ""
        let manifestURL = try writeManifestFile(in: home, minimumCLI: "1.0.0", recommendedCLI: "1.0.0")

        let service = MTPLXRuntimeUpdateService(manifestURL: manifestURL, environment: environment)
        let executable = try await service.prepareRuntimeForLaunch()

        XCTAssertTrue(
            executable.path.hasSuffix("/Library/Application Support/MTPLX/runtime-venv/bin/mtplx"),
            executable.path
        )
    }

    /// A stale app-owned venv (e.g. right after a Sparkle app update
    /// shipped a newer bundled wheel) refreshes from the wheel even
    /// when the published manifest floor would have accepted it.
    func testPrepareRuntimeForLaunchRefreshesStaleAppOwnedRuntime() async throws {
        let home = temporaryDirectory()
        let environment = try bundledWheelEnvironment(home: home, extraPath: nil)
        let appBin = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(environment: environment))
        try FileManager.default.createDirectory(at: appBin, withIntermediateDirectories: true)
        let staleVenvCLI = appBin.appendingPathComponent("mtplx")
        try "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'mtplx 0.9.0 (0.9.0)'; exit 0; fi\nexit 1\n"
            .data(using: .utf8)!.write(to: staleVenvCLI)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: staleVenvCLI.path)
        // Manifest floor would accept 0.9.0 — the app-version floor must win.
        let manifestURL = try writeManifestFile(in: home, minimumCLI: "0.9.0", recommendedCLI: "0.9.0")

        let service = MTPLXRuntimeUpdateService(manifestURL: manifestURL, environment: environment)
        let executable = try await service.prepareRuntimeForLaunch()

        XCTAssertEqual(
            MTPLXRuntimeUpdateService.runtimeVersion(executableURL: executable, environment: environment),
            "1.0.0",
            "The stale app-owned venv must be reinstalled from the bundled wheel"
        )
    }

    /// Shared fixture: bundled wheel + fake python that "installs" a
    /// 1.0.0 mtplx into the venv it creates (same shape as
    /// `testRuntimeBootstrapperRepairsStaleRuntimeFromBundledWheel`).
    private func bundledWheelEnvironment(
        home: URL,
        extraPath: String?,
        fakeInstallKind: String? = nil
    ) throws -> [String: String] {
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let wheel = home.appendingPathComponent("mtplx-1.0.0-py3-none-any.whl")
        try Data("fake wheel".utf8).write(to: wheel)
        let log = home.appendingPathComponent("runtime-install.log")
        let fakePython = home.appendingPathComponent("fake-python")
        try """
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        if [ "$1" = "--version" ]; then
          echo "Python 3.13.0"
          exit 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
          venv="$3"
          if [ "$venv" = "--clear" ]; then
            venv="$4"
            rm -rf "$venv"
          fi
          mkdir -p "$venv/bin"
          cat > "$venv/bin/python" <<'PYTHON'
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        if [ "$1" = "--version" ]; then
          echo "Python 3.13.0"
          exit 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
          case "$*" in
            *mtplx-1.0.0-py3-none-any.whl*)
              cat > "$(dirname "$0")/mtplx" <<'MTPLX'
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx 1.0.0 (1.0.0)"
          exit 0
        fi
        echo ok
        MTPLX
              chmod +x "$(dirname "$0")/mtplx"
              ;;
          esac
          exit 0
        fi
        exit 0
        PYTHON
          chmod +x "$venv/bin/python"
          exit 0
        fi
        exit 1
        """.data(using: .utf8)!.write(to: fakePython)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakePython.path)

        var environment: [String: String] = [
            "HOME": home.path,
            "PATH": (extraPath.map { "\($0):" } ?? "") + "/usr/bin:/bin",
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            "MTPLX_APP_REQUIRED_RUNTIME_VERSION": "1.0.0",
            "MTPLX_BUNDLED_RUNTIME_WHEEL": wheel.path,
            "MTPLX_APP_PYTHON_PATH": fakePython.path,
            "MTPLX_FAKE_LOG": log.path,
        ]
        if let fakeInstallKind {
            environment["MTPLX_APP_FAKE_INSTALL_KIND"] = fakeInstallKind
        }
        return environment
    }

    private func writeManifestFile(
        in directory: URL,
        minimumCLI: String,
        recommendedCLI: String
    ) throws -> URL {
        let manifest = """
        {
          "app_version": "1.0.0",
          "app_build": "10000",
          "minimum_cli_version": "\(minimumCLI)",
          "recommended_cli_version": "\(recommendedCLI)",
          "dmg_url": "https://github.com/youssofal/mtplx/releases/download/v1.0.0/MTPLX-1.0.0.dmg",
          "dmg_sha256": "abc123",
          "pypi_version": "1.0.0",
          "homebrew_formula_version": "1.0.0",
          "release_notes_url": "https://mtplx.com/releases/notes/v1.0.0.html"
        }
        """
        let url = directory.appendingPathComponent("latest.json")
        try manifest.data(using: .utf8)!.write(to: url)
        return url
    }

    func testRuntimePolicyAllowsCompatibleCLIWhenManifestIsUnavailable() throws {
        let fake = try makeExecutable(
            named: "mtplx",
            body: "#!/bin/sh\necho 'mtplx 1.0.0 (1.0.0)'\n"
        )
        let snapshot = MTPLXRuntimeUpdateService.snapshot(
            manifest: nil,
            environment: [
                "PATH": fake.deletingLastPathComponent().path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ]
        )

        XCTAssertEqual(snapshot.action, .useExisting)
        XCTAssertEqual(snapshot.cliVersion, "1.0.0")
        XCTAssertEqual(snapshot.title, "Runtime ready")
    }

    func testRuntimePolicyOffersInstallWhenRuntimeIsMissingAndHomebrewExists() throws {
        let root = temporaryDirectory()
        let fakeBin = root.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let fakeBrew = try makeExecutable(named: "brew")

        let snapshot = MTPLXRuntimeUpdateService.snapshot(
            manifest: releaseManifest(minimumCLI: "1.0.0", recommendedCLI: "1.0.0"),
            environment: [
                "PATH": fakeBin.path,
                // Isolate from the developer machine's real app-owned
                // runtime venv under ~/Library/Application Support/MTPLX.
                "HOME": root.path,
                "MTPLX_APP_HOMEBREW_PATH": fakeBrew.path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ]
        )

        XCTAssertEqual(snapshot.action, .installHomebrew)
        XCTAssertEqual(snapshot.cliInstallKind, .missing)
        XCTAssertTrue(snapshot.canUpdateRuntime)
    }

    func testCommandBuilderPrefersAppOwnedRuntimeBeforeHomebrewPath() throws {
        let home = temporaryDirectory()
        let environment = ["HOME": home.path]
        let appBin = URL(fileURLWithPath: MTPLXCommandBuilder.appRuntimeBinDirectory(environment: environment))
        try FileManager.default.createDirectory(at: appBin, withIntermediateDirectories: true)
        let appRuntime = appBin.appendingPathComponent("mtplx")
        try "#!/bin/sh\necho app runtime\n".data(using: .utf8)!.write(to: appRuntime)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: appRuntime.path)
        let staleHomebrew = try makeExecutable(
            named: "mtplx",
            body: "#!/bin/sh\necho 'mtplx 0.3.7 (0.3.7)'\n"
        )

        let resolved = try MTPLXCommandBuilder.resolveInstalledExecutable(environment: [
            "HOME": home.path,
            "PATH": staleHomebrew.deletingLastPathComponent().path,
        ])

        XCTAssertEqual(resolved.path, appRuntime.path)
    }

    func testRuntimeBootstrapperRepairsStaleRuntimeFromBundledWheel() throws {
        let home = temporaryDirectory()
        let fakeBin = home.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let staleRuntime = fakeBin.appendingPathComponent("mtplx")
        try """
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx 0.3.7 (0.3.7)"
          exit 0
        fi
        exit 1
        """.data(using: .utf8)!.write(to: staleRuntime)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: staleRuntime.path)

        let wheel = home.appendingPathComponent("mtplx-1.0.0-py3-none-any.whl")
        try Data("fake wheel".utf8).write(to: wheel)
        let log = home.appendingPathComponent("runtime-install.log")
        let fakePython = home.appendingPathComponent("fake-python")
        try """
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        if [ "$1" = "--version" ]; then
          echo "Python 3.13.0"
          exit 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
          venv="$3"
          if [ "$venv" = "--clear" ]; then
            venv="$4"
            rm -rf "$venv"
          fi
          mkdir -p "$venv/bin"
          cat > "$venv/bin/python" <<'PYTHON'
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        if [ "$1" = "--version" ]; then
          echo "Python 3.13.0"
          exit 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
          case "$*" in
            *mtplx-1.0.0-py3-none-any.whl*)
              cat > "$(dirname "$0")/mtplx" <<'MTPLX'
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx 1.0.0 (1.0.0)"
          exit 0
        fi
        echo ok
        MTPLX
              chmod +x "$(dirname "$0")/mtplx"
              ;;
          esac
          exit 0
        fi
        exit 0
        PYTHON
          chmod +x "$venv/bin/python"
          exit 0
        fi
        exit 1
        """.data(using: .utf8)!.write(to: fakePython)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakePython.path)

        let statuses = StatusCapture()
        let executable = try MTPLXRuntimeBootstrapper(environment: [
            "HOME": home.path,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            "MTPLX_APP_REQUIRED_RUNTIME_VERSION": "1.0.0",
            "MTPLX_BUNDLED_RUNTIME_WHEEL": wheel.path,
            "MTPLX_APP_PYTHON_PATH": fakePython.path,
            "MTPLX_FAKE_LOG": log.path,
        ]).installOrUpdate { statuses.append($0) }

        XCTAssertEqual(statuses.snapshot(), ["Checking MTPLX runtime", "Installing MTPLX runtime"])
        XCTAssertTrue(executable.path.hasSuffix("/Library/Application Support/MTPLX/runtime-venv/bin/mtplx"))
        XCTAssertEqual(MTPLXRuntimeUpdateService.runtimeVersion(executableURL: executable), "1.0.0")
        let installLog = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(installLog.contains("-m venv"), installLog)
        XCTAssertTrue(installLog.contains("mtplx-1.0.0-py3-none-any.whl[server]"), installLog)
    }

    // Issue #139: after an app update, the runtime venv's bin/python3
    // symlink chain can dangle (it pointed into the replaced bundle),
    // and a plain `python -m venv` over the corpse exits 1 with
    // "[Errno 2] No such file or directory: …/bin/python3". The venv
    // lives in Application Support, so DMG reinstalls never fix it —
    // the bootstrapper must rebuild with --clear itself.
    private func makeVenvFakePython(
        home: URL,
        log: URL,
        failPlainVenv: Bool = false
    ) throws -> URL {
        try FileManager.default.createDirectory(
            at: home, withIntermediateDirectories: true
        )
        let fakePython = home.appendingPathComponent("fake-python")
        let plainVenvExit = failPlainVenv ? 1 : 0
        try """
        #!/bin/sh
        echo "$*" >> "\(log.path)"
        if [ "$1" = "--version" ]; then
          echo "Python 3.13.0"
          exit 0
        fi
        if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
          if [ "$3" = "--clear" ]; then
            exit 0
          fi
          exit \(plainVenvExit)
        fi
        exit 0
        """.data(using: .utf8)!.write(to: fakePython)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: fakePython.path
        )
        return fakePython
    }

    func testRuntimeBootstrapperRebuildsBrokenVenvWithClear() throws {
        let home = temporaryDirectory()
        let log = home.appendingPathComponent("venv.log")
        let fakePython = try makeVenvFakePython(home: home, log: log)
        let runtimeDir = home.appendingPathComponent("runtime-venv")
        let bin = runtimeDir.appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        // Dangling symlink: the interpreter the venv was built from is gone.
        try FileManager.default.createSymbolicLink(
            at: bin.appendingPathComponent("python3"),
            withDestinationURL: home.appendingPathComponent("gone/python3.14")
        )

        let bootstrapper = MTPLXRuntimeBootstrapper(environment: ["HOME": home.path])
        try bootstrapper.createRuntimeVenv(python: fakePython, runtimeDir: runtimeDir)

        let calls = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(calls.contains("-m venv --clear \(runtimeDir.path)"), calls)
    }

    func testRuntimeBootstrapperRetriesVenvCreationWithClearOnFailure() throws {
        let home = temporaryDirectory()
        let log = home.appendingPathComponent("venv.log")
        let fakePython = try makeVenvFakePython(home: home, log: log, failPlainVenv: true)
        let runtimeDir = home.appendingPathComponent("runtime-venv")

        let bootstrapper = MTPLXRuntimeBootstrapper(environment: ["HOME": home.path])
        try bootstrapper.createRuntimeVenv(python: fakePython, runtimeDir: runtimeDir)

        let calls = try String(contentsOf: log, encoding: .utf8)
            .split(separator: "\n").map(String.init)
        XCTAssertEqual(calls, [
            "-m venv \(runtimeDir.path)",
            "-m venv --clear \(runtimeDir.path)",
        ])
    }

    func testRuntimeBootstrapperKeepsPlainVenvPathWhenHealthy() throws {
        let home = temporaryDirectory()
        let log = home.appendingPathComponent("venv.log")
        let fakePython = try makeVenvFakePython(home: home, log: log)
        let runtimeDir = home.appendingPathComponent("runtime-venv")
        let bin = runtimeDir.appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let venvPython = bin.appendingPathComponent("python")
        try "#!/bin/sh\nexit 0\n".data(using: .utf8)!.write(to: venvPython)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: venvPython.path
        )

        let bootstrapper = MTPLXRuntimeBootstrapper(environment: ["HOME": home.path])
        try bootstrapper.createRuntimeVenv(python: fakePython, runtimeDir: runtimeDir)

        let calls = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(calls.contains("-m venv \(runtimeDir.path)"), calls)
        XCTAssertFalse(calls.contains("--clear"), calls)
    }

    func testCommandBuilderPrefersInstalledRuntimeOverSourceWrapper() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": temporaryDirectory().path,
            "MTPLX_APP_ALLOW_SOURCE_WRAPPER": "0",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                model: "/models/qwen",
                profile: "sustained"
            )
        )

        XCTAssertEqual(command.executableURL.path, fake.path)
    }

    func testAppSubprocessEnvironmentStripsDeveloperPythonAndMLXOverrides() throws {
        let fake = try makeExecutable(named: "mtplx")

        let env = MTPLXCommandBuilder.appSubprocessEnvironment(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": "/Users/example",
            "PYTHONPATH": "/dev/mlx/python",
            "PYTHONHOME": "/dev/python",
            "VIRTUAL_ENV": "/dev/venv",
            "DYLD_LIBRARY_PATH": "/dev/lib",
            "PYTHONPYCACHEPREFIX": "/Applications/MTPLX.app/Contents/Resources/PythonRuntime/cache",
            "MTPLX_FAST_MLX_SOURCE_PATH_ACTIVE": "/dev/mlx/python",
            "MTPLX_APP_SOURCE_WRAPPER_PATH": "/dev/repo/bin/mtplx",
            "MTPLX_SESSION_BANK_MAX_ENTRIES": "16",
        ])

        XCTAssertNil(env["PYTHONPATH"])
        XCTAssertNil(env["PYTHONHOME"])
        XCTAssertNil(env["VIRTUAL_ENV"])
        XCTAssertNil(env["DYLD_LIBRARY_PATH"])
        XCTAssertNil(env["MTPLX_FAST_MLX_SOURCE_PATH_ACTIVE"])
        XCTAssertNil(env["MTPLX_APP_SOURCE_WRAPPER_PATH"])
        XCTAssertEqual(env["MTPLX_DISABLE_FAST_MLX_AUTODISCOVERY"], "1")
        XCTAssertEqual(
            env["PYTHONPYCACHEPREFIX"],
            "/Users/example/Library/Caches/MTPLX/PythonBytecode"
        )
        XCTAssertEqual(env["MTPLX_SESSION_BANK_MAX_ENTRIES"], "16")
        XCTAssertTrue(env["PATH"]?.contains(fake.deletingLastPathComponent().path) ?? false)
    }

    func testPythonBytecodeSafeEnvironmentPreservesCallerVariables() {
        let env = MTPLXCommandBuilder.pythonBytecodeSafeEnvironment(environment: [
            "HOME": "/Users/example",
            "PYTHONPATH": "/forge/python",
            "HF_TOKEN": "fixture-token",
            "PYTHONPYCACHEPREFIX": "/Applications/MTPLX.app/Contents/Resources/cache",
        ])

        XCTAssertEqual(env["PYTHONPATH"], "/forge/python")
        XCTAssertEqual(env["HF_TOKEN"], "fixture-token")
        XCTAssertEqual(
            env["PYTHONPYCACHEPREFIX"],
            "/Users/example/Library/Caches/MTPLX/PythonBytecode"
        )
    }

    func testHardwareInspectorRoutesPythonBytecodeOutsideSignedBundle() async throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let cacheLog = root.appendingPathComponent("pycache-prefix.log")
        let script = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            printf '%s' "$PYTHONPYCACHEPREFIX" > "$MTPLX_FAKE_LOG"
            printf '%s\n' '{"chip":"Apple M5 Max","apple_silicon_generation":"m5","model_identifier":"Mac16,1","unified_memory_bytes":137438953472,"gpu_cores":40,"cpu_cores":18}'
            """
        )
        let inspector = HardwareInspector(processEnvironment: [
            "HOME": root.path,
            "PATH": script.deletingLastPathComponent().path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            "MTPLX_FAKE_LOG": cacheLog.path,
        ])

        let hardware = await inspector.detect()

        XCTAssertEqual(hardware.chipName, "Apple M5 Max")
        XCTAssertEqual(
            try String(contentsOf: cacheLog, encoding: .utf8),
            root.appendingPathComponent("Library/Caches/MTPLX/PythonBytecode").path
        )
    }

    // MARK: - Launch-critical config sanitization (degraded-on-start class)

    private func decodeConfiguration(json: String) throws -> MTPLXAppConfiguration {
        try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
    }

    func testPersistedLegacyProfileStringsDecodeLaunchable() throws {
        // "auto" is a first-class persisted value since the 2026-07-03
        // turbo release (resolved per-model before argv).
        let auto = try decodeConfiguration(json: #"{"profile": "auto"}"#)
        XCTAssertEqual(auto.profile, "auto")

        // Junk normalizes straight to auto — never to a concrete profile
        // the user did not pick. The launch then omits --profile and the
        // engine resolves the per-model default.
        let unknown = try decodeConfiguration(
            json: #"{"profile": "banana", "generation_mode": "auto"}"#
        )
        XCTAssertEqual(unknown.profile, "auto")
        XCTAssertEqual(unknown.generationMode, "mtp")

        // Junk after the legacy migration must also land on auto, not on
        // a silent sustained (the pre-F20 behavior).
        let unknownMigrated = try decodeConfiguration(
            json: #"{"profile": "banana", "profile_legacy_default_migrated": true}"#
        )
        XCTAssertEqual(unknownMigrated.profile, "auto")

        // A post-migration explicit sustained is preserved verbatim.
        let chosen = try decodeConfiguration(
            json: #"{"profile": "sustained", "profile_legacy_default_migrated": true}"#
        )
        XCTAssertEqual(chosen.profile, "sustained")
    }

    func testPersistedSustainedMaxMigratesToSustainedPlusMaxFans() throws {
        let config = try decodeConfiguration(json: #"{"profile": "sustained-max"}"#)

        // sustained-max meant "fastest + pinned fans"; the profile half
        // now lands on auto (recommended per model — turbo for the
        // 27Bs), and the fan intent survives as before.
        XCTAssertEqual(config.profile, "auto")
        XCTAssertEqual(config.fanMode, MTPLXFanMode.max.rawValue)
        XCTAssertTrue(config.pinFansAtMaxOnStart)
    }

    func testServeCommandNeverEmitsUnlaunchableProfileOrMode() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": temporaryDirectory().path,
        ])
        var configuration = MTPLXAppConfiguration(
            model: "/models/qwen",
            profile: "sustained"
        )
        // In-memory state as an older build could hold it, before any
        // decode-time sanitization has a chance to run.
        configuration.profile = "auto"
        configuration.generationMode = "auto"

        let command = try builder.buildServeCommand(configuration: configuration)

        let arguments = command.arguments
        // "auto" (and anything not engine-launchable) emits NO --profile:
        // the engine owns default-profile resolution. The old behavior —
        // coercing to an explicit "--profile sustained" — blocked the
        // engine's per-artifact turbo promotion (F20).
        XCTAssertFalse(arguments.contains("--profile"), arguments.joined(separator: " "))
        XCTAssertFalse(arguments.contains("sustained"), arguments.joined(separator: " "))
        XCTAssertFalse(
            arguments.contains("--generation-mode"),
            arguments.joined(separator: " ")
        )
    }

    // MARK: - HF download mirror (issue #96)

    func testHFMirrorEnvironmentSetsEndpointAndWithholdsTokens() throws {
        let env = try XCTUnwrap(
            MTPLXAppConfiguration.hfMirrorEnvironment("https://hf-mirror.com")
        )

        XCTAssertEqual(env["HF_ENDPOINT"], "https://hf-mirror.com")
        XCTAssertEqual(env["HF_TOKEN"], "")
        XCTAssertEqual(env["HUGGING_FACE_HUB_TOKEN"], "")
    }

    func testHFMirrorEnvironmentRejectsInvalidAndOfficialEndpoints() {
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment(nil))
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment(""))
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment("   "))
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment("not a url"))
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment("ftp://hf-mirror.com"))
        // The official host is not a mirror; tokens must keep working.
        XCTAssertNil(MTPLXAppConfiguration.hfMirrorEnvironment("https://huggingface.co"))
    }

    func testServeCommandCarriesMirrorEnvironmentWhenConfigured() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": temporaryDirectory().path,
        ])
        var configuration = MTPLXAppConfiguration(
            model: "/models/qwen",
            profile: "sustained"
        )
        configuration.hfEndpoint = "https://hf-mirror.com"

        let command = try builder.buildServeCommand(configuration: configuration)

        XCTAssertEqual(command.environment["HF_ENDPOINT"], "https://hf-mirror.com")
        XCTAssertEqual(command.environment["HF_TOKEN"], "")

        configuration.hfEndpoint = nil
        let plain = try builder.buildServeCommand(configuration: configuration)
        XCTAssertNil(plain.environment["HF_ENDPOINT"])
        XCTAssertNil(plain.environment["HF_TOKEN"])
    }

    func testOnboardingDownloadFailureCopySuggestsMirrorOnlyForNetworkFailures() {
        let blocked = OnboardingOrchestrator.downloadFailureMessage(
            stderrTail: "ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded",
            mirrorActive: false
        )
        XCTAssertTrue(blocked.contains("download mirror"), blocked)

        let alreadyMirrored = OnboardingOrchestrator.downloadFailureMessage(
            stderrTail: "ConnectionError: Max retries exceeded",
            mirrorActive: true
        )
        XCTAssertFalse(alreadyMirrored.contains("download mirror"), alreadyMirrored)

        let disk = OnboardingOrchestrator.downloadFailureMessage(
            stderrTail: "OSError: No space left on device",
            mirrorActive: false
        )
        XCTAssertFalse(disk.contains("download mirror"), disk)
    }

    func testFailureIndicatesPortConflictMatchesBothDetectionLayers() {
        XCTAssertTrue(
            MTPLXBackendStore.failureIndicatesPortConflict(
                DaemonSupervisorError.portOccupied(pid: 123, launchID: "abc")
            )
        )
        XCTAssertTrue(
            MTPLXBackendStore.failureIndicatesPortConflict(
                DaemonSupervisorError.launchFailed(
                    "daemon exited before /health became ready: error: port 8000 is already in use"
                )
            )
        )
        XCTAssertTrue(
            MTPLXBackendStore.failureIndicatesPortConflict(
                DaemonSupervisorError.launchFailed("[Errno 48] Address already in use")
            )
        )
        XCTAssertFalse(
            MTPLXBackendStore.failureIndicatesPortConflict(
                DaemonSupervisorError.launchFailed("ImportError: missing module")
            )
        )
        XCTAssertFalse(
            MTPLXBackendStore.failureIndicatesPortConflict(
                DaemonSupervisorError.healthTimeout
            )
        )
    }

    func testAppSubprocessEnvironmentStripsInheritedPipOverrides() throws {
        let fake = try makeExecutable(named: "mtplx")

        let env = MTPLXCommandBuilder.appSubprocessEnvironment(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": "/Users/example",
            "PIP_USER": "1",
            "PIP_INDEX_URL": "https://mirror.example/simple",
            "PIP_CONFIG_FILE": "/Users/example/.config/pip/pip.conf",
        ])

        XCTAssertNil(env["PIP_USER"])
        XCTAssertNil(env["PIP_INDEX_URL"])
        XCTAssertNil(env["PIP_CONFIG_FILE"])
    }

    /// The Mac Mini first-run failure: a user-level pip.conf with
    /// `user = true` makes every venv install abort with "Can not
    /// perform a '--user' install." The bootstrapper env must keep
    /// pip away from user configuration entirely.
    func testBootstrapperHermeticEnvironmentNeutralizesPipUserInstalls() {
        let env = MTPLXRuntimeBootstrapper.hermeticSubprocessEnvironment(from: [
            "PATH": "/usr/bin",
            "HOME": "/Users/example",
            "PIP_USER": "1",
        ])

        XCTAssertEqual(env["PIP_CONFIG_FILE"], "/dev/null")
        XCTAssertEqual(env["PIP_USER"], "0")
        XCTAssertEqual(env["PYTHONNOUSERSITE"], "1")
        XCTAssertEqual(env["PIP_DISABLE_PIP_VERSION_CHECK"], "1")
    }

    func testAutoTunerPrefersInstalledRuntimeOverSourceWrapper() throws {
        let fake = try makeExecutable(named: "mtplx")
        let resolved = try AutoTuner.resolveMtplxExecutable(
            env: [
                "PATH": fake.deletingLastPathComponent().path,
                "HOME": temporaryDirectory().path,
                "MTPLX_APP_ALLOW_SOURCE_WRAPPER": "0",
            ]
        )

        XCTAssertEqual(resolved.path, fake.path)
    }

    func testCommandBuilderSkipsSourceWrapperSymlinkByDefault() throws {
        let fake = try makeExecutable(named: "mtplx")
        let sourceWrapper = try XCTUnwrap(sourceTreeWrapper())
        let symlinkDirectory = temporaryDirectory()
        try FileManager.default.createDirectory(at: symlinkDirectory, withIntermediateDirectories: true)
        let symlink = symlinkDirectory.appendingPathComponent("mtplx")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: sourceWrapper)
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": "\(symlinkDirectory.path):\(fake.deletingLastPathComponent().path)",
            "HOME": temporaryDirectory().path,
            "MTPLX_APP_ALLOW_SOURCE_WRAPPER": "0",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                model: "/models/qwen",
                profile: "sustained"
            )
        )

        XCTAssertEqual(command.executableURL.path, fake.path)
    }

    func testCommandBuilderSkipsExplicitSourceWrapperByDefault() throws {
        let fake = try makeExecutable(named: "mtplx")
        let sourceWrapper = try XCTUnwrap(sourceTreeWrapper())
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": temporaryDirectory().path,
            "MTPLX_APP_ALLOW_SOURCE_WRAPPER": "0",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: sourceWrapper.path,
                model: "/models/qwen",
                profile: "sustained"
            )
        )

        XCTAssertEqual(command.executableURL.path, fake.path)
    }

    func testCommandBuilderEmitsServeArgsWithoutBrowserFlags() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                host: "127.0.0.1",
                port: 8123,
                apiKey: "secret",
                enableThermalPolling: true
            )
        )

        XCTAssertEqual(command.executableURL.path, fake.path)
        XCTAssertEqual(
            command.arguments,
            [
                "serve",
                "--host", "127.0.0.1",
                "--port", "8123",
                "--model", "/models/qwen",
                "--profile", "sustained",
                "--scheduler-mode", "serial",
                "--batching-preset", "latency",
                "--ssd-session-cache", "off",
                "--api-key", "secret",
                "--enable-thermal-poll",
                "--fan-mode", "smart",
                "--unsafe-force-unverified",
                "--yes",
                "--reasoning", "auto",
                "--no-stats-footer",
            ]
        )
        XCTAssertFalse(command.arguments.contains("--open-browser"))
        XCTAssertFalse(command.arguments.contains("--open-dashboard"))
    }

    func testCommandBuilderEmitsRestartRequiredRuntimeSettings() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                generationMode: "ar",
                loadMTP: false,
                contextWindow: 8192
            )
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--generation-mode", "ar"]))
        XCTAssertTrue(command.arguments.contains("--no-load-mtp"))
        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--context-window", "8192"]))
    }

    func testCommandBuilderUsesMeasuredQwen35BOptimizedSpeedDefaults() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
                profile: "sustained"
            )
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "1"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--verify-strategy", "target_prefix"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "local_qwen36"]))
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen27BOptimizedSpeed() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
                profile: "auto"
            )
        )

        // Auto emits no --profile: the engine owns the per-model default
        // and promotes this flagship to turbo itself (it sits in
        // _TURBO_DEFAULT_PUBLIC_MODEL_IDS; chat lane 44.7 -> 58-60 tok/s,
        // 2026-07-02). An app-side "--profile" here would read as an
        // explicit user pick and block that promotion.
        XCTAssertFalse(command.arguments.contains("--profile"))
        // Qwen3.6 thinking-mode spec sampler (0.6/0.95/20), same as the
        // 35B and Step presets — the 27B fell through to 1.0 until the
        // launch family existed (founder-confirmed 2026-07-02).
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen27BSpeedFP16Sibling() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
                profile: "auto"
            )
        )

        // No --profile on auto: the engine's turbo-default set carries the
        // FP16 sibling (promoted 2026-07-07 after its first e2e
        // measurement: 1.98-2.08x over true AR under turbo vs 1.34x on
        // sustained) and resolves it per-artifact even for renamed dirs.
        XCTAssertFalse(command.arguments.contains("--profile"))
        // It still gets the Qwen3.6 thinking-mode sampler preset.
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen27BOptimizedQuality() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
                profile: "auto"
            )
        )

        // No --profile on auto: 8-bit Quality is in the engine's
        // turbo-default set (q8 verify_kernels ULP-exact, +22-40% on the
        // chat lane, 2026-07-03) — the engine applies it without an
        // app-side pin.
        XCTAssertFalse(command.arguments.contains("--profile"))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen38Family() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        for model in [
            "/Users/example/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed",
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
            // FP16 precision siblings (M1/M2 routing targets) share the family.
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
            "/Users/example/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
        ] {
            let command = try builder.buildServeCommand(
                configuration: MTPLXAppConfiguration(
                    executablePath: fake.path,
                    model: model,
                    profile: "auto"
                )
            )
            // Qwen3.8 27B launches with no --profile: the whole family
            // (all six artifacts, FP16 siblings included) sits in the
            // engine's turbo-default set, and the engine resolves it
            // per-artifact. The app still pins the model card's official
            // thinking sampler — 1.0/0.95/20, NOT the 3.6-era 0.6 coding
            // triple. The draft sampler is left to the artifact's
            // `recommended_draft_sampler` stamp (the CLI's zero-flag
            // path), so app and CLI serve the same draft sampler per artifact.
            XCTAssertFalse(command.arguments.contains("--profile"), model)
            XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]), model)
            XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]), model)
            XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]), model)
            XCTAssertFalse(command.arguments.contains("--draft-temperature"), model)
            XCTAssertFalse(command.arguments.contains("--draft-top-p"), model)
            XCTAssertFalse(command.arguments.contains("--draft-top-k"), model)
            // reasoning_effort / preserve_thinking stay unpinned: the
            // server's qwen3_8 family policy owns them (measured coding
            // default medium, preserve).
            XCTAssertFalse(command.arguments.contains("--reasoning-effort"), model)
        }
    }

    func testQwen38FamilyDetectionAndTuneSupport() {
        for ref in [
            "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
            "/Users/example/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
            "mtplx-qwen38-27b-bare-speed",
            "Qwen/Qwen3.8-27B",
        ] {
            XCTAssertEqual(MTPLXModelOption.modelFamily(for: ref), "qwen3_8", ref)
        }
        XCTAssertTrue(MTPLXModelOption.supportsTune(family: "qwen3_8"))
        XCTAssertTrue(MTPLXModelOption.supportsOnboardingTune(family: "qwen3_8"))
        XCTAssertEqual(TuneCandidate.candidates(forFamily: "qwen3_8"), [.ar, .d1, .d2, .d3])
        // The 3.6 flagship must NOT be swallowed by the 3.8 branch.
        XCTAssertEqual(
            MTPLXModelOption.modelFamily(for: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2"),
            "qwen3_6"
        )
        XCTAssertNotEqual(
            MTPLXModelOption.modelFamily(for: "Qwen/Qwen3-8B"),
            "qwen3_8",
            "the 8B parameter count must not masquerade as the Qwen 3.8 version"
        )
        XCTAssertEqual(
            MTPLXCommandBuilder.recommendedProfile(
                for: "/Users/example/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed"
            ),
            "turbo"
        )
    }

    func testQwen38ExplicitXHighReasoningEffortReachesServeCommand() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
                profile: "auto",
                reasoningEffort: "xhigh"
            )
        )
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-effort", "xhigh"]))
    }

    func testOnboardingTuneUsesTurboForQwen27BOptimizedModels() {
        for model in [
            "/Users/example/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
            "/Users/example/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
        ] {
            XCTAssertEqual(
                MTPLXCommandBuilder.recommendedProfile(for: model),
                "turbo",
                model
            )
        }
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen27BQualityFP16Sibling() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
                profile: "auto"
            )
        )

        // No --profile on auto: the Quality-FP16 sibling (M1/M2 quality
        // pick, built 2026-07-07; 2.5x over true AR under turbo on its
        // own artifact) is in the engine's turbo-default set.
        XCTAssertFalse(command.arguments.contains("--profile"))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForQwen359BOptimizedSpeed() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        for model in [
            "/Users/youssof/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed",
            "/Users/youssof/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
        ] {
            let command = try builder.buildServeCommand(
                configuration: MTPLXAppConfiguration(
                    executablePath: fake.path,
                    model: model,
                    profile: "auto"
                )
            )
            // No --profile on auto: both 9B artifacts are in the engine's
            // turbo-default set (earned 2026-07-07 with the 6-bit hexpack
            // split-K kernels: live ABBA MTP D3 110/102 tok/s under turbo
            // vs 90/69 sustained).
            XCTAssertFalse(command.arguments.contains("--profile"), model)
            XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]), model)
        }
    }

    func testCommandBuilderAutoProfileEmitsNoFlagForUnrecognizedModelDirs() throws {
        // F20 regression: models the path-substring family table cannot
        // classify — the legacy gdn8 hybrid and renamed/symlinked dirs of
        // flagships — used to launch with an explicit "--profile
        // sustained", which the engine had to obey. With no flag the
        // engine resolves identity from the artifact (#268) and applies
        // its own per-model default.
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        for model in [
            // Legacy 27B Optimized (gdn8 hybrid) — no -Speed/-Quality
            // suffix, so ModelLaunchFamily.detect has no case for it even
            // though the engine's turbo set carries it.
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized",
            // A renamed local dir of a flagship artifact.
            "/Users/youssof/models/my-favorite-model",
        ] {
            let command = try builder.buildServeCommand(
                configuration: MTPLXAppConfiguration(
                    executablePath: fake.path,
                    model: model,
                    profile: "auto"
                )
            )
            XCTAssertFalse(command.arguments.contains("--profile"), model)
            XCTAssertFalse(command.arguments.contains("sustained"), model)
        }
    }

    func testCommandBuilderHonorsExplicitProfileOverPreset() throws {
        // The Settings picker must never lie: an explicit user choice
        // beats the per-model preset (2026-07-03 turbo release).
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
                profile: "sustained",
                profileLegacyDefaultMigrated: true
            )
        )
        XCTAssertTrue(command.arguments.containsInOrder(["--profile", "sustained"]))
    }

    func testSanitizeMigratesLegacyProfileToAutoOnce() throws {
        var legacy = MTPLXAppConfiguration()
        legacy.profile = "sustained"
        legacy.profileLegacyDefaultMigrated = false
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertEqual(legacy.profile, "auto")
        XCTAssertTrue(legacy.profileLegacyDefaultMigrated)

        // A deliberate post-migration Sustained pick sticks.
        legacy.profile = "sustained"
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertEqual(legacy.profile, "sustained")

        // "auto" itself survives sanitize (persistable, not launchable).
        var auto = MTPLXAppConfiguration()
        auto.sanitizeLaunchCriticalFields()
        XCTAssertEqual(auto.profile, "auto")
    }

    func testSanitizeMigratesStreamCadenceOnce() throws {
        var legacy = MTPLXAppConfiguration()
        legacy.streamSnapshotIntervalMs = 250
        legacy.streamCadenceMigrated = false
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertEqual(legacy.streamSnapshotIntervalMs, 100)
        XCTAssertTrue(legacy.streamCadenceMigrated)

        // A deliberate post-migration 250 sticks.
        legacy.streamSnapshotIntervalMs = 250
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertEqual(legacy.streamSnapshotIntervalMs, 250)

        // A pre-migration non-default cadence is a choice and survives.
        var chosen = MTPLXAppConfiguration()
        chosen.streamSnapshotIntervalMs = 500
        chosen.streamCadenceMigrated = false
        chosen.sanitizeLaunchCriticalFields()
        XCTAssertEqual(chosen.streamSnapshotIntervalMs, 500)
    }

    func testSanitizeMigratesLegacyDefaultSamplerTripleOnce() throws {
        // The exact legacy triple (1.0/0.95/20) is the fingerprint of the
        // persisted override that clobbered the engine's 0.6 default. It
        // yields to preset authority exactly ONCE per config.
        var legacy = MTPLXAppConfiguration()
        legacy.temperature = 1.0
        legacy.topP = 0.95
        legacy.topK = 20
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertNil(legacy.temperature)
        XCTAssertNil(legacy.topP)
        XCTAssertNil(legacy.topK)
        XCTAssertTrue(legacy.samplerLegacyTripleMigrated)

        // After migration the user can deliberately choose exactly 1.0
        // (founder requirement 2026-07-02): it must stick.
        legacy.temperature = 1.0
        legacy.topP = 0.95
        legacy.topK = 20
        legacy.sanitizeLaunchCriticalFields()
        XCTAssertEqual(legacy.temperature, 1.0)
        XCTAssertEqual(legacy.topP, 0.95)
        XCTAssertEqual(legacy.topK, 20)

        // Any other combination is a deliberate choice and survives even
        // the first pass (and still flips the one-shot flag).
        var chosen = MTPLXAppConfiguration()
        chosen.temperature = 0.8
        chosen.topP = 0.95
        chosen.topK = 20
        chosen.sanitizeLaunchCriticalFields()
        XCTAssertEqual(chosen.temperature, 0.8)
        XCTAssertTrue(chosen.samplerLegacyTripleMigrated)

        var partial = MTPLXAppConfiguration()
        partial.temperature = 1.0
        partial.topP = 0.9
        partial.topK = 20
        partial.sanitizeLaunchCriticalFields()
        XCTAssertEqual(partial.temperature, 1.0)
        XCTAssertEqual(partial.topP, 0.9)
    }

    func testCommandBuilderEmitsLaunchOwnershipAndStrictFanArgs() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                fanMode: "max"
            ),
            launchID: "launch-123"
        )

        XCTAssertFalse(command.arguments.contains("--max"))
        XCTAssertTrue(command.arguments.containsInOrder(["--fan-mode", "max"]))
        XCTAssertTrue(command.arguments.contains("--require-max-fans"))
        XCTAssertTrue(command.arguments.containsInOrder(["--app-launch-id", "launch-123"]))
        XCTAssertEqual(command.environment["MTPLX_APP_LAUNCH_ID"], "launch-123")
        XCTAssertEqual(
            Int(command.environment["MTPLX_APP_PARENT_PID"] ?? ""),
            Int(ProcessInfo.processInfo.processIdentifier)
        )
    }

    func testAppConfigurationDefaultsToSmartFanMode() throws {
        let configuration = MTPLXAppConfiguration()

        XCTAssertEqual(configuration.fanMode, "smart")
        XCTAssertFalse(configuration.pinFansAtMaxOnStart)
    }

    func testAppConfigurationMigratesLegacyPinnedFansToMax() throws {
        let configuration = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: Data("""
            {
              "pin_fans_at_max_on_start": true
            }
            """.utf8)
        )

        XCTAssertEqual(configuration.fanMode, "max")
        XCTAssertTrue(configuration.pinFansAtMaxOnStart)
    }

    func testAppConfigurationExplicitFanModeWinsOverLegacyPinnedFans() throws {
        let configuration = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: Data("""
            {
              "fan_mode": "smart",
              "pin_fans_at_max_on_start": true
            }
            """.utf8)
        )

        XCTAssertEqual(configuration.fanMode, "smart")
        XCTAssertFalse(configuration.pinFansAtMaxOnStart)
    }

    @MainActor
    func testSetFanModeWhileStoppedPersistsNextLaunchMode() async throws {
        let root = temporaryDirectory()
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                port: try freeTCPPort(),
                fanMode: "max"
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        )

        try await backend.setFanMode("smart")

        XCTAssertEqual(backend.configuration.fanMode, "smart")
        XCTAssertFalse(backend.configuration.pinFansAtMaxOnStart)
        XCTAssertNil(backend.currentFanMode)
    }

    func testCommandBuilderOpenCodePresetKeepsMeasuredSamplerButUsesAppReasoning() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                temperature: 1.0,
                topP: 0.8,
                topK: 64,
                reasoning: "off"
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-max-size", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-min-prefix-tokens", "512"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "off"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "3"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.7"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--tool-prompt-mode", "hybrid"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "local_qwen36"]))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertFalse(command.arguments.contains("--adaptive-min-depth"))
        XCTAssertFalse(command.arguments.contains("--adaptive-ev-base-depth"))
        XCTAssertFalse(command.arguments.contains("--adaptive-ev-warmup-full-depth-cycles"))
        XCTAssertFalse(command.arguments.contains("--adaptive-ev-exploration-interval"))
        XCTAssertFalse(command.arguments.contains("--max-response-tokens"))
        XCTAssertTrue(command.arguments.contains("--no-stats-footer"))
        XCTAssertFalse(command.arguments.contains("--context-window"))
        XCTAssertFalse(command.arguments.containsInOrder(["--reasoning-parser", "on"]))
        XCTAssertFalse(command.arguments.containsInOrder(["--reasoning-parser", "off"]))
    }

    func testCommandBuilderCarriesQwenFamilySettingsAcrossForgeInstalledModelSwap() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
                temperature: 0.2,
                topP: 0.7,
                topK: 9,
                liveSettingsModelFamily: "qwen3_6"
            ),
            target: .chat,
            launchID: "qwen-family-swap"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.7"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "9"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.7"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "9"]))
    }

    func testConfigurationContextWindowCarriesAcrossQwenFamiliesOnly() {
        let qwen35 = MTPLXAppConfiguration(
            model: "/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
            contextWindow: 12_345,
            contextWindowModelFamily: "qwen3_6"
        )
        let gemma = MTPLXAppConfiguration(
            model: "/models/Gemma4-MTPLX-Optimized-Speed",
            contextWindow: 12_345,
            contextWindowModelFamily: "qwen3_6"
        )

        XCTAssertEqual(qwen35.compatibleContextWindowOverride(), 12_288)
        XCTAssertNil(gemma.compatibleContextWindowOverride())
    }

    func testCommandBuilderOpenCodeReasoningOnStillUsesMeasuredSampler() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                temperature: 1.0,
                topP: 0.8,
                topK: 64,
                reasoning: "on"
            ),
            target: .openCode,
            launchID: "opencode-reasoning-on"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.7"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
    }

    func testCommandBuilderOpenCodePresetKeepsLiteralD3OverTunedDepth() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                lastTunedDepth: 1
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "3"]))
        XCTAssertFalse(command.arguments.containsInOrder(["--depth", "1"]))
    }

    func testLaunchTargetIncludesHermesAgentMode() throws {
        XCTAssertTrue(LaunchTarget.allCases.contains(.hermes))
        XCTAssertEqual(LaunchTarget.hermes.title, "Hermes")
        XCTAssertEqual(
            LaunchTarget.hermes.tagline,
            "Use Hermes Desktop, powered by MTPLX — terminal, file, web, browser, and messaging tools."
        )
        XCTAssertEqual(LaunchTarget.hermes.systemImage, "sparkles")
        XCTAssertTrue(LaunchTarget.hermes.spawnsDaemon)
    }

    func testOpenCodeLaunchCopyIsClear() throws {
        XCTAssertEqual(
            LaunchTarget.openCode.tagline,
            "Use OpenCode Desktop, powered by MTPLX."
        )
    }

    func testCommandBuilderHermesPresetUsesFastSingleAgentLane() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                schedulingPreset: "agent",
                maxActiveRequests: 4,
                decodeBatchMax: 4,
                batchWaitMs: 50
            ),
            target: .hermes,
            launchID: "hermes-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertTrue(command.arguments.containsInOrder(["--prefill-chunk-tokens", "2048"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-max-size", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-min-prefix-tokens", "512"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--tool-prompt-mode", "hybrid"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "local_qwen36"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-policy", "expected_value"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-min-depth", "1"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-ev-base-depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-ev-warmup-full-depth-cycles", "4"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-ev-exploration-interval", "32"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--app-launch-id", "hermes-launch"]))
        XCTAssertEqual(command.environment["MTPLX_CLIENT"], "hermes")
        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"], "async_per_head")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "32")
    }

    func testCommandBuilderBenchmarkPresetStartsSoloBenchmarkDaemon() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained"
            ),
            target: .benchmark,
            launchID: "benchmark-launch"
        )

        XCTAssertTrue(LaunchTarget.benchmark.spawnsDaemon)
        XCTAssertTrue(command.arguments.containsInOrder(["--profile", "sustained"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--prefill-chunk-tokens", "2048"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--app-launch-id", "benchmark-launch"]))
    }

    func testCommandBuilderBenchmarkHonorsPersistedSamplerAndReasoning() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                temperature: 1.0,
                topP: 0.8,
                topK: 64,
                reasoning: "off"
            ),
            target: .benchmark,
            launchID: "benchmark-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.8"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.8"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "off"]))
    }

    func testCommandBuilderBenchmarkIgnoresBatchingOverrides() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulerMode: "ar_batch",
                batchingPreset: "throughput",
                schedulingPreset: "throughput",
                maxActiveRequests: 8,
                decodeBatchMax: 8,
                batchWaitMs: 20
            ),
            target: .benchmark,
            launchID: "benchmark-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
    }

    func testCommandBuilderBenchmarkKeepsConfiguredProfileForGemmaBundles() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained"
            ),
            target: .benchmark,
            launchID: "benchmark-gemma-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--profile", "sustained"]))
        // Depth 2: assistant-pair acceptance collapse makes deeper blocks
        // EV-negative (2026-07-03 in-app measurement, 23.2 vs ~31 tok/s).
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "gemma4"]))
    }

    func testCommandBuilderChatPresetUsesGemmaLaunchDefaultsForGemmaBundles() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained"
            ),
            target: .chat,
            launchID: "chat-gemma-launch"
        )

        XCTAssertEqual(MTPLXCommandBuilder.defaultReasoningMode(for: .chat), "auto")
        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "solo"]))
        // Depth 2: see the Gemma preset comment in MTPLXCommandBuilder.
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "gemma4"]))
    }

    func testCommandBuilderOpenCodeHonorsExplicitSSDOff() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ssdSessionCache: "off"
            ),
            target: .openCode,
            launchID: "opencode-ssd-off"
        )

        // "off" must be passed explicitly: the serve CLI default became
        // "on" in 2.0.0, so omitting the flag would silently re-enable
        // the cache the user disabled (issue #140).
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "off"]))
        XCTAssertFalse(command.arguments.contains("--ssd-session-cache-max-size"))
        XCTAssertFalse(command.arguments.contains("--ssd-session-cache-min-prefix-tokens"))
        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertFalse(command.arguments.contains("--adaptive-ev-base-depth"))
    }

    // Issue #140: Settings "Persistent Cache (SSD) = Off" + Play ->
    // "Other, Custom client" launched the daemon with the SSD cache ON,
    // because the .other preset says "on" and, before the fix, an
    // explicit user "off" was expressed by omitting the flag — which the
    // 2.0.0 serve CLI (default "on") reads as on with default limits.
    func testCommandBuilderOtherTargetHonorsExplicitSSDOff() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ssdSessionCache: "off"
            ),
            target: .other,
            launchID: "other-ssd-off"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "off"]))
        XCTAssertFalse(command.arguments.contains("--ssd-session-cache-max-size"))
        XCTAssertFalse(command.arguments.contains("--ssd-session-cache-min-prefix-tokens"))
    }

    func testCommandBuilderHonorsExplicitLatencySchedulingPresetForCodingAgents() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulingPreset: "latency"
            ),
            target: .openCode,
            launchID: "opencode-latency"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
    }

    func testCommandBuilderPiPresetKeepsServeAppOwned() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                profile: "sustained",
                host: "127.0.0.1",
                port: 8000
            ),
            target: .pi,
            launchID: "pi-launch"
        )

        XCTAssertFalse(command.arguments.contains("--launch-pi"))
        XCTAssertFalse(command.arguments.contains("--pi-launch-command"))
        XCTAssertTrue(command.arguments.containsInOrder(["--app-launch-id", "pi-launch"]))
        XCTAssertFalse(command.arguments.contains("--max-response-tokens"))
        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "ar_batch"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "agent"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--max-active-requests", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--decode-batch-max", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batch-wait-ms", "50.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--prefill-chunk-tokens", "2048"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--tool-prompt-mode", "hybrid"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "local_qwen36"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-policy", "expected_value"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--adaptive-ev-base-depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--preserve-thinking", "auto"]))
        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"], "async_per_head")
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT"])
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "32")
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"])
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"])
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH"])
        XCTAssertEqual(command.environment["MTPLX_LAZY_BONUS_VERIFY"], "1")
        XCTAssertEqual(command.environment["MTPLX_TOOL_RESULT_COMPACT_THRESHOLD_CHARS"], "1200")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES"], "32")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_LINE_MAX_CHARS"], "180")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES"], "72")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE"], "8")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS"], "120")
        XCTAssertEqual(command.environment["MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS"], "12")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES"], "32")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_TOOL_RESULT_LINE_MAX_CHARS"], "220")
        XCTAssertEqual(
            PiIntegration.launchCommand(for: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"),
            "pi --model mtplx/mtplx-qwen36-27b-optimized-speed --tools read,bash,edit,write,grep,find,ls "
                + "--append-system-prompt '\(PiIntegration.agentOperatingHintsURL().path)'"
        )
    }

    func testCommandBuilderPiPresetUsesGemmaLaunchDefaultsForGemmaBundles() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained",
                lastTunedDepth: 3
            ),
            target: .pi,
            launchID: "pi-gemma-launch"
        )

        // Depth 2 is the Gemma launch default (see MTPLXCommandBuilder);
        // the incompatible Qwen tune (3) above must NOT leak through.
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "gemma4"]))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertFalse(command.arguments.contains("--adaptive-ev-base-depth"))
        XCTAssertEqual(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"], "tokenizer")
    }

    func testCommandBuilderOpenCodePresetUsesGemmaLaunchDefaultsForGemmaBundles() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained"
            ),
            target: .openCode,
            launchID: "opencode-gemma-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        // Depth 2: Gemma launch default, see MTPLXCommandBuilder preset.
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "2"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "gemma4"]))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertEqual(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"], "tokenizer")
    }

    func testCommandBuilderOpenCodePresetUsesHy3NativeDefaults() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Hy3-295B-A21B-MTPLX-Optimized-Speed",
                profile: "turbo"
            ),
            target: .openCode,
            launchID: "opencode-hy3-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "1"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.9"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.9"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertFalse(command.arguments.containsInOrder(["--chat-template-profile", "local_qwen36"]))
        XCTAssertEqual(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"], "tokenizer")
    }

    func testCommandBuilderOpenCodePresetUsesStepLaunchDefaultsForStepfun() throws {
        let fake = try makeExecutable(named: "mtplx")
        let adapter = "/tmp/step37-134243.npz"
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
            "MTPLX_STEP_MTP_ADAPTER": adapter,
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                profile: "sustained",
                pagedKVQuantization: "q8",
                liveSettingsModelFamily: "step",
                lastTunedDepth: 3
            ),
            target: .openCode,
            launchID: "opencode-step-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "latency"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--depth", "1"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--verify-strategy", "trim_commit"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--verify-core", "stock"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--mtp-adapter", adapter]))
        XCTAssertTrue(command.arguments.containsInOrder(["--mtp-quant-bits", "4"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--mtp-quant-group-size", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--mtp-quant-mode", "affine"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "0.6"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "0.95"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "20"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--chat-template-profile", "tokenizer"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--preserve-thinking", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "step3p5"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-effort", "low"]))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"])
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"])
        XCTAssertEqual(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"], "tokenizer")
    }

    func testCommandBuilderOpenCodePresetUsesBundledStepAdapterWithoutEnvOverride() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                profile: "sustained",
                pagedKVQuantization: "q8",
                liveSettingsModelFamily: "step",
                lastTunedDepth: 3
            ),
            target: .openCode,
            launchID: "opencode-step-launch"
        )

        let adapterIndex = try XCTUnwrap(command.arguments.firstIndex(of: "--mtp-adapter"))
        let adapterPathIndex = command.arguments.index(after: adapterIndex)
        XCTAssertLessThan(adapterPathIndex, command.arguments.endIndex)
        let adapterPath = command.arguments[adapterPathIndex]
        XCTAssertTrue(adapterPath.hasSuffix("StepAdapters/c4-mtp-adapter-20260603-134243-r4.npz"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: adapterPath))
        XCTAssertTrue(command.arguments.containsInOrder(["--verify-strategy", "trim_commit"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--mtp-quant-bits", "4"]))
    }

    func testCommandBuilderUsesExplicitStepReasoningEffortWhenEnabled() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                reasoning: "on",
                reasoningEffort: "high",
                liveSettingsModelFamily: "step"
            ),
            target: .chat,
            launchID: "chat-step-reasoning"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--preserve-thinking", "auto"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "step3p5"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-effort", "high"]))
    }

    func testCommandBuilderStripsStepReasoningEffortWhenExplicitlyOff() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                reasoning: "off",
                reasoningEffort: "high",
                liveSettingsModelFamily: "step"
            ),
            target: .chat,
            launchID: "chat-step-no-reasoning"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "off"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--preserve-thinking", "off"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning-parser", "step3p5"]))
        XCTAssertFalse(command.arguments.contains("--reasoning-effort"))
    }

    func testCommandBuilderDoesNotPassLegacyQwenContextWindowToGemma() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained",
                contextWindow: 131_072
            ),
            target: .chat,
            launchID: "gemma-context-launch"
        )

        XCTAssertFalse(command.arguments.contains("--context-window"))
    }

    func testCommandBuilderPassesGemmaContextWindowWhenFamilyMatches() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Gemma4-MTPLX-Optimized-Speed",
                profile: "sustained",
                contextWindow: 131_072,
                contextWindowModelFamily: "gemma4"
            ),
            target: .chat,
            launchID: "gemma-context-launch"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--context-window", "131072"]))
    }

    func testCommandBuilderOpenCodePresetEnablesMeasuredLongContextGQARoute() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "137438953472",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained"
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertEqual(command.environment["MTPLX_APP_LAUNCH_ID"], "opencode-launch")
        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"], "async_per_head")
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT"])
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q"])
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q"])
        XCTAssertEqual(command.environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"], "1")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "32")
        // "auto": the engine budgets half the post-model RAM surplus.
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S"], "30.0")
        XCTAssertEqual(command.environment["MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS"], "4096")
        XCTAssertEqual(command.environment["MTPLX_LAZY_BONUS_VERIFY"], "1")
        XCTAssertEqual(command.environment["MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER"], "1")
        XCTAssertEqual(command.environment["MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE"], "1")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES"], "72")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE"], "8")
        XCTAssertEqual(command.environment["MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS"], "120")
        XCTAssertEqual(command.environment["MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS"], "12")
        XCTAssertEqual(command.environment["MTPLX_TOOL_PROMPT_MODE"], "hybrid")
        XCTAssertEqual(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"], "local_qwen36")
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"])
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"])
        XCTAssertNil(command.environment["MTPLX_LONG_CONTEXT_MTP_DEPTH"])
    }

    func testCommandBuilderOpenCodePresetKeepsConservativeRAMLimitsOnLowerMemoryMachines() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": "68719476736",
        ])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained"
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "6")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"], "async_per_head")
    }

    func testCommandBuilderSettingsOverrideOpenCodeRAMSessionCacheLimits() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ramSessionCachePolicy: "bounded",
                ramSessionBlockPrefixRestore: false,
                ramSessionCacheMaxEntries: 2,
                ramSessionCacheMaxSize: "4G",
                ramSessionCachePerSessionMaxSize: "2G"
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertEqual(command.environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"], "0")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "2")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_BYTES"], "4G")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"], "2G")
        XCTAssertEqual(command.environment["MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S"], "30.0")
        XCTAssertEqual(command.environment["MTPLX_TOOL_PROMPT_MODE"], "hybrid")
    }

    func testCommandBuilderMinimalRAMSessionCachePolicyUsesTinyBoundedEnv() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ramSessionCachePolicy: "minimal",
                ramSessionBlockPrefixRestore: true,
                ramSessionCacheMaxEntries: 8,
                ramSessionCacheMaxSize: "24G",
                ramSessionCachePerSessionMaxSize: "8G"
            ),
            target: .openCode
        )

        XCTAssertEqual(command.environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"], "0")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "1")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_BYTES"], "1G")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"], "1G")
    }

    func testCommandBuilderBoundedPolicyPassesAutoCacheSizesThrough() throws {
        // The default bounded selections are "auto": the engine budgets half
        // of the RAM left after the model weights. Explicit sizes still pass
        // through verbatim (previous test); auto must too, not be rewritten.
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ramSessionCachePolicy: "bounded",
                ramSessionCacheMaxEntries: 8,
                ramSessionCacheMaxSize: "auto",
                ramSessionCachePerSessionMaxSize: "auto"
            ),
            target: .openCode,
            launchID: "opencode-launch"
        )

        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_PER_SESSION_BYTES"], "auto")
        XCTAssertEqual(command.environment["MTPLX_SESSION_BANK_MAX_ENTRIES"], "8")
    }

    func testConfigurationDefaultsUseAutoCacheBudgets() throws {
        let defaults = MTPLXAppConfiguration()
        XCTAssertEqual(defaults.ramSessionCacheMaxSize, "auto")
        XCTAssertEqual(defaults.ramSessionCachePerSessionMaxSize, "auto")
        XCTAssertEqual(defaults.ssdSessionCacheMaxSize, "auto")

        // Persisted configs from older builds decode their explicit values.
        let legacy = try decodeConfiguration(json: """
        {"model": "/models/qwen", "ram_session_cache_max_size": "8G",
         "ssd_session_cache_max_size": "100GB"}
        """)
        XCTAssertEqual(legacy.ramSessionCacheMaxSize, "8G")
        XCTAssertEqual(legacy.ssdSessionCacheMaxSize, "100GB")
    }

    func testCommandBuilderChatPresetMirrorsWebUISoloServing() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained"
            ),
            target: .chat
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "solo"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertFalse(command.arguments.contains("--prefill-chunk-tokens"))
        // The chat preset is SSD-off by design; that must reach argv
        // explicitly now that the serve CLI defaults to "on".
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "off"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "auto"]))
        XCTAssertFalse(command.arguments.contains("--top-k"))
        XCTAssertFalse(command.arguments.contains("--draft-temperature"))
        XCTAssertFalse(command.arguments.contains("--draft-top-p"))
        XCTAssertFalse(command.arguments.contains("--draft-top-k"))
        XCTAssertFalse(command.arguments.contains("--tool-prompt-mode"))
        XCTAssertFalse(command.arguments.contains("--chat-template-profile"))
        XCTAssertFalse(command.arguments.contains("--adaptive-policy"))
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE"])
        XCTAssertNil(command.environment["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"])
        XCTAssertNil(command.environment["MTPLX_TOOL_PROMPT_MODE"])
        XCTAssertNil(command.environment["MTPLX_CHAT_TEMPLATE_PROFILE"])
        XCTAssertNil(command.environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"])
    }

    func testCommandBuilderChatCarriesPersistedQwenReasoningOn() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                reasoning: "on",
                liveSettingsModelFamily: "qwen3_6"
            ),
            target: .chat
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "on"]))
    }

    func testCommandBuilderChatPresetMigratesLegacyAgentPairToSolo() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        var configuration = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: Data("""
            {
              "scheduler_mode": "ar_batch",
              "batching_preset": "agent"
            }
            """.utf8)
        )
        configuration.executablePath = fake.path
        configuration.model = "/models/qwen"
        let command = try builder.buildServeCommand(
            configuration: configuration,
            target: .chat
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "solo"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
    }

    func testCommandBuilderChatIgnoresExplicitBatchingOverrides() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                schedulingPreset: "agent",
                maxActiveRequests: 4,
                decodeBatchMax: 4,
                batchWaitMs: 50
            ),
            target: .chat
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "solo"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
    }

    func testCommandBuilderOpenWebUIUsesAppOwnedSamplerButKeepsSoloScheduling() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                schedulingPreset: "throughput",
                maxActiveRequests: 8,
                decodeBatchMax: 8,
                batchWaitMs: 20,
                temperature: 1.0,
                topP: 1.0,
                topK: 64,
                reasoning: "off"
            ),
            target: .openWebUI
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "serial"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "solo"]))
        XCTAssertFalse(command.arguments.contains("--max-active-requests"))
        XCTAssertFalse(command.arguments.contains("--decode-batch-max"))
        XCTAssertFalse(command.arguments.contains("--batch-wait-ms"))
        XCTAssertFalse(command.arguments.contains("--tool-prompt-mode"))
        XCTAssertTrue(command.arguments.containsInOrder(["--temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-temperature", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-p", "1.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--draft-top-k", "64"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--reasoning", "off"]))
    }

    func testChatReasoningPolicyUsesModelDefaultForAuto() throws {
        let controls = ModelControls(
            modelFamily: "qwen3_6",
            reasoning: ReasoningPolicy(
                supported: true,
                parser: "qwen3",
                defaultMode: "off"
            )
        )

        XCTAssertEqual(
            ChatReasoningPolicy.enableThinking(
                explicitMode: "auto",
                liveMode: "auto",
                modelControls: controls,
                modelFamily: "qwen3_6"
            ),
            false
        )
    }

    func testChatReasoningPolicyDefaultsKnownFamiliesToAuto() throws {
        XCTAssertNil(ChatReasoningPolicy.enableThinking(
            explicitMode: "auto",
            modelFamily: "qwen3_6"
        ))
        XCTAssertNil(ChatReasoningPolicy.enableThinking(
            explicitMode: "auto",
            modelFamily: "qwen3_8"
        ))
        XCTAssertNil(ChatReasoningPolicy.enableThinking(
            explicitMode: "auto",
            modelFamily: "gemma4"
        ))
    }

    func testChatReasoningPolicyHonorsExplicitOnOff() throws {
        XCTAssertEqual(
            ChatReasoningPolicy.enableThinking(
                explicitMode: "on",
                modelFamily: "qwen3_6"
            ),
            true
        )
        XCTAssertEqual(
            ChatReasoningPolicy.enableThinking(
                explicitMode: "off",
                modelFamily: "gemma4"
            ),
            false
        )
    }

    func testHermesDesktopDiscoveryFindsBootstrapBundleAndReturnsNilWhenAbsent() throws {
        let root = temporaryDirectory()
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        // Override pins discovery away from LaunchServices so the test is
        // hermetic on machines that have a real Hermes installed.
        let missingOverride = root.appendingPathComponent("Nowhere/Hermes.app", isDirectory: true)
        let cliOnly = HermesIntegration(
            hermesHome: hermesHome,
            desktopApplicationOverride: missingOverride
        )
        XCTAssertNil(cliOnly.desktopApplicationURL())

        let bundle = hermesHome
            .appendingPathComponent("hermes-agent/apps/desktop/release/mac-arm64/Hermes.app", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        let bootstrapped = HermesIntegration(hermesHome: hermesHome)
        XCTAssertEqual(bootstrapped.desktopApplicationURL()?.path, bundle.path)
    }

    func testHermesActiveDesktopProfilePinWritesMTPLXAndReturnsPrevious() throws {
        let root = temporaryDirectory()
        let activeProfile = root
            .appendingPathComponent("Library/Application Support/Hermes", isDirectory: true)
            .appendingPathComponent("active-profile.json")
        let integration = HermesIntegration(
            hermesHome: root.appendingPathComponent(".hermes", isDirectory: true),
            activeProfileURL: activeProfile
        )

        // First pin: no previous file, directory created on demand.
        XCTAssertNil(try integration.writeActiveDesktopProfile())
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: activeProfile)) as? [String: Any]
        )
        XCTAssertEqual(object["profile"] as? String, "mtplx")

        // Re-pin over a user's own selection: previous name is surfaced,
        // file ends pinned to mtplx.
        try #"{"profile": "research"}"#.write(to: activeProfile, atomically: true, encoding: .utf8)
        XCTAssertEqual(try integration.writeActiveDesktopProfile(), "research")
        object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: activeProfile)) as? [String: Any]
        )
        XCTAssertEqual(object["profile"] as? String, "mtplx")
    }

    func testHermesIntegrationSyncsMTPLXProfileAndLaunchEnvironment() throws {
        let root = temporaryDirectory()
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        let profilesRoot = hermesHome.appendingPathComponent("profiles", isDirectory: true)
        try FileManager.default.createDirectory(
            at: profilesRoot.appendingPathComponent("research"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: profilesRoot.appendingPathComponent("invalid profile"),
            withIntermediateDirectories: true
        )
        try """
        TELEGRAM_BOT_TOKEN=fake-token
        TELEGRAM_ALLOWED_USERS=123,456
        TELEGRAM_HOME_CHANNEL=-10042
        """.write(to: hermesHome.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
        try """
        {"platforms":{"telegram":[{"id":"-10042","name":"launch-room","type":"channel"}]}}
        """.write(to: hermesHome.appendingPathComponent("channel_directory.json"), atomically: true, encoding: .utf8)
        let workspace = root.appendingPathComponent("Hermes Workspace", isDirectory: true)
        try FileManager.default.createDirectory(
            at: workspace,
            withIntermediateDirectories: true
        )
        let integration = HermesIntegration(
            hermesHome: hermesHome,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
            ],
            terminalCommandURL: root.appendingPathComponent(".mtplx").appendingPathComponent("open-hermes.command")
        )

        let profiles = integration.discoverProfiles()
        XCTAssertEqual(profiles.map(\.name), ["default", "research"])
        XCTAssertTrue(try XCTUnwrap(profiles.first).isDefault)

        let environment = integration.launchEnvironment(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                host: "0.0.0.0",
                port: 8123,
                apiKey: "",
                hermesWorkspacePath: workspace.path
            )
        )

        XCTAssertEqual(environment["OPENAI_BASE_URL"], "http://127.0.0.1:8123/v1")
        XCTAssertEqual(environment["CUSTOM_BASE_URL"], "http://127.0.0.1:8123/v1")
        XCTAssertEqual(environment["OPENAI_API_KEY"], PiIntegration.localAPIKey)
        XCTAssertEqual(environment["HERMES_MODEL"], "mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(environment["HERMES_INFERENCE_MODEL"], "mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(environment["HERMES_INFERENCE_PROVIDER"], "custom")
        XCTAssertEqual(environment["HERMES_YOLO_MODE"], "1")
        XCTAssertEqual(environment["HERMES_MTPLX_TOOLSETS"], "terminal,file,web,browser,messaging")
        XCTAssertEqual(environment["HERMES_MTPLX_CAPABILITIES"], HermesIntegration.capabilitySummary)
        XCTAssertEqual(environment["HERMES_MTPLX_MESSAGING_NOTE"], HermesIntegration.messagingSetupHint)
        XCTAssertEqual(environment["HERMES_MTPLX_GATEWAY_STATUS_COMMAND"], HermesIntegration.gatewayStatusCommand)
        XCTAssertEqual(environment["HERMES_MTPLX_GATEWAY_TRUTH_NOTE"], HermesIntegration.gatewayTruthHint)
        XCTAssertEqual(environment["HERMES_MTPLX_MESSAGING_SUMMARY"], "Telegram configured with a home channel.")
        XCTAssertEqual(environment["HERMES_SESSION_PLATFORM"], "mtplx-app")
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("send_message(action='list')"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("profile-local Gateway state"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("mirrors the root Gateway channel directory"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("distinguish configured credentials"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("env -u HERMES_HOME hermes gateway status"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("Never print token"))
        XCTAssertTrue(HermesIntegration.messagingSetupHint.contains("prefer `hermes gateway start`"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("send_message(action='list')"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("never use it by itself to conclude Telegram cannot connect"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("mirrors the root Gateway channel directory"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("env -u HERMES_HOME hermes gateway status"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("no connected or discovered destinations yet"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("sanitized presence check"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("MTPLXApp on macOS"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("do not recommend sudo"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("systemd"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("do not recommend sudo, systemd, system services, `hermes gateway install`"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("hermes gateway install --system"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("not valid UTF-8"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("Never print token"))
        XCTAssertTrue(HermesIntegration.systemPrompt.contains("explicitly name browser and messaging"))
        XCTAssertEqual(environment["HERMES_WORKSPACE"], workspace.path)
        XCTAssertEqual(environment["TERMINAL_CWD"], workspace.path)

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                host: "0.0.0.0",
                port: 8123,
                apiKey: "",
                hermesWorkspacePath: workspace.path
            )
        )

        XCTAssertEqual(result.profileName, "mtplx")
        XCTAssertEqual(result.baseURL, "http://127.0.0.1:8123/v1")
        XCTAssertEqual(result.modelReference, "mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(result.workspacePath, workspace.path)
        XCTAssertTrue(result.didChange)
        XCTAssertNil(result.configBackupPath)
        XCTAssertNil(result.envBackupPath)
        let configText = try String(contentsOfFile: result.configPath, encoding: .utf8)
        let envText = try String(contentsOfFile: result.envPath, encoding: .utf8)
        XCTAssertTrue(configText.contains("provider: custom"))
        XCTAssertTrue(configText.contains("base_url: 'http://127.0.0.1:8123/v1'"))
        XCTAssertTrue(configText.contains("  - terminal"))
        XCTAssertTrue(configText.contains("  - file"))
        XCTAssertTrue(configText.contains("  - web"))
        XCTAssertTrue(configText.contains("  - browser"))
        XCTAssertTrue(configText.contains("  - messaging"))
        XCTAssertTrue(configText.contains("system_prompt:"))
        XCTAssertTrue(configText.contains("hermes gateway setup"))
        XCTAssertFalse(configText.contains("gateway setup telegram"))
        XCTAssertTrue(configText.contains("env -u HERMES_HOME hermes gateway status"))
        XCTAssertTrue(configText.contains("profile-local Gateway state"))
        XCTAssertTrue(configText.contains("mirrors the root Gateway channel directory"))
        XCTAssertTrue(configText.contains("hermes gateway start"))
        XCTAssertTrue(configText.contains("MTPLXApp on macOS"))
        XCTAssertTrue(configText.contains("do not recommend sudo"))
        XCTAssertTrue(configText.contains("prefer `hermes gateway start`"))
        XCTAssertTrue(configText.contains("hermes gateway install --system"))
        XCTAssertTrue(configText.contains("not valid UTF-8"))
        XCTAssertTrue(configText.contains("send_message(action="))
        XCTAssertTrue(configText.contains("list"))
        XCTAssertTrue(configText.contains("Never print token"))
        XCTAssertTrue(configText.contains("TELEGRAM_BOT_TOKEN"))
        XCTAssertTrue(configText.contains("TELEGRAM_ALLOWED_USERS"))
        XCTAssertTrue(configText.contains("TELEGRAM_HOME_CHANNEL"))
        XCTAssertTrue(configText.contains("tool_use_enforcement: auto"))
        XCTAssertTrue(configText.contains("cwd: '\(workspace.path)'"))
        XCTAssertTrue(configText.contains("show_reasoning: true"))
        XCTAssertTrue(envText.contains("HERMES_INFERENCE_PROVIDER=custom"))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_REASONING=\"auto\""))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_SHOW_REASONING=1"))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_TOOLSETS=\"terminal,file,web,browser,messaging\""))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_MESSAGING_NOTE=\""))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_GATEWAY_STATUS_COMMAND=\"env -u HERMES_HOME hermes gateway status\""))
        XCTAssertTrue(envText.contains("HERMES_MTPLX_GATEWAY_TRUTH_NOTE=\""))
        XCTAssertFalse(envText.contains("'\\''"))
        XCTAssertTrue(envText.contains("hermes gateway setup"))
        XCTAssertFalse(envText.contains("gateway setup telegram"))
        XCTAssertTrue(envText.contains("env -u HERMES_HOME hermes gateway status"))
        XCTAssertTrue(envText.contains("profile-local Gateway state"))
        XCTAssertTrue(envText.contains("mirrors the root Gateway channel directory"))
        XCTAssertTrue(envText.contains("prefer `hermes gateway start`"))
        XCTAssertTrue(envText.contains("send_message(action="))
        XCTAssertTrue(envText.contains("list"))
        XCTAssertTrue(envText.contains("Never print token"))
        XCTAssertTrue(envText.contains("OPENAI_BASE_URL=\"http://127.0.0.1:8123/v1\""))
        XCTAssertTrue(envText.contains("HERMES_WORKSPACE=\"\(workspace.path)\""))
        XCTAssertTrue(envText.contains("TERMINAL_CWD=\"\(workspace.path)\""))
        XCTAssertTrue(envText.contains("HERMES_SESSION_PLATFORM=\"mtplx-app\""))
        XCTAssertTrue(envText.contains("TELEGRAM_BOT_TOKEN=\"fake-token\""))
        XCTAssertTrue(envText.contains("TELEGRAM_ALLOWED_USERS=\"123,456\""))
        XCTAssertTrue(envText.contains("TELEGRAM_HOME_CHANNEL=\"-10042\""))
        let channelDirectoryText = try String(
            contentsOf: URL(fileURLWithPath: result.profilePath).appendingPathComponent("channel_directory.json"),
            encoding: .utf8
        )
        XCTAssertTrue(channelDirectoryText.contains("launch-room"))
        XCTAssertEqual(integration.discoverProfiles().map(\.name), ["default", "mtplx", "research"])
    }

    // Issue #131: the app regenerated the Hermes profile config from its
    // template on every launch, silently dropping every top-level section
    // it does not own (memory/providers/delegation/…) and user-added child
    // keys such as model.max_tokens. These tests drive the reporter's exact
    // repro through the real sync() path.

    func testHermesSyncPreservesUserConfigSectionsAcrossLaunches() throws {
        let root = temporaryDirectory()
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        let workspace = root.appendingPathComponent("ws", isDirectory: true)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let integration = HermesIntegration(
            hermesHome: hermesHome,
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            terminalCommandURL: root.appendingPathComponent(".mtplx").appendingPathComponent("open-hermes.command")
        )
        let configuration = MTPLXAppConfiguration(
            model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
            host: "127.0.0.1",
            port: 8123,
            apiKey: "",
            hermesWorkspacePath: workspace.path
        )

        // First launch writes the template.
        let first = try integration.sync(configuration: configuration)
        let configURL = URL(fileURLWithPath: first.configPath)

        // The user adds non-template sections and a model override —
        // the issue #131 repro.
        var text = try String(contentsOf: configURL, encoding: .utf8)
        text = text.replacingOccurrences(
            of: "  api_mode: chat_completions\n",
            with: "  api_mode: chat_completions\n  max_tokens: 32768\n"
        )
        text += """
        # external memory (issue #131 repro)
        memory:
          provider: honcho
          workspace: mtplx
        providers:
          openrouter:
            api_key: sk-fake
        """ + "\n"
        try text.write(to: configURL, atomically: true, encoding: .utf8)

        // Next launch must preserve all of it.
        _ = try integration.sync(configuration: configuration)
        let preserved = try String(contentsOf: configURL, encoding: .utf8)
        XCTAssertTrue(preserved.contains("# external memory (issue #131 repro)"))
        XCTAssertTrue(preserved.contains("memory:\n  provider: honcho\n  workspace: mtplx"))
        XCTAssertTrue(preserved.contains("providers:\n  openrouter:\n    api_key: sk-fake"))
        XCTAssertTrue(preserved.contains("  max_tokens: 32768"))

        // A repeat launch with an unchanged configuration must not rewrite
        // the file at all (no rewrite loop, no backup spam).
        let repeated = try integration.sync(configuration: configuration)
        XCTAssertFalse(repeated.didChange)
        XCTAssertNil(repeated.configBackupPath)

        // App-owned keys still update in place while user content survives.
        let moved = MTPLXAppConfiguration(
            model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
            host: "127.0.0.1",
            port: 9001,
            apiKey: "",
            hermesWorkspacePath: workspace.path
        )
        let rewritten = try integration.sync(configuration: moved)
        XCTAssertTrue(rewritten.didChange)
        let final = try String(contentsOf: configURL, encoding: .utf8)
        XCTAssertTrue(final.contains("9001/v1"))
        XCTAssertFalse(final.contains("8123/v1"))
        XCTAssertTrue(final.contains("memory:\n  provider: honcho\n  workspace: mtplx"))
        XCTAssertTrue(final.contains("  max_tokens: 32768"))
    }

    func testHermesMergedConfigOwnsTemplateShapedSections() {
        let template = """
        model:
          default: "m"
          provider: custom
          base_url: "http://127.0.0.1:9001/v1"
        toolsets:
          - terminal
          - file
        display:
          streaming: true
        """ + "\n"
        let existing = """
        # user preamble comment
        model:
          default: "old"
          provider: custom
          base_url: "http://127.0.0.1:8123/v1"
          max_tokens: 32768
          reasoning_effort: "high"
        toolsets:
          - terminal
          - custom-extra
        memory:
          provider: honcho
        """ + "\n"

        let merged = HermesIntegration.mergedConfigYAML(existing: existing, template: template)

        // Preamble comment survives at the top.
        XCTAssertTrue(merged.hasPrefix("# user preamble comment\n"))
        // App-owned model keys come from the template; the user's unknown
        // child key is kept.
        XCTAssertTrue(merged.contains("base_url: \"http://127.0.0.1:9001/v1\""))
        XCTAssertFalse(merged.contains("8123"))
        XCTAssertTrue(merged.contains("  max_tokens: 32768"))
        // reasoning_effort is conditionally app-owned: the template omits
        // it, so a stale line must not be resurrected as user content.
        XCTAssertFalse(merged.contains("reasoning_effort"))
        // Sequence-shaped owned sections are rewritten wholly.
        XCTAssertFalse(merged.contains("custom-extra"))
        // Unknown sections survive verbatim.
        XCTAssertTrue(merged.contains("memory:\n  provider: honcho"))
        // The merge is idempotent.
        XCTAssertEqual(
            HermesIntegration.mergedConfigYAML(existing: merged, template: template),
            merged
        )
    }

    func testHermesMergedConfigWithoutExistingFileIsTemplate() {
        let template = "model:\n  default: \"m\"\n"
        XCTAssertEqual(
            HermesIntegration.mergedConfigYAML(existing: nil, template: template),
            template
        )
        XCTAssertEqual(
            HermesIntegration.mergedConfigYAML(existing: "  \n", template: template),
            template
        )
    }

    func testHermesIntegrationSurfacesInvalidMessagingEnvWarning() throws {
        let root = temporaryDirectory()
        let hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        try FileManager.default.createDirectory(at: hermesHome, withIntermediateDirectories: true)
        try Data([0x48, 0x9d, 0x49]).write(to: hermesHome.appendingPathComponent(".env"))
        let integration = HermesIntegration(
            hermesHome: hermesHome,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
            ],
            terminalCommandURL: root.appendingPathComponent(".mtplx").appendingPathComponent("open-hermes.command")
        )

        let environment = integration.launchEnvironment(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 8123,
                apiKey: "",
                hermesWorkspacePath: root.path
            )
        )

        XCTAssertTrue(environment["HERMES_MTPLX_MESSAGING_WARNINGS"]?.contains("invalid UTF-8") == true)
    }

    func testHermesReadyStatusSurfacesUpdateCommand() throws {
        let status = HermesInstallStatus.ready(
            executablePath: "/usr/local/bin/hermes",
            versionSummary: "Hermes Agent v0.5.0",
            updateSummary: "Update available: 7163 commits behind - run 'hermes update'"
        )

        XCTAssertEqual(status.updateCommand, "hermes update")
    }

    func testHermesInstallStatusChecksGatewayOutsideProfileHome() async throws {
        let fakeHermes = try makeExecutable(
            named: "hermes",
            body: """
            #!/bin/sh
            if [ "$1" = "--version" ]; then
              echo "Hermes Agent v0.5.0"
              exit 0
            fi
            if [ "$1" = "chat" ] && [ "$2" = "--help" ]; then
              echo "--query"
              echo "--source"
              exit 0
            fi
            if [ "$1" = "gateway" ] && [ "$2" = "status" ]; then
              if [ -n "$HERMES_HOME" ]; then
                echo "profile HERMES_HOME leaked into gateway status"
                exit 9
              fi
              echo "Gateway service is loaded"
              echo '{ "PID" = 42; };'
              exit 0
            fi
            echo ok
            """
        )
        let root = temporaryDirectory()
        let integration = HermesIntegration(
            hermesHome: root.appendingPathComponent(".hermes", isDirectory: true),
            environment: [
                "HOME": root.path,
                "PATH": fakeHermes.deletingLastPathComponent().path,
                "HERMES_HOME": root.appendingPathComponent(".hermes/profiles/mtplx").path,
            ]
        )

        let status = await integration.installStatus()

        XCTAssertEqual(status.kind, .ready)
        XCTAssertEqual(status.gatewayHealth, .healthy)
        XCTAssertEqual(status.gatewaySummary, "Gateway service loaded; PID 42")
    }

    func testHermesGatewayStatusClassifiesStaleLaunchAgent() throws {
        let output = """
        Launchd plist: /Users/youssof/Library/LaunchAgents/ai.hermes.gateway.plist
        Service definition is stale relative to the current Hermes install
          Run: hermes gateway start
        Gateway service is loaded
        {
            "PID" = 1374;
        };
        """

        XCTAssertEqual(HermesIntegration.gatewayHealth(fromStatusOutput: output), .warning)
        XCTAssertTrue(HermesIntegration.gatewayWarnings(fromStatusOutput: output).first?.contains("hermes gateway start") == true)
        let status = HermesInstallStatus.ready(
            executablePath: "/usr/local/bin/hermes",
            versionSummary: "Hermes Agent v0.5.0",
            gatewaySummary: "service definition stale",
            gatewayHealth: .warning
        )
        XCTAssertTrue(status.gatewayNeedsRepair)
    }

    func testHermesGatewayRepairStartsGatewayAndRefreshesStatus() async throws {
        let fakeHermes = try makeExecutable(
            named: "hermes",
            body: """
            #!/bin/sh
            if [ "$1" = "gateway" ] && [ -n "$HERMES_HOME" ]; then
              echo "profile HERMES_HOME leaked into gateway repair"
              exit 9
            fi
            if [ "$1" = "gateway" ] && [ "$2" = "start" ]; then
              echo "Gateway service loaded"
              exit 0
            fi
            if [ "$1" = "gateway" ] && [ "$2" = "status" ]; then
              echo "Gateway service is loaded"
              echo '{ "PID" = 42; };'
              exit 0
            fi
            echo ok
            """
        )
        let root = temporaryDirectory()
        let integration = HermesIntegration(
            hermesHome: root.appendingPathComponent(".hermes", isDirectory: true),
            environment: [
                "HOME": root.path,
                "PATH": fakeHermes.deletingLastPathComponent().path,
                "HERMES_HOME": root.appendingPathComponent(".hermes/profiles/mtplx").path,
            ]
        )

        let result = try await integration.repairGateway()

        XCTAssertEqual(result.startSummary, "Gateway service loaded")
        XCTAssertEqual(result.statusSummary, "Gateway service loaded; PID 42")
        XCTAssertEqual(result.statusHealth, .healthy)
    }

    func testHermesIntegrationLaunchCommandUsesProfileChatAndToolset() throws {
        let command = HermesIntegration.launchCommand(
            for: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
        )

        XCTAssertTrue(command.contains("'hermes' -p 'mtplx' chat"))
        XCTAssertTrue(command.contains("--model 'mtplx-qwen36-27b-optimized-speed'"))
        XCTAssertTrue(command.contains("--toolsets 'terminal,file,web,browser,messaging'"))
        XCTAssertTrue(command.contains("--yolo"))
        XCTAssertTrue(command.contains("--source 'mtplx-app'"))
    }

    func testHermesTerminalCleanupOnlyMatchesAppLaunchedChat() throws {
        XCTAssertTrue(
            HermesIntegration.isAppLaunchedTerminalAgentCommand(
                "/Users/youssof/.local/bin/hermes -p mtplx chat --model mtplx-qwen36-27b-optimized-speed --toolsets terminal,file,web,browser,messaging --yolo --source mtplx-app"
            )
        )
        XCTAssertTrue(
            HermesIntegration.isAppLaunchedTerminalAgentCommand(
                "/Users/youssof/.hermes/hermes-agent/venv/bin/python3 /Users/youssof/.local/bin/hermes -p mtplx chat --model mtplx-qwen36-27b-optimized-speed --toolsets terminal,file,web,browser,messaging --yolo --source mtplx-app"
            )
        )
        XCTAssertFalse(
            HermesIntegration.isAppLaunchedTerminalAgentCommand(
                "/Users/youssof/.local/bin/hermes -p personal chat --source cli"
            )
        )
        XCTAssertFalse(
            HermesIntegration.isAppLaunchedTerminalAgentCommand(
                "/Users/youssof/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace"
            )
        )
        XCTAssertFalse(
            HermesIntegration.isAppLaunchedTerminalAgentCommand(
                "/bin/zsh -lc ps -axo pid=,command= | awk 'index($0,\"hermes\") && index($0,\" chat\") && index($0,\"-p mtplx\") && index($0,\"--source mtplx-app\") {print}'"
            )
        )
    }

    func testCommandBuilderEmitsPagedKVQuantizationEnv() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                pagedKVQuantization: "q8"
            ),
            target: .openCode
        )

        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"], "q8")
        XCTAssertEqual(command.environment["MTPLX_TOOL_PROMPT_MODE"], "hybrid")
    }

    func testCommandBuilderKeepsPagedKVQuantizationForQwen38() throws {
        // 2.7.0 review: the KV-quant gate listed only qwen3_5/qwen3_6, so a
        // Qwen 3.8 launch silently exported nothing while Settings showed q8.
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
                profile: "turbo",
                pagedKVQuantization: "q8"
            ),
            target: .openCode
        )

        XCTAssertEqual(command.environment["MTPLX_VLLM_METAL_PAGED_KV_QUANT"], "q8")
    }

    func testOfficialModelCatalogIncludesOptimizedQuality() throws {
        let quality = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-27b-optimized-quality")
        )

        XCTAssertEqual(quality.id, "optimized-quality")
        XCTAssertEqual(quality.displayName, "Qwen 3.6 27B Optimized Quality")
        XCTAssertEqual(quality.shortName, "Qwen 3.6 27B Optimized Quality")
        XCTAssertEqual(
            MTPLXModelOption.displayName(
                for: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"
            ),
            "Qwen 3.6 27B Optimized Quality"
        )
        XCTAssertTrue(
            quality.matches("/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality")
        )
        XCTAssertTrue(
            MTPLXModelOption.officialCatalog.contains { $0.id == "optimized-speed" }
        )
    }

    func testOfficialModelCatalogIncludesQwen359BOptimizedSpeed() throws {
        let qwen9b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen35-9b-optimized-speed")
        )

        XCTAssertEqual(qwen9b.id, "qwen35-9b-optimized-speed")
        XCTAssertEqual(qwen9b.hfModelID, "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed")
        XCTAssertEqual(qwen9b.displayName, "Qwen 3.5 9B Optimized Speed")
        XCTAssertEqual(qwen9b.shortName, "Qwen 3.5 9B Optimized Speed")
        XCTAssertTrue(
            qwen9b.matches(
                "/Users/youssof/Documents/MTPLX/models/Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI"
            )
        )
        XCTAssertEqual(
            MTPLXModelOption.displayName(
                for: "/Users/youssof/Documents/MTPLX/models/Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI"
            ),
            "Qwen 3.5 9B Optimized Speed"
        )
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen9b.resolvedReference), "qwen3_5")
    }

    func testOfficialModelCatalogIncludesQwen359BOptimizedSpeedFP16() throws {
        let qwen9b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen35-9b-optimized-speed-fp16")
        )

        XCTAssertEqual(qwen9b.id, "qwen35-9b-optimized-speed-fp16")
        XCTAssertEqual(qwen9b.hfModelID, "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16")
        XCTAssertEqual(qwen9b.displayName, "Qwen 3.5 9B Optimized Speed FP16")
        XCTAssertEqual(qwen9b.shortName, "Qwen 3.5 9B Optimized Speed FP16")
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen9b.resolvedReference), "qwen3_5")
    }

    func testOfficialModelCatalogIncludesGemmaOptimizedSpeed() throws {
        let gemma = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx/gemma4-mtplx-optimized-speed")
        )

        XCTAssertEqual(gemma.id, "gemma4-optimized-speed")
        XCTAssertEqual(gemma.hfModelID, "Youssofal/Gemma4-MTPLX-Optimized-Speed")
        XCTAssertEqual(gemma.displayName, "Gemma 4 31B Optimized Speed")
        XCTAssertEqual(gemma.shortName, "Gemma 4 31B Optimized Speed")
        XCTAssertTrue(gemma.matches("/tmp/Gemma4-MTPLX-Optimized-Speed"))
        XCTAssertEqual(
            MTPLXModelOption.displayName(for: "gemma4-mtplx-optimized-speed"),
            "Gemma 4 31B Optimized Speed"
        )
    }

    func testOfficialModelCatalogIncludesQwen36ThirtyFiveBOptimizedSpeed() throws {
        let qwen35b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-35b-a3b-optimized-speed")
        )

        XCTAssertEqual(qwen35b.id, "qwen36-35b-a3b-optimized-speed")
        XCTAssertEqual(qwen35b.hfModelID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed")
        XCTAssertEqual(qwen35b.displayName, "Qwen 3.6 35B-A3B Optimized Speed")
        XCTAssertEqual(qwen35b.shortName, "Qwen 3.6 35B-A3B Optimized Speed")
        XCTAssertTrue(
            qwen35b.matches(
                "/Users/youssof/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe"
            )
        )
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen35b.resolvedReference), "qwen3_6")
    }

    func testOfficialModelCatalogIncludesQwen36ThirtyFiveBOptimizedSpeedFP16() throws {
        let qwen35b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-35b-a3b-optimized-speed-fp16")
        )

        XCTAssertEqual(qwen35b.id, "qwen36-35b-a3b-optimized-speed-fp16")
        XCTAssertEqual(qwen35b.hfModelID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
        XCTAssertEqual(qwen35b.displayName, "Qwen 3.6 35B-A3B Optimized Speed FP16")
        XCTAssertEqual(qwen35b.shortName, "Qwen 3.6 35B-A3B Optimized Speed FP16")
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen35b.resolvedReference), "qwen3_6")
    }

    func testOfficialModelCatalogIncludesQwen36ThirtyFiveBOptimizedBalance() throws {
        let qwen35b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-35b-a3b-optimized-balance")
        )

        XCTAssertEqual(qwen35b.id, "qwen36-35b-a3b-optimized-balance")
        XCTAssertEqual(qwen35b.hfModelID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance")
        XCTAssertEqual(qwen35b.displayName, "Qwen 3.6 35B-A3B Optimized Balance")
        XCTAssertEqual(qwen35b.shortName, "Qwen 3.6 35B-A3B Optimized Balance")
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen35b.resolvedReference), "qwen3_6")
    }

    func testOfficialModelCatalogIncludesQwen36ThirtyFiveBOptimizedBalanceFP16() throws {
        let qwen35b = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-35b-a3b-optimized-balance-fp16")
        )

        XCTAssertEqual(qwen35b.id, "qwen36-35b-a3b-optimized-balance-fp16")
        XCTAssertEqual(qwen35b.hfModelID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16")
        XCTAssertEqual(qwen35b.displayName, "Qwen 3.6 35B-A3B Optimized Balance FP16")
        XCTAssertEqual(qwen35b.shortName, "Qwen 3.6 35B-A3B Optimized Balance FP16")
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: qwen35b.resolvedReference), "qwen3_6")
    }

    func testDefaultAppModelIsPortableHuggingFaceReference() throws {
        let model = MTPLXAppConfiguration.defaultLocalModelPath()

        // Qwen 3.8 Optimized Speed is the recommended pick and fresh-install
        // default (2026-08-15 release); mirrors DEFAULT_HF_MODEL_ID.
        XCTAssertEqual(model, "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")
        XCTAssertFalse(model.contains("/Users/"))
        XCTAssertFalse(model.contains("Documents/MTPLX"))
    }

    func testAppConfigurationCompatibleTunedDepthMatchesQwen35BLocalPath() throws {
        let tunedAt = Date(timeIntervalSince1970: 1_780_000_000)
        let config = MTPLXAppConfiguration(
            model: "/Users/youssof/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
            tunedControlRecord: TunedControlRecord(
                modelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                modelFamily: "qwen3_6",
                backendID: "qwen3_next",
                controlField: "depth",
                controlValue: 1,
                candidates: ["AR", "D1", "D2", "D3"],
                tunedAt: tunedAt
            )
        )

        XCTAssertEqual(config.compatibleTunedDepth(), 1)
    }

    func testAppConfigurationCompatibleTunedDepthDoesNotLeakAcrossModels() throws {
        let tunedAt = Date(timeIntervalSince1970: 1_780_000_000)
        let record = TunedControlRecord(
            modelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            modelFamily: "qwen3_6",
            backendID: "qwen3_next",
            controlField: "depth",
            controlValue: 2,
            candidates: ["AR", "D1", "D2", "D3"],
            tunedAt: tunedAt
        )
        let qwen27B = MTPLXAppConfiguration(
            model: "/Users/youssof/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
            tunedControlRecord: record
        )
        let gemmaLegacy = MTPLXAppConfiguration(
            model: "/tmp/Gemma4-MTPLX-Optimized-Speed",
            lastTunedDepth: 2
        )

        XCTAssertNil(qwen27B.compatibleTunedDepth())
        XCTAssertNil(gemmaLegacy.compatibleTunedDepth())
    }

    func testAppConfigurationCompatibleTunedControlMatchesGemmaBlock() throws {
        let tunedAt = Date(timeIntervalSince1970: 1_780_000_000)
        let config = MTPLXAppConfiguration(
            model: "/tmp/Gemma4-MTPLX-Optimized-Speed",
            tunedControlRecord: TunedControlRecord(
                modelID: "Youssofal/Gemma4-MTPLX-Optimized-Speed",
                modelFamily: "gemma4",
                backendID: "gemma4_assistant",
                controlField: "draft_block_size",
                controlValue: 6,
                candidates: ["AR", "Block 2", "Block 3", "Block 4", "Block 5", "Block 6", "Block 7", "Block 8"],
                tunedAt: tunedAt
            )
        )

        XCTAssertEqual(config.compatibleTunedControlValue(controlField: "draft_block_size"), 6)
        XCTAssertNil(config.compatibleTunedDepth())
    }

    func testAppConfigurationSavesPostDownloadTuneByModelAlias() throws {
        let localPath = "/tmp/Example--Qwen3.6-Downloaded"
        let repoID = "Example/Qwen3.6-Downloaded"
        var config = MTPLXAppConfiguration(model: localPath, generationMode: "ar", loadMTP: false)

        config.saveTuneResult(
            modelPath: localPath,
            repoID: repoID,
            family: "qwen3_6",
            result: TuneResult(
                bestCandidate: .d1,
                bestDepth: 1,
                bestTokS: 70,
                bestMultiplierVsAR: 1.2,
                allCandidates: [
                    TuneCandidateResult(candidate: .ar, tokS: 58, multiplierVsAR: 1, acceptanceByDepth: []),
                    TuneCandidateResult(candidate: .d1, tokS: 70, multiplierVsAR: 1.2, acceptanceByDepth: [0.82]),
                ]
            )
        )

        XCTAssertEqual(config.generationMode, "mtp")
        XCTAssertTrue(config.loadMTP)
        XCTAssertEqual(config.compatibleTunedDepth(), 1)

        config.model = repoID
        XCTAssertEqual(config.compatibleTunedDepth(), 1)
    }

    func testAppConfigurationPostDownloadTuneDoesNotAutoPromoteARWinner() throws {
        var config = MTPLXAppConfiguration(
            model: "/tmp/Example--Qwen3.6-Downloaded",
            generationMode: "ar",
            loadMTP: false
        )

        config.saveTuneResult(
            modelPath: config.model,
            repoID: "Example/Qwen3.6-Downloaded",
            family: "qwen3_6",
            result: TuneResult(
                bestCandidate: .ar,
                bestDepth: 0,
                bestTokS: 132,
                bestMultiplierVsAR: 1,
                allCandidates: [
                    TuneCandidateResult(candidate: .ar, tokS: 132, multiplierVsAR: 1, acceptanceByDepth: []),
                    TuneCandidateResult(candidate: .d1, tokS: 91, multiplierVsAR: 0.69, acceptanceByDepth: [0.83]),
                    TuneCandidateResult(candidate: .d2, tokS: 99, multiplierVsAR: 0.75, acceptanceByDepth: [0.82, 0.54]),
                ]
            )
        )

        XCTAssertEqual(config.generationMode, "mtp")
        XCTAssertTrue(config.loadMTP)
        XCTAssertEqual(config.compatibleTunedDepth(), 2)
    }

    func testOnboardingQwen36ThirtyFiveBResolvesToTuneableCuratedModel() throws {
        let state = OnboardingFeatureState(
            step: .modelPick,
            pick: .curatedQwen35BSpeed
        )

        let model = try XCTUnwrap(state.resolvedModel)
        XCTAssertEqual(model.id, "qwen36-35b-a3b-optimized-speed")
        XCTAssertEqual(state.resolvedRepoID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed")
        XCTAssertEqual(state.resolvedModelFamily, "qwen3_6")
        XCTAssertTrue(state.supportsTune)
        XCTAssertTrue(model.supportsOnboardingTune)
        XCTAssertTrue(state.canAdvance)
    }

    func testOfficialModelCatalogExcludesStep37FlashMTPLX() throws {
        XCTAssertNil(MTPLXModelOption.option(matching: "stepfun-step37-flash"))
        XCTAssertNil(MTPLXModelOption.option(matching: "StepFun/Step-3.7-Flash-MTPLX-step3p5"))
        XCTAssertFalse(MTPLXModelOption.officialCatalog.contains { option in
            option.id.contains("step")
                || option.hfModelID.localizedCaseInsensitiveContains("StepFun")
        })
    }

    func testFourBPairCarriesModernTierMarkerSoInstalledCopiesStayVisible() throws {
        // The picker's installed-override hides any official entry whose
        // recommendedFor is empty (the orphan-protection that hid the broken
        // 4B). The rebuilt pair must carry the modern tier marker or an
        // installed copy is invisible on big-RAM Macs (2026-07-19 bug).
        let speed = try XCTUnwrap(MTPLXModelOption.option(matching: "qwen35-4b-optimized-speed"))
        let quality = try XCTUnwrap(MTPLXModelOption.option(matching: "qwen35-4b-optimized-quality"))

        XCTAssertEqual(speed.recommendedFor, [.modernApple])
        XCTAssertEqual(quality.recommendedFor, [.modernApple])
        // Unknown hardware keeps the conservative big-model-first list.
        XCTAssertFalse(MTPLXModelOption.recommendedCatalogIDs(for: nil).contains(speed.id))
    }

    func testFreshLegacySmallMemoryCatalogUses9BFP16AsMinimum() throws {
        let m2 = DetectedHardware(
            chipName: "Apple M2",
            appleSiliconGeneration: "m2",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m2,
            includeInstalledOverrides: false
        ).map(\.id)

        XCTAssertEqual(ids, ["qwen35-9b-optimized-speed-fp16"])
        XCTAssertFalse(ids.contains("qwen35-4b-optimized-speed"))
        XCTAssertFalse(ids.contains("qwen35-9b-optimized-speed"))
        XCTAssertFalse(ids.contains { $0.contains("step") })
    }

    func testFreshLegacyLargeMemoryCatalogRoutesQualityToFP16Sibling() throws {
        // The M1/M2 quality pick resolves the Quality-FP16 sibling
        // (2.0.1, 2026-07-07) — same policy as the speed lane.
        let m2 = DetectedHardware(
            chipName: "Apple M2 Ultra",
            appleSiliconGeneration: "m2",
            unifiedMemoryBytes: 64 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m2,
            includeInstalledOverrides: false
        ).map(\.id)

        // The Qwen 3.8 trio leads as its FP16 precision siblings (2026-08-15).
        XCTAssertEqual(ids, [
            "qwen38-27b-optimized-speed-fp16",
            "qwen38-27b-bare-speed-fp16",
            "qwen38-27b-optimized-quality-fp16",
            "optimized-speed-fp16",
            "optimized-quality-fp16",
            "qwen36-35b-a3b-optimized-speed-fp16",
            "qwen36-35b-a3b-optimized-balance-fp16",
            "gemma4-optimized-speed",
            "qwen35-9b-optimized-speed-fp16",
        ])
        XCTAssertFalse(ids.contains("optimized-quality"))
        XCTAssertFalse(ids.contains("qwen38-27b-optimized-speed"))
    }

    func testFreshLegacy32GiBCatalogLeadsWithQwen38FP16AndDropsQualityFP16() throws {
        // A 32 GiB M1/M2 keeps Optimized Speed FP16 (25 GiB peak) and Bare
        // Speed FP16 (20 GiB) but not Optimized Quality FP16 (33 GiB peak).
        let m1 = DetectedHardware(
            chipName: "Apple M1 Max",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 32 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m1,
            includeInstalledOverrides: false
        ).map(\.id)

        XCTAssertEqual(Array(ids.prefix(2)), ["qwen38-27b-optimized-speed-fp16", "qwen38-27b-bare-speed-fp16"])
        XCTAssertFalse(ids.contains("qwen38-27b-optimized-quality-fp16"))
        XCTAssertFalse(ids.contains("qwen38-27b-optimized-speed"))
    }

    func testQwen38FP16SiblingsMirrorParentsAndResolveOnLegacySilicon() throws {
        for base in ["qwen38-27b-bare-speed", "qwen38-27b-optimized-speed", "qwen38-27b-optimized-quality"] {
            let parent = try XCTUnwrap(MTPLXModelOption.option(matching: base))
            let sibling = try XCTUnwrap(MTPLXModelOption.option(matching: "mtplx-\(base)-fp16"))
            XCTAssertEqual(sibling.id, "\(base)-fp16")
            XCTAssertEqual(sibling.hfModelID, parent.hfModelID + "-FP16")
            XCTAssertEqual(sibling.peakMemoryGiB, parent.peakMemoryGiB)
            XCTAssertEqual(sibling.recommendedFor, [.legacyApple])
            XCTAssertEqual(parent.recommendedFor, [.modernApple])
            XCTAssertTrue(sibling.detail.contains("FP16 build for M1 and M2 Macs"))
        }
        // The OpenCode config names the id the server advertises for the sibling.
        XCTAssertEqual(
            OpenCodeIntegration.modelID(for: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"),
            "mtplx-qwen38-27b-optimized-speed-fp16"
        )
        XCTAssertEqual(
            OpenCodeIntegration.modelID(for: "/Users/example/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed-FP16"),
            "mtplx-qwen38-27b-bare-speed-fp16"
        )
        XCTAssertEqual(
            OpenCodeIntegration.modelID(for: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"),
            "mtplx-qwen38-27b-optimized-quality"
        )
    }

    func testOfficialModelCatalogIncludesOptimizedQualityFP16() throws {
        let quality = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen36-27b-optimized-quality-fp16")
        )
        XCTAssertEqual(quality.id, "optimized-quality-fp16")
        XCTAssertEqual(
            quality.hfModelID,
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16"
        )
        XCTAssertTrue(quality.recommendedFor.contains(.legacyApple))
    }

    func testOfficialModelCatalogIncludesQwen38BareSpeed() throws {
        let bare = try XCTUnwrap(
            MTPLXModelOption.option(matching: "mtplx-qwen38-27b-bare-speed")
        )

        XCTAssertEqual(bare.id, "qwen38-27b-bare-speed")
        XCTAssertEqual(bare.hfModelID, "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed")
        XCTAssertEqual(bare.displayName, "Qwen 3.8 27B Bare Speed")
        XCTAssertEqual(bare.shortName, "Qwen 3.8 27B Bare Speed")
        XCTAssertTrue(bare.recommendedFor.contains(.modernApple))
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: bare.hfModelID), "qwen3_8")
    }

    func testFreshModernSmallMemoryCatalogLeadsWith9BAndOffersFourBPair() throws {
        // The rebuilt 4B pair (2026-07-19) is recommendable again: the 16 GB
        // tier leads with the 9B and offers both 4B lanes behind it.
        let m5 = DetectedHardware(
            chipName: "Apple M5",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m5,
            includeInstalledOverrides: false
        ).map(\.id)

        XCTAssertEqual(ids, [
            "qwen35-9b-optimized-speed",
            "qwen35-4b-optimized-speed",
            "qwen35-4b-optimized-quality",
        ])
        XCTAssertFalse(ids.contains("qwen35-9b-optimized-speed-fp16"))
        XCTAssertFalse(ids.contains { $0.contains("step") })
    }

    func testFreshModernMidMemoryCatalogShowsStrongOptionsWithFourBTail() throws {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Pro",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 32 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m5,
            includeInstalledOverrides: false
        ).map(\.id)

        // 32 GiB: the Qwen 3.8 trio leads, minus Optimized Quality (33 GiB
        // peak) which the peak-memory filter hides on this tier.
        XCTAssertEqual(ids, [
            "qwen38-27b-optimized-speed",
            "qwen38-27b-bare-speed",
            "optimized-speed-v2",
            "optimized-speed",
            "qwen35-9b-optimized-speed",
            "gemma4-optimized-speed",
            "qwen36-35b-a3b-optimized-speed",
            "optimized-quality",
            "qwen35-4b-optimized-speed",
            "qwen35-4b-optimized-quality",
        ])
        XCTAssertFalse(ids.contains { $0.hasSuffix("-fp16") })
        XCTAssertFalse(ids.contains { $0.contains("step") })
    }

    func testFreshModern36GiBCatalogLeadsWithQwen38Trio() throws {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Pro",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 36 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m5,
            includeInstalledOverrides: false
        ).map(\.id)

        // 36 GiB clears Optimized Quality's 33 GiB peak, so the whole trio
        // leads, Optimized Speed (recommended) first, then the 3.6 V2 pair.
        XCTAssertEqual(Array(ids.prefix(5)), [
            "qwen38-27b-optimized-speed",
            "qwen38-27b-bare-speed",
            "qwen38-27b-optimized-quality",
            "optimized-speed-v2",
            "optimized-speed",
        ])
    }

    func testFreshModernLargeMemoryCatalogUnlocksBalanceWithoutFP16Siblings() throws {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m5,
            includeInstalledOverrides: false
        ).map(\.id)

        XCTAssertEqual(ids, [
            "qwen38-27b-optimized-speed",
            "qwen38-27b-bare-speed",
            "qwen38-27b-optimized-quality",
            "optimized-speed-v2",
            "optimized-speed",
            "optimized-quality",
            "qwen36-35b-a3b-optimized-speed",
            "qwen36-35b-a3b-optimized-balance",
            "gemma4-optimized-speed",
            "qwen35-9b-optimized-speed",
            "qwen35-4b-optimized-speed",
            "qwen35-4b-optimized-quality",
        ])
        XCTAssertFalse(ids.contains("qwen36-35b-a3b-optimized-speed-fp16"))
        XCTAssertFalse(ids.contains("qwen36-35b-a3b-optimized-balance-fp16"))
        XCTAssertFalse(ids.contains { $0.contains("step") })
    }

    func testCurrentModelStaysVisibleEvenWhenHardwareWouldHideIt() throws {
        let m5 = DetectedHardware(
            chipName: "Apple M5",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )

        let ids = MTPLXModelOption.hardwareAwareOfficialCatalog(
            hardware: m5,
            currentModel: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
            includeInstalledOverrides: false
        ).map(\.id)

        XCTAssertTrue(ids.contains("qwen36-35b-a3b-optimized-balance"))
    }

    func testModelInstallDetectionAcceptsGemmaAssistantPairBundle() throws {
        let root = temporaryDirectory()
        let pair = root.appendingPathComponent("Gemma4-MTPLX-Optimized-Speed", isDirectory: true)
        let target = pair.appendingPathComponent("target", isDirectory: true)
        let assistant = pair.appendingPathComponent("assistant", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: assistant, withIntermediateDirectories: true)
        try "{}".write(to: pair.appendingPathComponent("mtplx_pair.json"), atomically: true, encoding: .utf8)
        for directory in [target, assistant] {
            try "{}".write(to: directory.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
            try "{}".write(to: directory.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
            try Data([0]).write(to: directory.appendingPathComponent("model.safetensors"))
        }

        XCTAssertTrue(MTPLXModelOption.hasCompleteInstall(at: pair.path))
    }

    func testModelInstallDetectionAcceptsConfiguredNestedMTPSidecar() throws {
        let root = temporaryDirectory()
        let model = root.appendingPathComponent("Qwen3.6-27B-MTPLX-Optimized-Speed", isDirectory: true)
        let mtp = model.appendingPathComponent("mtp", isDirectory: true)
        try FileManager.default.createDirectory(at: mtp, withIntermediateDirectories: true)
        try """
        {"mlx_lm_extra_tensors": {"mtp_file": "mtp/weights.safetensors"}}
        """.write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("mtplx_runtime.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        try Data([0]).write(to: mtp.appendingPathComponent("weights.safetensors"))

        XCTAssertTrue(MTPLXModelOption.hasCompleteInstall(at: model.path))
    }

    func testModelInstallDetectionCanBeDisabledForFreshUserQA() throws {
        unsetenv("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN")
        let root = temporaryDirectory()
        let model = root.appendingPathComponent("Qwen3.6-27B-MTPLX-Optimized-Speed", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try "{}".write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("mtplx_runtime.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        let option = MTPLXModelOption(
            id: "qa",
            displayName: "QA",
            shortName: "QA",
            detail: "QA",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            localCandidates: [model.path]
        )

        XCTAssertEqual(option.installedLocalPath, model.path)

        setenv("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN", "1", 1)
        defer { unsetenv("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN") }

        XCTAssertNil(option.installedLocalPath)
        XCTAssertEqual(option.resolvedReference, "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
    }

    func testModelInstallDetectionRespectsLaunchHomeForTildeCandidates() throws {
        unsetenv("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN")
        let previousHome = getenv("HOME").map { String(cString: $0) }
        let root = temporaryDirectory()
        setenv("HOME", root.path, 1)
        defer {
            if let previousHome {
                setenv("HOME", previousHome, 1)
            } else {
                unsetenv("HOME")
            }
        }

        let model = root
            .appendingPathComponent(".mtplx/models/Youssofal--Qwen3.5-4B-MTPLX-Optimized-Speed", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try "{}".write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("mtplx_runtime.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        let option = MTPLXModelOption(
            id: "qwen35-4b-optimized-speed",
            displayName: "Qwen 3.5 4B Optimized Speed",
            shortName: "Qwen 3.5 4B Optimized Speed",
            detail: "QA",
            hfModelID: "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed",
            localCandidates: ["~/.mtplx/models/Youssofal--Qwen3.5-4B-MTPLX-Optimized-Speed"]
        )

        XCTAssertEqual(option.installedLocalPath, model.path)
        XCTAssertEqual(option.resolvedReference, model.path)
    }

    func testCustomHuggingFaceModelNormalizesAndMatchesCachePath() throws {
        let option = try XCTUnwrap(
            MTPLXModelOption.customHuggingFaceModel(
                repoID: "https://huggingface.co/Foo/Bar-7B/tree/main"
            )
        )

        XCTAssertEqual(option.id, "custom-foo--bar-7b")
        XCTAssertEqual(option.hfModelID, "Foo/Bar-7B")
        XCTAssertEqual(option.shortName, "Bar-7B")
        XCTAssertEqual(option.localCandidates.first, "~/.mtplx/models/Foo--Bar-7B")
        XCTAssertTrue(option.matches("Foo/Bar-7B"))
        XCTAssertTrue(option.matches("~/Documents/MTPLX/models/Bar-7B"))
        XCTAssertTrue(option.matches("~/Documents/MTPLX/hf-staging/Bar-7B"))
        XCTAssertTrue(option.matches("~/Documents/MTPLX/models/hf-release/Bar-7B"))
        XCTAssertTrue(option.matches("~/.mtplx/models/Foo--Bar-7B"))
        XCTAssertNil(MTPLXModelOption.customHuggingFaceModel(repoID: "not-a-repo"))
    }

    func testPickerCatalogAppendsUserModelsWithoutDuplicatingOfficialModels() throws {
        var config = MTPLXAppConfiguration()
        config.rememberCustomModel(repoID: "Foo/Bar")
        config.rememberCustomModel(repoID: "Foo/Bar")
        config.rememberCustomModel(repoID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")

        let catalog = MTPLXModelOption.pickerCatalog(
            customModels: config.customModels,
            currentModel: "Foo/Bar"
        )

        XCTAssertEqual(config.customModels.map(\.hfModelID), ["Foo/Bar"])
        XCTAssertEqual(catalog.filter { $0.matches("Foo/Bar") }.count, 1)
        XCTAssertEqual(
            catalog.filter { $0.matches("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed") }.count,
            1
        )
    }

    func testCommandBuilderEmitsBatchingRuntimeSettings() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulerMode: "ar_batch",
                batchingPreset: "agent",
                maxActiveRequests: 4,
                decodeBatchMax: 4,
                batchWaitMs: 3,
                prefillChunkTokens: 2048,
                experimentalMTPCohorts: true
            )
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "ar_batch"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "agent"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--max-active-requests", "4"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--decode-batch-max", "4"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batch-wait-ms", "3.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--prefill-chunk-tokens", "2048"]))
        XCTAssertTrue(command.arguments.contains("--experimental-mtp-cohorts"))
    }

    func testCommandBuilderEmitsSSDSessionCacheRuntimeSettings() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                ssdSessionCache: "on",
                ssdSessionCacheDir: "/tmp/mtplx-session-bank",
                ssdSessionCacheMaxSize: "100GB",
                ssdSessionCacheMinPrefixTokens: 512
            )
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache", "on"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-dir", "/tmp/mtplx-session-bank"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-max-size", "100GB"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--ssd-session-cache-min-prefix-tokens", "512"]))
    }

    func testSettingsStoreRoundTripsConfiguration() throws {
        let url = temporaryDirectory().appendingPathComponent("settings.json")
        let store = MTPLXSettingsStore(settingsURL: url)
        let configuration = MTPLXAppConfiguration(
            executablePath: "/usr/local/bin/mtplx",
            model: "model-a",
            profile: "performance-cold",
            host: "127.0.0.1",
            port: 9000,
            generationMode: "ar",
            loadMTP: false,
            schedulerMode: "ar_batch",
            batchingPreset: "agent",
            schedulingPreset: "agent",
            maxActiveRequests: 4,
            decodeBatchMax: 4,
            batchWaitMs: 3,
            prefillChunkTokens: 2048,
            experimentalMTPCohorts: true,
            ramSessionCachePolicy: "bounded",
            ramSessionBlockPrefixRestore: false,
            ramSessionCacheMaxEntries: 2,
            ramSessionCacheMaxSize: "4G",
            ramSessionCachePerSessionMaxSize: "2G",
            pagedKVQuantization: "q8",
            ssdSessionCache: "on",
            ssdSessionCacheDir: "/tmp/mtplx-session-bank",
            ssdSessionCacheMaxSize: "100GB",
            ssdSessionCacheMinPrefixTokens: 512,
            contextWindow: 8192,
            apiKey: "key",
            enableThermalPolling: true,
            streamSnapshotIntervalMs: 1000,
            performanceLock: true,
            launchDaemonOnOpen: false
        )

        try store.save(configuration)
        // Decode runs sanitize (one-shot migrations consume their
        // flags); the loaded config is the sanitized form.
        var expected = configuration
        expected.sanitizeLaunchCriticalFields()
        XCTAssertEqual(try store.load(), expected)
    }

    func testSettingsStoreSupportsEnvironmentOverride() throws {
        let url = MTPLXSettingsStore.defaultSettingsURL(
            environment: ["MTPLX_APP_SETTINGS_PATH": "~/tmp/mtplx-settings.json"],
            arguments: ["MTPLXApp"]
        )

        XCTAssertTrue(url.path.hasSuffix("/tmp/mtplx-settings.json"))
        XCTAssertFalse(url.path.contains("~"))
    }

    func testSettingsStoreSupportsArgumentOverride() throws {
        let expected = temporaryDirectory().appendingPathComponent("settings.json")
        let url = MTPLXSettingsStore.defaultSettingsURL(
            environment: [:],
            arguments: ["MTPLXApp", "--mtplx-app-settings", expected.path]
        )

        XCTAssertEqual(url.path, expected.path)
    }

    func testSettingsStoreSupportsEqualsArgumentOverride() throws {
        let expected = temporaryDirectory().appendingPathComponent("settings-equals.json")
        let url = MTPLXSettingsStore.defaultSettingsURL(
            environment: [:],
            arguments: ["MTPLXApp", "--mtplx-settings-path=\(expected.path)"]
        )

        XCTAssertEqual(url.path, expected.path)
    }

    @MainActor
    func testBackendStoreExposesActualSettingsURL() throws {
        let expected = temporaryDirectory().appendingPathComponent("settings-about.json")
        let backend = MTPLXBackendStore(
            settingsStore: MTPLXSettingsStore(settingsURL: expected)
        )

        XCTAssertEqual(backend.settingsURL, expected)
    }

    func testAppConfigurationPersistsHermesResumeState() throws {
        let url = temporaryDirectory().appendingPathComponent("settings.json")
        let store = MTPLXSettingsStore(settingsURL: url)
        let configuration = MTPLXAppConfiguration(
            lastLaunchTarget: LaunchTarget.hermes.rawValue,
            hermesWorkspacePath: "/tmp/hermes-workspace",
            lastHermesProfile: "research",
            lastHermesSessionID: "session-123",
            lastHermesSessionTitle: "Fix the app"
        )

        try store.save(configuration)
        let loaded = try store.load()

        XCTAssertEqual(loaded.lastLaunchTarget, LaunchTarget.hermes.rawValue)
        XCTAssertEqual(loaded.hermesWorkspacePath, "/tmp/hermes-workspace")
        XCTAssertEqual(loaded.lastHermesProfile, "research")
        XCTAssertEqual(loaded.lastHermesSessionID, "session-123")
        XCTAssertEqual(loaded.lastHermesSessionTitle, "Fix the app")
    }

    func testSettingsDecodeTreatsLegacySerialLatencyAsTargetDefault() throws {
        let data = Data("""
        {
          "scheduler_mode": "serial",
          "batching_preset": "latency"
        }
        """.utf8)

        let configuration = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)

        XCTAssertEqual(configuration.schedulingPreset, "target-default")
        XCTAssertEqual(configuration.schedulerMode, "serial")
        XCTAssertEqual(configuration.batchingPreset, "latency")
    }

    func testSettingsDecodeTreatsLegacyAgentPairAsTargetDefault() throws {
        let data = Data("""
        {
          "scheduler_mode": "ar_batch",
          "batching_preset": "agent"
        }
        """.utf8)

        let configuration = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)

        XCTAssertEqual(configuration.schedulingPreset, "target-default")
        XCTAssertEqual(configuration.schedulerMode, "ar_batch")
        XCTAssertEqual(configuration.batchingPreset, "agent")
    }

    func testSettingsDecodePreservesExplicitAgentSchedulingOverride() throws {
        let data = Data("""
        {
          "scheduler_mode": "ar_batch",
          "batching_preset": "agent",
          "scheduling_preset": "agent"
        }
        """.utf8)

        let configuration = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)

        XCTAssertEqual(configuration.schedulingPreset, "agent")
        XCTAssertEqual(configuration.schedulerMode, "ar_batch")
        XCTAssertEqual(configuration.batchingPreset, "agent")
    }

    func testApplySchedulingPresetClearsStaleAdvancedBatchOverrides() throws {
        var configuration = MTPLXAppConfiguration(
            schedulerMode: "ar_batch",
            batchingPreset: "agent",
            schedulingPreset: "agent",
            maxActiveRequests: 4,
            decodeBatchMax: 4,
            batchWaitMs: 50
        )

        configuration.applySchedulingPreset("latency")

        XCTAssertEqual(configuration.schedulingPreset, "latency")
        XCTAssertEqual(configuration.schedulerMode, "serial")
        XCTAssertEqual(configuration.batchingPreset, "latency")
        XCTAssertNil(configuration.maxActiveRequests)
        XCTAssertNil(configuration.decodeBatchMax)
        XCTAssertNil(configuration.batchWaitMs)
    }

    func testCommandBuilderThroughputOverrideUsesBackendThroughputPreset() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                schedulingPreset: "throughput"
            ),
            target: .openCode,
            launchID: "opencode-throughput"
        )

        XCTAssertTrue(command.arguments.containsInOrder(["--scheduler-mode", "ar_batch"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batching-preset", "throughput"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--max-active-requests", "8"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--decode-batch-max", "8"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--batch-wait-ms", "20.0"]))
        XCTAssertTrue(command.arguments.containsInOrder(["--prefill-chunk-tokens", "2048"]))
    }

    func testMutableSettingsIncludesPrefillChunkTokens() throws {
        let settings = MutableSettings(
            generationMode: "ar",
            depth: 3,
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            prefillChunkTokens: 8192
        )
        let data = try JSONEncoder().encode(settings)
        let root = try JSONDecoder().decode([String: JSONValue].self, from: data)

        XCTAssertEqual(root["generation_mode"]?.stringValue, "ar")
        XCTAssertEqual(root["depth"]?.intValue, 3)
        XCTAssertEqual(root["temperature"]?.doubleValue, 0.6)
        XCTAssertEqual(root["top_p"]?.doubleValue, 0.95)
        XCTAssertEqual(root["top_k"]?.intValue, 20)
        XCTAssertEqual(root["prefill_chunk_tokens"]?.intValue, 8192)

        let decoded = try JSONDecoder().decode(MutableSettings.self, from: data)
        XCTAssertEqual(decoded.generationMode, "ar")
        XCTAssertEqual(decoded.prefillChunkTokens, 8192)
    }

    func testMetricsLatestBuildsPerDepthAcceptanceRows() throws {
        let latest = try JSONDecoder().decode(
            MetricsLatest.self,
            from: Data("""
            {
              "accepted_by_depth": [4, 2],
              "drafted_by_depth": [5, 4],
              "accepted_drafts": 10,
              "drafted_tokens": 20
            }
            """.utf8)
        )

        XCTAssertEqual(
            latest.acceptanceCounterRows(),
            [
                AcceptanceCounterRow(label: "D1", accepted: 4, drafted: 5),
                AcceptanceCounterRow(label: "D2", accepted: 2, drafted: 4),
            ]
        )
    }

    func testMetricsLatestBuildsAggregateAcceptanceFallback() throws {
        let latest = try JSONDecoder().decode(
            MetricsLatest.self,
            from: Data("""
            {
              "accepted_drafts": 7,
              "drafted_tokens": 10
            }
            """.utf8)
        )

        let rows = latest.acceptanceCounterRows()

        XCTAssertEqual(rows, [AcceptanceCounterRow(label: "ALL", accepted: 7, drafted: 10)])
        let rate = try XCTUnwrap(rows.first?.rate)
        XCTAssertEqual(rate, 0.7, accuracy: 0.0001)
    }

    func testMetricsLatestDistinguishesMissingCacheVerdictFromMiss() throws {
        let unknown = try JSONDecoder().decode(
            MetricsLatest.self,
            from: Data(#"{}"#.utf8)
        )
        let miss = try JSONDecoder().decode(
            MetricsLatest.self,
            from: Data(#"{"session_cache_hit":false}"#.utf8)
        )
        let hit = try JSONDecoder().decode(
            MetricsLatest.self,
            from: Data(#"{"session_cache_hit":true}"#.utf8)
        )

        XCTAssertEqual(unknown.requestCacheVerdict, .unknown)
        XCTAssertEqual(miss.requestCacheVerdict, .miss)
        XCTAssertEqual(hit.requestCacheVerdict, .hit)
    }

    @MainActor
    func testStoppedLiveSettingsStayAvailableForBenchmarkStart() async throws {
        let settingsStore = MTPLXSettingsStore(
            settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
        )
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: 9),
            settingsStore: settingsStore
        )
        let wanted = MutableSettings(
            generationMode: "ar",
            depth: 3,
            temperature: 1.0,
            topP: 0.8,
            topK: 64,
            enableThinking: true,
            reasoning: "on"
        )

        try await backend.updateLiveSettings(wanted)
        XCTAssertEqual(backend.settings, wanted)

        let options = BenchmarkStartOptions(settings: backend.settings)
        XCTAssertEqual(options.temperature, 1.0)
        XCTAssertEqual(options.topP, 0.8)
        XCTAssertEqual(options.topK, 64)
        XCTAssertEqual(options.enableThinking, true)
        XCTAssertEqual(options.questionProcessIsolation, "per_question")

        let reloadedBackend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: 9),
            settingsStore: settingsStore
        )
        reloadedBackend.loadPersistedSettings()
        let reloadedOptions = BenchmarkStartOptions(settings: reloadedBackend.settings)
        XCTAssertEqual(reloadedBackend.settings?.generationMode, "ar")
        XCTAssertEqual(reloadedOptions.temperature, 1.0)
        XCTAssertEqual(reloadedOptions.topP, 0.8)
        XCTAssertEqual(reloadedOptions.topK, 64)
        XCTAssertEqual(reloadedOptions.enableThinking, true)
        XCTAssertEqual(reloadedOptions.questionProcessIsolation, "per_question")
    }

    @MainActor
    func testPartialLiveSettingsUpdatesPreserveExistingSamplerDraft() async throws {
        let settingsStore = MTPLXSettingsStore(
            settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
        )
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: "/models/qwen",
                port: 9
            ),
            settingsStore: settingsStore
        )

        try await backend.updateLiveSettings(MutableSettings(
            generationMode: "ar",
            depth: 3,
            temperature: 1.0,
            topP: 0.8,
            topK: 35,
            enableThinking: true,
            reasoning: "on",
            prefillChunkTokens: 2048
        ))
        try await backend.updateLiveSettings(MutableSettings(
            enableThinking: false,
            reasoning: "off"
        ))

        XCTAssertEqual(backend.settings?.temperature, 1.0)
        XCTAssertEqual(backend.settings?.generationMode, "ar")
        XCTAssertEqual(backend.settings?.topP, 0.8)
        XCTAssertEqual(backend.settings?.topK, 35)
        XCTAssertEqual(backend.settings?.depth, 3)
        XCTAssertEqual(backend.settings?.prefillChunkTokens, 2048)
        XCTAssertEqual(backend.settings?.reasoning, "off")
        XCTAssertEqual(backend.settings?.enableThinking, false)

        backend.clearLiveMetricsState()
        XCTAssertEqual(backend.settings?.temperature, 1.0)
        XCTAssertEqual(backend.settings?.generationMode, "ar")
        XCTAssertEqual(backend.settings?.topP, 0.8)
        XCTAssertEqual(backend.settings?.topK, 35)
        XCTAssertEqual(backend.settings?.reasoning, "off")
        XCTAssertEqual(backend.settings?.enableThinking, false)
    }

    @MainActor
    func testLiveSettingsRiderWithoutDepthDoesNotClobberTunedRecord() async throws {
        // QA-104 leg B: chat-open pushes reasoning/sampler riders whose
        // MERGED snapshot carries the daemon's current depth. Persisting
        // that as a "tuned" record erased the user's onboarding tune
        // with the launch preset's depth. Only an explicit depth in the
        // caller's own patch may write the tuned record.
        let settingsStore = MTPLXSettingsStore(
            settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
        )
        let model = "/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"
        let tunedAt = Date(timeIntervalSince1970: 1_780_000_000)
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: model,
                port: 9,
                lastTunedDepth: 2,
                tunedControlRecord: TunedControlRecord(
                    modelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                    modelFamily: "qwen3_6",
                    backendID: "qwen3_next",
                    controlField: "depth",
                    controlValue: 2,
                    candidates: ["Base speeds", "MTP 1", "MTP 2", "MTP 3"],
                    tunedAt: tunedAt
                )
            ),
            settingsStore: settingsStore
        )

        // Reasoning-only rider: no depth in the caller's patch.
        try await backend.updateLiveSettings(MutableSettings(
            reasoning: "off"
        ))

        XCTAssertEqual(
            backend.configuration.lastTunedDepth, 2,
            "a reasoning rider must not rewrite the tuned depth"
        )
        XCTAssertEqual(backend.configuration.tunedControlRecord?.controlValue, 2)
        XCTAssertEqual(backend.configuration.tunedControlRecord?.tunedAt, tunedAt)

        // An explicit depth choice still persists (existing contract).
        try await backend.updateLiveSettings(MutableSettings(
            generationMode: "mtp",
            depth: 3
        ))
        XCTAssertEqual(backend.configuration.lastTunedDepth, 3)
        XCTAssertEqual(backend.configuration.tunedControlRecord?.controlValue, 3)
    }

    @MainActor
    func testLiveSettingsPersistReasoningAndDraftDepthAcrossReload() async throws {
        let settingsStore = MTPLXSettingsStore(
            settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
        )
        let model = "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: model,
                port: 9
            ),
            settingsStore: settingsStore
        )

        try await backend.updateLiveSettings(MutableSettings(
            generationMode: "mtp",
            depth: 2,
            temperature: 0.7,
            topP: 0.9,
            topK: 32,
            enableThinking: true,
            reasoning: "on",
            prefillChunkTokens: 2048
        ))

        XCTAssertEqual(backend.configuration.reasoning, "on")
        XCTAssertEqual(backend.configuration.generationMode, "mtp")
        XCTAssertEqual(backend.configuration.lastTunedDepth, 2)
        XCTAssertEqual(backend.configuration.tunedControlRecord?.controlField, "depth")
        XCTAssertEqual(backend.configuration.tunedControlRecord?.controlValue, 2)

        let reloadedBackend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: model,
                port: 9
            ),
            settingsStore: settingsStore
        )
        reloadedBackend.loadPersistedSettings()

        XCTAssertEqual(reloadedBackend.settings?.generationMode, "mtp")
        XCTAssertEqual(reloadedBackend.settings?.depth, 2)
        XCTAssertEqual(reloadedBackend.settings?.temperature, 0.7)
        XCTAssertEqual(reloadedBackend.settings?.topP, 0.9)
        XCTAssertEqual(reloadedBackend.settings?.topK, 32)
        XCTAssertEqual(reloadedBackend.settings?.reasoning, "on")
        XCTAssertEqual(reloadedBackend.settings?.enableThinking, true)
    }

    func testLiveSettingsPatchDoesNotEchoDescriptorFields() throws {
        let fullSettings = MutableSettings(
            generationMode: "ar",
            depth: 1,
            depthMax: 3,
            draftControl: DraftControl(
                supported: true,
                requestField: "depth",
                displayLabel: "Draft depth",
                defaultValue: 1,
                minimum: 1,
                maximum: 3,
                unit: "depth",
                valueLabels: ["D1", "D2", "D3"]
            ),
            modelControls: ModelControls(
                schemaVersion: 1,
                modelFamily: "step",
                backendID: "step3p5_mtp",
                reasoning: ReasoningPolicy(
                    supported: true,
                    parser: "step3p5",
                    defaultMode: "auto",
                    effortLevels: ["low", "medium", "high"],
                    defaultEffort: "low"
                )
            ),
            modelFamily: "step",
            architectureID: "step3p5-mtp",
            supportLevel: "experimental_contract_gated",
            reasoningPolicy: ReasoningPolicy(
                supported: true,
                parser: "step3p5",
                defaultMode: "auto",
                effortLevels: ["low", "medium", "high"],
                defaultEffort: "low"
            ),
            samplingDefaults: SamplingDefaults(
                temperature: 0.6,
                topP: 0.95,
                topK: 20,
                familyDefaultReason: "Step native MTP sampler"
            ),
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            enableThinking: false,
            reasoningParser: "step3p5",
            reasoning: "off",
            reasoningEffort: "low",
            prefillChunkTokens: 2048
        )

        let patch = MTPLXBackendStore.liveMutableSettingsPatch(from: fullSettings)
        let data = try JSONEncoder().encode(patch)
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(root["reasoning"] as? String, "off")
        XCTAssertEqual(root["enable_thinking"] as? Bool, false)
        XCTAssertEqual(root["reasoning_effort"] as? String, "low")
        XCTAssertEqual(root["generation_mode"] as? String, "ar")
        XCTAssertNil(root["model_controls"])
        XCTAssertNil(root["draft_control"])
        XCTAssertNil(root["reasoning_policy"])
        XCTAssertNil(root["model_family"])
        XCTAssertNil(root["sampling_defaults"])
        XCTAssertNil(root["depth_max"])
    }

    func testChatTargetKeepsQwenReasoningWhenCarryingLiveSettings() throws {
        let settings = MutableSettings(
            generationMode: "mtp",
            depth: 3,
            temperature: 1.0,
            topP: 0.8,
            topK: 35,
            enableThinking: true,
            reasoningParser: "qwen3",
            reasoning: "on",
            reasoningEffort: "high",
            prefillChunkTokens: 2048
        )
        let configuration = MTPLXAppConfiguration(
            model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
            liveSettingsModelFamily: "qwen3_6",
            lastLaunchTarget: LaunchTarget.chat.rawValue
        )

        let carried = try XCTUnwrap(MTPLXBackendStore.liveSettingsCarriedIntoTarget(
            settings,
            target: .chat,
            configuration: configuration
        ))

        XCTAssertEqual(carried.temperature, 1.0)
        XCTAssertEqual(carried.topP, 0.8)
        XCTAssertEqual(carried.topK, 35)
        XCTAssertEqual(carried.prefillChunkTokens, 2048)
        XCTAssertEqual(carried.reasoningParser, "qwen3")
        XCTAssertEqual(carried.enableThinking, true)
        XCTAssertEqual(carried.reasoning, "on")
        XCTAssertEqual(carried.reasoningEffort, "high")
    }

    func testChatTargetKeepsStepReasoningWhenCarryingLiveSettings() throws {
        let settings = MutableSettings(
            temperature: 0.6,
            enableThinking: true,
            reasoning: "on",
            reasoningEffort: "high"
        )
        let configuration = MTPLXAppConfiguration(
            model: "/models/Step-3.7-Flash-MTPLX-step3p5",
            liveSettingsModelFamily: "step",
            lastLaunchTarget: LaunchTarget.chat.rawValue
        )

        let carried = try XCTUnwrap(MTPLXBackendStore.liveSettingsCarriedIntoTarget(
            settings,
            target: .chat,
            configuration: configuration
        ))

        XCTAssertEqual(carried.temperature, 0.6)
        XCTAssertEqual(carried.enableThinking, true)
        XCTAssertEqual(carried.reasoning, "on")
        XCTAssertEqual(carried.reasoningEffort, "high")
    }

    @MainActor
    func testLaunchTargetsCarryCompatibleAppOwnedSamplerOnStart() async throws {
        let settingsStore = MTPLXSettingsStore(
            settingsURL: temporaryDirectory().appendingPathComponent("settings.json")
        )
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: "/models/qwen",
                port: 9
            ),
            settingsStore: settingsStore
        )

        try await backend.updateLiveSettings(MutableSettings(
            temperature: 1.0,
            topP: 0.8,
            topK: 35,
            reasoning: "on"
        ))

        backend.clearLiveMetricsState(target: .openCode)

        XCTAssertEqual(backend.settings?.temperature, 1.0)
        XCTAssertEqual(backend.settings?.topP, 0.8)
        XCTAssertEqual(backend.settings?.topK, 35)
        XCTAssertEqual(backend.settings?.reasoning, "on")
    }

    func testOpenCodeIntegrationWritesCurrentPortProviderHeadersAndNoHiddenCaps() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let legacyPluginURL = url.deletingLastPathComponent()
            .appendingPathComponent("mtplx-session-headers.js")
        let existing = """
        {
          "$schema": "https://opencode.ai/config.json",
          "provider": {
            "lmstudio": {"name": "keep me"},
            "mtplx": {"options": {"baseURL": "http://127.0.0.1:18119/v1"}}
          },
          "plugin": ["/tmp/keep-plugin.js", "/tmp/mtplx-session-headers.js"],
          "model": "mtplx/stale"
        }
        """
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(existing.utf8).write(to: url)
        try Data("legacy plugin".utf8).write(to: legacyPluginURL)

        let desktopSettingsURL = temporaryDirectory().appendingPathComponent("default.dat")
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: desktopSettingsURL
        )
        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                host: "0.0.0.0",
                port: 8000,
                contextWindow: nil
            )
        )

        XCTAssertTrue(result.didChange)
        XCTAssertEqual(result.baseURL, "http://127.0.0.1:8000/v1")
        XCTAssertEqual(result.modelReference, "mtplx/mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(result.legacySessionHeadersPluginPath, legacyPluginURL.path)
        XCTAssertNotNil(result.backupPath)
        XCTAssertEqual(result.reasoningVisibilityPath, desktopSettingsURL.path)
        XCTAssertTrue(result.reasoningVisibilityDidChange)

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        XCTAssertEqual(root["plugin"]?.arrayValue, [.string("/tmp/keep-plugin.js")])
        XCTAssertEqual(root["model"]?.stringValue, "mtplx/mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(root["small_model"]?.stringValue, "mtplx/mtplx-qwen36-27b-optimized-speed")

        let providers = try XCTUnwrap(root["provider"]?.objectValue)
        XCTAssertNotNil(providers["lmstudio"])
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let options = try XCTUnwrap(mtplx["options"]?.objectValue)
        XCTAssertEqual(options["baseURL"]?.stringValue, "http://127.0.0.1:8000/v1")
        XCTAssertEqual(options["headers"]?.objectValue?["x-mtplx-client"]?.stringValue, "opencode")

        let models = try XCTUnwrap(mtplx["models"]?.objectValue)
        let model = try XCTUnwrap(models["mtplx-qwen36-27b-optimized-speed"]?.objectValue)
        XCTAssertEqual(model["reasoning"]?.boolValue, false)
        XCTAssertNil(model["interleaved"])
        XCTAssertEqual(model["tool_call"]?.boolValue, true)
        XCTAssertEqual(model["temperature"]?.boolValue, false)
        XCTAssertNil(model["options"])
        XCTAssertFalse(root.recursivelyContainsKey("maxTokens"))
        XCTAssertFalse(root.recursivelyContainsKey("max_response_tokens"))

        XCTAssertTrue(FileManager.default.fileExists(atPath: desktopSettingsURL.path))
        let desktopRoot = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: desktopSettingsURL))
        let settingsText = try XCTUnwrap(desktopRoot["settings.v3"]?.stringValue)
        let settingsData = try XCTUnwrap(settingsText.data(using: .utf8))
        let settingsRoot = try JSONDecoder().decode([String: JSONValue].self, from: settingsData)
        XCTAssertEqual(settingsRoot["general"]?.objectValue?["showReasoningSummaries"]?.boolValue, true)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: url.deletingLastPathComponent().appendingPathComponent("package.json").path
            )
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: result.legacySessionHeadersPluginPath))
    }

    func testOpenCodeIntegrationUsesGemmaModelIdentityForGemmaBundles() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let desktopSettingsURL = temporaryDirectory().appendingPathComponent("default.dat")
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: desktopSettingsURL
        )

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/hf-release/Gemma4-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 18095,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/gemma4-mtplx-optimized-speed")

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        XCTAssertEqual(root["model"]?.stringValue, "mtplx/gemma4-mtplx-optimized-speed")
        XCTAssertEqual(root["small_model"]?.stringValue, "mtplx/gemma4-mtplx-optimized-speed")
        let providers = try XCTUnwrap(root["provider"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let models = try XCTUnwrap(mtplx["models"]?.objectValue)
        let model = try XCTUnwrap(models["gemma4-mtplx-optimized-speed"]?.objectValue)
        XCTAssertEqual(model["reasoning"]?.boolValue, false)
        XCTAssertEqual(model["temperature"]?.boolValue, false)
        XCTAssertNil(model["interleaved"])
        XCTAssertNil(model["options"])
    }

    func testOpenCodeIntegrationKeepsQwen35BModelIdentity() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let desktopSettingsURL = temporaryDirectory().appendingPathComponent("default.dat")
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: desktopSettingsURL
        )

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 18097,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/mtplx-qwen36-35b-a3b-optimized-speed")

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        XCTAssertEqual(root["model"]?.stringValue, "mtplx/mtplx-qwen36-35b-a3b-optimized-speed")
        let providers = try XCTUnwrap(root["provider"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let models = try XCTUnwrap(mtplx["models"]?.objectValue)
        XCTAssertNotNil(models["mtplx-qwen36-35b-a3b-optimized-speed"])
    }

    func testOpenCodeIntegrationKeepsQwen4BModelIdentity() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let desktopSettingsURL = temporaryDirectory().appendingPathComponent("default.dat")
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: desktopSettingsURL
        )

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 18098,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/qwen3.5-4b-mtplx-optimized-speed")

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        XCTAssertEqual(root["model"]?.stringValue, "mtplx/qwen3.5-4b-mtplx-optimized-speed")
        let providers = try XCTUnwrap(root["provider"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let models = try XCTUnwrap(mtplx["models"]?.objectValue)
        XCTAssertNotNil(models["qwen3.5-4b-mtplx-optimized-speed"])
    }

    func testOpenCodeIntegrationUsesStepfunModelIdentityAndStepPolicy() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let desktopSettingsURL = temporaryDirectory().appendingPathComponent("default.dat")
        let integration = OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: desktopSettingsURL
        )

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                host: "127.0.0.1",
                port: 18096,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/step-3.7-flash-mtplx-step3p5")

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        XCTAssertEqual(root["model"]?.stringValue, "mtplx/step-3.7-flash-mtplx-step3p5")
        XCTAssertEqual(root["small_model"]?.stringValue, "mtplx/step-3.7-flash-mtplx-step3p5")
        let providers = try XCTUnwrap(root["provider"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let models = try XCTUnwrap(mtplx["models"]?.objectValue)
        let model = try XCTUnwrap(models["step-3.7-flash-mtplx-step3p5"]?.objectValue)
        XCTAssertEqual(model["reasoning"]?.boolValue, false)
        XCTAssertEqual(model["temperature"]?.boolValue, false)
        XCTAssertNil(model["interleaved"])
        XCTAssertNil(model["options"])
    }

    func testPiIntegrationWritesCurrentPortAndNoHiddenCaps() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let existing = """
        {
          "providers": {
            "anthropic": {"baseUrl": "https://api.anthropic.com"},
            "mtplx": {
              "baseUrl": "http://127.0.0.1:18119/v1",
              "models": [{"id": "stale", "maxTokens": 4096}]
            }
          }
        }
        """
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(existing.utf8).write(to: url)

        let integration = PiIntegration(configURL: url)
        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                host: "0.0.0.0",
                port: 8000,
                contextWindow: nil
            )
        )

        XCTAssertTrue(result.didChange)
        XCTAssertEqual(result.baseURL, "http://127.0.0.1:8000/v1")
        XCTAssertEqual(result.modelReference, "mtplx/mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(
            result.launchCommand,
            "pi --model mtplx/mtplx-qwen36-27b-optimized-speed --tools read,bash,edit,write,grep,find,ls "
                + "--append-system-prompt '\(PiIntegration.agentOperatingHintsURL().path)'"
        )
        XCTAssertNotNil(result.backupPath)

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let providers = try XCTUnwrap(root["providers"]?.objectValue)
        XCTAssertNotNil(providers["anthropic"])
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        XCTAssertEqual(mtplx["baseUrl"]?.stringValue, "http://127.0.0.1:8000/v1")
        XCTAssertEqual(mtplx["api"]?.stringValue, "openai-completions")
        XCTAssertEqual(mtplx["apiKey"]?.stringValue, PiIntegration.localAPIKey)
        XCTAssertEqual(mtplx["authHeader"]?.boolValue, true)
        XCTAssertEqual(mtplx["headers"]?.objectValue?["x-mtplx-client"]?.stringValue, "pi")
        XCTAssertEqual(mtplx["compat"]?.objectValue?["maxTokensField"]?.stringValue, "max_tokens")
        XCTAssertEqual(mtplx["compat"]?.objectValue?["supportsDeveloperRole"]?.boolValue, false)
        XCTAssertEqual(mtplx["compat"]?.objectValue?["supportsReasoningEffort"]?.boolValue, false)

        let models = try XCTUnwrap(mtplx["models"]?.arrayValue)
        let model = try XCTUnwrap(models.first?.objectValue)
        XCTAssertEqual(model["id"]?.stringValue, "mtplx-qwen36-27b-optimized-speed")
        XCTAssertEqual(model["reasoning"]?.boolValue, true)
        XCTAssertEqual(model["contextWindow"]?.intValue, 131_072)
        XCTAssertFalse(root.recursivelyContainsKey("maxTokens"))
        XCTAssertFalse(root.recursivelyContainsKey("max_response_tokens"))
    }

    func testPiIntegrationUsesGemmaModelIdentityForGemmaBundles() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/hf-release/Gemma4-MTPLX-Optimized-Speed",
                host: "127.0.0.1",
                port: 18095,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/gemma4-mtplx-optimized-speed")
        XCTAssertTrue(result.launchCommand.contains("--model mtplx/gemma4-mtplx-optimized-speed"))

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let providers = try XCTUnwrap(root["providers"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        let models = try XCTUnwrap(mtplx["models"]?.arrayValue)
        let model = try XCTUnwrap(models.first?.objectValue)
        XCTAssertEqual(model["id"]?.stringValue, "gemma4-mtplx-optimized-speed")
    }

    func testPiIntegrationUsesStepfunModelIdentity() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)

        let result = try integration.sync(
            configuration: MTPLXAppConfiguration(
                model: "/Users/youssof/Documents/MTPLX/models/Step-3.7-Flash-MTPLX-step3p5",
                host: "127.0.0.1",
                port: 18096,
                contextWindow: nil
            )
        )

        XCTAssertEqual(result.modelReference, "mtplx/step-3.7-flash-mtplx-step3p5")
        XCTAssertTrue(result.launchCommand.contains("--model mtplx/step-3.7-flash-mtplx-step3p5"))

        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let providers = try XCTUnwrap(root["providers"]?.objectValue)
        let mtplx = try XCTUnwrap(providers["mtplx"]?.objectValue)
        XCTAssertEqual(mtplx["compat"]?.objectValue?["supportsReasoningEffort"]?.boolValue, true)
        XCTAssertEqual(mtplx["compat"]?.objectValue?["reasoningEffort"]?.stringValue, "low")
        let models = try XCTUnwrap(mtplx["models"]?.arrayValue)
        let model = try XCTUnwrap(models.first?.objectValue)
        XCTAssertEqual(model["id"]?.stringValue, "step-3.7-flash-mtplx-step3p5")
        XCTAssertEqual(model["reasoning"]?.boolValue, true)
    }

    func testPiLaunchUsesConfiguredAgentWorkspace() throws {
        let workspace = temporaryDirectory()
            .appendingPathComponent("bowmasters", isDirectory: true)
        try FileManager.default.createDirectory(
            at: workspace,
            withIntermediateDirectories: true
        )

        let configuration = MTPLXAppConfiguration(
            hermesWorkspacePath: workspace.path
        )

        XCTAssertEqual(
            PiIntegration.resolvedWorkspacePath(configuration: configuration),
            workspace.path
        )
        XCTAssertTrue(
            PiIntegration.launchCommand(for: configuration.model)
                .contains("--tools read,bash,edit,write,grep,find,ls")
        )
        XCTAssertTrue(
            PiIntegration.launchCommand(for: configuration.model)
                .contains("--append-system-prompt")
        )
        XCTAssertTrue(
            PiIntegration.agentOperatingHints.contains("converge after roughly 10 to 14 tool calls")
        )
        XCTAssertTrue(PiIntegration.isPiAgentCommand("pi"))
        XCTAssertTrue(
            PiIntegration.isPiAgentCommand(
                "/opt/homebrew/bin/pi --model mtplx/mtplx-qwen36-27b-optimized-speed --tools read,bash,edit,write,grep,find,ls --append-system-prompt /Users/example/.mtplx/pi-agent-operating-hints.md"
            )
        )
        XCTAssertTrue(
            PiIntegration.isPiAgentCommand(
                "/opt/homebrew/bin/node /opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/dist/cli.js --model mtplx/mtplx-qwen36-27b-optimized-speed --tools read,bash"
            )
        )
        XCTAssertTrue(
            PiIntegration.isPiAgentCommand(
                "/opt/homebrew/bin/node /opt/homebrew/bin/pi --model mtplx/mtplx-qwen36-27b-optimized-speed"
            )
        )
        XCTAssertFalse(
            PiIntegration.isPiAgentCommand("/bin/zsh -lc ps -axo pid=,command= | rg pi")
        )
        XCTAssertFalse(
            PiIntegration.isPiAgentCommand(
                "/opt/homebrew/bin/node /tmp/other/cli.js --model mtplx/mtplx-qwen36-27b-optimized-speed"
            )
        )
    }

    @MainActor
    func testOpenCodeDesktopReloadReportsUnavailableWhenAppIsMissing() async {
        let integration = OpenCodeIntegration(
            configURL: temporaryDirectory().appendingPathComponent("opencode.json"),
            desktopApplicationURL: temporaryDirectory().appendingPathComponent("MissingOpenCode.app")
        )

        let result = await integration.reloadDesktopAfterDaemonReady()

        XCTAssertEqual(result.action, .unavailable)
        XCTAssertFalse(result.wasRunning)
        XCTAssertFalse(result.didOpen)
        XCTAssertTrue(result.detail.contains("OpenCode.app not found"))
    }

    func testClientHandoffNoticeSurfacesOpenCodeMissing() {
        let result = OpenCodeDesktopResult(
            action: .unavailable,
            wasRunning: false,
            didTerminateExistingInstance: false,
            didOpen: false,
            detail: "OpenCode.app not found at /Applications/OpenCode.app"
        )

        let notice = ClientHandoffNotice.openCode(result: result)

        XCTAssertEqual(notice.target, .openCode)
        XCTAssertEqual(notice.status, "Needs OpenCode Desktop")
        XCTAssertTrue(notice.detail.contains("/Applications/OpenCode.app"))
        XCTAssertTrue(notice.isWarning)
    }

    func testClientHandoffNoticeSurfacesPiTerminalWithoutAgent() {
        let result = PiLaunchResult(
            action: .launched,
            command: "pi --model mtplx/qwen3.5-4b-mtplx-optimized-speed",
            detail: "opened Pi in Terminal for the MTPLX daemon",
            launchedProcessIDs: []
        )

        let notice = ClientHandoffNotice.pi(result: result)

        XCTAssertEqual(notice?.target, .pi)
        XCTAssertEqual(notice?.status, "Pi not detected")
        XCTAssertTrue(notice?.detail.contains("no Pi agent process") == true)
        XCTAssertEqual(notice?.isWarning, true)
    }

    func testClientHandoffNoticeStaysQuietForRunningPiAgent() {
        let result = PiLaunchResult(
            action: .launched,
            command: "pi --model mtplx/qwen3.5-4b-mtplx-optimized-speed",
            detail: "opened Pi in Terminal for the MTPLX daemon",
            launchedProcessIDs: [12345]
        )

        XCTAssertNil(ClientHandoffNotice.pi(result: result))
    }

    func testClientHandoffNoticeSurfacesHermesUnavailable() {
        let result = HermesLaunchResult(
            action: .unavailable,
            command: "hermes chat --profile mtplx",
            detail: "Hermes is not installed or not on PATH."
        )

        let notice = ClientHandoffNotice.hermes(result: result)

        XCTAssertEqual(notice?.target, .hermes)
        XCTAssertEqual(notice?.status, "Hermes unavailable")
        XCTAssertTrue(notice?.detail.contains("not installed") == true)
        XCTAssertEqual(notice?.isWarning, true)
    }

    func testOpenCodeDesktopStateRepairPrunesMissingWorkspace() throws {
        let directory = temporaryDirectory()
        let storeURL = directory.appendingPathComponent("opencode.global.dat")
        let presentProject = directory.appendingPathComponent("live-project")
        try FileManager.default.createDirectory(at: presentProject, withIntermediateDirectories: true)
        let missingProject = "/private/tmp/mtplx-opencode-desktop-qa"

        func projectKey(_ path: String) -> String {
            Data(path.utf8)
                .base64EncodedString()
                .replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "/", with: "_")
                .trimmingCharacters(in: CharacterSet(charactersIn: "="))
        }

        let root: [String: String] = [
            "layout": """
            {"sessionTabs":{"\(projectKey(missingProject))/ses_dead":{"all":[]},"\(projectKey(presentProject.path))/ses_live":{"all":["context"]}},"sessionView":{"\(projectKey(missingProject))/ses_dead":{"scroll":{}},"\(projectKey(presentProject.path))/ses_live":{"scroll":{}}}}
            """,
            "layout.page": """
            {"lastProjectSession":{"\(missingProject)":{"directory":"\(missingProject)","id":"ses_dead"},"\(presentProject.path)":{"directory":"\(presentProject.path)","id":"ses_live"}},"workspaceExpanded":{"\(missingProject)":true,"\(presentProject.path)":true}}
            """,
            "server": """
            {"projects":{"local":[{"worktree":"\(missingProject)","expanded":true},{"worktree":"\(presentProject.path)","expanded":true}]},"lastProject":{"local":"\(presentProject.path)"}}
            """
        ]
        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted])
        try data.write(to: storeURL)

        let result = OpenCodeIntegration.repairDeadWorkspaceState(globalStoreURL: storeURL)

        XCTAssertEqual(result.status, "repaired")
        XCTAssertTrue(result.didChange)
        XCTAssertNotNil(result.backupPath)
        XCTAssertGreaterThanOrEqual(result.removedEntries, 4)
        let repaired = try String(contentsOf: storeURL, encoding: .utf8)
        XCTAssertFalse(repaired.contains("mtplx-opencode-desktop-qa"))
        let repairedRoot = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(repaired.utf8)) as? [String: String]
        )
        let serverText = try XCTUnwrap(repairedRoot["server"])
        let server = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(serverText.utf8)) as? [String: Any]
        )
        let projects = try XCTUnwrap(server["projects"] as? [String: Any])
        let local = try XCTUnwrap(projects["local"] as? [[String: Any]])
        XCTAssertEqual(local.count, 1)
        XCTAssertEqual(local.first?["worktree"] as? String, presentProject.path)
    }

    func testBackendStoreAppliesConfigurationToSettingsStore() async throws {
        let url = temporaryDirectory().appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: url)
        let backend = await MTPLXBackendStore(settingsStore: settingsStore)
        let configuration = MTPLXAppConfiguration(
            model: "model-b",
            profile: "sustained",
            port: 8124,
            generationMode: "ar",
            loadMTP: false,
            contextWindow: 16384
        )

        try await backend.applyConfiguration(configuration, restartIfRunning: true)

        // Loading decodes through sanitize (one-shot migrations consume
        // their flags); the backend's live copy holds the configuration
        // exactly as applied. Both are correct for their surface.
        var loadedExpected = configuration
        loadedExpected.sanitizeLaunchCriticalFields()
        XCTAssertEqual(try settingsStore.load(), loadedExpected)
        let observed = await backend.configuration
        XCTAssertEqual(observed, configuration)
    }

    @MainActor
    func testStartDaemonPromptsToDownloadPartialModelInsteadOfGoingDegraded() async throws {
        let root = temporaryDirectory()
        let partialModel = root.appendingPathComponent("PartialQuality", isDirectory: true)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        try FileManager.default.createDirectory(at: partialModel, withIntermediateDirectories: true)
        let option = MTPLXModelOption(
            id: "partial-quality",
            displayName: "Partial Quality",
            shortName: "Quality",
            detail: "Test partial model",
            hfModelID: "Example/PartialQuality",
            localCandidates: [partialModel.path],
            sizeBytes: 123_456_789
        )
        let settingsStore = MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: partialModel.path,
                customModels: [option]
            ),
            settingsStore: settingsStore,
            modelDownloader: ModelDownloader(modelCacheRoot: cacheRoot)
        )

        await backend.startDaemon(target: .chat)

        let pending = try XCTUnwrap(backend.pendingModelDownload)
        XCTAssertEqual(pending.repoID, "Example/PartialQuality")
        XCTAssertEqual(pending.shortName, "Quality")
        XCTAssertEqual(pending.target, .chat)
        XCTAssertEqual(pending.launchAction, .start)
        XCTAssertEqual(pending.totalBytes, 123_456_789)
        XCTAssertEqual(
            pending.destinationPath,
            cacheRoot.appendingPathComponent("Example--PartialQuality", isDirectory: true).path
        )
        XCTAssertEqual(backend.daemonState, .stopped)
        XCTAssertEqual(backend.startupPhase, .idle)
        XCTAssertNil(backend.modelDownloadFailure)
        XCTAssertNil(backend.modelDownloadProgress)
        XCTAssertEqual(try settingsStore.load().lastLaunchTarget, LaunchTarget.chat.rawValue)
    }

    @MainActor
    func testStartDaemonPromptsToDownloadCurrentHuggingFaceModel() async {
        let root = temporaryDirectory()
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/NewModel"),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            modelDownloader: ModelDownloader(modelCacheRoot: cacheRoot)
        )

        await backend.startDaemon(target: .openCode)

        let pending = backend.pendingModelDownload
        XCTAssertEqual(pending?.repoID, "Example/NewModel")
        XCTAssertEqual(pending?.displayName, "NewModel")
        XCTAssertEqual(pending?.target, .openCode)
        XCTAssertEqual(pending?.launchAction, .start)
        XCTAssertNil(pending?.totalBytes)
        XCTAssertEqual(
            pending?.destinationPath,
            cacheRoot.appendingPathComponent("Example--NewModel", isDirectory: true).path
        )
    }

    @MainActor
    func testModelDownloadRetryClearsFailureAndIncompleteFinishDoesNotStart() async throws {
        let root = temporaryDirectory()
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let incompletePath = cacheRoot.appendingPathComponent("Example--NewModel", isDirectory: true)
        try FileManager.default.createDirectory(at: incompletePath, withIntermediateDirectories: true)
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/NewModel"),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            modelDownloader: ModelDownloader(modelCacheRoot: cacheRoot)
        )

        await backend.startDaemon(target: .chat)
        let request = try XCTUnwrap(backend.pendingModelDownload)

        await backend.handleModelDownloadEvent(
            .failed(exitCode: 1, stderrTail: "network failed"),
            request: request
        )
        let failure = try XCTUnwrap(backend.modelDownloadFailure)
        XCTAssertTrue(
            failure.hasPrefix(
                "The download could not reach Hugging Face. Check the network connection and try again."
            ),
            failure
        )
        // No mirror is configured in this fixture, so the failure copy
        // points at the mirror option.
        XCTAssertTrue(failure.contains("HF download mirror"), failure)

        await backend.handleModelDownloadEvent(
            .progress(
                bytesOnDisk: 142 * 1_024 * 1_024,
                totalBytes: 28 * 1_024 * 1_024 * 1_024,
                smoothedBytesPerSecond: 24_000_000,
                etaSeconds: 120
            ),
            request: request
        )
        XCTAssertNil(backend.modelDownloadFailure)
        XCTAssertEqual(backend.modelDownloadProgress?.statusMessage, "Downloading")

        await backend.handleModelDownloadEvent(
            .complete(bytesOnDisk: 28 * 1_024 * 1_024 * 1_024, path: incompletePath.path),
            request: request
        )

        XCTAssertFalse(backend.modelDownloadProgress?.isComplete ?? true)
        XCTAssertEqual(backend.modelDownloadProgress?.statusMessage, "Incomplete")
        XCTAssertEqual(backend.pendingModelDownload, request)
        XCTAssertEqual(backend.daemonState, .stopped)
        XCTAssertTrue(backend.modelDownloadFailure?.contains("missing required MTPLX files") ?? false)
    }

    @MainActor
    func testModelDownloadCancelMarksProgressPaused() async throws {
        let root = temporaryDirectory()
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/NewModel"),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            modelDownloader: ModelDownloader(modelCacheRoot: cacheRoot)
        )

        await backend.startDaemon(target: .chat)
        let request = try XCTUnwrap(backend.pendingModelDownload)
        await backend.handleModelDownloadEvent(
            .progress(
                bytesOnDisk: 512 * 1_024 * 1_024,
                totalBytes: 28 * 1_024 * 1_024 * 1_024,
                smoothedBytesPerSecond: 30_000_000,
                etaSeconds: 920
            ),
            request: request
        )

        backend.cancelModelDownload()

        let progress = try XCTUnwrap(backend.modelDownloadProgress)
        XCTAssertEqual(progress.statusMessage, "Paused")
        XCTAssertEqual(progress.bytesPerSecond, 0)
        XCTAssertNil(progress.etaSeconds)
        XCTAssertFalse(progress.isComplete)
    }

    @MainActor
    func testDismissModelDownloadPromptClearsPromptState() async {
        let root = temporaryDirectory()
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/NewModel"),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            modelDownloader: ModelDownloader(modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true))
        )

        await backend.startDaemon(target: .chat)
        XCTAssertNotNil(backend.pendingModelDownload)

        backend.dismissModelDownloadPrompt()

        XCTAssertNil(backend.pendingModelDownload)
        XCTAssertNil(backend.modelDownloadProgress)
        XCTAssertNil(backend.modelDownloadFailure)
    }

    @MainActor
    func testCompletedQwenDownloadOffersPostDownloadTune() async throws {
        let root = temporaryDirectory()
        let modelDir = try makeCompleteModel(named: "Example--Qwen3.6-Downloaded")
        let settingsStore = MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/Qwen3.6-Downloaded"),
            settingsStore: settingsStore
        )
        let request = PendingModelDownload(
            repoID: "Example/Qwen3.6-Downloaded",
            displayName: "Qwen 3.6 Downloaded",
            shortName: "Qwen Downloaded",
            target: .chat,
            launchAction: .restart,
            totalBytes: 10,
            destinationPath: modelDir.path
        )

        await backend.handleModelDownloadEvent(
            .complete(bytesOnDisk: 10, path: modelDir.path),
            request: request
        )

        let pendingTune = try XCTUnwrap(backend.pendingModelTune)
        XCTAssertEqual(pendingTune.repoID, "Example/Qwen3.6-Downloaded")
        XCTAssertEqual(pendingTune.installedPath, modelDir.path)
        XCTAssertEqual(pendingTune.modelFamily, "qwen3_6")
        XCTAssertEqual(pendingTune.candidates, [.ar, .d1, .d2, .d3])
        XCTAssertTrue(backend.modelDownloadProgress?.isComplete ?? false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: settingsStore.settingsURL.path))
    }

    @MainActor
    func testCompletedUnsupportedDownloadSkipsTuneOffer() async throws {
        let root = temporaryDirectory()
        let modelDir = try makeCompleteModel(named: "PlainDownloadedModel", archID: nil)
        let settingsStore = MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/PlainDownloadedModel"),
            settingsStore: settingsStore
        )
        let request = PendingModelDownload(
            repoID: "Example/PlainDownloadedModel",
            displayName: "Plain Model",
            shortName: "Plain",
            target: .chat,
            launchAction: .restart,
            totalBytes: 10,
            destinationPath: modelDir.path
        )

        await backend.handleModelDownloadEvent(
            .complete(bytesOnDisk: 10, path: modelDir.path),
            request: request
        )

        XCTAssertNil(backend.pendingModelTune)
        XCTAssertNil(backend.pendingModelDownload)
        XCTAssertEqual(try settingsStore.load().model, modelDir.path)
    }

    @MainActor
    func testSkippingPostDownloadTunePersistsMTPDefaultInsteadOfAR() async throws {
        let root = temporaryDirectory()
        let modelDir = try makeCompleteModel(named: "Example--Qwen3.6-Downloaded")
        let settingsStore = MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                model: "Example/Qwen3.6-Downloaded",
                generationMode: "ar",
                loadMTP: false
            ),
            settingsStore: settingsStore
        )
        let request = PendingModelDownload(
            repoID: "Example/Qwen3.6-Downloaded",
            displayName: "Qwen 3.6 Downloaded",
            shortName: "Qwen Downloaded",
            target: .chat,
            launchAction: .restart,
            totalBytes: 10,
            destinationPath: modelDir.path
        )

        await backend.handleModelDownloadEvent(
            .complete(bytesOnDisk: 10, path: modelDir.path),
            request: request
        )
        backend.skipPendingModelTune()

        let deadline = Date().addingTimeInterval(2)
        while backend.pendingModelTune != nil && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        let saved = try settingsStore.load()
        XCTAssertEqual(saved.model, modelDir.path)
        XCTAssertEqual(saved.generationMode, "mtp")
        XCTAssertTrue(saved.loadMTP)
        XCTAssertEqual(saved.compatibleTunedDepth(), 2)
        XCTAssertNil(backend.pendingModelTune)
        XCTAssertNil(backend.pendingModelDownload)
    }

    func testModelOptionDecodesLegacyRowsWithoutLaunchPolicy() throws {
        let option = try JSONDecoder().decode(
            MTPLXModelOption.self,
            from: Data("""
            {
              "id": "custom-example",
              "displayName": "Example",
              "shortName": "Example",
              "detail": "Legacy row",
              "hfModelID": "Example/Model",
              "localCandidates": ["~/.mtplx/models/Example--Model"]
            }
            """.utf8)
        )

        XCTAssertEqual(option.id, "custom-example")
        XCTAssertEqual(option.aliases, [])
        XCTAssertEqual(option.sizeBytes, 0)
        XCTAssertEqual(option.recommendedFor, [])
    }

    func testOptimizedQualityCatalogEntryIsSelectable() throws {
        let option = try XCTUnwrap(MTPLXModelOption.option(matching: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"))

        XCTAssertEqual(option.id, "optimized-quality")
        XCTAssertEqual(option.shortName, "Qwen 3.6 27B Optimized Quality")
        XCTAssertEqual(option.hfModelID, "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality")
    }

    func testBenchmarkReadinessUsesReachableDaemon() async throws {
        let port = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }

        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: port),
            settingsStore: MTPLXSettingsStore(settingsURL: temporaryDirectory().appendingPathComponent("settings.json"))
        )

        let health = try await backend.ensureDaemonReadyForBenchmark()

        XCTAssertTrue(health.ok)
        let storedHealth = await backend.health
        let currentFanMode = await backend.currentFanMode
        let pendingModelDownload = await backend.pendingModelDownload
        XCTAssertEqual(storedHealth?.model, "mtplx-test-model")
        XCTAssertEqual(currentFanMode, "max")
        XCTAssertNil(pendingModelDownload)
    }

    func testStopDaemonRestoresFansWhenFanModeCacheIsStale() async throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let postLog = root.appendingPathComponent("fan-posts.jsonl")
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-mtplx-fans",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            HEALTH = json.loads(r'''\(Self.healthJSON)''')
            POST_LOG = r'''\(postLog.path)'''

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

                def do_GET(self):
                    if self.path == "/health":
                        self._json(HEALTH)
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                    if self.path == "/v1/mtplx/thermal/fan_mode":
                        length = int(self.headers.get("Content-Length", "0") or "0")
                        raw = self.rfile.read(length) if length else b"{}"
                        request = json.loads(raw.decode("utf-8") or "{}")
                        with open(POST_LOG, "a", encoding="utf-8") as handle:
                            json.dump(request, handle)
                            handle.write("\\n")
                        self._json({
                            "verified": True,
                            "current_mode": request.get("mode") or "auto",
                            "result": {"ok": True},
                            "fan_summary": {"ok": True},
                        })
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let server = Process()
        server.executableURL = script
        try server.run()
        defer { server.terminate() }

        let client = MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let health = try? await client.health(), health.ok {
                break
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }

        let probe = FanFallbackProbe()
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: port, pinFansAtMaxOnStart: true),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            localFanRestorer: {
                await probe.restore()
            }
        )

        await backend.stopDaemon()
        // Fan restore + process reap now run in a detached teardown task so
        // the Stop/Play button isn't frozen for the ~5s reap; await it so we
        // observe the restore the app's quit path also waits on.
        await backend.awaitDaemonTeardown()

        let postLogText = (try? String(contentsOf: postLog, encoding: .utf8)) ?? ""
        XCTAssertTrue(
            postLogText.isEmpty,
            "Stop should not depend on daemon HTTP fan restore, got POST log: \(postLogText)"
        )
        let currentFanMode = await backend.currentFanMode
        XCTAssertEqual(currentFanMode, "default")
        let fallbackCalls = await probe.count()
        XCTAssertGreaterThanOrEqual(fallbackCalls, 1)
    }

    func testStopDaemonUsesLocalThermalForgeFallbackWhenDaemonFanRestoreFails() async throws {
        let root = temporaryDirectory()
        let probe = FanFallbackProbe()
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: try freeTCPPort(), pinFansAtMaxOnStart: true),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            localFanRestorer: {
                await probe.restore()
            }
        )

        await backend.stopDaemon()
        await backend.awaitDaemonTeardown()

        let fallbackCalls = await probe.count()
        XCTAssertGreaterThanOrEqual(fallbackCalls, 1)
        let currentFanMode = await backend.currentFanMode
        XCTAssertEqual(currentFanMode, "default")
    }

    func testStopDaemonRestoresFansEvenWhenStateLooksDefault() async throws {
        let root = temporaryDirectory()
        let probe = FanFallbackProbe()
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                port: try freeTCPPort(),
                fanMode: MTPLXFanMode.default.rawValue
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            localFanRestorer: {
                await probe.restore()
            }
        )

        await backend.stopDaemon()
        await backend.awaitDaemonTeardown()

        let fallbackCalls = await probe.count()
        XCTAssertGreaterThanOrEqual(fallbackCalls, 1)
        let currentFanMode = await backend.currentFanMode
        XCTAssertEqual(currentFanMode, "default")
    }

    @MainActor
    func testCancelDuringStartupLeavesAppStoppedAndRestoresFans() async throws {
        let root = temporaryDirectory()
        let modelDir = root.appendingPathComponent("complete-model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for file in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try "{}".data(using: .utf8)!.write(to: modelDir.appendingPathComponent(file))
        }
        try Data().write(to: modelDir.appendingPathComponent("mtp.safetensors"))
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))
        let fake = try makeExecutable(
            named: "mtplx-slow-start",
            body: """
            #!/bin/sh
            echo "This is the long step; MTPLX is mapping the model into MLX."
            while true; do sleep 1; done
            """
        )
        let probe = FanFallbackProbe()
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: modelDir.path,
                port: try freeTCPPort(),
                pinFansAtMaxOnStart: true,
                customModels: [
                    MTPLXModelOption(
                        id: "complete-model",
                        displayName: "Complete Model",
                        shortName: "Complete",
                        detail: "Fixture model",
                        hfModelID: "Example/CompleteModel",
                        localCandidates: [modelDir.path]
                    )
                ]
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            localFanRestorer: {
                await probe.restore()
            }
        )

        let startTask = Task { @MainActor in
            await backend.startDaemon(target: .chat)
        }
        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline {
            if backend.daemonState == .starting {
                break
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertEqual(backend.daemonState, .starting)

        await backend.stopDaemon()
        await startTask.value
        await backend.awaitDaemonTeardown()

        XCTAssertEqual(backend.daemonState, .stopped)
        XCTAssertEqual(backend.startupPhase, .idle)
        XCTAssertEqual(backend.currentFanMode, "default")
        let fallbackCalls = await probe.count()
        XCTAssertGreaterThanOrEqual(fallbackCalls, 1)
    }

    func testBenchmarkReadinessPromptsForModelDownloadWhenCold() async throws {
        let root = temporaryDirectory()
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let port = try freeTCPPort()
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/NewModel", port: port),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            modelDownloader: ModelDownloader(modelCacheRoot: cacheRoot)
        )

        do {
            _ = try await backend.ensureDaemonReadyForBenchmark()
            XCTFail("expected model download readiness failure")
        } catch BenchmarkDaemonReadinessError.modelDownloadRequired(let model) {
            XCTAssertEqual(model, "NewModel")
            let pendingModelDownload = await backend.pendingModelDownload
            XCTAssertEqual(pendingModelDownload?.target, .benchmark)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testOpenWebUIStartFiresDaemonReadyHandoff() async throws {
        let root = temporaryDirectory()
        let modelDir = root.appendingPathComponent("complete-model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for file in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try "{}".data(using: .utf8)!.write(to: modelDir.appendingPathComponent(file))
        }
        try Data().write(to: modelDir.appendingPathComponent("mtp.safetensors"))
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))
        let port = try freeTCPPort()
        let healthJSON = Self.healthJSON.replacingOccurrences(of: "/models/test", with: modelDir.path)
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: healthJSON
        )
        defer { server.terminate() }

        let fake = try makeExecutable(named: "mtplx")
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: modelDir.path,
                port: port,
                customModels: [
                    MTPLXModelOption(
                        id: "complete-model",
                        displayName: "Complete Model",
                        shortName: "Complete",
                        detail: "Fixture model",
                        hfModelID: "Example/CompleteModel",
                        localCandidates: [modelDir.path]
                    )
                ]
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        )
        let handoff = expectation(description: "Open WebUI daemon-ready handoff")
        let readyTarget = LaunchTargetRecorder()
        await MainActor.run {
            backend.onDaemonReady = { target in
                readyTarget.target = target
                handoff.fulfill()
            }
        }

        await backend.startDaemon(target: .openWebUI)
        await fulfillment(of: [handoff], timeout: 2)

        let daemonState = await backend.daemonState
        let webChatURL = await backend.webChatURL.absoluteString

        let observedReadyTarget = await readyTarget.target

        XCTAssertEqual(observedReadyTarget, .openWebUI)
        XCTAssertEqual(daemonState, .running)
        XCTAssertEqual(webChatURL, "http://127.0.0.1:\(port)")
    }

    func testOpenWebUIHandoffSkipsIncompatiblePostStartSettingsPatch() async throws {
        let root = temporaryDirectory()
        let modelDir = root.appendingPathComponent("Qwen3.6-complete-model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        for file in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try "{}".data(using: .utf8)!.write(to: modelDir.appendingPathComponent(file))
        }
        try Data().write(to: modelDir.appendingPathComponent("mtp.safetensors"))
        try Data().write(to: modelDir.appendingPathComponent("model.safetensors"))
        let port = try freeTCPPort()
        let healthJSON = Self.healthJSON.replacingOccurrences(of: "/models/test", with: modelDir.path)
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: healthJSON
        )
        defer { server.terminate() }

        let fake = try makeExecutable(named: "mtplx")
        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: modelDir.path,
                port: port,
                reasoning: "off"
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
        )
        let handoff = expectation(description: "Open WebUI opens without an incompatible post-start settings patch")
        let readyTarget = LaunchTargetRecorder()
        await MainActor.run {
            backend.onDaemonReady = { target in
                readyTarget.target = target
                handoff.fulfill()
            }
        }

        await backend.startDaemon(target: .openWebUI)
        await fulfillment(of: [handoff], timeout: 2)
        await backend.refreshLogs()

        let daemonState = await backend.daemonState
        let logs = await backend.logs.map(\.message).joined(separator: "\n")
        let observedReadyTarget = await readyTarget.target

        XCTAssertEqual(observedReadyTarget, .openWebUI)
        XCTAssertEqual(daemonState, .running)
        XCTAssertFalse(logs.contains("post-start settings sync failed"))
    }

    @MainActor
    func testOpenWebUIBrowserURLIsChatRootNotDashboard() {
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 18083)
        )

        XCTAssertEqual(backend.webChatURL.absoluteString, "http://127.0.0.1:18083")
        XCTAssertEqual(backend.browserDashboardURL.absoluteString, "http://127.0.0.1:18083/dashboard")
    }

    @MainActor
    func testOpenWebUIBrowserURLBootstrapsBrowserAuthWhenAPIKeyIsConfigured() throws {
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                host: "127.0.0.1",
                port: 18083,
                apiKey: "dashboard-secret"
            )
        )

        let chatComponents = try XCTUnwrap(
            URLComponents(url: backend.webChatURL, resolvingAgainstBaseURL: false)
        )
        let chatQuery = Dictionary(
            uniqueKeysWithValues: (chatComponents.queryItems ?? []).compactMap { item in
                item.value.map { (item.name, $0) }
            }
        )
        XCTAssertEqual(chatComponents.path, "/mtplx/browser-auth")
        XCTAssertEqual(chatQuery["mtplx_api_key"], "dashboard-secret")
        XCTAssertEqual(chatQuery["next"], "/")

        let dashboardComponents = try XCTUnwrap(
            URLComponents(url: backend.browserDashboardURL, resolvingAgainstBaseURL: false)
        )
        let dashboardQuery = Dictionary(
            uniqueKeysWithValues: (dashboardComponents.queryItems ?? []).compactMap { item in
                item.value.map { (item.name, $0) }
            }
        )
        XCTAssertEqual(dashboardComponents.path, "/mtplx/browser-auth")
        XCTAssertEqual(dashboardQuery["mtplx_api_key"], "dashboard-secret")
        XCTAssertEqual(dashboardQuery["next"], "/dashboard/")
    }

    @MainActor
    func testOpenWebUIReadyHandoffOpensWebChatRoot() {
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 18083)
        )
        var openedURL: URL?
        let openWebChat = {
            openedURL = backend.webChatURL
        }
        backend.onDaemonReady = { target in
            if target == .openWebUI {
                openWebChat()
            }
        }

        backend.onDaemonReady?(.openWebUI)

        XCTAssertEqual(openedURL?.absoluteString, "http://127.0.0.1:18083")
    }

    @MainActor
    func testBackendStoreMigratesOpenCodeShapedChatSettings() throws {
        let url = temporaryDirectory().appendingPathComponent("settings.json")
        let settingsStore = MTPLXSettingsStore(settingsURL: url)
        let stale = MTPLXAppConfiguration(
            model: "/models/qwen",
            port: 18083,
            schedulerMode: "ar_batch",
            batchingPreset: "agent",
            ssdSessionCache: "on",
            lastLaunchTarget: LaunchTarget.chat.rawValue
        )
        try settingsStore.save(stale)
        let backend = MTPLXBackendStore(settingsStore: settingsStore)

        backend.loadPersistedSettings()

        XCTAssertEqual(backend.configuration.lastLaunchTarget, LaunchTarget.openCode.rawValue)
        XCTAssertEqual(try settingsStore.load().lastLaunchTarget, LaunchTarget.openCode.rawValue)
    }

    func testSSEParserAndDecoderHandleDashboardEvents() throws {
        let parser = SSEParser()
        let messages = parser.parse(
            """
            \(Self.sseBlock(event: "progress", json: #"{"kind":"progress","request_id":"r1","progress":{"decode_tok_s":42.0}}"#))

            \(Self.sseBlock(event: "snapshot", json: Self.snapshotJSON))
            """
        )
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0].event, "progress")
        XCTAssertEqual(messages[1].event, "snapshot")

        let client = MetricsStreamClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        )
        if case .progress(let payload) = try client.decode(message: messages[0]) {
            XCTAssertEqual(payload.values["request_id"]?.stringValue, "r1")
        } else {
            XCTFail("expected progress event")
        }
        if case .snapshot(let snapshot) = try client.decode(message: messages[1]) {
            XCTAssertEqual(snapshot.modelId, "mtplx-test-model")
            XCTAssertEqual(snapshot.latest?.decodeTokS, 55)
            XCTAssertEqual(snapshot.machine.chipName, "Apple M5 Max")
        } else {
            XCTFail("expected snapshot event")
        }
    }

    func testSSEDecoderHandlesFlatPrefillEvents() throws {
        let client = MetricsStreamClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        )
        let event = try client.decode(
            message: SSEMessage(
                event: "prefill",
                data: """
                {
                  "kind": "prefill",
                  "request_id": "r-prefill",
                  "session_id": "s1",
                  "phase": "chunk",
                  "tokens_done": 4096,
                  "tokens_total": 44988,
                  "live_prefill_tok_s": 212.5,
                  "elapsed_s": 19.2
                }
                """
            )
        )

        guard case .prefill(let payload) = event else {
            return XCTFail("expected prefill event")
        }
        XCTAssertEqual(payload.values["request_id"]?.stringValue, "r-prefill")
        XCTAssertEqual(payload.values["phase"]?.stringValue, "chunk")
        XCTAssertEqual(payload.values["live_prefill_tok_s"]?.doubleValue, 212.5)
    }

    func testAPIClientBuildsExpectedMetricsStreamURL() {
        let client = MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:8000")!)
        XCTAssertEqual(
            client.metricsStreamURL(snapshotIntervalMs: 500).absoluteString,
            "http://127.0.0.1:8000/v1/mtplx/metrics/stream?snapshot_interval_ms=500"
        )
    }

    func testMetricsStreamRequestCarriesAppAPIKey() throws {
        let client = MetricsStreamClient(
            apiClient: MTPLXAPIClient(
                baseURL: URL(string: "http://127.0.0.1:8000")!,
                apiKey: "dashboard-secret"
            )
        )

        let request = client.makeRequest(snapshotIntervalMs: 500)

        XCTAssertEqual(request.httpMethod, "GET")
        XCTAssertEqual(
            request.url?.absoluteString,
            "http://127.0.0.1:8000/v1/mtplx/metrics/stream?snapshot_interval_ms=500"
        )
        XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "text/event-stream")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Cache-Control"), "no-cache")
        XCTAssertEqual(
            request.value(forHTTPHeaderField: "Authorization"),
            "Bearer dashboard-secret"
        )
    }

    @MainActor
    func testChatViewModelUsesRequestLocalDecodeReading() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        self.rfile.read(length)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunks = [
                        {
                            "id": "chatcmpl-test",
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                        },
                        {
                            "id": "chatcmpl-test",
                            "choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}],
                            "mtplx_progress": {
                                "completion_tokens": 8,
                                "decode_tok_s": 36.5,
                                "display_decode_tok_s": 49.0,
                            },
                        },
                        {
                            "id": "chatcmpl-test",
                            "choices": [{"index": 0, "delta": {"content": "done"}}],
                            "mtplx_progress": {
                                "completion_tokens": 12,
                                "decode_tok_s": 36.5,
                                "display_decode_tok_s": 49.0,
                            },
                        },
                        {
                            "id": "chatcmpl-test",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 4,
                                "completion_tokens": 12,
                                "total_tokens": 16,
                            },
                            "mtplx_stats": {
                                "raw_decode_tok_s": 36.5,
                                "display_decode_tok_s": 49.0,
                            },
                        },
                    ]
                    for chunk in chunks:
                        self.wfile.write(sse(chunk))
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("hello")

        let deadline = Date().addingTimeInterval(5)
        var heldDecode: Double?
        while Date() < deadline {
            if case .held(let value, _) = viewModel.chatDecodeReading {
                heldDecode = value
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(heldDecode ?? -1, 36.5, accuracy: 0.01)
        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "done")
    }

    @MainActor
    func testChatViewModelDoesNotCapNormalChatOutputTokens() async throws {
        let port = try freeTCPPort()
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let requestURL = root.appendingPathComponent("request.json")
        let script = try makeExecutable(
            named: "fake-chat-request-capture",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            REQUEST = r'''\(requestURL.path)'''

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    body = self.rfile.read(length) if length else b"{}"
                    with open(REQUEST, "wb") as f:
                        f.write(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(sse({
                        "id": "chatcmpl-no-token-cap",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-no-token-cap",
                        "choices": [{"index": 0, "delta": {"content": "unrestricted"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-no-token-cap",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("write a detailed answer")

        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if !viewModel.isStreaming,
               FileManager.default.fileExists(atPath: requestURL.path) {
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        let payload = try chatRequestPayload(at: requestURL)
        XCTAssertNil(payload["max_tokens"])
        XCTAssertEqual(payload["stream"] as? Bool, true)
    }

    @MainActor
    func testChatViewModelPublishesSettledAssistantOnlyAfterStreamStops() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-delayed-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        self.rfile.read(length)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(sse({
                        "id": "chatcmpl-transition",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-transition",
                        "choices": [{"index": 0, "delta": {"reasoning_content": "thinking\\n"}}],
                    }))
                    self.wfile.flush()
                    time.sleep(0.20)
                    self.wfile.write(sse({
                        "id": "chatcmpl-transition",
                        "choices": [{"index": 0, "delta": {"content": "```python\\nprint('ok')\\n```"}}],
                    }))
                    self.wfile.flush()
                    time.sleep(0.35)
                    self.wfile.write(sse({
                        "id": "chatcmpl-transition",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 6, "total_tokens": 8},
                        "mtplx_stats": {"raw_decode_tok_s": 42.0},
                    }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("generate code")

        let reasoningDeadline = Date().addingTimeInterval(5)
        var sawLiveReasoning = false
        while Date() < reasoningDeadline {
            if viewModel.isStreaming, viewModel.hasStreamingReasoning {
                sawLiveReasoning = true
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertTrue(sawLiveReasoning)
        XCTAssertTrue(viewModel.streamingContent.isEmpty)
        XCTAssertFalse(viewModel.hasStreamingContent)
        XCTAssertEqual(viewModel.visibleMessages.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.last?.role, .user)

        let streamingDeadline = Date().addingTimeInterval(5)
        var sawLiveCode = false
        while Date() < streamingDeadline {
            if viewModel.isStreaming, viewModel.streamingContent.contains("print('ok')") {
                sawLiveCode = true
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertTrue(sawLiveCode)
        XCTAssertTrue(viewModel.isStreaming)
        XCTAssertTrue(viewModel.hasStreamingReasoning)
        XCTAssertTrue(viewModel.hasStreamingContent)
        XCTAssertTrue(viewModel.shouldRenderStreamingAssistant)
        XCTAssertFalse(viewModel.streamingReasoningDocument.isEmpty)
        XCTAssertFalse(viewModel.streamingContentDocument.isEmpty)
        XCTAssertGreaterThanOrEqual(viewModel.streamingContentDocument.blocks.count, 2)
        XCTAssertFalse(viewModel.streamingContentDocument.blocks.contains { block in
            if case .codeFence = block.kind {
                return true
            }
            return false
        })
        XCTAssertEqual(viewModel.visibleMessages.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.last?.role, .user)

        let finishedDeadline = Date().addingTimeInterval(5)
        while Date() < finishedDeadline {
            if !viewModel.isStreaming {
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertFalse(viewModel.hasStreamingReasoning)
        XCTAssertFalse(viewModel.hasStreamingContent)
        XCTAssertFalse(viewModel.shouldRenderStreamingAssistant)
        XCTAssertNil(viewModel.handoffAssistantMessageID)
        XCTAssertEqual(viewModel.visibleMessages.count, 2)
        XCTAssertEqual(viewModel.visibleMessages.last?.role, .assistant)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "```python\nprint('ok')\n```")
        XCTAssertEqual(viewModel.visibleMessages.last?.reasoningContent, "thinking\n")
        XCTAssertEqual(viewModel.visibleMessages.last?.finishReason, "stop")
    }

    @MainActor
    func testChatViewModelMovesLeakedThinkingTagsOutOfVisibleContent() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-thinking-tag-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        self.rfile.read(length)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for fragment in ["<thi", "nk>hidden plan", "</think>Visible answer."]:
                        self.wfile.write(sse({
                            "id": "chatcmpl-thinking-tags",
                            "choices": [{"index": 0, "delta": {"content": fragment}}],
                        }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-thinking-tags",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("answer visibly")

        let finishedDeadline = Date().addingTimeInterval(5)
        while Date() < finishedDeadline {
            if !viewModel.isStreaming,
               viewModel.visibleMessages.last?.role == .assistant {
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        let assistant = try XCTUnwrap(viewModel.visibleMessages.last)
        XCTAssertEqual(assistant.visibleContent, "Visible answer.")
        XCTAssertEqual(assistant.reasoningContent, "hidden plan")
        XCTAssertFalse(assistant.visibleContent.contains("<think>"))
        XCTAssertFalse(assistant.visibleContent.contains("</think>"))
    }

    @MainActor
    func testChatViewModelNotifiesBackendWhenDaemonIsUnreachable() async throws {
        let port = try freeTCPPort()
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        var notified = false
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" },
            onDaemonUnreachable: {
                notified = true
            }
        )

        viewModel.send("hello")

        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if viewModel.lastError == .daemonStopped {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(viewModel.lastError, .daemonStopped)
        XCTAssertTrue(notified)
        XCTAssertFalse(viewModel.isStreaming)
    }

    @MainActor
    func testChatViewModelRetriesHTTPErrorWithoutDuplicatingUserTurn() async throws {
        let port = try freeTCPPort()
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstRequestURL = root.appendingPathComponent("request-1.json")
        let secondRequestURL = root.appendingPathComponent("request-2.json")
        let script = try makeExecutable(
            named: "fake-retry-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            FIRST = r'''\(firstRequestURL.path)'''
            SECOND = r'''\(secondRequestURL.path)'''

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                count = 0

                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    body = self.rfile.read(length) if length else b"{}"
                    Handler.count += 1
                    with open(FIRST if Handler.count == 1 else SECOND, "wb") as f:
                        f.write(body)
                    if Handler.count == 1:
                        payload = b"temporary retry failure"
                        self.send_response(500)
                        self.send_header("Content-Type", "text/plain")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(sse({
                        "id": "chatcmpl-retry",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-retry",
                        "choices": [{"index": 0, "delta": {"content": "retry ok"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-retry",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("retry me")

        let errorDeadline = Date().addingTimeInterval(5)
        while Date() < errorDeadline {
            if viewModel.lastError == .http(500, "temporary retry failure") {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(viewModel.lastError, .http(500, "temporary retry failure"))
        XCTAssertTrue(viewModel.canRetryLastUserMessage)
        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.filter { $0.role == .user }.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "retry me")

        viewModel.retryLastUserMessage()

        let retryDeadline = Date().addingTimeInterval(5)
        while Date() < retryDeadline {
            if !viewModel.isStreaming,
               viewModel.visibleMessages.last?.visibleContent == "retry ok" {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertNil(viewModel.lastError)
        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.filter { $0.role == .user }.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.filter { $0.role == .assistant }.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "retry ok")

        let firstRequest = try chatRequestPayload(at: firstRequestURL)
        let secondRequest = try chatRequestPayload(at: secondRequestURL)
        XCTAssertEqual(chatRoles(in: firstRequest), ["user"])
        XCTAssertEqual(chatRoles(in: secondRequest), ["user"])
        XCTAssertEqual(chatMessages(in: firstRequest).first?["content"] as? String, "retry me")
        XCTAssertEqual(chatMessages(in: secondRequest).first?["content"] as? String, "retry me")
    }

    @MainActor
    func testChatViewModelForcesAnswerAfterOneWebToolRound() async throws {
        let port = try freeTCPPort()
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let firstRequestURL = root.appendingPathComponent("web-request-1.json")
        let secondRequestURL = root.appendingPathComponent("web-request-2.json")
        let thirdRequestURL = root.appendingPathComponent("web-request-3.json")
        let script = try makeExecutable(
            named: "fake-web-tool-loop-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            FIRST = r'''\(firstRequestURL.path)'''
            SECOND = r'''\(secondRequestURL.path)'''
            THIRD = r'''\(thirdRequestURL.path)'''

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                count = 0

                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    body = self.rfile.read(length) if length else b"{}"
                    Handler.count += 1
                    target = FIRST if Handler.count == 1 else SECOND if Handler.count == 2 else THIRD
                    with open(target, "wb") as f:
                        f.write(body)

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()

                    if Handler.count == 1:
                        self.wfile.write(sse({
                            "id": "chatcmpl-web",
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-web",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": "call_fetch",
                                        "type": "function",
                                        "function": {"name": "fetch_url", "arguments": ""},
                                    }],
                                },
                            }],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-web",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "function": {"arguments": "{\\"url\\":\\"https://example.com/release\\"}"},
                                    }],
                                },
                            }],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-web",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }))
                    else:
                        self.wfile.write(sse({
                            "id": "chatcmpl-web-final",
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-web-final",
                            "choices": [{"index": 0, "delta": {"content": "web answer"}}],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-web-final",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let toolFactory = MTPLXChatToolFactory(
            urlFetcher: URLFetcher(
                transport: FixtureWebTransport(
                    body: "<html><title>Release</title><body>MTPLX_WEB_SINGLE_ROUND_0606</body></html>"
                ),
                cache: URLFetchCache()
            )
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            toolFactory: toolFactory,
            modelName: { "mtplx-test-model" }
        )
        _ = viewModel.createNewConversation()
        viewModel.webSearchEnabled = true

        viewModel.send("Read the latest release page.")

        let finishedDeadline = Date().addingTimeInterval(5)
        while Date() < finishedDeadline {
            if !viewModel.isStreaming,
               viewModel.visibleMessages.last?.visibleContent == "web answer" {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.lastError, nil)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "web answer")

        let firstRequest = try chatRequestPayload(at: firstRequestURL)
        let secondRequest = try chatRequestPayload(at: secondRequestURL)
        XCTAssertEqual(firstRequest["tool_choice"] as? String, "auto")
        XCTAssertEqual(secondRequest["tool_choice"] as? String, "none")
        XCTAssertEqual(chatRoles(in: secondRequest), ["user", "assistant", "tool"])
        let toolMessage = try XCTUnwrap(chatMessages(in: secondRequest).first { $0["role"] as? String == "tool" })
        XCTAssertTrue((toolMessage["content"] as? String ?? "").contains("MTPLX_WEB_SINGLE_ROUND_0606"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: thirdRequestURL.path))
    }

    @MainActor
    func testUnreadableAttachmentDoesNotSendEmptyPrompt() throws {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )
        let unreadable = ChatAttachment(
            filename: "empty.pdf",
            mimeType: "application/pdf",
            sizeBytes: 12,
            extractedText: ""
        )
        viewModel.pendingAttachments = [unreadable]

        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
        viewModel.send("")

        XCTAssertNil(viewModel.current)
        XCTAssertTrue(viewModel.visibleMessages.isEmpty)
        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["empty.pdf"])
    }

    @MainActor
    func testRemovingUnreadableAttachmentRestoresComposerToPlainTextOnly() throws {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )
        let unreadable = ChatAttachment(
            filename: "empty.pdf",
            mimeType: "application/pdf",
            sizeBytes: 12,
            extractedText: ""
        )
        viewModel.pendingAttachments = [unreadable]

        XCTAssertFalse(viewModel.hasSendablePendingAttachments)
        viewModel.removeAttachment(unreadable)

        XCTAssertTrue(viewModel.pendingAttachments.isEmpty)
        XCTAssertFalse(viewModel.hasSendablePendingAttachments)

        viewModel.send("")

        XCTAssertNil(viewModel.current)
        XCTAssertTrue(viewModel.visibleMessages.isEmpty)

        viewModel.send("plain text still sends")

        XCTAssertEqual(viewModel.visibleMessages.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.first?.visibleContent, "plain text still sends")
        XCTAssertTrue(viewModel.visibleMessages.first?.attachments.isEmpty ?? false)
        XCTAssertTrue(viewModel.pendingAttachments.isEmpty)
    }

    @MainActor
    func testUnreadableAttachmentIsNotPersistedWithReadableAttachment() async throws {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )
        let unreadable = ChatAttachment(
            filename: "empty.pdf",
            mimeType: "application/pdf",
            sizeBytes: 12,
            extractedText: ""
        )
        let readable = ChatAttachment(
            filename: "notes.md",
            mimeType: "text/markdown",
            sizeBytes: 32,
            extractedText: "MTPLX_ATTACHMENT_MARKER"
        )
        viewModel.pendingAttachments = [unreadable, readable]

        XCTAssertTrue(viewModel.hasSendablePendingAttachments)
        viewModel.send("")

        XCTAssertEqual(viewModel.visibleMessages.count, 1)
        XCTAssertEqual(viewModel.visibleMessages.first?.attachments.map(\.filename), ["notes.md"])
        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["empty.pdf"])
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if viewModel.lastError == .daemonStopped {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }
        XCTAssertEqual(viewModel.lastError, .daemonStopped)
        XCTAssertFalse(viewModel.isStreaming)
    }

    func testFileExtractorReadsMarkdownDocxAndPDF() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-attachment-fixtures-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let mdURL = root.appendingPathComponent("notes.md")
        try "MTPLX_MARKDOWN_ATTACHMENT_0606".write(to: mdURL, atomically: true, encoding: .utf8)

        let docxURL = root.appendingPathComponent("brief.docx")
        try makeMinimalDocx(
            at: docxURL,
            text: "MTPLX_DOCX_ATTACHMENT_0606"
        )

        let pdfURL = root.appendingPathComponent("paper.pdf")
        try makeTextPDF(
            at: pdfURL,
            text: "MTPLX_PDF_ATTACHMENT_0606"
        )

        XCTAssertEqual(try FileExtractor.extract(from: mdURL).combinedText, "MTPLX_MARKDOWN_ATTACHMENT_0606")
        XCTAssertTrue(try FileExtractor.extract(from: docxURL).combinedText.contains("MTPLX_DOCX_ATTACHMENT_0606"))
        XCTAssertTrue(try FileExtractor.extract(from: pdfURL).combinedText.contains("MTPLX_PDF_ATTACHMENT_0606"))
    }

    @MainActor
    func testChatViewModelSendsReadableAttachmentFormatsOnly() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-attachment-send-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let mdURL = root.appendingPathComponent("notes.md")
        try "MTPLX_MARKDOWN_SEND_0606".write(to: mdURL, atomically: true, encoding: .utf8)
        let emptyURL = root.appendingPathComponent("empty.md")
        try "\n".write(to: emptyURL, atomically: true, encoding: .utf8)
        let docxURL = root.appendingPathComponent("brief.docx")
        try makeMinimalDocx(at: docxURL, text: "MTPLX_DOCX_SEND_0606")
        let pdfURL = root.appendingPathComponent("paper.pdf")
        try makeTextPDF(at: pdfURL, text: "MTPLX_PDF_SEND_0606")

        let port = try freeTCPPort()
        let captureURL = root.appendingPathComponent("request.json")
        let script = try makeExecutable(
            named: "fake-attachment-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            CAPTURE = r'''\(captureURL.path)'''

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    body = self.rfile.read(length) if length else b"{}"
                    with open(CAPTURE, "wb") as f:
                        f.write(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(sse({
                        "id": "chatcmpl-attachments",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-attachments",
                        "choices": [{"index": 0, "delta": {"content": "attachments ok"}}],
                    }))
                    self.wfile.write(sse({
                        "id": "chatcmpl-attachments",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }))
                    self.wfile.write(b"data: [DONE]\\n\\n")
                    self.wfile.flush()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        await viewModel.attach([mdURL, emptyURL, docxURL, pdfURL])
        XCTAssertEqual(viewModel.pendingAttachments.count, 4)
        XCTAssertTrue(viewModel.hasSendablePendingAttachments)

        viewModel.send("Use the attached files.")

        let finishedDeadline = Date().addingTimeInterval(5)
        while Date() < finishedDeadline {
            if !viewModel.isStreaming, FileManager.default.fileExists(atPath: captureURL.path) {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "attachments ok")
        XCTAssertEqual(viewModel.pendingAttachments.map(\.filename), ["empty.md"])

        let request = try String(contentsOf: captureURL, encoding: .utf8)
        XCTAssertTrue(request.contains("MTPLX_MARKDOWN_SEND_0606"))
        XCTAssertTrue(request.contains("MTPLX_DOCX_SEND_0606"))
        XCTAssertTrue(request.contains("MTPLX_PDF_SEND_0606"))
        XCTAssertTrue(request.contains("[Attached file: notes.md]"))
        XCTAssertTrue(request.contains("[Attached file: brief.docx]"))
        XCTAssertTrue(request.contains("[Attached file: paper.pdf]"))
        XCTAssertFalse(request.contains("[Attached file: empty.md]"))
    }

    @MainActor
    func testChatRequestMessagesCompactOldCodeHistoryWhenPromptWouldBeHuge() throws {
        let conversation = ChatConversation(title: "Long code thread")
        let oldCode = String(repeating: "OLD_HISTORY_LINE = 'not needed anymore'\n", count: 2_400)
        var persisted: [ChatMessage] = [
            ChatMessage(
                role: .user,
                visibleContent: "generate a game",
                createdAt: Date(timeIntervalSince1970: 100),
                conversation: conversation
            ),
            ChatMessage(
                role: .assistant,
                visibleContent: "```python\n\(oldCode)```",
                createdAt: Date(timeIntervalSince1970: 101),
                conversation: conversation
            ),
        ]
        for index in 0..<6 {
            persisted.append(
                ChatMessage(
                    role: index.isMultiple(of: 2) ? .user : .assistant,
                    visibleContent: "recent turn \(index)",
                    createdAt: Date(timeIntervalSince1970: Double(102 + index)),
                    conversation: conversation
                )
            )
        }
        persisted.append(
            ChatMessage(
                role: .assistant,
                visibleContent: "```python\nprint('LATEST_KEEP')\n```",
                createdAt: Date(timeIntervalSince1970: 108),
                conversation: conversation
            )
        )
        persisted.append(
            ChatMessage(
                role: .user,
                visibleContent: "fix the latest error",
                createdAt: Date(timeIntervalSince1970: 109),
                conversation: conversation
            )
        )

        let requestMessages = ChatViewModel.buildRequestMessages(
            from: persisted,
            overrideLastUserContent: nil
        )
        let joined = requestMessages.compactMap(\.content).joined(separator: "\n")

        XCTAssertFalse(joined.contains("OLD_HISTORY_LINE"))
        XCTAssertTrue(joined.contains("omitted historical python code block"))
        XCTAssertTrue(joined.contains("LATEST_KEEP"))
        XCTAssertEqual(requestMessages.last?.content, "fix the latest error")
    }

    @MainActor
    func testChatRetryRequestDropsTrailingFailedAssistantTurn() throws {
        let conversation = ChatConversation(title: "Retry thread")
        let firstUser = ChatMessage(
            role: .user,
            visibleContent: "first turn",
            createdAt: Date(timeIntervalSince1970: 300),
            conversation: conversation
        )
        let settledAssistant = ChatMessage(
            role: .assistant,
            visibleContent: "first answer",
            createdAt: Date(timeIntervalSince1970: 301),
            conversation: conversation
        )
        let retryUser = ChatMessage(
            role: .user,
            visibleContent: "please retry",
            createdAt: Date(timeIntervalSince1970: 302),
            conversation: conversation
        )
        let failedAssistant = ChatMessage(
            role: .assistant,
            visibleContent: "partial failed answer",
            finishReason: "error",
            createdAt: Date(timeIntervalSince1970: 303),
            conversation: conversation
        )
        let requestMessages = ChatViewModel.buildRetryRequestMessages(
            from: [firstUser, settledAssistant, retryUser, failedAssistant],
            retrying: retryUser,
            fullUserContent: "please retry"
        )
        let roles = requestMessages.map(\.role)
        let joined = requestMessages.compactMap(\.content).joined(separator: "\n")

        XCTAssertEqual(roles, ["user", "assistant", "user"])
        XCTAssertTrue(joined.contains("first answer"))
        XCTAssertTrue(joined.contains("please retry"))
        XCTAssertFalse(joined.contains("partial failed answer"))
    }

    @MainActor
    func testChatRequestMessagesCompactLargeToolResultsBeforePrefill() throws {
        let conversation = ChatConversation(title: "Web thread")
        let pageContent = """
        OpenAI News | OpenAI
        \(String(repeating: ".css-rule { color: var(--token); }\n", count: 700))
        LATEST_WEB_SIGNAL useful page text after stylesheet noise.
        """
        let payload: [String: Any] = [
            "query": "openai.com/news latest announcement",
            "results": [
                [
                    "title": "OpenAI News | OpenAI",
                    "url": "https://openai.com/news/",
                    "host": "openai.com",
                    "snippet": "LATEST_WEB_SIGNAL headline from search snippet.",
                    "page_content": pageContent,
                ],
            ],
        ]
        let toolData = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let toolContent = try XCTUnwrap(String(data: toolData, encoding: .utf8))
        let toolCallsJSON = try XCTUnwrap(String(
            data: JSONEncoder().encode([
                ToolCallRecord(
                    id: "call_web",
                    name: "web_search",
                    arguments: #"{"query":"openai.com/news latest announcement"}"#
                ),
            ]),
            encoding: .utf8
        ))
        let persisted = [
            ChatMessage(
                role: .user,
                visibleContent: "find current news",
                createdAt: Date(timeIntervalSince1970: 200),
                conversation: conversation
            ),
            ChatMessage(
                role: .assistant,
                visibleContent: "",
                toolCallsJSON: toolCallsJSON,
                createdAt: Date(timeIntervalSince1970: 201),
                conversation: conversation
            ),
            ChatMessage(
                role: .tool,
                visibleContent: toolContent,
                toolCallId: "call_web",
                createdAt: Date(timeIntervalSince1970: 202),
                conversation: conversation
            ),
            ChatMessage(
                role: .user,
                visibleContent: "answer from that web result",
                createdAt: Date(timeIntervalSince1970: 203),
                conversation: conversation
            ),
        ]

        let requestMessages = ChatViewModel.buildRequestMessages(
            from: persisted,
            overrideLastUserContent: nil
        )
        let toolMessage = try XCTUnwrap(requestMessages.first { $0.role == "tool" })
        let compactedToolContent = try XCTUnwrap(toolMessage.content)

        XCTAssertEqual(toolMessage.toolCallId, "call_web")
        XCTAssertLessThan(compactedToolContent.count, 3_000)
        XCTAssertTrue(compactedToolContent.contains("compact_notice"))
        XCTAssertTrue(compactedToolContent.contains("OpenAI News"))
        XCTAssertTrue(compactedToolContent.contains("LATEST_WEB_SIGNAL"))
        XCTAssertFalse(compactedToolContent.contains(".css-rule"))
        XCTAssertEqual(requestMessages.first { $0.role == "assistant" }?.toolCalls?.first?.id, "call_web")
        XCTAssertEqual(requestMessages.last?.content, "answer from that web result")
    }

    @MainActor
    func testToolResultCompactionKeepsUsefulWebSourceExcerpt() throws {
        let pageContent = String(repeating: "Detailed source paragraph with concrete evidence. ", count: 600)
        let payload: [String: Any] = [
            "query": "why is the answer shallow",
            "results": [
                [
                    "title": "Detailed Source",
                    "url": "https://example.com/source",
                    "host": "example.com",
                    "snippet": "Useful source snippet.",
                    "page_content": pageContent,
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let content = try XCTUnwrap(String(data: data, encoding: .utf8))

        let compacted = ChatViewModel.compactToolResultContent(content)
        let compactedData = try XCTUnwrap(compacted.data(using: .utf8))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: compactedData) as? [String: Any]
        )
        let results = try XCTUnwrap(object["results"] as? [[String: Any]])
        let excerpt = try XCTUnwrap(results.first?["page_excerpt"] as? String)

        XCTAssertGreaterThanOrEqual(excerpt.count, 2_000)
        XCTAssertTrue(excerpt.contains("Detailed source paragraph"))
    }

    @MainActor
    func testChatRequestMessagesKeepModerateToolResultFullForWebAnswering() throws {
        let conversation = ChatConversation(title: "RSS thread")
        let rssContent = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss><channel><title>OpenAI News</title>
        \(String(repeating: "<item><title>Older item</title></item>", count: 90))
        <item><title>KEEP_RSS_HEADLINE</title><link>https://openai.com/news/example</link></item>
        </channel></rss>
        """
        XCTAssertLessThan(rssContent.count, 6_000)
        let payload: [String: Any] = [
            "url": "https://openai.com/news/rss.xml",
            "content": rssContent,
        ]
        let toolData = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let toolContent = try XCTUnwrap(String(data: toolData, encoding: .utf8))
        let persisted = [
            ChatMessage(
                role: .tool,
                visibleContent: toolContent,
                toolCallId: "call_rss",
                createdAt: Date(timeIntervalSince1970: 210),
                conversation: conversation
            ),
        ]

        let requestMessages = ChatViewModel.buildRequestMessages(
            from: persisted,
            overrideLastUserContent: nil
        )
        let requestContent = try XCTUnwrap(requestMessages.first?.content)

        XCTAssertEqual(requestMessages.first?.toolCallId, "call_rss")
        XCTAssertEqual(requestContent, toolContent)
        XCTAssertTrue(requestContent.contains("KEEP_RSS_HEADLINE"))
        XCTAssertFalse(requestContent.contains("compact_notice"))
    }

    @MainActor
    func testChatViewModelMarksCancelledPartialAssistantTurn() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-cancellable-chat-stream",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            def sse(payload):
                return ("data: " + json.dumps(payload) + "\\n\\n").encode("utf-8")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length:
                        self.rfile.read(length)
                    if self.path.startswith("/v1/mtplx/cancel/"):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b"{\\"ok\\": true}")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    try:
                        self.wfile.write(sse({
                            "id": "chatcmpl-cancel-test",
                            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                        }))
                        self.wfile.write(sse({
                            "id": "chatcmpl-cancel-test",
                            "choices": [{"index": 0, "delta": {"content": "partial answer\\nline two"}}],
                        }))
                        self.wfile.flush()
                        time.sleep(3)
                    except BrokenPipeError:
                        return

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let startupDeadline = Date().addingTimeInterval(5)
        while Date() < startupDeadline {
            if (try? await URLSession.shared.data(from: baseURL)) != nil {
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        viewModel.send("start a long answer")

        let liveDeadline = Date().addingTimeInterval(5)
        while Date() < liveDeadline {
            if viewModel.hasStreamingContent {
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(viewModel.hasStreamingContent)

        await viewModel.cancel()

        let cancelDeadline = Date().addingTimeInterval(5)
        while Date() < cancelDeadline {
            if !viewModel.isStreaming {
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }

        XCTAssertFalse(viewModel.isStreaming)
        XCTAssertEqual(viewModel.visibleMessages.count, 2)
        XCTAssertEqual(viewModel.visibleMessages.last?.role, .assistant)
        XCTAssertEqual(viewModel.visibleMessages.last?.visibleContent, "partial answer\nline two")
        XCTAssertEqual(viewModel.visibleMessages.last?.finishReason, "cancelled")
    }

    func testChatStoreSupportsExplicitQAStorePathOverride() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-chat-store-\(UUID().uuidString)", isDirectory: true)
        let store = root.appendingPathComponent("chats.store")
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertEqual(
            ChatStore.explicitStorePath(
                environment: [ChatStore.storePathEnvironmentVariable: store.path],
                arguments: ["MTPLXApp"]
            ),
            store.path
        )
        XCTAssertEqual(
            ChatStore.explicitStorePath(
                environment: [:],
                arguments: ["MTPLXApp", "--mtplx-chat-store-path=\(store.path)"]
            ),
            store.path
        )
        XCTAssertEqual(
            ChatStore.explicitStorePath(
                environment: [:],
                arguments: ["MTPLXApp", "--mtplx-chat-store", store.path]
            ),
            store.path
        )

        let resolved = try ChatStore.explicitStoreURL(store.path)
        XCTAssertEqual(resolved.path, store.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.path))
    }

    @MainActor
    func testChatViewModelLoadsMessagesWithDirectFetchOnSelection() throws {
        let container = try ChatStore.makeInMemoryContainer()
        let context = container.mainContext
        let conversation = ChatConversation(
            id: UUID(),
            title: "Recovered transcript",
            createdAt: Date(timeIntervalSince1970: 100),
            updatedAt: Date(timeIntervalSince1970: 102)
        )
        let first = ChatMessage(
            role: .user,
            visibleContent: "first",
            createdAt: Date(timeIntervalSince1970: 101),
            conversation: conversation
        )
        let second = ChatMessage(
            role: .assistant,
            visibleContent: "second",
            createdAt: Date(timeIntervalSince1970: 102),
            conversation: conversation
        )
        context.insert(conversation)
        context.insert(first)
        context.insert(second)
        try context.save()

        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" }
        )

        XCTAssertEqual(viewModel.current?.id, conversation.id)
        XCTAssertEqual(viewModel.visibleMessages.map(\.visibleContent), ["first", "second"])
        XCTAssertEqual(viewModel.visibleMessages.map(\.conversationID), [conversation.id, conversation.id])
    }

    @MainActor
    func testChatReasoningOffStartsWithGeneratingPhase() async throws {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" },
            reasoningEnabledProvider: { false }
        )

        viewModel.send("hello")

        XCTAssertEqual(viewModel.streamingPhase, .generating)
        XCTAssertTrue(viewModel.isStreaming)
        await viewModel.cancel()
    }

    @MainActor
    func testChatReasoningOnStartsWithThinkingPhase() async throws {
        let container = try ChatStore.makeInMemoryContainer()
        let chatClient = MTPLXChatClient(
            apiClient: MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:9")!)
        )
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { chatClient },
            modelName: { "mtplx-test-model" },
            reasoningEnabledProvider: { true }
        )

        viewModel.send("hello")

        XCTAssertEqual(viewModel.streamingPhase, .thinking)
        XCTAssertTrue(viewModel.isStreaming)
        await viewModel.cancel()
    }

    func testMetricsStreamConnectsWithAppAPIKey() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-auth-metrics",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            SNAPSHOT = json.loads(r'''\(Self.snapshotJSON)''')

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    if self.path.startswith("/v1/mtplx/metrics/stream"):
                        if self.headers.get("Authorization") != "Bearer dashboard-secret":
                            self.send_response(401)
                            self.end_headers()
                            return
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        self.wfile.write(
                            ("event: snapshot\\n"
                             f"data: {json.dumps(SNAPSHOT)}\\n\\n").encode("utf-8")
                        )
                        self.wfile.flush()
                        time.sleep(1)
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let eventSeen = expectation(description: "authorized metrics snapshot")
        let client = MetricsStreamClient(
            apiClient: MTPLXAPIClient(
                baseURL: URL(string: "http://127.0.0.1:\(port)")!,
                apiKey: "dashboard-secret"
            )
        )
        let task = Task {
            await client.connect(
                snapshotIntervalMs: 500,
                onState: { _ in },
                onEvent: { event in
                    if case .snapshot(let snapshot) = event,
                       snapshot.modelId == "mtplx-test-model" {
                        eventSeen.fulfill()
                    }
                }
            )
        }

        await fulfillment(of: [eventSeen], timeout: 5)
        task.cancel()
    }

    @MainActor
    func testBackendHeadlineDecodeUsesRawCompletionTPSBeforeDisplayTPS() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-raw-tps-metrics",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    if self.path.startswith("/v1/mtplx/metrics/stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        payload = {
                            "kind": "completed",
                            "envelope": {
                                "request_id": "r-raw-tps",
                                "decode_tok_s": 30.0,
                                "display_decode_tok_s": 39.0,
                            },
                        }
                        self.wfile.write(("event: completed\\n"
                                          f"data: {json.dumps(payload)}\\n\\n").encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(1)
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: port),
            settingsStore: MTPLXSettingsStore(settingsURL: temporaryDirectory().appendingPathComponent("settings.json"))
        )

        backend.startMetricsStream()

        let deadline = Date().addingTimeInterval(5)
        var heldDecode: Double?
        while Date() < deadline {
            if case .held(let value, _) = backend.headlineDecode {
                heldDecode = value
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(heldDecode ?? -1, 30.0, accuracy: 0.01)
    }

    @MainActor
    func testBackendHeadlineDecodeIgnoresCumulativeAndStaleSnapshotMaxDuringLiveRequest() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-live-tps-bounce-metrics",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import copy
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            BASE = json.loads(r'''\(Self.snapshotJSON)''')

            def write_event(writer, event, payload):
                writer.write((f"event: {event}\\n"
                              f"data: {json.dumps(payload)}\\n\\n").encode("utf-8"))
                writer.flush()

            def zero_snapshot():
                snap = copy.deepcopy(BASE)
                snap["latest"] = None
                snap["active_requests"] = 0
                snap["in_flight"] = []
                snap["rolling"]["count"] = 0
                snap["rolling"]["min"] = None
                snap["rolling"]["max"] = None
                snap["rolling"]["mean"] = None
                snap["rolling"]["p50"] = None
                snap["rolling"]["p95"] = None
                snap["rolling"]["history"] = []
                snap["rolling"]["live_history"] = []
                snap["rolling"]["sticky_all_time_max"] = 0.0
                return snap

            def stale_peak_snapshot():
                snap = copy.deepcopy(BASE)
                snap["active_requests"] = 1
                snap["in_flight"] = [{
                    "request_id": "current-request",
                    "started_s": 1.0,
                    "age_s": 0.2,
                    "session_id": "current-session",
                    "prompt_preview": "current request",
                    "prompt_tokens": 64,
                    "last_progress": {
                        "request_id": "current-request",
                        "session_id": "current-session",
                        "decode_tok_s": 21.0,
                        "completion_tokens": 210,
                        "decode_elapsed_s": 5.8333333333
                    },
                    "prefill_state": None,
                    "cancelled": False
                }]
                snap["latest"] = {
                    "request_id": "old-completed-request",
                    "session_id": "old-session",
                    "decode_tok_s": 36.0,
                    "completion_tokens": 360,
                    "decode_elapsed_s": 10.0
                }
                snap["rolling"]["max"] = 36.0
                snap["rolling"]["sticky_all_time_max"] = 36.0
                return snap

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    if self.path.startswith("/v1/mtplx/metrics/stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        write_event(self.wfile, "snapshot", zero_snapshot())
                        progress = {
                            "kind": "progress",
                            "request_id": "current-request",
                            "progress": {
                                "request_id": "current-request",
                                "session_id": "current-session",
                                "decode_tok_s": 21.0,
                                "display_decode_tok_s": 35.0,
                                "completion_tokens": 210,
                                "decode_elapsed_s": 5.8333333333
                            }
                        }
                        write_event(self.wfile, "progress", progress)
                        write_event(self.wfile, "snapshot", stale_peak_snapshot())
                        time.sleep(1)
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        defer { process.terminate() }

        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: port),
            settingsStore: MTPLXSettingsStore(settingsURL: temporaryDirectory().appendingPathComponent("settings.json"))
        )

        backend.startMetricsStream()

        let deadline = Date().addingTimeInterval(5)
        var liveDecode: Double?
        while Date() < deadline {
            if case .live(let value) = backend.headlineDecode {
                liveDecode = value
            }
            if liveDecode != nil, backend.latest?.values["request_id"]?.stringValue == "current-request" {
                try await Task.sleep(for: .milliseconds(300))
                if case .live(let value) = backend.headlineDecode {
                    liveDecode = value
                }
                break
            }
            try await Task.sleep(for: .milliseconds(50))
        }

        XCTAssertEqual(liveDecode ?? -1, 21.0, accuracy: 0.01)
        XCTAssertEqual(backend.latest?.values["request_id"]?.stringValue, "current-request")
    }

    func testPrefillStateDecodesChunkTiming() throws {
        let state = try JSONDecoder().decode(
            PrefillState.self,
            from: Data(
                """
                {
                  "phase": "chunk",
                  "tokens_done": 4096,
                  "tokens_total": 113281,
                  "elapsed_s": 12.5,
                  "prompt_eval_time_s": 10.0,
                  "prefill_tok_s": 327.68,
                  "prefill_compute_tok_s": 409.6,
                  "prefill_wall_tok_s": 327.68,
                  "cumulative_prefill_tok_s": 327.68,
                  "live_prefill_tok_s": 256.0,
                  "chunk_size": 2048,
                  "chunk_elapsed_s": 8.0,
                  "chunk_prefill_tok_s": 256.0
                }
                """.utf8
            )
        )

        XCTAssertEqual(state.chunkElapsedS, 8.0)
        XCTAssertEqual(state.chunkPrefillTokS, 256.0)
        XCTAssertEqual(state.promptEvalTimeS, 10.0)
        XCTAssertEqual(state.prefillComputeTokS, 409.6)
        XCTAssertEqual(state.prefillWallTokS, 327.68)
        XCTAssertEqual(state.cumulativePrefillTokS, 327.68)
        XCTAssertEqual(state.livePrefillTokS, 256.0)
    }

    func testAppConfigurationOnboardingFieldsRoundTrip() throws {
        let when = Date(timeIntervalSince1970: 1_780_000_000)
        let original = MTPLXAppConfiguration(
            onboardingCompletedAt: when,
            lastTunedDepth: 2,
            lastTunedAt: when,
            customModels: [
                try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "Foo/Bar"))
            ]
        )
        let data = try JSONEncoder().encode(original)
        let root = try JSONDecoder().decode([String: JSONValue].self, from: data)
        XCTAssertTrue(root.keys.contains("onboarding_completed_at"))
        XCTAssertTrue(root.keys.contains("last_tuned_depth"))
        XCTAssertTrue(root.keys.contains("last_tuned_at"))
        XCTAssertTrue(root.keys.contains("custom_models"))
        if case .number(let depth) = root["last_tuned_depth"] {
            XCTAssertEqual(Int(depth), 2)
        } else {
            XCTFail("last_tuned_depth should encode as a JSON number")
        }
        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)
        XCTAssertEqual(decoded.onboardingCompletedAt, when)
        XCTAssertEqual(decoded.lastTunedDepth, 2)
        XCTAssertEqual(decoded.lastTunedAt, when)
        XCTAssertEqual(decoded.customModels.map(\.hfModelID), ["Foo/Bar"])
    }

    func testCommandBuilderForeignModelTuneDoesNotLeakDepth() throws {
        // QA-107: a 35B onboarding tune (record + legacy lastTunedDepth)
        // must not follow the user to a 27B launch — the legacy field
        // has no model identity and is the foreign record's residue.
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
                profile: "sustained",
                host: "127.0.0.1",
                port: 8000,
                lastTunedDepth: 2,
                tunedControlRecord: TunedControlRecord(
                    modelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                    modelFamily: "qwen3_6",
                    backendID: "qwen3_next",
                    controlField: "depth",
                    controlValue: 2,
                    candidates: ["Base speeds", "MTP 1", "MTP 2", "MTP 3"],
                    tunedAt: Date(timeIntervalSince1970: 1_780_000_000)
                )
            ),
            target: .chat,
            launchID: "foreign-tune-leak"
        )
        XCTAssertFalse(
            command.arguments.containsInOrder(["--depth", "2"]),
            "the 35B tune must not pull the 27B off its contract depth"
        )
    }

    func testCommandBuilderChatLaneTunedDepthBeatsModelPresetDepth() throws {
        // QA-104 regression: the 35B Optimized Speed model preset pins
        // depth 1; a fresh compatible tune (depth 2) must win on the
        // chat lane or onboarding's "found your sweet spot" is a lie.
        // OpenCode keeps its literal D3 preset (separate pinned test).
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                profile: "sustained",
                host: "127.0.0.1",
                port: 8000,
                lastTunedDepth: 2
            ),
            target: .chat,
            launchID: "chat-tuned-depth"
        )
        XCTAssertTrue(
            command.arguments.containsInOrder(["--depth", "2"]),
            "chat lane must honor the tuned depth over the 35B model preset depth"
        )
        XCTAssertFalse(command.arguments.containsInOrder(["--depth", "1"]))
    }

    func testCommandBuilderThreadsLastTunedDepthWhenSet() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                profile: "sustained",
                host: "127.0.0.1",
                port: 8000,
                lastTunedDepth: 2
            )
        )
        XCTAssertTrue(
            command.arguments.contains("--depth"),
            "Expected --depth argument when lastTunedDepth is set"
        )
        if let i = command.arguments.firstIndex(of: "--depth") {
            XCTAssertEqual(command.arguments[i + 1], "2")
        }
    }

    func testCommandBuilderThreadsGemmaTunedBlockWhenSet() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/Users/youssof/Documents/MTPLX/models/hf-release/Gemma4-MTPLX-Optimized-Speed",
                tunedControlRecord: TunedControlRecord(
                    modelID: "Youssofal/Gemma4-MTPLX-Optimized-Speed",
                    modelFamily: "gemma4",
                    backendID: "gemma4_assistant",
                    controlField: "draft_block_size",
                    controlValue: 5,
                    candidates: ["AR", "Block 2", "Block 3", "Block 4", "Block 5", "Block 6", "Block 7", "Block 8"],
                    tunedAt: Date(timeIntervalSince1970: 1_780_000_000)
                )
            )
        )

        let i = try XCTUnwrap(command.arguments.firstIndex(of: "--depth"))
        XCTAssertEqual(command.arguments[i + 1], "5")
    }

    func testCommandBuilderOmitsDepthWhenUnsetOrOutOfRange() throws {
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        let unset = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen"
            )
        )
        XCTAssertFalse(unset.arguments.contains("--depth"))

        let outOfRange = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/qwen",
                lastTunedDepth: 99
            )
        )
        XCTAssertFalse(outOfRange.arguments.contains("--depth"))
    }

    @MainActor
    func testOnboardingStartTuneIgnoresCachedTuneAndRunsFreshProcess() async throws {
        let home = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".mtplx"),
            withIntermediateDirectories: true
        )
        let thermalBin = home
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: thermalBin, withIntermediateDirectories: true)
        let thermalforge = thermalBin.appendingPathComponent("thermalforge")
        try "#!/bin/sh\nexit 0\n".data(using: .utf8)!.write(to: thermalforge)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: thermalforge.path)
        let tuningJSON = """
        {
          "records": {
            "cached": {
              "key_material": {"model": "/models/qwen"},
              "payload": {
                "best": {"depth": 2, "tok_s": 44.0, "multiplier_vs_ar": 1.6},
                "best_multiplier": {"ar_tok_s": 27.5},
                "results": [
                  {"candidate": "ar", "tok_s": 27.5},
                  {"candidate": "2", "tok_s": 44.0, "multiplier_vs_ar": 1.6}
                ]
              }
            }
          }
        }
        """
        try tuningJSON.data(using: .utf8)!.write(
            to: home.appendingPathComponent(".mtplx").appendingPathComponent("tuning.json")
        )
        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            case "$1" in
              max)
                if [ "$2" = "--status" ]; then
                  echo '{"ok":true,"detection":{"available":true}}'
                else
                  echo '{"ok":true,"message":"fan command ok"}'
                fi
                exit 0
                ;;
              tune)
                echo fresh tune was attempted >&2
                exit 42
                ;;
            esac
            exit 9
            """
        )
        let orchestrator = OnboardingOrchestrator(
            autoTuner: AutoTuner(
                processEnvironment: [
                    "HOME": home.path,
                    "PATH": fake.deletingLastPathComponent().path,
                ],
                pollInterval: 0.01,
                preferDevelopmentWrapper: false
            ),
            initialState: OnboardingFeatureState(step: .tune, pick: .other(hfRepo: "/models/qwen"))
        )

        orchestrator.startTune()

        XCTAssertNil(orchestrator.tuneResult)
        XCTAssertTrue(orchestrator.isTuning)
        let deadline = Date().addingTimeInterval(1)
        while orchestrator.isTuning && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(orchestrator.isTuning)
        XCTAssertNil(orchestrator.tuneResult)
        XCTAssertTrue(orchestrator.tuneFailure?.contains("fresh tune was attempted") == true)
    }

    func testAutoTunerPassesRetuneSoCachedTuneDoesNotSkipRunArtifacts() async throws {
        let home = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".mtplx").appendingPathComponent("bin", isDirectory: true),
            withIntermediateDirectories: true
        )
        let thermalforge = home
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
            .appendingPathComponent("thermalforge")
        try "#!/bin/sh\nexit 0\n".data(using: .utf8)!.write(to: thermalforge)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: thermalforge.path)
        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            echo "$*" >> "$MTPLX_FAKE_LOG"
            if [ "$1" = "max" ]; then
              if [ "$2" = "--status" ]; then
                echo '{"ok":true,"detection":{"available":true}}'
              else
                echo '{"ok":true,"message":"fan command ok"}'
              fi
              exit 0
            fi
            has_retune=0
            output_dir=""
            run_id=""
            while [ "$#" -gt 0 ]; do
              case "$1" in
                --retune)
                  has_retune=1
                  shift
                  ;;
                --output-dir)
                  output_dir="$2"
                  shift 2
                  ;;
                --run-id)
                  run_id="$2"
                  shift 2
                  ;;
                *)
                  shift
                  ;;
              esac
            done
            if [ "$has_retune" -ne 1 ]; then
              echo '{"from_cache":true,"best":{"depth":2,"tok_s":44.0},"results":[]}'
              exit 0
            fi
            run_dir="$output_dir/$run_id"
            mkdir -p "$run_dir"
            cat > "$run_dir/ar.json" <<'JSON'
            {"ar_rows":[{"tok_s":50.0}]}
            JSON
            cat > "$run_dir/d1.json" <<'JSON'
            {"depths":[{"rows":[{"tok_s":80.0,"acceptance_by_depth":[0.82]}]}]}
            JSON
            cat > "$run_dir/tune.json" <<'JSON'
            {"best":{"depth":1,"tok_s":80.0,"multiplier_vs_ar":1.6},"best_multiplier":{"ar_tok_s":50.0},"results":[{"candidate":"ar","tok_s":50.0},{"candidate":"1","tok_s":80.0,"multiplier_vs_ar":1.6,"acceptance_by_depth":[0.82]}]}
            JSON
            exit 0
            """
        )
        let log = home.appendingPathComponent("mtplx.log")
        let tuner = AutoTuner(
            processEnvironment: [
                "HOME": home.path,
                "PATH": "\(fake.deletingLastPathComponent().path):/usr/bin:/bin",
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                "MTPLX_FAKE_LOG": log.path,
            ],
            pollInterval: 0.01
        )
        var completed: TuneResult?
        var failure: String?

        let modelPath = "/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
        for await event in tuner.stream(modelPath: modelPath, candidates: [.ar, .d1]) {
            switch event {
            case .completed(let result):
                completed = result
            case .failed(_, let stderrTail):
                failure = stderrTail
            default:
                continue
            }
        }

        XCTAssertNil(failure)
        XCTAssertEqual(completed?.bestDepth, 1)
        let commands = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(commands.contains("--retune"), commands)
        XCTAssertTrue(commands.contains("--profile turbo"), commands)
    }

    func testAutoTunerParsesGemmaBlockCandidates() throws {
        let payload: [String: Any] = [
            "best": [
                "mode": "Block 6",
                "depth": 6,
                "control_field": "draft_block_size",
                "tok_s": 61.0,
                "multiplier_vs_ar": 1.42,
            ],
            "best_multiplier": ["ar_tok_s": 43.0],
            "results": [
                ["candidate": "ar", "tok_s": 43.0],
                ["candidate": "2", "tok_s": 50.0, "multiplier_vs_ar": 1.16, "acceptance_by_depth": [0.82]],
                ["candidate": "6", "tok_s": 61.0, "multiplier_vs_ar": 1.42, "acceptance_by_depth": [0.86, 0.78, 0.71]],
                ["candidate": "block8", "tok_s": 55.0, "multiplier_vs_ar": 1.28, "acceptance_by_depth": [0.80]],
            ],
        ]

        let result = try XCTUnwrap(AutoTuner.parseFinal(payload: payload, candidates: TuneCandidate.gemmaCandidates))

        XCTAssertEqual(result.bestCandidate, .block6)
        XCTAssertEqual(result.bestDepth, 6)
        XCTAssertEqual(result.allCandidates.map(\.candidate), [.ar, .block2, .block6, .block8])
    }

    func testAutoTunerParsesUserFacingQwenCandidateLabels() throws {
        let payload: [String: Any] = [
            "best": [
                "depth": 2 as Int,
                "tok_s": 80.0,
                "multiplier_vs_ar": 1.6,
            ] as [String: Any],
            "best_multiplier": ["ar_tok_s": 50.0],
            "results": [
                ["candidate": "Base speeds", "tok_s": 50.0],
                ["candidate": "MTP 1", "tok_s": 72.0, "multiplier_vs_ar": 1.44],
                ["candidate": "MTP 2", "tok_s": 80.0, "multiplier_vs_ar": 1.6],
                ["candidate": "D3", "tok_s": 76.0, "multiplier_vs_ar": 1.52],
            ],
        ]

        let result = try XCTUnwrap(AutoTuner.parseFinal(payload: payload))

        XCTAssertEqual(result.bestCandidate, .d2)
        XCTAssertEqual(result.allCandidates.map(\.candidate), [.ar, .d1, .d2, .d3])
    }

    func testAutoTunerParsesCompletedARWinsPayloadWithoutBest() throws {
        let payload: [String: Any] = [
            "best": NSNull(),
            "best_multiplier": [
                "ar_tok_s": 132.8,
                "winner": NSNull(),
                "failure_reasons": ["mtp_acceptance_collapsed", "no_mtp_depth_beat_ar"],
            ] as [String: Any],
            "results": [
                ["candidate": "ar", "tok_s": 132.8, "multiplier_vs_ar": 1.0],
                ["candidate": "1", "depth": 1, "tok_s": 99.1, "multiplier_vs_ar": 0.75],
                ["candidate": "2", "depth": 2, "tok_s": 79.8, "multiplier_vs_ar": 0.60],
                ["candidate": "3", "depth": 3, "tok_s": 62.3, "multiplier_vs_ar": 0.47],
            ],
        ]

        let result = try XCTUnwrap(AutoTuner.parseFinal(payload: payload))

        XCTAssertEqual(result.bestCandidate, .ar)
        XCTAssertEqual(result.bestDepth, 0)
        XCTAssertEqual(result.bestTokS, 132.8, accuracy: 0.001)
        XCTAssertEqual(result.bestMultiplierVsAR, 1.0, accuracy: 0.001)
        XCTAssertEqual(result.allCandidates.map(\.candidate), [.ar, .d1, .d2, .d3])
    }

    func testAutoTunerInstallsFanControlBeforeRunningTuneWhenMissing() async throws {
        let home = temporaryDirectory()
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            echo "$*" >> "$MTPLX_FAKE_LOG"
            case "$1" in
              max)
                mkdir -p "$HOME/.mtplx/bin"
                cat > "$HOME/.mtplx/bin/thermalforge" <<'THERMAL'
            #!/bin/sh
            exit 0
            THERMAL
                chmod +x "$HOME/.mtplx/bin/thermalforge"
                echo '{"ok":true,"message":"Fan control ready"}'
                exit 0
                ;;
              tune)
                output_dir=""
                run_id=""
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --output-dir)
                      output_dir="$2"
                      shift 2
                      ;;
                    --run-id)
                      run_id="$2"
                      shift 2
                      ;;
                    *)
                      shift
                      ;;
                  esac
                done
                run_dir="$output_dir/$run_id"
                mkdir -p "$run_dir"
                cat > "$run_dir/ar.json" <<'JSON'
            {"ar_rows":[{"tok_s":50.0}]}
            JSON
                cat > "$run_dir/d1.json" <<'JSON'
            {"depths":[{"rows":[{"tok_s":75.0,"acceptance_by_depth":[0.8]}]}]}
            JSON
                cat > "$run_dir/tune.json" <<'JSON'
            {"best":{"depth":1,"tok_s":75.0,"multiplier_vs_ar":1.5},"best_multiplier":{"ar_tok_s":50.0},"results":[{"candidate":"ar","tok_s":50.0},{"candidate":"1","tok_s":75.0,"multiplier_vs_ar":1.5,"acceptance_by_depth":[0.8]}]}
            JSON
                exit 0
                ;;
            esac
            exit 9
            """
        )
        let log = home.appendingPathComponent("mtplx.log")
        let tuner = AutoTuner(
            processEnvironment: [
                "HOME": home.path,
                "PATH": "\(fake.deletingLastPathComponent().path):/usr/bin:/bin",
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                "MTPLX_FAKE_LOG": log.path,
            ],
            pollInterval: 0.01
        )
        var statuses: [String] = []
        var completed: TuneResult?
        var failure: String?

        for await event in tuner.stream(modelPath: "/models/qwen", candidates: [.ar, .d1]) {
            switch event {
            case .installingFanControl(let message):
                statuses.append(message)
            case .completed(let result):
                completed = result
            case .failed(_, let stderrTail):
                failure = stderrTail
            default:
                continue
            }
        }

        XCTAssertEqual(statuses, [
            "Checking MTPLX runtime",
            "MTPLX runtime ready",
            "Checking fan control",
            "Installing fan control",
            "Fan control ready",
        ])
        XCTAssertNil(failure)
        XCTAssertEqual(completed?.bestDepth, 1)
        let commands = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(commands.contains("max --install --json"))
        XCTAssertFalse(commands.contains("max --on --json"))
        XCTAssertFalse(commands.contains("max --off --json"))
        XCTAssertTrue(commands.contains("tune --model /models/qwen"))
    }

    func testAutoTunerReportsCandidateLogFailureInsteadOfFanControlTail() async throws {
        let home = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".mtplx").appendingPathComponent("bin", isDirectory: true),
            withIntermediateDirectories: true
        )
        let thermalforge = home
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
            .appendingPathComponent("thermalforge")
        try "#!/bin/sh\nexit 0\n".data(using: .utf8)!.write(to: thermalforge)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: thermalforge.path)
        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            case "$1" in
              max)
                if [ "$2" = "--status" ]; then
                  echo '{"ok":true,"detection":{"available":true}}'
                  exit 0
                fi
                ;;
              tune)
                output_dir=""
                run_id=""
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --output-dir)
                      output_dir="$2"
                      shift 2
                      ;;
                    --run-id)
                      run_id="$2"
                      shift 2
                      ;;
                    *)
                      shift
                      ;;
                  esac
                done
                run_dir="$output_dir/$run_id"
                mkdir -p "$run_dir"
                cat > "$run_dir/ar.log" <<'LOG'
            Traceback (most recent call last):
              File "mtp_patch.py", line 49, in validate
                raise ValueError("mtp_quant_policy must be None, 'all', or 'cyankiwi'")
            ValueError: mtp_quant_policy must be None, 'all', or 'cyankiwi'
            LOG
                cat > "$run_dir/tune.json" <<JSON
            {"results":[{"candidate":"ar","command":["/opt/homebrew/var/mtplx/venv-0.3.7/bin/python","-m","mtplx.cli"],"error":"candidate did not write an artifact","returncode":1,"stdout":"$run_dir/ar.log"}]}
            JSON
                echo "[max] running thermalforge max ..." >&2
                exit 1
                ;;
            esac
            exit 9
            """
        )
        let tuner = AutoTuner(
            processEnvironment: [
                "HOME": home.path,
                "PATH": "\(fake.deletingLastPathComponent().path):/usr/bin:/bin",
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ],
            pollInterval: 0.01
        )
        var failure: String?

        for await event in tuner.stream(modelPath: "/models/qwen", candidates: [.ar]) {
            if case .failed(_, let stderrTail) = event {
                failure = stderrTail
            }
        }

        let message = try XCTUnwrap(failure)
        XCTAssertTrue(message.contains("MTPLX runtime failed while loading the model."), message)
        XCTAssertTrue(message.contains("venv-0.3.7"), message)
        XCTAssertTrue(message.contains("ValueError: mtp_quant_policy"), message)
        XCTAssertFalse(message.contains("[max] running thermalforge max"), message)
    }

    func testAutoTunerDoesNotToggleFansDuringReadinessCheck() async throws {
        let home = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".mtplx").appendingPathComponent("bin", isDirectory: true),
            withIntermediateDirectories: true
        )
        let thermalforge = home
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
            .appendingPathComponent("thermalforge")
        try "#!/bin/sh\nexit 0\n".data(using: .utf8)!.write(to: thermalforge)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: thermalforge.path)

        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            echo "$*" >> "$MTPLX_FAKE_LOG"
            marker="$HOME/.mtplx/fan-repaired"
            case "$1" in
              max)
                if [ "$2" = "--status" ]; then
                  echo '{"ok":true,"detection":{"available":true}}'
                  exit 0
                fi
                ;;
              tune)
                output_dir=""
                run_id=""
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --output-dir)
                      output_dir="$2"
                      shift 2
                      ;;
                    --run-id)
                      run_id="$2"
                      shift 2
                      ;;
                    *)
                      shift
                      ;;
                  esac
                done
                run_dir="$output_dir/$run_id"
                mkdir -p "$run_dir"
                cat > "$run_dir/tune.json" <<'JSON'
            {"best":{"depth":1,"tok_s":75.0,"multiplier_vs_ar":1.5},"best_multiplier":{"ar_tok_s":50.0},"results":[{"candidate":"ar","tok_s":50.0},{"candidate":"1","tok_s":75.0,"multiplier_vs_ar":1.5,"acceptance_by_depth":[0.8]}]}
            JSON
                exit 0
                ;;
            esac
            exit 9
            """
        )
        let log = home.appendingPathComponent("mtplx.log")
        let tuner = AutoTuner(
            processEnvironment: [
                "HOME": home.path,
                "PATH": "\(fake.deletingLastPathComponent().path):/usr/bin:/bin",
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                "MTPLX_FAKE_LOG": log.path,
            ],
            pollInterval: 0.01
        )
        var statuses: [String] = []
        var completed: TuneResult?
        var failure: String?

        for await event in tuner.stream(modelPath: "/models/qwen", candidates: [.ar, .d1]) {
            switch event {
            case .installingFanControl(let message):
                statuses.append(message)
            case .completed(let result):
                completed = result
            case .failed(_, let stderrTail):
                failure = stderrTail
            default:
                continue
            }
        }

        XCTAssertEqual(statuses, [
            "Checking MTPLX runtime",
            "MTPLX runtime ready",
            "Checking fan control",
            "Fan control ready",
        ])
        XCTAssertNil(failure)
        XCTAssertEqual(completed?.bestDepth, 1)
        let commands = try String(contentsOf: log, encoding: .utf8)
        XCTAssertTrue(commands.contains("max --status --json"), commands)
        XCTAssertFalse(commands.contains("max --install --json"), commands)
        XCTAssertFalse(commands.contains("max --on --json"), commands)
        XCTAssertFalse(commands.contains("max --off --json"), commands)
        XCTAssertTrue(commands.contains("tune --model /models/qwen"), commands)
    }

    @MainActor
    func testPostDownloadTuneShowsPreparingStatusBeforeCandidateFiles() async throws {
        let root = temporaryDirectory()
        let modelDir = try makeCompleteModel(named: "Example--Qwen3.6-Downloaded")
        let home = temporaryDirectory()
        let fake = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            case "$1" in
              max)
                if [ "$2" = "--status" ]; then
                  echo '{"ok":true,"detection":{"available":true}}'
                  exit 0
                fi
                ;;
              tune)
                sleep 10
                exit 0
                ;;
            esac
            exit 9
            """
        )
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(model: "Example/Qwen3.6-Downloaded"),
            settingsStore: MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json")),
            autoTuner: AutoTuner(
                processEnvironment: [
                    "HOME": home.path,
                    "PATH": "\(fake.deletingLastPathComponent().path):/usr/bin:/bin",
                    "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                ],
                pollInterval: 0.01
            )
        )
        let request = PendingModelDownload(
            repoID: "Example/Qwen3.6-Downloaded",
            displayName: "Qwen 3.6 Downloaded",
            shortName: "Qwen Downloaded",
            target: .chat,
            launchAction: .restart,
            totalBytes: 10,
            destinationPath: modelDir.path
        )

        await backend.handleModelDownloadEvent(
            .complete(bytesOnDisk: 10, path: modelDir.path),
            request: request
        )
        backend.runPendingModelTune()

        XCTAssertTrue(backend.isModelTuning)
        XCTAssertEqual(backend.modelTuneStatusMessage, "Preparing max fans and loading model")
        XCTAssertEqual(backend.modelTuneCandidatesLanded, [:])

        backend.cancelPendingModelTune()
    }

    @MainActor
    func testOnboardingSkipTuneUsesSafeDefaultWithoutStartingProcess() {
        let orchestrator = OnboardingOrchestrator(
            initialState: OnboardingFeatureState(step: .tune, pick: .other(hfRepo: "/models/qwen"))
        )

        orchestrator.skipTuneWithSafeDefault()

        XCTAssertFalse(orchestrator.isTuning)
        XCTAssertEqual(orchestrator.tuneResult?.bestDepth, 2)
        XCTAssertEqual(orchestrator.tuneResult?.bestCandidate, .d2)
        XCTAssertEqual(orchestrator.tuneResult?.allCandidates, [])
    }

    func testAppConfigurationBackCompatWithoutOnboardingFields() throws {
        let legacy = """
        {
          "model": "/models/qwen",
          "host": "127.0.0.1",
          "port": 8000,
          "last_launch_target": "chat"
        }
        """
        let decoded = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: Data(legacy.utf8)
        )
        XCTAssertNil(decoded.onboardingCompletedAt)
        XCTAssertNil(decoded.lastTunedDepth)
        XCTAssertNil(decoded.lastTunedAt)
        XCTAssertEqual(decoded.customModels, [])
        XCTAssertEqual(decoded.model, "/models/qwen")
        XCTAssertEqual(decoded.port, 8000)
    }

    func testAPIClientModelsDecodeBackendFixtures() throws {
        let decoder = JSONDecoder()
        let health = try decoder.decode(HealthPayload.self, from: Data(Self.healthJSON.utf8))
        XCTAssertEqual(health.model, "mtplx-test-model")
        XCTAssertEqual(health.chipName, "Apple M5 Max")
        XCTAssertEqual(health.machineModel, "Mac16,1")
        XCTAssertEqual(health.startup?.launchId, "fixture-launch")
        XCTAssertEqual(health.thermal?.actualRampVerified, true)

        let capabilities = try decoder.decode(
            AppCapabilities.self,
            from: Data(Self.capabilitiesJSON.utf8)
        )
        XCTAssertEqual(capabilities.apiVersion, 1)
        XCTAssertEqual(capabilities.snapshotInterval.nativeDefaultMs, 500)
        XCTAssertTrue(capabilities.features["sse_metrics"] == true)
        XCTAssertTrue(capabilities.features["ssd_session_cache"] == true)
        XCTAssertTrue(capabilities.features["startup_ownership"] == true)
        XCTAssertTrue(capabilities.features["strict_max_fan_startup"] == true)
    }

    func testDaemonSupervisorStartsFakeProcessAndKeepsLogsBounded() async throws {
        let script = try makeExecutable(
            named: "fake-mtplx",
            body: """
            #!/bin/sh
            echo fake daemon ready
            while true; do sleep 1; done
            """
        )
        let logs = BoundedLogStore(capacity: 4)
        let supervisor = DaemonSupervisor(logStore: logs)

        _ = try await supervisor.start(
            command: DaemonCommand(executableURL: script, arguments: []),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        XCTAssertTrue(supervisor.isRunning())
        try await Task.sleep(nanoseconds: 200_000_000)
        await supervisor.stop(graceSeconds: 0.1)
        XCTAssertFalse(supervisor.isRunning())
        let snapshot = await logs.snapshot()
        XCTAssertLessThanOrEqual(snapshot.count, 4)
        XCTAssertTrue(snapshot.contains { $0.message.contains("fake daemon ready") })
    }

    func testDaemonSupervisorStopKillsChildProcessFamily() async throws {
        let root = temporaryDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let childPIDFile = root.appendingPathComponent("child.pid")
        let script = try makeExecutable(
            named: "fake-mtplx-tree",
            body: """
            #!/bin/sh
            python3 -u - <<'PY' &
            import pathlib
            import signal
            import time
            import os

            pathlib.Path("\(childPIDFile.path)").write_text(str(os.getpid()))
            signal.signal(signal.SIGTERM, lambda *_: None)
            signal.signal(signal.SIGHUP, lambda *_: None)
            while True:
                time.sleep(1)
            PY
            echo fake daemon tree ready
            while true; do sleep 1; done
            """
        )
        let supervisor = DaemonSupervisor(logStore: BoundedLogStore(capacity: 8))

        _ = try await supervisor.start(
            command: DaemonCommand(executableURL: script, arguments: []),
            healthBaseURL: URL(string: "http://127.0.0.1:9")!,
            probeHealth: false
        )
        let childPID = try await waitForPIDFile(childPIDFile)
        XCTAssertTrue(pidIsAlive(childPID))

        await supervisor.stop(graceSeconds: 0.1)

        XCTAssertFalse(supervisor.isRunning())
        XCTAssertFalse(pidIsAlive(childPID), "Stop must reap child daemon processes, not only the wrapper")
    }

    func testDaemonSupervisorCleansLaunchedProcessAfterHealthTimeout() async throws {
        let script = try makeExecutable(
            named: "fake-mtplx-timeout",
            body: """
            #!/bin/sh
            while true; do sleep 1; done
            """
        )
        let supervisor = DaemonSupervisor(logStore: BoundedLogStore())

        do {
            _ = try await supervisor.start(
                command: DaemonCommand(executableURL: script, arguments: []),
                healthBaseURL: URL(string: "http://127.0.0.1:9")!,
                probeHealth: true,
                timeoutSeconds: 0.3
            )
            XCTFail("expected healthTimeout")
        } catch DaemonSupervisorError.healthTimeout {
            XCTAssertFalse(supervisor.isRunning())
        }
    }

    func testFakeDaemonHealthProbeAndMetricsStreamSmoke() async throws {
        let port = try freeTCPPort()
        let script = try makeExecutable(
            named: "fake-mtplx-http",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            HEALTH = json.loads(r'''\(Self.healthJSON)''')
            SNAPSHOT = json.loads(r'''\(Self.snapshotJSON)''')

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

                def do_GET(self):
                    if self.path == "/health":
                        self._json(HEALTH)
                    elif self.path == "/v1/mtplx/thermal/status":
                        self._json({
                            "ok": True,
                            "current_mode": "max",
                            "detection": {"available": True},
                            "fan_summary": {"ok": True},
                        })
                    elif self.path == "/v1/mtplx/app/capabilities":
                        self._json({
                            "ok": True,
                            "name": "MTPLX test daemon",
                            "api_version": 1,
                            "endpoints": {},
                            "mutable_settings": [],
                            "restart_required_settings": [],
                            "snapshot_interval": {
                                "default_ms": 500,
                                "min_ms": 250,
                                "max_ms": 5000,
                                "native_default_ms": 500,
                                "performance_lock_ms": 1000,
                            },
                            "features": {},
                            "scheduler": {},
                        })
                    elif self.path == "/admin/sessions":
                        self._json({"sessions": [], "count": 0})
                    elif self.path == "/v1/mtplx/prefill_history":
                        self._json({"capacity": 0, "history": []})
                    elif self.path == "/v1/models":
                        self._json({
                            "object": "list",
                            "data": [{
                                "id": "mtplx-test-model",
                                "object": "model",
                                "owned_by": "mtplx",
                            }],
                        })
                    elif self.path.startswith("/v1/mtplx/metrics/stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        self.wfile.write(
                            ("event: snapshot\\n"
                             f"data: {json.dumps(SNAPSHOT)}\\n\\n").encode("utf-8")
                        )
                        self.wfile.flush()
                        time.sleep(1)
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                    if self.path == "/v1/mtplx/thermal/fan_mode":
                        length = int(self.headers.get("Content-Length", "0") or "0")
                        raw = self.rfile.read(length) if length else b"{}"
                        try:
                            request = json.loads(raw.decode("utf-8") or "{}")
                        except Exception:
                            request = {}
                        mode = request.get("mode") or "max"
                        self._json({
                            "verified": True,
                            "current_mode": mode,
                            "result": {"ok": True},
                            "fan_summary": {"ok": True},
                        })
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let logs = BoundedLogStore(capacity: 16)
        let supervisor = DaemonSupervisor(logStore: logs)

        _ = try await supervisor.start(
            command: DaemonCommand(executableURL: script, arguments: []),
            healthBaseURL: baseURL,
            probeHealth: true,
            timeoutSeconds: 5
        )
        XCTAssertTrue(supervisor.isRunning())

        let eventSeen = expectation(description: "snapshot event seen")
        let stream = MetricsStreamClient(apiClient: MTPLXAPIClient(baseURL: baseURL))
        let streamTask = Task {
            await stream.connect(
                snapshotIntervalMs: 500,
                onState: { _ in },
                onEvent: { event in
                    if case .snapshot(let snapshot) = event,
                       snapshot.modelId == "mtplx-test-model" {
                        eventSeen.fulfill()
                    }
                }
            )
        }

        await fulfillment(of: [eventSeen], timeout: 5)
        streamTask.cancel()
        await supervisor.stop(graceSeconds: 0.1)
        XCTAssertFalse(supervisor.isRunning())
    }

    func testDaemonSupervisorRejectsOccupiedPortBeforeLaunch() async throws {
        let port = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }

        let script = try makeExecutable(named: "fake-mtplx", body: "#!/bin/sh\nsleep 5\n")
        let supervisor = DaemonSupervisor(logStore: BoundedLogStore())

        do {
            _ = try await supervisor.start(
                command: DaemonCommand(executableURL: script, arguments: []),
                healthBaseURL: URL(string: "http://127.0.0.1:\(port)")!,
                probeHealth: true,
                expectedLaunchID: "new-launch"
            )
            XCTFail("expected portOccupied")
        } catch DaemonSupervisorError.portOccupied(let pid, let launchID) {
            XCTAssertEqual(pid, 12345)
            XCTAssertEqual(launchID, "fixture-launch")
        }
    }

    func testDaemonSupervisorAdoptsHealthyAppOwnedDaemonOnOccupiedPort() async throws {
        let port = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }

        let script = try makeExecutable(named: "fake-mtplx", body: "#!/bin/sh\nsleep 5\n")
        let supervisor = DaemonSupervisor(logStore: BoundedLogStore())

        let health = try await supervisor.start(
            command: DaemonCommand(
                executableURL: script,
                arguments: ["--model", "/models/test"]
            ),
            healthBaseURL: URL(string: "http://127.0.0.1:\(port)")!,
            probeHealth: true,
            expectedLaunchID: "new-launch",
            requireActualFanRamp: true,
            adoptExistingAppOwnedDaemon: true
        )

        XCTAssertEqual(health?.startup?.launchId, "fixture-launch")
        XCTAssertTrue(supervisor.isRunning())
    }

    /// SYNC GUARD (report §10.6): the app's TargetPreset tables and the
    /// CLI's `mtplx start <target>` defaults must describe the same launch
    /// policy. Compares the OpenCode serve command the app would spawn
    /// against the CLI's own dry-run contract; skips cleanly when no
    /// runtime is resolvable so CI never flakes.
    func testTargetPresetParityWithCLIStartDryRun() throws {
        let runtime: URL
        if let wrapper = sourceTreeWrapper() {
            runtime = wrapper
        } else if let installed = try? MTPLXCommandBuilder.resolveInstalledExecutable(
            environment: ProcessInfo.processInfo.environment
        ) {
            runtime = installed
        } else {
            throw XCTSkip("no mtplx runtime resolvable; parity check needs the CLI")
        }

        let isolatedHome = temporaryDirectory()
        try FileManager.default.createDirectory(
            at: isolatedHome,
            withIntermediateDirectories: true
        )
        let process = Process()
        process.executableURL = runtime
        process.arguments = [
            "start", "opencode",
            "--dry-run", "--json",
            "--model", "/tmp/mtplx-parity-model",
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["HOME"] = isolatedHome.path
        environment["MTPLX_START_ATTACH_PROBE"] = "off"
        process.environment = environment
        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw XCTSkip(
                "mtplx start --dry-run exited \(process.terminationStatus); runtime not usable here"
            )
        }
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        let payload = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let opencode = try XCTUnwrap(payload["opencode"] as? [String: Any])
        let cliSampler = try XCTUnwrap(opencode["target_sampler"] as? [String: Any])

        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(
            environment: ["PATH": fake.deletingLastPathComponent().path]
        )
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/tmp/mtplx-parity-model"
            ),
            target: .openCode,
            launchID: "parity-launch"
        )

        func argumentValue(_ flag: String) -> String? {
            guard let index = command.arguments.firstIndex(of: flag),
                  index + 1 < command.arguments.count
            else { return nil }
            return command.arguments[index + 1]
        }

        XCTAssertEqual(
            argumentValue("--temperature").flatMap(Double.init),
            cliSampler["temperature"] as? Double,
            "app and CLI disagree on the OpenCode target temperature"
        )
        XCTAssertEqual(
            argumentValue("--top-p").flatMap(Double.init),
            cliSampler["top_p"] as? Double,
            "app and CLI disagree on the OpenCode target top_p"
        )
        XCTAssertEqual(
            argumentValue("--top-k").flatMap(Int.init),
            cliSampler["top_k"] as? Int,
            "app and CLI disagree on the OpenCode target top_k"
        )
        XCTAssertEqual(
            argumentValue("--tool-prompt-mode"),
            opencode["tool_prompt_mode"] as? String,
            "app and CLI disagree on the OpenCode tool prompt mode"
        )
        XCTAssertEqual(
            argumentValue("--chat-template-profile"),
            opencode["chat_template_profile"] as? String,
            "app and CLI disagree on the OpenCode chat template profile"
        )
        // The app emits no --profile on the default (auto) configuration,
        // so the CLI's own per-model resolution IS the launch profile and
        // parity holds by construction. Pin both halves of that contract:
        // the app side stays flagless (an accidental pin here re-creates
        // the F20 sustained lock) and the CLI side still resolves a
        // concrete engine profile for the same model.
        XCTAssertNil(
            argumentValue("--profile"),
            "auto must not pin a profile — the engine owns the default"
        )
        XCTAssertNotNil(
            payload["profile"] as? String,
            "CLI dry-run must resolve a concrete default launch profile"
        )
    }

    func testRuntimeBootstrapperPrefersBundledPythonOverSystem() throws {
        let bundled = try makeExecutable(
            named: "python3",
            body: "#!/bin/sh\necho \"Python 3.14.5\"\n"
        )
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: [
            "MTPLX_APP_BUNDLED_PYTHON": bundled.path,
            "PATH": "/nonexistent",
        ])

        let resolved = try bootstrapper.resolvePythonExecutable()

        XCTAssertEqual(resolved.path, bundled.path)
    }

    func testRuntimeBootstrapperExplicitPythonOverrideBeatsBundled() throws {
        let explicit = try makeExecutable(
            named: "python3-explicit",
            body: "#!/bin/sh\necho \"Python 3.13.2\"\n"
        )
        let bundled = try makeExecutable(
            named: "python3",
            body: "#!/bin/sh\necho \"Python 3.14.5\"\n"
        )
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: [
            "MTPLX_APP_PYTHON_PATH": explicit.path,
            "MTPLX_APP_BUNDLED_PYTHON": bundled.path,
        ])

        let resolved = try bootstrapper.resolvePythonExecutable()

        XCTAssertEqual(resolved.path, explicit.path)
    }

    func testBundledPythonPathRequiresExecutableOverride() throws {
        XCTAssertNil(
            MTPLXCommandBuilder.bundledPythonExecutablePath(
                environment: ["MTPLX_APP_BUNDLED_PYTHON": "/nonexistent/python3"]
            )
        )
        let bundled = try makeExecutable(
            named: "python3",
            body: "#!/bin/sh\necho \"Python 3.14.5\"\n"
        )
        XCTAssertEqual(
            MTPLXCommandBuilder.bundledPythonExecutablePath(
                environment: ["MTPLX_APP_BUNDLED_PYTHON": bundled.path]
            ),
            bundled.path
        )
    }

    func testPortPreflightClassifiesFreeMTPLXAndForeign() async throws {
        let freePort = try freeTCPPort()
        let freeKind = await PortPreflight.classify(
            baseURL: URL(string: "http://127.0.0.1:\(freePort)")!,
            apiKey: nil
        )
        XCTAssertEqual(freeKind, .free)

        let mtplxPort = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: mtplxPort,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }
        let mtplxKind = await PortPreflight.classify(
            baseURL: URL(string: "http://127.0.0.1:\(mtplxPort)")!,
            apiKey: nil
        )
        guard case .mtplxServer(let health) = mtplxKind else {
            XCTFail("expected mtplxServer, got \(mtplxKind)")
            return
        }
        XCTAssertEqual(health.startup?.launchId, "fixture-launch")

        let garbagePort = try freeTCPPort()
        let garbage = try startGarbageHTTPServer(port: garbagePort)
        defer { garbage.terminate() }
        let foreignKind = try await waitForNonFreeClassification(port: garbagePort)
        XCTAssertEqual(foreignKind, .foreign)
    }

    func testPortPreflightNextFreePortSkipsBoundPort() throws {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bindResult, 0)
        XCTAssertEqual(Darwin.listen(descriptor, 1), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        _ = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(descriptor, $0, &length)
            }
        }
        let boundPort = Int(UInt16(bigEndian: boundAddress.sin_port))

        let next = PortPreflight.nextFreePort(after: boundPort - 1)

        XCTAssertNotNil(next)
        XCTAssertNotEqual(next, boundPort)
    }

    // MARK: - Daemon watchdog liveness probes (QA-114)

    func testHealthWithinDeadlineTreatsWedgedServerAsMiss() async throws {
        // listen() with no accept(): connections complete the TCP
        // handshake in the kernel backlog and then hang — the wedge
        // shape from QA-113 (alive process, LISTENing port, HTTP that
        // never answers and never refuses).
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bindResult, 0)
        XCTAssertEqual(Darwin.listen(descriptor, 4), 0)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        _ = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(descriptor, $0, &length)
            }
        }
        let port = Int(UInt16(bigEndian: boundAddress.sin_port))

        let probe = MTPLXAPIClient.livenessProbe(
            baseURL: URL(string: "http://127.0.0.1:\(port)")!,
            apiKey: nil
        )
        let started = Date()
        let health = await probe.healthWithinDeadline(seconds: 1)
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertNil(health, "a wedged daemon must register as a miss")
        XCTAssertLessThan(elapsed, 5, "the deadline must bound the probe")
    }

    func testHealthWithinDeadlineReturnsPayloadFromResponsiveServer() async throws {
        let port = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: port,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }

        let probe = MTPLXAPIClient.livenessProbe(
            baseURL: URL(string: "http://127.0.0.1:\(port)")!,
            apiKey: nil
        )
        let health = await probe.healthWithinDeadline(seconds: 5)

        XCTAssertEqual(health?.ok, true)
        XCTAssertEqual(health?.startup?.launchId, "fixture-launch")
    }

    func testLivenessProbeClientIsIsolatedFromSharedSessionAndTight() {
        let probe = MTPLXAPIClient.livenessProbe(
            baseURL: URL(string: "http://127.0.0.1:1")!,
            apiKey: nil
        )
        XCTAssertFalse(probe.session === URLSession.shared)
        let configuration = probe.session.configuration
        XCTAssertLessThanOrEqual(configuration.timeoutIntervalForRequest, 5)
        XCTAssertLessThanOrEqual(configuration.timeoutIntervalForResource, 10)
        XCTAssertEqual(configuration.httpMaximumConnectionsPerHost, 1)
    }

    // MARK: - Runtime venv wheel fingerprint (auto-update integrity)

    private func makeRuntimeFixture(
        wheelContents: String
    ) throws -> (environment: [String: String], runtimeDir: URL, managedExecutable: URL, wheel: URL, home: URL) {
        let home = temporaryDirectory()
        let runtimeDir = home
            .appendingPathComponent("Library")
            .appendingPathComponent("Application Support")
            .appendingPathComponent("MTPLX")
            .appendingPathComponent("runtime-venv")
        let bin = runtimeDir.appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        let managed = bin.appendingPathComponent("mtplx")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: managed)
        let wheel = home.appendingPathComponent("mtplx-1.0.0-py3-none-any.whl")
        try Data(wheelContents.utf8).write(to: wheel)
        let environment = [
            "HOME": home.path,
            "MTPLX_BUNDLED_RUNTIME_WHEEL": wheel.path,
        ]
        return (environment, runtimeDir, managed, wheel, home)
    }

    func testManagedVenvWithoutFingerprintMarkerIsRefreshed() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: fixture.environment)
        XCTAssertFalse(
            bootstrapper.installedRuntimeMatchesBundledWheel(
                installedExecutable: fixture.managedExecutable
            ),
            "a venv with no recorded install fingerprint must reinstall once"
        )
    }

    func testManagedVenvMatchingBundledWheelIsReused() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        MTPLXRuntimeBootstrapper.recordWheelFingerprint(
            for: fixture.wheel,
            runtimeDir: fixture.runtimeDir
        )
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: fixture.environment)
        XCTAssertTrue(
            bootstrapper.installedRuntimeMatchesBundledWheel(
                installedExecutable: fixture.managedExecutable
            )
        )
    }

    func testSameVersionWheelRebuildForcesVenvRefresh() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        MTPLXRuntimeBootstrapper.recordWheelFingerprint(
            for: fixture.wheel,
            runtimeDir: fixture.runtimeDir
        )
        // Same filename (same semantic version), new contents — the
        // auto-update shape the version floor cannot see.
        try Data("wheel-B".utf8).write(to: fixture.wheel)
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: fixture.environment)
        XCTAssertFalse(
            bootstrapper.installedRuntimeMatchesBundledWheel(
                installedExecutable: fixture.managedExecutable
            ),
            "a rebuilt wheel under the same version must refresh the venv"
        )
    }

    func testShimSymlinkToManagedVenvIsTreatedAsManaged() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        let shimDir = fixture.home
            .appendingPathComponent(".mtplx")
            .appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: shimDir, withIntermediateDirectories: true)
        let shim = shimDir.appendingPathComponent("mtplx")
        try FileManager.default.createSymbolicLink(
            at: shim,
            withDestinationURL: fixture.managedExecutable
        )
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: fixture.environment)
        XCTAssertFalse(
            bootstrapper.installedRuntimeMatchesBundledWheel(installedExecutable: shim),
            "the ~/.mtplx/bin shim resolves into the managed venv and must honor the fingerprint"
        )
    }

    func testForeignRuntimeKeepsVersionFloorContract() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        let foreign = fixture.home.appendingPathComponent("homebrew-mtplx")
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: foreign)
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: fixture.environment)
        XCTAssertTrue(
            bootstrapper.installedRuntimeMatchesBundledWheel(installedExecutable: foreign),
            "non-app-managed runtimes are not fingerprint-gated"
        )
    }

    func testWheelFingerprintRoundTripAndChange() throws {
        let fixture = try makeRuntimeFixture(wheelContents: "wheel-A")
        let first = try MTPLXRuntimeBootstrapper.wheelFingerprint(of: fixture.wheel)
        MTPLXRuntimeBootstrapper.recordWheelFingerprint(
            for: fixture.wheel,
            runtimeDir: fixture.runtimeDir
        )
        XCTAssertEqual(
            MTPLXRuntimeBootstrapper.recordedWheelFingerprint(runtimeDir: fixture.runtimeDir),
            first
        )
        try Data("wheel-B".utf8).write(to: fixture.wheel)
        XCTAssertNotEqual(try MTPLXRuntimeBootstrapper.wheelFingerprint(of: fixture.wheel), first)
    }

    func testPreflightMovesPortAwayFromForeignOccupantAndPersists() async throws {
        let occupiedPort = try freeTCPPort()
        let garbage = try startGarbageHTTPServer(port: occupiedPort)
        defer { garbage.terminate() }
        _ = try await waitForNonFreeClassification(port: occupiedPort)
        let settingsURL = temporaryDirectory().appendingPathComponent("settings.json")

        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: occupiedPort),
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL)
        )
        let (port, notice) = await backend.preflightOutcomeForTest(
            target: nil,
            launchID: "test-launch"
        )

        XCTAssertNotEqual(port, occupiedPort)
        let fallbackNotice = try XCTUnwrap(notice)
        XCTAssertTrue(fallbackNotice.contains("another app"), fallbackNotice)
        XCTAssertTrue(fallbackNotice.contains("\(occupiedPort)"), fallbackNotice)
        XCTAssertTrue(fallbackNotice.contains("\(port)"), fallbackNotice)
        let persisted = try MTPLXSettingsStore(settingsURL: settingsURL).load()
        XCTAssertEqual(persisted.port, port)
    }

    func testPreflightMovesPortAwayFromExternalMTPLXServer() async throws {
        let occupiedPort = try freeTCPPort()
        let cliHealthJSON = Self.healthJSON.replacingOccurrences(
            of: "\"launch_id\": \"fixture-launch\"",
            with: "\"launch_id\": null"
        )
        let server = try await startFixtureHealthServer(
            port: occupiedPort,
            healthJSON: cliHealthJSON
        )
        defer { server.terminate() }
        let settingsURL = temporaryDirectory().appendingPathComponent("settings.json")

        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(port: occupiedPort),
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL)
        )
        let (port, notice) = await backend.preflightOutcomeForTest(
            target: nil,
            launchID: "test-launch"
        )

        XCTAssertNotEqual(port, occupiedPort)
        let fallbackNotice = try XCTUnwrap(notice)
        XCTAssertTrue(
            fallbackNotice.contains("an MTPLX server started outside the app"),
            fallbackNotice
        )
        let persisted = try MTPLXSettingsStore(settingsURL: settingsURL).load()
        XCTAssertEqual(persisted.port, port)
    }

    func testPreflightLeavesAdoptableAppOwnedDaemonAlone() async throws {
        let occupiedPort = try freeTCPPort()
        let server = try await startFixtureHealthServer(
            port: occupiedPort,
            healthJSON: Self.healthJSON
        )
        defer { server.terminate() }
        let fake = try makeExecutable(named: "mtplx")
        let settingsURL = temporaryDirectory().appendingPathComponent("settings.json")

        let backend = await MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: "/models/test",
                port: occupiedPort
            ),
            settingsStore: MTPLXSettingsStore(settingsURL: settingsURL)
        )
        let (port, notice) = await backend.preflightOutcomeForTest(
            target: nil,
            launchID: "new-launch"
        )

        XCTAssertEqual(port, occupiedPort)
        XCTAssertNil(notice)
    }

    func testHumanizedStartFailureExplainsPortOccupants() {
        let appOwned = MTPLXBackendStore.humanizedStartFailure(
            DaemonSupervisorError.portOccupied(pid: 1, launchID: "launch-1"),
            port: 8000
        )
        XCTAssertTrue(appOwned.contains("another MTPLX server"), appOwned)
        XCTAssertTrue(appOwned.contains("mtplx stop --port 8000"), appOwned)

        let external = MTPLXBackendStore.humanizedStartFailure(
            DaemonSupervisorError.portOccupied(pid: nil, launchID: nil),
            port: 8010
        )
        XCTAssertTrue(external.contains("started outside the app"), external)
        XCTAssertTrue(external.contains("Ctrl-C"), external)
        XCTAssertTrue(external.contains("mtplx stop --port 8010"), external)

        let other = MTPLXBackendStore.humanizedStartFailure(
            DaemonSupervisorError.healthTimeout,
            port: 8000
        )
        XCTAssertEqual(other, "MTPLX took too long to start up.")
    }

    func testDaemonSupervisorRejectsWrongLaunchID() async throws {
        let port = try freeTCPPort()
        let script = try makeHTTPFixtureScript(
            port: port,
            healthJSON: Self.healthJSON,
            snapshotJSON: Self.snapshotJSON
        )
        let supervisor = DaemonSupervisor(logStore: BoundedLogStore())

        do {
            _ = try await supervisor.start(
                command: DaemonCommand(executableURL: script, arguments: []),
                healthBaseURL: URL(string: "http://127.0.0.1:\(port)")!,
                probeHealth: true,
                timeoutSeconds: 5,
                expectedLaunchID: "expected-launch"
            )
            XCTFail("expected launchIdentityMismatch")
        } catch DaemonSupervisorError.launchIdentityMismatch(let expected, let observed) {
            XCTAssertEqual(expected, "expected-launch")
            XCTAssertEqual(observed, "fixture-launch")
        }

        await supervisor.stop(graceSeconds: 0.1)
    }

    func testBoundedLogStoreDropsOldEntries() async {
        let logs = BoundedLogStore(capacity: 2)
        await logs.append("one", stream: .stdout)
        await logs.append("two", stream: .stdout)
        await logs.append("three", stream: .stderr)
        let snapshot = await logs.snapshot()
        XCTAssertEqual(snapshot.map(\.message), ["two", "three"])
    }

    func testModelDownloaderIgnoresBriefStructuredStallEvents() async throws {
        let root = temporaryDirectory()
        let cacheLog = root.appendingPathComponent("pycache-prefix.log")
        let script = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            printf '%s' "$PYTHONPYCACHEPREFIX" > "$MTPLX_FAKE_LOG"
            cat <<'JSON'
            {"event":"start","path":"/tmp/model","size_bytes":0,"total_bytes":100}
            {"event":"progress","path":"/tmp/model","size_bytes":11,"total_bytes":100,"rate_bps":0,"stalled_s":1}
            {"event":"progress","path":"/tmp/model","size_bytes":11,"total_bytes":100,"rate_bps":0,"stalled_s":29}
            {"event":"complete","path":"/tmp/model","size_bytes":100,"total_bytes":100}
            JSON
            """
        )
        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "MTPLX_FAKE_LOG": cacheLog.path,
            ],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true),
            executableOverride: script
        )
        var progressCount = 0
        var stalledCount = 0

        for await event in downloader.stream(
            repo: "Example/Quality",
            totalBytes: 100,
            extraEnvironment: [
                "PYTHONPYCACHEPREFIX": "/Applications/MTPLX.app/Contents/Resources/cache"
            ]
        ) {
            switch event {
            case .progress:
                progressCount += 1
            case .stalled:
                stalledCount += 1
            default:
                continue
            }
        }

        XCTAssertEqual(progressCount, 2)
        XCTAssertEqual(stalledCount, 0)
        XCTAssertEqual(
            try String(contentsOf: cacheLog, encoding: .utf8),
            root.appendingPathComponent("Library/Caches/MTPLX/PythonBytecode").path
        )
    }

    func testModelDownloaderBootstrapsRuntimeWithHomebrewWhenMtplxIsMissing() async throws {
        let root = temporaryDirectory()
        let fakeBin = root.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let fakeBrew = fakeBin.appendingPathComponent("brew")
        let brewLog = root.appendingPathComponent("brew.log")
        try """
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        case "$1" in
          update)
            exit 0
            ;;
          install)
            cat > "$MTPLX_FAKE_BIN/mtplx" <<'SCRIPT'
        #!/bin/sh
        cat <<'JSON'
        {"event":"complete","path":"/tmp/model","size_bytes":100,"total_bytes":100}
        JSON
        SCRIPT
            chmod +x "$MTPLX_FAKE_BIN/mtplx"
            exit 0
            ;;
          upgrade|link)
            exit 0
            ;;
        esac
        exit 0
        """.data(using: .utf8)!.write(to: fakeBrew)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeBrew.path)

        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
                "MTPLX_APP_HOMEBREW_PATH": fakeBrew.path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                "MTPLX_FAKE_BIN": fakeBin.path,
                "MTPLX_FAKE_LOG": brewLog.path,
            ],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true)
        )
        var statuses: [String] = []
        var completed = false
        var failure: String?

        for await event in downloader.stream(repo: "Example/Quality", totalBytes: 100) {
            switch event {
            case .status(let message, _, _, _):
                statuses.append(message)
            case .complete:
                completed = true
            case .failed(_, let stderrTail):
                failure = stderrTail
            default:
                continue
            }
        }

        XCTAssertTrue(statuses.contains("Installing MTPLX runtime"))
        XCTAssertTrue(statuses.contains("MTPLX runtime ready"))
        XCTAssertNil(failure)
        XCTAssertTrue(completed, "statuses=\(statuses), failure=\(failure ?? "nil")")
        let log = try String(contentsOf: brewLog, encoding: .utf8)
        XCTAssertTrue(log.contains("update"))
        XCTAssertTrue(log.contains("install youssofal/mtplx/mtplx"))
        XCTAssertTrue(log.contains("upgrade youssofal/mtplx/mtplx"))
        XCTAssertTrue(log.contains("link --overwrite mtplx"))
    }

    func testModelDownloaderRepairsStaleHomebrewLinkWhenRuntimeIsStillMissing() async throws {
        let root = temporaryDirectory()
        let fakeBin = root.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let fakeBrew = fakeBin.appendingPathComponent("brew")
        let brewLog = root.appendingPathComponent("brew.log")
        let staleLinkState = root.appendingPathComponent("stale-link-state")
        try Data().write(to: staleLinkState)
        try """
        #!/bin/sh
        echo "$*" >> "$MTPLX_FAKE_LOG"
        case "$1" in
          update|install|upgrade)
            exit 0
            ;;
          unlink)
            rm -f "$MTPLX_FAKE_STALE_LINK"
            exit 0
            ;;
          link)
            if [ -f "$MTPLX_FAKE_STALE_LINK" ]; then
              echo "Warning: Already linked: /opt/homebrew/Cellar/mtplx/0.3.7" >&2
              exit 0
            fi
            cat > "$MTPLX_FAKE_BIN/mtplx" <<'SCRIPT'
        #!/bin/sh
        cat <<'JSON'
        {"event":"complete","path":"/tmp/model","size_bytes":100,"total_bytes":100}
        JSON
        SCRIPT
            chmod +x "$MTPLX_FAKE_BIN/mtplx"
            exit 0
            ;;
        esac
        exit 0
        """.data(using: .utf8)!.write(to: fakeBrew)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeBrew.path)

        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
                "MTPLX_APP_HOMEBREW_PATH": fakeBrew.path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                "MTPLX_FAKE_BIN": fakeBin.path,
                "MTPLX_FAKE_LOG": brewLog.path,
                "MTPLX_FAKE_STALE_LINK": staleLinkState.path,
            ],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true)
        )
        var completed = false
        var failure: String?

        for await event in downloader.stream(repo: "Example/Quality", totalBytes: 100) {
            switch event {
            case .complete:
                completed = true
            case .failed(_, let stderrTail):
                failure = stderrTail
            default:
                continue
            }
        }

        XCTAssertNil(failure)
        XCTAssertTrue(completed)
        let log = try String(contentsOf: brewLog, encoding: .utf8)
        XCTAssertTrue(log.contains("link --overwrite mtplx"))
        XCTAssertTrue(log.contains("unlink mtplx"))
    }

    func testModelDownloaderReportsMissingHomebrewWhenRuntimeCannotBootstrap() async throws {
        let root = temporaryDirectory()
        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "PATH": "\(root.appendingPathComponent("bin", isDirectory: true).path):/usr/bin:/bin",
                "MTPLX_APP_HOMEBREW_PATH": root.appendingPathComponent("missing-brew").path,
                "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
            ],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true)
        )
        var failure: String?

        for await event in downloader.stream(repo: "Example/Quality", totalBytes: 100) {
            if case .failed(_, let stderrTail) = event {
                failure = stderrTail
            }
        }

        XCTAssertEqual(
            failure,
            "Homebrew was not found, so MTPLX could not install its command-line runtime automatically. Install Homebrew from brew.sh, then press Retry."
        )
    }

    func testModelDownloaderSurfacesProgressJSONFailureDetail() async throws {
        let root = temporaryDirectory()
        let script = try makeExecutable(
            named: "mtplx",
            body: """
            #!/bin/sh
            cat <<'JSON'
            {"event":"failed","error":"pull_failed","detail":"cached model is incomplete: tokenizer.json"}
            JSON
            exit 1
            """
        )
        let downloader = ModelDownloader(
            processEnvironment: ["HOME": root.path],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true),
            executableOverride: script
        )
        var failure: String?

        for await event in downloader.stream(repo: "Example/Quality", totalBytes: 100) {
            if case .failed(_, let stderrTail) = event {
                failure = stderrTail
            }
        }

        XCTAssertEqual(failure, "cached model is incomplete: tokenizer.json")
    }

    private func makeExecutable(
        named name: String,
        body: String = "#!/bin/sh\necho ok\n"
    ) throws -> URL {
        let directory = temporaryDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(name)
        try body.data(using: .utf8)!.write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func makeCompleteModel(
        named name: String,
        archID: String? = "qwen3_next"
    ) throws -> URL {
        let model = temporaryDirectory().appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = archID.map { "{\"arch_id\":\"\($0)\"}" } ?? "{}"
        try config.write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("mtplx_runtime.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        return model
    }

    private func releaseManifest(
        minimumCLI: String,
        recommendedCLI: String
    ) -> MTPLXReleaseManifest {
        MTPLXReleaseManifest(
            appVersion: "1.0.0",
            appBuild: "10000",
            minimumCLIVersion: minimumCLI,
            recommendedCLIVersion: recommendedCLI,
            dmgURL: URL(string: "https://github.com/youssofal/mtplx/releases/download/v1.0.0/MTPLX-1.0.0.dmg")!,
            dmgSHA256: "sha",
            pypiVersion: "1.0.0",
            homebrewFormulaVersion: "1.0.0",
            releaseNotesURL: URL(string: "https://mtplx.com/releases/notes/v1.0.0.html")!,
            publishedAt: nil
        )
    }

    private func sourceTreeWrapper() -> URL? {
        var cursor = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            cursor.deleteLastPathComponent()
        }
        let candidate = cursor.appendingPathComponent("bin").appendingPathComponent("mtplx")
        return FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
    }

    private func chatRequestPayload(at url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let payload = try JSONSerialization.jsonObject(with: data)
        return try XCTUnwrap(payload as? [String: Any])
    }

    private func chatMessages(in payload: [String: Any]) -> [[String: Any]] {
        payload["messages"] as? [[String: Any]] ?? []
    }

    private func chatRoles(in payload: [String: Any]) -> [String] {
        chatMessages(in: payload).compactMap { $0["role"] as? String }
    }

    private struct FixtureWebTransport: WebTransport {
        let body: String

        func data(for request: URLRequest) async throws -> (Data, URLResponse) {
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "https://example.com")!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "text/html"]
            )!
            return (Data(body.utf8), response)
        }
    }

    private func makeHTTPFixtureScript(
        port: Int,
        healthJSON: String,
        snapshotJSON: String
    ) throws -> URL {
        try makeExecutable(
            named: "fake-mtplx-http",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            PORT = \(port)
            HEALTH = json.loads(r'''\(healthJSON)''')
            SNAPSHOT = json.loads(r'''\(snapshotJSON)''')

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

                def do_GET(self):
                    if self.path == "/health":
                        self._json(HEALTH)
                    elif self.path == "/v1/mtplx/thermal/status":
                        self._json({
                            "ok": True,
                            "current_mode": "max",
                            "detection": {"available": True},
                            "fan_summary": {"ok": True},
                        })
                    elif self.path == "/v1/mtplx/app/capabilities":
                        self._json({
                            "ok": True,
                            "name": "MTPLX test daemon",
                            "api_version": 1,
                            "endpoints": {},
                            "mutable_settings": [],
                            "restart_required_settings": [],
                            "snapshot_interval": {
                                "default_ms": 500,
                                "min_ms": 250,
                                "max_ms": 5000,
                                "native_default_ms": 500,
                                "performance_lock_ms": 1000,
                            },
                            "features": {},
                            "scheduler": {},
                        })
                    elif self.path == "/admin/sessions":
                        self._json({"sessions": [], "count": 0})
                    elif self.path == "/v1/mtplx/prefill_history":
                        self._json({"capacity": 0, "history": []})
                    elif self.path == "/v1/models":
                        self._json({
                            "object": "list",
                            "data": [{
                                "id": "mtplx-test-model",
                                "object": "model",
                                "owned_by": "mtplx",
                            }],
                        })
                    elif self.path.startswith("/v1/mtplx/metrics/stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        self.wfile.write(
                            ("event: snapshot\\n"
                             f"data: {json.dumps(SNAPSHOT)}\\n\\n").encode("utf-8")
                        )
                        self.wfile.flush()
                        time.sleep(1)
                    else:
                        self.send_response(404)
                        self.end_headers()

                def do_POST(self):
                    if self.path == "/v1/mtplx/thermal/fan_mode":
                        length = int(self.headers.get("Content-Length", "0") or "0")
                        raw = self.rfile.read(length) if length else b"{}"
                        try:
                            request = json.loads(raw.decode("utf-8") or "{}")
                        except Exception:
                            request = {}
                        mode = request.get("mode") or "max"
                        self._json({
                            "verified": True,
                            "current_mode": mode,
                            "result": {"ok": True},
                            "fan_summary": {"ok": True},
                        })
                    else:
                        self.send_response(404)
                        self.end_headers()

            ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
            PY
            """
        )
    }

    private func makeMinimalDocx(at url: URL, text: String) throws {
        let root = temporaryDirectory()
        let word = root.appendingPathComponent("word", isDirectory: true)
        try FileManager.default.createDirectory(at: word, withIntermediateDirectories: true)
        let xml = """
        <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>\(text)</w:t></w:r></w:p>
          </w:body>
        </w:document>
        """
        try xml.write(
            to: word.appendingPathComponent("document.xml"),
            atomically: true,
            encoding: .utf8
        )
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.currentDirectoryURL = root
        process.arguments = ["-q", "-r", url.path, "word"]
        try process.run()
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            throw NSError(domain: "MTPLXAppCoreTests", code: Int(process.terminationStatus))
        }
    }

    private func makeTextPDF(at url: URL, text: String) throws {
        let data = NSMutableData()
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(data: data),
            let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else {
            throw NSError(domain: "MTPLXAppCoreTests", code: 2)
        }

        context.beginPDFPage(nil)
        let font = CTFontCreateWithName("Helvetica" as CFString, 18, nil)
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(
                string: text,
                attributes: [.font: font]
            )
        )
        context.textPosition = CGPoint(x: 72, y: 720)
        CTLineDraw(line, context)
        context.endPDFPage()
        context.closePDF()
        try data.write(to: url)
    }

    private func startFixtureHealthServer(
        port: Int,
        healthJSON: String
    ) async throws -> Process {
        let script = try makeHTTPFixtureScript(
            port: port,
            healthJSON: healthJSON,
            snapshotJSON: Self.snapshotJSON
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        let client = MTPLXAPIClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!)
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let health = try? await client.health(), health.ok {
                return process
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        process.terminate()
        throw DaemonSupervisorError.healthTimeout
    }

    /// HTTP listener that answers every request with HTML — a stand-in for
    /// a foreign (non-MTPLX) app squatting on the daemon port.
    private func startGarbageHTTPServer(port: Int) throws -> Process {
        let script = try makeExecutable(
            named: "fake-garbage-http",
            body: """
            #!/bin/sh
            exec python3 -u - <<'PY'
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def do_GET(self):
                    body = b"<html>definitely not mtplx</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            ThreadingHTTPServer(("127.0.0.1", \(port)), Handler).serve_forever()
            PY
            """
        )
        let process = Process()
        process.executableURL = script
        try process.run()
        return process
    }

    private func waitForNonFreeClassification(port: Int) async throws -> PortOccupantKind {
        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            let kind = await PortPreflight.classify(baseURL: baseURL, apiKey: nil)
            if kind != .free {
                return kind
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        throw DaemonSupervisorError.healthTimeout
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-app-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
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
            Darwin.close(socketFD)
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }

        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var boundAddress = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &boundAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.getsockname(socketFD, $0, &length)
            }
        }
        Darwin.close(socketFD)
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: boundAddress.sin_port))
    }

    private func waitForPIDFile(_ url: URL) async throws -> pid_t {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let raw = try? String(contentsOf: url, encoding: .utf8),
               let pid = pid_t(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
               pid > 1 {
                return pid
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        throw DaemonSupervisorError.healthTimeout
    }

    private func pidIsAlive(_ pid: pid_t) -> Bool {
        if kill(pid, 0) != 0 {
            return errno != ESRCH
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/ps")
        process.arguments = ["-p", String(pid), "-o", "stat="]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return true
        }
        guard process.terminationStatus == 0 else { return false }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let state = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        return !state.contains("Z")
    }

    private static func sseBlock(event: String, json: String) -> String {
        let data = json
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { "data: \($0)" }
            .joined(separator: "\n")
        return "event: \(event)\n\(data)"
    }
}

@MainActor
private final class LaunchTargetRecorder {
    var target: LaunchTarget?
}

private extension MTPLXAppCoreTests {
    static let healthJSON = """
    {
      "ok": true,
      "model": "mtplx-test-model",
      "model_path": "/models/test",
      "generation_mode": "mtp",
      "load_mtp": true,
      "mtp_enabled": true,
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
        "launch_id": "fixture-launch",
        "pid": 12345,
        "started_at": 1.0,
        "model_id": "mtplx-test-model",
        "warmup": {"ok": true}
      },
      "thermal": {
        "max_requested": true,
        "max_verified": true,
        "actual_ramp_verified": true,
        "fan_summary": {"ok": true},
        "verified_at": "1.0",
        "verified": {"ok": true}
      }
    }
    """

    static let capabilitiesJSON = """
    {
      "ok": true,
      "name": "MTPLX App Backend",
      "api_version": 1,
      "endpoints": {
        "snapshot": "/v1/mtplx/snapshot",
        "metrics_stream": "/v1/mtplx/metrics/stream"
      },
      "mutable_settings": ["depth", "temperature"],
      "restart_required_settings": ["model", "profile"],
      "snapshot_interval": {
        "default_ms": 200,
        "min_ms": 100,
        "max_ms": 5000,
        "native_default_ms": 500,
        "performance_lock_ms": 1000
      },
      "features": {
        "sse_metrics": true,
        "request_cancel": true,
        "cache_clear": true,
        "ssd_session_cache": true,
        "ssd_cache_archive": true,
        "session_clear": true,
        "prefill_history": true,
        "thermal_polling": true,
        "dashboard_static_bundle": true,
        "startup_ownership": true,
        "strict_max_fan_startup": true,
        "thermal_actual_ramp_verification": true
      }
    }
    """

    static let snapshotJSON = """
    {
      "ts": 1.0,
      "model_id": "mtplx-test-model",
      "profile": {"name": "sustained"},
      "context_window": 4096,
      "active_requests": 0,
      "in_flight": [],
      "latest": {"decode_tok_s": 55.0, "session_id": "s1"},
      "recent": [],
      "rolling": {
        "window_s": 300.0,
        "count": 1,
        "min": 55.0,
        "max": 55.0,
        "mean": 55.0,
        "p50": 55.0,
        "p95": 55.0,
        "history": [{"t": 1.0, "tok_s": 55.0, "session_id": "s1"}],
        "live_history": [],
        "sticky_all_time_max": 55.0
      },
      "lifetime": {
        "started_at_s": 1.0,
        "uptime_s": 2.0,
        "prompt_tokens_total": 10,
        "completion_tokens_total": 20,
        "cached_tokens_total": 5,
        "tokens_total": 30,
        "requests_total": 1,
        "cancelled_total": 0
      },
      "sessions": {"sessions": [], "count": 0, "session_bank": {"prefixes": []}},
      "session_bank": {"prefixes": []},
      "mem": {"ok": true},
      "thermal": null,
      "thermal_when_s": 0.0,
      "settings": {"depth": 3, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
      "machine": {"chip": "Apple M5 Max", "machine_model": "Mac16,1", "unified_memory_bytes": 137438953472},
      "uptime_s": 2.0
    }
    """
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { value } else { nil }
    }
}

private extension Dictionary where Key == String, Value == JSONValue {
    func recursivelyContainsKey(_ wanted: String) -> Bool {
        for (key, value) in self {
            if key == wanted {
                return true
            }
            switch value {
            case .object(let object):
                if object.recursivelyContainsKey(wanted) {
                    return true
                }
            case .array(let array):
                if array.contains(where: { $0.recursivelyContainsKey(wanted) }) {
                    return true
                }
            default:
                continue
            }
        }
        return false
    }
}

private extension JSONValue {
    func recursivelyContainsKey(_ wanted: String) -> Bool {
        switch self {
        case .object(let object):
            return object.recursivelyContainsKey(wanted)
        case .array(let array):
            return array.contains(where: { $0.recursivelyContainsKey(wanted) })
        default:
            return false
        }
    }
}

private extension Array where Element == String {
    func containsInOrder(_ needle: [String]) -> Bool {
        guard !needle.isEmpty, needle.count <= count else { return false }
        return indices.contains { start in
            let end = start + needle.count
            guard end <= count else { return false }
            return Array(self[start..<end]) == needle
        }
    }
}
