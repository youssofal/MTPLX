import XCTest
@testable import MTPLXAppCore

final class HermesEmbeddedRuntimeTests: XCTestCase {
    private var root: URL!
    private var hermesHome: URL!
    private var sidecarRuntimeDirectory: URL!
    private var integration: HermesIntegration!
    private var configuration: MTPLXAppConfiguration!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("HermesEmbeddedRuntimeTests-\(UUID().uuidString)", isDirectory: true)
        hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        sidecarRuntimeDirectory = root.appendingPathComponent("sidecars", isDirectory: true)
        try FileManager.default.createDirectory(at: hermesHome, withIntermediateDirectories: true)
        integration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: "/usr/bin/true",
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "HERMES_HOME": "/inherited/hermes-home",
            ],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )
        configuration = MTPLXAppConfiguration(
            model: "/models/current-model",
            host: "127.0.0.1",
            port: 18080,
            apiKey: "test-api-key"
        )
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testNamedProfileLaunchUsesIsolatedServeAndProcessLocalMTPLXRoute() throws {
        let tempProfile = try makeProfile(named: "bernd", config: mtplxConfig())
        let profile = HermesProfile(name: "bernd", path: tempProfile.path, isDefault: false)

        let spec = try integration.serveLaunchSpec(
            profile: profile,
            configuration: configuration,
            token: "test-session-token",
            launchID: "0123456789abcdef",
            parentPID: 4242
        )

        XCTAssertEqual(spec.arguments, [
            "-p", "bernd", "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", "0123456789abcdef",
        ])
        XCTAssertEqual(spec.environment["HERMES_INFERENCE_PROVIDER"], "custom")
        XCTAssertEqual(spec.environment["CUSTOM_BASE_URL"], "http://127.0.0.1:18080/v1")
        XCTAssertEqual(spec.environment["HERMES_INFERENCE_MODEL"], "current-model")
        XCTAssertEqual(spec.environment["HERMES_DASHBOARD_SESSION_TOKEN"], "test-session-token")
        XCTAssertEqual(spec.environment["MTPLX_HERMES_PARENT_PID"], "4242")
        XCTAssertNil(spec.environment["HERMES_HOME"])
    }

    func testDefaultProfileLaunchOmitsProfileFlag() throws {
        let spec = try integration.serveLaunchSpec(
            profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
            configuration: configuration,
            token: "test-session-token",
            launchID: "fedcba9876543210",
            parentPID: 4242
        )

        XCTAssertEqual(Array(spec.arguments.prefix(5)), ["serve", "--isolated", "--host", "127.0.0.1", "--port"])
        XCTAssertFalse(spec.arguments.contains("-p"))
    }

    func testActiveSessionRegistryMapsLiveOwnedAndExternalSessions() throws {
        let profile = try makeProfile(named: "bernd", config: mtplxConfig())
        let registry = profile
            .appendingPathComponent("runtime", isDirectory: true)
            .appendingPathComponent("active_sessions.json")
        try FileManager.default.createDirectory(at: registry.deletingLastPathComponent(), withIntermediateDirectories: true)
        try """
        {"entries":[
          {"lease_id":"lease-ours","session_id":"ours","surface":"mtplx-app","pid":7001,"process_start_time":10.0,"started_at":100.0},
          {"lease_id":"lease-telegram","session_id":"telegram","surface":"telegram","pid":7002,"process_start_time":20.0,"started_at":200.0}
        ]}
        """.write(to: registry, atomically: true, encoding: .utf8)

        let inspector = HermesActiveSessionRegistryInspector(
            processIdentity: { pid in
                switch pid {
                case 7001: return .live(startedAt: 10)
                case 7002: return .live(startedAt: 20)
                default: return .dead
                }
            }
        )

        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "ours", ownedSidecarPID: 7001), .ownedByMTPLX)
        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "telegram", ownedSidecarPID: 7001), .external(surface: "telegram"))
        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "idle", ownedSidecarPID: 7001), .ready)
    }

    func testActiveSessionRegistryRejectsLegacyStartTimeSchema() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        try """
        {"entries":[
          {"session_id":"saved","surface":"telegram","pid":7001,"start_time":10.0}
        ]}
        """.write(to: registry, atomically: true, encoding: .utf8)

        let inspector = HermesActiveSessionRegistryInspector(processIdentity: { _ in .live(startedAt: 10) })

        XCTAssertEqual(
            inspector.ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .unknown("Session activity could not be inspected.")
        )
    }

    func testActiveSessionRegistryFailsClosedForMalformedOrUninspectableEntries() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        try "{not json".write(to: registry, atomically: true, encoding: .utf8)
        let inspector = HermesActiveSessionRegistryInspector(processIdentity: { _ in .unknown })

        XCTAssertEqual(
            inspector.ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .unknown("Session activity could not be inspected.")
        )
        XCTAssertEqual(
            HermesActiveSessionRegistryInspector(
                processIdentity: { _ in .unknown },
                readData: { _ in throw CocoaError(.fileReadNoPermission) }
            ).ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .registryUnavailable("Session activity could not be inspected.")
        )
    }

    func testActiveSessionRegistryIgnoresDeadAndPIDReusedEntries() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        try """
        {"entries":[
          {"lease_id":"lease-dead","session_id":"dead","surface":"telegram","pid":7001,"process_start_time":10.0,"started_at":100.0},
          {"lease_id":"lease-reused","session_id":"reused","surface":"telegram","pid":7002,"process_start_time":10.0,"started_at":100.0}
        ]}
        """.write(to: registry, atomically: true, encoding: .utf8)
        let inspector = HermesActiveSessionRegistryInspector(
            processIdentity: { pid in pid == 7001 ? .dead : .live(startedAt: 20) }
        )

        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "dead", ownedSidecarPID: nil), .ready)
        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "reused", ownedSidecarPID: nil), .ready)
    }

    func testActiveSessionRegistryKeepsUnknownProcessIdentityFailClosed() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        try """
        {"entries":[
          {"lease_id":"lease-unknown","session_id":"saved","surface":"telegram","pid":7001,"process_start_time":10.0,"started_at":100.0}
        ]}
        """.write(to: registry, atomically: true, encoding: .utf8)

        let inspector = HermesActiveSessionRegistryInspector(processIdentity: { _ in .unknown })

        XCTAssertEqual(
            inspector.ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: 7001),
            .unknown("Session activity could not be inspected.")
        )
    }

    func testActiveSessionRegistryRequiresNativeStartTimePrecisionAndOneLiveEntry() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        try """
        {"entries":[
          {"lease_id":"lease-close","session_id":"close","surface":"telegram","pid":7001,"process_start_time":10.0,"started_at":100.0},
          {"lease_id":"lease-edge","session_id":"edge","surface":"telegram","pid":7002,"process_start_time":20.0,"started_at":100.0},
          {"lease_id":"lease-first","session_id":"conflict","surface":"telegram","pid":7003,"process_start_time":30.0,"started_at":100.0},
          {"lease_id":"lease-second","session_id":"conflict","surface":"desktop","pid":7004,"process_start_time":40.0,"started_at":100.0}
        ]}
        """.write(to: registry, atomically: true, encoding: .utf8)
        let inspector = HermesActiveSessionRegistryInspector(
            processIdentity: { pid in
                switch pid {
                case 7001: return .live(startedAt: 10.0009)
                case 7002: return .live(startedAt: 20.001)
                case 7003: return .live(startedAt: 30)
                case 7004: return .live(startedAt: 40)
                default: return .dead
                }
            }
        )

        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "close", ownedSidecarPID: nil), .external(surface: "telegram"))
        XCTAssertEqual(inspector.ownership(registryURL: registry, sessionID: "edge", ownedSidecarPID: nil), .ready)
        XCTAssertEqual(
            inspector.ownership(registryURL: registry, sessionID: "conflict", ownedSidecarPID: nil),
            .unknown("Session activity could not be inspected.")
        )
    }

    func testActiveSessionRegistryFailsClosedWhenRegistryCannotBeRead() throws {
        let registry = root.appendingPathComponent("active_sessions.json")
        let notFound = HermesActiveSessionRegistryInspector(
            processIdentity: { _ in .unknown },
            readData: { _ in throw CocoaError(.fileNoSuchFile) }
        )
        let unreadable = HermesActiveSessionRegistryInspector(
            processIdentity: { _ in .unknown },
            readData: { _ in throw CocoaError(.fileReadNoPermission) }
        )

        XCTAssertEqual(
            notFound.ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .registryUnavailable("Session activity could not be inspected.")
        )
        XCTAssertEqual(
            unreadable.ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .registryUnavailable("Session activity could not be inspected.")
        )

        try FileManager.default.createDirectory(at: registry, withIntermediateDirectories: true)
        XCTAssertEqual(
            HermesActiveSessionRegistryInspector(processIdentity: { _ in .unknown })
                .ownership(registryURL: registry, sessionID: "saved", ownedSidecarPID: nil),
            .registryUnavailable("Session activity could not be inspected.")
        )
    }

    func testIntegrationFailsClosedWhenProfileRegistryIsMissing() throws {
        let profile = try makeProfile(named: "bernd", config: mtplxConfig())
        let integration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: "/usr/bin/true",
            environment: [:],
            activeSessionRegistryInspector: HermesActiveSessionRegistryInspector(processIdentity: { _ in .unknown })
        )

        XCTAssertEqual(
            integration.sessionOwnership(
                profile: HermesProfile(name: "bernd", path: profile.path, isDefault: false),
                sessionID: "saved",
                ownedSidecarPID: nil
            ),
            .registryUnavailable("Session activity could not be inspected.")
        )
    }

    func testEmbeddedLaunchDoesNotInheritRootMessagingCredentials() throws {
        try """
        TELEGRAM_BOT_TOKEN=root-telegram-token
        TELEGRAM_ALLOWED_USERS=123456
        DISCORD_BOT_TOKEN=root-discord-token
        SLACK_BOT_TOKEN=root-slack-token
        SIGNAL_ACCOUNT=root-signal-account
        """.write(
            to: hermesHome.appendingPathComponent(".env"),
            atomically: true,
            encoding: .utf8
        )

        let credentialedIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: "/usr/bin/true",
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "HERMES_HOME": "/inherited/hermes-home",
                "TELEGRAM_BOT_TOKEN": "root-telegram-token",
                "TELEGRAM_ALLOWED_USERS": "123456",
                "DISCORD_BOT_TOKEN": "root-discord-token",
                "SLACK_BOT_TOKEN": "root-slack-token",
                "SIGNAL_ACCOUNT": "root-signal-account",
            ]
        )

        let spec = try credentialedIntegration.serveLaunchSpec(
            profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
            configuration: configuration,
            token: "test-session-token",
            launchID: "fedcba9876543210",
            parentPID: 4242
        )

        for key in [
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "DISCORD_BOT_TOKEN",
            "SLACK_BOT_TOKEN", "SIGNAL_ACCOUNT",
        ] {
            XCTAssertNil(spec.environment[key], "Embedded child must not receive \(key)")
        }
    }

    func testServeLaunchSpecRejectsNonHexOrWrongLengthLaunchID() {
        XCTAssertThrowsError(
            try integration.serveLaunchSpec(
                profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
                configuration: configuration,
                token: "test-session-token",
                launchID: "not-a-safe-launch-id",
                parentPID: 4242
            )
        )
    }

    func testProfileRoutingClassifiesMTPLXExternalAndUnavailableIndependently() throws {
        let mtplxProfile = HermesProfile(
            name: "mtplx",
            path: try makeProfile(named: "mtplx", config: mtplxConfig()).path,
            isDefault: false
        )
        let externalProfile = HermesProfile(
            name: "external",
            path: try makeProfile(
                named: "external",
                config: "model:\n  default: external-model\n  provider: anthropic\n  base_url: https://api.example.test/v1\n"
            ).path,
            isDefault: false
        )
        let unreadableProfile = HermesProfile(
            name: "unavailable",
            path: try makeProfile(named: "unavailable", config: "not: [valid").path,
            isDefault: false
        )

        XCTAssertEqual(integration.routingState(for: mtplxProfile, configuration: configuration), .mtplx)
        XCTAssertEqual(integration.routingState(for: externalProfile, configuration: configuration), .external)
        guard case .unavailable = integration.routingState(for: unreadableProfile, configuration: configuration) else {
            return XCTFail("Unreadable profile must remain visible as unavailable")
        }
    }

    func testProfileRoutingRejectsDuplicateModelBlocksAndRoutingKeys() throws {
        let duplicateModel = HermesProfile(
            name: "duplicate-model",
            path: try makeProfile(
                named: "duplicate-model",
                config: mtplxConfig() + "\n" + mtplxConfig()
            ).path,
            isDefault: false
        )
        let duplicateBaseURL = HermesProfile(
            name: "duplicate-base-url",
            path: try makeProfile(
                named: "duplicate-base-url",
                config: "model:\n  default: current-model\n  provider: custom\n",
                env: "CUSTOM_BASE_URL=http://127.0.0.1:18080/v1\nCUSTOM_BASE_URL=http://127.0.0.1:18080/v1\n"
            ).path,
            isDefault: false
        )

        for profile in [duplicateModel, duplicateBaseURL] {
            guard case .unavailable = integration.routingState(for: profile, configuration: configuration) else {
                return XCTFail("Duplicate routing configuration must fail closed")
            }
        }
    }

    func testReadyParserAcceptsHeadlessSentinelOnly() {
        XCTAssertEqual(HermesBackendReadyParser.port(from: "HERMES_BACKEND_READY port=45123"), 45123)
        XCTAssertNil(HermesBackendReadyParser.port(from: "Hermes backend listening on 0.0.0.0:45123"))
        XCTAssertNil(HermesBackendReadyParser.port(from: "HERMES_BACKEND_READY port=0"))
        XCTAssertNil(HermesBackendReadyParser.port(from: "HERMES_BACKEND_READY port=45123 extra"))
    }

    func testSidecarUsesCallerTokenAndRemovesOwnershipRecordOnStopWithoutChangingProfileFiles() async throws {
        let profileURL = try makeProfile(
            named: "fixture",
            config: mtplxConfig(),
            env: "EXTERNAL_PROVIDER_TOKEN=keep-this-byte-identical\n"
        )
        let configURL = profileURL.appendingPathComponent("config.yaml")
        let envURL = profileURL.appendingPathComponent(".env")
        let configBefore = try Data(contentsOf: configURL)
        let envBefore = try Data(contentsOf: envURL)
        let environmentCaptureURL = root.appendingPathComponent("fixture-environment.txt")
        let fixture = try makeSidecarFixture(environmentCaptureURL: environmentCaptureURL)
        let fixtureIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: fixture.path,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "MTPLX_FIXTURE_ENV_FILE": environmentCaptureURL.path,
            ],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )
        let profile = HermesProfile(name: "fixture", path: profileURL.path, isDefault: false)

        let sidecar = try await fixtureIntegration.startEmbeddedSidecar(
            profile: profile,
            configuration: configuration
        )

        XCTAssertEqual(sidecar.webSocketURL.host, "127.0.0.1")
        XCTAssertEqual(sidecar.webSocketURL.path, "/api/ws")
        let token = URLComponents(url: sidecar.webSocketURL, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first(where: { $0.name == "token" })?
            .value
        XCTAssertNotNil(token)
        XCTAssertEqual(token?.count, 43)
        XCTAssertTrue(FileManager.default.fileExists(atPath: sidecar.ownershipRecordURL.path))
        XCTAssertEqual(try Data(contentsOf: configURL), configBefore)
        XCTAssertEqual(try Data(contentsOf: envURL), envBefore)
        XCTAssertEqual(
            try String(contentsOf: environmentCaptureURL, encoding: .utf8),
            "HERMES_DASHBOARD_SESSION_TOKEN=present\nOPENAI_API_KEY=present\n"
        )

        sidecar.stop()

        XCTAssertFalse(FileManager.default.fileExists(atPath: sidecar.ownershipRecordURL.path))
        XCTAssertFalse(sidecar.isRunning)
        XCTAssertEqual(try Data(contentsOf: configURL), configBefore)
        XCTAssertEqual(try Data(contentsOf: envURL), envBefore)
    }

    func testExecWrapperRecordsVerifiedPostLaunchIdentity() async throws {
        let environmentCaptureURL = root.appendingPathComponent("wrapper-environment.txt")
        let fixture = try makeExecWrapperFixture(environmentCaptureURL: environmentCaptureURL)
        let wrapperIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: fixture.wrapper.path,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "MTPLX_FIXTURE_ENV_FILE": environmentCaptureURL.path,
            ],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )

        let sidecar = try await wrapperIntegration.startEmbeddedSidecar(
            profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
            configuration: configuration
        )
        defer { sidecar.stop() }
        let record = try JSONDecoder().decode(
            HermesSidecarOwnershipRecord.self,
            from: Data(contentsOf: sidecar.ownershipRecordURL)
        )

        XCTAssertEqual(record.executablePath, fixture.binary.standardizedFileURL.resolvingSymlinksInPath().path)
        XCTAssertEqual(record.argv0, fixture.binary.path)
        XCTAssertEqual(Array(record.arguments.suffix(6)), ["--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", record.launchID])
    }

    func testConsoleScriptPrefixStartsAndRecordsFullVerifiedPostLaunchIdentity() async throws {
        let fixture = try makeConsoleScriptLauncherFixture()
        let consoleIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: fixture.launcher.path,
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )

        let sidecar = try await consoleIntegration.startEmbeddedSidecar(
            profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
            configuration: configuration
        )
        defer { sidecar.stop() }
        let record = try JSONDecoder().decode(
            HermesSidecarOwnershipRecord.self,
            from: Data(contentsOf: sidecar.ownershipRecordURL)
        )

        XCTAssertEqual(record.executablePath, fixture.interpreter.standardizedFileURL.resolvingSymlinksInPath().path)
        XCTAssertEqual(record.argv0, fixture.interpreter.path)
        XCTAssertEqual(record.arguments.first, fixture.consoleScript.standardizedFileURL.resolvingSymlinksInPath().path)
        XCTAssertEqual(Array(record.arguments.dropFirst()), [
            "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", record.launchID,
        ])
    }

    func testConsoleEntrypointPrefixRejectsUnexpectedArgumentsAndUnsafePaths() throws {
        let fixture = try makeConsoleScriptLauncherFixture()
        let launchID = "1111111111111111"
        let canonical = [
            "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", launchID,
        ]
        let valid = [fixture.consoleScript.standardizedFileURL.resolvingSymlinksInPath().path] + canonical

        XCTAssertTrue(HermesOrphanSidecarScanner.hasVerifiedArguments(
            valid,
            profileName: "default",
            launchID: launchID
        ))
        XCTAssertFalse(HermesOrphanSidecarScanner.hasVerifiedArguments(
            [fixture.consoleScript.path, "--unexpected"] + canonical,
            profileName: "default",
            launchID: launchID
        ))
        XCTAssertFalse(HermesOrphanSidecarScanner.hasVerifiedArguments(
            [fixture.consoleScript.path] + canonical + ["--unexpected"],
            profileName: "default",
            launchID: launchID
        ))
        let wrongBasename = fixture.consoleScript.deletingLastPathComponent().appendingPathComponent("not-hermes")
        try FileManager.default.copyItem(at: fixture.consoleScript, to: wrongBasename)
        XCTAssertFalse(HermesOrphanSidecarScanner.hasVerifiedArguments(
            [wrongBasename.path] + canonical,
            profileName: "default",
            launchID: launchID
        ))
        XCTAssertFalse(HermesOrphanSidecarScanner.hasVerifiedArguments(
            ["relative/hermes"] + canonical,
            profileName: "default",
            launchID: launchID
        ))

        let identity = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: fixture.interpreter.standardizedFileURL.resolvingSymlinksInPath().path,
            argv0: fixture.interpreter.path,
            arguments: valid
        )
        let spec = HermesServeLaunchSpec(
            executableURL: fixture.launcher,
            arguments: canonical,
            environment: [:],
            token: "test-token",
            launchID: launchID,
            parentPID: 8001
        )
        XCTAssertTrue(HermesOrphanSidecarScanner.isVerifiedPostLaunchIdentity(
            identity,
            spec: spec,
            profileName: "default",
            hermesHome: hermesHome
        ))
        let outsideEntrypoint = root.appendingPathComponent("outside/hermes")
        try FileManager.default.createDirectory(at: outsideEntrypoint.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixture.consoleScript, to: outsideEntrypoint)
        XCTAssertFalse(HermesOrphanSidecarScanner.isVerifiedPostLaunchIdentity(
            HermesSidecarProcessSnapshot(
                pid: 7101,
                executablePath: identity.executablePath,
                argv0: identity.argv0,
                arguments: [outsideEntrypoint.path] + canonical
            ),
            spec: spec,
            profileName: "default",
            hermesHome: hermesHome
        ))
    }

    func testConsoleScriptOrphanScannerRequiresExactRecordAndRejectsGenericNearMiss() throws {
        let fixture = try makeConsoleScriptLauncherFixture()
        let launchID = "1111111111111111"
        let arguments = [fixture.consoleScript.standardizedFileURL.resolvingSymlinksInPath().path,
            "serve", "--isolated", "--host", "127.0.0.1",
            "--port", "0", "--ssh-owner-nonce", launchID,
        ]
        let record = HermesSidecarOwnershipRecord(
            launchID: launchID,
            pid: 7101,
            parentPID: 8001,
            profileName: "default",
            createdAt: .now,
            executablePath: fixture.interpreter.standardizedFileURL.resolvingSymlinksInPath().path,
            argv0: fixture.interpreter.path,
            arguments: arguments
        )
        let exact = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: record.executablePath,
            argv0: record.argv0,
            arguments: record.arguments
        )
        let genericNearMiss = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: record.executablePath,
            argv0: record.argv0,
            arguments: Array(record.arguments.dropFirst())
        )

        XCTAssertEqual(HermesOrphanSidecarScanner.orphanPIDs(
            records: [record],
            processes: [exact],
            livePIDs: []
        ), [7101])
        XCTAssertEqual(HermesOrphanSidecarScanner.orphanPIDs(
            records: [record],
            processes: [genericNearMiss],
            livePIDs: []
        ), [])
    }

    func testStartupFailureRedactsTokenSplitAcrossStderrWrites() async throws {
        let tokenCaptureURL = root.appendingPathComponent("split-token.txt")
        let fixture = try makeSplitSecretFailureFixture(tokenCaptureURL: tokenCaptureURL)
        let failingIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: fixture.path,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "MTPLX_FIXTURE_TOKEN_FILE": tokenCaptureURL.path,
            ],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )

        do {
            _ = try await failingIntegration.startEmbeddedSidecar(
                profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
                configuration: configuration
            )
            XCTFail("The fixture exits before readiness and must fail startup")
        } catch {
            let diagnostic = error.localizedDescription
            let token = try String(contentsOf: tokenCaptureURL, encoding: .utf8)
            XCTAssertFalse(diagnostic.contains(token))
            XCTAssertFalse(diagnostic.contains(String(token.prefix(20))))
            XCTAssertFalse(diagnostic.contains(String(token.suffix(20))))
        }
    }

    func testStartupFailureRedactsShortSecretSplitAcrossStderrWrites() async throws {
        let secretCaptureURL = root.appendingPathComponent("short-secret.txt")
        let fixture = try makeShortSplitSecretFailureFixture(secretCaptureURL: secretCaptureURL)
        let shortSecretConfiguration = MTPLXAppConfiguration(
            model: "/models/current-model",
            host: "127.0.0.1",
            port: 18080,
            apiKey: "qq"
        )
        let failingIntegration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: fixture.path,
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "MTPLX_FIXTURE_SECRET_FILE": secretCaptureURL.path,
            ],
            sidecarRuntimeDirectory: sidecarRuntimeDirectory
        )

        do {
            _ = try await failingIntegration.startEmbeddedSidecar(
                profile: HermesProfile(name: "default", path: hermesHome.path, isDefault: true),
                configuration: shortSecretConfiguration
            )
            XCTFail("The fixture exits before readiness and must fail startup")
        } catch {
            let diagnostic = error.localizedDescription
            XCTAssertEqual(try String(contentsOf: secretCaptureURL, encoding: .utf8), "qq")
            XCTAssertFalse(diagnostic.contains("qq"))
            XCTAssertFalse(diagnostic.contains("q"))
        }
    }

    func testOrphanCleanupRequiresExactMarkerCommandAndDeadParent() {
        let records = [
            HermesSidecarOwnershipRecord(launchID: "1111111111111111", pid: 7101, parentPID: 8001, profileName: "one", createdAt: .now, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "one", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "1111111111111111"]),
            HermesSidecarOwnershipRecord(launchID: "2222222222222222", pid: 7102, parentPID: 8002, profileName: "two", createdAt: .now, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "two", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "2222222222222222"]),
            HermesSidecarOwnershipRecord(launchID: "3333333333333333", pid: 7103, parentPID: 8003, profileName: "three", createdAt: .now, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "three", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "3333333333333333"]),
            HermesSidecarOwnershipRecord(launchID: "4444444444444444", pid: 7104, parentPID: 9001, profileName: "four", createdAt: .now, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "four", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "4444444444444444"]),
        ]
        let processes = [
            HermesSidecarProcessSnapshot(pid: 7101, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "one", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "1111111111111111"]),
            HermesSidecarProcessSnapshot(pid: 7102, executablePath: "/usr/local/bin/hermes", arguments: ["serve", "--host", "127.0.0.1", "--port", "0"]),
            HermesSidecarProcessSnapshot(pid: 7103, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "three", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "wrong-marker"]),
            HermesSidecarProcessSnapshot(pid: 7104, executablePath: "/usr/local/bin/hermes", arguments: ["-p", "four", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "4444444444444444"]),
        ]

        let killed = HermesOrphanSidecarScanner.orphanPIDs(
            records: records,
            processes: processes,
            livePIDs: [9001]
        )

        XCTAssertEqual(killed, [7101])
        XCTAssertFalse(killed.contains(7102))
        XCTAssertFalse(killed.contains(7103))
        XCTAssertFalse(killed.contains(7104))
    }

    func testOrphanCleanupRejectsExecutableOrFullArgumentMismatch() {
        let arguments = ["-p", "one", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "1111111111111111"]
        let record = HermesSidecarOwnershipRecord(
            launchID: "1111111111111111",
            pid: 7101,
            parentPID: 8001,
            profileName: "one",
            createdAt: .now,
            executablePath: "/usr/local/bin/hermes",
            arguments: arguments
        )
        let wrongExecutable = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: "/usr/local/bin/not-hermes",
            arguments: arguments
        )
        let extraArgument = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: "/usr/local/bin/hermes",
            arguments: arguments + ["--unexpected"]
        )
        let wrongArgv0 = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: "/usr/local/bin/hermes",
            argv0: "/tmp/not-the-recorded-hermes",
            arguments: arguments
        )

        XCTAssertEqual(HermesOrphanSidecarScanner.orphanPIDs(records: [record], processes: [wrongExecutable], livePIDs: []), [])
        XCTAssertEqual(HermesOrphanSidecarScanner.orphanPIDs(records: [record], processes: [extraArgument], livePIDs: []), [])
        XCTAssertEqual(HermesOrphanSidecarScanner.orphanPIDs(records: [record], processes: [wrongArgv0], livePIDs: []), [])
    }

    func testOrphanCleanupFailsClosedForDuplicatePIDRecords() {
        let first = HermesSidecarOwnershipRecord(
            launchID: "1111111111111111",
            pid: 7101,
            parentPID: 8001,
            profileName: "one",
            createdAt: .now,
            executablePath: "/usr/local/bin/hermes",
            arguments: ["-p", "one", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "1111111111111111"]
        )
        let conflicting = HermesSidecarOwnershipRecord(
            launchID: "2222222222222222",
            pid: 7101,
            parentPID: 8002,
            profileName: "two",
            createdAt: .now,
            executablePath: "/usr/local/bin/hermes",
            arguments: ["-p", "two", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "2222222222222222"]
        )
        let process = HermesSidecarProcessSnapshot(
            pid: 7101,
            executablePath: "/usr/local/bin/hermes",
            arguments: first.arguments
        )

        XCTAssertEqual(
            HermesOrphanSidecarScanner.orphanPIDs(records: [first, conflicting], processes: [process, process], livePIDs: []),
            []
        )
    }

    private func makeProfile(named name: String, config: String, env: String? = nil) throws -> URL {
        let profile = hermesHome.appendingPathComponent("profiles/\(name)", isDirectory: true)
        try FileManager.default.createDirectory(at: profile, withIntermediateDirectories: true)
        try config.write(to: profile.appendingPathComponent("config.yaml"), atomically: true, encoding: .utf8)
        if let env {
            try env.write(to: profile.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
        }
        return profile
    }

    private func mtplxConfig() -> String {
        """
        model:
          default: current-model
          provider: custom
          base_url: http://127.0.0.1:18080/v1
        """
    }

    private func makeSidecarFixture(environmentCaptureURL: URL) throws -> URL {
        _ = environmentCaptureURL
        return try makeSidecarBinary()
    }

    private func makeExecWrapperFixture(environmentCaptureURL: URL) throws -> (wrapper: URL, binary: URL) {
        let binary = try makeSidecarBinary()
        let wrapper = root.appendingPathComponent("hermes-exec-wrapper.sh")
        let script = """
        #!/bin/sh
        exec "\(binary.path)" "$@"
        """
        try script.write(to: wrapper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: wrapper.path)
        _ = environmentCaptureURL
        return (wrapper, binary)
    }

    private func makeConsoleScriptLauncherFixture() throws -> (launcher: URL, interpreter: URL, consoleScript: URL) {
        let interpreter = try makeSidecarBinary()
        let consoleDirectory = hermesHome
            .appendingPathComponent("hermes-agent/venv/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: consoleDirectory, withIntermediateDirectories: true)
        let consoleScript = consoleDirectory.appendingPathComponent("hermes")
        try "#!/bin/sh\n# Hermes console entrypoint fixture\n".write(
            to: consoleScript,
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: consoleScript.path)

        let source = root.appendingPathComponent("console-script-launcher.c")
        let launcher = root.appendingPathComponent("console-script-launcher")
        let code = """
        #include <stdlib.h>
        #include <unistd.h>

        int main(int argc, char **argv) {
            char **arguments = calloc((size_t)argc + 2, sizeof(char *));
            if (arguments == NULL) return 2;
            arguments[0] = \"\(interpreter.path)\";
            arguments[1] = \"\(consoleScript.path)\";
            for (int index = 1; index < argc; index++) arguments[index + 1] = argv[index];
            execv(arguments[0], arguments);
            return 3;
        }
        """
        try code.write(to: source, atomically: true, encoding: .utf8)
        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["clang", source.path, "-o", launcher.path]
        try compiler.run()
        compiler.waitUntilExit()
        guard compiler.terminationStatus == 0 else {
            throw NSError(domain: "HermesEmbeddedRuntimeTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Could not compile console-script launcher fixture.",
            ])
        }
        return (launcher, interpreter, consoleScript)
    }

    private func makeSidecarBinary() throws -> URL {
        let source = root.appendingPathComponent("hermes-sidecar-fixture.c")
        let binary = root.appendingPathComponent("hermes-sidecar-fixture")
        let code = """
        #include <signal.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <unistd.h>

        static volatile sig_atomic_t running = 1;
        static void stop(int signal) { (void)signal; running = 0; }

        int main(void) {
            const char *capture = getenv("MTPLX_FIXTURE_ENV_FILE");
            if (capture != NULL) {
                FILE *file = fopen(capture, "w");
                if (file != NULL) {
                    if (getenv("HERMES_DASHBOARD_SESSION_TOKEN") != NULL) {
                        fputs("HERMES_DASHBOARD_SESSION_TOKEN=present\\n", file);
                    }
                    if (getenv("OPENAI_API_KEY") != NULL) {
                        fputs("OPENAI_API_KEY=present\\n", file);
                    }
                    fclose(file);
                }
            }
            fputs("HERMES_BACKEND_READY port=45123\\n", stdout);
            fflush(stdout);
            signal(SIGTERM, stop);
            signal(SIGINT, stop);
            while (running) { pause(); }
            return 0;
        }
        """
        try code.write(to: source, atomically: true, encoding: .utf8)
        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["clang", source.path, "-o", binary.path]
        let output = Pipe()
        compiler.standardOutput = output
        compiler.standardError = output
        let watchdog = SubprocessWatchdog(compiler)
        try compiler.run()
        let drain = SubprocessPipeDrain(output, capacity: 65_536)
        guard watchdog.wait(for: compiler, timeout: 20), compiler.terminationStatus == 0 else {
            drain.join(timeout: 1)
            throw NSError(domain: "HermesEmbeddedRuntimeTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Could not compile sidecar fixture: \(drain.snapshot())",
            ])
        }
        drain.join(timeout: 1)
        return binary
    }

    private func makeSplitSecretFailureFixture(tokenCaptureURL: URL) throws -> URL {
        let fixture = root.appendingPathComponent("hermes-split-secret-fixture.sh")
        let script = """
        #!/bin/sh
        token="$HERMES_DASHBOARD_SESSION_TOKEN"
        printf '%s' "$token" > "$MTPLX_FIXTURE_TOKEN_FILE"
        printf '%s' "${token%????????????????????}" >&2
        printf '%s\\n' "${token#???????????????????????}" >&2
        exit 1
        """
        try script.write(to: fixture, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: fixture.path)
        return fixture
    }

    private func makeShortSplitSecretFailureFixture(secretCaptureURL: URL) throws -> URL {
        let fixture = root.appendingPathComponent("hermes-short-secret-fixture.sh")
        let script = """
        #!/bin/sh
        secret="$OPENAI_API_KEY"
        printf '%s' "$secret" > "$MTPLX_FIXTURE_SECRET_FILE"
        printf '%s' "${secret%?}" >&2
        printf '%s\\n' "${secret#?}" >&2
        exit 1
        """
        try script.write(to: fixture, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: fixture.path)
        _ = secretCaptureURL
        return fixture
    }
}
