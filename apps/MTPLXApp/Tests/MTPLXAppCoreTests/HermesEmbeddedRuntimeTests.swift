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

    func testOrphanCleanupRequiresExactMarkerCommandAndDeadParent() {
        let records = [
            HermesSidecarOwnershipRecord(launchID: "1111111111111111", pid: 7101, parentPID: 8001, profileName: "one", createdAt: .now),
            HermesSidecarOwnershipRecord(launchID: "2222222222222222", pid: 7102, parentPID: 8002, profileName: "two", createdAt: .now),
            HermesSidecarOwnershipRecord(launchID: "3333333333333333", pid: 7103, parentPID: 8003, profileName: "three", createdAt: .now),
            HermesSidecarOwnershipRecord(launchID: "4444444444444444", pid: 7104, parentPID: 9001, profileName: "four", createdAt: .now),
        ]
        let processes = [
            HermesSidecarProcessSnapshot(pid: 7101, arguments: ["-p", "one", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", "1111111111111111"]),
            HermesSidecarProcessSnapshot(pid: 7102, arguments: ["serve", "--host", "127.0.0.1", "--port", "0"]),
            HermesSidecarProcessSnapshot(pid: 7103, arguments: ["serve", "--isolated", "--ssh-owner-nonce", "wrong-marker"]),
            HermesSidecarProcessSnapshot(pid: 7104, arguments: ["serve", "--isolated", "--ssh-owner-nonce", "4444444444444444"]),
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
        let fixture = root.appendingPathComponent("hermes-fixture.sh")
        let script = """
        #!/bin/sh
        if [ -n \"$HERMES_DASHBOARD_SESSION_TOKEN\" ]; then
          printf 'HERMES_DASHBOARD_SESSION_TOKEN=present\\n' > \"\(environmentCaptureURL.path)\"
        fi
        if [ -n \"$OPENAI_API_KEY\" ]; then
          printf 'OPENAI_API_KEY=present\\n' >> \"\(environmentCaptureURL.path)\"
        fi
        printf 'HERMES_BACKEND_READY port=45123\\n'
        trap 'exit 0' TERM INT
        while :; do sleep 1; done
        """
        try script.write(to: fixture, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: fixture.path)
        return fixture
    }
}
