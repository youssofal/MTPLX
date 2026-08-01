import XCTest
@testable import MTPLXAppCore

final class HermesEmbeddedRuntimeTests: XCTestCase {
    private var root: URL!
    private var hermesHome: URL!
    private var integration: HermesIntegration!
    private var configuration: MTPLXAppConfiguration!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("HermesEmbeddedRuntimeTests-\(UUID().uuidString)", isDirectory: true)
        hermesHome = root.appendingPathComponent(".hermes", isDirectory: true)
        try FileManager.default.createDirectory(at: hermesHome, withIntermediateDirectories: true)
        integration = HermesIntegration(
            hermesHome: hermesHome,
            executablePath: "/usr/bin/true",
            environment: [
                "HOME": root.path,
                "PATH": "/usr/bin:/bin",
                "HERMES_HOME": "/inherited/hermes-home",
            ]
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
}
