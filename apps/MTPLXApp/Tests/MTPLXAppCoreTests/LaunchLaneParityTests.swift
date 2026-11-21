import XCTest

@testable import MTPLXAppCore

/// One resolver, three entry points (PX.1, 2026-09-18).
///
/// The engine owns the launch lane (`mtplx/launch_lane.py`): scheduler mode,
/// SSD session cache and its cap, the agent env block, the rule that the
/// prefill chunk is never a launch flag. `scripts/write_launch_lane_matrix.py`
/// writes that resolver's answers for the app to
/// `Tests/Fixtures/launch_lane_matrix.json`, and this test builds the app's
/// real daemon argv for every (client, model family, Adaptive depth switch)
/// on every simulated RAM seat from 16 to 128 GB and compares.
///
/// Before this test the same client took a different engine path depending
/// on how the daemon was started: Pi was `ar_batch` here and `serial` from
/// `mtplx start pi`, and nobody saw it.
final class LaunchLaneParityTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Row: Decodable {
            let family: String
            let client: String
            let adaptiveDepthSwitch: Bool?
            let argv: [String: JSONValue]
            let effectiveAdaptivePolicy: String
            let effectiveDepth: Int
            let prefillChunkIsALaunchFlag: Bool
            let agentEnv: Bool

            enum CodingKeys: String, CodingKey {
                case family, client, argv
                case adaptiveDepthSwitch = "adaptive_depth_switch"
                case effectiveAdaptivePolicy = "effective_adaptive_policy"
                case effectiveDepth = "effective_depth"
                case prefillChunkIsALaunchFlag = "prefill_chunk_is_a_launch_flag"
                case agentEnv = "agent_env"
            }
        }

        let seatsGib: [Int]
        let piSchedulerMode: String
        let familiesWithOneCompiledDepth: [String]
        let agentEnvBySeatGib: [String: [String: [String: String]]]
        let rows: [Row]

        enum CodingKeys: String, CodingKey {
            case rows
            case seatsGib = "seats_gib"
            case piSchedulerMode = "pi_scheduler_mode"
            case familiesWithOneCompiledDepth = "families_with_one_compiled_depth"
            case agentEnvBySeatGib = "agent_env_by_seat_gib"
        }
    }

    private static let modelByFamily = [
        "qwen4_exp": "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
        "qwen3_8": "/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
        "inkling2": "/models/MTPLX-Inkling-2-Mini-Bare-Speed",
    ]

    private static let targetByClient: [String: LaunchTarget] = [
        "chat": .chat,
        "openwebui": .openWebUI,
        "opencode": .openCode,
        "pi": .pi,
        "hermes": .hermes,
        "claude-code": .claudeCode,
        "codex": .codex,
        "deepseek-harness": .deepseekHarness,
        "other": .other,
        "benchmark": .benchmark,
    ]

    private func loadFixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/launch_lane_matrix.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func makeExecutable() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-lane-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("mtplx")
        try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func expectedText(_ value: JSONValue?) -> String? {
        switch value {
        case .some(.string(let text)): return text
        case .some(.number(let number)):
            return number == number.rounded() ? String(Int(number)) : String(number)
        default: return nil
        }
    }

    /// Dumps the argv for every cell and compares it with the resolver.
    func testAppArgvAgreesWithTheEngineResolverOnEverySeat() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture.seatsGib, [16, 24, 36, 48, 64, 96, 128])
        XCTAssertEqual(fixture.piSchedulerMode, "serial")
        let fake = try makeExecutable()
        var cells = 0

        for seat in fixture.seatsGib {
            let builder = MTPLXCommandBuilder(environment: [
                "PATH": fake.deletingLastPathComponent().path,
                "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": String(UInt64(seat) * 1024 * 1024 * 1024),
            ])
            for row in fixture.rows {
                guard let model = Self.modelByFamily[row.family],
                      let target = Self.targetByClient[row.client]
                else {
                    XCTFail("fixture names an unknown family or client: \(row.family) \(row.client)")
                    continue
                }
                let command = try builder.buildServeCommand(
                    configuration: MTPLXAppConfiguration(
                        executablePath: fake.path,
                        model: model,
                        adaptiveDepth: row.adaptiveDepthSwitch
                    ),
                    target: target,
                    launchID: "lane-parity"
                )
                let argv = command.arguments
                let cell = "\(row.client) / \(row.family) / \(seat) GB / switch \(String(describing: row.adaptiveDepthSwitch))"
                cells += 1

                for (key, flag) in [
                    ("scheduler_mode", "--scheduler-mode"),
                    ("batching_preset", "--batching-preset"),
                    ("ssd_session_cache", "--ssd-session-cache"),
                ] {
                    XCTAssertEqual(
                        MTPLXCommandBuilder.flagValue(flag, in: argv),
                        expectedText(row.argv[key]),
                        "\(flag) for \(cell)"
                    )
                }
                for (key, flag) in [
                    ("max_active_requests", "--max-active-requests"),
                    ("decode_batch_max", "--decode-batch-max"),
                ] {
                    XCTAssertEqual(
                        MTPLXCommandBuilder.flagValue(flag, in: argv),
                        expectedText(row.argv[key]),
                        "\(flag) for \(cell)"
                    )
                }
                if let wait = MTPLXCommandBuilder.flagValue("--batch-wait-ms", in: argv) {
                    XCTAssertEqual(Double(wait), Double(expectedText(row.argv["batch_wait_ms"]) ?? ""), cell)
                } else {
                    XCTAssertNil(expectedText(row.argv["batch_wait_ms"]), "--batch-wait-ms for \(cell)")
                }
                if MTPLXCommandBuilder.flagValue("--ssd-session-cache", in: argv) != "off" {
                    XCTAssertEqual(
                        MTPLXCommandBuilder.flagValue("--ssd-session-cache-max-size", in: argv),
                        expectedText(row.argv["ssd_session_cache_max_size"]),
                        "SSD cap for \(cell)"
                    )
                    XCTAssertEqual(
                        MTPLXCommandBuilder.flagValue("--ssd-session-cache-min-prefix-tokens", in: argv),
                        expectedText(row.argv["ssd_session_cache_min_prefix_tokens"]),
                        "SSD minimum prefix for \(cell)"
                    )
                }

                // The prefill chunk is a model-tuned value the engine's family
                // block owns; a preset must never pass it (PX.0).
                XCTAssertFalse(row.prefillChunkIsALaunchFlag)
                XCTAssertFalse(argv.contains("--prefill-chunk-tokens"), "prefill chunk for \(cell)")

                // Depth policy. The engine resolves a request on a family
                // with one compiled verify depth to static depth, so there
                // the app may ask for anything. Elsewhere the argv is the
                // effective policy.
                let requested = MTPLXCommandBuilder.flagValue("--adaptive-policy", in: argv) ?? "none"
                if !fixture.familiesWithOneCompiledDepth.contains(row.family) {
                    XCTAssertEqual(requested, row.effectiveAdaptivePolicy, "adaptive policy for \(cell)")
                }
                if let depth = MTPLXCommandBuilder.flagValue("--depth", in: argv) {
                    XCTAssertEqual(Int(depth), row.effectiveDepth, "depth for \(cell)")
                }

                // The agent env block, seat by seat.
                let expectedEnv = fixture.agentEnvBySeatGib[String(seat)]?[row.client] ?? [:]
                XCTAssertEqual(row.agentEnv, !expectedEnv.isEmpty)
                for (key, value) in expectedEnv {
                    XCTAssertEqual(command.environment[key], value, "\(key) for \(cell)")
                }
                for key in command.environment.keys
                where key.contains("READ_INSPECTION") || key.contains("READ_ONLY_INSPECTION") {
                    XCTFail("\(key) re-arms a transcript compactor (#282): \(cell)")
                }
            }
        }
        // 3 families (Flash-Next, Qwen 3.8, Inkling 2 Mini) x 10 clients x 3 switch states x 7 seats.
        XCTAssertEqual(cells, Self.modelByFamily.count * 10 * 3 * 7)
        XCTAssertEqual(Set(fixture.rows.map(\.family)), Set(Self.modelByFamily.keys))
    }
}
