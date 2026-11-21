import Foundation
import XCTest

@testable import MTPLXAppCore

/// Inkling 2 Mini launches leave the sampler, the tool-call format and the chat template to the
/// engine's family law, from every launch target, for every Inkling pack. Its descriptor requires
/// native tool calls and the tokenizer's own template, so `mtplx serve` refuses to start Inkling
/// when a launch names `--tool-prompt-mode hybrid` or `--chat-template-profile local_qwen36` (the
/// OpenCode and Hermes presets did, so those buttons could not start it), and an explicit sampler
/// replaces the model's own (temperature 1.0 over the whole distribution). Bare `mtplx serve` and
/// `mtplx start <client>` pass none of them; the app must not either.
/// Mirrors tests/test_app_inkling_launch_contract.py, which resolves the fixture this test writes
/// through the CLI's own serve defaults.
final class InklingLaunchContractTests: XCTestCase {
    private let packs = [
        "inkling2-mini-bare-speed",
        "inkling2-mini-optimized-speed",
        "inkling2-mini-optimized-quality",
        "inkling2-mini-bare-speed-fp16",
        "inkling2-mini-optimized-speed-fp16",
        "inkling2-mini-optimized-quality-fp16",
    ]
    private let engineOwnedFlags: Set<String> = [
        "--temperature", "--top-p", "--top-k",
        "--draft-temperature", "--draft-top-p", "--draft-top-k",
        "--tool-prompt-mode", "--chat-template-profile",
    ]

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-inkling-launch-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeBuilder(home: URL) throws -> MTPLXCommandBuilder {
        let directory = temporaryDirectory()
        let fake = directory.appendingPathComponent("mtplx")
        try "#!/bin/sh\nexit 0\n".write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)
        return MTPLXCommandBuilder(environment: [
            "PATH": directory.path,
            "HOME": home.path,
            "MTPLX_APP_TEST_PHYSICAL_MEMORY_BYTES": String(128 * 1_073_741_824),
        ])
    }

    /// Every reference a user's configuration can hold for each pack: catalog id, Hugging Face id,
    /// the local folder names the catalog looks for.
    private func references() throws -> [(pack: String, reference: String)] {
        try packs.flatMap { id -> [(pack: String, reference: String)] in
            let option = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == id }, id)
            return ([id, option.hfModelID] + option.localCandidates).map { (pack: id, reference: $0) }
        }
    }

    private struct LaunchRow: Codable, Equatable {
        let pack: String
        let target: String
        /// The serve argv after `serve --host H --port P --model M`, temporary paths as `$HOME`.
        let flags: [String]
        let environment: [String: String]
    }

    /// The lane-level part of an app launch: the argv after the per-launch host, port and model,
    /// and the MTPLX_* environment the launch exports (less the app's own process id), with the
    /// test's temporary HOME spelled `$HOME` so the fixture is stable.
    private func launchRow(pack: String, target: LaunchTarget, command: DaemonCommand, home: String) -> LaunchRow {
        func stable(_ value: String) -> String { value.replacingOccurrences(of: home, with: "$HOME") }
        var flags: [String] = []
        var index = 0
        while index < command.arguments.count {
            let token = command.arguments[index]
            if index == 0 && token == "serve" { index += 1; continue }
            if ["--host", "--port", "--model"].contains(token) { index += 2; continue }
            flags.append(stable(token))
            index += 1
        }
        var environment: [String: String] = [:]
        for (key, value) in command.environment where key.hasPrefix("MTPLX_") && key != "MTPLX_APP_PARENT_PID" {
            environment[key] = stable(value)
        }
        return LaunchRow(pack: pack, target: target.rawValue, flags: flags, environment: environment)
    }

    func testEveryInklingLaunchLeavesTheSamplerToolFormatAndTemplateToTheEngine() throws {
        let builder = try makeBuilder(home: temporaryDirectory())
        for (pack, reference) in try references() {
            for target in LaunchTarget.allCases {
                let configuration = MTPLXAppConfiguration(model: reference, profile: "auto")
                let command = try builder.buildServeCommand(configuration: configuration, target: target)
                let named = Set(command.arguments).intersection(engineOwnedFlags)
                XCTAssertTrue(named.isEmpty, "\(pack) via \(reference) from \(target.rawValue) passes \(named.sorted())")
            }
        }
    }

    /// The fixture the CLI-side test resolves (tests/test_app_inkling_launch_contract.py): each
    /// pack's app launch per target. Written when MTPLX_WRITE_APP_LAUNCH_FIXTURE=1, compared
    /// otherwise, so a change on either side shows up in a test.
    func testLaunchFixtureMatchesTheBuilder() throws {
        let home = temporaryDirectory()
        let builder = try makeBuilder(home: home)
        var rows: [LaunchRow] = []
        for pack in packs {
            for target in LaunchTarget.allCases {
                let configuration = MTPLXAppConfiguration(model: pack, profile: "auto")
                let command = try builder.buildServeCommand(configuration: configuration, target: target)
                rows.append(launchRow(pack: pack, target: target, command: command, home: home.path))
            }
        }
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let fixture = root.appendingPathComponent("tests/fixtures/app_inkling_launches.json")
        if ProcessInfo.processInfo.environment["MTPLX_WRITE_APP_LAUNCH_FIXTURE"] == "1" {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(rows).write(to: fixture)
            return
        }
        let stored = try JSONDecoder().decode([LaunchRow].self, from: Data(contentsOf: fixture))
        XCTAssertEqual(stored.count, rows.count, "regenerate with MTPLX_WRITE_APP_LAUNCH_FIXTURE=1")
        for (old, new) in zip(stored, rows) {
            XCTAssertEqual(old, new, "regenerate with MTPLX_WRITE_APP_LAUNCH_FIXTURE=1, then run the CLI-side test")
        }
    }

    /// Every folder of an official Inkling pack names one served id, the same one the engine serves
    /// (default_models._public_model_id_from_name): a download (Youssofal--<repo>), a folder named like
    /// the repository, the folder the packs were built under, the repository id and the catalog id.
    /// A derivative folder keeps its own id.
    func testEveryInklingFolderNamesTheServedID() throws {
        for id in packs {
            let option = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == id })
            let served = "mtplx-" + id.replacingOccurrences(of: "inkling2-mini", with: "inkling-2-mini")
            XCTAssertEqual(option.aliases.first, served)
            let repo = String(option.hfModelID.split(separator: "/")[1])
            XCTAssertTrue(repo.hasPrefix("Inkling-2-Mini-MTPLX-"), repo)
            let built = "MTPLX-Inkling-2-Mini-" + repo.dropFirst("Inkling-2-Mini-MTPLX-".count)
            for ref in [option.hfModelID, id, "/Users/u/.mtplx/models/Youssofal--\(repo)",
                        "/Users/u/.mtplx/models/\(repo)", "/Users/u/.mtplx/models/\(built)"] {
                XCTAssertEqual(OpenCodeIntegration.modelID(for: ref), served, ref)
                XCTAssertEqual(MTPLXModelOption.option(matching: ref)?.id, id, ref)
            }
        }
        XCTAssertEqual(
            OpenCodeIntegration.modelID(for: "/Users/u/Models/Inkling-2-Mini-MTPLX-Native-Experts-Quality"),
            "inkling-2-mini-mtplx-native-experts-quality"
        )
    }
}
