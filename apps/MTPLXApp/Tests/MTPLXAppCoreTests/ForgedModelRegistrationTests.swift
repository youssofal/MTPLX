import XCTest
@testable import MTPLXAppCore

final class ForgedModelRegistrationTests: XCTestCase {
    // MARK: forgedModel factory

    func testForgedModelFactoryProducesPickerReadyOption() {
        let option = MTPLXModelOption.forgedModel(
            brandedName: "Qwen3.6-35B-A3B-MTPLX-Speed",
            localPath: "/tmp/MTPLX-models/Qwen3.6-35B-A3B-MTPLX-Speed",
            sizeBytes: 24_000_000_000,
            peakMemoryGiB: 28.0
        )
        XCTAssertNotNil(option)
        let opt = option!
        XCTAssertEqual(opt.id, "forged-qwen3.6-35b-a3b-mtplx-speed")
        XCTAssertEqual(opt.displayName, "Qwen3.6-35B-A3B-MTPLX-Speed")
        XCTAssertEqual(opt.shortName, "Qwen3.6-35B-A3B-MTPLX-Speed")
        XCTAssertEqual(opt.hfModelID, "Qwen3.6-35B-A3B-MTPLX-Speed")
        XCTAssertEqual(opt.localCandidates, ["/tmp/MTPLX-models/Qwen3.6-35B-A3B-MTPLX-Speed"])
        XCTAssertEqual(opt.sizeBytes, 24_000_000_000)
        XCTAssertEqual(opt.peakMemoryGiB, 28.0)
        XCTAssertTrue(opt.aliases.contains("Qwen3.6-35B-A3B-MTPLX-Speed"))
    }

    func testForgedModelFactoryRejectsEmptyBrandedName() {
        XCTAssertNil(MTPLXModelOption.forgedModel(brandedName: "", localPath: "/x"))
        XCTAssertNil(MTPLXModelOption.forgedModel(brandedName: "   ", localPath: "/x"))
    }

    func testForgedModelHfModelIdIsBrandedNameNotPath() {
        // Confirms the chrome-strip picker shows the branded name in
        // its label, not the absolute local path — `hfModelID` is
        // what feeds the picker's display when no shortName helper
        // is overridden.
        let opt = MTPLXModelOption.forgedModel(
            brandedName: "Foo-MTPLX-Speed",
            localPath: "/Users/x/Documents/MTPLX/models/Foo-MTPLX-Speed"
        )
        XCTAssertEqual(opt?.hfModelID, "Foo-MTPLX-Speed")
        XCTAssertNotEqual(opt?.hfModelID, "/Users/x/Documents/MTPLX/models/Foo-MTPLX-Speed")
    }

    func testDisplayNamePrefersRegisteredForgedNameForLocalPath() throws {
        let path = "/Users/x/Documents/MTPLX/models/Foo-MTPLX-Speed"
        let option = try XCTUnwrap(MTPLXModelOption.forgedModel(
            brandedName: "Friendly Foo MTPLX",
            localPath: path
        ))

        XCTAssertEqual(
            MTPLXModelOption.displayName(for: path, customModels: [option]),
            "Friendly Foo MTPLX"
        )
    }

    // MARK: AppConfiguration.rememberForgedModel

    func testRememberForgedModelAppendsToCustomModels() {
        var config = MTPLXAppConfiguration()
        XCTAssertTrue(config.customModels.isEmpty)
        config.rememberForgedModel(
            brandedName: "Qwen3.6-27B-MTPLX-Speed",
            localPath: "/tmp/forged/Qwen3.6-27B-MTPLX-Speed",
            sizeBytes: 16_000_000_000,
            peakMemoryGiB: 17.0
        )
        XCTAssertEqual(config.customModels.count, 1)
        XCTAssertEqual(config.customModels[0].displayName, "Qwen3.6-27B-MTPLX-Speed")
    }

    func testRememberForgedModelDedupesByLocalPath() {
        var config = MTPLXAppConfiguration()
        let path = "/tmp/forged/Foo-MTPLX-Quality"
        config.rememberForgedModel(brandedName: "Foo-MTPLX-Quality", localPath: path)
        config.rememberForgedModel(brandedName: "Foo-MTPLX-Quality", localPath: path)
        XCTAssertEqual(config.customModels.count, 1, "Re-registering the same path collapses to one entry")
    }

    func testRememberForgedModelOverwritesPreviousEntryAtSamePath() {
        var config = MTPLXAppConfiguration()
        let path = "/tmp/forged/Foo-MTPLX-Quality"
        config.rememberForgedModel(brandedName: "Foo-MTPLX-Quality", localPath: path, sizeBytes: 100)
        config.rememberForgedModel(brandedName: "Foo-MTPLX-Quality", localPath: path, sizeBytes: 200)
        XCTAssertEqual(config.customModels.count, 1)
        XCTAssertEqual(config.customModels[0].sizeBytes, 200)
    }

    func testRememberForgedModelRejectsEmptyName() {
        var config = MTPLXAppConfiguration()
        config.rememberForgedModel(brandedName: "", localPath: "/x")
        XCTAssertTrue(config.customModels.isEmpty)
    }

    // MARK: Picker round-trip — the whole point of this commit

    func testForgedModelShowsUpInPickerCatalog() {
        guard let opt = MTPLXModelOption.forgedModel(
            brandedName: "Qwen3.6-35B-A3B-MTPLX-Speed",
            localPath: "/tmp/forged/Qwen3.6-35B-A3B-MTPLX-Speed"
        ) else { return XCTFail("Factory returned nil for valid input") }

        let catalog = MTPLXModelOption.pickerCatalog(customModels: [opt])
        let baseline = MTPLXModelOption.pickerCatalog(customModels: [])

        XCTAssertEqual(catalog.count, baseline.count + 1)
        XCTAssertTrue(catalog.contains { $0.id == opt.id },
                      "Forged option's id should appear in the picker catalog")
        XCTAssertTrue(catalog.contains { $0.displayName == "Qwen3.6-35B-A3B-MTPLX-Speed" })
    }

    func testForgedModelDeduplicatesAgainstOfficialCatalog() {
        // Crafting a forged entry whose hfModelID collides with an
        // official model id is contrived (the branded name pattern
        // doesn't allow it normally), but the dedup rule from
        // `pickerCatalog` should still hold.
        guard let opt = MTPLXModelOption.forgedModel(
            brandedName: "Qwen3.6 27B Optimized Speed",
            localPath: "/tmp/forged/X"
        ) else { return XCTFail("Factory returned nil") }
        let catalog = MTPLXModelOption.pickerCatalog(customModels: [opt])
        let baseline = MTPLXModelOption.pickerCatalog(customModels: [])
        // Official Optimized Speed exists by displayName match;
        // appendCustom drops dupes that collide with the official
        // catalog via matches().
        XCTAssertEqual(
            catalog.count,
            baseline.count,
            "Forged entry colliding with an official model should be deduped out"
        )
        XCTAssertFalse(catalog.contains { $0.id == opt.id })
    }

    func testForgedModelCodableRoundTripPreservesEverything() throws {
        guard let original = MTPLXModelOption.forgedModel(
            brandedName: "X-MTPLX-Balanced",
            localPath: "/y/X-MTPLX-Balanced",
            sizeBytes: 12_345_678,
            peakMemoryGiB: 8.5
        ) else { return XCTFail("Factory returned nil") }
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MTPLXModelOption.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testArbitraryForgedNameReadsModelFamilyFromRuntimeMetadata() throws {
        let modelDir = temporaryDirectory().appendingPathComponent("Tiny-Test-MTPLX", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let runtime = """
        {
          "mtplx_version": "1.0.0",
          "arch_id": "qwen3-next-mtp",
          "mtp_depth_max": 3,
          "forge_provenance": {
            "source_repo": "Qwen/Qwen3.5-2B",
            "source_format": "bf16_native",
            "forge_recipe": {
              "body_bits": 4,
              "body_group_size": 64,
              "body_mode": "affine",
              "mtp_policy": "keep_bf16"
            },
            "forge_inputs": {},
            "forged_at": "2026-06-05T00:00:00+0000",
            "mtplx_version": "1.0.0",
            "forged_locally": true
          }
        }
        """
        try runtime.write(
            to: modelDir.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertEqual(MTPLXModelOption.modelFamily(for: modelDir.path), "qwen3_5")
    }

    func testNeutralForgeNameUsesDeclaredQwen38IdentityAndLaunchDefaults() throws {
        let modelDir = temporaryDirectory().appendingPathComponent("Bare-Speed-Beta", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let runtime = """
        {
          "public_model_id": "mtplx-qwen38-27b-bare-speed-beta",
          "arch_id": "qwen3-next-mtp"
        }
        """
        try runtime.write(
            to: modelDir.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )

        XCTAssertEqual(MTPLXModelOption.modelFamily(for: modelDir.path), "qwen3_8")

        let fake = modelDir.appendingPathComponent("mtplx")
        try "#!/bin/sh\n".write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)
        let command = try MTPLXCommandBuilder(environment: ["PATH": modelDir.path])
            .buildServeCommand(configuration: MTPLXAppConfiguration(
                executablePath: fake.path,
                model: modelDir.path,
                profile: "auto"
            ))
        let temperatureIndex = try XCTUnwrap(command.arguments.firstIndex(of: "--temperature"))
        XCTAssertEqual(command.arguments[temperatureIndex + 1], "1.0")
        XCTAssertFalse(command.arguments.contains("--draft-temperature"))
    }

    func testArbitraryForgedNameKeepsTunedDepthCompatible() throws {
        let modelDir = temporaryDirectory().appendingPathComponent("My-Custom-Name-MTPLX", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let runtime = """
        {
          "mtplx_version": "1.0.0",
          "arch_id": "qwen3-next-mtp",
          "forge_provenance": {
            "source_repo": "Qwen/Qwen3.5-2B",
            "source_format": "bf16_native",
            "forge_recipe": {
              "body_bits": 4,
              "body_group_size": 64,
              "body_mode": "affine",
              "mtp_policy": "keep_bf16"
            },
            "forge_inputs": {},
            "forged_at": "2026-06-05T00:00:00+0000",
            "mtplx_version": "1.0.0",
            "forged_locally": true
          }
        }
        """
        try runtime.write(
            to: modelDir.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )

        var config = MTPLXAppConfiguration()
        config.model = modelDir.path
        config.liveSettingsModelFamily = "qwen3_5"
        config.tunedControlRecord = TunedControlRecord(
            modelID: modelDir.path,
            modelFamily: "qwen3_5",
            backendID: "qwen3_next",
            controlField: "depth",
            controlValue: 2,
            candidates: ["1", "2", "3"],
            tunedAt: Date()
        )

        XCTAssertEqual(config.compatibleTunedDepth(), 2)
    }

    func testForgeLocalEntryUsesWinningSpeedEvidenceDepth() throws {
        let runtimeJSON = try runtimeMetadataJSON(
            sourceRepo: "Qwen/Qwen3.5-2B",
            bestDepth: 1,
            arTokS: 237.11574516823444,
            rows: [
                (0, 237.11574516823444),
                (1, 313.6033431297458),
                (2, 291.1426701662897),
                (3, 222.35626688266214)
            ]
        )
        let metadata = try XCTUnwrap(MTPLXRuntimeMetadata.parse(runtimeJSON))
        let entry = ForgeLocalEntry(
            localPath: "/tmp/forged/Qwen-Qwen3.5-2B-MTPLX-Speed",
            directoryName: "Qwen-Qwen3.5-2B-MTPLX-Speed",
            metadata: metadata,
            modelOption: nil,
            sizeOnDisk: 0
        )

        XCTAssertEqual(entry.depth, 1)
        XCTAssertEqual(entry.verification?.bestDepth, 1)
        XCTAssertEqual(entry.verificationMultiplier ?? 0, 313.6033431297458 / 237.11574516823444, accuracy: 0.0001)
    }

    func testApplyForgeRuntimeDefaultsUsesMeasuredWinningDepth() throws {
        let modelDir = temporaryDirectory().appendingPathComponent("Friendly-Name-MTPLX", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let runtimeJSON = try runtimeMetadataJSON(
            sourceRepo: "Qwen/Qwen3.5-2B",
            bestDepth: 1,
            arTokS: 237.0,
            rows: [(0, 237.0), (1, 313.0), (2, 291.0), (3, 222.0)]
        )
        let runtimeData = try JSONSerialization.data(withJSONObject: runtimeJSON, options: [.prettyPrinted, .sortedKeys])
        try runtimeData.write(to: modelDir.appendingPathComponent("mtplx_runtime.json"))
        let verification = try XCTUnwrap(ForgeVerification.fromRuntimeMetadata(runtimeJSON))

        var config = MTPLXAppConfiguration()
        config.applyForgeRuntimeDefaults(
            modelPath: modelDir.path,
            verification: verification,
            sourceRepo: "Qwen/Qwen3.5-2B"
        )

        XCTAssertEqual(config.model, modelDir.path)
        XCTAssertEqual(config.generationMode, "mtp")
        XCTAssertTrue(config.loadMTP)
        XCTAssertEqual(config.liveSettingsModelFamily, "qwen3_5")
        XCTAssertEqual(config.compatibleTunedDepth(), 1)
        XCTAssertEqual(config.tunedControlRecord?.controlValue, 1)
        XCTAssertEqual(config.tunedControlRecord?.controlField, "depth")
    }

    @MainActor
    func testExistingForgePublishHydratesVerificationFromRuntimeMetadata() throws {
        let modelDir = temporaryDirectory().appendingPathComponent("Friendly-Name-MTPLX", isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }
        let runtime = """
        {
          "mtplx_version": "1.0.0",
          "arch_id": "qwen3-next-mtp",
          "mtp_depth_max": 3,
          "sampler": { "temperature": 0.6, "top_p": 0.95, "top_k": 20 },
          "verified_on": { "hardware": "Apple M5 Max" },
          "speed_evidence": {
            "depth": 3,
            "verdict": "mtp_depth_wins",
            "forge_verify_rows": [
              { "depth": 0, "tok_s": 50.0, "multiplier_vs_ar": 1.0, "acceptance_by_position": [] },
              { "depth": 3, "tok_s": 75.0, "multiplier_vs_ar": 1.5, "acceptance_by_position": [0.9, 0.8, 0.7] }
            ]
          },
          "forge_provenance": {
            "source_repo": "Qwen/Qwen3.5-2B",
            "source_format": "bf16_native",
            "forge_recipe": {
              "body_bits": 4,
              "body_group_size": 64,
              "body_mode": "affine",
              "mtp_policy": "keep_bf16"
            },
            "forge_inputs": {},
            "forged_at": "2026-06-05T00:00:00+0000",
            "mtplx_version": "1.0.0",
            "forged_locally": true
          }
        }
        """
        try runtime.write(
            to: modelDir.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )

        let orchestrator = ForgeOrchestrator()
        orchestrator.startPublishForExistingForge(
            brandedName: "Friendly-Name-MTPLX",
            localPath: modelDir.path
        )

        XCTAssertEqual(orchestrator.state.step, .publishing)
        XCTAssertNil(orchestrator.publishFailure)
        XCTAssertTrue(orchestrator.state.hasSpeedWinningVerification)
        XCTAssertEqual(orchestrator.state.verification?.bestDepth, 3)
        XCTAssertEqual(orchestrator.state.verification?.multiplierVsAr, 1.5)
        XCTAssertEqual(orchestrator.state.sourceProbe?.hfRepo, "Qwen/Qwen3.5-2B")
    }

    // MARK: settings JSON round-trip ensures persistence survives quit/relaunch

    func testForgedRegistrationSurvivesSettingsRoundTrip() throws {
        var config = MTPLXAppConfiguration()
        config.rememberForgedModel(
            brandedName: "Foo-MTPLX-Speed",
            localPath: "/tmp/forged/Foo-MTPLX-Speed",
            sizeBytes: 1234,
            peakMemoryGiB: 5.5
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(config)
        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)
        XCTAssertEqual(decoded.customModels.count, 1)
        XCTAssertEqual(decoded.customModels[0].displayName, "Foo-MTPLX-Speed")
        XCTAssertEqual(decoded.customModels[0].sizeBytes, 1234)
        XCTAssertEqual(decoded.customModels[0].peakMemoryGiB, 5.5)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-forged-registration-\(UUID().uuidString)", isDirectory: true)
    }

    private func runtimeMetadataJSON(
        sourceRepo: String,
        bestDepth: Int,
        arTokS: Double,
        rows: [(Int, Double)]
    ) throws -> [String: Any] {
        [
            "mtplx_version": "1.0.0",
            "arch_id": "qwen3-next-mtp",
            "mtp_depth_max": 3,
            "sampler": ["temperature": 0.6, "top_p": 0.95, "top_k": 20],
            "verified_on": ["hardware": "Apple M5 Max"],
            "speed_evidence": [
                "depth": bestDepth,
                "verdict": "mtp_depth_wins",
                "greedy_diagnostic": ["tok_s": arTokS],
                "forge_verify_rows": rows.map { depth, tokS in
                    [
                        "depth": depth,
                        "tok_s": tokS,
                        "multiplier_vs_ar": arTokS > 0 ? tokS / arTokS : 1.0,
                        "acceptance_by_position": depth == 0
                            ? []
                            : Array(repeating: 0.8, count: depth)
                    ] as [String: Any]
                }
            ],
            "forge_provenance": [
                "source_repo": sourceRepo,
                "source_format": "bf16_native",
                "forge_recipe": [
                    "body_bits": 4,
                    "body_group_size": 64,
                    "body_mode": "affine",
                    "mtp_policy": "keep_bf16"
                ],
                "forge_inputs": [:],
                "forged_at": "2026-06-05T00:00:00+0000",
                "mtplx_version": "1.0.0",
                "forged_locally": true
            ]
        ]
    }
}
