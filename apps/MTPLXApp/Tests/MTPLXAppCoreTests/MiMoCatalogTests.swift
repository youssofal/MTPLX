import Foundation
import XCTest
@testable import MTPLXAppCore

/// MiMo V2.6 Qwen 9B (2026-09-23) is Xiaomi's Qwen3.5-9B fine-tune. "MiMo"
/// in its names is the publisher, so every identity must resolve to the
/// qwen3_5 family and its served id, and the pack rides right ahead of the
/// Qwen 3.5 9B in every modern recommendation list without changing any
/// default model. Mirrors tests/test_mimo_catalog.py.
final class MiMoCatalogTests: XCTestCase {
    private let mimoID = "mimo-v26-qwen-9b-optimized-speed"
    private let nineID = "qwen35-9b-optimized-speed"
    private let bonsaiID = "bonsai-2-27b-optimized-speed"
    private let publicID = "mtplx-mimo-v26-qwen-9b-optimized-speed"
    private let hfID = "Youssofal/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed"
    private let tinyIDs = ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]

    private func mac(_ generation: String, gib: Double) -> DetectedHardware {
        DetectedHardware(
            chipName: "Apple \(generation.uppercased())",
            appleSiliconGeneration: generation,
            unifiedMemoryBytes: Int64(gib * 1_073_741_824)
        )
    }

    private func mimo() throws -> MTPLXModelOption {
        try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == mimoID })
    }

    private func refs() throws -> [String] {
        let model = try mimo()
        return [mimoID, model.hfModelID, publicID]
            + model.aliases.filter { !$0.contains(" ") }
            + model.localCandidates
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-mimo-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String) throws -> URL {
        let url = temporaryDirectory().appendingPathComponent(name)
        try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    func testCatalogEntryMirrorsTheSixBitNineB() throws {
        let model = try mimo()
        let nine = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == nineID })
        XCTAssertEqual(model.displayName, "MiMo V2.6 Qwen 9B Optimized Speed")
        XCTAssertEqual(model.shortName, "MiMo V2.6 Qwen 9B Optimized Speed")
        XCTAssertEqual(model.detail, "6-bit quantization. Xiaomi's agentic coding distill of Qwen 3.5 9B.")
        XCTAssertEqual(model.hfModelID, hfID)
        XCTAssertEqual(model.localCandidates, [
            "~/.mtplx/models/Youssofal--MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
            "~/Documents/MTPLX/models/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
        ])
        XCTAssertEqual(model.aliases, [publicID, "MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed"])
        XCTAssertEqual(model.sizeBytes, 8_695_116_595)
        XCTAssertEqual(model.peakMemoryGiB, nine.peakMemoryGiB)
        XCTAssertEqual(model.peakMemoryGiB, 10.0)
        // No FP16 sibling: M1 and M2 are never offered it.
        XCTAssertEqual(model.recommendedFor, [.modernApple])
        XCTAssertFalse(model.arOnly)
    }

    func testEveryIdentityResolvesToQwen35AndTheServedID() throws {
        for ref in try refs() {
            XCTAssertEqual(MTPLXModelOption.option(matching: ref)?.id, mimoID, ref)
            XCTAssertEqual(MTPLXModelOption.modelFamily(for: ref), "qwen3_5", ref)
            XCTAssertEqual(OpenCodeIntegration.modelID(for: ref), publicID, ref)
            XCTAssertEqual(PiIntegration.modelID(for: ref), publicID, ref)
            XCTAssertTrue(HermesIntegration.launchCommand(for: ref).contains(publicID), ref)
        }
        let model = try mimo()
        XCTAssertEqual(model.modelFamily, "qwen3_5")
        XCTAssertTrue(model.supportsOnboardingTune)
        XCTAssertTrue(MTPLXModelOption.supportsTune(family: model.modelFamily))
        XCTAssertEqual(TuneCandidate.candidates(forFamily: model.modelFamily), [.ar, .d1, .d2, .d3])
        XCTAssertEqual(MTPLXModelOption.displayName(for: publicID), "MiMo V2.6 Qwen 9B Optimized Speed")

        // A derivative never acquires the first-party identity.
        let derivative = hfID + "-third-party"
        XCTAssertNil(MTPLXModelOption.option(matching: derivative))
        XCTAssertNotEqual(OpenCodeIntegration.modelID(for: derivative), publicID)
    }

    func testReasoningHasNoEffortDialAndNeverTheQwen38Codec() throws {
        for ref in try refs() {
            let modelID = OpenCodeIntegration.modelID(for: ref)
            // [] = Qwen think tags without an effort dial (the Qwen 3.5 contract).
            XCTAssertEqual(OpenCodeIntegration.reasoningEffortLevels(forModelID: modelID), [], ref)
            XCTAssertEqual(OpenCodeIntegration.reasoningEffortLevels(forModelID: ref), [], ref)
            XCTAssertNil(OpenCodeIntegration.reasoningEffort(forModelID: modelID), ref)
            XCTAssertNil(OpenCodeIntegration.resolvedReasoningEffort(forModelID: modelID, configuredEffort: nil))
            XCTAssertFalse(MTPLXModelOption.isBonsai2Model(ref), ref)
            XCTAssertEqual(OpenCodeIntegration.samplerTemperature(forModelID: modelID), 0.6)
            XCTAssertEqual(OpenCodeIntegration.samplerTopK(forModelID: modelID), 20)
        }
    }

    func testPublishedRuntimeContractKeepsQwen35AfterARename() throws {
        let directory = temporaryDirectory().appendingPathComponent("renamed-pack")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let metadata: [String: Any] = [
            "arch_id": "qwen3-next-mtp",
            "public_model_id": publicID,
            "served_model_id": publicID,
            "model_family": "qwen3_5",
            "base_trunk": "Qwen/Qwen3.5-9B",
            "mtp_depth_default": 2,
        ]
        try JSONSerialization.data(withJSONObject: metadata)
            .write(to: directory.appendingPathComponent("mtplx_runtime.json"))
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: directory.path), "qwen3_5")
        // Without the family field the served id alone still names Qwen 3.5,
        // not the qwen3_6 fallback every other "qwen" hint lands on.
        let bare = temporaryDirectory().appendingPathComponent("renamed-pack-no-family")
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["public_model_id": publicID])
            .write(to: bare.appendingPathComponent("mtplx_runtime.json"))
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: bare.path), "qwen3_5")
    }

    func testRecommendationOrderPerModernTier() {
        let expectedSmall = [bonsaiID, mimoID, nineID] + tinyIDs
        XCTAssertEqual(MTPLXModelOption.recommendedCatalogIDs(for: mac("m5", gib: 15.9)), tinyIDs)
        XCTAssertEqual(MTPLXModelOption.recommendedCatalogIDs(for: mac("m5", gib: 16)), expectedSmall)
        // From 24 GiB Inkling 2 Mini Bare Speed rides second (inklingTightMacOffer, on since
        // 2026-11-21); MiMo stays right ahead of the Qwen 3.5 9B and Bonsai 2 stays first.
        XCTAssertEqual(
            MTPLXModelOption.recommendedCatalogIDs(for: mac("m5", gib: 24)),
            [bonsaiID, "inkling2-mini-bare-speed", mimoID, nineID] + tinyIDs
        )
        // A 16 GB Mac is never recommended, offered or pre-selected an Inkling pack: the offer
        // starts at 24 GiB and Bare Speed's 18.21 GiB peak is over the machine anyway.
        let sixteen = mac("m5", gib: 16)
        let offeredAt16 = MTPLXModelOption.hardwareAwareOfficialCatalog(hardware: sixteen, includeInstalledOverrides: false).map(\.id)
        XCTAssertFalse(MTPLXModelOption.recommendedCatalogIDs(for: sixteen).contains { $0.hasPrefix("inkling") })
        XCTAssertFalse(offeredAt16.contains { $0.hasPrefix("inkling") })
        XCTAssertEqual(offeredAt16.first, bonsaiID)
        for gib in [16.0, 24, 32, 48, 64, 96, 128, 256] {
            let hardware = mac("m5", gib: gib)
            let raw = MTPLXModelOption.recommendedCatalogIDs(for: hardware)
            let visible = MTPLXModelOption.hardwareAwareOfficialCatalog(
                hardware: hardware,
                includeInstalledOverrides: false
            ).map(\.id)
            for ids in [raw, visible] {
                XCTAssertEqual(ids.filter { $0 == mimoID }.count, 1, "\(gib)")
                // Immediately ahead of the Qwen 3.5 9B, never the first pick.
                XCTAssertEqual(ids.firstIndex(of: mimoID).map { $0 + 1 }, ids.firstIndex(of: nineID), "\(gib)")
                XCTAssertNotEqual(ids.first, mimoID, "\(gib)")
            }
        }
        let unknown = MTPLXModelOption.recommendedCatalogIDs(for: nil)
        XCTAssertEqual(unknown.firstIndex(of: mimoID).map { $0 + 1 }, unknown.firstIndex(of: nineID))
        XCTAssertEqual(Array(unknown.suffix(3)), [mimoID, nineID, bonsaiID])
    }

    func testLegacyTierNeverListsMiMo() {
        for generation in ["m1", "m2"] {
            for gib in [8.0, 15.9, 16, 24, 32, 48, 64, 96, 128, 256] {
                let hardware = mac(generation, gib: gib)
                XCTAssertFalse(MTPLXModelOption.recommendedCatalogIDs(for: hardware).contains(mimoID), "\(generation) \(gib)")
                XCTAssertFalse(
                    MTPLXModelOption.hardwareAwareOfficialCatalog(hardware: hardware, includeInstalledOverrides: false)
                        .contains { $0.id == mimoID },
                    "\(generation) \(gib)"
                )
            }
        }
    }

    func testNoDefaultModelChanges() {
        for (gib, expected) in [
            (12.0, "qwen35-4b-optimized-speed"),
            (16.0, bonsaiID),
            (24.0, bonsaiID),
            (32.0, "qwen38-27b-optimized-speed"),
            (128.0, "qwen38-27b-optimized-speed"),
            (256.0, "flash-next-optimized-speed"),
        ] {
            let path = MTPLXAppConfiguration.defaultLocalModelPath(for: mac("m5", gib: gib))
            XCTAssertEqual(MTPLXModelOption.option(matching: path)?.id, expected, "\(gib)")
        }
        XCTAssertEqual(MTPLXModelOption.recommendedCatalogIDs(for: nil).first, "qwen38-27b-optimized-speed")
    }

    func testOnboardingChoiceResolvesToMiMoOnTheQwen35Contract() throws {
        for hardware in [mac("m5", gib: 16), mac("m4", gib: 24), mac("m3", gib: 64)] {
            var state = OnboardingFeatureState()
            state.hardware = hardware
            state.step = .modelPick
            state.select(.curatedMiMoQwen9BOptimizedSpeed)
            XCTAssertEqual(state.resolvedModel?.id, mimoID)
            XCTAssertEqual(state.resolvedRepoID, hfID)
            XCTAssertEqual(state.resolvedModelFamily, "qwen3_5")
            XCTAssertTrue(state.supportsTune)
            XCTAssertEqual(state.tuneCandidates, [.ar, .d1, .d2, .d3])
            XCTAssertTrue(state.canAdvance)
        }
        XCTAssertNotEqual(ModelPickChoice.curatedMiMoQwen9BOptimizedSpeed, .curatedQwen35NineBSpeed)
    }

    func testLaunchLeavesProfileAndSamplerToTheEngine() throws {
        // Turbo is the 9B's measured promotion, not MiMo's: the pack takes the
        // engine's profile resolution (sustained until measured) and the
        // qwen3_5 family sampler, exactly as `mtplx serve` does with no flags.
        let fake = try makeExecutable(named: "mtplx")
        let builder = MTPLXCommandBuilder(environment: ["PATH": fake.deletingLastPathComponent().path])
        for model in try refs() {
            let command = try builder.buildServeCommand(configuration: MTPLXAppConfiguration(
                executablePath: fake.path, model: model, profile: "auto"
            ))
            XCTAssertFalse(command.arguments.contains("--profile"), model)
            XCTAssertFalse(command.arguments.contains("--temperature"), model)
            XCTAssertFalse(command.arguments.contains("--reasoning-effort"), model)
            XCTAssertEqual(MTPLXCommandBuilder.recommendedProfile(for: model), "sustained", model)
        }
        XCTAssertEqual(MTPLXCommandBuilder.recommendedProfile(for: "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"), "turbo")
    }
}
