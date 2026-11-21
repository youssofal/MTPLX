import Foundation
import XCTest
@testable import MTPLXAppCore

final class Release2114CatalogTests: XCTestCase {
    private struct MatrixRow: Decodable {
        let ram_gib: Int
        let tier: String
        let raw: [String]
        let visible: [String]
    }

    func testSharedPythonSwiftRAMMatrix() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/release_2114_recommendations.json"))
        for row in try JSONDecoder().decode([MatrixRow].self, from: data) {
            let hardware = DetectedHardware(
                chipName: row.tier == "legacy" ? "Apple M2" : "Apple M5",
                appleSiliconGeneration: row.tier == "legacy" ? "m2" : "m5",
                unifiedMemoryBytes: Int64(row.ram_gib) * 1_073_741_824
            )
            XCTAssertEqual(MTPLXModelOption.recommendedCatalogIDs(for: hardware), row.raw, "\(row.tier) \(row.ram_gib)")
            XCTAssertEqual(MTPLXModelOption.hardwareAwareOfficialCatalog(hardware: hardware, includeInstalledOverrides: false).map(\.id), row.visible)
            if let firstID = row.visible.first {
                XCTAssertEqual(MTPLXModelOption.option(matching: MTPLXAppConfiguration.defaultLocalModelPath(for: hardware))?.id, firstID)
            }
        }
        XCTAssertEqual(MTPLXModelOption.officialCatalog.count, 32)
    }

    func testBonsaiBoundCanMoveWithoutDroppingTheOption() {
        XCTAssertEqual(MTPLXModelOption.bonsaiRecommendationMinGiB, 16)
        for ram in [16, 18, 24] {
            let hardware = DetectedHardware(chipName: "Apple M5", appleSiliconGeneration: "m5", unifiedMemoryBytes: Int64(ram) * 1_073_741_824)
            let ids = MTPLXModelOption.recommendedCatalogIDs(for: hardware, bonsaiMinimumGiB: 24)
            // Below a moved bound Bonsai follows the 9B-class picks, led by MiMo.
            XCTAssertEqual(ids.first, ram < 24 ? "mimo-v26-qwen-9b-optimized-speed" : "bonsai-2-27b-optimized-speed")
            XCTAssertTrue(ids.contains("bonsai-2-27b-optimized-speed"))
        }
    }

    func testNewIdentitiesFamiliesAndClientIDs() throws {
        for (id, family) in [("flash-next-optimized-quality", "qwen4_exp"), ("bonsai-2-27b-optimized-speed", "qwen3_8")] {
            let model = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == id })
            for ref in [id, model.hfModelID] + model.aliases.filter({ !$0.contains(" ") }) + model.localCandidates {
                XCTAssertEqual(MTPLXModelOption.option(matching: ref)?.id, id, ref)
                XCTAssertEqual(MTPLXModelOption.modelFamily(for: ref), family, ref)
                XCTAssertEqual(OpenCodeIntegration.modelID(for: ref), "mtplx-\(id)")
                XCTAssertEqual(PiIntegration.modelID(for: ref), "mtplx-\(id)")
                XCTAssertTrue(HermesIntegration.launchCommand(for: ref).contains("mtplx-\(id)"))
            }
            let derivative = model.hfModelID + "-third-party"
            XCTAssertNil(MTPLXModelOption.option(matching: derivative))
            XCTAssertNotEqual(OpenCodeIntegration.modelID(for: derivative), "mtplx-\(id)")
        }
    }

    func testLegacyMetadataKeepsBonsaiFamilyAfterRename() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let metadata = ["public_model_id": "mtplx-bonsai-38-27b-optimized-speed", "arch_id": "qwen3-next-mtp"]
        try JSONSerialization.data(withJSONObject: metadata).write(to: directory.appendingPathComponent("mtplx_runtime.json"))
        XCTAssertEqual(MTPLXModelOption.modelFamily(for: directory.path), "qwen3_8")
        XCTAssertEqual(OpenCodeIntegration.reasoningEffortLevels(forModelID: "mtplx-bonsai-2-27b-optimized-speed"), ["xhigh", "medium"])
        XCTAssertEqual(OpenCodeIntegration.reasoningEffortLevels(forModelID: directory.path), ["xhigh", "medium"])
    }

    func testNewCuratedChoicesResolveAndCanAdvance() throws {
        for (choice, id) in [(ModelPickChoice.curatedQwen35FourBit, "qwen35-4b-optimized-speed"), (.curatedQwen35FourBQuality, "qwen35-4b-optimized-quality"), (.curatedBonsaiOptimizedSpeed, "bonsai-2-27b-optimized-speed"), (.curatedFlashNextOptimizedQuality, "flash-next-optimized-quality")] {
            var state = OnboardingFeatureState()
            state.step = .modelPick
            state.select(choice)
            XCTAssertEqual(state.resolvedModel?.id, id)
            XCTAssertEqual(state.resolvedRepoID, state.resolvedModel?.hfModelID)
            XCTAssertTrue(state.canAdvance)
        }
        XCTAssertEqual(OnboardingFeatureState().pick, .none)
        XCTAssertEqual(MTPLXAppConfiguration(model: "User/SavedModel").model, "User/SavedModel")
    }

    func testNewPackFeasibilityAndUnchangedBadgeBoundaries() throws {
        let evaluator = ModelFeasibility()
        for (id, ram, verdict) in [("flash-next-optimized-quality", 256.0, ModelFeasibilityVerdict.recommended), ("bonsai-2-27b-optimized-speed", 16.0, .tightFit), ("bonsai-2-27b-optimized-speed", 24.0, .recommended)] {
            let model = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == id })
            XCTAssertEqual(evaluator.evaluate(model: model, chipTier: .modernApple, ramGiB: ram, diskFreeGiB: 1000), verdict)
            if case .insufficientDisk = evaluator.evaluate(model: model, chipTier: .modernApple, ramGiB: ram, diskFreeGiB: 1) {} else { XCTFail("Disk gate must apply") }
        }
        var model = try XCTUnwrap(MTPLXModelOption.officialCatalog.first)
        for (ram, peak, expected) in [(256.0, 170.7, ModelFeasibilityVerdict.tightFit), (256, 170.6, .recommended), (16, 10.7, .tightFit), (16, 10.6, .recommended)] {
            model.peakMemoryGiB = peak
            XCTAssertEqual(evaluator.evaluate(model: model, chipTier: .modernApple, ramGiB: ram, diskFreeGiB: 1000), expected)
        }
    }
}
