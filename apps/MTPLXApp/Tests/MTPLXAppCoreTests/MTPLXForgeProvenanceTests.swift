import XCTest
@testable import MTPLXAppCore

final class MTPLXForgeProvenanceTests: XCTestCase {
    // MARK: Real fixture parses without forge_provenance

    /// Verbatim copy of
    /// `/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP/mtplx_runtime.json`
    /// — the production runtime metadata that ships with the
    /// Optimized-Speed flagship. Lives here so the Codable extension
    /// can be verified against the actual on-disk schema without a
    /// resource-bundle indirection.
    private static let productionGDN8RuntimeJSON = """
    {
      "mtplx_version": "0.1.0-preview",
      "arch_id": "qwen3-next-mtp",
      "mtp_depth_max": 3,
      "recommended_profile": "stable",
      "exactness_baseline": {
        "gate": "Phase 0H paged-verifier smoke",
        "context": 2048,
        "attention_impl": "mlx_vector_paged",
        "max_abs_diff": 0.0,
        "evidence": "LOG.md records Phase 0H exactness gates at max_diff=0.0 for the champion stack"
      },
      "verified_on": {
        "timestamp": "2026-05-02T02:23:23+0100",
        "hardware": "Apple M5 Max, 128 GB unified memory",
        "machine_arch": "arm64",
        "macos": "26.3.1",
        "model": "Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP",
        "sampler": {
          "temperature": 0.6,
          "top_p": 0.95,
          "top_k": 20
        }
      }
    }
    """

    /// Verbatim copy of
    /// `/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Flat4-CyanKiwiMTP/mtplx_runtime.json`
    /// — the Flat4 variant the existing build_flat4_cyankiwi_mtp_requant.py
    /// script writes. Has the richer optional fields (artifact_role,
    /// base_trunk, mtp_sidecar, speed_evidence) the Forge will set.
    private static let flat4RuntimeJSON = """
    {
      "arch_id": "qwen3-next-mtp",
      "artifact_role": "maximum-speed-flat4-candidate",
      "base_trunk": "mlx-community/Qwen3.6-27B-4bit",
      "exactness_baseline": {
        "attention_impl": "mlx_vector_paged",
        "context": 64,
        "gate": "phase0h-paged-verifier-exactness",
        "max_abs_diff": 0.0,
        "mode": "decode-from-stock-prefix",
        "sample_agreement": 1.0,
        "status": "passed",
        "topk_overlap_ratio": 1.0,
        "total_variation": 0.0,
        "verify_tokens": 4
      },
      "mtp_depth_max": 3,
      "mtp_sidecar": "Qwen3.6-27B-MTPLX-CyanKiwi-Packed-BF16-INT4-v3",
      "mtplx_version": "0.1.0-preview",
      "recommended_profile": "performance-cold"
    }
    """

    func testProductionFixtureParsesWithoutForgeProvenance() throws {
        let data = Self.productionGDN8RuntimeJSON.data(using: .utf8)!
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let meta = try XCTUnwrap(MTPLXRuntimeMetadata.parse(json))

        XCTAssertEqual(meta.mtplxVersion, "0.1.0-preview")
        XCTAssertEqual(meta.archId, "qwen3-next-mtp")
        XCTAssertEqual(meta.mtpDepthMax, 3)
        XCTAssertEqual(meta.recommendedProfile, "stable")
        XCTAssertNil(meta.forgeProvenance, "Existing flagships have no forge_provenance — additive schema")
        XCTAssertNotNil(meta.rawJSON["exactness_baseline"], "Unmodelled fields preserved in rawJSON for the detail panel")
        XCTAssertNotNil(meta.rawJSON["verified_on"])
    }

    func testRuntimeMetadataParsesExplicitModelIdentity() throws {
        let meta = try XCTUnwrap(MTPLXRuntimeMetadata.parse([
            "public_model_id": "mtplx-qwen38-27b-bare-speed-beta",
            "model_family": "qwen3_8",
            "arch_id": "qwen3-next-mtp",
        ]))

        XCTAssertEqual(meta.publicModelID, "mtplx-qwen38-27b-bare-speed-beta")
        XCTAssertEqual(meta.modelFamily, "qwen3_8")
    }

    func testFlat4FixtureParsesWithOptionalFields() throws {
        let data = Self.flat4RuntimeJSON.data(using: .utf8)!
        let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let meta = try XCTUnwrap(MTPLXRuntimeMetadata.parse(json))

        XCTAssertEqual(meta.artifactRole, "maximum-speed-flat4-candidate")
        XCTAssertEqual(meta.baseTrunk, "mlx-community/Qwen3.6-27B-4bit")
        XCTAssertEqual(meta.mtpSidecar, "Qwen3.6-27B-MTPLX-CyanKiwi-Packed-BF16-INT4-v3")
        XCTAssertEqual(meta.recommendedProfile, "performance-cold")
        XCTAssertNil(meta.forgeProvenance)
    }

    // MARK: Forge provenance Codable round-trip

    func testForgeProvenanceRoundTripsThroughJSON() throws {
        let original = MTPLXForgeProvenance(
            sourceRepo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            sourceSha: "7a1c0c26c56ee56f98bfdb77124acf5b239eabf3",
            sourceFormat: .compressedTensorsAwq,
            forgeRecipe: ForgeRecipe(
                bodyBits: 4,
                bodyGroupSize: 64,
                bodyMode: .affine,
                mtpPolicy: .extractFromSidecar
            ),
            forgeInputs: [
                "trunk_path": "/Users/x/Documents/MTPLX/models/Qwen3.6-35B-A3B-AWQ-4bit",
                "mtp_source_path": "/Users/x/Documents/MTPLX/models/Qwen3.6-35B-A3B-AWQ-4bit/model_mtp.safetensors"
            ],
            forgedAt: "2026-05-25T22:45:00+0100",
            mtplxVersion: "1.0.0",
            forgedLocally: true,
            publishedToHf: nil
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(original)

        let decoded = try JSONDecoder().decode(MTPLXForgeProvenance.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testForgeProvenanceRoundTripsWithPublishedBlock() throws {
        let published = MTPLXForgeProvenance.PublishedToHF(
            repo: "youssofal/Qwen3.6-35B-A3B-MTPLX-Speed",
            revision: "abc123",
            visibility: .publicRepo,
            licenseSpdx: "apache-2.0",
            uploadedAt: "2026-05-25T23:00:00+0100"
        )
        let original = MTPLXForgeProvenance(
            sourceRepo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            sourceFormat: .compressedTensorsAwq,
            forgeRecipe: ForgeRecipe(),
            forgedAt: "2026-05-25T22:45:00+0100",
            mtplxVersion: "1.0.0",
            forgedLocally: true,
            publishedToHf: published
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MTPLXForgeProvenance.self, from: data)
        XCTAssertEqual(decoded.publishedToHf, published)
        XCTAssertEqual(decoded.publishedToHf?.licenseSpdx, "apache-2.0")
        XCTAssertEqual(decoded.publishedToHf?.visibility, .publicRepo)
    }

    func testForgeProvenanceJSONKeysAreSnakeCase() throws {
        let provenance = MTPLXForgeProvenance(
            sourceRepo: "a/b",
            sourceFormat: .bf16Native,
            forgeRecipe: ForgeRecipe(),
            forgedAt: "2026-05-25T00:00:00Z",
            mtplxVersion: "1.0.0"
        )
        let data = try JSONEncoder().encode(provenance)
        let text = String(data: data, encoding: .utf8) ?? ""

        // Spec from plan section 3.4 — these keys MUST appear verbatim
        // so the Python agent's writer and the Swift reader stay in
        // sync without a separate schema doc.
        for key in [
            "\"source_repo\"",
            "\"source_format\"",
            "\"forge_recipe\"",
            "\"forge_inputs\"",
            "\"forged_at\"",
            "\"mtplx_version\"",
            "\"forged_locally\""
        ] {
            XCTAssertTrue(text.contains(key), "Missing canonical JSON key \(key) in \(text)")
        }
    }

    func testForgeRecipeJSONKeysAreSnakeCase() throws {
        let recipe = ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, mtpPolicy: .keepBf16)
        let data = try JSONEncoder().encode(recipe)
        let text = String(data: data, encoding: .utf8) ?? ""
        for key in ["\"body_bits\"", "\"body_group_size\"", "\"body_mode\"", "\"mtp_policy\""] {
            XCTAssertTrue(text.contains(key), "Missing canonical JSON key \(key) in \(text)")
        }
        XCTAssertTrue(text.contains("\"keep_bf16\""), "MTP policy value should be snake_case raw")
    }

    // MARK: Extending an existing runtime metadata with forge_provenance

    func testForgeProvenanceCanBeAddedToExistingRuntimeMetadata() throws {
        let baseData = Self.flat4RuntimeJSON.data(using: .utf8)!
        var baseJSON = try XCTUnwrap(try JSONSerialization.jsonObject(with: baseData) as? [String: Any])

        let provenance = MTPLXForgeProvenance(
            sourceRepo: "mlx-community/Qwen3.6-27B-4bit",
            sourceFormat: .mlxAffine,
            forgeRecipe: ForgeRecipe(
                bodyBits: 4,
                bodyGroupSize: 64,
                bodyMode: .affine,
                mtpPolicy: .extractFromSidecar
            ),
            forgeInputs: ["trunk_path": "/tmp/trunk", "mtp_source_path": "/tmp/mtp.safetensors"],
            forgedAt: "2026-05-25T23:00:00+0100",
            mtplxVersion: "1.0.0",
            forgedLocally: true
        )

        let provenanceData = try JSONEncoder().encode(provenance)
        let provenanceJSON = try XCTUnwrap(try JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
        baseJSON["forge_provenance"] = provenanceJSON

        let merged = try XCTUnwrap(MTPLXRuntimeMetadata.parse(baseJSON))

        // Spine fields still present.
        XCTAssertEqual(merged.archId, "qwen3-next-mtp")
        XCTAssertEqual(merged.artifactRole, "maximum-speed-flat4-candidate")
        XCTAssertEqual(merged.baseTrunk, "mlx-community/Qwen3.6-27B-4bit")

        // Forge block round-trips through the parse boundary.
        let recoveredProv = try XCTUnwrap(merged.forgeProvenance)
        XCTAssertEqual(recoveredProv.sourceRepo, "mlx-community/Qwen3.6-27B-4bit")
        XCTAssertEqual(recoveredProv.sourceFormat, .mlxAffine)
        XCTAssertEqual(recoveredProv.forgeRecipe.mtpPolicy, .extractFromSidecar)
        XCTAssertTrue(recoveredProv.forgedLocally)
    }

    func testRuntimeMetadataReadReturnsNilForMissingFile() {
        XCTAssertNil(MTPLXRuntimeMetadata.read(at: "/no/such/path/mtplx_runtime.json"))
    }
}
