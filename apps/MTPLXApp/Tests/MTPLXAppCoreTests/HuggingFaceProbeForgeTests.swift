import Foundation
import XCTest
@testable import MTPLXAppCore

final class HuggingFaceProbeForgeTests: XCTestCase {
    // MARK: - HTTPRunner harness

    /// Routes URLs to canned JSON responses for the three endpoints
    /// HuggingFaceProbe touches:
    ///
    ///   /<repo>/resolve/main/mtplx_runtime.json
    ///   /<repo>/resolve/main/config.json
    ///   /api/models/<repo>?expand=config&expand=sha
    ///   /api/models/<repo>/tree/main
    ///
    /// Missing entries return 404; throwing entries simulate network
    /// errors (URLError-style). Comparisons are case-insensitive on
    /// path so cases like cyankiwi vs CyanKiwi don't break tests.
    private final class FakeRunner: @unchecked Sendable {
        var responses: [String: (Int, Data)] = [:]
        var errors: Set<String> = []

        func install(url: String, status: Int = 200, body: String) {
            responses[url] = (status, body.data(using: .utf8) ?? Data())
        }

        func runner() -> HuggingFaceProbe.HTTPRunner {
            let snapshot = responses
            let snapshotErrors = errors
            return { url, _ in
                let key = url.absoluteString
                if snapshotErrors.contains(key) {
                    throw URLError(.notConnectedToInternet)
                }
                if let (status, data) = snapshot[key] {
                    return (status, data)
                }
                return (404, Data())
            }
        }
    }

    // MARK: - Already-MTPLX short-circuit

    func testForgeProbeAlreadyMTPLXShortCircuitsBeforeFetchingConfig() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed/resolve/main/mtplx_runtime.json",
            body: """
            {
              "mtplx_version": "0.1.0-preview",
              "arch_id": "qwen3-next-mtp",
              "mtp_depth_max": 3,
              "mtp_sidecar": "Qwen3.6-27B-MTPLX-CyanKiwi-Packed-BF16-INT4-v3"
            }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
        XCTAssertEqual(result.verdict, .alreadyMTPLX)
        XCTAssertEqual(result.sourceFormat, .mlxAffineWithMtp)
        XCTAssertTrue(result.message.contains("depth 3"), "Verdict surfaces the verified MTP depth")
    }

    func testForgeProbeAlreadyMTPLXWithoutSidecarFlagsMlxAffine() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/foo/Qwen3.6-MTPLX-NoSidecar/resolve/main/mtplx_runtime.json",
            body: """
            { "mtplx_version": "1.0.0", "arch_id": "qwen3-next-mtp", "mtp_depth_max": 1 }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "foo/Qwen3.6-MTPLX-NoSidecar")
        XCTAssertEqual(result.verdict, .alreadyMTPLX)
        XCTAssertEqual(result.sourceFormat, .mlxAffine,
                       "No mtp_sidecar key → can't claim mlxAffineWithMtp")
    }

    // MARK: - Forgeable verdicts (no mtplx_runtime.json present)

    func testForgeProbeCompressedTensorsAwqDetected() async {
        let fake = FakeRunner()
        // mtplx_runtime.json absent → fall through to config.json
        fake.install(
            url: "https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3MoeForCausalLM"],
              "num_nextn_predict_layers": 1,
              "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized"
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.sourceFormat, .compressedTensorsAwq)
        XCTAssertTrue(result.message.contains("AWQ"))
    }

    func testForgeProbeAcceptsHuggingFaceModelURL() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3MoeForCausalLM"],
              "num_nextn_predict_layers": 1,
              "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized"
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/cyankiwi/Qwen3.6-27B-AWQ-INT4/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4")

        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.hfRepo, "cyankiwi/Qwen3.6-27B-AWQ-INT4")
        XCTAssertEqual(result.sourceFormat, .compressedTensorsAwq)
        XCTAssertNil(result.diagnostic)
    }

    func testForgeProbeBf16NativeDetectedWhenNoQuantBlock() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3NextForCausalLM"],
              "num_nextn_predict_layers": 1
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Qwen/Qwen3.6-27B")
        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.sourceFormat, .bf16Native)
        XCTAssertTrue(result.message.contains("BF16"))
    }

    func testForgeProbeMlxAffineWithMtpDetectedFromExtraTensors() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/mlx-community/Qwen3.6-27B-MTP/resolve/main/config.json",
            body: """
            {
              "num_nextn_predict_layers": 1,
              "mlx_lm_extra_tensors": { "mtp_file": "mtp.safetensors", "mtp_tensor_count": 29 }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/mlx-community/Qwen3.6-27B-MTP/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "mlx-community/Qwen3.6-27B-MTP")
        XCTAssertEqual(result.sourceFormat, .mlxAffineWithMtp)
        XCTAssertTrue(result.hasMtpWeights)
    }

    func testForgeProbeNoMtpHeadsRefusedEvenWhenSourceIsClean() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/mlx-community/Llama-3-8B/resolve/main/config.json",
            body: """
            { "architectures": ["LlamaForCausalLM"] }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "mlx-community/Llama-3-8B")
        XCTAssertEqual(result.verdict, .noMtpHeads)
        XCTAssertTrue(result.message.contains("Forge cannot synthesize"))
    }

    // MARK: - Error paths

    func testForgeProbeInvalidRepoIdShortCircuits() async {
        let probe = HuggingFaceProbe(runner: FakeRunner().runner())
        let result = await probe.forgeProbe(repo: "not-a-valid-repo")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "invalid_repo_id")
    }

    func testForgeProbe404OnConfigJsonReturnsProbeFailed() async {
        let fake = FakeRunner()
        // Nothing installed → all requests 404
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "nonexistent/repo")
        XCTAssertEqual(result.verdict, .probeFailed)
    }

    func testForgeProbeFallsBackToModelAPIWhenLargeHy3ConfigTimesOut() async {
        let fake = FakeRunner()
        let repo = "philipjohnbasile/hy3-demolition-mlx-reap25-v1-mtp"
        fake.errors.insert(
            "https://huggingface.co/\(repo)/resolve/main/config.json"
        )
        fake.install(
            url: "https://huggingface.co/api/models/\(repo)?expand=config&expand=sha",
            body: """
            {
              "sha": "ac84bc50a90acdcbbffc632877f60be4b7efdddf",
              "config": {
                "model_type": "hy_v3"
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/\(repo)/resolve/ac84bc50a90acdcbbffc632877f60be4b7efdddf/config.json",
            body: """
            {
              "model_type": "hy_v3",
              "num_nextn_predict_layers": 1,
              "quantization": {
                "bits": 4,
                "group_size": 64
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/\(repo)/tree/main",
            body: "[]"
        )

        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: repo)

        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.sourceFormat, .mlxAffine)
        XCTAssertEqual(result.hfRepo, repo)
        XCTAssertNil(result.diagnostic)
    }

    func testForgeProbePreservesRawFetchDiagnosticWhenAPIFallbackAlsoFails() async {
        let fake = FakeRunner()
        let repo = "someone/flaky-model"
        fake.errors.insert(
            "https://huggingface.co/\(repo)/resolve/main/config.json"
        )

        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: repo)

        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.message, "Couldn't fetch config.json.")
        XCTAssertEqual(result.diagnostic, URLError(.notConnectedToInternet).localizedDescription)
    }

    // MARK: - classifySourceFormat unit (config-only, no IO)

    func testClassifySourceFormatPrefersCompressedTensorsOverMlxHints() {
        let config: [String: Any] = [
            "quantization_config": ["quant_method": "compressed-tensors"],
            "mlx_lm_extra_tensors": ["mtp_file": "mtp.safetensors"]
        ]
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: config, hasMTP: true),
            .compressedTensorsAwq,
            "compressed-tensors is the dominant signal — those repos can't be loaded as MLX even if they ship an mtp marker"
        )
    }

    func testClassifySourceFormatDetectsAwqMethod() {
        let config: [String: Any] = [
            "quantization_config": ["quant_method": "awq"]
        ]
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: config, hasMTP: true),
            .compressedTensorsAwq
        )
    }

    func testClassifySourceFormatFallsBackToBf16NativeWhenMtpPresent() {
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: [:], hasMTP: true),
            .bf16Native
        )
    }

    func testClassifySourceFormatFallsBackToHfVllmWhenNoMtp() {
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: [:], hasMTP: false),
            .hfVllm
        )
    }

    // MARK: - Onboarding flow is untouched

    func testOnboardingProbeStillReturnsReadyForVanillaMtpRepo() async {
        // Sanity check that the extension didn't break the existing
        // onboarding contract.
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            { "num_nextn_predict_layers": 1 }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "Qwen/Qwen3.6-27B")
        XCTAssertEqual(result.verdict, .ready)
    }

    func testOnboardingProbeAlsoAcceptsHuggingFaceModelURL() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            { "num_nextn_predict_layers": 1 }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "https://huggingface.co/Qwen/Qwen3.6-27B/tree/main")
        XCTAssertEqual(result.verdict, .ready)
        XCTAssertEqual(result.hfRepo, "Qwen/Qwen3.6-27B")
    }

    // MARK: - Missing-config triage (GGUF and friends)

    func testProbeExplainsGGUFRepoAndSuggestsSource() async {
        let fake = FakeRunner()
        // config.json 404s (FakeRunner default); the metadata endpoint
        // says the repo exists, is GGUF, and names its source.
        fake.install(
            url: "https://huggingface.co/api/models/Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF",
            body: """
            {
              "id": "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF",
              "tags": [
                "gguf",
                "llama.cpp",
                "base_model:Jackrong/Qwopus3.6-27B-Coder",
                "base_model:quantized:Jackrong/Qwopus3.6-27B-Coder"
              ],
              "siblings": [ { "rfilename": "Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf" } ]
            }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "gguf_repo")
        XCTAssertTrue(result.message.contains("GGUF"))
        XCTAssertTrue(result.message.contains("made from Jackrong/Qwopus3.6-27B-Coder."))
        XCTAssertFalse(result.message.lowercased().contains("not found"))
    }

    func testProbeDetectsGGUFFromSiblingsAndPointsAtSourceGenerically() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/api/models/someone/quant-dump",
            body: """
            {
              "id": "someone/quant-dump",
              "tags": [],
              "siblings": [ { "rfilename": "model-Q5_K_S.GGUF" } ]
            }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "someone/quant-dump")
        XCTAssertEqual(result.diagnostic, "gguf_repo")
        XCTAssertTrue(result.message.contains("Paste the original repo"))
    }

    func testProbeMissingRepoSaysCheckTheName() async {
        // FakeRunner 404s everything: config.json AND the metadata
        // endpoint, which is the genuine-typo case.
        let fake = FakeRunner()
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "nosuch/repo-at-all")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "repo_not_found")
        XCTAssertTrue(result.message.contains("doesn't exist"))
    }

    func testProbeMissingRepoBehindAuthBlurSaysExistOrGated() async {
        // Live huggingface.co answers 401 (not 404) on the metadata
        // endpoint for repos that do not exist, deliberately blurring
        // missing and private. The message must cover both readings.
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/api/models/nosuch/repo-at-all",
            status: 401,
            body: "{}"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "nosuch/repo-at-all")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "repo_not_found_or_gated")
        XCTAssertTrue(result.message.contains("doesn't exist"))
        XCTAssertTrue(result.message.contains("private or gated"))
    }

    func testProbeExistingRepoWithoutConfigOrGGUFMentionsExport() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/api/models/someone/raw-export",
            body: """
            { "id": "someone/raw-export", "tags": ["pytorch"], "siblings": [ { "rfilename": "model.bin" } ] }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "someone/raw-export")
        XCTAssertEqual(result.diagnostic, "config_missing")
        XCTAssertTrue(result.message.contains("no config.json"))
    }

    // MARK: - Rootless pair bundles (official Gemma 4 layout)

    func testProbeAcceptsRootlessPairBundle() async {
        let fake = FakeRunner()
        // config.json 404s (FakeRunner default) — pair bundles have no
        // root config by design. The metadata listing names
        // mtplx_pair.json at the bundle root and both pair files fetch
        // cleanly, exactly the shape `_inspect_hf_model` accepts.
        fake.install(
            url: "https://huggingface.co/api/models/Youssofal/Gemma4-MTPLX-Optimized-Quality",
            body: """
            {
              "id": "Youssofal/Gemma4-MTPLX-Optimized-Quality",
              "tags": ["mlx"],
              "siblings": [
                { "rfilename": "mtplx_pair.json" },
                { "rfilename": "target/config.json" },
                { "rfilename": "draft/config.json" }
              ]
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/Youssofal/Gemma4-MTPLX-Optimized-Quality/resolve/main/mtplx_pair.json",
            body: """
            { "layout": { "target": "target", "assistant": "draft" } }
            """
        )
        fake.install(
            url: "https://huggingface.co/Youssofal/Gemma4-MTPLX-Optimized-Quality/resolve/main/target/config.json",
            body: """
            { "architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3" }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "Youssofal/Gemma4-MTPLX-Optimized-Quality")
        XCTAssertEqual(result.verdict, .ready)
        XCTAssertNil(result.diagnostic)
        XCTAssertTrue(result.message.contains("pair bundle"))
        XCTAssertTrue(
            result.message.contains("Gemma3ForConditionalGeneration"),
            "Architecture is extracted from target/config.json"
        )
    }

    func testProbePairBundleWithoutTargetConfigFallsBackToUnreadable() async {
        let fake = FakeRunner()
        // The listing advertises mtplx_pair.json and the manifest
        // fetches, but target/config.json 404s (FakeRunner default).
        // The daemon requires BOTH files, so the probe must fall back
        // to the existing unreadable classification.
        fake.install(
            url: "https://huggingface.co/api/models/someone/broken-pair",
            body: """
            { "id": "someone/broken-pair", "tags": [], "siblings": [ { "rfilename": "mtplx_pair.json" } ] }
            """
        )
        fake.install(
            url: "https://huggingface.co/someone/broken-pair/resolve/main/mtplx_pair.json",
            body: "{ }"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "someone/broken-pair")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "config_missing")
        XCTAssertTrue(result.message.contains("no config.json"))
    }

    func testProbePairBundleManifestNetworkErrorFallsBackToUnreadable() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/api/models/someone/flaky-pair",
            body: """
            { "id": "someone/flaky-pair", "tags": [], "siblings": [ { "rfilename": "mtplx_pair.json" } ] }
            """
        )
        fake.errors.insert(
            "https://huggingface.co/someone/flaky-pair/resolve/main/mtplx_pair.json"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "someone/flaky-pair")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "config_missing")
    }

    func testForgeProbeRoutesPairBundleToInstallInstead() async {
        let fake = FakeRunner()
        // No mtplx_runtime.json at the bundle root, so Forge falls
        // through to the config fetch and the 404 triage recognises
        // the pair bundle. That is a finished official artifact —
        // SourceStage must offer "Install instead", not a failure.
        fake.install(
            url: "https://huggingface.co/api/models/Youssofal/Gemma4-MTPLX-Optimized-Speed",
            body: """
            {
              "id": "Youssofal/Gemma4-MTPLX-Optimized-Speed",
              "tags": ["mlx"],
              "siblings": [ { "rfilename": "mtplx_pair.json" } ]
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/Youssofal/Gemma4-MTPLX-Optimized-Speed/resolve/main/mtplx_pair.json",
            body: """
            { "layout": { "target": "target", "assistant": "draft" } }
            """
        )
        fake.install(
            url: "https://huggingface.co/Youssofal/Gemma4-MTPLX-Optimized-Speed/resolve/main/target/config.json",
            body: """
            { "architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3" }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Youssofal/Gemma4-MTPLX-Optimized-Speed")
        XCTAssertEqual(result.verdict, .alreadyMTPLX)
        XCTAssertFalse(
            result.hasMtpWeights,
            "Pair bundles speculate via the draft model, not an MTP sidecar"
        )
        XCTAssertTrue(result.message.contains("Install it instead"))
    }

    func testForgeProbeSurfacesGGUFTriageMessage() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/api/models/Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF",
            body: """
            {
              "tags": ["gguf", "base_model:Jackrong/Qwopus3.6-27B-Coder"],
              "siblings": []
            }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "gguf_repo")
        XCTAssertTrue(result.message.contains("GGUF"))
    }

    func testProbeRoutesThroughConfiguredMirrorEndpoint() async {
        let fake = FakeRunner()
        // Only the mirror host serves config.json; reaching .noMTP
        // proves every probe URL was built against the mirror.
        fake.install(
            url: "https://hf-mirror.example/mirror-org/some-model/resolve/main/config.json",
            body: """
            { "num_mtp_modules": 0 }
            """
        )
        let probe = HuggingFaceProbe(endpoint: "https://hf-mirror.example/", runner: fake.runner())
        let result = await probe.probe(repo: "mirror-org/some-model")
        XCTAssertEqual(result.verdict, .noMTP)
    }
}
