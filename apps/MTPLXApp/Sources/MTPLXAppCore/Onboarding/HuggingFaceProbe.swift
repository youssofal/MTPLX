import Foundation

// MARK: - HuggingFaceProbe
//
// Pre-download verification for the `Other` model-pick path. Reads
// the public `config.json` first for the MTP arch fields, then checks
// the Hugging Face tree listing for the `mtp.safetensors` sidecar file
// when needed. The probe deliberately avoids a separate auth/existence
// preflight: public model downloads should not be blocked by a brittle
// metadata endpoint.
//
// Verdict rules mirror the daemon's `_classify_scanned_model` at
// `mtplx/ui/onboarding.py:314-496` but for a remote repository:
//   .ready          arch supports MTP AND mtp.safetensors is published,
//                    OR the repo is a rootless assistant-pair bundle
//                    (mtplx_pair.json + target/config.json, the official
//                    Gemma 4 layout — mirrors `_inspect_hf_model` in
//                    `mtplx/artifacts.py`)
//   .missingSidecar arch supports MTP but no sidecar weights in tree
//   .noMTP          architecture does not declare MTP at all
//   .probeFailed    network / 404 / private-or-gated / malformed config

public struct HuggingFaceProbe: Sendable {
    /// Injectable for tests. Returns (httpStatusCode, body).
    public typealias HTTPRunner = @Sendable (URL, String) async throws -> (Int, Data)

    private let runner: HTTPRunner
    private let endpointBase: String

    /// `endpoint` is the user's HF download mirror from Settings
    /// (nil/empty = huggingface.co). The probe must follow it: on
    /// networks where huggingface.co is blocked, probing the blocked
    /// host would fail every Check & Add even though downloads work.
    public init(endpoint: String? = nil, runner: @escaping HTTPRunner = Self.defaultRunner) {
        self.endpointBase = Self.normalizedEndpoint(endpoint)
        self.runner = runner
    }

    static func normalizedEndpoint(_ raw: String?) -> String {
        var value = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        while value.hasSuffix("/") { value = String(value.dropLast()) }
        return value.isEmpty ? "https://huggingface.co" : value
    }

    // MARK: - Onboarding flow (unchanged)

    public func probe(repo rawRepo: String) async -> OtherModelProbe {
        guard let repo = Self.normalizedRepoID(rawRepo) else {
            return OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: rawRepo.trimmingCharacters(in: .whitespacesAndNewlines),
                message: "Paste a Hugging Face repo or link.",
                diagnostic: "invalid_repo_id"
            )
        }

        let configOutcome = await fetchConfig(repo: repo)
        let config: [String: Any]
        switch configOutcome {
        case .failed(let probe):
            return probe
        case .ok(let dict):
            config = dict
        }

        if !Self.configDeclaresMTP(config) {
            return OtherModelProbe(
                verdict: .noMTP,
                hfRepo: repo,
                message: "No MTP heads. You'll lose speculative decoding (~2-3× slower)."
            )
        }

        let hasSidecar = await checkSidecarPublished(repo: repo)
        if hasSidecar {
            return OtherModelProbe(
                verdict: .ready,
                hfRepo: repo,
                message: "MTP heads detected. Ready to download."
            )
        }
        return OtherModelProbe(
            verdict: .missingSidecar,
            hfRepo: repo,
            message: "Model declares MTP but mtp.safetensors isn't published. Speed will drop to standard decoding."
        )
    }

    // MARK: - Forge flow
    //
    // Returns a Forge-shaped verdict (`ForgeSourceProbe`) rather than
    // the onboarding-shaped `OtherModelProbe`. Forge has different
    // routing needs:
    //
    //   .alreadyMTPLX   repo already carries an mtplx_runtime.json,
    //                    so SourceStage swaps the primary CTA from
    //                    "Forge" to "Install instead"
    //   .forgeable      ready to forge, with the detected source
    //                    format threaded through so PlanStage can
    //                    auto-pick the right recipe
    //   .noMtpHeads     refuse — Forge cannot synthesize MTP heads
    //                    from nothing
    //   .probeFailed    network/404/private/malformed
    //
    // Implementation reuses the same fetchConfig +
    // checkSidecarPublished + configDeclaresMTP helpers as the
    // onboarding flow; the extra work is one HEAD-ish GET for the
    // mtplx_runtime.json marker and source-format classification
    // against `quantization_config` + `mlx_lm_extra_tensors` hints.
    // We deliberately do NOT pollute OtherModelProbe with Forge-
    // specific verdict cases — onboarding's switch statements stay
    // untouched and the two flows evolve independently.

    public func forgeProbe(repo rawRepo: String) async -> ForgeSourceProbe {
        guard let repo = Self.normalizedRepoID(rawRepo) else {
            return ForgeSourceProbe(
                verdict: .probeFailed,
                hfRepo: rawRepo.trimmingCharacters(in: .whitespacesAndNewlines),
                sourceFormat: .unknown,
                hasMtpWeights: false,
                message: "Paste a Hugging Face repo or link.",
                diagnostic: "invalid_repo_id"
            )
        }

        // Cheapest signal first: is this an already-MTPLX-branded
        // artifact? If yes, short-circuit before doing the heavier
        // config + tree fetches.
        if let runtimeJSON = await fetchMtplxRuntimeJSON(repo: repo) {
            let format: ForgeSourceFormat = runtimeJSON["mtp_sidecar"] != nil
                ? .mlxAffineWithMtp
                : .mlxAffine
            let depth = (runtimeJSON["mtp_depth_max"] as? Int) ?? 1
            return ForgeSourceProbe(
                verdict: .alreadyMTPLX,
                hfRepo: repo,
                sourceFormat: format,
                hasMtpWeights: true,
                message: "Already MTPLX-branded — depth \(depth) verified. Install instead of rebuilding.",
                diagnostic: nil
            )
        }

        let configOutcome = await fetchConfig(repo: repo)
        let config: [String: Any]
        switch configOutcome {
        case .failed(let probe):
            if probe.verdict == .ready {
                // The 404 triage recognised a rootless pair bundle —
                // an already-finished official MTPLX artifact whose
                // speculation comes from the draft model, not an MTP
                // sidecar. Forge routes it to install, same as the
                // mtplx_runtime.json short-circuit above; there is
                // nothing to rebuild.
                return ForgeSourceProbe(
                    verdict: .alreadyMTPLX,
                    hfRepo: repo,
                    sourceFormat: .unknown,
                    hasMtpWeights: false,
                    message: "Official MTPLX pair bundle (target + draft). Install it instead of rebuilding.",
                    diagnostic: nil
                )
            }
            return ForgeSourceProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                sourceFormat: .unknown,
                hasMtpWeights: false,
                message: probe.message,
                diagnostic: probe.diagnostic
            )
        case .ok(let dict):
            config = dict
        }

        let hasMTP = Self.configDeclaresMTP(config)
        if !hasMTP {
            return ForgeSourceProbe(
                verdict: .noMtpHeads,
                hfRepo: repo,
                sourceFormat: Self.classifySourceFormat(config: config, hasMTP: false),
                hasMtpWeights: false,
                message: "Architecture has no MTP heads. Forge cannot synthesize them — pick a model with `mtp_num_hidden_layers > 0` in config.json."
            )
        }

        let hasSidecar = await checkSidecarPublished(repo: repo)
        let format = Self.classifySourceFormat(config: config, hasMTP: true)
        let hasMtpWeights = hasSidecar || format == .mlxAffineWithMtp
        return ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: repo,
            sourceFormat: format,
            hasMtpWeights: hasMtpWeights,
            message: Self.forgeableMessage(format: format, hasSidecar: hasSidecar)
        )
    }

    private func fetchMtplxRuntimeJSON(repo: String) async -> [String: Any]? {
        await fetchRepoJSON(repo: repo, path: "mtplx_runtime.json")
    }

    /// GET `<endpoint>/<repo>/resolve/main/<path>` and decode a JSON
    /// object. `nil` on any failure — callers treat these fetches as
    /// best-effort signals, never hard errors.
    private func fetchRepoJSON(repo: String, path: String) async -> [String: Any]? {
        await fetchRepoJSON(repo: repo, revision: "main", path: path)
    }

    private func fetchRepoJSON(
        repo: String,
        revision: String,
        path: String
    ) async -> [String: Any]? {
        guard let url = URL(string: "\(endpointBase)/\(repo)/resolve/\(revision)/\(path)") else {
            return nil
        }
        do {
            let (status, body) = try await runner(url, "GET")
            guard status == 200 else { return nil }
            return try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        } catch {
            return nil
        }
    }

    /// Classify the source artifact's quantization layout from the
    /// `config.json` we already fetched. Detection priority:
    ///
    /// 1. `quantization_config.quant_method == "compressed-tensors"`
    ///    or weight format hints → compressed-tensors AWQ (vLLM/SGLang
    ///    target; cyankiwi pattern). Needs the AWQ → MLX-affine path.
    /// 2. `mlx_lm_extra_tensors.mtp_file` present → already an MLX
    ///    artifact with packed MTP sidecar. Verify-only.
    /// 3. `quantization` block with MLX-style fields → mlxAffine.
    /// 4. Default: BF16 native if MTP, otherwise `.hfVllm`.
    ///
    /// `unknown` is reserved for genuinely uncategorisable cases —
    /// PlanStage warns the user and defaults to the BF16-native
    /// recipe.
    static func classifySourceFormat(config: [String: Any], hasMTP: Bool) -> ForgeSourceFormat {
        if let quantConfig = config["quantization_config"] as? [String: Any] {
            let method = (quantConfig["quant_method"] as? String)?.lowercased() ?? ""
            let format = (quantConfig["format"] as? String)?.lowercased() ?? ""
            let isCompressedTensors = method == "compressed-tensors"
                || method == "compressed_tensors"
                || format.contains("compressed-tensors")
                || format.contains("compressed_tensors")
            let isAWQ = method == "awq" || method.contains("awq")
            if isCompressedTensors || isAWQ {
                return .compressedTensorsAwq
            }
        }
        if let extras = config["mlx_lm_extra_tensors"] as? [String: Any],
           extras["mtp_file"] != nil {
            return .mlxAffineWithMtp
        }
        if config["quantization"] is [String: Any] {
            return .mlxAffine
        }
        return hasMTP ? .bf16Native : .hfVllm
    }

    private static func forgeableMessage(format: ForgeSourceFormat, hasSidecar: Bool) -> String {
        switch format {
        case .compressedTensorsAwq:
            return "AWQ source detected (vLLM/SGLang format). Forge will convert the body to MLX-affine and extract the MTP sidecar."
        case .mlxAffineWithMtp:
            return "MLX-affine artifact with packed MTP sidecar. Forge will requantize the body and re-pack."
        case .mlxAffine:
            return "MLX-affine source. Forge will package the MTP sidecar from " + (hasSidecar ? "mtp.safetensors." : "the main shards.")
        case .bf16Native:
            return "BF16 source with MTP heads. Forge will quantize the body and keep MTP weights at BF16 (safest)."
        case .hfVllm:
            return "Hugging Face source. Forge will convert to MLX and pack the MTP sidecar."
        case .unknown:
            return "Source detected but format is unfamiliar. Forge will attempt a BF16-native recipe; review the Plan step carefully."
        }
    }

    private enum ConfigOutcome {
        case ok([String: Any])
        case failed(OtherModelProbe)
    }

    // MARK: - Step 1: GET <endpoint>/<repo>/resolve/main/config.json
    //
    // The Hub's raw-file route can occasionally stall long enough to
    // trip URLSession's short onboarding timeout, especially for
    // generated MLX configs with large per-tensor quantization maps.
    // On transport errors, transient HTTP failures, or malformed raw
    // responses, fall back to the model API's `config` expansion. The
    // raw file remains authoritative and auth/404 handling stays
    // unchanged.

    private func fetchConfig(repo: String) async -> ConfigOutcome {
        guard let url = URL(string: "\(endpointBase)/\(repo)/resolve/main/config.json") else {
            return .failed(OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "Could not build URL for config.json.",
                diagnostic: "url_build_failed"
            ))
        }
        do {
            let (status, body) = try await runner(url, "GET")
            switch status {
            case 200:
                break
            case 401, 403:
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "This repo is private or gated. Public Hugging Face models download without a login.",
                    diagnostic: "http_\(status)"
                ))
            case 404:
                // A missing config.json does not mean a missing repo:
                // GGUF exports carry no config.json at all, and they
                // are exactly what users paste when a model trends
                // with "MTP" in its name. One metadata GET separates
                // "typo" from "wrong format" so the message can help
                // instead of claiming the repo does not exist.
                return .failed(await classifyMissingConfig(repo: repo))
            default:
                if let config = await fetchConfigFromModelAPI(repo: repo) {
                    return .ok(config)
                }
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "config.json is unavailable (HTTP \(status)).",
                    diagnostic: "http_\(status)"
                ))
            }
            guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                if let config = await fetchConfigFromModelAPI(repo: repo) {
                    return .ok(config)
                }
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "config.json was malformed.",
                    diagnostic: "config_decode_failed"
                ))
            }
            return .ok(json)
        } catch {
            if let config = await fetchConfigFromModelAPI(repo: repo) {
                return .ok(config)
            }
            return .failed(OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "Couldn't fetch config.json.",
                diagnostic: error.localizedDescription
            ))
        }
    }

    /// Uses Hugging Face's compact model metadata to recover from a
    /// raw-file failure. The indexed config is accepted only when it
    /// carries a positive MTP marker; the Hub intentionally omits many
    /// model-specific fields from that representation, so treating its
    /// absence as "no MTP" would create a false negative. Prefer the
    /// returned immutable revision to retry the complete raw config,
    /// retaining the indexed config only as a last positive signal.
    /// Failures return nil so the caller preserves the original raw-file
    /// diagnostic.
    private func fetchConfigFromModelAPI(repo: String) async -> [String: Any]? {
        guard var components = URLComponents(string: "\(endpointBase)/api/models/\(repo)") else {
            return nil
        }
        components.queryItems = [
            URLQueryItem(name: "expand", value: "config"),
            URLQueryItem(name: "expand", value: "sha"),
        ]
        guard let url = components.url else { return nil }
        guard let (status, body) = try? await runner(url, "GET"), status == 200 else {
            return nil
        }
        guard let metadata = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return nil
        }
        let indexedConfig = (metadata["config"] as? [String: Any]) ?? [:]

        if let revision = metadata["sha"] as? String,
           !revision.isEmpty,
           revision.unicodeScalars.allSatisfy({
               CharacterSet.alphanumerics.contains($0)
           }),
           let config = await fetchRepoJSON(
               repo: repo,
               revision: revision,
               path: "config.json"
           ) {
            return config
        }
        return Self.configDeclaresMTP(indexedConfig) ? indexedConfig : nil
    }

    // MARK: - Missing-config triage

    /// Decides what a config.json 404 actually means by asking the
    /// repo metadata endpoint. Three honest outcomes instead of one
    /// wrong one: the repo is GGUF (llama.cpp format we cannot run),
    /// the repo exists but is unreadable, or the repo truly is not
    /// there.
    private func classifyMissingConfig(repo: String) async -> OtherModelProbe {
        let fallback = OtherModelProbe(
            verdict: .probeFailed,
            hfRepo: repo,
            message: "Repository or config.json not found on Hugging Face.",
            diagnostic: "http_404"
        )
        guard let url = URL(string: "\(endpointBase)/api/models/\(repo)") else {
            return fallback
        }
        guard let (status, body) = try? await runner(url, "GET") else {
            return fallback
        }
        switch status {
        case 200:
            break
        case 401, 403:
            // Live behavior: the metadata endpoint answers 401 for
            // repos that do not exist at all. HF deliberately blurs
            // missing and private, so the message covers both.
            return OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "This repo doesn't exist on Hugging Face, or it's private or gated. Check the name; public models download without a login.",
                diagnostic: "repo_not_found_or_gated"
            )
        case 404:
            return OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "This repo doesn't exist on Hugging Face. Check the name and try again.",
                diagnostic: "repo_not_found"
            )
        default:
            return fallback
        }
        let metadata = (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:]
        let tags = (metadata["tags"] as? [String]) ?? []
        let siblings = (metadata["siblings"] as? [[String: Any]]) ?? []
        // Rootless assistant-pair bundles (the official Gemma 4 repos)
        // publish NO root config.json by design: configs live under
        // target/ and draft/ with mtplx_pair.json at the bundle root.
        // The daemon accepts exactly this shape (`_inspect_hf_model`,
        // mtplx/artifacts.py): the file listing names mtplx_pair.json
        // and both the pair manifest and target/config.json load. The
        // probe must reach the same verdict instead of refusing what
        // the engine can run. Fetch/parse failures fall through to the
        // honest classifications below.
        let hasPairManifest = siblings.contains {
            (($0["rfilename"] as? String) ?? "") == "mtplx_pair.json"
        }
        if hasPairManifest, let pairBundle = await classifyPairBundle(repo: repo) {
            return pairBundle
        }
        let hasGGUF = tags.contains { $0.lowercased() == "gguf" }
            || siblings.contains {
                (($0["rfilename"] as? String) ?? "").lowercased().hasSuffix(".gguf")
            }
        if hasGGUF {
            let source = Self.baseModelSuggestion(
                tags: tags,
                cardData: metadata["cardData"] as? [String: Any]
            )
            let pointer = source.map {
                "This one was made from \($0). Paste that repo instead and Forge can convert it."
            } ?? "Paste the original repo it was made from instead and Forge can convert it."
            return OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "GGUF repos aren't supported: GGUF is llama.cpp's format and MTPLX runs MLX models. \(pointer)",
                diagnostic: "gguf_repo"
            )
        }
        return OtherModelProbe(
            verdict: .probeFailed,
            hfRepo: repo,
            message: "This repo exists but has no config.json, so MTPLX can't read it. If it's a converted export, paste the original repo instead.",
            diagnostic: "config_missing"
        )
    }

    /// Validates a rootless pair bundle the way the daemon does
    /// (`_inspect_hf_model`, mtplx/artifacts.py): the pair manifest
    /// AND target/config.json must BOTH fetch and parse before the
    /// repo is accepted. Returns `nil` on any failure so the caller
    /// falls back to the existing "exists but unreadable" outcome.
    private func classifyPairBundle(repo: String) async -> OtherModelProbe? {
        guard await fetchRepoJSON(repo: repo, path: "mtplx_pair.json") != nil,
              let targetConfig = await fetchRepoJSON(repo: repo, path: "target/config.json")
        else {
            return nil
        }
        // Mirror the daemon's identity extraction: architectures read
        // the root config first, then text_config; model_type checks
        // text_config first, then the root.
        let textConfig = (targetConfig["text_config"] as? [String: Any]) ?? targetConfig
        let architecture = (targetConfig["architectures"] as? [String])?.first
            ?? (textConfig["architectures"] as? [String])?.first
        let modelType = (textConfig["model_type"] as? String)
            ?? (targetConfig["model_type"] as? String)
        let described = (architecture ?? modelType).map { " (\($0) target + draft)" }
            ?? " (target + draft)"
        return OtherModelProbe(
            verdict: .ready,
            hfRepo: repo,
            message: "MTPLX pair bundle detected\(described). Ready to download."
        )
    }

    /// Pulls the source repo out of HF metadata. Plain
    /// `base_model:Org/Name` tags win; relation-prefixed tags
    /// (`base_model:quantized:Org/Name`) and cardData are fallbacks.
    static func baseModelSuggestion(tags: [String], cardData: [String: Any]?) -> String? {
        var relationFallback: String?
        for tag in tags {
            guard tag.lowercased().hasPrefix("base_model:") else { continue }
            let value = String(tag.dropFirst("base_model:".count))
            if let colon = value.firstIndex(of: ":") {
                let repoPart = String(value[value.index(after: colon)...])
                if relationFallback == nil, repoPart.contains("/") {
                    relationFallback = repoPart
                }
            } else if value.contains("/") {
                return value
            }
        }
        if let card = cardData?["base_model"] as? String, card.contains("/") {
            return card
        }
        if let cards = cardData?["base_model"] as? [String],
           let first = cards.first(where: { $0.contains("/") }) {
            return first
        }
        return relationFallback
    }

    // MARK: - Step 2: GET <endpoint>/api/models/<repo>/tree/main

    private func checkSidecarPublished(repo: String) async -> Bool {
        guard let url = URL(string: "\(endpointBase)/api/models/\(repo)/tree/main") else {
            return false
        }
        guard let (status, body) = try? await runner(url, "GET"), status == 200 else {
            // On any tree-listing failure we conservatively assume the
            // sidecar is missing — the user-visible verdict drops to
            // amber `.missingSidecar` rather than crashing the probe.
            return false
        }
        guard let entries = try? JSONSerialization.jsonObject(with: body) as? [[String: Any]] else {
            return false
        }
        for entry in entries {
            let path = (entry["path"] as? String)?.lowercased() ?? ""
            if path == "mtp.safetensors"
                || path == "model-mtp.safetensors"
                || path == "mtp/weights.safetensors"
                || path.hasSuffix("/mtp.safetensors")
                || (path.contains("/mtp/") && path.hasSuffix(".safetensors"))
            {
                return true
            }
        }
        return false
    }

    // MARK: - Architecture detection (config-only, no weight inspection)

    static func configDeclaresMTP(_ config: [String: Any]) -> Bool {
        // Three indicator fields the daemon recognises, in priority
        // order. Each is an Int > 0 on MTP-capable model configs.
        if let n = configInt(config, ["text_config", "mtp_num_hidden_layers"]), n > 0 {
            return true
        }
        if let n = configInt(config, ["num_nextn_predict_layers"]), n > 0 {
            return true
        }
        if let n = configInt(config, ["num_mtp_modules"]), n > 0 {
            return true
        }
        return false
    }

    /// Walks a key path inside a JSON dict and coerces the leaf to Int.
    /// Accepts both `Int` and `Double` leaf values because HF configs
    /// sometimes serialise integers as floats.
    private static func configInt(_ config: [String: Any], _ keys: [String]) -> Int? {
        var cursor: Any = config
        for key in keys {
            guard let dict = cursor as? [String: Any], let next = dict[key] else {
                return nil
            }
            cursor = next
        }
        if let int = cursor as? Int { return int }
        if let double = cursor as? Double { return Int(double) }
        return nil
    }

    // MARK: - Repo id sanity

    static func normalizedRepoID(_ rawValue: String) -> String? {
        MTPLXModelOption.normalizedHuggingFaceRepoID(rawValue)
    }

    static func looksLikeRepoID(_ s: String) -> Bool {
        normalizedRepoID(s) != nil
    }

    // MARK: - Default URLSession runner

    @Sendable
    public static func defaultRunner(_ url: URL, _ method: String) async throws -> (Int, Data) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        // Generated MLX configs can exceed 500 KB because they carry
        // per-tensor quantization maps. Twelve seconds produced false
        // "Unknown format" failures on otherwise public, forgeable
        // repos; keep the probe bounded but allow slow Hub responses.
        request.timeoutInterval = 30
        // Hugging Face honors a User-Agent and uses it to gate scraping
        // heuristics. Identify ourselves clearly so a probe failure is
        // traceable in HF logs back to MTPLX.
        request.setValue("MTPLXApp/1 OnboardingProbe", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, data)
    }
}
