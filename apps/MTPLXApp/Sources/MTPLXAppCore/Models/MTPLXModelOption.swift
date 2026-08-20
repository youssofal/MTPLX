import Foundation
import os

public struct MTPLXModelOption: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var displayName: String
    public var shortName: String
    public var detail: String
    public var hfModelID: String
    public var localCandidates: [String]
    public var aliases: [String]
    /// Approximate on-disk download size for the artifact in bytes.
    /// Used by the onboarding download step for percentage + ETA, and
    /// by `ModelFeasibility` for disk-space pre-flight (it multiplies
    /// by 2.5 to mirror the daemon's `required_download_free_bytes`).
    /// Measured from real on-disk symlink-resolved sizes (Speed) or HF
    /// staging manifests (Quality); FP16 is the exact sum of the
    /// published HF repo files (2026-07-03 audit — FP16 keeps INT4
    /// packs and only downcasts BF16 floats, so it tracks its sibling).
    public var sizeBytes: Int64
    /// Approximate runtime peak unified-memory cost in GiB at the
    /// daemon's default `sustained` profile and a 16k context.
    /// `ModelFeasibility` multiplies by 1.5 for the safe-fit ceiling.
    /// Quality is the measured value from `benchmark_summary.peak_gib`;
    /// Speed and FP16 are conservative estimates from on-disk weight
    /// size + typical KV-cache overhead because their runtime jsons
    /// don't carry a `peak_gib` field.
    public var peakMemoryGiB: Double
    /// Which Mac tiers the model is recommended for. Drives the green
    /// "Recommended for your Mac" badge on the model-pick step. M1/M2
    /// Speed → FP16 routing happens in `ModelPickStep`, not here.
    public var recommendedFor: [ChipTier]
    /// Target-only AR model (no native MTP head). The engine hard-rejects
    /// MTP loads for these (Laguna), so launches must carry `--no-mtp` and
    /// the depth/auto-tune lane must stay out of the serve command.
    public var arOnly: Bool
    public init(
        id: String,
        displayName: String,
        shortName: String,
        detail: String,
        hfModelID: String,
        localCandidates: [String],
        aliases: [String] = [],
        sizeBytes: Int64 = 0,
        peakMemoryGiB: Double = 0,
        recommendedFor: [ChipTier] = [],
        arOnly: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.shortName = shortName
        self.detail = detail
        self.hfModelID = hfModelID
        self.localCandidates = localCandidates
        self.aliases = aliases
        self.sizeBytes = sizeBytes
        self.peakMemoryGiB = peakMemoryGiB
        self.recommendedFor = recommendedFor
        self.arOnly = arOnly
    }

    enum CodingKeys: String, CodingKey {
        case id
        case displayName
        case shortName
        case detail
        case hfModelID
        case localCandidates
        case aliases
        case sizeBytes
        case peakMemoryGiB
        case recommendedFor
        case arOnly
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        displayName = try container.decode(String.self, forKey: .displayName)
        shortName = try container.decode(String.self, forKey: .shortName)
        detail = try container.decode(String.self, forKey: .detail)
        hfModelID = try container.decode(String.self, forKey: .hfModelID)
        localCandidates = try container.decode([String].self, forKey: .localCandidates)
        aliases = try container.decodeIfPresent([String].self, forKey: .aliases) ?? []
        sizeBytes = try container.decodeIfPresent(Int64.self, forKey: .sizeBytes) ?? 0
        peakMemoryGiB = try container.decodeIfPresent(Double.self, forKey: .peakMemoryGiB) ?? 0
        recommendedFor = try container.decodeIfPresent([ChipTier].self, forKey: .recommendedFor) ?? []
        arOnly = try container.decodeIfPresent(Bool.self, forKey: .arOnly) ?? false
    }

    /// True when `reference` (a model string from configuration: catalog id,
    /// alias, HF id, or a local path) resolves to a target-only AR model.
    /// Used by the command builder so every app launch path carries the
    /// correct `--no-mtp` shape without each caller re-deriving it.
    public static func isAROnlyReference(_ reference: String) -> Bool {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let lower = trimmed.lowercased()
        for option in MTPLXModelOption.officialCatalog where option.arOnly {
            if option.id.lowercased() == lower { return true }
            if option.hfModelID.lowercased() == lower { return true }
            if option.aliases.contains(where: { $0.lowercased() == lower }) { return true }
            let tail = (trimmed as NSString).lastPathComponent.lowercased()
            let hfTail = (option.hfModelID as NSString).lastPathComponent.lowercased()
            if !tail.isEmpty, tail == hfTail || tail == hfTail.replacingOccurrences(of: "/", with: "--") {
                return true
            }
            if option.localCandidates.contains(where: {
                Self.expand($0).lowercased() == Self.expand(trimmed).lowercased()
            }) {
                return true
            }
        }
        return false
    }

    public var resolvedReference: String {
        installedLocalPath ?? hfModelID
    }

    /// First `localCandidates` entry that is a **completely
    /// downloaded** MTPLX install on disk — has the metadata files,
    /// the MTP sidecar, AND every weight shard referenced by the
    /// safetensors index. Runtime launchability belongs to the daemon:
    /// the app should attempt the selected complete model and surface
    /// the real startup result.
    public var installedLocalPath: String? {
        guard Self.localModelScanEnabled else { return nil }
        for candidate in localCandidates {
            let expanded = Self.expand(candidate)
            guard FileManager.default.fileExists(atPath: expanded) else { continue }
            if Self.hasCompleteInstall(at: expanded) {
                return expanded
            }
        }
        return nil
    }

    public var isInstalled: Bool {
        installedLocalPath != nil
    }

    public var modelFamily: String {
        Self.modelFamily(for: hfModelID)
    }

    public var supportsOnboardingTune: Bool {
        Self.supportsOnboardingTune(family: modelFamily)
    }

    public var maxContextWindow: Int {
        Self.maxContextWindow(forFamily: modelFamily)
    }

    /// Lenient first-match for any local candidate directory that
    /// merely exists — used when we need to *recognise* an existing
    /// settings.json model path (which may be pointing at a partial
    /// install) without claiming it's loadable. Never use this as
    /// the source of truth for what to launch the daemon with.
    public var anyLocalCandidatePath: String? {
        guard Self.localModelScanEnabled else { return nil }
        for candidate in localCandidates {
            let expanded = Self.expand(candidate)
            if FileManager.default.fileExists(atPath: expanded) {
                return expanded
            }
        }
        return nil
    }

    private static var localModelScanEnabled: Bool {
        !environmentFlag("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN")
    }

    private static func environmentFlag(_ name: String) -> Bool {
        guard let rawPointer = getenv(name) else { return false }
        let raw = String(cString: rawPointer)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return ["1", "true", "yes", "on"].contains(raw)
    }

    /// Verify that `directory` contains a fully-downloaded MTPLX
    /// model. The contract mirrors the daemon's
    /// `REQUIRED_MTPLX_MODEL_FILES` check plus trunk-shard
    /// verification so partial downloads (HuggingFace uploads
    /// `model.safetensors.index.json` long before the actual shards
    /// land) don't masquerade as installed. Symlinked installs
    /// resolve transparently because `FileManager.fileExists` follows
    /// symlinks.
    public static func hasCompleteInstall(at directory: String) -> Bool {
        let fm = FileManager.default
        let url = URL(fileURLWithPath: directory)

        if fm.fileExists(atPath: url.appendingPathComponent("mtplx_pair.json").path) {
            let target = url.appendingPathComponent("target", isDirectory: true)
            let assistant = url.appendingPathComponent("assistant", isDirectory: true)
            return Self.hasCompleteModelDirectory(at: target)
                && Self.hasCompleteModelDirectory(at: assistant)
        }

        let coreFiles = ["config.json", "tokenizer.json", "mtplx_runtime.json"]
        for name in coreFiles {
            if !fm.fileExists(atPath: url.appendingPathComponent(name).path) {
                return false
            }
        }

        if !Self.hasMTPSidecar(at: url) {
            return false
        }

        return Self.hasCompleteWeightSet(at: url)
    }

    private static func hasMTPSidecar(at url: URL) -> Bool {
        let fm = FileManager.default
        for rel in Self.mtpSidecarCandidates(at: url) {
            if fm.fileExists(atPath: url.appendingPathComponent(rel).path) {
                return true
            }
        }
        return false
    }

    private static func mtpSidecarCandidates(at url: URL) -> [String] {
        var result: [String] = []
        if let configured = Self.configuredMTPSidecar(at: url) {
            result.append(configured)
        }
        result.append(contentsOf: ["mtp.safetensors", "mtp/weights.safetensors", "model-mtp.safetensors"])
        var seen = Set<String>()
        return result.filter { seen.insert($0).inserted }
    }

    private static func configuredMTPSidecar(at url: URL) -> String? {
        let configURL = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let extras = json["mlx_lm_extra_tensors"] as? [String: Any],
              let value = extras["mtp_file"] as? String,
              !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return nil
        }
        return value
    }

    private static func hasCompleteModelDirectory(at url: URL) -> Bool {
        let fm = FileManager.default
        for name in ["config.json", "tokenizer.json"] {
            if !fm.fileExists(atPath: url.appendingPathComponent(name).path) {
                return false
            }
        }
        return Self.hasCompleteWeightSet(at: url)
    }

    private static func hasCompleteWeightSet(at url: URL) -> Bool {
        let fm = FileManager.default
        // Trunk weights: a single-file model lives in `model.safetensors`;
        // every other build is sharded with an index that lists every
        // `model-XXXXX-of-NNNNN.safetensors` shard the loader needs.
        // Either path is acceptable; missing shards aren't.
        let singleWeights = url.appendingPathComponent("model.safetensors").path
        if fm.fileExists(atPath: singleWeights) {
            return true
        }

        let indexPath = url.appendingPathComponent("model.safetensors.index.json").path
        guard
            let data = fm.contents(atPath: indexPath),
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = parsed["weight_map"] as? [String: String]
        else {
            return false
        }

        let shardNames = Set(weightMap.values)
        guard !shardNames.isEmpty else { return false }
        for name in shardNames {
            if !fm.fileExists(atPath: url.appendingPathComponent(name).path) {
                return false
            }
        }
        return true
    }

    public func matches(_ model: String) -> Bool {
        let normalized = Self.normalized(model)
        let basename = Self.normalized(URL(fileURLWithPath: model).lastPathComponent)
        if normalized == Self.normalized(id) { return true }
        if normalized == Self.normalized(displayName) { return true }
        if normalized == Self.normalized(shortName) { return true }
        if normalized == Self.normalized(hfModelID) { return true }
        if aliases.contains(where: { Self.normalized($0) == normalized }) { return true }
        return localCandidates.contains { candidate in
            let expanded = Self.expand(candidate)
            return Self.normalized(expanded) == normalized
                || Self.normalized(URL(fileURLWithPath: expanded).lastPathComponent) == basename
        }
    }

    public static func modelsMatch(_ lhs: String, _ rhs: String) -> Bool {
        let normalizedLHS = normalized(lhs)
        let normalizedRHS = normalized(rhs)
        if normalizedLHS == normalizedRHS { return true }
        if let option = option(matching: lhs), option.matches(rhs) { return true }
        if let option = option(matching: rhs), option.matches(lhs) { return true }
        return URL(fileURLWithPath: lhs).lastPathComponent
            .lowercased()
            .contains(URL(fileURLWithPath: rhs).lastPathComponent.lowercased())
            || URL(fileURLWithPath: rhs).lastPathComponent
                .lowercased()
                .contains(URL(fileURLWithPath: lhs).lastPathComponent.lowercased())
    }

    // SYNC PAIR: mtplx/model_catalog.py mirrors this catalog (entries,
    // sizes, peak memory, RAM tiers in recommendationIDs below) so the CLI
    // offers the same models as the app. Update both sides together.
    public static let officialCatalog: [MTPLXModelOption] = [
        MTPLXModelOption(
            id: "qwen35-4b-optimized-speed",
            displayName: "Qwen 3.5 4B Optimized Speed",
            shortName: "Qwen 3.5 4B Optimized Speed",
            detail: "4-bit quantization. Fastest fit for smaller Macs.",
            hfModelID: "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Qwen3.5-4B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-qwen35-4b-optimized-speed",
                "qwen3.5-4b-mtplx-optimized-speed",
                "Qwen3.5 4B Optimized Speed",
                "Qwen 3.5 4B",
                "Small Qwen",
            ],
            sizeBytes: 2_567_456_776,
            peakMemoryGiB: 2.86,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-4b-optimized-quality",
            displayName: "Qwen 3.5 4B Optimized Quality",
            shortName: "Qwen 3.5 4B Optimized Quality",
            detail: "8-bit quantization. Highest-fidelity 4B; 2x MTP multiplier.",
            hfModelID: "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Quality",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.5-4B-MTPLX-Optimized-Quality",
                "~/.mtplx/models/Youssofal--Qwen3.5-4B-MTPLX-Optimized-Quality",
            ],
            aliases: [
                "mtplx-qwen35-4b-optimized-quality",
                "qwen3.5-4b-mtplx-optimized-quality",
                "Qwen3.5 4B Optimized Quality",
                "Qwen 3.5 4B Quality",
            ],
            sizeBytes: 4_576_426_401,
            peakMemoryGiB: 4.75,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-9b-optimized-speed",
            displayName: "Qwen 3.5 9B Optimized Speed",
            shortName: "Qwen 3.5 9B Optimized Speed",
            detail: "6-bit quantization. Strong small-Mac speed pick.",
            hfModelID: "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
                "~/Documents/MTPLX/models/Qwen3.5-9B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
            ],
            aliases: [
                "mtplx-qwen35-9b-optimized-speed",
                "mtplx-qwen35-9b-speed-6bit",
                "Qwen-Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
                "Qwen3.5-9B-MTPLX-Speed-6bit-OfficialCLI",
                "Qwen 3.5 9B Speed 6-bit",
                "Qwen 3.5 9B Speed",
            ],
            sizeBytes: 8_695_118_657,
            peakMemoryGiB: 10.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-9b-optimized-speed-fp16",
            displayName: "Qwen 3.5 9B Optimized Speed FP16",
            shortName: "Qwen 3.5 9B Optimized Speed FP16",
            detail: "FP16-friendly 9B speed artifact for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
                "~/.mtplx/models/Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
            ],
            aliases: [
                "mtplx-qwen35-9b-optimized-speed-fp16",
                "Qwen3.5 9B Optimized Speed FP16",
                "Qwen 3.5 9B Speed FP16",
            ],
            sizeBytes: 7_783_301_179,
            peakMemoryGiB: 10.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-bare-speed",
            displayName: "Qwen 3.8 27B Bare Speed",
            shortName: "Qwen 3.8 27B Bare Speed",
            detail: "Quickest burst chat speeds. Lower quality and slower on long coding tasks.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Bare-Speed",
            ],
            aliases: [
                "mtplx-qwen38-27b-bare-speed",
                "Qwen3.8 27B Bare Speed",
                "Qwen 3.8 Bare Speed",
                "Bare Speed",
            ],
            // Exact byte sum of the published HF repo files (2026-08-15 tree API).
            sizeBytes: 16_313_698_865,
            // Measured 2026-08-14: request-log MLX high-water 19.6 GiB during
            // quiet-window 2.4k-context serving (boot + Flappy arms + rung).
            peakMemoryGiB: 20.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-speed",
            displayName: "Qwen 3.8 27B Optimized Speed",
            shortName: "Qwen 3.8 27B Optimized Speed",
            detail: "4-bit dynamic quant. Great coding speeds and good quality. Recommended.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-qwen38-27b-optimized-speed",
                "Qwen3.8 27B Optimized Speed",
                "Qwen 3.8 Optimized Speed",
            ],
            // Exact byte sum of the published HF repo files (2026-08-15 tree API).
            sizeBytes: 20_703_484_600,
            // Measured 2026-08-14: request-log MLX high-water 24.6 GiB during
            // quiet-window 2.4k-context serving (boot + Flappy arms + rung).
            peakMemoryGiB: 25.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-quality",
            displayName: "Qwen 3.8 27B Optimized Quality",
            shortName: "Qwen 3.8 27B Optimized Quality",
            detail: "8-bit dynamic quant. Good coding speeds and perfect quality.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality",
            ],
            aliases: [
                "mtplx-qwen38-27b-optimized-quality",
                "Qwen3.8 27B Optimized Quality",
                "Qwen 3.8 Optimized Quality",
            ],
            // Exact byte sum of the published HF repo files (2026-08-15 tree API).
            sizeBytes: 29_972_712_041,
            // Measured 2026-08-14: request-log MLX high-water 32.9 GiB during
            // quiet-window 2.4k-context serving (boot + Flappy arms + rung).
            peakMemoryGiB: 33.0,
            recommendedFor: [.modernApple]
        ),
        // Qwen 3.8 FP16 precision siblings (built 2026-08-15): the same
        // quantized packs byte for byte, every 16-bit tensor cast
        // bf16 -> fp16, so M1 and M2 Macs (no native bf16) run the
        // identical model at full speed. They are the legacy tier's face
        // of the trio; the modern tier never sees them. Mirrors
        // model_catalog.OFFICIAL_CATALOG.
        MTPLXModelOption(
            id: "qwen38-27b-bare-speed-fp16",
            displayName: "Qwen 3.8 27B Bare Speed FP16",
            shortName: "Qwen 3.8 27B Bare Speed FP16",
            detail: "Quickest burst chat speeds. Lower quality and slower on long coding tasks. FP16 build for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
            ],
            aliases: [
                "mtplx-qwen38-27b-bare-speed-fp16",
                "Qwen3.8 27B Bare Speed FP16",
                "Qwen 3.8 Bare Speed FP16",
                "Bare Speed FP16",
            ],
            // Exact byte sum of the local sibling at build time (2026-08-15).
            sizeBytes: 16_314_182_467,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 20.0,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-speed-fp16",
            displayName: "Qwen 3.8 27B Optimized Speed FP16",
            shortName: "Qwen 3.8 27B Optimized Speed FP16",
            detail: "4-bit dynamic quant. Great coding speeds and good quality. FP16 build for M1 and M2 Macs. Recommended.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
            ],
            aliases: [
                "mtplx-qwen38-27b-optimized-speed-fp16",
                "Qwen3.8 27B Optimized Speed FP16",
                "Qwen 3.8 Optimized Speed FP16",
            ],
            // Exact byte sum of the local sibling at build time (2026-08-15).
            sizeBytes: 20_703_969_110,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 25.0,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-quality-fp16",
            displayName: "Qwen 3.8 27B Optimized Quality FP16",
            shortName: "Qwen 3.8 27B Optimized Quality FP16",
            detail: "8-bit dynamic quant. Good coding speeds and perfect quality. FP16 build for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
                "~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
            ],
            aliases: [
                "mtplx-qwen38-27b-optimized-quality-fp16",
                "Qwen3.8 27B Optimized Quality FP16",
                "Qwen 3.8 Optimized Quality FP16",
            ],
            // Exact byte sum of the local sibling at build time (2026-08-15).
            sizeBytes: 29_973_197_540,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 33.0,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed-v2",
            displayName: "Qwen 3.6 27B Optimized Speed V2",
            shortName: "Qwen 3.6 27B Optimized Speed V2",
            detail: "Much higher quality for coding. Dynamic 4-bit hybrid quantization keeps hand-tuned sensitive parts at up to 16-bit. Faster on long agent tasks, slightly larger, and a little slower for short chats.",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
                "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
                "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
            ],
            aliases: [
                "mtplx-qwen36-27b-optimized-speed-v2",
                "Qwen3.6 27B Optimized Speed V2",
                "Optimized Speed V2",
            ],
            sizeBytes: 19_887_455_619,
            peakMemoryGiB: 21.5,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed",
            displayName: "Qwen 3.6 27B Optimized Speed",
            shortName: "Qwen 3.6 27B Optimized Speed",
            detail: "Smaller 4-bit model. A little faster for short chats.",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-qwen36-27b-optimized-speed",
                "Qwen3.6 27B Optimized Speed",
                "Optimized Speed",
            ],
            sizeBytes: 16_419_081_846,
            peakMemoryGiB: 17.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed-fp16",
            displayName: "Qwen 3.6 27B Optimized Speed FP16",
            shortName: "Qwen 3.6 27B Optimized Speed FP16",
            detail: "FP16 speed artifact recommended for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
                "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
                "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
            ],
            aliases: [
                "mtplx-qwen36-27b-optimized-speed-fp16",
                "Qwen3.6 27B Optimized Speed FP16",
                "Optimized Speed FP16",
            ],
            sizeBytes: 16_419_644_366,
            peakMemoryGiB: 17.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-speed",
            displayName: "Qwen 3.6 35B-A3B Optimized Speed",
            shortName: "Qwen 3.6 35B-A3B Optimized Speed",
            detail: "4-bit quantization. Blazingly fast and quite smart.",
            hfModelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Flat4-CyanKiwiMTP-ForgeRepairClean",
                "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-qwen36-35b-a3b-optimized-speed",
                "Qwen3.6 35B-A3B Optimized Speed",
                "Qwen3.6 35B Speed",
                "Qwen3.6-35B-A3B-MTPLX-Optimized-Speed",
                "Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe",
                "Qwen3.6-35B-A3B-MTPLX-Flat4-CyanKiwiMTP-ForgeRepairClean",
            ],
            sizeBytes: 21_014_908_550,
            peakMemoryGiB: 28.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-speed-fp16",
            displayName: "Qwen 3.6 35B-A3B Optimized Speed FP16",
            shortName: "Qwen 3.6 35B-A3B Optimized Speed FP16",
            detail: "FP16-friendly 35B speed artifact for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
                "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16",
            ],
            aliases: [
                "mtplx-qwen36-35b-a3b-optimized-speed-fp16",
                "Qwen3.6 35B-A3B Optimized Speed FP16",
                "Qwen3.6 35B Speed FP16",
            ],
            sizeBytes: 21_016_116_512,
            peakMemoryGiB: 28.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-balance",
            displayName: "Qwen 3.6 35B-A3B Optimized Balance",
            shortName: "Qwen 3.6 35B-A3B Optimized Balance",
            detail: "6-bit quantization. Stronger balance of speed and quality.",
            hfModelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
                "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
            ],
            aliases: [
                "mtplx-qwen36-35b-a3b-optimized-balance",
                "Qwen3.6 35B-A3B Optimized Balance",
                "Qwen3.6 35B Balance",
            ],
            sizeBytes: 29_671_037_161,
            peakMemoryGiB: 32.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-balance-fp16",
            displayName: "Qwen 3.6 35B-A3B Optimized Balance FP16",
            shortName: "Qwen 3.6 35B-A3B Optimized Balance FP16",
            detail: "FP16-friendly 35B balance artifact for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16",
                "~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16",
            ],
            aliases: [
                "mtplx-qwen36-35b-a3b-optimized-balance-fp16",
                "Qwen3.6 35B-A3B Optimized Balance FP16",
                "Qwen3.6 35B Balance FP16",
            ],
            sizeBytes: 29_672_249_552,
            peakMemoryGiB: 32.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "gemma4-optimized-speed",
            displayName: "Gemma 4 31B Optimized Speed",
            shortName: "Gemma 4 31B Optimized Speed",
            detail: "High quality. Moderate speeds.",
            hfModelID: "Youssofal/Gemma4-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/Documents/MTPLX/models/hf-release/Gemma4-MTPLX-Optimized-Speed",
                "~/Documents/MTPLX/models/Gemma4-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Gemma4-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "Gemma4-MTPLX-Optimized-Speed",
                "Gemma4 Optimized Speed",
                "Gemma4 Speed",
                "gemma4-mtplx-optimized-speed",
                "mtplx/gemma4-mtplx-optimized-speed",
                "mtplx-gemma4-optimized-speed",
            ],
            sizeBytes: 17_715_574_395,
            peakMemoryGiB: 18.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-quality",
            displayName: "Qwen 3.6 27B Optimized Quality",
            shortName: "Qwen 3.6 27B Optimized Quality",
            detail: "Maximum quality. Moderate speeds.",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality",
            localCandidates: [
                "~/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Quality",
                "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality",
                "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality",
            ],
            aliases: [
                "mtplx-qwen36-27b-optimized-quality",
                "Qwen3.6 27B Optimized Quality",
                "Optimized Quality",
            ],
            sizeBytes: 30_016_961_493,
            peakMemoryGiB: 27.62,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-quality-fp16",
            displayName: "Qwen 3.6 27B Optimized Quality FP16",
            shortName: "Qwen 3.6 27B Optimized Quality FP16",
            detail: "FP16 quality artifact recommended for M1 and M2 Macs.",
            hfModelID: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
            localCandidates: [
                "~/Documents/MTPLX/hf-staging/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
                "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
            ],
            aliases: [
                "mtplx-qwen36-27b-optimized-quality-fp16",
                "Qwen3.6 27B Optimized Quality FP16",
                "Optimized Quality FP16",
            ],
            sizeBytes: 30_017_528_922,
            peakMemoryGiB: 28.12,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "laguna-s21-oq4e",
            displayName: "Laguna S-2.1 (community oQ4e)",
            shortName: "Laguna S-2.1",
            detail: "Poolside coding model, mixed-precision 4-bit. AR-only (no MTP head yet).",
            hfModelID: "mlx-community/Laguna-S-2.1-oQ4e",
            localCandidates: [
                "~/.mtplx/models/mlx-community--Laguna-S-2.1-oQ4e",
            ],
            aliases: [
                "mtplx-laguna-s21",
                "Laguna S-2.1",
                "Laguna-S-2.1-oQ4e",
            ],
            sizeBytes: 64_129_781_104,
            peakMemoryGiB: 74.0,
            recommendedFor: [.modernApple],
            arOnly: true
        ),
    ]

    public static func option(matching model: String) -> MTPLXModelOption? {
        officialCatalog.first { $0.matches(model) }
    }

    public static func pickerCatalog(
        customModels: [MTPLXModelOption],
        currentModel: String? = nil,
        hardware: DetectedHardware? = nil
    ) -> [MTPLXModelOption] {
        var rows = hardwareAwareOfficialCatalog(
            hardware: hardware,
            currentModel: currentModel
        )
        for custom in customModels {
            appendCustom(custom, to: &rows)
        }
        if let currentModel,
           option(matching: currentModel) == nil,
           let current = customHuggingFaceModel(repoID: currentModel)
        {
            appendCustom(current, to: &rows)
        }
        return rows
    }

    public static func hardwareAwareOfficialCatalog(
        hardware: DetectedHardware?,
        currentModel: String? = nil,
        includeInstalledOverrides: Bool = true
    ) -> [MTPLXModelOption] {
        var rows = recommendedCatalogIDs(for: hardware).compactMap(optionWithID)
        if let hardware {
            rows = rows.filter { option in
                shouldShowOfficialOption(option, hardware: hardware)
            }
        }
        if includeInstalledOverrides {
            for option in officialCatalog where option.isInstalled {
                if option.recommendedFor.isEmpty,
                   currentModel.map({ !option.matches($0) }) ?? true {
                    continue
                }
                appendUnique(option, to: &rows)
            }
        }
        if let currentModel {
            for option in officialCatalog where option.matches(currentModel) {
                appendUnique(option, to: &rows)
            }
        }
        return rows
    }

    /// Ordered fresh-user model matrix shared by the top-left picker
    /// and first-run onboarding. The rebuilt 4B pair (2026-07-19) leads
    /// the sub-16GB tiers and trails every larger modern tier: bigger
    /// Macs should be steered at the 27B/35B class first, but the tiny
    /// pair stays discoverable as the fast-small pick everywhere. No
    /// fp16 4B siblings exist yet, so the legacy (M1/M2) matrix keeps
    /// its fp16-only entries.
    public static func recommendedCatalogIDs(for hardware: DetectedHardware?) -> [String] {
        guard let hardware else { return modernTopRecommendationIDs }
        switch hardware.tier {
        case .intel:
            return []
        case .legacyApple:
            return recommendationIDs(
                memoryGiB: hardware.unifiedMemoryGiB,
                small: "qwen35-9b-optimized-speed-fp16",
                speed27: "optimized-speed-fp16",
                speed27V2: nil,
                speed35: "qwen36-35b-a3b-optimized-speed-fp16",
                balance35: "qwen36-35b-a3b-optimized-balance-fp16",
                quality27: "optimized-quality-fp16",
                trio38: qwen38TrioIDs.map { "\($0)-fp16" }
            )
        case .modernApple, .unknown:
            let tinyIDs = ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]
            if hardware.unifiedMemoryGiB < 16 {
                return tinyIDs
            }
            var ids = recommendationIDs(
                memoryGiB: hardware.unifiedMemoryGiB,
                small: "qwen35-9b-optimized-speed",
                speed27: "optimized-speed",
                speed27V2: "optimized-speed-v2",
                speed35: "qwen36-35b-a3b-optimized-speed",
                balance35: "qwen36-35b-a3b-optimized-balance",
                quality27: "optimized-quality",
                trio38: qwen38TrioIDs
            )
            ids.append(contentsOf: tinyIDs)
            return ids
        }
    }

    private static func shouldShowOfficialOption(
        _ option: MTPLXModelOption,
        hardware: DetectedHardware
    ) -> Bool {
        if hardware.tier == .intel { return false }
        return hardware.unifiedMemoryGiB >= option.peakMemoryGiB
    }

    private static let modernTopRecommendationIDs = [
        "qwen38-27b-optimized-speed",
        "qwen38-27b-bare-speed",
        "qwen38-27b-optimized-quality",
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "qwen35-9b-optimized-speed",
    ]

    /// Qwen 3.8 trio (2026-08-15 release): Optimized Speed is the
    /// recommended pick and leads every tier with at least 32 GiB, then
    /// Bare Speed and Optimized Quality (the latter drops out of the
    /// 32-33 GiB band via the peak-memory filter). The legacy (M1/M2) tier
    /// gets the same three picks as their FP16 precision siblings, same
    /// order. Mirrors model_catalog.recommended_catalog_ids.
    private static let qwen38TrioIDs = [
        "qwen38-27b-optimized-speed",
        "qwen38-27b-bare-speed",
        "qwen38-27b-optimized-quality",
    ]

    private static func recommendationIDs(
        memoryGiB: Double,
        small: String,
        speed27: String,
        speed27V2: String?,
        speed35: String,
        balance35: String,
        quality27: String,
        trio38: [String] = []
    ) -> [String] {
        if memoryGiB < 32 {
            return [small]
        }
        if memoryGiB < 48 {
            guard let speed27V2 else {
                return trio38 + [small, speed27, "gemma4-optimized-speed", speed35, quality27]
            }
            return trio38 + [speed27V2, speed27, small, "gemma4-optimized-speed", speed35, quality27]
        }
        return trio38 + (speed27V2.map { [$0] } ?? [])
            + [speed27, quality27, speed35, balance35, "gemma4-optimized-speed", small]
    }

    private static func optionWithID(_ id: String) -> MTPLXModelOption? {
        officialCatalog.first { $0.id == id }
    }

    private static func appendUnique(_ option: MTPLXModelOption, to rows: inout [MTPLXModelOption]) {
        guard !rows.contains(where: { $0.id == option.id }) else { return }
        rows.append(option)
    }

    public static func customHuggingFaceModel(repoID rawRepoID: String) -> MTPLXModelOption? {
        guard let repoID = normalizedHuggingFaceRepoID(rawRepoID) else { return nil }
        let repoName = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let safeID = repoID
            .lowercased()
            .replacingOccurrences(of: "/", with: "--")
            .replacingOccurrences(of: "_", with: "-")
        return MTPLXModelOption(
            id: "custom-\(safeID)",
            displayName: repoName,
            shortName: repoName,
            detail: "Custom Hugging Face model. MTPLX will use MTP when the repo includes a sidecar.",
            hfModelID: repoID,
            localCandidates: [
                "~/.mtplx/models/\(repoID.replacingOccurrences(of: "/", with: "--"))",
                "~/Documents/MTPLX/models/\(repoName)",
                "~/Documents/MTPLX/hf-staging/\(repoName)",
                "~/Documents/MTPLX/models/hf-release/\(repoName)",
            ],
            aliases: [repoID]
        )
    }

    /// Factory for a locally-forged model. Unlike `customHuggingFaceModel`,
    /// the artifact lives only on disk — there's no HF repo behind it
    /// (yet; the Publish flow can flip that later). `hfModelID` carries
    /// the branded name as the well-known identifier so:
    ///   (a) `pickerCatalog(customModels:)` dedup via `matches(_:)`
    ///       works against a settings.json that points at the local path
    ///   (b) the chrome strip's model label shows the branded name
    ///       rather than the absolute local path
    /// Returns nil for an empty branded name.
    public static func forgedModel(
        brandedName: String,
        localPath: String,
        sizeBytes: Int64 = 0,
        peakMemoryGiB: Double = 0
    ) -> MTPLXModelOption? {
        let trimmedName = brandedName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return nil }
        let safeID = trimmedName
            .lowercased()
            .replacingOccurrences(of: "/", with: "--")
            .replacingOccurrences(of: "_", with: "-")
        return MTPLXModelOption(
            id: "forged-\(safeID)",
            displayName: trimmedName,
            shortName: trimmedName,
            detail: "Forged locally with MTPLX Forge.",
            hfModelID: trimmedName,
            localCandidates: [localPath],
            aliases: [trimmedName, localPath],
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakMemoryGiB
        )
    }

    public static func normalizedHuggingFaceRepoID(_ rawValue: String) -> String? {
        var value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefix = "https://huggingface.co/"
        if value.lowercased().hasPrefix(prefix) {
            value = String(value.dropFirst(prefix.count))
        }
        if let queryIndex = value.firstIndex(where: { $0 == "?" || $0 == "#" }) {
            value = String(value[..<queryIndex])
        }
        let pathMarkers = ["/tree/", "/resolve/", "/blob/"]
        for marker in pathMarkers {
            if let range = value.range(of: marker) {
                value = String(value[..<range.lowerBound])
                break
            }
        }
        value = value.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let parts = value.split(separator: "/", omittingEmptySubsequences: true)
        guard parts.count == 2 else { return nil }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        for part in parts {
            guard part.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
                return nil
            }
        }
        return parts.joined(separator: "/")
    }

    /// Memoized: chrome-strip bodies call this on every 10 Hz metrics
    /// tick, and a miss used to walk the 23-entry catalog with 5+
    /// string normalizations per entry PLUS `URL(fileURLWithPath:)` —
    /// which syscalls `getcwd()` — inside a view body (2026-08-17 field
    /// regression). The result is a pure function of the id.
    private static let displayNameCache = OSAllocatedUnfairLock<[String: String]>(
        initialState: [:]
    )

    public static func displayName(for model: String) -> String {
        if let cached = displayNameCache.withLock({ $0[model] }) {
            return cached
        }
        let resolved: String
        if let option = option(matching: model) {
            resolved = option.displayName
        } else {
            // Plain path-tail split — no URL(fileURLWithPath:), which
            // hits the filesystem to resolve the working directory.
            let tail = model.split(separator: "/").last.map(String.init) ?? model
            resolved = tail.isEmpty ? model : tail
        }
        displayNameCache.withLock { cache in
            if cache.count > 512 { cache.removeAll() }
            cache[model] = resolved
        }
        return resolved
    }

    public static func displayName(for model: String, customModels: [MTPLXModelOption]) -> String {
        if let custom = customModels.first(where: { $0.matches(model) }) {
            return custom.displayName
        }
        return displayName(for: model)
    }

    public static func modelFamily(for model: String) -> String {
        // PRECISE forge provenance (source repo, assistant-pair marker)
        // outranks any name marker: a renamed or symlinked dir keeps the
        // family its artifact declares (engine F22 twin). Architecture ids
        // are NOT precise — every qwen3-next build of any version reports
        // the same arch — so arch/model_type-derived family stays a
        // fallback BELOW the name markers: a version token in the name
        // must beat the coarse arch mapping (the 3.5 9B pack is a
        // qwen3-next arch but a qwen3_5 family).
        if let preciseFamily = modelFamilyFromLocalMetadata(model, preciseOnly: true) {
            return preciseFamily
        }
        let normalized = Self.normalized(model)
            .replacingOccurrences(of: "_", with: "-")
        if normalized.contains("gemma4") || normalized.contains("gemma-4") {
            return "gemma4"
        }
        if matchesQwen38VersionToken(normalized) {
            return "qwen3_8"
        }
        if normalized.contains("qwen3.6") || normalized.contains("qwen36") || normalized.contains("qwen3-6") {
            return "qwen3_6"
        }
        if normalized == "qwen" || normalized.hasSuffix("/qwen") {
            return "qwen3_6"
        }
        if normalized.contains("qwen3.5") || normalized.contains("qwen35") || normalized.contains("qwen3-5") {
            return "qwen3_5"
        }
        if normalized.contains("step3.7") || normalized.contains("step-3.7")
            || normalized.contains("step3.5") || normalized.contains("step-3.5")
            || normalized.contains("step")
        {
            return "step"
        }
        if normalized.contains("deepseek") {
            return "deepseek"
        }
        if normalized.contains("laguna") {
            return "laguna"
        }
        if normalized.contains("glm") {
            return "glm"
        }

        // Coarse metadata (arch id, model_type): only when no name marker
        // resolved a version family above.
        if let metadataFamily = modelFamilyFromLocalMetadata(model, preciseOnly: false) {
            return metadataFamily
        }

        let marker = URL(fileURLWithPath: NSString(string: model).expandingTildeInPath)
            .appendingPathComponent("mtplx_pair.json")
            .path
        if FileManager.default.fileExists(atPath: marker) {
            return "gemma4"
        }
        return "unknown"
    }

    /// The 3.8 version token must not be a parameter count: stock
    /// `Qwen/Qwen3-8B` and `Qwen3-80B` (digit run ending in "b" right after
    /// the token) never claim the qwen3_8 family. Engine twin:
    /// descriptors.py `qwen3[._-]?8(?!\d*b)` (F21).
    private static func matchesQwen38VersionToken(_ dashNormalized: String) -> Bool {
        return dashNormalized.range(
            of: "qwen3[.-]?8(?![0-9]*b)",
            options: .regularExpression
        ) != nil
    }

    /// `preciseOnly: true` evaluates only the signals that identify one
    /// exact family (the assistant-pair marker and the forge source repo);
    /// `preciseOnly: false` evaluates only the coarse signals (arch ids and
    /// model_type, which are shared across model versions). The caller runs
    /// precise above the name markers and coarse below them.
    private static func modelFamilyFromLocalMetadata(_ model: String, preciseOnly: Bool) -> String? {
        let expanded = NSString(string: model).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return nil
        }

        if preciseOnly {
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("mtplx_pair.json").path) {
                return "gemma4"
            }
            if let runtime = MTPLXRuntimeMetadata.read(
                at: url.appendingPathComponent("mtplx_runtime.json").path
            ) {
                // Artifact-declared identity outranks its folder name and
                // the shared qwen3-next architecture id. Forge users are
                // free to brand a model "Bare Speed Beta"; the stable
                // public id / family must still expose the Qwen 3.8
                // reasoning contract without a curated catalog row.
                let declaredControls = runtime.rawJSON["model_controls"] as? [String: Any]
                for hint in [
                    runtime.modelFamily,
                    stringValue(declaredControls?["model_family"]),
                    runtime.publicModelID,
                    stringValue(runtime.rawJSON["served_model_id"]),
                    stringValue(runtime.rawJSON["model_id"]),
                    runtime.forgeProvenance?.sourceRepo,
                ].compactMap({ $0 }) {
                    let family = modelFamilyFromHint(hint)
                    if family != "unknown" { return family }
                }
            }
            return nil
        }

        if let runtime = MTPLXRuntimeMetadata.read(at: url.appendingPathComponent("mtplx_runtime.json").path) {
            if let archFamily = modelFamily(forArchitectureID: runtime.archId) {
                return archFamily
            }
        }

        guard let config = readJSONObject(at: url.appendingPathComponent("config.json").path) else {
            return nil
        }
        if let archFamily = modelFamily(forArchitectureID: stringValue(config["arch_id"]) ?? stringValue(config["architecture_id"])) {
            return archFamily
        }
        if let mtpArch = (config["mtp"] as? [String: Any]).flatMap({ stringValue($0["arch_id"]) ?? stringValue($0["architecture_id"]) ?? stringValue($0["mtp_arch"]) }),
           let archFamily = modelFamily(forArchitectureID: mtpArch)
        {
            return archFamily
        }
        if let modelType = stringValue(config["model_type"]) {
            return modelFamilyFromHint(modelType)
        }
        if let architectures = config["architectures"] as? [String] {
            for architecture in architectures {
                let family = modelFamilyFromHint(architecture)
                if family != "unknown" { return family }
            }
        }
        return nil
    }

    private static func modelFamily(forArchitectureID archID: String?) -> String? {
        guard let archID else { return nil }
        let normalized = archID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return nil }
        if normalized.contains("gemma") { return "gemma4" }
        if normalized.contains("step") { return "step" }
        if normalized.contains("deepseek") { return "deepseek" }
        if normalized.contains("glm") { return "glm" }
        if normalized.contains("qwen") { return "qwen3_6" }
        return nil
    }

    private static func modelFamilyFromHint(_ raw: String) -> String {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.contains("gemma") { return "gemma4" }
        if normalized.contains("step") { return "step" }
        if normalized.contains("deepseek") { return "deepseek" }
        if normalized.contains("glm") { return "glm" }
        if normalized.range(
            of: "qwen3[._-]?8(?![0-9]*b)",
            options: .regularExpression
        ) != nil {
            return "qwen3_8"
        }
        if normalized.contains("qwen3.5") || normalized.contains("qwen3_5")
            || normalized.contains("qwen3-5") || normalized.contains("qwen35")
        {
            return "qwen3_5"
        }
        if normalized.contains("qwen") { return "qwen3_6" }
        return "unknown"
    }

    private static func readJSONObject(at path: String) -> [String: Any]? {
        guard let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return json
    }

    private static func stringValue(_ value: Any?) -> String? {
        value as? String
    }

    public static func supportsTune(family: String) -> Bool {
        family == "qwen3_5" || family == "qwen3_6" || family == "qwen3_8"
    }

    public static func settingsFamiliesCompatible(stored: String, current: String) -> Bool {
        let storedFamily = stored.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentFamily = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if storedFamily.isEmpty || currentFamily.isEmpty { return false }
        if storedFamily == currentFamily { return true }
        return qwenDepthFamily(storedFamily) && qwenDepthFamily(currentFamily)
    }

    private static func qwenDepthFamily(_ family: String) -> Bool {
        family == "qwen3_5" || family == "qwen3_6"
    }

    public static func supportsOnboardingTune(family: String) -> Bool {
        supportsTune(family: family) || family == "gemma4"
    }

    public static func maxContextWindow(for model: String) -> Int {
        maxContextWindow(forFamily: modelFamily(for: model))
    }

    public static func maxContextWindow(forFamily family: String) -> Int {
        switch family {
        case "qwen3_5", "qwen3_6", "qwen3_8", "gemma4", "step", "glm", "deepseek":
            return 262_144
        default:
            return 262_144
        }
    }

    private static func expand(_ path: String) -> String {
        if path == "~" || path.hasPrefix("~/") {
            let fallback = NSHomeDirectory()
            let home = getenv("HOME").map { String(cString: $0) } ?? fallback
            let normalizedHome = home.isEmpty ? fallback : home
            if path == "~" { return normalizedHome }
            return URL(fileURLWithPath: normalizedHome)
                .appendingPathComponent(String(path.dropFirst(2)))
                .path
        }
        return (path as NSString).expandingTildeInPath
    }

    private static func appendCustom(_ custom: MTPLXModelOption, to rows: inout [MTPLXModelOption]) {
        guard option(matching: custom.hfModelID) == nil else { return }
        guard !rows.contains(where: { existing in
            existing.matches(custom.hfModelID) || custom.matches(existing.hfModelID)
        }) else { return }
        rows.append(custom)
    }

    private static func normalized(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
            .replacingOccurrences(of: "--", with: "/")
    }
}
