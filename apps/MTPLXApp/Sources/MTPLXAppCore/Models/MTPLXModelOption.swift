import Foundation
import os

public struct MTPLXModelOption: Codable, Equatable, Identifiable, Sendable {
    /// Match the engine's vision-spec probe: metadata and actual tower weights,
    /// never a model-name guess. The picker resolves downloaded aliases first.
    static func supportsVision(model: String) -> Bool {
        let explicitPath = NSString(string: model).expandingTildeInPath
        let path = FileManager.default.fileExists(atPath: explicitPath)
            ? explicitPath : (MTPLXModelOption.option(matching: model)?.installedLocalPath ?? model)
        let directory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        func object(_ name: String) -> [String: Any]? {
            guard let data = try? Data(contentsOf: directory.appendingPathComponent(name)) else { return nil }
            return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        }
        guard object("config.json")?["vision_config"] is [String: Any],
              let weights = object("model.safetensors.index.json")?["weight_map"] as? [String: String]
        else { return false }
        return weights.keys.contains { $0.hasPrefix("vision_tower.") || $0.hasPrefix("model.visual.") }
    }

    public var id: String
    public var displayName: String
    public var shortName: String
    /// Literal fallback text. Catalog and app-created rows also retain the
    /// English localization key separately so descriptions can resolve in
    /// whatever language is active when a picker renders.
    public var detail: String
    private var detailLocalizationKey: String?
    /// Set from the current folder scan. This state is never persisted in a model preference.
    public private(set) var isOnDisconnectedDrive = false
    public var hfModelID: String
    public var localCandidates: [String]
    public var aliases: [String]
    /// Approximate on-disk download size for the artifact in bytes.
    /// Used by the onboarding download step for percentage + ETA, and
    /// by `ModelFeasibility` for disk-space pre-flight (it adds the
    /// 5 GiB headroom `mtplx pull` keeps).
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
        self.detailLocalizationKey = nil
        self.hfModelID = hfModelID
        self.localCandidates = localCandidates
        self.aliases = aliases
        self.sizeBytes = sizeBytes
        self.peakMemoryGiB = peakMemoryGiB
        self.recommendedFor = recommendedFor
        self.arOnly = arOnly
    }

    public init(
        id: String,
        displayName: String,
        shortName: String,
        localizedDetailKey: String,
        hfModelID: String,
        localCandidates: [String],
        aliases: [String] = [],
        sizeBytes: Int64 = 0,
        peakMemoryGiB: Double = 0,
        recommendedFor: [ChipTier] = [],
        arOnly: Bool = false
    ) {
        self.init(
            id: id,
            displayName: displayName,
            shortName: shortName,
            detail: localizedDetailKey,
            hfModelID: hfModelID,
            localCandidates: localCandidates,
            aliases: aliases,
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakMemoryGiB,
            recommendedFor: recommendedFor,
            arOnly: arOnly
        )
        detailLocalizationKey = localizedDetailKey
    }

    /// A row whose pack is on a drive that is not plugged in says so in its
    /// detail line (already localized) instead of its catalog text.
    mutating func markNotConnected(_ text: String) {
        detail = text
        detailLocalizationKey = nil
        isOnDisconnectedDrive = true
    }

    /// Resolves only app-authored catalog text. Arbitrary details supplied by
    /// clients remain literal instead of accidentally becoming table keys.
    public var localizedDetail: String {
        guard let detailLocalizationKey else { return detail }
        return tr(detailLocalizationKey)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case displayName
        case shortName
        case detail
        case detailLocalizationKey
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
        detailLocalizationKey = try container.decodeIfPresent(String.self, forKey: .detailLocalizationKey)
            ?? Self.legacyDetailLocalizationKey(for: id)
        hfModelID = try container.decode(String.self, forKey: .hfModelID)
        localCandidates = try container.decode([String].self, forKey: .localCandidates)
        aliases = try container.decodeIfPresent([String].self, forKey: .aliases) ?? []
        sizeBytes = try container.decodeIfPresent(Int64.self, forKey: .sizeBytes) ?? 0
        peakMemoryGiB = try container.decodeIfPresent(Double.self, forKey: .peakMemoryGiB) ?? 0
        recommendedFor = try container.decodeIfPresent([ChipTier].self, forKey: .recommendedFor) ?? []
        arOnly = try container.decodeIfPresent(Bool.self, forKey: .arOnly) ?? false
    }

    /// Custom rows created before `detailLocalizationKey` shipped persisted
    /// the already-rendered text. Their stable id prefix identifies the
    /// semantic description without guessing from that old language.
    private static func legacyDetailLocalizationKey(for id: String) -> String? {
        if id.hasPrefix("custom-") {
            return "Custom Hugging Face model. MTPLX will use MTP when the repo includes a sidecar."
        }
        if id.hasPrefix("forged-") {
            return "Forged locally with MTPLX Forge."
        }
        if id.hasPrefix("local:") {
            return "Local MTPLX model in a configured library."
        }
        if id.hasPrefix("local-") {
            return "Local model folder on this Mac."
        }
        return nil
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

    public func resolvedReference(in library: ModelLibrary) -> String {
        installedLocalPath(in: library) ?? hfModelID
    }

    /// A folder the user chose on this Mac (`localFolderModel(path:)`):
    /// its only identity is its path, so the picker and the chrome label
    /// treat it differently from a forged or Hugging Face entry.
    public var isLocalFolder: Bool {
        id.hasPrefix("local-")
    }

    /// First `localCandidates` entry that is a **completely
    /// downloaded** MTPLX install on disk — has the metadata files,
    /// the MTP sidecar, AND every weight shard referenced by the
    /// safetensors index. Runtime launchability belongs to the daemon:
    /// the app should attempt the selected complete model and surface
    /// the real startup result.
    public var installedLocalPath: String? {
        installedLocalPath(in: .default)
    }

    public func installedLocalPath(in library: ModelLibrary) -> String? {
        guard Self.localModelScanEnabled else { return nil }
        var seen: Set<String> = []
        let candidates = library.candidatePaths(for: hfModelID) + localCandidates
        for candidate in candidates {
            let expanded = Self.expand(candidate)
            let canonical = ModelLibrary.canonicalURL(for: expanded).path
            guard seen.insert(canonical).inserted else { continue }
            guard FileManager.default.fileExists(atPath: expanded) else { continue }
            if Self.hasCompleteInstall(at: expanded), packIdentityProblem(at: expanded) == nil {
                return canonical
            }
        }
        return nil
    }

    /// What a pack says about itself in its own `mtplx-pack.json`.
    public struct PackIdentity: Equatable, Sendable {
        public var source: String
        public var recipe: String
        public var precisionVariant: String?
    }

    /// A folder that carries an Inkling pack's name is taken for that pack only when its
    /// mtplx-pack.json says it is that build: the source, the recipe, and the FP16 siblings'
    /// precision variant. The Mac the packs were built on kept two research builds under the
    /// published repository names (20 November), and a name alone served them as the catalog
    /// pack. The upstream revision is not pinned: a pack rebuilt from new weights moves it, and an
    /// older app would then refuse the new download. Mirrors `PACK_IDENTITIES` in
    /// mtplx/model_catalog.py.
    static let packIdentities: [String: PackIdentity] = {
        let source = "thinkingmachines/Inkling-2-Mini"
        return [
            "inkling2-mini-bare-speed": PackIdentity(source: source, recipe: "bare-speed"),
            "inkling2-mini-optimized-speed": PackIdentity(source: source, recipe: "speed"),
            "inkling2-mini-optimized-quality": PackIdentity(source: source, recipe: "quality"),
            "inkling2-mini-bare-speed-fp16": PackIdentity(source: source, recipe: "bare-speed", precisionVariant: "fp16"),
            "inkling2-mini-optimized-speed-fp16": PackIdentity(source: source, recipe: "speed", precisionVariant: "fp16"),
            "inkling2-mini-optimized-quality-fp16": PackIdentity(source: source, recipe: "quality", precisionVariant: "fp16"),
        ]
    }()

    /// Why the folder at `path` is not this pack although it carries its name; nil when it is,
    /// or when the catalog pins no identity for this pack. Mirrors `pack_identity_problem`.
    public func packIdentityProblem(at path: String) -> String? {
        guard let expected = Self.packIdentities[id] else { return nil }
        let stampURL = URL(fileURLWithPath: Self.expand(path), isDirectory: true)
            .appendingPathComponent("mtplx-pack.json")
        guard let data = try? Data(contentsOf: stampURL) else {
            return "it has no readable mtplx-pack.json, so it cannot be shown to be \(displayName)"
        }
        guard let stamp = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return "its mtplx-pack.json is not a pack description, so it cannot be shown to be \(displayName)"
        }
        func text(_ key: String) -> String? {
            guard let value = stamp[key] as? String, !value.isEmpty else { return nil }
            return value
        }
        let found = PackIdentity(
            source: text("source") ?? "",
            recipe: text("recipe") ?? "",
            precisionVariant: text("precision_variant")
        )
        if found == expected { return nil }
        let built = text("display_name")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let precision = found.precisionVariant.map { ", \($0)" } ?? ""
        let recipe = found.recipe.isEmpty ? "missing" : found.recipe
        return "its mtplx-pack.json says it is \(built.isEmpty ? "another build" : built) (recipe \(recipe)\(precision)), not \(displayName)"
    }

    public var isInstalled: Bool {
        installedLocalPath != nil
    }

    public func isInstalled(in library: ModelLibrary) -> Bool {
        installedLocalPath(in: library) != nil
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

        // Core completeness: a runnable checkpoint needs its config, a
        // tokenizer in any standard form, and every trunk shard.
        // mtplx_runtime.json and the MTP sidecar are MTPLX branding, not
        // load requirements — the engine serves unbranded checkpoints
        // autoregressive. Requiring them here turned byte-complete
        // third-party downloads into fake "incomplete download" failures
        // whose Retry could never succeed (issue #359).
        if !fm.fileExists(atPath: url.appendingPathComponent("config.json").path) {
            return false
        }
        let tokenizerForms = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
        let hasTokenizer = tokenizerForms.contains { name in
            fm.fileExists(atPath: url.appendingPathComponent(name).path)
        }
        if !hasTokenizer {
            return false
        }

        // True download completeness: every file the source repo is known
        // to ship (recorded in .mtplx-source.json at pull time) must
        // exist. A curated MTPLX repo therefore still requires its
        // runtime contract and MTP sidecar — because the repo actually
        // contains them — while a repo that never shipped them is not
        // punished for their absence.
        for name in Self.sourceMarkerFileList(at: url) {
            if !fm.fileExists(atPath: url.appendingPathComponent(name).path) {
                return false
            }
        }

        return Self.hasCompleteWeightSet(at: url)
    }

    private static func sourceMarkerFileList(at url: URL) -> [String] {
        let markerURL = url.appendingPathComponent(".mtplx-source.json")
        guard
            let data = try? Data(contentsOf: markerURL),
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let files = parsed["files"] as? [String: Any]
        else {
            return []
        }
        return Array(files.keys)
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
        guard nameMatches(model) else { return false }
        // A folder under this pack's name is this pack only if it says so (packIdentities).
        guard Self.packIdentities[id] != nil else { return true }
        var isDirectory: ObjCBool = false
        let path = Self.expand(model)
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return true }
        return packIdentityProblem(at: path) == nil
    }

    private func nameMatches(_ model: String) -> Bool {
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
            localizedDetailKey: "4-bit quantization. Fastest fit for smaller Macs.",
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
            sizeBytes: 2_567_456_768,
            peakMemoryGiB: 2.86,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-4b-optimized-quality",
            displayName: "Qwen 3.5 4B Optimized Quality",
            shortName: "Qwen 3.5 4B Optimized Quality",
            localizedDetailKey: "8-bit quantization. Highest-fidelity 4B; 2x MTP multiplier.",
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
            sizeBytes: 4_576_426_393,
            peakMemoryGiB: 4.75,
            recommendedFor: [.modernApple]
        ),
        // MiMo V2.6 Qwen 9B (2026-09-23): Xiaomi's supervised fine-tune of
        // Qwen3.5-9B for agentic coding (XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B,
        // MIT), packed like the 6-bit 9B: 6-bit group-64 body, the Qwen3.5-9B
        // MTP head and the vision tower in BF16. The architecture is plain
        // Qwen 3.5, so it takes the Qwen 3.5 contract, never the MiMo one. No
        // FP16 sibling exists, so M1 and M2 are not offered it. Mirrors
        // model_catalog.OFFICIAL_CATALOG.
        MTPLXModelOption(
            id: "mimo-v26-qwen-9b-optimized-speed",
            displayName: "MiMo V2.6 Qwen 9B Optimized Speed",
            shortName: "MiMo V2.6 Qwen 9B Optimized Speed",
            localizedDetailKey: "6-bit quantization. Xiaomi's agentic coding distill of Qwen 3.5 9B.",
            hfModelID: "Youssofal/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
                "~/Documents/MTPLX/models/MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-mimo-v26-qwen-9b-optimized-speed",
                "MiMo-V2.6-Qwen-9B-MTPLX-Optimized-Speed",
            ],
            // Published Hub bytes (2026-09-23). Same geometry and bytes as the
            // 6-bit 9B, so its peak carries over; measured 8.70 GiB at 15K tokens.
            sizeBytes: 8_695_116_595,
            peakMemoryGiB: 10.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-9b-optimized-speed",
            displayName: "Qwen 3.5 9B Optimized Speed",
            shortName: "Qwen 3.5 9B Optimized Speed",
            localizedDetailKey: "6-bit quantization. Strong small-Mac speed pick.",
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
            sizeBytes: 8_695_118_659,
            peakMemoryGiB: 10.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen35-9b-optimized-speed-fp16",
            displayName: "Qwen 3.5 9B Optimized Speed FP16",
            shortName: "Qwen 3.5 9B Optimized Speed FP16",
            localizedDetailKey: "FP16-friendly 9B speed artifact for M1 and M2 Macs.",
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
            sizeBytes: 7_783_301_181,
            peakMemoryGiB: 10.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-bare-speed",
            displayName: "Qwen 3.8 27B Bare Speed",
            shortName: "Qwen 3.8 27B Bare Speed",
            localizedDetailKey: "Quickest burst chat speeds. Lower quality and slower on long coding tasks.",
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
            sizeBytes: 16_313_700_871,
            // Measured 2026-08-14: request-log MLX high-water 19.6 GiB during
            // quiet-window 2.4k-context serving (boot + Flappy arms + rung).
            peakMemoryGiB: 20.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-speed",
            displayName: "Qwen 3.8 27B Optimized Speed",
            shortName: "Qwen 3.8 27B Optimized Speed",
            localizedDetailKey: "4-bit dynamic quant. Great coding speeds and good quality.",
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
            sizeBytes: 20_703_486_605,
            // Measured 2026-08-14: request-log MLX high-water 24.6 GiB during
            // quiet-window 2.4k-context serving (boot + Flappy arms + rung).
            peakMemoryGiB: 25.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-quality",
            displayName: "Qwen 3.8 27B Optimized Quality",
            shortName: "Qwen 3.8 27B Optimized Quality",
            localizedDetailKey: "8-bit dynamic quant. Good coding speeds and perfect quality.",
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
            sizeBytes: 29_972_714_046,
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
            localizedDetailKey: "Quickest burst chat speeds. Lower quality and slower on long coding tasks. FP16 build for M1 and M2 Macs.",
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
            sizeBytes: 16_314_184_539,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 20.0,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-speed-fp16",
            displayName: "Qwen 3.8 27B Optimized Speed FP16",
            shortName: "Qwen 3.8 27B Optimized Speed FP16",
            localizedDetailKey: "4-bit dynamic quant. Great coding speeds and good quality. FP16 build for M1 and M2 Macs.",
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
            sizeBytes: 20_703_971_173,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 25.0,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen38-27b-optimized-quality-fp16",
            displayName: "Qwen 3.8 27B Optimized Quality FP16",
            shortName: "Qwen 3.8 27B Optimized Quality FP16",
            localizedDetailKey: "8-bit dynamic quant. Good coding speeds and perfect quality. FP16 build for M1 and M2 Macs.",
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
            sizeBytes: 29_973_199_603,
            // Same packs and tensor bytes as the parent; peak carried over.
            peakMemoryGiB: 33.0,
            recommendedFor: [.legacyApple]
        ),
        // Qwen 3.8 Flash-Next (day-0 native, 2026-08-26): the 125B-A6B
        // Qwen4-generation preview (GDN hybrid MoE + Qwen Sparse Attention +
        // n-gram memory sidecar). The 32 GB n-gram table streams from SSD by
        // default, so resident peak is weights + MTP + working set, not the
        // full download size. Big-Mac exclusive: the peak-memory filter
        // hides both entries below ~96 GB unified memory.
        MTPLXModelOption(
            id: "flash-next-bare-speed",
            displayName: "Qwen 3.8 Flash-Next Bare Speed",
            shortName: "Flash-Next Bare Speed",
            localizedDetailKey: "Flat 4-bit quantization. Quickest Flash-Next speeds for chat and coding.",
            hfModelID: "Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
                "~/.mtplx/models/Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
            ],
            aliases: [
                "mtplx-flash-next-bare-speed",
                "Qwen3.8 Flash-Next Bare Speed",
                "Flash-Next Bare Speed",
                "Qwen3.8-Flash-Next-MTPLX-Bare-Speed",
            ],
            // Exact byte sum of the local pack (2026-08-27 audit); includes
            // the 32 GB SSD-streamed n-gram table and the vision tower.
            sizeBytes: 106_336_814_165,
            // Weights 72.6 GB + MTP 1.7 GB resident (n-gram on SSD) plus
            // KV/working headroom at the default profile.
            peakMemoryGiB: 78.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "flash-next-optimized-speed",
            displayName: "Qwen 3.8 Flash-Next Optimized Speed",
            shortName: "Flash-Next Optimized Speed",
            localizedDetailKey: "Dynamic 4-bit quant with 8-bit attention. Higher quality and slightly slower.",
            hfModelID: "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-flash-next-optimized-speed",
                "Qwen3.8 Flash-Next Optimized Speed",
                "Flash-Next Optimized Speed",
                "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
            ],
            // Exact byte sum of the local pack (2026-08-27 audit); includes
            // the 32 GB SSD-streamed n-gram table and the vision tower.
            sizeBytes: 115_061_255_377,
            // Weights 81.4 GB + MTP 1.7 GB resident (n-gram on SSD) plus
            // KV/working headroom at the default profile.
            peakMemoryGiB: 87.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "flash-next-optimized-quality",
            displayName: "Qwen 3.8 Flash-Next Optimized Quality",
            shortName: "Flash-Next Optimized Quality",
            localizedDetailKey: "8-bit body and MTP head, BF16 structural tensors, and a 4-bit n-gram table. Higher-fidelity Flash-Next build.",
            hfModelID: "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
                "~/.mtplx/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
            ],
            aliases: [
                "mtplx-flash-next-optimized-quality",
                "Qwen3.8-Flash-Next-MTPLX-Optimized-Quality",
                "Flash-Next Optimized Quality",
            ],
            // Recipe: body/MTP 8-bit group 64, BF16 structural tensors, n-gram 4-bit group 32.
            // Planner need at 128K with the n-gram table streamed from SSD (calculated,
            // not yet measured on a 256 GB Mac).
            sizeBytes: 169_958_537_520, peakMemoryGiB: 136.4,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "bonsai-2-27b-optimized-speed",
            displayName: "Bonsai 2 27B Optimized Speed",
            shortName: "Bonsai 2 27B Optimized Speed",
            localizedDetailKey: "Prism ML ternary 27B model with vision and MTP. Compact weights for smaller Macs.",
            hfModelID: "Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/.mtplx/models/Youssofal--Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Bonsai-3.8-27B-MTPLX-Optimized-Speed",
            ],
            aliases: [
                "mtplx-bonsai-2-27b-optimized-speed",
                "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
                "Bonsai-3.8-27B-MTPLX-Optimized-Speed",
                "mtplx-bonsai-38-27b-optimized-speed",
            ],
            // Published Hub bytes (2026-09-22) and the measured 8K context peak.
            sizeBytes: 8_847_919_618, peakMemoryGiB: 11.80,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed-v2",
            displayName: "Qwen 3.6 27B Optimized Speed V2",
            shortName: "Qwen 3.6 27B Optimized Speed V2",
            localizedDetailKey: "Much higher quality for coding. Dynamic 4-bit hybrid quantization keeps hand-tuned sensitive parts at up to 16-bit. Faster on long agent tasks, slightly larger, and a little slower for short chats.",
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
            sizeBytes: 19_887_455_621,
            peakMemoryGiB: 21.5,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed",
            displayName: "Qwen 3.6 27B Optimized Speed",
            shortName: "Qwen 3.6 27B Optimized Speed",
            localizedDetailKey: "Smaller 4-bit model. A little faster for short chats.",
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
            sizeBytes: 16_419_081_848,
            peakMemoryGiB: 17.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-speed-fp16",
            displayName: "Qwen 3.6 27B Optimized Speed FP16",
            shortName: "Qwen 3.6 27B Optimized Speed FP16",
            localizedDetailKey: "FP16 speed artifact for M1 and M2 Macs.",
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
            sizeBytes: 16_419_644_386,
            peakMemoryGiB: 17.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-speed",
            displayName: "Qwen 3.6 35B-A3B Optimized Speed",
            shortName: "Qwen 3.6 35B-A3B Optimized Speed",
            localizedDetailKey: "4-bit quantization. Blazingly fast and quite smart.",
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
            sizeBytes: 21_014_908_552,
            peakMemoryGiB: 28.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-speed-fp16",
            displayName: "Qwen 3.6 35B-A3B Optimized Speed FP16",
            shortName: "Qwen 3.6 35B-A3B Optimized Speed FP16",
            localizedDetailKey: "FP16-friendly 35B speed artifact for M1 and M2 Macs.",
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
            sizeBytes: 21_016_116_514,
            peakMemoryGiB: 28.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-balance",
            displayName: "Qwen 3.6 35B-A3B Optimized Balance",
            shortName: "Qwen 3.6 35B-A3B Optimized Balance",
            localizedDetailKey: "6-bit quantization. Stronger balance of speed and quality.",
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
            sizeBytes: 29_671_037_163,
            peakMemoryGiB: 32.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "qwen36-35b-a3b-optimized-balance-fp16",
            displayName: "Qwen 3.6 35B-A3B Optimized Balance FP16",
            shortName: "Qwen 3.6 35B-A3B Optimized Balance FP16",
            localizedDetailKey: "FP16-friendly 35B balance artifact for M1 and M2 Macs.",
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
            sizeBytes: 29_672_249_554,
            peakMemoryGiB: 32.5,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "gemma4-optimized-speed",
            displayName: "Gemma 4 31B Optimized Speed",
            shortName: "Gemma 4 31B Optimized Speed",
            localizedDetailKey: "High quality. Moderate speeds.",
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
            sizeBytes: 17_715_574_397,
            peakMemoryGiB: 18.0,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-quality",
            displayName: "Qwen 3.6 27B Optimized Quality",
            shortName: "Qwen 3.6 27B Optimized Quality",
            localizedDetailKey: "Maximum quality. Moderate speeds.",
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
            sizeBytes: 30_016_961_495,
            peakMemoryGiB: 27.62,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "optimized-quality-fp16",
            displayName: "Qwen 3.6 27B Optimized Quality FP16",
            shortName: "Qwen 3.6 27B Optimized Quality FP16",
            localizedDetailKey: "FP16 quality artifact for M1 and M2 Macs.",
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
            sizeBytes: 30_017_528_942,
            peakMemoryGiB: 28.12,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "laguna-s21-oq4e",
            displayName: "Laguna S-2.1 (community oQ4e)",
            shortName: "Laguna S-2.1",
            localizedDetailKey: "Poolside coding model, mixed-precision 4-bit. AR-only (no MTP head yet).",
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
        MTPLXModelOption(
            id: "inkling2-mini-bare-speed",
            displayName: "Inkling 2 Mini Bare Speed",
            shortName: "Inkling 2 Mini Bare Speed",
            localizedDetailKey: "Fastest Inkling 2 Mini: 4-bit attention and output, with vision and speculative decoding.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Bare-Speed",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Bare-Speed",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Bare-Speed",
            ],
            aliases: ["mtplx-inkling-2-mini-bare-speed", "Inkling 2 Mini Bare Speed", "MTPLX-Inkling-2-Mini-Bare-Speed", "Inkling-2-Mini-MTPLX-Bare-Speed"],
            sizeBytes: 16_023_243_308,
            peakMemoryGiB: 18.21,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "inkling2-mini-optimized-speed",
            displayName: "Inkling 2 Mini Optimized Speed",
            shortName: "Inkling 2 Mini Optimized Speed",
            localizedDetailKey: "Closer to the original weights: 8-bit attention, with vision and speculative decoding.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Optimized-Speed",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Speed",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Optimized-Speed",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Speed",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Optimized-Speed",
            ],
            aliases: ["mtplx-inkling-2-mini-optimized-speed", "Inkling 2 Mini Optimized Speed", "MTPLX-Inkling-2-Mini-Optimized-Speed", "Inkling-2-Mini-MTPLX-Optimized-Speed"],
            sizeBytes: 17_242_658_297,
            peakMemoryGiB: 19.92,
            recommendedFor: [.modernApple]
        ),
        MTPLXModelOption(
            id: "inkling2-mini-optimized-quality",
            displayName: "Inkling 2 Mini Optimized Quality",
            shortName: "Inkling 2 Mini Optimized Quality",
            localizedDetailKey: "Original expert and attention weights, with vision and speculative decoding.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Optimized-Quality",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Quality",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Optimized-Quality",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Optimized-Quality",
            ],
            aliases: ["mtplx-inkling-2-mini-optimized-quality", "Inkling 2 Mini", "MTPLX-Inkling-2-Mini-Optimized-Quality", "Inkling-2-Mini-MTPLX-Optimized-Quality"],
            sizeBytes: 20_191_646_421,
            peakMemoryGiB: 23.52,
            recommendedFor: [.modernApple]
        ),
        // FP16 siblings for M1 and M2 Macs (no native bfloat16): the same packs with every BF16
        // tensor cast to FP16. Recommended on the legacy tier (inklingLegacyOffer, on).
        MTPLXModelOption(
            id: "inkling2-mini-bare-speed-fp16",
            displayName: "Inkling 2 Mini Bare Speed FP16",
            shortName: "Inkling 2 Mini Bare Speed FP16",
            localizedDetailKey: "Fastest Inkling 2 Mini: 4-bit attention and output, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed-FP16",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Bare-Speed-FP16",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Bare-Speed-FP16",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed-FP16",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Bare-Speed-FP16",
            ],
            aliases: ["mtplx-inkling-2-mini-bare-speed-fp16", "Inkling 2 Mini Bare Speed FP16", "MTPLX-Inkling-2-Mini-Bare-Speed-FP16", "Inkling-2-Mini-MTPLX-Bare-Speed-FP16"],
            sizeBytes: 16_023_246_321,
            peakMemoryGiB: 18.21,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "inkling2-mini-optimized-speed-fp16",
            displayName: "Inkling 2 Mini Optimized Speed FP16",
            shortName: "Inkling 2 Mini Optimized Speed FP16",
            localizedDetailKey: "Closer to the original weights: 8-bit attention, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Optimized-Speed-FP16",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Speed-FP16",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Optimized-Speed-FP16",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Speed-FP16",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Optimized-Speed-FP16",
            ],
            aliases: ["mtplx-inkling-2-mini-optimized-speed-fp16", "Inkling 2 Mini Optimized Speed FP16", "MTPLX-Inkling-2-Mini-Optimized-Speed-FP16", "Inkling-2-Mini-MTPLX-Optimized-Speed-FP16"],
            sizeBytes: 17_242_661_312,
            peakMemoryGiB: 19.92,
            recommendedFor: [.legacyApple]
        ),
        MTPLXModelOption(
            id: "inkling2-mini-optimized-quality-fp16",
            displayName: "Inkling 2 Mini Optimized Quality FP16",
            shortName: "Inkling 2 Mini Optimized Quality FP16",
            localizedDetailKey: "Original expert and attention weights, with vision and speculative decoding. FP16 build for M1 and M2 Macs.",
            hfModelID: "Youssofal/Inkling-2-Mini-MTPLX-Optimized-Quality-FP16",
            localCandidates: [
                "~/.mtplx/models/MTPLX-Inkling-2-Mini-Optimized-Quality-FP16",
                "~/.mtplx/models/Youssofal--MTPLX-Inkling-2-Mini-Optimized-Quality-FP16",
                "~/.mtplx/models/Youssofal--Inkling-2-Mini-MTPLX-Optimized-Quality-FP16",
                "~/.mtplx/models/Inkling-2-Mini-MTPLX-Optimized-Quality-FP16",
            ],
            aliases: ["mtplx-inkling-2-mini-optimized-quality-fp16", "Inkling 2 Mini Optimized Quality FP16", "MTPLX-Inkling-2-Mini-Optimized-Quality-FP16", "Inkling-2-Mini-MTPLX-Optimized-Quality-FP16"],
            sizeBytes: 20_191_648_443,
            peakMemoryGiB: 23.52,
            recommendedFor: [.legacyApple]
        ),
    ]

    public static func option(matching model: String) -> MTPLXModelOption? {
        officialCatalog.first { $0.matches(model) }
    }

    public static func pickerCatalog(
        customModels: [MTPLXModelOption],
        currentModel: String? = nil,
        hardware: DetectedHardware? = nil,
        modelLibrary: ModelLibrary = .default
    ) -> [MTPLXModelOption] {
        var rows = hardwareAwareOfficialCatalog(
            hardware: hardware,
            currentModel: currentModel
        )
        for custom in customModels {
            appendCustom(custom, to: &rows)
        }
        // The model settings.json points at always has a row, even when
        // nothing remembers it yet: a pasted Hugging Face id, or a folder
        // chosen before the app remembered folders. Without the row the
        // user has to type the path again after every switch away.
        if let currentModel,
           let current = customHuggingFaceModel(repoID: currentModel)
               ?? localFolderModel(path: currentModel)
        {
            appendCustom(current, to: &rows)
        }
        // A discovered folder joins a catalog, remembered or current row when
        // it is that model: the row already lists the folder, or the folder's
        // name (see ModelLibrary) is the row's. Discovered folders never join
        // each other by name, so two builds keep two rows; only the same
        // install found under two library roots (same folder name, same
        // declared name) shares one row, first root first.
        var listedInstalls: [String: Int] = [:]
        for (index, row) in rows.enumerated() {
            for candidate in row.localCandidates {
                let path = ModelLibrary.canonicalURL(for: expand(candidate)).path
                if listedInstalls[path] == nil {
                    listedInstalls[path] = index
                }
            }
        }
        var discoveredRows: [String: Int] = [:]
        let discovered = modelLibrary.discoverCompleteModels()
        // Remember what each readable folder holds, so its packs stay listed
        // (as not connected) while its drive is unplugged.
        ModelFolderIndex.record(library: modelLibrary, discovered: discovered)
        for local in discovered {
            if let index = listedInstalls[local.path] ?? rows.indices.first(where: {
                !rows[$0].id.hasPrefix("local:") && rows[$0].matches(local.reference)
            }) {
                rows[index].localCandidates = insertLibraryPath(
                    local.path,
                    to: rows[index].localCandidates,
                    library: modelLibrary
                )
                continue
            }
            if var official = officialCatalog.first(where: { $0.matches(local.reference) }) {
                official.localCandidates = insertLibraryPath(
                    local.path,
                    to: official.localCandidates,
                    library: modelLibrary
                )
                appendUnique(official, to: &rows)
                continue
            }
            let installKey = normalized(local.reference) + "\n"
                + normalized(URL(fileURLWithPath: local.path).lastPathComponent)
            if let index = discoveredRows[installKey] {
                rows[index].localCandidates = insertLibraryPath(
                    local.path,
                    to: rows[index].localCandidates,
                    library: modelLibrary
                )
                continue
            }
            discoveredRows[installKey] = rows.count
            rows.append(MTPLXModelOption(
                id: "local:\(local.path)",
                displayName: local.displayName,
                shortName: local.displayName,
                localizedDetailKey: "Local MTPLX model in a configured library.",
                hfModelID: local.reference,
                localCandidates: [local.path],
                aliases: [local.reference, local.path]
            ))
        }
        // Packs last seen on a drive that is unplugged: listed and marked, so
        // they neither vanish nor look like something to download again.
        for remembered in ModelFolderIndex.rememberedPacks(library: modelLibrary) {
            let text = ModelFolderIndex.notConnectedDetail(driveName: remembered.driveName)
            if let index = rows.indices.first(where: {
                rows[$0].localCandidates.contains(remembered.model.path)
                    || rows[$0].matches(remembered.model.reference)
            }) {
                if rows[index].installedLocalPath(in: modelLibrary) == nil {
                    // A selected folder is synthesized again on each scan. Its pack metadata is now off the Mac.
                    if rows[index].isLocalFolder, rows[index].hfModelID == remembered.model.path,
                       !customModels.contains(where: { $0.id == rows[index].id }) {
                        rows[index].displayName = remembered.model.displayName
                        rows[index].shortName = remembered.model.displayName
                    }
                    rows[index].markNotConnected(text)
                }
                continue
            }
            var row = MTPLXModelOption(
                id: "local:\(remembered.model.path)",
                displayName: remembered.model.displayName,
                shortName: remembered.model.displayName,
                detail: text,
                hfModelID: remembered.model.reference,
                localCandidates: [remembered.model.path],
                aliases: [remembered.model.reference, remembered.model.path]
            )
            row.markNotConnected(text)
            rows.append(row)
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
    public static let bonsaiRecommendationMinGiB: Double = 16.0
    /// Inkling 2 Mini Bare Speed on 24 to 31 GiB Macs, directly after the tier's first pick. On
    /// (21 November). In a simulated 24 GiB envelope the memory plan serves a 24,576-token window,
    /// one in which every conversation can take its next turn (22 November): a 20K prompt, turns of
    /// 532 and 3,729 new tokens, an image, inside the 18 GiB engine budget (peak 17.92 GiB), no swap.
    /// True is the whole switch. Mirrors model_catalog.INKLING_TIGHT_MAC_OFFER.
    public static let inklingTightMacOffer = true
    /// The Inkling 2 Mini FP16 siblings on M1 and M2 Macs (32 GiB and up), where the modern tier
    /// gives the trio. On by the founder's order of 21 November; their FP16 numbers are checked on
    /// an M5 and the M1-M4 rehearsal switch, not yet on an M1 or M2 itself. True is the whole
    /// switch. Mirrors model_catalog.INKLING_LEGACY_OFFER.
    public static let inklingLegacyOffer = true

    public static func recommendedCatalogIDs(
        for hardware: DetectedHardware?,
        bonsaiMinimumGiB: Double = bonsaiRecommendationMinGiB
    ) -> [String] {
        guard let hardware else { return modernTopRecommendationIDs }
        switch hardware.tier {
        case .intel:
            return []
        case .legacyApple:
            var ids = recommendationIDs(
                memoryGiB: hardware.unifiedMemoryGiB,
                small: ["qwen35-9b-optimized-speed-fp16"],
                speed27: "optimized-speed-fp16",
                speed27V2: nil,
                speed35: "qwen36-35b-a3b-optimized-speed-fp16",
                balance35: "qwen36-35b-a3b-optimized-balance-fp16",
                quality27: "optimized-quality-fp16",
                trio38: qwen38TrioIDs.map { "\($0)-fp16" }
            )
            if inklingLegacyOffer && hardware.unifiedMemoryGiB >= 48 {
                ids.append(contentsOf: inklingIDs.map { "\($0)-fp16" })
            } else if inklingLegacyOffer && hardware.unifiedMemoryGiB >= 32 {
                ids.insert(contentsOf: inklingIDs.map { "\($0)-fp16" }, at: min(1, ids.count))
            }
            return ids
        case .modernApple, .unknown:
            let tinyIDs = ["qwen35-4b-optimized-speed", "qwen35-4b-optimized-quality"]
            if hardware.unifiedMemoryGiB < 16 {
                return tinyIDs
            }
            var ids = recommendationIDs(
                memoryGiB: hardware.unifiedMemoryGiB,
                small: modernSmallIDs,
                speed27: "optimized-speed",
                speed27V2: "optimized-speed-v2",
                speed35: "qwen36-35b-a3b-optimized-speed",
                balance35: "qwen36-35b-a3b-optimized-balance",
                quality27: "optimized-quality",
                trio38: qwen38TrioIDs
            )
            let bonsai = "bonsai-2-27b-optimized-speed"
            if hardware.unifiedMemoryGiB < 32 {
                // Bonsai leads from its bound and follows the 9B-class picks below it.
                let position = hardware.unifiedMemoryGiB >= bonsaiMinimumGiB ? 0 : modernSmallIDs.count
                ids.insert(bonsai, at: position)
            } else {
                ids.append(bonsai)
            }
            if hardware.unifiedMemoryGiB >= 48 {
                ids.append(contentsOf: inklingIDs)
            } else if hardware.unifiedMemoryGiB < 32 {
                if inklingTightMacOffer && hardware.unifiedMemoryGiB >= 24 {
                    ids.insert("inkling2-mini-bare-speed", at: min(1, ids.count))
                }
            } else {
                // From 32 to 47 GiB the Inkling trio rides directly after the tier's first pick.
                ids.insert(contentsOf: inklingIDs, at: min(1, ids.count))
            }
            // The raw list and the peak-filtered list both mirror Python.
            if hardware.unifiedMemoryGiB >= 96,
               let trioEnd = ids.lastIndex(where: { qwen38TrioIDs.contains($0) }) {
                ids.insert(contentsOf: flashNextIDs, at: ids.index(after: trioEnd))
            }
            if hardware.unifiedMemoryGiB < 128 {
                // Its wired floor (80.3 GiB) is above a 96 GB Mac's stock GPU limit (about 72 GiB).
                ids.removeAll { $0 == "flash-next-optimized-speed" }
            }
            if hardware.unifiedMemoryGiB >= 256 {
                // Optimized Speed leads and Optimized Quality follows, as the 27B trio does.
                let leading = ["flash-next-optimized-speed", "flash-next-optimized-quality"]
                ids.removeAll { leading.contains($0) }
                ids.insert(contentsOf: leading, at: 0)
            }
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
        "flash-next-bare-speed",
        "flash-next-optimized-speed",
        "flash-next-optimized-quality",
        "optimized-speed-v2",
        "optimized-speed",
        "optimized-quality",
        "qwen36-35b-a3b-optimized-speed",
        "qwen36-35b-a3b-optimized-balance",
        "gemma4-optimized-speed",
        "mimo-v26-qwen-9b-optimized-speed",
        "qwen35-9b-optimized-speed",
        "bonsai-2-27b-optimized-speed",
    ]

    /// The 9B-class speed picks on modern chips. MiMo V2.6 Qwen 9B rides
    /// immediately ahead of the Qwen 3.5 9B wherever the 9B is offered, so
    /// it is a top onboarding suggestion (founder, 2026-09-23) without
    /// replacing any tier's first pick. Mirrors model_catalog._MODERN_SMALL_IDS.
    private static let modernSmallIDs = [
        "mimo-v26-qwen-9b-optimized-speed",
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

    /// Inkling 2 Mini trio on modern chips (2.13): Bare Speed is the
    /// recommended pack. From 32 to 47 GiB the trio rides directly after the
    /// tier's first pick (founder, 2026-11-21: recommend Bare Speed to 36 GB
    /// Macs), Bare Speed, then Optimized Speed, then Optimized Quality; the
    /// peak-memory filter drops a pack where it does not fit. From 48 GiB it
    /// keeps the family's slot near the end; below 32 GiB only Bare Speed,
    /// behind inklingTightMacOffer. Mirrors model_catalog._INKLING_IDS.
    private static let inklingIDs = [
        "inkling2-mini-bare-speed",
        "inkling2-mini-optimized-speed",
        "inkling2-mini-optimized-quality",
    ]

    /// Flash-Next options on modern chips; Optimized Speed and Quality lead
    /// from 256 GiB. Mirrors model_catalog._FLASH_NEXT_IDS.
    private static let flashNextIDs = [
        "flash-next-bare-speed",
        "flash-next-optimized-speed",
        "flash-next-optimized-quality",
    ]

    private static func recommendationIDs(
        memoryGiB: Double,
        small: [String],
        speed27: String,
        speed27V2: String?,
        speed35: String,
        balance35: String,
        quality27: String,
        trio38: [String] = []
    ) -> [String] {
        if memoryGiB < 32 {
            return small
        }
        if memoryGiB < 48 {
            guard let speed27V2 else {
                return trio38 + small + [speed27, "gemma4-optimized-speed", speed35, quality27]
            }
            return trio38 + [speed27V2, speed27] + small + ["gemma4-optimized-speed", speed35, quality27]
        }
        return trio38 + (speed27V2.map { [$0] } ?? [])
            + [speed27, quality27, speed35, balance35, "gemma4-optimized-speed"] + small
    }

    private static func optionWithID(_ id: String) -> MTPLXModelOption? {
        officialCatalog.first { $0.id == id }
    }

    private static func appendUnique(_ option: MTPLXModelOption, to rows: inout [MTPLXModelOption]) {
        guard !rows.contains(where: { $0.id == option.id }) else { return }
        rows.append(option)
    }

    private static func insertLibraryPath(
        _ path: String,
        to paths: [String],
        library: ModelLibrary
    ) -> [String] {
        let canonical = ModelLibrary.canonicalURL(for: path).path
        let roots = Set(library.directories.map(\.path))
        var libraryPaths: [String] = []
        var fallbackPaths: [String] = []
        for existing in paths {
            let existingCanonical = ModelLibrary.canonicalURL(for: Self.expand(existing)).path
            guard existingCanonical != canonical else { continue }
            let parent = URL(fileURLWithPath: existingCanonical).deletingLastPathComponent().path
            if roots.contains(parent) {
                libraryPaths.append(existingCanonical)
            } else {
                fallbackPaths.append(existing)
            }
        }
        return libraryPaths + [canonical] + fallbackPaths
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
            localizedDetailKey: "Custom Hugging Face model. MTPLX will use MTP when the repo includes a sidecar.",
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
            localizedDetailKey: "Forged locally with MTPLX Forge.",
            hfModelID: trimmedName,
            localCandidates: [localPath],
            aliases: [trimmedName, localPath],
            sizeBytes: sizeBytes,
            peakMemoryGiB: peakMemoryGiB
        )
    }

    /// Factory for a model folder the user chose on this Mac. Like a
    /// forged model the artifact lives only on disk, so `hfModelID`
    /// carries the absolute path as the well-known identifier: `matches`
    /// finds the row from a settings.json that points at the folder, and
    /// selecting the row launches `--model <path>` exactly like
    /// onboarding's local pick. `~` and `file://` input canonicalize to
    /// one path so one folder is one entry. The folder's basename is the
    /// only name it has; a folder carrying a catalog model's folder name
    /// is folded into that catalog row by `pickerCatalog`. Returns nil
    /// unless `path` names a filesystem location.
    public static func localFolderModel(path rawPath: String) -> MTPLXModelOption? {
        guard let folder = canonicalFolderPath(rawPath) else { return nil }
        let name = (folder as NSString).lastPathComponent
        guard !name.isEmpty, name != "/" else { return nil }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: ".-"))
        let safeID = name.lowercased().unicodeScalars
            .map { allowed.contains($0) ? String($0) : "-" }
            .joined()
        let typed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        var aliases = [typed]
        if folder != typed {
            aliases.append(folder)
        }
        let displayName = ModelLibrary.packDisplayName(in: URL(fileURLWithPath: folder)) ?? name
        return MTPLXModelOption(
            id: "local-\(safeID)",
            displayName: displayName,
            shortName: displayName,
            localizedDetailKey: "Local model folder on this Mac.",
            hfModelID: folder,
            localCandidates: [folder],
            aliases: aliases
        )
    }

    /// Canonical absolute form of a user-supplied folder: a `file://` URL
    /// or `~` unwrapped, trailing slash dropped. Nil unless the input
    /// names a filesystem location (`/…`, `~…`, `file://…`) — a Hugging
    /// Face id is never a folder, so the two add-model parsers can never
    /// both accept one input.
    static func canonicalFolderPath(_ raw: String) -> String? {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.lowercased().hasPrefix("file://") {
            guard let url = URL(string: value), url.isFileURL else { return nil }
            value = url.path
        }
        guard value.hasPrefix("/") || value.hasPrefix("~") else { return nil }
        value = expand(value)
        while value.count > 1, value.hasSuffix("/") {
            value.removeLast()
        }
        return value
    }

    /// True when `path` (canonical, see `canonicalFolderPath`) is one of
    /// this option's local candidate directories.
    func hasLocalCandidate(at path: String) -> Bool {
        localCandidates.contains { Self.canonicalFolderPath($0) == path }
    }

    public static func normalizedHuggingFaceRepoID(_ rawValue: String) -> String? {
        var value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let prefix = "https://huggingface.co/"
        if value.lowercased().hasPrefix(prefix) {
            value = String(value.dropFirst(prefix.count))
        }
        // A filesystem path is never a repo id. Without this, "/Volumes/Foo"
        // survived the slash trim below as the repo "Volumes/Foo".
        if value.hasPrefix("/") || value.hasPrefix("~") {
            return nil
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
        // A chosen folder has no name beyond its basename, which is what
        // `displayName(for:)` derives anyway — and when the folder carries
        // a catalog model's name, that lookup answers with the catalog
        // name, matching the row the picker shows for it.
        if let custom = customModels.first(where: { $0.matches(model) }),
           !custom.isLocalFolder
        {
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
        // Flash-Next must resolve BEFORE the 3.8 version token: the pack
        // names carry "Qwen3.8-Flash-Next", which would otherwise be
        // swallowed into the dense-27B qwen3_8 contract (engine F21/#268
        // twin — the families share a marketing version, not a behavior
        // contract).
        if normalized.contains("flash-next") || normalized.contains("flashnext") {
            return "qwen4_exp"
        }
        if isBonsaiFamilyHint(normalized) || matchesQwen38VersionToken(normalized) {
            return "qwen3_8"
        }
        if isMiMoQwen35FamilyHint(normalized) {
            return "qwen3_5"
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
        if normalized.contains("inkling-2-mini") || normalized.contains("inkling2-mini") || normalized.contains("inkling2_loop") {
            return "inkling2"
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
                    stringValue(runtime.rawJSON["base_trunk"]),
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
        if normalized == "inkling2-loop" { return "inkling2" }
        if normalized.contains("gemma") { return "gemma4" }
        if normalized.contains("step") { return "step" }
        if normalized.contains("deepseek") { return "deepseek" }
        if normalized.contains("glm") { return "glm" }
        if normalized.contains("qwen4") { return "qwen4_exp" }
        if normalized.contains("qwen") { return "qwen3_6" }
        return nil
    }

    static func isBonsaiFamilyHint(_ text: String) -> Bool {
        ["bonsai-2-27b", "bonsai-3.8-27b", "bonsai-38-27b"].contains { text.contains($0) }
    }

    /// The MiMo V2.6 Qwen 9B catalog pack (repo and folder names, catalog
    /// and served ids) is Xiaomi's Qwen3.5-9B fine-tune, and none of its
    /// names carries a Qwen version token. "MiMo" names the publisher, not
    /// the architecture, so these resolve to qwen3_5 (engine twin:
    /// descriptors._MIMO_QWEN35_MARKERS).
    static func isMiMoQwen35FamilyHint(_ text: String) -> Bool {
        let dashed = text.lowercased().replacingOccurrences(of: "_", with: "-")
        return ["mimo-v2.6-qwen-9b", "mimo-v26-qwen-9b"].contains { dashed.contains($0) }
    }

    /// Memoized like the engine's lru_cached `_artifact_family_texts`: the
    /// effort picker asks on every render, and a miss resolves symlinks and
    /// reads mtplx_runtime.json.
    private static let bonsai2ModelCache = OSAllocatedUnfairLock<[String: Bool]>(
        initialState: [:]
    )

    public static func isBonsai2Model(_ model: String) -> Bool {
        if let cached = bonsai2ModelCache.withLock({ $0[model] }) {
            return cached
        }
        let resolved = resolveIsBonsai2Model(model)
        bonsai2ModelCache.withLock { cache in
            if cache.count > 512 { cache.removeAll() }
            cache[model] = resolved
        }
        return resolved
    }

    private static func resolveIsBonsai2Model(_ model: String) -> Bool {
        if isBonsaiFamilyHint(model.lowercased()) { return true }
        let path = URL(fileURLWithPath: NSString(string: model).expandingTildeInPath)
            .resolvingSymlinksInPath()
        if isBonsaiFamilyHint(path.path.lowercased()) { return true }
        guard let runtime = MTPLXRuntimeMetadata.read(
            at: path.appendingPathComponent("mtplx_runtime.json").path
        ) else { return false }
        return [runtime.publicModelID, runtime.forgeProvenance?.sourceRepo,
                stringValue(runtime.rawJSON["base_trunk"])]
            .compactMap { $0 }.contains { isBonsaiFamilyHint($0.lowercased()) }
    }

    private static func modelFamilyFromHint(_ raw: String) -> String {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.contains("inkling2loop") || normalized.contains("inkling2_loop") { return "inkling2" }
        if normalized.contains("gemma") { return "gemma4" }
        if normalized.contains("step") { return "step" }
        if normalized.contains("deepseek") { return "deepseek" }
        if normalized.contains("glm") { return "glm" }
        if normalized.contains("qwen4_exp") || normalized.contains("qwen4-exp")
            || normalized.contains("flash-next") || normalized.contains("flash_next")
        {
            return "qwen4_exp"
        }
        if isBonsaiFamilyHint(normalized) { return "qwen3_8" }
        if normalized.range(
            of: "qwen3[._-]?8(?![0-9]*b)",
            options: .regularExpression
        ) != nil {
            return "qwen3_8"
        }
        if isMiMoQwen35FamilyHint(normalized) { return "qwen3_5" }
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
        case "inkling2":
            return 1_048_576
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
        if let official = option(matching: custom.hfModelID) {
            // A chosen folder that carries a catalog model's folder name is
            // that model at a non-standard location (LM Studio's org/repo
            // layout, an external drive). The catalog row learns the folder
            // as an install location so selecting the row launches it: a
            // second row would duplicate the model, and the row's HF id on
            // its own is not cached on this Mac.
            if custom.isLocalFolder {
                adoptLocalFolder(custom, into: official, rows: &rows)
            }
            return
        }
        guard !rows.contains(where: { existing in
            existing.matches(custom.hfModelID) || custom.matches(existing.hfModelID)
        }) else { return }
        rows.append(custom)
    }

    private static func adoptLocalFolder(
        _ folder: MTPLXModelOption,
        into official: MTPLXModelOption,
        rows: inout [MTPLXModelOption]
    ) {
        let index = rows.firstIndex { $0.id == official.id }
        var row = index.map { rows[$0] } ?? official
        for path in folder.localCandidates where !row.localCandidates.contains(path) {
            row.localCandidates.append(path)
        }
        if let index {
            rows[index] = row
        } else {
            rows.append(row)
        }
    }

    private static func normalized(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
            .replacingOccurrences(of: "--", with: "/")
    }
}
