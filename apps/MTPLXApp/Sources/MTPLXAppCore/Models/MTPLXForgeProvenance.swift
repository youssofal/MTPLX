import Foundation

// MARK: - MTPLXForgeProvenance
//
// Additive extension to the existing `mtplx_runtime.json` schema.
// Forge writes this block alongside the verified spine fields
// (mtplx_version, arch_id, mtp_depth_max, recommended_profile,
// verified_on, exactness_baseline, speed_evidence, mtp_sidecar,
// base_trunk, artifact_role); none of those are owned by Forge so
// none of them appear here.
//
// JSON shape (mirrors plan section 3.4 exactly):
//
//   {
//     "forge_provenance": {
//       "source_repo": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
//       "source_sha": "7a1c0c26c56ee56f98bfdb77124acf5b239eabf3",
//       "source_format": "compressed_tensors_awq",
//       "forge_recipe": {
//         "body_bits": 4,
//         "body_group_size": 64,
//         "body_mode": "affine",
//         "mtp_policy": "keep_bf16"
//       },
//       "forge_inputs": {
//         "trunk_path": "...",
//         "mtp_source_path": "..."
//       },
//       "forged_at": "2026-05-25T22:45:00+0100",
//       "mtplx_version": "0.x.x",
//       "forged_locally": true,
//       "published_to_hf": null
//     }
//   }
//
// `forged_locally: true` is the filter key for the My Models browser
// (anything in `~/Documents/MTPLX/models/` with this flag set is
// shown there). `published_to_hf` flips to a non-null object after
// a successful HF upload.

public struct MTPLXForgeProvenance: Codable, Equatable, Sendable {
    public var sourceRepo: String
    public var sourceSha: String?
    public var sourceFormat: ForgeSourceFormat
    public var forgeRecipe: ForgeRecipe
    public var forgeInputs: [String: String]
    /// ISO 8601 timestamp string. Kept as String (not Date) so the
    /// on-disk JSON matches the existing `verified_on.timestamp`
    /// convention (`"2026-05-02T02:23:23+0100"`) without imposing a
    /// per-decoder strategy.
    public var forgedAt: String
    public var mtplxVersion: String
    public var forgedLocally: Bool
    public var publishedToHf: PublishedToHF?

    public init(
        sourceRepo: String,
        sourceSha: String? = nil,
        sourceFormat: ForgeSourceFormat,
        forgeRecipe: ForgeRecipe,
        forgeInputs: [String: String] = [:],
        forgedAt: String,
        mtplxVersion: String,
        forgedLocally: Bool = true,
        publishedToHf: PublishedToHF? = nil
    ) {
        self.sourceRepo = sourceRepo
        self.sourceSha = sourceSha
        self.sourceFormat = sourceFormat
        self.forgeRecipe = forgeRecipe
        self.forgeInputs = forgeInputs
        self.forgedAt = forgedAt
        self.mtplxVersion = mtplxVersion
        self.forgedLocally = forgedLocally
        self.publishedToHf = publishedToHf
    }

    public struct PublishedToHF: Codable, Equatable, Sendable {
        public var repo: String
        public var revision: String?
        public var visibility: ForgePublishOptions.Visibility
        public var licenseSpdx: String
        public var uploadedAt: String

        public init(
            repo: String,
            revision: String? = nil,
            visibility: ForgePublishOptions.Visibility,
            licenseSpdx: String,
            uploadedAt: String
        ) {
            self.repo = repo
            self.revision = revision
            self.visibility = visibility
            self.licenseSpdx = licenseSpdx
            self.uploadedAt = uploadedAt
        }

        enum CodingKeys: String, CodingKey {
            case repo
            case revision
            case visibility
            case licenseSpdx = "license_spdx"
            case uploadedAt = "uploaded_at"
        }
    }

    enum CodingKeys: String, CodingKey {
        case sourceRepo = "source_repo"
        case sourceSha = "source_sha"
        case sourceFormat = "source_format"
        case forgeRecipe = "forge_recipe"
        case forgeInputs = "forge_inputs"
        case forgedAt = "forged_at"
        case mtplxVersion = "mtplx_version"
        case forgedLocally = "forged_locally"
        case publishedToHf = "published_to_hf"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sourceRepo = try c.decode(String.self, forKey: .sourceRepo)
        sourceSha = try c.decodeIfPresent(String.self, forKey: .sourceSha)
        sourceFormat = try c.decode(ForgeSourceFormat.self, forKey: .sourceFormat)
        forgeRecipe = try c.decode(ForgeRecipe.self, forKey: .forgeRecipe)
        forgeInputs = try c.decodeIfPresent([String: String].self, forKey: .forgeInputs) ?? [:]
        forgedAt = try c.decode(String.self, forKey: .forgedAt)
        mtplxVersion = try c.decode(String.self, forKey: .mtplxVersion)
        forgedLocally = try c.decodeIfPresent(Bool.self, forKey: .forgedLocally) ?? true
        publishedToHf = try c.decodeIfPresent(PublishedToHF.self, forKey: .publishedToHf)
    }
}

// MARK: - MTPLXRuntimeMetadata
//
// Reader-side view of an `mtplx_runtime.json` file. Models only the
// spine fields the frontend actually reads (My Models detail panel,
// Discover card chip derivations); everything else stays as raw JSON
// in `additionalFields` and is rendered by the generic
// RuntimeMetadataTable component.
//
// We deliberately do NOT model `exactness_baseline` or
// `speed_evidence` as Swift structs — both have shapes that drift
// across model variants (see Qwen3.6-27B-MTPLX-Flat4-CyanKiwiMTP vs
// Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP) and the frontend's job
// is to display them, not to interpret them.

// Sendability note: `rawJSON` is `[String: Any]` because that's what
// `JSONSerialization` returns and the table renderer wants to walk
// it directly. We never mutate it after `parse(_:)` — it's display
// payload only. `@unchecked Sendable` is the honest annotation for
// that pattern (Swift's type system can't prove the dict's deep
// immutability through `Any`, but the parser owns and freezes it).
public struct MTPLXRuntimeMetadata: Equatable, @unchecked Sendable {
    private final class ReadCache: @unchecked Sendable {
        struct Entry {
            let modificationDate: Date?
            let size: UInt64
            let metadata: MTPLXRuntimeMetadata
        }

        private let lock = NSLock()
        private var entries: [String: Entry] = [:]

        func value(for path: String, modificationDate: Date?, size: UInt64) -> MTPLXRuntimeMetadata? {
            lock.lock()
            defer { lock.unlock() }
            guard let entry = entries[path],
                  entry.modificationDate == modificationDate,
                  entry.size == size else { return nil }
            return entry.metadata
        }

        func store(_ metadata: MTPLXRuntimeMetadata, for path: String, modificationDate: Date?, size: UInt64) {
            lock.lock()
            entries[path] = Entry(
                modificationDate: modificationDate,
                size: size,
                metadata: metadata
            )
            lock.unlock()
        }

        func remove(_ path: String) {
            lock.lock()
            entries[path] = nil
            lock.unlock()
        }
    }

    private static let readCache = ReadCache()

    public var mtplxVersion: String?
    public var publicModelID: String?
    public var modelFamily: String?
    public var archId: String?
    public var mtpDepthMax: Int?
    public var recommendedProfile: String?
    public var artifactRole: String?
    public var baseTrunk: String?
    public var mtpSidecar: String?
    public var forgeProvenance: MTPLXForgeProvenance?

    /// The full raw JSON of the source file, used by RuntimeMetadataTable
    /// to show everything (including the un-modelled fields like
    /// `exactness_baseline`, `speed_evidence`, `verified_on`).
    public var rawJSON: [String: Any]

    public init(
        mtplxVersion: String? = nil,
        publicModelID: String? = nil,
        modelFamily: String? = nil,
        archId: String? = nil,
        mtpDepthMax: Int? = nil,
        recommendedProfile: String? = nil,
        artifactRole: String? = nil,
        baseTrunk: String? = nil,
        mtpSidecar: String? = nil,
        forgeProvenance: MTPLXForgeProvenance? = nil,
        rawJSON: [String: Any] = [:]
    ) {
        self.mtplxVersion = mtplxVersion
        self.publicModelID = publicModelID
        self.modelFamily = modelFamily
        self.archId = archId
        self.mtpDepthMax = mtpDepthMax
        self.recommendedProfile = recommendedProfile
        self.artifactRole = artifactRole
        self.baseTrunk = baseTrunk
        self.mtpSidecar = mtpSidecar
        self.forgeProvenance = forgeProvenance
        self.rawJSON = rawJSON
    }

    public static func == (lhs: MTPLXRuntimeMetadata, rhs: MTPLXRuntimeMetadata) -> Bool {
        lhs.mtplxVersion == rhs.mtplxVersion
            && lhs.publicModelID == rhs.publicModelID
            && lhs.modelFamily == rhs.modelFamily
            && lhs.archId == rhs.archId
            && lhs.mtpDepthMax == rhs.mtpDepthMax
            && lhs.recommendedProfile == rhs.recommendedProfile
            && lhs.artifactRole == rhs.artifactRole
            && lhs.baseTrunk == rhs.baseTrunk
            && lhs.mtpSidecar == rhs.mtpSidecar
            && lhs.forgeProvenance == rhs.forgeProvenance
        // rawJSON is intentionally excluded from equality — it's
        // a display payload, not part of the model identity.
    }

    /// Parses a runtime-metadata JSON dictionary into the typed view.
    /// Unknown fields are preserved in `rawJSON`. Returns nil only if
    /// the input is empty / clearly not a runtime metadata blob.
    public static func parse(_ json: [String: Any]) -> MTPLXRuntimeMetadata? {
        guard !json.isEmpty else { return nil }

        var provenance: MTPLXForgeProvenance?
        if let raw = json["forge_provenance"] as? [String: Any],
           let data = try? JSONSerialization.data(withJSONObject: raw) {
            provenance = try? JSONDecoder().decode(MTPLXForgeProvenance.self, from: data)
        }

        return MTPLXRuntimeMetadata(
            mtplxVersion: json["mtplx_version"] as? String,
            publicModelID: json["public_model_id"] as? String,
            modelFamily: json["model_family"] as? String,
            archId: json["arch_id"] as? String,
            mtpDepthMax: json["mtp_depth_max"] as? Int,
            recommendedProfile: json["recommended_profile"] as? String,
            artifactRole: json["artifact_role"] as? String,
            baseTrunk: json["base_trunk"] as? String,
            mtpSidecar: json["mtp_sidecar"] as? String,
            forgeProvenance: provenance,
            rawJSON: json
        )
    }

    /// Reads + parses a runtime-metadata file from disk. Returns nil
    /// on missing file, IO error, or invalid JSON.
    public static func read(at path: String) -> MTPLXRuntimeMetadata? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path) else {
            readCache.remove(path)
            return nil
        }
        let modificationDate = attributes[.modificationDate] as? Date
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        if let cached = readCache.value(
            for: path,
            modificationDate: modificationDate,
            size: size
        ) {
            return cached
        }
        guard let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let metadata = parse(json) else { return nil }
        readCache.store(
            metadata,
            for: path,
            modificationDate: modificationDate,
            size: size
        )
        return metadata
    }
}
