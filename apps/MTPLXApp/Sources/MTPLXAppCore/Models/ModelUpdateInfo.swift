import Foundation

// MARK: - ModelUpdateInfo
//
// One row of `mtplx models --check --json` — the model-pack counterpart of a
// Sparkle appcast item. The engine compares each cached pack's pull marker
// (exact commit sha recorded at download) against the published revision
// (models manifest on mtplx.com, Hugging Face API fallback) and reports a
// state per pack. Decoded verbatim from the CLI's JSON so the app never
// reimplements the freshness logic.

public struct ModelUpdateInfo: Codable, Equatable, Sendable, Identifiable {
    public var id: String { repoID }

    public let repoID: String
    public let path: String?
    public let state: String
    public let localRevision: String?
    public let remoteRevision: String?
    public let source: String?
    public let note: String?
    public let minEngineVersion: String?
    public let updateBytes: Int64?
    public let changedFiles: [String]?

    public var isUpdateAvailable: Bool { state == "update-available" }
    public var requiresEngineUpdate: Bool { state == "engine-update-required" }

    /// "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed" -> the pack name.
    public var shortName: String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case path
        case state
        case localRevision = "local_revision"
        case remoteRevision = "remote_revision"
        case source
        case note
        case minEngineVersion = "min_engine_version"
        case updateBytes = "update_bytes"
        case changedFiles = "changed_files"
    }

    public init(
        repoID: String,
        path: String? = nil,
        state: String,
        localRevision: String? = nil,
        remoteRevision: String? = nil,
        source: String? = nil,
        note: String? = nil,
        minEngineVersion: String? = nil,
        updateBytes: Int64? = nil,
        changedFiles: [String]? = nil
    ) {
        self.repoID = repoID
        self.path = path
        self.state = state
        self.localRevision = localRevision
        self.remoteRevision = remoteRevision
        self.source = source
        self.note = note
        self.minEngineVersion = minEngineVersion
        self.updateBytes = updateBytes
        self.changedFiles = changedFiles
    }
}

public struct ModelUpdateCheckPayload: Codable, Equatable, Sendable {
    public let cacheDir: String?
    public let engineVersion: String?
    public let updatesAvailable: Int?
    public let models: [ModelUpdateInfo]

    enum CodingKeys: String, CodingKey {
        case cacheDir = "cache_dir"
        case engineVersion = "engine_version"
        case updatesAvailable = "updates_available"
        case models
    }
}
