import Foundation

/// Ordered model storage policy for the app. The primary directory is the
/// only write target. Additional directories are read-only discovery roots.
public struct ModelLibrary: Equatable, Sendable {
    public struct LocalModel: Equatable, Identifiable, Sendable {
        public var id: String { path }
        public let path: String
        public let reference: String
        public let displayName: String

        public init(path: String, reference: String, displayName: String) {
            self.path = path
            self.reference = reference
            self.displayName = displayName
        }
    }

    public let primaryDirectory: URL
    public let additionalDirectories: [URL]
    public let huggingfaceHubCacheDirectory: URL?

    public var directories: [URL] {
        [primaryDirectory] + additionalDirectories
    }

    public init(
        primaryDirectory: String,
        additionalDirectories: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default,
        includeHuggingFaceCache: Bool? = nil
    ) {
        let fallback = Self.defaultPrimaryDirectory(environment: environment)
        let primary = Self.canonicalURL(
            for: primaryDirectory,
            fallback: fallback,
            environment: environment,
            fileManager: fileManager
        )
        var seen = Set([primary.path])
        var extras: [URL] = []
        for raw in additionalDirectories {
            let url = Self.canonicalURL(
                for: raw,
                fallback: fallback,
                environment: environment,
                fileManager: fileManager
            )
            guard seen.insert(url.path).inserted else { continue }
            extras.append(url)
        }
        self.primaryDirectory = primary
        self.additionalDirectories = extras
        let disableHF = ["1", "true", "yes", "on"].contains(
            (environment["MTPLX_DISABLE_HF_HOME_REUSE"] ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
        )
        let inXCTest = NSClassFromString("XCTestCase") != nil
            || environment["XCTestConfigurationFilePath"] != nil
        let scanHF = !disableHF && (
            includeHuggingFaceCache ?? (!inXCTest)
        )
        if scanHF {
            let rawHF = SharedModelFolders.huggingfaceHubCacheURL(environment: environment)
            let canonicalHF = Self.canonicalURL(
                for: rawHF.path,
                fallback: rawHF,
                environment: environment,
                fileManager: fileManager
            )
            self.huggingfaceHubCacheDirectory = seen.contains(canonicalHF.path) ? nil : canonicalHF
        } else {
            self.huggingfaceHubCacheDirectory = nil
        }
    }

    public static var `default`: ModelLibrary {
        ModelLibrary(primaryDirectory: defaultPrimaryDirectory().path)
    }

    public static func defaultPrimaryDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let override = environment["MTPLX_MODEL_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty
        {
            return expandedURL(for: override, environment: environment).standardizedFileURL
        }
        let home = environment["HOME"].flatMap { $0.isEmpty ? nil : $0 } ?? NSHomeDirectory()
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .standardizedFileURL
    }

    public static func canonicalURL(
        for rawPath: String,
        fallback: URL = defaultPrimaryDirectory(),
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL {
        let trimmed = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expanded = trimmed.isEmpty ? fallback : expandedURL(for: trimmed, environment: environment)
        let absolute: URL
        if expanded.path.hasPrefix("/") {
            absolute = expanded
        } else {
            absolute = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent(expanded.path, isDirectory: true)
        }
        let standardized = absolute.standardizedFileURL
        guard fileManager.fileExists(atPath: standardized.path) else { return standardized }
        return standardized.resolvingSymlinksInPath().standardizedFileURL
    }

    public static func isAvailable(
        _ directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    public func candidatePaths(for repoID: String) -> [String] {
        let safeName = repoID.replacingOccurrences(of: "/", with: "--")
        var paths: [String] = []
        for root in directories {
            paths.append(root.appendingPathComponent(safeName, isDirectory: true).path)
            if let snap = Self.resolveHFSnapshot(repoID: repoID, in: root) {
                paths.append(snap.path)
            }
        }
        if let hfRoot = huggingfaceHubCacheDirectory,
           let snap = Self.resolveHFSnapshot(repoID: repoID, in: hfRoot) {
            paths.append(snap.path)
        }
        return paths
    }

    static func resolveAnyHFSnapshot(
        inRepoDirectory repoDir: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        let snapshotsDir = repoDir.appendingPathComponent("snapshots", isDirectory: true)
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: snapshotsDir.path, isDirectory: &isDir),
              isDir.boolValue else { return nil }
        var ordered: [URL] = []
        var seen = Set<String>()
        let refsDir = repoDir.appendingPathComponent("refs", isDirectory: true)
        if let refFiles = try? fileManager.contentsOfDirectory(
            at: refsDir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            let sortedRefs = refFiles.sorted { a, b in
                if a.lastPathComponent == "main" { return true }
                if b.lastPathComponent == "main" { return false }
                return a.lastPathComponent < b.lastPathComponent
            }
            for refFile in sortedRefs {
                guard let rev = try? String(contentsOf: refFile, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                      !rev.isEmpty,
                      !rev.contains("/"),
                      !rev.contains("\\"),
                      rev != ".", rev != ".." else { continue }
                let snap = snapshotsDir.appendingPathComponent(rev, isDirectory: true).standardizedFileURL
                if seen.insert(snap.path).inserted {
                    ordered.append(snap)
                }
            }
        }
        if let snapDirs = try? fileManager.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) {
            let byDate = snapDirs.sorted { a, b in
                let dA = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let dB = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return dA > dB
            }
            for snap in byDate {
                let std = snap.standardizedFileURL
                if seen.insert(std.path).inserted {
                    ordered.append(std)
                }
            }
        }
        return ordered.first { Self.isAvailable($0, fileManager: fileManager) }
    }

    public static func resolveHFSnapshot(
        repoID: String,
        in hubCache: URL,
        fileManager: FileManager = .default
    ) -> URL? {
        let safeName = repoID.replacingOccurrences(of: "/", with: "--")
        let repoDir = hubCache.appendingPathComponent("models--\(safeName)", isDirectory: true)
        guard Self.isAvailable(repoDir, fileManager: fileManager) else { return nil }
        let snapshotsDir = repoDir.appendingPathComponent("snapshots", isDirectory: true)
        var candidates: [URL] = []
        var seen = Set<String>()
        let mainRef = repoDir.appendingPathComponent("refs", isDirectory: true)
            .appendingPathComponent("main")
        if let rev = try? String(contentsOf: mainRef, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !rev.isEmpty, !rev.contains("/"), !rev.contains("\\") {
            let snap = snapshotsDir.appendingPathComponent(rev, isDirectory: true).standardizedFileURL
            if seen.insert(snap.path).inserted { candidates.append(snap) }
        }
        if let snapDirs = try? fileManager.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) {
            let byDate = snapDirs.sorted { a, b in
                let dA = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let dB = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return dA > dB
            }
            for snap in byDate {
                let std = snap.standardizedFileURL
                if seen.insert(std.path).inserted { candidates.append(std) }
            }
        }
        return candidates.first { snap in
            Self.isAvailable(snap, fileManager: fileManager)
                && MTPLXModelOption.hasCompleteInstall(at: snap.path)
        }
    }

    static func hfRepoID(fromRepoDirectory repoDir: URL, snapshot: URL) -> String? {
        let sourceURL = snapshot.appendingPathComponent(".mtplx-source.json")
        if let data = try? Data(contentsOf: sourceURL),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let recorded = parsed["repo_id"] as? String,
           recorded.contains("/") {
            return recorded
        }
        let name = repoDir.lastPathComponent
        guard name.hasPrefix("models--") else { return nil }
        let rest = String(name.dropFirst("models--".count))
        guard let sep = rest.range(of: "--") else { return nil }
        let org = String(rest[..<sep.lowerBound])
        let repo = String(rest[sep.upperBound...])
        guard !org.isEmpty, !repo.isEmpty else { return nil }
        return "\(org)/\(repo)"
    }

    /// Discovers complete direct-child and HF-cache snapshot installs in root
    /// order. Canonical path identity prevents overlapping roots and symlink
    /// aliases from duplicating one physical model in the picker.
    public func discoverCompleteModels(
        fileManager: FileManager = .default
    ) -> [LocalModel] {
        var seen = Set<String>()
        var result: [LocalModel] = []
        let searchRoots = directories + (huggingfaceHubCacheDirectory.map { [$0] } ?? [])
        for root in searchRoots {
            guard let children = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if child.lastPathComponent.hasPrefix("models--") {
                    guard let snap = Self.resolveAnyHFSnapshot(inRepoDirectory: child, fileManager: fileManager),
                          let repoID = Self.hfRepoID(fromRepoDirectory: child, snapshot: snap) else { continue }
                    let canonical = Self.canonicalURL(for: snap.path, fallback: snap, fileManager: fileManager)
                    guard seen.insert(canonical.path).inserted,
                          MTPLXModelOption.hasCompleteInstall(at: canonical.path) else { continue }
                    let runtimePath = canonical.appendingPathComponent("mtplx_runtime.json").path
                    let metadata = MTPLXRuntimeMetadata.read(at: runtimePath)
                    let reference = explicitModelReference(metadata: metadata) ?? repoID
                    result.append(LocalModel(
                        path: canonical.path,
                        reference: reference,
                        displayName: Self.packDisplayName(in: canonical)
                            ?? reference.split(separator: "/").last.map(String.init) ?? reference
                    ))
                    continue
                }
                let canonical = Self.canonicalURL(
                    for: child.path,
                    fallback: child,
                    fileManager: fileManager
                )
                guard seen.insert(canonical.path).inserted,
                      MTPLXModelOption.hasCompleteInstall(at: canonical.path)
                else { continue }
                let runtimePath = canonical.appendingPathComponent("mtplx_runtime.json").path
                let metadata = MTPLXRuntimeMetadata.read(at: runtimePath)
                let reference = modelReference(metadata: metadata, directoryName: canonical.lastPathComponent)
                result.append(LocalModel(
                    path: canonical.path,
                    reference: reference,
                    displayName: Self.packDisplayName(in: canonical)
                        ?? reference.split(separator: "/").last.map(String.init) ?? reference
                ))
            }
        }
        return result
    }

    static func packDisplayName(in directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("mtplx-pack.json")),
              let metadata = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = metadata["display_name"] as? String,
              !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return name
    }

    public func nearestExistingDirectory(
        to directory: URL,
        fileManager: FileManager = .default
    ) -> URL {
        var candidate = directory.standardizedFileURL
        while !fileManager.fileExists(atPath: candidate.path), candidate.path != "/" {
            candidate.deleteLastPathComponent()
        }
        return candidate
    }

    private static func expandedURL(
        for path: String,
        environment: [String: String]
    ) -> URL {
        guard path == "~" || path.hasPrefix("~/") else {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        let home = environment["HOME"].flatMap { $0.isEmpty ? nil : $0 } ?? NSHomeDirectory()
        guard path != "~" else { return URL(fileURLWithPath: home, isDirectory: true) }
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent(String(path.dropFirst(2)), isDirectory: true)
    }

    /// The name a folder is listed under: the public name the pack declares
    /// for itself, else the folder name. Never the source it was built from:
    /// `forge_provenance.source_repo` names the source repo or folder, which
    /// every build of that source shares, so a 4-bit and a 6-bit build of one
    /// source got one name and one picker row, and an official download was
    /// listed a second time under its base model's name.
    private func explicitModelReference(
        metadata: MTPLXRuntimeMetadata?
    ) -> String? {
        let candidates: [String?] = [
            metadata?.publicModelID,
            metadata?.rawJSON["served_model_id"] as? String,
        ]
        return candidates.compactMap({ $0 })
            .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
            .first(where: { !$0.isEmpty })
    }

    private func modelReference(
        metadata: MTPLXRuntimeMetadata?,
        directoryName: String
    ) -> String {
        if let value = explicitModelReference(metadata: metadata) {
            return value
        }
        return directoryName.replacingOccurrences(of: "--", with: "/")
    }
}
