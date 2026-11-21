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

    public var directories: [URL] {
        [primaryDirectory] + additionalDirectories
    }

    public init(
        primaryDirectory: String,
        additionalDirectories: [String] = [],
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
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
        return directories.map {
            $0.appendingPathComponent(safeName, isDirectory: true).path
        }
    }

    /// Discovers complete direct-child installs in root order. Canonical path
    /// identity prevents overlapping roots and symlink aliases from duplicating
    /// one physical model in the picker.
    public func discoverCompleteModels(
        fileManager: FileManager = .default
    ) -> [LocalModel] {
        var seen = Set<String>()
        var result: [LocalModel] = []
        for root in directories {
            guard let children = try? fileManager.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
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
    private func modelReference(
        metadata: MTPLXRuntimeMetadata?,
        directoryName: String
    ) -> String {
        let candidates: [String?] = [
            metadata?.publicModelID,
            metadata?.rawJSON["served_model_id"] as? String,
        ]
        if let value = candidates.compactMap({ $0 })
            .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
            .first(where: { !$0.isEmpty })
        {
            return value
        }
        return directoryName.replacingOccurrences(of: "--", with: "/")
    }
}
