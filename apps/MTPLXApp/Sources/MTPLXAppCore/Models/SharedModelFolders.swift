import Foundation

/// The model folders the app and the CLI share: `model_dir` (where downloads
/// and Forge builds go) and `model_dirs` (more folders searched for packs)
/// in `~/.mtplx/config.toml` (`MTPLX_CONFIG` names another file). Swift side
/// of `mtplx/model_folders.py`; `mtplx folders` edits the same two keys.
///
/// Nothing here copies, moves or deletes a pack. A folder on a drive that is
/// not plugged in stays in the list and reports `.notConnected`.
public struct SharedModelFolderSetting: Equatable, Sendable {
    /// nil: the default download folder (`~/.mtplx/models`).
    public var modelDir: String?
    /// nil: never set by either side (the app fills it from its own
    /// settings once); an empty array: set to no extra folders.
    public var modelDirs: [String]?

    public init(modelDir: String? = nil, modelDirs: [String]? = nil) {
        self.modelDir = modelDir
        self.modelDirs = modelDirs
    }

    public var isSet: Bool { modelDir != nil || modelDirs != nil }
}

public enum ModelFolderAvailability: Equatable, Sendable {
    case available
    /// On `/Volumes/<name>`, which is not mounted now.
    case notConnected(volume: String)
    case missing
    case notAFolder
}

public enum SharedModelFolders {
    public static func configURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        if let raw = environment["MTPLX_CONFIG"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !raw.isEmpty {
            return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
        }
        let home = environment["HOME"].flatMap { $0.isEmpty ? nil : $0 } ?? NSHomeDirectory()
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("config.toml")
    }

    /// nil when the file does not exist.
    public static func read(from url: URL) throws -> SharedModelFolderSetting? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let text = try String(contentsOf: url, encoding: .utf8)
        var setting = SharedModelFolderSetting()
        if case .some(.some(.string(let dir))) = TOMLLines.topLevelValue(text, key: "model_dir"),
           !dir.isEmpty {
            setting.modelDir = dir
        }
        switch TOMLLines.topLevelValue(text, key: "model_dirs") {
        case .some(.some(.strings(let dirs))):
            setting.modelDirs = dirs.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        case .some(.some(.string(let dir))):
            setting.modelDirs = [dir]
        default:
            break
        }
        return setting
    }

    /// Writes both keys, keeping every other line of the file. The default
    /// download folder is written as no `model_dir` line; `model_dirs` is
    /// always written, an empty list included.
    public static func write(
        primary: String,
        additional: [String],
        to url: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let fileManager = FileManager.default
        // A file that exists but cannot be read is never replaced: it holds
        // the user's other CLI settings.
        let existing: String? = fileManager.fileExists(atPath: url.path)
            ? try String(contentsOf: url, encoding: .utf8)
            : nil
        let defaultPrimary = ModelLibrary.defaultPrimaryDirectory(environment: environment)
        let isDefault = ModelLibrary.canonicalURL(for: primary, environment: environment).path
            == ModelLibrary.canonicalURL(for: defaultPrimary.path, environment: environment).path
        let text = TOMLLines.setTopLevelValues(
            existing ?? "# MTPLX user configuration\n",
            [
                ("model_dir", isDefault ? nil : .string(primary)),
                ("model_dirs", .strings(additional)),
            ]
        )
        guard text != existing else { return }
        let mode = (try? fileManager.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber)?
            .intValue ?? 0o644
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data(text.utf8).write(to: url, options: [.atomic])
        try fileManager.setAttributes([.posixPermissions: mode], ofItemAtPath: url.path)
    }

    public static func volume(of path: String) -> String? {
        let components = URL(fileURLWithPath: (path as NSString).expandingTildeInPath).pathComponents
        guard components.count >= 3, components[0] == "/", components[1] == "Volumes" else {
            return nil
        }
        return "/Volumes/\(components[2])"
    }

    static func isMounted(_ volume: String) -> Bool {
        let url = URL(fileURLWithPath: volume, isDirectory: true)
        return (try? url.resourceValues(forKeys: [.isVolumeKey]))?.isVolume == true
    }

    public static func availability(
        of path: String,
        fileManager: FileManager = .default
    ) -> ModelFolderAvailability {
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: expanded, isDirectory: &isDirectory) {
            return isDirectory.boolValue ? .available : .notAFolder
        }
        // A link onto an external drive that is unplugged (the #466 layout).
        var target = expanded
        if let destination = try? fileManager.destinationOfSymbolicLink(atPath: expanded) {
            target = destination.hasPrefix("/")
                ? destination
                : ((expanded as NSString).deletingLastPathComponent as NSString)
                    .appendingPathComponent(destination)
        }
        if let volume = volume(of: target), !isMounted(volume) {
            return .notConnected(volume: volume)
        }
        return .missing
    }

    public static func huggingfaceHubCacheURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
            if let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
               !raw.isEmpty {
                return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath, isDirectory: true)
                    .standardizedFileURL
            }
        }
        if let hfHome = environment["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hfHome.isEmpty {
            return URL(fileURLWithPath: (hfHome as NSString).expandingTildeInPath, isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
                .standardizedFileURL
        }
        let home = environment["HOME"].flatMap { $0.isEmpty ? nil : $0 } ?? NSHomeDirectory()
        return URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent(".cache", isDirectory: true)
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
            .standardizedFileURL
    }

    public static func huggingfaceCacheConfigured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"].contains { key in
            guard let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                return false
            }
            return !raw.isEmpty
        }
    }

    /// Model folders directly inside a folder: direct pack children or HF hub
    /// cache `models--Org--Repo/snapshots/<rev>` entries (0 when unreadable).
    public static func packCount(in path: String, fileManager: FileManager = .default) -> Int {
        let root = URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        guard let children = try? fileManager.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        let markers = ["config.json", "mtplx_pair.json"]
        var count = 0
        for child in children {
            if child.lastPathComponent.hasPrefix("models--") {
                if let snap = ModelLibrary.resolveAnyHFSnapshot(inRepoDirectory: child, fileManager: fileManager),
                   markers.contains(where: { fileManager.fileExists(atPath: snap.appendingPathComponent($0).path) }) {
                    count += 1
                }
            } else if markers.contains(where: { fileManager.fileExists(atPath: child.appendingPathComponent($0).path) }) {
                count += 1
            }
        }
        return count
    }
}

extension MTPLXAppConfiguration {
    /// Takes the shared setting when either side has set it; otherwise the
    /// app's own folders stay (and are written to the shared file once by
    /// the settings store).
    mutating func adoptSharedModelFolders(_ shared: SharedModelFolderSetting) {
        guard shared.isSet else { return }
        primaryModelDirectory = shared.modelDir ?? ModelLibrary.defaultPrimaryDirectory().path
        if let dirs = shared.modelDirs {
            additionalModelDirectories = dirs
        }
        normalizeModelDirectories()
    }
}
