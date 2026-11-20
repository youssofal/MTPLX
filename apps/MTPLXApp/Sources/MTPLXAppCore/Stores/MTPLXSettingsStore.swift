import Foundation

/// What the settings store did about a settings file it could not read at
/// all (not JSON, not an object, unreadable bytes). Shown to the user as a
/// one-line notice so a reset to defaults is never silent, and so they can
/// find the file that was set aside.
public enum SettingsRecoveryNotice: Equatable, Sendable {
    /// The unreadable file was renamed beside the settings file, untouched,
    /// and the app started from defaults.
    case unreadableFileKept(at: URL, reason: String)
    /// The file could not be read and could not be moved aside either
    /// (`moveFailure`). The app still started from defaults; the next save
    /// will replace the file, so the notice names it while it still exists.
    case unreadableFileLeftInPlace(at: URL, reason: String, moveFailure: String)

    /// The file the notice is about.
    public var fileURL: URL {
        switch self {
        case .unreadableFileKept(let url, _), .unreadableFileLeftInPlace(let url, _, _):
            return url
        }
    }
}

/// Outcome of a settings load that never fails: the configuration to run
/// with, the fields that had to fall back to defaults, and whether the file
/// itself had to be set aside.
public struct SettingsLoadResult: Sendable {
    public var configuration: MTPLXAppConfiguration
    public var degradedFields: [SettingsDecodeIssue]
    public var recovery: SettingsRecoveryNotice?

    public init(
        configuration: MTPLXAppConfiguration,
        degradedFields: [SettingsDecodeIssue] = [],
        recovery: SettingsRecoveryNotice? = nil
    ) {
        self.configuration = configuration
        self.degradedFields = degradedFields
        self.recovery = recovery
    }
}

public struct MTPLXSettingsStore: Sendable {
    public enum ModelFolderSharing: Sendable {
        /// The app's real settings file shares its model folders with the
        /// CLI; a scratch or test settings path (and any XCTest run) never
        /// touches the shared file.
        case automatic
        case off
        case at(URL)
    }

    public var settingsURL: URL
    public var encoder: JSONEncoder
    public var decoder: JSONDecoder
    /// `~/.mtplx/config.toml` (`model_dir`, `model_dirs`), the model folders
    /// the app and `mtplx folders` share; nil when this store must not use it.
    public var sharedModelFoldersURL: URL?

    public init(
        settingsURL: URL = MTPLXSettingsStore.defaultSettingsURL(),
        encoder: JSONEncoder = JSONEncoder(),
        decoder: JSONDecoder = JSONDecoder(),
        modelFolderSharing: ModelFolderSharing = .automatic
    ) {
        self.settingsURL = settingsURL
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        switch modelFolderSharing {
        case .automatic:
            let environment = ProcessInfo.processInfo.environment
            let isProductionSettings = Self.settingsURLOverride(
                environment: environment,
                arguments: CommandLine.arguments
            ) == nil && settingsURL == Self.defaultSettingsURL()
            let underTest = environment["XCTestConfigurationFilePath"] != nil
                || NSClassFromString("XCTestCase") != nil
            self.sharedModelFoldersURL = isProductionSettings && !underTest
                ? SharedModelFolders.configURL(environment: environment)
                : nil
            if isProductionSettings && !underTest {
                ModelFolderIndex.location = settingsURL.deletingLastPathComponent()
                    .appendingPathComponent("model-folders-seen.json")
            }
        case .off:
            self.sharedModelFoldersURL = nil
        case .at(let url):
            self.sharedModelFoldersURL = url
        }
    }

    /// Absent file: defaults. Readable file: the decoded configuration, with
    /// any wrong-typed field already degraded to its default. Throws only
    /// when the file cannot be read or is not a JSON object at all.
    public func load() throws -> MTPLXAppConfiguration {
        try load(reporting: nil)
    }

    /// `load()` that also reports every field the decoder had to degrade
    /// into `issues`.
    public func load(reporting issues: SettingsDecodeIssues?) throws -> MTPLXAppConfiguration {
        var configuration: MTPLXAppConfiguration
        if FileManager.default.fileExists(atPath: settingsURL.path) {
            let data = try Data(contentsOf: settingsURL)
            if let issues {
                decoder.userInfo[SettingsDecodeIssues.userInfoKey] = issues
            }
            defer { decoder.userInfo.removeValue(forKey: SettingsDecodeIssues.userInfoKey) }
            configuration = try decoder.decode(MTPLXAppConfiguration.self, from: data)
        } else {
            configuration = MTPLXAppConfiguration()
        }
        applySharedModelFolders(to: &configuration, issues: issues)
        return configuration
    }

    /// The shared file wins once either side has named folders in it. Until
    /// then the app's own folders stay and are written there once, so the
    /// CLI sees what the app was already using. An unreadable shared file is
    /// reported and left as it is.
    func applySharedModelFolders(
        to configuration: inout MTPLXAppConfiguration,
        issues: SettingsDecodeIssues?
    ) {
        guard let url = sharedModelFoldersURL else { return }
        do {
            if let shared = try SharedModelFolders.read(from: url), shared.isSet {
                configuration.adoptSharedModelFolders(shared)
                return
            }
            let defaults = MTPLXAppConfiguration()
            let library = configuration.modelLibrary
            if library.primaryDirectory != defaults.modelLibrary.primaryDirectory
                || !library.additionalDirectories.isEmpty {
                try SharedModelFolders.write(
                    primary: library.primaryDirectory.path,
                    additional: library.additionalDirectories.map(\.path),
                    to: url
                )
            }
        } catch {
            issues?.record(SettingsDecodeIssue(
                path: "model folders (\(url.path))",
                reason: String(describing: error)
            ))
        }
    }

    /// The app's launch-time load. Distinguishes an absent file (fresh
    /// install: defaults, nothing to say) from an unreadable one: that file
    /// is moved beside itself as `settings.json.unreadable-<yyyyMMdd-HHmmss>`
    /// (never deleted, never overwritten) before the app falls back to
    /// defaults, and the caller gets a notice to show. A file that reads but
    /// has some bad fields is not a recovery; those fields are listed in
    /// `degradedFields` for the log.
    public func loadWithRecovery(now: Date = Date()) -> SettingsLoadResult {
        let issues = SettingsDecodeIssues()
        do {
            let configuration = try load(reporting: issues)
            return SettingsLoadResult(configuration: configuration, degradedFields: issues.all)
        } catch {
            let reason = SettingsDecodeIssue.describe(error)
            let recovery: SettingsRecoveryNotice
            do {
                let preservedAt = try setAsideUnreadableFile(now: now)
                recovery = .unreadableFileKept(at: preservedAt, reason: reason)
            } catch let moveFailure {
                recovery = .unreadableFileLeftInPlace(
                    at: settingsURL,
                    reason: reason,
                    moveFailure: moveFailure.localizedDescription
                )
            }
            return SettingsLoadResult(configuration: MTPLXAppConfiguration(), recovery: recovery)
        }
    }

    /// Rename the settings file to `settings.json.unreadable-<stamp>` next
    /// to itself. A name that already exists gets a numeric suffix rather
    /// than being replaced.
    func setAsideUnreadableFile(now: Date) throws -> URL {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let base = settingsURL.path + ".unreadable-" + formatter.string(from: now)
        var destination = URL(fileURLWithPath: base)
        var attempt = 1
        while FileManager.default.fileExists(atPath: destination.path) {
            attempt += 1
            destination = URL(fileURLWithPath: "\(base)-\(attempt)")
        }
        try FileManager.default.moveItem(at: settingsURL, to: destination)
        return destination
    }

    public func save(_ configuration: MTPLXAppConfiguration) throws {
        let directory = settingsURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let data = try encoder.encode(configuration)
        try data.write(to: settingsURL, options: [.atomic])
        // The same folders for `mtplx folders` and every CLI command. The
        // write is skipped when the file already says exactly this; a shared
        // file that cannot be read is left alone (load() reports it).
        if let url = sharedModelFoldersURL {
            let exists = FileManager.default.fileExists(atPath: url.path)
            guard !exists || (try? SharedModelFolders.read(from: url)) != nil else { return }
            let library = configuration.modelLibrary
            try SharedModelFolders.write(
                primary: library.primaryDirectory.path,
                additional: library.additionalDirectories.map(\.path),
                to: url
            )
        }
    }

    public static func defaultSettingsURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        arguments: [String] = CommandLine.arguments
    ) -> URL {
        if let override = settingsURLOverride(
            environment: environment,
            arguments: arguments
        ) {
            return override
        }
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base
            .appendingPathComponent("MTPLX", isDirectory: true)
            .appendingPathComponent("settings.json")
    }

    public static func settingsURLOverride(
        environment: [String: String],
        arguments: [String]
    ) -> URL? {
        if let raw = environment["MTPLX_APP_SETTINGS_PATH"],
           let url = settingsURL(fromOverride: raw) {
            return url
        }

        let names = ["--mtplx-app-settings", "--mtplx-settings-path"]
        for index in arguments.indices {
            let argument = arguments[index]
            for name in names {
                if argument == name {
                    let next = arguments.index(after: index)
                    guard arguments.indices.contains(next) else { continue }
                    if let url = settingsURL(fromOverride: arguments[next]) {
                        return url
                    }
                } else if argument.hasPrefix("\(name)=") {
                    let value = String(argument.dropFirst(name.count + 1))
                    if let url = settingsURL(fromOverride: value) {
                        return url
                    }
                }
            }
        }
        return nil
    }

    private static func settingsURL(fromOverride raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let expanded = (trimmed as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(expanded)
    }
}
