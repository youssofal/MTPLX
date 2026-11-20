import Foundation

/// What each model folder held the last time it was readable, so a pack on a
/// drive that is unplugged is listed as "not connected" instead of vanishing
/// from the picker, and Start never offers to download it again onto this Mac.
///
/// The picker records it whenever it scans the folders (off the main actor);
/// a folder that is not connected keeps its last entry, a folder no longer
/// configured is dropped. One small JSON file beside the app's settings,
/// written only for the app's real settings (`location` is nil in tests
/// unless a test sets it).
public enum ModelFolderIndex {
    struct Entry: Codable, Equatable, Sendable {
        let path: String
        let reference: String
        let displayName: String
    }

    struct File: Codable, Equatable, Sendable {
        var version = 1
        var folders: [String: [Entry]] = [:]
    }

    /// A pack remembered on a folder whose drive is not plugged in.
    public struct RememberedPack: Equatable, Sendable {
        public let model: ModelLibrary.LocalModel
        /// `/Volumes/<drive>`.
        public let volume: String

        public var driveName: String { (volume as NSString).lastPathComponent }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var storedLocation: URL?

    /// Where the index lives; nil means nothing is recorded or read.
    public static var location: URL? {
        get { lock.withLock { storedLocation } }
        set { lock.withLock { storedLocation = newValue } }
    }

    static func read(_ url: URL) -> File {
        guard let data = try? Data(contentsOf: url),
              let file = try? JSONDecoder().decode(File.self, from: data),
              file.version == 1
        else { return File() }
        return file
    }

    /// Records the readable folders of `library` from `discovered` (the
    /// packs a scan of the same library just found).
    public static func record(
        library: ModelLibrary,
        discovered: [ModelLibrary.LocalModel],
        at url: URL? = location
    ) {
        guard let url else { return }
        let previous = read(url)
        var next = File()
        for root in library.directories {
            let key = root.path
            switch SharedModelFolders.availability(of: key) {
            case .available:
                let prefix = key.hasSuffix("/") ? key : key + "/"
                next.folders[key] = discovered
                    .filter { $0.path.hasPrefix(prefix) }
                    .map { Entry(path: $0.path, reference: $0.reference, displayName: $0.displayName) }
            case .notConnected:
                if let kept = previous.folders[key] { next.folders[key] = kept }
            case .missing, .notAFolder:
                continue
            }
        }
        guard next != previous else { return }
        guard let data = try? JSONEncoder().encode(next) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try? data.write(to: url, options: [.atomic])
    }

    /// Packs remembered in the folders of `library` that are not connected.
    public static func rememberedPacks(
        library: ModelLibrary,
        at url: URL? = location
    ) -> [RememberedPack] {
        guard let url else { return [] }
        let file = read(url)
        var packs: [RememberedPack] = []
        for root in library.directories {
            guard case .notConnected(let volume) = SharedModelFolders.availability(of: root.path),
                  let entries = file.folders[root.path]
            else { continue }
            for entry in entries {
                packs.append(RememberedPack(
                    model: ModelLibrary.LocalModel(
                        path: entry.path, reference: entry.reference, displayName: entry.displayName
                    ),
                    volume: volume
                ))
            }
        }
        return packs
    }

    /// The not-connected drive a pack (by repo id or folder path) was last
    /// seen on, if any.
    public static func notConnectedPack(
        for model: String,
        library: ModelLibrary,
        at url: URL? = location
    ) -> RememberedPack? {
        let wanted = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !wanted.isEmpty else { return nil }
        return rememberedPacks(library: library, at: url).first {
            $0.model.reference.lowercased() == wanted || $0.model.path.lowercased() == wanted
        }
    }

    /// The line a picker row shows for a remembered pack.
    public static func notConnectedDetail(driveName: String) -> String {
        tr("On the drive %@, which is not connected. Plug it in to use this model.", driveName)
    }
}
