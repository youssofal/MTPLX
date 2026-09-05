import Foundation

/// The volume that will actually hold downloaded models.
///
/// The wizard, the download step and Forge measured free space on the home
/// volume. A user who keeps `~/.mtplx/models` on an external drive (a symlink,
/// or `MTPLX_MODEL_DIR`) was refused with "insufficient space" while that
/// drive had hundreds of gigabytes free (#466). Measure the models
/// directory's own volume instead: resolve symlinks and, while the directory
/// does not exist yet (a first launch has no `~/.mtplx`), the nearest existing
/// ancestor — which is the home volume exactly when nothing is redirected.
public enum ModelStoreVolume {
    /// The existing, symlink-resolved location whose volume is measured for
    /// `root`. Nothing is created on disk.
    public static func measurementURL(
        for root: URL,
        fileManager: FileManager = .default
    ) -> URL {
        var url = root.standardizedFileURL
        while !fileManager.fileExists(atPath: url.path) {
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path { break }
            url = parent
        }
        return url.resolvingSymlinksInPath()
    }

    /// Free space, in GiB, on the volume that holds the model store.
    public static func freeGiB(
        env: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> Double {
        let root = ModelDownloader.defaultCacheRoot(env: env)
        return freeGiB(at: measurementURL(for: root, fileManager: fileManager))
    }

    static func freeGiB(at url: URL) -> Double {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let bytes = values?.volumeAvailableCapacityForImportantUsage ?? 0
        return Double(bytes) / 1_073_741_824.0
    }
}
