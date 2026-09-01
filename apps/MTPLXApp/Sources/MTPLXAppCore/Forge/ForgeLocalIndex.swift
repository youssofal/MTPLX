import Foundation

// MARK: - ForgeLocalIndex
//
// Read-only filesystem scanner that surfaces every locally-installed
// MTPLX model so the Forge "My Models" browser can render it. Two
// admission rules, OR'd:
//
//   • The model dir contains an `mtplx_runtime.json` with
//     `forge_provenance.forged_locally == true` — the canonical
//     forged-locally signal stamped by the Python agent on every
//     `mtplx forge build` artifact.
//   • The model is registered in `customModels` via
//     `AppConfiguration.rememberForgedModel` (covers the case where
//     the file wasn't stamped yet but the user just clicked
//     "Use it now" on the Registered stage).
//
// Scanner roots are configurable so tests can point at a temp dir.
// In production we walk `~/Documents/MTPLX/models/` and the
// `hf-staging/` sibling — both directories the existing flagships
// already use.

// FileManager is not Sendable in Swift 6, so the scanner doesn't
// claim Sendability. It's only invoked from @MainActor view code
// (ForgeMineView.reload()) where the actor-isolation already serializes
// concurrent calls.
public struct ForgeLocalIndex {
    public init(
        roots: [URL]? = nil,
        modelLibrary: ModelLibrary = .default,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager
        if let roots {
            self.roots = Self.canonicalRoots(roots, fileManager: fileManager)
        } else {
            let home = fileManager.homeDirectoryForCurrentUser
            self.roots = Self.canonicalRoots(modelLibrary.directories + [
                home.appendingPathComponent("Documents/MTPLX/models", isDirectory: true),
                home.appendingPathComponent("Documents/MTPLX/hf-staging", isDirectory: true)
            ], fileManager: fileManager)
        }
    }

    private let roots: [URL]
    private let fileManager: FileManager

    public func scan(includingRegistered registered: [MTPLXModelOption] = []) -> [ForgeLocalEntry] {
        var seen: Set<String> = []
        var entries: [ForgeLocalEntry] = []

        // Walk each root once.
        for root in roots {
            guard let directoryEnumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey, .nameKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            for case let url as URL in directoryEnumerator {
                let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                guard isDir else { continue }
                let runtimeJSON = url.appendingPathComponent("mtplx_runtime.json")
                guard fileManager.fileExists(atPath: runtimeJSON.path) else { continue }
                directoryEnumerator.skipDescendants() // don't recurse into the model

                let metadata = MTPLXRuntimeMetadata.read(at: runtimeJSON.path)
                let isForgedLocal = metadata?.forgeProvenance?.forgedLocally == true
                let canonicalURL = ModelLibrary.canonicalURL(for: url.path, fileManager: fileManager)
                let matchedOption = registered.first {
                    $0.localCandidates.contains {
                        ModelLibrary.canonicalURL(for: $0, fileManager: fileManager).path
                            == canonicalURL.path
                    }
                }

                if isForgedLocal || matchedOption != nil {
                    if !seen.insert(canonicalURL.path).inserted { continue }
                    entries.append(ForgeLocalEntry(
                        localPath: canonicalURL.path,
                        directoryName: canonicalURL.lastPathComponent,
                        metadata: metadata,
                        modelOption: matchedOption,
                        sizeOnDisk: ModelDownloaderSize.recursive(of: canonicalURL, fm: fileManager)
                    ))
                }
            }
        }

        // Finally, surface any customModels whose localCandidate wasn't
        // visited (e.g. user pointed the picker at a forge under
        // ~/AltLocation/ that we didn't scan).
        for option in registered where option.id.hasPrefix("forged-") {
            for candidate in option.localCandidates {
                let expanded = (candidate as NSString).expandingTildeInPath
                guard fileManager.fileExists(atPath: expanded) else { continue }
                let canonicalURL = ModelLibrary.canonicalURL(for: expanded, fileManager: fileManager)
                if seen.insert(canonicalURL.path).inserted {
                    let runtimeJSON = canonicalURL
                        .appendingPathComponent("mtplx_runtime.json")
                    let metadata = MTPLXRuntimeMetadata.read(at: runtimeJSON.path)
                    entries.append(ForgeLocalEntry(
                        localPath: canonicalURL.path,
                        directoryName: canonicalURL.lastPathComponent,
                        metadata: metadata,
                        modelOption: option,
                        sizeOnDisk: ModelDownloaderSize.recursive(of: canonicalURL, fm: fileManager)
                    ))
                }
            }
        }

        return entries.sorted { ($0.forgedAt ?? .distantPast) > ($1.forgedAt ?? .distantPast) }
    }

    private static func canonicalRoots(
        _ roots: [URL],
        fileManager: FileManager
    ) -> [URL] {
        var seen: Set<String> = []
        return roots.compactMap { root in
            let canonical = ModelLibrary.canonicalURL(for: root.path, fileManager: fileManager)
            return seen.insert(canonical.path).inserted ? canonical : nil
        }
    }
}

// MARK: - ForgeLocalEntry

public struct ForgeLocalEntry: Identifiable, Equatable, @unchecked Sendable {
    public var id: String { localPath }
    public var localPath: String
    public var directoryName: String
    public var metadata: MTPLXRuntimeMetadata?
    public var modelOption: MTPLXModelOption?
    public var sizeOnDisk: Int64

    public init(
        localPath: String,
        directoryName: String,
        metadata: MTPLXRuntimeMetadata?,
        modelOption: MTPLXModelOption?,
        sizeOnDisk: Int64
    ) {
        self.localPath = localPath
        self.directoryName = directoryName
        self.metadata = metadata
        self.modelOption = modelOption
        self.sizeOnDisk = sizeOnDisk
    }

    public var displayName: String {
        modelOption?.displayName
            ?? metadata?.forgeProvenance?.sourceRepo
            ?? directoryName
    }

    public var brandedName: String {
        modelOption?.displayName ?? directoryName
    }

    public var forgedAt: Date? {
        guard let raw = metadata?.forgeProvenance?.forgedAt else { return nil }
        // Two ISO8601 variants because the spine fixtures use the
        // non-fractional form ("2026-05-02T02:23:23+0100") while
        // freshly-stamped forges may include fractional seconds.
        // Allocating a formatter per call sidesteps ISO8601DateFormatter
        // not being Sendable (Swift 6 strict-concurrency).
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        if let date = plain.date(from: raw) { return date }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return fractional.date(from: raw)
    }

    public var sourceRepo: String? {
        metadata?.forgeProvenance?.sourceRepo
    }

    public var verification: ForgeVerification? {
        guard let rawJSON = metadata?.rawJSON else { return nil }
        return ForgeVerification.fromRuntimeMetadata(rawJSON)
    }

    public var verificationMultiplier: Double? {
        verification?.multiplierVsAr
    }

    public var depth: Int? {
        verification?.bestDepth ?? metadata?.mtpDepthMax
    }

    public var publishedToHF: Bool {
        metadata?.forgeProvenance?.publishedToHf != nil
    }

    public var publishedRepo: String? {
        metadata?.forgeProvenance?.publishedToHf?.repo
    }

    public static func == (lhs: ForgeLocalEntry, rhs: ForgeLocalEntry) -> Bool {
        lhs.localPath == rhs.localPath
            && lhs.directoryName == rhs.directoryName
            && lhs.sizeOnDisk == rhs.sizeOnDisk
            && lhs.modelOption == rhs.modelOption
        // metadata.rawJSON intentionally excluded — see
        // MTPLXRuntimeMetadata's Equatable note in todo 04.
    }
}

// MARK: - Recursive size helper (kept fileprivate to avoid clashing
// with the public ModelDownloader.recursiveSize)

private enum ModelDownloaderSize {
    static func recursive(of url: URL, fm: FileManager) -> Int64 {
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if values?.isRegularFile == true {
                total += Int64(values?.fileSize ?? 0)
            }
        }
        return total
    }
}
