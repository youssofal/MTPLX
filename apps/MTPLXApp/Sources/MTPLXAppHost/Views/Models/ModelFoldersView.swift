import SwiftUI
import MTPLXAppCore
#if canImport(AppKit)
import AppKit
#endif

// MARK: - ModelFoldersView
//
// The model folders, for the Settings tab (APPUI places it; it edits the
// Settings draft and the normal Save writes it). The folders are shared with
// the `mtplx folders` command: `model_dir` (where downloads and Forge builds
// go) and `model_dirs` (more folders searched for packs) in
// ~/.mtplx/config.toml, through MTPLXSettingsStore.
//
// A folder on a drive that is unplugged stays listed as "Not connected";
// its packs show again when the drive is back. Removing a folder only stops
// searching it, and choosing a new download folder keeps the old one
// searched: nothing is copied, moved or deleted.

struct ModelFoldersView: View {
    @Binding var configuration: MTPLXAppConfiguration

    struct Row: Identifiable, Equatable, Sendable {
        var id: String { path }
        let path: String
        let isDownloads: Bool
        let isHuggingFaceCache: Bool
        let availability: ModelFolderAvailability
        let packs: Int
    }

    @State private var rows: [Row] = []

    private var folderKey: [String] {
        [
            configuration.primaryModelDirectory,
            SharedModelFolders.huggingfaceHubCacheURL().path,
        ] + configuration.additionalModelDirectories
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(rows) { row in
                folderRow(row)
                if row.id != rows.last?.id {
                    Divider().overlay(Brand.separator)
                }
            }
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Button {
                    addFolders()
                } label: {
                    Label(tr("Add folder"), systemImage: "plus")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                Text(tr("Shared with the mtplx folders command. Nothing is copied or moved."))
                    .font(.caption)
                    .foregroundStyle(Brand.typeTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 2)
        }
        .task(id: folderKey) { await refresh() }
    }

    // MARK: Row

    @ViewBuilder
    private func folderRow(_ row: Row) -> some View {
        let onVolume = SharedModelFolders.volume(of: row.path) != nil
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: onVolume ? "externaldrive" : "folder")
                .font(.system(size: 13))
                .foregroundStyle(Brand.typeSecondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(displayName(row.path))
                        .font(.callout.weight(.medium))
                        .foregroundStyle(Brand.typeBody)
                    if row.isDownloads {
                        Text(tr("Downloads"))
                            .font(.system(size: 9, weight: .heavy, design: .rounded))
                            .foregroundStyle(Brand.accent)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Capsule().stroke(Brand.accent.opacity(0.6), lineWidth: 0.75))
                    } else if row.isHuggingFaceCache {
                        Text(tr("Hugging Face"))
                            .font(.system(size: 9, weight: .heavy, design: .rounded))
                            .foregroundStyle(Brand.typeSecondary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Capsule().stroke(Brand.typeTertiary.opacity(0.6), lineWidth: 0.75))
                    }
                    status(row)
                }
                Text(row.path)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 8)
            Menu {
                if !row.isHuggingFaceCache {
                    Button(tr("Use for Downloads")) { useForDownloads(row.path) }
                        .disabled(row.isDownloads || row.availability != .available)
                }
                Button(tr("Show in Finder")) { showInFinder(row.path) }
                    .disabled(row.availability != .available)
                if !row.isHuggingFaceCache {
                    Divider()
                    Button(tr("Remove"), role: .destructive) { remove(row.path) }
                        .disabled(row.isDownloads)
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func status(_ row: Row) -> some View {
        switch row.availability {
        case .available:
            Text(tr("Packs: %lld", row.packs))
                .font(.caption)
                .foregroundStyle(Brand.typeTertiary)
        case .notConnected:
            Label(tr("Not connected"), systemImage: "externaldrive.badge.xmark")
                .font(.caption)
                .foregroundStyle(Brand.warning)
                .help(tr("Plug the drive back in and its packs show again."))
        case .missing:
            Label(tr("Missing"), systemImage: "questionmark.folder")
                .font(.caption)
                .foregroundStyle(Brand.warning)
        case .notAFolder:
            Label(tr("Not a folder"), systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(Brand.warning)
        }
    }

    private func displayName(_ path: String) -> String {
        let name = (path as NSString).lastPathComponent
        if let volume = SharedModelFolders.volume(of: path) {
            let volumeName = (volume as NSString).lastPathComponent
            return volumeName == name ? volumeName : "\(name) (\(volumeName))"
        }
        return name
    }

    // MARK: Actions

    private func refresh() async {
        let primary = configuration.modelLibrary.primaryDirectory.path
        let additional = configuration.modelLibrary.additionalDirectories.map(\.path)
        let hfHub = configuration.modelLibrary.huggingfaceHubCacheDirectory?.path
        // File checks off the main actor: a slow or sleeping drive must never
        // stall Settings.
        let computed = await Task.detached(priority: .utility) { () -> [Row] in
            var result = ([primary] + additional).enumerated().map { index, path in
                let availability = SharedModelFolders.availability(of: path)
                return Row(
                    path: path,
                    isDownloads: index == 0,
                    isHuggingFaceCache: false,
                    availability: availability,
                    packs: availability == .available ? SharedModelFolders.packCount(in: path) : 0
                )
            }
            if let hfHub,
               SharedModelFolders.availability(of: hfHub) == .available {
                result.append(Row(
                    path: hfHub,
                    isDownloads: false,
                    isHuggingFaceCache: true,
                    availability: .available,
                    packs: SharedModelFolders.packCount(in: hfHub)
                ))
            }
            return result
        }.value
        rows = computed
    }

    private func addFolders() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.canCreateDirectories = true
        panel.prompt = tr("Add")
        panel.message = tr("Choose one or more model folders to search.")
        if panel.runModal() == .OK {
            configuration.addModelDirectories(panel.urls.map(\.path))
        }
        #endif
    }

    private func useForDownloads(_ path: String) {
        configuration.additionalModelDirectories.removeAll {
            ModelLibrary.canonicalURL(for: $0).path == ModelLibrary.canonicalURL(for: path).path
        }
        configuration.setPrimaryModelDirectory(path, preservePrevious: true)
    }

    private func remove(_ path: String) {
        configuration.additionalModelDirectories.removeAll {
            ModelLibrary.canonicalURL(for: $0).path == ModelLibrary.canonicalURL(for: path).path
        }
        configuration.normalizeModelDirectories()
    }

    private func showInFinder(_ path: String) {
        #if canImport(AppKit)
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
        #endif
    }
}
