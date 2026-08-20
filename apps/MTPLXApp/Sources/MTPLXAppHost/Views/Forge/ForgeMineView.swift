import SwiftUI
import MTPLXAppCore

// MARK: - ForgeMineView
//
// Browser for every model the user has forged locally (or that
// AppConfiguration.customModels registered via rememberForgedModel).
// Two-column layout in the same dashboard rhythm as Cache + Sessions:
// list of cards on the left, detail panel on the right surfacing the
// full mtplx_runtime.json via RuntimeMetadataTable.
//
// Per-entry actions: Use, Open in Finder, Publish to HF, Remove.

struct ForgeMineView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var router: AppRouter

    @State private var entries: [ForgeLocalEntry] = []
    @State private var selectedID: String?
    @State private var pendingRemoveID: String?

    var body: some View {
        Group {
            if entries.isEmpty {
                EmptyStateView(
                    symbol: "tray.full",
                    title: "No forged models yet",
                    message: "Run a Forge build under the Create tab and your models will land here for re-verification, publishing, and inspection."
                )
            } else {
                splitView
            }
        }
        .onAppear { reload() }
        .confirmationDialog(
            "Remove this model from MTPLX?",
            isPresented: removePresented,
            titleVisibility: .visible
        ) {
            Button("Remove from picker", role: .destructive) {
                if let id = pendingRemoveID { removeFromPicker(entryID: id) }
                pendingRemoveID = nil
            }
            Button("Cancel", role: .cancel) {
                pendingRemoveID = nil
            }
        } message: {
            Text("This unregisters the model from the picker only — the files on disk stay intact. Delete the folder from Finder if you want to free the space.")
        }
    }

    // MARK: - Layout

    private var splitView: some View {
        HStack(alignment: .top, spacing: 0) {
            entriesList
                .frame(width: 360)
                .frame(maxHeight: .infinity, alignment: .top)
            Divider().overlay(Brand.separator)
            detailPanel
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    @ViewBuilder
    private var entriesList: some View {
        ScrollView {
            VStack(spacing: 8) {
                ForEach(entries) { entry in
                    Button {
                        selectedID = entry.id
                    } label: {
                        entryRow(entry, selected: selectedID == entry.id)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(16)
        }
    }

    @ViewBuilder
    private func entryRow(_ entry: ForgeLocalEntry, selected: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "hammer.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(selected ? Brand.accentChrome : Brand.typeTertiary)
                Text(entry.brandedName)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            if let repo = entry.sourceRepo {
                Text(repo)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(1)
            }
            HStack(spacing: 5) {
                if let depth = entry.depth, depth > 0 {
                    chip(text: "D\(depth)")
                }
                if let multiplier = entry.verificationMultiplier, multiplier > 1 {
                    chip(text: String(format: "%.2f× baseline", multiplier))
                }
                if entry.sizeOnDisk > 0 {
                    chip(text: formatGiB(entry.sizeOnDisk))
                }
                if entry.publishedToHF {
                    chip(text: "published", systemImage: "checkmark.seal")
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(selected ? Brand.accentChrome.opacity(0.10) : Brand.raisedSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(selected ? Brand.accentChrome.opacity(0.45) : Brand.separator, lineWidth: 0.5)
                )
        )
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    @ViewBuilder
    private var detailPanel: some View {
        if let entry = selectedEntry {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    detailHeader(entry: entry)
                    actionRow(entry: entry)
                    if let metadata = entry.metadata {
                        metadataCard(metadata: metadata)
                    } else {
                        Text("mtplx_runtime.json missing or unreadable.")
                            .font(.caption)
                            .foregroundStyle(Brand.typeTertiary)
                    }
                }
                .padding(20)
            }
        } else if let first = entries.first {
            // Auto-select the first entry on first render.
            Color.clear.onAppear { selectedID = first.id }
        } else {
            Color.clear
        }
    }

    private func detailHeader(entry: ForgeLocalEntry) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(entry.brandedName)
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
            if let forgedAt = entry.forgedAt {
                Text("Forged \(Self.dateFormatter.string(from: forgedAt))")
                    .font(.caption)
                    .foregroundStyle(Brand.typeSecondary)
            }
            Text(entry.localPath)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .truncationMode(.middle)
                .lineLimit(1)
        }
    }

    private func actionRow(entry: ForgeLocalEntry) -> some View {
        HStack(spacing: 10) {
            actionButton("Use", icon: "play.fill", emphasis: true) { useNow(entry) }
            actionButton("Open in Finder", icon: "folder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.localPath)])
            }
            actionButton(entry.publishedToHF ? "Re-publish" : "Publish to HF", icon: "arrow.up.circle.fill") {
                publish(entry)
            }
            actionButton("Remove", icon: "trash", emphasis: false, destructive: true) {
                pendingRemoveID = entry.id
            }
            Spacer(minLength: 0)
        }
    }

    private func metadataCard(metadata: MTPLXRuntimeMetadata) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("MTPLX_RUNTIME.JSON")
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(Brand.typeTertiary)
            RuntimeMetadataTable(json: metadata.rawJSON)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Brand.bgInner.opacity(0.6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.5)
                        )
                )
        }
    }

    @ViewBuilder
    private func actionButton(
        _ title: String,
        icon: String,
        emphasis: Bool = false,
        destructive: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
                Text(title)
                    .font(.system(size: 11.5, weight: .semibold))
            }
            .foregroundStyle(emphasis ? Brand.bgOuter : (destructive ? Brand.danger : Brand.typeBody))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                Capsule(style: .continuous)
                    .fill(emphasis ? AnyShapeStyle(Brand.typeBody) : AnyShapeStyle(Brand.cardSurface))
                    .overlay(
                        Capsule(style: .continuous)
                            .stroke(emphasis ? Color.clear : Brand.separator, lineWidth: 0.5)
                    )
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Helpers

    private var selectedEntry: ForgeLocalEntry? {
        entries.first(where: { $0.id == selectedID }) ?? entries.first
    }

    private var removePresented: Binding<Bool> {
        Binding(
            get: { pendingRemoveID != nil },
            set: { if !$0 { pendingRemoveID = nil } }
        )
    }

    private func chip(text: String, systemImage: String? = nil) -> some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .medium))
            }
            Text(text)
                .font(.system(size: 10, weight: .medium))
        }
        .foregroundStyle(Brand.typeSecondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 1.5)
        .background(
            Capsule(style: .continuous)
                .strokeBorder(Brand.separator, lineWidth: 0.5)
        )
    }

    private func formatGiB(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        if gib >= 1 { return String(format: "%.1f GB", gib) }
        let mib = Double(bytes) / 1_048_576.0
        if mib > 0, mib < 1 { return "<1 MB" }
        return String(format: "%.0f MB", mib)
    }

    // MARK: - Actions

    private func reload() {
        let scanner = ForgeLocalIndex()
        entries = scanner.scan(includingRegistered: backend.configuration.customModels)
        if selectedID == nil { selectedID = entries.first?.id }
    }

    private func useNow(_ entry: ForgeLocalEntry) {
        var config = backend.configuration
        config.rememberForgedModel(
            brandedName: entry.brandedName,
            localPath: entry.localPath,
            sizeBytes: entry.sizeOnDisk
        )
        if let verification = entry.verification {
            config.applyForgeRuntimeDefaults(
                modelPath: entry.localPath,
                verification: verification,
                sourceRepo: entry.sourceRepo
            )
        } else {
            config.model = entry.localPath
        }
        try? backend.saveSettings(config)
        Task {
            try? await backend.applyConfiguration(config, restartIfRunning: true)
            await MainActor.run {
                router.select(.live)
                router.primaryMode = .dashboard
            }
        }
    }

    private func publish(_ entry: ForgeLocalEntry) {
        // Hand off to the wizard's publish stage with the entry's
        // branded name + local path pre-seeded. Orchestrator owns
        // the state mutation so the @Published private(set) contract
        // is preserved.
        orchestrator.startPublishForExistingForge(
            brandedName: entry.brandedName,
            localPath: entry.localPath
        )
    }

    private func removeFromPicker(entryID: String) {
        guard let entry = entries.first(where: { $0.id == entryID }) else { return }
        var config = backend.configuration
        config.customModels.removeAll { $0.localCandidates.contains(entry.localPath) }
        try? backend.saveSettings(config)
        reload()
        if selectedID == entryID {
            selectedID = entries.first?.id
        }
    }

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f
    }()
}
