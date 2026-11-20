import SwiftUI
import MTPLXAppCore

// MARK: - ForgeMineView
//
// Browser for every model the user has forged locally (or that
// AppConfiguration.customModels registered via rememberForgedModel).
// Two-column layout in the same dashboard rhythm as Cache + Sessions:
// list of cards on the left, detail panel on the right: a summary of
// the model, then the full mtplx_runtime.json (RuntimeMetadataTable)
// behind a disclosure.
//
// Per-entry actions: Use, Open in Finder, Publish to HF, Remove.

struct ForgeMineView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var router: AppRouter

    @State private var entries: [ForgeLocalEntry] = []
    @State private var selectedID: String?
    @State private var pendingRemoveID: String?
    @State private var showsRuntimeFile = false

    var body: some View {
        Group {
            if entries.isEmpty {
                EmptyStateView(
                    symbol: "tray.full",
                    title: tr("No forged models yet"),
                    message: tr("Run a Forge build under the Create tab and your models will land here for re-verification, publishing, and inspection.")
                )
            } else {
                splitView
            }
        }
        .onAppear { reload() }
        .confirmationDialog(
            tr("Remove this model from MTPLX?"),
            isPresented: removePresented,
            titleVisibility: .visible
        ) {
            Button(tr("Remove from picker"), role: .destructive) {
                if let id = pendingRemoveID { removeFromPicker(entryID: id) }
                pendingRemoveID = nil
            }
            Button(tr("Cancel"), role: .cancel) {
                pendingRemoveID = nil
            }
        } message: {
            Text(tr("This unregisters the model from the picker only — the files on disk stay intact. Delete the folder from Finder if you want to free the space."))
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
                    chip(text: tr("D%lld", depth))
                }
                if let multiplier = entry.verificationMultiplier, multiplier > 1 {
                    chip(text: tr("%.2f× baseline", multiplier))
                }
                if entry.sizeOnDisk > 0 {
                    chip(text: formatGiB(entry.sizeOnDisk))
                }
                if entry.publishedToHF {
                    chip(text: tr("published"), systemImage: "checkmark.seal")
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
                        metadataCard(entry: entry, metadata: metadata)
                    } else {
                        Text(tr("mtplx_runtime.json missing or unreadable."))
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
                Text(tr("Forged %@", Self.dateFormatter.string(from: forgedAt)))
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
            actionButton(tr("Use"), icon: "play.fill", emphasis: true) { useNow(entry) }
            actionButton(tr("Open in Finder"), icon: "folder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.localPath)])
            }
            actionButton(entry.publishedToHF ? tr("Re-publish") : tr("Publish to HF"), icon: "arrow.up.circle.fill") {
                publish(entry)
            }
            actionButton(tr("Remove"), icon: "trash", emphasis: false, destructive: true) {
                pendingRemoveID = entry.id
            }
            Spacer(minLength: 0)
        }
    }

    /// What a person asks of a model first (what it is built from, how
    /// fast it measured, how big it is, how it samples), then the whole
    /// runtime file one click away. The file used to fill the panel.
    private func metadataCard(entry: ForgeLocalEntry, metadata: MTPLXRuntimeMetadata) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            let facts = summaryFacts(entry: entry, metadata: metadata)
            if !facts.isEmpty {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 18, verticalSpacing: 11) {
                    ForEach(facts) { fact in
                        GridRow {
                            Text(fact.label)
                                .textCase(.uppercase)
                                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                                .tracking(1.5)
                                .foregroundStyle(Brand.typeTertiary)
                                .gridColumnAlignment(.leading)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(fact.value)
                                    .font(.system(size: 12.5, weight: .medium))
                                    .foregroundStyle(Brand.typeHi)
                                    .textSelection(.enabled)
                                if let detail = fact.detail {
                                    Text(detail)
                                        .font(.system(size: 11))
                                        .foregroundStyle(Brand.typeTertiary)
                                }
                            }
                        }
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(panelBackground)
            }
            runtimeFileToggle
            if showsRuntimeFile {
                RuntimeMetadataTable(json: metadata.rawJSON)
                    .padding(12)
                    .background(panelBackground)
            }
        }
    }

    private var runtimeFileToggle: some View {
        Button {
            showsRuntimeFile.toggle()
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .bold))
                    .rotationEffect(.degrees(showsRuntimeFile ? 90 : 0))
                Text(tr("Runtime file"))
                    .textCase(.uppercase)
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                Text(verbatim: "mtplx_runtime.json")
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            .foregroundStyle(Brand.typeSecondary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(showsRuntimeFile ? [.isSelected] : [])
    }

    private var panelBackground: some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(Brand.bgInner.opacity(0.6))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(Brand.separator, lineWidth: 0.5)
            )
    }

    private struct Fact: Identifiable {
        let label: String
        let value: String
        var detail: String?
        var id: String { label }
    }

    /// Only what the model's own files say; a row with nothing behind it
    /// is left out rather than guessed.
    private func summaryFacts(entry: ForgeLocalEntry, metadata: MTPLXRuntimeMetadata) -> [Fact] {
        var facts: [Fact] = []
        if let base = entry.sourceRepo ?? metadata.baseTrunk, !base.isEmpty {
            facts.append(Fact(label: tr("Base model"), value: base))
        }
        if let verification = entry.verification, verification.bestDepth > 0, verification.multiplierVsAr > 1 {
            // The stamped "hardware" is a platform string
            // ("macOS-26.3.1-arm64-..."), not a Mac a person would name,
            // so the rate is labelled by when it was measured instead.
            let tokS = verification.tokSByDepth[verification.bestDepth]
            facts.append(Fact(
                label: tr("Speed"),
                value: tr("%.2f× baseline at depth %lld", verification.multiplierVsAr, verification.bestDepth),
                detail: tokS.map { tr("%.1f tok/s at build time", $0) }
            ))
        }
        if let recipe = metadata.forgeProvenance?.forgeRecipe {
            facts.append(Fact(
                label: tr("Weights"),
                value: tr("%lld-bit, group size %lld", recipe.bodyBits, recipe.bodyGroupSize)
            ))
        }
        if entry.sizeOnDisk > 0 {
            facts.append(Fact(label: tr("Size on disk"), value: formatGiB(entry.sizeOnDisk)))
        }
        if let sampler = metadata.rawJSON["sampler"] as? [String: Any],
           let temperature = (sampler["temperature"] as? NSNumber)?.doubleValue,
           let topP = (sampler["top_p"] as? NSNumber)?.doubleValue,
           let topK = (sampler["top_k"] as? NSNumber)?.intValue {
            facts.append(Fact(
                label: tr("Sampler"),
                value: tr(
                    "temperature %@ · top-p %@ · top-k %lld",
                    Self.samplerNumber(temperature), Self.samplerNumber(topP), topK
                )
            ))
        }
        if let profile = metadata.recommendedProfile, !profile.isEmpty {
            facts.append(Fact(label: tr("Profile"), value: profile))
        }
        if let version = metadata.mtplxVersion, !version.isEmpty {
            facts.append(Fact(label: tr("Built with"), value: tr("MTPLX %@", version)))
        }
        facts.append(Fact(label: tr("Hugging Face"), value: entry.publishedRepo ?? tr("Not published")))
        return facts
    }

    /// "1.0", "0.6", "0.95": at least one decimal so a temperature of
    /// 1.0 never reads as an integer setting; separators follow the app
    /// language, as `tr` does.
    private static func samplerNumber(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(1...2)).locale(L10n.language.locale))
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
        if gib >= 1 { return tr("%.1f GB", gib) }
        let mib = Double(bytes) / 1_048_576.0
        if mib > 0, mib < 1 { return tr("<1 MB") }
        return tr("%.0f MB", mib)
    }

    // MARK: - Actions

    private func reload() {
        let scanner = ForgeLocalIndex(modelLibrary: backend.configuration.modelLibrary)
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
        let modelIDs = config.customModels
            .filter { $0.localCandidates.contains(entry.localPath) }
            .map(\.id)
        for modelID in modelIDs {
            config.removeCustomModel(id: modelID)
        }
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
