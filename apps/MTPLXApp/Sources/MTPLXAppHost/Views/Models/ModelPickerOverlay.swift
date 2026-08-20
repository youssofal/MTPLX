import Foundation
import SwiftUI
import MTPLXAppCore

// MARK: - ModelPickerOverlay
//
// Top-left model selector. It mirrors the inference popover language:
// notch, raised surface, monospaced section labels, and row reveal.

struct ModelPickerOverlay: View, Equatable {
    let backend: MTPLXBackendStore
    let configuration: MTPLXAppConfiguration
    let daemonState: DaemonState
    let modelUpdates: [ModelUpdateInfo]
    let modelPackUpdatingRepoID: String?
    let modelPackUpdateStatus: String?
    let modelPackUpdateNeedsRestart: ModelUpdateInfo?

    @EnvironmentObject private var themeStore: ThemeStore

    @Binding var presented: Bool
    private let presentedValue: Bool

    init(
        backend: MTPLXBackendStore,
        configuration: MTPLXAppConfiguration,
        daemonState: DaemonState,
        presented: Binding<Bool>,
        modelUpdates: [ModelUpdateInfo] = [],
        modelPackUpdatingRepoID: String? = nil,
        modelPackUpdateStatus: String? = nil,
        modelPackUpdateNeedsRestart: ModelUpdateInfo? = nil
    ) {
        self.backend = backend
        self.configuration = configuration
        self.daemonState = daemonState
        self.modelUpdates = modelUpdates
        self.modelPackUpdatingRepoID = modelPackUpdatingRepoID
        self.modelPackUpdateStatus = modelPackUpdateStatus
        self.modelPackUpdateNeedsRestart = modelPackUpdateNeedsRestart
        _presented = presented
        presentedValue = presented.wrappedValue
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.presentedValue == rhs.presentedValue
            && lhs.configuration == rhs.configuration
            && lhs.daemonState == rhs.daemonState
            && lhs.modelUpdates == rhs.modelUpdates
            && lhs.modelPackUpdatingRepoID == rhs.modelPackUpdatingRepoID
            && lhs.modelPackUpdateStatus == rhs.modelPackUpdateStatus
            && lhs.modelPackUpdateNeedsRestart == rhs.modelPackUpdateNeedsRestart
    }

    @State private var borderProgress: CGFloat = 0
    @State private var headerVisible: Bool = false
    @State private var rowsVisibleCount: Int = 0
    @State private var applyingModelID: String? = nil
    @State private var errorMessage: String? = nil
    @State private var addRowExpanded: Bool = false
    @State private var customRepoInput: String = ""
    @State private var customProbe: OtherModelProbe? = nil
    @State private var customNoMTPAcknowledged: Bool = false
    @State private var checkingCustomRepo: Bool = false
    @State private var preparedRows: [ModelPickerPreparedOption] = []
    @State private var preparedRowsSignature: ModelPickerCatalogSignature?
    @State private var prepareRowsTask: Task<Void, Never>?
    @State private var detectedHardware: DetectedHardware?
    @State private var hardwareTask: Task<Void, Never>?
    @FocusState private var customRepoFocused: Bool

    private let popoverWidth: CGFloat = 420
    private let modelListMaxHeight: CGFloat = 340
    private let modelRowEstimatedHeight: CGFloat = 76
    private let cornerRadius: CGFloat = Motion.overlayCornerRadius
    private let topOffset: CGFloat = 50
    private let leadingOffset: CGFloat = 152

    var body: some View {
        ZStack(alignment: .topLeading) {
            backdrop
            if presented {
                popoverColumn
                    .padding(.top, topOffset)
                    .padding(.leading, leadingOffset)
                    .transition(.identity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(presented)
        .onAppear {
            preparePickerRows()
            detectHardwareForCatalog()
        }
        .onDisappear {
            prepareRowsTask?.cancel()
            prepareRowsTask = nil
            hardwareTask?.cancel()
            hardwareTask = nil
        }
        .onChange(of: configuration.model) { _, _ in
            preparePickerRows()
        }
        .onChange(of: configuration.customModels) { _, _ in
            preparePickerRows()
        }
        .onChange(of: detectedHardware) { _, _ in
            preparePickerRows()
        }
        .onChange(of: presented) { _, isOn in
            if isOn { runEnterChoreography() } else { runExitChoreography() }
        }
        .task(id: presented) {
            // Opening the picker is the natural moment to look for pack
            // updates; the store throttles to one network check per 6 h.
            guard presented else { return }
            await backend.refreshModelUpdates()
        }
    }

    @ViewBuilder
    private var backdrop: some View {
        Color.clear
            .contentShape(Rectangle())
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onTapGesture { presented = false }
            .opacity(presented ? 1 : 0)
    }

    @ViewBuilder
    private var popoverColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            notch
            popoverSurface
                .frame(width: popoverWidth)
        }
        .frame(width: popoverWidth, alignment: .leading)
        .opacity(borderProgress)
        .scaleEffect(borderProgress > 0 ? 1 : 0.94, anchor: .topLeading)
    }

    @ViewBuilder
    private var notch: some View {
        ModelPickerNotch()
            .fill(Brand.raisedSurface)
            .frame(width: 12, height: 7)
            .padding(.leading, 22)
            .padding(.bottom, -1)
            .opacity(borderProgress > 0.3 ? 1 : 0)
    }

    @ViewBuilder
    private var popoverSurface: some View {
        let rows = preparedRows
        VStack(alignment: .leading, spacing: 0) {
            header
            if modelPackUpdateNeedsRestart != nil || !availablePackUpdates.isEmpty {
                modelUpdatesBanner
            }
            sectionDivider(precedesRow: 1)
            ScrollView(.vertical, showsIndicators: rows.count > 4) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(rows.enumerated()), id: \.element.id) { index, row in
                        modelRow(row, visible: rowsVisibleCount > index)
                        if index < rows.count - 1 {
                            sectionDivider(precedesRow: index + 2)
                        }
                    }
                }
                .frame(width: popoverWidth, alignment: .leading)
            }
            .frame(height: modelListHeight(for: rows.count))
            sectionDivider(precedesRow: rows.count + 1)
            addModelRow(visible: rowsVisibleCount > rows.count)
            if let errorMessage {
                errorBar(errorMessage)
            }
        }
        // Clip the content stack to the popover's rounded shape so the
        // edge-to-edge row fills (selection + hover) naturally follow
        // the curve at the top/bottom rows instead of poking past it.
        // Without this, a flat-stripe fill would render rectangular
        // and clash with the popover's outer corners.
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Brand.raisedSurface)
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .trim(from: 0, to: borderProgress)
                    .stroke(Brand.separatorStrong, lineWidth: 0.75)
            }
            .shadow(color: .black.opacity(0.55), radius: 18, x: 0, y: 10)
        )
    }

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Model")
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeBody)
            Text(restartHint)
                .font(.caption2)
                .foregroundStyle(Brand.typeTertiary)
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .opacity(headerVisible ? 1 : 0)
        .offset(y: headerVisible ? 0 : -6)
    }

    private var availablePackUpdates: [ModelUpdateInfo] {
        modelUpdates.filter(\.isUpdateAvailable)
    }

    private func updateSizeText(_ update: ModelUpdateInfo) -> String? {
        guard let bytes = update.updateBytes, bytes > 0 else { return nil }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Sparkle-for-models strip: one row per pack with a newer published
    /// revision, plus the restart affordance once an update has landed for
    /// the pack the running daemon serves.
    @ViewBuilder
    private var modelUpdatesBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let restart = modelPackUpdateNeedsRestart {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.typeBody)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("\(restart.shortName) updated")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(Brand.typeBody)
                        Text("Restart MTPLX to load the updated files.")
                            .font(.caption2)
                            .foregroundStyle(Brand.typeTertiary)
                    }
                    Spacer(minLength: 8)
                    Button("Restart") {
                        Task { await backend.restartToApplyModelUpdate() }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
            }
            ForEach(availablePackUpdates.prefix(3)) { update in
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.typeBody)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(
                            updateSizeText(update).map {
                                "Update available: \(update.shortName) (\($0))"
                            } ?? "Update available: \(update.shortName)"
                        )
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Brand.typeBody)
                        .lineLimit(1)
                        if let note = update.note, !note.isEmpty {
                            Text(note)
                                .font(.caption2)
                                .foregroundStyle(Brand.typeTertiary)
                                .lineLimit(2)
                        }
                        if modelPackUpdatingRepoID == update.repoID,
                           let status = modelPackUpdateStatus {
                            Text(status)
                                .font(.caption2)
                                .foregroundStyle(Brand.typeTertiary)
                                .lineLimit(1)
                        }
                    }
                    Spacer(minLength: 8)
                    if modelPackUpdatingRepoID == update.repoID {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Button("Update") {
                            backend.updateModelPack(update)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(modelPackUpdatingRepoID != nil)
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(width: popoverWidth, alignment: .leading)
        .background(Brand.separatorStrong.opacity(0.18))
    }

    @ViewBuilder
    private func modelRow(_ row: ModelPickerPreparedOption, visible: Bool) -> some View {
        ModelRowView(
            displayName: row.displayName,
            detail: row.detail,
            isInstalled: row.isInstalled,
            selected: row.selected,
            applying: applyingModelID == row.id,
            restartRequired: restartRequired,
            disabled: applyingModelID != nil || checkingCustomRepo || isTransitioning,
            visible: visible,
            motionEnabled: motionEnabled,
            action: { select(row) }
        )
    }

    @ViewBuilder
    private func addModelRow(visible: Bool) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                guard applyingModelID == nil, !checkingCustomRepo, !isTransitioning else { return }
                withAnimation(motionEnabled ? .spring(response: 0.30, dampingFraction: 0.86) : nil) {
                    addRowExpanded.toggle()
                    customProbe = nil
                    customNoMTPAcknowledged = false
                    errorMessage = nil
                }
                if addRowExpanded {
                    Task { @MainActor in
                        try? await Task.sleep(for: .milliseconds(160))
                        customRepoFocused = true
                    }
                }
            } label: {
                HStack(alignment: .center, spacing: 12) {
                    // Hairline circle + thin plus, same vocabulary as
                    // the chat sidebar's "+ New Chat" affordance. The
                    // previous filled `plus.circle.fill` in
                    // accent-blue stole the eye away from the selection
                    // signal (which is the *only* blue element on the
                    // picker).
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(addRowExpanded ? Brand.typeBody : Brand.typeSecondary)
                        .frame(width: 22, height: 22)
                        .background(
                            Circle()
                                .fill(Color.white.opacity(0.04))
                                .overlay(Circle().stroke(Brand.separator, lineWidth: 0.5))
                        )
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Add a model from Hugging Face")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundStyle(Brand.typeBody)
                        Text("Paste any org/repo. Added models stay in this list.")
                            .font(.caption2)
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(2)
                    }
                    Spacer(minLength: 10)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                        .rotationEffect(.degrees(addRowExpanded ? 180 : 0))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .contentShape(Rectangle())
                // No expanded-state background tint. The chevron
                // rotation is already a single, clear expand indicator;
                // doubling it with an accent-blue fill made the row
                // look "active" even when the user wasn't interacting
                // with it, and the tint stopped abruptly at the row
                // boundary so the form below felt detached.
            }
            .buttonStyle(.plain)
            .disabled(applyingModelID != nil || checkingCustomRepo || isTransitioning)

            if addRowExpanded {
                // Hairline divider separates the trigger row from the
                // form so the focused text field's stroke can't appear
                // to overlap the subtitle above. Matches the picker's
                // existing `sectionDivider` look (0.5pt Brand.separator)
                // for visual continuity with the rest of the popover.
                Rectangle()
                    .fill(Brand.separator)
                    .frame(height: 0.5)
                    .padding(.horizontal, 14)
                customModelForm
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .opacity(visible ? 1 : 0)
        .offset(y: visible ? 0 : 8)
        .animation(motionEnabled ? .smooth(duration: 0.16) : nil, value: visible)
    }

    @ViewBuilder
    private var customModelForm: some View {
        let canSubmit = !customRepoInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !checkingCustomRepo
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                TextField("org/repo", text: $customRepoInput)
                    .textFieldStyle(.plain)
                    .focused($customRepoFocused)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeHi)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Brand.bgOuter)
                            .overlay(
                                // Softer focus stroke: 0.75pt at 60%
                                // opacity reads as "this field is
                                // active" without looking like a
                                // 1pt hard outline that visually
                                // overlaps the row above.
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .stroke(
                                        customRepoFocused
                                            ? Brand.accentChrome.opacity(0.6)
                                            : Brand.separator,
                                        lineWidth: customRepoFocused ? 0.75 : 0.5
                                    )
                            )
                    )
                    .onChange(of: customRepoInput) { _, _ in
                        customProbe = nil
                        customNoMTPAcknowledged = false
                        errorMessage = nil
                    }
                    .onSubmit { checkAndAddCustomModel() }
                Button {
                    checkAndAddCustomModel()
                } label: {
                    HStack(spacing: 5) {
                        if checkingCustomRepo {
                            ProgressView()
                                .controlSize(.mini)
                                .tint(Brand.bgOuter)
                        }
                        Text(checkingCustomRepo ? "Checking…" : "Check & Add")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .foregroundStyle(canSubmit ? Brand.bgOuter : Brand.typeTertiary)
                    .background(
                        Capsule(style: .continuous)
                            .fill(canSubmit ? AnyShapeStyle(Brand.accentChrome) : AnyShapeStyle(Color.clear))
                            .overlay(
                                Capsule(style: .continuous)
                                    .stroke(
                                        canSubmit ? Color.clear : Brand.separator,
                                        lineWidth: 0.5
                                    )
                            )
                    )
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .disabled(!canSubmit)
                .animation(motionEnabled ? .smooth(duration: 0.16) : nil, value: canSubmit)
            }
            if let probe = customProbe {
                CustomModelProbeRow(
                    probe: probe,
                    acknowledgedNoMTP: $customNoMTPAcknowledged,
                    onAddAnyway: { addCustomModel(repoID: probe.hfRepo) }
                )
            }
        }
        .padding(.horizontal, 14)
        // Generous top + bottom padding so the field's focus stroke
        // has clear air above and below — was the source of the
        // "overlap" complaint when the divider wasn't there to
        // visually separate the trigger row from the form.
        .padding(.top, 12)
        .padding(.bottom, 14)
    }

    @ViewBuilder
    private func errorBar(_ message: String) -> some View {
        Divider().overlay(Brand.separator)
        Text(message)
            .font(.caption2)
            .foregroundStyle(Brand.danger)
            .lineLimit(2)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
    }

    @ViewBuilder
    private func sectionDivider(precedesRow row: Int) -> some View {
        // Matches LaunchOverlay.rowDivider: the divider appears at the
        // same moment the row immediately below it spawns in, so the
        // stagger animation reads as one cohesive reveal.
        let visible = rowsVisibleCount >= row
        Rectangle()
            .fill(Brand.separator)
            .frame(height: 0.5)
            .scaleEffect(x: visible ? 1 : 0, y: 1, anchor: .leading)
            .opacity(visible ? 1 : 0)
    }

    private func select(_ row: ModelPickerPreparedOption) {
        select(option: row.option, resolvedReference: row.resolvedReference)
    }

    private func select(_ option: MTPLXModelOption) {
        select(option: option, resolvedReference: option.resolvedReference)
    }

    private func select(option: MTPLXModelOption, resolvedReference: String) {
        guard applyingModelID == nil, !isTransitioning else { return }
        errorMessage = nil
        applyingModelID = option.id
        var next = backend.configuration
        next.rememberCustomModel(repoID: option.hfModelID)
        next.model = resolvedReference
        normalizeModelScopedDefaults(&next)
        Task {
            do {
                try await backend.applyConfiguration(next, restartIfRunning: true)
                await MainActor.run {
                    applyingModelID = nil
                    presented = false
                }
            } catch {
                print("MTPLX: model switch failed: \(error)")
                await MainActor.run {
                    applyingModelID = nil
                    errorMessage = "Couldn't switch models. Try again."
                }
            }
        }
    }

    private func checkAndAddCustomModel() {
        guard !checkingCustomRepo, applyingModelID == nil, !isTransitioning else { return }
        guard let option = MTPLXModelOption.customHuggingFaceModel(repoID: customRepoInput) else {
            customProbe = nil
            errorMessage = "Enter a Hugging Face repo id like org/name."
            return
        }
        if let existing = preparedRows.first(where: { $0.matches(option.hfModelID) }) {
            select(existing)
            return
        }
        errorMessage = nil
        checkingCustomRepo = true
        let probe = HuggingFaceProbe(endpoint: backend.configuration.hfEndpoint)
        Task {
            let result = await probe.probe(repo: option.hfModelID)
            await MainActor.run {
                checkingCustomRepo = false
                customProbe = result
                switch result.verdict {
                case .ready, .missingSidecar:
                    addCustomModel(repoID: result.hfRepo)
                case .noMTP:
                    if customNoMTPAcknowledged {
                        addCustomModel(repoID: result.hfRepo)
                    }
                case .probeFailed:
                    break
                }
            }
        }
    }

    private func addCustomModel(repoID: String) {
        guard let option = MTPLXModelOption.customHuggingFaceModel(repoID: repoID) else {
            errorMessage = "Enter a Hugging Face repo id like org/name."
            return
        }
        guard applyingModelID == nil, !isTransitioning else { return }
        applyingModelID = option.id
        errorMessage = nil
        var next = backend.configuration
        next.rememberCustomModel(repoID: option.hfModelID)
        next.model = option.resolvedReference
        normalizeModelScopedDefaults(&next)
        Task {
            do {
                try await backend.applyConfiguration(next, restartIfRunning: true)
                await MainActor.run {
                    applyingModelID = nil
                    customRepoInput = ""
                    customProbe = nil
                    customNoMTPAcknowledged = false
                    addRowExpanded = false
                    presented = false
                }
            } catch {
                print("MTPLX: add custom model failed: \(error)")
                await MainActor.run {
                    applyingModelID = nil
                    errorMessage = "Couldn't add that model. Check the repo and try again."
                }
            }
        }
    }

    private func normalizeModelScopedDefaults(_ config: inout MTPLXAppConfiguration) {
        let family = MTPLXModelOption.modelFamily(for: config.model)
        let storedFamily = config.liveSettingsModelFamily?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let alreadyCompatible = storedFamily == family

        guard !alreadyCompatible else { return }

        config.temperature = nil
        config.topP = nil
        config.topK = nil
        config.reasoning = nil
        config.reasoningEffort = nil
        config.prefillChunkTokens = nil

        switch family {
        case "qwen3_5", "qwen3_6", "qwen3_8", "gemma4", "step":
            config.generationMode = "mtp"
            config.loadMTP = true
            config.liveSettingsModelFamily = family
        default:
            config.generationMode = "mtp"
            config.loadMTP = true
            config.liveSettingsModelFamily = nil
        }
    }

    private var restartRequired: Bool {
        switch daemonState.kind {
        case .running: return true
        default: return false
        }
    }

    private var isTransitioning: Bool {
        switch daemonState.kind {
        case .starting, .warming, .stopping: return true
        default: return false
        }
    }

    private var restartHint: String {
        if isTransitioning {
            return "Wait for startup or shutdown to finish."
        }
        if restartRequired {
            return "Switching models restarts the model server."
        }
        return "This model loads next time you start."
    }

    private var motionEnabled: Bool {
        !configuration.performanceLock && !themeStore.reduceMotionPreference
    }

    private var catalogSignature: ModelPickerCatalogSignature {
        ModelPickerCatalogSignature(
            currentModel: configuration.model,
            customModels: configuration.customModels,
            hardware: detectedHardware
        )
    }

    private func modelListHeight(for rowCount: Int) -> CGFloat {
        min(modelListMaxHeight, CGFloat(max(rowCount, 1)) * modelRowEstimatedHeight)
    }

    private func runEnterChoreography() {
        errorMessage = nil
        let signature = catalogSignature
        guard preparedRowsSignature == signature, !preparedRows.isEmpty else {
            preparePickerRows(startChoreographyWhenReady: true)
            return
        }
        startEnterChoreography(rowCount: preparedRows.count + 1)
    }

    private func startEnterChoreography(rowCount: Int) {
        OverlayChoreography.runEnter(
            motionEnabled: motionEnabled,
            rowCount: rowCount,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount
        )
    }

    private func preparePickerRows(startChoreographyWhenReady: Bool = false) {
        let signature = catalogSignature
        let customModels = signature.customModels
        let currentModel = signature.currentModel
        let hardware = signature.hardware

        prepareRowsTask?.cancel()
        prepareRowsTask = Task { @MainActor in
            let rows = await Task.detached(priority: .userInitiated) {
                MTPLXModelOption.pickerCatalog(
                    customModels: customModels,
                    currentModel: currentModel,
                    hardware: hardware
                )
                .map { option in
                    ModelPickerPreparedOption(option: option, currentModel: currentModel)
                }
            }.value

            guard !Task.isCancelled else { return }
            preparedRows = rows
            preparedRowsSignature = signature
            if startChoreographyWhenReady, presented {
                startEnterChoreography(rowCount: rows.count + 1)
            }
        }
    }

    private func detectHardwareForCatalog() {
        guard hardwareTask == nil, detectedHardware == nil else { return }
        hardwareTask = Task { @MainActor in
            let hardware = await HardwareInspector().detect()
            guard !Task.isCancelled else { return }
            detectedHardware = hardware
            hardwareTask = nil
        }
    }

    private func runExitChoreography() {
        addRowExpanded = false
        customProbe = nil
        customNoMTPAcknowledged = false
        checkingCustomRepo = false
        customRepoFocused = false
        OverlayChoreography.runExit(
            motionEnabled: motionEnabled,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount
        )
    }
}

private struct ModelPickerCatalogSignature: Equatable, Sendable {
    let currentModel: String
    let customModels: [MTPLXModelOption]
    let hardware: DetectedHardware?
}

private struct ModelPickerPreparedOption: Equatable, Identifiable, Sendable {
    let option: MTPLXModelOption
    let id: String
    let displayName: String
    let detail: String
    let isInstalled: Bool
    let selected: Bool
    let resolvedReference: String

    init(option: MTPLXModelOption, currentModel: String) {
        let installedLocalPath = option.installedLocalPath

        self.option = option
        self.id = option.id
        self.displayName = option.displayName
        self.detail = option.detail
        self.isInstalled = installedLocalPath != nil
        self.resolvedReference = installedLocalPath ?? option.hfModelID
        self.selected = Self.matches(
            option: option,
            model: currentModel,
            resolvedReference: installedLocalPath ?? option.hfModelID
        )
    }

    func matches(_ model: String) -> Bool {
        Self.matches(option: option, model: model, resolvedReference: resolvedReference)
    }

    private static func matches(
        option: MTPLXModelOption,
        model: String,
        resolvedReference: String
    ) -> Bool {
        let normalizedModel = normalized(model)
        let basename = normalized(URL(fileURLWithPath: model).lastPathComponent)
        if normalizedModel == normalized(option.id) { return true }
        if normalizedModel == normalized(option.displayName) { return true }
        if normalizedModel == normalized(option.shortName) { return true }
        if normalizedModel == normalized(option.hfModelID) { return true }
        if normalizedModel == normalized(resolvedReference) { return true }
        if option.aliases.contains(where: { normalized($0) == normalizedModel }) { return true }
        return option.localCandidates.contains { candidate in
            let expanded = (candidate as NSString).expandingTildeInPath
            return normalized(expanded) == normalizedModel
                || normalized(URL(fileURLWithPath: expanded).lastPathComponent) == basename
        }
    }

    private static func normalized(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "\\", with: "/")
            .replacingOccurrences(of: "--", with: "/")
    }
}

// MARK: - ModelRowView
//
// One row in the model picker. Owns its own hover state so the picker
// doesn't have to track which id the cursor is over. The selection fill
// stays flat and full-width, while hover adds the tactile scale/shadow
// the app expects. That keeps the row feeling alive without going back
// to an inner rounded card competing with the popover surface.
//
// Selection signal lives in three quiet places stacked on top of each
// other: a thin tinted full-width fill (Brand.accentChrome at 8%
// opacity), a brighter title colour (Brand.typeHi vs typeBody), and a
// filled blue checkmark on the trailing edge. Hover is an even
// quieter Color.white opacity-0.04 stripe that the selection fill
// composites on top of, so hovering the selected row still reads as
// "selected" but with a subtle lift.

private struct ModelRowView: View {
    let displayName: String
    let detail: String
    let isInstalled: Bool
    let selected: Bool
    let applying: Bool
    let restartRequired: Bool
    let disabled: Bool
    let visible: Bool
    let motionEnabled: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .firstTextBaseline, spacing: 9) {
                        Text(displayName)
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundStyle(selected ? Brand.typeHi : Brand.typeBody)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        statusBadge
                    }
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(2)
                }
                Spacer(minLength: 10)
                trailingIcon
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .contentShape(Rectangle())
            // Two flat full-width fills stacked: hover sits on top of
            // the selection tint so a hovered selected row reads as
            // selected + lifted. No `RoundedRectangle` here — the
            // popover's `.clipShape` rounds these stripes at the top
            // and bottom of the list automatically.
            .background {
                ZStack {
                    if selected {
                        Brand.accentChrome.opacity(0.10)
                    }
                    if hovering {
                        Color.white.opacity(0.05)
                    }
                }
            }
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(visible ? 1 : 0)
        .offset(y: visible ? 0 : 8)
        .scaleEffect(hovering ? 1.015 : 1.0, anchor: .center)
        .shadow(
            color: .black.opacity(hovering ? 0.32 : 0),
            radius: hovering ? 6 : 0,
            x: 0,
            y: hovering ? 3 : 0
        )
        .animation(motionEnabled ? .spring(response: 0.24, dampingFraction: 0.85) : nil, value: hovering)
        .animation(motionEnabled ? .smooth(duration: 0.16) : nil, value: selected)
        .onHover { isHovering in
            if isHovering { Haptics.tick(.levelChange) }
            hovering = isHovering
        }
    }

    @ViewBuilder
    private var trailingIcon: some View {
        if applying {
            ProgressView()
                .controlSize(.small)
                .scaleEffect(0.76)
                .tint(Brand.typeBody)
        } else if selected {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 15, weight: .semibold))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(Brand.accentChrome)
        } else {
            Image(systemName: restartRequired ? "arrow.clockwise" : "checkmark.circle")
                .font(.system(size: 14, weight: .semibold))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(Brand.typeTertiary)
        }
    }

    /// Status chip — hairline outline only, no fill. The previous
    /// filled treatment (tinted background + same-tint text) read as
    /// "kid's-sticker duotone" because two saturated layers of the
    /// same colour stacked with no neutral anchor. Outline-only lets
    /// the dark popover surface carry the contrast, the text
    /// supplies the colour, and the chip sits as a quiet pill
    /// instead of a stamped badge. Matches the size/weight
    /// vocabulary of the app-wide `PillBadge`.
    @ViewBuilder
    private var statusBadge: some View {
        let tint = isInstalled ? Brand.accentChrome : Brand.warning
        let label = isInstalled ? "Installed" : "HF"
        Text(label)
            .font(.caption2.weight(.medium))
            .foregroundStyle(tint.opacity(0.9))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(
                Capsule(style: .continuous)
                    .strokeBorder(tint.opacity(0.35), lineWidth: 0.5)
            )
    }
}

private struct CustomModelProbeRow: View {
    let probe: OtherModelProbe
    @Binding var acknowledgedNoMTP: Bool
    let onAddAnyway: () -> Void

    var body: some View {
        let (symbol, color) = icon
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: symbol)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(color)
                    .padding(.top, 2)
                Text(probe.message)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.typeHi)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let diagnostic = probe.diagnostic, !diagnostic.isEmpty {
                Text(diagnostic)
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                    .padding(.leading, 20)
            }
            if probe.verdict == .noMTP {
                HStack(alignment: .center, spacing: 10) {
                    Toggle(isOn: $acknowledgedNoMTP) {
                        Text("Add anyway without the speed boost")
                            .font(.caption)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    .toggleStyle(.checkbox)
                    Spacer(minLength: 8)
                    Button("Add") {
                        onAddAnyway()
                    }
                    .font(.system(size: 11, weight: .semibold))
                    .buttonStyle(.plain)
                    .foregroundStyle(acknowledgedNoMTP ? Brand.typeBody : Brand.typeTertiary)
                    .disabled(!acknowledgedNoMTP)
                }
                .padding(.leading, 20)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(color.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(color.opacity(0.36), lineWidth: 0.5)
                )
        )
    }

    private var icon: (String, Color) {
        switch probe.verdict {
        case .ready:
            return ("checkmark.circle.fill", Brand.success)
        case .missingSidecar:
            return ("exclamationmark.triangle.fill", Brand.warning)
        case .noMTP:
            return ("xmark.octagon.fill", Brand.danger)
        case .probeFailed:
            return ("wifi.exclamationmark", Brand.danger)
        }
    }
}

private struct ModelPickerNotch: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
