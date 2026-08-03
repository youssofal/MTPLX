import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - ModelPickStep
//
// Device-aware recommendation list for first-run onboarding. This is
// intentionally narrower than the global app catalog: onboarding should
// suggest good first choices for this Mac, then provide escape hatches
// for Hugging Face repos and complete local MTPLX folders.

struct ModelPickStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var otherInput: String = ""
    @State private var localPathInput: String = ""
    @State private var preparedRows: [PreparedRecommendedModelRow] = []
    @State private var preparedRowsSignature: ModelPickPreparationSignature?
    @State private var prepareRowsTask: Task<Void, Never>?
    @State private var hoveringChoice: ModelPickChoice?
    @FocusState private var otherInputFocused: Bool
    @FocusState private var localPathFocused: Bool

    var body: some View {
        OnboardingStepContainer(
            title: "Recommended models",
            subtitle: subtitleForHardware,
            stepIndex: 2,
            stepCount: OnboardingStep.allCases.count,
            onBack: { orchestrator.goBack() },
            primary: {
                OnboardingPrimaryButton("Next", isEnabled: orchestrator.state.canAdvance) {
                    orchestrator.goNext()
                }
            },
            content: {
                ScrollViewReader { proxy in
                    ScrollView(showsIndicators: false) {
                        VStack(spacing: 12) {
                            if preparedRows.isEmpty {
                                preparingRowsPlaceholder
                            }
                            ForEach(preparedRows) { row in
                                curatedCard(row)
                            }
                            otherRow(proxy: proxy)
                                .id(ModelPickerScrollID.huggingFace)
                            localRow(proxy: proxy)
                                .id(ModelPickerScrollID.localFolder)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                    }
                }
            }
        )
        .onAppear { prepareRecommendedRows() }
        .onDisappear {
            prepareRowsTask?.cancel()
            prepareRowsTask = nil
        }
        .onChange(of: orchestrator.state.hardware) { _, _ in
            prepareRecommendedRows(force: true)
        }
    }

    // MARK: - Subtitle

    private var subtitleForHardware: String {
        guard let hardware = orchestrator.state.hardware else {
            return "Chosen for Apple Silicon and MTPLX speed."
        }
        switch hardware.tier {
        case .legacyApple:
            return String(format: "Chosen for %@ with %.0f GB unified memory.", hardware.chipName, hardware.unifiedMemoryGiB)
        case .modernApple:
            return String(format: "Chosen for %@ with %.0f GB unified memory.", hardware.chipName, hardware.unifiedMemoryGiB)
        case .intel:
            return "MTPLX is built for Apple Silicon. Use a local folder if you want to experiment."
        case .unknown:
            return String(format: "Chosen for this Mac with %.0f GB unified memory.", hardware.unifiedMemoryGiB)
        }
    }

    // MARK: - Curated recommendations

    private var preparingRowsPlaceholder: some View {
        HStack(spacing: 10) {
            ProgressView()
                .controlSize(.small)
                .scaleEffect(0.72)
            Text("Preparing recommendations")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Brand.typeSecondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func prepareRecommendedRows(force: Bool = false) {
        let hardware = orchestrator.state.hardware
        let signature = ModelPickPreparationSignature(hardware: hardware)
        guard force || preparedRowsSignature != signature else { return }

        prepareRowsTask?.cancel()
        prepareRowsTask = Task { @MainActor in
            let rows = await Task.detached(priority: .userInitiated) {
                Self.makePreparedRows(for: hardware)
            }.value

            guard !Task.isCancelled else { return }
            var transaction = Transaction()
            transaction.animation = nil
            withTransaction(transaction) {
                preparedRows = rows
                preparedRowsSignature = signature
            }
        }
    }

    private nonisolated static func makePreparedRows(for hardware: DetectedHardware?) -> [PreparedRecommendedModelRow] {
        let rows = RecommendedModelRow.rows(
            for: MTPLXModelOption.recommendedCatalogIDs(for: hardware)
        )
        let prepared = rows.compactMap { prepare(row: $0, hardware: hardware) }
        let visible = prepared.filter(\.shouldShow)
        return visible.isEmpty ? Array(prepared.prefix(1)) : visible
    }

    private nonisolated static func prepare(
        row: RecommendedModelRow,
        hardware: DetectedHardware?
    ) -> PreparedRecommendedModelRow? {
        guard let model = model(for: row, hardware: hardware) else { return nil }
        let isInstalled = model.isInstalled
        let verdict = Self.verdict(for: model, hardware: hardware, isInstalled: isInstalled)
        let shouldShow: Bool
        if hardware == nil || isInstalled {
            shouldShow = true
        } else {
            switch verdict {
            case .recommended, .tightFit, .insufficientDisk:
                shouldShow = true
            case .insufficientMemory:
                shouldShow = false
            }
        }
        return PreparedRecommendedModelRow(
            id: model.id,
            choice: row.choice,
            model: model,
            logo: row.logo,
            title: row.title,
            detail: row.detail,
            verdict: verdict,
            isInstalled: isInstalled,
            shouldShow: shouldShow,
            isUsable: Self.isUsable(verdict)
        )
    }

    private nonisolated static func verdict(
        for model: MTPLXModelOption,
        hardware: DetectedHardware?,
        isInstalled: Bool
    ) -> ModelFeasibilityVerdict {
        let diskFreeGiB = isInstalled ? Double.greatestFiniteMagnitude : freeDiskGiB()
        return ModelFeasibility().evaluate(
            model: model,
            chipTier: hardware?.tier ?? .unknown,
            ramGiB: hardware?.unifiedMemoryGiB ?? 0,
            diskFreeGiB: diskFreeGiB
        )
    }

    private nonisolated static func freeDiskGiB() -> Double {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let values = try? home.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let bytes = values?.volumeAvailableCapacityForImportantUsage ?? 0
        return Double(bytes) / 1_073_741_824.0
    }

    private nonisolated static func model(for row: RecommendedModelRow, hardware: DetectedHardware?) -> MTPLXModelOption? {
        let id: String
        if row.choice == .curatedQwen35NineBSpeed {
            id = hardware?.tier == .legacyApple
                ? "qwen35-9b-optimized-speed-fp16"
                : row.modelID
        } else if row.choice == .curatedSpeed {
            id = hardware?.tier == .legacyApple ? "optimized-speed-fp16" : row.modelID
        } else if row.choice == .curatedQwen35BSpeed {
            id = hardware?.tier == .legacyApple
                ? "qwen36-35b-a3b-optimized-speed-fp16"
                : row.modelID
        } else if row.choice == .curatedQwen35BBalance {
            id = hardware?.tier == .legacyApple
                ? "qwen36-35b-a3b-optimized-balance-fp16"
                : row.modelID
        } else {
            id = row.modelID
        }
        return MTPLXModelOption.officialCatalog.first { $0.id == id }
    }

    private func curatedCard(_ row: PreparedRecommendedModelRow) -> some View {
        let model = row.model
        let selected = orchestrator.state.pick == row.choice
        let verdict = row.verdict
        let isUsable = row.isUsable
        let hovering = hoveringChoice == row.choice && isUsable

        return Button {
            guard isUsable else { return }
            withAnimation(.spring(response: 0.35, dampingFraction: 0.82)) {
                orchestrator.select(row.choice)
            }
        } label: {
            HStack(alignment: .top, spacing: 14) {
                ProviderLogoMark(kind: row.logo, selected: selected)
                    .frame(width: 38, height: 38)
                    .scaleEffect(selected ? 1.06 : 1.0)
                    .animation(.spring(response: 0.32, dampingFraction: 0.78), value: selected)

                VStack(alignment: .leading, spacing: 7) {
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 8) {
                            titleText(row.title)
                            badgeLine(
                                model: model,
                                selected: selected,
                                verdict: verdict,
                                isInstalled: row.isInstalled
                            )
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            titleText(row.title)
                            badgeLine(
                                model: model,
                                selected: selected,
                                verdict: verdict,
                                isInstalled: row.isInstalled
                            )
                        }
                    }

                    Text(row.detail)
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeSecondary)
                        .fixedSize(horizontal: false, vertical: true)

                    verdictMessage(verdict)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Brand.cardSurface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(selected ? Brand.accentChrome : Brand.separator, lineWidth: selected ? 1.4 : 0.5)
                    )
            )
            .scaleEffect(hovering ? 1.012 : 1.0)
            .shadow(
                color: .black.opacity(hovering ? 0.22 : 0),
                radius: hovering ? 7 : 0,
                x: 0,
                y: hovering ? 4 : 0
            )
            .opacity(isUsable ? 1.0 : 0.55)
            .contentShape(Rectangle())
            .animation(
                themeStore.reduceMotionPreference ? nil : .spring(response: 0.28, dampingFraction: 0.84),
                value: hovering
            )
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity)
        .disabled(!isUsable)
        .zIndex(hovering ? 1 : 0)
        .onHover { isHovering in
            hoveringChoice = isHovering ? row.choice : nil
        }
    }

    private func titleText(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(Brand.typeHi)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func badgeLine(
        model: MTPLXModelOption?,
        selected: Bool,
        verdict: ModelFeasibilityVerdict?,
        isInstalled: Bool
    ) -> some View {
        HStack(spacing: 8) {
            if model != nil, isInstalled {
                badge("Installed", color: Brand.success)
            } else if selected, let verdict, case .recommended = verdict {
                badge("Recommended", color: Brand.accentChrome)
            }
            if let model {
                badge(Self.formatBytes(model.sizeBytes), color: Brand.typeTertiary)
            }
        }
    }

    @ViewBuilder
    private func verdictMessage(_ verdict: ModelFeasibilityVerdict) -> some View {
        switch verdict {
        case .tightFit:
            Text("Will run, but memory will be tight on long chats.")
                .font(.caption2)
                .foregroundStyle(Brand.warning)
        case .insufficientMemory(let needs):
            Text(String(format: "Needs at least %.0f GB of memory.", needs))
                .font(.caption2)
                .foregroundStyle(Brand.danger)
        case .insufficientDisk(let needs):
            Text(String(format: "Needs at least %.0f GB of free disk space.", needs))
                .font(.caption2)
                .foregroundStyle(Brand.danger)
        case .recommended:
            EmptyView()
        }
    }

    // MARK: - Hugging Face row

    private func otherRow(proxy: ScrollViewProxy) -> some View {
        let isOther: Bool
        if case .other = orchestrator.state.pick { isOther = true } else { isOther = false }
        let disclosureSpring = Animation.spring(response: 0.42, dampingFraction: 0.82)

        return VStack(alignment: .leading, spacing: 10) {
            disclosureButton(
                title: "Use a different model from Hugging Face",
                detail: "Paste any org/repo with MTPLX weights.",
                icon: .huggingFace,
                isExpanded: isOther
            ) {
                withAnimation(themeStore.reduceMotionPreference ? nil : disclosureSpring) {
                    orchestrator.select(isOther ? .none : .other(hfRepo: otherInput))
                }
                if !isOther {
                    reveal(proxy, id: .huggingFace) {
                        otherInputFocused = true
                    }
                }
            }

            if isOther {
                otherExpanded
                    .transition(.asymmetric(
                        insertion: .move(edge: .top)
                            .combined(with: .opacity)
                            .combined(with: .scale(scale: 0.97, anchor: .top)),
                        removal: .opacity.combined(with: .move(edge: .top))
                    ))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var otherExpanded: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                inputField(
                    placeholder: "org/repo",
                    text: $otherInput,
                    isFocused: otherInputFocused
                )
                .focused($otherInputFocused)
                .onChange(of: otherInput) { _, newValue in
                    orchestrator.select(.other(hfRepo: newValue))
                }
                .onSubmit {
                    if !otherInput.isEmpty {
                        orchestrator.probeOther(repo: otherInput)
                    }
                }

                checkButton(
                    title: orchestrator.isProbingOther ? "Checking..." : "Check Model",
                    isBusy: orchestrator.isProbingOther,
                    isEnabled: !otherInput.isEmpty && !orchestrator.isProbingOther
                ) {
                    orchestrator.probeOther(repo: otherInput)
                }
            }

            if let probe = orchestrator.state.otherProbe {
                probeResultRow(probe)
                    .transition(.opacity.combined(with: .offset(y: -2)))
            }
        }
    }

    // MARK: - Local folder row

    private func localRow(proxy: ScrollViewProxy) -> some View {
        let isLocal: Bool
        if case .local = orchestrator.state.pick { isLocal = true } else { isLocal = false }
        let disclosureSpring = Animation.spring(response: 0.42, dampingFraction: 0.82)

        return VStack(alignment: .leading, spacing: 10) {
            disclosureButton(
                title: "Use a local model folder",
                detail: "Paste a complete MTPLX model directory on this Mac.",
                icon: .localFolder,
                isExpanded: isLocal
            ) {
                withAnimation(themeStore.reduceMotionPreference ? nil : disclosureSpring) {
                    orchestrator.select(isLocal ? .none : .local(path: localPathInput))
                }
                if !isLocal {
                    reveal(proxy, id: .localFolder) {
                        localPathFocused = true
                    }
                }
            }

            if isLocal {
                localExpanded
                    .transition(.asymmetric(
                        insertion: .move(edge: .top)
                            .combined(with: .opacity)
                            .combined(with: .scale(scale: 0.97, anchor: .top)),
                        removal: .opacity.combined(with: .move(edge: .top))
                    ))
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var localExpanded: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                inputField(
                    placeholder: "/Users/you/.mtplx/models/...",
                    text: $localPathInput,
                    isFocused: localPathFocused
                )
                .focused($localPathFocused)
                .onChange(of: localPathInput) { _, newValue in
                    orchestrator.select(.local(path: newValue))
                }
                .onSubmit {
                    if !localPathInput.isEmpty {
                        orchestrator.probeLocal(path: localPathInput)
                    }
                }

                checkButton(
                    title: "Check Folder",
                    isBusy: false,
                    isEnabled: !localPathInput.isEmpty
                ) {
                    orchestrator.probeLocal(path: localPathInput)
                }
            }

            if let probe = orchestrator.state.localProbe {
                localProbeResultRow(probe)
                    .transition(.opacity.combined(with: .offset(y: -2)))
            }
        }
    }

    private func reveal(
        _ proxy: ScrollViewProxy,
        id: ModelPickerScrollID,
        focus: @escaping @MainActor () -> Void
    ) {
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(160))
            focus()
            withAnimation(themeStore.reduceMotionPreference ? nil : .easeOut(duration: 0.22)) {
                proxy.scrollTo(id, anchor: .bottom)
            }
        }
    }

    // MARK: - Shared input rows

    private func disclosureButton(
        title: String,
        detail: String,
        icon: ProviderLogoKind,
        isExpanded: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 12) {
                ProviderLogoMark(kind: icon, selected: isExpanded)
                    .frame(width: 30, height: 30)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Brand.typeHi)
                    Text(detail)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: "chevron.down")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.typeTertiary)
                    .rotationEffect(.degrees(isExpanded ? 180 : 0))
                    .animation(.spring(response: 0.42, dampingFraction: 0.82), value: isExpanded)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Brand.cardSurface)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(isExpanded ? Brand.accentChrome : Brand.separator, lineWidth: isExpanded ? 1.4 : 0.5)
                    )
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func inputField(
        placeholder: String,
        text: Binding<String>,
        isFocused: Bool
    ) -> some View {
        TextField(placeholder, text: text)
            .textFieldStyle(.plain)
            .font(.system(size: 13, design: .monospaced))
            .foregroundStyle(Brand.typeHi)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Brand.bgOuter)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(
                                isFocused ? Brand.accentChrome : Brand.separator,
                                lineWidth: isFocused ? 1 : 0.5
                            )
                    )
            )
            .animation(
                themeStore.reduceMotionPreference ? nil : .easeOut(duration: 0.15),
                value: isFocused
            )
    }

    private func checkButton(
        title: String,
        isBusy: Bool,
        isEnabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if isBusy {
                    ProgressView().controlSize(.mini)
                }
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .foregroundStyle(Brand.typeBody)
            .background(Capsule().stroke(Brand.separator, lineWidth: 0.5))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1.0 : 0.5)
    }

    @ViewBuilder
    private func probeResultRow(_ probe: OtherModelProbe) -> some View {
        let (symbol, color) = probeIcon(for: probe.verdict)
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
                Toggle(isOn: Binding(
                    get: { orchestrator.state.hasAcknowledgedOtherWarning },
                    set: { newValue in if newValue { orchestrator.acknowledgeOtherWarning() } }
                )) {
                    Text("Continue anyway - I know it'll be slower")
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                }
                .toggleStyle(.checkbox)
                .padding(.leading, 20)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(color.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(color.opacity(0.4), lineWidth: 0.5)
                )
        )
    }

    private func localProbeResultRow(_ probe: LocalModelProbe) -> some View {
        let (symbol, color) = localProbeIcon(for: probe.verdict)
        return VStack(alignment: .leading, spacing: 8) {
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
                    .lineLimit(3)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(color.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(color.opacity(0.4), lineWidth: 0.5)
                )
        )
    }

    private func probeIcon(for verdict: OtherModelProbe.Verdict) -> (String, Color) {
        switch verdict {
        case .ready: return ("checkmark.circle.fill", Brand.success)
        case .missingSidecar: return ("exclamationmark.triangle.fill", Brand.warning)
        case .noMTP: return ("xmark.octagon.fill", Brand.danger)
        case .probeFailed: return ("wifi.exclamationmark", Brand.danger)
        }
    }

    private func localProbeIcon(for verdict: LocalModelProbe.Verdict) -> (String, Color) {
        switch verdict {
        case .ready: return ("checkmark.circle.fill", Brand.success)
        case .notFound: return ("folder.badge.questionmark", Brand.danger)
        case .incomplete: return ("exclamationmark.triangle.fill", Brand.warning)
        }
    }

    // MARK: - Helpers

    private nonisolated static func isUsable(_ verdict: ModelFeasibilityVerdict) -> Bool {
        switch verdict {
        case .recommended, .tightFit: return true
        case .insufficientMemory, .insufficientDisk: return false
        }
    }

    private nonisolated static func formatBytes(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        return String(format: "%.0f GB", gib.rounded())
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(color.opacity(0.9))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(
                Capsule(style: .continuous)
                    .strokeBorder(color.opacity(0.35), lineWidth: 0.5)
            )
    }
}

private enum ModelPickerScrollID: Hashable {
    case huggingFace
    case localFolder
}

private struct ModelPickPreparationSignature: Equatable {
    let chipName: String?
    let appleSiliconGeneration: String?
    let memoryBytes: Int64?
    let gpuCoreCount: Int?
    let cpuCoreCount: Int?

    init(hardware: DetectedHardware?) {
        chipName = hardware?.chipName
        appleSiliconGeneration = hardware?.appleSiliconGeneration
        memoryBytes = hardware?.unifiedMemoryBytes
        gpuCoreCount = hardware?.gpuCoreCount
        cpuCoreCount = hardware?.cpuCoreCount
    }
}

private struct PreparedRecommendedModelRow: Identifiable, Equatable, Sendable {
    let id: String
    let choice: ModelPickChoice
    let model: MTPLXModelOption
    let logo: ProviderLogoKind
    let title: String
    let detail: String
    let verdict: ModelFeasibilityVerdict
    let isInstalled: Bool
    let shouldShow: Bool
    let isUsable: Bool
}

private struct RecommendedModelRow: Identifiable, Sendable {
    var id: String { modelID }
    let choice: ModelPickChoice
    let modelID: String
    let logo: ProviderLogoKind
    let title: String
    let detail: String

    static func rows(for catalogIDs: [String]) -> [RecommendedModelRow] {
        var seen = Set<ModelPickChoice>()
        return catalogIDs.compactMap(row(for:)).filter { row in
            seen.insert(row.choice).inserted
        }
    }

    static func row(for catalogID: String) -> RecommendedModelRow? {
        switch catalogID {
        case "qwen35-9b-optimized-speed", "qwen35-9b-optimized-speed-fp16":
            return .qwen9B
        case "optimized-speed-v2":
            return .qwen27SpeedV2
        case "optimized-speed", "optimized-speed-fp16":
            return .qwen27Speed
        case "qwen36-35b-a3b-optimized-speed", "qwen36-35b-a3b-optimized-speed-fp16":
            return .qwen35Speed
        case "qwen36-35b-a3b-optimized-balance", "qwen36-35b-a3b-optimized-balance-fp16":
            return .qwen35Balance
        case "optimized-quality":
            return .qwen27Quality
        case "gemma4-optimized-speed":
            return .gemma31
        default:
            return nil
        }
    }

    static let qwen9B = RecommendedModelRow(
        choice: .curatedQwen35NineBSpeed,
        modelID: "qwen35-9b-optimized-speed",
        logo: .qwen,
        title: "Qwen 3.5 9B Optimized Speed",
        detail: "6-bit quantization. Strong small-Mac speed pick."
    )

    static let qwen27Speed = RecommendedModelRow(
        choice: .curatedSpeed,
        modelID: "optimized-speed",
        logo: .qwen,
        title: "Qwen 3.6 27B Optimized Speed",
        detail: "Smaller 4-bit model. A little faster for short chats."
    )

    static let qwen27SpeedV2 = RecommendedModelRow(
        choice: .curatedSpeedV2,
        modelID: "optimized-speed-v2",
        logo: .qwen,
        title: "Qwen 3.6 27B Optimized Speed V2",
        detail: "Much higher quality for coding. Dynamic 4-bit hybrid quantization keeps hand-tuned sensitive parts at up to 16-bit. Faster on long agent tasks, slightly larger, and a little slower for short chats."
    )

    static let qwen35Speed = RecommendedModelRow(
        choice: .curatedQwen35BSpeed,
        modelID: "qwen36-35b-a3b-optimized-speed",
        logo: .qwen,
        title: "Qwen 3.6 35B-A3B Optimized Speed",
        detail: "4-bit quantization. Blazingly fast and quite smart."
    )

    static let qwen35Balance = RecommendedModelRow(
        choice: .curatedQwen35BBalance,
        modelID: "qwen36-35b-a3b-optimized-balance",
        logo: .qwen,
        title: "Qwen 3.6 35B-A3B Optimized Balance",
        detail: "6-bit quantization. Stronger balance of speed and quality."
    )

    static let qwen27Quality = RecommendedModelRow(
        choice: .curatedQuality,
        modelID: "optimized-quality",
        logo: .qwen,
        title: "Qwen 3.6 27B Optimized Quality",
        detail: "Maximum quality. Moderate speeds."
    )

    static let gemma31 = RecommendedModelRow(
        choice: .curatedGemmaSpeed,
        modelID: "gemma4-optimized-speed",
        logo: .google,
        title: "Gemma 4 31B Optimized Speed",
        detail: "High quality. Moderate speeds."
    )
}

private enum ProviderLogoKind: Equatable, Sendable {
    case qwen
    case google
    case huggingFace
    case localFolder
}

private struct ProviderLogoMark: View {
    let kind: ProviderLogoKind
    let selected: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(selected ? Brand.typeBody : Brand.cardSurface)
                .overlay(
                    Circle()
                        .stroke(selected ? Brand.typeBody.opacity(0.2) : Brand.separator, lineWidth: 0.5)
                )
            content
        }
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private var content: some View {
        switch kind {
        case .qwen:
            qwenMark
        case .google:
            googleMark
        case .huggingFace:
            Image(systemName: "link")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(selected ? Brand.bgOuter : Brand.typeBody)
        case .localFolder:
            Image(systemName: "folder.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(selected ? Brand.bgOuter : Brand.typeBody)
        }
    }

    private var qwenMark: some View {
        Group {
            if let image = Self.cachedQwenIcon {
                (selected ? Brand.bgOuter : Brand.typeBody)
                    .mask {
                        Image(nsImage: image)
                            .resizable()
                            .interpolation(.high)
                            .scaledToFit()
                    }
            } else {
                Image(systemName: "sparkles")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(selected ? Brand.bgOuter : Brand.typeBody)
            }
        }
        .padding(4)
    }

    private var googleMark: some View {
        let tint = selected ? Brand.bgOuter : Brand.typeBody
        let cutout = selected ? Brand.typeBody : Brand.cardSurface
        return ZStack {
            Circle()
                .trim(from: 0.00, to: 0.25)
                .stroke(tint, style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
                .rotationEffect(.degrees(-18))
            Circle()
                .trim(from: 0.25, to: 0.43)
                .stroke(tint, style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
                .rotationEffect(.degrees(-18))
            Circle()
                .trim(from: 0.43, to: 0.63)
                .stroke(tint, style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
                .rotationEffect(.degrees(-18))
            Circle()
                .trim(from: 0.63, to: 0.91)
                .stroke(tint, style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
                .rotationEffect(.degrees(-18))
            Rectangle()
                .fill(tint)
                .frame(width: 10, height: 3.2)
                .offset(x: 5, y: 1)
            Rectangle()
                .fill(cutout)
                .frame(width: 7, height: 8)
                .offset(x: 8, y: -3)
        }
        .frame(width: 19, height: 19)
    }

    private static let cachedQwenIcon: NSImage? = {
        let name = "QwenIcon"
        let ext = "png"
        if let url = Bundle.main.url(forResource: name, withExtension: ext),
           let img = NSImage(contentsOf: url) {
            return img
        }
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: name, withExtension: ext),
               let img = NSImage(contentsOf: url) {
                return img
            }
        }
        return nil
    }()
}
