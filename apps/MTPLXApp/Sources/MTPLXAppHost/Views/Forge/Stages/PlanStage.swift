import SwiftUI
import MTPLXAppCore

// MARK: - PlanStage
//
// Read-only summary of what Forge is about to do, with an Advanced
// disclosure that exposes the recipe knobs (body bits, group size,
// MTP policy) for power users. The default-collapsed pattern keeps
// the page calm; users who want to tune don't have to dig.
//
// Hard rule baked in: setting `mtpPolicy = .requantize` triggers a
// loud warning chip + a checkbox the user has to tick before the
// footer "Start build" pill unlocks. Evidence: mlx-lm PR #990 review
// shows quantising MTP weights collapses MoE acceptance to 5-11%.
// The orchestrator's `state.canAdvance` already enforces the gate;
// the warning chip is the user-visible half.

struct PlanStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    @State private var showingAdvanced: Bool = false
    @State private var localRecipe: ForgeRecipe = ForgeRecipe()
    @State private var modelNameEdit: String = ""
    @State private var pendingDaemonConfirm: Bool = false

    private var motionEnabled: Bool {
        !themeStore.reduceMotionPreference
    }

    private var feasibility: ModelFeasibilityVerdict? {
        orchestrator.evaluateFeasibility(modelLibrary: backend.configuration.modelLibrary)
    }

    var body: some View {
        ForgeStageShell(
            title: "Review the build",
            subtitle: "Defaults are picked for your Mac. Open Advanced if you want to tweak.",
            step: .plan,
            symbol: "checklist",
            symbolTint: Brand.typeBody,
            onBack: { orchestrator.goBack() }
        ) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    sourceRow
                    modelNameRow
                    feasibilityRow
                    diskPreflightRow
                    daemonCoexistenceRow
                    recipeSummaryRow
                    advancedDisclosure
                    if localRecipe.degradesMtp {
                        degradedMtpWarning
                    }
                    Spacer(minLength: 0)
                }
            }
            .onAppear {
                localRecipe = orchestrator.state.recipe
                loadModelNameFromState()
            }
            .onChange(of: orchestrator.state.recipe) { _, newValue in
                if localRecipe != newValue { localRecipe = newValue }
            }
        } footer: {
            ForgePrimaryButton(
                "Start build",
                icon: "hammer.fill",
                isEnabled: orchestrator.state.canAdvance && !modelNameCollisionDetected
            ) {
                attemptStartBuild()
            }
        }
        .confirmationDialog(
            "Build a model while one is running?",
            isPresented: $pendingDaemonConfirm,
            titleVisibility: .visible
        ) {
            Button("Unload the model and build", role: .destructive) {
                pauseDaemonAndForge()
            }
            Button("Build anyway (memory may run tight)") {
                syncModelNameToState()
                orchestrator.startBuild(modelLibrary: backend.configuration.modelLibrary)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Building a model and running one share the same memory. Unloading the running model first is safer.")
        }
    }

    // MARK: - Build entry point

    private func attemptStartBuild() {
        syncModelNameToState()
        guard !modelNameCollisionDetected else { return }
        if backend.daemonState.kind == .running {
            pendingDaemonConfirm = true
            return
        }
        orchestrator.startBuild(modelLibrary: backend.configuration.modelLibrary)
    }

    private func pauseDaemonAndForge() {
        Task {
            await stopCoordinator.stopAll(reason: "forge_pause_daemon")
            await MainActor.run {
                syncModelNameToState()
                orchestrator.startBuild(modelLibrary: backend.configuration.modelLibrary)
            }
        }
    }

    // MARK: - Source summary row

    @ViewBuilder
    private var sourceRow: some View {
        if let probe = orchestrator.state.sourceProbe {
            section(title: "Source") {
                VStack(alignment: .leading, spacing: 6) {
                    Text(probe.hfRepo)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Brand.typeHi)
                    HStack(spacing: 6) {
                        chip(text: sourceFormatLabel(probe.sourceFormat))
                        if probe.hasMtpWeights {
                            chip(text: "MTP weights", systemImage: "rays")
                        }
                        if let bytes = probe.estimatedSizeBytes, bytes > 0 {
                            chip(text: formatGiB(Double(bytes) / 1_073_741_824.0))
                        }
                        if let peak = probe.estimatedPeakGiB, peak > 0 {
                            chip(text: String(format: "~%.0f GB peak", peak), systemImage: "memorychip")
                        }
                    }
                }
            }
        }
    }

    // MARK: - Model name

    @ViewBuilder
    private var modelNameRow: some View {
        section(title: "Name") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    TextField("Model name", text: $modelNameEdit)
                        .textFieldStyle(.plain)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Brand.typeHi)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(textFieldBackground(focused: !modelNameEdit.isEmpty))
                        .onChange(of: modelNameEdit) { _, _ in
                            syncModelNameToState()
                        }
                    Text("-MTPLX")
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .foregroundStyle(Brand.bgOuter)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 7)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Brand.typeBody)
                        )
                        .accessibilityLabel("MTPLX suffix locked")
                }
                Text(modelPathPreview)
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if modelNameCollisionDetected {
                    modelNameCollisionWarning
                }
            }
        }
    }

    @ViewBuilder
    private var modelNameCollisionWarning: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "folder.fill.badge.questionmark")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Brand.warning)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text("A model with this name already exists")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text("Pick the next available name so Forge does not collide with an existing build.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Use \(nextAvailableBaseName())") {
                    modelNameEdit = nextAvailableBaseName()
                    syncModelNameToState()
                }
                .font(.caption2)
                .buttonStyle(.plain)
                .foregroundStyle(Brand.accentChrome)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.warning.opacity(0.40), lineWidth: 0.5)
                )
        )
    }

    // MARK: - Hardware feasibility row

    @ViewBuilder
    private var feasibilityRow: some View {
        section(title: "Hardware fit") {
            VStack(alignment: .leading, spacing: 6) {
                if let verdict = feasibility {
                    feasibilityChip(verdict: verdict)
                } else {
                    Text("Detecting hardware…")
                        .font(.caption)
                        .foregroundStyle(Brand.typeTertiary)
                }
                if let hw = orchestrator.state.hardware {
                    Text("\(hw.chipName) · \(Int(hw.unifiedMemoryGiB)) GB unified memory")
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                }
            }
        }
    }

    private func feasibilityChip(verdict: ModelFeasibilityVerdict) -> some View {
        let (text, color, icon): (String, Color, String)
        switch verdict {
        case .recommended:
            text = "Recommended for this Mac"; color = Brand.success; icon = "checkmark.circle.fill"
        case .tightFit:
            text = "Tight fit — may swap under long context"; color = Brand.warning; icon = "exclamationmark.triangle.fill"
        case .insufficientMemory(let needs):
            text = String(format: "Insufficient memory — needs ~%.0f GB safely", needs); color = Brand.danger; icon = "xmark.octagon.fill"
        case .insufficientDisk(let needs):
            text = String(format: "Insufficient disk — needs ~%.0f GB free", needs); color = Brand.danger; icon = "internaldrive"
        }
        return HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(color)
            Text(text)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Brand.typeHi)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule(style: .continuous)
                .fill(color.opacity(0.10))
                .overlay(
                    Capsule(style: .continuous)
                        .stroke(color.opacity(0.35), lineWidth: 0.5)
                )
        )
    }

    // MARK: - Disk pre-flight + daemon coexistence

    @ViewBuilder
    private var diskPreflightRow: some View {
        if let probe = orchestrator.state.sourceProbe,
           let sizeBytes = probe.estimatedSizeBytes,
           sizeBytes > 0
        {
            let neededGiB = Double(sizeBytes) / 1_073_741_824.0 * 2.5
            let freeGiB = orchestrator.freeDiskGiB(modelLibrary: backend.configuration.modelLibrary)
            if freeGiB < neededGiB {
                bannerCard(
                    icon: "internaldrive",
                    tint: Brand.warning,
                    headline: String(format: "Need ~%.0f GB free — you have %.0f GB", neededGiB, freeGiB),
                    detail: "The build downloads the model and writes a converted copy. Free up disk space or pick a smaller model."
                )
            }
        }
    }

    @ViewBuilder
    private var daemonCoexistenceRow: some View {
        if backend.daemonState.kind == .running {
            bannerCard(
                icon: "bolt.fill",
                tint: Brand.accentChrome,
                headline: "A model is loaded",
                detail: "Building will share memory with the running model. You can unload it first when you start the build."
            )
        }
    }

    private func bannerCard(icon: String, tint: Color, headline: String, detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(tint)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text(headline)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(tint.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(tint.opacity(0.30), lineWidth: 0.5)
                )
        )
    }

    private func textFieldBackground(focused: Bool) -> some View {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
            .fill(Brand.bgOuter)
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(
                        focused ? Brand.accentChrome.opacity(0.6) : Brand.separator,
                        lineWidth: focused ? 0.75 : 0.5
                    )
            )
    }

    // MARK: - Recipe summary

    @ViewBuilder
    private var recipeSummaryRow: some View {
        section(title: "Recipe") {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    chip(text: "\(localRecipe.bodyBits)-bit body")
                    chip(text: "g\(localRecipe.bodyGroupSize)")
                    chip(text: localRecipe.bodyMode.rawValue)
                    chip(text: dtypeChipLabel(localRecipe.bodyDtype))
                    chip(text: mtpPolicyLabel(localRecipe.mtpPolicy), systemImage: "shield.lefthalf.filled")
                }
                Text("Picked automatically for your Mac. Open Advanced to override.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
    }

    // MARK: - Advanced (collapsed by default)

    @ViewBuilder
    private var advancedDisclosure: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(motionEnabled ? .spring(response: 0.30, dampingFraction: 0.86) : nil) {
                    showingAdvanced.toggle()
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .rotationEffect(.degrees(showingAdvanced ? 90 : 0))
                    Text("Advanced")
                        .font(.system(size: 11, weight: .semibold))
                        .tracking(0.4)
                }
                .foregroundStyle(Brand.typeSecondary)
                .padding(.vertical, 4)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if showingAdvanced {
                RecipeAdvancedEditor(
                    recipe: $localRecipe,
                    onChange: { newRecipe in
                        orchestrator.updateRecipe(newRecipe)
                    }
                )
                .transition(.opacity.combined(with: .offset(y: 4)))
            }
        }
    }

    // MARK: - Degraded MTP warning

    @ViewBuilder
    private var degradedMtpWarning: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Brand.warning)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 6) {
                Text("This setting kills the speed boost")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text("Compressing the speed-prediction weights drops accuracy from ~80% to ~10%. The built model won't actually run faster than the slow baseline.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Toggle(isOn: ackBinding) {
                    Text("I understand — build it anyway")
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                }
                .toggleStyle(.checkbox)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.warning.opacity(0.40), lineWidth: 0.5)
                )
        )
    }

    private var ackBinding: Binding<Bool> {
        Binding(
            get: { orchestrator.state.hasAcknowledgedDegradedMTP },
            set: { newValue in
                if newValue { orchestrator.acknowledgeDegradedMTP() }
            }
        )
    }

    // MARK: - Helpers

    @ViewBuilder
    private func section<C: View>(title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(Brand.typeTertiary)
            content()
        }
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
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(
            Capsule(style: .continuous)
                .strokeBorder(Brand.separator, lineWidth: 0.5)
        )
    }

    private func sourceFormatLabel(_ format: ForgeSourceFormat) -> String {
        switch format {
        case .bf16Native: return "BF16 native"
        case .mlxAffine: return "MLX-affine"
        case .mlxAffineWithMtp: return "MLX-affine + MTP"
        case .compressedTensorsAwq: return "compressed-tensors AWQ"
        case .hfVllm: return "HF / vLLM"
        case .unknown: return "Unknown"
        }
    }

    private func mtpPolicyLabel(_ policy: ForgeRecipe.MTPPolicy) -> String {
        switch policy {
        case .keepBf16: return "MTP: keep BF16"
        case .extractFromSidecar: return "MTP: extract from sidecar"
        case .requantize: return "MTP: requantise (degraded)"
        }
    }

    private func dtypeChipLabel(_ dtype: ForgeRecipe.BodyDtype) -> String {
        switch dtype {
        case .auto:
            return HardwareInspector.hostPrefersFP16Forge() ? "auto fp16" : "auto bf16"
        case .bf16: return "bf16"
        case .fp16: return "fp16"
        }
    }

    private func formatGiB(_ gib: Double) -> String {
        String(format: "%.1f GB", gib)
    }

    private func loadModelNameFromState() {
        let base = ForgeBrandInfo.baseName(fromBrandedName: orchestrator.state.brand.brandedName)
        let fallback = orchestrator.state.sourceProbe.map { ForgeBrandInfo.defaultBaseName(sourceRepo: $0.hfRepo) } ?? "Model"
        let next = base.isEmpty ? fallback : base
        if modelNameEdit != next {
            modelNameEdit = next
        }
    }

    private var finalBrandedName: String {
        ForgeBrandInfo.resolvedBrandedName(
            userName: modelNameEdit,
            fallbackSourceRepo: orchestrator.state.sourceProbe?.hfRepo
        )
    }

    private func syncModelNameToState() {
        let next = ForgeBrandInfo(brandedName: finalBrandedName)
        if orchestrator.state.brand != next {
            orchestrator.updateBrand(next)
        }
    }

    private var modelPathPreview: String {
        "Forge will save \(expandedModelPath)/"
    }

    private var expandedModelPath: String {
        backend.configuration.modelLibrary.primaryDirectory
            .appendingPathComponent(finalBrandedName, isDirectory: true)
            .path
    }

    private var modelNameCollisionDetected: Bool {
        FileManager.default.fileExists(atPath: expandedModelPath)
    }

    private func nextAvailableBaseName() -> String {
        let base = ForgeBrandInfo.sanitizedBaseName(
            modelNameEdit,
            fallback: orchestrator.state.sourceProbe.map { ForgeBrandInfo.defaultBaseName(sourceRepo: $0.hfRepo) } ?? "Model"
        )
        for i in 1...20 {
            let candidateBase = "\(base)-\(i)"
            let candidateName = ForgeBrandInfo.resolvedBrandedName(userName: candidateBase)
            let path = backend.configuration.modelLibrary.primaryDirectory
                .appendingPathComponent(candidateName, isDirectory: true)
                .path
            if !FileManager.default.fileExists(atPath: path) {
                return candidateBase
            }
        }
        return "\(base)-new"
    }
}

// MARK: - RecipeAdvancedEditor
//
// Three controls in a small, dense card: body bits picker, body
// group size picker, MTP policy segmented control. Every change
// propagates immediately to the orchestrator so the surrounding
// summary chips + degraded-MTP warning update in sync.

private struct RecipeAdvancedEditor: View {
    @Binding var recipe: ForgeRecipe
    var onChange: (ForgeRecipe) -> Void

    private let bitsOptions: [Int] = [3, 4, 5, 6, 8]
    private let groupSizeOptions: [Int] = [32, 64, 128]
    private let hostPrefersFP16 = HardwareInspector.hostPrefersFP16Forge()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            row(label: "Body bits") {
                Picker("", selection: bitsBinding) {
                    ForEach(bitsOptions, id: \.self) { v in
                        Text("\(v)-bit").tag(v)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
            }
            row(label: "Group size") {
                Picker("", selection: groupBinding) {
                    ForEach(groupSizeOptions, id: \.self) { v in
                        Text("g\(v)").tag(v)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
            }
            row(label: "MTP policy") {
                Picker("", selection: policyBinding) {
                    Text("Keep BF16").tag(ForgeRecipe.MTPPolicy.keepBf16)
                    Text("From sidecar").tag(ForgeRecipe.MTPPolicy.extractFromSidecar)
                    Text("Requantise").tag(ForgeRecipe.MTPPolicy.requantize)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
            }
            row(label: "Precision") {
                VStack(alignment: .leading, spacing: 4) {
                    Picker("", selection: dtypeBinding) {
                        Text(hostPrefersFP16 ? "Auto (FP16)" : "Auto (BF16)")
                            .tag(ForgeRecipe.BodyDtype.auto)
                        Text("BF16").tag(ForgeRecipe.BodyDtype.bf16)
                        Text("FP16").tag(ForgeRecipe.BodyDtype.fp16)
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 320)
                    Text("Dtype for the non-quantized weights. FP16 prompt-processes faster on M1/M2 Macs, which have no native BF16.")
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
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

    @ViewBuilder
    private func row<C: View>(label: String, @ViewBuilder content: () -> C) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Brand.typeSecondary)
                .frame(width: 100, alignment: .leading)
            content()
            Spacer(minLength: 0)
        }
    }

    private var bitsBinding: Binding<Int> {
        Binding(
            get: { recipe.bodyBits },
            set: {
                recipe.bodyBits = $0
                Haptics.tick(.alignment)
                onChange(recipe)
            }
        )
    }

    private var groupBinding: Binding<Int> {
        Binding(
            get: { recipe.bodyGroupSize },
            set: {
                recipe.bodyGroupSize = $0
                Haptics.tick(.alignment)
                onChange(recipe)
            }
        )
    }

    private var policyBinding: Binding<ForgeRecipe.MTPPolicy> {
        Binding(
            get: { recipe.mtpPolicy },
            set: {
                recipe.mtpPolicy = $0
                Haptics.tick(.levelChange)
                onChange(recipe)
            }
        )
    }

    private var dtypeBinding: Binding<ForgeRecipe.BodyDtype> {
        Binding(
            get: { recipe.bodyDtype },
            set: {
                recipe.bodyDtype = $0
                Haptics.tick(.levelChange)
                onChange(recipe)
            }
        )
    }
}
