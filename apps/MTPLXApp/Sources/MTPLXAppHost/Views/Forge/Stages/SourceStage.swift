import SwiftUI
import MTPLXAppCore

// MARK: - SourceStage
//
// First step of the Forge Create wizard. Mirrors the visual language
// of ModelPickerOverlay.customModelForm (lines 236-319) and its
// CustomModelProbeRow (lines 642-709) verbatim so the muscle memory
// from the "Add Hugging Face model" flow transfers — same monospaced
// repo field, same soft focus stroke, same "Check" pill, same tinted
// result tile per verdict.
//
// Verdict routing:
//   .forgeable      → footer Next pill enabled; wizard advances to Plan
//   .alreadyMTPLX   → CTA pill swaps to "Install instead" inside the
//                     result tile; clicking it registers the repo as a
//                     custom model and exits the wizard
//   .noMtpHeads     → red banner, footer Next disabled (state.canAdvance
//                     already enforces this; banner explains why)
//   .probeFailed    → amber banner with diagnostic line, retry inline
//                     via the Check pill (the field stays populated)

struct SourceStage: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    @State private var repoInput: String = ""
    @FocusState private var fieldFocused: Bool

    private var canCheck: Bool {
        !repoInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !orchestrator.isProbing
    }

    private var motionEnabled: Bool {
        !themeStore.reduceMotionPreference && !backend.configuration.performanceLock
    }

    var body: some View {
        ForgeStageShell(
            title: "Pick a model",
            subtitle: "Paste a Hugging Face link. We'll check the model before downloading anything.",
            step: .source,
            symbol: "link.circle.fill",
            symbolTint: Brand.accentChrome
        ) {
            VStack(alignment: .leading, spacing: 12) {
                inputRow
                helperLine
                if let probe = orchestrator.state.sourceProbe {
                    ForgeSourceProbeRow(
                        probe: probe,
                        onInstallInstead: { installAlreadyMTPLX(probe: probe) }
                    )
                    .transition(.opacity.combined(with: .offset(y: 4)))
                }
            }
            .animation(motionEnabled ? .smooth(duration: 0.22) : nil, value: orchestrator.state.sourceProbe)
            .onAppear {
                if repoInput.isEmpty {
                    repoInput = orchestrator.state.sourceRepoInput
                }
                // Auto-focus the field a beat after the stage
                // transition completes so the keyboard caret doesn't
                // flash mid-animation.
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(180))
                    fieldFocused = true
                }
            }
            .onChange(of: repoInput) { _, newValue in
                orchestrator.setSourceRepo(newValue)
            }
        } footer: {
            ForgePrimaryButton(
                "Next",
                isEnabled: orchestrator.state.canAdvance
            ) {
                orchestrator.goNext()
            }
        }
    }

    // MARK: - Input row

    @ViewBuilder
    private var inputRow: some View {
        HStack(spacing: 8) {
            TextField("org/repo", text: $repoInput)
                .textFieldStyle(.plain)
                .focused($fieldFocused)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Brand.bgOuter)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(
                                    fieldFocused
                                        ? Brand.accentChrome.opacity(0.6)
                                        : Brand.separator,
                                    lineWidth: fieldFocused ? 0.75 : 0.5
                                )
                        )
                )
                .onSubmit {
                    if canCheck { orchestrator.probeSource() }
                }

            Button {
                orchestrator.probeSource()
            } label: {
                HStack(spacing: 5) {
                    if orchestrator.isProbing {
                        ProgressView()
                            .controlSize(.mini)
                            .tint(Brand.bgOuter)
                    }
                    Text((orchestrator.isProbing ? "Checking…" : "Check").mtplxLocalized)
                        .font(.system(size: 12, weight: .semibold))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .foregroundStyle(canCheck ? Brand.bgOuter : Brand.typeTertiary)
                .background(
                    Capsule(style: .continuous)
                        .fill(canCheck ? AnyShapeStyle(Brand.accentChrome) : AnyShapeStyle(Color.clear))
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(canCheck ? Color.clear : Brand.separator, lineWidth: 0.5)
                        )
                )
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .disabled(!canCheck)
            .animation(motionEnabled ? .smooth(duration: 0.16) : nil, value: canCheck)
        }
    }

    @ViewBuilder
    private var helperLine: some View {
        Text("Example: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` or `https://huggingface.co/Qwen/Qwen3.6-27B`.")
            .font(.caption2)
            .foregroundStyle(Brand.typeTertiary)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Install-instead handoff

    private func installAlreadyMTPLX(probe: ForgeSourceProbe) {
        // Already-MTPLX repos belong in the standard picker / install
        // path — not the Forge build pipeline. Stash the repo on the
        // user's custom-models list and bail back to the top of the
        // tab. The model picker overlay (chrome-strip button) will
        // surface it on its next render.
        var config = backend.configuration
        config.rememberCustomModel(repoID: probe.hfRepo)
        try? backend.saveSettings(config)
        orchestrator.resetWizard()
    }
}

// MARK: - ForgeSourceProbeRow
//
// Result tile rendered under the input row. Tinted background +
// matching stroke per verdict (same vocabulary as
// CustomModelProbeRow). Forge-specific verdicts get their own copy
// + CTAs:
//   .alreadyMTPLX → "Install instead" pill that hands off via
//                   onInstallInstead
//   .forgeable    → no inline CTA; the footer "Next" pill is the
//                   advancement path
//   .noMtpHeads   → red banner with the Forge-specific message about
//                   why we refuse
//   .probeFailed  → amber banner + monospace diagnostic for the
//                   underlying HTTP failure

private struct ForgeSourceProbeRow: View {
    let probe: ForgeSourceProbe
    let onInstallInstead: () -> Void

    var body: some View {
        let (symbol, color) = iconForVerdict
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
            sourceFormatChipRow
                .padding(.leading, 20)
            if probe.verdict == .alreadyMTPLX {
                HStack {
                    Spacer(minLength: 0)
                    Button(action: onInstallInstead) {
                        HStack(spacing: 5) {
                            Image(systemName: "arrow.down.circle.fill")
                                .font(.system(size: 11, weight: .semibold))
                            Text("Install instead")
                                .font(.system(size: 11, weight: .semibold))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .foregroundStyle(Brand.bgOuter)
                        .background(
                            Capsule(style: .continuous).fill(color)
                        )
                        .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.top, 2)
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

    private var iconForVerdict: (String, Color) {
        switch probe.verdict {
        case .forgeable:
            return ("checkmark.circle.fill", Brand.success)
        case .alreadyMTPLX:
            return ("checkmark.seal.fill", Brand.accentChrome)
        case .noMtpHeads:
            return ("xmark.octagon.fill", Brand.danger)
        case .probeFailed:
            return ("wifi.exclamationmark", Brand.warning)
        }
    }

    @ViewBuilder
    private var sourceFormatChipRow: some View {
        HStack(spacing: 6) {
            chip(text: sourceFormatLabel)
            if probe.hasMtpWeights {
                chip(text: "MTP weights", systemImage: "rays")
            }
            if let bytes = probe.estimatedSizeBytes, bytes > 0 {
                chip(text: formatGiB(bytes))
            }
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

    private var sourceFormatLabel: String {
        switch probe.sourceFormat {
        case .bf16Native: return "BF16 native"
        case .mlxAffine: return "MLX-affine"
        case .mlxAffineWithMtp: return "MLX-affine + MTP"
        case .compressedTensorsAwq: return "compressed-tensors AWQ"
        case .hfVllm: return "HF / vLLM"
        case .unknown: return "Unknown format"
        }
    }

    private func formatGiB(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        return String(format: "%.1f GB", gib)
    }
}
