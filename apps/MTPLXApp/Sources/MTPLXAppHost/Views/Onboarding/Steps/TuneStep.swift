import SwiftUI
import MTPLXAppCore

// MARK: - TuneStep
//
// Runs `mtplx tune` in the background, shows the selected model
// family's candidate checklist and reveals
// the result card with a throughput grid + "Start chatting" CTA.
//
// First-run onboarding waits for an explicit Start tuning click. If
// fan control is missing, the app installs it before running the
// measured candidates.
//   • Completed tune → reveal measured speed in sequence and allow an
//     explicit rerun without substituting old saved numbers.

struct TuneStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    let onFinish: () -> Void
    @State private var hasAutoStarted: Bool = false

    var body: some View {
        OnboardingStepContainer(
            title: title,
            subtitle: subtitle,
            stepIndex: 5,
            stepCount: OnboardingStep.allCases.count,
            onBack: orchestrator.isTuning ? nil : { orchestrator.returnToModelPick() },
            primary: { primaryButton },
            content: { content }
        )
        .onAppear { onAppearStart() }
    }

    private var title: String {
        if !supportsTune {
            return "Defaults configured"
        }
        if orchestrator.tuneResult != nil {
            return "You're set"
        }
        return "Tuning for your Mac"
    }

    private var subtitle: String? {
        if !supportsTune {
            return String(
                format: "%@ uses its backend defaults for this release.".mtplxLocalized,
                modelFamilyLabel
            )
        }
        if let status = orchestrator.tuneStatusMessage {
            return status
        }
        if orchestrator.tuneResult != nil {
            return nil
        }
        return "Finding the fastest setting for your Mac."
    }

    @ViewBuilder
    private var primaryButton: some View {
        if !supportsTune || orchestrator.tuneResult != nil {
            OnboardingPrimaryButton("Open Dashboard") { onFinish() }
        } else if orchestrator.isTuning {
            OnboardingPrimaryButton("Stop") { orchestrator.cancelTune() }
        } else if orchestrator.tuneFailure != nil {
            OnboardingPrimaryButton("Retry") { orchestrator.startTune() }
        } else {
            OnboardingPrimaryButton("Start tuning") { orchestrator.startTune() }
        }
    }

    @ViewBuilder
    private var content: some View {
        if !supportsTune {
            defaultsConfiguredCard
        } else if let result = orchestrator.tuneResult {
            resultCard(result)
        } else {
            checklistView
        }
    }

    private var defaultsConfiguredCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(Brand.success)
            Text(String(format: "%@ is ready.".mtplxLocalized, modelFamilyLabel))
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
            Text("MTPLX will use the model's runtime defaults for this release.")
                .font(.system(size: 13))
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 12)
            HStack(spacing: 6) {
                pillTag(modelTag)
                pillTag("Tune skipped")
                Spacer()
            }
        }
        .onAppear {
            orchestrator.skipTuneForModelDefaults()
        }
    }

    // MARK: - Per-candidate checklist

    private var checklistView: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let status = orchestrator.tuneStatusMessage {
                tuneStatusBanner(status)
            }
            ForEach(orchestrator.tuneCandidates, id: \.self) { candidate in
                candidateRow(for: candidate)
            }
            if let failure = orchestrator.tuneFailure {
                failureBanner(failure)
            }
            Spacer(minLength: 0)
            VStack(alignment: .leading, spacing: 8) {
                Text("Takes a few minutes. Your fans will spin up for accurate timing.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                if !orchestrator.isTuning {
                    Button {
                        orchestrator.skipTuneWithSafeDefault()
                    } label: {
                        Text("Skip for now")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Skip tuning and use a safe default")
                }
            }
        }
    }

    @ViewBuilder
    private func candidateRow(for candidate: TuneCandidate) -> some View {
        let landed = orchestrator.tuneCandidatesLanded[candidate]
        let isRunning = orchestrator.isTuning
            && orchestrator.tuneStatusMessage == nil
            && landed == nil
            && previousCandidatesDone(before: candidate)
        HStack(spacing: 14) {
            statusIcon(landed: landed != nil, isRunning: isRunning)
            VStack(alignment: .leading, spacing: 2) {
                Text(candidate.displayLabel.mtplxLocalized)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
            }
            Spacer()
            if let landed {
                Text(formatTokS(landed.tokS))
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Brand.typeBody)
                    .monospacedDigit()
            } else if isRunning {
                Text("running…")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
            } else {
                Text("—")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    private func statusIcon(landed: Bool, isRunning: Bool) -> some View {
        Group {
            if landed {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(Brand.success)
            } else if isRunning {
                ProgressView().controlSize(.small)
            } else {
                Image(systemName: "circle.dotted")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
        .frame(width: 20, height: 20)
    }

    private func previousCandidatesDone(before candidate: TuneCandidate) -> Bool {
        let all = orchestrator.tuneCandidates
        guard let i = all.firstIndex(of: candidate), i > 0 else { return true }
        for j in 0..<i {
            if orchestrator.tuneCandidatesLanded[all[j]] == nil { return false }
        }
        return true
    }

    // MARK: - Result card

    private func resultCard(_ result: TuneResult) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(headlineForResult(result))
                    .font(.system(.title3, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeHi)
                Spacer(minLength: 10)
                if supportsTune {
                    rerunButton(isSafeDefault(result) ? "Run tuning" : "Rerun")
                }
            }
            Text(result.bestDepth == 0 && result.allCandidates.isEmpty
                ? "Ready. Open the dashboard to use this model's backend defaults."
                : "Ready. Open the dashboard to save this setup.")
                .font(.system(size: 13))
                .foregroundStyle(Brand.typeSecondary)
            if isSafeDefault(result) {
                Text("You can run tuning later for measured speed.")
                    .font(.system(size: 12))
                    .foregroundStyle(Brand.typeTertiary)
            }
            if !result.allCandidates.isEmpty {
                TuneThroughputGrid(result: result)
            }
            Spacer(minLength: 12)
            tpsReveal(for: result)
            Spacer(minLength: 12)
            centeredResultTags(for: result)
        }
    }

    private func rerunButton(_ title: String) -> some View {
        Button {
            orchestrator.startTune()
        } label: {
            Label(title.mtplxLocalized, systemImage: "arrow.clockwise")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Brand.typeSecondary)
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(
                    Capsule(style: .continuous)
                        .fill(Brand.cardSurface)
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.5)
                        )
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel((title == "Run tuning" ? "Run tuning" : "Rerun tuning").mtplxLocalized)
    }

    private func centeredResultTags(for result: TuneResult) -> some View {
        HStack(spacing: 6) {
            Spacer(minLength: 0)
            pillTag(modelTag)
            pillTag(depthTag(for: result))
            if let chip = orchestrator.state.hardware?.chipName {
                pillTag(chip)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .center)
    }

    private func tuneAnimationID(for result: TuneResult) -> String {
        [
            String(format: "%.3f", result.bestTokS),
            String(format: "%.3f", result.bestMultiplierVsAR),
            result.bestCandidate.rawValue,
            result.allCandidates
                .map { "\($0.candidate.rawValue):\(String(format: "%.3f", $0.tokS))" }
                .joined(separator: "|")
        ].joined(separator: "::")
    }

    // MARK: - TPS reveal (the dopamine moment)

    @ViewBuilder
    private func tpsReveal(for result: TuneResult) -> some View {
        let arRow = result.allCandidates.first { $0.candidate == .ar }
        let bestRow = result.allCandidates.first { $0.candidate == result.bestCandidate }
        let arTokS = arRow?.tokS ?? 0
        let bestTokS = bestRow?.tokS ?? result.bestTokS
        let showReveal = arTokS > 0 && bestTokS > 0 && result.bestMultiplierVsAR > 1.0

        if showReveal {
            TuneResultReveal(
                beforeTokS: arTokS,
                nowTokS: bestTokS,
                multiplier: result.bestMultiplierVsAR,
                animationID: tuneAnimationID(for: result)
            )
        }
    }

    private func headlineForResult(_ result: TuneResult) -> String {
        if result.bestDepth == 0 && result.allCandidates.isEmpty {
            return "Model defaults selected."
        }
        if isSafeDefault(result) {
            return "Safe default selected."
        }
        if result.bestDepth == 0 {
            return "Standard mode is fastest on your Mac."
        }
        return "Found your sweet spot."
    }

    private func isSafeDefault(_ result: TuneResult) -> Bool {
        result.allCandidates.isEmpty && result.bestTokS <= 0
    }

    private var modelTag: String {
        orchestrator.state.resolvedModel?.shortName
            ?? orchestrator.state.resolvedRepoID
            ?? "Model"
    }

    private var supportsTune: Bool {
        orchestrator.state.supportsTune
    }

    private var modelFamilyLabel: String {
        switch orchestrator.state.resolvedModelFamily {
        case "gemma4": return "Gemma"
        case "step": return "Step"
        case "glm": return "GLM"
        case "deepseek": return "DeepSeek"
        case "qwen3_5": return "Qwen 3.5"
        case "qwen3_6": return "Qwen 3.6"
        case "qwen3_8": return "Qwen 3.8"
        default: return "This model"
        }
    }

    private func depthTag(for result: TuneResult) -> String {
        if result.bestDepth == 0 { return "Base" }
        if orchestrator.state.resolvedModelFamily == "gemma4" {
            return "Block \(result.bestDepth)"
        }
        return "MTP \(result.bestDepth)"
    }

    /// Same hairline-outline chip vocabulary as `ModelPickStep.badge`
    /// and `ModelPickerOverlay.statusBadge` — caption2 medium SF Pro,
    /// 7/2 padding, separator stroke, no fill. Monospaced semibold
    /// at 10pt read as a stamp; SF Pro at caption2 reads as the
    /// macOS-native chrome label the result panel needs.
    private func pillTag(_ text: String) -> some View {
        Text(text.mtplxLocalized)
            .font(.caption2.weight(.medium))
            .foregroundStyle(Brand.typeSecondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(
                Capsule(style: .continuous)
                    .strokeBorder(Brand.separator, lineWidth: 0.5)
            )
    }

    private func failureBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Brand.warning)
            VStack(alignment: .leading, spacing: 4) {
                Text("Tuning didn't finish")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(message)
                    .font(.system(size: 11))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.warning.opacity(0.4), lineWidth: 0.5)
                )
        )
    }

    private func tuneStatusBanner(_ message: String) -> some View {
        HStack(alignment: .center, spacing: 10) {
            ProgressView()
                .controlSize(.small)
            Text(message)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Brand.typeBody)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    // MARK: - Lifecycle

    private func onAppearStart() {
        guard !hasAutoStarted else { return }
        hasAutoStarted = true
        guard supportsTune else {
            orchestrator.skipTuneForModelDefaults()
            return
        }
    }

    // MARK: - Formatting

    private func formatTokS(_ tokS: Double) -> String {
        if tokS <= 0 { return "—" }
        return String(format: "%.1f tok/s", tokS)
    }
}

// MARK: - TuneThroughputGrid

private struct TuneThroughputGrid: View {
    let result: TuneResult
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var visibleRows: Int = 0
    @State private var barProgress: Double = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("THROUGHPUT")
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.0)
                .foregroundStyle(Brand.typeSecondary)
            ForEach(Array(result.allCandidates.enumerated()), id: \.element.candidate) { index, candidate in
                throughputBar(
                    for: candidate,
                    maxTokS: maxTokS,
                    isBest: candidate.candidate == result.bestCandidate,
                    progress: barProgress
                )
                .opacity(index < visibleRows ? 1 : 0)
                .offset(y: index < visibleRows ? 0 : 5)
            }
        }
        .task(id: animationID) {
            await runReveal()
        }
    }

    private var maxTokS: Double {
        max(0.001, result.allCandidates.map(\.tokS).max() ?? 0.001)
    }

    private var animationID: String {
        result.allCandidates
            .map { "\($0.candidate.rawValue):\(String(format: "%.3f", $0.tokS))" }
            .joined(separator: "|")
    }

    @MainActor
    private func runReveal() async {
        visibleRows = 0
        barProgress = 0
        guard !themeStore.reduceMotionPreference else {
            visibleRows = result.allCandidates.count
            barProgress = 1
            return
        }
        for index in result.allCandidates.indices {
            try? await Task.sleep(nanoseconds: 45_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.spring(response: 0.28, dampingFraction: 0.86)) {
                visibleRows = index + 1
            }
        }
        try? await Task.sleep(nanoseconds: 70_000_000)
        guard !Task.isCancelled else { return }
        withAnimation(.easeOut(duration: 0.7)) {
            barProgress = 1
        }
    }

    private func throughputBar(
        for candidate: TuneCandidateResult,
        maxTokS: Double,
        isBest: Bool,
        progress: Double
    ) -> some View {
        let ratio = max(0, min(1, candidate.tokS / maxTokS)) * max(0, min(1, progress))
        return HStack(spacing: 10) {
            Text(candidate.candidate.compactLabel)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .foregroundStyle(isBest ? Brand.typeHi : Brand.typeSecondary)
                .frame(width: 42, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Brand.separator.opacity(0.4))
                    Capsule()
                        .fill(isBest ? Brand.success : Brand.accentChrome)
                        .frame(width: max(4, geo.size.width * ratio))
                }
            }
            .frame(height: 6)
            Text(String(format: "%.1f tok/s", candidate.tokS))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(isBest ? Brand.typeHi : Brand.typeSecondary)
                .monospacedDigit()
                .frame(width: 78, alignment: .trailing)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(String(
            format: "%@ %.1f tokens per second%@",
            candidate.candidate.displayLabel,
            candidate.tokS,
            isBest ? ", selected" : ""
        ))
    }
}

// MARK: - TuneResultReveal

private struct TuneResultReveal: View {
    let beforeTokS: Double
    let nowTokS: Double
    let multiplier: Double
    let animationID: String
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var revealStage: Int = 0
    @State private var animatedBefore: Double = 0
    @State private var animatedNow: Double = 0
    @State private var animatedMultiplier: Double = 1

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 6) {
                metricBefore
                Image(systemName: "arrow.right")
                    .font(.system(size: 16, weight: .heavy))
                    .foregroundStyle(Brand.typeTertiary)
                    .opacity(revealStage >= 2 ? 1 : 0)
                    .offset(x: revealStage >= 2 ? 0 : -5)
                metricNow
            }
            .frame(maxWidth: .infinity, alignment: .center)
            multiplierBlock
                .padding(.top, 6)
        }
        .frame(maxWidth: .infinity)
        .task(id: animationID) {
            await runReveal()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(String(
            format: "From %.1f tokens per second to %.1f tokens per second, %.2f times faster",
            beforeTokS,
            nowTokS,
            multiplier
        ))
    }

    private var metricBefore: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(String(format: "%.1f", animatedBefore))
                .font(.system(size: 22, weight: .medium, design: .rounded))
                .foregroundStyle(Brand.typeSecondary)
                .monospacedDigit()
                .strikethrough(true, color: Brand.typeTertiary.opacity(0.6))
            Text("BEFORE · tok/s")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.0)
                .foregroundStyle(Brand.typeTertiary)
        }
        .opacity(revealStage >= 1 ? 1 : 0)
        .offset(y: revealStage >= 1 ? 0 : 6)
    }

    private var metricNow: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f", animatedNow))
                .font(.system(size: 32, weight: .heavy, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
            Text("NOW · tok/s")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.0)
                .foregroundStyle(Brand.typeBody)
        }
        .opacity(revealStage >= 2 ? 1 : 0)
        .offset(y: revealStage >= 2 ? 0 : 8)
    }

    private var multiplierBlock: some View {
        VStack(spacing: 4) {
            Text(String(format: "%.2f×", animatedMultiplier))
                .font(.system(size: 40, weight: .heavy, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
                .shadow(color: Brand.typeBody.opacity(0.20), radius: 18, x: 0, y: 6)
            Text("FASTER")
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.6)
                .foregroundStyle(Brand.typeSecondary)
        }
        .opacity(revealStage >= 3 ? 1 : 0)
        .scaleEffect(revealStage >= 3 ? 1 : 0.94)
    }

    @MainActor
    private func runReveal() async {
        revealStage = 0
        animatedBefore = 0
        animatedNow = beforeTokS
        animatedMultiplier = 1
        guard !themeStore.reduceMotionPreference else {
            revealStage = 3
            animatedBefore = beforeTokS
            animatedNow = nowTokS
            animatedMultiplier = multiplier
            return
        }
        try? await Task.sleep(nanoseconds: 160_000_000)
        guard !Task.isCancelled else { return }
        withAnimation(.spring(response: 0.36, dampingFraction: 0.84)) {
            revealStage = 1
        }
        await tickBefore(to: beforeTokS, duration: 720_000_000)
        try? await Task.sleep(nanoseconds: 620_000_000)
        guard !Task.isCancelled else { return }
        withAnimation(.spring(response: 0.38, dampingFraction: 0.84)) {
            revealStage = 2
        }
        await tickNow(to: nowTokS, duration: 860_000_000)
        try? await Task.sleep(nanoseconds: 720_000_000)
        guard !Task.isCancelled else { return }
        withAnimation(.spring(response: 0.44, dampingFraction: 0.78)) {
            revealStage = 3
        }
        await tickMultiplier(to: multiplier, duration: 900_000_000)
    }

    @MainActor
    private func tickBefore(to target: Double, duration: UInt64) async {
        let start = animatedBefore
        await tickValue(from: start, to: target, duration: duration) { value in
            animatedBefore = value
        }
    }

    @MainActor
    private func tickNow(to target: Double, duration: UInt64) async {
        let start = animatedNow
        await tickValue(from: start, to: target, duration: duration) { value in
            animatedNow = value
        }
    }

    @MainActor
    private func tickMultiplier(to target: Double, duration: UInt64) async {
        let start = animatedMultiplier
        await tickValue(from: start, to: target, duration: duration) { value in
            animatedMultiplier = value
        }
    }

    @MainActor
    private func tickValue(
        from start: Double,
        to target: Double,
        duration: UInt64,
        update: @MainActor (Double) -> Void
    ) async {
        let steps = 34
        let sleep = max(1, duration / UInt64(steps))
        for step in 1...steps {
            try? await Task.sleep(nanoseconds: sleep)
            guard !Task.isCancelled else { return }
            let progress = Double(step) / Double(steps)
            let eased = Self.easeOutCubic(progress)
            update(start + (target - start) * eased)
        }
        update(target)
    }

    private static func easeOutCubic(_ progress: Double) -> Double {
        let t = max(0, min(1, progress))
        let inverse = 1 - t
        return 1 - inverse * inverse * inverse
    }
}
