import SwiftUI
import MTPLXAppCore

// MARK: - HardwareScanStep
//
// Animates a circular progress ring while `HardwareInspector` runs,
// then reveals the detected chip name, unified memory, GPU cores,
// and tier label. The inspector finishes in ~250 ms but we hold the
// ring at 0.85 for ~0.8 s so the reveal feels earned rather than
// instantaneous.
//
// Mirrors the Aphanes V2 `HardwareScanOnboardingStep` visual rhythm
// but re-themed with MTPLX `Brand` tokens.

struct HardwareScanStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var ringProgress: CGFloat = 0
    @State private var revealHardware: Bool = false

    var body: some View {
        OnboardingStepContainer(
            title: "Your Mac",
            subtitle: "Optimizing setup for your hardware.",
            stepIndex: 1,
            stepCount: OnboardingStep.allCases.count,
            onBack: { orchestrator.goBack() },
            primary: {
                OnboardingPrimaryButton("Next", isEnabled: revealHardware) {
                    orchestrator.goNext()
                }
            },
            content: {
                VStack(spacing: 20) {
                    Spacer(minLength: 12)
                    scanRing
                    if let hardware = orchestrator.state.hardware, revealHardware {
                        hardwareSummary(for: hardware)
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity)
            }
        )
        .onAppear { runScan() }
    }

    // MARK: - Ring + checkmark

    private var scanRing: some View {
        ZStack {
            Circle()
                .stroke(Brand.separator, lineWidth: 4)
                .frame(width: 80, height: 80)
            Circle()
                .trim(from: 0, to: ringProgress)
                .stroke(
                    Brand.typeBody,
                    style: StrokeStyle(lineWidth: 4, lineCap: .round)
                )
                .frame(width: 80, height: 80)
                .rotationEffect(.degrees(-90))
            if revealHardware {
                Image(systemName: "checkmark")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(Brand.typeBody)
                    .transition(.scale.combined(with: .opacity))
            } else {
                Image(systemName: "desktopcomputer")
                    .font(.system(size: 28, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: revealHardware)
        .accessibilityHidden(true)
    }

    // MARK: - Reveal

    private func hardwareSummary(for hardware: DetectedHardware) -> some View {
        VStack(spacing: 14) {
            Text(hardware.chipName)
                .font(.system(size: 22, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .accessibilityLabel(hardware.chipName)
            HStack(spacing: 24) {
                stat(value: Self.formatMemory(hardware.unifiedMemoryBytes), label: "Unified Memory")
                if let cores = hardware.gpuCoreCount, cores > 0 {
                    stat(value: "\(cores)", label: "GPU Cores")
                }
                if let cores = hardware.cpuCoreCount, cores > 0 {
                    stat(value: "\(cores)", label: "CPU Cores")
                }
            }
        }
    }

    private func stat(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
            Text(label.mtplxLocalized.uppercased())
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.0)
                .foregroundStyle(Brand.typeTertiary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(value) \(label)")
    }

    // MARK: - Choreography

    private func runScan() {
        // Avoid re-running on transition back from the next step.
        guard !revealHardware else { return }
        ringProgress = 0
        orchestrator.detectHardware()

        let motionEnabled = !themeStore.reduceMotionPreference
        let rampDuration: Double = motionEnabled ? 0.85 : 0.0
        let holdDuration: Double = motionEnabled ? 0.30 : 0.0

        withAnimation(.easeInOut(duration: rampDuration)) {
            ringProgress = 0.85
        }
        // Wait until the inspector actually returns before snapping to
        // 100% and unveiling the chip summary. Modern Swift Concurrency
        // (was DispatchQueue.main.asyncAfter chains per swiftui-pro
        // swift.md).
        let totalDelay = rampDuration + holdDuration
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(totalDelay))
            await waitForHardwareThenReveal()
        }
    }

    @MainActor
    private func waitForHardwareThenReveal() async {
        if orchestrator.state.hardware != nil {
            withAnimation(.easeInOut(duration: 0.2)) { ringProgress = 1.0 }
            withAnimation(.spring(response: 0.35, dampingFraction: 0.75).delay(0.05)) {
                revealHardware = true
            }
            return
        }
        // Inspector hasn't returned yet — poll every 80 ms. Bounded
        // by the inspector's own ~3 s timeout for the subprocess.
        try? await Task.sleep(for: .milliseconds(80))
        await waitForHardwareThenReveal()
    }

    private static func formatMemory(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1_073_741_824.0
        if gib >= 100 { return String(format: "%.0f GB", gib) }
        return String(format: "%.0f GB", gib.rounded())
    }
}
