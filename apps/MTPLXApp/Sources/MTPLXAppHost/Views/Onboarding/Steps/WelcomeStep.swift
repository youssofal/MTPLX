import SwiftUI
import MTPLXAppCore

// MARK: - WelcomeStep
//
// One cohesive centered group: wordmark → tagline → three pill chips
// → CTA. Designed to read as a single composition rather than a
// fragmented hero / content / footer triptych. The footer's CTA is
// horizontally centered for the welcome step only (other steps still
// use right-anchored Next so the Back / Next pattern stays).
//
// Entrance choreography stagger-reveals each element with subtle
// scale + opacity springs, then settles. Reduce-motion collapses
// every animation to identity.

struct WelcomeStep: View {
    @ObservedObject var orchestrator: OnboardingOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var heroVisible = false
    @State private var taglineVisible = false
    @State private var pillsVisible = false
    @State private var ctaVisible = false

    private let pills: [(symbol: String, text: String)] = [
        ("bolt.fill", "2–3× faster"),
        ("cpu.fill", "Auto-tuned"),
        ("lock.shield.fill", "On-device"),
    ]

    var body: some View {
        OnboardingStepContainer(
            stepIndex: 0,
            stepCount: OnboardingStep.allCases.count,
            centerPrimary: true,
            primary: {
                OnboardingPrimaryButton("Get Started") { orchestrator.goNext() }
                    .opacity(ctaVisible ? 1 : 0)
                    .offset(y: ctaVisible ? 0 : 8)
            },
            content: {
                // Tight intrinsic-height stack. No outer Spacers, no
                // min-height — the container sizes to this content and
                // the footer CTA sits directly below, with the whole
                // compact card centered in the window.
                VStack(spacing: 16) {
                    wordmark
                    tagline
                    pillRow
                        .padding(.top, 8)
                }
                .frame(maxWidth: .infinity)
            }
        )
        .onAppear { runEntrance() }
    }

    // MARK: - Pieces

    private var wordmark: some View {
        WordmarkView(height: 64)
            .accessibilityLabel("MTPLX")
            .scaleEffect(heroVisible ? 1.0 : 0.94)
            .opacity(heroVisible ? 1 : 0)
            .shadow(
                color: heroVisible ? Brand.typeBody.opacity(0.18) : .clear,
                radius: 28,
                x: 0,
                y: 14
            )
    }

    private var tagline: some View {
        Text("The fastest way to run local AI.")
            .font(.system(size: 15, weight: .medium, design: .rounded))
            .foregroundStyle(Brand.typeSecondary)
            .opacity(taglineVisible ? 1 : 0)
            .offset(y: taglineVisible ? 0 : 6)
    }

    private var pillRow: some View {
        HStack(spacing: 10) {
            ForEach(Array(pills.enumerated()), id: \.offset) { _, item in
                pill(symbol: item.symbol, text: item.text)
            }
        }
        .opacity(pillsVisible ? 1 : 0)
        .offset(y: pillsVisible ? 0 : 8)
    }

    private func pill(symbol: String, text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Brand.typeBody)
                .accessibilityHidden(true)
            Text(text.mtplxLocalized)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Brand.typeHi)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    Capsule(style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(text)
    }

    // MARK: - Entrance choreography

    private func runEntrance() {
        guard !heroVisible else { return }
        let motionEnabled = !themeStore.reduceMotionPreference
        guard motionEnabled else {
            heroVisible = true
            taglineVisible = true
            pillsVisible = true
            ctaVisible = true
            return
        }
        withAnimation(.spring(response: 0.55, dampingFraction: 0.82)) {
            heroVisible = true
        }
        // Staggered entrance via modern Swift Concurrency (was three
        // separate DispatchQueue.main.asyncAfter chains, per
        // swiftui-pro swift.md "Never use Grand Central Dispatch").
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(180))
            withAnimation(.spring(response: 0.42, dampingFraction: 0.85)) {
                taglineVisible = true
            }
            try? await Task.sleep(for: .milliseconds(120))
            withAnimation(.spring(response: 0.42, dampingFraction: 0.85)) {
                pillsVisible = true
            }
            try? await Task.sleep(for: .milliseconds(120))
            withAnimation(.spring(response: 0.42, dampingFraction: 0.85)) {
                ctaVisible = true
            }
        }
    }
}
