import SwiftUI
import MTPLXAppCore

// MARK: - OnboardingStepContainer
//
// Shared shell used by every onboarding step. Anchors the layout so
// the back/next buttons never shift between steps and the progress
// capsule always reads at the same eye level.
//
// Layout, top to bottom:
//   • Header — title + optional subtitle (auto-sizes vertically)
//   • Content — fixed-height slot the step view fills
//   • Footer — Back (left ghost) + primary CTA (right pill)
//   • Progress capsule — "Step N of M" with a thin fill
//
// All tokens come from `Brand` so this matches the rest of the app.

struct OnboardingStepContainer<Content: View, Primary: View>: View {
    let title: String?
    let subtitle: String?
    let stepIndex: Int
    let stepCount: Int
    let onBack: (() -> Void)?
    /// When true the primary CTA sits centered in the footer (used for
    /// the Welcome step where right-alignment leaves the button looking
    /// orphaned). Default is `false` so subsequent steps keep the
    /// Back-on-left / Next-on-right anchor.
    let centerPrimary: Bool
    @ViewBuilder let primary: () -> Primary
    @ViewBuilder let content: () -> Content

    init(
        title: String? = nil,
        subtitle: String? = nil,
        stepIndex: Int,
        stepCount: Int,
        onBack: (() -> Void)? = nil,
        centerPrimary: Bool = false,
        @ViewBuilder primary: @escaping () -> Primary,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.title = title
        self.subtitle = subtitle
        self.stepIndex = stepIndex
        self.stepCount = stepCount
        self.onBack = onBack
        self.centerPrimary = centerPrimary
        self.primary = primary
        self.content = content
    }

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)
            card
            Spacer(minLength: 12)
            progressCapsule
                .padding(.bottom, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Brand.bgOuter.ignoresSafeArea())
    }

    // MARK: - Card

    private var card: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let title, !title.isEmpty {
                header(title: title)
                    .padding(.top, 28)
                    .padding(.horizontal, 32)
                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(minHeight: 320, idealHeight: 320, alignment: .topLeading)
                    .padding(.top, 24)
                    .padding(.horizontal, 32)
            } else {
                // Header-less variant — Welcome. NO min-height. The
                // card sizes to the content's intrinsic height so the
                // CTA in the footer sits tight underneath the hero,
                // and the whole compact card is then vertically
                // centered in the window by the outer Spacers.
                content()
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.top, 32)
                    .padding(.horizontal, 32)
            }
            footer
                .padding(.top, 24)
                .padding(.horizontal, 28)
                .padding(.bottom, 22)
        }
        .frame(width: Self.cardWidth)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Brand.raisedSurface)
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Brand.separatorStrong, lineWidth: 0.75)
            }
            .shadow(color: .black.opacity(0.55), radius: 22, x: 0, y: 14)
        )
    }

    private func header(title: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.mtplxLocalized)
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
            if let subtitle, !subtitle.isEmpty {
                Text(subtitle.mtplxLocalized)
                    .font(.system(size: 13))
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var footer: some View {
        if centerPrimary {
            // Welcome variant: single CTA visually centered. No
            // Back button (you're on step 1) and no other footer
            // chrome means the button is the only thing in the row.
            HStack {
                Spacer(minLength: 0)
                primary()
                Spacer(minLength: 0)
            }
        } else {
            HStack(spacing: 12) {
                if let onBack {
                    Button(action: onBack) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                                .font(.system(size: 11, weight: .semibold))
                            Text("Back")
                                .font(.system(size: 13, weight: .medium))
                        }
                        .foregroundStyle(Brand.typeSecondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Back to previous step")
                }
                Spacer(minLength: 0)
                primary()
            }
        }
    }

    // MARK: - Progress capsule

    private var progressCapsule: some View {
        let stepProgress = String(
            format: "Step %d of %d".mtplxLocalized,
            stepIndex + 1,
            stepCount
        )
        return VStack(spacing: 8) {
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Brand.separator.opacity(0.5))
                    .frame(width: 220, height: 4)
                Capsule()
                    .fill(Brand.typeBody)
                    .frame(width: max(8, 220 * CGFloat(progressFraction)), height: 4)
                    .animation(.spring(response: 0.45, dampingFraction: 0.85), value: progressFraction)
            }
            .accessibilityHidden(true)
            Text(stepProgress)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(0.6)
                .foregroundStyle(Brand.typeTertiary)
                .contentTransition(.numericText())
                .accessibilityLabel(stepProgress)
        }
    }

    private var progressFraction: Double {
        guard stepCount > 0 else { return 0 }
        return Double(stepIndex + 1) / Double(stepCount)
    }

    // MARK: - Layout constants

    static var cardWidth: CGFloat { 540 }
}

// MARK: - OnboardingPrimaryButton
//
// The pill-style "Next" / "Get Started" / "Start chatting" button used
// across the steps. Lives here so every step renders an identical
// control without copy-pasting the styling.

struct OnboardingPrimaryButton: View {
    let title: String
    let isEnabled: Bool
    let action: () -> Void
    @State private var isHovering: Bool = false
    @State private var isPressed: Bool = false
    @EnvironmentObject private var themeStore: ThemeStore

    init(_ title: String, isEnabled: Bool = true, action: @escaping () -> Void) {
        self.title = title
        self.isEnabled = isEnabled
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Text(title.mtplxLocalized)
                    .font(.system(size: 13, weight: .semibold))
                    .contentTransition(.opacity)
                Image(systemName: "arrow.right")
                    .font(.system(size: 11, weight: .heavy))
                    .opacity(isEnabled ? 1 : 0)
                    .offset(x: isHovering && isEnabled ? 3 : 0)
            }
            .foregroundStyle(isEnabled ? Brand.bgOuter : Brand.typeTertiary)
            .padding(.horizontal, 22)
            .padding(.vertical, 10)
            .background(
                Capsule(style: .continuous)
                    .fill(isEnabled ? AnyShapeStyle(Brand.typeBody) : AnyShapeStyle(Brand.cardSurface))
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(Brand.separator, lineWidth: 0.5)
            )
            .shadow(
                color: isEnabled && isHovering ? Brand.typeBody.opacity(0.35) : .clear,
                radius: isEnabled && isHovering ? 14 : 0,
                x: 0,
                y: isEnabled && isHovering ? 6 : 0
            )
            .scaleEffect(scale)
            .contentShape(Capsule())
            .animation(
                themeStore.reduceMotionPreference
                    ? nil
                    : .spring(response: 0.28, dampingFraction: 0.78),
                value: isHovering
            )
            .animation(
                themeStore.reduceMotionPreference
                    ? nil
                    : .spring(response: 0.18, dampingFraction: 0.72),
                value: isPressed
            )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(title.mtplxLocalized)
        .onHover { hovering in isHovering = hovering && isEnabled }
        // Tracks a real press separately from `.buttonStyle` because
        // `.plain` swallows the in-progress press state we need to
        // drive the scale-down. A DragGesture with minimumDistance 0
        // gives us press-in / press-out events we can spring on.
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in if isEnabled { isPressed = true } }
                .onEnded { _ in isPressed = false }
        )
    }

    private var scale: CGFloat {
        if !isEnabled { return 1.0 }
        if isPressed { return 0.96 }
        if isHovering { return 1.04 }
        return 1.0
    }
}
