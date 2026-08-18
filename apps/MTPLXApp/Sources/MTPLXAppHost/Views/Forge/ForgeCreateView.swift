import SwiftUI
import MTPLXAppCore

enum ForgeStageLayout {
    static let cardWidth: CGFloat = 560
    static let contentPadding: CGFloat = 28
    static let footerPadding: CGFloat = 24
    static let contentWidth: CGFloat = cardWidth - contentPadding * 2
    static let footerWidth: CGFloat = cardWidth - footerPadding * 2
}

// MARK: - ForgeCreateView
//
// Root of the Forge Create sub-mode. Switches on the orchestrator's
// `state.step` and renders one stage view at a time. Lives inside
// the Forge tab; the segmented header (Create | Discover | My Models)
// + `+ New` CTA arrive in a subsequent todo and wrap this view.
//
// Each stage is a separate file under Forge/Stages/. They all share
// the same outer `ForgeStageShell` chrome (header, content slot,
// footer with Back + primary CTA) so the wizard reads as one
// linear flow regardless of which step is active.
//
// Step transitions reuse the crossfade-scale pattern from
// OnboardingExperienceView so the Forge wizard feels like a sibling
// of the first-launch experience visually.

struct ForgeCreateView: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        ZStack {
            Brand.bgOuter
            Group {
                switch orchestrator.state.step {
                case .source:
                    SourceStage()
                        .transition(stepTransition)
                case .plan:
                    PlanStage()
                        .transition(stepTransition)
                case .convert:
                    ConvertStage()
                        .transition(stepTransition)
                case .calibrate:
                    CalibrateStage()
                        .transition(stepTransition)
                case .verify:
                    VerifyStage()
                        .transition(stepTransition)
                case .brand:
                    BrandStage()
                        .transition(stepTransition)
                case .registered:
                    RegisteredStage()
                        .transition(stepTransition)
                case .publishing:
                    PublishStage()
                        .transition(stepTransition)
                }
            }
            .animation(orchestrator.isBuilding ? nil : stepAnimation, value: orchestrator.state.step)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // `.focusable` is needed for the Esc key handler to receive
        // key events, but the default AppKit focus ring (the cyan
        // selection halo around the whole content area) is the
        // single ugliest visual on the surface. `.focusEffectDisabled`
        // suppresses the ring without killing the focus, so Esc still
        // steps backwards through the wizard.
        .focusable(true)
        .focusEffectDisabled()
        .onKeyPress(.escape) {
            if [ForgeStep.convert, .calibrate, .verify].contains(orchestrator.state.step) {
                return .ignored
            }
            if orchestrator.state.step != .source {
                orchestrator.goBack()
                return .handled
            }
            return .ignored
        }
        .onAppear {
            orchestrator.detectHardwareIfNeeded()
            Task { await orchestrator.checkBackendAvailability() }
        }
    }

    /// Matches OnboardingExperienceView's transition. Outgoing card
    /// scales up slightly + fades; incoming scales up from 0.96 + fades.
    private var stepTransition: AnyTransition {
        .asymmetric(
            insertion: .opacity.combined(with: .scale(scale: 0.96, anchor: .center)),
            removal: .opacity.combined(with: .scale(scale: 1.02, anchor: .center))
        )
    }

    private var stepAnimation: Animation? {
        themeStore.reduceMotionPreference
            ? nil
            : .spring(response: 0.42, dampingFraction: 0.86)
    }
}

// MARK: - ForgeStageShell
//
// Shared shell every Forge stage wraps its content in. Mirrors
// `OnboardingStepContainer` (same card width, same Brand surface,
// same Back + primary CTA footer) but indexed against ForgeStep
// not OnboardingStep so the progress capsule reads accurately.
//
// Stages that don't want a footer (e.g. RegisteredStage with three
// peer CTAs) can pass `footer: { EmptyView() }`; the progress
// capsule is still rendered.

struct ForgeStageShell<Content: View, Footer: View>: View {
    let title: String
    let subtitle: String?
    let step: ForgeStep
    /// Optional SF Symbol shown as a 44pt hero "chip" above the title.
    /// Gives sparse stages (e.g. Source — just a text field) visual
    /// weight so the card feels intentional instead of empty. Mirrors
    /// the SettingsTab section-header chip and the EmptyStateView
    /// hero glyph rhythm.
    let symbol: String?
    let symbolTint: Color
    let onBack: (() -> Void)?
    @ViewBuilder let content: () -> Content
    @ViewBuilder let footer: () -> Footer

    init(
        title: String,
        subtitle: String? = nil,
        step: ForgeStep,
        symbol: String? = nil,
        symbolTint: Color = Brand.typeBody,
        onBack: (() -> Void)? = nil,
        @ViewBuilder content: @escaping () -> Content,
        @ViewBuilder footer: @escaping () -> Footer = { EmptyView() }
    ) {
        self.title = title
        self.subtitle = subtitle
        self.step = step
        self.symbol = symbol
        self.symbolTint = symbolTint
        self.onBack = onBack
        self.content = content
        self.footer = footer
    }

    var body: some View {
        // Layout invariants:
        //   • The progress capsule is anchored OUTSIDE the ScrollView
        //     so it never gets clipped by tight viewports.
        //   • The card lives inside a centering ScrollView. When the
        //     viewport is tall enough the spacers expand and the card
        //     is vertically centered; when it isn't, the ScrollView
        //     scrolls instead of pushing the bottom tab bar off-screen.
        //   • The card sizes to its NATURAL CONTENT (no forced
        //     minHeight). Earlier builds forced a 180-220pt minimum
        //     on every content slot, which gave simple stages like
        //     Source a giant empty rectangle below the form. The hero
        //     icon now carries the visual weight instead.
        VStack(spacing: 10) {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 0) {
                    Spacer(minLength: 18)
                    card
                    Spacer(minLength: 18)
                }
                .frame(maxWidth: .infinity)
                .containerRelativeFrame(.vertical) { length, _ in length }
            }
            .scrollBounceBehavior(.basedOnSize)
            progressCapsule
                .padding(.bottom, 14)
        }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let symbol {
                heroChip(symbol)
                    .padding(.top, 24)
                    .frame(width: ForgeStageLayout.contentWidth, alignment: .leading)
                    .padding(.horizontal, ForgeStageLayout.contentPadding)
            }
            header
                .padding(.top, symbol == nil ? 24 : 16)
                .frame(width: ForgeStageLayout.contentWidth, alignment: .leading)
                .padding(.horizontal, ForgeStageLayout.contentPadding)
            content()
                .frame(width: ForgeStageLayout.contentWidth, alignment: .leading)
                .padding(.top, 18)
                .padding(.horizontal, ForgeStageLayout.contentPadding)
                .clipped()
            footerRow
                .padding(.top, 20)
                .frame(width: ForgeStageLayout.footerWidth, alignment: .leading)
                .padding(.horizontal, ForgeStageLayout.footerPadding)
                .padding(.bottom, 18)
        }
        .frame(width: ForgeStageLayout.cardWidth)
        .clipped()
        .background(cardSurface)
    }

    /// Card background — routed through the shared `PanelChrome`
    /// primitive so every Forge stage shell reads with the same
    /// surface tone, hairline weight, and elevation as the
    /// BenchmarkOverlay panels and the LiveTab cards.
    private var cardSurface: some View {
        PanelChrome(cornerRadius: Brand.Radii.l, elevation: Brand.Elevation.hi)
    }

    @ViewBuilder
    private func heroChip(_ symbol: String) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .fill(symbolTint.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                        .stroke(symbolTint.opacity(0.25), lineWidth: 0.5)
                )
            Image(systemName: symbol)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(symbolTint)
                .symbolRenderingMode(.hierarchical)
        }
        .frame(width: 44, height: 44)
        .accessibilityHidden(true)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.mtplxLocalized)
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
            if let subtitle, !subtitle.isEmpty {
                Text(subtitle.mtplxLocalized)
                    .font(.system(size: 13))
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(2)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var footerRow: some View {
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
            footer()
        }
    }

    private var progressCapsule: some View {
        let allCases = ForgeStep.allCases
        let stepIndex = allCases.firstIndex(of: step) ?? 0
        let stepProgress = String(
            format: "Step %d of %d".mtplxLocalized,
            stepIndex + 1,
            allCases.count
        )
        return VStack(spacing: 8) {
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Brand.separator.opacity(0.5))
                    .frame(width: 240, height: 4)
                Capsule()
                    .fill(Brand.typeBody)
                    .frame(width: max(8, 240 * CGFloat(Double(stepIndex + 1) / Double(allCases.count))), height: 4)
                    .animation(.spring(response: 0.45, dampingFraction: 0.85), value: stepIndex)
            }
            .accessibilityHidden(true)
            Text("\(progressLabel(for: step).mtplxLocalized)  ·  \(stepProgress)")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .tracking(0.6)
                .foregroundStyle(Brand.typeTertiary)
                .contentTransition(.numericText())
                .accessibilityLabel(stepProgress)
        }
    }

    private func progressLabel(for step: ForgeStep) -> String {
        switch step {
        case .source: return "Source"
        case .plan: return "Plan"
        case .convert: return "Convert"
        case .calibrate: return "Calibrate"
        case .verify: return "Verify"
        case .brand: return "Brand"
        case .registered: return "Done"
        case .publishing: return "Publish"
        }
    }
}

// MARK: - ForgePrimaryButton
//
// The pill CTA shared across every stage. Mirrors
// `OnboardingPrimaryButton`'s shape (same Brand tokens, same hover
// lift + press scale + arrow nudge) so the Forge wizard reads
// visually identical to the onboarding wizard.

struct ForgePrimaryButton: View {
    let title: String
    var icon: String? = "arrow.right"
    let isEnabled: Bool
    let action: () -> Void
    @State private var isHovering: Bool = false
    @State private var isPressed: Bool = false
    @EnvironmentObject private var themeStore: ThemeStore

    init(_ title: String, icon: String? = "arrow.right", isEnabled: Bool = true, action: @escaping () -> Void) {
        self.title = title
        self.icon = icon
        self.isEnabled = isEnabled
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Text(title.mtplxLocalized)
                    .font(.system(size: 13.5, weight: .semibold))
                    .contentTransition(.opacity)
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .heavy))
                        .opacity(isEnabled ? 1 : 0.5)
                        .offset(x: isHovering && isEnabled ? 3 : 0)
                }
            }
            .foregroundStyle(isEnabled ? Brand.bgOuter : Brand.typeSecondary)
            .padding(.horizontal, 22)
            .padding(.vertical, 11)
            .background(
                Capsule(style: .continuous)
                    .fill(
                        isEnabled
                            ? AnyShapeStyle(Brand.typeBody)
                            : AnyShapeStyle(Color.white.opacity(0.06))
                    )
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(
                        isEnabled ? Brand.separator : Brand.separatorStrong,
                        lineWidth: isEnabled ? 0.5 : 0.75
                    )
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
                themeStore.reduceMotionPreference ? nil : .spring(response: 0.28, dampingFraction: 0.78),
                value: isHovering
            )
            .animation(
                themeStore.reduceMotionPreference ? nil : .spring(response: 0.18, dampingFraction: 0.72),
                value: isPressed
            )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(title.mtplxLocalized)
        .onHover { hovering in isHovering = hovering && isEnabled }
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

// MARK: - Stage files
//
// Every stage now lives in its own file under Forge/Stages/.
// Source, Plan, Convert, Calibrate, Verify, Brand, Registered, and
// Publishing are all implemented in dedicated files; ForgeCreateView
// just routes state.step → the right view.
