import SwiftUI
import MTPLXAppCore

// MARK: - StarPromptLayer
//
// The one-time GitHub star prompt, centred in the main window over a dim
// floor. `StarPromptPresenter` decides when it appears; this layer only
// draws it. The floor takes clicks without closing the prompt, so a stray
// click on the chat behind does not use up the one time it is shown; the
// close button and Escape close it. The floor is a flat tint, not a blur,
// so a dashboard still animating behind it costs nothing extra to draw.

struct StarPromptLayer: View {
    @EnvironmentObject private var presenter: StarPromptPresenter
    @EnvironmentObject private var themeStore: ThemeStore
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion

    let performanceLock: Bool

    private var motionEnabled: Bool {
        !performanceLock && !themeStore.reduceMotionPreference && !systemReduceMotion
    }

    var body: some View {
        ZStack {
            if presenter.isPresented {
                Rectangle()
                    .fill(Brand.bgOuter.opacity(0.62))
                    .ignoresSafeArea()
                    .contentShape(Rectangle())
                    .onTapGesture {}
                    .accessibilityHidden(true)
                    .transition(.opacity)
                StarPromptCard { outcome in
                    presenter.complete(outcome)
                }
                .padding(Brand.Spacing.s6)
                .transition(
                    motionEnabled
                        ? .opacity.combined(with: .scale(scale: 0.96, anchor: .center))
                        : .opacity
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(presenter.isPresented)
        .animation(
            motionEnabled ? .spring(response: 0.42, dampingFraction: 0.86) : nil,
            value: presenter.isPresented
        )
    }
}

// MARK: - StarPromptCard

struct StarPromptCard: View {
    let onAnswer: (StarPromptPresenter.Outcome) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            StarPromptMedallion()
                .padding(.bottom, Brand.Spacing.s5)
            Text(tr("Is everything working well?"))
                .font(.system(.title2, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeHi)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
            VStack(alignment: .leading, spacing: Brand.Spacing.s2) {
                Text(tr("If you like MTPLX, a star on GitHub would help the project a lot. Stars are how other Mac users find it."))
                Text(tr("If something is not right, tell us and we will fix it."))
            }
            .font(.system(size: 13))
            .foregroundStyle(Brand.typeSecondary)
            .lineSpacing(2)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, Brand.Spacing.s2)
            HStack(spacing: Brand.Spacing.s3) {
                Button {
                    onAnswer(.fileIssue)
                } label: {
                    Label {
                        Text(tr("File an issue"))
                    } icon: {
                        Image(systemName: "exclamationmark.bubble")
                    }
                    .labelStyle(.titleAndIcon)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(MTPLXGhostButton())
                .accessibilityLabel(tr("File an issue"))
                .accessibilityHint(tr("Opens a new issue on GitHub in your browser."))

                Button {
                    onAnswer(.starred)
                } label: {
                    HStack(spacing: 7) {
                        StarPromptButtonStar()
                        Text(tr("Star on GitHub"))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(MTPLXPillButton())
                .accessibilityLabel(tr("Star on GitHub"))
                .accessibilityHint(tr("Opens the MTPLX page on GitHub in your browser."))
            }
            .padding(.top, Brand.Spacing.s6)
        }
        .padding(28)
        .frame(maxWidth: 440, alignment: .leading)
        .background { PanelChrome(cornerRadius: Brand.Radii.panel) }
        .overlay(alignment: .topTrailing) {
            StarPromptCloseButton {
                onAnswer(.dismissed)
            }
            .padding(Brand.Spacing.s3)
        }
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(.isModal)
        .accessibilityLabel(tr("Is everything working well?"))
        .accessibilityAction(.escape) { onAnswer(.dismissed) }
    }
}

// MARK: - Illustrations

/// A chrome star on a raised tile, with two sparkles: the app's own steel
/// treatment, drawn from SF Symbols.
private struct StarPromptMedallion: View {
    var body: some View {
        let shape = RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
        ZStack {
            shape
                .fill(
                    LinearGradient(
                        colors: [Brand.panelSurfaceTop, Brand.cardSurface],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .overlay {
                    shape.strokeBorder(Brand.separatorStrong, lineWidth: Brand.hairlineStrong)
                }
                .shadow(
                    color: Brand.Elevation.mid.color,
                    radius: Brand.Elevation.mid.radius,
                    x: Brand.Elevation.mid.x,
                    y: Brand.Elevation.mid.y
                )
            Image(systemName: "star.fill")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(Brand.chromeAccent)
                .shadow(color: Brand.chromeHalo, radius: 8)
            Image(systemName: "sparkle")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(Brand.typeSecondary)
                .offset(x: 17, y: -16)
            Image(systemName: "sparkle")
                .font(.system(size: 7, weight: .bold))
                .foregroundStyle(Brand.typeTertiary)
                .offset(x: -17, y: 15)
        }
        .frame(width: 56, height: 56)
        .accessibilityHidden(true)
    }
}

/// The star on the primary button, with one small sparkle.
private struct StarPromptButtonStar: View {
    var body: some View {
        Image(systemName: "star.fill")
            .font(.system(size: 12, weight: .semibold))
            .overlay(alignment: .topTrailing) {
                Image(systemName: "sparkle")
                    .font(.system(size: 6, weight: .bold))
                    .offset(x: 5, y: -4)
            }
            .accessibilityHidden(true)
    }
}

private struct StarPromptCloseButton: View {
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: "xmark")
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
                .frame(width: 24, height: 24)
                .background {
                    Circle().fill(Brand.wash.opacity(hovering ? 0.10 : 0.05))
                }
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.15), value: hovering)
        .accessibilityLabel(tr("Close"))
        .accessibilityHint(tr("Closes this message. It will not appear again."))
    }
}
