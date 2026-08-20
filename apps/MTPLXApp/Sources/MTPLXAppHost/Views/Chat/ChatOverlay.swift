import SwiftUI
import MTPLXAppCore

// MARK: - ChatOverlay
//
// Wraps `ChatView` so it presents as a real overlay over the
// dashboard. The overlay slides UP from the bottom edge of the
// dashboard area on open and collapses DOWN on close (the
// transition is wired by the call-site `.move(edge: .bottom)`
// on the chat slot). Carries a `ChatCloseButton` pinned to the
// top-left — the Mac convention for "close this surface"
// (mirrors the red traffic-light position on a window). A
// peer `ChatExpandTab` (in `ContentView`'s ZStack) renders at
// the bottom-centre of the dashboard area when chat is closed
// so the user can pull the drawer back up.

struct ChatOverlay: View, Equatable {
    let daemonState: DaemonState
    let startupPhase: DaemonStartupPhase
    let selectedModel: String
    let visionEnabled: Bool
    let performanceLock: Bool
    let onCollapse: () -> Void

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.daemonState == rhs.daemonState
            && lhs.startupPhase == rhs.startupPhase
            && lhs.selectedModel == rhs.selectedModel
            && lhs.visionEnabled == rhs.visionEnabled
            && lhs.performanceLock == rhs.performanceLock
    }

    var body: some View {
        VStack(spacing: 0) {
            closeBar
            ChatView(
                daemonState: daemonState,
                startupPhase: startupPhase,
                selectedModel: selectedModel,
                visionEnabled: visionEnabled,
                performanceLock: performanceLock
            )
                .background(Brand.bgOuter)
        }
    }

    /// Thin top bar that anchors `ChatCloseButton` at the leftmost
    /// position so the close affordance lives in the top-left of
    /// the chat overlay regardless of whether the sidebar is open
    /// or collapsed. Single hairline separator below the bar so the
    /// chat content reads as a layered surface.
    private var closeBar: some View {
        HStack(spacing: 8) {
            ChatCloseButton(action: onCollapse)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(height: Brand.hairline),
                    alignment: .bottom
                )
        )
    }
}

// MARK: - ChatCloseButton
//
// Chrome pill at the top-left of the chat overlay. Single click
// (or Esc) collapses the chat surface back down to the dashboard.
// `chevron.down` glyph so the icon matches the collapse direction
// (chat is about to slide down and out of view).

struct ChatCloseButton: View {
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .heavy))
                Text("Close chat")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(0.3)
            }
            .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background {
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(0.04))
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(
                                hovering ? Brand.separatorStrong : Brand.separator,
                                lineWidth: Brand.hairlineStrong
                            )
                    }
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .keyboardShortcut(.escape, modifiers: [])
        .help("Close chat (Esc)")
        .accessibilityLabel("Close chat")
        .onHover { hovering = $0 }
        .animation(
            reduceMotion ? nil : .easeOut(duration: 0.16),
            value: hovering
        )
    }
}

// MARK: - ChatExpandTab
//
// Counter-affordance to `ChatCloseButton`. A small chrome pill
// pinned to the bottom-centre of the dashboard area, visible only
// when chat is closed. Single click pulls the chat drawer up from
// below. Panel-surface gradient + hairlineStrong stroke + soft
// elevation so it reads as a chrome handle attached to the
// surrounding chrome system, not as a separate cheap button.

struct SurfaceExpandTab: View {
    let surface: AppExpandableSurface
    let action: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.up")
                    .font(.system(size: 10, weight: .heavy))
                Text("Expand \(surface.title)")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(0.3)
            }
            .foregroundStyle(hovering ? Brand.typeBody : Brand.typeSecondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background {
                Capsule(style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [Brand.panelSurfaceTop, Brand.panelSurfaceBottom],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .overlay {
                        Capsule(style: .continuous)
                            .strokeBorder(
                                hovering ? Brand.separatorStrong : Brand.separator,
                                lineWidth: Brand.hairlineStrong
                            )
                    }
                    .shadow(
                        color: Brand.Elevation.mid.color,
                        radius: Brand.Elevation.mid.radius,
                        x: 0,
                        y: Brand.Elevation.mid.y
                    )
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Expand \(surface.title)")
        .accessibilityLabel("Expand \(surface.title)")
        .onHover { hovering = $0 }
        .animation(
            reduceMotion ? nil : .easeOut(duration: 0.16),
            value: hovering
        )
    }
}
