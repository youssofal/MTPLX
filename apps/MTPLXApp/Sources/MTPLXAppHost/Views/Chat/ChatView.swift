import SwiftUI
import MTPLXAppCore

// MARK: - ChatView
//
// Root of the in-app chat primary mode. ContentView swaps the dashboard
// surface for this view when `router.primaryMode == .chat`. The chrome
// (TopChromeStrip + ConnectionIssueBanner) sits above; ChatView owns
// everything below.
//
// Layout:
//   HStack {
//     ChatSidebarView   // left rail, collapsible
//     VStack {
//       ChatHeaderView          // title + live TPS chip
//       ChatConversationView    // messages + streaming bubble
//       ChatComposerView        // composer pill at bottom
//     }
//   }

private struct MTPLXPerformanceLockKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    /// Chat render surfaces read this instead of observing the whole
    /// backend store: an @EnvironmentObject subscription re-evaluated
    /// every transcript bubble on every 10 Hz metrics tick for one
    /// static Bool (2026-08-17 field regression). An environment value
    /// re-evaluates readers only when it actually flips.
    var mtplxPerformanceLock: Bool {
        get { self[MTPLXPerformanceLockKey.self] }
        set { self[MTPLXPerformanceLockKey.self] = newValue }
    }
}

struct ChatView: View {
    @EnvironmentObject private var chatViewModel: ChatViewModel
    @EnvironmentObject private var router: AppRouter

    let daemonState: DaemonState
    let startupPhase: DaemonStartupPhase
    let selectedModel: String
    let visionEnabled: Bool
    let performanceLock: Bool

    var body: some View {
        HStack(spacing: 0) {
            ChatSidebarView(
                viewModel: chatViewModel,
                collapsed: $router.chatSidebarCollapsed
            )
            VStack(spacing: 0) {
                ChatHeaderView(
                    viewModel: chatViewModel,
                    sidebarCollapsed: $router.chatSidebarCollapsed
                )
                ChatConversationView(
                    viewModel: chatViewModel,
                    daemonState: daemonState,
                    startupPhase: startupPhase,
                    selectedModel: selectedModel
                )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                ChatComposerView(
                    viewModel: chatViewModel,
                    daemonState: daemonState,
                    selectedModel: selectedModel,
                    visionEnabled: visionEnabled
                )
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
                    .padding(.top, 8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Brand.bgOuter)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .environment(\.mtplxPerformanceLock, performanceLock)
        .overlay(alignment: .bottomTrailing) {
            if chatViewModel.uiPerfProbe.showsHUD {
                UIPerfHUDView(probe: chatViewModel.uiPerfProbe)
                    .padding(.trailing, 14)
                    .padding(.bottom, 96)
                    .allowsHitTesting(false)
            }
        }
        .onAppear {
            // Ensure there is a conversation to send into, so the user
            // can type immediately without having to click "+".
            if chatViewModel.current == nil {
                _ = chatViewModel.createNewConversation()
            }
        }
        .onChange(of: performanceLock, initial: true) { _, locked in
            // Mirror for render leaves that can't take the flag as a
            // parameter (theme closures, NSView viewports).
            ChatRenderPreferences.plainTextOnly = locked
        }
    }
}

// MARK: - UIPerfHUDView
//
// Live frontend-perf chip (MTPLX_UI_PERF_HUD=1): main-thread stalls,
// flush cadence, last document-apply cost, and realized block count —
// the numbers behind "does streaming feel smooth", on screen while it
// streams. Read-only; sits above the composer, ignores clicks.
private struct UIPerfHUDView: View {
    @ObservedObject var probe: UIStreamPerfProbe

    var body: some View {
        let hud = probe.hud
        VStack(alignment: .trailing, spacing: 2) {
            Text(hud.isStreaming ? "STREAMING" : "IDLE")
                .font(.system(size: 8, weight: .heavy, design: .monospaced))
                .foregroundStyle(hud.isStreaming ? Brand.success : Brand.typeTertiary)
            Text(String(format: "flush %4.1f/s  apply %5.1f ms", hud.flushesPerSecond, hud.lastAppendMs))
            Text(String(format: "stalls %d/s  worst %4.0f ms", hud.stallsLastSecond, hud.worstStallMsLastSecond))
            Text(String(format: "blocks %d  chars %d", hud.documentBlocks, hud.turnChars))
        }
        .font(.system(size: 9, weight: .medium, design: .monospaced))
        .foregroundStyle(Brand.typeSecondary)
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Color.black.opacity(0.72))
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }
}
