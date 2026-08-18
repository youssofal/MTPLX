import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - MenuBarContent
//
// Compact popover content for the MenuBarExtra. ~280pt wide. Brand mini
// wordmark + connection dot at the top, mini gauge in the middle,
// contextual start/stop/restart, profile picker, fan toggle, perf lock,
// and an action list (Open Dashboard, Open Browser, About, Quit).
//
// The popover uses `.menuBarExtraStyle(.window)` so we have full design
// control — piano-black background, chrome accents, hairline separators.

struct MenuBarContent: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            divider
            gaugeSection
            divider
            controlsSection
            divider
            actionsSection
        }
        .frame(width: 300)
        .background(Brand.pianoRadial)
        .preferredColorScheme(.dark)
        .tint(Brand.accent)
    }

    // MARK: - Sections

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 10) {
            WordmarkView(height: 16, fallbackTracking: true)
            Spacer()
            ConnectionDot(
                daemonState: backend.daemonState,
                connectionState: backend.connectionState
            )
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var gaugeSection: some View {
        VStack(spacing: 6) {
            MenuBarMiniGauge()
                .padding(.vertical, 6)
            // Same warm-up gate as the hero caption: don't surface an
            // all-time max until a real request has actually completed,
            // so the menubar never opens on a misleading warm-up record.
            if backend.observedCompletionCount > 0,
               let max = backend.rolling?.stickyAllTimeMax, max > 0 {
                Text("ALL-TIME MAX \(Format.tps(max)) TPS")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private var controlsSection: some View {
        VStack(spacing: 8) {
            primaryActionButton
            HStack {
                Text("FAN")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.textHighlight.opacity(0.6))
                Spacer()
                FanModeToggle()
            }
            HStack {
                Text("PERF LOCK")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.textHighlight.opacity(0.6))
                Spacer()
                PerformanceLockToggle()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var primaryActionButton: some View {
        let kind = backend.daemonState.kind
        let (label, symbol, action): (String, String, () -> Void) = {
            switch kind {
            case .stopped, .crashed, .degraded:
                return ("Start MTPLX", "play.fill", {
                    Task { await backend.startDaemon() }
                })
            case .starting, .warming:
                return ("Stop (loading…)", "stop.fill", {
                    Task {
                        await stopCoordinator.stopAll(reason: "menu_bar_stop_loading")
                    }
                })
            case .running:
                return ("Stop MTPLX", "stop.fill", {
                    Task {
                        await stopCoordinator.stopAll(reason: "menu_bar_stop_running")
                    }
                })
            case .stopping:
                return ("Stopping…", "hourglass", {})
            }
        }()

        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: symbol)
                    .font(.system(size: 11, weight: .semibold))
                Text(label.mtplxLocalized)
                    .font(.system(size: 12, weight: .heavy, design: .monospaced))
                    .tracking(1)
                Spacer()
            }
            .foregroundStyle(Brand.bgOuter)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Brand.chromeFill)
                    .shadow(color: Brand.Depth.ambient.color, radius: 6, x: 0, y: 4)
            )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var actionsSection: some View {
        VStack(spacing: 0) {
            actionRow("Open MTPLX", symbol: "macwindow") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "main")
            }
            actionRow("Open Logs", symbol: "doc.text") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "main")
                router.presentLogs()
            }
            actionRow("About MTPLX…", symbol: "info.circle") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "main")
                router.presentAbout()
            }
            Divider().overlay(Brand.separator).padding(.vertical, 4)
            actionRow("Quit MTPLX", symbol: "power") {
                NSApp.terminate(nil)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private func actionRow(_ label: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: symbol)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Brand.textHighlight)
                    .frame(width: 18)
                Text(label.mtplxLocalized)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.textHighlight)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.clear)
            )
        }
        .buttonStyle(MenuBarRowButtonStyle())
    }

    @ViewBuilder
    private var divider: some View {
        Rectangle()
            .fill(Brand.separator)
            .frame(height: 0.5)
    }
}

// MARK: - MenuBarRowButtonStyle

private struct MenuBarRowButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(configuration.isPressed
                          ? Brand.accent.opacity(0.18)
                          : Color.clear)
            )
    }
}
