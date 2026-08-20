import SwiftUI
import MTPLXAppCore

// MARK: - TopChromeStrip
//
// Replaces the macOS toolbar with a custom 48pt strip whose contents
// are vertically centred. Reading left → right:
//
//   [ WordmarkView(20) ]  [ ● Running ]  [ model name ]
//                                                       Spacer
//   [ Refresh ]  [ InferenceParams ]  [ LaunchButton ]
//
// The previous design had two competing status badges (a "RUNNING"
// text pill and a separate green "live" dot). The pulse + label are
// now a single indicator that goes green only when the daemon is
// fully up AND the SSE stream is open, sitting close to the wordmark
// so the user reads "MTPLX • Running" as one phrase.

struct TopChromeStrip: View {
    let backend: MTPLXBackendStore
    let daemonState: DaemonState
    let connectionState: MetricsConnectionState
    let activeModelLabel: String
    let configuration: MTPLXAppConfiguration

    @EnvironmentObject private var router: AppRouter

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            // Left cluster reads as three distinct beats:
            //   [MTPLX]  ·····  [● Stopped]  ·  [MODEL-NAME ▾]
            //
            // The wordmark gets 18pt of breathing room before the
            // status group; inside the status group the daemon
            // status and the model selector are separated by a
            // middle-dot in tertiary tint with 10pt on each side.
            // Without that dot they appeared as one continuous
            // monospace phrase because both labels share the same
            // size/weight/tracking — the eye had nothing to anchor
            // on between "Stopped" and the model id.
            HStack(alignment: .center, spacing: Brand.Spacing.s4) {
                WordmarkView(height: 24)
                HStack(alignment: .center, spacing: 10) {
                    ConnectionDot(
                        daemonState: daemonState,
                        connectionState: connectionState
                    )
                    Text("\u{00B7}")
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                        .accessibilityHidden(true)
                    Button {
                        router.modelPickerPresented.toggle()
                    } label: {
                        HStack(spacing: 5) {
                            Text(modelShort(activeModelLabel))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .tracking(1)
                                .lineLimit(1)
                                // .tail, not .middle: on macOS 26, middle
                                // truncation on a tracked single-line Text in
                                // a flexible frame can spin layout at 100%
                                // CPU (streamwar A7; Settings-click suspect).
                                .truncationMode(.tail)
                            Image(systemName: "chevron.down")
                                .font(.system(size: 8, weight: .bold))
                        }
                        .foregroundStyle(Brand.typeSecondary)
                        .frame(maxWidth: 280, alignment: .leading)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Change model")
                    .opacity(router.modelPickerPresented ? 1 : 0.82)
                    .animation(.smooth(duration: 0.2), value: router.modelPickerPresented)
                }
            }

            Spacer(minLength: 12)

            HStack(alignment: .center, spacing: 8) {
                RefreshButton {
                    Task {
                        try? await backend.refreshStaticState()
                        try? await backend.refreshSnapshot()
                    }
                }
                InferenceParamsButton(
                    performanceLock: configuration.performanceLock
                )
                LaunchButton(
                    backend: backend,
                    daemonState: daemonState
                )
            }
        }
        // Top inset is intentionally larger than the bottom inset
        // because `windowStyle(.hiddenTitleBar)` doesn't reserve any
        // room for the traffic lights — they overlay directly on top
        // of the content. Without the extra top padding, the wordmark
        // sits at the same vertical line as the close/min/max dots and
        // its uppercase ascenders look clipped against the window edge.
        .padding(.horizontal, 14)
        .padding(.top, 12)
        .padding(.bottom, 8)
        .frame(minHeight: 52)
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

    /// Show catalog-backed names where possible so the top strip stays
    /// readable even when the backend reports a full path or HF id.
    private func modelShort(_ raw: String) -> String {
        let stripped = MTPLXModelOption.displayName(
            for: raw,
            customModels: configuration.customModels
        )
        return stripped.uppercased()
    }
}

// MARK: - RefreshButton (32x32 circle, monochrome white)

struct RefreshButton: View {
    let action: () -> Void

    @State private var spinning: Bool = false

    var body: some View {
        Button {
            spinning = true
            action()
            Task { @MainActor in
                try? await Task.sleep(for: .milliseconds(600))
                spinning = false
            }
        } label: {
            Image(systemName: "arrow.clockwise")
                .font(.system(size: 13, weight: .semibold))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(Brand.typeBody)
                .rotationEffect(.degrees(spinning ? 360 : 0))
                .animation(spinning ? .easeInOut(duration: 0.55) : nil, value: spinning)
                .frame(width: Brand.controlSize, height: Brand.controlSize)
                .background(
                    Circle()
                        .stroke(Brand.separatorStrong, lineWidth: 1.0)
                        .background(Circle().fill(Color.white.opacity(0.04)))
                )
        }
        .buttonStyle(.plain)
        .help("Refresh snapshot + capabilities")
    }
}
