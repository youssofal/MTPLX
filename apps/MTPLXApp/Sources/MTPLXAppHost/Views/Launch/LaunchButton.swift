import SwiftUI
import MTPLXAppCore

// MARK: - LaunchButton
//
// Single state-driven button in the top chrome strip. Replaces the V0
// separate Play / Stop pair. The icon morphs between states via
// SwiftUI 6's `contentTransition(.symbolEffect(.replace))`:
//
//   stopped / crashed / degraded  →  play.fill        click → picker
//   starting                       →  circle.dotted   click → stop
//   warming                        →  hourglass       click → stop
//   running                        →  stop.fill       click → stop
//   stopping                       →  spinner         click → no-op
//
// "Premium puck" treatment (Jet Chrome pass): the polished chrome
// chassis is a `ButtonStyle` (`PremiumPuckStyle`) so press / hover
// state rides on `ButtonStyle.Configuration.isPressed` instead of a
// `simultaneousGesture` that would race with the Button's own tap
// recognition — the previous View-wrapper approach silently ate the
// click on macOS. The chrome look (dual-ring, radial inner fill,
// chromeAccent hover halo, press collapse) is identical.

struct LaunchButton: View {
    let backend: MTPLXBackendStore
    let daemonState: DaemonState

    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    var body: some View {
        Button(action: handlePress) {
            Group {
                if case .stopping = daemonState {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.7)
                        .tint(Brand.typeHi)
                } else {
                    Image(systemName: glyphName)
                        .font(.system(size: 13, weight: .semibold))
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(Brand.typeHi)
                        .contentTransition(.symbolEffect(.replace))
                }
            }
        }
        .buttonStyle(PremiumPuckStyle())
        .disabled(disabled)
        .help(helpText)
        .accessibilityLabel(accessibilityLabel)
        .animation(.smooth(duration: 0.25), value: stateKey)
    }

    private var stateKey: String { daemonState.kind.rawValue }

    private var glyphName: String {
        switch daemonState.kind {
        case .stopped, .crashed, .degraded: return "play.fill"
        case .starting: return "circle.dotted"
        case .warming: return "hourglass"
        case .running: return "stop.fill"
        case .stopping: return "stop.fill"
        }
    }

    private var disabled: Bool {
        if stopCoordinator.isStoppingEverything { return true }
        if case .stopping = daemonState { return true }
        return false
    }

    private var helpText: String {
        switch daemonState.kind {
        case .stopped, .crashed, .degraded:
            return "Pick how you want to use it, then start"
        case .starting: return "Starting — click to cancel"
        case .warming: return "Loading the model — click to cancel"
        case .running: return "Stop"
        case .stopping: return "Stopping…"
        }
    }

    private var accessibilityLabel: String {
        switch daemonState.kind {
        case .stopped, .crashed, .degraded: return "Start MTPLX"
        case .starting: return "Starting — tap to stop"
        case .warming: return "Loading — tap to stop"
        case .running: return "Stop MTPLX"
        case .stopping: return "Stopping"
        }
    }

    private func handlePress() {
        switch daemonState.kind {
        case .stopped, .crashed, .degraded:
            router.launchPickerPresented.toggle()
        case .starting, .warming, .running:
            Task {
                await stopCoordinator.stopAll(reason: "launch_button_stop")
            }
        case .stopping:
            break
        }
    }
}

// MARK: - PremiumPuckStyle
//
// Canonical chrome puck ButtonStyle shared by `LaunchButton` and
// `InferenceParamsButton` so the two top-strip controls read as a
// matched pair of circular pucks. The press / hover state rides on
// `ButtonStyle.Configuration.isPressed` + an internal @State for
// hover — no gesture recognizers in the chrome layer, so the Button's
// tap recognition isn't fighting anything.

struct PremiumPuckStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        PremiumPuckBody(configuration: configuration)
    }
}

private struct PremiumPuckBody: View {
    let configuration: ButtonStyle.Configuration

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false

    var body: some View {
        ZStack {
            Circle()
                .fill(
                    RadialGradient(
                        colors: [Brand.raisedSurface, Brand.bgInner],
                        center: .center,
                        startRadius: 1,
                        endRadius: Brand.controlSize * 0.55
                    )
                )
                .overlay {
                    Circle()
                        .strokeBorder(
                            Color.white.opacity(0.18),
                            lineWidth: Brand.hairlineStrong
                        )
                }
                .overlay {
                    Circle()
                        .trim(from: 0.0, to: 0.5)
                        .stroke(
                            Color.white.opacity(0.06),
                            style: StrokeStyle(lineWidth: 1, lineCap: .round)
                        )
                        .rotationEffect(.degrees(180))
                }

            configuration.label
        }
        .frame(width: Brand.controlSize, height: Brand.controlSize)
        .scaleEffect(scale)
        .shadow(
            color: Brand.accentChrome.opacity(haloOpacity),
            radius: 16
        )
        .animation(
            reduceMotion ? nil : .spring(response: 0.30, dampingFraction: 0.74),
            value: hovering
        )
        .animation(
            reduceMotion ? nil : .spring(response: 0.18, dampingFraction: 0.62),
            value: configuration.isPressed
        )
        .contentShape(Circle())
        .onHover { hovering = $0 }
    }

    private var scale: CGFloat {
        if configuration.isPressed { return 0.94 }
        if hovering { return 1.06 }
        return 1.0
    }

    private var haloOpacity: Double {
        guard hovering && !configuration.isPressed else { return 0 }
        return 0.10
    }
}
