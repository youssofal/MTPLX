import SwiftUI
import MTPLXAppCore

// MARK: - VerifyWaterfallExpander
//
// Folded over from the old V0 SpeculativeTab. A collapsed disclosure
// under the acceptance bars; tap to expand the stacked-bar breakdown of
// per-verify-call wall time. Stays out of the way until the user wants
// it — keeps the LiveTab visually quiet.

struct VerifyWaterfallExpander: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore
    @State private var expanded: Bool = false

    /// Last link in the LiveTab lift chain (TileRow 0…4,
    /// AcceptanceSection 5, DecodeChart 6, waterfall here at 7). The
    /// waterfall sits at the very bottom of the dashboard so this
    /// closes out the powering-on wave as it travels down the page.
    static let liftStaggerIndex: Int = 7

    var body: some View {
        let latest = backend.hasObservedCurrentRunMetrics ? backend.latest : nil
        let summary = VerifyWaterfallSummary(latest: latest)
        let daemonKind = backend.daemonState.kind
        let isRunning = daemonKind == .running
        let liftAnimation: Animation? = themeStore.reduceMotionPreference
            ? nil
            : Motion.surfaceLiftSpring
        let liftDelay: TimeInterval = themeStore.reduceMotionPreference
            ? 0
            : Double(Self.liftStaggerIndex) * Motion.surfaceLiftStaggerStep

        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.smooth(duration: 0.25)) { expanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Brand.textHighlight.opacity(0.6))
                    Text(tr("VERIFY WATERFALL"))
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .tracking(3)
                        .foregroundStyle(Brand.textHighlight)
                    Spacer()
                    if let perCall = summary.msPerCallSeconds {
                        Text(tr("%@ / call", Format.milliseconds(perCall)))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.bottom, expanded ? 12 : 0)

            if expanded {
                content(summary: summary, daemonKind: daemonKind)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .background {
            LiftedSurface(
                lifted: isRunning,
                cornerRadius: Brand.Radii.m,
                animation: liftAnimation,
                delay: liftDelay
            )
        }
    }

    @ViewBuilder
    private func content(summary: VerifyWaterfallSummary, daemonKind: DaemonStateKind) -> some View {
        let segments = verifySegments(summary: summary)
        if summary.totalSeconds > 0 && !segments.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                StackedBar(segments: segments, total: summary.totalSeconds)
                if let footer = summary.footerSummary {
                    HStack {
                        Label(footer, systemImage: "stopwatch")
                            .font(.caption)
                            .foregroundStyle(Brand.textHighlight.opacity(0.65))
                        Spacer()
                    }
                }
            }
        } else {
            Text(VerifyWaterfallSummary.emptyStateMessage(daemonState: daemonKind))
                .font(.callout)
                .foregroundStyle(Brand.textHighlight.opacity(0.6))
        }
    }

    /// Verify-waterfall tonal palette. Maps draft / verify_forward /
    /// target_dist / verify_hidden / verify_overhead onto the chrome scale
    /// with semantic warmth reserved for draft vs unattributed overhead.
    private func verifySegments(summary: VerifyWaterfallSummary) -> [StackedBarSegment] {
        summary.segments.map { segment in
            StackedBarSegment(
                label: segment.label,
                value: segment.seconds,
                tint: segmentTint(for: segment.key),
                valueLabel: Format.milliseconds(segment.seconds)
            )
        }
    }

    private func segmentTint(for key: String) -> Color {
        switch key {
        case "draft":
            return Brand.accentWarm
        case "verify_forward", "verify":
            return Brand.coolChrome
        case "target_dist":
            return Brand.typeTertiary
        case "verify_hidden":
            return Brand.typeSecondary
        case "verify_overhead", "verify_other":
            return Brand.warning
        default:
            return Brand.accentChrome
        }
    }
}
