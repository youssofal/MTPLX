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
        let isRunning = backend.daemonState.kind == .running
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
                    if let calls = latest?.verifyCalls, let total = latest?.verifyTimeS, total > 0, calls > 0 {
                        Text(tr("%@ / call", Format.milliseconds(total / Double(calls))))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.bottom, expanded ? 12 : 0)

            if expanded {
                content(latest: latest)
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
    private func content(latest: MetricsLatest?) -> some View {
        let segments = verifySegments(latest: latest)
        let total = latest?.verifyTimeS ?? 0
        if total > 0 && !segments.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                StackedBar(segments: segments, total: total)
                HStack {
                    Label(tr("Total verify_time_s: %@", Format.duration(total)),
                          systemImage: "stopwatch")
                        .font(.caption)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    Spacer()
                }
            }
        } else {
            Text(tr("Detailed timing will show here once a model is running."))
                .font(.callout)
                .foregroundStyle(Brand.textHighlight.opacity(0.6))
        }
    }

    /// Verify-waterfall tonal palette. Distinguishes 12 different
    /// pipeline phases by warmth and opacity rather than by hue —
    /// previously this strip rendered as a literal blue/indigo/purple/
    /// pink rainbow ("AI app debug palette") despite the rest of the
    /// dashboard being polished chrome. Each phase now maps to a tone
    /// on the chrome scale: chromeAccent → accentChrome → typeBody →
    /// typeSecondary → typeTertiary etc., with semantic colors
    /// reserved for the binary outcomes (accept = success, repair /
    /// rollback = danger) and warnings.
    private func verifySegments(latest: MetricsLatest?) -> [StackedBarSegment] {
        let candidates: [(label: String, value: Double?, tint: Color)] = [
            ("verify_forward", latest?.verifyForwardTimeS, Brand.coolChrome),
            ("target_dist", latest?.verifyTargetDistTimeS ?? latest?.verifyLogitsEvalTimeS, Brand.typeTertiary),
            ("verify_hidden", latest?.verifyHiddenEvalTimeS, Brand.typeSecondary),
            ("verify_other", latest?.verifyEvalUnattributedS, Brand.warning),
        ]
        return candidates.compactMap { entry in
            guard let v = entry.value, v > 0 else { return nil }
            return StackedBarSegment(
                label: entry.label,
                value: v,
                tint: entry.tint,
                valueLabel: Format.milliseconds(v)
            )
        }
    }
}
