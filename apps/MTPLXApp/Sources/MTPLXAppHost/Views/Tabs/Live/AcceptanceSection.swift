import SwiftUI
import MTPLXAppCore

// MARK: - AcceptanceSection
//
// Per-depth acceptance bars (D1/D2/D3…) — folded out of the old V0
// SpeculativeTab and onto the LiveTab so the metric MTPLX competes on
// (acceptance probability) shares a screen with the rate the user feels
// (TPS). Reads smoothed acceptance values from the store so the bars
// glide rather than strobe between integers per progress frame.

struct AcceptanceSection: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    /// Last non-empty acceptance rows for the current run. The live
    /// `accepted_by_depth` payload occasionally arrives empty for a
    /// frame (a snapshot poll that omits it, or the gap between one
    /// request finishing and the next starting). Holding the last good
    /// rows and only animating their widths keeps the bars steady
    /// instead of collapsing to the empty state and snapping back — the
    /// "appear / disappear / appear" strobe. Cleared when the daemon
    /// stops so a fresh session starts clean.
    @State private var heldRows: [AcceptanceRow] = []

    /// Tail of the row-lift stagger. The five `TileRow` tiles consume
    /// indices 0…4, so the Acceptance panel picks up at index 5 and
    /// the chain continues through `DecodeChart` and the verify
    /// waterfall — keeping the powering-on read on a single visual
    /// beat rather than three independent reveals.
    static let liftStaggerIndex: Int = 5

    var body: some View {
        let smoothed = backend.smoothedMetrics
        let liveRows = acceptanceRows(latest: backend.latest, smoothed: smoothed)
        // Render the live rows when present, otherwise the last good
        // rows we held — never an empty collapse mid-run.
        let rows = liveRows.isEmpty ? heldRows : liveRows
        let isRunning = backend.daemonState.kind == .running
        let liftAnimation: Animation? = themeStore.reduceMotionPreference
            ? nil
            : Motion.surfaceLiftSpring
        let liftDelay: TimeInterval = themeStore.reduceMotionPreference
            ? 0
            : Double(Self.liftStaggerIndex) * Motion.surfaceLiftStaggerStep

        VStack(alignment: .leading, spacing: 12) {
            header(smoothed: smoothed)

            if backend.observedCompletionCount > 0, !rows.isEmpty {
                VStack(spacing: 10) {
                    ForEach(rows) { row in
                        bar(
                            label: row.label,
                            rate: row.rate,
                            accepted: row.accepted,
                            drafted: row.drafted,
                            meanProbability: row.meanProbability
                        )
                    }
                }
            } else {
                emptyState
            }
        }
        .padding(Brand.Spacing.s4)
        .background {
            LiftedSurface(
                lifted: isRunning,
                cornerRadius: Brand.Radii.l,
                animation: liftAnimation,
                delay: liftDelay
            )
        }
        .onChange(of: liveRows) { _, newRows in
            // Hold the last non-empty rows so the bars keep showing the
            // current/previous request's acceptance and re-render in place
            // when new rows arrive — never blanking on a new send. Only a
            // daemon stop clears them (below).
            if !newRows.isEmpty { heldRows = newRows }
        }
        .onChange(of: backend.daemonState.kind) { _, kind in
            if kind == .stopped || kind == .crashed { heldRows = [] }
        }
    }

    @ViewBuilder
    private func header(smoothed: SmoothedMetrics) -> some View {
        HStack(spacing: 8) {
            Text("ACCEPTANCE")
                .font(.system(size: 11, weight: .heavy, design: .monospaced))
                .tracking(3)
                .foregroundStyle(Brand.textHighlight)
                .accessibilityHidden(true)
            Spacer()
            if backend.observedCompletionCount > 0,
               let meanAcceptance = smoothed.meanAcceptance {
                Text("mean P=\(Format.percent(meanAcceptance))")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    .contentTransition(.numericText())
            }
        }
    }

    @ViewBuilder
    private func bar(label: String, rate: Double, accepted: Int, drafted: Int, meanProbability: Double?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.system(size: 12, weight: .heavy, design: .rounded))
                    .foregroundStyle(Brand.textHighlight)
                Spacer()
                Text("\(Format.percent(rate)) (\(accepted)/\(drafted))")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .monospacedDigit()
                    .foregroundStyle(Brand.textHighlight.opacity(0.85))
                    .frame(minWidth: 96, alignment: .trailing)
                    .contentTransition(.numericText())
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Brand.separator)
                    Capsule()
                        .fill(barGradient(rate: rate))
                        .frame(width: proxy.size.width * max(0, min(1, rate)))
                        .animation(Motion.metricBar, value: rate)
                }
            }
            .frame(height: 10)
            Text(meanProbability.map { "mean P=\(Format.percent($0))" } ?? " ")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(Brand.textHighlight.opacity(0.45))
                .lineLimit(1)
                .frame(height: 11, alignment: .leading)
                .contentTransition(.numericText())
        }
        .frame(minHeight: 47, alignment: .top)
    }

    /// Acceptance bar palette — tonal chrome instead of the V0 "blue
    /// vs warm" split. The high band reads as polished chrome, the
    /// upper-mid as warm steel, then the warning amber + danger red
    /// for the failing bands. Distinguished by warmth, not hue.
    private func barGradient(rate: Double) -> LinearGradient {
        if rate >= 0.9 {
            return Brand.chromeAccent
        }
        if rate >= 0.7 {
            return LinearGradient(
                colors: [Brand.accentWarm, Brand.accentWarm.opacity(0.65)],
                startPoint: .leading,
                endPoint: .trailing
            )
        }
        let amber = LinearGradient(
            colors: [Brand.warning.opacity(0.95), Brand.warning.opacity(0.6)],
            startPoint: .leading,
            endPoint: .trailing
        )
        return rate >= 0.5 ? amber : LinearGradient(
            colors: [Brand.danger.opacity(0.85), Brand.danger.opacity(0.55)],
            startPoint: .leading,
            endPoint: .trailing
        )
    }

    @ViewBuilder
    private var emptyState: some View {
        Text(emptyStateText.mtplxLocalized)
            .font(.callout)
            .foregroundStyle(Brand.textHighlight.opacity(0.6))
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var emptyStateText: String {
        if isARMode {
            return "MTP acceptance appears when draft depth is on."
        }
        return "Counters appear after your first MTP response."
    }

    private var isARMode: Bool {
        let candidates = [
            backend.latest?.values["generation_mode"]?.stringValue,
            backend.settings?.generationMode,
            backend.health?.generationMode,
        ]
        return candidates.contains { value in
            value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "ar"
        }
    }

    private struct AcceptanceRow: Identifiable, Equatable {
        let label: String
        let accepted: Int
        let drafted: Int
        let rate: Double
        let meanProbability: Double?

        var id: String { label }
    }

    /// Build the rendered rows from raw `accepted`/`drafted` counts and
    /// the smoothed bar widths / mean probabilities. The counter
    /// numbers stay precise; the bar width and mean caption read off
    /// the EMA so the visuals stay steady.
    private func acceptanceRows(latest: MetricsLatest?, smoothed: SmoothedMetrics) -> [AcceptanceRow] {
        guard let latest else { return [] }

        let counters = latest.acceptanceCounterRows()
        guard !counters.isEmpty else { return [] }

        let smoothedRates = smoothed.acceptanceRateByDepth
        let smoothedMeans = smoothed.meanAcceptByDepth

        return counters.enumerated().map { idx, counter in
            let smoothedRate = smoothedRates.indices.contains(idx)
                ? smoothedRates[idx]
                : counter.rate
            let mean = smoothedMeans.indices.contains(idx) ? smoothedMeans[idx] : nil
            return AcceptanceRow(
                label: counter.label,
                accepted: counter.accepted,
                drafted: counter.drafted,
                rate: smoothedRate,
                meanProbability: mean
            )
        }
    }
}
