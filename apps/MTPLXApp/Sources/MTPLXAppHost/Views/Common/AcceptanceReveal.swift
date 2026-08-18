import SwiftUI
import MTPLXAppCore

// MARK: - AcceptanceReveal
//
// Shared rendering primitives for the "AR vs MTP acceptance + TPS"
// reveal both the onboarding TuneStep and the Forge VerifyStage
// display. Extracted here so visual consistency is enforced by the
// shared codepath rather than by two parallel inline implementations
// drifting out of sync.
//
// Data input is the small `AcceptanceRevealData` value type below;
// the two callsites construct it via per-source adapters (one from
// `TuneResult`, one from `ForgeVerification`). The components
// themselves are display-only — they don't reach back into any
// orchestrator or state machine.
//
// Onboarding's TuneStep currently keeps its existing inline copies
// of these primitives intact; a future commit can refactor it to
// consume these shared components without risk to the proven
// onboarding flow. The Forge VerifyStage uses them straight away.

// MARK: - AcceptanceRevealData

public struct AcceptanceRevealData: Equatable, Sendable {
    public struct Row: Identifiable, Equatable, Sendable {
        public let id: String
        public var label: String          // "Base", "D1", "D2", "D3"
        public var tokS: Double
        public var meanAcceptance: Double // 0...1
        public var isWinner: Bool

        public init(id: String, label: String, tokS: Double, meanAcceptance: Double, isWinner: Bool = false) {
            self.id = id
            self.label = label
            self.tokS = tokS
            self.meanAcceptance = meanAcceptance
            self.isWinner = isWinner
        }
    }

    public var headline: String
    public var arTokS: Double
    public var bestTokS: Double
    public var multiplierVsAr: Double
    public var rows: [Row]

    public init(
        headline: String,
        arTokS: Double,
        bestTokS: Double,
        multiplierVsAr: Double,
        rows: [Row]
    ) {
        self.headline = headline
        self.arTokS = arTokS
        self.bestTokS = bestTokS
        self.multiplierVsAr = multiplierVsAr
        self.rows = rows
    }

    /// Build from a ForgeVerification record. Rows are AR first, then
    /// each MTP depth in ascending order.
    public static func from(_ verification: ForgeVerification) -> AcceptanceRevealData {
        var rows: [Row] = []
        rows.append(Row(
            id: "ar",
            label: "Base",
            tokS: verification.arTokS,
            meanAcceptance: 1.0,
            isWinner: verification.bestDepth == 0
        ))
        let depths = verification.tokSByDepth.keys.sorted()
        for depth in depths {
            let tokS = verification.tokSByDepth[depth] ?? 0
            let acceptances = verification.acceptanceByDepth[depth] ?? []
            let mean = acceptances.isEmpty
                ? 0
                : acceptances.reduce(0, +) / Double(acceptances.count)
            rows.append(Row(
                id: "d\(depth)",
                label: "D\(depth)",
                tokS: tokS,
                meanAcceptance: mean,
                isWinner: depth == verification.bestDepth
            ))
        }
        let headline: String
        if verification.bestDepth == 0 {
            headline = "Baseline is fastest on this Mac."
        } else {
            headline = "Depth \(verification.bestDepth) is fastest on this Mac."
        }
        let bestTokS = verification.bestDepth == 0
            ? verification.arTokS
            : (verification.tokSByDepth[verification.bestDepth] ?? verification.tokSByDepth.values.max() ?? 0)
        return AcceptanceRevealData(
            headline: headline,
            arTokS: verification.arTokS,
            bestTokS: bestTokS,
            multiplierVsAr: verification.multiplierVsAr,
            rows: rows
        )
    }
}

// MARK: - AcceptanceRevealTPSPanel
//
// "Before → After" hero. The "after" number is the loudest
// typographic element so the user gets a visual reward for sitting
// through the verification pass; the multiplier hovers to the right
// as the headline secondary payoff. The previous "huge 40pt 1.89×
// stacked below the before/after row" felt like two reveals fighting
// for the same eye-line; this single horizontal composition reads as
// one phrase: "23.6 → 44.7 · 1.89× faster."

public struct AcceptanceRevealTPSPanel: View {
    public let data: AcceptanceRevealData

    public init(data: AcceptanceRevealData) {
        self.data = data
    }

    public var body: some View {
        let showReveal = data.arTokS > 0
            && data.bestTokS > 0
            && data.multiplierVsAr > 1.0
        Group {
            if showReveal {
                HStack(alignment: .center, spacing: 22) {
                    beforeColumn
                    arrow
                    afterColumn
                    Spacer(minLength: 8)
                    multiplierColumn
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(String(format: "%.1f", data.bestTokS))
                        .font(.system(size: 32, weight: .heavy, design: .rounded))
                        .foregroundStyle(Brand.typeHi)
                        .monospacedDigit()
                    Text("tok/s")
                        .font(.system(size: 12, weight: .semibold, design: .rounded))
                        .foregroundStyle(Brand.typeSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }

    private var beforeColumn: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("BEFORE")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeTertiary)
            Text(String(format: "%.1f", data.arTokS))
                .font(.system(size: 22, weight: .medium, design: .rounded))
                .foregroundStyle(Brand.typeTertiary)
                .monospacedDigit()
        }
    }

    private var arrow: some View {
        Image(systemName: "arrow.right")
            .font(.system(size: 13, weight: .heavy))
            .foregroundStyle(Brand.typeTertiary.opacity(0.7))
            .padding(.top, 10)
    }

    private var afterColumn: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("NOW")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeBody)
            Text(String(format: "%.1f", data.bestTokS))
                .font(.system(size: 32, weight: .heavy, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .monospacedDigit()
                .contentTransition(.numericText())
        }
    }

    private var multiplierColumn: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text(String(format: "%.2f×", data.multiplierVsAr))
                .font(.system(size: 28, weight: .heavy, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .contentTransition(.numericText())
            Text("FASTER")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeSecondary)
        }
    }

    private var accessibilityLabel: String {
        if data.arTokS > 0 && data.bestTokS > 0 {
            return String(
                format: "From %.1f tokens per second to %.1f tokens per second, %.2f times faster",
                data.arTokS, data.bestTokS, data.multiplierVsAr
            )
        }
        return String(format: "%.1f tokens per second", data.bestTokS)
    }
}

// MARK: - AcceptanceBarGrid
//
// Per-depth horizontal bars showing mean acceptance, with tok/s
// printed on the right. Same colour ladder as TuneStep
// (green ≥ 0.9, blue ≥ 0.7, amber ≥ 0.5, red below). Winner row
// gets a subtle accent ring so the eye finds it without a label.
//
// The AR row is rendered specially: it has no real "acceptance" to
// report (AR generates one token at a time — there are no MTP
// predictions to accept or reject), so we show its tok/s as a
// monospace baseline value and skip the meaningless full-width bar
// that earlier versions painted as 100% green. That bar was the
// reason the reveal card seemed to grow a stray thick horizontal
// line.

public struct AcceptanceBarGrid: View {
    public let data: AcceptanceRevealData

    public init(data: AcceptanceRevealData) {
        self.data = data
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("MTP ACCEPTANCE")
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeTertiary)
            VStack(spacing: 7) {
                ForEach(data.rows) { row in
                    if row.id == "ar" {
                        ARBaselineRow(row: row)
                    } else {
                        AcceptanceBarRow(row: row)
                    }
                }
            }
        }
    }
}

private struct ARBaselineRow: View {
    let row: AcceptanceRevealData.Row

    var body: some View {
        HStack(spacing: 10) {
            Text(row.label.mtplxLocalized)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .frame(width: 24, alignment: .leading)
            Text("baseline — no MTP predictions")
                .font(.system(size: 11, weight: .regular))
                .foregroundStyle(Brand.typeTertiary)
            Spacer(minLength: 4)
            Text(String(format: "%.1f tok/s", row.tokS))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .monospacedDigit()
        }
    }
}

private struct AcceptanceBarRow: View {
    let row: AcceptanceRevealData.Row

    var body: some View {
        HStack(spacing: 10) {
            Text(row.label.mtplxLocalized)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .foregroundStyle(row.isWinner ? Brand.typeHi : Brand.typeSecondary)
                .frame(width: 24, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Brand.separator.opacity(0.35))
                    Capsule()
                        .fill(acceptanceColor(row.meanAcceptance))
                        .frame(width: max(4, geo.size.width * max(0, min(1, row.meanAcceptance))))
                }
            }
            .frame(height: 6)
            Text(String(format: "%.0f%%", row.meanAcceptance * 100))
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .monospacedDigit()
                .frame(width: 36, alignment: .trailing)
            Text(String(format: "%.1f tok/s", row.tokS))
                .font(.system(size: 11, weight: row.isWinner ? .semibold : .medium, design: .monospaced))
                .foregroundStyle(row.isWinner ? Brand.typeHi : Brand.typeSecondary)
                .monospacedDigit()
                .frame(width: 70, alignment: .trailing)
        }
    }
}

func acceptanceColor(_ rate: Double) -> Color {
    if rate >= 0.9 { return Brand.success }
    if rate >= 0.7 { return Brand.accentChrome }
    if rate >= 0.5 { return Brand.warning }
    return Brand.danger
}
