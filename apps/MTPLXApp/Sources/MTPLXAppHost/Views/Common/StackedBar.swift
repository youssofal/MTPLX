import SwiftUI
import MTPLXAppCore

// MARK: - StackedBarSegment
//
// One segment of a `StackedBar`. The `id` is generated so callers
// can compose segments from heterogeneous sources without inventing
// their own identifier scheme.

struct StackedBarSegment: Identifiable {
    let id = UUID()
    let label: String
    let value: Double
    let tint: Color
    var valueLabel: String? = nil
}

// MARK: - StackedBar
//
// Horizontal stacked bar (memory, verify-time waterfall, etc.).
// Segments render proportionally to their values; `total` anchors
// the scale (otherwise it sums the values). Inner GeometryReader is
// load-bearing for the per-segment width math, same as `HBarRow`.

struct StackedBar: View {
    let segments: [StackedBarSegment]
    var total: Double? = nil
    var height: CGFloat = 18
    /// Legend-only entries drawn after the segments' own (a peak, say):
    /// they label the bar without taking a share of it.
    var legendExtras: [StackedBarSegment] = []

    private var resolvedTotal: Double {
        if let total, total > 0 { return total }
        let sum = segments.map(\.value).reduce(0, +)
        return sum > 0 ? sum : 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            GeometryReader { proxy in
                HStack(spacing: 1) {
                    ForEach(segments) { segment in
                        Rectangle()
                            .fill(segment.tint.gradient)
                            .frame(width: max(0, proxy.size.width * (segment.value / resolvedTotal)))
                    }
                    Spacer(minLength: 0)
                }
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .background {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Brand.separator)
                }
            }
            .frame(height: height)

            FlowLayout(spacing: 12) {
                ForEach(segments + legendExtras) { segment in
                    HStack(spacing: 6) {
                        Circle()
                            .fill(segment.tint)
                            .frame(width: 8, height: 8)
                        Text(segment.label)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Brand.typeBody.opacity(0.7))
                        Text(segment.valueLabel ?? Format.percent(segment.value / resolvedTotal))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(Brand.accentChrome)
                            .monospacedDigit()
                    }
                }
            }
        }
    }
}
