import SwiftUI

// MARK: - BenchHeaderStat
//
// One numeric column in the header stats cluster (ELAPSED / RESOLVED
// / ACCURACY). Label is a tracked monospaced tag in the quietest
// type tier; value is a rounded-heavy headline with monospacedDigit
// for stable column width across digit changes.

struct BenchHeaderStat: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.mtplxLocalized)
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeTertiary)
            Text(value)
                .font(.system(size: 17, weight: .heavy, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(Brand.typeBody)
                .lineLimit(1)
                .contentTransition(.numericText())
        }
        .accessibilityElement(children: .combine)
    }
}
