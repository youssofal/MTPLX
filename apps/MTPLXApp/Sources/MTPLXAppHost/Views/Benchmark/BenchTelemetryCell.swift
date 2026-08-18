import SwiftUI

// MARK: - BenchTelemetryCell
//
// Mini-LiveTile for the 4-column telemetry strip under the live
// card. Each cell renders a tracked monospaced label cap and a
// rounded-heavy headline value with monospacedDigit so digit changes
// don't shift column width. Avoid animated numeric transitions here:
// the benchmark receives live metrics every second, and animation makes
// a normal telemetry update read as flicker. Value rendering routes through
// `Text(value, format:)` so the C-style %.1f / %.0f hangovers from
// V0 are gone (per swiftui-pro swift.md "Never use C-style number
// formatting").

struct BenchTelemetryCell: View {
    let label: String
    private let valueText: Text
    private let unit: String?
    private let emphasised: Bool

    init(label: String, value: Double?, fractionDigits: Int, unit: String?, emphasised: Bool = false) {
        self.label = label
        if let value {
            self.valueText = Text(value, format: .number.precision(.fractionLength(fractionDigits)))
        } else {
            self.valueText = Text("—")
        }
        self.unit = unit
        self.emphasised = emphasised
    }

    init(label: String, integer: Int, unit: String?, emphasised: Bool = false) {
        self.label = label
        self.valueText = Text(integer, format: .number)
        self.unit = unit
        self.emphasised = emphasised
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.mtplxLocalized)
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeTertiary)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                valueText
                    .font(.system(size: emphasised ? 20 : 16, weight: .heavy, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(emphasised ? Brand.typeBody : Brand.typeBody.opacity(0.85))
                if let unit {
                    Text(unit)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(0.8)
                        .foregroundStyle(Brand.typeSecondary)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}
