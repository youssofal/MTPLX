import SwiftUI

// MARK: - MetricRow
//
// Label / value row used inside `Card` for key-value pairs. Default
// value tint is the polished chrome accent so the right-aligned
// number reads as Jet Chrome; callers override (success / warning /
// danger) when the value carries semantic state.

struct MetricRow: View {
    let label: String
    let value: String
    var valueTint: Color = Brand.accentChrome

    var body: some View {
        HStack {
            Text(label.mtplxLocalized)
                .font(.callout)
                .foregroundStyle(Brand.typeBody.opacity(0.75))
            Spacer(minLength: 8)
            Text(value)
                .font(.system(.callout, design: .rounded).weight(.medium))
                .foregroundStyle(valueTint)
                .monospacedDigit()
                .contentTransition(.numericText())
        }
    }
}
