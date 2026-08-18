import SwiftUI

// MARK: - StatTile
//
// Single big-number tile. SF Mono digit width keeps the value from
// jittering across updates. Default tint is the polished chrome
// accent so the metric reads as Jet Chrome by default; callers can
// override (success / warning / danger) when the metric carries
// semantic state.

struct StatTile: View {
    let label: String
    let value: String
    var unit: String? = nil
    var systemImage: String? = nil
    var tint: Color = Brand.accentChrome
    var caption: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Brand.typeTertiary)
                }
                Text(label.uppercased().mtplxLocalized)
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
            }
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(.title2, design: .rounded).weight(.bold))
                    .foregroundStyle(tint)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                    .layoutPriority(1)
                if let unit {
                    Text(unit)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(1)
                        .foregroundStyle(Brand.typeSecondary)
                }
            }
            if let caption {
                Text(caption.mtplxLocalized)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
