import SwiftUI
import MTPLXAppCore

// MARK: - HBarRow
//
// Horizontal bar for "depth N: 95%" style displays. Used by the
// speculative tab's per-depth acceptance row. Bar fills from 0 to 1
// against `tint` (defaults to the polished chrome accent so the bar
// reads as Jet Chrome; callers override semantically).
//
// The inner GeometryReader is intentional: the bar fill width is a
// fraction of the row's render width, and `Layout` / `containerRelativeFrame`
// don't propagate the right anchor cleanly into the Capsule. Flagged
// load-bearing per `references/design.md`.

struct HBarRow: View {
    let label: String
    let value: Double
    var valueLabel: String? = nil
    var tint: Color = Brand.accentChrome
    var trackHeight: CGFloat = 14

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label.mtplxLocalized)
                    .font(.system(.caption, design: .rounded).weight(.bold))
                    .foregroundStyle(Brand.typeBody)
                Spacer()
                Text(valueLabel ?? Format.percent(value))
                    .font(.system(.caption, design: .rounded).weight(.medium))
                    .foregroundStyle(Brand.accentChrome)
                    .monospacedDigit()
                    .contentTransition(.numericText())
            }
            GeometryReader { proxy in
                ZStack(alignment: .leading) {
                    Capsule(style: .continuous)
                        .fill(Brand.separator)
                    Capsule(style: .continuous)
                        .fill(tint.gradient)
                        .frame(width: proxy.size.width * max(0, min(1, value)))
                }
            }
            .frame(height: trackHeight)
        }
    }
}
