import SwiftUI

// MARK: - FormRow + FormToggleRow
//
// The two canonical Settings-card row shapes. Use only these inside
// `Card` so every row in the app has the same label-column width,
// the same caption placement, and the same right-edge alignment for
// trailing controls. Mirrors macOS System Settings.
//
// Layout:
//   ┌────────────────────┬───────────────────────────────────────────┐
//   │ Label              │ Control                                 ↥ │
//   │ caption (smaller)  │                                           │
//   └────────────────────┴───────────────────────────────────────────┘
//
// `labelColumn` is fixed at 200pt so every row in every card aligns
// vertically. Trailing content is left-aligned in the remaining flex
// space; toggles use `FormToggleRow` which spacers the toggle to the
// far right.

struct FormRow<Content: View>: View {
    let label: String
    var caption: String? = nil
    @ViewBuilder let content: Content

    static var labelColumnWidth: CGFloat { 200 }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(label.mtplxLocalized)
                    .font(.system(.callout))
                    .foregroundStyle(Brand.typeBody)
                if let caption {
                    Text(caption.mtplxLocalized)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(width: Self.labelColumnWidth, alignment: .leading)

            content
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
    }
}

/// Toggle row. Label + optional caption on the left in the standard
/// label column; switch flush against the right edge of the card.
struct FormToggleRow: View {
    let label: String
    var caption: String? = nil
    @Binding var isOn: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(label.mtplxLocalized)
                    .font(.system(.callout))
                    .foregroundStyle(Brand.typeBody)
                if let caption {
                    Text(caption.mtplxLocalized)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            Toggle("", isOn: $isOn)
                .toggleStyle(.switch)
                .labelsHidden()
        }
        .padding(.vertical, 4)
    }
}
