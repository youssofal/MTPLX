import SwiftUI

// MARK: - BenchPrimaryCTA
//
// Thin wrapper around the canonical `MTPLXPillButton` button style so
// the AIME header's call-to-action cluster routes through the same
// chrome pill recipe as every other primary action in the app
// (Start serving, Run again, Apply parameters). The wrapper keeps
// the existing call-site API (title + icon + isDanger + isEnabled +
// accessibilityHint + action) intact so swapping the style was a
// surgical change here, not 6 edits in BenchHeader.

struct BenchPrimaryCTA: View {
    enum Style {
        case primary
        case secondary
        case danger
    }

    let title: String
    let icon: String
    var style: Style = .primary
    let isDanger: Bool
    let isEnabled: Bool
    var accessibilityHint: String = ""
    let action: () -> Void

    var body: some View {
        Group {
            if isDanger || style == .danger {
                Button(action: action) { labelContent }
                    .buttonStyle(.mtplxDanger)
            } else if style == .secondary {
                Button(action: action) { labelContent }
                    .buttonStyle(.mtplxGhost)
            } else {
                Button(action: action) { labelContent }
                    .buttonStyle(.mtplxPrimary)
            }
        }
        .disabled(!isEnabled)
        .accessibilityLabel(title)
        .accessibilityHint(accessibilityHint)
    }

    @ViewBuilder
    private var labelContent: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .heavy))
            Text(title.mtplxLocalized)
                .lineLimit(1)
                .contentTransition(.opacity)
        }
        // Keep the pill at its natural single-line width. Without this the
        // header HStack compresses the label under space pressure and the
        // text wraps character-by-character ("Pa / us / e").
        .fixedSize(horizontal: true, vertical: false)
    }
}
