import SwiftUI
import MTPLXAppCore

// MARK: - LaunchPopover
//
// Popover body. SwiftUI's `.popover` already handles the speech-bubble
// chrome (arrow, shadow, NSPanel hosting), so this view is just the
// stacked list of target choices — no card background, no custom
// arrow, no shadow drawing. Width is intentionally narrow (260pt) so
// the bubble reads as a compact menu rather than a panel.

struct LaunchPopover: View {
    let lastTarget: String
    let onPick: (LaunchTarget) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ForEach(LaunchTarget.allCases) { target in
                row(target: target)
                if target != LaunchTarget.allCases.last {
                    Divider().padding(.leading, 40)
                }
            }
        }
        .frame(width: 260)
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Start MTPLX")
                .font(.system(.caption, design: .rounded).weight(.semibold))
                .foregroundStyle(.primary)
            Text("Pick what you're serving.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 6)
    }

    @ViewBuilder
    private func row(target: LaunchTarget) -> some View {
        let isLast = lastTarget == target.rawValue
        Button {
            onPick(target)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: target.systemImage)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(isLast ? Color.accentColor : .secondary)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 6) {
                        Text(target.title.mtplxLocalized)
                            .font(.system(.callout).weight(.medium))
                            .foregroundStyle(.primary)
                        if isLast {
                            Text("last")
                                .font(.system(size: 9, weight: .heavy))
                                .foregroundStyle(Color.accentColor)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(
                                    Capsule().stroke(Color.accentColor.opacity(0.6), lineWidth: 0.75)
                                )
                        }
                    }
                    Text(target.tagline.mtplxLocalized)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(LaunchRowButtonStyle())
    }
}

// MARK: - LaunchRowButtonStyle

private struct LaunchRowButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                Rectangle()
                    .fill(configuration.isPressed ? Color.accentColor.opacity(0.15) : Color.clear)
            )
    }
}
