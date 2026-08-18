import AppKit
import SwiftUI
import MTPLXAppCore

// MARK: - SourcesFooterView
//
// Where a searched turn's web sources live: ONE quiet capsule under
// the answer — "12 sources" — that expands in place into the numbered
// domain pills on click. Collapsed by default so a searched answer
// ends in a single tidy line instead of a wall of websites
// (2026-07-03 chat-UX redesign; the always-on pill wall was the
// founder's "bunch of websites" complaint).
//
// Clicking a pill opens the page; hovering shows the page title.

struct SourcesFooterView: View {
    let sources: [SourceRecord]

    @State private var isExpanded = false

    private var disclosureAnimation: Animation {
        .spring(response: 0.34, dampingFraction: 0.88, blendDuration: 0.12)
    }

    var body: some View {
        if !sources.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                summaryCapsule

                if isExpanded {
                    FlowRow(horizontalSpacing: 6, verticalSpacing: 6) {
                        ForEach(Array(sources.enumerated()), id: \.element.id) { index, source in
                            sourcePill(index: index + 1, source: source)
                        }
                    }
                    .transition(.opacity.combined(with: .offset(y: -4)))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .animation(disclosureAnimation, value: isExpanded)
            .animation(disclosureAnimation, value: sources.count)
        }
    }

    private var summaryCapsule: some View {
        Button {
            withAnimation(disclosureAnimation) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "link")
                    .font(.system(size: 10, weight: .medium))
                Text(sourceCountLabel)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Brand.typeTertiary)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
            }
            .foregroundStyle(isExpanded ? Brand.typeHi.opacity(0.85) : Brand.typeSecondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(isExpanded ? 0.10 : 0.06))
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(isExpanded ? "Hide sources".mtplxLocalized : showAllSourcesLabel)
        .accessibilityLabel(
            isExpanded ? "Hide sources".mtplxLocalized : showSourcesLabel
        )
    }

    private var sourceCountLabel: String {
        let key = sources.count == 1 ? "%d source" : "%d sources"
        return String(format: key.mtplxLocalized, sources.count)
    }

    private var showAllSourcesLabel: String {
        String(format: "Show all %d sources".mtplxLocalized, sources.count)
    }

    private var showSourcesLabel: String {
        String(format: "Show %d sources".mtplxLocalized, sources.count)
    }

    private func sourcePill(index: Int, source: SourceRecord) -> some View {
        Button {
            guard let url = URL(string: source.url) else { return }
            NSWorkspace.shared.open(url)
        } label: {
            HStack(spacing: 5) {
                Text("\(index)")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                Text(source.domain)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(0.05))
            )
            .overlay(
                Capsule(style: .continuous)
                    .stroke(Brand.separator.opacity(0.7), lineWidth: 0.5)
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(source.title.isEmpty ? source.url : source.title)
        .accessibilityLabel(
            String(
                format: "Open source %1$d: %2$@".mtplxLocalized,
                index,
                source.domain
            )
        )
    }
}
