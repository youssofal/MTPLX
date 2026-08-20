import AppKit
import Combine
import SwiftUI
import MTPLXAppCore

// MARK: - TurnActivityStrip
//
// THE surface for an assistant turn's activity (2026-07-03 chat-UX
// redesign, founder-directed). One strip serves both the streaming
// turn and the settled transcript bubble:
//
//   [ 🧠 Thinking ⋯ ]  [ 🌐 Searched ×3 ]              ← content-hugging chips
//   ┌───────────────────────────────────────────┐
//   │ …one detail well, only ever one open…     │     ← morphs per phase
//   └───────────────────────────────────────────┘
//
// Rules:
//   - Chips sit BESIDE each other (never stacked) and hug their
//     content — the classic chunky capsules (founder reverted the
//     brief equal-width experiment on sight, 2026-07-03 ~02:50).
//   - Exactly one well below the row. While streaming it auto-follows
//     the active tool (thinking → thought lines, searching → query
//     rows); the first answer token closes it. After the turn settles
//     the chips become manual toggles.
//   - The strip renders in the SAME geometry live and settled, so the
//     handoff from streaming surface to persisted bubble doesn't jump.
//
// This replaces the ThinkingCard + AssistantTraceSurface pair, whose
// separate stacked cards (plus the mid-turn persisted rounds rendering
// as a second, settled copy of the same turn) produced the cluttered
// transcript in the founder's 03:00 screenshots.

struct TurnActivityStrip<ThoughtWell: View, SearchWell: View>: View {
    let model: TurnActivityModel
    @Binding var expandedDetail: TurnActivityModel.Detail
    @ViewBuilder var thoughtWell: () -> ThoughtWell
    @ViewBuilder var searchWell: () -> SearchWell

    private var disclosureAnimation: Animation {
        .spring(response: 0.34, dampingFraction: 0.88, blendDuration: 0.12)
    }

    var body: some View {
        if !model.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    ForEach(model.chips) { chip in
                        chipView(chip)
                            .transition(.opacity.combined(with: .scale(scale: 0.96)))
                    }
                }

                switch activeDetail {
                case .thought:
                    wellContainer { thoughtWell() }
                case .search:
                    wellContainer { searchWell() }
                case .none:
                    EmptyView()
                }
            }
            // Value-scoped so these survive the transcript's
            // streaming-wide `transaction { animation = nil }` guard:
            // chip arrivals and well swaps animate, token repaints
            // never do.
            .animation(disclosureAnimation, value: model)
            .animation(disclosureAnimation, value: expandedDetail)
        }
    }

    /// A well only opens for a chip that exists — a stale `.search`
    /// selection on a turn that never searched renders as closed.
    private var activeDetail: TurnActivityModel.Detail {
        switch expandedDetail {
        case .thought: return model.hasChip(.thought) ? .thought : .none
        case .search: return model.hasChip(.search) ? .search : .none
        case .none: return .none
        }
    }

    private func toggle(_ chip: TurnActivityModel.Chip) {
        withAnimation(disclosureAnimation) {
            expandedDetail = expandedDetail == chip.kind.detail ? .none : chip.kind.detail
        }
    }

    private func chipView(_ chip: TurnActivityModel.Chip) -> some View {
        let isOpen = activeDetail == chip.kind.detail
        return Button {
            toggle(chip)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: chip.systemName)
                    .font(.system(size: 11, weight: .medium))
                Text(chip.label)
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .contentTransition(.opacity)
                if let caption = chip.caption {
                    Text(caption)
                        .font(.system(size: 11, design: .rounded))
                        .foregroundStyle(Brand.typeTertiary)
                }
                if chip.isLive {
                    ThinkingIndicatorDots()
                }
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Brand.typeTertiary)
                    .rotationEffect(.degrees(isOpen ? 90 : 0))
            }
            .foregroundStyle(chip.isLive || isOpen ? Brand.typeHi.opacity(0.85) : Brand.typeSecondary)
            .lineLimit(1)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(
                Capsule(style: .continuous)
                    .fill(Color.white.opacity(chip.isLive || isOpen ? 0.10 : 0.06))
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            isOpen ? "Collapse \(chip.label)" : "Expand \(chip.label)"
        )
    }

    private func wellContainer<Content: View>(
        @ViewBuilder _ content: () -> Content
    ) -> some View {
        content()
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(0.04))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(Brand.separator, lineWidth: 1)
                    )
            )
            .transition(.opacity.combined(with: .offset(y: -4)))
    }
}

// MARK: - Search well

/// One line of tool activity inside the search well
/// ("claude opus 4.8 release · Found 5 results").
struct ThinkingActivityRow: Identifiable, Equatable {
    let id: String
    var systemName: String
    var text: String
    var detail: String
    var isLive: Bool
}

struct SearchActivityWell: View {
    let rows: [ThinkingActivityRow]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(rows) { row in
                HStack(spacing: 7) {
                    Image(systemName: row.systemName)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Brand.typeTertiary)
                        .frame(width: 12)
                    Text(row.text)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    if row.isLive {
                        ThinkingIndicatorDots()
                    } else if !row.detail.isEmpty {
                        Text(row.detail)
                            .font(.system(size: 11))
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .transition(.opacity.combined(with: .offset(y: -2)))
            }
        }
        .animation(.spring(response: 0.3, dampingFraction: 0.9), value: rows)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Thought wells
//
// The 72pt / 3-line fading tail viewport and its line shaping are the
// Aphanes V2 port previously hosted in ThinkingCard — constants and
// ramp values carried over verbatim (they were hand-tuned there).

enum ThoughtViewportMetrics {
    static let viewportHeight: CGFloat = 72
    static let settledMaxHeight: CGFloat = 360
}

/// Live reasoning is one literal append-only text surface. It intentionally
/// bypasses Markdown and SwiftUI line shaping: the previous three-slot view
/// rewrapped words, changed each slot's font/inset/opacity, and concatenated a
/// parent-captured pending suffix with the already-flushed document. That
/// stale overlap briefly duplicated and reordered text (for example
/// `offline. L` becoming `Loffline`) before a later repaint corrected it.
struct StreamingThoughtWell: View {
    let document: StreamingDocumentStore

    var body: some View {
        ThoughtStreamViewport(document: document)
            .frame(height: ThoughtViewportMetrics.viewportHeight)
    }
}

/// Settled thought well: the whole turn's reasoning as PLAIN TEXT,
/// scrollable past `settledMaxHeight`. Deliberately not markdown
/// (founder order, 2026-08-18): reasoning is the model talking to
/// itself, styling it buys nothing, and running MarkdownUI over a
/// 10k-char thought dump is real parse + layout work.
struct SettledThoughtWell: View {
    let content: String

    var body: some View {
        ScrollView {
            Text(verbatim: content)
                .font(.system(size: 13, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .frame(maxHeight: ThoughtViewportMetrics.settledMaxHeight)
    }
}

private struct ThoughtStreamViewport: NSViewRepresentable {
    let document: StreamingDocumentStore

    @MainActor
    final class Coordinator {
        private static let highWaterCharacters = 4_096
        private static let lowWaterCharacters = 2_048

        weak var document: StreamingDocumentStore?
        weak var surface: LiveTailTextSurface?
        var appliedText = ""
        var mutationCancellable: AnyCancellable?
        let textAttributes: [NSAttributedString.Key: Any] = {
            let paragraph = NSMutableParagraphStyle()
            paragraph.lineBreakMode = .byCharWrapping
            paragraph.minimumLineHeight = 24
            paragraph.maximumLineHeight = 24
            return [
                .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .regular),
                .foregroundColor: NSColor.secondaryLabelColor,
                .paragraphStyle: paragraph,
            ]
        }()

        func attach(document: StreamingDocumentStore, surface: LiveTailTextSurface) {
            let surfaceChanged = self.surface !== surface
            self.surface = surface
            if self.document !== document {
                mutationCancellable?.cancel()
                self.document = document
                replace(with: document.recentText(characterLimit: Self.lowWaterCharacters))
                mutationCancellable = document.mutationPublisher.sink { [weak self] mutation in
                    self?.apply(mutation)
                }
            } else if surfaceChanged {
                replace(with: appliedText)
            }
            surface.textDidChange()
        }

        private func apply(_ mutation: StreamingDocumentStore.Mutation) {
            switch mutation {
            case .reset:
                replace(with: "")
            case .append(let delta):
                append(delta)
            }
        }

        private func append(_ delta: String) {
            guard !delta.isEmpty, let surface else { return }
            if appliedText.isEmpty {
                surface.textStorage.setAttributedString(NSAttributedString())
            }
            appliedText.append(delta)
            surface.textStorage.append(NSAttributedString(
                string: delta,
                attributes: textAttributes
            ))
            if appliedText.count > Self.highWaterCharacters {
                var tail = String(appliedText.suffix(Self.lowWaterCharacters))
                if let newline = tail.firstIndex(of: "\n") {
                    tail = String(tail[tail.index(after: newline)...])
                }
                replace(with: tail)
                return
            }
            surface.textDidChange()
        }

        private func replace(with text: String) {
            guard let surface else {
                appliedText = text
                return
            }
            appliedText = text
            let visible = text.isEmpty ? "Processing…" : text
            surface.textStorage.setAttributedString(NSAttributedString(
                string: visible,
                attributes: textAttributes
            ))
            surface.textDidChange()
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> LiveTailTextSurface {
        let surface = LiveTailTextSurface(frame: .zero)
        surface.setAccessibilityElement(false)
        context.coordinator.attach(document: document, surface: surface)
        return surface
    }

    func updateNSView(_ surface: LiveTailTextSurface, context: Context) {
        context.coordinator.attach(document: document, surface: surface)
    }
}
