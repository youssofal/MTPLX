import SwiftUI
import MTPLXAppCore

// MARK: - StreamingAssistantView
//
// The live surface for the in-flight assistant TURN — and, while the
// turn streams, its ONLY representation in the transcript (the render
// plan excludes the turn's already-persisted rounds, so nothing shows
// twice). One stack, however many think/search rounds the tool loop
// runs:
//
//   [ 🧠 Thinking ⋯ ] [ 🌐 Searching the web ⋯ ]   ← activity strip;
//   ┌ one detail well ──────────────────────────┐   the well follows
//   │ thought tail  ⇄  search query rows        │   the active phase
//   └────────────────────────────────────────────┘
//   ┌ answer bubble ─────────────────────────────┐  appears with the
//   │ markdown…                                  │  first answer token;
//   └────────────────────────────────────────────┘  wells close then
//   [ 12 sources ]                                   compact capsule
//
// Phase choreography (founder-directed, 2026-07-03): thinking expands
// the thought well; a tool round collapses it and expands the search
// well; the next think round swaps back; the first answer token closes
// both. Chips stay mounted for the rest of the turn, so the settled
// bubble that replaces this view at persist time has the exact same
// shape — the handoff doesn't jump.

struct StreamingAssistantView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var backend: MTPLXBackendStore

    /// The open well. Auto-follows `streamingPhase`; chip taps can
    /// override until the next phase change reasserts the live tool.
    @State private var expandedDetail: TurnActivityModel.Detail = .thought

    /// The final answer has started streaming into the bubble.
    private var contentHasStarted: Bool {
        viewModel.hasStreamingContent
    }

    private var activityModel: TurnActivityModel {
        TurnActivityModel.live(
            phase: viewModel.streamingPhase,
            hasReasoning: viewModel.hasStreamingReasoning,
            traces: viewModel.pendingToolTraces
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TurnActivityStrip(
                model: activityModel,
                expandedDetail: $expandedDetail,
                thoughtWell: {
                    StreamingThoughtWell(
                        document: viewModel.streamingReasoningDocument,
                        fallback: viewModel.streamingReasoning
                    )
                },
                searchWell: {
                    SearchActivityWell(
                        rows: Self.activityRows(from: viewModel.pendingToolTraces)
                    )
                }
            )
            .frame(maxWidth: 576, alignment: .leading)
            .id("streaming-turn-activity")

            if contentHasStarted {
                HStack(alignment: .top, spacing: 0) {
                    StreamingAssistantMarkdownView(
                        document: viewModel.streamingContentDocument,
                        fallbackText: viewModel.streamingContent,
                        plainTextOnly: backend.configuration.performanceLock
                    )
                    .frame(maxWidth: 576, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                    .background(assistantBubbleBackground)
                    Spacer(minLength: 60)
                }

                SourcesFooterView(sources: viewModel.liveTurnSources)
                    .frame(maxWidth: 576, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear {
            expandedDetail = TurnActivityModel.autoDetail(for: viewModel.streamingPhase)
        }
        .onChange(of: viewModel.streamingPhase) { _, phase in
            expandedDetail = TurnActivityModel.autoDetail(for: phase)
        }
    }

    private var assistantBubbleBackground: some View {
        UnevenRoundedRectangle(
            topLeadingRadius: 14,
            bottomLeadingRadius: 4,
            bottomTrailingRadius: 14,
            topTrailingRadius: 14,
            style: .continuous
        )
        .fill(Brand.cardSurface)
        .overlay(
            UnevenRoundedRectangle(
                topLeadingRadius: 14,
                bottomLeadingRadius: 4,
                bottomTrailingRadius: 14,
                topTrailingRadius: 14,
                style: .continuous
            )
            .stroke(Brand.separator, lineWidth: 1)
        )
    }

    /// Search-well rows for the WHOLE turn — `pendingToolTraces`
    /// accumulates across rounds, so earlier rounds' searches stay
    /// listed while the current one pulses.
    static func activityRows(
        from traces: [PendingToolTrace]
    ) -> [ThinkingActivityRow] {
        traces.map { trace in
            ThinkingActivityRow(
                id: trace.id,
                systemName: icon(for: trace.name),
                text: activityText(for: trace),
                detail: trace.status == .pending ? "" : trace.detail,
                isLive: trace.status == .pending
            )
        }
    }

    private static func activityText(for trace: PendingToolTrace) -> String {
        let subtitle = trace.subtitle.trimmingCharacters(in: .whitespacesAndNewlines)
        if !subtitle.isEmpty {
            // "Searched: x" reads as a receipt; inside the live well
            // the bare query reads as the action itself.
            return subtitle
                .replacingOccurrences(of: "Searched: ", with: "")
                .replacingOccurrences(of: "Searching: ", with: "")
        }
        switch trace.name {
        case "web_search": return "Searching the web"
        case "fetch_url": return "Reading page"
        default: return trace.name.replacingOccurrences(of: "_", with: " ")
        }
    }

    private static func icon(for toolName: String) -> String {
        switch toolName {
        case "web_search": return "magnifyingglass"
        case "fetch_url": return "doc.text"
        default: return "wrench.and.screwdriver"
        }
    }
}
