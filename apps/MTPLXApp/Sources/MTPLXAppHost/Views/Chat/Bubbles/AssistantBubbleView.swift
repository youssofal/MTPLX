import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - AssistantBubbleView
//
// Left-anchored surface for one persisted assistant TURN — which may
// span several stored messages when web search ran (think → search →
// think → answer). Composes (top to bottom): ONE `TurnActivityStrip`
// (equal-width Thought / Searched chips whose detail wells expand on
// tap), the answer markdown, then a compact `SourcesFooterView`
// capsule. The strip has the exact same geometry as the streaming
// surface's, so the live→settled handoff doesn't jump (2026-07-03
// chat-UX redesign).
//
// Container: 576pt max width, `Brand.cardSurface` fill, `Brand.separator`
// border, mirrored asymmetric corners (small 4pt on bottom-leading —
// the tail side; large 14pt elsewhere).

struct AssistantBubbleView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    let group: AssistantTurnGroup
    private let message: ChatMessage
    private let combinedReasoning: String
    private let sources: [SourceRecord]
    private let searchQueries: [String]
    private let fetchedPageCount: Int
    private let thinkingTimeMs: Int?
    private let metricItems: [MetricItem]
    private let replyCopyText: String
    private let isInterruptedReply: Bool
    private let longReplyPreviewText: String?
    @State private var isHovered: Bool = false
    @State private var expandedLongReply: Bool = false
    @State private var expandedDetail: TurnActivityModel.Detail = .none

    init(group: AssistantTurnGroup) {
        self.group = group
        let finalMessage = group.finalMessage
        self.message = finalMessage
        self.combinedReasoning = group.combinedReasoning
        self.sources = group.sources
        self.searchQueries = group.searchQueries
        self.fetchedPageCount = group.fetchedPageCount
        self.thinkingTimeMs = group.thinkingTimeMs
        self.metricItems = Self.formattedMetrics(from: finalMessage.statsJSON)
        self.replyCopyText = finalMessage.visibleContent
            .trimmingCharacters(in: .whitespacesAndNewlines)
        self.isInterruptedReply = Self.isInterruptedFinishReason(finalMessage.finishReason)
        self.longReplyPreviewText = Self.longReplyPreview(for: self.replyCopyText)
    }

    init(message: ChatMessage) {
        self.init(group: AssistantTurnGroup(id: message.id, members: [message]))
    }

    private var activityModel: TurnActivityModel {
        TurnActivityModel.settled(
            hasThought: !combinedReasoning.isEmpty,
            thinkingTimeMs: thinkingTimeMs,
            searchCount: searchQueries.count,
            fetchedPageCount: fetchedPageCount,
            hasOtherToolActivity: !group.traces.isEmpty
        )
    }

    /// Search-well receipt rows: one per query, plus a page-read
    /// summary line when the turn fetched pages.
    private var settledActivityRows: [ThinkingActivityRow] {
        var rows = searchQueries.enumerated().map { index, query in
            ThinkingActivityRow(
                id: "query-\(index)",
                systemName: "magnifyingglass",
                text: query,
                detail: "",
                isLive: false
            )
        }
        if fetchedPageCount > 0 {
            rows.append(
                ThinkingActivityRow(
                    id: "fetched-pages",
                    systemName: "doc.text",
                    text: fetchedPageCount == 1
                        ? "Read 1 page"
                        : "Read \(fetchedPageCount) pages",
                    detail: "",
                    isLive: false
                )
            )
        }
        return rows
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // The turn's whole activity record — reasoning and tool
            // work — as ONE strip of tap-to-expand chips above the
            // answer, matching the streaming surface exactly.
            TurnActivityStrip(
                model: activityModel,
                expandedDetail: $expandedDetail,
                thoughtWell: {
                    SettledThoughtWell(content: combinedReasoning)
                },
                searchWell: {
                    SearchActivityWell(rows: settledActivityRows)
                }
            )
            .frame(maxWidth: 576, alignment: .leading)
            let hasVisibleAnswer = !message.visibleContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let hasReasoning = !combinedReasoning.isEmpty
            let hasTrace = !group.traces.isEmpty
            let hasToolCalls = message.toolCallsJSON?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            if hasVisibleAnswer {
                HStack(alignment: .top, spacing: 0) {
                    Group {
                        if isInterruptedReply && !expandedLongReply {
                            LongAssistantReplyPreview(
                                title: "Interrupted reply",
                                previewText: longReplyPreviewText ?? replyCopyText,
                                characterCount: message.visibleContent.count,
                                onCopy: { copyToPasteboard(replyCopyText) },
                                onExpand: { expandedLongReply = true }
                            )
                        } else {
                            AssistantMarkdownView(
                                message.visibleContent,
                                isStreaming: false,
                                plainTextOnly: backend.configuration.performanceLock
                            )
                        }
                    }
                        .frame(maxWidth: 576, alignment: .leading)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 11)
                        .background(
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
                        )
                    Spacer(minLength: 60)
                }
            } else if !hasReasoning && !hasTrace && !hasToolCalls {
                HStack(alignment: .top, spacing: 0) {
                    Text("No visible answer generated.")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary)
                        .frame(maxWidth: 576, alignment: .leading)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 11)
                        .background(
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
                        )
                    Spacer(minLength: 60)
                }
            }
            // Where the turn's web sources live — one quiet pill row,
            // not a card per fetched page.
            SourcesFooterView(sources: sources)
                .frame(maxWidth: 576, alignment: .leading)
            // Hover-revealed metrics footer (web-dashboard parity).
            // Renders in a fixed-height slot so the layout below
            // doesn't shift when it appears.
            metricsFooter
                .frame(maxWidth: 576, alignment: .leading)
                .frame(height: 20, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .onHover { hovering in
            isHovered = hovering
        }
    }

    // MARK: - Hover-revealed metrics footer
    //
    // Reads `ChatMessage.statsJSON` (persisted on every assistant turn
    // by `ChatViewModel.persistAssistantTurn`) and renders a compact
    // metrics strip beneath the bubble. The strip occupies a fixed
    // 20pt slot so the layout doesn't shift when it fades in. Items
    // are separated by a dim middle-dot, monospaced for stable digit
    // widths.

    @ViewBuilder
    private var metricsFooter: some View {
        if !metricItems.isEmpty || !replyCopyText.isEmpty {
            HStack(spacing: 8) {
                ForEach(Array(metricItems.enumerated()), id: \.offset) { index, item in
                    if index > 0 {
                        Text("·")
                            .font(.system(size: 10))
                            .foregroundStyle(Brand.typeTertiary.opacity(0.6))
                    }
                    metricItem(label: item.label, value: item.value)
                }
                if !replyCopyText.isEmpty {
                    if !metricItems.isEmpty {
                        Text("·")
                            .font(.system(size: 10))
                            .foregroundStyle(Brand.typeTertiary.opacity(0.6))
                    }
                    copyButton(for: replyCopyText)
                }
            }
            .padding(.leading, 4)
            .opacity(isHovered ? 1.0 : 0.0)
            .animation(.smooth(duration: 0.18), value: isHovered)
        }
    }

    private struct MetricItem {
        let label: String
        let value: String
    }

    private static func isInterruptedFinishReason(_ reason: String?) -> Bool {
        switch reason?.lowercased() {
        case "cancelled", "error":
            return true
        default:
            return false
        }
    }

    private static func longReplyPreview(for text: String) -> String? {
        guard text.count > 1_200 else { return nil }
        let head = text.prefix(760)
        let tail = text.suffix(320)
        return """
        \(head)

        ...

        \(tail)
        """
    }

    private static func formattedMetrics(from statsJSON: String?) -> [MetricItem] {
        guard let json = statsJSON,
            let data = json.data(using: .utf8),
            let stats = try? JSONDecoder().decode(ChatTurnStats.self, from: data)
        else { return [] }
        var items: [MetricItem] = []
        if let tps = stats.rawDecodeTokS ?? stats.displayDecodeTokS, tps > 0 {
            items.append(MetricItem(label: "tok/s", value: String(format: "%.1f", tps)))
        }
        if let completion = stats.completionTokens, completion > 0 {
            items.append(MetricItem(label: "out", value: Self.formatCount(completion)))
        }
        if let prompt = stats.promptTokens, prompt > 0 {
            items.append(MetricItem(label: "in", value: Self.formatCount(prompt)))
        }
        if let ttft = stats.ttftS, ttft > 0 {
            items.append(MetricItem(label: "TTFT", value: Self.formatSeconds(ttft)))
        }
        if let verifyCalls = stats.verifyCalls, verifyCalls > 0 {
            items.append(MetricItem(label: "verify", value: "\(verifyCalls)"))
        }
        return items
    }

    @ViewBuilder
    private func metricItem(label: String, value: String) -> some View {
        HStack(spacing: 4) {
            Text(value)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
            Text(label)
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(0.6)
                .foregroundStyle(Brand.typeTertiary)
        }
    }

    @ViewBuilder
    private func copyButton(for text: String) -> some View {
        Button {
            copyToPasteboard(text)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 9, weight: .semibold))
                Text("copy")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(0.6)
            }
            .foregroundStyle(Brand.typeTertiary)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Copy reply to clipboard")
    }

    private func copyToPasteboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private struct LongAssistantReplyPreview: View {
        let title: String
        let previewText: String
        let characterCount: Int
        let onCopy: () -> Void
        let onExpand: () -> Void

        var body: some View {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                    Text(title)
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .tracking(0.8)
                        .foregroundStyle(Brand.typeSecondary)
                    Text(Self.formatCount(characterCount))
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                    Spacer(minLength: 8)
                    Button(action: onCopy) {
                        Label("Copy", systemImage: "doc.on.doc")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Brand.typeSecondary)
                    .help("Copy full reply")
                    Button(action: onExpand) {
                        Label("Show", systemImage: "arrow.down.right.and.arrow.up.left")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Brand.typeSecondary)
                    .help("Show full reply")
                }

                AssistantReplyPreviewViewport(text: previewText)
                    .frame(height: 220)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }

        private static func formatCount(_ value: Int) -> String {
            if value >= 1000 {
                return String(format: "%.1fk chars", Double(value) / 1000.0)
            }
            return "\(value) chars"
        }
    }

    private struct AssistantReplyPreviewViewport: NSViewRepresentable {
        let text: String

        func makeNSView(context: Context) -> NSScrollView {
            let scrollView = NSScrollView()
            scrollView.drawsBackground = false
            scrollView.borderType = .noBorder
            scrollView.hasVerticalScroller = true
            scrollView.hasHorizontalScroller = false
            scrollView.autohidesScrollers = true

            let textView = NSTextView()
            textView.drawsBackground = false
            textView.isEditable = false
            textView.isSelectable = true
            textView.isRichText = false
            textView.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
            textView.textColor = NSColor(calibratedWhite: 0.88, alpha: 1.0)
            textView.textContainerInset = NSSize(width: 0, height: 0)
            textView.textContainer?.lineFragmentPadding = 0
            textView.textContainer?.widthTracksTextView = true
            textView.isHorizontallyResizable = false
            textView.isVerticallyResizable = true
            textView.minSize = NSSize(width: 0, height: 0)
            textView.maxSize = NSSize(
                width: CGFloat.greatestFiniteMagnitude,
                height: CGFloat.greatestFiniteMagnitude
            )
            textView.string = text

            scrollView.documentView = textView
            return scrollView
        }

        func updateNSView(_ scrollView: NSScrollView, context: Context) {
            guard let textView = scrollView.documentView as? NSTextView else { return }
            if textView.string != text {
                textView.string = text
            }
        }
    }

    private static func formatCount(_ value: Int) -> String {
        if value >= 1000 {
            return String(format: "%.1fk", Double(value) / 1000.0)
        }
        return "\(value)"
    }

    private static func formatSeconds(_ value: Double) -> String {
        if value < 1.0 {
            return String(format: "%.0fms", value * 1000)
        }
        return String(format: "%.1fs", value)
    }

}
