import AppKit
import QuartzCore
import SwiftUI
import MTPLXAppCore

// MARK: - ChatConversationView
//
// Scrollable column of messages. Persisted user / assistant / tool
// turns render as bubbles; the in-flight assistant turn renders as
// `StreamingAssistantView` at the bottom. Empty state takes over when
// the conversation has no messages.
//
// Auto-scroll: every render scrolls to the bottom anchor if streaming
// and `policy.shouldAutoScrollForStreamingUpdate` is true. The user
// can scroll up to detach (>120pt) and back to the bottom to reattach
// (<28pt), matching Aphanes' tuning.

/// Plain (non-observed) box for auto-scroll pacing state. Deliberately
/// NOT individual `@State` vars: policy + task bookkeeping mutate on
/// every revision/scroll tick (up to ~62 Hz while streaming), and as
/// `@State` each write invalidated the whole conversation view — every
/// bubble's body re-ran per tick (2026-08-17 field regression). Nothing
/// in `body` reads these; scrolling goes through the AppKit driver.
@MainActor
final class ChatConversationScrollState {
    var policy = ConversationAutoScrollPolicy()
    var autoScrollTask: Task<Void, Never>?
    var deferredScrollTask: Task<Void, Never>?
    var finishScrollRepairTask: Task<Void, Never>?
    var lastAutoScrollAt: ContinuousClock.Instant?

    func cancelTasks() {
        autoScrollTask?.cancel()
        deferredScrollTask?.cancel()
        finishScrollRepairTask?.cancel()
        autoScrollTask = nil
        deferredScrollTask = nil
        finishScrollRepairTask = nil
    }
}

struct ChatConversationView: View {
    @ObservedObject var viewModel: ChatViewModel
    let daemonState: DaemonState
    let startupPhase: DaemonStartupPhase
    let selectedModel: String
    @State private var scroll = ChatConversationScrollState()
    @State private var scrollDriver = ChatConversationScrollDriver()
    @State private var userScroll = ChatConversationUserScrollState()
    @State private var showFullHeavyTranscript = false
    @State private var renderPlan = ChatConversationRenderPlan(
        messages: [],
        showFullHeavyTranscript: false,
        excludedTurnGroupID: nil
    )

    var body: some View {
        let plan = activeRenderPlan
        ScrollView(.vertical, showsIndicators: false) {
            // MUST be a plain (non-lazy) VStack. LazyVStack decides
            // which rows to realize from SwiftUI's own scroll-position
            // bookkeeping — but this transcript is scrolled by AppKit
            // (ChatConversationScrollDriver moves the clip view
            // directly, including synchronously inside the document's
            // frameDidChange during layout). Under 60 Hz content growth
            // the lazy container's realization window desyncs from the
            // actual visible rect and culls rows that are on screen —
            // intermittent flicker escalating to a fully BLANK
            // transcript mid-generation (founder screenshot,
            // 2026-08-18). Row count is already bounded without
            // laziness: ChatConversationRenderPlan slices heavy
            // transcripts to a 4-item tail behind the "earlier history"
            // card, so eager realization stays viewport-scale.
            VStack(alignment: .leading, spacing: 16) {
                if let hiddenTranscriptSummary = plan.hiddenTranscriptSummary {
                    HiddenTranscriptSummaryView(
                        summary: hiddenTranscriptSummary,
                        onShow: revealFullHeavyTranscript
                    )
                    .id("hidden-heavy-transcript-summary")
                }
                ForEach(plan.transcriptItems) { item in
                    switch item {
                    case .user(let message):
                        UserBubbleView(message: message)
                            .id(item.id)
                    case .assistantTurn(let group):
                        // One surface per TURN: a searched answer renders
                        // a single thinking chip + activity chip + bubble
                        // + sources footer, however many think/search
                        // rounds the tool loop persisted.
                        AssistantBubbleView(group: group)
                            .id(item.id)
                    }
                }
                if viewModel.shouldRenderStreamingAssistant {
                    StreamingAssistantView(viewModel: viewModel)
                        .id("streaming-bubble")
                }
                if let error = viewModel.lastError {
                    errorCard(error)
                        .id("error-card")
                }
            }
            .frame(maxWidth: 768)
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            .frame(maxWidth: .infinity)
            .transaction { transaction in
                if viewModel.isStreaming {
                    transaction.animation = nil
                }
            }
            .background(
                ChatConversationScrollObserverView(
                    userScroll: userScroll,
                    onScrollViewResolved: { scrollView in
                        scrollDriver.updateScrollView(scrollView)
                        scrollDriver.userScroll = userScroll
                        if scrollView != nil, scroll.policy.shouldAutoScrollForStreamingUpdate {
                            scheduleDeferredBottomScroll(delays: [.milliseconds(40)])
                        }
                    },
                    onScroll: { distanceToBottom, isUserInitiated in
                        viewModel.uiPerfProbe.scrollTick(
                            distanceToBottom: distanceToBottom,
                            userInitiated: isUserInitiated
                        )
                        performScrollActions(
                            scroll.policy.didScroll(
                                distanceToBottom: distanceToBottom,
                                isUserInitiated: isUserInitiated
                            )
                        )
                    },
                    onDocumentFrameChanged: {
                        synchronousBottomPinIfNeeded()
                    }
                )
            )
        }
        .overlay(alignment: .center) {
            if plan.renderableMessages.isEmpty && !viewModel.isStreaming {
                ChatConversationEmptyStateView(
                    daemonState: daemonState,
                    startupPhase: startupPhase,
                    selectedModel: selectedModel
                )
            }
        }
        .background(Brand.bgOuter)
        .onReceive(viewModel.streamingContentDocument.revisionPublisher) { _ in
            scrollToBottom()
        }
        .onChange(of: viewModel.current?.id) { _, _ in
            handleConversationChange()
        }
        .onChange(of: viewModel.visibleMessages.count) { _, _ in
            updateRenderPlan()
            if viewModel.visibleMessages.last?.role == .user {
                performScrollActions(scroll.policy.didSendUserMessage())
            } else if !viewModel.isStreaming && scroll.policy.shouldAutoScrollForStreamingUpdate {
                performScrollActions([.immediate, .deferred])
                scheduleFinishScrollRepair()
            } else {
                scrollToBottom()
            }
        }
        .onChange(of: viewModel.isStreaming) { _, streaming in
            if streaming {
                scroll.finishScrollRepairTask?.cancel()
                scroll.finishScrollRepairTask = nil
                performScrollActions(scroll.policy.didStartStreaming())
            } else {
                performScrollActions(scroll.policy.didFinishStreaming())
                scheduleFinishScrollRepair()
            }
        }
        .onAppear {
            updateRenderPlan()
            performScrollActions(scroll.policy.didAppear())
        }
        .onDisappear {
            scroll.autoScrollTask?.cancel()
            scroll.deferredScrollTask?.cancel()
            scroll.finishScrollRepairTask?.cancel()
            scroll.autoScrollTask = nil
            scroll.deferredScrollTask = nil
            scroll.finishScrollRepairTask = nil
        }
    }

    // MARK: Synchronous bottom pin (2026-07-31 sawtooth fix)
    //
    // The founder's clip showed the streaming bubble's bottom edge
    // oscillating at ~6 Hz with ±34 px amplitude: content grows in the
    // flush's layout pass, but the scroll correction ran in an async
    // Task gated to a 24/50 ms cadence — so for one-to-three frames the
    // document was taller with the viewport unmoved (new line renders
    // low), then the deferred task yanked it back up. This handler runs
    // inside the document view's frameDidChange notification, i.e. in
    // the SAME layout pass that grew the content: the clip origin moves
    // with the growth and a grown-but-unscrolled frame never reaches
    // the screen. No @State is touched here (the notification fires
    // during AppKit layout); the cadenced path stays as a safety net
    // and simply no-ops once the pin has already glued the bottom.
    private func synchronousBottomPinIfNeeded() {
        guard viewModel.isStreaming,
              scroll.policy.shouldAutoScrollForStreamingUpdate else { return }
        if scrollDriver.scrollToBottom(animated: false) {
            viewModel.uiPerfProbe.scrollPinned()
        }
    }

    private func scrollToBottom(force: Bool = false) {
        guard force || scroll.policy.shouldAutoScrollForStreamingUpdate else { return }
        if force {
            scroll.autoScrollTask?.cancel()
            scroll.autoScrollTask = nil
            performAutoScroll(animated: false)
            return
        }

        guard scroll.autoScrollTask == nil else { return }

        let minimumCadence: Duration
        if viewModel.isStreaming {
            minimumCadence = usesHeavyTranscriptScrollGuard
                ? .milliseconds(50)
                : .milliseconds(24)
        } else {
            minimumCadence = usesHeavyTranscriptScrollGuard
                ? .milliseconds(120)
                : .milliseconds(50)
        }
        let now = ContinuousClock.now
        let delay: Duration
        if let lastScrollAt = scroll.lastAutoScrollAt {
            let elapsed = now - lastScrollAt
            delay = elapsed >= minimumCadence ? .zero : minimumCadence - elapsed
        } else {
            delay = .zero
        }

        scroll.autoScrollTask = Task { @MainActor in
            if delay > .zero {
                try? await Task.sleep(for: delay)
            } else {
                await Task.yield()
            }
            guard !Task.isCancelled else { return }
            scroll.autoScrollTask = nil
            guard scroll.policy.shouldAutoScrollForStreamingUpdate else { return }
            performAutoScroll(animated: false)
        }
    }

    private func performScrollActions(_ actions: [ConversationAutoScrollAction]) {
        guard !actions.isEmpty else { return }
        let guarded = usesHeavyTranscriptScrollGuard
        if actions.contains(.immediate) {
            if guarded {
                scheduleDeferredBottomScroll(delays: [.milliseconds(140)])
            } else {
                scrollToBottom(force: true)
            }
        }
        if actions.contains(.deferred) {
            scheduleDeferredBottomScroll(
                delays: guarded
                    ? [.milliseconds(260)]
                    : [.milliseconds(60), .milliseconds(120), .milliseconds(240)]
            )
        }
    }

    private func scheduleDeferredBottomScroll(delays: [Duration]) {
        scroll.deferredScrollTask?.cancel()
        scroll.deferredScrollTask = Task { @MainActor in
            for delay in delays {
                try? await Task.sleep(for: delay)
                guard !Task.isCancelled, scroll.policy.shouldAutoScrollForStreamingUpdate else { return }
                performAutoScroll(animated: false)
            }
        }
    }

    private func performAutoScroll(animated: Bool) {
        scroll.lastAutoScrollAt = ContinuousClock.now
        _ = scrollDriver.scrollToBottom(animated: animated)
    }

    private func scheduleFinishScrollRepair() {
        scroll.finishScrollRepairTask?.cancel()
        scroll.finishScrollRepairTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            performFinishScrollRepairTick()
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled else { return }
            performFinishScrollRepairTick()
        }
    }

    private func performFinishScrollRepairTick() {
        guard !viewModel.isStreaming, scroll.policy.shouldAutoScrollForStreamingUpdate else { return }
        scroll.lastAutoScrollAt = ContinuousClock.now
        _ = scrollDriver.clampToValidOffset()
        _ = scrollDriver.scrollToBottom(animated: false)
    }

    private func handleConversationChange() {
        scroll.autoScrollTask?.cancel()
        scroll.deferredScrollTask?.cancel()
        scroll.finishScrollRepairTask?.cancel()
        scroll.autoScrollTask = nil
        scroll.deferredScrollTask = nil
        scroll.finishScrollRepairTask = nil
        performScrollActions(scroll.policy.didOpenConversation())
        showFullHeavyTranscript = false
        updateRenderPlan(showFullHeavyTranscript: false)
    }

    private var usesHeavyTranscriptScrollGuard: Bool {
        activeRenderPlan.usesHeavyTranscriptScrollGuard
    }

    /// While the tool loop streams, the in-flight turn's already-
    /// persisted rounds are excluded from the settled transcript — the
    /// live surface at the bottom is the turn's ONE representation.
    /// Lifts automatically the moment streaming ends (including error
    /// and cancel paths, which publish and stop streaming).
    private var liveExcludedTurnGroupID: UUID? {
        viewModel.shouldRenderStreamingAssistant ? viewModel.currentTurnGroupID : nil
    }

    private var activeRenderPlan: ChatConversationRenderPlan {
        if renderPlan.matches(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: showFullHeavyTranscript,
            excludedTurnGroupID: liveExcludedTurnGroupID
        ) {
            return renderPlan
        }
        return ChatConversationRenderPlan(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: showFullHeavyTranscript,
            excludedTurnGroupID: liveExcludedTurnGroupID
        )
    }

    private func updateRenderPlan(showFullHeavyTranscript override: Bool? = nil) {
        renderPlan = ChatConversationRenderPlan(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: override ?? showFullHeavyTranscript,
            excludedTurnGroupID: liveExcludedTurnGroupID
        )
    }

    private func revealFullHeavyTranscript() {
        showFullHeavyTranscript = true
        updateRenderPlan(showFullHeavyTranscript: true)
    }

    private func errorCard(_ error: ChatError) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Brand.warning)
            VStack(alignment: .leading, spacing: 4) {
                Text(error.errorDescription ?? "Something went wrong.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if case .daemonStopped = error {
                    Text("Hit the play button to start a model, then send again.")
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeSecondary)
                }
            }
            if offersRetry(for: error) {
                Button(action: { viewModel.retryLastUserMessage() }) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.system(size: 11, weight: .semibold))
                        .labelStyle(.titleAndIcon)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Brand.warning.opacity(0.14))
                        )
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Brand.warning.opacity(0.38), lineWidth: 0.5)
                        )
                }
                .buttonStyle(.plain)
                .foregroundStyle(Brand.typeHi)
                .disabled(!viewModel.canRetryLastUserMessage)
                .help("Retry the last message")
                .accessibilityLabel("Retry last message")
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.warning.opacity(0.40), lineWidth: 0.5)
                )
        )
        .frame(maxWidth: 768)
    }

    private func offersRetry(for error: ChatError) -> Bool {
        if case .daemonStopped = error {
            return false
        }
        return true
    }
}

private struct ChatConversationEmptyStateView: View {
    let daemonState: DaemonState
    let startupPhase: DaemonStartupPhase
    let selectedModel: String

    var body: some View {
        if let startupState {
            ChatStartupStatusView(state: startupState)
        } else {
            ChatEmptyView()
        }
    }

    private var startupState: ChatStartupStatusView.State? {
        switch daemonState.kind {
        case .starting, .warming:
            return ChatStartupStatusView.State(
                title: startupTitle,
                detail: startupDetail
            )
        case .stopping:
            return ChatStartupStatusView.State(
                title: "Stopping MTPLX",
                detail: "The app is closing the model server and restoring fans."
            )
        default:
            return nil
        }
    }

    private var startupTitle: String {
        switch startupPhase {
        case .launching:
            return "Starting \(selectedModelName)"
        case .waitingForOwnedHealth:
            return "Loading \(selectedModelName)"
        case .rampingFans:
            return "Preparing cooling"
        case .warming:
            return "Warming up \(selectedModelName)"
        case .ready:
            return "\(selectedModelName) is ready"
        case .failed:
            return "Startup failed"
        case .idle:
            return "Starting \(selectedModelName)"
        }
    }

    private var startupDetail: String {
        switch startupPhase {
        case .launching:
            return "Starting the local model…"
        case .waitingForOwnedHealth:
            return "Mapping weights and building the draft head. Large Step loads can take a minute or two cold."
        case .rampingFans:
            return "Waiting for the requested fan profile."
        case .warming:
            return "Running the first warmup tokens before chat opens."
        case .failed(let message):
            return message
        case .ready:
            return "You can send now."
        case .idle:
            return "Preparing the local engine."
        }
    }

    private var selectedModelName: String {
        if let option = MTPLXModelOption.option(matching: selectedModel) {
            return option.shortName
        }
        let expanded = NSString(string: selectedModel).expandingTildeInPath
        let last = URL(fileURLWithPath: expanded).lastPathComponent
        return last.isEmpty ? selectedModel : last
    }
}

private struct HiddenTranscriptSummary: Equatable {
    let messageCount: Int
    let characterCount: Int
}

private struct ChatConversationRenderPlan {
    private static let heavyMessageCharacterThreshold = 3_000
    private static let heavyTranscriptCharacterThreshold = 18_000
    private static let heavyTranscriptTailItemCount = 4

    let renderableMessages: [ChatMessage]
    /// Grouped transcript rows: one item per user message and one per
    /// assistant TURN (a searched turn's several stored messages fold
    /// into a single item, so heavy-tail slicing can never cut a turn
    /// in half).
    let transcriptItems: [ChatTranscriptItem]
    let hiddenTranscriptSummary: HiddenTranscriptSummary?
    let usesHeavyTranscriptScrollGuard: Bool

    private let sourceMessageCount: Int
    private let firstSourceMessageID: UUID?
    private let lastSourceMessageID: UUID?
    private let showFullHeavyTranscript: Bool
    private let excludedTurnGroupID: UUID?

    init(
        messages: [ChatMessage],
        showFullHeavyTranscript: Bool,
        excludedTurnGroupID: UUID? = nil
    ) {
        self.sourceMessageCount = messages.count
        self.firstSourceMessageID = messages.first?.id
        self.lastSourceMessageID = messages.last?.id
        self.showFullHeavyTranscript = showFullHeavyTranscript
        self.excludedTurnGroupID = excludedTurnGroupID

        var totalCharacters = 0
        var heavy = false
        var renderable: [ChatMessage] = []
        renderable.reserveCapacity(messages.count)

        for message in messages {
            if Self.isRenderableMessage(message) {
                renderable.append(message)
            }

            if !heavy {
                let messageCharacters = message.visibleContent.count + (message.reasoningContent?.count ?? 0)
                if messageCharacters >= Self.heavyMessageCharacterThreshold {
                    heavy = true
                } else {
                    totalCharacters += messageCharacters
                    if totalCharacters >= Self.heavyTranscriptCharacterThreshold {
                        heavy = true
                    }
                }
            }
        }

        self.renderableMessages = renderable
        self.usesHeavyTranscriptScrollGuard = heavy

        let allItems = ChatTranscriptGrouping.items(
            from: renderable,
            excludingTurnGroupID: excludedTurnGroupID
        )

        guard !showFullHeavyTranscript,
              heavy,
              allItems.count > Self.heavyTranscriptTailItemCount
        else {
            self.transcriptItems = allItems
            self.hiddenTranscriptSummary = nil
            return
        }

        let hiddenItems = allItems.dropLast(Self.heavyTranscriptTailItemCount)
        var hiddenMessageCount = 0
        var hiddenCharacters = 0
        for item in hiddenItems {
            switch item {
            case .user(let message):
                hiddenMessageCount += 1
                hiddenCharacters += message.visibleContent.count
                    + (message.reasoningContent?.count ?? 0)
            case .assistantTurn(let group):
                hiddenMessageCount += group.members.count
                for member in group.members {
                    hiddenCharacters += member.visibleContent.count
                        + (member.reasoningContent?.count ?? 0)
                }
            }
        }
        self.transcriptItems = Array(allItems.suffix(Self.heavyTranscriptTailItemCount))
        self.hiddenTranscriptSummary = HiddenTranscriptSummary(
            messageCount: hiddenMessageCount,
            characterCount: hiddenCharacters
        )
    }

    func matches(
        messages: [ChatMessage],
        showFullHeavyTranscript: Bool,
        excludedTurnGroupID: UUID?
    ) -> Bool {
        sourceMessageCount == messages.count
            && firstSourceMessageID == messages.first?.id
            && lastSourceMessageID == messages.last?.id
            && self.showFullHeavyTranscript == showFullHeavyTranscript
            && self.excludedTurnGroupID == excludedTurnGroupID
    }

    private static func isRenderableMessage(_ message: ChatMessage) -> Bool {
        switch message.role {
        case .user, .assistant:
            return true
        case .tool, .system:
            return false
        }
    }
}

private struct HiddenTranscriptSummaryView: View {
    let summary: HiddenTranscriptSummary
    let onShow: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
            VStack(alignment: .leading, spacing: 3) {
                Text("Earlier heavy history hidden")
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(0.6)
                    .foregroundStyle(Brand.typeSecondary)
                Text("\(summary.messageCount) messages · \(Self.formatCount(summary.characterCount))")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer(minLength: 8)
            Button(action: onShow) {
                Label("Show", systemImage: "arrow.down.right.and.arrow.up.left")
                    .font(.system(size: 10, weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(Brand.typeSecondary)
            .help("Show older messages")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.cardSurface.opacity(0.74))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .frame(maxWidth: 576, alignment: .leading)
    }

    private static func formatCount(_ value: Int) -> String {
        if value >= 1000 {
            return String(format: "%.1fk chars", Double(value) / 1000.0)
        }
        return "\(value) chars"
    }
}

private struct ChatConversationScrollObserverView: NSViewRepresentable {
    let userScroll: ChatConversationUserScrollState
    let onScrollViewResolved: @MainActor (NSScrollView?) -> Void
    let onScroll: @MainActor (CGFloat, Bool) -> Void
    let onDocumentFrameChanged: @MainActor () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(
            userScroll: userScroll,
            onScrollViewResolved: onScrollViewResolved,
            onScroll: onScroll,
            onDocumentFrameChanged: onDocumentFrameChanged
        )
    }

    func makeNSView(context: Context) -> HostView {
        let view = HostView()
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ nsView: HostView, context: Context) {
        context.coordinator.onScrollViewResolved = onScrollViewResolved
        context.coordinator.onScroll = onScroll
        context.coordinator.onDocumentFrameChanged = onDocumentFrameChanged
        nsView.coordinator = context.coordinator
        DispatchQueue.main.async {
            context.coordinator.attachIfNeeded(from: nsView)
        }
    }

    static func dismantleNSView(_ nsView: HostView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject {
        let userScroll: ChatConversationUserScrollState
        var onScrollViewResolved: @MainActor (NSScrollView?) -> Void
        var onScroll: @MainActor (CGFloat, Bool) -> Void
        var onDocumentFrameChanged: @MainActor () -> Void
        weak var scrollView: NSScrollView?
        private var boundsObserver: NSObjectProtocol?
        private var documentFrameObserver: NSObjectProtocol?
        private var liveScrollStartObserver: NSObjectProtocol?
        private var liveScrollEndObserver: NSObjectProtocol?
        private var wheelMonitor: Any?

        init(
            userScroll: ChatConversationUserScrollState,
            onScrollViewResolved: @escaping @MainActor (NSScrollView?) -> Void,
            onScroll: @escaping @MainActor (CGFloat, Bool) -> Void,
            onDocumentFrameChanged: @escaping @MainActor () -> Void
        ) {
            self.userScroll = userScroll
            self.onScrollViewResolved = onScrollViewResolved
            self.onScroll = onScroll
            self.onDocumentFrameChanged = onDocumentFrameChanged
        }

        func attachIfNeeded(from hostView: HostView) {
            guard let resolvedScrollView = Self.findEnclosingScrollView(from: hostView) else {
                return
            }

            guard scrollView !== resolvedScrollView else { return }
            detach()

            scrollView = resolvedScrollView
            onScrollViewResolved(resolvedScrollView)
            resolvedScrollView.contentView.postsBoundsChangedNotifications = true
            // The live-scroll flag MUST flip synchronously in the
            // notification block (queue .main delivers on the main
            // thread): the frameDidChange pin below runs synchronously
            // inside layout, and the old `Task { }` wrapper let the pin
            // race a scroll the user had already started — the yank the
            // founder felt as the app "grabbing the wheel back".
            liveScrollStartObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.willStartLiveScrollNotification,
                object: resolvedScrollView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                MainActor.assumeIsolated {
                    guard let coordinator = hostView?.coordinator,
                          let resolvedScrollView else { return }
                    coordinator.userScroll.beginLiveScroll()
                    coordinator.onScroll(Self.distanceToBottom(for: resolvedScrollView), true)
                }
            }
            liveScrollEndObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.didEndLiveScrollNotification,
                object: resolvedScrollView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                MainActor.assumeIsolated {
                    guard let coordinator = hostView?.coordinator,
                          let resolvedScrollView else { return }
                    coordinator.onScroll(Self.distanceToBottom(for: resolvedScrollView), true)
                    coordinator.userScroll.endLiveScroll()
                }
            }
            // Live-scroll notifications only cover phased (trackpad)
            // gestures between touch-down and finger-lift. Momentum
            // events and classic non-phased wheel mice bypass them
            // entirely, so the pin used to fight both. A local monitor
            // sees every wheel event before dispatch; any wheel over
            // the transcript extends the inhibit window.
            wheelMonitor = NSEvent.addLocalMonitorForEvents(
                matching: .scrollWheel
            ) { [weak hostView, weak resolvedScrollView] event in
                MainActor.assumeIsolated {
                    guard let coordinator = hostView?.coordinator,
                          let resolvedScrollView,
                          event.window === resolvedScrollView.window else { return }
                    let point = resolvedScrollView.convert(event.locationInWindow, from: nil)
                    guard resolvedScrollView.bounds.contains(point) else { return }
                    coordinator.userScroll.noteWheelEvent()
                }
                return event
            }
            boundsObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification,
                object: resolvedScrollView.contentView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                Task { @MainActor [weak hostView, weak resolvedScrollView] in
                    guard let hostView, let resolvedScrollView, let coordinator = hostView.coordinator else { return }
                    // isActive (not just live-scrolling) so momentum and
                    // classic-wheel scrolls count as user-initiated and
                    // the policy can detach past 120pt.
                    coordinator.onScroll(
                        Self.distanceToBottom(for: resolvedScrollView),
                        coordinator.userScroll.isActive
                    )
                }
            }
            if let documentView = resolvedScrollView.documentView {
                // queue: nil ⇒ the block runs SYNCHRONOUSLY on the
                // posting thread. Frame changes post on the main thread
                // during layout, which is the whole point: the bottom
                // pin runs in the same display cycle that grew the
                // content, so no grown-but-unscrolled frame is ever
                // presented (the founder's ±34 px 6 Hz sawtooth).
                documentView.postsFrameChangedNotifications = true
                documentFrameObserver = NotificationCenter.default.addObserver(
                    forName: NSView.frameDidChangeNotification,
                    object: documentView,
                    queue: nil
                ) { [weak hostView] _ in
                    guard Thread.isMainThread else { return }
                    MainActor.assumeIsolated {
                        guard let coordinator = hostView?.coordinator,
                              !coordinator.userScroll.isActive else { return }
                        coordinator.onDocumentFrameChanged()
                    }
                }
            }
        }

        func detach() {
            if let boundsObserver {
                NotificationCenter.default.removeObserver(boundsObserver)
            }
            if let documentFrameObserver {
                NotificationCenter.default.removeObserver(documentFrameObserver)
            }
            if let liveScrollStartObserver {
                NotificationCenter.default.removeObserver(liveScrollStartObserver)
            }
            if let liveScrollEndObserver {
                NotificationCenter.default.removeObserver(liveScrollEndObserver)
            }
            if let wheelMonitor {
                NSEvent.removeMonitor(wheelMonitor)
            }
            wheelMonitor = nil
            boundsObserver = nil
            documentFrameObserver = nil
            liveScrollStartObserver = nil
            liveScrollEndObserver = nil
            scrollView = nil
            onScrollViewResolved(nil)
        }

        private static func findEnclosingScrollView(from view: NSView?) -> NSScrollView? {
            var candidate = view?.superview
            while let current = candidate {
                if let scrollView = current as? NSScrollView {
                    return scrollView
                }
                candidate = current.superview
            }
            return nil
        }

        private static func distanceToBottom(for scrollView: NSScrollView) -> CGFloat {
            guard let documentView = scrollView.documentView else { return 0 }
            let visibleRect = scrollView.contentView.documentVisibleRect
            return max(0, documentView.bounds.maxY - visibleRect.maxY)
        }
    }

    final class HostView: NSView {
        weak var coordinator: Coordinator?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                guard let self, let coordinator else { return }
                coordinator.attachIfNeeded(from: self)
            }
        }

        override func viewDidMoveToSuperview() {
            super.viewDidMoveToSuperview()
            DispatchQueue.main.async { [weak self] in
                guard let self, let coordinator else { return }
                coordinator.attachIfNeeded(from: self)
            }
        }
    }
}

// MARK: User-scroll arbitration (founder stutter round three, 2026-08-18)
//
// Why this exists: the bottom pin used to lose every fight with a human.
// The live-scroll flag was set inside a `Task { @MainActor }`, so the
// SYNCHRONOUS frameDidChange pin raced it and yanked the viewport while
// the user's fingers were still on the trackpad; the momentum phase ran
// entirely unguarded (didEndLiveScroll fires at finger-lift); and a
// classic non-phased wheel mouse never posts live-scroll notifications
// at all. Meanwhile the policy's 28pt re-attach meant every yank reset
// the user's escape distance — scrolling up mid-stream felt like the
// app was grabbing the wheel back. One shared state object, updated
// synchronously, consulted by every pin path:
// - live-scroll begin/end set the flag in the notification block itself
// - every wheel event over the transcript (phased, momentum, or classic)
//   extends a short inhibit window, so momentum and wheel mice are
//   covered by the same signal
// - bounds changes report `isActive` as user-initiated, so the policy
//   can legitimately detach (>120pt) during momentum/wheel scrolls.
@MainActor
final class ChatConversationUserScrollState {
    private(set) var isLiveScrolling = false
    private var inhibitUntil: CFTimeInterval = 0

    var isActive: Bool {
        isLiveScrolling || CACurrentMediaTime() < inhibitUntil
    }

    func beginLiveScroll() {
        isLiveScrolling = true
    }

    func endLiveScroll() {
        isLiveScrolling = false
        // Momentum keeps delivering wheel events after finger-lift; the
        // grace covers the gap until the first momentum event lands.
        inhibitUntil = max(inhibitUntil, CACurrentMediaTime() + 0.35)
    }

    func noteWheelEvent() {
        inhibitUntil = CACurrentMediaTime() + 0.30
    }
}

@MainActor
private final class ChatConversationScrollDriver {
    weak var scrollView: NSScrollView?
    var userScroll: ChatConversationUserScrollState?

    func updateScrollView(_ scrollView: NSScrollView?) {
        self.scrollView = scrollView
    }

    @discardableResult
    func scrollToBottom(animated: Bool) -> Bool {
        // Never pin against an active user scroll: the user wins, the
        // policy detaches past 120pt, and streaming follows resume only
        // when they return to the bottom.
        if userScroll?.isActive == true { return false }
        _ = clampToValidOffset()
        guard let scrollView,
              let documentView = scrollView.documentView else { return false }

        let clipView = scrollView.contentView
        let targetY: CGFloat
        if documentView.isFlipped {
            targetY = max(documentView.bounds.minY, documentView.bounds.maxY - clipView.bounds.height)
        } else {
            targetY = documentView.bounds.minY
        }
        let targetRect = clipView.constrainBoundsRect(
            NSRect(
                x: clipView.bounds.origin.x,
                y: targetY,
                width: clipView.bounds.width,
                height: clipView.bounds.height
            )
        )
        let targetPoint = targetRect.origin

        if abs(clipView.bounds.origin.y - targetPoint.y) < 0.5 {
            return true
        }

        if animated {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.14
                context.timingFunction = CAMediaTimingFunction(name: .easeOut)
                clipView.animator().setBoundsOrigin(targetPoint)
            } completionHandler: {
                Task { @MainActor in
                    scrollView.reflectScrolledClipView(clipView)
                }
            }
        } else {
            clipView.scroll(to: targetPoint)
            scrollView.reflectScrolledClipView(clipView)
        }

        return true
    }

    @discardableResult
    func clampToValidOffset() -> Bool {
        guard let scrollView,
              scrollView.documentView != nil else { return false }

        let clipView = scrollView.contentView
        let currentRect = NSRect(
            x: clipView.bounds.origin.x,
            y: clipView.bounds.origin.y,
            width: clipView.bounds.width,
            height: clipView.bounds.height
        )
        let constrainedPoint = clipView.constrainBoundsRect(currentRect).origin

        if abs(clipView.bounds.origin.x - constrainedPoint.x) < 0.5,
           abs(clipView.bounds.origin.y - constrainedPoint.y) < 0.5 {
            return true
        }

        clipView.scroll(to: constrainedPoint)
        scrollView.reflectScrolledClipView(clipView)
        return true
    }
}

private struct ChatStartupStatusView: View {
    struct State: Equatable {
        var title: String
        var detail: String
    }

    let state: State

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.regular)
                .tint(Brand.warning)
                .accessibilityHidden(true)
            Text(state.title)
                .font(BrandFont.subtitle())
                .foregroundStyle(Brand.typeHi)
                .multilineTextAlignment(.center)
            Text(state.detail)
                .font(.system(size: 11))
                .foregroundStyle(Brand.typeSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 40)
    }
}
