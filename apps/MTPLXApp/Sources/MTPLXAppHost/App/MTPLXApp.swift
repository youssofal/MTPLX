import SwiftUI
import SwiftData
import AppKit
import Dispatch
import MTPLXAppCore

@MainActor
final class AppStopCoordinator: ObservableObject {
    @Published private(set) var isStoppingEverything = false

    var stopAllHandler: ((_ reason: String) async -> Void)?

    func stopAll(reason: String = "unspecified") async {
        guard !isStoppingEverything else { return }
        isStoppingEverything = true
        defer { isStoppingEverything = false }
        await stopAllHandler?(reason)
    }
}

@MainActor
private final class AppTerminationCoordinator {
    static let shared = AppTerminationCoordinator()

    var cleanup: (() async -> Void)?
    var isCleaningUp = false
}

private final class MTPLXApplicationDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let coordinator = AppTerminationCoordinator.shared
        guard !coordinator.isCleaningUp else {
            return .terminateNow
        }
        coordinator.isCleaningUp = true
        Task { @MainActor in
            await coordinator.cleanup?()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

private final class AppMemoryPressureMonitor {
    private var source: DispatchSourceMemoryPressure?

    func start() {
        guard source == nil else { return }
        let pressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .main
        )
        pressureSource.setEventHandler {
            ChatRenderCaches.clearMemoryPressureSensitiveCaches()
        }
        pressureSource.resume()
        source = pressureSource
    }
}

@main
struct MTPLXApp: App {
    @NSApplicationDelegateAdaptor(MTPLXApplicationDelegate.self) private var appDelegate
    @StateObject private var backend: MTPLXBackendStore
    @StateObject private var themeStore = ThemeStore()
    @StateObject private var router = AppRouter()
    @StateObject private var chatViewModel: ChatViewModel
    @StateObject private var hermesAgentStore: HermesAgentStore
    @StateObject private var stopCoordinator: AppStopCoordinator
    @StateObject private var appUpdater = MTPLXAppUpdater()
    /// Forge orchestrator lives at the app root so wizard state +
    /// in-flight build progress survive tab switches. Without this
    /// the user would lose mid-build progress every time they
    /// switched to Live or Settings.
    @StateObject private var forgeOrchestrator = ForgeOrchestrator()
    /// Benchmark orchestrator lives at the app root so an in-flight
    /// AIME run survives overlay close (the BenchmarkOverlay is
    /// dismissable but does NOT cancel the run). Same lifetime
    /// guarantee as forgeOrchestrator.
    @StateObject private var benchmarkOrchestrator: BenchmarkOrchestrator
    private let chatContainer: ModelContainer
    private let memoryPressureMonitor = AppMemoryPressureMonitor()

    init() {
        // Existing users skip onboarding, so migrate the legacy terminal
        // symlink on every app launch before it can run the bundled Python
        // without the signature-safe bytecode cache environment.
        _ = try? RuntimeSetupService.migrateLegacyTerminalShimIfNeeded()
        let backend = MTPLXBackendStore()
        let hermesAgentStore = HermesAgentStore()
        let benchmarkOrchestrator = BenchmarkOrchestrator()
        let stopCoordinator = AppStopCoordinator()
        let container: ModelContainer
        do {
            container = try ChatStore.makeContainer()
        } catch {
            // SwiftData store is corrupt or inaccessible. Fall back to
            // an in-memory store so the app can still launch; the user
            // gets a fresh chat history but the dashboard still works.
            container = try! ChatStore.makeInMemoryContainer()
        }
        let viewModel = ChatViewModel(
            container: container,
            chatClientProvider: { [backend] in
                MTPLXChatClient(apiClient: backend.apiClient)
            },
            modelName: { [backend] in
                backend.health?.model
                    ?? backend.snapshot?.modelId
                    ?? backend.configuration.model
            },
            reasoningEnabledProvider: { [backend] in
                if let enableThinking = backend.settings?.enableThinking {
                    return enableThinking
                }
                return ChatReasoningPolicy.enableThinking(
                    explicitMode: backend.settings?.reasoning
                        ?? backend.configuration.reasoning,
                    modelControls: backend.settings?.modelControls,
                    modelFamily: backend.settings?.modelFamily
                        ?? backend.configuration.liveSettingsModelFamily
                )
            },
            onDaemonUnreachable: { [backend] in
                backend.markDaemonUnreachable(
                    reason: "MTPLX lost contact with the model server. Start it again."
                )
            }
        )
        _backend = StateObject(wrappedValue: backend)
        _chatViewModel = StateObject(wrappedValue: viewModel)
        _hermesAgentStore = StateObject(wrappedValue: hermesAgentStore)
        _stopCoordinator = StateObject(wrappedValue: stopCoordinator)
        _benchmarkOrchestrator = StateObject(wrappedValue: benchmarkOrchestrator)
        self.chatContainer = container
        memoryPressureMonitor.start()
        stopCoordinator.stopAllHandler = { [backend, viewModel, hermesAgentStore, benchmarkOrchestrator] reason in
            let stack = Thread.callStackSymbols.prefix(12).joined(separator: "\n")
            AIMEDiagnostics.record(
                "app_stop_all_invoked",
                fields: [
                    "reason": .string(reason),
                    "benchmark_run_id": .string(benchmarkOrchestrator.runID ?? ""),
                    "benchmark_state": .string(benchmarkOrchestrator.state.rawValue),
                    "daemon_state": .string(backend.daemonState.kind.rawValue),
                    "call_stack": .string(stack)
                ],
                flushImmediately: true,
                force: true
            )
            if benchmarkOrchestrator.runID != nil || benchmarkOrchestrator.state.isLive {
                await benchmarkOrchestrator.cancel(reason: "app_stop_all:\(reason)")
            }
            await viewModel.cancel()
            await backend.stopDaemon()
            await hermesAgentStore.stop()
        }
        AppTerminationCoordinator.shared.cleanup = { [stopCoordinator, backend] in
            await stopCoordinator.stopAll(reason: "app_termination")
            // stopDaemon() now detaches the slow process reap; wait for it
            // here so quitting the app never orphans the serve child.
            await backend.awaitDaemonTeardown()
        }
    }

    var body: some Scene {
        WindowGroup("MTPLX", id: "main") {
            ContentView(backend: backend)
                .environmentObject(backend)
                .environmentObject(themeStore)
                .environmentObject(router)
                .environmentObject(chatViewModel)
                .environmentObject(hermesAgentStore)
                .environmentObject(stopCoordinator)
                .environmentObject(forgeOrchestrator)
                .environmentObject(benchmarkOrchestrator)
                .modelContainer(chatContainer)
                .task {
                    // Wire the benchmark orchestrator to read the
                    // freshest MTPLXAPIClient every call so mid-run
                    // port/key changes in Settings don't strand it.
                    // The readiness hook starts/adopts the daemon path
                    // before the first AIME request, so the benchmark
                    // button is product-real even from a cold app.
                    benchmarkOrchestrator.attach(
                        apiClientProvider: { [backend] in backend.apiClient },
                        daemonReadinessProvider: { [backend] in
                            _ = try await backend.ensureDaemonReadyForBenchmark()
                            try? await backend.refreshSnapshot()
                        },
                        startOptionsProvider: { [backend] in
                            BenchmarkStartOptions(settings: backend.settings)
                        }
                    )
                    backend.loadPersistedSettings()
                    await backend.refreshRuntimeUpdateStatus()
                    // Onboarding gate: if the user has never finished
                    // the first-launch flow, the entire window takes
                    // over with `OnboardingExperienceView` and the
                    // daemon stays stopped — there's nothing useful to
                    // do until they pick a model and download it.
                    if backend.configuration.onboardingCompletedAt == nil {
                        router.onboardingPhase = .onboarding
                    } else {
                        router.onboardingPhase = .completed
                    }
                    // Daemon-ready handoff: launch targets with their own
                    // user surface should open only after the server is
                    // actually responding. Chat flips into the native
                    // surface; Open WebUI opens the old browser chat root.
                    backend.onDaemonReady = { [weak backend, router] target in
                        switch target {
                        case .chat:
                            router.showChat()
                        case .openWebUI:
                            backend?.openWebChat()
                        case .hermes:
                            router.showHermesBrowse()
                        default:
                            break
                        }
                    }
                    // No auto-start during onboarding — the user hasn't
                    // even picked a model yet. After onboarding, the
                    // launch-on-open preference still applies for power
                    // users who set it true.
                    if router.onboardingPhase == .onboarding {
                        return
                    }
                    if backend.configuration.launchDaemonOnOpen {
                        await backend.startDaemon()
                    } else {
                        await backend.attachExistingDaemonIfOwned()
                    }
                }
                .overlay(alignment: .bottomTrailing) {
                    if let pending = appUpdater.pendingUpdate {
                        UpdateAvailableToast(
                            update: pending,
                            onInstall: { appUpdater.installPendingUpdate() },
                            onLater: { appUpdater.dismissPendingUpdate() }
                        )
                        .padding(.trailing, 20)
                        // Clear the tab bar instead of covering
                        // Forge/Settings.
                        .padding(.bottom, 84)
                        .transition(
                            .asymmetric(
                                insertion: .move(edge: .bottom).combined(with: .opacity),
                                removal: .opacity
                            )
                        )
                    }
                }
                .animation(
                    .spring(response: 0.4, dampingFraction: 0.8),
                    value: appUpdater.pendingUpdate
                )
        }
        .windowStyle(.hiddenTitleBar)
        // The 420×540 floor is pinned as an explicit AppKit
        // `contentMinSize` by WindowSizingTuner. `.contentMinSize`
        // resizability kept `.minSize` in the hosting view's sizing
        // options, which re-derives window extrema with a FULL
        // `sizeThatFits` walk of the transcript on every constraint
        // invalidation (34% of the idle main thread on a long chat;
        // the 2026-08-17 streaming-freeze field regression).
        .windowResizability(.automatic)
        // Open at a generous default. Without this the window opens close to
        // its 420pt minimum, which crushed wide surfaces like the AIME
        // benchmark header on first launch ("default sizing" looked squashed).
        .defaultSize(width: 1240, height: 820)
        .commands {
            CommandGroup(replacing: .newItem) {}

            // `replacing` (not `after`): the system's default About
            // item would otherwise sit alongside ours (QA-124).
            CommandGroup(replacing: .appInfo) {
                Button("About MTPLX…") { router.presentAbout() }
                CheckForUpdatesCommand(updater: appUpdater)
            }

            CommandGroup(after: .appInfo) {
                Divider()

                Button("Start MTPLX") {
                    Task { await backend.startDaemon() }
                }
                .keyboardShortcut("r", modifiers: [.command])

                Button("Stop MTPLX") {
                    Task {
                        await stopCoordinator.stopAll(reason: "menu_stop_daemon")
                    }
                }
                .keyboardShortcut(".", modifiers: [.command])

                Button("Restart MTPLX") {
                    Task {
                        await stopCoordinator.stopAll(reason: "menu_restart_daemon")
                        await backend.startDaemon()
                    }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])

                Divider()

                Button("Toggle Performance Lock") {
                    Task {
                        var config = backend.configuration
                        config.performanceLock.toggle()
                        try? backend.saveSettings(config)
                        backend.startMetricsStream()
                    }
                }
                .keyboardShortcut("l", modifiers: [.command, .option])
            }

            CommandMenu("View") {
                ForEach(Array(AppTab.allCases.enumerated()), id: \.element.id) { index, tab in
                    Button(tab.title) {
                        router.select(tab)
                    }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: [.command])
                }
                Divider()
                Button("Next Tab") { router.nextTab() }
                    .keyboardShortcut("]", modifiers: [.command])
                Button("Previous Tab") { router.previousTab() }
                    .keyboardShortcut("[", modifiers: [.command])
                Divider()
                Button(router.benchmarkOverlayPresented ? "Close Benchmark" : "Open Benchmark") {
                    if router.benchmarkOverlayPresented {
                        router.closeBenchmark()
                    } else {
                        router.openBenchmark()
                    }
                }
                .keyboardShortcut("b", modifiers: [.command])
                Divider()
                Button("Open Logs…") { router.presentLogs() }
                    .keyboardShortcut("l", modifiers: [.command, .shift])
                Button("Refresh") {
                    Task {
                        try? await backend.refreshStaticState()
                        try? await backend.refreshSnapshot()
                    }
                }
                .keyboardShortcut("r", modifiers: [.command, .option])
            }

            CommandMenu("Chat") {
                Button("New Chat") {
                    if router.primaryMode != .chat { router.showChat() }
                    _ = chatViewModel.createNewConversation()
                }
                .keyboardShortcut("n", modifiers: [.command])
                Button(router.chatSidebarCollapsed ? "Show Sidebar" : "Hide Sidebar") {
                    withAnimation(.smooth(duration: 0.22)) {
                        router.chatSidebarCollapsed.toggle()
                    }
                }
                .keyboardShortcut("/", modifiers: [.command])
                .disabled(router.primaryMode != .chat)
                Divider()
                Button("Toggle Web Search") {
                    chatViewModel.webSearchEnabled.toggle()
                }
                .keyboardShortcut("g", modifiers: [.command, .shift])
                .disabled(chatViewModel.current == nil)
                Divider()
                Button("Stop Generating") {
                    Task { await chatViewModel.cancel() }
                }
                .keyboardShortcut(.escape, modifiers: [])
                .disabled(!chatViewModel.isStreaming)
            }
        }

        // MenuBarExtra source remains in Views/MenuBar, but the scene is
        // intentionally not installed until the mini controller is release-ready.
    }
}
