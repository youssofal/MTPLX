import AppKit
import Combine
import SwiftUI
import MTPLXAppCore

// MARK: - StarPromptMoment
//
// Whether now is a moment the star prompt may appear. It is offered once,
// after the third successful model load (`shouldOfferStarPrompt`), and it
// waits for a quiet moment: never during onboarding, never over a sheet,
// popover or overlay of the app, never while a benchmark, a download, a
// tune, a Forge build or a request is running, and never while the server
// is starting or stopping. A load that completes at a busy moment is
// remembered (the count is in the settings file) and the prompt shows at
// the next quiet one.

struct StarPromptMoment: Equatable {
    var offered: Bool = false
    var onboarding: Bool = false
    var surfacePresented: Bool = false
    var benchmarkLive: Bool = false
    var downloadOrBuildBusy: Bool = false
    var requestInFlight: Bool = false
    var daemonTransitioning: Bool = false

    var isQuiet: Bool {
        !onboarding && !surfacePresented && !benchmarkLive && !downloadOrBuildBusy
            && !requestInFlight && !daemonTransitioning
    }

    var shouldPresent: Bool { offered && isQuiet }
}

// MARK: - StarPromptPresenter
//
// Decides when the one-time GitHub star prompt appears and records its
// answer. It watches only while the prompt is owed: before the third load
// it follows the settings alone, after the answer it watches nothing.
//
// It never touches the server, the model or the network. The only network
// effect is the browser opening the GitHub page when the user clicks a
// button. It never takes keyboard focus: a chat being typed keeps its
// focus and every keystroke, and the prompt waits for a few seconds
// without typing before it appears. Escape closes it.

@MainActor
final class StarPromptPresenter: ObservableObject {
    enum Outcome: String {
        case starred
        case fileIssue
        case dismissed
    }

    nonisolated static let repositoryURL = URL(string: "https://github.com/youssofal/MTPLX")!
    nonisolated static let newIssueURL = URL(string: "https://github.com/youssofal/MTPLX/issues/new/choose")!
    /// A quiet moment must last this long before the prompt appears, so it
    /// does not land on the same frame a model becomes ready or a sheet
    /// closes.
    static let settleSeconds: Double = 1.5
    /// Seconds since the last key press in the app, so it never lands
    /// mid-word.
    static let typingQuietSeconds: Double = 3
    /// QA only: when set, the URL a button would open is appended to this
    /// file instead of being opened, so a drive can read what the app
    /// handed over without opening a browser on the tester's desktop.
    static let qaOpenURLLogKey = "MTPLX_APP_QA_OPEN_URL_LOG"

    @Published private(set) var isPresented = false

    private weak var backend: MTPLXBackendStore?
    private weak var router: AppRouter?
    private weak var benchmark: BenchmarkOrchestrator?
    private weak var forge: ForgeOrchestrator?
    private var offeredCancellable: AnyCancellable?
    private var watchCancellables: Set<AnyCancellable> = []
    private var evaluationPending = false
    private var presentTask: Task<Void, Never>?
    private var keyMonitor: Any?
    private var lastKeyDown: Date = .distantPast
    private let openURL: @MainActor (URL) -> Void

    init(openURL: @escaping @MainActor (URL) -> Void = StarPromptPresenter.openExternally) {
        self.openURL = openURL
    }

    /// Wire to the app's stores. Idempotent: the window's startup task runs
    /// again when a window reopens.
    func attach(
        backend: MTPLXBackendStore,
        router: AppRouter,
        benchmark: BenchmarkOrchestrator,
        forge: ForgeOrchestrator
    ) {
        guard offeredCancellable == nil else { return }
        self.backend = backend
        self.router = router
        self.benchmark = benchmark
        self.forge = forge
        offeredCancellable = backend.$configuration
            .map(\.shouldOfferStarPrompt)
            .removeDuplicates()
            .sink { [weak self] offered in
                self?.setWatching(offered)
            }
    }

    /// The user answered: either button, the close button or Escape. The
    /// answer is stamped first, so the prompt never comes back whatever
    /// the browser does.
    func complete(_ outcome: Outcome) {
        guard isPresented else { return }
        backend?.markStarPromptCompleted()
        isPresented = false
        stopWatching()
        switch outcome {
        case .starred:
            openURL(Self.repositoryURL)
        case .fileIssue:
            openURL(Self.newIssueURL)
        case .dismissed:
            break
        }
    }

    static func openExternally(_ url: URL) {
        let environment = ProcessInfo.processInfo.environment
        if let path = environment[qaOpenURLLogKey], !path.isEmpty {
            let line = Data((url.absoluteString + "\n").utf8)
            let fileURL = URL(fileURLWithPath: path)
            if let handle = try? FileHandle(forWritingTo: fileURL) {
                handle.seekToEndOfFile()
                handle.write(line)
                try? handle.close()
            } else {
                try? line.write(to: fileURL)
            }
            return
        }
        NSWorkspace.shared.open(url)
    }

    // MARK: - Watching

    private func setWatching(_ offered: Bool) {
        if offered {
            startWatching()
        } else if !isPresented {
            stopWatching()
        }
    }

    private func startWatching() {
        guard watchCancellables.isEmpty, let backend, let router else { return }
        var triggers: [AnyPublisher<Void, Never>] = [
            backend.$configuration.map { _ in () }.eraseToAnyPublisher(),
            backend.$daemonState.map { _ in () }.eraseToAnyPublisher(),
            backend.$inFlight.map { _ in () }.eraseToAnyPublisher(),
            backend.$pendingModelDownload.map { _ in () }.eraseToAnyPublisher(),
            backend.$isModelDownloading.map { _ in () }.eraseToAnyPublisher(),
            backend.$isModelTuning.map { _ in () }.eraseToAnyPublisher(),
            backend.$modelPackUpdatingRepoID.map { _ in () }.eraseToAnyPublisher(),
            router.objectWillChange.map { _ in () }.eraseToAnyPublisher(),
        ]
        if let benchmark {
            triggers.append(benchmark.$state.map { _ in () }.eraseToAnyPublisher())
        }
        if let forge {
            triggers.append(forge.$isBuilding.map { _ in () }.eraseToAnyPublisher())
            triggers.append(forge.$isPublishing.map { _ in () }.eraseToAnyPublisher())
        }
        Publishers.MergeMany(triggers)
            .sink { [weak self] in self?.scheduleEvaluation() }
            .store(in: &watchCancellables)
        installKeyMonitor()
        scheduleEvaluation()
    }

    private func stopWatching() {
        watchCancellables.removeAll()
        presentTask?.cancel()
        presentTask = nil
        if let keyMonitor {
            NSEvent.removeMonitor(keyMonitor)
            self.keyMonitor = nil
        }
    }

    /// `objectWillChange` and `@Published` fire before the new value is
    /// stored: read the stores on the next turn, once per burst.
    private func scheduleEvaluation() {
        guard !evaluationPending else { return }
        evaluationPending = true
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else { return }
            evaluationPending = false
            evaluate()
        }
    }

    private func evaluate() {
        guard !isPresented, !watchCancellables.isEmpty else { return }
        if currentMoment().shouldPresent {
            guard presentTask == nil else { return }
            presentTask = Task { @MainActor [weak self] in
                await self?.presentAfterQuietPeriod()
            }
        } else {
            presentTask?.cancel()
            presentTask = nil
        }
    }

    private func presentAfterQuietPeriod() async {
        try? await Task.sleep(nanoseconds: UInt64(Self.settleSeconds * 1_000_000_000))
        while !Task.isCancelled {
            let sinceKey = Date().timeIntervalSince(lastKeyDown)
            if sinceKey >= Self.typingQuietSeconds { break }
            let wait = Self.typingQuietSeconds - sinceKey
            try? await Task.sleep(nanoseconds: UInt64(wait * 1_000_000_000))
        }
        guard !Task.isCancelled else { return }
        presentTask = nil
        guard !isPresented, currentMoment().shouldPresent else { return }
        isPresented = true
        AccessibilityNotification.Announcement(tr("Is everything working well?")).post()
    }

    func currentMoment() -> StarPromptMoment {
        guard let backend else { return StarPromptMoment() }
        var moment = StarPromptMoment()
        moment.offered = backend.configuration.shouldOfferStarPrompt
        if let router {
            moment.onboarding = router.onboardingPhase == .onboarding
            moment.surfacePresented = router.logsSheetPresented
                || router.aboutSheetPresented
                || router.languagePromptPresented
                || router.launchPickerPresented
                || router.inferenceParamsPresented
                || router.modelPickerPresented
                || router.benchmarkOverlayPresented
                || backend.pendingModelDownload != nil
        }
        moment.benchmarkLive = benchmark?.state.isLive ?? false
        moment.downloadOrBuildBusy = backend.isModelDownloading
            || backend.isModelTuning
            || backend.modelPackUpdatingRepoID != nil
            || (forge?.isBuilding ?? false)
            || (forge?.isPublishing ?? false)
        moment.requestInFlight = !backend.inFlight.isEmpty
        switch backend.daemonState {
        case .starting, .warming, .stopping:
            moment.daemonTransitioning = true
        case .stopped, .running, .degraded, .crashed:
            moment.daemonTransitioning = false
        }
        return moment
    }

    // MARK: - Keys

    /// Stamps the last key press (the prompt waits for a pause in typing)
    /// and closes the prompt on Escape in the main window. Every other key
    /// passes through untouched; installed only while the prompt is owed.
    private func installKeyMonitor() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self else { return false }
                self.lastKeyDown = Date()
                let inMainWindow = event.window.map {
                    !$0.isSheet && $0.attachedSheet == nil && !($0 is NSPanel)
                } ?? false
                guard self.isPresented, event.keyCode == 53, inMainWindow else { return false }
                self.complete(.dismissed)
                return true
            }
            return consumed ? nil : event
        }
    }
}
