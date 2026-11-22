import XCTest
@testable import MTPLXAppCore

@testable import MTPLXAppHost

/// When the star prompt may appear, and what each answer does.
final class StarPromptPresenterTests: XCTestCase {
    private let stamp = Date(timeIntervalSinceReferenceDate: 800_000_000)

    func testPresentsOnlyWhenOwedAndQuiet() {
        var moment = StarPromptMoment(offered: true)
        XCTAssertTrue(moment.shouldPresent)
        moment.offered = false
        XCTAssertFalse(moment.shouldPresent, "not owed before the third load or after the answer")
    }

    func testEveryBusyMomentHoldsThePromptBack() {
        let busy: [(String, WritableKeyPath<StarPromptMoment, Bool>)] = [
            ("onboarding", \.onboarding),
            ("a sheet, popover or overlay", \.surfacePresented),
            ("a benchmark", \.benchmarkLive),
            ("a download, tune or build", \.downloadOrBuildBusy),
            ("a request in flight", \.requestInFlight),
            ("the server starting or stopping", \.daemonTransitioning),
        ]
        for (name, keyPath) in busy {
            var moment = StarPromptMoment(offered: true)
            moment[keyPath: keyPath] = true
            XCTAssertFalse(moment.shouldPresent, "must wait during \(name)")
        }
    }

    func testURLsAreTheProjectsGitHubPages() {
        XCTAssertEqual(StarPromptPresenter.repositoryURL.absoluteString, "https://github.com/youssofal/MTPLX")
        XCTAssertEqual(
            StarPromptPresenter.newIssueURL.absoluteString,
            "https://github.com/youssofal/MTPLX/issues/new/choose"
        )
    }

    @MainActor
    func testAnAnswerIsIgnoredWhileThePromptIsNotShowing() {
        var opened: [URL] = []
        let presenter = StarPromptPresenter(openURL: { opened.append($0) })
        presenter.complete(.starred)
        XCTAssertTrue(opened.isEmpty)
        XCTAssertFalse(presenter.isPresented)
    }

    @MainActor
    func testTheMomentReadsTheStores() {
        let backend = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(onboardingCompletedAt: stamp, successfulModelLoads: 3),
            settingsStore: MTPLXSettingsStore(settingsURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("mtplx-star-host-\(UUID().uuidString)")
                .appendingPathComponent("settings.json"))
        )
        let router = AppRouter()
        let presenter = StarPromptPresenter(openURL: { _ in })
        presenter.attach(
            backend: backend,
            router: router,
            benchmark: BenchmarkOrchestrator(),
            forge: ForgeOrchestrator()
        )
        XCTAssertTrue(presenter.currentMoment().shouldPresent)
        router.modelPickerPresented = true
        XCTAssertFalse(presenter.currentMoment().shouldPresent, "never over the model picker")
        router.modelPickerPresented = false
        router.onboardingPhase = .onboarding
        XCTAssertFalse(presenter.currentMoment().shouldPresent, "never during onboarding")
        router.onboardingPhase = .completed
        backend.setDaemonStateForTesting(.starting)
        XCTAssertFalse(presenter.currentMoment().shouldPresent, "never while a model loads")
        backend.setDaemonStateForTesting(.running)
        XCTAssertTrue(presenter.currentMoment().shouldPresent)
        backend.markStarPromptCompleted(at: stamp)
        XCTAssertFalse(presenter.currentMoment().shouldPresent, "never after the answer")
    }
}
