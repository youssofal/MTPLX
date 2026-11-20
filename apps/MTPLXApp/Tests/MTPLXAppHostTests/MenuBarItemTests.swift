import AppKit
import Combine
import SwiftUI
import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// The menu bar item (#692): what its switch announces, its default, and
/// what its readout shows.
final class MenuBarItemTests: XCTestCase {
    private static let key = "mtplx.app.menuBarItem"
    private var savedSwitch: Any?

    override func setUp() {
        super.setUp()
        savedSwitch = UserDefaults.standard.object(forKey: Self.key)
    }

    override func tearDown() {
        if let savedSwitch {
            UserDefaults.standard.set(savedSwitch, forKey: Self.key)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.key)
        }
        super.tearDown()
    }

    /// Why nothing in the Scene is bound to the switch any more: ThemeStore
    /// announces a write even of the value it already holds (five
    /// announcements for one write, 16 November), and a Scene bound to it
    /// crashed at launch. The AppKit item follows the stored setting instead.
    @MainActor
    func testThemeStoreAnnouncesAWriteOfTheValueItHolds() {
        let store = ThemeStore()
        var announcements = 0
        let subscription = store.objectWillChange.sink { _ in announcements += 1 }
        store.showMenuBarItem = store.showMenuBarItem
        subscription.cancel()
        XCTAssertGreaterThan(announcements, 0)
    }

    /// On for a new user: it costs nothing measurable during a decode
    /// (Theme.swift has the rows).
    @MainActor
    func testTheItemIsOnForANewUser() {
        UserDefaults.standard.removeObject(forKey: Self.key)
        XCTAssertTrue(ThemeStore().showMenuBarItem)
        XCTAssertTrue(MenuBarController.isEnabled(UserDefaults(suiteName: "MenuBarItemTests-\(UUID().uuidString)")!))
    }

    /// The number beside the glyph means "decoding now": a running engine
    /// with a live reading while a request in flight has tokens out. A held
    /// reading, the engine's own warm-up (no request in flight), a stopping
    /// engine or a reading that is not a speed shows the glyph alone.
    @MainActor
    func testTheReadoutShowsASpeedOnlyWhileARunningEngineDecodes() {
        XCTAssertEqual(
            MenuBarReadout.reading(decode: .live(85.4), state: .running, decoding: true),
            MenuBarReadout.Reading(state: .running, decodeTokS: 85)
        )
        XCTAssertNil(MenuBarReadout.reading(decode: .live(31), state: .running, decoding: false).decodeTokS)
        XCTAssertNil(
            MenuBarReadout.reading(
                decode: .held(value: 85.4, completedAt: Date()), state: .running, decoding: true
            ).decodeTokS
        )
        XCTAssertNil(MenuBarReadout.reading(decode: .absent, state: .running, decoding: true).decodeTokS)
        XCTAssertNil(MenuBarReadout.reading(decode: .live(0.2), state: .running, decoding: true).decodeTokS)
        XCTAssertNil(MenuBarReadout.reading(decode: .live(.nan), state: .running, decoding: true).decodeTokS)
        XCTAssertEqual(
            MenuBarReadout.reading(decode: .live(85.4), state: .stopping, decoding: true),
            MenuBarReadout.Reading(state: .stopping, decodeTokS: nil)
        )
        XCTAssertEqual(MenuBarReadout.reading(decode: .absent, state: .stopped, decoding: false).state, .stopped)
    }

    private func request(done: Int? = nil, total: Int = 0, generated: Double? = nil) -> InFlightRequest {
        InFlightRequest(
            requestId: "r", startedS: 0, ageS: 1, sessionId: nil, model: nil, promptPreview: "",
            promptTokens: total,
            lastProgress: DynamicObject(values: generated.map { ["completion_tokens": .number($0)] } ?? [:]),
            prefillState: total > 0 ? PrefillState(phase: "prefill", tokensDone: done, tokensTotal: total) : nil,
            cancelled: false
        )
    }

    /// While a request reads its prompt the item shows how much is read,
    /// never "100%" before the first token; once tokens are out, the speed.
    @MainActor
    func testTheReadoutShowsHowMuchOfThePromptIsRead() {
        let reading = MenuBarReadout.reading(decode: .absent, state: .running, inFlight: [request(done: 370, total: 1_000)])
        XCTAssertEqual(reading.prefillPercent, 37)
        XCTAssertTrue(reading.requestActive)
        XCTAssertNil(reading.decodeTokS)
        XCTAssertEqual(
            MenuBarReadout.reading(decode: .absent, state: .running, inFlight: [request(done: 1_000, total: 1_000)])
                .prefillPercent,
            99
        )
        let decoding = MenuBarReadout.reading(
            decode: .live(74.4), state: .running, inFlight: [request(done: 1_000, total: 1_000, generated: 12)]
        )
        XCTAssertEqual(decoding.decodeTokS, 74)
        XCTAssertNil(decoding.prefillPercent)
        XCTAssertTrue(decoding.requestActive)
        XCTAssertEqual(
            MenuBarReadout.reading(decode: .held(value: 74, completedAt: Date()), state: .running, inFlight: []),
            MenuBarReadout.Reading(state: .running, decodeTokS: nil)
        )
        XCTAssertFalse(
            MenuBarReadout.reading(decode: .absent, state: .stopping, inFlight: [request(done: 5, total: 10)])
                .requestActive
        )
    }

    /// The AppKit item: its width never changes while a request runs, it
    /// writes its title only when the text changes, nothing at all while
    /// idle, and gives the wide length back only after the hold, so a tool
    /// loop's next request inside the hold keeps it (the redraws that cost
    /// the SwiftUI item about 43 ms of window-server CPU a second).
    @MainActor
    func testTheItemKeepsItsWidthWhileARequestRunsAndWritesOnlyChangedText() async throws {
        typealias Reading = MenuBarReadout.Reading
        let controller = MenuBarController(readout: MenuBarReadout(), holdSeconds: 0.2) { AnyView(EmptyView()) }
        let surface = RecordingSurface()
        controller.attach(surface: surface)
        XCTAssertEqual(surface.images, 1)
        XCTAssertTrue(surface.titles.isEmpty)
        XCTAssertTrue(surface.lengths.isEmpty)

        let idle = Reading(state: .running, decodeTokS: nil)
        for _ in 0..<5 { controller.apply(idle) }
        XCTAssertEqual(surface.images, 2, "one image for the change to running, none after")
        XCTAssertTrue(surface.titles.isEmpty, "nothing written while idle")

        for reading in [
            Reading(state: .running, decodeTokS: nil, prefillPercent: 12, requestActive: true),
            Reading(state: .running, decodeTokS: nil, prefillPercent: 60, requestActive: true),
            Reading(state: .running, decodeTokS: 74, requestActive: true),
            Reading(state: .running, decodeTokS: 74, requestActive: true),
            Reading(state: .running, decodeTokS: 75, requestActive: true),
        ] {
            controller.apply(reading)
        }
        XCTAssertEqual(surface.lengths, [controller.wideLength])
        XCTAssertEqual(surface.titles, ["12%", "60%", tr("%@ tok/s", "74"), tr("%@ tok/s", "75")])
        XCTAssertEqual(surface.labels[2], tr("MTPLX — %@", tr("%@ tok/s", "74")))

        controller.apply(idle)
        XCTAssertEqual(surface.titles.last, "", "blank at once, never a stale number")
        controller.apply(Reading(state: .running, decodeTokS: nil, prefillPercent: 5, requestActive: true))
        controller.apply(idle)
        XCTAssertEqual(surface.lengths, [controller.wideLength], "a request inside the hold keeps the width")
        try await Task.sleep(nanoseconds: 600_000_000)
        XCTAssertEqual(surface.lengths, [controller.wideLength, controller.narrowLength])
        XCTAssertEqual(surface.images, 2)
        XCTAssertGreaterThan(controller.wideLength, controller.narrowLength)
    }

    /// The switch: the default for a new user, and the stored value wins
    /// once the user sets it.
    @MainActor
    func testTheSettingDecidesWhetherTheItemIsInstalled() {
        let defaults = UserDefaults(suiteName: "MenuBarItemTests-\(UUID().uuidString)")!
        XCTAssertEqual(MenuBarController.isEnabled(defaults), ThemeStore.menuBarItemDefault)
        defaults.set(true, forKey: MenuBarController.settingKey)
        XCTAssertTrue(MenuBarController.isEnabled(defaults))
        defaults.set(false, forKey: MenuBarController.settingKey)
        XCTAssertFalse(MenuBarController.isEnabled(defaults))
    }

    /// Turning the switch on makes one item, though making a status item
    /// announces a defaults change (AppKit writes the item's saved
    /// visibility while it makes it). Each such change is one the controller
    /// follows, and before the guard each made another item until the stack
    /// ran out: the 17 November QA copy crashed at launch with the switch on.
    /// This surface announces a change while it is made, as AppKit does.
    @MainActor
    func testTurningTheSwitchOnMakesOneItemThoughMakingItAnnouncesAChange() {
        let defaults = UserDefaults(suiteName: "MenuBarItemTests-\(UUID().uuidString)")!
        func announce() {
            NotificationCenter.default.post(name: UserDefaults.didChangeNotification, object: defaults)
        }
        defaults.set(true, forKey: MenuBarController.settingKey)
        var made = 0
        let controller = MenuBarController(
            readout: MenuBarReadout(),
            defaults: defaults,
            makeSurface: {
                made += 1
                if made < 4 { announce() }  // bounded, so the unguarded code fails here instead of crashing
                return RecordingSurface()
            }
        ) { AnyView(EmptyView()) }
        controller.start()
        XCTAssertEqual(made, 1)
        XCTAssertTrue(controller.isInstalled)

        defaults.set(false, forKey: MenuBarController.settingKey)
        announce()
        XCTAssertFalse(controller.isInstalled)
        defaults.set(true, forKey: MenuBarController.settingKey)
        announce()
        XCTAssertTrue(controller.isInstalled)
        XCTAssertEqual(made, 2)
        controller.stop()
        XCTAssertFalse(controller.isInstalled)
    }

    /// The M is a template image (macOS tints it for the bar) in every state.
    @MainActor
    func testTheMarkIsATemplateImageInEveryState() {
        for state in [DaemonStateKind.running, .starting, .warming, .stopping, .degraded, .crashed, .stopped] {
            let image = MenuBarMark.image(for: state)
            XCTAssertTrue(image.isTemplate)
            XCTAssertEqual(image.size.height, 15)
        }
    }
}

@MainActor
private final class RecordingSurface: MenuBarItemSurface {
    var titles: [String] = []
    var labels: [String] = []
    var images = 0
    var lengths: [CGFloat] = []

    func show(title: String, accessibilityLabel: String) {
        titles.append(title)
        labels.append(accessibilityLabel)
    }

    func show(image: NSImage) {
        images += 1
    }

    func set(length: CGFloat) {
        lengths.append(length)
    }
}
