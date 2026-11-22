import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// The Settings page reads every row's label, help and place from one
/// catalog, and search runs over the same catalog. These pin the catalog's
/// shape and the search a user relies on to find a setting by any word.
final class SettingsCatalogTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    override func tearDown() {
        L10n.activate(.english)
        super.tearDown()
    }

    func testEverySectionHasGroupsAndEveryGroupHasRows() {
        for section in SettingsSection.allCases {
            XCTAssertFalse(section.groups.isEmpty, "\(section) has no groups")
            XCTAssertFalse(section.title.isEmpty)
            XCTAssertFalse(section.subtitle.isEmpty)
        }
        for group in SettingsGroupID.allCases {
            XCTAssertFalse(group.items.isEmpty, "\(group) has no rows")
            XCTAssertFalse(group.title.isEmpty)
        }
    }

    func testEveryRowHasALabelAndItsGroupBelongsToItsSection() {
        for item in SettingsItem.allCases {
            XCTAssertFalse(item.label.isEmpty, "\(item) has no label")
            XCTAssertEqual(item.section, item.group.section)
            // A row this release does not offer (a client of MTPLX 3) is in no section or group.
            XCTAssertEqual(item.section.items.contains(item), item.isOffered, "\(item)")
            XCTAssertEqual(item.group.items.contains(item), item.isOffered, "\(item)")
        }
    }

    func testTheOpeningsOfMTPLX3ClientsAreNotShown() {
        // Claude Code, Codex and DeepSeek Harness arrive in MTPLX 3 (LaunchTarget.v3AgentClients).
        for item in [SettingsItem.claudeCodeOpensIn, .codexOpensIn, .deepseekHarnessOpensIn] {
            XCTAssertFalse(item.isOffered, "\(item)")
        }
        XCTAssertEqual(SettingsGroupID.openIn.items, [.piOpensIn, .openCodeOpensIn, .hermesOpensIn])
    }

    func testEmptySearchShowsEverything() {
        let search = SettingsSearch(query: "   ")
        XCTAssertFalse(search.isActive)
        XCTAssertTrue(SettingsItem.allCases.allSatisfy(search.matches))
    }

    func testSearchFindsSettingsByLabelKeywordAndGroup() {
        XCTAssertTrue(SettingsSearch(query: "port").matches(.hostPort))
        XCTAssertTrue(SettingsSearch(query: "PORT").matches(.hostPort), "case-insensitive")
        XCTAssertTrue(SettingsSearch(query: "swap").matches(.allowSwap))
        XCTAssertTrue(SettingsSearch(query: "ssd").matches(.ssdMaxSize))
        XCTAssertTrue(SettingsSearch(query: "fan").matches(.fanMode))
        // Group and section titles count: "Persistent Cache (SSD)".
        XCTAssertTrue(SettingsSearch(query: "persistent").matches(.ssdFolder))
        // Every word must match somewhere in the row.
        XCTAssertTrue(SettingsSearch(query: "ssd size").matches(.ssdMaxSize))
        XCTAssertFalse(SettingsSearch(query: "ssd size").matches(.ramTotalCap))
        XCTAssertFalse(SettingsSearch(query: "zzzz").matches(.hostPort))
    }

    func testSidebarAndGroupVisibilityFollowTheSearch() {
        let search = SettingsSearch(query: "port")
        XCTAssertTrue(search.hasMatches(.clients))
        XCTAssertFalse(search.hasMatches(.cooling))
        XCTAssertTrue(search.matches(SettingsGroupID.connection))
        let none = SettingsSearch(query: "zzzz")
        for section in SettingsSection.allCases {
            XCTAssertFalse(none.hasMatches(section))
        }
        XCTAssertFalse(none.matches(SettingsGroupID.connection))
        let all = SettingsSearch(query: "")
        XCTAssertTrue(SettingsSection.allCases.allSatisfy(all.hasMatches))
    }

    func testSearchForAHiddenCacheRowShowsThePolicyThatRevealsIt() {
        // The SSD size row shows only once the SSD cache is on; its policy
        // row must show for the same search, or the group would be empty.
        let search = SettingsSearch(query: "ssd size")
        XCTAssertTrue(search.matches(.ssdMaxSize))
        XCTAssertTrue(search.matches(.ssdPolicy))
        XCTAssertFalse(search.matches(.ramPolicy))
        XCTAssertTrue(SettingsSearch(query: "entries").matches(.ramPolicy))
    }

    func testSearchWorksInTheActiveLanguage() {
        L10n.activate(.german)
        // "Speicherlimit" is the German label of Memory limit.
        XCTAssertTrue(SettingsSearch(query: "Speicherlimit").matches(.memoryLimit))
    }

    func testSettingsLeavesTheBottomBarButKeepsItsShortcutCase() {
        XCTAssertFalse(AppTab.barTabs.contains(.settings))
        XCTAssertTrue(AppTab.allCases.contains(.settings))
        XCTAssertEqual(AppTab.barTabs, [.live, .activity, .system, .forge])
    }

    @MainActor
    func testGearOpensSettingsAndTakesYouBackWhereYouWere() {
        let router = AppRouter()
        let previousSection = router.settingsSection
        let previousTab = router.selection
        defer {
            router.settingsSection = previousSection
            router.selection = previousTab
        }
        router.select(.system)
        router.toggleSettings()
        XCTAssertEqual(router.selection, .settings)
        router.toggleSettings()
        XCTAssertEqual(router.selection, .system)
        router.openSettings(section: .cooling)
        XCTAssertEqual(router.selection, .settings)
        XCTAssertEqual(router.settingsSection, .cooling)
    }

    @MainActor
    func testNextAndPreviousTabCycleTheDashboardsOnly() {
        let router = AppRouter()
        let previousTab = router.selection
        defer { router.selection = previousTab }
        router.select(.forge)
        router.nextTab()
        XCTAssertEqual(router.selection, .live, "wraps past Forge without stopping at Settings")
        router.previousTab()
        XCTAssertEqual(router.selection, .forge)
        // From Settings, the step starts at the dashboard Settings came from.
        router.select(.activity)
        router.openSettings()
        router.nextTab()
        XCTAssertEqual(router.selection, .system)
    }
}
