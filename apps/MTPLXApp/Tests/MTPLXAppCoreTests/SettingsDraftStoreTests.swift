import XCTest

@testable import MTPLXAppCore

/// The Settings draft lives above the tab. These tests drive the store the
/// way the tab does — `adoptIfUnedited` on every appearance and on every
/// persisted-configuration change, `reset` after a save or a revert — with
/// the tab itself modelled as a throwaway object that is destroyed and
/// recreated between appearances, exactly as the dashboard does.
@MainActor
final class SettingsDraftStoreTests: XCTestCase {
    /// Stand-in for the tab view: it has no state of its own any more,
    /// only a reference to the shared store, and it syncs on appearance.
    @MainActor
    private struct TabScoped {
        let drafts: SettingsDraftStore

        func appear(with persisted: MTPLXAppConfiguration) {
            drafts.adoptIfUnedited(persisted)
        }
    }

    private func persisted() -> MTPLXAppConfiguration {
        MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained", host: "127.0.0.1", port: 8000)
    }

    func testEditedFieldSurvivesTheTabBeingDestroyedAndRecreated() {
        let drafts = SettingsDraftStore()
        let saved = persisted()
        var tab: TabScoped? = TabScoped(drafts: drafts)
        tab?.appear(with: saved)
        XCTAssertEqual(drafts.draft, saved, "first appearance seeds the draft from the saved configuration")
        XCTAssertFalse(drafts.hasUnsavedChanges)

        drafts.draft.port = 9001
        drafts.draft.host = "0.0.0.0"
        XCTAssertTrue(drafts.hasUnsavedChanges)

        // The user looks at another tab: the Settings tab is torn down...
        tab = nil
        // ...and rebuilt when they come back.
        tab = TabScoped(drafts: drafts)
        tab?.appear(with: saved)

        XCTAssertEqual(drafts.draft.port, 9001, "the pending edit is still on screen")
        XCTAssertEqual(drafts.draft.host, "0.0.0.0")
        XCTAssertTrue(drafts.hasUnsavedChanges, "and it still reads as unsaved")
        XCTAssertEqual(drafts.lastSynced, saved, "while the saved values are remembered for Revert")
    }

    func testUneditedDraftFollowsAConfigurationChangedElsewhere() {
        let drafts = SettingsDraftStore()
        let first = persisted()
        TabScoped(drafts: drafts).appear(with: first)

        var moved = first
        moved.port = 8001 // a port fallback persisted from the launch path
        drafts.adoptIfUnedited(moved)

        XCTAssertEqual(drafts.draft, moved)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }

    func testPendingEditsAreKeptOverAnExternalChangeUntilSavedOrReverted() {
        let drafts = SettingsDraftStore()
        let first = persisted()
        TabScoped(drafts: drafts).appear(with: first)
        drafts.draft.contextWindow = 32_768

        var moved = first
        moved.port = 8001
        drafts.adoptIfUnedited(moved)

        XCTAssertEqual(drafts.draft.contextWindow, 32_768, "the user's edit is not thrown away")
        XCTAssertEqual(drafts.draft.port, 8000, "the draft is left as the user sees it")
        XCTAssertTrue(drafts.hasUnsavedChanges)

        drafts.reset(to: moved)
        XCTAssertEqual(drafts.draft, moved)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }

    func testResetAfterSaveAlignsBothHalvesAndClearsUnsaved() {
        let drafts = SettingsDraftStore()
        TabScoped(drafts: drafts).appear(with: persisted())
        drafts.draft.memoryLimitGB = 64
        XCTAssertTrue(drafts.hasUnsavedChanges)

        let savedNow = drafts.draft
        drafts.reset(to: savedNow)

        XCTAssertEqual(drafts.draft, savedNow)
        XCTAssertEqual(drafts.lastSynced, savedNow)
        XCTAssertFalse(drafts.hasUnsavedChanges)
        // A later appearance with the same saved values changes nothing.
        TabScoped(drafts: drafts).appear(with: savedNow)
        XCTAssertEqual(drafts.draft.memoryLimitGB, 64)
        XCTAssertFalse(drafts.hasUnsavedChanges)
    }

    func testImmediateFanModeCommitNeverMarksDraftDirtyAndPreservesOtherEdits() {
        let drafts = SettingsDraftStore()
        let initial = persisted()
        TabScoped(drafts: drafts).appear(with: initial)

        drafts.commitImmediateFanMode(.max)
        XCTAssertEqual(drafts.draft.fanMode, "max")
        XCTAssertTrue(drafts.draft.pinFansAtMaxOnStart)
        XCTAssertEqual(drafts.lastSynced.fanMode, "max")
        XCTAssertTrue(drafts.lastSynced.pinFansAtMaxOnStart)
        XCTAssertFalse(drafts.hasUnsavedChanges)

        // An unrelated edit in Settings remains dirty when Fan Mode changes.
        drafts.draft.memoryLimitGB = 96
        XCTAssertTrue(drafts.hasUnsavedChanges)
        drafts.commitImmediateFanMode(.smart)
        XCTAssertEqual(drafts.draft.memoryLimitGB, 96)
        XCTAssertEqual(drafts.draft.fanMode, "smart")
        XCTAssertFalse(drafts.draft.pinFansAtMaxOnStart)
        XCTAssertTrue(drafts.hasUnsavedChanges)
    }

    func testMergedForSavePreservesExternalFanModeReasoningAndModelPickOverStaleDraft() {
        let drafts = SettingsDraftStore()
        let initial = persisted()
        TabScoped(drafts: drafts).appear(with: initial)

        // User edits a Settings-only field without saving yet.
        drafts.draft.port = 9002
        XCTAssertTrue(drafts.hasUnsavedChanges)

        // Elsewhere in the app, the user selects Max fans, switches to Inkling,
        // and sets reasoning/effort via Model Controls.
        var external = initial
        external.fanMode = "max"
        external.pinFansAtMaxOnStart = true
        external.model = "inkling2-mini-optimized-speed"
        external.liveSettingsModelFamily = "inkling2"
        external.reasoning = "off"
        external.reasoningEffort = "low"

        drafts.syncNonDraftedFields(from: external)
        XCTAssertEqual(drafts.draft.port, 9002, "pending Settings edit stays intact")
        XCTAssertEqual(drafts.draft.fanMode, "max")
        XCTAssertEqual(drafts.draft.reasoning, "off")
        XCTAssertEqual(drafts.draft.reasoningEffort, "low")

        let merged = drafts.mergedForSave(against: external)
        XCTAssertEqual(merged.port, 9002)
        XCTAssertEqual(merged.fanMode, "max")
        XCTAssertTrue(merged.pinFansAtMaxOnStart)
        XCTAssertEqual(merged.model, "inkling2-mini-optimized-speed")
        XCTAssertEqual(merged.liveSettingsModelFamily, "inkling2")
        XCTAssertEqual(merged.reasoning, "off")
        XCTAssertEqual(merged.reasoningEffort, "low")

        // Explicit edits to the Model field inside Settings still win over external.model.
        drafts.draft.model = "/models/custom-override"
        let mergedManualModel = drafts.mergedForSave(against: external)
        XCTAssertEqual(mergedManualModel.model, "/models/custom-override")
    }
}
