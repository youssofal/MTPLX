import Foundation

/// The Settings tab's editing session: the working copy of the persisted
/// configuration, the configuration it was last synced to, and the state
/// of a save in flight.
///
/// It lives above the tab (owned for the app's lifetime and handed down
/// through the environment) because the dashboard rebuilds the Settings
/// tab every time it is selected. When this was the tab's own `@State`,
/// switching to another tab and back silently threw away every pending
/// edit and showed the saved values again, which read as "the app failed
/// to save".
@MainActor
public final class SettingsDraftStore: ObservableObject {
    /// The values the controls edit. Committed by Save, discarded by Revert.
    @Published public var draft: MTPLXAppConfiguration
    /// The persisted configuration the draft was last aligned with. The
    /// difference between the two is exactly the user's pending edits.
    @Published public var lastSynced: MTPLXAppConfiguration
    /// A save/apply is running; the buttons show progress and stay
    /// disabled even if the user leaves and comes back mid-apply.
    @Published public var isApplying = false
    /// The last save's failure, shown until the next save attempt.
    @Published public var lastSaveError: String?

    public init(configuration: MTPLXAppConfiguration = MTPLXAppConfiguration()) {
        draft = configuration
        lastSynced = configuration
    }

    /// True while the draft carries edits the user has not saved.
    public var hasUnsavedChanges: Bool {
        draft != lastSynced
    }

    /// Align the draft with `configuration`, dropping any pending edits.
    /// Used after a successful save (with the saved values) and by Revert.
    public func reset(to configuration: MTPLXAppConfiguration) {
        draft = configuration
        lastSynced = configuration
    }

    /// Take a configuration that changed outside the tab (a port fallback,
    /// a model pick, another session's save) unless the user has pending
    /// edits, which are kept until they save or revert. Called whenever
    /// the tab appears and whenever the persisted configuration changes
    /// while it is visible.
    public func adoptIfUnedited(_ configuration: MTPLXAppConfiguration) {
        if draft == configuration {
            lastSynced = configuration
        } else if draft == lastSynced {
            draft = configuration
            lastSynced = configuration
        }
    }

    /// Commit an immediate-apply Fan Mode selection into both halves of the
    /// draft store so choosing a fan mode never marks Settings dirty or
    /// conflicts with pending edits in other sections.
    public func commitImmediateFanMode(_ mode: MTPLXFanMode) {
        let canonical = mode.rawValue
        let pinned = mode == .max
        if draft.fanMode != canonical || draft.pinFansAtMaxOnStart != pinned {
            var nextDraft = draft
            nextDraft.fanMode = canonical
            nextDraft.pinFansAtMaxOnStart = pinned
            draft = nextDraft
        }
        if lastSynced.fanMode != canonical || lastSynced.pinFansAtMaxOnStart != pinned {
            var nextSynced = lastSynced
            nextSynced.fanMode = canonical
            nextSynced.pinFansAtMaxOnStart = pinned
            lastSynced = nextSynced
        }
    }

    /// Synchronize fields that are not edited via the Settings tab's draft/save
    /// bar (immediate-apply controls like Fan Mode & Scheduling Mode, live
    /// inference/reasoning settings from Model Controls, and app metadata) into
    /// both `draft` and `lastSynced`. That way a pending edit to e.g. memory
    /// limit or port never reverts a newer Fan Mode, reasoning choice, or
    /// custom-model list when Save is pressed.
    public func syncNonDraftedFields(from persisted: MTPLXAppConfiguration) {
        let nextDraft = Self.copyingNonDraftedFields(from: persisted, into: draft)
        let nextSynced = Self.copyingNonDraftedFields(from: persisted, into: lastSynced)
        if nextDraft != draft {
            draft = nextDraft
        }
        if nextSynced != lastSynced {
            lastSynced = nextSynced
        }
    }

    /// Produce the configuration to compare/save against `persisted`: starts
    /// from `draft`, refreshes all non-drafted/immediate fields from
    /// `persisted`, and keeps `persisted.model` if the user did not edit the
    /// Model text field in Settings (e.g. they switched models in the top-bar
    /// model picker while another Settings section had an unsaved edit).
    public func mergedForSave(against persisted: MTPLXAppConfiguration) -> MTPLXAppConfiguration {
        var merged = Self.copyingNonDraftedFields(from: persisted, into: draft)
        if draft.model == lastSynced.model {
            merged.model = persisted.model
        }
        return merged
    }

    private static func copyingNonDraftedFields(
        from source: MTPLXAppConfiguration,
        into target: MTPLXAppConfiguration
    ) -> MTPLXAppConfiguration {
        var result = target
        let fanMode = MTPLXFanMode.normalized(source.fanMode)
        result.fanMode = fanMode.rawValue
        result.pinFansAtMaxOnStart = fanMode == .max
        result.schedulingPreset = source.schedulingPreset
        result.schedulerMode = source.schedulerMode
        result.batchingPreset = source.batchingPreset
        result.temperature = source.temperature
        result.topP = source.topP
        result.topK = source.topK
        result.presencePenalty = source.presencePenalty
        result.reasoning = source.reasoning
        result.reasoningEffort = source.reasoningEffort
        result.controlClientSettings = source.controlClientSettings
        result.liveSettingsModelFamily = source.liveSettingsModelFamily
        result.prefillChunkTokens = source.prefillChunkTokens
        result.adaptiveDepth = source.adaptiveDepth
        result.performanceLock = source.performanceLock
        result.hermesAutoApprove = source.hermesAutoApprove
        result.lastLaunchTarget = source.lastLaunchTarget
        result.lastHermesProfile = source.lastHermesProfile
        result.lastHermesSessionID = source.lastHermesSessionID
        result.lastHermesSessionTitle = source.lastHermesSessionTitle
        result.onboardingCompletedAt = source.onboardingCompletedAt
        result.languagePromptCompletedAt = source.languagePromptCompletedAt
        result.successfulModelLoads = max(target.successfulModelLoads, source.successfulModelLoads)
        result.starPromptCompletedAt = target.starPromptCompletedAt ?? source.starPromptCompletedAt
        result.lastTunedDepth = source.lastTunedDepth
        result.lastTunedAt = source.lastTunedAt
        result.tunedControlRecord = source.tunedControlRecord
        result.tunedControlRecordsByModel = source.tunedControlRecordsByModel
        result.customModels = source.customModels
        result.huggingFaceHandle = source.huggingFaceHandle
        return result
    }
}
