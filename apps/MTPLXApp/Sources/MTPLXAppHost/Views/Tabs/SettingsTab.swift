import Combine
import SwiftUI
import MTPLXAppCore
#if canImport(AppKit)
import AppKit
#endif

// MARK: - SettingsTab
//
// The Settings page: a sidebar of sections (General, Model, Performance,
// Memory, Cache, Clients, Cooling, Advanced) beside one section's rows,
// search across all of them, and one save bar for every pending edit.
// Opened from the gear in the top bar and Cmd-,. The controls, their
// bindings and what they save are the same ones the page always had; the
// catalog (`SettingsCatalog.swift`) says where each one lives.

private struct SettingsBackendSnapshot: Equatable {
    let configuration: MTPLXAppConfiguration
    let configuredModelFamily: String
    let daemonRunning: Bool
    let effectiveFanMode: MTPLXFanMode
    let lastLaunchScheduling: DaemonLaunchScheduling?
    let daemonRestartStatus: DaemonRestartStatus
    let daemonRestartEligibility: DaemonRestartEligibility
    let snapshotIntervalBounds: AppCapabilities.SnapshotInterval?
    let memoryPlan: MemoryPlanStatus?
    let settings: MutableSettings?
    let startupModelControls: ModelControls?

    @MainActor
    init(backend: MTPLXBackendStore, configuredModelFamily: String) {
        configuration = backend.configuration
        self.configuredModelFamily = configuredModelFamily
        daemonRunning = backend.daemonState.kind == .running || backend.daemonState.kind == .warming
        effectiveFanMode = backend.effectiveFanMode
        lastLaunchScheduling = backend.lastLaunchScheduling
        daemonRestartStatus = backend.daemonRestartStatus
        daemonRestartEligibility = backend.daemonRestartEligibility
        snapshotIntervalBounds = backend.capabilities?.snapshotInterval
        memoryPlan = backend.memoryPlan
        settings = backend.settings
        startupModelControls = backend.health?.startup?.modelControls
    }
}

/// Projects `MTPLXBackendStore` down to the low-frequency fields `SettingsTab`
/// actually renders. High-frequency 500ms SSE metrics, thermal RPM polls, and
/// health checks produce equal snapshots and never trigger a Settings re-render
/// or scroll/layout jump.
@MainActor
private final class SettingsBackendProjection: ObservableObject {
    @Published private(set) var snapshot: SettingsBackendSnapshot

    private weak var backend: MTPLXBackendStore?
    private var backendCancellable: AnyCancellable?
    private var refreshPending = false
    private var familyModel: String
    private var configuredModelFamily: String

    init(backend: MTPLXBackendStore) {
        let model = backend.configuration.model
        let family = MTPLXModelOption.modelFamily(for: model)
        self.backend = backend
        familyModel = model
        configuredModelFamily = family
        snapshot = SettingsBackendSnapshot(
            backend: backend,
            configuredModelFamily: family
        )
        backendCancellable = backend.objectWillChange.sink { [weak self] _ in
            self?.scheduleRefresh()
        }
    }

    func refreshNow() {
        refreshPending = false
        guard let backend else { return }
        let model = backend.configuration.model
        if model != familyModel {
            familyModel = model
            configuredModelFamily = MTPLXModelOption.modelFamily(for: model)
        }
        let next = SettingsBackendSnapshot(
            backend: backend,
            configuredModelFamily: configuredModelFamily
        )
        if next != snapshot {
            snapshot = next
        }
    }

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else { return }
            refreshNow()
        }
    }
}

struct SettingsTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    var body: some View {
        SettingsTabBridge(backend: backend).equatable()
    }
}

private struct SettingsTabBridge: View, Equatable {
    let backend: MTPLXBackendStore

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.backend === rhs.backend
    }

    var body: some View {
        SettingsTabContent(backend: backend)
    }
}

private struct SettingsTabContent: View {
    private let backend: MTPLXBackendStore
    @StateObject private var projection: SettingsBackendProjection
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var languageStore: LanguageStore
    @State private var languagePopoverPresented = false

    // Working copy of the persisted configuration. Saved on demand.
    // Live-mutable sampling/depth/reasoning knobs live in the
    // chrome-strip Model controls popover — they aren't duplicated here.
    // The draft is owned above this tab (`SettingsDraftStore`, injected
    // at the app root): the dashboard rebuilds the tab on every
    // selection, so a draft kept in `@State` here died — with every
    // pending edit — the moment the user looked at another tab.
    @EnvironmentObject private var drafts: SettingsDraftStore
    @State private var pendingClearAll = false
    @State private var clearingCache = false
    @State private var searchQuery = ""
    /// A section the content should scroll to (sidebar click while the
    /// search shows several sections).
    @State private var scrollTarget: SettingsSection?
    /// Optimistic Fan Mode selection while a live `/thermal/fan_mode` update is
    /// in flight, preventing a single-frame bounce back to the previous segment.
    @State private var optimisticFanMode: MTPLXFanMode?
    @State private var fanModePending = false

    @EnvironmentObject private var router: AppRouter

    @MainActor
    init(backend: MTPLXBackendStore) {
        self.backend = backend
        _projection = StateObject(
            wrappedValue: SettingsBackendProjection(backend: backend)
        )
    }

    private var persistedConfiguration: MTPLXAppConfiguration {
        projection.snapshot.configuration
    }

    private var draftConfig: MTPLXAppConfiguration {
        get { drafts.draft }
        nonmutating set { drafts.draft = newValue }
    }

    private var isApplying: Bool {
        get { drafts.isApplying }
        nonmutating set { drafts.isApplying = newValue }
    }

    private var lastSaveError: String? {
        get { drafts.lastSaveError }
        nonmutating set { drafts.lastSaveError = newValue }
    }

    private var search: SettingsSearch { SettingsSearch(query: searchQuery) }

    /// Below this width the sidebar becomes a row of chips over the
    /// content, so the rows keep their room on a narrow window.
    private static let sidebarMinimumWidth: CGFloat = 720
    private static let sidebarWidth: CGFloat = 210
    private static let contentMaxWidth: CGFloat = 760

    var body: some View {
        GeometryReader { geo in
            let unsaved = unsavedSections
            Group {
                if geo.size.width < Self.sidebarMinimumWidth {
                    VStack(alignment: .leading, spacing: 10) {
                        SettingsSearchField(query: $searchQuery)
                            .padding(.horizontal, 16)
                            .padding(.top, 14)
                        SettingsSectionChips(
                            selection: $router.settingsSection,
                            search: search,
                            unsaved: unsaved,
                            onSelect: selectSection
                        )
                        contentPane
                    }
                } else {
                    HStack(spacing: 0) {
                        SettingsSidebar(
                            selection: $router.settingsSection,
                            query: $searchQuery,
                            search: search,
                            unsaved: unsaved,
                            onSelect: selectSection
                        )
                        .frame(width: Self.sidebarWidth)
                        .background(Brand.bgInner.opacity(0.55))
                        Rectangle()
                            .fill(Brand.separator)
                            .frame(width: Brand.hairline)
                        contentPane
                    }
                }
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
        }
        .background(Brand.bgOuter)
        .environment(\.settingsSearch, search)
        .onAppear {
            // Pending edits survive a tab switch; only an unedited draft
            // follows the persisted configuration, while immediate/external
            // controls (Fan Mode, Scheduling, live reasoning/sampler) stay
            // aligned so a later Save never reverts them.
            projection.refreshNow()
            let current = backend.configuration
            drafts.adoptIfUnedited(current)
            drafts.syncNonDraftedFields(from: current)
        }
        .onChange(of: projection.snapshot.configuration) { _, newConfiguration in
            drafts.adoptIfUnedited(newConfiguration)
            drafts.syncNonDraftedFields(from: newConfiguration)
        }
        .confirmationDialog(
            tr("Clear all SessionBank entries?"),
            isPresented: $pendingClearAll
        ) {
            Button(tr("Clear All"), role: .destructive) {
                clearingCache = true
                Task {
                    defer { Task { @MainActor in clearingCache = false } }
                    try? await backend.clearCache()
                }
            }
            Button(tr("Cancel"), role: .cancel) {}
        } message: {
            Text(tr("Clears every saved prompt cache. Anything mid-flight keeps running."))
        }
    }

    private func selectSection(_ section: SettingsSection) {
        if search.isActive {
            scrollTarget = section
        } else {
            router.settingsSection = section
        }
    }

    // MARK: - Content

    private var contentPane: some View {
        let showSaveBar = settingsDirty || lastSaveError != nil
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 36) {
                    if search.isActive {
                        let matching = SettingsSection.allCases.filter(search.hasMatches)
                        if matching.isEmpty {
                            noMatches
                        }
                        ForEach(matching) { section in
                            sectionPage(section)
                                .id(section)
                        }
                    } else {
                        sectionPage(router.settingsSection)
                    }
                }
                .padding(.horizontal, 28)
                .padding(.top, 24)
                // Reserve constant bottom room so showing or hiding the
                // floating save bar never changes the scroll viewport height
                // or triggers a layout reflow across the Settings page.
                .padding(.bottom, 104)
                .frame(maxWidth: Self.contentMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .center)
            }
            // A new section starts at its top, not where the last one
            // was scrolled to.
            .id(search.isActive ? "search" : router.settingsSection.rawValue)
            .scrollIndicators(.automatic)
            .onChange(of: scrollTarget) { _, target in
                guard let target else { return }
                withAnimation(.smooth(duration: 0.25)) {
                    proxy.scrollTo(target, anchor: .top)
                }
                scrollTarget = nil
            }
        }
        .overlay(alignment: .bottom) {
            Group {
                if showSaveBar {
                    SettingsSaveBar(
                        running: daemonRunning,
                        applying: isApplying,
                        error: lastSaveError,
                        onSave: { saveAndMaybeRestart(restart: daemonRunning) },
                        onRevert: {
                            lastSaveError = nil
                            revertDraft()
                        }
                    )
                    .frame(maxWidth: Self.contentMaxWidth)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 14)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.smooth(duration: 0.18), value: showSaveBar)
        }
    }

    private var noMatches: some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(Brand.typeTertiary)
            Text(tr("No settings match “%@”.", searchQuery.trimmingCharacters(in: .whitespaces)))
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Brand.typeSecondary)
        }
        .padding(.top, 40)
        .frame(maxWidth: .infinity, alignment: .center)
    }

    @ViewBuilder
    private func sectionPage(_ section: SettingsSection) -> some View {
        VStack(alignment: .leading, spacing: 22) {
            SettingsSectionHeader(section: section) {
                sectionHeaderAccessory(section)
            }
            switch section {
            case .general: generalSection
            case .model: modelSection
            case .performance: performanceSection
            case .memory: memorySection
            case .cache: cacheSection
            case .clients: clientsSection
            case .cooling: coolingSection
            case .advanced: advancedSection
            }
        }
    }

    @ViewBuilder
    private func sectionHeaderAccessory(_ section: SettingsSection) -> some View {
        switch section {
        case .performance:
            // Mode saves the moment it is picked but only reaches a running
            // engine on its next start: say so where the control is.
            if schedulingRestartPending && daemonRunning {
                PillBadge(text: tr("restart to apply"), systemImage: "arrow.clockwise.circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        default:
            EmptyView()
        }
    }

    /// Which sections hold edits that are not saved yet, for the
    /// sidebar's dots. The save bar itself keys off `settingsDirty`, so a
    /// field missing here can never hide an unsaved edit.
    private var unsavedSections: Set<SettingsSection> {
        let draft = normalizedDraftConfigurationForSave(against: persistedConfiguration)
        let saved = normalizedConfigurationForSave(persistedConfiguration)
        guard draft != saved else { return [] }
        var sections: Set<SettingsSection> = []
        if draft.launchDaemonOnOpen != saved.launchDaemonOnOpen
            || draft.automaticDaemonRestart != saved.automaticDaemonRestart {
            sections.insert(.general)
        }
        if draft.model != saved.model
            || draft.profile != saved.profile
            || draft.generationMode != saved.generationMode
            || draft.loadMTP != saved.loadMTP
            || draft.contextWindow != saved.contextWindow
            || draft.primaryModelDirectory != saved.primaryModelDirectory
            || draft.additionalModelDirectories != saved.additionalModelDirectories
            || draft.hfEndpoint != saved.hfEndpoint {
            sections.insert(.model)
        }
        if draft.maxActiveRequests != saved.maxActiveRequests
            || draft.decodeBatchMax != saved.decodeBatchMax
            || draft.batchWaitMs != saved.batchWaitMs
            || draft.experimentalMTPCohorts != saved.experimentalMTPCohorts {
            sections.insert(.performance)
        }
        if draft.memoryLimitGB != saved.memoryLimitGB
            || draft.allowSwap != saved.allowSwap
            || draft.pagedKVQuantization != saved.pagedKVQuantization {
            sections.insert(.memory)
        }
        if draft.ramSessionCachePolicy != saved.ramSessionCachePolicy
            || draft.ramSessionBlockPrefixRestore != saved.ramSessionBlockPrefixRestore
            || draft.ramSessionCacheMaxEntries != saved.ramSessionCacheMaxEntries
            || draft.ramSessionCacheMaxSize != saved.ramSessionCacheMaxSize
            || draft.ramSessionCachePerSessionMaxSize != saved.ramSessionCachePerSessionMaxSize
            || draft.ssdSessionCache != saved.ssdSessionCache
            || draft.ssdSessionCacheDir != saved.ssdSessionCacheDir
            || draft.ssdSessionCacheMaxSize != saved.ssdSessionCacheMaxSize
            || draft.ssdSessionCacheMinPrefixTokens != saved.ssdSessionCacheMinPrefixTokens {
            sections.insert(.cache)
        }
        if draft.host != saved.host
            || draft.port != saved.port
            || draft.apiKey != saved.apiKey
            || draft.hermesWorkspacePath != saved.hermesWorkspacePath {
            sections.insert(.clients)
        }
        if draft.enableThermalPolling != saved.enableThermalPolling {
            sections.insert(.cooling)
        }
        if draft.executablePath != saved.executablePath
            || draft.streamSnapshotIntervalMs != saved.streamSnapshotIntervalMs
            || draft.streamStallDeadlineSeconds != saved.streamStallDeadlineSeconds
            || draft.embeddingModels != saved.embeddingModels
            || draft.rerankerModels != saved.rerankerModels
            || draft.retrievalIdleTimeout != saved.retrievalIdleTimeout
            || draft.retrievalMaxResident != saved.retrievalMaxResident {
            sections.insert(.advanced)
        }
        return sections
    }

    // MARK: - General

    @ViewBuilder
    private var generalSection: some View {
        SettingsGroup(group: .appearance) {
            SettingsRow(item: .appearance) {
                Picker(tr("Appearance"), selection: $themeStore.appearance) {
                    ForEach(AppAppearance.allCases) { option in
                        Text(tr(option.title)).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            SettingsRow(item: .language) {
                languageControl
            }
            SettingsToggleRow(item: .speedRecordSound, isOn: $themeStore.soundEnabled)
            SettingsToggleRow(item: .menuBarItem, isOn: $themeStore.showMenuBarItem)
        }

        SettingsGroup(group: .startup) {
            SettingsToggleRow(item: .launchOnOpen, isOn: $drafts.draft.launchDaemonOnOpen)
            SettingsToggleRow(item: .autoRestart, isOn: $drafts.draft.automaticDaemonRestart)
            if let restartStatusText {
                SettingsInfoRow(item: .autoRestart) {
                    Text(restartStatusText)
                }
            }
        }

        SettingsGroup(group: .info) {
            SettingsLinkRow(item: .about) {
                router.presentAbout()
            }
        }
    }

    /// Same searchable picker as the onboarding step, in a popover so the
    /// row stays one line tall. Picking applies immediately: the store
    /// activates the language and the app shell re-renders.
    private var languageControl: some View {
        let language = languageStore.language
        return HStack(spacing: 10) {
            Text(language.flag)
                .font(.system(size: 18))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(language.nativeName)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                // One line when the two names match ("English", not
                // "English / English").
                if language.englishName != language.nativeName {
                    Text(language.englishName)
                        .font(.system(size: 10))
                        .foregroundStyle(Brand.typeTertiary)
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel(tr("Current language: %@", language.nativeName))
            Button(tr("Change…")) {
                languagePopoverPresented = true
            }
            .buttonStyle(MTPLXGhostButton())
            .popover(isPresented: $languagePopoverPresented, arrowEdge: .bottom) {
                // The popover is its own window on macOS, so it applies the
                // appearance preference itself; without this it follows the
                // macOS setting and renders dark inside a light app.
                LanguagePickerList(onPick: { _ in languagePopoverPresented = false })
                    .frame(width: 320, height: 400)
                    .padding(12)
                    .environmentObject(languageStore)
                    .environmentObject(themeStore)
                    .appliesAppearance()
            }
        }
    }

    // MARK: - Model

    @ViewBuilder
    private var modelSection: some View {
        SettingsGroup(group: .modelIdentity) {
            SettingsRow(item: .model, layout: .stacked) {
                TextField("", text: $drafts.draft.model)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))
            }
            SettingsRow(item: .profile) {
                // Only persistable profiles may appear here; a stray
                // tag value persists into config and kills serve at
                // argparse ("auto" is resolved to a concrete engine
                // profile before launch). Max fans is the Fan mode
                // row, not a profile.
                Picker(tr("Profile"), selection: $drafts.draft.profile) {
                    Text(tr("Auto (recommended)")).tag("auto")
                    Text(tr("Turbo")).tag("turbo")
                    Text(tr("Sustained")).tag("sustained")
                    Text(tr("Performance Cold (Burst)")).tag("performance-cold")
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 220)
            }
            SettingsRow(item: .generationMode) {
                Picker(tr("Mode"), selection: $drafts.draft.generationMode) {
                    Text(tr("MTP")).tag("mtp")
                    Text(tr("Baseline")).tag("ar")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            SettingsToggleRow(item: .loadMTP, isOn: $drafts.draft.loadMTP)
            SettingsRow(item: .contextWindow) {
                HStack(spacing: 8) {
                    // Empty (nil) shows "Auto": the engine serves the
                    // largest window this Mac fits. A zero here read as a
                    // zero-token window.
                    TextField(
                        "",
                        value: contextWindowBinding,
                        format: .number,
                        prompt: Text(tr("Auto"))
                    )
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 110)
                    Text(tr("tokens"))
                        .font(.caption)
                        .foregroundStyle(Brand.typeTertiary)
                    Text(tr("max %@", Self.formatTokens(settingsModelMaxContext)))
                        .font(.caption)
                        .foregroundStyle(Brand.typeTertiary)
                }
                .fixedSize()
            }
        }

        // Model folders. APPFX hands over a self-contained view for this
        // group; until then it holds the folder list the page always had.
        SettingsGroup(group: .modelFolders) {
            SettingsRow(item: .modelFolders, layout: .stacked) {
                modelFoldersContent
            }
        }

        SettingsGroup(group: .downloads) {
            SettingsRow(item: .downloadMirror) {
                // Verbatim prompt: a URL in a localized title renders as a
                // blue link and reads as a value that is already set.
                TextField(
                    "",
                    text: Binding(
                        get: { draftConfig.hfEndpoint ?? "" },
                        set: { draftConfig.hfEndpoint = $0.isEmpty ? nil : $0 }
                    ),
                    prompt: Text(verbatim: "https://hf-mirror.com")
                )
                .textFieldStyle(.roundedBorder)
                .frame(width: 240)
            }
        }
    }

    private var modelFoldersContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            modelDirectoryRow(
                title: tr("Primary"),
                path: draftConfig.primaryModelDirectory
            ) {
                Button(tr("Choose")) { choosePrimaryModelDirectory() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }

            if !draftConfig.additionalModelDirectories.isEmpty {
                Divider().overlay(Brand.separator)
            }

            ForEach(
                Array(draftConfig.additionalModelDirectories.enumerated()),
                id: \.offset
            ) { index, path in
                modelDirectoryRow(title: tr("Additional %lld", index + 1), path: path) {
                    HStack(spacing: 4) {
                        Button {
                            moveAdditionalModelDirectory(from: index, by: -1)
                        } label: {
                            Image(systemName: "chevron.up")
                        }
                        .disabled(index == 0)

                        Button {
                            moveAdditionalModelDirectory(from: index, by: 1)
                        } label: {
                            Image(systemName: "chevron.down")
                        }
                        .disabled(index == draftConfig.additionalModelDirectories.count - 1)

                        Button {
                            draftConfig.additionalModelDirectories.remove(at: index)
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                    }
                    .buttonStyle(.borderless)
                }
            }

            Button {
                addModelDirectories()
            } label: {
                Label(tr("Add folder"), systemImage: "plus")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
    }

    // MARK: - Performance (concurrency + batching)
    //
    // Maps to: schedulerMode (serial|ar_batch|continuous), batchingPreset
    // (latency|throughput|agent), maxActiveRequests, decodeBatchMax,
    // batchWaitMs, experimentalMTPCohorts. These are restart-required so the
    // same Save+Restart button is used. Prefill chunk size lives under Quick
    // settings because the daemon can apply it to the next request.

    @ViewBuilder
    private var performanceSection: some View {
        SettingsGroup(group: .requests) {
            SettingsRow(item: .schedulingMode) {
                Picker(tr("Mode"), selection: schedulerPresetBinding) {
                    Text(tr("Auto")).tag("target-default")
                    Text(tr("Fastest response")).tag("latency")
                    Text(tr("Handle multiple at once")).tag("throughput")
                    Text(tr("Long agent tasks")).tag("agent")
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 220)
            }
            // The picker says what the next launch will use; this says
            // what the daemon in front of you actually launched with.
            // Issue #398 could not tell those two apart at all.
            if daemonRunning, let launched = projection.snapshot.lastLaunchScheduling {
                SettingsInfoRow(item: .schedulingMode) {
                    Text(tr(
                        "Running now: %@ (%@ / %@).",
                        schedulingPresetLabel(launched.selectedPreset),
                        launched.schedulerMode,
                        launched.batchingPreset
                    ))
                    .textSelection(.enabled)
                }
            }
            SettingsRow(item: .concurrencyCap) {
                optionalIntStepper(
                    value: $drafts.draft.maxActiveRequests,
                    defaultValue: defaultMaxActiveRequests,
                    range: 1...16
                )
            }
        }

        SettingsGroup(group: .batching) {
            SettingsRow(item: .decodeBatchMax) {
                optionalIntStepper(
                    value: $drafts.draft.decodeBatchMax,
                    defaultValue: defaultDecodeBatchMax,
                    range: 1...16
                )
            }
            SettingsRow(item: .admissionWindow) {
                optionalDoubleStepper(
                    value: $drafts.draft.batchWaitMs,
                    defaultValue: defaultBatchWaitMs,
                    range: 0...500,
                    step: 10,
                    suffix: " ms",
                    width: 64
                )
            }
            SettingsToggleRow(item: .mtpCohorts, isOn: $drafts.draft.experimentalMTPCohorts)
        }
    }

    // MARK: - Memory (limit + swap, KV quantization)
    //
    // Issue #431: a 128 GB Mac hit "reached memory limit" long before its
    // real ceiling, because the engine plans against 75% of RAM and the app
    // exposed no way past it. The reporter had to launch the bundle by hand
    // with MTPLX_MEMORY_LIMIT_BYTES to get their model served. Issue #427:
    // a 32 GB Mac that used to run 60-70k contexts under 2.9.x was clamped
    // to 40960 by the same fit calculation, and asked for an "I know what
    // I'm doing" escape hatch. Both are engine env-only knobs, so the rows
    // write them into the daemon's launch environment.

    @ViewBuilder
    private var memorySection: some View {
        SettingsGroup(group: .memoryLimits) {
            SettingsRow(item: .memoryLimit) {
                HStack(spacing: 6) {
                    TextField("", text: memoryLimitBinding, prompt: Text(tr("Auto")))
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                        .multilineTextAlignment(.trailing)
                        .frame(width: 80)
                    Text(tr("GB"))
                        .font(.callout)
                        .foregroundStyle(Brand.typeSecondary)
                }
            }
            SettingsToggleRow(item: .allowSwap, isOn: $drafts.draft.allowSwap)
            SettingsInfoRow(item: .memoryLimit) {
                Label(memoryPlanCaption, systemImage: "memorychip")
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        let policy = settingsKVQuantPolicy
        let modes = settingsKVQuantModes(policy)
        let supported = settingsKVQuantSupported(policy)
        SettingsGroup(group: .kvQuantization) {
            SettingsRow(
                item: .kvQuantization,
                caption: supported
                    ? tr("Off is the speed path. q8 saves memory when the selected model supports it; q4 is experimental.")
                    : policy.disabledReason ?? tr("KV quantization is not supported for this model.")
            ) {
                Picker(tr("Quantization"), selection: kvQuantSelectionBinding(policy)) {
                    ForEach(modes, id: \.self) { mode in
                        Text(Self.kvQuantDisplayLabel(mode)).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(!supported)
                .fixedSize()
            }
            if supported {
                SettingsInfoRow(item: .kvQuantization) {
                    Label(kvQuantCaption, systemImage: "memorychip")
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    // MARK: - Cache (RAM SessionBank, SSD bank, reset)

    @ViewBuilder
    private var cacheSection: some View {
        SettingsGroup(group: .ramCache) {
            SettingsRow(item: .ramPolicy) {
                Picker(tr("Allocation policy"), selection: $drafts.draft.ramSessionCachePolicy) {
                    Text(tr("Target default")).tag("target-default")
                    Text(tr("Bounded")).tag("bounded")
                    Text(tr("Minimal")).tag("minimal")
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 180)
            }

            if draftConfig.ramSessionCachePolicy == "minimal" {
                SettingsInfoRow(item: .ramPolicy) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text(tr("Effective cap"))
                            Text(tr("1 entry / 1 GB"))
                                .monospacedDigit()
                        }
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Brand.typeBody)
                        Text(tr("Keeps one tiny RAM entry and disables block-prefix restore."))
                    }
                }
            } else if draftConfig.ramSessionCachePolicy != "target-default" {
                SettingsToggleRow(
                    item: .ramBlockPrefix,
                    isOn: $drafts.draft.ramSessionBlockPrefixRestore
                )
                SettingsRow(item: .ramMaxEntries) {
                    Stepper(
                        value: $drafts.draft.ramSessionCacheMaxEntries,
                        in: 1...64
                    ) {
                        Text("\(draftConfig.ramSessionCacheMaxEntries)")
                            .font(.system(.body, design: .rounded).weight(.semibold))
                            .monospacedDigit()
                            .frame(width: 32, alignment: .trailing)
                    }
                    .fixedSize()
                }
                SettingsRow(item: .ramTotalCap) {
                    cacheSizePicker(
                        selection: $drafts.draft.ramSessionCacheMaxSize,
                        values: ["auto", "1G", "2G", "4G", "8G", "16G", "24G", "32G", "48G"]
                    )
                }
                SettingsRow(item: .ramPerSessionCap) {
                    cacheSizePicker(
                        selection: $drafts.draft.ramSessionCachePerSessionMaxSize,
                        values: ["auto", "1G", "2G", "4G", "8G", "16G", "24G", "32G"]
                    )
                }
            }

            SettingsRAMCacheUsageRow(backend: backend)
        }

        SettingsGroup(group: .ssdCache) {
            SettingsRow(item: .ssdPolicy, caption: ssdPolicyCaption) {
                Picker(tr("Policy"), selection: $drafts.draft.ssdSessionCache) {
                    Text(tr("Target default")).tag("target-default")
                    Text(tr("Off")).tag("off")
                    Text(tr("Read + write")).tag("on")
                    Text(tr("Write-only")).tag("write-only")
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 180)
            }

            if effectiveSSDSessionCache != "off" {
                SettingsRow(item: .ssdFolder, layout: .stacked) {
                    HStack(spacing: 6) {
                        TextField("",
                                  text: Binding(
                                    get: { draftConfig.ssdSessionCacheDir ?? "" },
                                    set: { draftConfig.ssdSessionCacheDir = $0.isEmpty ? nil : $0 }
                                  ))
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.caption, design: .monospaced))
                        if let dir = draftConfig.ssdSessionCacheDir, !dir.isEmpty {
                            Button {
                                NSWorkspace.shared.open(URL(fileURLWithPath: dir))
                            } label: { Image(systemName: "folder") }
                                .buttonStyle(.borderless)
                                .help(tr("Open folder in Finder"))
                                .accessibilityLabel(tr("Reveal in Finder"))
                        }
                    }
                }
                SettingsRow(item: .ssdMaxSize) {
                    Picker(tr("Max size"), selection: $drafts.draft.ssdSessionCacheMaxSize) {
                        Text(tr("Auto")).tag("auto")
                        Text(tr("10 GB")).tag("10GB")
                        Text(tr("50 GB")).tag("50GB")
                        Text(tr("100 GB")).tag("100GB")
                        Text(tr("250 GB")).tag("250GB")
                        Text(tr("500 GB")).tag("500GB")
                        Text(tr("1 TB")).tag("1TB")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(width: 140)
                }
                SettingsRow(item: .ssdMinPrompt) {
                    Stepper(
                        value: $drafts.draft.ssdSessionCacheMinPrefixTokens,
                        in: 128...8192,
                        step: 128
                    ) {
                        Text(tr("%lld tok", draftConfig.ssdSessionCacheMinPrefixTokens))
                            .font(.system(.body, design: .rounded).weight(.semibold))
                            .monospacedDigit()
                            .frame(width: 96, alignment: .trailing)
                    }
                    .fixedSize()
                }
                SettingsSSDCacheUsageRow(backend: backend)
            }
        }

        SettingsGroup(
            group: .cacheReset,
            footer: tr("To stop or restart the model, use the play button in the top bar.")
        ) {
            SettingsRow(item: .clearCache) {
                Button(role: .destructive) {
                    pendingClearAll = true
                } label: {
                    HStack(spacing: 6) {
                        if clearingCache {
                            ProgressView().controlSize(.mini)
                        } else {
                            Image(systemName: "trash")
                        }
                        Text(tr("Clear All"))
                    }
                }
                .buttonStyle(.bordered)
                .tint(Brand.danger)
                .disabled(clearingCache)
            }
        }
    }

    // MARK: - Clients

    @ViewBuilder
    private var clientsSection: some View {
        SettingsGroup(group: .connection) {
            SettingsRow(item: .hostPort) {
                HStack(spacing: 6) {
                    TextField("127.0.0.1", text: $drafts.draft.host)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 150)
                    Text(verbatim: ":")
                        .foregroundStyle(Brand.typeTertiary)
                    TextField(
                        "8000",
                        value: $drafts.draft.port,
                        format: .number.grouping(.never)
                    )
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 72)
                }
            }
            if !MTPLXServerURLs.isLoopbackBind(draftConfig.host) {
                SettingsInfoRow(item: .hostPort) {
                    Label(
                        MTPLXServerURLs.isWildcardBind(draftConfig.host)
                            ? tr("Serves on every interface (LAN). An API key is required.")
                            : tr("Non-localhost host: an API key is required."),
                        systemImage: "exclamationmark.triangle"
                    )
                    .foregroundStyle(
                        (draftConfig.apiKey ?? "").isEmpty
                            ? Brand.warning
                            : Brand.typeTertiary
                    )
                }
            }
            SettingsRow(item: .apiKey) {
                SecureField(
                    "none",
                    text: Binding(
                        get: { draftConfig.apiKey ?? "" },
                        set: { draftConfig.apiKey = $0.isEmpty ? nil : $0 }
                    )
                )
                .textFieldStyle(.roundedBorder)
                .frame(width: 240)
            }
        }

        SettingsGroup(
            group: .openIn,
            footer: tr("Shared with mtplx start: `mtplx clients` lists them, `mtplx config set pi_opens_in Ghostty` changes one.")
        ) {
            ClientOpeningRows()
        }

        SettingsHermesGroup(
            backend: backend,
            hermesWorkspacePath: $drafts.draft.hermesWorkspacePath,
            onChooseWorkspace: chooseHermesWorkspace
        )
    }

    // MARK: - Cooling

    @ViewBuilder
    private var coolingSection: some View {
        SettingsGroup(group: .fans) {
            SettingsRow(item: .fanMode) {
                FanModeToggle(
                    activeMode: activeFanMode,
                    pending: fanModePending,
                    onSelect: commitFanMode
                )
            }
            SettingsInfoRow(item: .fanMode) {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 4) {
                    ForEach(MTPLXFanMode.allCases, id: \.self) { mode in
                        let selected = mode == activeFanMode
                        GridRow {
                            Text(mode.title)
                                .font(.system(size: 11, weight: selected ? .semibold : .medium))
                                .foregroundStyle(selected ? Brand.typeHi : Brand.typeTertiary)
                            Text(mode.help)
                                .foregroundStyle(selected ? Brand.typeBody : Brand.typeTertiary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
            SettingsToggleRow(item: .thermalPolling, isOn: $drafts.draft.enableThermalPolling)
            SettingsCoolingFanStatusRow(
                backend: backend,
                thermalPollingEnabled: persistedConfiguration.enableThermalPolling
            )
        }
    }

    // MARK: - Advanced

    @ViewBuilder
    private var advancedSection: some View {
        SettingsGroup(group: .engine) {
            SettingsRow(item: .executablePath, layout: .stacked) {
                TextField(
                    "mtplx (in PATH)",
                    text: Binding(
                        get: { draftConfig.executablePath ?? "" },
                        set: { draftConfig.executablePath = $0.isEmpty ? nil : $0 }
                    )
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(.callout, design: .monospaced))
            }
            streamCadenceRow
            SettingsRow(item: .stallWatchdog) {
                TextField(
                    "300",
                    value: Binding(
                        get: { draftConfig.streamStallDeadlineSeconds },
                        set: { draftConfig.streamStallDeadlineSeconds = max(0, $0) }
                    ),
                    format: .number.precision(.fractionLength(0))
                )
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(width: 80)
            }
        }

        retrievalGroup

        SettingsGroup(group: .diagnostics) {
            SettingsLinkRow(item: .logs, trailingHint: "⇧⌘L") {
                router.presentLogs()
            }
            SettingsRow(item: .settingsFile, caption: settingsFilePathHint) {
                Button(tr("Reveal in Finder")) {
                    NSWorkspace.shared.activateFileViewerSelecting([backend.settingsURL])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    @ViewBuilder
    private var streamCadenceRow: some View {
        let bounds = projection.snapshot.snapshotIntervalBounds
        let minMs = bounds?.minMs ?? 100
        let maxMs = bounds?.maxMs ?? 5000
        let perfLockMs = bounds?.performanceLockMs ?? 1000
        SettingsRow(
            item: .streamCadence,
            caption: tr("Performance Lock overrides this to %lld ms.", perfLockMs)
        ) {
            Stepper(
                value: $drafts.draft.streamSnapshotIntervalMs,
                in: minMs...maxMs,
                step: 50
            ) {
                Text(tr("%lld ms", draftConfig.streamSnapshotIntervalMs))
                    .font(.system(.body, design: .rounded).weight(.medium))
                    .monospacedDigit()
                    .frame(minWidth: 72, alignment: .trailing)
            }
            .fixedSize()
        }
    }

    // MARK: - Retrieval (embeddings + reranking)
    //
    // Maps to: embeddingModels, rerankerModels, retrievalMaxResident. Purely
    // additive — with nothing configured the endpoints answer 404 and the chat
    // runtime is untouched. Restart-required, because the served set is built
    // when the daemon starts.

    @ViewBuilder
    private var retrievalGroup: some View {
        SettingsGroup(
            group: .retrieval,
            footer: tr("Serve embeddings and reranking from this daemon. Leave empty to keep MTPLX chat-only.")
        ) {
            SettingsRow(item: .embeddingModels, layout: .stacked) {
                TextField(
                    "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
                    text: modelListBinding(\.embeddingModels),
                    axis: .vertical
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(2...6)
            }
            SettingsRow(item: .rerankerModels, layout: .stacked) {
                TextField(
                    "vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
                    text: modelListBinding(\.rerankerModels),
                    axis: .vertical
                )
                .textFieldStyle(.roundedBorder)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(2...6)
            }
            SettingsRow(item: .retrievalIdle) {
                TextField(
                    "0",
                    value: Binding(
                        get: { draftConfig.retrievalIdleTimeout / 60 },
                        set: { draftConfig.retrievalIdleTimeout = max(0, $0) * 60 }
                    ),
                    format: .number.precision(.fractionLength(0))
                )
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(width: 80)
            }
            SettingsRow(item: .retrievalResident) {
                Stepper(
                    value: $drafts.draft.retrievalMaxResident,
                    in: 1...8
                ) {
                    Text("\(draftConfig.retrievalMaxResident)")
                        .font(.system(.callout, design: .monospaced))
                        .frame(width: 24, alignment: .trailing)
                }
                .fixedSize()
            }
        }
    }
    // MARK: - Helpers (bindings, defaults, formatting)

    private func revertDraft() {
        drafts.reset(to: backend.configuration)
    }

    /// Mode saves the moment it is picked (issue #398). It was first made
    /// to do so because the tab's draft died on every tab switch and a
    /// draft-only Mode was silently discarded by the very navigation the
    /// user needs to make to restart the model. The draft now survives tab
    /// switches (`SettingsDraftStore`), but Mode stays save-on-pick on
    /// purpose: it is one discrete global, like Appearance and Language,
    /// which also commit on selection, and the card's caption promises it.
    private var schedulerPresetBinding: Binding<String> {
        Binding(
            get: {
                normalizedSchedulingPreset(draftConfig.schedulingPreset)
            },
            set: { preset in
                commitSchedulingPreset(normalizedSchedulingPreset(preset))
            }
        )
    }

    private func commitSchedulingPreset(_ preset: String) {
        // Applied to both halves of the draft store so the pick never
        // reads as an unsaved edit: it is persisted right below.
        drafts.draft.applySchedulingPreset(preset)
        drafts.lastSynced.applySchedulingPreset(preset)
        do {
            try backend.applySchedulingPresetSelection(preset)
            projection.refreshNow()
            lastSaveError = nil
        } catch {
            lastSaveError = tr("Apply failed: %@", String(describing: error))
        }
    }

    private func normalizedSchedulingPreset(_ raw: String) -> String {
        MTPLXAppConfiguration.schedulingPresetSelection(raw)
    }

    private func schedulingPresetLabel(_ raw: String) -> String {
        switch normalizedSchedulingPreset(raw) {
        case "latency":
            return tr("Fastest response")
        case "throughput":
            return tr("Handle multiple at once")
        case "agent":
            return tr("Long agent tasks")
        default:
            return tr("Auto")
        }
    }

    /// True when the running daemon launched under a different Mode than
    /// the one now saved. This is what makes "restart to apply" honest for
    /// a control that no longer goes through the draft/dirty path.
    private var schedulingRestartPending: Bool {
        guard let launched = projection.snapshot.lastLaunchScheduling else { return false }
        return launched.selectedPreset != normalizedSchedulingPreset(draftConfig.schedulingPreset)
    }

    private var defaultMaxActiveRequests: Int {
        switch normalizedSchedulingPreset(draftConfig.schedulingPreset) {
        case "latency":
            return 1
        case "throughput":
            return 8
        case "agent":
            return 4
        default:
            return launchTargetDefaultMaxActiveRequests
        }
    }

    private var defaultDecodeBatchMax: Int {
        switch normalizedSchedulingPreset(draftConfig.schedulingPreset) {
        case "latency":
            return 1
        case "throughput":
            return 8
        case "agent":
            return 4
        default:
            return launchTargetDefaultDecodeBatchMax
        }
    }

    private var defaultBatchWaitMs: Double {
        switch normalizedSchedulingPreset(draftConfig.schedulingPreset) {
        case "latency":
            return 0
        case "throughput":
            return 20
        case "agent":
            return 50
        default:
            return launchTargetDefaultBatchWaitMs
        }
    }

    private var draftLaunchTarget: LaunchTarget? {
        LaunchTarget(rawValue: draftConfig.lastLaunchTarget)
    }

    private var launchTargetDefaultMaxActiveRequests: Int {
        switch draftLaunchTarget {
        case .chat, .pi:
            return 2
        case .other:
            return 4
        case .openCode, .hermes, .claudeCode, .codex, .deepseekHarness:
            return 1
        case .openWebUI, .benchmark, nil:
            // Benchmark is single-stream like Open WebUI: one math run
            // owns the decode lane, with no agent batching.
            return 1
        }
    }

    private var launchTargetDefaultDecodeBatchMax: Int {
        switch draftLaunchTarget {
        case .chat, .pi:
            return 2
        case .other:
            return 4
        case .openCode, .hermes, .claudeCode, .codex, .deepseekHarness:
            return 1
        case .openWebUI, .benchmark, nil:
            return 1
        }
    }

    private var launchTargetDefaultBatchWaitMs: Double {
        switch draftLaunchTarget {
        case .chat, .pi, .other:
            return 50
        case .openCode, .hermes, .claudeCode, .codex, .deepseekHarness:
            return 0
        case .openWebUI, .benchmark, nil:
            return 0
        }
    }

    private func optionalIntStepper(
        value: Binding<Int?>,
        defaultValue: Int,
        range: ClosedRange<Int>
    ) -> some View {
        HStack(spacing: 8) {
            if value.wrappedValue == nil {
                Text(tr("Preset default"))
                    .font(.system(.body, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeSecondary)
                Button(tr("Override")) {
                    value.wrappedValue = defaultValue
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Stepper(
                    value: Binding(
                        get: { value.wrappedValue ?? defaultValue },
                        set: { value.wrappedValue = $0 }
                    ),
                    in: range
                ) {
                    Text("\(value.wrappedValue ?? defaultValue)")
                        .font(.system(.body, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                        .frame(width: 32, alignment: .leading)
                }
                Button {
                    value.wrappedValue = nil
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                }
                .buttonStyle(.borderless)
                .help(tr("Use the preset default"))
                .accessibilityLabel(tr("Reset to preset default"))
            }
        }
    }

    private func optionalDoubleStepper(
        value: Binding<Double?>,
        defaultValue: Double,
        range: ClosedRange<Double>,
        step: Double,
        suffix: String,
        width: CGFloat
    ) -> some View {
        HStack(spacing: 8) {
            if value.wrappedValue == nil {
                Text(tr("Preset default"))
                    .font(.system(.body, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeSecondary)
                Button(tr("Override")) {
                    value.wrappedValue = defaultValue
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            } else {
                Stepper(
                    value: Binding(
                        get: { value.wrappedValue ?? defaultValue },
                        set: { value.wrappedValue = $0 }
                    ),
                    in: range,
                    step: step
                ) {
                    Text("\(Int(value.wrappedValue ?? defaultValue))\(suffix)")
                        .font(.system(.body, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                        .frame(width: width, alignment: .leading)
                }
                Button {
                    value.wrappedValue = nil
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                }
                .buttonStyle(.borderless)
                .help(tr("Use the preset default"))
                .accessibilityLabel(tr("Reset to preset default"))
            }
        }
    }

    private var memoryLimitBinding: Binding<String> {
        Binding(
            get: { draftConfig.memoryLimitGB.map(String.init) ?? "" },
            set: { raw in
                let digits = raw.filter(\.isNumber)
                draftConfig.memoryLimitGB = MTPLXAppConfiguration.normalizedMemoryLimitGB(
                    Int(digits)
                )
            }
        )
    }

    /// What the running engine actually planned, so the number in the field
    /// can be compared against a measured effect instead of a guess.
    private var memoryPlanCaption: String {
        guard let plan = projection.snapshot.memoryPlan, plan.available else {
            return tr("Start the model to see the plan the engine computes for this Mac.")
        }
        let ram = Format.bytes(plan.totalRamBytes)
        let usable = Format.bytes(plan.usableBytes)
        guard let window = plan.contextWindowResolved, window > 0 else {
            return tr("Engine plan: %@ usable of %@ memory.", usable, ram)
        }
        return tr(
            "Engine plan: %@ usable of %@ memory, context window %lld tokens.",
            usable,
            ram,
            window
        )
    }

    private var kvQuantCaption: String {
        let policy = settingsKVQuantPolicy
        guard settingsKVQuantSupported(policy) else {
            return policy.disabledReason ?? tr("KV quantization is not supported for this model.")
        }
        switch kvQuantSelectionBinding(policy).wrappedValue {
        case "q8":
            return tr("Stores KV in int8 with per-token scales. Use it when memory matters more than 20k decode speed.")
        case "q4":
            return tr("Packs KV into 4-bit nibbles with per-token scales; keep as an experiment until measured.")
        default:
            return tr("Uses the model's native KV dtype.")
        }
    }

    private func cacheSizePicker(selection: Binding<String>, values: [String]) -> some View {
        Picker(tr("Cache size"), selection: selection) {
            ForEach(values, id: \.self) { value in
                Text(Self.cacheSizeDisplayLabel(value)).tag(value)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .frame(maxWidth: 140, alignment: .leading)
    }

    static func cacheSizeDisplayLabel(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        if trimmed.lowercased() == "auto" {
            return tr("Auto")
        }
        for (suffix, unit) in [("GB", "GB"), ("TB", "TB"), ("G", "GB"), ("T", "TB")] {
            if trimmed.uppercased().hasSuffix(suffix) {
                let number = trimmed.dropLast(suffix.count)
                    .trimmingCharacters(in: .whitespaces)
                return "\(number) \(unit)"
            }
        }
        return trimmed
    }

    private var ssdPolicyCaption: String {
        let target = draftLaunchTarget?.title ?? tr("this target")
        let targetDefault = defaultSSDSessionCache(for: LaunchTarget(rawValue: draftConfig.lastLaunchTarget))
        let targetDefaultLabel = targetDefault == "off"
            ? "off"
            : (targetDefault == "write-only" ? "write-only" : "read + write")
        return tr("Target default is %@ for %@. Explicit choices override every launch target.", targetDefaultLabel, target)
    }

    private var daemonRunning: Bool {
        projection.snapshot.daemonRunning
    }

    private var effectiveSSDSessionCache: String {
        let normalized = draftConfig.ssdSessionCache.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalized {
        case "target-default", "default", "":
            return defaultSSDSessionCache(for: LaunchTarget(rawValue: draftConfig.lastLaunchTarget))
        case "off", "on", "write-only":
            return normalized
        default:
            return defaultSSDSessionCache(for: LaunchTarget(rawValue: draftConfig.lastLaunchTarget))
        }
    }

    private func defaultSSDSessionCache(for target: LaunchTarget?) -> String {
        switch target {
        case .openCode, .hermes, .claudeCode, .codex, .deepseekHarness, .other:
            return "on"
        default:
            return "off"
        }
    }

    /// Bridge the stored string list to a multi-line text field.
    ///
    /// Splitting on newlines and commas means a pasted `a, b` works as well as
    /// one-per-line, and blank lines are dropped rather than becoming empty
    /// model references.
    private func modelListBinding(
        _ keyPath: WritableKeyPath<MTPLXAppConfiguration, [String]>
    ) -> Binding<String> {
        Binding(
            get: { draftConfig[keyPath: keyPath].joined(separator: "\n") },
            set: { newValue in
                draftConfig[keyPath: keyPath] = newValue
                    .split(whereSeparator: { $0 == "\n" || $0 == "," })
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            }
        )
    }

    private func modelDirectoryRow<Trailing: View>(
        title: String,
        path: String,
        @ViewBuilder trailing: () -> Trailing
    ) -> some View {
        let url = ModelLibrary.canonicalURL(for: path)
        let available = ModelLibrary.isAvailable(url)
        return HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Text(title).font(.callout.weight(.medium))
                    Label(
                        available ? tr("Available") : tr("Unavailable"),
                        systemImage: available ? "checkmark.circle.fill" : "externaldrive.badge.questionmark"
                    )
                    .font(.caption)
                    .foregroundStyle(available ? Brand.success : Brand.warning)
                }
                Text(url.path)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 8)
            trailing()
        }
    }

    private func choosePrimaryModelDirectory() {
        #if canImport(AppKit)
        let panel = modelDirectoryPanel(allowsMultipleSelection: false)
        let current = draftConfig.modelLibrary.primaryDirectory
        if ModelLibrary.isAvailable(current) {
            panel.directoryURL = current
        }
        if panel.runModal() == .OK, let url = panel.url {
            draftConfig.setPrimaryModelDirectory(url.path, preservePrevious: true)
        }
        #endif
    }

    private func addModelDirectories() {
        #if canImport(AppKit)
        let panel = modelDirectoryPanel(allowsMultipleSelection: true)
        if panel.runModal() == .OK {
            draftConfig.addModelDirectories(panel.urls.map(\.path))
        }
        #endif
    }

    #if canImport(AppKit)
    private func modelDirectoryPanel(allowsMultipleSelection: Bool) -> NSOpenPanel {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = allowsMultipleSelection
        panel.canCreateDirectories = true
        panel.prompt = allowsMultipleSelection ? tr("Add") : tr("Use")
        panel.message = allowsMultipleSelection
            ? tr("Choose one or more model folders to search.")
            : tr("Choose the folder for downloads and Forge output.")
        return panel
    }
    #endif

    private func moveAdditionalModelDirectory(from index: Int, by offset: Int) {
        let destination = index + offset
        guard draftConfig.additionalModelDirectories.indices.contains(index),
              draftConfig.additionalModelDirectories.indices.contains(destination)
        else { return }
        draftConfig.additionalModelDirectories.swapAt(index, destination)
    }

    private var settingsFilePathHint: String {
        backend.settingsURL.path
    }

    private var restartStatusText: String? {
        switch projection.snapshot.daemonRestartStatus {
        case .idle:
            guard draftConfig.automaticDaemonRestart else { return nil }
            switch projection.snapshot.daemonRestartEligibility {
            case .adoptedPriorSession:
                return tr("This adopted prior-session daemon is not protected. Start a fresh daemon.")
            case .currentSessionUnprotected:
                return tr("This daemon was launched before restart protection was enabled. Start a fresh daemon.")
            case .noDaemon, .currentSessionProtected:
                return nil
            }
        case .scheduled(let attempt, let delay):
            return tr("Restart %lld scheduled in %.1fs.", attempt, delay)
        case .restarting(let attempt):
            return tr("Restarting MTPLX (attempt %lld).", attempt)
        case .runningAfterRestart(let attempt):
            return tr("Recovered automatically on restart %lld.", attempt)
        case .exhausted(let attempts, _):
            return tr("Automatic recovery stopped after %lld attempts.", attempts)
        }
    }

    private func chooseHermesWorkspace() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = tr("Use")
        panel.message = tr("Choose the project folder Hermes should use for terminal and file tools.")
        let current = MTPLXAppConfiguration.normalizedHermesWorkspacePath(
            draftConfig.hermesWorkspacePath
        )
        if FileManager.default.fileExists(atPath: current) {
            panel.directoryURL = URL(fileURLWithPath: current, isDirectory: true)
        }
        if panel.runModal() == .OK, let url = panel.url {
            draftConfig.hermesWorkspacePath = url.path
        }
        #endif
    }

    private func saveAndMaybeRestart(restart: Bool) {
        isApplying = true
        lastSaveError = nil
        let persisted = backend.configuration
        var config = normalizedDraftConfigurationForSave(against: persisted)
        if config.model != persisted.model {
            ModelPickerOverlay.normalizeModelScopedDefaults(&config, previousModel: persisted.model)
        }
        // The draft store outlives this view, so a save that finishes
        // after the user has moved to another tab still lands.
        let drafts = self.drafts
        let projection = self.projection
        Task {
            do {
                try await backend.applyConfiguration(config, restartIfRunning: restart)
                projection.refreshNow()
                drafts.reset(to: backend.configuration)
            } catch {
                drafts.lastSaveError = tr("Apply failed: %@", String(describing: error))
            }
            drafts.isApplying = false
        }
    }

    private var settingsDirty: Bool {
        normalizedDraftConfigurationForSave(against: persistedConfiguration)
            != normalizedConfigurationForSave(persistedConfiguration)
    }

    private var settingsModelFamily: String {
        if draftConfig.model == persistedConfiguration.model {
            return projection.snapshot.configuredModelFamily
        }
        return MTPLXModelOption.modelFamily(for: draftConfig.model)
    }

    private var compatibleSettings: MutableSettings? {
        guard let settings = projection.snapshot.settings else { return nil }
        let settingsFamily = settings.modelControls?.modelFamily ?? settings.modelFamily
        guard let settingsFamily else {
            return MTPLXModelOption.supportsTune(family: settingsModelFamily) ? settings : nil
        }
        return settingsFamily == settingsModelFamily ? settings : nil
    }

    private var compatibleStartupControls: ModelControls? {
        guard let controls = projection.snapshot.startupModelControls else { return nil }
        if let modelRef = controls.modelRef {
            return MTPLXModelOption.modelsMatch(modelRef, draftConfig.model)
                ? controls
                : nil
        }
        return controls.modelFamily == settingsModelFamily ? controls : nil
    }

    private var settingsModelControls: ModelControls? {
        compatibleSettings?.modelControls ?? compatibleStartupControls
    }

    private var settingsKVQuantPolicy: KVQuantPolicy {
        settingsModelControls?.kvQuant
            ?? compatibleSettings?.kvQuantPolicy
            ?? fallbackKVQuantPolicy(for: settingsModelFamily)
    }

    private func fallbackKVQuantPolicy(for family: String) -> KVQuantPolicy {
        switch family {
        case "qwen3_5", "qwen3_6", "qwen3_8":
            return KVQuantPolicy(
                supported: true,
                modes: ["off", "q8", "q4"],
                restartRequired: true,
                proofLevel: "qwen_only"
            )
        case "gemma4":
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: tr("KV quantization is not supported for Gemma.")
            )
        case "step":
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: tr("KV quantization is not supported for Step.")
            )
        case "qwen4_exp":
            // Mirrors QWEN4_EXP_KV_QUANT_POLICY in backends/descriptors.py:
            // the paged KV-quant lane never converts this family's QSA
            // caches, and the hybrid design keeps KV small by construction.
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: tr("Flash-Next keeps KV on 12 of 48 layers (~24 KB/token), and its QSA attention has no validated quantized-cache lane yet.")
            )
        default:
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: tr("KV quantization is not supported for this model.")
            )
        }
    }

    private func settingsKVQuantSupported(_ policy: KVQuantPolicy) -> Bool {
        policy.supported && policy.modes.contains { $0 != "off" }
    }

    private func settingsKVQuantModes(_ policy: KVQuantPolicy) -> [String] {
        let normalized = policy.modes
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
        let modes = normalized.isEmpty ? ["off"] : normalized
        return modes.contains("off") ? modes : ["off"] + modes
    }

    private func kvQuantSelectionBinding(_ policy: KVQuantPolicy) -> Binding<String> {
        let modes = settingsKVQuantModes(policy)
        let supported = settingsKVQuantSupported(policy)
        return Binding(
            get: {
                let value = draftConfig.pagedKVQuantization
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return supported && modes.contains(value) ? value : "off"
            },
            set: { value in
                let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                draftConfig.pagedKVQuantization = supported && modes.contains(normalized)
                    ? normalized
                    : "off"
            }
        )
    }

    private static func kvQuantDisplayLabel(_ mode: String) -> String {
        mode == "off" ? tr("Off") : mode
    }

    private var settingsModelMaxContext: Int {
        MTPLXModelOption.maxContextWindow(forFamily: settingsModelFamily)
    }

    private var compatibleDraftContextWindow: Int? {
        guard let value = draftConfig.contextWindow, value > 0 else { return nil }
        if let family = draftConfig.contextWindowModelFamily {
            return family == settingsModelFamily ? value : nil
        }
        return MTPLXModelOption.supportsTune(family: settingsModelFamily) ? value : nil
    }

    private var contextWindowBinding: Binding<Int?> {
        Binding(
            get: { compatibleDraftContextWindow },
            set: { value in
                guard let raw = value, raw > 0 else {
                    draftConfig.contextWindow = nil
                    draftConfig.contextWindowModelFamily = nil
                    return
                }
                draftConfig.contextWindow = Self.clampContextWindow(
                    raw,
                    maximum: settingsModelMaxContext
                )
                draftConfig.contextWindowModelFamily = settingsModelFamily
            }
        )
    }

    private func normalizedDraftConfigurationForSave(
        against persisted: MTPLXAppConfiguration
    ) -> MTPLXAppConfiguration {
        normalizedConfigurationForSave(drafts.mergedForSave(against: persisted))
    }

    private func normalizedConfigurationForSave(_ source: MTPLXAppConfiguration) -> MTPLXAppConfiguration {
        var config = source
        let normalizedFanMode = MTPLXFanMode.normalized(config.fanMode)
        config.fanMode = normalizedFanMode.rawValue
        config.pinFansAtMaxOnStart = normalizedFanMode == .max
        let family = source.model == persistedConfiguration.model
            ? projection.snapshot.configuredModelFamily
            : MTPLXModelOption.modelFamily(for: source.model)
        if let value = compatibleContextWindow(in: source, family: family) {
            config.contextWindow = Self.clampContextWindow(
                value,
                maximum: MTPLXModelOption.maxContextWindow(forFamily: family)
            )
            config.contextWindowModelFamily = family
        } else if config.contextWindow != nil {
            config.contextWindow = nil
            config.contextWindowModelFamily = nil
        }
        let kvPolicy = source.model == draftConfig.model
            ? settingsKVQuantPolicy
            : fallbackKVQuantPolicy(for: family)
        let kvModes = settingsKVQuantModes(kvPolicy)
        let kvValue = config.pagedKVQuantization
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if !settingsKVQuantSupported(kvPolicy) || !kvModes.contains(kvValue) {
            config.pagedKVQuantization = "off"
        } else {
            config.pagedKVQuantization = kvValue
        }
        config.normalizeModelDirectories()
        return config
    }

    private func compatibleContextWindow(in config: MTPLXAppConfiguration, family: String) -> Int? {
        guard let value = config.contextWindow, value > 0 else { return nil }
        if let storedFamily = config.contextWindowModelFamily {
            return storedFamily == family ? value : nil
        }
        return MTPLXModelOption.supportsTune(family: family) ? value : nil
    }

    private static func clampContextWindow(_ value: Int, maximum: Int) -> Int {
        let snapped = Int((Double(value) / 1024.0).rounded()) * 1024
        return max(4_096, min(maximum, snapped))
    }

    private static func formatTokens(_ value: Int) -> String {
        if value >= 1_000_000 {
            return tr("%.1fM", Double(value) / 1_000_000.0)
        }
        if value >= 1_000 {
            return "\(value / 1_000)K"
        }
        return "\(value)"
    }

    // Fan mode: single authoritative immediate-apply control (stopped or live).

    private var activeFanMode: MTPLXFanMode {
        optimisticFanMode ?? projection.snapshot.effectiveFanMode
    }

    private func commitFanMode(_ mode: MTPLXFanMode) {
        guard !fanModePending else { return }
        optimisticFanMode = mode
        fanModePending = true
        drafts.commitImmediateFanMode(mode)
        let drafts = self.drafts
        let projection = self.projection
        Task { @MainActor in
            do {
                try await backend.setFanMode(mode.rawValue)
                projection.refreshNow()
                drafts.commitImmediateFanMode(backend.effectiveFanMode)
                drafts.lastSaveError = nil
            } catch {
                projection.refreshNow()
                drafts.commitImmediateFanMode(backend.effectiveFanMode)
                drafts.lastSaveError = tr("Apply failed: %@", String(describing: error))
            }
            optimisticFanMode = nil
            fanModePending = false
        }
    }
}

// MARK: - Isolated live-telemetry rows for Settings
//
// Each row subscribes to a tiny Equatable slice of `MTPLXBackendStore` so
// 500ms SSE snapshots and 2s thermal polls only update that one info row
// when its displayed values actually change, never the whole Settings page.

private struct SettingsRAMCacheSummary: Equatable {
    let totalNbytes: Int?
    let entries: Int
    let maxBytes: Int?

    @MainActor
    init?(backend: MTPLXBackendStore) {
        guard let bank = backend.snapshot?.sessionBank else { return nil }
        totalNbytes = bank.totalNbytes
        entries = bank.entries ?? bank.prefixes?.count ?? 0
        maxBytes = bank.maxBytes
    }
}

@MainActor
private final class SettingsRAMCacheProjection: ObservableObject {
    @Published private(set) var summary: SettingsRAMCacheSummary?

    private weak var backend: MTPLXBackendStore?
    private var cancellable: AnyCancellable?
    private var refreshPending = false

    init(backend: MTPLXBackendStore) {
        self.backend = backend
        summary = SettingsRAMCacheSummary(backend: backend)
        cancellable = backend.objectWillChange.sink { [weak self] _ in
            self?.scheduleRefresh()
        }
    }

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else { return }
            refreshPending = false
            guard let backend else { return }
            let next = SettingsRAMCacheSummary(backend: backend)
            if next != summary {
                summary = next
            }
        }
    }
}

private struct SettingsRAMCacheUsageRow: View {
    @StateObject private var projection: SettingsRAMCacheProjection

    @MainActor
    init(backend: MTPLXBackendStore) {
        _projection = StateObject(
            wrappedValue: SettingsRAMCacheProjection(backend: backend)
        )
    }

    var body: some View {
        if let summary = projection.summary {
            SettingsInfoRow(item: .ramPolicy) {
                HStack(spacing: 12) {
                    Image(systemName: "memorychip")
                        .foregroundStyle(Brand.typeSecondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(tr("Current RAM cache"))
                            .font(.caption2)
                            .foregroundStyle(Brand.typeTertiary)
                        HStack(spacing: 6) {
                            Text(Format.bytes(summary.totalNbytes))
                                .font(.system(.callout, design: .rounded).weight(.semibold))
                                .monospacedDigit()
                                .foregroundStyle(Brand.typeBody)
                            Text(summary.entries == 1 ? tr("· 1 entry") : tr("· %lld entries", summary.entries))
                                .font(.caption)
                                .foregroundStyle(Brand.typeSecondary)
                            if let maxBytes = summary.maxBytes {
                                Text(tr("· cap %@", Format.bytes(maxBytes)))
                                    .font(.caption)
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                        }
                    }
                    Spacer()
                }
                .padding(.top, 6)
            }
        }
    }
}

@MainActor
private final class SettingsSSDCacheProjection: ObservableObject {
    @Published private(set) var coldTier: SessionBankColdTier?

    private weak var backend: MTPLXBackendStore?
    private var cancellable: AnyCancellable?
    private var refreshPending = false

    init(backend: MTPLXBackendStore) {
        self.backend = backend
        coldTier = backend.snapshot?.sessionBank.coldTier
        cancellable = backend.objectWillChange.sink { [weak self] _ in
            self?.scheduleRefresh()
        }
    }

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else { return }
            refreshPending = false
            guard let backend else { return }
            let next = backend.snapshot?.sessionBank.coldTier
            if next != coldTier {
                coldTier = next
            }
        }
    }
}

private struct SettingsSSDCacheUsageRow: View {
    @StateObject private var projection: SettingsSSDCacheProjection

    @MainActor
    init(backend: MTPLXBackendStore) {
        _projection = StateObject(
            wrappedValue: SettingsSSDCacheProjection(backend: backend)
        )
    }

    var body: some View {
        if let cold = projection.coldTier {
            SettingsInfoRow(item: .ssdPolicy) {
                let livePhysicalBytes = cold.livePhysicalBytes ?? cold.physicalBytes ?? cold.bytes ?? 0
                let diskBytes = cold.managedDiskBytes ?? cold.managedFileBytes ?? livePhysicalBytes
                let untrackedBytes = cold.untrackedFileBytes ?? cold.untrackedDiskBytes ?? 0
                let logicalBytes = cold.logicalBytes
                let maxBytes = cold.maxBytes
                let scanPending = cold.diskUsageScanPending == true
                HStack(spacing: 12) {
                    Image(systemName: "internaldrive")
                        .foregroundStyle(Brand.typeSecondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(tr("Current usage"))
                            .font(.caption2)
                            .foregroundStyle(Brand.typeTertiary)
                        FlowLayout(spacing: 6) {
                            Text(tr("disk %@", Format.bytes(diskBytes)))
                                .font(.system(.callout, design: .rounded).weight(.semibold))
                                .monospacedDigit()
                                .foregroundStyle(Brand.typeBody)
                            if let maxBytes {
                                Text(tr("/ cap %@", Format.bytes(maxBytes)))
                                    .font(.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            if livePhysicalBytes != diskBytes {
                                Text(tr("· live %@", Format.bytes(livePhysicalBytes)))
                                    .font(.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            if untrackedBytes > 0 {
                                Text(tr("· untracked %@", Format.bytes(untrackedBytes)))
                                    .font(.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(Brand.warning)
                            }
                            if scanPending {
                                Text(tr("\u{00B7} scanning"))
                                    .font(.caption)
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            if let logicalBytes, logicalBytes != livePhysicalBytes {
                                Text(tr("· logical %@", Format.bytes(logicalBytes)))
                                    .font(.caption)
                                    .monospacedDigit()
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            if let entries = cold.entries {
                                Text(entries == 1 ? tr("· 1 entry") : tr("· %lld entries", entries))
                                    .font(.caption)
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            if let ratio = cold.dedupeRatio, ratio > 0 {
                                Text(tr("· %lld%% deduped", Int((ratio * 100).rounded())))
                                    .font(.caption)
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                        }
                    }
                    Spacer()
                    if cold.restorable == true {
                        PillBadge(
                            text: tr("restorable"),
                            systemImage: "checkmark.seal",
                            tint: Brand.success
                        )
                    }
                }
                .padding(.top, 6)
            }
        }
    }
}

@MainActor
private final class SettingsFanStatusProjection: ObservableObject {
    @Published private(set) var fanRampVerified: Bool?

    private weak var backend: MTPLXBackendStore?
    private var cancellable: AnyCancellable?
    private var refreshPending = false

    init(backend: MTPLXBackendStore) {
        self.backend = backend
        fanRampVerified = Self.evaluate(backend.thermal)
        cancellable = backend.objectWillChange.sink { [weak self] _ in
            self?.scheduleRefresh()
        }
    }

    private func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self else { return }
            refreshPending = false
            guard let backend else { return }
            let next = Self.evaluate(backend.thermal)
            if next != fanRampVerified {
                fanRampVerified = next
            }
        }
    }

    private static func evaluate(_ thermal: ThermalSnapshot?) -> Bool? {
        guard let thermal, thermal.ok, let max = thermal.maxRpm else { return nil }
        let fanRpms = thermal.fans.compactMap(\.actualRpm)
        guard !fanRpms.isEmpty else { return nil }
        return fanRpms.allSatisfy { Double($0) >= Double(max) * 0.9 }
    }
}

private struct SettingsCoolingFanStatusRow: View {
    let thermalPollingEnabled: Bool
    @StateObject private var projection: SettingsFanStatusProjection

    @MainActor
    init(backend: MTPLXBackendStore, thermalPollingEnabled: Bool) {
        self.thermalPollingEnabled = thermalPollingEnabled
        _projection = StateObject(
            wrappedValue: SettingsFanStatusProjection(backend: backend)
        )
    }

    var body: some View {
        if let verified = projection.fanRampVerified {
            SettingsInfoRow(item: .fanMode) {
                PillBadge(
                    text: verified ? tr("fan ramp verified") : tr("fan ramp pending"),
                    systemImage: verified ? "checkmark.seal.fill" : "clock",
                    tint: verified ? Brand.success : Brand.warning,
                    emphasized: !verified
                )
            }
        } else if !thermalPollingEnabled {
            SettingsInfoRow(item: .thermalPolling) {
                Text(tr("Turn on Thermal Polling above to confirm fan state."))
            }
        }
    }
}

private struct SettingsHermesGroup: View {
    let backend: MTPLXBackendStore
    @Binding var hermesWorkspacePath: String
    let onChooseWorkspace: () -> Void
    @EnvironmentObject private var hermes: HermesAgentStore

    var body: some View {
        SettingsGroup(group: .agents) {
            SettingsRow(item: .agentWorkspace, layout: .stacked) {
                HStack(spacing: 8) {
                    TextField(
                        MTPLXAppConfiguration.defaultHermesWorkspacePath(),
                        text: $hermesWorkspacePath
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))

                    Button {
                        onChooseWorkspace()
                    } label: {
                        Image(systemName: "folder")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help(tr("Choose agent workspace"))
                    .accessibilityLabel(tr("Choose agent workspace"))
                }
            }
            hermesRows
        }
        // Hermes is checked when its status is on screen, not on every
        // visit to Settings: the check runs the Hermes command line.
        .task {
            await hermes.prepare(configuration: backend.configuration)
        }
    }

    @ViewBuilder
    private var hermesRows: some View {
        SettingsRow(item: .hermes) {
            HStack(spacing: 8) {
                // The first check shows its spinner in the status line
                // below; this one covers a refresh of a loaded status.
                if case .checkingInstall = hermes.connectionState,
                   hermes.installStatus != nil {
                    ProgressView()
                        .controlSize(.small)
                }
                Button {
                    Task { await hermes.prepare(configuration: backend.configuration) }
                } label: {
                    Label(tr("Refresh"), systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        SettingsInfoRow(item: .hermes) {
            if let status = hermes.installStatus {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 6) {
                    hermesStatusLine(tr("Tools"), status.enabledToolsets.joined(separator: ", "))
                    if let version = status.versionSummary {
                        hermesStatusLine(tr("Version"), version)
                    }
                    if let update = status.updateSummary {
                        hermesStatusLine(tr("Update"), update, color: Brand.warning)
                    }
                    if let updateCommand = status.updateCommand, status.updateSummary != nil {
                        hermesStatusLine(tr("Command"), updateCommand)
                    }
                    if let gateway = status.gatewaySummary {
                        hermesStatusLine(tr("Gateway"), gateway, color: hermesGatewayColor(for: status.gatewayHealth))
                    }
                    if status.gatewayNeedsRepair {
                        GridRow {
                            Text(tr("Repair"))
                                .foregroundStyle(Brand.typeTertiary)
                            Button {
                                Task { await hermes.repairGateway() }
                            } label: {
                                Label(
                                    hermes.gatewayRepairInFlight ? tr("Repairing") : tr("Repair Gateway"),
                                    systemImage: "arrow.triangle.2.circlepath"
                                )
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(hermes.gatewayRepairInFlight)
                        }
                    }
                    if let repairMessage = hermes.gatewayRepairMessage {
                        hermesStatusLine(tr("Repair result"), repairMessage)
                    }
                    ForEach(status.integrationSummaries.prefix(3), id: \.self) { item in
                        hermesStatusLine(tr("Messaging"), item)
                    }
                    ForEach(status.warnings.prefix(2), id: \.self) { warning in
                        hermesStatusLine(tr("Warning"), warning, color: Brand.warning)
                    }
                    hermesStatusLine(tr("Capability"), status.capabilitySummary)
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.small)
                    Text(tr("Checking Hermes"))
                }
            }
        }
    }

    private func hermesStatusLine(
        _ label: String,
        _ value: String,
        color: Color = Brand.typeSecondary
    ) -> some View {
        GridRow {
            Text(label)
                .foregroundStyle(Brand.typeTertiary)
            Text(value)
                .font(.system(.callout, design: .rounded))
                .foregroundStyle(color)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func hermesGatewayColor(for health: HermesInstallStatus.GatewayHealth?) -> Color {
        switch health {
        case .healthy:
            return Brand.success
        case .warning, .unavailable:
            return Brand.warning
        case nil:
            return Brand.typeSecondary
        }
    }
}
