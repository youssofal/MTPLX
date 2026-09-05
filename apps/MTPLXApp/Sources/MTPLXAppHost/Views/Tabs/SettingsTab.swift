import SwiftUI
import MTPLXAppCore
#if canImport(AppKit)
import AppKit
#endif

struct SettingsTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var hermes: HermesAgentStore
    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var languageStore: LanguageStore
    @State private var languagePopoverPresented = false

    // Working copy of the persisted configuration. Saved on demand.
    // Live-mutable sampling/depth/reasoning knobs live in the
    // chrome-strip Inference popover — they aren't duplicated here.
    // The draft is owned above this tab (`SettingsDraftStore`, injected
    // at the app root): the dashboard rebuilds the tab on every
    // selection, so a draft kept in `@State` here died — with every
    // pending edit — the moment the user looked at another tab.
    @EnvironmentObject private var drafts: SettingsDraftStore
    @State private var pendingClearAll = false
    @State private var clearingCache = false

    @EnvironmentObject private var router: AppRouter

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

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if settingsDirty {
                    unsavedChangesRow
                }
                appearanceCard
                languageCard
                performanceCard
                ramCacheCard
                kvQuantCard
                memoryCard
                ssdCacheCard
                retrievalCard
                restartRequiredCard
                hermesToolTruthCard
                thermalCard
                adminCard
                aboutAndLogsCard
                if let error = lastSaveError {
                    Card(tr("Last save error")) {
                        Text(error)
                            .font(.callout)
                            .foregroundStyle(Color.mtplxDanger)
                    }
                }
            }
            .padding(20)
        }
        .onAppear {
            // Pending edits survive a tab switch; only an unedited draft
            // follows the persisted configuration.
            drafts.adoptIfUnedited(backend.configuration)
            Task { await hermes.prepare(configuration: backend.configuration) }
        }
        .onChange(of: backend.configuration) { _, newConfiguration in
            drafts.adoptIfUnedited(newConfiguration)
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

    /// Shown at the top of the tab while the draft differs from what is
    /// saved, so a user coming back from another tab can see that the
    /// values on screen are pending edits, and act on them without
    /// scrolling to the restart-required card.
    private var unsavedChangesRow: some View {
        let running = backend.daemonState.kind == .running || backend.daemonState.kind == .warming
        return HStack(spacing: 10) {
            PillBadge(text: tr("Unsaved changes"), systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
            Text(tr("Your edits are kept while you use other tabs. Save them to apply, or revert to the saved values."))
                .font(.caption)
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(2)
            Spacer(minLength: 8)
            Button {
                saveAndMaybeRestart(restart: running)
            } label: {
                if isApplying {
                    ProgressView().controlSize(.mini)
                } else {
                    Label(running ? tr("Apply + Restart") : tr("Save"),
                          systemImage: running ? "arrow.clockwise" : "checkmark.circle")
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isApplying)
            Button(tr("Revert")) { revertDraft() }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isApplying)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.mtplxWarning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.mtplxWarning.opacity(0.28), lineWidth: Brand.hairline)
                )
        )
    }

    private func revertDraft() {
        drafts.reset(to: backend.configuration)
    }

    // MARK: - Appearance

    @ViewBuilder
    private var appearanceCard: some View {
        Card(tr("Behavior"),
             subtitle: tr("App preferences saved on your Mac.")) {
            VStack(alignment: .leading, spacing: 8) {
                FormRow(
                    label: "Appearance",
                    caption: "Jet black, warm cream, or follow macOS."
                ) {
                    Picker("Appearance", selection: $themeStore.appearance) {
                        ForEach(AppAppearance.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 260, alignment: .leading)
                }

                Divider().overlay(Brand.separator)

                FormToggleRow(
                    label: tr("Sound on new speed record"),
                    caption: tr("Plays a soft chime when your Mac hits a new top speed."),
                    isOn: $themeStore.soundEnabled
                )
            }
        }
    }

    // MARK: - Language

    /// Same searchable picker as the onboarding step, in a popover so the
    /// card stays one row tall. Picking applies immediately: the store
    /// activates the language and the app shell re-renders.
    @ViewBuilder
    private var languageCard: some View {
        let language = languageStore.language
        Card(tr("Language"),
             subtitle: tr("The language MTPLX uses across the app. Changes apply immediately.")) {
            HStack(spacing: 12) {
                Text(language.flag)
                    .font(.system(size: 22))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(language.nativeName)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(Brand.typeHi)
                    Text(language.englishName)
                        .font(.caption)
                        .foregroundStyle(Brand.typeTertiary)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel(tr("Current language: %@", language.nativeName))
                Spacer(minLength: 0)
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
    }

    // MARK: - Performance (concurrency + batching)
    //
    // Maps to: schedulerMode (serial|ar_batch|continuous), batchingPreset
    // (latency|throughput|agent), maxActiveRequests, decodeBatchMax,
    // batchWaitMs, experimentalMTPCohorts. These are restart-required so the
    // same Save+Restart button is used. Prefill chunk size lives under Live
    // Settings because the daemon can apply it to the next request.

    @State private var performanceAdvancedExpanded = false

    @ViewBuilder
    private var performanceCard: some View {
        // Surface a pending indicator right on the card whose controls are
        // restart-required, so changing Mode / MTP cohorts while the daemon
        // is running no longer looks instantly applied with no feedback —
        // it reads "restart to apply" until the user hits Apply + Restart.
        let dirty = settingsDirty
        let running = backend.daemonState.kind == .running || backend.daemonState.kind == .warming
        Card(tr("Performance"),
             subtitle: tr("Speed and batching. Needs a restart to apply.")) {
            if (dirty || schedulingRestartPending) && running {
                PillBadge(text: tr("restart to apply"), systemImage: "arrow.clockwise.circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: tr("Mode"),
                    caption: tr("Auto picks the best mode for what you're using. Pick a mode below to use it everywhere. Saved as soon as you pick it. Benchmark runs keep their own single-stream setup.")
                ) {
                    Picker(tr("Mode"), selection: schedulerPresetBinding) {
                        Text(tr("Auto")).tag("target-default")
                        Text(tr("Fastest response")).tag("latency")
                        Text(tr("Handle multiple at once")).tag("throughput")
                        Text(tr("Long agent tasks")).tag("agent")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 280, alignment: .leading)
                }

                // The picker says what the next launch will use; this says
                // what the daemon in front of you actually launched with.
                // Issue #398 could not tell those two apart at all.
                if running, let launched = backend.lastLaunchScheduling {
                    Text(tr(
                        "Running now: %@ (%@ / %@).",
                        schedulingPresetLabel(launched.selectedPreset),
                        launched.schedulerMode,
                        launched.batchingPreset
                    ))
                    .font(.caption)
                    .foregroundStyle(Brand.typeSecondary)
                    .textSelection(.enabled)
                }

                Divider().overlay(Brand.separator)

                FormRow(
                    label: tr("Concurrency cap"),
                    caption: tr("Max parallel completions in flight.")
                ) {
                    optionalIntStepper(
                        value: $drafts.draft.maxActiveRequests,
                        defaultValue: defaultMaxActiveRequests,
                        range: 1...16
                    )
                }

                Divider().overlay(Brand.separator)

                FormToggleRow(
                    label: tr("Experimental MTP cohorts"),
                    caption: tr("Batch MTP verify steps across requests. Off = solo MTP (exactness preserved), on = experimental cohort batching."),
                    isOn: $drafts.draft.experimentalMTPCohorts
                )

                DisclosureGroup(isExpanded: $performanceAdvancedExpanded) {
                    VStack(alignment: .leading, spacing: 6) {
                        Divider().overlay(Brand.separator).padding(.top, 6)
                        FormRow(
                            label: tr("Decode batch max"),
                            caption: tr("Requests in a single decode step.")
                        ) {
                            optionalIntStepper(
                                value: $drafts.draft.decodeBatchMax,
                                defaultValue: defaultDecodeBatchMax,
                                range: 1...16
                            )
                        }
                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("Admission window"),
                            caption: tr("How long the scheduler waits for peers before firing.")
                        ) {
                            optionalDoubleStepper(
                                value: $drafts.draft.batchWaitMs,
                                defaultValue: defaultBatchWaitMs,
                                range: 0...500,
                                step: 10,
                                suffix: " ms",
                                width: 64
                            )
                        }
                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("Stall watchdog"),
                            caption: tr("Seconds a response may wait on a model that is making no progress before it is cancelled. 0 turns the watchdog off; long prompts and deep reasoning keep reporting progress and are never cut short by this.")
                        ) {
                            TextField(
                                "300",
                                value: Binding(
                                    get: { draftConfig.streamStallDeadlineSeconds },
                                    set: { draftConfig.streamStallDeadlineSeconds = max(0, $0) }
                                ),
                                format: .number.precision(.fractionLength(0))
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 90)
                        }
                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("HF download mirror"),
                            caption: tr("Hugging Face endpoint for model downloads. Your HF token is never sent to a mirror.")
                        ) {
                            TextField(
                                "https://hf-mirror.com",
                                text: Binding(
                                    get: { draftConfig.hfEndpoint ?? "" },
                                    set: { draftConfig.hfEndpoint = $0.isEmpty ? nil : $0 }
                                )
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 220)
                        }
                    }
                } label: {
                    Text(tr("Advanced"))
                        .font(.callout)
                        .foregroundStyle(Brand.typeSecondary)
                }
                .padding(.top, 4)
            }
        }
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
        guard let launched = backend.lastLaunchScheduling else { return false }
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
        case .openCode, .hermes:
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
        case .openCode, .hermes:
            return 1
        case .openWebUI, .benchmark, nil:
            return 1
        }
    }

    private var launchTargetDefaultBatchWaitMs: Double {
        switch draftLaunchTarget {
        case .chat, .pi, .other:
            return 50
        case .openCode, .hermes:
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

    // MARK: - RAM SessionBank Cache

    @ViewBuilder
    private var ramCacheCard: some View {
        Card(tr("Memory Cache (RAM)"),
             subtitle: tr("Warm prompt restore and bounded SessionBank allocation. Restart required.")) {
            EmptyView()
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: tr("Allocation policy"),
                    caption: tr("Target default budgets half the RAM left after the model loads; bounded uses the limits below.")
                ) {
                    Picker(tr("Allocation policy"), selection: $drafts.draft.ramSessionCachePolicy) {
                        Text(tr("Target default")).tag("target-default")
                        Text(tr("Bounded")).tag("bounded")
                        Text(tr("Minimal")).tag("minimal")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 180, alignment: .leading)
                }

                if draftConfig.ramSessionCachePolicy != "target-default" {
                    Divider().overlay(Brand.separator)

                    if draftConfig.ramSessionCachePolicy == "minimal" {
                        FormRow(
                            label: tr("Effective cap"),
                            caption: tr("Keeps one tiny RAM entry and disables block-prefix restore.")
                        ) {
                            Text(tr("1 entry / 1 GB"))
                                .font(.system(.body, design: .rounded).weight(.semibold))
                                .monospacedDigit()
                                .foregroundStyle(Brand.typeBody)
                        }
                    } else {
                        FormToggleRow(
                            label: tr("Block-prefix restore"),
                            caption: tr("Restore a shared prompt prefix and prefill only the changed suffix."),
                            isOn: $drafts.draft.ramSessionBlockPrefixRestore
                        )

                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("Max entries"),
                            caption: tr("Number of warm prefixes allowed to stay in RAM.")
                        ) {
                            Stepper(
                                value: $drafts.draft.ramSessionCacheMaxEntries,
                                in: 1...64
                            ) {
                                Text("\(draftConfig.ramSessionCacheMaxEntries)")
                                    .font(.system(.body, design: .rounded).weight(.semibold))
                                    .monospacedDigit()
                                    .frame(width: 32, alignment: .leading)
                            }
                        }

                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("Total RAM cap"),
                            caption: tr("Auto budgets half the RAM left after the model loads. Old prompts are evicted past the cap.")
                        ) {
                            cacheSizePicker(
                                selection: $drafts.draft.ramSessionCacheMaxSize,
                                values: ["auto", "1G", "2G", "4G", "8G", "16G", "24G", "32G", "48G"]
                            )
                        }

                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: tr("Per-session cap"),
                            caption: tr("Maximum RAM cache held by one conversation. Auto keeps it at 2/3 of the total cap.")
                        ) {
                            cacheSizePicker(
                                selection: $drafts.draft.ramSessionCachePerSessionMaxSize,
                                values: ["auto", "1G", "2G", "4G", "8G", "16G", "24G", "32G"]
                            )
                        }
                    }
                }

                if let bank = backend.snapshot?.sessionBank {
                    Divider().overlay(Brand.separator).padding(.top, 4)
                    ramUsageRow(bank: bank)
                }
            }
        }
    }

    // MARK: - KV Quantization

    @ViewBuilder
    private var kvQuantCard: some View {
        let policy = settingsKVQuantPolicy
        let modes = settingsKVQuantModes(policy)
        let supported = settingsKVQuantSupported(policy)
        let kvDirty = normalizedDraftConfigurationForSave().pagedKVQuantization
            != normalizedConfigurationForSave(backend.configuration).pagedKVQuantization
        Card(tr("KV Quantization"),
             subtitle: tr("Paged-attention KV cache precision. Restart required.")) {
            if kvDirty {
                PillBadge(text: tr("unsaved"), systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: tr("Quantization"),
                    caption: supported
                        ? "Off is the speed path. q8 saves memory when the selected model supports it; q4 is experimental."
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
                    .frame(maxWidth: 220, alignment: .leading)
                }

                Divider().overlay(Brand.separator)

                HStack(spacing: 8) {
                    Image(systemName: "memorychip")
                        .foregroundStyle(Brand.typeSecondary)
                    Text(kvQuantCaption)
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                    Spacer()
                }
            }
        }
    }

    // MARK: - Memory (limit + swap)
    //
    // Issue #431: a 128 GB Mac hit "reached memory limit" long before its
    // real ceiling, because the engine plans against 75% of RAM and the app
    // exposed no way past it. The reporter had to launch the bundle by hand
    // with MTPLX_MEMORY_LIMIT_BYTES to get their model served. Issue #427:
    // a 32 GB Mac that used to run 60-70k contexts under 2.9.x was clamped
    // to 40960 by the same fit calculation, and asked for an "I know what
    // I'm doing" escape hatch. Both are engine env-only knobs, so the card
    // writes them into the daemon's launch environment.

    @ViewBuilder
    private var memoryCard: some View {
        let memoryDirty = draftConfig.memoryLimitGB != backend.configuration.memoryLimitGB
            || draftConfig.allowSwap != backend.configuration.allowSwap
        Card(tr("Memory"),
             subtitle: tr("How much of this Mac the engine may plan with. Restart required.")) {
            if memoryDirty {
                PillBadge(text: tr("unsaved"), systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: tr("Memory limit"),
                    caption: tr("Empty uses the engine's own plan, about three quarters of this Mac's memory. Raise it to serve longer contexts on a Mac with headroom.")
                ) {
                    HStack(spacing: 6) {
                        TextField("", text: memoryLimitBinding)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.callout, design: .monospaced))
                            .frame(maxWidth: 90)
                        Text(tr("GB"))
                            .font(.callout)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                }

                Divider().overlay(Brand.separator)

                FormToggleRow(
                    label: tr("Allow swap"),
                    caption: tr("Serve contexts larger than what fits in memory. macOS pages to SSD, so speed drops sharply, but long sessions stop being refused."),
                    isOn: $drafts.draft.allowSwap
                )

                Divider().overlay(Brand.separator)

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "memorychip")
                        .foregroundStyle(Brand.typeSecondary)
                    Text(memoryPlanCaption)
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                }
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
        guard let plan = backend.memoryPlan, plan.available else {
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

    @ViewBuilder
    private func ramUsageRow(bank: SessionBank) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "memorychip")
                .foregroundStyle(Brand.typeSecondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(tr("Current RAM cache"))
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                HStack(spacing: 6) {
                    Text(Format.bytes(bank.totalNbytes))
                        .font(.system(.callout, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(Brand.typeBody)
                    let entries = bank.entries ?? bank.prefixes?.count ?? 0
                    Text(entries == 1 ? tr("· 1 entry") : tr("· %lld entries", entries))
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                    if let maxBytes = bank.maxBytes {
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

    // MARK: - SSD Persistent Cache

    @ViewBuilder
    private var ssdCacheCard: some View {
        Card(tr("Persistent Cache (SSD)"),
             subtitle: tr("Cached prompts survive an engine restart. Restart required after changes.")) {
            EmptyView()
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: tr("Policy"),
                    caption: ssdPolicyCaption
                ) {
                    Picker(tr("Policy"), selection: $drafts.draft.ssdSessionCache) {
                        Text(tr("Target default")).tag("target-default")
                        Text(tr("Off")).tag("off")
                        Text(tr("Read + write")).tag("on")
                        Text(tr("Write-only")).tag("write-only")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 180, alignment: .leading)
                }

                if ssdCacheDirty {
                    Divider().overlay(Brand.separator)
                    HStack(spacing: 8) {
                        PillBadge(
                            text: tr("unsaved"),
                            systemImage: "circle.fill",
                            tint: .mtplxWarning,
                            emphasized: true
                        )
                        Spacer()
                        Button {
                            saveAndMaybeRestart(restart: daemonRunning)
                        } label: {
                            if isApplying {
                                ProgressView().controlSize(.mini)
                            } else {
                                Label(daemonRunning ? tr("Apply + Restart") : tr("Save"),
                                      systemImage: daemonRunning ? "arrow.clockwise" : "checkmark.circle")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(isApplying)
                    }
                }

                if effectiveSSDSessionCache != "off" {
                    Divider().overlay(Brand.separator)
                    FormRow(
                        label: tr("Folder"),
                        caption: tr("Defaults to ~/.mtplx/session-bank when blank.")
                    ) {
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

                    Divider().overlay(Brand.separator)
                    FormRow(
                        label: tr("Max size"),
                        caption: tr("Auto scales with your Mac's RAM tier (16 GB to 100 GB). Old entries are evicted to stay under the cap.")
                    ) {
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
                        .frame(maxWidth: 140, alignment: .leading)
                    }

                    Divider().overlay(Brand.separator)
                    FormRow(
                        label: tr("Save prompts ≥"),
                        caption: tr("Shorter prompts aren't worth the write churn.")
                    ) {
                        Stepper(
                            value: $drafts.draft.ssdSessionCacheMinPrefixTokens,
                            in: 128...8192,
                            step: 128
                        ) {
                            Text(tr("%lld tok", draftConfig.ssdSessionCacheMinPrefixTokens))
                                .font(.system(.body, design: .rounded).weight(.semibold))
                                .monospacedDigit()
                                .frame(width: 96, alignment: .leading)
                        }
                    }

                    if let cold = backend.snapshot?.sessionBank.coldTier {
                        Divider().overlay(Brand.separator).padding(.top, 4)
                        usageRow(cold: cold)
                    }
                }
            }
        }
    }

    private var ssdPolicyCaption: String {
        let target = draftLaunchTarget?.title ?? tr("this target")
        let targetDefault = defaultSSDSessionCache(for: LaunchTarget(rawValue: draftConfig.lastLaunchTarget))
        let targetDefaultLabel = targetDefault == "off"
            ? "off"
            : (targetDefault == "write-only" ? "write-only" : "read + write")
        return tr("Target default is %@ for %@. Explicit choices override every launch target.", targetDefaultLabel, target)
    }

    private var ssdCacheDirty: Bool {
        draftConfig.ssdSessionCache != backend.configuration.ssdSessionCache
            || draftConfig.ssdSessionCacheDir != backend.configuration.ssdSessionCacheDir
            || draftConfig.ssdSessionCacheMaxSize != backend.configuration.ssdSessionCacheMaxSize
            || draftConfig.ssdSessionCacheMinPrefixTokens != backend.configuration.ssdSessionCacheMinPrefixTokens
    }

    private var daemonRunning: Bool {
        backend.daemonState.kind == .running || backend.daemonState.kind == .warming
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
        case .openCode, .hermes, .other:
            return "on"
        default:
            return "off"
        }
    }

    @ViewBuilder
    private func usageRow(cold: SessionBankColdTier) -> some View {
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

    // MARK: - Restart-required

    // MARK: - Retrieval (embeddings + reranking)
    //
    // Maps to: embeddingModels, rerankerModels, retrievalMaxResident. Purely
    // additive — with nothing configured the endpoints answer 404 and the chat
    // runtime is untouched. Restart-required, because the served set is built
    // when the daemon starts.

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

    @ViewBuilder
    private var retrievalCard: some View {
        Card(
            tr("Retrieval endpoints"),
            subtitle: tr("Serve embeddings and reranking from this daemon. Leave empty to keep MTPLX chat-only.")
        ) {
            VStack(alignment: .leading, spacing: 10) {
                FormRow(
                    label: tr("Embedding models"),
                    caption: tr("One per line — a Hugging Face id or local path, optionally REF=served-id. Serves /v1/embeddings.")
                ) {
                    TextField(
                        "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
                        text: modelListBinding(\.embeddingModels),
                        axis: .vertical
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))
                    .lineLimit(2...6)
                }

                Divider().overlay(Brand.separator)

                FormRow(
                    label: tr("Reranker models"),
                    caption: tr("Serves /v1/rerank. Listing the same reference here and above loads one copy of the weights for both roles.")
                ) {
                    TextField(
                        "vserifsaglam/Qwen3-Reranker-4B-4bit-MLX",
                        text: modelListBinding(\.rerankerModels),
                        axis: .vertical
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))
                    .lineLimit(2...6)
                }

                Divider().overlay(Brand.separator)

                FormRow(
                    label: tr("Release when idle"),
                    caption: tr("Minutes of inactivity before retrieval weights are unloaded. 0 keeps them resident; they reload on the next request.")
                ) {
                    TextField(
                        "0",
                        value: Binding(
                            get: { draftConfig.retrievalIdleTimeout / 60 },
                            set: { draftConfig.retrievalIdleTimeout = max(0, $0) * 60 }
                        ),
                        format: .number.precision(.fractionLength(0))
                    )
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 90)
                }

                Divider().overlay(Brand.separator)

                FormRow(
                    label: tr("Models kept resident"),
                    caption: tr("Retrieval models load on first request. Beyond this count the least recently used one is unloaded.")
                ) {
                    Stepper(
                        value: $drafts.draft.retrievalMaxResident,
                        in: 1...8
                    ) {
                        Text("\(draftConfig.retrievalMaxResident)")
                            .font(.system(.callout, design: .monospaced))
                    }
                    .frame(maxWidth: 160, alignment: .leading)
                }
            }
        }
    }

    @ViewBuilder
    private var restartRequiredCard: some View {
        let dirty = settingsDirty
        let running = backend.daemonState.kind == .running || backend.daemonState.kind == .warming

        Card(tr("Restart-Required Settings"),
             subtitle: tr("Changing these restarts the engine. Changes are saved to %@.", settingsFilePathHint)) {
            HStack(spacing: 8) {
                if dirty {
                    PillBadge(text: tr("unsaved"), systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
                }
                Button {
                    saveAndMaybeRestart(restart: running)
                } label: {
                    if isApplying {
                        ProgressView().controlSize(.mini)
                    } else {
                        Label(running ? tr("Apply + Restart") : tr("Save"),
                              systemImage: running ? "arrow.clockwise" : "checkmark.circle")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!dirty || isApplying)
                Button(tr("Revert")) { revertDraft() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(!dirty)
            }
        } content: {
            // Single `VStack(alignment: .leading)` is the whole fix —
            // without an explicit alignment SwiftUI's VStack defaults
            // to .center, which made short-value rows (Profile,
            // Host/Port, Generation mode, Context window) silently
            // float toward the middle of the card while wide-value
            // rows (Model, Executable path) stayed flush-left. Every
            // row now uses `FormRow` / `FormToggleRow` so the label
            // column is the same 200pt across every card in the tab.
            VStack(alignment: .leading, spacing: 4) {
                FormRow(label: tr("Model")) {
                    TextField("", text: $drafts.draft.model)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                }

                FormRow(
                    label: tr("Profile"),
                    caption: tr("Auto picks the recommended profile for the selected model — Turbo for the 27B and Flash-Next models.")
                ) {
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
                    .frame(maxWidth: 220, alignment: .leading)
                }

                FormRow(
                    label: tr("Executable path"),
                    caption: tr("Defaults to mtplx in PATH if blank.")
                ) {
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

                FormRow(label: tr("Host / Port")) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            TextField("127.0.0.1", text: $drafts.draft.host)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 160)
                            Text(":")
                                .foregroundStyle(Brand.typeTertiary)
                            TextField(
                                "8000",
                                value: $drafts.draft.port,
                                format: .number.grouping(.never)
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 80)
                        }
                        if !MTPLXServerURLs.isLoopbackBind(draftConfig.host) {
                            Text(
                                MTPLXServerURLs.isWildcardBind(draftConfig.host)
                                    ? "Serves on every interface (LAN). An API key is required."
                                    : tr("Non-localhost host: an API key is required.")
                            )
                            .font(.caption)
                            .foregroundStyle(
                                (draftConfig.apiKey ?? "").isEmpty
                                    ? Brand.warning
                                    : Brand.typeTertiary
                            )
                        }
                    }
                }

                FormRow(label: tr("Generation mode")) {
                    Picker(tr("Mode"), selection: $drafts.draft.generationMode) {
                        Text(tr("MTP")).tag("mtp")
                        Text(tr("Baseline")).tag("ar")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 160, alignment: .leading)
                }

                FormRow(label: tr("Context window")) {
                    HStack(spacing: 8) {
                        TextField(
                            "auto",
                            value: contextWindowBinding,
                            format: .number
                        )
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 120)
                        Text(tr("tokens"))
                            .font(.caption)
                            .foregroundStyle(Brand.typeTertiary)
                        Text(tr("max %@", Self.formatTokens(settingsModelMaxContext)))
                            .font(.caption)
                            .foregroundStyle(Brand.typeTertiary)
                    }
                }

                FormRow(label: tr("API key (optional)")) {
                    SecureField(
                        "none",
                        text: Binding(
                            get: { draftConfig.apiKey ?? "" },
                            set: { draftConfig.apiKey = $0.isEmpty ? nil : $0 }
                        )
                    )
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                }

                FormRow(
                    label: tr("Agent workspace"),
                    caption: tr("Pi and Hermes terminal tools start in this folder.")
                ) {
                    HStack(spacing: 8) {
                        TextField(
                            MTPLXAppConfiguration.defaultHermesWorkspacePath(),
                            text: $drafts.draft.hermesWorkspacePath
                        )
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                        .frame(maxWidth: 460)

                        Button {
                            chooseHermesWorkspace()
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

                Divider().overlay(Brand.separator).padding(.vertical, 4)

                FormToggleRow(
                    label: tr("Load MTP head"),
                    caption: tr("Disable to fall back to baseline (no speculation)."),
                    isOn: $drafts.draft.loadMTP
                )

                FormToggleRow(
                    label: tr("Enable thermal polling"),
                    caption: tr("Required to verify fan ramp before benchmarks."),
                    isOn: $drafts.draft.enableThermalPolling
                )

                FormToggleRow(
                    label: tr("Start MTPLX when the app opens"),
                    caption: tr("Otherwise you start MTPLX from the toolbar manually."),
                    isOn: $drafts.draft.launchDaemonOnOpen
                )

                FormToggleRow(
                    label: tr("Restart this session's MTPLX after a crash"),
                    caption: tr("Applies to daemons launched after enabling this setting. Retries up to three times with backoff; Stop cancels pending retries."),
                    isOn: $drafts.draft.automaticDaemonRestart
                )
                if let restartStatusText {
                    Text(restartStatusText)
                        .font(.caption)
                        .foregroundStyle(Brand.typeTertiary)
                        .padding(.leading, 200)
                }

                Divider().overlay(Brand.separator).padding(.vertical, 4)

                streamCadenceRow
            }
        }
    }

    private var settingsFilePathHint: String {
        backend.settingsURL.path
    }

    private var restartStatusText: String? {
        switch backend.daemonRestartStatus {
        case .idle:
            guard draftConfig.automaticDaemonRestart else { return nil }
            switch backend.daemonRestartEligibility {
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

    @ViewBuilder
    private var streamCadenceRow: some View {
        let bounds = backend.capabilities?.snapshotInterval
        let minMs = bounds?.minMs ?? 100
        let maxMs = bounds?.maxMs ?? 5000
        let perfLockMs = bounds?.performanceLockMs ?? 1000
        FormRow(
            label: tr("Stream cadence"),
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
                    .frame(minWidth: 72, alignment: .leading)
            }
        }
    }

    private func saveAndMaybeRestart(restart: Bool) {
        isApplying = true
        lastSaveError = nil
        let config = normalizedDraftConfigurationForSave()
        // The draft store outlives this view, so a save that finishes
        // after the user has moved to another tab still lands.
        let drafts = self.drafts
        Task {
            do {
                try await backend.applyConfiguration(config, restartIfRunning: restart)
                drafts.reset(to: config)
            } catch {
                drafts.lastSaveError = tr("Apply failed: %@", String(describing: error))
            }
            drafts.isApplying = false
        }
    }

    private var settingsDirty: Bool {
        normalizedDraftConfigurationForSave() != normalizedConfigurationForSave(backend.configuration)
    }

    private var settingsModelFamily: String {
        MTPLXModelOption.modelFamily(for: draftConfig.model)
    }

    private var compatibleSettings: MutableSettings? {
        guard let settings = backend.settings else { return nil }
        let settingsFamily = settings.modelControls?.modelFamily ?? settings.modelFamily
        guard let settingsFamily else {
            return MTPLXModelOption.supportsTune(family: settingsModelFamily) ? settings : nil
        }
        return settingsFamily == settingsModelFamily ? settings : nil
    }

    private var compatibleStartupControls: ModelControls? {
        guard let controls = backend.health?.startup?.modelControls else { return nil }
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

    private var contextWindowBinding: Binding<Int> {
        Binding(
            get: { compatibleDraftContextWindow ?? 0 },
            set: { value in
                let raw = Int(value)
                guard raw > 0 else {
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

    private func normalizedDraftConfigurationForSave() -> MTPLXAppConfiguration {
        normalizedConfigurationForSave(draftConfig)
    }

    private func normalizedConfigurationForSave(_ source: MTPLXAppConfiguration) -> MTPLXAppConfiguration {
        var config = source
        let family = MTPLXModelOption.modelFamily(for: source.model)
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

    // MARK: - Hermes

    @ViewBuilder
    private var hermesToolTruthCard: some View {
        Card("Hermes", subtitle: tr("Agent handoff, tools, and gateway state.")) {
            HStack(spacing: 8) {
                Button {
                    Task { await hermes.prepare(configuration: backend.configuration) }
                } label: {
                    Label(tr("Refresh"), systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                if case .checkingInstall = hermes.connectionState {
                    ProgressView()
                        .controlSize(.small)
                }
            }
        } content: {
            VStack(alignment: .leading, spacing: 4) {
                if let status = hermes.installStatus {
                    FormRow(label: tr("Tools")) {
                        statusText(status.enabledToolsets.joined(separator: ", "))
                    }
                    if let version = status.versionSummary {
                        FormRow(label: tr("Version")) { statusText(version) }
                    }
                    if let update = status.updateSummary {
                        FormRow(label: tr("Update")) { statusText(update, color: Brand.warning) }
                    }
                    if let updateCommand = status.updateCommand, status.updateSummary != nil {
                        FormRow(label: tr("Command")) { statusText(updateCommand) }
                    }
                    if let gateway = status.gatewaySummary {
                        FormRow(label: tr("Gateway")) {
                            statusText(gateway, color: hermesGatewayColor(for: status.gatewayHealth))
                        }
                    }
                    if status.gatewayNeedsRepair {
                        FormRow(label: tr("Repair")) {
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
                        FormRow(label: tr("Repair result")) {
                            statusText(repairMessage)
                        }
                    }
                    ForEach(status.integrationSummaries.prefix(3), id: \.self) { item in
                        FormRow(label: tr("Messaging")) { statusText(item) }
                    }
                    ForEach(status.warnings.prefix(2), id: \.self) { warning in
                        FormRow(label: tr("Warning")) { statusText(warning, color: Brand.warning) }
                    }
                    FormRow(label: tr("Capability")) {
                        statusText(status.capabilitySummary)
                    }
                } else {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(tr("Checking Hermes"))
                            .font(.callout)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    .task {
                        await hermes.prepare(configuration: backend.configuration)
                    }
                }
            }
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

    private func statusText(
        _ value: String,
        color: Color = Brand.typeSecondary
    ) -> some View {
        Text(value)
            .font(.system(.callout, design: .rounded))
            .foregroundStyle(color)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Thermal (V1 fan control)

    private var fanModeSelectionBinding: Binding<String> {
        Binding(
            get: { MTPLXFanMode.normalized(draftConfig.fanMode).rawValue },
            set: { rawMode in
                let mode = MTPLXFanMode.normalized(rawMode)
                draftConfig.fanMode = mode.rawValue
                draftConfig.pinFansAtMaxOnStart = mode == .max
            }
        )
    }

    @ViewBuilder
    private var thermalCard: some View {
        Card(tr("Thermal"),
             subtitle: tr("Smart boosts fans only during generation; Max stays available for sustained benchmark runs.")) {
            FanModeToggle()
        } content: {
            VStack(alignment: .leading, spacing: 10) {
                FormRow(
                    label: tr("Fan Mode"),
                    caption: tr("Default uses Apple's curve. Smart boosts during requests. Max pins verified fans.")
                ) {
                    Picker(tr("Fan Mode"), selection: fanModeSelectionBinding) {
                        ForEach(MTPLXFanMode.allCases, id: \.self) { mode in
                            Text(mode.title).tag(mode.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 280, alignment: .leading)
                }
                if let thermal = backend.thermal, thermal.ok {
                    let fanRpms = thermal.fans.compactMap(\.actualRpm)
                    if let max = thermal.maxRpm, !fanRpms.isEmpty {
                        let verified = fanRpms.allSatisfy { Double($0) >= Double(max) * 0.9 }
                        PillBadge(
                            text: verified ? tr("fan ramp verified") : tr("fan ramp pending"),
                            systemImage: verified ? "checkmark.seal.fill" : "clock",
                            tint: verified ? Brand.success : Brand.warning,
                            emphasized: !verified
                        )
                    }
                } else if !backend.configuration.enableThermalPolling {
                    Text(tr("Turn on Thermal Polling above to confirm fan state."))
                        .font(.caption)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                }
            }
        }
    }

    // MARK: - Admin

    @ViewBuilder
    private var adminCard: some View {
        Card(tr("Reset"), subtitle: tr("Can't be undone.")) {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    pendingClearAll = true
                } label: {
                    HStack {
                        Label(tr("Clear prompt cache"), systemImage: "trash")
                        Spacer()
                        if clearingCache {
                            ProgressView().controlSize(.mini)
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                Text(tr("To stop or restart the model, use the play button in the top bar."))
                    .font(.caption2)
                    .foregroundStyle(Brand.textHighlight.opacity(0.55))
            }
        }
    }

    // MARK: - About + Logs

    @ViewBuilder
    private var aboutAndLogsCard: some View {
        Card(tr("Info")) {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    router.presentAbout()
                } label: {
                    HStack {
                        Label(tr("About MTPLX"), systemImage: "info.circle")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption)
                            .foregroundStyle(Brand.textHighlight.opacity(0.4))
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)

                Button {
                    router.presentLogs()
                } label: {
                    HStack {
                        Label(tr("Open Logs (Cmd-Shift-L)"), systemImage: "doc.text")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption)
                            .foregroundStyle(Brand.textHighlight.opacity(0.4))
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
            }
        }
    }
}
