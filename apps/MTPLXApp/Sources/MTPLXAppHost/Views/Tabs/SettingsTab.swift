import SwiftUI
import MTPLXAppCore
#if canImport(AppKit)
import AppKit
#endif

struct SettingsTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var hermes: HermesAgentStore
    @EnvironmentObject private var themeStore: ThemeStore

    // Working copy of the persisted configuration. Saved on demand.
    // Live-mutable sampling/depth/reasoning knobs live in the
    // chrome-strip Inference popover — they aren't duplicated here.
    @State private var draftConfig: MTPLXAppConfiguration = MTPLXAppConfiguration()
    @State private var lastSyncedConfig: MTPLXAppConfiguration = MTPLXAppConfiguration()
    @State private var isApplying = false
    @State private var lastSaveError: String? = nil
    @State private var pendingClearAll = false
    @State private var clearingCache = false

    @EnvironmentObject private var router: AppRouter

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                appearanceCard
                performanceCard
                ramCacheCard
                kvQuantCard
                ssdCacheCard
                retrievalCard
                restartRequiredCard
                hermesToolTruthCard
                thermalCard
                adminCard
                aboutAndLogsCard
                if let error = lastSaveError {
                    Card("Last save error") {
                        Text(error)
                            .font(.callout)
                            .foregroundStyle(Color.mtplxDanger)
                    }
                }
            }
            .padding(20)
        }
        .onAppear {
            syncDrafts()
            Task { await hermes.prepare(configuration: backend.configuration) }
        }
        .onChange(of: backend.configuration) { _, newConfiguration in
            syncDraftsIfUnedited(newConfiguration)
        }
        .confirmationDialog(
            "Clear all SessionBank entries?",
            isPresented: $pendingClearAll
        ) {
            Button("Clear All", role: .destructive) {
                clearingCache = true
                Task {
                    defer { Task { @MainActor in clearingCache = false } }
                    try? await backend.clearCache()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Clears every saved prompt cache. Anything mid-flight keeps running.")
        }
    }

    private func syncDrafts() {
        draftConfig = backend.configuration
        lastSyncedConfig = backend.configuration
    }

    private func syncDraftsIfUnedited(_ newConfiguration: MTPLXAppConfiguration) {
        if draftConfig == newConfiguration {
            lastSyncedConfig = newConfiguration
        } else if draftConfig == lastSyncedConfig {
            draftConfig = newConfiguration
            lastSyncedConfig = newConfiguration
        }
    }

    // MARK: - Appearance

    @ViewBuilder
    private var appearanceCard: some View {
        Card("Behavior",
             subtitle: "App preferences saved on your Mac.") {
            VStack(alignment: .leading, spacing: 8) {
                FormToggleRow(
                    label: "Sound on new speed record",
                    caption: "Plays a soft chime when your Mac hits a new top speed.",
                    isOn: $themeStore.soundEnabled
                )
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
        Card("Performance",
             subtitle: "Speed and batching. Needs a restart to apply.") {
            if dirty && running {
                PillBadge(text: "restart to apply", systemImage: "arrow.clockwise.circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: "Mode",
                    caption: "Auto picks the best mode for what you're using. Pick a mode below to use it everywhere."
                ) {
                    Picker("Mode", selection: schedulerPresetBinding) {
                        Text("Auto").tag("target-default")
                        Text("Fastest response").tag("latency")
                        Text("Handle multiple at once").tag("throughput")
                        Text("Long agent tasks").tag("agent")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 280, alignment: .leading)
                }

                Divider().overlay(Brand.separator)

                FormRow(
                    label: "Concurrency cap",
                    caption: "Max parallel completions in flight."
                ) {
                    optionalIntStepper(
                        value: $draftConfig.maxActiveRequests,
                        defaultValue: defaultMaxActiveRequests,
                        range: 1...16
                    )
                }

                Divider().overlay(Brand.separator)

                FormToggleRow(
                    label: "Experimental MTP cohorts",
                    caption: "Batch MTP verify steps across requests. Off = solo MTP (exactness preserved), on = experimental cohort batching.",
                    isOn: $draftConfig.experimentalMTPCohorts
                )

                DisclosureGroup(isExpanded: $performanceAdvancedExpanded) {
                    VStack(alignment: .leading, spacing: 6) {
                        Divider().overlay(Brand.separator).padding(.top, 6)
                        FormRow(
                            label: "Decode batch max",
                            caption: "Requests in a single decode step."
                        ) {
                            optionalIntStepper(
                                value: $draftConfig.decodeBatchMax,
                                defaultValue: defaultDecodeBatchMax,
                                range: 1...16
                            )
                        }
                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: "Admission window",
                            caption: "How long the scheduler waits for peers before firing."
                        ) {
                            optionalDoubleStepper(
                                value: $draftConfig.batchWaitMs,
                                defaultValue: defaultBatchWaitMs,
                                range: 0...500,
                                step: 10,
                                suffix: " ms",
                                width: 64
                            )
                        }
                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: "HF download mirror",
                            caption: "Hugging Face endpoint for model downloads. Your HF token is never sent to a mirror."
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
                    Text("Advanced")
                        .font(.callout)
                        .foregroundStyle(Brand.typeSecondary)
                }
                .padding(.top, 4)
            }
        }
    }

    private var schedulerPresetBinding: Binding<String> {
        Binding(
            get: {
                normalizedSchedulingPreset(draftConfig.schedulingPreset)
            },
            set: { preset in
                draftConfig.applySchedulingPreset(normalizedSchedulingPreset(preset))
            }
        )
    }

    private func normalizedSchedulingPreset(_ raw: String) -> String {
        switch raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        {
        case "latency", "serial-latency":
            return "latency"
        case "throughput", "ar-batch-throughput":
            return "throughput"
        case "agent", "ar-batch-agent":
            return "agent"
        default:
            return "target-default"
        }
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
                Text("Preset default")
                    .font(.system(.body, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeSecondary)
                Button("Override") {
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
                .help("Use the preset default")
                .accessibilityLabel("Reset to preset default")
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
                Text("Preset default")
                    .font(.system(.body, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeSecondary)
                Button("Override") {
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
                .help("Use the preset default")
                .accessibilityLabel("Reset to preset default")
            }
        }
    }

    // MARK: - RAM SessionBank Cache

    @ViewBuilder
    private var ramCacheCard: some View {
        Card("Memory Cache (RAM)",
             subtitle: "Warm prompt restore and bounded SessionBank allocation. Restart required.") {
            EmptyView()
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: "Allocation policy",
                    caption: "Target default budgets half the RAM left after the model loads; bounded uses the limits below."
                ) {
                    Picker("Allocation policy", selection: $draftConfig.ramSessionCachePolicy) {
                        Text("Target default").tag("target-default")
                        Text("Bounded").tag("bounded")
                        Text("Minimal").tag("minimal")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 180, alignment: .leading)
                }

                if draftConfig.ramSessionCachePolicy != "target-default" {
                    Divider().overlay(Brand.separator)

                    if draftConfig.ramSessionCachePolicy == "minimal" {
                        FormRow(
                            label: "Effective cap",
                            caption: "Keeps one tiny RAM entry and disables block-prefix restore."
                        ) {
                            Text("1 entry / 1 GB")
                                .font(.system(.body, design: .rounded).weight(.semibold))
                                .monospacedDigit()
                                .foregroundStyle(Brand.typeBody)
                        }
                    } else {
                        FormToggleRow(
                            label: "Block-prefix restore",
                            caption: "Restore a shared prompt prefix and prefill only the changed suffix.",
                            isOn: $draftConfig.ramSessionBlockPrefixRestore
                        )

                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: "Max entries",
                            caption: "Number of warm prefixes allowed to stay in RAM."
                        ) {
                            Stepper(
                                value: $draftConfig.ramSessionCacheMaxEntries,
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
                            label: "Total RAM cap",
                            caption: "Auto budgets half the RAM left after the model loads. Old prompts are evicted past the cap."
                        ) {
                            cacheSizePicker(
                                selection: $draftConfig.ramSessionCacheMaxSize,
                                values: ["auto", "1G", "2G", "4G", "8G", "16G", "24G", "32G", "48G"]
                            )
                        }

                        Divider().overlay(Brand.separator)
                        FormRow(
                            label: "Per-session cap",
                            caption: "Maximum RAM cache held by one conversation. Auto keeps it at 2/3 of the total cap."
                        ) {
                            cacheSizePicker(
                                selection: $draftConfig.ramSessionCachePerSessionMaxSize,
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
        Card("KV Quantization",
             subtitle: "Paged-attention KV cache precision. Restart required.") {
            if kvDirty {
                PillBadge(text: "unsaved", systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
            }
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: "Quantization",
                    caption: supported
                        ? "Off is the speed path. q8 saves memory when the selected model supports it; q4 is experimental."
                        : policy.disabledReason ?? "KV quantization is not supported for this model."
                ) {
                    Picker("Quantization", selection: kvQuantSelectionBinding(policy)) {
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

    private var kvQuantCaption: String {
        let policy = settingsKVQuantPolicy
        guard settingsKVQuantSupported(policy) else {
            return policy.disabledReason ?? "KV quantization is not supported for this model."
        }
        switch kvQuantSelectionBinding(policy).wrappedValue {
        case "q8":
            return "Stores KV in int8 with per-token scales. Use it when memory matters more than 20k decode speed."
        case "q4":
            return "Packs KV into 4-bit nibbles with per-token scales; keep as an experiment until measured."
        default:
            return "Uses the model's native KV dtype."
        }
    }

    private func cacheSizePicker(selection: Binding<String>, values: [String]) -> some View {
        Picker("Cache size", selection: selection) {
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
            return "Auto"
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
                Text("Current RAM cache")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                HStack(spacing: 6) {
                    Text(Format.bytes(bank.totalNbytes))
                        .font(.system(.callout, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(Brand.typeBody)
                    let entries = bank.entries ?? bank.prefixes?.count ?? 0
                    Text("· \(entries) entr\(entries == 1 ? "y" : "ies")")
                        .font(.caption)
                        .foregroundStyle(Brand.typeSecondary)
                    if let maxBytes = bank.maxBytes {
                        Text("· cap \(Format.bytes(maxBytes))")
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
        Card("Persistent Cache (SSD)",
             subtitle: "Cached prompts survive an engine restart. Restart required after changes.") {
            EmptyView()
        } content: {
            VStack(alignment: .leading, spacing: 6) {
                FormRow(
                    label: "Policy",
                    caption: ssdPolicyCaption
                ) {
                    Picker("Policy", selection: $draftConfig.ssdSessionCache) {
                        Text("Target default").tag("target-default")
                        Text("Off").tag("off")
                        Text("Read + write").tag("on")
                        Text("Write-only").tag("write-only")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 180, alignment: .leading)
                }

                if ssdCacheDirty {
                    Divider().overlay(Brand.separator)
                    HStack(spacing: 8) {
                        PillBadge(
                            text: "unsaved",
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
                                Label(daemonRunning ? "Apply + Restart" : "Save",
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
                        label: "Folder",
                        caption: "Defaults to ~/.mtplx/session-bank when blank."
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
                                    .help("Open folder in Finder")
                                    .accessibilityLabel("Reveal in Finder")
                            }
                        }
                    }

                    Divider().overlay(Brand.separator)
                    FormRow(
                        label: "Max size",
                        caption: "Auto scales with your Mac's RAM tier (16 GB to 100 GB). Old entries are evicted to stay under the cap."
                    ) {
                        Picker("Max size", selection: $draftConfig.ssdSessionCacheMaxSize) {
                            Text("Auto").tag("auto")
                            Text("10 GB").tag("10GB")
                            Text("50 GB").tag("50GB")
                            Text("100 GB").tag("100GB")
                            Text("250 GB").tag("250GB")
                            Text("500 GB").tag("500GB")
                            Text("1 TB").tag("1TB")
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .frame(maxWidth: 140, alignment: .leading)
                    }

                    Divider().overlay(Brand.separator)
                    FormRow(
                        label: "Save prompts \u{2265}",
                        caption: "Shorter prompts aren't worth the write churn."
                    ) {
                        Stepper(
                            value: $draftConfig.ssdSessionCacheMinPrefixTokens,
                            in: 128...8192,
                            step: 128
                        ) {
                            Text("\(draftConfig.ssdSessionCacheMinPrefixTokens) tok")
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
        let target = draftLaunchTarget?.title ?? "this target"
        let targetDefault = defaultSSDSessionCache(for: LaunchTarget(rawValue: draftConfig.lastLaunchTarget))
        let targetDefaultLabel = targetDefault == "off"
            ? "off"
            : (targetDefault == "write-only" ? "write-only" : "read + write")
        return "Target default is \(targetDefaultLabel) for \(target). Explicit choices override every launch target."
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
                Text("Current usage")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
                FlowLayout(spacing: 6) {
                    Text("disk \(Format.bytes(diskBytes))")
                        .font(.system(.callout, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(Brand.typeBody)
                    if let maxBytes {
                        Text("/ cap \(Format.bytes(maxBytes))")
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    if livePhysicalBytes != diskBytes {
                        Text("\u{00B7} live \(Format.bytes(livePhysicalBytes))")
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    if untrackedBytes > 0 {
                        Text("\u{00B7} untracked \(Format.bytes(untrackedBytes))")
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(Brand.warning)
                    }
                    if scanPending {
                        Text("\u{00B7} scanning")
                            .font(.caption)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    if let logicalBytes, logicalBytes != livePhysicalBytes {
                        Text("\u{00B7} logical \(Format.bytes(logicalBytes))")
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    if let entries = cold.entries {
                        Text("\u{00B7} \(entries) entr\(entries == 1 ? "y" : "ies")")
                            .font(.caption)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                    if let ratio = cold.dedupeRatio, ratio > 0 {
                        Text("\u{00B7} \(Int((ratio * 100).rounded()))% deduped")
                            .font(.caption)
                            .foregroundStyle(Brand.typeSecondary)
                    }
                }
            }
            Spacer()
            if cold.restorable == true {
                PillBadge(
                    text: "restorable",
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
            "Retrieval endpoints",
            subtitle: "Serve embeddings and reranking from this daemon. Leave empty to keep MTPLX chat-only."
        ) {
            VStack(alignment: .leading, spacing: 10) {
                FormRow(
                    label: "Embedding models",
                    caption: "One per line — a Hugging Face id or local path, optionally REF=served-id. Serves /v1/embeddings."
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
                    label: "Reranker models",
                    caption: "Serves /v1/rerank. Listing the same reference here and above loads one copy of the weights for both roles."
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
                    label: "Release when idle",
                    caption: "Minutes of inactivity before retrieval weights are unloaded. 0 keeps them resident; they reload on the next request."
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
                    label: "Models kept resident",
                    caption: "Retrieval models load on first request. Beyond this count the least recently used one is unloaded."
                ) {
                    Stepper(
                        value: $draftConfig.retrievalMaxResident,
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

        Card("Restart-Required Settings",
             subtitle: "Changing these restarts the engine. Changes are saved to \(settingsFilePathHint).") {
            HStack(spacing: 8) {
                if dirty {
                    PillBadge(text: "unsaved", systemImage: "circle.fill", tint: .mtplxWarning, emphasized: true)
                }
                Button {
                    saveAndMaybeRestart(restart: running)
                } label: {
                    if isApplying {
                        ProgressView().controlSize(.mini)
                    } else {
                        Label(running ? "Apply + Restart" : "Save",
                              systemImage: running ? "arrow.clockwise" : "checkmark.circle")
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!dirty || isApplying)
                Button("Revert") { draftConfig = backend.configuration }
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
                FormRow(label: "Model") {
                    TextField("", text: $draftConfig.model)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                }

                FormRow(
                    label: "Profile",
                    caption: "Auto picks the recommended profile for the "
                        + "selected model — Turbo for the 27B models."
                ) {
                    // Only persistable profiles may appear here; a stray
                    // tag value persists into config and kills serve at
                    // argparse ("auto" is resolved to a concrete engine
                    // profile before launch). Max fans is the Fan mode
                    // row, not a profile.
                    Picker("Profile", selection: $draftConfig.profile) {
                        Text("Auto (recommended)").tag("auto")
                        Text("Turbo").tag("turbo")
                        Text("Sustained").tag("sustained")
                        Text("Performance Cold (Burst)").tag("performance-cold")
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .frame(maxWidth: 220, alignment: .leading)
                }

                FormRow(
                    label: "Executable path",
                    caption: "Defaults to mtplx in PATH if blank."
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

                FormRow(label: "Host / Port") {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            TextField("127.0.0.1", text: $draftConfig.host)
                                .textFieldStyle(.roundedBorder)
                                .frame(maxWidth: 160)
                            Text(":")
                                .foregroundStyle(Brand.typeTertiary)
                            TextField(
                                "8000",
                                value: $draftConfig.port,
                                format: .number.grouping(.never)
                            )
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 80)
                        }
                        if !MTPLXServerURLs.isLoopbackBind(draftConfig.host) {
                            Text(
                                MTPLXServerURLs.isWildcardBind(draftConfig.host)
                                    ? "Serves on every interface (LAN). An API key is required."
                                    : "Non-localhost host: an API key is required."
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

                FormRow(label: "Generation mode") {
                    Picker("Mode", selection: $draftConfig.generationMode) {
                        Text("MTP").tag("mtp")
                        Text("Baseline").tag("ar")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 160, alignment: .leading)
                }

                FormRow(label: "Context window") {
                    HStack(spacing: 8) {
                        TextField(
                            "auto",
                            value: contextWindowBinding,
                            format: .number
                        )
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 120)
                        Text("tokens")
                            .font(.caption)
                            .foregroundStyle(Brand.typeTertiary)
                        Text("max \(Self.formatTokens(settingsModelMaxContext))")
                            .font(.caption)
                            .foregroundStyle(Brand.typeTertiary)
                    }
                }

                FormRow(label: "API key (optional)") {
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
                    label: "Agent workspace",
                    caption: "Pi and Hermes terminal tools start in this folder."
                ) {
                    HStack(spacing: 8) {
                        TextField(
                            MTPLXAppConfiguration.defaultHermesWorkspacePath(),
                            text: $draftConfig.hermesWorkspacePath
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
                        .help("Choose agent workspace")
                        .accessibilityLabel("Choose agent workspace")
                    }
                }

                Divider().overlay(Brand.separator).padding(.vertical, 4)

                FormToggleRow(
                    label: "Load MTP head",
                    caption: "Disable to fall back to baseline (no speculation).",
                    isOn: $draftConfig.loadMTP
                )

                FormToggleRow(
                    label: "Enable thermal polling",
                    caption: "Required to verify fan ramp before benchmarks.",
                    isOn: $draftConfig.enableThermalPolling
                )

                FormToggleRow(
                    label: "Start MTPLX when the app opens",
                    caption: "Otherwise you start MTPLX from the toolbar manually.",
                    isOn: $draftConfig.launchDaemonOnOpen
                )

                Divider().overlay(Brand.separator).padding(.vertical, 4)

                streamCadenceRow
            }
        }
    }

    private var settingsFilePathHint: String {
        backend.settingsURL.path
    }

    private func chooseHermesWorkspace() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.prompt = "Use"
        panel.message = "Choose the project folder Hermes should use for terminal and file tools."
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
            label: "Stream cadence",
            caption: "Performance Lock overrides this to \(perfLockMs) ms."
        ) {
            Stepper(
                value: $draftConfig.streamSnapshotIntervalMs,
                in: minMs...maxMs,
                step: 50
            ) {
                Text("\(draftConfig.streamSnapshotIntervalMs) ms")
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
        Task {
            do {
                try await backend.applyConfiguration(config, restartIfRunning: restart)
                await MainActor.run {
                    draftConfig = config
                    lastSyncedConfig = config
                }
            } catch {
                lastSaveError = "Apply failed: \(error)"
            }
            await MainActor.run { isApplying = false }
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
        case "qwen3_5", "qwen3_6":
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
                disabledReason: "KV quantization is not supported for Gemma."
            )
        case "step":
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: "KV quantization is not supported for Step."
            )
        default:
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_supported",
                disabledReason: "KV quantization is not supported for this model."
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
        mode == "off" ? "Off" : mode
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
            return String(format: "%.1fM", Double(value) / 1_000_000.0)
        }
        if value >= 1_000 {
            return "\(value / 1_000)K"
        }
        return "\(value)"
    }

    // MARK: - Hermes

    @ViewBuilder
    private var hermesToolTruthCard: some View {
        Card("Hermes", subtitle: "Agent handoff, tools, and gateway state.") {
            HStack(spacing: 8) {
                Button {
                    Task { await hermes.prepare(configuration: backend.configuration) }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
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
                    FormRow(label: "Tools") {
                        statusText(status.enabledToolsets.joined(separator: ", "))
                    }
                    if let version = status.versionSummary {
                        FormRow(label: "Version") { statusText(version) }
                    }
                    if let update = status.updateSummary {
                        FormRow(label: "Update") { statusText(update, color: Brand.warning) }
                    }
                    if let updateCommand = status.updateCommand, status.updateSummary != nil {
                        FormRow(label: "Command") { statusText(updateCommand) }
                    }
                    if let gateway = status.gatewaySummary {
                        FormRow(label: "Gateway") {
                            statusText(gateway, color: hermesGatewayColor(for: status.gatewayHealth))
                        }
                    }
                    if status.gatewayNeedsRepair {
                        FormRow(label: "Repair") {
                            Button {
                                Task { await hermes.repairGateway() }
                            } label: {
                                Label(
                                    hermes.gatewayRepairInFlight ? "Repairing" : "Repair Gateway",
                                    systemImage: "arrow.triangle.2.circlepath"
                                )
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(hermes.gatewayRepairInFlight)
                        }
                    }
                    if let repairMessage = hermes.gatewayRepairMessage {
                        FormRow(label: "Repair result") {
                            statusText(repairMessage)
                        }
                    }
                    ForEach(status.integrationSummaries.prefix(3), id: \.self) { item in
                        FormRow(label: "Messaging") { statusText(item) }
                    }
                    ForEach(status.warnings.prefix(2), id: \.self) { warning in
                        FormRow(label: "Warning") { statusText(warning, color: Brand.warning) }
                    }
                    FormRow(label: "Capability") {
                        statusText(status.capabilitySummary)
                    }
                } else {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Checking Hermes")
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
        Card("Thermal",
             subtitle: "Smart boosts fans only during generation; Max stays available for sustained benchmark runs.") {
            FanModeToggle()
        } content: {
            VStack(alignment: .leading, spacing: 10) {
                FormRow(
                    label: "Fan Mode",
                    caption: "Default uses Apple's curve. Smart boosts during requests. Max pins verified fans."
                ) {
                    Picker("Fan Mode", selection: fanModeSelectionBinding) {
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
                            text: verified ? "fan ramp verified" : "fan ramp pending",
                            systemImage: verified ? "checkmark.seal.fill" : "clock",
                            tint: verified ? Brand.success : Brand.warning,
                            emphasized: !verified
                        )
                    }
                } else if !backend.configuration.enableThermalPolling {
                    Text("Turn on Thermal Polling above to confirm fan state.")
                        .font(.caption)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                }
            }
        }
    }

    // MARK: - Admin

    @ViewBuilder
    private var adminCard: some View {
        Card("Reset", subtitle: "Can't be undone.") {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    pendingClearAll = true
                } label: {
                    HStack {
                        Label("Clear prompt cache", systemImage: "trash")
                        Spacer()
                        if clearingCache {
                            ProgressView().controlSize(.mini)
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                Text("To stop or restart the model, use the play button in the top bar.")
                    .font(.caption2)
                    .foregroundStyle(Brand.textHighlight.opacity(0.55))
            }
        }
    }

    // MARK: - About + Logs

    @ViewBuilder
    private var aboutAndLogsCard: some View {
        Card("Info") {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    router.presentAbout()
                } label: {
                    HStack {
                        Label("About MTPLX", systemImage: "info.circle")
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
                        Label("Open Logs (Cmd-Shift-L)", systemImage: "doc.text")
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
