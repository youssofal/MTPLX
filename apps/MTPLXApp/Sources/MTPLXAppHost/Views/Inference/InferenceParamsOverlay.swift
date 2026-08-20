import SwiftUI
import AppKit
import MTPLXAppCore

// MARK: - InferenceParamsOverlay
//
// Sibling to `LaunchOverlay` — same draw-in border + staggered row
// reveal, same outside-tap dismiss. Lets the user tune the inference
// knobs they care about most without leaving the dashboard:
//
//   - Performance Mode             (re-opens SSE stream, no restart)
//   - Temperature                  (live mutable, no restart)
//   - Top P                        (live mutable, no restart)
//   - Top K                        (live mutable, no restart)
//   - Reasoning Auto|On|Off        (live mutable, no restart)
//   - MTP heads (depth)            (live mutable, no restart)
//   - Prefill batch step size      (live mutable, applies to next request)
//   - Context window               (restart required, applies on next launch)
//   - KV quantization              (restart required)
//
// Sampling/depth/reasoning/prefill sliders commit on release: live
// settings update for the current daemon, plus a single persisted
// settings write so the value sticks across relaunches. The
// context-window slider and KV quantization picker commit through the
// Apply bar because they need a daemon restart, so the user can stage
// both at once before paying the warm-up cost. Performance Mode toggles
// immediately. There is no max-response-tokens control: every live
// commit clears `max_response_tokens` so the daemon never truncates a
// reply — generation only stops on EOS or when the context window is
// exhausted.
//
// Haptics: slider step crossings fire `NSHapticFeedbackManager` ticks.
// Temperature, Top P, Top K, and Prefill use `.alignment` (a micro click);
// MTP depth (only D1/D2/D3) uses `.levelChange` so the three positions
// feel like firm detents rather than continuous tuning.
//
// Hover: nothing in the popover paints a background highlight on
// hover. The only thing that animates is the *dial itself* — the
// `Slider` track + thumb, the segmented `Picker` pill, the `Toggle`
// switch — which floats up ~1.5pt with a soft shadow underneath.
// Label and value text never move. Adjacent dials lift independently
// because each owns its own hover state. No card. No fill. No scale.

struct InferenceParamsSnapshot: Equatable, Sendable {
    let configuration: MTPLXAppConfiguration
    let configuredModelFamily: String
    let settings: MutableSettings?
    let startupControls: ModelControls?
    let healthDepth: Int?
    let healthGenerationMode: String?
    let healthContextWindow: Int?
    let dashboardContextWindow: Int?
    let currentFanMode: String?

    @MainActor
    init(backend: MTPLXBackendStore, configuredModelFamily: String) {
        configuration = backend.configuration
        self.configuredModelFamily = configuredModelFamily
        settings = backend.settings
        startupControls = backend.health?.startup?.modelControls
        healthDepth = backend.health?.depth
        healthGenerationMode = backend.health?.generationMode
        healthContextWindow = backend.health?.contextWindow
        dashboardContextWindow = backend.snapshot?.contextWindow
        currentFanMode = backend.currentFanMode
    }
}

/// SwiftUI can revisit a sibling's body whenever the chat's AppKit bridge
/// requests a display pass. Keep that cheap outer pass from rebuilding the
/// entire settings control tree unless one of its actual inputs changed.
private struct EquatableViewBuilder<Key: Equatable & Sendable, Content: View>: View, Equatable {
    nonisolated let key: Key
    let content: () -> Content

    init(key: Key, @ViewBuilder content: @escaping () -> Content) {
        self.key = key
        self.content = content
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.key == rhs.key
    }

    var body: some View { content() }
}

struct InferenceParamsOverlay: View, Equatable {
    let backend: MTPLXBackendStore
    let snapshot: InferenceParamsSnapshot
    @EnvironmentObject private var themeStore: ThemeStore

    @Binding var presented: Bool
    private let presentedValue: Bool

    init(
        backend: MTPLXBackendStore,
        snapshot: InferenceParamsSnapshot,
        presented: Binding<Bool>
    ) {
        self.backend = backend
        self.snapshot = snapshot
        _presented = presented
        presentedValue = presented.wrappedValue
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.presentedValue == rhs.presentedValue && lhs.snapshot == rhs.snapshot
    }

    @State private var borderProgress: CGFloat = 0
    @State private var headerVisible: Bool = false
    @State private var rowsVisibleCount: Int = 0

    // Sampling drafts — live-mutable, applied immediately on slider
    // release.
    @State private var temperature: Double = 0.6
    @State private var topP: Double = 0.95
    @State private var topK: Int = 20
    @State private var presencePenalty: Double = 0
    @State private var depth: Int = 3

    // Reasoning draft — live-mutable. "auto" lets the daemon decide per
    // turn, "on" always reasons, "off" suppresses reasoning entirely.
    // Mirrors the wire field `MutableSettings.reasoning`.
    @State private var reasoningMode: String = "auto"
    @State private var reasoningEffort: String = "auto"
    @State private var fanMode: String = MTPLXFanMode.smart.rawValue

    // Prefill draft — live-mutable per request. Commits on slider
    // release (live settings + persisted config) like temperature, so
    // there is no apply-bar warning to make the user think prefill
    // requires a restart. It doesn't.
    @State private var prefillChunk: Int = 2048

    // Context window draft — restart required (daemon `--context-window`).
    // Slider runs 4 096 … model max, snaps to 1 024-token increments.
    @State private var contextWindow: Int = 16384
    @State private var contextWindowDirty: Bool = false

    @State private var kvQuantization: String = "off"
    @State private var kvDirty: Bool = false
    @State private var applying: Bool = false

    private struct PopoverRenderKey: Equatable, Sendable {
        let snapshot: InferenceParamsSnapshot
        let borderProgress: CGFloat
        let headerVisible: Bool
        let rowsVisibleCount: Int
        let temperature: Double
        let topP: Double
        let topK: Int
        let presencePenalty: Double
        let depth: Int
        let reasoningMode: String
        let reasoningEffort: String
        let fanMode: String
        let prefillChunk: Int
        let contextWindow: Int
        let contextWindowDirty: Bool
        let kvQuantization: String
        let kvDirty: Bool
        let applying: Bool
        let reduceMotion: Bool
    }

    private let popoverWidth: CGFloat = 340
    private let cornerRadius: CGFloat = 12
    // Strip layout right→left: [LaunchButton 32pt] · 8pt · [Params
    // 32pt] · 8pt · [Refresh 32pt] · 14pt right padding. Params
    // centre = right - 14 - 32 - 8 - 16 = right - 70. Notch sits
    // 8pt from popover's right edge, so rightOffset = 70 - 8 = 62.
    private let topOffset: CGFloat = 50
    private let rightOffset: CGFloat = 62

    var body: some View {
        ZStack(alignment: .topTrailing) {
            backdrop
            if presented {
                EquatableViewBuilder(key: popoverRenderKey) {
                    popoverColumn
                        .padding(.top, topOffset)
                        .padding(.trailing, rightOffset)
                        .transition(.identity)
                }
                .equatable()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        .allowsHitTesting(presented)
        .onChange(of: presented) { _, isOn in
            if isOn {
                runEnterChoreography()
                Task {
                    try? await backend.refreshLiveSettingsFromDaemon(persist: true)
                }
            } else {
                runExitChoreography()
            }
        }
        .onChange(of: snapshot.settings) { _, _ in
            guard presented else { return }
            seedDraftsFromCurrentState()
        }
        .onChange(of: snapshot.startupControls) { _, _ in
            guard presented else { return }
            seedDraftsFromCurrentState()
        }
    }

    private var popoverRenderKey: PopoverRenderKey {
        PopoverRenderKey(
            snapshot: snapshot,
            borderProgress: borderProgress,
            headerVisible: headerVisible,
            rowsVisibleCount: rowsVisibleCount,
            temperature: temperature,
            topP: topP,
            topK: topK,
            presencePenalty: presencePenalty,
            depth: depth,
            reasoningMode: reasoningMode,
            reasoningEffort: reasoningEffort,
            fanMode: fanMode,
            prefillChunk: prefillChunk,
            contextWindow: contextWindow,
            contextWindowDirty: contextWindowDirty,
            kvQuantization: kvQuantization,
            kvDirty: kvDirty,
            applying: applying,
            reduceMotion: themeStore.reduceMotionPreference
        )
    }

    // MARK: - Layers

    @ViewBuilder
    private var backdrop: some View {
        Color.clear
            .contentShape(Rectangle())
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onTapGesture { presented = false }
            .opacity(presented ? 1 : 0)
    }

    @ViewBuilder
    private var popoverColumn: some View {
        VStack(alignment: .trailing, spacing: 0) {
            notch
            popoverSurface
                .frame(width: popoverWidth)
        }
        .frame(width: popoverWidth, alignment: .trailing)
        .opacity(borderProgress)
        .scaleEffect(borderProgress > 0 ? 1 : 0.94, anchor: .topTrailing)
    }

    @ViewBuilder
    private var notch: some View {
        UpNotch()
            .fill(Brand.raisedSurface)
            .frame(width: 12, height: 7)
            .padding(.trailing, 8)
            .padding(.bottom, -1)
            .opacity(borderProgress > 0.3 ? 1 : 0)
    }

    @ViewBuilder
    private var popoverSurface: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header pinned at the top so the popover always identifies
            // itself; never scrolls away.
            header
                .padding(.top, 6)
            sectionDivider(precedesRow: 1)
            // Scrollable middle: every setting section lives here so the
            // popover can stay bounded vertically. macOS supplies the
            // hover-scrollbar automatically; we don't paint a fake one.
            ScrollView(.vertical, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 0) {
                    performanceModeRow
                    sectionDivider(precedesRow: 2)
                    samplingSection
                    sectionDivider(precedesRow: 3)
                    reasoningSection
                    sectionDivider(precedesRow: 4)
                    fanModeSection
                    sectionDivider(precedesRow: 5)
                    depthSection
                    sectionDivider(precedesRow: 6)
                    prefillSection
                    sectionDivider(precedesRow: 7)
                    contextWindowSection
                    sectionDivider(precedesRow: 8)
                    kvQuantizationSection
                }
                .padding(.bottom, 6)
            }
            .frame(maxHeight: scrollAreaMaxHeight)
            .scrollIndicators(.automatic)
            // Apply bar pinned at the bottom for the *only* two changes
            // that need a daemon restart. Live-mutable sliders
            // (temperature/topP/topK/depth/prefill) commit silently on
            // release — surfacing a "click Apply" call-to-action for
            // them would be a lie about what's actually pending.
            if kvDirty || contextWindowDirty {
                applyBar
            }
        }
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Brand.raisedSurface)
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .trim(from: 0, to: borderProgress)
                    .stroke(Brand.separatorStrong, lineWidth: 0.75)
            }
            .shadow(color: .black.opacity(0.55), radius: 18, x: 0, y: 10)
        )
    }

    /// Cap on the inner scroll area so the whole popover stays bounded
    /// — header + apply bar live outside it. Tuned to fit comfortably
    /// inside the default app window without ever clipping at the top
    /// or covering the bottom strip.
    private let scrollAreaMaxHeight: CGFloat = 440

    // MARK: - Sections

    @ViewBuilder
    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Settings")
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeBody)
            Text(modelControls?.displayName ?? fallbackDisplayName)
                .font(.caption2)
                .foregroundStyle(Brand.typeTertiary)
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .opacity(headerVisible ? 1 : 0)
        .offset(y: headerVisible ? 0 : -6)
    }

    @ViewBuilder
    private var samplingSection: some View {
        InferenceSection(visible: rowsVisibleCount > 1) {
            sectionHeader("SAMPLING")
            paramSlider(
                title: "Temperature",
                value: Binding(get: { temperature }, set: { temperature = $0 }),
                range: 0...Self.temperatureMax,
                step: 0.05,
                valueText: { v in
                    Text(v, format: .number.precision(.fractionLength(2)))
                },
                hapticPattern: .alignment,
                onCommit: { commitLiveSettings() }
            )
            paramSlider(
                title: "Top P",
                value: Binding(get: { topP }, set: { topP = $0 }),
                range: Self.topPMin...Self.topPMax,
                step: 0.05,
                valueText: { v in
                    Text(v, format: .number.precision(.fractionLength(2)))
                },
                hapticPattern: .alignment,
                onCommit: { commitLiveSettings() }
            )
            // Top K uses 0…200 step 5 so the tick density on the
            // slider matches Temperature (40 ticks). With step 1 over
            // 0…1000 the filled portion fused into a solid white bar.
            // 200 covers every practical Top K value: 0 (off), the
            // 20–50 coding band, 100/200 for creative sampling.
            // Values above 200 are rare and clamped on slider entry.
            paramSlider(
                title: "Top K",
                value: Binding(
                    get: { Double(topK) },
                    set: { topK = Int($0.rounded()) }
                ),
                range: 0...Double(Self.topKMax),
                step: 5,
                valueText: { v in
                    Text(Int(v.rounded()), format: .number)
                },
                hapticPattern: .alignment,
                onCommit: { commitLiveSettings() }
            )
            paramSlider(
                title: "Presence Penalty",
                value: Binding(get: { presencePenalty }, set: { presencePenalty = $0 }),
                range: 0...Self.presencePenaltyMax,
                step: 0.05,
                valueText: { v in
                    Text(v, format: .number.precision(.fractionLength(2)))
                },
                hapticPattern: .alignment,
                onCommit: { commitLiveSettings() }
            )
            Text("Discourages reusing tokens the reply already contains. 0 is exact and best for coding; try 0.5–1.5 for creative or repetitive output.")
                .font(.caption2)
                .foregroundStyle(Brand.typeTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var reasoningSection: some View {
        InferenceSection(visible: rowsVisibleCount > 2) {
            sectionHeader("REASONING")
            Picker("Reasoning", selection: Binding(
                get: { reasoningMode },
                set: { mode in
                    guard reasoningSupported else { return }
                    reasoningMode = mode
                    commitReasoning()
                }
            )) {
                Text("Auto").tag("auto")
                Text("On").tag("on")
                Text("Off").tag("off")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlHoverLift(motionEnabled: motionEnabled)
            .disabled(!reasoningSupported)
            if reasoningEffortSupported && reasoningMode != "off" {
                Picker("Reasoning effort", selection: Binding(
                    get: { reasoningEffort },
                    set: { effort in
                        reasoningEffort = normalizedReasoningEffort(effort)
                        commitReasoning()
                    }
                )) {
                    ForEach(reasoningEffortLevels, id: \.self) { effort in
                        Text(effort.capitalized).tag(effort)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlHoverLift(motionEnabled: motionEnabled)
            }
            Text(reasoningStatusCopy)
                .font(.caption2)
                .foregroundStyle(reasoningSupported ? Brand.typeTertiary : Brand.warning)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var fanModeSection: some View {
        InferenceSection(visible: rowsVisibleCount > 3) {
            sectionHeader("FAN MODE")
            Picker("Fan Mode", selection: Binding(
                get: { fanMode },
                set: { mode in
                    fanMode = MTPLXFanMode.normalized(mode).rawValue
                    commitFanMode()
                }
            )) {
                ForEach(MTPLXFanMode.allCases, id: \.self) { mode in
                    Text(mode.title).tag(mode.rawValue)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlHoverLift(motionEnabled: motionEnabled)
        }
    }

    /// Per-model speculative-depth descriptor reported by the daemon
    /// (`draft_control` / `depth_max`). Lets the slider adapt to the
    /// loaded model — Qwen/Step "MTP off + D1-D3", Gemma "Draft block"
    /// 2-8 — instead of being hardcoded to Qwen's D1-D3.
    private var configuredModelFamily: String {
        snapshot.configuredModelFamily
    }
    private var compatibleSettings: MutableSettings? {
        guard let settings = snapshot.settings else { return nil }
        let settingsFamily = settings.modelControls?.modelFamily ?? settings.modelFamily
        guard let settingsFamily else {
            return MTPLXModelOption.supportsTune(family: configuredModelFamily) ? settings : nil
        }
        return settingsFamily == configuredModelFamily ? settings : nil
    }
    private var compatibleStartupControls: ModelControls? {
        guard let controls = snapshot.startupControls else { return nil }
        if let modelRef = controls.modelRef {
            return MTPLXModelOption.modelsMatch(modelRef, snapshot.configuration.model)
                ? controls
                : nil
        }
        return controls.modelFamily == configuredModelFamily ? controls : nil
    }
    private var modelControls: ModelControls? {
        compatibleSettings?.modelControls ?? compatibleStartupControls
    }
    private var draftControl: DraftControl? {
        modelControls?.draftControl ?? compatibleSettings?.draftControl ?? fallbackDraftControl
    }
    private var reasoningPolicy: ReasoningPolicy? {
        modelControls?.reasoning ?? compatibleSettings?.reasoningPolicy ?? fallbackReasoningPolicy
    }
    private var kvQuantPolicy: KVQuantPolicy? {
        modelControls?.kvQuant ?? compatibleSettings?.kvQuantPolicy ?? fallbackKVQuantPolicy
    }
    private var contextWindowPolicy: ContextWindowPolicy? {
        modelControls?.contextWindow
            ?? compatibleSettings?.contextWindowPolicy
            ?? fallbackContextWindowPolicy
    }
    private var samplingDefaults: SamplingDefaults? {
        modelControls?.sampling ?? compatibleSettings?.samplingDefaults ?? fallbackSamplingDefaults
    }
    private var selectedModelFamily: String {
        modelControls?.modelFamily
            ?? compatibleSettings?.modelFamily
            ?? configuredModelFamily
    }
    private var compatibleConfigurationReasoning: String? {
        let family = selectedModelFamily
        if let storedFamily = snapshot.configuration.liveSettingsModelFamily,
           !storedFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return storedFamily == family ? snapshot.configuration.reasoning : nil
        }
        return MTPLXModelOption.supportsTune(family: family) ? snapshot.configuration.reasoning : nil
    }
    private var compatibleConfigurationGenerationMode: String? {
        let family = selectedModelFamily
        let mode = normalizedGenerationMode(snapshot.configuration.generationMode)
        if let storedFamily = snapshot.configuration.liveSettingsModelFamily,
           !storedFamily.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return storedFamily == family ? mode : nil
        }
        // Legacy settings files came from the old Qwen-only era where
        // `mtp` was just the global default. Preserve a legacy explicit
        // AR/off choice, but don't let unversioned `mtp` force Qwen/Step
        // depth controls to open on D1.
        return MTPLXModelOption.supportsTune(family: family) && mode == "ar" ? mode : nil
    }
    private var compatibleConfigurationTunedDraftValue: Int? {
        guard let field = draftControl?.requestField else { return nil }
        if field == "depth" {
            return snapshot.configuration.compatibleTunedDepth()
        }
        return snapshot.configuration.compatibleTunedControlValue(controlField: field)
    }
    private var defaultGenerationModeForSelectedControl: String {
        depthControlSupportsMtpOff ? "ar" : "mtp"
    }
    private var selectedLaunchTarget: LaunchTarget {
        LaunchTarget(rawValue: snapshot.configuration.lastLaunchTarget) ?? .chat
    }
    private var launchDefaultReasoning: String? {
        MTPLXCommandBuilder.defaultReasoningMode(for: selectedLaunchTarget)
    }
    private var fallbackDisplayName: String {
        switch selectedModelFamily {
        case "gemma4": return "Gemma assistant MTP"
        case "step": return "Step experimental MTP"
        case "qwen3_5", "qwen3_6", "qwen3_8": return "Qwen native MTP"
        case "glm": return "GLM MTP"
        case "deepseek": return "DeepSeek MTP"
        default: return "Custom model"
        }
    }
    private var fallbackSamplingDefaults: SamplingDefaults {
        switch selectedModelFamily {
        case "gemma4":
            return SamplingDefaults(
                temperature: 1.0,
                topP: 0.95,
                topK: 64,
                familyDefaultReason: "Gemma sampler defaults"
            )
        case "step":
            return SamplingDefaults(
                temperature: 0.6,
                topP: 0.95,
                topK: 20,
                familyDefaultReason: "Step sampler defaults"
            )
        case "qwen3_8":
            return SamplingDefaults(
                temperature: 1.0,
                topP: 0.95,
                topK: 20,
                familyDefaultReason: "Qwen 3.8 native sampler"
            )
        default:
            return SamplingDefaults(
                temperature: 0.6,
                topP: 0.95,
                topK: 20,
                familyDefaultReason: "Qwen coding sampler"
            )
        }
    }
    private var fallbackDraftControl: DraftControl {
        switch selectedModelFamily {
        case "gemma4":
            return DraftControl(
                supported: true,
                requestField: "draft_block_size",
                displayLabel: "Draft block",
                defaultValue: 6,
                minimum: 2,
                maximum: 8,
                unit: "block",
                valueLabels: (2...8).map { "Block \($0)" }
            )
        case "unknown":
            return DraftControl(
                supported: false,
                requestField: nil,
                displayLabel: "Draft control",
                defaultValue: nil,
                minimum: 1,
                maximum: 1,
                unit: "depth"
            )
        case "step":
            return DraftControl(
                supported: true,
                requestField: "depth",
                displayLabel: "Draft depth",
                defaultValue: 1,
                minimum: 1,
                maximum: 3,
                unit: "depth",
                valueLabels: ["D1", "D2", "D3"]
            )
        default:
            return DraftControl(
                supported: true,
                requestField: "depth",
                displayLabel: "Draft depth",
                defaultValue: 3,
                minimum: 1,
                maximum: 3,
                unit: "depth",
                valueLabels: ["D1", "D2", "D3"]
            )
        }
    }
    private var fallbackReasoningPolicy: ReasoningPolicy {
        switch selectedModelFamily {
        case "gemma4":
            return ReasoningPolicy(
                supported: true,
                parser: "gemma4",
                defaultMode: "auto",
                historyPolicy: "preserve_when_enabled"
            )
        case "qwen3_8":
            return ReasoningPolicy(
                supported: true,
                parser: "qwen3",
                defaultMode: "auto",
                historyPolicy: "preserve_when_enabled",
                effortLevels: ["xhigh", "medium", "low"],
                defaultEffort: "medium"
            )
        case "step", "unknown":
            if selectedModelFamily == "step" {
                return ReasoningPolicy(
                    supported: true,
                    parser: "step3p5",
                    defaultMode: "auto",
                    historyPolicy: "preserve_when_enabled",
                    effortLevels: ["low", "medium", "high"],
                    defaultEffort: "low"
                )
            }
            return ReasoningPolicy(
                supported: false,
                parser: "none",
                modes: [],
                defaultMode: "off",
                historyPolicy: "visible_content_only"
            )
        default:
            return ReasoningPolicy(
                supported: true,
                parser: "qwen3",
                defaultMode: "auto",
                historyPolicy: "preserve_when_enabled"
            )
        }
    }
    private var fallbackKVQuantPolicy: KVQuantPolicy {
        switch selectedModelFamily {
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
                proofLevel: "not_validated",
                disabledReason: "KV quantization is not supported for Gemma."
            )
        case "step":
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_validated",
                disabledReason: "KV quantization is not supported for Step."
            )
        default:
            return KVQuantPolicy(
                supported: false,
                modes: ["off"],
                restartRequired: true,
                proofLevel: "not_validated",
                disabledReason: "KV quantization is not supported for this model."
            )
        }
    }
    private var fallbackContextWindowPolicy: ContextWindowPolicy {
        ContextWindowPolicy(
            supported: true,
            minimum: Self.contextWindowMin,
            maximum: MTPLXModelOption.maxContextWindow(forFamily: selectedModelFamily),
            defaultValue: MTPLXModelOption.maxContextWindow(forFamily: selectedModelFamily),
            step: 1024,
            source: "app_catalog",
            unit: "tokens"
        )
    }
    private var reasoningSupported: Bool {
        reasoningPolicy?.supported ?? true
    }
    private var reasoningEffortLevels: [String] {
        let raw = reasoningPolicy?.effortLevels ?? []
        var seen = Set<String>()
        return raw.compactMap { rawLevel in
            let level = rawLevel.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            guard ["low", "medium", "high", "xhigh"].contains(level), !seen.contains(level) else {
                return nil
            }
            seen.insert(level)
            return level
        }
    }
    private var reasoningEffortSupported: Bool {
        reasoningSupported && !reasoningEffortLevels.isEmpty
    }
    private var reasoningStatusCopy: String {
        guard reasoningSupported else {
            return "Reasoning is not supported for \(contextWindowModelLabel)."
        }
        if reasoningEffortSupported && reasoningMode != "off" {
            return "Reasoning effort: \(reasoningEffort.capitalized)."
        }
        return Self.reasoningHint(for: reasoningMode)
    }
    private var draftControlSupported: Bool {
        draftControl?.supported ?? true
    }
    private var kvQuantSupported: Bool {
        kvQuantPolicy?.supported ?? true
    }
    private var kvQuantModes: [String] {
        let modes = kvQuantPolicy?.modes.filter { ["off", "q8", "q4"].contains($0) }
        return (modes?.isEmpty == false) ? modes! : ["off", "q8", "q4"]
    }
    private var depthControlSupportsMtpOff: Bool {
        draftControlSupported
            && (draftControl?.unit ?? "depth") == "depth"
            && (draftControl?.requestField ?? "depth") == "depth"
    }
    private var depthMin: Int {
        depthControlSupportsMtpOff ? 0 : max(1, draftControl?.minimum ?? 1)
    }
    private var depthMax: Int {
        max(depthMin, compatibleSettings?.depthMax ?? draftControl?.maximum ?? 3)
    }
    /// A slider needs at least two distinct values; SwiftUI's
    /// `Normalizing` traps on a zero-width interval. Unsupported models
    /// and descriptors whose minimum equals maximum must not construct
    /// a `Slider` at all — `.disabled` is applied too late to help.
    private var depthSliderRange: ClosedRange<Int>? {
        guard draftControlSupported, depthMax > depthMin else { return nil }
        return depthMin...depthMax
    }
    private var depthDefault: Int {
        min(depthMax, max(depthMin, draftControl?.defaultValue ?? depthMax))
    }
    private var depthValuePrefix: String {
        draftControl?.unit == "block" ? "Block " : "D"
    }
    private var draftLabelBase: Int {
        max(1, draftControl?.minimum ?? 1)
    }
    private func draftValueLabel(for value: Int) -> String {
        if depthControlSupportsMtpOff && value <= 0 {
            return "MTP off"
        }
        if let labels = draftControl?.valueLabels {
            let index = value - draftLabelBase
            if labels.indices.contains(index) {
                return labels[index]
            }
        }
        return "\(depthValuePrefix)\(value)"
    }

    @ViewBuilder
    private var depthSection: some View {
        InferenceSection(visible: rowsVisibleCount > 4) {
            sectionHeader(draftControl?.displayLabel?.uppercased() ?? "MTP HEADS")
            // Range/label/unit come from the loaded backend's draft-control
            // descriptor, so a model with more MTP heads (e.g. Gemma's
            // draft blocks 2-8) is no longer clamped to Qwen's D1-D3. Each
            // detent is a real structural change to the speculative-decode
            // pipeline, so the haptic stays a firm `.levelChange`.
            if let sliderRange = depthSliderRange {
                paramSlider(
                    title: draftControl?.displayLabel ?? "Depth",
                    value: Binding(
                        get: { Double(depth) },
                        set: {
                            guard draftControlSupported else { return }
                            depth = Int($0.rounded())
                        }
                    ),
                    range: Double(sliderRange.lowerBound)...Double(sliderRange.upperBound),
                    step: 1,
                    valueText: { v in
                        Text(draftValueLabel(for: Int(v.rounded())))
                    },
                    hapticPattern: .levelChange,
                    onCommit: { if draftControlSupported { commitLiveSettings() } }
                )
            } else if draftControlSupported {
                // Supported but only one valid value — show it, don't
                // build a degenerate slider.
                HStack {
                    Text(draftControl?.displayLabel ?? "Depth")
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeBody)
                    Spacer()
                    Text(draftValueLabel(for: depthMin))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                        .monospacedDigit()
                }
            } else {
                Text("Draft control is not available for this model.")
                    .font(.caption2)
                    .foregroundStyle(Brand.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private var prefillSection: some View {
        InferenceSection(visible: rowsVisibleCount > 5) {
            sectionHeader("PREFILL", hint: "next request")
            paramSlider(
                title: "Batch step size",
                value: Binding(
                    get: { Double(prefillChunk) },
                    set: { prefillChunk = Int($0.rounded()) }
                ),
                range: 256...32768,
                step: 256,
                valueText: { v in
                    Text(Int(v.rounded()), format: .number) + Text(" tok")
                },
                hapticPattern: .alignment,
                onCommit: { commitPrefill() }
            )
            prefillPresetChips
        }
    }

    @ViewBuilder
    private var contextWindowSection: some View {
        InferenceSection(visible: rowsVisibleCount > 6) {
            sectionHeader("CONTEXT WINDOW", hint: "restart")
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Window")
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeBody)
                    Spacer()
                    Text(Self.formatTokensVerbose(contextWindow))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                        .monospacedDigit()
                }
                if let sliderRange = contextWindowSliderRange {
                    Slider(
                        value: Binding(
                            get: { Double(contextWindow) },
                            set: { newValue in
                                let clamped = clampContextWindow(Int(newValue.rounded()))
                                if clamped != contextWindow {
                                    Haptics.tick(.alignment)
                                }
                                contextWindow = clamped
                                contextWindowDirty = clamped != currentContextWindow
                            }
                        ),
                        in: Double(sliderRange.lowerBound)...Double(sliderRange.upperBound),
                        step: 1024
                    )
                    .tint(Brand.typeBody)
                    .controlHoverLift(motionEnabled: motionEnabled)
                }
            }
            contextPresetChips
            Text("Max for \(contextWindowModelLabel): \(Self.formatTokensVerbose(modelMaxContext)).")
                .font(.caption2)
                .foregroundStyle(Brand.typeTertiary)
                .fixedSize(horizontal: false, vertical: true)
            Text("Responses run until the model stops or fills its context. No length cap.")
                .font(.caption2)
                .foregroundStyle(Brand.typeTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var contextPresetChips: some View {
        // Three well-spaced presets cover the meaningful jumps —
        // short chat, long doc, near-max — and `Max` adapts to whatever
        // the loaded model actually supports. The slider handles
        // everything in between, so we don't crowd the row with chips.
        let presets = Self.contextPresets.filter { $0 < modelMaxContext }
        HStack(spacing: 6) {
            ForEach(presets, id: \.self) { preset in
                contextChip(label: Self.formatTokensShort(preset), isOn: contextWindow == preset) {
                    contextWindow = preset
                    contextWindowDirty = preset != currentContextWindow
                }
            }
            contextChip(label: "Max", isOn: contextWindow == modelMaxContext) {
                contextWindow = modelMaxContext
                contextWindowDirty = modelMaxContext != currentContextWindow
            }
            Spacer(minLength: 0)
        }
        .padding(.top, 4)
    }

    @ViewBuilder
    private func contextChip(label: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(isOn ? Brand.bgOuter : Brand.typeBody)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule(style: .continuous)
                        .fill(isOn ? AnyShapeStyle(Brand.typeBody) : AnyShapeStyle(Color.clear))
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.5)
                        )
                )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var prefillPresetChips: some View {
        HStack(spacing: 6) {
            ForEach([2048, 4096, 8192, 16384], id: \.self) { preset in
                Button {
                    guard prefillChunk != preset else { return }
                    prefillChunk = preset
                    Haptics.tick(.alignment)
                    commitPrefill()
                } label: {
                    Text("\(preset)")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(prefillChunk == preset ? Brand.bgOuter : Brand.typeBody)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule(style: .continuous)
                                .fill(prefillChunk == preset
                                      ? AnyShapeStyle(Brand.typeBody)
                                      : AnyShapeStyle(Color.clear))
                                .overlay(
                                    Capsule(style: .continuous)
                                        .stroke(Brand.separator, lineWidth: 0.5)
                                )
                        )
                }
                .buttonStyle(.plain)
            }
            Spacer(minLength: 0)
        }
        .padding(.top, 4)
    }

    @ViewBuilder
    private var performanceModeRow: some View {
        InferenceSection(visible: rowsVisibleCount > 0) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Performance mode")
                        .font(.system(.callout))
                        .foregroundStyle(Brand.typeBody)
                    Text("Calms the UI so it doesn't slow down the model. Turn on for accurate benchmarks.")
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 12)
                Toggle("", isOn: performanceLockBinding)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlHoverLift(motionEnabled: motionEnabled)
            }
        }
    }

    @ViewBuilder
    private var kvQuantizationSection: some View {
        InferenceSection(visible: rowsVisibleCount > 7) {
            sectionHeader("KV QUANTIZATION", hint: "restart")
            Picker("KV quantization", selection: Binding(
                get: { kvQuantization },
                set: { mode in
                    guard kvQuantSupported else { return }
                    if mode != kvQuantization {
                        Haptics.tick(.levelChange)
                    }
                    kvQuantization = mode
                    kvDirty = mode != currentKVQuantization
                }
            )) {
                ForEach(kvQuantModes, id: \.self) { mode in
                    Text(Self.kvQuantLabel(mode)).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlHoverLift(motionEnabled: motionEnabled)
            .disabled(!kvQuantSupported)
            if !kvQuantSupported {
                Text(kvQuantPolicy?.disabledReason ?? "KV quantization is not supported for this model.")
                    .font(.caption2)
                    .foregroundStyle(Brand.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private var applyBar: some View {
        Divider().overlay(Brand.separator)
        HStack(spacing: 8) {
            Text(applyBarMessage)
                .font(.caption2)
                .foregroundStyle(Brand.warning)
            Spacer()
            Button("Revert") {
                kvQuantization = currentKVQuantization
                kvDirty = false
                contextWindow = currentContextWindow
                contextWindowDirty = false
            }
            .buttonStyle(.borderless)
            .disabled(applying)
            Button("Apply") {
                applyPendingChanges()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .tint(Brand.accentChrome)
            .disabled(applying)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    /// Apply bar only ever surfaces restart-required settings (KV
    /// quantization + context window). Prefill is live-mutable and
    /// commits silently on slider release, so we don't lie to the user
    /// about it needing an Apply step.
    private var applyBarMessage: String {
        let restartParts: [String] = [
            kvDirty ? "KV quantization" : nil,
            contextWindowDirty ? "context window" : nil,
        ].compactMap { $0 }
        let joined = restartParts.joined(separator: " and ")
        return "Applying \(joined) restarts the engine."
    }

    // MARK: - Helpers

    @ViewBuilder
    private func sectionHeader(_ title: String, hint: String? = nil) -> some View {
        HStack {
            Text(title)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.6)
                .foregroundStyle(Brand.typeSecondary)
            if let hint {
                Text("\u{00B7} \(hint)")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(Brand.warning.opacity(0.85))
            }
            Spacer()
        }
        .padding(.bottom, 2)
    }

    /// Build a labelled slider. The label + value text are static; the
    /// `controlHoverLift` modifier is applied *only* to the Slider
    /// itself so the dial floats up on hover and the surrounding text
    /// stays put. If `hapticPattern` is supplied, every step crossing
    /// during a drag fires that haptic on the trackpad — `.alignment`
    /// for micro tuning (temperature, top K, prefill), `.levelChange`
    /// for big detents (depth, KV quantization).
    ///
    /// The caller supplies the value formatter as a `(Double) -> Text`
    /// closure so each slider can route through swiftui-pro
    /// `Text(value, format: .number.precision(.fractionLength(N)))`
    /// instead of the C-style `String(format: "%.2f", value)` the V0
    /// helper used (per swiftui-pro swift.md).
    @ViewBuilder
    private func paramSlider(
        title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        valueText: @escaping (Double) -> Text,
        hapticPattern: NSHapticFeedbackManager.FeedbackPattern? = nil,
        onCommit: (() -> Void)?
    ) -> some View {
        // Wrap the source binding so the setter fires a haptic
        // whenever the step-snapped value actually changes. The Slider
        // already snaps to the step grid, so a `!=` check after the
        // snap is exactly one tick per step crossing.
        let hapticValue = Binding<Double>(
            get: { value.wrappedValue },
            set: { newValue in
                if let pattern = hapticPattern, newValue != value.wrappedValue {
                    Haptics.tick(pattern)
                }
                value.wrappedValue = newValue
            }
        )
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.system(size: 12))
                    .foregroundStyle(Brand.typeBody)
                Spacer()
                valueText(value.wrappedValue)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                    .monospacedDigit()
            }
            Slider(
                value: hapticValue,
                in: range,
                step: step,
                onEditingChanged: { editing in
                    if !editing { onCommit?() }
                }
            )
            .tint(Brand.typeBody)
            .controlHoverLift(motionEnabled: motionEnabled)
        }
    }

    @ViewBuilder
    private func sectionDivider(precedesRow row: Int) -> some View {
        let visible = rowsVisibleCount > row
        Rectangle()
            .fill(Brand.separator)
            .frame(height: 0.5)
            .scaleEffect(x: visible ? 1 : 0, y: 1, anchor: .leading)
            .opacity(visible ? 1 : 0)
    }

    private var performanceLockBinding: Binding<Bool> {
        Binding(
            get: { snapshot.configuration.performanceLock },
            set: { newValue in
                var config = backend.configuration
                config.performanceLock = newValue
                try? backend.saveSettings(config)
                backend.startMetricsStream()
            }
        )
    }

    // MARK: - Commit

    private func commitLiveSettings() {
        let draft = currentLiveSettingsDraft()
        Task {
            try? await backend.updateLiveSettings(draft)
        }
    }

    private func commitReasoning() {
        let draft = currentLiveSettingsDraft()
        Task {
            try? await backend.updateLiveSettings(draft)
        }
    }

    private func commitFanMode() {
        let canonical = MTPLXFanMode.normalized(fanMode)
        Task {
            try? await backend.setFanMode(canonical.rawValue)
        }
    }

    /// Prefill is live-mutable: the daemon picks up the new chunk size
    /// on the next request without a restart. We persist the same
    /// value into `configuration` so the change sticks across
    /// relaunches — one disk write per drag (on slider release), not
    /// per step. Failure to persist or push live silently no-ops; the
    /// next slider release retries.
    private func commitPrefill() {
        let live = currentLiveSettingsDraft()

        var config = backend.configuration
        config.prefillChunkTokens = prefillChunk

        try? backend.saveSettings(config)
        Task {
            try? await backend.updateLiveSettings(live)
        }
    }

    private func currentLiveSettingsDraft() -> MutableSettings {
        var draft = compatibleSettings ?? MutableSettings()
        draft.temperature = temperature
        draft.topP = topP
        draft.topK = topK
        draft.presencePenalty = presencePenalty
        if depthControlSupportsMtpOff {
            if depth <= 0 {
                draft.generationMode = "ar"
                draft.depth = nil
            } else {
                draft.generationMode = "mtp"
                draft.depth = min(depthMax, max(draftLabelBase, depth))
            }
        } else {
            draft.depth = min(depthMax, max(depthMin, depth))
        }
        if reasoningSupported {
            draft.reasoning = reasoningMode
            if reasoningEffortSupported {
                draft.reasoningEffort = reasoningEffort
            }
            draft.enableThinking = ChatReasoningPolicy.enableThinking(
                explicitMode: reasoningMode,
                modelControls: modelControls,
                modelFamily: selectedModelFamily
            )
        } else {
            draft.reasoning = nil
            draft.reasoningEffort = nil
            draft.enableThinking = nil
        }
        draft.prefillChunkTokens = prefillChunk
        // Force the response-length cap to null on every live commit so the
        // daemon never truncates a reply. Generation stops on EOS or when
        // the context window is exhausted — nothing in between.
        draft.maxResponseTokens = nil
        return draft
    }

    private func applyPendingChanges() {
        applying = true
        var config = backend.configuration
        if kvDirty {
            guard kvQuantSupported else {
                kvDirty = false
                applying = false
                return
            }
            config.pagedKVQuantization = kvQuantization
        }
        if contextWindowDirty {
            config.contextWindow = contextWindow > 0 ? contextWindow : nil
            config.contextWindowModelFamily = selectedModelFamily
        }
        // Apply bar only fires for restart-required changes. KV
        // quantization and context window both bake into the daemon
        // launch args, so we always restart when one of them is dirty.
        let needsRestart = kvDirty || contextWindowDirty
        Task {
            var success = false
            do {
                try await backend.applyConfiguration(config, restartIfRunning: needsRestart)
                success = true
            } catch {
                // Keep the dirty state visible so the user can retry.
            }
            await MainActor.run {
                kvDirty = !success && kvDirty
                contextWindowDirty = !success && contextWindowDirty
                applying = false
            }
        }
    }

    // MARK: - Choreography

    private var motionEnabled: Bool {
        !snapshot.configuration.performanceLock && !themeStore.reduceMotionPreference
    }

    private func runEnterChoreography() {
        seedDraftsFromCurrentState()

        OverlayChoreography.runEnter(
            motionEnabled: motionEnabled,
            rowCount: 8,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount
        )
    }

    private func seedDraftsFromCurrentState() {
        // Seed drafts from current state when the popover opens. Top K
        // is clamped into the slider's 0…200 visible range so a
        // historical value of e.g. 500 (set when the slider used the
        // 0…1000 range) snaps to the slider's max instead of falling
        // off-screen with an invisible thumb.
        let settings = compatibleSettings
        temperature = clampTemperature(settings?.temperature ?? samplingDefaults?.temperature ?? 0.6)
        topP = clampTopP(settings?.topP ?? samplingDefaults?.topP ?? 0.95)
        topK = clampTopK(settings?.topK ?? samplingDefaults?.topK ?? 20)
        presencePenalty = clampPresencePenalty(settings?.presencePenalty ?? 0)
        let liveDepth = compatibleStartupControls == nil ? nil : snapshot.healthDepth
        let tunedDraftValue = compatibleConfigurationTunedDraftValue
        let generationMode = normalizedGenerationMode(
            settings?.generationMode
                ?? (compatibleStartupControls == nil ? nil : snapshot.healthGenerationMode)
                ?? compatibleConfigurationGenerationMode
                ?? (tunedDraftValue == nil ? nil : "mtp")
                ?? defaultGenerationModeForSelectedControl
        )
        if depthControlSupportsMtpOff && generationMode == "ar" {
            depth = 0
        } else {
            depth = min(depthMax, max(draftLabelBase, settings?.depth ?? liveDepth ?? tunedDraftValue ?? depthDefault))
        }
        reasoningMode = normalizedReasoningMode(
            settings?.reasoning
                ?? compatibleConfigurationReasoning
                ?? reasoningPolicy?.defaultMode
                ?? launchDefaultReasoning
        )
        reasoningEffort = normalizedReasoningEffort(
            settings?.reasoningEffort ?? reasoningPolicy?.defaultEffort
        )
        fanMode = MTPLXFanMode.normalized(
            snapshot.currentFanMode ?? snapshot.configuration.fanMode
        ).rawValue
        prefillChunk = currentPrefillChunk
        contextWindow = currentContextWindow
        contextWindowDirty = false
        kvQuantization = kvQuantSupported ? currentKVQuantization : "off"
        kvDirty = false
    }

    private func clampTopK(_ value: Int) -> Int {
        max(0, min(Self.topKMax, value))
    }

    private func clampTopP(_ value: Double) -> Double {
        max(Self.topPMin, min(Self.topPMax, value))
    }

    private func clampTemperature(_ value: Double) -> Double {
        max(0, min(Self.temperatureMax, value))
    }

    private func clampPresencePenalty(_ value: Double) -> Double {
        max(0, min(Self.presencePenaltyMax, value))
    }

    private static func kvQuantLabel(_ mode: String) -> String {
        switch mode {
        case "q8": return "q8"
        case "q4": return "q4"
        default: return "Off"
        }
    }

    private func runExitChoreography() {
        OverlayChoreography.runExit(
            motionEnabled: motionEnabled,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount
        )
    }

    private var currentPrefillChunk: Int {
        compatibleSettings?.prefillChunkTokens
            ?? snapshot.configuration.prefillChunkTokens
            ?? 2048
    }

    private var currentKVQuantization: String {
        guard kvQuantSupported else { return "off" }
        switch snapshot.configuration.pagedKVQuantization {
        case "q8", "q4":
            return snapshot.configuration.pagedKVQuantization
        default:
            return "off"
        }
    }

    /// What the user thinks of as the "currently in-use" context window.
    /// Falls back through: compatible explicit persisted launch arg →
    /// compatible daemon value → the selected model's max. We never show
    /// stale context from a different model family.
    private var currentContextWindow: Int {
        if let explicit = compatibleConfigurationContextWindow {
            return clampContextWindow(explicit)
        }
        let resolved = compatibleHealthContextWindow ?? modelMaxContext
        return clampContextWindow(resolved)
    }

    /// Upper bound for the slider. Comes from `model_controls` when a
    /// daemon is live and falls back to the app catalog for stopped-state
    /// model switches.
    private var modelMaxContext: Int {
        let reported = contextWindowPolicy?.maximum
            ?? MTPLXModelOption.maxContextWindow(forFamily: selectedModelFamily)
        return max(Self.contextWindowMin, reported)
    }

    /// Same contract as `depthSliderRange`: a `Slider` needs two distinct
    /// detents, and SwiftUI's `Normalizing` traps on a zero-width interval.
    /// A model whose window is pinned at the floor keeps the value row and
    /// the `Max` chip; there is simply nothing to slide.
    private var contextWindowSliderRange: ClosedRange<Int>? {
        guard modelMaxContext > Self.contextWindowMin else { return nil }
        return Self.contextWindowMin...modelMaxContext
    }

    private var compatibleConfigurationContextWindow: Int? {
        snapshot.configuration.compatibleContextWindowOverride()
    }

    private var compatibleHealthContextWindow: Int? {
        guard compatibleStartupControls != nil else { return nil }
        return snapshot.healthContextWindow ?? snapshot.dashboardContextWindow
    }

    private var contextWindowModelLabel: String {
        switch selectedModelFamily {
        case "gemma4": return "Gemma"
        case "step": return "Step"
        case "qwen3_5", "qwen3_6", "qwen3_8": return "Qwen"
        case "glm": return "GLM"
        case "deepseek": return "DeepSeek"
        default: return "this model"
        }
    }

    private func clampContextWindow(_ value: Int) -> Int {
        let snapped = Int((Double(value) / 1024.0).rounded()) * 1024
        return max(Self.contextWindowMin, min(modelMaxContext, snapped))
    }

    private func normalizedReasoningMode(_ raw: String?) -> String {
        switch raw?.lowercased() {
        case "on": return "on"
        case "off": return "off"
        case "auto": return "auto"
        default: return "auto"
        }
    }

    private func normalizedGenerationMode(_ raw: String?) -> String {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "ar": return "ar"
        default: return "mtp"
        }
    }

    private func normalizedReasoningEffort(_ raw: String?) -> String {
        let levels = reasoningEffortLevels
        let fallback = reasoningPolicy?.defaultEffort ?? levels.first ?? "auto"
        let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case .some(let effort) where ["low", "medium", "high", "xhigh"].contains(effort):
            return levels.contains(effort) ? effort : fallback
        case "auto":
            return levels.contains(fallback) ? fallback : levels.first ?? "auto"
        default:
            return levels.contains(fallback) ? fallback : levels.first ?? "auto"
        }
    }

    // MARK: - Static utilities

    static let contextWindowMin: Int = 4096
    static let contextWindowFallbackMax: Int = 262_144
    /// Three meaningful jumps. `Max` is rendered separately and always
    /// snaps to the loaded model's real ceiling, so we don't list
    /// 256k explicitly here — it would just duplicate the Max chip.
    static let contextPresets: [Int] = [8192, 32768, 131072]

    /// Top K slider upper bound. Chosen so the filled portion of the
    /// track never blurs into a solid white bar the way it did with
    /// the previous 0…1000 range. Top K above 200 is rare; we clamp
    /// on overlay open so historical settings still land on the
    /// slider.
    static let topKMax: Int = 200

    /// Nucleus sampling must stay > 0 for the server sampler. The lower bound
    /// is low enough for focused runs, high enough that a drag cannot commit an
    /// invalid value.
    static let topPMin: Double = 0.05
    static let topPMax: Double = 1.0

    /// Temperature slider upper bound. Capped at 1.0 because nothing
    /// above 1.0 produces useful text from this class of model — past
    /// ~1.2 the distribution is essentially uniform and the output is
    /// noise. Historical settings above 1.0 are clamped on overlay
    /// open so the thumb never pins off-screen with a stale value
    /// label.
    static let temperatureMax: Double = 1.0

    /// OpenAI's documented presence_penalty range is [-2, 2]; the dial
    /// exposes the useful positive half (0 = off/exact, up to 2 = max
    /// anti-repetition). Negative values encourage repetition, which no
    /// product flow wants from a slider.
    static let presencePenaltyMax: Double = 2.0

    private static func reasoningHint(for mode: String) -> String {
        switch mode {
        case "on":
            return "Always reasons before answering. Best quality, more tokens."
        case "off":
            return "Skips reasoning. Fastest replies, weaker on hard prompts."
        default:
            return "Model decides per turn based on prompt difficulty."
        }
    }

    private static func formatTokensVerbose(_ tokens: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        let formatted = formatter.string(from: NSNumber(value: tokens)) ?? "\(tokens)"
        return "\(formatted) tok"
    }

    private static func formatTokensShort(_ tokens: Int) -> String {
        if tokens >= 1024 && tokens % 1024 == 0 {
            return "\(tokens / 1024)k"
        }
        return "\(tokens)"
    }
}

// MARK: - InferenceSection
//
// Pure layout container for a single section in the inference popover.
// Holds the section header + the section's body. *Never* lifts and
// *never* paints a background highlight on hover — the section is
// structural, not interactive. Hover lift belongs to the widget the
// cursor is over (see `controlHoverLift`). The choreographed entrance
// (opacity + 8pt drop) rides at the section level because each section
// is one beat in the staggered reveal.

private struct InferenceSection<Content: View>: View {
    let visible: Bool
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            content
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .opacity(visible ? 1 : 0)
        .offset(y: visible ? 0 : 8)
    }
}

// MARK: - controlHoverLift
//
// Modifier applied *directly to the dial itself* — the `Slider`, the
// `Picker`, the `Toggle`. Pure translation + shadow, no background
// fill, no scale, no card. The widget rises 1.5pt with a soft shadow
// underneath so it reads as "this control floated off the surface".
// Surrounding label text and value text aren't wrapped, so they don't
// move when the cursor enters/exits — exactly the mental model the
// user asked for ("only the dials themselves").

extension View {
    func controlHoverLift(motionEnabled: Bool) -> some View {
        modifier(ControlHoverLift(motionEnabled: motionEnabled))
    }
}

private struct ControlHoverLift: ViewModifier {
    let motionEnabled: Bool
    @State private var hovering = false

    func body(content: Content) -> some View {
        content
            .offset(y: hovering ? Motion.controlHoverOffsetY : 0)
            .shadow(
                color: .black.opacity(hovering ? Motion.controlHoverShadowOpacity : 0),
                radius: hovering ? Motion.controlHoverShadowRadius : 0,
                x: 0,
                y: hovering ? Motion.controlHoverShadowYOffset : 0
            )
            .animation(motionEnabled ? Motion.controlHoverSpring : nil, value: hovering)
            .onHover { hovering = $0 }
    }
}

// MARK: - UpNotch (12pt triangle pointing UP toward the trigger button)

private struct UpNotch: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
