import SwiftUI
import MTPLXAppCore

// MARK: - LiveTab
//
// V1 main page. Single hero gauge (morphs TPS ↔ Prefill) + 5 tiles +
// per-depth acceptance + 5-min decode chart. Replaces V0 OverviewTab
// and absorbs the V0 SpeculativeTab's acceptance bars.

struct LiveTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var themeStore: ThemeStore

    var body: some View {
        GeometryReader { geo in
            // Content width after the 20pt page padding. Handed to the
            // hero gauge and the tile row so they reflow (gauge shrinks,
            // tiles wrap) when the window is dragged narrow.
            let contentWidth = max(0, geo.size.width - 40)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let notice = backend.clientHandoffNotice, notice.isWarning {
                        ClientHandoffWarningBanner(notice: notice)
                    }

                    if let portNotice = backend.portFallbackNotice {
                        PortFallbackBanner(message: portNotice)
                    }

                    MemoryGuardBanner(
                        pressureLevel: backend.memoryPressureLevel,
                        recentShed: backend.memoryGuardRecentShed,
                        pressureSource: backend.memoryPressureSource,
                        plan: backend.memoryPlan
                    )

                    heroSection(availableWidth: contentWidth)
                    TileRow(availableWidth: contentWidth)
                    AcceptanceSection()
                    DecodeChart()
                    VerifyWaterfallExpander()
                }
                .padding(20)
                .frame(width: geo.size.width, alignment: .topLeading)
            }
            .scrollIndicators(.hidden)
            .background(Brand.bgOuter)
        }
    }

    // MARK: - Hero

    @ViewBuilder
    private func heroSection(availableWidth: CGFloat) -> some View {
        let mode = currentMode()
        let rolling = backend.rolling
        let stickyMax = rolling?.stickyAllTimeMax ?? 0
        // Shrink the hero gauge on narrow windows so it never clips the
        // ring; the canonical 280pt hero holds on normal/wide layouts.
        let diameter = min(280, max(150, availableWidth - 16))

        VStack(spacing: 14) {
            GaugeView(
                mode: mode,
                // Pass the raw sticky max in the same units as the
                // mode's axis. GaugeView maps it through the same log
                // arcFraction it uses for the fill so the tick stays
                // pinned to the right number, not a percent.
                stickyMax: scaledStickyMax(stickyMax, mode: mode),
                ceiling: 1.0,
                motionEnabled: motionEnabled,
                borderTint: fanBoostVisible ? Brand.coolChrome : nil,
                diameter: diameter,
                onPowerTap: { quickStart() }
            )
            .padding(.top, 8)

            heroCaption(mode: mode)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }

    /// Speedtest-style "GO" — quick-starts the daemon with the user's
    /// most-recent launch target (or `.chat` if none has been chosen
    /// yet). The top-strip Play button is still available for explicit
    /// target selection.
    private func quickStart() {
        let target = LaunchTarget(rawValue: backend.configuration.lastLaunchTarget) ?? .chat
        Task { await backend.startDaemon(target: target) }
    }

    private var fanBoostVisible: Bool {
        if backend.health?.fanBoostActive == true {
            return true
        }
        return MTPLXFanMode.normalized(backend.currentFanMode) == .max
    }

    @ViewBuilder
    private func heroCaption(mode: GaugeMode) -> some View {
        if case .prefill = mode, let prefill = currentPrefill() {
            HStack(spacing: 6) {
                Image(systemName: cacheSymbol(for: prefill))
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(Brand.warning)
                Text(prefillTokenCaption(prefill))
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(1)
                    .foregroundStyle(Brand.textHighlight.opacity(0.75))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
        } else {
            // Same warm-up gate as `currentMode()` — don't show the
            // ALL-TIME MAX badge until a real user request has landed,
            // or the first thing the user sees is a misleading "30.9 TPS
            // max" from the model's warm-up snapshot.
            let hasRealRequest = backend.observedCompletionCount > 0
            let rolling = backend.rolling
            if hasRealRequest, let max = rolling?.stickyAllTimeMax, max > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(Brand.warning)
                    Text(tr("ALL-TIME MAX %@ TPS", Format.tps(max)))
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .tracking(2)
                        .chromeText()
                }
            }
        }
    }

    // MARK: - Mode computation

    private func currentMode() -> GaugeMode {
        switch backend.daemonState.kind {
        case .stopped, .crashed: return .dim
        case .degraded: return .degraded
        case .starting: return .loading(phase: loadingPhase())
        case .warming: return .loading(phase: loadingPhase())
        case .stopping: return .loading(phase: .stopping)
        case .running: break
        }

        if let prefill = currentPrefill(), prefill.isActive,
           HeroPrefillGate.showsPrefill(
               prefill: prefill,
               promptTokens: currentPrefillRequest?.promptTokens,
               sessionPrefixLen: currentPrefillSessionPrefixLen
           ) {
            return .prefill(
                progress: prefill.progress,
                // Sanity-filter the rate so the gauge never shows
                // "18000" from a sub-second cumulative chunk read on
                // a fresh OpenCode prompt. When the filter rejects
                // the value, the headline falls back to the percent
                // label and the caption still shows tokens + ETA.
                livePrefillTPS: prefill.sanePrefillTokS(),
                eta: prefill.etaSeconds,
                requestId: backend.inFlight.first?.requestId
            )
        }

        // Headline decode is owned by the backend's lifecycle state
        // machine. It survives snapshot updates that drop the decode
        // key (the value stays on the gauge after a request finishes),
        // it holds the final average between requests, and it resets
        // cleanly when the daemon stops/restarts. The gauge therefore
        // never falls to zero just because an SSE snapshot omitted
        // `decode_tok_s` for a frame.
        //
        // Warm-up gate: loading a model runs a warm-up decode whose
        // reading (~30 tok/s) lands in `headlineDecode` before the user
        // has run anything — so a fresh chat used to spawn the dial at
        // 30. Suppress the headline until a real user request is
        // actually decoding, or one has completed; then the dial starts
        // at 0 and rises with the first real response.
        let hasRealDecode = backend.observedCompletionCount > 0
            || backend.inFlight.contains { $0.hasDecodeProgress }
        let decode = hasRealDecode ? (backend.headlineDecode.value ?? 0) : 0
        return .tps(decode: decode, max: 100)
    }

    /// The in-flight request that owns the active prefill, for the gate's
    /// prompt size and session identity.
    private var currentPrefillRequest: InFlightRequest? {
        backend.inFlight.first(where: { $0.prefillState?.isActive == true }) ?? backend.inFlight.first
    }

    /// What the resolved session already holds, so a tool turn's
    /// `started` frame (fired before the restore runs) is not read as a
    /// full re-prefill.
    private var currentPrefillSessionPrefixLen: Int? {
        guard let sessionId = currentPrefillRequest?.sessionId else { return nil }
        return backend.sessions?.sessions.first(where: { $0.sessionId == sessionId })?.prefixLen
    }

    private func loadingPhase() -> LoadingPhase {
        switch backend.startupPhase {
        case .launching: return .starting
        case .waitingForOwnedHealth: return .waitingForServer
        case .rampingFans: return .rampingFans
        case .warming: return .warming
        default: return .starting
        }
    }

    /// Builds a live `PrefillState` from the authoritative in-flight
    /// snapshot and the freshest current SSE prefill frame. Completed
    /// history is intentionally excluded from the hero gauge; a previous
    /// prefill should never make the running UI look stuck in prefill.
    private func currentPrefill() -> PrefillState? {
        let snapshotState = backend.inFlight.compactMap(\.prefillState).first(where: \.isActive)
        var eventState: PrefillState?
        if let values = currentPrefillPayloadValues(),
           let state = prefillState(from: values),
           state.isActive,
           prefillPayloadIsCurrent(values) {
            eventState = state
        }
        if let snapshotState, let eventState {
            return freshestPrefill(snapshotState, eventState)
        }
        if let eventState { return eventState }
        if let snapshotState { return snapshotState }
        return nil
    }

    private func freshestPrefill(_ left: PrefillState, _ right: PrefillState) -> PrefillState {
        let leftDone = left.tokensDone ?? -1
        let rightDone = right.tokensDone ?? -1
        if rightDone > leftDone { return right }
        if leftDone > rightDone { return left }
        if right.displayPrefillTokS != nil, left.displayPrefillTokS == nil {
            return right
        }
        return left
    }

    private func prefillTokenCaption(_ state: PrefillState) -> String {
        let done = state.tokensDone ?? 0
        let total = state.tokensTotal
        var parts = [tr("%@/%@ tok", Format.integer(done), Format.integer(total))]
        if let cached = state.cachedTokens, cached > 0 {
            parts.append(tr("cached %@", Format.integer(cached)))
        }
        if let newPrefill = state.newPrefillTokens {
            parts.append(tr("new %@", Format.integer(newPrefill)))
        }
        if let restore = state.ssdRestoreS, restore > 0 {
            parts.append(tr("SSD %@", Format.duration(restore)))
        } else if let source = state.cacheSource, !source.isEmpty, source != "none" {
            parts.append(source.uppercased())
        }
        if let eta = state.etaSeconds {
            parts.append(tr("ETA %@", Format.duration(eta)))
        }
        return parts.joined(separator: " · ")
    }

    private func cacheSymbol(for state: PrefillState) -> String {
        if state.ssdCacheHit == true { return "externaldrive.fill" }
        if let source = state.cacheSource, source == "ram" { return "memorychip.fill" }
        return "gauge.with.dots.needle.bottom.50percent"
    }

    private func currentPrefillPayloadValues() -> [String: JSONValue]? {
        guard let status = backend.prefillStatus else { return nil }
        if let nested = status.object("prefill") {
            return nested
        }
        return status.values
    }

    private func prefillPayloadIsCurrent(_ values: [String: JSONValue]) -> Bool {
        guard !backend.inFlight.isEmpty else { return false }
        guard let requestId = values["request_id"]?.stringValue else { return false }
        guard let request = backend.inFlight.first(where: { $0.requestId == requestId }) else {
            return false
        }
        return !request.hasDecodeProgress
    }

    /// Map a raw prefill `DynamicObject` payload onto `PrefillState`
    /// using untyped accessors. Keeps us decoupled from the upstream
    /// schema — fields we don't understand are simply ignored.
    private func prefillState(from values: [String: JSONValue]) -> PrefillState? {
        let phase = values["phase"]?.stringValue ?? "unknown"
        let tokensTotal = (values["tokens_total"]?.doubleValue).map(Int.init) ?? 0
        return PrefillState(
            phase: phase,
            tokensDone: (values["tokens_done"]?.doubleValue).map(Int.init),
            tokensTotal: tokensTotal,
            cachedTokens: (values["cached_tokens"]?.doubleValue).map(Int.init),
            newPrefillTokens: (values["new_prefill_tokens"]?.doubleValue).map(Int.init),
            elapsedS: values["elapsed_s"]?.doubleValue,
            promptEvalTimeS: values["prompt_eval_time_s"]?.doubleValue,
            prefillTokS: values["prefill_tok_s"]?.doubleValue,
            prefillComputeTokS: values["prefill_compute_tok_s"]?.doubleValue,
            prefillWallTokS: values["prefill_wall_tok_s"]?.doubleValue,
            cumulativePrefillTokS: values["cumulative_prefill_tok_s"]?.doubleValue,
            livePrefillTokS: values["live_prefill_tok_s"]?.doubleValue,
            chunkSize: (values["chunk_size"]?.doubleValue).map(Int.init),
            chunkElapsedS: values["chunk_elapsed_s"]?.doubleValue,
            chunkPrefillTokS: values["chunk_prefill_tok_s"]?.doubleValue,
            cacheHit: values["cache_hit"]?.boolValue,
            cacheSource: values["cache_source"]?.stringValue,
            ssdCacheHit: values["ssd_cache_hit"]?.boolValue,
            ssdCachedTokens: (values["ssd_cached_tokens"]?.doubleValue).map(Int.init),
            ssdRestoreS: values["ssd_restore_s"]?.doubleValue,
            ssdSuffixTokens: (values["ssd_suffix_tokens"]?.doubleValue).map(Int.init),
            startedS: values["started_s"]?.doubleValue
        )
    }

    /// Scale sticky max to 0...1 along the visible arc using the same
    /// piecewise-linear tick mapping the fill needle uses. The sticky
    /// tick is hidden in prefill mode because the sticky cap is a
    /// decode-band concept.
    private func scaledStickyMax(_ max: Double, mode: GaugeMode) -> Double {
        if case .prefill = mode { return 0 }
        return GaugeMode.arcFraction(value: max, tickStops: mode.tickStops)
    }

    private var motionEnabled: Bool {
        !backend.configuration.performanceLock && !themeStore.reduceMotionPreference
    }
}

private struct PortFallbackBanner: View {
    let message: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "rectangle.connected.to.line.below")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Color.mtplxWarning)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 4) {
                Text(tr("Port changed"))
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.mtplxWarning.opacity(0.10))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.mtplxWarning.opacity(0.28), lineWidth: Brand.hairline)
                }
        }
    }
}

private struct ClientHandoffWarningBanner: View {
    let notice: ClientHandoffNotice

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Color.mtplxWarning)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 4) {
                Text(notice.status)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(notice.detail)
                    .font(.caption)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            PillBadge(
                text: notice.target.title,
                systemImage: notice.target.systemImage,
                tint: .mtplxWarning,
                emphasized: true
            )
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.mtplxWarning.opacity(0.10))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Color.mtplxWarning.opacity(0.28), lineWidth: Brand.hairline)
                }
        }
    }
}
