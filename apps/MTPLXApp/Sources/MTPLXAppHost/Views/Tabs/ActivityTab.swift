import SwiftUI
import MTPLXAppCore

// MARK: - ActivityTab
//
// Merged surface that replaces the V0 Cache + Requests pair. One
// dashboard tab that answers "what is the engine doing right NOW and
// what just happened?" — in-flight first because that's what the user
// usually came to see, then recent completions, then SessionBank
// state.
//
// The original `CacheTab.swift` and `RequestsTab.swift` are kept
// intact (no longer referenced by `AppTab` / `DashboardSurface` after
// the V1 merge) so this file is a self-contained recomposition of
// their body content. Helpers below are private duplicates of the
// originals on purpose — keeping the old files unmutated makes the
// merge trivially revertible if we ever want to split them again.

struct ActivityTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    @State private var pendingClearAll = false
    @State private var clearingAll = false
    @State private var clearingSession: String? = nil
    @State private var cancellingId: String? = nil

    var body: some View {
        let sessions = backend.sessions
        let sessionBank = backend.sessionBank ?? sessions?.sessionBank
        let inFlight = backend.inFlight
        let recent = backend.snapshot?.recent ?? []

        Group {
            if backend.daemonState.kind == .stopped {
                EmptyStateView(
                    symbol: "waveform.path.ecg",
                    title: "No activity yet",
                    message: "Start a model to see live requests, recent answers, and cache stats."
                ) {
                    Task { await backend.startDaemon() }
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        inFlightCard(requests: inFlight)
                        if let retrieval = backend.snapshot?.retrieval, retrieval.enabled {
                            retrievalCard(retrieval)
                        }
                        if let notice = backend.clientHandoffNotice {
                            clientHandoffCard(notice)
                        }
                        if shouldShowPiHandoff {
                            piHandoffCard()
                        }
                        recentRequestsCard(recent: recent)
                        speedTruthCard(latest: backend.latest)
                        cacheSummaryCard(sessions: sessions, sessionBank: sessionBank)
                        cacheTruthCard(latest: backend.latest, sessionBank: sessionBank)
                        bankCard(sessionBank: sessionBank)
                        sessionsListCard(sessions: sessions)
                    }
                    .padding(20)
                }
            }
        }
        .navigationTitle("Activity")
        .confirmationDialog(
            "Clear all cache entries?",
            isPresented: $pendingClearAll
        ) {
            Button("Clear All", role: .destructive) {
                clearingAll = true
                Task {
                    defer { Task { @MainActor in clearingAll = false } }
                    try? await backend.clearCache()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Clears every saved prompt cache. Anything mid-flight keeps running. Next chats will start from scratch.")
        }
    }

    // MARK: Agent handoffs

    @ViewBuilder
    private func clientHandoffCard(_ notice: ClientHandoffNotice) -> some View {
        let tint: Color = notice.isWarning ? .mtplxWarning : .mtplxSuccess
        Card("External Client", subtitle: "\(notice.target.title) handoff from MTPLX.") {
            PillBadge(
                text: notice.status,
                systemImage: notice.target.systemImage,
                tint: tint,
                emphasized: true
            )
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                truthRow(
                    "Target",
                    notice.target.title,
                    systemImage: notice.target.systemImage,
                    tint: tint
                )
                truthRow(
                    "Status",
                    notice.detail,
                    systemImage: notice.isWarning ? "exclamationmark.triangle.fill" : "checkmark.circle.fill",
                    tint: tint
                )
            }
        }
    }

    private var shouldShowPiHandoff: Bool {
        backend.piTerminalAgentRunning
            || backend.piTerminalLaunchDetail?.isEmpty == false
            || backend.piTerminalLaunchCommand?.isEmpty == false
    }

    @ViewBuilder
    private func piHandoffCard() -> some View {
        Card("Agent Handoff", subtitle: "External terminal client launched by MTPLX.") {
            PillBadge(
                text: backend.piTerminalAgentRunning ? "Pi running" : "Pi not detected",
                systemImage: "pi",
                tint: backend.piTerminalAgentRunning ? .mtplxSuccess : .mtplxWarning,
                emphasized: backend.piTerminalAgentRunning
            )
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                truthRow(
                    "Pi Terminal",
                    piHandoffStatusText(),
                    systemImage: "terminal",
                    tint: backend.piTerminalAgentRunning ? .mtplxSuccess : .mtplxWarning
                )
                truthRow(
                    "Workspace",
                    PiIntegration.resolvedWorkspacePath(configuration: backend.configuration),
                    systemImage: "folder",
                    tint: Brand.accentChrome
                )
                if let command = backend.piTerminalLaunchCommand, !command.isEmpty {
                    truthRow(
                        "Command",
                        command,
                        systemImage: "chevron.left.forwardslash.chevron.right",
                        tint: .secondary
                    )
                }
            }
        }
    }

    private func piHandoffStatusText() -> String {
        var parts: [String] = []
        if let detail = backend.piTerminalLaunchDetail, !detail.isEmpty {
            parts.append(detail)
        }
        if !backend.piTerminalAgentProcessIDs.isEmpty {
            parts.append(
                "pid " + backend.piTerminalAgentProcessIDs
                    .map(String.init)
                    .joined(separator: ", ")
            )
        }
        return parts.isEmpty ? "No Pi handoff yet." : parts.joined(separator: " · ")
    }

    // MARK: Requests half

    @ViewBuilder
    private func inFlightCard(requests: [InFlightRequest]) -> some View {
        Card("In Flight", subtitle: requests.isEmpty ? "No active requests." : "\(requests.count) active") {
            if requests.isEmpty {
                PillBadge(text: "idle", systemImage: "moon.stars", tint: .secondary)
            }
        } content: {
            if requests.isEmpty {
                Text("Live requests will show up here. Hit Cancel to stop one.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .leading)
            } else {
                VStack(spacing: 8) {
                    ForEach(requests) { request in
                        inFlightRow(request)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func inFlightRow(_ request: InFlightRequest) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Image(systemName: request.cancelled ? "stop.circle.fill" : "circle.fill")
                    .foregroundStyle(request.cancelled ? Color.mtplxDanger : Color.mtplxSuccess)
                    .font(.caption2)
                Text(request.shortId)
                    .font(.system(.callout, design: .monospaced).weight(.semibold))
                Text("· \(Format.duration(request.ageS)) ago")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                if let session = request.sessionId {
                    PillBadge(text: "session " + String(session.prefix(8)),
                              systemImage: "person.crop.circle",
                              tint: Brand.accentChrome)
                }
                if let prefill = request.prefillState, prefill.isActive {
                    PillBadge(text: "PREFILL " + Format.percent(prefill.progress, fractionDigits: 0),
                              systemImage: "gauge.with.dots.needle.bottom.50percent",
                              tint: .mtplxWarning,
                              emphasized: true)
                }
                Spacer()
                if backend.capabilities?.features["request_cancel"] != false {
                    Button {
                        cancellingId = request.requestId
                        Task {
                            defer { Task { @MainActor in cancellingId = nil } }
                            try? await backend.cancel(requestId: request.requestId)
                        }
                    } label: {
                        if cancellingId == request.requestId {
                            ProgressView().controlSize(.mini)
                        } else {
                            Label("Cancel", systemImage: "xmark.circle")
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(request.cancelled || cancellingId == request.requestId)
                }
            }
            Text(request.promptDigest)
                .font(.callout)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            HStack(spacing: 14) {
                Label("\(Format.integer(request.promptTokens)) prompt tok",
                      systemImage: "text.alignleft")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
                if let model = request.model {
                    Label(model,
                          systemImage: "cube.box")
                        .labelStyle(.titleAndIcon)
                        .font(.caption2)
                        .lineLimit(1)
                }
            }
            .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.raisedSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    @ViewBuilder
    private func recentRequestsCard(recent: [MetricsLatest]) -> some View {
        Card("Recent", subtitle: recent.isEmpty ? "Nothing recorded yet." : "\(recent.count) requests") {
            if recent.isEmpty {
                EmptyView()
            } else {
                PillBadge(text: "live updates", systemImage: "antenna.radiowaves.left.and.right", tint: .mtplxSuccess)
            }
        } content: {
            if recent.isEmpty {
                Text("Finished requests will show up here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .leading)
            } else {
                VStack(spacing: 0) {
                    recentHeaderRow
                    Divider()
                    ForEach(Array(recent.enumerated()), id: \.offset) { _, metric in
                        recentRow(metric)
                        Divider().opacity(0.3)
                    }
                }
            }
        }
    }

    private var recentHeaderRow: some View {
        HStack {
            Text("Session").frame(width: 120, alignment: .leading)
            Text("Decode").frame(width: 80, alignment: .trailing)
            Text("Prefill").frame(width: 80, alignment: .trailing)
            Text("TTFT").frame(width: 70, alignment: .trailing)
            Text("Tokens").frame(width: 70, alignment: .trailing)
            Text("Cache").frame(width: 60, alignment: .trailing)
            Spacer()
        }
        .font(.system(size: 10, weight: .semibold))
        .tracking(0.6)
        .foregroundStyle(.tertiary)
        .padding(.bottom, 4)
    }

    private func recentRow(_ metric: MetricsLatest) -> some View {
        HStack {
            Text(metric.sessionId.map { String($0.prefix(12)) } ?? "—")
                .font(.system(.caption, design: .monospaced))
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)
            Text(Format.tps(metric.decodeTokS))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .frame(width: 80, alignment: .trailing)
            Text(Format.tps(metric.prefillTokS))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text(Format.duration(metric.ttftS))
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .trailing)
            Text(Format.integer(metric.generatedTokens ?? metric.completionTokens))
                .font(.system(.caption, design: .rounded))
                .monospacedDigit()
                .frame(width: 70, alignment: .trailing)
            Text(cacheVerdictLabel(metric.requestCacheVerdict))
                .font(.caption2)
                .foregroundStyle(cacheVerdictTint(metric.requestCacheVerdict))
                .frame(width: 60, alignment: .trailing)
            Spacer()
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func speedTruthCard(latest: MetricsLatest?) -> some View {
        Card("Speed Truth", subtitle: "Where the last agent turn spent time.") {
            if let latest {
                PillBadge(
                    text: speedVerdict(latest).label,
                    systemImage: speedVerdict(latest).symbol,
                    tint: speedVerdict(latest).tint,
                    emphasized: speedVerdict(latest).emphasized
                )
            }
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                truthRow(
                    "User speed",
                    userSpeedText(latest),
                    systemImage: "speedometer",
                    tint: speedVerdict(latest).tint
                )
                truthRow(
                    "Agent lane",
                    agentLaneText(latest),
                    systemImage: "point.3.connected.trianglepath.dotted",
                    tint: agentLaneTint(latest)
                )
                truthRow(
                    "Decode",
                    decodeSpeedText(latest),
                    systemImage: "bolt.fill",
                    tint: decodeTint(latest)
                )
                truthRow(
                    "Prefill",
                    prefillSpeedText(latest),
                    systemImage: "gauge.with.dots.needle.bottom.50percent",
                    tint: prefillTint(latest)
                )
                truthRow(
                    "Verify cost",
                    verifyCostText(latest),
                    systemImage: "checkmark.seal",
                    tint: verifyTint(latest)
                )
                truthRow(
                    "Prompt pressure",
                    promptPressureText(latest),
                    systemImage: "text.alignleft",
                    tint: promptPressureTint(latest)
                )
            }
        }
    }

    private func userSpeedText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No request completed yet." }
        var parts: [String] = []
        if let raw = latest.decodeTokS {
            parts.append("\(Format.tps(raw)) tok/s generation")
        }
        if let request = latest.requestTokS {
            parts.append("\(Format.tps(request)) tok/s end-to-end")
        }
        if let elapsed = latest.requestElapsedS ?? latest.elapsedS {
            parts.append("elapsed \(Format.duration(elapsed))")
        }
        if let completion = latest.completionTokens ?? latest.generatedTokens {
            parts.append("\(Format.integer(completion)) output tok")
        }
        return parts.isEmpty ? "No speed data yet." : parts.joined(separator: " · ")
    }

    private func agentLaneText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No agent lane data yet." }
        var parts: [String] = []
        if let client = latest.requestClientHint, !client.isEmpty {
            parts.append(client)
        }
        if let mode = latest.requestGenerationMode, !mode.isEmpty {
            parts.append(mode.uppercased())
        }
        if let depth = latest.requestEffectiveMtpDepth
            ?? latest.requestDepth
            ?? latest.mtpDepth
            ?? latest.speculativeDepth {
            parts.append("depth \(depth)")
        }
        if let reasoning = latest.requestReasoningMode, !reasoning.isEmpty {
            parts.append("reasoning \(reasoning)")
        } else if let thinking = latest.requestEnableThinking {
            parts.append(thinking ? "thinking on" : "thinking off")
        }
        if let filteredCount = latest.requestFilteredToolCount {
            parts.append("\(Format.integer(filteredCount)) active tools")
        } else if let filtered = latest.requestFilteredToolNames {
            parts.append("\(Format.integer(filtered.count)) active tools")
        } else if let tools = latest.requestToolNames {
            parts.append("\(tools.count) tools")
        } else if let toolCount = latest.requestToolCount {
            parts.append("\(Format.integer(toolCount)) tools")
        }
        if let hidden = latest.requestHiddenToolNames, !hidden.isEmpty {
            parts.append("\(hidden.count) hidden")
        }
        return parts.isEmpty ? "No agent lane data yet." : parts.joined(separator: " · ")
    }

    private func decodeSpeedText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No decode data yet." }
        var parts: [String] = []
        if let raw = latest.decodeTokS {
            parts.append("generation \(Format.tps(raw))")
        }
        if let display = latest.displayDecodeTokS, display != latest.decodeTokS {
            parts.append("window \(Format.tps(display))")
        }
        if let server = latest.serverTokS {
            parts.append("server \(Format.tps(server))")
        }
        if let decodeElapsed = latest.decodeElapsedS {
            parts.append("decode \(Format.duration(decodeElapsed))")
        }
        return parts.isEmpty ? "No decode data yet." : parts.joined(separator: " · ")
    }

    private func prefillSpeedText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No prefill data yet." }
        var parts: [String] = []
        if let ttft = latest.ttftS {
            parts.append("TTFT \(Format.duration(ttft))")
        }
        if let prefill = latest.prefillTokS {
            parts.append("\(Format.tps(prefill)) tok/s")
        }
        if let cached = latest.cachedTokens, cached > 0 {
            parts.append("\(Format.integer(cached)) cached")
        }
        if let newPrefill = latest.newPrefillTokens {
            parts.append("\(Format.integer(newPrefill)) new")
        }
        if latest.ssdCacheHit == true, let restore = latest.ssdRestoreS {
            parts.append("SSD restore \(Format.duration(restore))")
        }
        return parts.isEmpty ? "No prefill data yet." : parts.joined(separator: " · ")
    }

    private func verifyCostText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No verify data yet." }
        guard let calls = latest.verifyCalls, calls > 0 else {
            return "Baseline or no speculative verify calls."
        }
        var parts = ["\(Format.integer(calls)) calls"]
        if let verify = latest.verifyTimeS {
            parts.append("total \(Format.duration(verify))")
            parts.append("\(Format.milliseconds(verify / Double(max(1, calls))))/call")
            if let completion = latest.completionTokens ?? latest.generatedTokens,
               completion > 0 {
                parts.append("\(Format.milliseconds(verify / Double(completion)))/output tok")
            }
        }
        if let forward = latest.verifyForwardTimeS {
            parts.append("forward \(Format.duration(forward))")
        }
        return parts.joined(separator: " · ")
    }

    private func promptPressureText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No prompt data yet." }
        var parts: [String] = []
        if let prompt = latest.promptTokens {
            parts.append("\(Format.integer(prompt)) prompt tok")
        }
        if let raw = latest.transcriptRawMessageChars,
           let canonical = latest.transcriptCanonicalMessageChars,
           raw > 0 {
            let saved = max(0, raw - canonical)
            parts.append("\(Format.integer(canonical)) chars")
            if saved > 0 {
                parts.append("saved \(Format.integer(saved))")
            }
        }
        if latest.transcriptCanonicalized == true {
            parts.append("compacted")
        }
        if let activeRead = latest.transcriptCompactedActiveReadChars,
           activeRead > 0 {
            parts.append("read saved \(Format.integer(activeRead))")
        }
        return parts.isEmpty ? "No prompt data yet." : parts.joined(separator: " · ")
    }

    private func speedVerdict(_ latest: MetricsLatest?) -> (label: String, symbol: String, tint: Color, emphasized: Bool) {
        guard let latest else {
            return ("waiting", "clock", .secondary, false)
        }
        let userRate = latest.decodeTokS ?? latest.requestTokS
        let displayRate = latest.decodeTokS ?? latest.displayDecodeTokS
        let prompt = latest.promptTokens ?? 0
        if let userRate {
            if prompt >= 10_000 && userRate >= 25 {
                return ("long turn healthy", "checkmark.seal.fill", Brand.success, false)
            }
            if prompt >= 10_000 && userRate < 20 {
                return ("long turn sludge", "exclamationmark.triangle.fill", Brand.warning, true)
            }
            if userRate >= 30 {
                return ("healthy", "checkmark.seal.fill", Brand.success, false)
            }
            if let displayRate, displayRate >= 30, userRate >= 25 {
                return ("decode healthy", "checkmark.seal.fill", Brand.success, false)
            }
            if userRate < 15 {
                return ("slow", "exclamationmark.triangle.fill", Brand.warning, true)
            }
            return ("watch", "gauge.medium", Brand.warning, false)
        }
        return ("no speed sample", "clock", .secondary, false)
    }

    private func agentLaneTint(_ latest: MetricsLatest?) -> Color {
        guard let latest else { return .secondary }
        if latest.requestEnableThinking == false { return Brand.warning }
        if latest.requestToolsHiddenByBridge == true { return Brand.success }
        return Brand.accentChrome
    }

    private func decodeTint(_ latest: MetricsLatest?) -> Color {
        guard let rate = latest?.decodeTokS ?? latest?.displayDecodeTokS else {
            return .secondary
        }
        if rate >= 30 { return Brand.success }
        if rate < 20 { return Brand.warning }
        return Brand.accentChrome
    }

    private func prefillTint(_ latest: MetricsLatest?) -> Color {
        guard let latest else { return .secondary }
        if latest.sessionCacheHit == true || latest.cachedTokens ?? 0 > 0 {
            return Brand.success
        }
        if (latest.ttftS ?? 0) > 8 {
            return Brand.warning
        }
        return Brand.accentChrome
    }

    private func verifyTint(_ latest: MetricsLatest?) -> Color {
        guard let latest,
              let calls = latest.verifyCalls,
              calls > 0,
              let verify = latest.verifyTimeS else {
            return .secondary
        }
        let msPerCall = verify * 1000 / Double(max(1, calls))
        if msPerCall > 100 { return Brand.warning }
        return Brand.accentChrome
    }

    private func promptPressureTint(_ latest: MetricsLatest?) -> Color {
        guard let latest else { return .secondary }
        let prompt = latest.promptTokens ?? 0
        if prompt >= 16_000 { return Brand.warning }
        if latest.transcriptCanonicalized == true { return Brand.success }
        return .secondary
    }

    // MARK: Cache half

    @ViewBuilder
    private func cacheSummaryCard(sessions: SessionsPayload?, sessionBank: SessionBank?) -> some View {
        Card("Cache Summary") {
            HStack(spacing: 24) {
                StatTile(
                    label: "Sessions",
                    value: Format.integer(sessions?.count ?? sessions?.sessions.count),
                    systemImage: "person.2.fill",
                    tint: Brand.accentChrome
                )
                Divider().frame(height: 36)
                StatTile(
                    label: "Bank size",
                    value: Format.bytes(sessionBank?.totalNbytes),
                    systemImage: "tray.full",
                    tint: .secondary
                )
                Divider().frame(height: 36)
                StatTile(
                    label: "Bank capacity",
                    value: Format.integer(sessionBank?.maxEntries),
                    systemImage: "square.stack.3d.up",
                    tint: .secondary
                )
                Divider().frame(height: 36)
                StatTile(
                    label: "Last miss",
                    value: sessionBank?.lastMissReason ?? "—",
                    systemImage: "questionmark.circle",
                    tint: (sessionBank?.lastMissReason).flatMap { $0.isEmpty ? nil : $0 } == nil
                        ? .secondary : .mtplxWarning
                )
            }
        }
    }

    @ViewBuilder
    private func cacheTruthCard(latest: MetricsLatest?, sessionBank: SessionBank?) -> some View {
        Card("Cache Truth", subtitle: "Why the last agent turn was fast or slow.") {
            if let label = frontierBadgeText(latest) {
                PillBadge(
                    text: label,
                    systemImage: latest?.liveFrontierHit == true ? "bolt.fill" : "point.3.connected.trianglepath.dotted",
                    tint: latest?.liveFrontierHit == false ? .mtplxWarning : Brand.accentChrome,
                    emphasized: latest?.liveFrontierHit == true
                )
            }
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                truthRow(
                    "Request cache",
                    cacheTruthText(latest),
                    systemImage: cacheTruthSymbol(latest?.requestCacheVerdict),
                    tint: cacheVerdictTint(latest?.requestCacheVerdict)
                )
                truthRow(
                    "Live frontier",
                    frontierTruthText(latest),
                    systemImage: latest?.requestSessionKeepLiveRef == true ? "bolt.horizontal.fill" : "square.stack.3d.down.right",
                    tint: latest?.liveFrontierHit == false ? .mtplxWarning : Brand.accentChrome
                )
                truthRow(
                    "Transcript",
                    transcriptTruthText(latest),
                    systemImage: "text.line.first.and.arrowtriangle.forward",
                    tint: latest?.transcriptCanonicalized == true ? .mtplxSuccess : .secondary
                )
                truthRow(
                    "Tool surface",
                    toolSurfaceText(latest),
                    systemImage: latest?.requestToolsHiddenByBridge == true
                        ? "slider.horizontal.3"
                        : "wrench.and.screwdriver",
                    tint: latest?.requestToolsHiddenByBridge == true ? .mtplxSuccess : .secondary
                )
                truthRow(
                    "Generation",
                    generationTruthText(latest),
                    systemImage: "brain.head.profile",
                    tint: latest?.requestEnableThinking == false ? .mtplxWarning : Brand.accentChrome
                )
                truthRow(
                    "Response cap",
                    responseCapText(latest),
                    systemImage: "line.3.horizontal.decrease.circle",
                    tint: latest?.serverCapApplied == true
                        || latest?.contextCapApplied == true
                        || latest?.uncappedResponseLeaseApplied == true
                        ? .mtplxWarning
                        : .secondary
                )
                if let diagnostic = sessionBank?.lastPrefixDiagnostic {
                    truthRow(
                        "Prefix diagnostic",
                        prefixDiagnosticText(diagnostic),
                        systemImage: "scope",
                        tint: diagnostic.string("miss_reason") == nil ? .secondary : .mtplxWarning
                    )
                }
            }
        }
    }

    // MARK: - Retrieval

    /// Retrieval models load on first request, so a configured model that has
    /// never been asked anything is a normal state — not an error, and not
    /// "working" either. The card distinguishes the three cases explicitly
    /// instead of leaving the user to guess whether the setting took effect.
    private func retrievalCard(_ retrieval: RetrievalStatus) -> some View {
        let loaded = retrieval.models.filter(\.loaded).count
        return Card(
            "Retrieval",
            subtitle: "Embedding and reranking served by this daemon."
        ) {
            PillBadge(
                text: "\(loaded)/\(retrieval.models.count) loaded",
                systemImage: loaded > 0 ? "checkmark.circle.fill" : "moon.zzz",
                tint: loaded > 0 ? Color.mtplxSuccess : .secondary,
                emphasized: loaded > 0
            )
        } content: {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(retrieval.models) { model in
                    retrievalModelRow(model)
                }
                if retrieval.maxResident > 0 {
                    truthRow(
                        "Resident cap",
                        "\(retrieval.resident.count) of \(retrieval.maxResident) slots in use",
                        systemImage: "memorychip",
                        tint: .secondary
                    )
                }
            }
        }
    }

    private func retrievalModelRow(_ model: RetrievalModelStatus) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: model.isEmbedding ? "square.grid.3x3.topleft.filled" : "arrow.up.arrow.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(model.loaded ? Color.mtplxSuccess : Color.secondary)
                Text(model.servedId)
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .textSelection(.enabled)
                PillBadge(text: model.isEmbedding ? "embeddings" : "rerank", tint: .secondary)
                Spacer(minLength: 0)
                PillBadge(
                    text: retrievalStateLabel(model),
                    systemImage: retrievalStateSymbol(model),
                    tint: retrievalStateTint(model),
                    emphasized: model.loaded
                )
            }
            truthRow(
                "Throughput",
                retrievalThroughputText(model),
                systemImage: "speedometer",
                tint: model.hasBeenUsed ? Color.mtplxSuccess : .secondary
            )
            truthRow("Work", retrievalWorkText(model), systemImage: "sum", tint: .secondary)
            if let error = model.lastError {
                truthRow("Last error", error, systemImage: "exclamationmark.triangle.fill", tint: Color.mtplxDanger)
            }
        }
        .padding(.vertical, 2)
    }

    private func retrievalStateLabel(_ model: RetrievalModelStatus) -> String {
        if model.hasFailed && !model.hasBeenUsed { return "failing" }
        if !model.loaded { return "idle" }
        return model.hasBeenUsed ? "ready" : "loaded"
    }

    private func retrievalStateSymbol(_ model: RetrievalModelStatus) -> String {
        if model.hasFailed && !model.hasBeenUsed { return "exclamationmark.triangle.fill" }
        return model.loaded ? "checkmark.circle.fill" : "moon.zzz"
    }

    private func retrievalStateTint(_ model: RetrievalModelStatus) -> Color {
        if model.hasFailed && !model.hasBeenUsed { return Color.mtplxDanger }
        return model.loaded ? Color.mtplxSuccess : .secondary
    }

    private func retrievalThroughputText(_ model: RetrievalModelStatus) -> String {
        guard model.hasBeenUsed else {
            return model.loaded
                ? "loaded, no requests yet"
                : "not loaded — loads on first request"
        }
        var parts: [String] = []
        if let rate = model.itemsPerSecond {
            parts.append(String(format: "%.1f %@/s", rate, model.isEmbedding ? "texts" : "docs"))
        }
        if let latency = model.avgLatencyMs {
            parts.append(String(format: "%.0f ms avg", latency))
        }
        return parts.isEmpty ? "—" : parts.joined(separator: " · ")
    }

    private func retrievalWorkText(_ model: RetrievalModelStatus) -> String {
        guard model.hasBeenUsed else {
            return model.loadSeconds > 0
                ? String(format: "loaded in %.1f s", model.loadSeconds)
                : "no work yet"
        }
        let unit = model.isEmbedding ? "texts" : "documents"
        var text = "\(model.requests) requests · \(model.items) \(unit) · \(model.tokens) tokens"
        if model.loadSeconds > 0 {
            text += String(format: " · loaded in %.1f s", model.loadSeconds)
        }
        return text
    }

    private func truthRow(
        _ label: String,
        _ value: String,
        systemImage: String,
        tint: Color
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: systemImage)
                .font(.caption.weight(.semibold))
                .foregroundStyle(tint)
                .frame(width: 16)
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.tertiary)
                .frame(width: 118, alignment: .leading)
            Text(value)
                .font(.system(.callout, design: .rounded))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    private func cacheTruthText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No request completed yet." }
        switch latest.requestCacheVerdict {
        case .hit:
            var parts = ["hit"]
            if let source = latest.cacheSource, !source.isEmpty {
                parts.append(source.uppercased())
            }
            if let cached = latest.cachedTokens, cached > 0 {
                parts.append("\(Format.integer(cached)) cached")
            }
            if let newPrefill = latest.newPrefillTokens {
                parts.append("\(Format.integer(newPrefill)) new")
            }
            if let restore = latest.ssdRestoreS, restore > 0 {
                parts.append("restore \(Format.duration(restore))")
            }
            return parts.joined(separator: " · ")
        case .miss:
            return "miss · \(latest.cacheMissReason ?? "new or uncached request")"
        case .unknown:
            if let reason = latest.cacheMissReason, !reason.isEmpty {
                return "unknown · \(reason)"
            }
            return "unknown · no cache verdict from the model"
        }
    }

    private func cacheVerdictLabel(_ verdict: RequestCacheVerdict?) -> String {
        switch verdict {
        case .hit: return "hit"
        case .miss: return "miss"
        case .unknown, nil: return "—"
        }
    }

    private func cacheVerdictTint(_ verdict: RequestCacheVerdict?) -> Color {
        switch verdict {
        case .hit: return .mtplxSuccess
        case .miss: return .secondary
        case .unknown, nil: return .secondary.opacity(0.65)
        }
    }

    private func cacheTruthSymbol(_ verdict: RequestCacheVerdict?) -> String {
        switch verdict {
        case .hit: return "memorychip.fill"
        case .miss: return "exclamationmark.triangle"
        case .unknown, nil: return "questionmark.circle"
        }
    }

    private func frontierTruthText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No frontier data yet." }
        let policy = latest.liveFrontierPolicy ?? "none"
        if latest.liveFrontierResultTurn == true {
            if latest.liveFrontierHit == true {
                let mode = latest.liveFrontierRestoreMode ?? latest.sessionRestoreMode ?? policy
                if latest.opencodeToolHistoryLiveFrontierRestore == true || mode == "reference_lease" {
                    return "hit · exact live frontier"
                }
                if mode.contains("near_prefix") {
                    return "hit · near prefix restore"
                }
                if mode.contains("clone") {
                    return "hit · snapshot restore"
                }
                return "hit · \(mode)"
            }
            return "miss · \(latest.liveFrontierMissReason ?? latest.cacheMissReason ?? policy)"
        }
        if latest.requestSessionKeepLiveRef == true {
            if latest.opencodeToolHistoryLiveFrontierRestore == true || policy.contains("live_reference") {
                return "armed for next tool result · exact live frontier"
            }
            return "armed for next tool result · \(policy)"
        }
        if let reason = latest.requestSessionKeepLiveRefReason {
            return "\(policy) · \(reason)"
        }
        return policy
    }

    private func frontierBadgeText(_ latest: MetricsLatest?) -> String? {
        guard let latest else { return nil }
        if latest.liveFrontierResultTurn == true {
            return latest.liveFrontierHit == true ? "frontier hit" : "frontier miss"
        }
        if latest.requestSessionKeepLiveRef == true {
            return "frontier armed"
        }
        return nil
    }

    private func transcriptTruthText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No transcript data yet." }
        let raw = latest.transcriptRawMessageChars ?? 0
        let canonical = latest.transcriptCanonicalMessageChars ?? raw
        let saved = max(0, raw - canonical)
        let toolSaved = (latest.transcriptCompactedToolResultChars ?? 0)
            + (latest.transcriptCompactedActiveToolResultChars ?? 0)
            + (latest.transcriptCompactedActiveReadChars ?? 0)
        if raw <= 0 {
            return "No transcript data yet."
        }
        var parts = ["\(Format.integer(canonical)) chars"]
        if saved > 0 {
            parts.append("saved \(Format.integer(saved))")
        }
        if toolSaved > 0 {
            parts.append("tool output compacted \(Format.integer(toolSaved))")
        }
        if let readHints = latest.transcriptCompactedActiveToolResultReadHints,
           readHints > 0 {
            parts.append("\(Format.integer(readHints)) next-read hints")
        }
        if let files = latest.transcriptInspectionReadBudgetCandidateMessages,
           files > 1,
           let maxLines = latest.transcriptInspectionReadBudgetMaxLinesPerFile,
           maxLines > 0 {
            parts.append("read digest budget \(files)x\(maxLines) lines")
        }
        return parts.joined(separator: " · ")
    }

    private func toolSurfaceText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No tool data yet." }
        let client = latest.requestClientHint ?? "client"
        let hidden = latest.requestHiddenToolNames ?? []
        if let filteredCount = latest.requestFilteredToolCount {
            var active = "\(client) · \(Format.integer(filteredCount)) active"
            if let filtered = latest.requestFilteredToolNames, !filtered.isEmpty {
                active += ": \(compactToolList(filtered))"
            }
            var parts = [active]
            if !hidden.isEmpty {
                parts.append("hid \(Format.integer(hidden.count)): \(compactToolList(hidden))")
            }
            return parts.joined(separator: " · ")
        }
        if let filtered = latest.requestFilteredToolNames, !filtered.isEmpty {
            var parts = ["\(client) · \(Format.integer(filtered.count)) active: \(compactToolList(filtered))"]
            if let hidden = latest.requestHiddenToolNames, !hidden.isEmpty {
                parts.append("hid \(Format.integer(hidden.count)): \(compactToolList(hidden))")
            }
            return parts.joined(separator: " · ")
        }
        if latest.requestToolsHiddenByBridge == true, !hidden.isEmpty {
            return "\(client) · 0 active · hid \(Format.integer(hidden.count)): \(compactToolList(hidden))"
        }
        if let requested = latest.requestToolNames, !requested.isEmpty {
            return "\(client) · \(Format.integer(requested.count)) advertised: \(compactToolList(requested))"
        }
        if let requested = latest.requestToolCount, requested > 0 {
            return "\(client) · \(Format.integer(requested)) advertised"
        }
        return "\(client) · no tools advertised"
    }

    private func compactToolList(_ names: [String], limit: Int = 5) -> String {
        let clean = names
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !clean.isEmpty else { return "none" }
        let visibleCount = min(clean.count, max(1, limit))
        let visible = clean.prefix(visibleCount).joined(separator: ", ")
        let remaining = clean.count - visibleCount
        return remaining > 0 ? "\(visible), +\(remaining)" : visible
    }

    private func generationTruthText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No generation data yet." }
        var parts: [String] = []
        let mode = (latest.requestGenerationMode ?? "mtp").lowercased()
        if mode == "ar" {
            parts.append("Baseline")
        } else {
            let depth = latest.requestEffectiveMtpDepth
                ?? latest.mtpDepth
                ?? latest.requestDepth
                ?? latest.speculativeDepth
            parts.append(depth.map { "MTP D\($0)" } ?? "MTP")
        }
        let reasoningMode = latest.requestReasoningMode
            ?? (latest.requestEnableThinking == false ? "off" : "on")
        var reasoning = "reasoning \(reasoningMode)"
        if latest.requestEnableThinking == false, reasoningMode != "off" {
            reasoning += " -> off"
        }
        if latest.requestEnableThinkingOverride == true {
            reasoning += " override"
        }
        parts.append(reasoning)
        if latest.requestReasoningParser == "none" {
            parts.append("parser off")
        }
        if latest.stripAssistantReasoningHistory == true {
            parts.append("history stripped")
        } else if latest.preserveThinkingEffective == false {
            parts.append("history not preserved")
        }
        return parts.joined(separator: " · ")
    }

    private func responseCapText(_ latest: MetricsLatest?) -> String {
        guard let latest else { return "No cap data yet." }
        var parts: [String] = []
        if latest.uncappedResponseRequested == true {
            parts.append("request uncapped")
        } else if let requested = latest.requestMaxTokens {
            parts.append("request \(Format.integer(requested))")
        } else {
            parts.append("request —")
        }
        if let effective = latest.effectiveMaxTokens {
            parts.append("effective \(Format.integer(effective))")
        }
        if let lease = latest.decodeLeaseTokens,
           latest.effectiveMaxTokens != lease {
            parts.append("lease \(Format.integer(lease))")
        }
        if latest.serverCapApplied == true {
            if let serverCap = latest.serverMaxResponseTokens {
                parts.append("server cap \(Format.integer(serverCap))")
            } else {
                parts.append("server cap")
            }
        }
        if latest.contextCapApplied == true {
            parts.append("context cap")
        }
        if latest.uncappedResponseLeaseApplied == true {
            parts.append("uncapped lease")
        }
        return parts.joined(separator: " · ")
    }

    private func prefixDiagnosticText(_ diagnostic: DynamicObject) -> String {
        let reason = diagnostic.string("miss_reason")
            ?? diagnostic.string("reason")
            ?? "no miss"
        var parts = [reason]
        if let matched = diagnostic.int("common_prefix_tokens")
            ?? diagnostic.int("matched_prefix_len") {
            parts.append("matched \(Format.integer(matched))")
        }
        if let boundary = diagnostic.int("nearest_boundary_tokens") {
            parts.append("boundary \(Format.integer(boundary))")
        }
        if let newPrefill = diagnostic.int("new_prefill_tokens") {
            parts.append("new \(Format.integer(newPrefill))")
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func bankCard(sessionBank: SessionBank?) -> some View {
        Card("SessionBank", subtitle: "Block-prefix reuse across requests and restarts.") {
            if backend.capabilities?.features["cache_clear"] != false {
                Button {
                    pendingClearAll = true
                } label: {
                    Label("Clear All", systemImage: "trash")
                        .font(.caption.weight(.medium))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(clearingAll)
            }
        } content: {
            if let prefixes = sessionBank?.prefixes, !prefixes.isEmpty {
                LazyVGrid(
                    columns: [
                        GridItem(.adaptive(minimum: 220, maximum: 280), spacing: 12)
                    ],
                    spacing: 12
                ) {
                    ForEach(prefixes, id: \.sessionId) { prefix in
                        prefixTile(prefix: prefix)
                    }
                }
            } else {
                Text("Cached chats will show up here as you use the app.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 100)
            }
            if let evictions = sessionBank?.evictionLog, !evictions.isEmpty {
                Divider().padding(.vertical, 4)
                MicroHeader("Recent evictions", systemImage: "tray.and.arrow.up")
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(evictions.prefix(6).enumerated()), id: \.offset) { _, eviction in
                        HStack(spacing: 8) {
                            Image(systemName: "minus.circle")
                                .foregroundStyle(Color.mtplxWarning)
                                .font(.caption2)
                            Text(eviction.string("reason") ?? "unknown")
                                .font(.caption)
                            Spacer()
                            if let bytes = eviction.int("bytes") {
                                Text(Format.bytes(bytes))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .monospacedDigit()
                            }
                            if let session = eviction.string("session_id") {
                                Text(String(session.prefix(8)))
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                                    .monospacedDigit()
                            }
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func prefixTile(prefix: SessionBankPrefix) -> some View {
        let age = max(0, Date().timeIntervalSince1970 - prefix.lastAccessS)
        let isHot = age < 60
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(isHot ? Color.mtplxSuccess : Color.gray)
                    .frame(width: 8, height: 8)
                Text(String(prefix.sessionId.prefix(12)))
                    .font(.system(.caption, design: .monospaced).weight(.medium))
                    .lineLimit(1)
                Spacer()
                Text(Format.relative(from: age))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            HStack(spacing: 10) {
                Label("\(Format.integer(prefix.prefixLen)) tok", systemImage: "text.alignleft")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
                Label(Format.bytes(prefix.nbytes), systemImage: "internaldrive")
                    .labelStyle(.titleAndIcon)
                    .font(.caption2)
            }
            .foregroundStyle(.secondary)
            HStack {
                if prefix.hasLiveRef == true {
                    PillBadge(text: "live", systemImage: "bolt.fill", tint: Brand.accentChrome)
                } else {
                    PillBadge(text: "cached", systemImage: "checkmark", tint: .secondary)
                }
                Text("\(prefix.hits) hits")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.raisedSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(
                            isHot ? Brand.success.opacity(0.55) : Brand.separator,
                            lineWidth: 0.75
                        )
                )
        )
    }

    @ViewBuilder
    private func sessionsListCard(sessions: SessionsPayload?) -> some View {
        Card("Engine Sessions") {
            if let sessions, !sessions.sessions.isEmpty {
                VStack(spacing: 4) {
                    ForEach(sessions.sessions) { session in
                        sessionRow(session)
                        Divider().opacity(0.3)
                    }
                }
            } else {
                Text("No active sessions.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 80)
            }
        }
    }

    @ViewBuilder
    private func sessionRow(_ session: SessionRow) -> some View {
        let canClearSession = backend.capabilities?.features["session_clear"] != false
        HStack {
            Image(systemName: "circle.fill")
                .font(.caption2)
                .foregroundStyle(session.inFlight == true ? Color.mtplxSuccess : .secondary)
            Text(session.sessionId)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(1)
            Spacer()
            Label("\(Format.integer(session.prefixLen)) tok",
                  systemImage: "text.alignleft")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Label(Format.bytes(session.bytes),
                  systemImage: "internaldrive")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(Format.relative(from: session.ageSeconds))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .frame(width: 64, alignment: .trailing)
            if canClearSession {
                Button {
                    clearingSession = session.sessionId
                    Task {
                        defer { Task { @MainActor in clearingSession = nil } }
                        try? await backend.clearSession(sessionId: session.sessionId)
                    }
                } label: {
                    if clearingSession == session.sessionId {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: "trash")
                    }
                }
                .buttonStyle(.borderless)
                .help("Drop this session's cached prefix")
                .accessibilityLabel("Clear session cache")
            }
        }
        .padding(.vertical, 4)
    }
}
