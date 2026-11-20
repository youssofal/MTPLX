import SwiftUI
import MTPLXAppCore

// MARK: - SystemTab
//
// V1 consolidation of V0 MemoryTab + ThermalTab. Memory pressure on top,
// thermal/fan rings below — both surfaces of the same "is this machine
// happy?" question. Piano-black + chrome styling, FanRing redrawn with
// Brand palette.

struct SystemTab: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        Group {
            if backend.daemonState.kind == .stopped {
                EmptyStateView(
                    symbol: "cpu",
                    title: tr("Nothing to show yet"),
                    message: tr("Start a model to see memory and thermal data.")
                ) {
                    Task { await backend.startDaemon() }
                }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        hardwareBanner
                        memoryStackedCard
                        memoryDetailsCard
                        ruleCard
                        fansCard
                    }
                    .padding(20)
                }
            }
        }
    }

    // MARK: - Hardware

    @ViewBuilder
    private var hardwareBanner: some View {
        let machine = backend.snapshot?.machine
        let health = backend.health
        let chip = machine?.chipName ?? health?.chipName
        let model = machine?.machineModel ?? health?.machineModel
        let unified = machine?.unifiedMemoryBytes ?? health?.unifiedMemoryBytes

        Card(tr("Hardware")) {
            ViewThatFits(in: .horizontal) {
                // Top-aligned so every tile's label sits on one line; the
                // chip tile's extra caption used to push its label above
                // the others.
                HStack(alignment: .top, spacing: 24) {
                    hardwareTiles(
                        chip: chip,
                        model: model,
                        unified: unified,
                        health: health,
                        includeDividers: true
                    )
                }

                LazyVGrid(
                    columns: [
                        GridItem(.adaptive(minimum: 132), alignment: .leading),
                    ],
                    alignment: .leading,
                    spacing: 16
                ) {
                    hardwareTiles(
                        chip: chip,
                        model: model,
                        unified: unified,
                        health: health,
                        includeDividers: false
                    )
                }
            }
        }
    }

    @ViewBuilder
    private func hardwareTiles(
        chip: String?,
        model: String?,
        unified: Int?,
        health: HealthPayload?,
        includeDividers: Bool
    ) -> some View {
        StatTile(
            label: chip == nil ? tr("Mac model") : tr("Chip"),
            value: chip ?? model ?? tr("Apple Silicon"),
            systemImage: "cpu",
            caption: chip == nil ? nil : model
        )
        if includeDividers {
            Divider().frame(height: 36).background(Brand.separator)
        }
        StatTile(
            label: tr("Unified Memory"),
            value: Format.gigabytes(unified),
            systemImage: "memorychip",
            tint: Brand.accent
        )
        if let health {
            if includeDividers {
                Divider().frame(height: 36).background(Brand.separator)
            }
            StatTile(
                label: tr("Context window"),
                value: Format.integer(health.contextWindow),
                unit: "tok",
                systemImage: "text.line.first.and.arrowtriangle.forward"
            )
            if includeDividers {
                Divider().frame(height: 36).background(Brand.separator)
            }
            StatTile(
                label: tr("Generation"),
                value: health.generationMode.uppercased(),
                systemImage: "arrow.triangle.branch",
                tint: health.mtpEnabled ? Brand.success : Brand.textHighlight
            )
        }
    }

    // MARK: - Memory

    @ViewBuilder
    private var memoryStackedCard: some View {
        let snapshot = backend.snapshot
        let health = backend.health
        let mem = backend.mem
        let unified = snapshot?.machine.unifiedMemoryBytes ?? health?.unifiedMemoryBytes

        Card(tr("Memory"),
             subtitle: tr("What MTPLX holds, against the limit its memory guard works to.")) {
            if let mem, mem.ok, let engine = backend.engineMemoryReading {
                engineMemoryBar(mem: mem, engine: engine)
            } else if let mem, mem.ok, let total = unified, total > 0 {
                // Daemons that report no memory limit: MLX's own account
                // against the Mac's RAM, as before.
                let active = max(0, Double(mem.activeMemoryBytes ?? 0))
                let cache = max(0, Double(mem.cacheMemoryBytes ?? 0))
                let peak = Double(mem.peakMemoryBytes ?? 0)
                let weights = max(0, Double(mem.modelWeightsBytes ?? 0))
                let bank = max(0, Double(mem.sessionBankBytes ?? 0))
                // Derived remainder of active; recomputed here so the three
                // segments always sum to the active bar even if the daemon
                // snapshot raced a prefill.
                let working = max(0, active - weights - bank)
                let used = active + cache
                let headroom = max(0, Double(total) - used)
                let hasAttribution = weights > 0
                // One legend: each part with its size (the bar shows the
                // share); it used to repeat every part twice, once as a
                // percent and once in GB.
                StackedBar(
                    segments: (hasAttribution
                        ? [
                            memorySegment(tr("Model"), min(weights, active), Brand.accent),
                            memorySegment(tr("Sessions"), bank, Brand.coolChrome),
                            memorySegment(tr("Working"), working, Brand.warning.opacity(0.75)),
                            memorySegment(tr("Reusable"), cache, Brand.textHighlight.opacity(0.5)),
                        ]
                        : [
                            memorySegment(tr("Active"), active, Brand.accent),
                            memorySegment(tr("Cache"), cache, Brand.coolChrome),
                        ])
                        + [memorySegment(tr("Headroom"), headroom, Brand.textHighlight.opacity(0.35))],
                    total: Double(total),
                    height: 22,
                    legendExtras: peakLegend(peak)
                )
            } else if let mem, !mem.ok {
                Text(mem.error ?? tr("Memory snapshot unavailable."))
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            } else {
                Text(tr("Memory stats appear once the model loads."))
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            }
        }
    }

    /// The bar is the engine's limit, the line the memory guard trims the
    /// warm cache and sheds work against; "Room" is what is left under it.
    /// The Mac's whole RAM was the wrong whole: it showed 33.8% headroom
    /// while the guard was trimming (12 November).
    @ViewBuilder
    private func engineMemoryBar(mem: MemSnapshot, engine: EngineMemoryReading) -> some View {
        let active = max(0, Double(mem.activeMemoryBytes ?? 0))
        let pool = max(0, Double(mem.cacheMemoryBytes ?? 0))
        let weights = min(max(0, Double(mem.modelWeightsBytes ?? 0)), active)
        let bank = max(0, Double(mem.sessionBankBytes ?? 0))
        let hasAttribution = weights > 0
        // Derived remainder of active, so the segments always sum to it
        // even when the snapshot raced a prefill.
        let working = max(0, active - weights - bank)
        // Host memory the guard counts, shown once it is a tenth of a GB:
        // under that it reads "0.0 GB" and only adds noise to the legend.
        let outside = Double(engine.chargedOutsideMLXBytes)
        let showsOutside = outside >= 0.1 * 1_073_741_824
        let room = Double(max(0, engine.roomBytes))
        let over = max(0, -engine.roomBytes)
        let level = backend.memoryPressureLevel
        let statusTint: Color = over > 0 || level >= 4
            ? Brand.danger
            : (level >= 2 ? Brand.warning : Brand.success)
        let segments: [StackedBarSegment] = (hasAttribution
            ? [
                memorySegment(tr("Model"), weights, Brand.accent),
                memorySegment(tr("Sessions"), bank, Brand.coolChrome),
                memorySegment(tr("Working"), working, Brand.warning.opacity(0.75)),
            ]
            : [memorySegment(tr("Active"), active, Brand.accent)])
            + [memorySegment(tr("Reusable"), pool, Brand.textHighlight.opacity(0.5))]
            + (showsOutside
                ? [memorySegment(tr("Host memory"), outside, Brand.danger.opacity(0.6))]
                : [])
            + [memorySegment(tr("Room"), room, Brand.textHighlight.opacity(0.35))]

        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Circle()
                    .fill(statusTint)
                    .frame(width: 8, height: 8)
                Text(tr(
                    "%@ of the %@ limit",
                    Format.gigabytes(engine.usedBytes),
                    Format.gigabytes(engine.limitBytes)
                ))
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(Brand.typeHi)
                Text(over > 0
                     ? tr("over by %@", Format.gigabytes(over))
                     : tr("%@ room", Format.gigabytes(engine.roomBytes)))
                    .font(.system(.caption, design: .rounded).weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(over > 0 ? Brand.danger : Brand.typeSecondary)
                Spacer(minLength: 8)
                if backend.memoryGuardRecentShed {
                    PillBadge(
                        text: tr("Guard trimming cache"),
                        systemImage: "scissors",
                        tint: Brand.warning,
                        emphasized: true
                    )
                }
            }
            StackedBar(
                segments: segments,
                total: Double(max(engine.limitBytes, engine.usedBytes)),
                height: 22,
                legendExtras: peakLegend(Double(mem.peakMemoryBytes ?? 0))
            )
        }
    }

    /// A bar part labelled with its size in GB (binary, as macOS labels memory).
    private func memorySegment(_ label: String, _ bytes: Double, _ tint: Color) -> StackedBarSegment {
        StackedBarSegment(label: label, value: bytes, tint: tint, valueLabel: Format.gigabytes(Int(bytes)))
    }

    /// The process's peak, named in the legend without a share of the bar.
    private func peakLegend(_ peakBytes: Double) -> [StackedBarSegment] {
        guard peakBytes > 0 else { return [] }
        return [StackedBarSegment(
            label: tr("Peak"),
            value: 0,
            tint: Brand.danger,
            valueLabel: Format.gigabytes(Int(peakBytes))
        )]
    }

    @ViewBuilder
    private var memoryDetailsCard: some View {
        let snapshot = backend.snapshot
        let health = backend.health
        let mem = backend.mem
        let unified = snapshot?.machine.unifiedMemoryBytes ?? health?.unifiedMemoryBytes

        Card(tr("Memory Detail")) {
            VStack(spacing: 6) {
                if let weights = mem?.modelWeightsBytes, weights > 0 {
                    MetricRow(label: tr("Model weights"), value: Format.bytes(weights))
                    MetricRow(label: tr("Session cache (RAM)"), value: Format.bytes(mem?.sessionBankBytes ?? 0))
                    MetricRow(label: tr("Generation working set"), value: Format.bytes(mem?.generationWorkingBytes ?? 0))
                }
                // Flash-Next n-gram table in streamed mode: ~30 GB of the
                // pack that reads from SSD as reclaimable file-backed pages
                // — deliberately not wired, so it never appears in Active
                // and the pack's disk size would otherwise not add up.
                if let table = backend.memoryPlan?.ngramTableStreamedBytes,
                   table > 0
                {
                    MetricRow(
                        label: tr("N-gram table (streamed from SSD)"),
                        value: Format.bytes(table)
                    )
                }
                // Retrieval weights are not part of the chat model's shard
                // total, so without this row several GB would be unattributed.
                if let retrieval = snapshot?.retrieval, retrieval.enabled {
                    MetricRow(
                        label: tr("Retrieval models"),
                        value: retrieval.residentBytes > 0
                            ? Format.bytes(retrieval.residentBytes)
                            : tr("none loaded")
                    )
                }
                MetricRow(label: tr("Active (total in use)"), value: Format.bytes(mem?.activeMemoryBytes))
                MetricRow(label: tr("Reusable buffer pool"), value: Format.bytes(mem?.cacheMemoryBytes))
                // The process as macOS counts it, and the part MLX's own
                // account does not explain (host memory: Python, the session
                // bank's host buffers, leaks). The guard charges what passes
                // the seat's allowance.
                if let footprint = mem?.physFootprintBytes, footprint > 0 {
                    MetricRow(label: tr("Process footprint (macOS)"), value: Format.bytes(footprint))
                }
                if let outside = mem?.hostOverhangBytes {
                    // The bar's "Host memory" is the counted part: what is
                    // past the allowance the guard grants every seat.
                    let counted = backend.engineMemoryReading?.chargedOutsideMLXBytes ?? 0
                    MetricRow(
                        label: tr("Host memory outside MLX"),
                        value: counted > 0
                            ? tr("%@ (%@ counted)", Format.bytes(outside), Format.bytes(counted))
                            : Format.bytes(outside)
                    )
                }
                MetricRow(
                    label: tr("Peak"),
                    value: Format.bytes(mem?.peakMemoryBytes),
                    valueTint: (mem?.peakMemoryBytes ?? 0) > Int(Double(unified ?? 0) * 0.85)
                        ? Brand.warning : Brand.accent
                )
                if let engine = backend.engineMemoryReading {
                    MetricRow(
                        label: tr("Memory guard reading"),
                        value: tr(
                            "%@ (%@ of limit)",
                            Format.bytes(engine.usedBytes),
                            Format.percent(engine.fraction, fractionDigits: 0)
                        ),
                        valueTint: engine.roomBytes < 0 ? Brand.danger : Brand.accent
                    )
                }
                if let limit = backend.engineMemoryLimitBytes {
                    MetricRow(label: tr("Engine memory limit"), value: Format.bytes(limit))
                    if let unified, unified > limit {
                        MetricRow(label: tr("Kept for macOS and other apps"), value: Format.bytes(unified - limit))
                    }
                }
                MetricRow(label: tr("Unified total"), value: Format.bytes(unified))
                if let mem, !mem.ok {
                    MetricRow(label: tr("Error"), value: mem.error ?? "unknown", valueTint: Brand.danger)
                }
            }
        }
    }

    // MARK: - Thermal

    @ViewBuilder
    private var ruleCard: some View {
        let pollingEnabled = backend.configuration.enableThermalPolling

        Card(tr("Cooling"), padding: 16) {
            PillBadge(
                text: pollingEnabled ? tr("Monitoring fans") : tr("Fans not monitored"),
                systemImage: pollingEnabled ? "checkmark.seal.fill" : "exclamationmark.triangle",
                tint: pollingEnabled ? Brand.success : Brand.warning,
                emphasized: !pollingEnabled
            )
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                Text(tr("For accurate benchmarks, your fans need to be at max. Thermal throttling makes numbers meaningless."))
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight)
                Text(tr("Turn on Thermal Polling in Settings to confirm. Performance Lock also calms the UI so it doesn't compete with the model."))
                    .font(.caption)
                    .foregroundStyle(Brand.textHighlight.opacity(0.65))
                if !pollingEnabled {
                    openCoolingSettingsButton
                }
            }
        }
    }

    private var openCoolingSettingsButton: some View {
        Button {
            router.openSettings(section: .cooling)
        } label: {
            Label(tr("Open Cooling settings"), systemImage: "fan")
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    @ViewBuilder
    private var fansCard: some View {
        let thermal = backend.thermal
        let pollingEnabled = backend.configuration.enableThermalPolling

        Card(tr("Fans")) {
            if pollingEnabled, let thermal, thermal.ok, !thermal.fans.isEmpty {
                HStack(alignment: .top, spacing: 24) {
                    ForEach(Array(thermal.fans.enumerated()), id: \.offset) { idx, fan in
                        FanRing(
                            label: tr("Fan %lld", idx + 1),
                            actualRpm: fan.actualRpm,
                            targetRpm: fan.targetRpm,
                            minRpm: thermal.minRpm,
                            maxRpm: fan.maxCapacityRpm ?? thermal.maxRpm,
                            mode: fan.mode
                        )
                    }
                    Spacer()
                }
            } else if pollingEnabled, let thermal, !thermal.ok {
                Text(tr("Can't read fan data. The helper might be missing or doesn't have permission."))
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            } else if !pollingEnabled {
                VStack(alignment: .leading, spacing: 8) {
                    Text(tr("Fan monitoring is off by default so things stay quiet."))
                        .font(.callout)
                        .foregroundStyle(Brand.textHighlight)
                    Text(tr("Turn it on in Settings → Cooling to see live fan speeds."))
                        .font(.caption)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                }
            } else {
                Text(tr("Waiting for fan data…"))
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            }
        }
    }
}

// MARK: - FanRing
//
// Circular gauge showing actual vs target RPM relative to the fan's max
// capacity. Same control logic as the V0 ThermalTab's FanRing, but
// repainted with Brand chrome-on-piano colors. Canvas is appropriate
// here because it draws once on data change (not on every frame).

struct FanRing: View {
    let label: String
    let actualRpm: Int?
    let targetRpm: Int?
    let minRpm: Int?
    let maxRpm: Int?
    let mode: String?

    @State private var animated: Double = 0

    var body: some View {
        let actual = Double(actualRpm ?? 0)
        let target = Double(targetRpm ?? 0)
        let maxR = Double(maxRpm ?? Int(max(1, actual, target)))
        let ratio = maxR > 0 ? min(1, actual / maxR) : 0
        let targetRatio = maxR > 0 ? min(1, target / maxR) : 0
        let verifiedMax = ratio >= 0.90
        let activeColor = verifiedMax ? Brand.success : Brand.warning

        VStack(spacing: 10) {
            ZStack {
                Canvas { context, size in
                    let center = CGPoint(x: size.width / 2, y: size.height / 2)
                    let radius = min(size.width, size.height) / 2 - 8
                    let lineWidth: CGFloat = max(10, radius * 0.14)

                    var track = Path()
                    track.addArc(
                        center: center,
                        radius: radius,
                        startAngle: .degrees(135),
                        endAngle: .degrees(45),
                        clockwise: false
                    )
                    context.stroke(track,
                        with: .color(Brand.separator),
                        style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                    )

                    let sweep = 270.0 * animated
                    if sweep > 0 {
                        var arc = Path()
                        arc.addArc(
                            center: center,
                            radius: radius,
                            startAngle: .degrees(135),
                            endAngle: .degrees(135 + sweep),
                            clockwise: false
                        )
                        context.stroke(arc,
                            with: .color(activeColor),
                            style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                        )
                    }

                    if targetRatio > 0 && targetRatio <= 1 {
                        let angleDeg = 135.0 + 270.0 * targetRatio
                        let rad = angleDeg * .pi / 180.0
                        let cosV = CGFloat(cos(rad))
                        let sinV = CGFloat(sin(rad))
                        let innerR = radius - lineWidth / 2 - 2
                        let outerR = radius + lineWidth / 2 + 2
                        let inner = CGPoint(x: center.x + innerR * cosV, y: center.y + innerR * sinV)
                        let outer = CGPoint(x: center.x + outerR * cosV, y: center.y + outerR * sinV)
                        var tick = Path()
                        tick.move(to: inner)
                        tick.addLine(to: outer)
                        context.stroke(tick,
                            with: .color(Brand.accent),
                            style: StrokeStyle(lineWidth: 2, lineCap: .round)
                        )
                    }
                }
                .frame(width: 160, height: 160)

                VStack(spacing: 2) {
                    Text(Format.integer(actualRpm))
                        .font(.system(.title2, design: .rounded).weight(.bold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                        .foregroundStyle(Brand.chromeFill)
                    Text(tr("rpm"))
                        .font(.system(size: 10, weight: .heavy, design: .monospaced))
                        .tracking(1.5)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    if let mode {
                        Text(mode.uppercased())
                            .font(.system(size: 10, weight: .heavy, design: .monospaced))
                            .tracking(1.5)
                            .foregroundStyle(verifiedMax ? Brand.success : Brand.textHighlight.opacity(0.7))
                    }
                }
            }
            VStack(spacing: 4) {
                Text(label)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Brand.textHighlight)
                if let target = targetRpm {
                    Text(tr("target %@ rpm", Format.integer(target)))
                        .font(.caption2)
                        .foregroundStyle(Brand.textHighlight.opacity(0.5))
                }
                if let minR = minRpm, let maxR = maxRpm {
                    Text(tr("range %lld–%lld", minR, maxR))
                        .font(.caption2)
                        .foregroundStyle(Brand.textHighlight.opacity(0.4))
                }
            }
        }
        .onAppear { animated = ratio }
        .onChange(of: ratio) { _, newValue in
            withAnimation(.easeOut(duration: 0.35)) { animated = newValue }
        }
    }
}
