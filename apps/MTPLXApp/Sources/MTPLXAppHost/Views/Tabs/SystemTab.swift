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

    var body: some View {
        Group {
            if backend.daemonState.kind == .stopped {
                EmptyStateView(
                    symbol: "cpu",
                    title: "Nothing to show yet",
                    message: "Start a model to see memory and thermal data."
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

        Card("Hardware") {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 24) {
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
            label: chip == nil ? "Mac model" : "Chip",
            value: chip ?? model ?? "Apple Silicon",
            systemImage: "cpu",
            caption: chip == nil ? nil : model
        )
        if includeDividers {
            Divider().frame(height: 36).background(Brand.separator)
        }
        StatTile(
            label: "Unified Memory",
            value: Format.gigabytes(unified),
            systemImage: "memorychip",
            tint: Brand.accent
        )
        if let health {
            if includeDividers {
                Divider().frame(height: 36).background(Brand.separator)
            }
            StatTile(
                label: "Context window",
                value: Format.integer(health.contextWindow),
                unit: "tok",
                systemImage: "text.line.first.and.arrowtriangle.forward"
            )
            if includeDividers {
                Divider().frame(height: 36).background(Brand.separator)
            }
            StatTile(
                label: "Generation",
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

        Card("Memory",
             subtitle: "What the engine is actually holding vs your Mac's total.") {
            if let mem, mem.ok, let total = unified, total > 0 {
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
                VStack(alignment: .leading, spacing: 14) {
                    if hasAttribution {
                        StackedBar(
                            segments: [
                                StackedBarSegment(label: "Model", value: min(weights, active), tint: Brand.accent),
                                StackedBarSegment(label: "Sessions", value: bank, tint: Brand.coolChrome),
                                StackedBarSegment(label: "Working", value: working, tint: Brand.warning.opacity(0.75)),
                                StackedBarSegment(label: "Reusable", value: cache, tint: Brand.textHighlight.opacity(0.5)),
                                StackedBarSegment(label: "Headroom", value: headroom, tint: Brand.textHighlight.opacity(0.35)),
                            ],
                            total: Double(total),
                            height: 22
                        )
                        HStack(spacing: 14) {
                            memoryTag(color: Brand.accent, label: "Model", value: Format.gigabytes(Int(min(weights, active))))
                            memoryTag(color: Brand.coolChrome, label: "Sessions", value: Format.gigabytes(Int(bank)))
                            memoryTag(color: Brand.warning.opacity(0.75), label: "Working", value: Format.gigabytes(Int(working)))
                            memoryTag(color: Brand.textHighlight.opacity(0.6), label: "Reusable", value: Format.gigabytes(Int(cache)))
                            if peak > 0 {
                                memoryTag(color: Brand.danger, label: "Peak", value: Format.gigabytes(Int(peak)))
                            }
                        }
                    } else {
                        StackedBar(
                            segments: [
                                StackedBarSegment(label: "Active", value: active, tint: Brand.accent),
                                StackedBarSegment(label: "Cache", value: cache, tint: Brand.coolChrome),
                                StackedBarSegment(label: "Headroom", value: headroom, tint: Brand.textHighlight.opacity(0.35)),
                            ],
                            total: Double(total),
                            height: 22
                        )
                        HStack(spacing: 18) {
                            memoryTag(color: Brand.accent, label: "Active", value: Format.gigabytes(Int(active)))
                            memoryTag(color: Brand.coolChrome, label: "Cache", value: Format.gigabytes(Int(cache)))
                            memoryTag(color: Brand.textHighlight.opacity(0.7), label: "Headroom", value: Format.gigabytes(Int(headroom)))
                            if peak > 0 {
                                memoryTag(color: Brand.warning, label: "Peak", value: Format.gigabytes(Int(peak)))
                            }
                        }
                    }
                }
            } else if let mem, !mem.ok {
                Text(mem.error ?? "Memory snapshot unavailable.")
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            } else {
                Text("Memory stats appear once the model loads.")
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            }
        }
    }

    private func memoryTag(color: Color, label: String, value: String) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Brand.textHighlight.opacity(0.7))
            Text(value)
                .font(.system(.caption, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(Brand.accent)
        }
    }

    @ViewBuilder
    private var memoryDetailsCard: some View {
        let snapshot = backend.snapshot
        let health = backend.health
        let mem = backend.mem
        let unified = snapshot?.machine.unifiedMemoryBytes ?? health?.unifiedMemoryBytes

        Card("Memory Detail") {
            VStack(spacing: 6) {
                if let weights = mem?.modelWeightsBytes, weights > 0 {
                    MetricRow(label: "Model weights", value: Format.bytes(weights))
                    MetricRow(label: "Session cache (RAM)", value: Format.bytes(mem?.sessionBankBytes ?? 0))
                    MetricRow(label: "Generation working set", value: Format.bytes(mem?.generationWorkingBytes ?? 0))
                }
                // Retrieval weights are not part of the chat model's shard
                // total, so without this row several GB would be unattributed.
                if let retrieval = snapshot?.retrieval, retrieval.enabled {
                    MetricRow(
                        label: "Retrieval models",
                        value: retrieval.residentBytes > 0
                            ? Format.bytes(retrieval.residentBytes)
                            : "none loaded"
                    )
                }
                MetricRow(label: "Active (total in use)", value: Format.bytes(mem?.activeMemoryBytes))
                MetricRow(label: "Reusable buffer pool", value: Format.bytes(mem?.cacheMemoryBytes))
                MetricRow(
                    label: "Peak",
                    value: Format.bytes(mem?.peakMemoryBytes),
                    valueTint: (mem?.peakMemoryBytes ?? 0) > Int(Double(unified ?? 0) * 0.85)
                        ? Brand.warning : Brand.accent
                )
                MetricRow(label: "Unified total", value: Format.bytes(unified))
                if let mem, !mem.ok {
                    MetricRow(label: "Error", value: mem.error ?? "unknown", valueTint: Brand.danger)
                }
            }
        }
    }

    // MARK: - Thermal

    @ViewBuilder
    private var ruleCard: some View {
        let pollingEnabled = backend.configuration.enableThermalPolling

        Card("Cooling", padding: 16) {
            PillBadge(
                text: pollingEnabled ? "Monitoring fans" : "Fans not monitored",
                systemImage: pollingEnabled ? "checkmark.seal.fill" : "exclamationmark.triangle",
                tint: pollingEnabled ? Brand.success : Brand.warning,
                emphasized: !pollingEnabled
            )
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                Text("For accurate benchmarks, your fans need to be at max. Thermal throttling makes numbers meaningless.")
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight)
                Text("Turn on Thermal Polling in Settings to confirm. Performance Lock also calms the UI so it doesn't compete with the model.")
                    .font(.caption)
                    .foregroundStyle(Brand.textHighlight.opacity(0.65))
            }
        }
    }

    @ViewBuilder
    private var fansCard: some View {
        let thermal = backend.thermal
        let pollingEnabled = backend.configuration.enableThermalPolling

        Card("Fans") {
            if pollingEnabled, let thermal, thermal.ok, !thermal.fans.isEmpty {
                HStack(alignment: .top, spacing: 24) {
                    ForEach(Array(thermal.fans.enumerated()), id: \.offset) { idx, fan in
                        FanRing(
                            label: "Fan \(idx + 1)",
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
                Text("Can't read fan data. The helper might be missing or doesn't have permission.")
                    .font(.callout)
                    .foregroundStyle(Brand.textHighlight.opacity(0.7))
            } else if !pollingEnabled {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Fan monitoring is off by default so things stay quiet.")
                        .font(.callout)
                        .foregroundStyle(Brand.textHighlight)
                    Text("Turn it on in Settings → Thermal to see live fan speeds.")
                        .font(.caption)
                        .foregroundStyle(Brand.textHighlight.opacity(0.65))
                }
            } else {
                Text("Waiting for fan data…")
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
                    Text("rpm")
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
                    Text("target \(Format.integer(target)) rpm")
                        .font(.caption2)
                        .foregroundStyle(Brand.textHighlight.opacity(0.5))
                }
                if let minR = minRpm, let maxR = maxRpm {
                    Text("range \(minR)–\(maxR)")
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
