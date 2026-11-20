import AppKit
import SwiftUI
import MTPLXAppCore

// MARK: - MenuBarDashboard
//
// The panel the menu bar item opens: MTPLX's speed at a glance. A
// monochrome speedometer for decode (prefill while a prompt is read), then
// prefill speed, first token, context used, memory, cache reuse and draft
// acceptance, the model's name, Start or Stop, Open MTPLX and Quit. It
// reads only what the app already receives from the engine's stream (no
// request, no endpoint), and it exists only while open: MenuBarController
// builds it on open and releases it on close. It paints no background of
// its own, so the popover's system material shows through; on macOS 26 its
// cards are glass.

struct MenuBarDashboard: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            MenuBarSpeedometer(reading: dialReading)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .menuBarGlassCard()
            tiles
            primaryActionButton
            HStack {
                Button(tr("Open MTPLX")) { MenuBarController.showMainWindow() }
                Spacer()
                Button(tr("Quit MTPLX")) { NSApp.terminate(nil) }
            }
            .buttonStyle(.borderless)
            .font(.system(size: 12, weight: .medium))
        }
        .padding(14)
        .frame(width: 320)
        .appliesAppearance()
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                WordmarkView(height: 15, fallbackTracking: true)
                Spacer()
                MenuBarStatus(daemonState: backend.daemonState, connectionState: backend.connectionState)
            }
            if let model = backend.snapshot?.modelId, !model.isEmpty {
                Text(MTPLXModelOption.displayName(for: model))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    // MARK: The dial

    private var prefilling: PrefillState? {
        guard !backend.inFlight.contains(where: { $0.hasDecodeProgress }) else { return nil }
        return backend.inFlight.lazy.compactMap(\.prefillState).first { $0.tokensTotal > 0 }
    }

    private var dialReading: MenuBarSpeedometer.Reading {
        let best = backend.observedCompletionCount > 0 ? backend.rolling?.stickyAllTimeMax : nil
        if let prefill = prefilling {
            return .init(value: prefill.displayPrefillTokS, phase: tr("PREFILL"), live: true, best: nil)
        }
        if backend.daemonState.kind == .running, case .live(let value) = backend.headlineDecode,
           backend.inFlight.contains(where: { $0.hasDecodeProgress }) {
            return .init(value: value, phase: tr("DECODE"), live: true, best: best)
        }
        return .init(value: backend.headlineDecode.value ?? backend.latest?.decodeTokS, phase: tr("DECODE"), live: false, best: best)
    }

    // MARK: The six figures
    //
    // From the readouts the Live tab uses (Readouts.swift), behind the same
    // gate, so the panel and the window never disagree: nothing a request
    // decides shows until a real request has finished in this session (the
    // engine's own warm-up is not a turn). First text and cache reuse are the
    // last finished turn's, which hold still while the next one runs.

    private var tiles: some View {
        let hasRealRequest = backend.observedCompletionCount > 0
        let snapshot = backend.snapshot
        let lastTurn = hasRealRequest ? (snapshot?.recent.last ?? backend.latest) : nil
        let prefill: PrefillSpeedSummary? = hasRealRequest
            ? PrefillSpeedSummary(lifetime: snapshot?.lifetime) ?? PrefillSpeedSummary(recent: snapshot?.recent ?? [])
            : nil
        let context: String = {
            guard let window = backend.health?.contextWindow, window > 0 else { return "—" }
            let used = backend.inFlight.lazy.compactMap(\.contextTokens).max()
                ?? backend.sessions?.sessions.map(\.prefixLen).max()
                ?? 0
            return "\(Format.compactTokens(used)) / \(Format.compactTokens(window))"
        }()
        let memory: String = {
            if let engine = backend.engineMemoryReading {
                return "\(Format.gigabytes(engine.usedBytes)) / \(Format.gigabytes(engine.limitBytes))"
            }
            let used = (backend.mem?.activeMemoryBytes ?? 0) + (backend.mem?.cacheMemoryBytes ?? 0)
            return used > 0 ? Format.gigabytes(used) : "—"
        }()
        let reuse: String = {
            guard let lastTurn, let prompt = lastTurn.values["prompt_tokens"]?.intValue, prompt > 0 else { return "—" }
            let cached = lastTurn.values["cached_tokens"]?.intValue ?? 0
            return Format.percent(Double(cached) / Double(prompt), fractionDigits: 0)
        }()
        return Grid(horizontalSpacing: 8, verticalSpacing: 8) {
            GridRow {
                MenuBarTile(
                    label: tr("Avg Prefill"),
                    value: prefill.map { tr("%@ tok/s", Format.tps($0.averageTokS)) } ?? "—"
                )
                MenuBarTile(label: tr("TTFT"), value: lastTurn?.ttftS.map { Format.duration($0) } ?? "—")
            }
            GridRow {
                MenuBarTile(label: tr("Context"), value: context)
                MenuBarTile(label: tr("Memory"), value: memory)
            }
            GridRow {
                MenuBarTile(label: tr("Cached"), value: reuse)
                MenuBarTile(
                    label: tr("ACCEPTANCE"),
                    value: hasRealRequest ? Format.percent(backend.smoothedMetrics.meanAcceptance) : "—"
                )
            }
        }
    }

    // MARK: Start or stop

    @ViewBuilder
    private var primaryActionButton: some View {
        let kind = backend.daemonState.kind
        let (label, action): (String, () -> Void) = {
            switch kind {
            case .stopped, .crashed, .degraded:
                return (tr("Start MTPLX"), { Task { await backend.startDaemon() } })
            case .starting, .warming:
                return (tr("Stop (loading…)"), {
                    Task { await stopCoordinator.stopAll(reason: "menu_bar_stop_loading") }
                })
            case .running:
                return (tr("Stop MTPLX"), {
                    Task { await stopCoordinator.stopAll(reason: "menu_bar_stop_running") }
                })
            case .stopping:
                return (tr("Stopping…"), {})
            }
        }()
        Button(action: action) {
            Text(label)
                .font(.system(size: 12, weight: .semibold))
                .frame(maxWidth: .infinity)
        }
        .controlSize(.large)
        .disabled(kind == .stopping)
    }
}

// MARK: - MenuBarTile

private struct MenuBarTile: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .textCase(.uppercase)
                .font(.system(size: 9, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .menuBarGlassCard()
        .accessibilityElement(children: .combine)
    }
}

// MARK: - MenuBarStatus

/// The engine's state in the panel's header, in the panel's two tones: a
/// filled dot while it serves, a faint one otherwise, and the state's name.
private struct MenuBarStatus: View {
    let daemonState: DaemonState
    let connectionState: MetricsConnectionState

    var body: some View {
        let badge = DaemonStatusBadge(daemonState: daemonState, connectionState: connectionState)
        HStack(spacing: 5) {
            Circle()
                .fill(badge.tone == .healthy ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary.opacity(0.5)))
                .frame(width: 6, height: 6)
            Text(badge.label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .help(badge.help)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - MenuBarSpeedometer

/// A 240-degree monochrome dial: a hairline track, the value arc, a thin
/// needle with the M as its hub, the number in heavy digits under it, and a
/// small mark at the model's best decode so far. White on dark, near-black
/// on light (the primary colour), the track in the secondary colour.
struct MenuBarSpeedometer: View {
    struct Reading: Equatable {
        var value: Double?
        var phase: String
        var live: Bool
        var best: Double?
    }

    let reading: Reading

    static let sweep = 240.0
    static let startAngle = 150.0  // degrees, 0 = east, clockwise on screen
    static let radius: CGFloat = 58
    /// The pivot sits high so the number reads under it, inside the dial's
    /// open bottom, never across the M.
    static let pivotY: CGFloat = radius + 12
    static let height: CGFloat = 128

    /// The scale's top: the next round number above the larger of the value
    /// and the model's best, so a needle never pins.
    static func scaleTop(value: Double?, best: Double?) -> Double {
        let peak = max(value ?? 0, best ?? 0, 1)
        for step in [25.0, 50, 100, 150, 200, 300, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000] where peak <= step * 0.92 {
            return step
        }
        return (peak / 10_000).rounded(.up) * 10_000
    }

    var body: some View {
        let top = Self.scaleTop(value: reading.value, best: reading.best)
        let fraction = min(1, max(0, (reading.value ?? 0) / top))
        ZStack(alignment: .top) {
            Canvas { context, size in
                let center = CGPoint(x: size.width / 2, y: Self.pivotY)
                let radius = Self.radius
                func point(_ f: Double, _ r: Double) -> CGPoint {
                    let angle = (Self.startAngle + Self.sweep * f) * .pi / 180
                    return CGPoint(x: center.x + r * cos(angle), y: center.y + r * sin(angle))
                }
                var track = Path()
                track.addArc(center: center, radius: radius, startAngle: .degrees(Self.startAngle),
                             endAngle: .degrees(Self.startAngle + Self.sweep), clockwise: false)
                context.stroke(track, with: .style(.secondary.opacity(0.45)), lineWidth: 2)
                for tick in 0...4 {
                    let f = Double(tick) / 4
                    var mark = Path()
                    mark.move(to: point(f, radius - 7))
                    mark.addLine(to: point(f, radius - 3))
                    context.stroke(mark, with: .style(.secondary.opacity(0.6)), lineWidth: 1)
                }
                if fraction > 0 {
                    var value = Path()
                    value.addArc(center: center, radius: radius, startAngle: .degrees(Self.startAngle),
                                 endAngle: .degrees(Self.startAngle + Self.sweep * fraction), clockwise: false)
                    context.stroke(value, with: .style(.primary.opacity(reading.live ? 1 : 0.55)),
                                   style: StrokeStyle(lineWidth: 4, lineCap: .round))
                }
                if let best = reading.best, best > 0 {
                    let f = min(1, best / top)
                    var mark = Path()
                    mark.move(to: point(f, radius + 3))
                    mark.addLine(to: point(f, radius + 8))
                    context.stroke(mark, with: .style(.primary.opacity(0.7)), lineWidth: 1.5)
                }
                var needle = Path()
                needle.move(to: center)
                needle.addLine(to: point(fraction, radius - 10))
                context.stroke(needle, with: .style(.primary.opacity(reading.live ? 1 : 0.55)),
                               style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                let hub = MenuBarMark.image(height: 12, dimmed: false, badge: .none)
                let hubSize = hub.size
                // A template image in a Canvas takes no colour of its own: it
                // drew white in light mode too. The primary colour, as the needle.
                var hubImage = context.resolve(Image(nsImage: hub).renderingMode(.template))
                hubImage.shading = .style(.primary)
                context.draw(
                    hubImage,
                    in: CGRect(x: center.x - hubSize.width / 2, y: center.y - hubSize.height / 2,
                               width: hubSize.width, height: hubSize.height)
                )
            }
            .frame(height: Self.height)
            VStack(spacing: 1) {
                Text(reading.value.map { Format.tps($0) } ?? "—")
                    .font(.system(size: 26, weight: .heavy, design: .rounded))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                    .frame(maxWidth: Self.radius * 1.5)  // inside the arc's two ends
                    .foregroundStyle(reading.live ? .primary : .secondary)
                Text("\(tr("tok/s")) · \(reading.phase)")
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(1)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, Self.pivotY + 10)
        }
        .frame(height: Self.height)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(reading.phase)
        .accessibilityValue(reading.value.map { tr("%@ tok/s", Format.tps($0)) } ?? "—")
    }
}

// MARK: - Glass

private struct MenuBarGlassCard: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        } else {
            content.background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }
}

extension View {
    fileprivate func menuBarGlassCard() -> some View {
        modifier(MenuBarGlassCard())
    }
}
