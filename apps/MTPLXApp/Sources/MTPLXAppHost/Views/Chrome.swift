import SwiftUI
import MTPLXAppCore

// MARK: - DaemonStatePill

/// Compact status chip showing what the daemon is currently doing. Lives
/// in the top chrome strip.
struct DaemonStatePill: View {
    let state: DaemonState

    var body: some View {
        let kind = state.kind
        let detail = state.detail
        return HStack(spacing: 6) {
            stateDot
                .frame(width: 8, height: 8)
            Text(kind.label.uppercased())
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
            if let detail {
                Text("·")
                    .foregroundStyle(Brand.textHighlight.opacity(0.4))
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(Brand.textHighlight.opacity(0.65))
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule(style: .continuous)
                .fill(tint.opacity(0.18))
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(tint.opacity(0.45), lineWidth: 0.75)
                )
        )
        .foregroundStyle(tint)
        .help(helpText)
    }

    private var tint: Color {
        switch state.kind {
        case .running: Brand.success
        case .starting, .warming, .stopping: Brand.warning
        case .degraded, .crashed: Brand.danger
        case .stopped: Brand.textHighlight.opacity(0.7)
        }
    }

    @ViewBuilder
    private var stateDot: some View {
        switch state.kind {
        case .starting, .warming, .stopping:
            ProgressView()
                .controlSize(.mini)
                .scaleEffect(0.55)
                .tint(tint)
        case .running:
            Circle()
                .fill(tint)
                .shadow(color: tint.opacity(0.55), radius: 3)
        default:
            Circle().fill(tint)
        }
    }

    private var helpText: String {
        switch state {
        case .stopped: "MTPLX is not running."
        case .starting: "Spawning the hidden mtplx serve process."
        case .warming: "MTPLX is up, checking health."
        case .running: "MTPLX healthy."
        case .degraded(let msg): "Degraded: \(msg)"
        case .stopping: "Shutting down."
        case .crashed(let status?): "Crashed with status \(status)."
        case .crashed: "Crashed."
        }
    }
}

// MARK: - RunningIndicator
//
// Single combined daemon + stream pulse that replaces the old "RUNNING"
// text pill plus the separate "live" dot. Goes green and pulses only
// when the daemon is fully running AND the SSE metrics stream is open;
// otherwise it shows the most informative intermediate state (starting,
// reconnecting, offline) without splitting attention across two widgets.
struct ConnectionDot: View {
    let daemonState: DaemonState
    let connectionState: MetricsConnectionState

    @State private var pulseKey: Int = 0

    var body: some View {
        HStack(spacing: 6) {
            ZStack {
                Circle()
                    .fill(tint)
                    .frame(width: 8, height: 8)
                    .shadow(color: tint.opacity(0.5), radius: 3)
                if isHealthy {
                    Circle()
                        .stroke(tint.opacity(0.45), lineWidth: 1.5)
                        .frame(width: 14, height: 14)
                        .phaseAnimator(
                            [1.0, 1.8],
                            trigger: pulseKey
                        ) { content, scale in
                            content
                                .scaleEffect(scale)
                                .opacity(2 - scale)
                        } animation: { _ in .easeOut(duration: 1.2) }
                }
            }
            .frame(width: 14, height: 14)
            Text(label)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .tracking(0.5)
                .foregroundStyle(tint.opacity(0.95))
        }
        .onChange(of: stateKey) { _, _ in pulseKey &+= 1 }
        .help(helpText)
    }

    /// True only when both the daemon is `.running` and the SSE stream
    /// is `.open`. Used to gate the pulse so the indicator never lies
    /// about health.
    private var isHealthy: Bool {
        if case .running = daemonState, case .open = connectionState { return true }
        return false
    }

    /// Equatable proxy so `.onChange` fires only on a meaningful state
    /// transition, not on the `reconnecting(n)` counter increment.
    private var stateKey: String {
        "\(daemonState.kind.rawValue)|\(connectionLabel)"
    }

    private var connectionLabel: String {
        switch connectionState {
        case .open: return "open"
        case .connecting: return "connecting"
        case .reconnecting: return "reconnecting"
        case .failed: return "failed"
        case .idle: return "idle"
        }
    }

    private var tint: Color {
        if isHealthy { return .mtplxSuccess }
        switch daemonState.kind {
        case .running:
            // Daemon up but stream not open yet.
            switch connectionState {
            case .failed: return .mtplxDanger
            case .reconnecting, .connecting: return .mtplxWarning
            default: return .mtplxWarning
            }
        case .starting, .warming:
            return .mtplxWarning
        case .stopping:
            return .mtplxWarning
        case .degraded, .crashed:
            return .mtplxDanger
        case .stopped:
            return Brand.textHighlight.opacity(0.55)
        }
    }

    private var label: String {
        if isHealthy { return "Running" }
        // The degraded reason was invisible outside a hover tooltip; users
        // (and their screenshots) only ever saw the bare word "Degraded".
        // Surface a capped reason inline; the full text stays in helpText.
        if case .degraded(let reason) = daemonState {
            let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return "Degraded" }
            let capped = trimmed.count > 44
                ? String(trimmed.prefix(44)).trimmingCharacters(in: .whitespaces) + "…"
                : trimmed
            return "Degraded — \(capped)"
        }
        switch daemonState.kind {
        case .running:
            switch connectionState {
            case .open: return "Running"
            case .connecting: return "Connecting…"
            case .reconnecting(let n): return "Reconnect #\(n)"
            case .failed: return "Offline"
            case .idle: return "Idle"
            }
        case .starting: return "Starting"
        case .warming: return "Warming"
        case .stopping: return "Stopping"
        case .degraded: return "Degraded"
        case .crashed: return "Crashed"
        case .stopped: return "Stopped"
        }
    }

    private var helpText: String {
        if isHealthy { return "Running and ready." }
        switch daemonState {
        case .running:
            switch connectionState {
            case .open: return "Running."
            case .connecting: return "Connecting to live stats…"
            case .reconnecting(let n): return "Reconnecting (attempt \(n))."
            case .failed(let msg): return "Live stats offline: \(msg)"
            case .idle: return "Running. Waiting for stats."
            }
        case .starting: return "Starting up…"
        case .warming: return "Loading the model…"
        case .degraded(let msg): return "Degraded: \(msg)"
        case .stopping: return "Stopping…"
        case .crashed(let status?): return "Crashed (exit code \(status))."
        case .crashed: return "Crashed."
        case .stopped: return "Not running."
        }
    }
}

// MARK: - ProfilePill

struct ProfilePill: View {
    let profile: DynamicObject?
    let model: String?

    var body: some View {
        let name = profile?.profileName ?? "—"
        let family = profile?.profileFamily
        return HStack(spacing: 6) {
            Image(systemName: "speedometer")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(name.capitalized)
                .font(.caption.weight(.medium))
            if let family, family != name {
                Text("·")
                    .foregroundStyle(.tertiary)
                Text(family)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule(style: .continuous)
                .fill(.thinMaterial)
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(Color.mtplxSeparator, lineWidth: 0.5)
                )
        )
        .help(model ?? "Active profile")
    }
}

// MARK: - PerformanceLockToggle

/// Toggle that flips Performance Lock at the configuration level. When
/// on: SSE cadence drops to 1000 ms and the UI suppresses every animation.
/// Critical for the thermal-rule compliance story.
///
/// V1 fix: stream-cadence-only change ⇒ no daemon restart. Just save the
/// new configuration locally and reopen the metrics stream with the
/// updated interval.
struct PerformanceLockToggle: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @State private var isUpdating = false

    var body: some View {
        let active = backend.configuration.performanceLock
        Button {
            apply(!active)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: active ? "lock.fill" : "lock.open")
                    .symbolRenderingMode(.hierarchical)
                    .font(.system(size: 11, weight: .medium))
                Text(active ? "Locked" : "Lock")
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(1)
                if isUpdating {
                    ProgressView().controlSize(.mini)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .fill(active ? Brand.warning.opacity(0.18) : Color.clear)
                    .overlay(
                        Capsule(style: .continuous)
                            .strokeBorder(
                                active ? Brand.warning.opacity(0.55) : Brand.separator,
                                lineWidth: 0.75
                            )
                    )
            )
            .foregroundStyle(active ? Brand.warning : Brand.textHighlight)
        }
        .buttonStyle(.plain)
        .disabled(isUpdating)
        .help(
            active
                ? "Performance Lock is on: 1 Hz polling, animations paused."
                : "Performance Lock pauses animations and drops polling to 1 Hz so the dashboard never steals GPU from the model."
        )
    }

    private func apply(_ next: Bool) {
        isUpdating = true
        var config = backend.configuration
        config.performanceLock = next
        do {
            try backend.saveSettings(config)
            backend.startMetricsStream()
        } catch {
            // The save is best-effort; the SSE restart still picks up the
            // new in-memory configuration.
            backend.startMetricsStream()
        }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(200))
            isUpdating = false
        }
    }
}

// MARK: - DaemonControls

/// Start / stop / restart trio for the top chrome strip. Buttons enable
/// or disable based on daemon state so users can't double-start or stop
/// an idle daemon.
struct DaemonControls: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var stopCoordinator: AppStopCoordinator

    var body: some View {
        HStack(spacing: 6) {
            LaunchButton(
                backend: backend,
                daemonState: backend.daemonState
            )
            ControlButton(
                systemImage: "arrow.clockwise",
                tint: Brand.accent,
                enabled: canStop,
                help: "Restart MTPLX with the last target."
            ) {
                Task {
                    await stopCoordinator.stopAll(reason: "top_chrome_restart")
                    await backend.startDaemon()
                }
            }
        }
    }

    private var canStop: Bool {
        switch backend.daemonState {
        case .running, .warming, .degraded: true
        default: false
        }
    }
}

// MARK: - ControlButton

/// Compact icon-only button used by DaemonControls. Chrome-bezeled, tint
/// matches semantic action (Start = success, Stop = warning, Restart =
/// chrome). Disabled state fades to a muted gray.
struct ControlButton: View {
    let systemImage: String
    let tint: Color
    let enabled: Bool
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(enabled ? tint : Brand.textHighlight.opacity(0.35))
                .frame(width: 26, height: 22)
                .background(
                    Capsule(style: .continuous)
                        .stroke(
                            enabled ? tint.opacity(0.45) : Brand.separator,
                            lineWidth: 0.75
                        )
                )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .help(help)
        .accessibilityLabel(help)
    }
}

// MARK: - ThermalRuleBanner

/// Universal Thermal Rule banner. Shows when there's at least one in-flight
/// request and the fan ramp isn't verified (either thermal polling is off or
/// the fan rpm hasn't reached its expected max).
struct ThermalRuleBanner: View {
    let inFlight: [InFlightRequest]
    let thermal: ThermalSnapshot?
    let thermalPollingEnabled: Bool
    let verifiedFanMode: String?

    var body: some View {
        if !inFlight.isEmpty && !isFanRampVerified {
            bannerContent
        }
    }

    @ViewBuilder
    private var bannerContent: some View {
        let title = thermalPollingEnabled
            ? "Fans haven't ramped to verified max"
            : "Thermal polling is off"
        let message = thermalPollingEnabled
            ? "RULES.md requires verified max-fan mode for any real model probe. Headline numbers measured now are diagnostic-only."
            : "Enable thermal polling in Settings to verify fan ramp before benchmarking. Headline numbers without verified fans are diagnostic-only."

        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "thermometer.high")
                .font(.title3)
                .foregroundStyle(Brand.warning)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(Brand.warning)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(Brand.textHighlight.opacity(0.75))
            }
            Spacer()
        }
        .padding(12)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                .fill(Brand.warning.opacity(0.12))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                        .strokeBorder(Brand.warning.opacity(0.45), lineWidth: Brand.hairlineStrong)
                }
        }
    }

    private var isFanRampVerified: Bool {
        if verifiedFanMode == "max" {
            return true
        }
        guard let thermal else { return false }
        guard let max = thermal.maxRpm, max > 0 else { return false }
        let fanRpms = thermal.fans.compactMap(\.actualRpm)
        guard !fanRpms.isEmpty else { return false }
        return fanRpms.allSatisfy { Double($0) >= Double(max) * 0.90 }
    }
}

// MARK: - ConnectionIssueBanner

/// Top-of-window banner when the SSE connection is reconnecting or failed.
struct ConnectionIssueBanner: View {
    let state: MetricsConnectionState

    var body: some View {
        if let message {
            HStack(spacing: 10) {
                Image(systemName: "antenna.radiowaves.left.and.right.slash")
                    .font(.callout)
                    .foregroundStyle(Brand.danger)
                Text(message)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(Brand.textHighlight)
                Spacer()
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Brand.danger.opacity(0.14))
            .overlay(
                Rectangle()
                    .fill(Brand.danger.opacity(0.35))
                    .frame(height: 0.5),
                alignment: .bottom
            )
        }
    }

    private var message: String? {
        switch state {
        case .reconnecting(let n): "Stream reconnecting (attempt #\(n))…"
        case .failed(let m): "Stream offline: \(m)"
        default: nil
        }
    }
}
