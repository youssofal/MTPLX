import SwiftUI
import MTPLXAppCore

// MARK: - BenchHeader
//
// Top row of the AIME overlay: close puck, state-tinted hero chip,
// title stack, live stats cluster, and the contextual CTA cluster
// (Run / Pause + Cancel / Resume + Cancel). State-driven tint flips
// the chip + accent: idle uses the polished chromeAccent so the
// overlay reads as Jet Chrome by default; done = success green,
// error = danger red, cancelled = warning amber stay semantic.

struct BenchHeader: View {
    let state: BenchRunState
    let elapsed: TimeInterval
    let resolved: Int
    let total: Int
    let score: Int
    let accuracy: Double?
    let pausePending: Bool
    let skipPending: Bool
    let startTitle: String
    let startIcon: String
    let startEnabled: Bool
    let performanceLock: Bool
    /// Rendered content width of the panel, threaded down so the header can
    /// reflow gracefully instead of letting its CTA cluster get crushed and
    /// wrap. The controls (CTAs / settings / close) are rigid; the branding
    /// and stats yield first, then the title truncates.
    let availableWidth: CGFloat
    let onClose: () -> Void
    let onStart: () -> Void
    let onQuickStart: () -> Void
    let onPause: () -> Void
    let onSkip: () -> Void
    let onResume: () -> Void
    let onCancel: () -> Void

    /// Wordmark + hero chip only when there's comfortable room — they're
    /// pure identity and the first thing to drop when space is tight.
    private var showsBranding: Bool { availableWidth >= 820 }
    /// The live stats cluster yields next, before the title is touched.
    private var showsStats: Bool { availableWidth >= 640 }
    /// The title subtitle is the smallest thing to shed.
    private var showsSubtitle: Bool { availableWidth >= 560 }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            if showsBranding {
                WordmarkView(height: 20)
                    .accessibilityHidden(true)
                heroChip
            }
            titleStack
            Spacer(minLength: 12)
            if showsStats {
                statsCluster
                    .fixedSize()
                Spacer(minLength: 12)
            }
            ctaCluster
            InferenceParamsButton(performanceLock: performanceLock)
            closeButton
        }
    }

    // MARK: - Pieces

    private var closeButton: some View {
        Button(action: onClose) {
            Image(systemName: "xmark")
                .font(.system(size: 12, weight: .heavy))
                .foregroundStyle(Brand.typeBody)
                .frame(width: 32, height: 32)
                .background {
                    Circle()
                        .fill(Brand.cardSurface)
                        .overlay {
                            Circle().strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                        }
                }
        }
        .buttonStyle(.plain)
        .help("Close (Esc). The run keeps going — press Cmd+B to bring it back.")
        .accessibilityLabel("Close benchmark")
        .accessibilityHint("Hides this view. The run keeps going. Press Cmd+B to reopen.")
    }

    /// 44pt hero chip with state-tinted glyph. Default tint is the
    /// polished chrome accent so the overlay's hero reads as Jet
    /// Chrome at rest; done / error / cancelled use the semantic
    /// colors so the eye picks up state changes at a glance.
    private var heroChip: some View {
        let tint: Color = {
            switch state {
            case .done: return Brand.success
            case .error: return Brand.danger
            case .cancelled: return Brand.warning
            default: return Brand.accentChrome
            }
        }()
        return ZStack {
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .fill(tint.opacity(0.10))
                .overlay {
                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                        .strokeBorder(tint.opacity(0.25), lineWidth: Brand.hairline)
                }
            Image(systemName: "function")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(tint)
                .symbolRenderingMode(.hierarchical)
        }
        .frame(width: 44, height: 44)
        .accessibilityHidden(true)
    }

    private var titleStack: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("AIME 2026")
                .font(.system(size: 17, weight: .heavy, design: .rounded))
                .foregroundStyle(Brand.typeBody)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            if showsSubtitle {
                Text(subtitle)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var statsCluster: some View {
        HStack(alignment: .center, spacing: 18) {
            BenchHeaderStat(
                label: "ELAPSED",
                value: formattedElapsed
            )
            BenchHeaderStat(
                label: "RESOLVED",
                value: total > 0 ? "\(resolved) / \(total)" : "—"
            )
            BenchHeaderStat(
                label: trailingStatLabel,
                value: accuracyLabel
            )
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "Elapsed \(formattedElapsed), \(resolved) of \(total) resolved, \(trailingStatLabel.lowercased()) \(accuracyLabel)"
        )
    }

    private var ctaCluster: some View {
        HStack(spacing: 10) {
            switch state {
            case .idle, .done, .cancelled, .error:
                BenchPrimaryCTA(
                    title: "Quick Check",
                    icon: "bolt.fill",
                    style: .secondary,
                    isDanger: false,
                    isEnabled: startEnabled,
                    accessibilityHint: "Runs the first 3 AIME 2026 problems as a bounded health check.",
                    action: onQuickStart
                )
                BenchPrimaryCTA(
                    title: startTitle,
                    icon: startIcon,
                    isDanger: false,
                    isEnabled: startEnabled,
                    accessibilityHint: "Load the model if needed, then start a new 30-problem AIME 2026 run.",
                    action: onStart
                )
            case .running:
                BenchPrimaryCTA(
                    title: pausePending ? "Pausing…" : "Pause",
                    icon: "pause.fill",
                    isDanger: false,
                    isEnabled: !pausePending && !skipPending,
                    accessibilityHint: "Stop the current decode and pause this problem.",
                    action: onPause
                )
                BenchPrimaryCTA(
                    title: skipPending ? "Skipping…" : "Skip",
                    icon: "forward.end.fill",
                    isDanger: false,
                    isEnabled: !skipPending,
                    accessibilityHint: "Mark this problem unanswered and continue with the next one.",
                    action: onSkip
                )
                BenchPrimaryCTA(
                    title: "Cancel",
                    icon: "xmark",
                    isDanger: true,
                    isEnabled: true,
                    accessibilityHint: "Stop now. Saves the answers you have so far.",
                    action: onCancel
                )
            case .paused:
                BenchPrimaryCTA(
                    title: "Resume",
                    icon: "play.fill",
                    isDanger: false,
                    isEnabled: true,
                    accessibilityHint: "Retry the paused problem from a fresh prompt.",
                    action: onResume
                )
                BenchPrimaryCTA(
                    title: "Cancel",
                    icon: "xmark",
                    isDanger: true,
                    isEnabled: true,
                    accessibilityHint: "Stop now. Saves the answers you have so far.",
                    action: onCancel
                )
            }
        }
    }

    // MARK: - Formatting

    private var formattedElapsed: String {
        let totalSeconds = Int(elapsed)
        if totalSeconds >= 3600 {
            let h = totalSeconds / 3600
            let m = (totalSeconds % 3600) / 60
            let s = totalSeconds % 60
            return String(format: "%d:%02d:%02d", h, m, s)
        }
        let m = totalSeconds / 60
        let s = totalSeconds % 60
        return String(format: "%d:%02d", m, s)
    }

    private var accuracyLabel: String {
        if state == .cancelled {
            return "\(score) correct"
        }
        guard let accuracy else { return "—" }
        return percentFormatter.string(from: NSNumber(value: accuracy)) ?? "—"
    }

    private var trailingStatLabel: String {
        state == .cancelled ? "SCORE" : "ACCURACY"
    }

    private var subtitle: String {
        let runLabel = total > 0 && total < 30
            ? "\(total)-problem Quick Check"
            : "30 competition problems"
        let stateLabel: String = {
            switch state {
            case .idle:
                return "ready"
            case .running, .paused:
                return "live"
            case .done:
                return "complete"
            case .cancelled:
                return "cancelled"
            case .error:
                return "needs review"
            }
        }()
        return "\(runLabel), \(stateLabel)"
    }

    private var percentFormatter: NumberFormatter {
        let f = NumberFormatter()
        f.numberStyle = .percent
        f.minimumFractionDigits = 0
        f.maximumFractionDigits = 0
        return f
    }
}
