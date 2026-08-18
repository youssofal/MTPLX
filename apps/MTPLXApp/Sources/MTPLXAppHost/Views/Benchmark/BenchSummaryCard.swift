import SwiftUI
import MTPLXAppCore

// MARK: - BenchSummaryCard
//
// Post-run summary. Hero block (state label + score / total + accuracy),
// meta block (duration / model / state), and action cluster (Run again
// + clear). The card background routes through PanelChrome so the
// summary surface lives in the same panel system as the live card and
// the overlay shell. Confetti fires only on a 30/30 perfect run.
//
// Jet Chrome pass: the 70%+ band's state + score color flips from
// cool-blue to chromeAccent; perfect run keeps success green; failure
// modes keep their semantic colors; the action cluster is replaced by
// explicit Quick / full-run actions + MTPLXGhostButton (Clear run).

struct BenchSummaryCard: View {
    let state: BenchRunState
    let score: Int
    let resolved: Int
    let total: Int
    let accuracy: Double?
    let elapsed: TimeInterval
    let model: String
    let onQuickCheck: () -> Void
    let onRunFull: () -> Void
    let onClear: () -> Void

    @State private var confettiStart: Date = .distantPast

    var body: some View {
        ZStack {
            content
            if isPerfectRun {
                BenchConfettiView(start: confettiStart)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
        }
        .padding(22)
        .background {
            PanelChrome(cornerRadius: Brand.Radii.l, elevation: Brand.Elevation.hi)
        }
        .onAppear {
            if isPerfectRun {
                confettiStart = Date.now
            }
        }
    }

    private var content: some View {
        HStack(alignment: .top, spacing: 24) {
            heroBlock
            Divider().overlay(Brand.separator).frame(height: 120)
            metaBlock
            Spacer()
            actions
        }
    }

    private var heroBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(stateLabel)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(stateColor)
            Text("\(score) / \(total)")
                .font(.system(size: 56, weight: .heavy, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(scoreColor)
                .contentTransition(.numericText())
            Text(accuracyLabel)
                .font(.system(size: 14, weight: .heavy, design: .monospaced))
                .foregroundStyle(scoreColor.opacity(0.85))
                .contentTransition(.numericText())
        }
    }

    private var metaBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            metaRow(label: "DURATION", value: formattedElapsed)
            metaRow(label: "MODEL", value: modelShort)
            metaRow(label: "STATE", value: state.rawValue.uppercased())
        }
        .frame(width: 240, alignment: .leading)
    }

    private var actions: some View {
        VStack(alignment: .trailing, spacing: 10) {
            Button(action: onQuickCheck) {
                HStack(spacing: 7) {
                    Image(systemName: "bolt.fill")
                        .font(.system(size: 11, weight: .heavy))
                    Text("Quick Check")
                }
            }
            .buttonStyle(.mtplxPrimary)

            Button(action: onRunFull) {
                HStack(spacing: 7) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 11, weight: .heavy))
                    Text("Run AIME 2026")
                }
            }
            .buttonStyle(.mtplxGhost)

            Button(action: onClear) {
                Text("Clear run")
            }
            .buttonStyle(.mtplxGhost)
        }
    }

    private func metaRow(label: String, value: String) -> some View {
        HStack(spacing: 10) {
            Text(label.mtplxLocalized)
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeTertiary)
                .frame(width: 72, alignment: .leading)
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeBody)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    // MARK: - Derived values

    private var isPerfectRun: Bool {
        state == .done && score == total && total > 0
    }

    private var stateLabel: String {
        switch state {
        case .done where isPerfectRun: return "PERFECT RUN"
        case .done: return "RUN COMPLETE"
        case .cancelled: return "CANCELLED"
        case .error: return "ERRORED"
        default: return "—"
        }
    }

    private var stateColor: Color {
        switch state {
        case .done: return isPerfectRun ? Brand.success : Brand.accentChrome
        case .cancelled: return Brand.warning
        case .error: return Brand.danger
        default: return Brand.typeTertiary
        }
    }

    private var scoreColor: Color {
        let rate = total > 0 ? Double(score) / Double(total) : 0
        if state == .error { return Brand.danger }
        if state == .cancelled { return Brand.warning }
        switch rate {
        case 0.90...: return Brand.success
        case 0.70...: return Brand.accentChrome
        case 0.50...: return Brand.warning
        default: return Brand.danger
        }
    }

    private var accuracyLabel: String {
        if state == .cancelled {
            let missed = max(0, resolved - score)
            let remaining = max(0, total - resolved)
            if missed > 0 {
                return "\(score) correct · \(missed) missed · \(remaining) left"
            }
            return "\(score) correct · \(remaining) left"
        }
        guard let accuracy else { return "—" }
        return Self.percentFormatter.string(from: NSNumber(value: accuracy)) ?? "—"
    }

    private static let percentFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .percent
        f.minimumFractionDigits = 1
        f.maximumFractionDigits = 1
        return f
    }()

    private var formattedElapsed: String {
        let totalSeconds = Int(elapsed)
        let h = totalSeconds / 3600
        let m = (totalSeconds % 3600) / 60
        let s = totalSeconds % 60
        if h > 0 {
            return String(format: "%d:%02d:%02d", h, m, s)
        }
        return String(format: "%d:%02d", m, s)
    }

    private var modelShort: String {
        model.split(separator: "/").last.map(String.init) ?? model
    }
}
