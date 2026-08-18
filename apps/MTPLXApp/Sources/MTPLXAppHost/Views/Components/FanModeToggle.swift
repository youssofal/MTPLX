import SwiftUI
import MTPLXAppCore

// MARK: - FanModeToggle
//
// Segmented Default | Smart | Max control wired to `backend.currentFanMode` +
// `backend.setFanMode(_:)`. Hidden only when the daemon is not running.
//
// Loading spinner shows during the async POST.

struct FanModeToggle: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    @State private var pending: Bool = false

    var body: some View {
        if !shouldShow {
            EmptyView()
        } else {
            HStack(spacing: 6) {
                Image(systemName: "fan.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(active == MTPLXFanMode.max.rawValue ? Brand.coolChrome : Brand.textHighlight.opacity(0.55))
                ForEach(MTPLXFanMode.allCases, id: \.self) { mode in
                    segment(mode.title.uppercased(), value: mode.rawValue)
                }
                if pending {
                    ProgressView().controlSize(.mini)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                Capsule(style: .continuous)
                    .stroke(Brand.separator, lineWidth: 0.75)
            )
            .help("Default uses Apple's fan curve, Smart boosts during generation, Max pins verified max.")
        }
    }

    /// Feature gate: must have a running daemon because live fan-mode changes
    /// are applied through the daemon endpoint.
    private var shouldShow: Bool {
        switch backend.daemonState.kind {
        case .running, .stopping, .warming: return true
        default: return false
        }
    }

    private var active: String {
        MTPLXFanMode.normalized(backend.currentFanMode ?? backend.configuration.fanMode).rawValue
    }

    @ViewBuilder
    private func segment(_ label: String, value: String) -> some View {
        Button {
            apply(value)
        } label: {
            Text(label.mtplxLocalized)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1)
                .foregroundStyle(active == value ? Brand.bgOuter : Brand.textHighlight)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule(style: .continuous)
                        .fill(active == value ? AnyShapeStyle(Brand.chromeFill) : AnyShapeStyle(Color.clear))
                )
        }
        .buttonStyle(.plain)
        .disabled(pending)
    }

    private func apply(_ mode: String) {
        let canonical = MTPLXFanMode.normalized(mode).rawValue
        guard canonical != active, !pending else { return }
        pending = true
        Task {
            defer { Task { @MainActor in pending = false } }
            try? await backend.setFanMode(canonical)
        }
    }
}
