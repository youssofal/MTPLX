import SwiftUI
import MTPLXAppCore

// MARK: - FanModeToggle
//
// Single authoritative segmented Default | Smart | Max control. Works
// consistently whether the daemon is stopped (persists immediately for the
// next launch) or running (applies live through `/v1/mtplx/thermal/fan_mode`
// and persists). Reserves a fixed-width spinner slot so applying a live
// change never shifts the picker or triggers a parent reflow.

struct FanModeToggle: View {
    let activeMode: MTPLXFanMode
    let pending: Bool
    let onSelect: (MTPLXFanMode) -> Void

    var body: some View {
        HStack(spacing: 8) {
            // Fixed-size slot so toggling `pending` never shifts the
            // segmented control horizontally.
            ZStack {
                if pending {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.72)
                }
            }
            .frame(width: 14, height: 14)

            Picker(tr("Fan Mode"), selection: Binding(
                get: { activeMode },
                set: { next in
                    guard next != activeMode, !pending else { return }
                    onSelect(next)
                }
            )) {
                ForEach(MTPLXFanMode.allCases, id: \.self) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            .disabled(pending)
        }
        .help(tr("Default uses Apple's fan curve, Smart boosts during generation, Max pins verified max."))
    }
}
