import SwiftUI
import MTPLXAppCore

// MARK: - InferenceParamsButton
//
// 32x32 chrome puck in the top strip that opens the inference-params
// overlay. Shares the `PremiumPuckStyle` chassis with `LaunchButton`
// so the two right-side controls read as a matched pair of polished
// pucks (dual-ring chrome, radial inner fill, hover halo, press
// collapse). The puck is a ButtonStyle — press state rides on
// `ButtonStyle.Configuration.isPressed`, not a `simultaneousGesture`,
// so the click is delivered to the Button reliably.

struct InferenceParamsButton: View {
    let performanceLock: Bool

    @EnvironmentObject private var router: AppRouter

    var body: some View {
        Button {
            router.inferenceParamsPresented.toggle()
        } label: {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 13, weight: .semibold))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(Brand.typeHi)
        }
        .buttonStyle(PremiumPuckStyle())
        .help(performanceLock ? "Settings · Performance mode on" : "Settings")
        .accessibilityLabel("Settings")
    }
}
