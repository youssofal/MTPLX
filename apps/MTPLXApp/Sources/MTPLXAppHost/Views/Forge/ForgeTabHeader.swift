import SwiftUI
import MTPLXAppCore

// MARK: - ForgeTabHeader
//
// Top strip of the Forge tab. Two pieces:
//
//   • A bespoke segmented pill on the left switching between the
//     three sub-modes (Create / Discover / My Models). Mirrors
//     PrimaryModeToggle (TopChromeStrip.swift:129-201) in shape, fill,
//     and the matchedGeometryEffect highlight-slide so the muscle
//     memory from the Monitor|Chat toggle transfers.
//   • A "+ New" CTA on the right that snaps the user back to the
//     Create wizard from any sub-mode. If a build is already
//     in-flight, the button labels itself "Continue build" to make
//     the destination honest.

struct ForgeTabHeader: View {
    @EnvironmentObject private var orchestrator: ForgeOrchestrator
    @Namespace private var highlight

    var body: some View {
        ZStack {
            segmentedControl
            HStack(alignment: .center, spacing: 16) {
                title
                Spacer(minLength: 12)
                newCTA
                    .frame(width: 132, alignment: .trailing)
            }
        }
        .padding(.horizontal, 22)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    // MARK: - Title

    private var title: some View {
        HStack(spacing: 8) {
            Image(systemName: "hammer.fill")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Brand.typeBody)
            Text("Forge")
                .font(.system(size: 16, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
        }
    }

    // MARK: - Segmented sub-mode pill

    private var segmentedControl: some View {
        HStack(spacing: 2) {
            segment(.create, label: "Create", icon: "wand.and.sparkles")
            segment(.discover, label: "Discover", icon: "globe")
            segment(.mine, label: "My Models", icon: "tray.full")
        }
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(Color.white.opacity(0.03))
                .overlay(
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    @ViewBuilder
    private func segment(_ mode: ForgeSubMode, label: String, icon: String) -> some View {
        let isActive = orchestrator.state.subMode == mode
        Button {
            guard orchestrator.state.subMode != mode else { return }
            Haptics.tick(.alignment)
            withAnimation(.smooth(duration: 0.24)) {
                orchestrator.selectSubMode(mode)
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: icon)
                    .font(.system(size: 10.5, weight: .semibold))
                Text(label.mtplxLocalized)
                    .font(.system(size: 11.5, weight: .semibold))
            }
            .foregroundStyle(isActive ? Brand.typeHi : Brand.typeSecondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                ZStack {
                    if isActive {
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .fill(Brand.cardSurface)
                            .overlay(
                                RoundedRectangle(cornerRadius: 7, style: .continuous)
                                    .stroke(Brand.separatorStrong, lineWidth: 0.5)
                            )
                            .matchedGeometryEffect(id: "forge_sub_mode_highlight", in: highlight)
                    }
                }
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - + New CTA

    @ViewBuilder
    private var newCTA: some View {
        let isMidBuild = orchestrator.isBuilding
        Button {
            if !isMidBuild {
                orchestrator.resetWizard()
            }
            withAnimation(isMidBuild ? nil : .smooth(duration: 0.24)) {
                orchestrator.selectSubMode(.create)
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: isMidBuild ? "arrow.forward" : "plus")
                    .font(.system(size: 10.5, weight: .heavy))
                Text((isMidBuild ? "Continue build" : "New build").mtplxLocalized)
                    .font(.system(size: 11.5, weight: .semibold))
                    .lineLimit(1)
            }
            .foregroundStyle(Brand.bgOuter)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(Brand.typeBody)
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help((isMidBuild
              ? "A build is in progress — jump back to it"
              : "Start a new build").mtplxLocalized)
    }
}
