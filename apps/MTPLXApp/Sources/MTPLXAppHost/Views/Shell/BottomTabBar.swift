import SwiftUI
import MTPLXAppCore

// MARK: - BottomTabBar
//
// Custom bottom-edge tab bar. Renders the V1 dashboard surfaces
// (Live / Activity / System / Forge / Settings) with a chrome
// underline under the selected tab and a soft cross-fade rail on
// hovered items. Items auto-derive from `AppTab.allCases` — adding or
// removing a case in AppRouter.swift updates the bar with zero edits
// here.
//
// Jet Chrome pass: the previous whole-button scaleEffect on hover
// scaled icon, label, and underline as one block — "the fucking text
// and the icon hover at the same time" feel. The treatment now is
// decoupled: the icon brightens on hover (no scale), the label stays
// completely still, and a thin chromeAccent rail cross-fades in
// underneath the icon. The selected underline lives at the parent so
// it can glide between tabs via `matchedGeometryEffect`.

struct BottomTabBar: View {
    let inFlightCount: Int
    let daemonState: DaemonState
    let performanceLock: Bool

    @EnvironmentObject private var router: AppRouter
    @EnvironmentObject private var themeStore: ThemeStore

    @Namespace private var tabHighlight

    var body: some View {
        HStack(spacing: 0) {
            ForEach(AppTab.allCases) { tab in
                TabBarItem(
                    tab: tab,
                    isSelected: router.selection == tab,
                    badge: badgeCount(for: tab),
                    motionEnabled: !themeStore.reduceMotionPreference
                        && !performanceLock,
                    highlight: tabHighlight
                ) {
                    withAnimation(tabNavigationAnimation) {
                        router.select(tab)
                        router.showDashboard()
                    }
                }
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, Brand.Spacing.s4)
        .padding(.vertical, 8)
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(height: Brand.hairline),
                    alignment: .top
                )
                .shadow(
                    color: Brand.Elevation.hi.color,
                    radius: Brand.Elevation.hi.radius,
                    x: 0,
                    y: -Brand.Elevation.hi.y
                )
        )
    }

    private func badgeCount(for tab: AppTab) -> Int? {
        switch tab {
        case .activity:
            // Activity owns the in-flight counter that used to live on
            // the old Requests tab. Cache pressure deliberately does
            // NOT get a badge — too noisy; users open the tab to look
            // at cache state when they want it.
            let n = inFlightCount
            return n > 0 ? n : nil
        case .system:
            // Surface a "!" badge when degraded / thermal alarm.
            switch daemonState.kind {
            case .degraded, .crashed: return 0
            default: break
            }
            return nil
        default:
            return nil
        }
    }

    private var tabNavigationAnimation: Animation? {
        guard !themeStore.reduceMotionPreference,
              !performanceLock
        else { return nil }
        return .spring(response: 0.36, dampingFraction: 0.86)
    }
}

// MARK: - TabBarItem

struct TabBarItem: View {
    let tab: AppTab
    let isSelected: Bool
    /// `nil` = no badge. `0` = warning glyph instead of counter. `>0` =
    /// numeric counter.
    let badge: Int?
    let motionEnabled: Bool
    let highlight: Namespace.ID
    let action: () -> Void

    @State private var hovering: Bool = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                ZStack {
                    Image(systemName: tab.systemImage)
                        .font(.system(size: 18, weight: .medium))
                        .symbolRenderingMode(.monochrome)
                        .foregroundStyle(symbolForeground)
                        .opacity(iconOpacity)

                    if let badge {
                        badgeView(count: badge)
                            .offset(x: 14, y: -10)
                    }
                }
                .frame(height: 22)

                // Hover rail. 1px chrome line under the icon that
                // cross-fades in on hover for unselected tabs only;
                // the selected tab uses the chromeAccent underline
                // below as its anchor.
                Rectangle()
                    .fill(Brand.accentChrome.opacity(0.12))
                    .frame(width: 28, height: 1)
                    .opacity((hovering && !isSelected) ? 1 : 0)

                Text(tab.title)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .tracking(1.2)
                    .foregroundStyle(labelForeground)

                // Selected underline. Only the selected tab renders
                // the filled capsule; matchedGeometryEffect glides it
                // between tabs when selection changes. Unselected
                // slots reserve the layout space with a transparent
                // placeholder so the row doesn't reflow.
                ZStack {
                    Capsule()
                        .fill(Color.clear)
                        .frame(width: 24, height: 2)
                    if isSelected {
                        Capsule()
                            .fill(Brand.chromeAccent)
                            .frame(width: 24, height: 2)
                            .matchedGeometryEffect(id: "tab_underline", in: highlight)
                    }
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(
            motionEnabled ? .spring(response: 0.32, dampingFraction: 0.78) : nil,
            value: isSelected
        )
        .animation(
            motionEnabled ? .easeOut(duration: 0.18) : nil,
            value: hovering
        )
        .help(tab.title)
        .accessibilityLabel(tab.title)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    /// Icon-only hover step. Selected = 1.0; hovered = 0.95;
    /// resting = 0.65. The label stays still on hover so the row
    /// doesn't feel like text and icon are bouncing in sync.
    private var iconOpacity: Double {
        if isSelected { return 1.0 }
        if hovering { return 0.95 }
        return 0.65
    }

    private var symbolForeground: AnyShapeStyle {
        if isSelected { return AnyShapeStyle(Brand.chromeAccent) }
        return AnyShapeStyle(Brand.typeBody)
    }

    private var labelForeground: Color {
        if isSelected { return Brand.typeHi }
        return Brand.typeBody.opacity(0.55)
    }

    @ViewBuilder
    private func badgeView(count: Int) -> some View {
        if count == 0 {
            Circle()
                .fill(Brand.danger)
                .frame(width: 8, height: 8)
                .overlay {
                    Circle().strokeBorder(Brand.bgInner, lineWidth: 1)
                }
        } else {
            Text(count > 99 ? "99+" : String(count))
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(Brand.bgOuter)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background {
                    Capsule()
                        .fill(Brand.accentChrome)
                        .overlay {
                            Capsule().strokeBorder(Brand.bgInner, lineWidth: 1)
                        }
                }
        }
    }
}
