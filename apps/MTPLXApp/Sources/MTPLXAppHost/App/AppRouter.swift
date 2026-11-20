import SwiftUI
import Combine
import MTPLXAppCore

// MARK: - AppTab
//
// MTPLX V1 dashboard surfaces. The bottom-bar order mirrors how a
// customer thinks about the engine: what's happening NOW (Live), what
// just happened across requests and cache (Activity — merged from the
// V0 Cache + Requests pair so the dashboard stops reading like a
// debug console), what hardware/thermals look like (System), what to
// BUILD (Forge — the MTP-model creation surface), and how to
// configure everything (Settings).
//
// Tab dispatch lives in `DashboardSurface` (ContentView.swift). The
// bottom bar and Cmd-1..N auto-derive from `AppTab.allCases` — adding
// or removing a case Just Works for those two surfaces.
//
// Backward compatibility: the old `.cache` and `.requests` raw values
// can still appear in a returning user's UserDefaults
// ("mtplx.app.selectedTab"); the router's init decodes unknown values
// back to `.live`, so removing the cases is safe.
//
// Settings is a destination, not a dashboard: it opens from the gear in
// the top bar (and Cmd-,), so the bottom bar shows `barTabs` only. The
// case stays in `allCases` so a saved "settings" selection still
// restores and Cmd-5 keeps opening it.
enum AppTab: String, Hashable, CaseIterable, Identifiable, Sendable {
    case live
    case activity
    case system
    case forge
    case settings

    /// The dashboards the bottom tab bar shows.
    static let barTabs: [AppTab] = [.live, .activity, .system, .forge]

    var id: String { rawValue }

    var title: String {
        switch self {
        case .live: return tr("Live")
        case .activity: return tr("Activity")
        case .system: return tr("System")
        case .forge: return tr("Forge")
        case .settings: return tr("Settings")
        }
    }

    /// SF Symbol used in the bottom tab bar + Cmd-1..N menu items.
    var systemImage: String {
        switch self {
        case .live: return "gauge.with.dots.needle.bottom.50percent"
        case .activity: return "waveform.path.ecg"
        case .system: return "cpu"
        case .forge: return "hammer.fill"
        case .settings: return "gearshape"
        }
    }
}

// MARK: - AppPrimaryMode
//
// Two top-level surfaces the window can show: the live monitoring
// dashboard (default, existing behaviour) and the in-app chat surface
// (new). Surfaces are mutually exclusive at the root — the chrome
// toggle and the daemon-ready handoff for `.chat` flip this. The
// existing `selection: AppTab` still controls which dashboard tab is
// visible when `primaryMode == .dashboard`.
enum AppPrimaryMode: String, Hashable, Sendable {
    case dashboard
    case chat
    case hermes
}

enum AppExpandableSurface: String, Hashable, Sendable {
    case chat
    case hermes
    case benchmark

    var title: String {
        switch self {
        case .chat: return "chat"
        case .hermes: return tr("Hermes")
        case .benchmark: return tr("AIME")
        }
    }
}

enum HermesLaunchIntent: String, Hashable, Sendable {
    case browse
    case resumeLast
}

// MARK: - OnboardingPhase
//
// Above both `primaryMode` and `selection`: gates whether the user
// sees the existing app shell at all. `.onboarding` means the first-
// launch experience replaces the entire window (no chrome, no tab
// bar). `.completed` means the existing shell renders as today.
// Derived on startup from `MTPLXAppConfiguration.onboardingCompletedAt`,
// never persisted to `UserDefaults` (the source of truth is the
// settings file).
enum OnboardingPhase: String, Hashable, Sendable {
    case onboarding
    case completed
}

// MARK: - AppRouter

/// Shared routing/selection state. Lifted out of `ContentView` so menu
/// commands (Cmd-1..5, Cmd-]/[) and the MenuBarExtra can mutate the
/// current tab from anywhere.
///
/// The selected tab persists to `UserDefaults` so quit/relaunch restores
/// the last view. Sheet presentation flags are runtime-only.
@MainActor
final class AppRouter: ObservableObject {
    private static let defaultsKey = "mtplx.app.selectedTab"
    private static let settingsSectionKey = "mtplx.app.settingsSection"

    @Published var selection: AppTab {
        didSet {
            UserDefaults.standard.set(selection.rawValue, forKey: Self.defaultsKey)
            if selection != .settings {
                lastDashboardTab = selection
            }
        }
    }
    /// The dashboard to return to when the Settings gear is pressed again.
    private(set) var lastDashboardTab: AppTab = .live
    /// The Settings page's open section. Lives here, like `selection`,
    /// because the dashboard rebuilds the Settings view on every visit.
    @Published var settingsSection: SettingsSection {
        didSet {
            UserDefaults.standard.set(settingsSection.rawValue, forKey: Self.settingsSectionKey)
        }
    }
    /// Which top-level surface the window shows. Session-local — NOT
    /// persisted to UserDefaults. Dashboard is always the home
    /// destination on launch; chat is something the user opens
    /// deliberately via the Play button or the Monitor/Chat toggle.
    /// Persisting it caused a UX bug where the welcome ON button or
    /// onboarding-completion would land the user in chat on a fresh
    /// launch with no way to opt out.
    @Published var primaryMode: AppPrimaryMode = .dashboard
    /// Whether the chat sidebar is collapsed. Lives on the router so
    /// menu commands (Cmd+/) can toggle it without piercing into the
    /// chat surface's local state.
    @Published var chatSidebarCollapsed: Bool = false
    @Published var hermesLaunchIntent: HermesLaunchIntent = .browse
    @Published var logsSheetPresented: Bool = false
    @Published var aboutSheetPresented: Bool = false
    /// Whether the Play-button LaunchOverlay is showing. Lives on the
    /// router so the button (in the chrome strip) and the overlay (in
    /// ContentView's root ZStack) can share state without inserting the
    /// overlay into the toolbar's layout flow.
    @Published var launchPickerPresented: Bool = false
    /// Whether the Inference-Params overlay is showing (the slider/
    /// toggle popover attached to the top-strip slider-3 button).
    @Published var inferenceParamsPresented: Bool = false
    /// Whether the model picker attached to the top-left model label is
    /// showing.
    @Published var modelPickerPresented: Bool = false
    /// Whether the AIME 2026 benchmark overlay is showing. Lives on the
    /// router so the launcher row (in `LaunchOverlay.handlePick`) can
    /// flip it on, the overlay's close button (and ESC) can flip it off,
    /// and the orchestrator state survives any number of overlay
    /// open/close cycles (the orchestrator lives at MTPLXApp root).
    @Published var benchmarkOverlayPresented: Bool = false
    /// The foreground work surface the bottom handle should reopen after
    /// the user collapses it back to the dashboard.
    @Published var expandableSurface: AppExpandableSurface = .chat
    /// Whether the user is in the first-launch onboarding experience or
    /// the regular app shell. Defaults to `.completed` so a returning
    /// user with a valid settings file never sees a transient onboarding
    /// flash. `MTPLXApp.swift`'s startup `.task` flips this to
    /// `.onboarding` synchronously after reading `onboardingCompletedAt`.
    @Published var onboardingPhase: OnboardingPhase = .completed
    /// Whether the one-time language prompt sheet is showing. Raised by
    /// the startup `.task` when `shouldOfferLanguagePrompt` says this
    /// install finished onboarding before the Language step existed;
    /// ContentView stamps the configuration on dismiss so it never
    /// comes back.
    @Published var languagePromptPresented: Bool = false

    init() {
        let raw = UserDefaults.standard.string(forKey: Self.defaultsKey) ?? AppTab.live.rawValue
        let restored = AppTab(rawValue: raw) ?? .live
        self.selection = restored
        if restored != .settings { lastDashboardTab = restored }
        let section = UserDefaults.standard.string(forKey: Self.settingsSectionKey) ?? ""
        self.settingsSection = SettingsSection(rawValue: section) ?? .general
    }

    /// Open Settings on the dashboard surface, at `section` when given.
    func openSettings(section: SettingsSection? = nil) {
        if let section { settingsSection = section }
        selection = .settings
        primaryMode = .dashboard
    }

    /// The gear in the top bar: opens Settings, or, when Settings is
    /// already showing, goes back to the dashboard the user came from.
    func toggleSettings() {
        if selection == .settings, primaryMode == .dashboard {
            selection = lastDashboardTab
        } else {
            openSettings()
        }
    }

    /// Flip into chat mode (called after `.chat` daemon reaches running,
    /// or by the chrome Monitor/Chat toggle).
    func showChat() {
        expandableSurface = .chat
        primaryMode = .chat
    }

    func showHermesBrowse() {
        expandableSurface = .hermes
        hermesLaunchIntent = .browse
        primaryMode = .hermes
    }

    func showHermesResumeLast() {
        expandableSurface = .hermes
        hermesLaunchIntent = .resumeLast
        primaryMode = .hermes
    }

    /// Flip back to the live dashboard.
    func showDashboard() {
        primaryMode = .dashboard
    }

    func select(_ tab: AppTab) {
        selection = tab
    }

    func nextTab() { stepTab(by: 1) }

    func previousTab() { stepTab(by: -1) }

    /// Cycles the bottom bar's dashboards. Settings is not one of them: from
    /// Settings the step starts at the dashboard it was opened from.
    private func stepTab(by offset: Int) {
        let tabs = AppTab.barTabs
        let current = tabs.contains(selection) ? selection : lastDashboardTab
        guard let idx = tabs.firstIndex(of: current) else { return }
        selection = tabs[(idx + offset + tabs.count) % tabs.count]
    }

    func presentLogs() {
        logsSheetPresented = true
    }

    func presentAbout() {
        aboutSheetPresented = true
    }

    /// Open the AIME 2026 benchmark overlay. Idempotent.
    func openBenchmark() {
        expandableSurface = .benchmark
        benchmarkOverlayPresented = true
    }

    /// Close the benchmark overlay. Does NOT cancel an active run —
    /// the orchestrator keeps streaming behind the dismissed overlay.
    func closeBenchmark() {
        expandableSurface = .benchmark
        benchmarkOverlayPresented = false
    }

    func reopenExpandableSurface() {
        switch expandableSurface {
        case .chat:
            showChat()
        case .hermes:
            showHermesBrowse()
        case .benchmark:
            openBenchmark()
        }
    }
}
