import AppKit
import Combine
import SwiftUI
import MTPLXAppCore

// MARK: - MenuBarController
//
// MTPLX in the menu bar (#692) as one AppKit status item. Last night's
// SwiftUI version cost the window server about 43 ms of CPU a second during
// a decode (rows in `2f9d7cce`): its label re-measured and re-laid out the
// bar on every new reading. This item never changes width while a request
// runs, writes only its title, and only when the shown text changes:
//
// - Two fixed lengths: the M alone, and the M with room for "888 tok/s".
//   The wide one is taken when a request starts and given back 5 s after
//   the last one ends, so an agent's tool loop never flaps the bar.
// - The title is "37%" while a prompt is read, "74 tok/s" while it decodes,
//   blank otherwise; the readout changes at most once a second and only
//   when that text changes, and not at all while nothing runs.
// - The M's image is set once per engine state.
// - A click opens the panel; its SwiftUI content is built on open and
//   released on close, so nothing observes the store while it is closed.
//   AppKit's button also opens on an accessibility press (VoiceOver).

@MainActor
protocol MenuBarItemSurface: AnyObject {
    func show(title: String, accessibilityLabel: String)
    func show(image: NSImage)
    func set(length: CGFloat)
}

@MainActor
final class MenuBarController: NSObject, NSPopoverDelegate {
    static let settingKey = "mtplx.app.menuBarItem"
    /// Opens the main window when none exists (SwiftUI's openWindow, handed
    /// over by the main window's root view: an AppKit popover has none).
    static var openMainWindow: (() -> Void)?

    private let readout: MenuBarReadout
    private let defaults: UserDefaults
    private let makeSurface: (() -> MenuBarItemSurface)?
    private let makePanel: () -> AnyView
    private let holdSeconds: Double
    private var statusItem: NSStatusItem?
    private var surface: MenuBarItemSurface?
    private var popover: NSPopover?
    private var readingSubscription: AnyCancellable?
    private var defaultsObserver: NSObjectProtocol?
    private var isSyncing = false
    private(set) var isInstalled = false
    private var shownState: DaemonStateKind?
    private var shownTitle = ""
    private(set) var isWide = false
    private var holdTask: Task<Void, Never>?
    let narrowLength: CGFloat
    let wideLength: CGFloat

    /// What the item has written, for the tests and the cost arms' logs.
    private(set) var titleWrites = 0
    private(set) var imageWrites = 0
    private(set) var lengthWrites = 0

    /// `makeSurface` stands in for the real status item in the tests.
    init(
        readout: MenuBarReadout,
        defaults: UserDefaults = .standard,
        holdSeconds: Double = 5,
        makeSurface: (() -> MenuBarItemSurface)? = nil,
        makePanel: @escaping () -> AnyView
    ) {
        self.readout = readout
        self.defaults = defaults
        self.holdSeconds = holdSeconds
        self.makeSurface = makeSurface
        self.makePanel = makePanel
        let font = Self.titleFont
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let widest = [tr("%@ tok/s", "888"), "99%"]
            .map { ($0 as NSString).size(withAttributes: attributes).width }
            .max() ?? 60
        let mark = MenuBarMark.image(for: .running).size.width
        narrowLength = NSStatusItem.squareLength
        wideLength = (mark + 4 + widest + 12).rounded(.up)
        super.init()
    }

    static var titleFont: NSFont {
        NSFont.monospacedDigitSystemFont(ofSize: NSFont.menuBarFont(ofSize: 0).pointSize, weight: .regular)
    }

    static func isEnabled(_ defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: settingKey) as? Bool ?? ThemeStore.menuBarItemDefault
    }

    // MARK: Install and remove with the setting

    /// Follows Settings > General's switch from now on.
    func start() {
        guard defaultsObserver == nil else { return }
        defaultsObserver = NotificationCenter.default.addObserver(
            forName: UserDefaults.didChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.syncWithSetting() }
        }
        syncWithSetting()
    }

    /// Stops following the switch and takes the item out.
    func stop() {
        if let defaultsObserver { NotificationCenter.default.removeObserver(defaultsObserver) }
        defaultsObserver = nil
        if isInstalled { remove() }
    }

    /// AppKit writes a new status item's saved visibility to the defaults
    /// while it makes the item, and that write posts the very change this
    /// follows, before `install` has the item: unguarded, each post made
    /// another item until the stack ran out (a launch crash with the switch
    /// on, 17 November). A change that arrives while the item is made or
    /// removed is the item's own, so it is dropped.
    private func syncWithSetting() {
        guard !isSyncing else { return }
        isSyncing = true
        defer { isSyncing = false }
        let wanted = Self.isEnabled(defaults)
        if wanted, !isInstalled {
            install()
        } else if !wanted, isInstalled {
            remove()
        }
    }

    private func install() {
        let made = makeSurface?() ?? makeStatusItem()
        isInstalled = true
        attach(surface: made)
    }

    private func makeStatusItem() -> MenuBarItemSurface {
        let item = NSStatusBar.system.statusItem(withLength: narrowLength)
        item.autosaveName = "mtplx.menuBar"
        // AppKit makes every item under the name "Item-0" first and applies
        // that name's saved visibility before this name is set. A Mac that
        // ran a build whose menu bar item was off has "Item-0" saved hidden
        // (his installed app's preferences do, 17 November), and the item
        // stayed hidden under its own name. The switch is the user's word.
        item.isVisible = true
        if let button = item.button {
            button.imagePosition = .imageLeading
            button.font = Self.titleFont
            button.target = self
            button.action = #selector(togglePanel(_:))
        }
        statusItem = item
        return StatusItemSurface(item: item)
    }

    private func remove() {
        popover?.performClose(nil)
        readingSubscription = nil
        holdTask?.cancel()
        holdTask = nil
        if let statusItem { NSStatusBar.system.removeStatusItem(statusItem) }
        statusItem = nil
        isInstalled = false
        surface = nil
        shownState = nil
        shownTitle = ""
        isWide = false
    }

    /// Wires a surface to the readout (the real status item, or a test's).
    func attach(surface: MenuBarItemSurface) {
        self.surface = surface
        shownState = nil
        shownTitle = ""
        isWide = false
        apply(readout.reading)
        readingSubscription = readout.$reading
            .dropFirst()
            .sink { [weak self] reading in self?.apply(reading) }
    }

    // MARK: Drawing

    func apply(_ reading: MenuBarReadout.Reading) {
        guard let surface else { return }
        if shownState != reading.state {
            shownState = reading.state
            surface.show(image: MenuBarMark.image(for: reading.state))
            imageWrites += 1
        }
        if reading.requestActive {
            holdTask?.cancel()
            holdTask = nil
            if !isWide {
                isWide = true
                surface.set(length: wideLength)
                lengthWrites += 1
            }
        } else if isWide, holdTask == nil {
            let seconds = holdSeconds
            holdTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                guard !Task.isCancelled else { return }
                self?.narrowAfterHold()
            }
        }
        let title = Self.title(for: reading)
        if title != shownTitle {
            shownTitle = title
            surface.show(title: title, accessibilityLabel: Self.accessibilityLabel(for: reading, title: title))
            titleWrites += 1
        }
    }

    private func narrowAfterHold() {
        holdTask = nil
        guard isWide, let surface else { return }
        isWide = false
        surface.set(length: narrowLength)
        lengthWrites += 1
    }

    static func title(for reading: MenuBarReadout.Reading) -> String {
        if let tokS = reading.decodeTokS { return tr("%@ tok/s", String(tokS)) }
        if let percent = reading.prefillPercent { return "\(percent)%" }
        return ""
    }

    static func accessibilityLabel(for reading: MenuBarReadout.Reading, title: String) -> String {
        tr("MTPLX — %@", title.isEmpty ? reading.state.label : title)
    }

    // MARK: The panel

    @objc private func togglePanel(_ sender: Any?) {
        if let popover, popover.isShown {
            popover.performClose(sender)
            return
        }
        guard let button = statusItem?.button else { return }
        let popover = self.popover ?? {
            let made = NSPopover()
            made.behavior = .transient
            made.animates = false
            made.delegate = self
            self.popover = made
            return made
        }()
        popover.contentViewController = NSHostingController(rootView: makePanel())
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        popover.contentViewController?.view.window?.makeKey()
    }

    /// Built on open, released on close: nothing observes the store while closed.
    func popoverDidClose(_ notification: Notification) {
        popover?.contentViewController = nil
    }

    var panelIsBuilt: Bool { popover?.contentViewController != nil }

    /// MTPLX is one window: bring it forward when it is open (minimized,
    /// hidden with the app or behind other apps), and open it only when it
    /// was closed. Titled, not a panel, not a sheet; of those, the one the
    /// "main" group opened before another titled window such as the update
    /// dialog. Not filtered on `canBecomeMain`: AppKit says false for a
    /// minimized window (the 16 Nov build opened a second window).
    static func showMainWindow() {
        NSApp.unhide(nil)
        NSApp.activate(ignoringOtherApps: true)
        let candidates = NSApp.windows.filter { window in
            window.styleMask.contains(.titled) && !(window is NSPanel) && window.sheetParent == nil
                && (window.isVisible || window.isMiniaturized)
        }
        let open = candidates.first { $0.identifier?.rawValue.hasPrefix("main") == true }
            ?? candidates.first
        guard let window = open else {
            openMainWindow?()
            return
        }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
    }
}

// MARK: - The real status item

@MainActor
private final class StatusItemSurface: MenuBarItemSurface {
    private let item: NSStatusItem

    init(item: NSStatusItem) {
        self.item = item
    }

    func show(title: String, accessibilityLabel: String) {
        item.button?.title = title
        item.button?.setAccessibilityLabel(accessibilityLabel)
    }

    func show(image: NSImage) {
        item.button?.image = image
    }

    func set(length: CGFloat) {
        item.length = length
    }
}

// MARK: - OpenWindowBridge

/// Hands SwiftUI's openWindow to the menu bar controller from the main
/// window's root view, for "Open MTPLX" when no window is open.
struct MenuBarOpenWindowBridge: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onAppear {
                MenuBarController.openMainWindow = { openWindow(id: "main") }
            }
    }
}
