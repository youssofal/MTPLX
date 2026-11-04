import AppKit
import SwiftUI

// MARK: - WindowVisibilityReporter
//
// Reports whether the window that hosts it can be seen, from AppKit's
// occlusion state: false while the window is minimized, hidden with the
// app, fully covered by other windows or on another Space; true as soon as
// any part of it is on screen again. The metrics stream uses it so a
// window nobody can see does not keep the engine building dashboard
// snapshots, or the app redrawing the Live tab, at full cadence while a
// request decodes (MTPLXBackendStore.setDashboardVisible).

struct WindowVisibilityReporter: NSViewRepresentable {
    let onChange: @MainActor (Bool) -> Void

    func makeNSView(context: Context) -> ReporterView {
        let view = ReporterView()
        view.onChange = onChange
        return view
    }

    func updateNSView(_ nsView: ReporterView, context: Context) {
        nsView.onChange = onChange
    }

    static func dismantleNSView(_ nsView: ReporterView, coordinator: ()) {
        nsView.tearDown()
    }

    final class ReporterView: NSView {
        var onChange: (@MainActor (Bool) -> Void)?
        private var observers: [NSObjectProtocol] = []
        private var lastReported: Bool?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            stopObserving()
            // Detached while SwiftUI moves the view: report nothing, so a
            // move does not blink the cadence. A window that closes says so
            // through willClose, and final teardown reports false: AppKit
            // may deliver the last occlusion change after the views are
            // gone, and the engine outlives the window.
            guard let window else { return }
            // A replacement window always reports its own state.
            lastReported = nil
            observers = [
                NotificationCenter.default.addObserver(
                    forName: NSWindow.didChangeOcclusionStateNotification,
                    object: window,
                    queue: .main
                ) { [weak self] _ in
                    MainActor.assumeIsolated {
                        self?.reportCurrentState()
                    }
                },
                NotificationCenter.default.addObserver(
                    forName: NSWindow.willCloseNotification,
                    object: window,
                    queue: .main
                ) { [weak self] _ in
                    MainActor.assumeIsolated {
                        self?.report(false)
                    }
                },
            ]
            reportCurrentState()
        }

        func stopObserving() {
            for observer in observers {
                NotificationCenter.default.removeObserver(observer)
            }
            observers = []
        }

        /// SwiftUI dismantled the view: nothing it hosted can be seen.
        func tearDown() {
            stopObserving()
            report(false)
        }

        private func reportCurrentState() {
            guard let window else { return }
            report(window.occlusionState.contains(.visible))
        }

        private func report(_ visible: Bool) {
            guard visible != lastReported else { return }
            lastReported = visible
            onChange?(visible)
        }
    }
}
