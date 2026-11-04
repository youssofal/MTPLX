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
        nsView.stopObserving()
    }

    final class ReporterView: NSView {
        var onChange: (@MainActor (Bool) -> Void)?
        private var observer: NSObjectProtocol?
        private var lastReported: Bool?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            stopObserving()
            // Detached (the window closed, or SwiftUI is moving the view):
            // report nothing. A closing window turns occluded before its
            // views leave it, and a replacement reports on arrival.
            guard let window else { return }
            observer = NotificationCenter.default.addObserver(
                forName: NSWindow.didChangeOcclusionStateNotification,
                object: window,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.reportCurrentState()
                }
            }
            reportCurrentState()
        }

        func stopObserving() {
            if let observer {
                NotificationCenter.default.removeObserver(observer)
            }
            observer = nil
        }

        private func reportCurrentState() {
            guard let window else { return }
            let visible = window.occlusionState.contains(.visible)
            guard visible != lastReported else { return }
            lastReported = visible
            onChange?(visible)
        }
    }
}
