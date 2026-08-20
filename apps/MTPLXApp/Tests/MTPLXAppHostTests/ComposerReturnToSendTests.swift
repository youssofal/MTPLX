import AppKit
import SwiftUI
import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// 2026-08-19 release blocker: Return in the composer silently did nothing.
///
/// macOS 14+ inline predictions arrive as marked text during plain typing;
/// the IME publish gate then never syncs the NSTextView string into the
/// SwiftUI binding, so the send guard reads an empty binding and swallows
/// the submit. These tests pin the two-part fix: predictions are disabled
/// on the composer, and Return syncs the authoritative view string into the
/// binding before invoking onSubmit — a stale binding can never eat a send.
final class ComposerReturnToSendTests: XCTestCase {
    @MainActor
    private final class Box {
        var text = ""
        var submitted = 0
        var textAtSubmit: [String] = []
    }

    @MainActor
    private func mountComposer(
        box: Box
    ) throws -> (host: NSHostingView<ComposerInputTextView>, textView: NSTextView) {
        let view = ComposerInputTextView(
            text: Binding(get: { box.text }, set: { box.text = $0 }),
            measuredHeight: .constant(44),
            minHeight: 44,
            maxHeight: 160,
            onSubmit: {
                box.submitted += 1
                box.textAtSubmit.append(box.text)
            },
            onFileDrop: { _ in }
        )
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 420, height: 64)
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        guard let textView = Self.firstTextView(in: host) else {
            throw XCTSkip("composer NSTextView did not mount under NSHostingView")
        }
        return (host, textView)
    }

    private static func firstTextView(in view: NSView) -> NSTextView? {
        if let textView = view as? NSTextView { return textView }
        for child in view.subviews {
            if let found = firstTextView(in: child) { return found }
        }
        return nil
    }

    @MainActor
    func testInlinePredictionsDisabledOnComposer() throws {
        let box = Box()
        let (host, textView) = try mountComposer(box: box)
        defer { _ = host }
        XCTAssertEqual(
            textView.inlinePredictionType,
            .no,
            "inline predictions deliver marked text during ASCII typing and starve the send guard"
        )
    }

    @MainActor
    func testReturnSubmitsWithAuthoritativeTextEvenWhenBindingIsStale() throws {
        let box = Box()
        let (host, textView) = try mountComposer(box: box)
        defer { _ = host }
        // Reproduce the field bug: the view holds text the binding never saw.
        textView.string = "ship it"
        XCTAssertEqual(box.text, "", "precondition: binding is stale")
        textView.doCommand(by: #selector(NSResponder.insertNewline(_:)))
        XCTAssertEqual(box.submitted, 1, "Return must submit")
        XCTAssertEqual(
            box.textAtSubmit.first,
            "ship it",
            "submit must read the committed view string, not the stale binding"
        )
    }
}
