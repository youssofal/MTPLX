import AppKit
import SwiftUI
import MTPLXAppCore

// MARK: - ComposerInputTextView
//
// AppKit NSTextView bridge for the chat composer. Provides:
//   - Auto-growing height between `minHeight` and `maxHeight`
//   - Enter → submit, Shift+Enter → newline
//   - Drag-drop of file URLs (PDF/docx/txt/md) onto the text field
//   - Brand-themed colours and SF Pro typography (re-themed from
//     Aphanes V2's DMSans variant)
//
// Mirrors Aphanes V2's `ComposerInputTextView` + `ComposerNSTextView`
// (AppViews.swift ~8410-8590). The two-style variant (`composer` vs
// `userPromptEditor`) is collapsed here to a single composer style;
// MTPLX chat does not edit historical user prompts.

struct ComposerInputTextView: NSViewRepresentable {
    @Binding var text: String
    @Binding var measuredHeight: CGFloat
    let minHeight: CGFloat
    let maxHeight: CGFloat
    let onSubmit: () -> Void
    let onFileDrop: ([URL]) -> Void
    var shouldFocus: Bool = false

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true

        let textView = ComposerNSTextView()
        textView.delegate = context.coordinator
        textView.isEditable = true
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        // macOS 14+ inline predictions arrive as marked text during plain
        // ASCII typing; the IME publish gate below then never syncs the
        // binding, canSend stays false, and Return silently does nothing.
        // A submit-on-Return composer opts out. Real IME composition (CJK,
        // dead-key accents) is unaffected — its marked text comes from the
        // input method, not this trait.
        textView.inlinePredictionType = .no
        textView.font = .systemFont(ofSize: 14)
        textView.textColor = NSColor(Brand.typeHi)
        textView.insertionPointColor = NSColor(Brand.typeHi)
        textView.textContainerInset = NSSize(width: 0, height: 10)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.heightTracksTextView = false
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.minSize = NSSize(width: 0, height: minHeight)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.frame = NSRect(
            origin: .zero,
            size: NSSize(width: max(scrollView.contentSize.width, 1), height: minHeight)
        )
        textView.autoresizingMask = [.width]
        textView.onSubmit = onSubmit
        textView.onSyncText = { [coordinator = context.coordinator] committed in
            coordinator.parent.text = committed
        }
        textView.onFileDrop = onFileDrop
        textView.string = text
        textView.appearance = NSAppearance(named: .darkAqua)
        textView.selectedTextAttributes = [
            .backgroundColor: NSColor(Brand.accentChrome.opacity(0.35)),
            .foregroundColor: NSColor(Brand.typeHi),
        ]
        textView.registerForDraggedTypes([.fileURL])

        scrollView.documentView = textView
        scrollView.contentView.postsBoundsChangedNotifications = true
        syncDocumentFrame(for: textView)

        Task { @MainActor in
            if shouldFocus {
                textView.window?.makeFirstResponder(textView)
            }
            recalculateHeight(for: textView)
        }
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? ComposerNSTextView else { return }
        context.coordinator.parent = self
        textView.onSubmit = onSubmit
        textView.onSyncText = { [coordinator = context.coordinator] committed in
            coordinator.parent.text = committed
        }
        textView.onFileDrop = onFileDrop
        syncDocumentFrame(for: textView)
        // Never overwrite the text view while an IME composition is in flight:
        // doing so tears down the marked-text session and drops the in-progress
        // input (CJK, dead-key accents, etc.). Defer until the composition commits.
        if ComposerTextSync.shouldApplyBindingToTextView(
            hasMarkedText: textView.hasMarkedText(),
            textViewString: textView.string,
            binding: text
        ) {
            context.coordinator.isApplyingProgrammaticText = true
            textView.string = text
            let cursor = text.utf16.count
            textView.setSelectedRange(NSRange(location: cursor, length: 0))
            textView.scrollRangeToVisible(NSRange(location: cursor, length: 0))
            if text.isEmpty {
                textView.undoManager?.removeAllActions()
            }
            context.coordinator.isApplyingProgrammaticText = false
        }
        if shouldFocus, textView.window?.firstResponder !== textView {
            Task { @MainActor in
                guard textView.window?.firstResponder !== textView else { return }
                textView.window?.makeFirstResponder(textView)
            }
        }
        recalculateHeight(for: textView)
    }

    fileprivate func syncDocumentFrame(for textView: NSTextView) {
        let visibleWidth = max(
            textView.enclosingScrollView?.contentSize.width ?? textView.bounds.width,
            1
        )
        textView.textContainer?.containerSize = NSSize(
            width: visibleWidth,
            height: CGFloat.greatestFiniteMagnitude
        )
        let targetHeight = max(measuredHeight, minHeight)
        if abs(textView.frame.width - visibleWidth) > 0.5
            || abs(textView.frame.height - targetHeight) > 0.5 {
            textView.frame = NSRect(
                origin: .zero,
                size: NSSize(width: visibleWidth, height: targetHeight)
            )
        }
    }

    fileprivate func recalculateHeight(for textView: NSTextView) {
        syncDocumentFrame(for: textView)
        guard let layoutManager = textView.layoutManager,
            let textContainer = textView.textContainer
        else { return }
        layoutManager.ensureLayout(for: textContainer)
        let usedHeight = layoutManager.usedRect(for: textContainer).height
        let contentHeight = usedHeight + textView.textContainerInset.height * 2
        let clampedHeight = min(max(contentHeight, minHeight), maxHeight)
        let visibleWidth = max(
            textView.enclosingScrollView?.contentSize.width ?? textView.bounds.width,
            1
        )
        if abs(textView.frame.width - visibleWidth) > 0.5
            || abs(textView.frame.height - clampedHeight) > 0.5 {
            textView.frame = NSRect(
                origin: .zero,
                size: NSSize(width: visibleWidth, height: clampedHeight)
            )
        }
        if abs(measuredHeight - clampedHeight) > 0.5 {
            Task { @MainActor in
                measuredHeight = clampedHeight
            }
        }
        textView.enclosingScrollView?.hasVerticalScroller = contentHeight > maxHeight
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ComposerInputTextView
        var isApplyingProgrammaticText = false

        init(parent: ComposerInputTextView) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard !isApplyingProgrammaticText else { return }
            guard let textView = notification.object as? NSTextView else { return }
            // While marked text is active the string is provisional IME preedit.
            // Keep the box auto-growing, but don't publish the preedit up into the
            // binding — that round-trip races (and kills) the live composition.
            guard ComposerTextSync.shouldPublishEdit(hasMarkedText: textView.hasMarkedText()) else {
                parent.recalculateHeight(for: textView)
                return
            }
            parent.text = textView.string
            parent.recalculateHeight(for: textView)
        }
    }
}

// MARK: - ComposerNSTextView

private final class ComposerNSTextView: NSTextView {
    var onSubmit: (() -> Void)?
    var onSyncText: ((String) -> Void)?
    var onFileDrop: (([URL]) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.mouseDown(with: event)
    }

    override func doCommand(by selector: Selector) {
        if selector == #selector(insertNewline(_:)) {
            let modifiers = NSApp.currentEvent?.modifierFlags
                .intersection(.deviceIndependentFlagsMask) ?? []
            if modifiers.contains(.shift) {
                // insertLineBreak inserts U+2028 (LINE SEPARATOR), which
                // rides into the sent payload; this inserts a real "\n".
                super.doCommand(by: #selector(insertNewlineIgnoringFieldEditor(_:)))
            } else {
                // Sync the authoritative view string into the binding before
                // submitting: the IME publish gate can leave the binding
                // stale (marked text at submit time), and a submit that
                // reads a stale empty binding is silently swallowed.
                onSyncText?(string)
                onSubmit?()
            }
            return
        }
        super.doCommand(by: selector)
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard !droppedFileURLs(from: sender).isEmpty else {
            return super.draggingEntered(sender)
        }
        return .copy
    }

    override func prepareForDragOperation(_ sender: NSDraggingInfo) -> Bool {
        !droppedFileURLs(from: sender).isEmpty
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = droppedFileURLs(from: sender)
        guard !urls.isEmpty else { return super.performDragOperation(sender) }
        onFileDrop?(urls)
        return true
    }

    private func droppedFileURLs(from draggingInfo: NSDraggingInfo) -> [URL] {
        let pasteboard = draggingInfo.draggingPasteboard
        return pasteboard.readObjects(forClasses: [NSURL.self])?
            .compactMap { $0 as? URL } ?? []
    }
}
