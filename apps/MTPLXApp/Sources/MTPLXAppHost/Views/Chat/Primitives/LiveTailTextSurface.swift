import AppKit
import MTPLXAppCore

/// A fixed-size TextKit surface that draws the bottom of bounded text storage.
///
/// Unlike an `NSScrollView`/`NSClipView`, this view has no off-screen document
/// canvas and never scrolls on token arrival. TextKit lays out the bounded
/// projection, then this view draws only the glyphs intersecting its visible
/// bounds. The complete answer remains in `StreamingDocumentStore`.
@MainActor
final class LiveTailTextSurface: NSView {
    let textStorage = NSTextStorage()

    var contentInsets: NSSize = .zero {
        didSet {
            guard oldValue != contentInsets else { return }
            updateContainerWidth()
            needsDisplay = true
        }
    }

    /// While the text is shorter than the viewport, draw it from the top
    /// (the code card reserves its full slot at fence-open and fills down).
    /// Off by default: the reasoning ticker keeps its bottom-up look.
    var anchorsTopWhenShort = false

    private let layoutManager = NSLayoutManager()
    private let textContainer = NSTextContainer(size: .zero)

    override var isFlipped: Bool { true }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        // Isolate token repaints to this fixed viewport. A non-layer-backed
        // representable dirties the hosting window, which makes SwiftUI walk
        // the complete transcript layout even though this view's size never
        // changes. The old regression was a layer-backed 8K/16K *canvas*;
        // this layer is exactly the visible 72pt/420pt surface.
        wantsLayer = true
        layer?.masksToBounds = true
        layerContentsRedrawPolicy = .onSetNeedsDisplay
        textContainer.lineFragmentPadding = 0
        textContainer.lineBreakMode = .byCharWrapping
        textContainer.widthTracksTextView = false
        textContainer.heightTracksTextView = false
        layoutManager.allowsNonContiguousLayout = true
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        updateContainerWidth()
    }

    func textDidChange() {
        // The view has fixed geometry, so mutation does not need a
        // synchronous layout pass. AppKit calls `draw` in the next display
        // cycle and that path lays out the tail once before reading its line
        // fragment. Doing both here and in `draw` doubled TextKit work for
        // every streamed append.
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard textStorage.length > 0 else { return }

        UIStreamPerfProbe.renderTimed(.draw, size: textStorage.length) {
            ensureTailLayout()
            let textHeight = laidOutTextHeight()
            let bottomAnchoredY = bounds.height - contentInsets.height - textHeight
            let origin = NSPoint(
                x: contentInsets.width,
                y: anchorsTopWhenShort
                    ? min(contentInsets.height, bottomAnchoredY)
                    : bottomAnchoredY
            )
            let textDirtyRect = dirtyRect.offsetBy(dx: -origin.x, dy: -origin.y)
            let glyphRange = layoutManager.glyphRange(
                forBoundingRect: textDirtyRect,
                in: textContainer
            )
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(rect: bounds).addClip()
            layoutManager.drawBackground(forGlyphRange: glyphRange, at: origin)
            layoutManager.drawGlyphs(forGlyphRange: glyphRange, at: origin)
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    private func updateContainerWidth() {
        let width = max(1, bounds.width - contentInsets.width * 2)
        let size = NSSize(width: width, height: .greatestFiniteMagnitude)
        guard textContainer.containerSize != size else { return }
        textContainer.containerSize = size
        layoutManager.invalidateLayout(
            forCharacterRange: NSRange(location: 0, length: textStorage.length),
            actualCharacterRange: nil
        )
        needsDisplay = true
    }

    private func ensureTailLayout() {
        guard textStorage.length > 0 else { return }
        layoutManager.ensureLayout(
            forCharacterRange: NSRange(location: textStorage.length - 1, length: 1)
        )
    }

    private func laidOutTextHeight() -> CGFloat {
        guard textStorage.length > 0 else { return 0 }
        let characterRange = NSRange(location: textStorage.length - 1, length: 1)
        let glyphRange = layoutManager.glyphRange(
            forCharacterRange: characterRange,
            actualCharacterRange: nil
        )
        var height: CGFloat = 0
        if glyphRange.length > 0 {
            height = layoutManager.lineFragmentUsedRect(
                forGlyphAt: NSMaxRange(glyphRange) - 1,
                effectiveRange: nil
            ).maxY
        }
        if layoutManager.extraLineFragmentTextContainer === textContainer {
            height = max(height, layoutManager.extraLineFragmentUsedRect.maxY)
        }
        return ceil(height)
    }
}
