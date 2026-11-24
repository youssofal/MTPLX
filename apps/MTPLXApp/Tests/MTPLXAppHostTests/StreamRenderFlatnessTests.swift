import AppKit
import Combine
import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// Streamwar 2026-08-19 O(n^2) tripwires.
///
/// The installed 2.8.3 shipped with the live code card's TextKit storage
/// holding the ENTIRE growing fence behind a fixed 420 pt viewport: every
/// draw ran far-end glyph queries over the whole storage, so per-frame cost
/// grew with fence length and the freeze landed exactly at line boundaries.
/// Six commits shipped green because nothing asserted flatness. These tests
/// do — they drive the REAL apply/draw/derive pipeline with a synthetic
/// 500-line stream and fail on any cost or storage growth curve.
final class StreamRenderFlatnessTests: XCTestCase {

    private static let lineBody =
        "let value = compute(input) + offset // synthetic streamed line"

    private func fragment(id: Int, text: String) -> StreamingCodeFragment {
        StreamingCodeFragment(
            block: StreamingDocumentBlock(
                id: id,
                text: text,
                kind: .unfinished,
                finalized: true
            ),
            entryTag: MTPLXCodeHighlighter.LexState.none.cacheTag
        )
    }

    /// Grow a synthetic fence to `lineCount` lines, folding the oldest 32
    /// single lines into one merged segment every 32 new lines — the same
    /// restructuring StreamingDocumentStore's coalesce performs.
    @MainActor
    private func growFence(
        to lineCount: Int,
        surface: LiveTailTextSurface,
        coordinator: StreamingCodeTextViewport.Coordinator,
        onLine: ((Int, [StreamingCodeFragment]) -> Void)? = nil
    ) {
        var fragments: [StreamingCodeFragment] = []
        var nextMergeBoundary = 64
        for line in 1...lineCount {
            fragments.append(fragment(id: line, text: "\(Self.lineBody) #\(line)"))
            if line >= nextMergeBoundary {
                // Fold the oldest run of 32 single lines into one merged
                // segment (id = first constituent's), exactly like the
                // store's coalesce: runs AFTER earlier merged segments
                // keep folding as the fence grows.
                if let firstSingle = fragments.firstIndex(where: { !$0.text.contains("\n") }) {
                    let run = fragments[firstSingle...].prefix { !$0.text.contains("\n") }
                    if run.count >= 32 {
                        let merged = fragment(
                            id: run.first!.id,
                            text: run.prefix(32).map(\.text).joined(separator: "\n")
                        )
                        fragments.replaceSubrange(
                            firstSingle..<(firstSingle + 32), with: [merged]
                        )
                    }
                }
                nextMergeBoundary += 32
            }
            StreamingCodeTextViewport.apply(
                fragments: fragments,
                language: .generic,
                to: surface,
                coordinator: coordinator
            )
            onLine?(line, fragments)
        }
    }

    @MainActor
    private func makeSurface() -> LiveTailTextSurface {
        let surface = LiveTailTextSurface(frame: NSRect(x: 0, y: 0, width: 420, height: 420))
        surface.contentInsets = NSSize(width: 12, height: 10)
        surface.anchorsTopWhenShort = true
        surface.layout()
        return surface
    }

    // MARK: Storage bound — the deterministic A1 tripwire

    @MainActor
    func testLiveCodeStorageStaysBoundedWhileFenceGrows() {
        let surface = makeSurface()
        let coordinator = StreamingCodeTextViewport.Coordinator()
        var worstLength = 0
        growFence(to: 500, surface: surface, coordinator: coordinator) { _, _ in
            worstLength = max(worstLength, surface.textStorage.length)
        }
        // Window invariant: >= 48 rendered lines, plus at most one unfolded
        // merge run (32) and the fresh window margin before the next fold.
        // 130 lines is comfortably above the design bound; the unbounded
        // regression reaches 500 lines here and fails by 4x.
        let lineUTF16 = (Self.lineBody + " #500\n").utf16.count
        let boundLines = 130
        XCTAssertLessThanOrEqual(
            worstLength,
            boundLines * lineUTF16,
            "live-fence TextKit storage grew past the bounded window — O(fence) draw cost is back"
        )
        // And the window must still end with the newest line (bottom-anchored tail).
        XCTAssertTrue(
            surface.textStorage.string.hasSuffix("#500"),
            "windowed storage lost the newest line"
        )
    }

    @MainActor
    func testWindowedStorageMatchesDocumentTail() {
        let surface = makeSurface()
        let coordinator = StreamingCodeTextViewport.Coordinator()
        var lastFragments: [StreamingCodeFragment] = []
        growFence(to: 200, surface: surface, coordinator: coordinator) { _, fragments in
            lastFragments = fragments
        }
        let full = lastFragments.map(\.text).joined(separator: "\n")
        let windowed = surface.textStorage.string
        XCTAssertTrue(
            full.hasSuffix(windowed),
            "windowed storage must be byte-identical to the fence tail"
        )
        XCTAssertGreaterThanOrEqual(
            windowed.split(separator: "\n").count, 48,
            "window must keep at least two viewports of lines"
        )
    }

    // MARK: Cost flatness — line 400 vs line 40

    @MainActor
    func testApplyPlusDrawCostStaysFlatAcrossFenceGrowth() {
        let surface = makeSurface()
        let coordinator = StreamingCodeTextViewport.Coordinator()
        let image = NSImage(size: NSSize(width: 420, height: 420))

        func drawOnce() {
            image.lockFocus()
            surface.draw(surface.bounds)
            image.unlockFocus()
        }

        var costAt40: [Double] = []
        var costAt400: [Double] = []
        growFence(to: 460, surface: surface, coordinator: coordinator) { line, _ in
            guard (36...42).contains(line) || (396...402).contains(line) else { return }
            let started = ProcessInfo.processInfo.systemUptime
            drawOnce()
            let ms = (ProcessInfo.processInfo.systemUptime - started) * 1000
            if line <= 42 { costAt40.append(ms) } else { costAt400.append(ms) }
        }

        let early = costAt40.sorted()[costAt40.count / 2]
        let late = costAt400.sorted()[costAt400.count / 2]
        // Flat = the tripwire. Allow 1.5x plus a small absolute epsilon so
        // micro-costs (<1 ms) can't flake the gate.
        XCTAssertLessThanOrEqual(
            late,
            max(early * 1.5, early + 0.5),
            "draw cost grew with fence length (line 400: \(late) ms vs line 40: \(early) ms) — the O(n^2) curve is back"
        )
    }

    @MainActor
    func testRenderItemsCostStaysFlatAcrossDocumentGrowth() {
        let store = StreamingDocumentStore(mode: .plainLines)
        let lexChain = StreamingFenceLexChain()
        store.append("```swift\n")

        func deriveCostMs() -> Double {
            let started = ProcessInfo.processInfo.systemUptime
            _ = StreamingAssistantMarkdownView.renderItems(
                for: store.blocks, lexChain: lexChain
            )
            return (ProcessInfo.processInfo.systemUptime - started) * 1000
        }

        var costAt40 = [Double]()
        var costAt400 = [Double]()
        for line in 1...420 {
            store.append("\(Self.lineBody) #\(line)\n")
            if (36...42).contains(line) { costAt40.append(deriveCostMs()) }
            if (396...402).contains(line) { costAt400.append(deriveCostMs()) }
            _ = StreamingAssistantMarkdownView.renderItems(
                for: store.blocks, lexChain: lexChain
            )
        }
        let early = costAt40.sorted()[costAt40.count / 2]
        let late = costAt400.sorted()[costAt400.count / 2]
        XCTAssertLessThanOrEqual(
            late,
            max(early * 1.5, early + 0.5),
            "renderItems cost grew with document length (line 400: \(late) ms vs line 40: \(early) ms)"
        )
    }

    // MARK: SwiftUI publish gate — bounded height ramp, then the transcript sleeps

    @MainActor
    func testRenderModelPublishesOnlyBoundedHeightRampWhileFenceGrows() {
        let store = StreamingDocumentStore(mode: .plainLines)
        store.append("```swift\n")
        store.append("\(Self.lineBody) #1\n")
        let model = StreamingRichRenderModel(document: store)

        var publishes = 0
        var cancellables: Set<AnyCancellable> = []
        model.objectWillChange.sink { _ in publishes += 1 }.store(in: &cancellables)

        // Ramp phase: the card grows in 4-line buckets, so the model may
        // publish — but only the bounded handful of height steps to the
        // 420 pt cap, never per line.
        for line in 2...30 {
            store.append("\(Self.lineBody) #\(line)\n")
        }
        XCTAssertLessThanOrEqual(
            publishes, 7,
            "SwiftUI republished \(publishes) times during the height ramp — the bucket must publish at most once per 4-line step"
        )
        XCTAssertEqual(model.openFenceLineBucket, 24, "30 lines must saturate the height cap")

        // Past the cap the transcript must sleep: zero publishes while the
        // fence grows from line 31 to 200 — this is the O(n^2) tripwire.
        let publishesAtCap = publishes
        for line in 31...200 {
            store.append("\(Self.lineBody) #\(line)\n")
        }
        XCTAssertEqual(
            publishes, publishesAtCap,
            "SwiftUI republished past the height cap — the live card owns growth; the transcript must stay static"
        )

        // The fence CLOSE must still publish (open card -> settled card),
        // and the bucket must reset for the next fence.
        store.append("```\n")
        XCTAssertGreaterThan(publishes, publishesAtCap, "fence close must republish the transcript")
        XCTAssertEqual(model.openFenceLineBucket, 0, "closed fence must clear the ramp bucket")
    }

    @MainActor
    func testOpenFenceLineBucketRampIsMonotoneAndCapped() {
        let store = StreamingDocumentStore(mode: .plainLines)
        store.append("```swift\n")
        let model = StreamingRichRenderModel(document: store)

        // A just-opened fence gets a small box, not the whole empty slot.
        XCTAssertLessThanOrEqual(model.openFenceLineBucket, 4)

        var previous = model.openFenceLineBucket
        for line in 1...40 {
            store.append("\(Self.lineBody) #\(line)\n")
            let bucket = model.openFenceLineBucket
            XCTAssertGreaterThanOrEqual(bucket, previous, "height ramp must never shrink mid-fence")
            XCTAssertLessThanOrEqual(bucket, 24, "ramp must cap at the 420 pt slot")
            XCTAssertEqual(bucket % 4, 0, "ramp must move in 4-line steps")
            previous = bucket
        }
        XCTAssertEqual(previous, 24, "40 lines must saturate the cap")
    }

    // MARK: Rich streaming prose — bounded SwiftUI churn & zero mid-line relayouts

    @MainActor
    func testRichMarkdown200LinesPublishesBoundedSwiftUIItems() {
        let store = StreamingDocumentStore(mode: .plainLines)
        let model = StreamingRichRenderModel(document: store)

        var objectWillChangeCount = 0
        var cancellables: Set<AnyCancellable> = []
        model.objectWillChange.sink { _ in objectWillChangeCount += 1 }.store(in: &cancellables)

        // Stream 200 lines of mixed Markdown prose + an interleaved code
        // fence in sub-line chunks (simulating 120 Hz typewriter reveals).
        for line in 1...200 {
            if line == 95 {
                store.append("```swift\n")
                for c in 1...12 {
                    store.append("let step\(c) = \(c) * 2\n")
                }
                store.append("```\n")
                continue
            }
            let content: String
            switch line % 7 {
            case 0: content = "## Section \(line)"
            case 1: content = "Paragraph with **bold**, *italic*, and `inlineCode(\(line))`."
            case 2: content = "- Bullet item **\(line)** with [reference](https://example.com)"
            case 3: content = "\(line). Ordered item with ~~old~~ and $x^2 + 1$"
            case 4: content = "> Quoted commentary on line \(line)"
            case 5: content = "---"
            default: content = "Plain body sentence continuing across line \(line)."
            }
            // Deliver each line in two sub-line chunks to mimic mid-line ticks.
            let mid = content.index(content.startIndex, offsetBy: content.count / 2)
            store.append(String(content[..<mid]))
            store.append(String(content[mid...]) + "\n")
        }

        // Before StreamingProseTextSurface, 200 lines produced ~54 separate
        // `.plain` blocks and 400+ SwiftUI `@Published items` publishes.
        // With bounded `.prose` chunks fed over `perRevision`, publishes
        // only happen on chunk boundaries and code-fence open/close.
        XCTAssertLessThanOrEqual(
            model.itemsPublishCount, 12,
            "SwiftUI @Published items churned \(model.itemsPublishCount) times over 200 rich markdown lines"
        )
        XCTAssertLessThanOrEqual(
            model.items.count, 8,
            "Render item count (\(model.items.count)) grew unbounded instead of grouping prose runs"
        )
        XCTAssertLessThanOrEqual(
            objectWillChangeCount, 16,
            "StreamingRichRenderModel fired objectWillChange \(objectWillChangeCount) times"
        )
    }

    @MainActor
    func testStreamingProseSurfaceDoesNotInvalidateIntrinsicSizeMidLine() {
        let surface = StreamingProseNSView(frame: NSRect(x: 0, y: 0, width: 548, height: 100))
        surface.layout()
        let coordinator = StreamingProseTextSurface.Coordinator()

        // First character establishes the initial line height.
        let initialBlock = StreamingDocumentBlock(id: 0, text: "H", kind: .unfinished, finalized: false)
        StreamingProseTextSurface.apply(blocks: [initialBlock], to: surface, coordinator: coordinator)
        let initialHeight = surface.measuredHeight
        let invalidationsAfterFirstChar = surface.intrinsicSizeInvalidationCount

        // Append characters along the same single visual line — height does
        // not change, so intrinsic content size MUST NOT invalidate.
        var growing = "H"
        for ch in "ello streaming prose at 120 Hz!" {
            growing.append(ch)
            let block = StreamingDocumentBlock(id: 0, text: growing, kind: .unfinished, finalized: false)
            StreamingProseTextSurface.apply(blocks: [block], to: surface, coordinator: coordinator)
        }
        XCTAssertEqual(
            surface.measuredHeight, initialHeight,
            "single-line typing should keep the same measured height"
        )
        XCTAssertEqual(
            surface.intrinsicSizeInvalidationCount, invalidationsAfterFirstChar,
            "mid-line character appends must trigger ZERO intrinsic size invalidations"
        )

        // Adding a second line grows height and MUST invalidate once so
        // SwiftUI expands the bubble and pins the scroll view synchronously.
        let blocksTwoLines = [
            StreamingDocumentBlock(id: 0, text: growing, kind: .plain, finalized: true),
            StreamingDocumentBlock(id: 1, text: "Second line arrives", kind: .unfinished, finalized: false),
        ]
        StreamingProseTextSurface.apply(blocks: blocksTwoLines, to: surface, coordinator: coordinator)
        XCTAssertGreaterThan(surface.measuredHeight, initialHeight)
        XCTAssertEqual(
            surface.intrinsicSizeInvalidationCount,
            invalidationsAfterFirstChar + 1,
            "new line should invalidate intrinsic size exactly once"
        )
    }

    @MainActor
    func testStreamingProseFormatterStylesMarkdownIncrementallyWithoutPartialFlicker() {
        // Unclosed bold stays literal while streaming; closed bold strips markers and bolds.
        let partial = StreamingProseMarkdownFormatter.formatLine("Hello **wor", finalized: false)
        XCTAssertEqual(partial.string, "Hello **wor")

        let closed = StreamingProseMarkdownFormatter.formatLine("Hello **world** and `code`", finalized: true)
        XCTAssertEqual(closed.string, "Hello world and code")
        let worldRange = (closed.string as NSString).range(of: "world")
        let worldFont = closed.attribute(.font, at: worldRange.location, effectiveRange: nil) as? NSFont
        XCTAssertNotNil(worldFont)
        XCTAssertTrue(
            NSFontManager.shared.traits(of: worldFont!).contains(.boldFontMask)
                || (worldFont?.fontDescriptor.symbolicTraits.contains(.bold) ?? false)
                || (worldFont?.fontName.lowercased().contains("semibold") ?? false)
                || (worldFont?.fontName.lowercased().contains("bold") ?? false)
        )
        let codeRange = (closed.string as NSString).range(of: "code")
        let codeFont = closed.attribute(.font, at: codeRange.location, effectiveRange: nil) as? NSFont
        XCTAssertTrue(codeFont?.isFixedPitch ?? false, "inline code must use a monospace font")

        // Heading strips # markers and enlarges font.
        let heading = StreamingProseMarkdownFormatter.formatLine("# Big Title", finalized: true)
        XCTAssertEqual(heading.string, "Big Title")
        let headingFont = heading.attribute(.font, at: 0, effectiveRange: nil) as? NSFont
        XCTAssertGreaterThanOrEqual(headingFont?.pointSize ?? 0, 20)

        // Bullet replaces leading '-' with '•\t'.
        let bullet = StreamingProseMarkdownFormatter.formatLine("- Item one", finalized: true)
        XCTAssertTrue(bullet.string.hasPrefix("•\tItem one"))

        // Intra-word underscores stay untouched (snake_case).
        let snake = StreamingProseMarkdownFormatter.formatLine("use foo_bar_baz here", finalized: true)
        XCTAssertEqual(snake.string, "use foo_bar_baz here")
    }

    @MainActor
    func testStreamingProseApplyPlusDrawCostStaysFlatAcross240Lines() {
        let store = StreamingDocumentStore(mode: .plainLines)
        let model = StreamingRichRenderModel(document: store)
        var surfaces: [Int: (StreamingProseNSView, StreamingProseTextSurface.Coordinator)] = [:]
        let image = NSImage(size: NSSize(width: 548, height: 1200))

        func syncAndDrawActive() -> Double {
            let started = ProcessInfo.processInfo.systemUptime
            if case let .prose(id, blocks)? = model.latestItems.last {
                let pair = surfaces[id] ?? {
                    let v = StreamingProseNSView(frame: NSRect(x: 0, y: 0, width: 548, height: 1200))
                    v.layout()
                    let c = StreamingProseTextSurface.Coordinator()
                    surfaces[id] = (v, c)
                    return (v, c)
                }()
                StreamingProseTextSurface.apply(blocks: blocks, to: pair.0, coordinator: pair.1)
                image.lockFocus()
                pair.0.draw(NSRect(x: 0, y: 0, width: 548, height: min(1200, pair.0.measuredHeight)))
                image.unlockFocus()
            }
            return (ProcessInfo.processInfo.systemUptime - started) * 1000
        }

        var costAt30: [Double] = []
        var costAt220: [Double] = []
        for line in 1...240 {
            store.append("Line \(line) with **bold** and `code_\(line)` streaming smoothly.\n")
            let ms = syncAndDrawActive()
            if (26...34).contains(line) { costAt30.append(ms) }
            if (216...224).contains(line) { costAt220.append(ms) }
        }

        let early = costAt30.sorted()[costAt30.count / 2]
        let late = costAt220.sorted()[costAt220.count / 2]
        XCTAssertLessThanOrEqual(
            late,
            max(early * 2.0, early + 1.5),
            "prose apply+draw grew with document length (line 220: \(late) ms vs line 30: \(early) ms)"
        )
    }

    @MainActor
    func testPlainTextOnlyRenderModelSkipsCodeCardsAndRendersRawMarkdownLiterals() {
        let store = StreamingDocumentStore(mode: .plainLines)
        let model = StreamingRichRenderModel(document: store, plainTextOnly: true)
        var surfaces: [Int: (StreamingProseNSView, StreamingProseTextSurface.Coordinator)] = [:]

        for line in 1...200 {
            if line == 1 {
                store.append("# Heading `code`\n")
            } else if line == 10 {
                store.append("```swift\n")
            } else if line == 25 {
                store.append("```\n")
            } else {
                store.append("Line \(line) with **bold** and `inline` literal text\n")
            }
            for item in model.latestItems {
                if case let .prose(id, blocks) = item {
                    let pair = surfaces[id] ?? {
                        let v = StreamingProseNSView(frame: NSRect(x: 0, y: 0, width: 548, height: 1200))
                        v.layout()
                        let c = StreamingProseTextSurface.Coordinator()
                        surfaces[id] = (v, c)
                        return (v, c)
                    }()
                    StreamingProseTextSurface.apply(
                        blocks: blocks,
                        to: pair.0,
                        coordinator: pair.1,
                        plainTextOnly: true
                    )
                }
            }
        }

        XCTAssertFalse(model.latestItems.isEmpty)
        for item in model.latestItems {
            switch item {
            case .prose:
                break
            case .openCode, .closedCode, .plain, .settled:
                XCTFail("plainTextOnly render model must emit .prose items only, got \(item)")
            }
        }
        XCTAssertLessThanOrEqual(
            model.itemsPublishCount,
            6,
            "plainTextOnly AppKit routing must keep SwiftUI items publishes bounded (<= 6 across 200 lines)"
        )

        let combinedRendered = surfaces.keys.sorted()
            .compactMap { surfaces[$0]?.0.textStorage.string }
            .joined(separator: "\n")
        XCTAssertTrue(combinedRendered.contains("# Heading `code`"))
        XCTAssertTrue(combinedRendered.contains("```swift"))
        XCTAssertTrue(combinedRendered.contains("**bold**"))
    }
}
