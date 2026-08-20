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
}
