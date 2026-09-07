import AppKit
import CoreGraphics
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// ⌘V in the composer, at the AppKit seam: the real `ComposerNSTextView`
/// mounted under `NSHostingView`, fed from a private named pasteboard so
/// the user's clipboard is never read or written. An image or a Finder
/// copy must reach the attachment callbacks, and Edit ▸ Paste must
/// validate for them — a disabled Paste swallows ⌘V before `paste(_:)`
/// ever runs, which is exactly how the composer used to ignore
/// screenshots.
final class ComposerPasteTests: XCTestCase {
    @MainActor
    private final class Box {
        var text = ""
        var droppedURLs: [[URL]] = []
        var pastedImages: [Data] = []
    }

    @MainActor
    private func mountComposer(
        box: Box,
        pasteboard: NSPasteboard
    ) throws -> (host: NSHostingView<ComposerInputTextView>, textView: ComposerNSTextView) {
        let view = ComposerInputTextView(
            text: Binding(get: { box.text }, set: { box.text = $0 }),
            measuredHeight: .constant(44),
            minHeight: 44,
            maxHeight: 160,
            onSubmit: {},
            onFileDrop: { box.droppedURLs.append($0) },
            onImagePaste: { box.pastedImages.append($0) }
        )
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 420, height: 64)
        host.layoutSubtreeIfNeeded()
        RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        guard let textView = Self.firstComposerTextView(in: host) else {
            throw XCTSkip("composer NSTextView did not mount under NSHostingView")
        }
        textView.pasteboard = pasteboard
        return (host, textView)
    }

    @MainActor
    private static func firstComposerTextView(in view: NSView) -> ComposerNSTextView? {
        if let textView = view as? ComposerNSTextView { return textView }
        for child in view.subviews {
            if let found = firstComposerTextView(in: child) { return found }
        }
        return nil
    }

    /// A pasteboard of our own; `NSPasteboard.general` stays untouched.
    private func privatePasteboard() -> NSPasteboard {
        let name = "mtplx.tests.composer-paste.\(UUID().uuidString)"
        let pasteboard = NSPasteboard(name: NSPasteboard.Name(name))
        addTeardownBlock { NSPasteboard(name: NSPasteboard.Name(name)).releaseGlobally() }
        pasteboard.clearContents()
        return pasteboard
    }

    private func scratchFile(named name: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-composer-paste-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(name)
        try "attached".write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    /// Edit ▸ Paste as the menu presents it for validation.
    @MainActor
    private static func pasteItem() -> NSMenuItem {
        NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
    }

    // MARK: paste(_:)

    @MainActor
    func testPastingAnImageAttachesItAndLeavesTheTextAlone() throws {
        let box = Box()
        let pasteboard = privatePasteboard()
        let png = try Self.pngData(width: 5, height: 4)
        pasteboard.setData(png, forType: .png)
        let (host, textView) = try mountComposer(box: box, pasteboard: pasteboard)
        defer { _ = host }

        textView.paste(nil)

        XCTAssertEqual(box.pastedImages, [png], "the screenshot reaches the composer as its exact PNG bytes")
        XCTAssertTrue(box.droppedURLs.isEmpty)
        XCTAssertEqual(textView.string, "", "nothing is inserted into the draft")
        XCTAssertEqual(box.text, "")
    }

    @MainActor
    func testPastingFilesCopiedInFinderTakesTheDropPath() throws {
        let box = Box()
        let pasteboard = privatePasteboard()
        let url = try scratchFile(named: "brief.md")
        XCTAssertTrue(pasteboard.writeObjects([url as NSURL]))
        let (host, textView) = try mountComposer(box: box, pasteboard: pasteboard)
        defer { _ = host }

        textView.paste(nil)

        XCTAssertEqual(box.droppedURLs, [[url]])
        XCTAssertTrue(box.pastedImages.isEmpty)
        XCTAssertEqual(textView.string, "")
    }

    /// A copied image FILE is a file: the read must not decode it into
    /// image bytes the classifier would only discard (a large photo would
    /// stall ⌘V).
    @MainActor
    func testACopiedImageFileIsReadAsAFileNotDecoded() throws {
        let pasteboard = privatePasteboard()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-composer-paste-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("photo.png")
        try Self.pngData(width: 3, height: 3).write(to: url)
        XCTAssertTrue(pasteboard.writeObjects([url as NSURL]))

        let contents = ComposerNSTextView.pasteboardContents(pasteboard)

        XCTAssertEqual(contents.urls, [url])
        XCTAssertNil(contents.imageData, "types on the pasteboard: \(pasteboard.types ?? [])")
        XCTAssertEqual(ComposerPasteClassifier.classify(contents), .files([url]))
    }

    // MARK: Edit ▸ Paste validation

    @MainActor
    func testPasteIsEnabledForAnImageOnlyPasteboard() throws {
        let box = Box()
        let pasteboard = privatePasteboard()
        pasteboard.setData(try Self.pngData(width: 2, height: 2), forType: .png)
        let (host, textView) = try mountComposer(box: box, pasteboard: pasteboard)
        defer { _ = host }

        XCTAssertTrue(textView.validateUserInterfaceItem(Self.pasteItem()))
    }

    @MainActor
    func testPasteIsEnabledForAFinderCopy() throws {
        let box = Box()
        let pasteboard = privatePasteboard()
        XCTAssertTrue(pasteboard.writeObjects([try scratchFile(named: "deck.pdf") as NSURL]))
        let (host, textView) = try mountComposer(box: box, pasteboard: pasteboard)
        defer { _ = host }

        XCTAssertTrue(textView.validateUserInterfaceItem(Self.pasteItem()))
    }

    @MainActor
    func testPasteIsNotClaimedForAnEmptyPasteboard() throws {
        let box = Box()
        let pasteboard = privatePasteboard()
        let (host, textView) = try mountComposer(box: box, pasteboard: pasteboard)
        defer { _ = host }

        XCTAssertFalse(ComposerNSTextView.pasteboardMayHoldAttachment(pasteboard))
    }

    // MARK: The pasteboard read

    @MainActor
    func testTextOnlyPasteboardReadsAsPassthrough() throws {
        let pasteboard = privatePasteboard()
        pasteboard.setString("just words, and a link https://example.com/a.png", forType: .string)

        let contents = ComposerNSTextView.pasteboardContents(pasteboard)

        XCTAssertTrue(contents.urls.isEmpty, "plain text is not a URL object")
        XCTAssertNil(contents.imageData)
        XCTAssertEqual(ComposerPasteClassifier.classify(contents), .passthrough)
        XCTAssertFalse(ComposerNSTextView.pasteboardMayHoldAttachment(pasteboard))
    }

    @MainActor
    func testPasteboardReadPrefersPNGOverTIFF() throws {
        let pasteboard = privatePasteboard()
        let png = try Self.pngData(width: 3, height: 3)
        let tiff = try Self.tiffData(width: 3, height: 3)
        pasteboard.setData(tiff, forType: .tiff)
        pasteboard.setData(png, forType: .png)

        XCTAssertEqual(ComposerNSTextView.pasteboardContents(pasteboard).imageData, png)
    }

    /// Only a JPEG was written. The pasteboard server may translate it to
    /// PNG/TIFF on request or hand back the JPEG itself; either way the
    /// read finds an image and the classifier delivers PNG.
    @MainActor
    func testPasteboardReadFallsBackToAnyImageNSImageDecodes() throws {
        let pasteboard = privatePasteboard()
        let jpeg = try Self.encoded(try Self.solidImage(width: 3, height: 3), as: .jpeg)
        pasteboard.setData(jpeg, forType: NSPasteboard.PasteboardType(UTType.jpeg.identifier))

        let contents = ComposerNSTextView.pasteboardContents(pasteboard)
        XCTAssertNotNil(contents.imageData)
        guard case .image(let data) = ComposerPasteClassifier.classify(contents) else {
            return XCTFail("a JPEG on the pasteboard must classify as an image")
        }
        XCTAssertEqual(Array(data.prefix(8)), [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], "normalized to PNG")
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        XCTAssertEqual(properties[kCGImagePropertyPixelWidth] as? Int, 3)
        XCTAssertEqual(properties[kCGImagePropertyPixelHeight] as? Int, 3)
        XCTAssertTrue(ComposerNSTextView.pasteboardMayHoldAttachment(pasteboard))
    }

    // MARK: Fixtures

    private static func solidImage(width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw NSError(domain: "ComposerPasteTests", code: 1) }
        context.setFillColor(CGColor(red: 0.1, green: 0.6, blue: 0.4, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else { throw NSError(domain: "ComposerPasteTests", code: 2) }
        return image
    }

    private static func encoded(_ image: CGImage, as type: UTType) throws -> Data {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type.identifier as CFString, 1, nil) else {
            throw NSError(domain: "ComposerPasteTests", code: 3)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw NSError(domain: "ComposerPasteTests", code: 4) }
        return output as Data
    }

    private static func pngData(width: Int, height: Int) throws -> Data {
        try encoded(try solidImage(width: width, height: height), as: .png)
    }

    private static func tiffData(width: Int, height: Int) throws -> Data {
        try encoded(try solidImage(width: width, height: height), as: .tiff)
    }
}
