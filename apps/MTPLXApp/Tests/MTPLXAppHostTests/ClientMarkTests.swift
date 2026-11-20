import AppKit
import SwiftUI
import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

/// The client marks: every launch target has a one-colour glyph drawn in
/// MTPLX's style, inside its frame, and the menu copies are templates (so
/// macOS gives them the menu's own text colour, never a colour of theirs).
final class ClientMarkTests: XCTestCase {
    @MainActor
    func testClientMarksUseTheBundledArtworkAtEveryMenuSize() throws {
        let artwork: [LaunchTarget: String] = [
            .claudeCode: "ClaudeMark", .codex: "ChatGPTMark",
            .openCode: "OpenCodeMark", .hermes: "HermesLogo",
        ]
        for (target, name) in artwork {
            let url = try XCTUnwrap(ClientMarkImages.artworkURL(for: target))
            XCTAssertEqual(url.deletingPathExtension().lastPathComponent, name)
            XCTAssertNotNil(NSImage(contentsOf: url))
        }
        for target in LaunchTarget.allCases {
            for size: CGFloat in [14, 16, 18, 22] {
                let image = ClientMarkImages.templateImage(for: target, size: size)
                XCTAssertTrue(image.isTemplate)
                if artwork[target] != nil || target == .deepseekHarness {
                    XCTAssertEqual(image.size, NSSize(width: size, height: size))
                } else {
                    XCTAssertNotNil(NSImage(systemSymbolName: target.systemImage, accessibilityDescription: nil))
                }
            }
        }
    }

    @MainActor
    func testMenuImagesAreTemplates() {
        for target in LaunchTarget.allCases {
            let image = ClientMarkImages.templateImage(for: target)
            XCTAssertTrue(image.isTemplate, "\(target)")
            XCTAssertGreaterThan(image.size.width, 0, "\(target)")
        }
    }

    /// The marks are drawn in one colour: rendered white on black, every
    /// pixel is a grey (no hue), and some pixels are lit.
    @MainActor
    func testMarksRenderInOneColour() throws {
        for target in LaunchTarget.allCases {
            let bitmap = try XCTUnwrap(Self.render(target, size: 36, tint: .white, background: .black))
            var lit = 0
            for y in 0..<bitmap.pixelsHigh {
                for x in 0..<bitmap.pixelsWide {
                    guard let colour = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                    let spread = max(colour.redComponent, colour.greenComponent, colour.blueComponent)
                        - min(colour.redComponent, colour.greenComponent, colour.blueComponent)
                    XCTAssertLessThan(spread, 0.02, "\(target) has a coloured pixel at \(x),\(y)")
                    if colour.redComponent > 0.5 { lit += 1 }
                }
            }
            XCTAssertGreaterThan(lit, 20, "\(target) drew too little")
        }
    }

    /// Writes every mark at 18 and 36 points on the dark and light floors
    /// to `MTPLX_MARKS_OUT` for a person to look at; skipped without it.
    @MainActor
    func testWriteMarksForReview() throws {
        guard let folder = ProcessInfo.processInfo.environment["MTPLX_MARKS_OUT"] else {
            throw XCTSkip("set MTPLX_MARKS_OUT to write the review sheet")
        }
        let sheet = HStack(spacing: 0) {
            ForEach([ColorScheme.dark, .light], id: \.self) { scheme in
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(LaunchTarget.allCases) { target in
                        HStack(spacing: 10) {
                            ClientMark(target: target, size: 18)
                            ClientMark(target: target, size: 36)
                            Text(target.title).foregroundStyle(Brand.typeBody)
                        }
                    }
                }
                .padding(16)
                .background(Brand.cardSurface)
                .environment(\.colorScheme, scheme)
            }
        }
        let renderer = ImageRenderer(content: sheet)
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.nsImage)
        let data = try XCTUnwrap(
            NSBitmapImageRep(data: try XCTUnwrap(image.tiffRepresentation))?
                .representation(using: .png, properties: [:])
        )
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        try data.write(to: URL(fileURLWithPath: folder).appendingPathComponent("client-marks.png"))
    }

    @MainActor
    private static func render(
        _ target: LaunchTarget, size: CGFloat, tint: Color, background: Color
    ) -> NSBitmapImageRep? {
        let view = ClientMark(target: target, size: size, tint: tint)
            .padding(2)
            .background(background)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        guard let cgImage = renderer.cgImage else { return nil }
        return NSBitmapImageRep(cgImage: cgImage)
    }
}
