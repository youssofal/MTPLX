import AppKit
import CoreText
import SwiftUI
import MTPLXAppCore

// MARK: - RollingDigits
//
// The hero number with its digit roll drawn from `GaugeRoll` at a given
// time, the way the counter has always rolled: a digit that changed slides
// up out of the band the digits fill, shrinking, blurring and fading, while
// its replacement rises into the band from below, growing to full size and
// sharpening; both are cut at the band's edges. Digits that did not change
// stay exactly where they are. At rest it is one plain `Text`.
//
// Each layer is the whole string with the characters it does not draw made
// transparent, so every glyph sits where the resting text puts it: nothing
// shifts when a roll starts or lands. Font, colour and fitting come from the
// caller's modifiers; lengths here are fractions of the text's line box, so
// a gauge that shrinks its number rolls in proportion.

struct RollingDigits: View {
    let roll: GaugeRoll
    let time: Double

    /// Blur on a digit at the far end of its travel, as a fraction of the
    /// line height.
    static let maxBlur: CGFloat = 0.08

    var body: some View {
        if roll.isRolling(at: time) {
            let progress = CGFloat(roll.progress(at: time))
            let fade = CGFloat(roll.fade(at: time))
            let gone = min(1, max(0, progress))
            let shrink = CGFloat(1 - GaugeRoll.minimumScale)
            let newCenters = Self.characterCenters(roll.to)
            let oldCenters = Self.characterCenters(roll.from)
            ZStack {
                Self.layer(roll.to) { !roll.changes(at: $0) }
                ForEach(Self.changedIndices(roll, in: roll.to), id: \.self) { index in
                    Self.rolling(
                        Self.layer(roll.to) { $0 == index },
                        anchorX: newCenters[safe: index],
                        offset: 1 - progress, blur: 1 - fade, opacity: fade,
                        scale: 1 - shrink * (1 - fade)
                    )
                }
                ForEach(Self.changedIndices(roll, in: roll.from), id: \.self) { index in
                    Self.rolling(
                        Self.layer(roll.from) { $0 == index },
                        anchorX: oldCenters[safe: index],
                        offset: -progress, blur: min(1, 3 * gone), opacity: 1 - gone,
                        scale: 1 - shrink * gone
                    )
                }
            }
        } else {
            Text(roll.to)
        }
    }

    /// One changing digit on its way: sized about its own centre, moved by
    /// `offset` band heights, blurred by `blur` of the maximum, and cut to
    /// the band.
    private static func rolling(
        _ text: Text, anchorX: CGFloat, offset: CGFloat, blur: CGFloat, opacity: CGFloat, scale: CGFloat
    ) -> some View {
        let band = digitBand
        let maxBlur = maxBlur
        return text
            .scaleEffect(scale, anchor: UnitPoint(x: anchorX, y: band.top + band.height / 2))
            .visualEffect { content, proxy in
                content
                    .offset(y: proxy.size.height * band.height * offset)
                    .blur(radius: proxy.size.height * maxBlur * blur)
            }
            .opacity(opacity)
            .mask {
                GeometryReader { proxy in
                    Rectangle()
                        .frame(height: proxy.size.height * band.height)
                        .offset(y: proxy.size.height * band.top)
                }
            }
    }

    /// The characters of `text` this roll changes: those that differ, or all
    /// of them when the length changes.
    private static func changedIndices(_ roll: GaugeRoll, in text: String) -> [Int] {
        (0..<text.count).filter { roll.changes(at: $0) }
    }

    /// `text` with the characters `shows` rejects drawn transparent.
    private static func layer(_ text: String, shows: (Int) -> Bool) -> Text {
        var attributed = AttributedString()
        for (index, character) in text.enumerated() {
            var run = AttributedString(String(character))
            if !shows(index) { run.swiftUI.foregroundColor = .clear }
            attributed.append(run)
        }
        return Text(attributed)
    }

    /// Where each character's centre sits across `text`, as fractions of its
    /// width, in the hero's face.
    private static func characterCenters(_ text: String) -> [CGFloat] {
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: [.font: heroFont()])
        )
        let width = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        guard width > 0 else { return [] }
        var centers: [CGFloat] = []
        var offset = 0
        for character in text {
            let length = character.utf16.count
            let start = CTLineGetOffsetForStringIndex(line, offset, nil)
            let end = CTLineGetOffsetForStringIndex(line, offset + length, nil)
            centers.append((start + end) / 2 / width)
            offset += length
        }
        return centers
    }

    /// The hero's face: rounded heavy at 64 pt with figures of one width,
    /// as `GaugeView` sets it.
    private static func heroFont() -> NSFont {
        let base = NSFont.systemFont(ofSize: 64, weight: .heavy)
        let rounded = base.fontDescriptor.withDesign(.rounded) ?? base.fontDescriptor
        let descriptor = rounded.addingAttributes([
            .featureSettings: [[
                NSFontDescriptor.FeatureKey.typeIdentifier: kNumberSpacingType,
                NSFontDescriptor.FeatureKey.selectorIdentifier: kMonospacedNumbersSelector,
            ]],
        ])
        return NSFont(descriptor: descriptor, size: 64) ?? base
    }

    /// The band the digits' ink fills, from the top of the text's line box,
    /// as fractions of its height.
    static let digitBand: (top: CGFloat, height: CGFloat) = {
        let font = heroFont()
        var characters = Array("0123456789".utf16)
        var glyphs = [CGGlyph](repeating: 0, count: characters.count)
        var rects = [CGRect](repeating: .zero, count: characters.count)
        CTFontGetGlyphsForCharacters(font as CTFont, &characters, &glyphs, characters.count)
        CTFontGetBoundingRectsForGlyphs(font as CTFont, .horizontal, glyphs, &rects, glyphs.count)
        let lineHeight = font.ascender - font.descender + font.leading
        guard lineHeight > 0,
              let inkTop = rects.map(\.maxY).max(),
              let inkBottom = rects.map(\.minY).min(),
              inkTop > inkBottom
        else { return (0.2, 0.63) }
        return ((font.ascender - inkTop) / lineHeight, (inkTop - inkBottom) / lineHeight)
    }()

    /// The band's height in points for the hero's full-size digits: how far
    /// a digit travels to leave it.
    static let fullSizeTravel: Double = {
        let font = heroFont()
        return Double((font.ascender - font.descender + font.leading) * digitBand.height)
    }()
}

private extension Array where Element == CGFloat {
    /// The element at `index`, or the middle when there is none.
    subscript(safe index: Int) -> CGFloat {
        indices.contains(index) ? self[index] : 0.5
    }
}
