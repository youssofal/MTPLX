import SwiftUI

// MARK: - MicroHeader
//
// Tiny tracked uppercase header for inline sections inside a `Card`.
// Reads as the smallest typographic anchor in the type scale.

struct MicroHeader: View {
    let text: String
    var systemImage: String? = nil

    init(_ text: String, systemImage: String? = nil) {
        self.text = text
        self.systemImage = systemImage
    }

    var body: some View {
        HStack(spacing: 6) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Text(text.uppercased().mtplxLocalized)
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.5)
                .foregroundStyle(Brand.typeTertiary)
        }
    }
}
