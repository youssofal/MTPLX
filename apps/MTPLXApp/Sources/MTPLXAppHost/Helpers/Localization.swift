import Foundation

extension String {
    /// Looks up a runtime-provided UI label in the app's localization table.
    /// SwiftUI localizes string literals automatically, but labels passed
    /// through reusable components arrive as plain `String` values.
    var mtplxLocalized: String {
        Bundle.main.localizedString(forKey: self, value: self, table: nil)
    }
}
