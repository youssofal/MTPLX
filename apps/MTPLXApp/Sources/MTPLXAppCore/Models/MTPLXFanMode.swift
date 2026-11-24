import Foundation

public enum MTPLXFanMode: String, CaseIterable, Codable, Equatable, Sendable {
    case `default`
    case smart
    case max

    public var title: String {
        switch self {
        case .default: return tr("Default")
        case .smart: return tr("Smart")
        case .max: return tr("Max")
        }
    }

    public var help: String {
        switch self {
        case .default:
            return tr("Apple's fan curve owns idle and generation.")
        case .smart:
            return tr("Fans stay quiet on short turns, ramp gently as heat builds, then return to auto.")
        case .max:
            return tr("Fans pin to verified max until you change mode or stop.")
        }
    }

    public static func normalized(_ raw: String?) -> MTPLXFanMode {
        switch (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "auto", "apple", "apple-default", "system", "off", "default":
            return .default
        case "max", "maximum", "performance", "sustained-max":
            return .max
        case "smart", "request", "request-scoped":
            return .smart
        default:
            return .smart
        }
    }
}
