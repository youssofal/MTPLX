import Foundation
import MTPLXAppCore

// MARK: - Number / Rate formatters

enum Format {
    /// Decode/prefill TPS (tokens per second). Returns an em-dash for
    /// nil/non-finite values. Pair with the "TPS" unit suffix in UI.
    static func tps(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        if value >= 100 {
            // Prefill reads run into the thousands: group them like every
            // token count beside them ("1,560", not "1560").
            let whole = value.rounded()
            return whole >= 1_000 ? integer(whole) : String(format: "%.0f", whole)
        }
        if value >= 10 { return String(format: "%.1f", value) }
        return String(format: "%.2f", value)
    }

    static func integer(_ value: Int?) -> String {
        guard let value else { return "—" }
        return value.formatted(.number.grouping(.automatic))
    }

    static func integer(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return Int(value).formatted(.number.grouping(.automatic))
    }

    /// Compact token count for narrow tiles. Below 1000 returns the raw
    /// number, then "12.3K" / "262K" / "1.2M" / "10M". Trims the
    /// decimal once the magnitude is large enough that the extra digit
    /// adds noise. Designed for the Context tile in the LiveTab where
    /// "12,345 / 262,144" overflows but "12.3K / 262K" fits cleanly.
    static func compactTokens(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "—" }
        if value < 1_000 { return String(value) }
        let thousands = Double(value) / 1_000.0
        if value < 10_000 { return tr("%.1fK", thousands) }
        if value < 1_000_000 { return tr("%.0fK", thousands) }
        let millions = Double(value) / 1_000_000.0
        if value < 10_000_000 { return tr("%.1fM", millions) }
        if value < 1_000_000_000 { return tr("%.0fM", millions) }
        let billions = Double(value) / 1_000_000_000.0
        return tr("%.1fB", billions)
    }

    /// Percentage from a 0…1 value. Returns "—" for nil.
    static func percent(_ value: Double?, fractionDigits: Int = 1) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.\(fractionDigits)f%%", locale: L10n.language.locale, value * 100)
    }

    static func ratio(_ value: Double?, fractionDigits: Int = 2) -> String {
        guard let value, value.isFinite else { return "—" }
        return String(format: "%.\(fractionDigits)f×", locale: L10n.language.locale, value)
    }

    /// Byte count with binary units. Mirrors macOS Finder style.
    static func bytes(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "—" }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        return formatter.string(fromByteCount: Int64(value))
    }

    /// Compact GB-only with one decimal: e.g. "27.5 GB".
    static func gigabytes(_ value: Int?) -> String {
        guard let value, value >= 0 else { return "—" }
        let gb = Double(value) / 1024.0 / 1024.0 / 1024.0
        return tr("%.1f GB", gb)
    }

    /// Duration in seconds, picks the most readable unit.
    static func duration(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
        if seconds < 1 {
            return tr("%.0f ms", seconds * 1000)
        }
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        }
        if seconds < 3600 {
            let m = Int(seconds) / 60
            let s = Int(seconds) % 60
            return tr("%dm %02ds", m, s)
        }
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        return tr("%dh %02dm", h, m)
    }

    static func milliseconds(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
        return tr("%.1f ms", seconds * 1000)
    }

    /// "12:34:56" / "1:23" elapsed clock.
    static func clock(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "—" }
        let total = Int(seconds)
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, s) }
        return String(format: "%d:%02d", m, s)
    }

    /// Short request-id rendering: first six characters with leading "#".
    static func shortId(_ id: String?) -> String {
        guard let id, !id.isEmpty else { return "—" }
        let cleaned = id.hasPrefix("req_") ? String(id.dropFirst(4)) : id
        return "#" + cleaned.prefix(8)
    }

    /// "now", "3s ago", "12m ago". Used for last-access / age timestamps.
    static func relative(from age: Double?) -> String {
        guard let age, age.isFinite, age >= 0 else { return "—" }
        if age < 2 { return "now" }
        if age < 60 { return tr("%llds ago", Int(age)) }
        if age < 3600 { return tr("%lldm ago", Int(age / 60)) }
        return tr("%lldh ago", Int(age / 3600))
    }

    /// Multiplier strings like "2.54× vs AR" for headline copy.
    static func multiplier(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        return tr("%.2f×", value)
    }
}
