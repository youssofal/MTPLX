import Foundation

/// `/health` `thermal_pressure`: macOS's own verdict on how hard the Mac may
/// run (nominal, moderate, heavy, trapping, sleeping; the levels behind
/// `ProcessInfo.thermalState`), which the engine samples every two seconds
/// (`mtplx/thermal_pressure.py`). It is not the fan state: the fans can be at
/// full speed while macOS slows the GPU. Read leniently, like the SSD tier's
/// stats, so a field of an unexpected type never makes `/health`
/// undecodable.
public struct ThermalPressureHealth: Codable, Equatable, Sendable {
    public var values: [String: JSONValue]

    public init(values: [String: JSONValue] = [:]) {
        self.values = values
    }

    public init(from decoder: Decoder) throws {
        values = (try? decoder.singleValueContainer().decode([String: JSONValue].self)) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }

    /// "nominal", "moderate", "heavy", "trapping", "sleeping", or
    /// "unavailable" where the engine cannot read it.
    public var level: String? { values["level"]?.stringValue }
}

/// The Live tab's heat notice. Under heavy pressure and above, macOS slows
/// the GPU and decode speed drops with the engine doing the same work, so
/// the notice says why instead of leaving a falling gauge unexplained.
/// Nominal and moderate pressure leave the GPU's speed alone and show
/// nothing.
public struct ThermalPressureNotice: Equatable, Sendable {
    public var level: String

    public init(level: String) {
        self.level = level
    }

    /// nil below heavy pressure, or when the engine reports no level.
    public static func from(_ health: ThermalPressureHealth?) -> ThermalPressureNotice? {
        guard let raw = health?.level else { return nil }
        let level = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch level {
        case "heavy", "trapping", "sleeping":
            return ThermalPressureNotice(level: level)
        default:
            return nil
        }
    }

    public var message: String {
        tr("macOS is slowing the GPU because the Mac is hot. Speed returns as it cools.")
    }
}
