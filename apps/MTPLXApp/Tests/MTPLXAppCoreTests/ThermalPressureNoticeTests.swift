import XCTest
import Foundation
@testable import MTPLXAppCore

/// The Live tab's heat notice reads `/health` `thermal_pressure`, the level
/// the engine samples from macOS. Heavy and worse slow the GPU and show the
/// notice; nominal and moderate do not; a missing or odd field never breaks
/// `/health` decoding (the watchdog would count a live daemon as gone).
final class ThermalPressureNoticeTests: XCTestCase {
    private func health(thermalPressure: String?) throws -> HealthPayload {
        let field = thermalPressure.map { #", "thermal_pressure": \#($0)"# } ?? ""
        let json = """
        {"ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
         "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
         "context_window": 262144, "active_requests": 1, "reasoning_parser": "qwen3"\(field)}
        """
        return try JSONDecoder().decode(HealthPayload.self, from: Data(json.utf8))
    }

    private func level(_ name: String) -> String {
        #"{"level": "\#(name)", "available": true, "sampled_age_s": 0.4, "interval_s": 2.0, "read_errors": 0, "last_error": null}"#
    }

    func testHeavyPressureAndWorseShowTheNotice() throws {
        for name in ["heavy", "trapping", "sleeping", "Heavy "] {
            let notice = ThermalPressureNotice.from(try health(thermalPressure: level(name)).thermalPressure)
            XCTAssertNotNil(notice, name)
        }
        let notice = try XCTUnwrap(
            ThermalPressureNotice.from(try health(thermalPressure: level("heavy")).thermalPressure)
        )
        L10n.activate(.english)
        XCTAssertEqual(
            notice.message,
            "macOS is slowing the GPU because the Mac is hot. Speed returns as it cools."
        )
    }

    func testNominalModerateAndUnavailableShowNothing() throws {
        for name in ["nominal", "moderate", "unavailable", "level_7"] {
            XCTAssertNil(ThermalPressureNotice.from(try health(thermalPressure: level(name)).thermalPressure), name)
        }
        XCTAssertNil(ThermalPressureNotice.from(try health(thermalPressure: nil).thermalPressure))
        XCTAssertNil(ThermalPressureNotice.from(try health(thermalPressure: "null").thermalPressure))
    }

    func testAnUnexpectedShapeNeverBreaksHealthDecoding() throws {
        for odd in [#"{"level": 2}"#, #""heavy""#, "[1, 2]", "7"] {
            let payload = try health(thermalPressure: odd)
            XCTAssertTrue(payload.ok, odd)
            XCTAssertNil(ThermalPressureNotice.from(payload.thermalPressure), odd)
        }
    }

    func testTheNoticeIsTranslated() {
        let key = "macOS is slowing the GPU because the Mac is hot. Speed returns as it cools."
        for language in AppLanguage.allCases where language != .english {
            XCTAssertNotEqual(L10n.string(key, language: language), key, language.code)
        }
    }
}
