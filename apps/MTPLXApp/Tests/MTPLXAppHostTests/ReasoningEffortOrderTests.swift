import XCTest
import MTPLXAppCore

@testable import MTPLXAppHost

/// The Model controls effort picker reads lowest on the left to highest on
/// the right for every family, whatever order the family declares.
final class ReasoningEffortOrderTests: XCTestCase {
    func testQwenFamiliesReadLowToHigh() {
        // Qwen 3.8, Flash-Next and Bonsai 2 declare xhigh first.
        XCTAssertEqual(
            InferenceParamsOverlay.effortLevelsLowToHigh(["xhigh", "medium", "low"]),
            ["low", "medium", "xhigh"]
        )
        XCTAssertEqual(
            InferenceParamsOverlay.effortLevelsLowToHigh(["xhigh", "medium"]),
            ["medium", "xhigh"]
        )
    }

    func testFamiliesAlreadyLowFirstKeepTheirOrder() {
        // Inkling (after the picker drops none, minimal and max) and Step.
        XCTAssertEqual(
            InferenceParamsOverlay.effortLevelsLowToHigh(["low", "medium", "high", "xhigh"]),
            ["low", "medium", "high", "xhigh"]
        )
        XCTAssertEqual(
            InferenceParamsOverlay.effortLevelsLowToHigh(["low", "medium", "high"]),
            ["low", "medium", "high"]
        )
    }

    func testDeclaredFamilyListsAreNotReordered() {
        // The order shown changes; what each family declares, and therefore
        // what the server, OpenCode and Pi receive, does not.
        XCTAssertEqual(
            OpenCodeIntegration.reasoningEffortLevels(forModelID: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"),
            ["xhigh", "medium", "low"]
        )
        XCTAssertEqual(
            OpenCodeIntegration.reasoningEffortLevels(forModelID: "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"),
            ["xhigh", "medium", "low"]
        )
    }
}
