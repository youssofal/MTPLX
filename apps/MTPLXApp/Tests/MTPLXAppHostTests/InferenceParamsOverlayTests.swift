import XCTest

@testable import MTPLXAppCore
@testable import MTPLXAppHost

final class InferenceParamsOverlayTests: XCTestCase {
    func testDeepSeekDescriptorMaxSurvivesPickerMapping() {
        let policy = ReasoningPolicy(
            supported: true,
            modes: ["auto", "on", "off"],
            defaultMode: "auto",
            effortLevels: [" low ", "high", "max", "MAX", "unsupported"],
            defaultEffort: "low"
        )

        XCTAssertEqual(
            InferenceParamsOverlay.reasoningEffortLevels(for: policy),
            ["low", "high", "max"]
        )
        XCTAssertEqual(
            InferenceParamsOverlay.normalizedReasoningEffort(" MAX ", policy: policy),
            "max"
        )
        XCTAssertEqual(
            InferenceParamsOverlay.normalizedReasoningEffort("xhigh", policy: policy),
            "low"
        )
    }
}
