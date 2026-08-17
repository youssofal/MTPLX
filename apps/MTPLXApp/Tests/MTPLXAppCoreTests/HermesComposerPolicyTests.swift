import XCTest
@testable import MTPLXAppCore

final class HermesComposerPolicyTests: XCTestCase {
    func testSelectedProfileAcceptsFirstPromptBeforeSessionExists() {
        XCTAssertTrue(HermesComposerPolicy.acceptsInput(
            gatewayReady: true,
            hasPendingRequest: false,
            activeSessionWritable: false,
            hasActiveSession: false,
            hasAvailableProfile: true
        ))
    }

    func testExistingWritableSessionAcceptsInput() {
        XCTAssertTrue(HermesComposerPolicy.acceptsInput(
            gatewayReady: true,
            hasPendingRequest: false,
            activeSessionWritable: true,
            hasActiveSession: true,
            hasAvailableProfile: true
        ))
    }

    func testExistingReadOnlySessionCannotUseFreshSessionFallback() {
        XCTAssertFalse(HermesComposerPolicy.acceptsInput(
            gatewayReady: true,
            hasPendingRequest: false,
            activeSessionWritable: false,
            hasActiveSession: true,
            hasAvailableProfile: true
        ))
    }

    func testGatewayPendingRequestAndMissingProfileRemainDisabled() {
        XCTAssertFalse(HermesComposerPolicy.acceptsInput(
            gatewayReady: false,
            hasPendingRequest: false,
            activeSessionWritable: false,
            hasActiveSession: false,
            hasAvailableProfile: true
        ))
        XCTAssertFalse(HermesComposerPolicy.acceptsInput(
            gatewayReady: true,
            hasPendingRequest: true,
            activeSessionWritable: true,
            hasActiveSession: true,
            hasAvailableProfile: true
        ))
        XCTAssertFalse(HermesComposerPolicy.acceptsInput(
            gatewayReady: true,
            hasPendingRequest: false,
            activeSessionWritable: false,
            hasActiveSession: false,
            hasAvailableProfile: false
        ))
    }
}
