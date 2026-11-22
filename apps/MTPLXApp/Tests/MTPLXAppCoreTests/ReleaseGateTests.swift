import XCTest

@testable import MTPLXAppCore

/// The founder, 22 November 2026: MTPLX 2.13's Start button offers what 2.12.3 offered and nothing new; Claude
/// Code, Codex and DeepSeek Harness arrive in MTPLX 3 behind one switch (`LaunchTarget.v3AgentClients`, with the
/// CLI's `V3_AGENT_CLIENTS` in mtplx/release_gates.py). Their launch code keeps its own tests (the lane parity, the
/// client plans and the command builder enumerate every case); these say what a user can pick.
final class ReleaseGateTests: XCTestCase {
    private let v3Clients: [LaunchTarget] = [.claudeCode, .codex, .deepseekHarness]

    func testTheReleaseStartListIsWhat2123OfferedAndNothingNew() {
        XCTAssertFalse(LaunchTarget.v3AgentClients)
        XCTAssertEqual(LaunchTarget.offered(), [.chat, .openWebUI, .pi, .openCode, .hermes, .benchmark, .other])
        for target in v3Clients {
            XCTAssertFalse(target.isOffered(), target.rawValue)
        }
    }

    func testTheSwitchOffersEveryTargetAgain() {
        XCTAssertEqual(LaunchTarget.offered(v3AgentClients: true), LaunchTarget.allCases)
        for target in v3Clients {
            XCTAssertTrue(target.isOffered(v3AgentClients: true), target.rawValue)
        }
    }
}
