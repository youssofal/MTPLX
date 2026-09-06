import XCTest
@testable import MTPLXAppCore
@testable import MTPLXAppHost

final class PrefillDisplayTests: XCTestCase {
    func testLiveDialUsesMeasuredChunkInsteadOfSetupDilutedRate() throws {
        // Recorded Hermes first chunk: 1.27 seconds of model work after
        // earlier setup made the cumulative wall reading only 296 TPS.
        var state = PrefillState(phase: "chunk", tokensDone: 2048,
            tokensTotal: 15143, prefillTokS: 295.996,
            cumulativePrefillTokS: 295.996, livePrefillTokS: 1613.065,
            chunkSize: 2048, chunkElapsedS: 1.2696327920129988)
        XCTAssertEqual(try XCTUnwrap(state.sanePrefillTokS()), 1613.0648, accuracy: 0.001)
        // Cached tokens never become measured work; the axis is not a cap.
        state.cachedTokens = 100_000
        state.tokensDone = 102_048
        state.chunkElapsedS = 1
        XCTAssertEqual(state.sanePrefillTokS(), 2048)
    }

    func testAverageAndPeakCardUseTheSameChunkWork() throws {
        let summary = try JSONDecoder().decode(PrefillRateSummary.self, from: Data("""
            {"tokens": 6144, "compute_time_s": 4.0, "peak_tok_s": 2048,
             "samples": 2, "capacity": 100}
            """.utf8))
        XCTAssertEqual(summary.averageTokS, 1536)
        XCTAssertEqual(summary.peakTokS, 2048)
        XCTAssertEqual(Format.tps(summary.averageTokS), "1536")
    }
}
