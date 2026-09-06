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

    func testHeroGaugeKeepsDecodeFaceForShortSuffixPrefills() throws {
        // A Hermes tool turn: the started frame claims the whole prompt before
        // the restore runs; the session's known prefix says 625 tokens are new.
        let started = PrefillState(phase: "started", tokensDone: 0, tokensTotal: 15953,
            cachedTokens: 0, newPrefillTokens: 15953)
        XCTAssertFalse(HeroPrefillGate.showsPrefill(prefill: started, promptTokens: 15953, sessionPrefixLen: 15328))
        // The same frame for a cold prompt, or for an 8,715-token file read, morphs at once.
        XCTAssertTrue(HeroPrefillGate.showsPrefill(prefill: started, promptTokens: 15953, sessionPrefixLen: nil))
        XCTAssertTrue(HeroPrefillGate.showsPrefill(prefill: started, promptTokens: 24752, sessionPrefixLen: 16037))
        // An edited history shorter than the session's prefix is unknown work: shown.
        XCTAssertTrue(HeroPrefillGate.showsPrefill(prefill: started, promptTokens: 9000, sessionPrefixLen: 15328))
        // Once a chunk or the completion reports the real split, the frame decides.
        var chunk = PrefillState(phase: "chunk", tokensDone: 35857, tokensTotal: 35857,
            cachedTokens: 35600, newPrefillTokens: 257)
        XCTAssertFalse(HeroPrefillGate.showsPrefill(prefill: chunk, promptTokens: 35857, sessionPrefixLen: nil))
        chunk.newPrefillTokens = nil
        XCTAssertEqual(HeroPrefillGate.newWorkTokens(prefill: chunk, promptTokens: nil, sessionPrefixLen: nil), 257)
        chunk.phase = "completed"
        XCTAssertFalse(HeroPrefillGate.showsPrefill(prefill: chunk, promptTokens: 35857, sessionPrefixLen: nil))
    }
}
