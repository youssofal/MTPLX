import XCTest
@testable import MTPLXAppCore

final class OnboardingFeatureStateTests: XCTestCase {
    // MARK: goNext / goBack walk the canonical case order

    func testGoNextWalksThroughEveryStep() {
        var s = OnboardingFeatureState()
        XCTAssertEqual(s.step, .welcome)
        s.goNext(); XCTAssertEqual(s.step, .hardwareScan)
        s.goNext(); XCTAssertEqual(s.step, .modelPick)
        s.goNext(); XCTAssertEqual(s.step, .runtimeSetup)
        s.goNext(); XCTAssertEqual(s.step, .download)
        s.goNext(); XCTAssertEqual(s.step, .tune)
    }

    func testGoNextOnLastStepIsNoOp() {
        var s = OnboardingFeatureState(step: .tune)
        s.goNext()
        XCTAssertEqual(s.step, .tune)
    }

    func testGoBackWalksReverseAndStopsAtWelcome() {
        var s = OnboardingFeatureState(step: .tune)
        s.goBack(); XCTAssertEqual(s.step, .download)
        s.goBack(); XCTAssertEqual(s.step, .runtimeSetup)
        s.goBack(); XCTAssertEqual(s.step, .modelPick)
        s.goBack(); XCTAssertEqual(s.step, .hardwareScan)
        s.goBack(); XCTAssertEqual(s.step, .welcome)
        s.goBack(); XCTAssertEqual(s.step, .welcome) // clamped
    }

    func testRuntimeSetupSitsBetweenModelPickAndDownload() {
        let all = OnboardingStep.allCases
        let pick = all.firstIndex(of: .modelPick)
        let setup = all.firstIndex(of: .runtimeSetup)
        let download = all.firstIndex(of: .download)
        XCTAssertNotNil(pick)
        XCTAssertNotNil(setup)
        XCTAssertNotNil(download)
        XCTAssertEqual(setup, pick.map { $0 + 1 }, "Runtime setup must directly follow model pick")
        XCTAssertEqual(download, setup.map { $0 + 1 }, "Download (which shells mtplx pull) must follow runtime setup")
    }

    func testRuntimeSetupIsServiceGated() {
        XCTAssertFalse(
            OnboardingFeatureState(step: .runtimeSetup).canAdvance,
            "Runtime setup advances via orchestrator completion, not user input"
        )
    }

    // MARK: canAdvance reflects the user-input gates

    func testWelcomeCanAlwaysAdvance() {
        XCTAssertTrue(OnboardingFeatureState(step: .welcome).canAdvance)
    }

    func testHardwareScanRequiresDetectedHardware() {
        var s = OnboardingFeatureState(step: .hardwareScan)
        XCTAssertFalse(s.canAdvance)
        s.hardware = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 36 * 1_073_741_824
        )
        XCTAssertTrue(s.canAdvance)
    }

    func testModelPickStartsWithoutSelection() {
        let s = OnboardingFeatureState(step: .modelPick)
        XCTAssertEqual(s.pick, .none)
        XCTAssertFalse(s.canAdvance)
        XCTAssertNil(s.resolvedModel)
        XCTAssertNil(s.resolvedRepoID)
    }

    func testModelPickCuratedAlwaysResolves() {
        var s = OnboardingFeatureState(step: .modelPick, pick: .curatedSpeedV2)
        XCTAssertTrue(s.canAdvance, "Curated Speed V2 always resolves to a catalog entry")
        s.pick = .curatedSpeed
        XCTAssertTrue(s.canAdvance, "Curated Speed always resolves to a catalog entry")
        s.pick = .curatedQwen35FourBit
        XCTAssertTrue(s.canAdvance, "Curated Qwen 4B always resolves to a catalog entry")
        s.pick = .curatedQwen35NineBSpeed
        XCTAssertTrue(s.canAdvance, "Curated Qwen 9B Speed always resolves to a catalog entry")
        s.pick = .curatedQwen35BSpeed
        XCTAssertTrue(s.canAdvance, "Curated 35B Speed always resolves to a catalog entry")
        s.pick = .curatedQwen35BBalance
        XCTAssertTrue(s.canAdvance, "Curated 35B Balance always resolves to a catalog entry")
        s.pick = .curatedQuality
        XCTAssertTrue(s.canAdvance, "Curated Quality always resolves to a catalog entry")
        s.pick = .curatedGemmaSpeed
        XCTAssertTrue(s.canAdvance, "Curated Gemma Speed always resolves to a catalog entry")
        s.pick = .curatedStepFlash
        XCTAssertFalse(s.canAdvance, "StepFun is held out of the release catalog")
    }

    func testModelPickOtherRequiresSuccessfulProbe() {
        var s = OnboardingFeatureState(
            step: .modelPick,
            pick: .other(hfRepo: "Foo/Bar")
        )
        XCTAssertFalse(s.canAdvance, "No probe yet")
        s.otherProbe = OtherModelProbe(verdict: .probeFailed, hfRepo: "Foo/Bar", message: "404")
        XCTAssertFalse(s.canAdvance, "Failed probe blocks advance")
        s.otherProbe = OtherModelProbe(verdict: .ready, hfRepo: "Foo/Bar", message: "OK")
        XCTAssertTrue(s.canAdvance, "Ready probe allows advance")
    }

    func testModelPickNoMTPRequiresExplicitAcknowledgement() {
        var s = OnboardingFeatureState(
            step: .modelPick,
            pick: .other(hfRepo: "Foo/Bar"),
            otherProbe: OtherModelProbe(verdict: .noMTP, hfRepo: "Foo/Bar", message: "No MTP")
        )
        XCTAssertFalse(s.canAdvance, "noMTP blocks until acknowledged")
        s.hasAcknowledgedOtherWarning = true
        XCTAssertTrue(s.canAdvance, "Acknowledged noMTP allows advance")
    }

    func testModelPickLocalRequiresReadyProbe() {
        var s = OnboardingFeatureState(
            step: .modelPick,
            pick: .local(path: "/models/qwen")
        )
        XCTAssertFalse(s.canAdvance, "No local probe yet")
        s.localProbe = LocalModelProbe(
            verdict: .incomplete,
            path: "/models/qwen",
            message: "Missing sidecar"
        )
        XCTAssertFalse(s.canAdvance, "Incomplete local folders block advance")
        s.localProbe = LocalModelProbe(
            verdict: .ready,
            path: "/models/qwen",
            message: "Ready"
        )
        XCTAssertTrue(s.canAdvance, "Complete MTPLX local folders allow advance")
    }

    // MARK: select(_:) wipes stale probe + acknowledgement

    func testSelectChoiceClearsProbeAndAcknowledgement() {
        var s = OnboardingFeatureState(
            step: .modelPick,
            pick: .other(hfRepo: "Foo/Bar"),
            otherProbe: OtherModelProbe(verdict: .ready, hfRepo: "Foo/Bar", message: "OK"),
            localProbe: LocalModelProbe(verdict: .ready, path: "/models/qwen", message: "Ready"),
            hasAcknowledgedOtherWarning: true
        )
        s.select(.curatedSpeed)
        XCTAssertEqual(s.pick, .curatedSpeed)
        XCTAssertNil(s.otherProbe)
        XCTAssertNil(s.localProbe)
        XCTAssertFalse(s.hasAcknowledgedOtherWarning)
    }

    func testRecordProbeWipesAcknowledgement() {
        var s = OnboardingFeatureState(
            step: .modelPick,
            pick: .other(hfRepo: "Foo/Bar"),
            otherProbe: OtherModelProbe(verdict: .noMTP, hfRepo: "Foo/Bar", message: "No MTP"),
            hasAcknowledgedOtherWarning: true
        )
        s.record(OtherModelProbe(verdict: .ready, hfRepo: "Foo/Bar", message: "OK"))
        XCTAssertEqual(s.otherProbe?.verdict, .ready)
        XCTAssertFalse(
            s.hasAcknowledgedOtherWarning,
            "A new probe must force a fresh acknowledgement"
        )
    }

    // MARK: resolvedModel applies M1/M2 FP16 routing

    func testResolvedModelRoutesSpeedToFP16OnLegacyApple() {
        let m1 = DetectedHardware(
            chipName: "Apple M1 Pro",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "optimized-speed-fp16")
    }

    func testResolvedModelKeepsSpeedQ4OnModernApple() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 36 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "optimized-speed")
    }

    func testResolvedModelKeepsSpeedV2DistinctFromV1() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 64 * 1_073_741_824
        )
        let v2 = OnboardingFeatureState(hardware: m5, pick: .curatedSpeedV2)
        let v1 = OnboardingFeatureState(hardware: m5, pick: .curatedSpeed)

        XCTAssertEqual(v2.resolvedModel?.id, "optimized-speed-v2")
        XCTAssertEqual(v1.resolvedModel?.id, "optimized-speed")
    }

    func testResolvedModelForQualityRoutesToFP16OnLegacyApple() {
        // Until 2026-07-07 quality never swapped because no Quality-FP16
        // artifact existed. It exists now (2.0.1) and is measured (2.5x
        // over true AR under turbo), so the M1/M2 quality pick resolves
        // the FP16 sibling exactly like the speed lane.
        let m1 = DetectedHardware(
            chipName: "Apple M1",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 8 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedQuality)
        XCTAssertEqual(s.resolvedModel?.id, "optimized-quality-fp16")
    }

    func testResolvedModelForQualityKeepsBF16OnModernApple() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 64 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedQuality)
        XCTAssertEqual(s.resolvedModel?.id, "optimized-quality")
    }

    func testResolvedModelForQwen35FourBitUsesSmallCatalogEntry() {
        let m1 = DetectedHardware(
            chipName: "Apple M1",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedQwen35FourBit)
        XCTAssertEqual(s.resolvedModel?.id, "qwen35-4b-optimized-speed")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_5")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen359BRoutesToFP16OnLegacyApple() {
        let m1 = DetectedHardware(
            chipName: "Apple M1 Pro",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedQwen35NineBSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "qwen35-9b-optimized-speed-fp16")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed-FP16")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_5")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen359BKeepsBaseOnModernApple() {
        let m5 = DetectedHardware(
            chipName: "Apple M5",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 16 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedQwen35NineBSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "qwen35-9b-optimized-speed")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_5")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen35BUsesQwenCatalogAndRunsTune() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedQwen35BSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "qwen36-35b-a3b-optimized-speed")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_6")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen35BUsesFP16OnLegacyApple() {
        let m1 = DetectedHardware(
            chipName: "Apple M1 Max",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 64 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedQwen35BSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "qwen36-35b-a3b-optimized-speed-fp16")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_6")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen35BBalanceUsesBaseOnModernApple() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedQwen35BBalance)
        XCTAssertEqual(s.resolvedModel?.id, "qwen36-35b-a3b-optimized-balance")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_6")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForQwen35BBalanceUsesFP16OnLegacyApple() {
        let m1 = DetectedHardware(
            chipName: "Apple M1 Max",
            appleSiliconGeneration: "m1",
            unifiedMemoryBytes: 64 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m1, pick: .curatedQwen35BBalance)
        XCTAssertEqual(s.resolvedModel?.id, "qwen36-35b-a3b-optimized-balance-fp16")
        XCTAssertEqual(s.resolvedRepoID, "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance-FP16")
        XCTAssertEqual(s.resolvedModelFamily, "qwen3_6")
        XCTAssertTrue(s.supportsTune)
    }

    func testResolvedModelForGemmaUsesGemmaCatalogAndTunesBlocks() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )
        let s = OnboardingFeatureState(hardware: m5, pick: .curatedGemmaSpeed)
        XCTAssertEqual(s.resolvedModel?.id, "gemma4-optimized-speed")
        XCTAssertEqual(s.resolvedModelFamily, "gemma4")
        XCTAssertTrue(s.supportsTune)
        XCTAssertEqual(s.tuneCandidates, [
            .ar,
            .block2,
            .block3,
            .block4,
            .block5,
            .block6,
            .block7,
            .block8,
        ])
    }

    func testResolvedModelForStepIsHeldOutOfReleaseCatalog() {
        let m5 = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )
        let s = OnboardingFeatureState(
            step: .modelPick,
            hardware: m5,
            pick: .curatedStepFlash
        )
        XCTAssertNil(s.resolvedModel)
        XCTAssertNil(s.resolvedRepoID)
        XCTAssertEqual(s.resolvedModelFamily, "unknown")
        XCTAssertFalse(s.supportsTune)
        XCTAssertEqual(s.tuneCandidates, [])
        XCTAssertFalse(s.canAdvance)
    }

    @MainActor
    func testInstalledModelBypassesDownloadDiskFeasibility() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-installed-feasibility-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try "{}".write(to: root.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        try Data([0]).write(to: root.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: root.appendingPathComponent("model.safetensors"))

        let hugeInstalledModel = MTPLXModelOption(
            id: "huge-installed",
            displayName: "Huge Installed",
            shortName: "Huge Installed",
            detail: "Installed model with an intentionally huge download footprint.",
            hfModelID: "Example/HugeInstalled",
            localCandidates: [root.path],
            sizeBytes: 10_000_000_000_000,
            peakMemoryGiB: 1
        )
        let hardware = DetectedHardware(
            chipName: "Apple M5 Max",
            appleSiliconGeneration: "m5",
            unifiedMemoryBytes: 128 * 1_073_741_824
        )
        let orchestrator = OnboardingOrchestrator(
            initialState: OnboardingFeatureState(hardware: hardware)
        )

        XCTAssertEqual(orchestrator.evaluateFeasibility(for: hugeInstalledModel), .recommended)
    }

    func testResolvedRepoIDForOtherReturnsTrimmedInput() {
        let s = OnboardingFeatureState(pick: .other(hfRepo: "  Foo/Bar  "))
        XCTAssertEqual(s.resolvedRepoID, "Foo/Bar")
        XCTAssertNil(s.resolvedModel, "Other never resolves to a catalog entry")
    }

    func testResolvedRepoIDForOtherEmptyInputIsNil() {
        let s = OnboardingFeatureState(pick: .other(hfRepo: "   "))
        XCTAssertNil(s.resolvedRepoID)
    }

    func testResolvedRepoIDForLocalReturnsTrimmedInput() {
        let s = OnboardingFeatureState(pick: .local(path: "  /models/qwen  "))
        XCTAssertEqual(s.resolvedRepoID, "/models/qwen")
        XCTAssertNil(s.resolvedModel, "Local folders never resolve to a catalog entry")
    }

    @MainActor
    func testOrchestratorSkipsDownloadForValidatedLocalFolder() {
        let orchestrator = OnboardingOrchestrator(
            initialState: OnboardingFeatureState(
                step: .modelPick,
                pick: .local(path: "/models/qwen"),
                localProbe: LocalModelProbe(
                    verdict: .ready,
                    path: "/models/qwen",
                    message: "Ready"
                )
            )
        )

        XCTAssertTrue(orchestrator.state.canAdvance)
        orchestrator.goNext()
        XCTAssertEqual(
            orchestrator.state.step, .runtimeSetup,
            "Local folders still need the runtime set up before tuning"
        )
        orchestrator.goNext()
        XCTAssertEqual(
            orchestrator.state.step, .tune,
            "The download step is skipped — the bytes are already on disk"
        )
        orchestrator.goBack()
        XCTAssertEqual(
            orchestrator.state.step, .runtimeSetup,
            "Back from tune mirrors the forward jump for local picks"
        )
    }

    // MARK: HuggingFaceProbe

    func testHuggingFaceProbeDoesNotAuthGatePublicConfig() async {
        let calls = ProbeCallRecorder()
        let probe = HuggingFaceProbe { url, method in
            await calls.record(url: url.absoluteString, method: method)
            if url.absoluteString.contains("/resolve/main/config.json") {
                return (200, #"{"model_type":"qwen3"}"#.data(using: .utf8)!)
            }
            return (401, Data())
        }

        let result = await probe.probe(repo: "Qwen/Qwen3-0.6B")
        let recorded = await calls.snapshot()

        XCTAssertEqual(result.verdict, .noMTP)
        XCTAssertEqual(recorded.count, 1)
        XCTAssertEqual(recorded.first?.method, "GET")
        XCTAssertTrue(recorded.first?.url.contains("/resolve/main/config.json") == true)
    }

    func testHuggingFaceProbeReportsAuthOnlyWhenConfigIsPrivateOrGated() async {
        let calls = ProbeCallRecorder()
        let probe = HuggingFaceProbe { url, method in
            await calls.record(url: url.absoluteString, method: method)
            return (403, Data())
        }

        let result = await probe.probe(repo: "Private/Model")
        let recorded = await calls.snapshot()

        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "http_403")
        XCTAssertEqual(recorded.count, 1)
        XCTAssertTrue(result.message.contains("Public Hugging Face models download without a login."))
    }

    func testHuggingFaceProbeDetectsPublishedMTPSidecar() async {
        let calls = ProbeCallRecorder()
        let probe = HuggingFaceProbe { url, method in
            await calls.record(url: url.absoluteString, method: method)
            if url.absoluteString.contains("/resolve/main/config.json") {
                return (
                    200,
                    #"{"text_config":{"mtp_num_hidden_layers":1}}"#.data(using: .utf8)!
                )
            }
            if url.absoluteString.contains("/tree/main") {
                return (200, #"[{"path":"mtp.safetensors"}]"#.data(using: .utf8)!)
            }
            return (404, Data())
        }

        let result = await probe.probe(repo: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
        let recorded = await calls.snapshot()

        XCTAssertEqual(result.verdict, .ready)
        XCTAssertEqual(recorded.map(\.method), ["GET", "GET"])
    }
}

private struct ProbeCall: Equatable, Sendable {
    var url: String
    var method: String
}

private actor ProbeCallRecorder {
    private var calls: [ProbeCall] = []

    func record(url: String, method: String) {
        calls.append(ProbeCall(url: url, method: method))
    }

    func snapshot() -> [ProbeCall] {
        calls
    }
}
