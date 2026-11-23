import XCTest
@testable import MTPLXAppCore

final class ForgeFeatureStateTests: XCTestCase {
    // MARK: goNext / goBack walk the canonical case order

    func testGoNextWalksThroughEveryStep() {
        var s = ForgeFeatureState()
        XCTAssertEqual(s.step, .source)
        s.goNext(); XCTAssertEqual(s.step, .plan)
        s.goNext(); XCTAssertEqual(s.step, .convert)
        s.goNext(); XCTAssertEqual(s.step, .calibrate)
        s.goNext(); XCTAssertEqual(s.step, .verify)
        s.goNext(); XCTAssertEqual(s.step, .brand)
        s.goNext(); XCTAssertEqual(s.step, .registered)
        s.goNext(); XCTAssertEqual(s.step, .publishing)
    }

    func testGoNextOnLastStepIsNoOp() {
        var s = ForgeFeatureState(step: .publishing)
        s.goNext()
        XCTAssertEqual(s.step, .publishing)
    }

    func testGoBackWalksReverseAndStopsAtSource() {
        var s = ForgeFeatureState(step: .registered)
        s.goBack(); XCTAssertEqual(s.step, .brand)
        s.goBack(); XCTAssertEqual(s.step, .verify)
        s.goBack(); XCTAssertEqual(s.step, .calibrate)
        s.goBack(); XCTAssertEqual(s.step, .convert)
        s.goBack(); XCTAssertEqual(s.step, .plan)
        s.goBack(); XCTAssertEqual(s.step, .source)
        s.goBack(); XCTAssertEqual(s.step, .source) // clamped
    }

    // MARK: Source step gates on probe verdict

    func testSourceStepRequiresProbeBeforeAdvancing() {
        var s = ForgeFeatureState()
        XCTAssertFalse(s.canAdvance, "Empty input cannot advance")
        s.setSourceRepo("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        XCTAssertFalse(s.canAdvance, "Repo without probe cannot advance")
    }

    func testSourceForgeableProbeAllowsAdvance() {
        var s = ForgeFeatureState()
        s.setSourceRepo("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            sourceFormat: .compressedTensorsAwq,
            hasMtpWeights: true,
            message: "Ready to forge"
        ))
        XCTAssertTrue(s.canAdvance)
    }

    func testSourceAlreadyMTPLXBlocksAdvance() {
        // alreadyMTPLX means "install this instead, don't rebuild"
        // — the wizard does NOT advance to Plan. SourceStage swaps
        // its primary CTA to "Install instead" instead.
        var s = ForgeFeatureState()
        s.setSourceRepo("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
        s.recordProbe(ForgeSourceProbe(
            verdict: .alreadyMTPLX,
            hfRepo: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            sourceFormat: .mlxAffineWithMtp,
            hasMtpWeights: true,
            message: "Already MTPLX-branded — install instead"
        ))
        XCTAssertFalse(s.canAdvance)
    }

    func testSourceNoMtpHeadsBlocksAdvance() {
        var s = ForgeFeatureState()
        s.setSourceRepo("mlx-community/Llama-3-8B-Instruct-4bit")
        s.recordProbe(ForgeSourceProbe(
            verdict: .noMtpHeads,
            hfRepo: "mlx-community/Llama-3-8B-Instruct-4bit",
            sourceFormat: .mlxAffine,
            hasMtpWeights: false,
            message: "Architecture has no MTP heads — Forge cannot build this"
        ))
        XCTAssertFalse(s.canAdvance)
    }

    func testSourceProbeFailedBlocksAdvance() {
        var s = ForgeFeatureState()
        s.setSourceRepo("nonexistent/repo")
        s.recordProbe(ForgeSourceProbe(
            verdict: .probeFailed,
            hfRepo: "nonexistent/repo",
            sourceFormat: .unknown,
            hasMtpWeights: false,
            message: "Probe failed",
            diagnostic: "404"
        ))
        XCTAssertFalse(s.canAdvance)
    }

    // MARK: Setting a new source resets downstream state

    func testSetSourceResetsRecipeAndProbe() {
        var s = ForgeFeatureState()
        s.setSourceRepo("a/b")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "a/b",
            sourceFormat: .compressedTensorsAwq,
            hasMtpWeights: true,
            message: "ok"
        ))
        s.updateRecipe(ForgeRecipe(bodyBits: 3, bodyGroupSize: 32, bodyMode: .affine, mtpPolicy: .requantize))
        s.acknowledgeDegradedMTP()
        XCTAssertTrue(s.hasAcknowledgedDegradedMTP)

        s.setSourceRepo("c/d")
        XCTAssertNil(s.sourceProbe, "Probe cleared when source changes")
        XCTAssertEqual(s.recipe, ForgeRecipe(), "Recipe reset to default when source changes")
        XCTAssertFalse(s.hasAcknowledgedDegradedMTP, "Degraded-MTP ack reset when source changes")
    }

    // MARK: Recipe defaults map source format → reasonable knobs

    func testRecipeDefaultForCompressedTensorsAwq() {
        let r = ForgeRecipe.defaultFor(format: .compressedTensorsAwq)
        XCTAssertEqual(r.bodyBits, 4)
        XCTAssertEqual(r.bodyGroupSize, 64)
        XCTAssertEqual(r.mtpPolicy, .extractFromSidecar,
                       "AWQ source: MTP comes from the embedded sidecar; we extract, never requant by default")
    }

    func testRecipeDefaultForBf16NativePicksKeepBf16() {
        let r = ForgeRecipe.defaultFor(format: .bf16Native)
        XCTAssertEqual(r.mtpPolicy, .keepBf16,
                       "BF16 native: safest path is to keep MTP weights at BF16 (mlx-lm PR 990 evidence)")
    }

    func testRecipeDefaultForMlxAffineWithMtpIsVerifyOnly() {
        let r = ForgeRecipe.defaultFor(format: .mlxAffineWithMtp)
        XCTAssertFalse(r.degradesMtp)
    }

    // MARK: Probe recording auto-derives recipe + brand

    func testRecordProbeSetsRecipeBasedOnSourceFormat() {
        var s = ForgeFeatureState()
        s.setSourceRepo("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            sourceFormat: .compressedTensorsAwq,
            hasMtpWeights: true,
            message: "ok"
        ))
        XCTAssertEqual(s.recipe.mtpPolicy, .extractFromSidecar)
    }

    func testRecordProbeSetsBrandedNameWithLockedMTPLXSuffix() {
        var s = ForgeFeatureState()
        s.setSourceRepo("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
            sourceFormat: .compressedTensorsAwq,
            hasMtpWeights: true,
            message: "ok"
        ))
        XCTAssertEqual(s.brand.brandedName, "cyankiwi-Qwen3.6-35B-A3B-AWQ-4bit-MTPLX")
    }

    func testDerivedBrandedNameMovesMTPLXToFinalSuffix() {
        let name = ForgeBrandInfo.derivedBrandedName(
            sourceRepo: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            role: .speed
        )
        XCTAssertEqual(name, "Youssofal-Qwen3.6-27B-Optimized-Speed-MTPLX",
                       "Forge names always end with one locked MTPLX suffix")
    }

    func testDerivedBrandedNameIgnoresLegacyCustomLabel() {
        let name = ForgeBrandInfo.derivedBrandedName(
            sourceRepo: "foo/bar",
            role: .custom,
            customRoleLabel: "Fast"
        )
        XCTAssertEqual(name, "foo-bar-MTPLX")
    }

    func testResolvedBrandedNameSanitizesUserNameAndLocksSuffix() {
        let name = ForgeBrandInfo.resolvedBrandedName(userName: "  Tiny Test / MTPLX  ")
        XCTAssertEqual(name, "Tiny-Test-MTPLX")
    }

    // MARK: Plan step gates on degraded-MTP override

    func testPlanRequiresAcknowledgementWhenMtpPolicyIsRequantize() {
        var s = ForgeFeatureState(step: .plan)
        s.setSourceRepo("a/b")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "a/b",
            sourceFormat: .bf16Native,
            hasMtpWeights: true,
            message: "ok"
        ))
        s.step = .plan
        XCTAssertTrue(s.canAdvance, "Default recipe is safe; can advance")
        s.updateRecipe(ForgeRecipe(bodyBits: 4, bodyGroupSize: 64, mtpPolicy: .requantize))
        XCTAssertFalse(s.canAdvance, "Degraded MTP requires explicit acknowledgement")
        s.acknowledgeDegradedMTP()
        XCTAssertTrue(s.canAdvance, "After acknowledgement, advance is allowed")
    }

    // MARK: Brand step gates on non-empty name + completed verification

    func testBrandRequiresNonEmptyNameAndVerification() {
        var s = ForgeFeatureState(step: .brand)
        s.updateBrand(ForgeBrandInfo(role: .speed, brandedName: ""))
        XCTAssertFalse(s.canAdvance, "Empty branded name blocks")
        s.updateBrand(ForgeBrandInfo(role: .speed, brandedName: "x-MTPLX"))
        XCTAssertFalse(s.canAdvance, "Name set but verification missing")
        s.recordVerification(ForgeVerification(
            arTokS: 22.0,
            tokSByDepth: [2: 50.0],
            acceptanceByDepth: [2: [0.8, 0.5]],
            bestDepth: 2,
            multiplierVsAr: 2.27,
            verifiedOnHardware: "Apple M5 Max",
            sampler: ForgeSampler()
        ))
        XCTAssertTrue(s.canAdvance)
    }

    func testBrandRequiresMtpSpeedWinNotJustVerification() {
        var s = ForgeFeatureState(step: .brand)
        s.updateBrand(ForgeBrandInfo(role: .speed, brandedName: "x-MTPLX"))
        s.recordVerification(ForgeVerification(
            arTokS: 94.65,
            tokSByDepth: [1: 67.23, 2: 54.87, 3: 45.48],
            acceptanceByDepth: [1: [0.0], 2: [0.0, 0.0], 3: [0.0, 0.0, 0.0]],
            bestDepth: 0,
            multiplierVsAr: 1.0,
            verifiedOnHardware: "Apple Silicon",
            sampler: ForgeSampler()
        ))

        XCTAssertFalse(s.hasSpeedWinningVerification)
        XCTAssertFalse(s.canAdvance, "Converted-but-not-accelerated artifacts cannot reach Registered")
    }

    func testDiscoveryQueryDefaultsTo500AndMatchesMultiTokenQueries() {
        let query = ForgeDiscoveryService.Query()
        XCTAssertEqual(query.limit, 500)

        let inkling = DiscoveryEntry(
            repo: "Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed",
            owner: "Youssofal",
            brandedName: "MTPLX-Inkling-2-Mini-Bare-Speed",
            downloads: 420
        )
        XCTAssertTrue(inkling.matchesSearchQuery(""))
        XCTAssertTrue(inkling.matchesSearchQuery("inkling 2 mini"))
        XCTAssertTrue(inkling.matchesSearchQuery("youssofal inkling"))
        XCTAssertTrue(inkling.matchesSearchQuery("inkling2mini"))
        XCTAssertFalse(inkling.matchesSearchQuery("bonsai 27b"))
    }

    func testBuildOutcomeParserHandlesFailedSpeedPayload() {
        let payload: [String: Any] = [
            "converted_path": "/tmp/Qwen-Qwen3.5-9B-MTPLX-Speed",
            "phase": "verify",
            "verdict": "mtp_acceptance_collapsed",
            "failure_reasons": ["mtp_acceptance_collapsed", "no_mtp_depth_beat_ar"],
            "message": "MTP did not accelerate this model; draft acceptance collapsed.",
            "architecture_id": "qwen3-next-mtp",
            "ar_tok_s": 94.65,
            "best_mtp_depth": 1,
            "best_mtp_tok_s": 67.23,
            "best_mtp_multiplier_vs_ar": 0.71,
            "verify_rows": [
                [
                    "depth": 0,
                    "tok_s": 94.65,
                    "multiplier_vs_ar": 1.0,
                    "acceptance_by_position": [],
                    "verify_time_s": 1.2,
                ],
                [
                    "depth": 1,
                    "tok_s": 67.23,
                    "multiplier_vs_ar": 0.71,
                    "acceptance_by_position": [0.0],
                    "verify_time_s": 1.4,
                ],
            ],
        ]

        let outcome = ForgeBuildOutcome.parse(payload)

        XCTAssertEqual(outcome?.verdict, "mtp_acceptance_collapsed")
        XCTAssertEqual(outcome?.arTokS, 94.65)
        XCTAssertEqual(outcome?.bestMTPDepth, 1)
        XCTAssertEqual(outcome?.verifyRows.count, 2)
        XCTAssertEqual(outcome?.verifyRows[1].acceptanceByPosition, [0.0])
        XCTAssertFalse(outcome?.isSpeedWin ?? true)
    }

    // MARK: Service-driven steps never self-advance

    func testServiceDrivenStepsAreAlwaysGatedExternally() {
        for step in [ForgeStep.convert, .calibrate, .verify, .publishing] {
            let s = ForgeFeatureState(step: step)
            XCTAssertFalse(s.canAdvance, "Step \(step) should be orchestrator-driven, not self-advancing")
        }
    }

    // MARK: SubMode is independent of step

    func testSelectSubModeDoesNotResetStep() {
        var s = ForgeFeatureState(step: .verify, subMode: .create)
        s.selectSubMode(.discover)
        XCTAssertEqual(s.subMode, .discover)
        XCTAssertEqual(s.step, .verify, "Switching sub-modes does NOT cancel an in-flight wizard")
    }

    // MARK: resetWizard fully clears

    func testResetWizardClearsEverything() {
        var s = ForgeFeatureState(step: .brand)
        s.setSourceRepo("a/b")
        s.recordProbe(ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: "a/b",
            sourceFormat: .bf16Native,
            hasMtpWeights: true,
            message: "ok"
        ))
        s.updateBrand(ForgeBrandInfo(role: .quality, brandedName: "x"))
        s.recordVerification(ForgeVerification(
            arTokS: 22, tokSByDepth: [3: 60], acceptanceByDepth: [3: [0.9, 0.7, 0.5]],
            bestDepth: 3, multiplierVsAr: 2.7, verifiedOnHardware: "x", sampler: ForgeSampler()
        ))
        s.subMode = .mine
        s.resetWizard()
        XCTAssertEqual(s.step, .source)
        XCTAssertEqual(s.subMode, .create)
        XCTAssertEqual(s.sourceRepoInput, "")
        XCTAssertNil(s.sourceProbe)
        XCTAssertNil(s.verification)
        XCTAssertEqual(s.brand, ForgeBrandInfo())
    }

    // MARK: body dtype (#166)

    func testForgeRecipeDecodesLegacyPayloadWithoutBodyDtype() throws {
        let legacy = #"{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}"#
        let recipe = try JSONDecoder().decode(ForgeRecipe.self, from: Data(legacy.utf8))
        XCTAssertEqual(recipe.bodyDtype, .auto)
        XCTAssertEqual(recipe.bodyBits, 4)
        XCTAssertEqual(recipe.mtpPolicy, .keepBf16)
    }

    func testForgeRecipeEncodesBodyDtypeUnderSnakeCaseKey() throws {
        var recipe = ForgeRecipe()
        recipe.bodyDtype = .fp16
        let data = try JSONEncoder().encode(recipe)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["body_dtype"] as? String, "fp16")
    }

    func testForgeRecipeDefaultBodyDtypeIsAuto() {
        XCTAssertEqual(ForgeRecipe().bodyDtype, .auto)
    }
}
