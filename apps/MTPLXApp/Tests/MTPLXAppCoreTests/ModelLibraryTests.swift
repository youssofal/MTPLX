import XCTest
@testable import MTPLXAppCore

final class ModelLibraryTests: XCTestCase {
    func testMalformedLibrarySettingsPreserveValidRootsAndOtherSettings() throws {
        let root = temporaryDirectory()
        let settingsURL = root.appendingPathComponent("settings.json")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let first = root.appendingPathComponent("first").path
        let second = root.appendingPathComponent("second").path
        let data = try JSONSerialization.data(withJSONObject: [
            "primary_model_directory": 42,
            "additional_model_directories": [first, 42, second, first],
            "port": 8765,
            "onboarding_completed_at": 800_000_000,
        ])
        try data.write(to: settingsURL)

        let result = MTPLXSettingsStore(settingsURL: settingsURL).loadWithRecovery()

        XCTAssertNil(result.recovery)
        XCTAssertEqual(result.configuration.primaryModelDirectory, ModelLibrary.default.primaryDirectory.path)
        XCTAssertEqual(result.configuration.additionalModelDirectories, [first, second])
        XCTAssertEqual(result.configuration.port, 8765)
        XCTAssertNotNil(result.configuration.onboardingCompletedAt)
        XCTAssertEqual(result.degradedFields.map(\.path).sorted(), [
            "additional_model_directories[1]", "primary_model_directory",
        ])
    }

    func testLegacyConfigurationDecodesDefaultLibrary() throws {
        let config = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: Data("{}".utf8)
        )

        XCTAssertEqual(config.primaryModelDirectory, ModelLibrary.default.primaryDirectory.path)
        XCTAssertEqual(config.additionalModelDirectories, [])
    }

    func testConfigurationRoundTripsOrderedCanonicalDirectories() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let secondary = root.appendingPathComponent("secondary", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondary, withIntermediateDirectories: true)
        let alias = root.appendingPathComponent("primary-alias", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: primary)

        let config = MTPLXAppConfiguration(
            primaryModelDirectory: primary.path,
            additionalModelDirectories: [secondary.path, alias.path, secondary.path]
        )
        let decoded = try JSONDecoder().decode(
            MTPLXAppConfiguration.self,
            from: JSONEncoder().encode(config)
        )

        XCTAssertEqual(decoded.primaryModelDirectory, primary.path)
        XCTAssertEqual(decoded.additionalModelDirectories, [secondary.path])
    }

    func testUnavailableAdditionalDirectoryIsPreserved() {
        let root = temporaryDirectory()
        let missing = root.appendingPathComponent("offline-volume/models").path
        var config = MTPLXAppConfiguration(
            primaryModelDirectory: root.path,
            additionalModelDirectories: [missing]
        )

        config.normalizeModelDirectories()

        XCTAssertEqual(config.additionalModelDirectories, [missing])
        XCTAssertFalse(ModelLibrary.isAvailable(URL(fileURLWithPath: missing)))
    }

    func testChangingPrimaryPreservesPreviousRootWithoutAliases() throws {
        let root = temporaryDirectory()
        let old = root.appendingPathComponent("old", isDirectory: true)
        let new = root.appendingPathComponent("new", isDirectory: true)
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: new, withIntermediateDirectories: true)
        var config = MTPLXAppConfiguration(primaryModelDirectory: old.path)

        config.setPrimaryModelDirectory(new.path)

        XCTAssertEqual(config.primaryModelDirectory, new.path)
        XCTAssertEqual(config.additionalModelDirectories, [old.path])
    }

    func testDiscoveryUsesRootOrderAndPathStableIdentity() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let secondary = root.appendingPathComponent("secondary", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondary, withIntermediateDirectories: true)
        let first = try makeCompleteModel(
            under: primary,
            directoryName: "Acme--First",
            publicModelID: "Acme/First"
        )
        let second = try makeCompleteModel(
            under: secondary,
            directoryName: "Acme--Second",
            publicModelID: "Acme/Second"
        )
        let library = ModelLibrary(
            primaryDirectory: primary.path,
            additionalDirectories: [secondary.path]
        )

        XCTAssertEqual(library.discoverCompleteModels().map(\.path), [first.path, second.path])
        let catalog = MTPLXModelOption.pickerCatalog(
            customModels: [],
            modelLibrary: library
        )
        XCTAssertEqual(catalog.first(where: { $0.hfModelID == "Acme/First" })?.id, "local:\(first.path)")
        XCTAssertEqual(catalog.first(where: { $0.hfModelID == "Acme/Second" })?.resolvedReference, second.path)
    }

    func testKnownPickerModelPrefersConfiguredLibraryCandidate() throws {
        let root = temporaryDirectory()
        let repo = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
        let model = try makeCompleteModel(
            under: root,
            directoryName: repo.replacingOccurrences(of: "/", with: "--"),
            publicModelID: repo
        )
        let library = ModelLibrary(primaryDirectory: root.path)

        let catalog = MTPLXModelOption.pickerCatalog(
            customModels: [],
            currentModel: repo,
            modelLibrary: library
        )
        let option = try XCTUnwrap(catalog.first(where: { $0.matches(repo) }))

        XCTAssertEqual(option.localCandidates.first, model.path)
        XCTAssertEqual(option.installedLocalPath(in: library), model.path)
    }

    func testDuplicateRepositoryUsesFirstCompleteLibraryCopy() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let secondary = root.appendingPathComponent("secondary", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondary, withIntermediateDirectories: true)
        let repo = "Acme/Duplicate"
        let first = try makeCompleteModel(
            under: primary,
            directoryName: "Acme--Duplicate",
            publicModelID: repo
        )
        _ = try makeCompleteModel(
            under: secondary,
            directoryName: "Acme--Duplicate",
            publicModelID: repo
        )
        let library = ModelLibrary(
            primaryDirectory: primary.path,
            additionalDirectories: [secondary.path]
        )

        let option = try XCTUnwrap(
            MTPLXModelOption.pickerCatalog(
                customModels: [],
                modelLibrary: library
            ).first(where: { $0.hfModelID == repo })
        )

        XCTAssertEqual(option.localCandidates.first, first.path)
        XCTAssertEqual(option.installedLocalPath(in: library), first.path)
    }

    func testDownloaderArgumentsCarrySnapshottedPrimaryRoot() {
        let root = URL(fileURLWithPath: "/Volumes/Models A", isDirectory: true)

        XCTAssertEqual(
            ModelDownloader.streamArguments(
                repo: "owner/pack",
                update: false,
                cacheRoot: root
            ),
            ["pull", "owner/pack", "--progress-json", "--cache-dir", root.path]
        )
        XCTAssertEqual(
            ModelDownloader.streamArguments(
                repo: "owner/pack",
                update: true,
                destinationPath: "/legacy/pack",
                cacheRoot: root
            ),
            [
                "models", "--update", "owner/pack", "--progress-json",
                "--installed-path", "/legacy/pack", "--cache-dir", root.path,
            ]
        )
    }

    func testForgeIndexDedupesSymlinkedRootAliases() throws {
        let outer = temporaryDirectory()
        let root = outer.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let model = root.appendingPathComponent("Local-Forge", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try "{\"public_model_id\":\"Local Forge\"}".write(
            to: model.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )
        let alias = outer.appendingPathComponent("models-alias", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: root)
        let registered = try XCTUnwrap(
            MTPLXModelOption.forgedModel(brandedName: "Local Forge", localPath: model.path)
        )

        let entries = ForgeLocalIndex(roots: [root, alias])
            .scan(includingRegistered: [registered])

        XCTAssertEqual(entries.map(\.localPath), [model.path])
    }

    @MainActor
    func testOnboardingUsesConfiguredLibraryForInstalledModels() throws {
        let root = temporaryDirectory()
        let repo = "Acme/Onboarding"
        let model = try makeCompleteModel(
            under: root,
            directoryName: "Acme--Onboarding",
            publicModelID: repo
        )
        let option = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: repo))
        let orchestrator = OnboardingOrchestrator(
            modelLibrary: ModelLibrary(primaryDirectory: root.path)
        )

        XCTAssertTrue(orchestrator.isModelInstalled(option))
        XCTAssertEqual(orchestrator.installedLocalPath(for: option), model.path)
    }

    // Two Forge builds of one source (a 4-bit and a 6-bit) both record that
    // source as forge_provenance.source_repo. Each is its own model: it is
    // listed under its folder name, in its own row, and selecting the row
    // launches that folder.
    func testBuildsOfOneSourceKeepTheirFolderNamesAndTheirOwnRows() throws {
        let root = temporaryDirectory()
        let tag = String(UUID().uuidString.prefix(8))
        for source in [root.appendingPathComponent("_src-Qwen3.5-9B-V26-Distill-graft").path, "Qwen/Qwen3.5-9B"] {
            let runtime: [String: Any] = ["forge_provenance": forgeProvenance(source: source)]
            let variant = source.hasPrefix("/") ? "local" : "hub"
            let fourBit = try makeCompleteModel(
                under: root,
                directoryName: "MiMo-9B-namecheck-\(variant)-\(tag)",
                runtime: runtime
            )
            let sixBit = try makeCompleteModel(
                under: root,
                directoryName: "MiMo-9B-6bit-\(variant)-\(tag)",
                runtime: runtime
            )
            let library = ModelLibrary(primaryDirectory: root.path)
            XCTAssertEqual(recordedSource(of: sixBit), source, "the fixture must parse as Forge provenance")

            let names = Dictionary(
                uniqueKeysWithValues: library.discoverCompleteModels().map { ($0.path, $0.displayName) }
            )
            XCTAssertEqual(names[fourBit.path], fourBit.lastPathComponent)
            XCTAssertEqual(names[sixBit.path], sixBit.lastPathComponent)

            let rows = MTPLXModelOption.pickerCatalog(customModels: [], modelLibrary: library)
            for folder in [fourBit, sixBit] {
                let listing = rows.filter { $0.localCandidates.contains(folder.path) }
                XCTAssertEqual(listing.count, 1, folder.lastPathComponent)
                XCTAssertEqual(listing.first?.displayName, folder.lastPathComponent)
                XCTAssertEqual(listing.first?.localCandidates, [folder.path])
                XCTAssertEqual(listing.first?.resolvedReference(in: library), folder.path)
                XCTAssertEqual(listing.first?.installedLocalPath, folder.path)
            }
            XCTAssertFalse(rows.contains { $0.displayName.hasPrefix("_src-") || $0.matches(source) })
        }
    }

    // Official packs record the base model they were built from. A download of
    // one belongs to its catalog row only, not to a second row named after the
    // base model.
    func testOfficialDownloadIsNotListedAgainUnderItsBaseModel() throws {
        let root = temporaryDirectory()
        let repo = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
        let official = try makeCompleteModel(
            under: root,
            directoryName: repo.replacingOccurrences(of: "/", with: "--"),
            runtime: ["forge_provenance": forgeProvenance(source: "Qwen/Qwen3.8-27B")]
        )
        let library = ModelLibrary(primaryDirectory: root.path)
        XCTAssertEqual(recordedSource(of: official), "Qwen/Qwen3.8-27B")

        let rows = MTPLXModelOption.pickerCatalog(
            customModels: [],
            currentModel: repo,
            modelLibrary: library
        )

        XCTAssertEqual(rows.filter { $0.localCandidates.contains(official.path) }.map(\.hfModelID), [repo])
        XCTAssertFalse(rows.contains { $0.id.hasPrefix("local:") })
        XCTAssertFalse(rows.contains { $0.matches("Qwen/Qwen3.8-27B") })
    }

    // Only a name the pack declares for itself replaces the folder name.
    func testOnlyADeclaredPublicNameReplacesTheFolderName() throws {
        let root = temporaryDirectory()
        let tag = String(UUID().uuidString.prefix(8))
        let provenance = forgeProvenance(source: "Acme/Base")
        let publicName = try makeCompleteModel(
            under: root,
            directoryName: "build-a-\(tag)",
            runtime: ["public_model_id": "acme-coder-a-\(tag)", "forge_provenance": provenance]
        )
        let servedName = try makeCompleteModel(
            under: root,
            directoryName: "build-b-\(tag)",
            runtime: ["served_model_id": "acme-coder-b-\(tag)", "forge_provenance": provenance]
        )
        let undeclared = try makeCompleteModel(
            under: root,
            directoryName: "build-c-\(tag)",
            runtime: ["model_id": "Acme/Base", "forge_provenance": provenance]
        )
        let library = ModelLibrary(primaryDirectory: root.path)
        XCTAssertEqual(recordedSource(of: undeclared), "Acme/Base")

        let names = Dictionary(
            uniqueKeysWithValues: library.discoverCompleteModels().map { ($0.path, $0.displayName) }
        )

        XCTAssertEqual(names[publicName.path], "acme-coder-a-\(tag)")
        XCTAssertEqual(names[servedName.path], "acme-coder-b-\(tag)")
        XCTAssertEqual(names[undeclared.path], "build-c-\(tag)")
    }

    // The same install under two roots (same folder name) is one model with an
    // ordered fallback, as for a duplicated repository. Two different folders
    // stay two rows even when they declare the same name.
    func testOnlyTheSameInstallUnderTwoRootsSharesARow() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let secondary = root.appendingPathComponent("secondary", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondary, withIntermediateDirectories: true)
        let tag = String(UUID().uuidString.prefix(8))
        let first = try makeCompleteModel(under: primary, directoryName: "Local-Build-\(tag)", runtime: nil)
        let copy = try makeCompleteModel(under: secondary, directoryName: "Local-Build-\(tag)", runtime: nil)
        let shared = ["public_model_id": "shared-name-\(tag)"]
        let variantA = try makeCompleteModel(under: primary, directoryName: "Variant-A-\(tag)", runtime: shared)
        let variantB = try makeCompleteModel(under: secondary, directoryName: "Variant-B-\(tag)", runtime: shared)
        let library = ModelLibrary(
            primaryDirectory: primary.path,
            additionalDirectories: [secondary.path]
        )

        let rows = MTPLXModelOption.pickerCatalog(customModels: [], modelLibrary: library)

        let install = rows.filter { $0.localCandidates.contains(first.path) }
        XCTAssertEqual(install.map(\.localCandidates), [[first.path, copy.path]])
        XCTAssertEqual(install.first?.resolvedReference(in: library), first.path)
        for variant in [variantA, variantB] {
            let listing = rows.filter { $0.localCandidates.contains(variant.path) }
            XCTAssertEqual(listing.map(\.localCandidates), [[variant.path]])
            XCTAssertEqual(listing.first?.resolvedReference(in: library), variant.path)
        }
    }

    // The folder settings.json points at already has its row; discovering it
    // under a declared name that differs from the folder name adds no second.
    func testCurrentFolderWithItsOwnDeclaredNameHasOneRow() throws {
        let root = temporaryDirectory()
        let tag = String(UUID().uuidString.prefix(8))
        let folder = try makeCompleteModel(
            under: root,
            directoryName: "Acme--Current-\(tag)",
            runtime: ["public_model_id": "acme-current-build-\(tag)"]
        )
        let library = ModelLibrary(primaryDirectory: root.path)

        let rows = MTPLXModelOption.pickerCatalog(
            customModels: [],
            currentModel: folder.path,
            modelLibrary: library
        )

        XCTAssertEqual(rows.filter { $0.localCandidates.contains(folder.path) }.count, 1)
    }

    func testHFHomeAndHFHubCacheLayoutsReuseExistingCompleteSnapshots() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let hfHome = root.appendingPathComponent("hf-home", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)

        let repo = "Youssofal/Inkling-2-Mini-MTPLX-Bare-Speed"
        let sha = "112233445566"
        let snap = hfHome
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(sha, isDirectory: true)
        let refs = hfHome
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--Youssofal--Inkling-2-Mini-MTPLX-Bare-Speed", isDirectory: true)
            .appendingPathComponent("refs", isDirectory: true)
        try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try sha.write(to: refs.appendingPathComponent("main"), atomically: true, encoding: .utf8)
        try "{}".write(to: snap.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: snap.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: snap.appendingPathComponent("model.safetensors"))
        try Data([0]).write(to: snap.appendingPathComponent("mtp.safetensors"))
        try JSONSerialization.data(withJSONObject: [
            "source": "thinkingmachines/Inkling-2-Mini",
            "recipe": "bare-speed",
            "display_name": "Inkling 2 Mini Bare Speed",
        ]).write(to: snap.appendingPathComponent("mtplx-pack.json"))

        let customRepo = "Vendor/HFReuseTestModel"
        let customSnap = hfHome
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--Vendor--HFReuseTestModel", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(sha, isDirectory: true)
        let customRefs = hfHome
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--Vendor--HFReuseTestModel", isDirectory: true)
            .appendingPathComponent("refs", isDirectory: true)
        try FileManager.default.createDirectory(at: customSnap, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: customRefs, withIntermediateDirectories: true)
        try sha.write(to: customRefs.appendingPathComponent("main"), atomically: true, encoding: .utf8)
        try "{}".write(to: customSnap.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: customSnap.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: customSnap.appendingPathComponent("model.safetensors"))

        let library = ModelLibrary(
            primaryDirectory: primary.path,
            environment: ["HF_HOME": hfHome.path],
            includeHuggingFaceCache: true
        )
        let resolvedSnap = ModelLibrary.canonicalURL(for: snap.path).path
        XCTAssertTrue(library.candidatePaths(for: repo).contains(resolvedSnap))

        let resolvedCustomSnap = ModelLibrary.canonicalURL(for: customSnap.path).path
        let option = MTPLXModelOption(
            id: "custom-hf-reuse",
            displayName: "HF Reuse Model",
            shortName: "HF Reuse",
            localizedDetailKey: "Custom",
            hfModelID: customRepo,
            localCandidates: []
        )
        XCTAssertEqual(option.installedLocalPath(in: library), resolvedCustomSnap)

        let discovered = library.discoverCompleteModels()
        XCTAssertTrue(discovered.contains { $0.reference == repo && $0.path == resolvedSnap })

        // Incomplete HF snapshots must never count as installed.
        try FileManager.default.removeItem(at: customSnap.appendingPathComponent("model.safetensors"))
        XCTAssertNil(option.installedLocalPath(in: library))
    }

    func testHFLayoutInsideAddedModelDirIsDiscoveredAndCounted() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let sharedHF = root.appendingPathComponent("shared-hf", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        let snap = sharedHF
            .appendingPathComponent("models--Vendor--SampleModel", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("mainsha", isDirectory: true)
        try FileManager.default.createDirectory(at: snap, withIntermediateDirectories: true)
        try "{}".write(to: snap.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: snap.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: snap.appendingPathComponent("model.safetensors"))

        XCTAssertEqual(SharedModelFolders.packCount(in: sharedHF.path), 1)
        let library = ModelLibrary(
            primaryDirectory: primary.path,
            additionalDirectories: [sharedHF.path]
        )
        let discovered = library.discoverCompleteModels()
        XCTAssertEqual(discovered.map(\.reference), ["Vendor/SampleModel"])
    }

    func testResolveAnyHFSnapshotSkipsPartialMainToFindOlderCompleteSnapshot() throws {
        let root = temporaryDirectory()
        let primary = root.appendingPathComponent("primary", isDirectory: true)
        let hfHome = root.appendingPathComponent("hf_home", isDirectory: true)
        try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)

        let partialHub = root.appendingPathComponent("partial-only-hub", isDirectory: true)
        let partialOnlySnap = partialHub
            .appendingPathComponent("models--Vendor--PartialOnly", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent("partialsha", isDirectory: true)
        try FileManager.default.createDirectory(at: partialOnlySnap, withIntermediateDirectories: true)
        try "{}".write(to: partialOnlySnap.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        XCTAssertEqual(SharedModelFolders.packCount(in: partialHub.path), 0)

        let repoDir = hfHome
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--Vendor--MixedSnapshots", isDirectory: true)
        let oldComplete = repoDir.appendingPathComponent("snapshots/oldcomplete", isDirectory: true)
        let newPartial = repoDir.appendingPathComponent("snapshots/newpartial", isDirectory: true)
        let refs = repoDir.appendingPathComponent("refs", isDirectory: true)
        try FileManager.default.createDirectory(at: oldComplete, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: newPartial, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: refs, withIntermediateDirectories: true)
        try "newpartial".write(to: refs.appendingPathComponent("main"), atomically: true, encoding: .utf8)
        try "{}".write(to: newPartial.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: oldComplete.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: oldComplete.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: oldComplete.appendingPathComponent("model.safetensors"))

        let library = ModelLibrary(
            primaryDirectory: primary.path,
            environment: ["HF_HOME": hfHome.path],
            includeHuggingFaceCache: true
        )
        let discovered = library.discoverCompleteModels()
        let canonicalOld = ModelLibrary.canonicalURL(for: oldComplete.path).path
        XCTAssertTrue(discovered.contains { $0.reference == "Vendor/MixedSnapshots" && $0.path == canonicalOld })
    }

    /// The provenance block Forge writes into mtplx_runtime.json, complete
    /// enough for MTPLXRuntimeMetadata to decode (a partial block decodes to
    /// nil and would hide the naming path under test).
    private func forgeProvenance(source: String) -> [String: Any] {
        [
            "source_repo": source,
            "source_format": "bf16_native",
            "forge_recipe": [
                "body_bits": 4,
                "body_group_size": 64,
                "body_mode": "affine",
                "mtp_policy": "keep_bf16",
            ],
            "forge_inputs": [String: String](),
            "forged_at": "2026-09-22T15:03:46-07:00",
            "mtplx_version": "2.12.0",
            "forged_locally": true,
        ]
    }

    private func recordedSource(of model: URL) -> String? {
        MTPLXRuntimeMetadata.read(
            at: model.appendingPathComponent("mtplx_runtime.json").path
        )?.forgeProvenance?.sourceRepo
    }

    private func makeCompleteModel(
        under root: URL,
        directoryName: String,
        publicModelID: String
    ) throws -> URL {
        try makeCompleteModel(
            under: root,
            directoryName: directoryName,
            runtime: ["public_model_id": publicModelID]
        )
    }

    private func makeCompleteModel(
        under root: URL,
        directoryName: String,
        runtime: [String: Any]?
    ) throws -> URL {
        let model = root.appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try "{}".write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        if let runtime {
            try JSONSerialization.data(withJSONObject: runtime)
                .write(to: model.appendingPathComponent("mtplx_runtime.json"))
        }
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))
        return model
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelLibraryTests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
