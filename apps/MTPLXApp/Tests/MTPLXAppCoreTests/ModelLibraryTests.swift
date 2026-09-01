import XCTest
@testable import MTPLXAppCore

final class ModelLibraryTests: XCTestCase {
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

    private func makeCompleteModel(
        under root: URL,
        directoryName: String,
        publicModelID: String
    ) throws -> URL {
        let model = root.appendingPathComponent(directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try "{}".write(to: model.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try "{\"public_model_id\":\"\(publicModelID)\"}".write(
            to: model.appendingPathComponent("mtplx_runtime.json"),
            atomically: true,
            encoding: .utf8
        )
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
