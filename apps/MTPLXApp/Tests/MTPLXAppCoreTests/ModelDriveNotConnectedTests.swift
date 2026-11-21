import XCTest

@testable import MTPLXAppCore

/// Model folders on an external drive (night 4, D5 follow-up): the two past
/// failures of this feature, proven not to come back through the shared
/// `model_dir` / `model_dirs` setting.
///
/// #466: the wizard measured free space on the home volume while downloads
/// went to an external drive. #573: a pack the user picked from another
/// folder was swapped for a fresh download on the next start. Scratch
/// folders and a `/Volumes` name that is never mounted stand in for the
/// drive; no real drive, model or server is touched.
final class ModelDriveNotConnectedTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-drive-\(UUID().uuidString.lowercased())", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        ModelFolderIndex.location = nil
        try? FileManager.default.removeItem(at: scratch)
    }

    private func seedIndex(_ folders: [String: [ModelFolderIndex.Entry]]) throws -> URL {
        let url = scratch.appendingPathComponent("model-folders-seen.json")
        try JSONEncoder().encode(ModelFolderIndex.File(folders: folders)).write(to: url)
        ModelFolderIndex.location = url
        return url
    }

    private func completePack(in folder: URL, named name: String) throws -> URL {
        let pack = folder.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: pack.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: pack.appendingPathComponent("tokenizer.json"))
        try Data("w".utf8).write(to: pack.appendingPathComponent("model.safetensors"))
        return pack
    }

    /// The picker keeps listing the packs of a folder whose drive is
    /// unplugged, marked as not connected: a row for that model (here a
    /// remembered Hugging Face id) is marked instead of offered as a fresh
    /// download, and a local build keeps its own row.
    func testThePickerListsPacksOnAnUnpluggedDriveAsNotConnected() throws {
        let root = unplugged + "/models"
        let drive = ((root as NSString).deletingLastPathComponent as NSString).lastPathComponent
        _ = try seedIndex([root: [
            .init(path: root + "/someone--example-pack",
                  reference: "someone/example-pack",
                  displayName: "example-pack"),
            .init(path: root + "/my-forged-pack", reference: "local/my-forged-pack", displayName: "my-forged-pack"),
        ]])
        let library = ModelLibrary(
            primaryDirectory: scratch.appendingPathComponent("primary").path,
            additionalDirectories: [root]
        )
        let remembered = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "someone/example-pack"))
        let rows = MTPLXModelOption.pickerCatalog(customModels: [remembered], modelLibrary: library)
        let row = try XCTUnwrap(rows.first { $0.matches("someone/example-pack") })
        XCTAssertEqual(row.localizedDetail, ModelFolderIndex.notConnectedDetail(driveName: drive))
        XCTAssertNil(row.installedLocalPath(in: library))
        let forged = try XCTUnwrap(rows.first { $0.id == "local:\(root)/my-forged-pack" })
        XCTAssertEqual(forged.localizedDetail, ModelFolderIndex.notConnectedDetail(driveName: drive))
        // With no index (a fresh install, or tests) nothing extra is listed.
        ModelFolderIndex.location = nil
        XCTAssertFalse(MTPLXModelOption.pickerCatalog(customModels: [], modelLibrary: library)
            .contains { $0.id.hasPrefix("local:\(root)") })
    }

    /// A readable folder's packs are recorded; an unplugged folder keeps its
    /// last entry; a folder no longer configured is dropped.
    func testTheIndexKeepsAnUnpluggedFolderAndDropsAnUnconfiguredOne() throws {
        let ssd = scratch.appendingPathComponent("ssd", isDirectory: true)
        let pack = try completePack(in: ssd, named: "someone--example")
        let root = unplugged + "/models"
        let url = try seedIndex([
            root: [.init(path: root + "/a", reference: "x/a", displayName: "a")],
            "/gone/folder": [.init(path: "/gone/folder/b", reference: "x/b", displayName: "b")],
        ])
        let library = ModelLibrary(primaryDirectory: ssd.path, additionalDirectories: [root])
        _ = MTPLXModelOption.pickerCatalog(customModels: [], modelLibrary: library)
        let file = ModelFolderIndex.read(url)
        XCTAssertEqual(Set(file.folders.keys), [library.primaryDirectory.path, root])
        XCTAssertEqual(file.folders[library.primaryDirectory.path]?.map(\.path),
                       [ModelLibrary.canonicalURL(for: pack.path).path])
        XCTAssertEqual(file.folders[root]?.map(\.reference), ["x/a"])
    }

    private var unplugged: String { "/Volumes/mtplx-test-unplugged-\(UUID().uuidString)" }

    func testSelectedLocalPackKeepsItsRecordedNameAcrossEjectAndReconnect() throws {
        let fm = FileManager.default
        let folder = scratch.resolvingSymlinksInPath().appendingPathComponent("library")
        let pack = try completePack(in: folder, named: "repo--pack")
        let title = "Removable test pack"
        try JSONEncoder().encode(["display_name": title]).write(to: pack.appendingPathComponent("mtplx-pack.json"))
        ModelFolderIndex.location = scratch.appendingPathComponent("model-folders-seen.json")
        func selectedPack() throws -> MTPLXModelOption {
            let library = ModelLibrary(primaryDirectory: folder.path)
            let rows = MTPLXModelOption.pickerCatalog(customModels: [], currentModel: pack.path, modelLibrary: library)
                .filter { $0.localCandidates.contains(pack.path) }
            XCTAssertEqual(rows.count, 1)
            return try XCTUnwrap(rows.first)
        }

        let mounted = try selectedPack()
        XCTAssertEqual(mounted.displayName, title)
        XCTAssertEqual(mounted.installedLocalPath, pack.path)
        let parked = scratch.appendingPathComponent("parked")
        try fm.moveItem(at: folder, to: parked)
        let drive = unplugged
        try fm.createSymbolicLink(at: folder, withDestinationURL: URL(fileURLWithPath: drive + "/library"))

        let offline = try selectedPack()
        XCTAssertEqual(offline.id, mounted.id)
        XCTAssertEqual(offline.displayName, title)
        XCTAssertNil(offline.installedLocalPath)
        XCTAssertTrue(offline.isOnDisconnectedDrive)
        XCTAssertEqual(offline.localizedDetail, ModelFolderIndex.notConnectedDetail(driveName: (drive as NSString).lastPathComponent))

        try fm.removeItem(at: folder) // The test's dangling link.
        try fm.moveItem(at: parked, to: folder)
        XCTAssertEqual(try selectedPack(), mounted)
    }

    /// #466 through the shared setting: `mtplx folders downloads` (or the
    /// app's "Use for Downloads") writes `model_dir`; every free-space read
    /// and the downloader's own default then use that folder, not the home
    /// volume.
    func testTheSharedDownloadFolderIsTheStoreThatIsMeasured() throws {
        let home = scratch.appendingPathComponent("home", isDirectory: true)
        let ssd = scratch.appendingPathComponent("ssd/models", isDirectory: true)
        try FileManager.default.createDirectory(at: ssd, withIntermediateDirectories: true)
        let config = home.appendingPathComponent(".mtplx/config.toml")
        try SharedModelFolders.write(
            primary: ssd.path, additional: [], to: config, environment: ["HOME": home.path]
        )
        let env = ["HOME": home.path]
        XCTAssertEqual(ModelDownloader.defaultCacheRoot(env: env).path, ssd.path)
        XCTAssertEqual(
            ModelStoreVolume.measurementURL(for: ModelDownloader.defaultCacheRoot(env: env)).path,
            ssd.resolvingSymlinksInPath().path
        )
        XCTAssertGreaterThan(ModelStoreVolume.freeGiB(env: env), 0)
        // MTPLX_MODEL_DIR still wins, and without the setting the default
        // is ~/.mtplx/models as before.
        XCTAssertEqual(
            ModelDownloader.defaultCacheRoot(env: ["HOME": home.path, "MTPLX_MODEL_DIR": "/tmp/x"]).path,
            "/tmp/x"
        )
        let bare = scratch.appendingPathComponent("bare", isDirectory: true)
        XCTAssertEqual(
            ModelDownloader.defaultCacheRoot(env: ["HOME": bare.path]).path,
            bare.appendingPathComponent(".mtplx/models").path
        )
    }

    /// A download folder on a drive that is unplugged has no free space for
    /// MTPLX: walking up from /Volumes/<drive> must not measure the startup
    /// disk instead.
    func testAnUnpluggedDownloadFolderHasNoFreeSpace() {
        XCTAssertEqual(ModelStoreVolume.freeGiB(forStore: URL(fileURLWithPath: unplugged + "/models")), 0)
        XCTAssertGreaterThan(ModelStoreVolume.freeGiB(forStore: scratch), 0)
    }

    /// #573's shape in the app: the selected pack lives on an unplugged
    /// drive. Start says which drive to plug in; it neither launches nor
    /// offers to download the pack again onto this Mac.
    @MainActor
    func testStartWithThePacksDriveUnpluggedSaysSoAndNeverDownloads() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        var configuration = daemon.configuration(fanMode: .default)
        let drive = unplugged
        configuration.model = drive + "/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
        let store = daemon.makeStore(configuration: configuration, fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        let message = try XCTUnwrap(store.modelDriveNotConnectedMessage(for: configuration))
        XCTAssertTrue(message.contains((drive as NSString).lastPathComponent), message)

        await store.startDaemon()

        XCTAssertNil(store.pendingModelDownload, "a pack on an unplugged drive is never downloaded again")
        XCTAssertEqual(store.startupPhase, .failed(message))
        XCTAssertEqual(store.daemonState, .degraded(message))
        XCTAssertNil(store.health, "nothing was launched")

        // A repo id whose pack was last seen on that drive: the same message,
        // never a download.
        let root = drive + "/models"
        _ = try seedIndex([root: [.init(
            path: root + "/someone--example-pack",
            reference: "someone/example-pack",
            displayName: "example-pack"
        )]])
        var byRepo = configuration
        byRepo.model = "someone/example-pack"
        byRepo.additionalModelDirectories = [root]
        let byRepoMessage = try XCTUnwrap(store.modelDriveNotConnectedMessage(for: byRepo))
        XCTAssertTrue(byRepoMessage.contains((drive as NSString).lastPathComponent), byRepoMessage)
        ModelFolderIndex.location = nil

        // A plain path that is simply missing, or a repo id, is not this case.
        var missing = configuration
        missing.model = scratch.appendingPathComponent("gone/Youssofal--X").path
        XCTAssertNil(store.modelDriveNotConnectedMessage(for: missing))
        missing.model = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
        XCTAssertNil(store.modelDriveNotConnectedMessage(for: missing))
    }

    /// #573's shape through the shared folders: a pack the user keeps in a
    /// folder named in `model_dirs` is found there by the app on every
    /// start (the app adopts the shared list), so it is never queued for a
    /// download into the primary folder.
    func testAPackInASharedFolderIsFoundNotDownloaded() throws {
        let ssd = scratch.appendingPathComponent("ssd", isDirectory: true)
        let pack = ssd.appendingPathComponent("Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed", isDirectory: true)
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        let settings = scratch.appendingPathComponent("settings.json")
        let shared = scratch.appendingPathComponent("config.toml")
        try SharedModelFolders.write(
            primary: scratch.appendingPathComponent("primary").path,
            additional: [ssd.path],
            to: shared,
            environment: ["HOME": scratch.path]
        )
        let loaded = try MTPLXSettingsStore(settingsURL: settings, modelFolderSharing: .at(shared)).load()
        XCTAssertEqual(loaded.additionalModelDirectories, [ssd.path])
        XCTAssertTrue(
            loaded.modelLibrary.candidatePaths(for: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")
                .map { ModelLibrary.canonicalURL(for: $0).path }
                .contains(ModelLibrary.canonicalURL(for: pack.path).path),
            "the shared folder is searched for the picked pack"
        )
    }
}
