import Foundation
import XCTest
@testable import MTPLXAppCore

/// #466: the wizard measured free space on the home volume, so a model store
/// redirected to an external drive was refused as "insufficient space".
final class ModelStoreVolumeTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelStoreVolumeTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testMissingModelsDirectoryMeasuresTheNearestExistingAncestor() throws {
        let base = try makeTempDir()
        let missing = base
            .appendingPathComponent("first-launch", isDirectory: true)
            .appendingPathComponent(".mtplx", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)

        let measured = ModelStoreVolume.measurementURL(for: missing)

        XCTAssertEqual(measured.path, base.resolvingSymlinksInPath().path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path), "must not create the store")
    }

    func testSymlinkedModelsDirectoryMeasuresItsTarget() throws {
        let base = try makeTempDir()
        let external = base.appendingPathComponent("external-drive", isDirectory: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        let link = base.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: external)

        let measured = ModelStoreVolume.measurementURL(for: link)

        XCTAssertEqual(measured.path, external.resolvingSymlinksInPath().path)
    }

    func testModelDirOverrideDrivesTheMeasurement() throws {
        let base = try makeTempDir()
        let store = base.appendingPathComponent("store", isDirectory: true)
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)

        let root = ModelDownloader.defaultCacheRoot(env: ["MTPLX_MODEL_DIR": store.path])
        XCTAssertEqual(root.path, store.path)
        XCTAssertGreaterThan(ModelStoreVolume.freeGiB(env: ["MTPLX_MODEL_DIR": store.path]), 0)
    }

    func testDefaultStoreReportsPositiveFreeSpace() {
        XCTAssertGreaterThan(ModelStoreVolume.freeGiB(env: ["HOME": NSHomeDirectory()]), 0)
    }
}
