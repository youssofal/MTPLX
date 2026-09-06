import XCTest
import MTPLXAppCore

@testable import MTPLXAppHost

final class ModelPickerOverlayTests: XCTestCase {
    func testMissingPersistedCustomModelCanBeRemoved() throws {
        let option = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "Foo/Bar"))
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertFalse(row.isInstalled)
        XCTAssertTrue(row.canRemoveFromPicker)
    }

    func testInstalledPersistedCustomModelCannotBeRemoved() throws {
        let directory = temporaryDirectory().appendingPathComponent("installed-model", isDirectory: true)
        try makeCompleteModelFolder(at: directory)
        defer { try? FileManager.default.removeItem(at: directory) }

        let option = MTPLXModelOption(
            id: "custom-installed",
            displayName: "Installed Custom",
            shortName: "Installed Custom",
            detail: "Test model",
            hfModelID: "Foo/Installed",
            localCandidates: [directory.path]
        )
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertTrue(row.isInstalled)
        XCTAssertFalse(row.canRemoveFromPicker)
    }

    func testOfficialModelCannotBeRemoved() throws {
        let option = try XCTUnwrap(MTPLXModelOption.officialCatalog.first)
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: "Other/Model",
            customModels: [option]
        )

        XCTAssertFalse(row.canRemoveFromPicker)
    }

    func testCurrentSynthesizedModelCannotBeRemovedAndRemainsRepresentable() throws {
        let option = try XCTUnwrap(MTPLXModelOption.customHuggingFaceModel(repoID: "Foo/Current"))
        let row = ModelPickerPreparedOption(
            option: option,
            currentModel: option.hfModelID,
            customModels: []
        )

        XCTAssertTrue(row.selected)
        XCTAssertFalse(row.canRemoveFromPicker)
        XCTAssertEqual(row.resolvedReference, option.hfModelID)
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-model-picker-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func makeCompleteModelFolder(at folder: URL) throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try "{}".write(to: folder.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: folder.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: folder.appendingPathComponent("model.safetensors"))
    }
}