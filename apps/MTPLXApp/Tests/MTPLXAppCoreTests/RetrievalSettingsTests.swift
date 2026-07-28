import XCTest

@testable import MTPLXAppCore

/// The retrieval endpoints are configured from the app, so the settings have to
/// survive a round trip through `settings.json` and reach the daemon's argv.
/// Without these, a user could set an embedding model in the UI and get a
/// chat-only server back with no error anywhere.
final class RetrievalSettingsTests: XCTestCase {
    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-retrieval-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeExecutable(named name: String) throws -> URL {
        let directory = temporaryDirectory()
        let url = directory.appendingPathComponent(name)
        try "#!/bin/sh\nexit 0\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func makeBuilder() throws -> MTPLXCommandBuilder {
        let fake = try makeExecutable(named: "mtplx")
        return MTPLXCommandBuilder(environment: [
            "PATH": fake.deletingLastPathComponent().path,
            "HOME": temporaryDirectory().path,
        ])
    }

    // MARK: - Persistence

    func testRetrievalSettingsDefaultToChatOnly() {
        let configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        XCTAssertTrue(configuration.embeddingModels.isEmpty)
        XCTAssertTrue(configuration.rerankerModels.isEmpty)
        XCTAssertEqual(configuration.retrievalMaxResident, 2)
    }

    func testRetrievalSettingsRoundTripThroughJSON() throws {
        var configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        configuration.embeddingModels = ["org/embed", "/models/local-embed=fast"]
        configuration.rerankerModels = ["org/rank"]
        configuration.retrievalMaxResident = 4

        let data = try JSONEncoder().encode(configuration)
        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)

        XCTAssertEqual(decoded.embeddingModels, ["org/embed", "/models/local-embed=fast"])
        XCTAssertEqual(decoded.rerankerModels, ["org/rank"])
        XCTAssertEqual(decoded.retrievalMaxResident, 4)
    }

    func testSettingsWrittenByAnOlderBuildStillDecode() throws {
        // A settings.json from before the retrieval feature has none of these
        // keys; decoding must fall back to the defaults rather than throwing.
        let json = """
        {"model":"/models/qwen","profile":"sustained","host":"127.0.0.1","port":8000}
        """
        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
        XCTAssertTrue(decoded.embeddingModels.isEmpty)
        XCTAssertTrue(decoded.rerankerModels.isEmpty)
        XCTAssertEqual(decoded.retrievalMaxResident, 2)
    }

    func testRetrievalKeysUseSnakeCaseInSettingsJSON() throws {
        var configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        configuration.embeddingModels = ["org/embed"]
        let data = try JSONEncoder().encode(configuration)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("embedding_models"), text)
        XCTAssertTrue(text.contains("reranker_models"), text)
        XCTAssertTrue(text.contains("retrieval_max_resident"), text)
    }

    // MARK: - Daemon arguments

    func testChatOnlyConfigurationPassesNoRetrievalFlags() throws {
        let builder = try makeBuilder()
        let command = try builder.buildServeCommand(
            configuration: MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        )
        XCTAssertFalse(command.arguments.contains("--embedding-model"), command.arguments.joined(separator: " "))
        XCTAssertFalse(command.arguments.contains("--reranker-model"), command.arguments.joined(separator: " "))
        XCTAssertFalse(command.arguments.contains("--retrieval-max-resident"), command.arguments.joined(separator: " "))
    }

    func testConfiguredRetrievalModelsReachTheDaemonArguments() throws {
        let builder = try makeBuilder()
        var configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        configuration.embeddingModels = ["org/embed", "/models/local=fast"]
        configuration.rerankerModels = ["org/rank"]
        configuration.retrievalMaxResident = 3

        let command = try builder.buildServeCommand(configuration: configuration)
        let arguments = command.arguments

        let embeddingValues = arguments.enumerated()
            .filter { $0.element == "--embedding-model" }
            .map { arguments[$0.offset + 1] }
        XCTAssertEqual(embeddingValues, ["org/embed", "/models/local=fast"])

        let rerankValues = arguments.enumerated()
            .filter { $0.element == "--reranker-model" }
            .map { arguments[$0.offset + 1] }
        XCTAssertEqual(rerankValues, ["org/rank"])

        guard let residentIndex = arguments.firstIndex(of: "--retrieval-max-resident") else {
            return XCTFail("resident cap missing: \(arguments.joined(separator: " "))")
        }
        XCTAssertEqual(arguments[residentIndex + 1], "3")
    }

    func testBlankModelEntriesAreNotPassedAsArguments() throws {
        // The settings field is free text, so stray blank lines are expected;
        // forwarding one would make the daemon fail to start.
        let builder = try makeBuilder()
        var configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        configuration.embeddingModels = ["org/embed", "   ", ""]

        let command = try builder.buildServeCommand(configuration: configuration)
        let embeddingValues = command.arguments.enumerated()
            .filter { $0.element == "--embedding-model" }
            .map { command.arguments[$0.offset + 1] }
        XCTAssertEqual(embeddingValues, ["org/embed"])
    }

    func testResidentCapIsClampedToAtLeastOne() throws {
        let builder = try makeBuilder()
        var configuration = MTPLXAppConfiguration(model: "/models/qwen", profile: "sustained")
        configuration.rerankerModels = ["org/rank"]
        configuration.retrievalMaxResident = 0

        let command = try builder.buildServeCommand(configuration: configuration)
        guard let index = command.arguments.firstIndex(of: "--retrieval-max-resident") else {
            return XCTFail("resident cap missing")
        }
        XCTAssertEqual(command.arguments[index + 1], "1")
    }
}
