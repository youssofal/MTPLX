import XCTest
@testable import MTPLXAppCore

final class ChatSlashCommandsTests: XCTestCase {
    func testParsesReasoningAndThinkAlias() throws {
        let reasoning = try XCTUnwrap(ChatSlashCommands.parse("/reasoning high"))
        XCTAssertEqual(reasoning.definition.name, "reasoning")
        XCTAssertEqual(reasoning.argument, "high")

        let think = try XCTUnwrap(ChatSlashCommands.parse("/think medium"))
        XCTAssertEqual(think.definition.name, "reasoning")
        XCTAssertEqual(think.argument, "medium")
    }

    func testSuggestionsMatchCommandNameAndTitle() {
        XCTAssertTrue(ChatSlashCommands.suggestions(for: "/me").contains { $0.name == "memory" })
        XCTAssertTrue(ChatSlashCommands.suggestions(for: "/plan").contains { $0.name == "plan" })
        XCTAssertTrue(ChatSlashCommands.suggestions(for: "/").contains { $0.name == "mcp" })
    }

    func testUnknownAndNaturalLanguageInputDoNotBecomeCommands() {
        XCTAssertNil(ChatSlashCommands.parse("/not-a-command"))
        XCTAssertNil(ChatSlashCommands.parse("please /help"))
        XCTAssertEqual(ChatSlashCommands.parse("/help now please")?.argument, "now please")
    }

    func testHelpContainsHighValueNativeControls() {
        let help = ChatSlashCommands.helpText
        XCTAssertTrue(help.contains("/goal"))
        XCTAssertTrue(help.contains("/mcp"))
        XCTAssertTrue(help.contains("/status"))
        XCTAssertTrue(help.contains("/usage"))
        XCTAssertTrue(help.contains("/feedback"))
    }
}
