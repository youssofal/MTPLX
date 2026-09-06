import Foundation
import XCTest
@testable import MTPLXAppCore

final class MTPLXSkillStoreTests: XCTestCase {
    func testDiscoversWorkspaceSkillWithReferencesScriptsAndHash() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-skill-(UUID().uuidString)")
        let skill = root.appendingPathComponent("skills/reviewer")
        try FileManager.default.createDirectory(
            at: skill.appendingPathComponent("references"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: skill.appendingPathComponent("scripts"),
            withIntermediateDirectories: true
        )
        try Data("# Reviewer\nInspect the diff.".utf8)
            .write(to: skill.appendingPathComponent("SKILL.md"))
        try Data("reference".utf8)
            .write(to: skill.appendingPathComponent("references/checklist.md"))
        try Data("#!/bin/sh".utf8)
            .write(to: skill.appendingPathComponent("scripts/check.sh"))
        defer { try? FileManager.default.removeItem(at: root) }

        let definition = try XCTUnwrap(
            MTPLXSkillStore(workspaceRoots: [root.path]).load(named: "reviewer")
        )
        XCTAssertEqual(definition.summary, "Reviewer")
        XCTAssertEqual(definition.references.count, 1)
        XCTAssertEqual(definition.scripts.count, 1)
        XCTAssertFalse(definition.sha256.isEmpty)
        XCTAssertTrue(
            MTPLXSkillStore(workspaceRoots: [root.path])
                .promptContext()?
                .contains("reviewer") == true
        )
    }
}
