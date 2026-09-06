import Foundation
import XCTest
@testable import MTPLXAppCore

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { value } else { nil }
    }
    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { value } else { nil }
    }
}

final class ClientVisionIntegrationTests: XCTestCase {
    func testAllClientWritersUpgradeVisionFromTheSamePackMetadata() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let pack = root.appendingPathComponent("Qwen3.8-Flash-Next-MTPLX-Optimized-Speed")
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        try Data(#"{"weight_map":{"vision_tower.patch_embed.weight":"model-vision.safetensors"}}"#.utf8)
            .write(to: pack.appendingPathComponent("model.safetensors.index.json"))
        let ocURL = root.appendingPathComponent("opencode/config.json")
        let piURL = root.appendingPathComponent("pi/models.json")
        let hermesHome = root.appendingPathComponent("hermes")
        let oc = OpenCodeIntegration(configURL: ocURL, desktopSettingsStoreURL: root.appendingPathComponent("desktop.dat"))
        let pi = PiIntegration(configURL: piURL)
        let hermes = HermesIntegration(hermesHome: hermesHome,
            environment: ["HOME": root.path, "PATH": "/usr/bin:/bin"],
            terminalCommandURL: root.appendingPathComponent("hermes.command"))
        let config = MTPLXAppConfiguration(model: pack.path, host: "127.0.0.1", port: 8000, contextWindow: 262144)
        for vision in [false, true] {
            try Data((vision ? #"{"vision_config":{}}"# : "{}").utf8).write(to: pack.appendingPathComponent("config.json"))
            _ = try oc.sync(configuration: config)
            _ = try pi.sync(configuration: config)
            _ = try hermes.sync(configuration: config)
            let ocRoot = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: ocURL))
            let ocModel = try XCTUnwrap(ocRoot["provider"]?.objectValue?["mtplx"]?.objectValue?["models"]?.objectValue?.values.first?.objectValue)
            XCTAssertEqual(ocModel["modalities"]?.objectValue?["input"]?.arrayValue?.contains(.string("image")), vision)
            let piRoot = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: piURL))
            let piModel = try XCTUnwrap(piRoot["providers"]?.objectValue?["mtplx"]?.objectValue?["models"]?.arrayValue?.first?.objectValue)
            XCTAssertEqual(piModel["input"]?.arrayValue?.contains(.string("image")), vision)
            let yaml = try String(contentsOf: hermesHome.appendingPathComponent("profiles/mtplx/config.yaml"), encoding: .utf8)
            XCTAssertTrue(yaml.contains("supports_vision: \(vision)"))
        }
    }
}
