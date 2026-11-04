import XCTest
import Foundation

/// Guards the app bundle's two release stamps.
///
/// - CFBundleVersion: Sparkle ranks updates by it alone. 2.12.2 shipped as
///   2012011 while the derivation gave 2.12.3 2012003, so 2.12.2 users would
///   never have been offered 2.12.3. `script/app_build_number.sh` is the one
///   derivation; these tests pin its order against what shipped.
/// - The SDK stamp: Xcode 27's `swift build` stamped the binary's SDK with
///   the deployment target (14.0) and macOS drew the app without Liquid
///   Glass. The build script passes the real SDK to the linker and refuses a
///   binary stamped below 26.
final class AppBuildScriptTests: XCTestCase {
    /// apps/MTPLXApp/Tests/MTPLXAppCoreTests/<this file> -> apps/MTPLXApp.
    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var repoRoot: URL {
        packageRoot.deletingLastPathComponent().deletingLastPathComponent()
    }

    private struct Run {
        let status: Int32
        let stdout: String
    }

    private func buildNumber(_ arguments: [String]) throws -> Run {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [Self.packageRoot.appendingPathComponent("script/app_build_number.sh").path] + arguments
        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        try process.run()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return Run(
            status: process.terminationStatus,
            stdout: String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    private func number(_ arguments: String...) throws -> Int {
        let run = try buildNumber(arguments)
        XCTAssertEqual(run.status, 0, "app_build_number.sh \(arguments)")
        return try XCTUnwrap(Int(run.stdout), "app_build_number.sh \(arguments) printed \(run.stdout)")
    }

    // MARK: CFBundleVersion

    func testTheReleaseOf2123RanksAboveEveryShipped2122Build() throws {
        // 2.12.0 to 2.12.2 shipped as 2012009, 2012010 and 2012011.
        let shipped2122 = 2_012_011
        XCTAssertEqual(try number("2.12.3"), 2_012_030)
        XCTAssertEqual(try number("2.12.3", "1"), 2_012_031)
        XCTAssertGreaterThanOrEqual(try number("2.12.3", "1"), 2_012_012)
        // A QA build of 2.12.3 already ranks above 2.12.2, so it is never
        // offered 2.12.2 as an update, and below its own release.
        XCTAssertGreaterThan(try number("2.12.3"), shipped2122)
        XCTAssertLessThan(try number("2.12.3"), try number("2.12.3", "1"))
    }

    func testEveryBuildOfANewerVersionRanksAboveEveryBuildOfAnOlderOne() throws {
        let versions = ["2.12.3", "2.12.4", "2.12.10", "2.12.99", "2.13.0", "2.99.99", "3.0.0", "3.0.1"]
        for (older, newer) in zip(versions, versions.dropFirst()) {
            XCTAssertLessThan(try number(older, "9"), try number(newer, "0"), "\(older) -> \(newer)")
        }
        // A dev build's dotted suffix ranks with its base version, as before.
        XCTAssertEqual(try number("2.12.4.dev0"), try number("2.12.4"))
    }

    func testMalformedOrOverflowingVersionsFailClosed() throws {
        for arguments in [["2.12"], ["2.12.3rc1"], ["v2.12.3"], ["2.12.100"], ["2.1000.0"], ["2.12.3", "10"], ["2.12.3", "x"], []] {
            let run = try buildNumber(arguments)
            XCTAssertNotEqual(run.status, 0, "\(arguments) printed \(run.stdout)")
            XCTAssertEqual(run.stdout, "", "\(arguments)")
        }
    }

    func testReleaseAndRehearsalScriptsDeriveTheNumberInsteadOfTypingIt() throws {
        let release = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("scripts/release_macos_v1.sh"),
            encoding: .utf8
        )
        XCTAssertNil(
            release.range(of: #"MTPLX_RELEASE_BUILD:-[0-9]"#, options: .regularExpression),
            "the release must derive its build number, not default to a typed one"
        )
        XCTAssertTrue(release.contains("app_build_number.sh"))
        let rehearsal = try String(
            contentsOf: Self.repoRoot.appendingPathComponent("scripts/sparkle_rehearsal_kit.sh"),
            encoding: .utf8
        )
        XCTAssertTrue(rehearsal.contains("app_build_number.sh"))
        let build = try String(
            contentsOf: Self.packageRoot.appendingPathComponent("script/build_and_run.sh"),
            encoding: .utf8
        )
        XCTAssertTrue(build.contains(#"script/app_build_number.sh" "$APP_VERSION""#))
    }

    // MARK: SDK stamp

    func testTheBuildPassesTheRealSDKToTheLinkerAndGatesTheStamp() throws {
        let build = try String(
            contentsOf: Self.packageRoot.appendingPathComponent("script/build_and_run.sh"),
            encoding: .utf8
        )
        XCTAssertTrue(build.contains(#"MACOS_SDK_VERSION="$(/usr/bin/xcrun --show-sdk-version)""#))
        XCTAssertTrue(build.contains("-Xlinker -platform_version -Xlinker macos"))
        XCTAssertTrue(build.contains(#"-Xlinker "$MACOS_DEPLOYMENT_TARGET" -Xlinker "$MACOS_SDK_VERSION""#))
        XCTAssertTrue(build.contains(#"/usr/bin/vtool -show-build "$BUNDLE_DIR/Contents/MacOS/$APP_NAME""#))
        XCTAssertTrue(build.contains("STAMPED_SDK_MAJOR < 26"))
        // One deployment target for the plist and the linker, matching the
        // package's platform.
        XCTAssertTrue(build.contains(#"MACOS_DEPLOYMENT_TARGET="14.0""#))
        XCTAssertTrue(build.contains("<string>$MACOS_DEPLOYMENT_TARGET</string>"))
        let manifest = try String(contentsOf: Self.packageRoot.appendingPathComponent("Package.swift"), encoding: .utf8)
        XCTAssertTrue(manifest.contains(".macOS(.v14)"))
    }

    // MARK: Build-machine rpaths

    /// Xcode 27's build adds a back-deploy rpath into every toolchain,
    /// including the Metal toolchain's /var/run cryptex mount. The script
    /// drops them and refuses any other absolute path left in the binary.
    func testToolchainRpathsAreStrippedAndAnyOtherAbsoluteRpathFailsTheBuild() throws {
        let build = try String(
            contentsOf: Self.packageRoot.appendingPathComponent("script/build_and_run.sh"),
            encoding: .utf8
        )
        XCTAssertTrue(build.contains(#""$ROOT/script/strip_build_rpaths.sh" "$BUNDLE_DIR/Contents/MacOS/$APP_NAME""#))

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-rpaths-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let xcodeRpath = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift-6.2/macosx"
        let metalRpath = "/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.TEST01/Metal.xctoolchain/usr/lib/swift-6.2/macosx"
        let kept = ["/usr/lib/swift", "@loader_path", "@executable_path/../Frameworks"]

        let clean = try linkBinary(in: scratch, named: "clean", rpaths: ["/usr/lib/swift", "@loader_path", metalRpath, xcodeRpath, "@executable_path/../Frameworks"])
        XCTAssertEqual(try run("/bin/bash", [Self.packageRoot.appendingPathComponent("script/strip_build_rpaths.sh").path, clean.path]).status, 0)
        XCTAssertEqual(try rpaths(of: clean), kept)

        let stray = try linkBinary(in: scratch, named: "stray", rpaths: kept + ["/opt/local/lib"])
        let refused = try run("/bin/bash", [Self.packageRoot.appendingPathComponent("script/strip_build_rpaths.sh").path, stray.path])
        XCTAssertNotEqual(refused.status, 0, "an absolute rpath outside the toolchains must fail the build")
    }

    private func linkBinary(in directory: URL, named name: String, rpaths: [String]) throws -> URL {
        let source = directory.appendingPathComponent("\(name).c")
        try "int main(void) { return 0; }\n".write(to: source, atomically: true, encoding: .utf8)
        let output = directory.appendingPathComponent(name)
        let linked = try run("/usr/bin/xcrun", ["clang", source.path, "-o", output.path] + rpaths.flatMap { ["-Wl,-rpath,\($0)"] })
        guard linked.status == 0 else { throw XCTSkip("clang is not available to link a test binary") }
        return output
    }

    private func rpaths(of binary: URL) throws -> [String] {
        let loadCommands = try run("/usr/bin/otool", ["-l", binary.path]).stdout.components(separatedBy: "\n")
        var found: [String] = []
        for (index, line) in loadCommands.enumerated() where line.trimmingCharacters(in: .whitespaces) == "cmd LC_RPATH" {
            guard let pathLine = loadCommands.dropFirst(index + 1).first(where: { $0.contains(" path ") }) else { continue }
            let path = pathLine.trimmingCharacters(in: .whitespaces).dropFirst("path ".count)
            found.append(String(path.components(separatedBy: " (offset").first ?? ""))
        }
        return found
    }

    private func run(_ executable: String, _ arguments: [String]) throws -> Run {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        try process.run()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return Run(status: process.terminationStatus, stdout: String(decoding: data, as: UTF8.self))
    }
}
