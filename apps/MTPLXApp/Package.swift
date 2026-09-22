// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MTPLXApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MTPLXAppCore", targets: ["MTPLXAppCore"]),
        .executable(name: "MTPLXApp", targets: ["MTPLXAppHost"]),
    ],
    dependencies: [
        // Markdown renderer for settled chat replies. Live generation stays
        // plain text so token streaming does not pay a per-delta parse cost.
        .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.4.1"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.9.2"),
    ],
    targets: [
        .target(
            name: "MTPLXAppCore",
            path: "Sources/MTPLXAppCore",
            exclude: ["Resources"]
        ),
        .executableTarget(
            name: "MTPLXAppHost",
            dependencies: [
                "MTPLXAppCore",
                .product(name: "MarkdownUI", package: "swift-markdown-ui"),
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/MTPLXAppHost",
            exclude: ["Resources"]
        ),
        .testTarget(
            name: "MTPLXAppCoreTests",
            dependencies: ["MTPLXAppCore"],
            path: "Tests/MTPLXAppCoreTests",
            // Real /health, /snapshot and /settings responses captured from a
            // live `mtplx serve --engine splash`, so the decoding tests run
            // against what the daemon actually sends rather than a guess.
            resources: [.process("Fixtures")]
        ),
        .testTarget(
            name: "MTPLXAppHostTests",
            dependencies: ["MTPLXAppHost"],
            path: "Tests/MTPLXAppHostTests"
        ),
    ]
)
