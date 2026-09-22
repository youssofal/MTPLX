import Foundation

/// A Splash package as the model picker presents it.
///
/// Splash loads only its own packages, so it has a short fixed catalog of its
/// own instead of sharing the MLX one, which is built by scanning checkpoint
/// folders. The shape mirrors what the picker needs from an `MTPLXModelOption`
/// (name, one-line detail, size, on-disk state) so both engines render through
/// the same row.
public struct SplashPackageOption: Identifiable, Equatable, Sendable {
    /// The Hugging Face repo id, which is also what `--model` takes.
    public let id: String
    public let displayName: String
    public let detail: String
    public let downloadBytes: Int64

    /// Speeds below are Inco's published single-request figures on a 48 GB
    /// M5 Pro (https://inco.ai/blog/splash/). They are quoted as ceilings
    /// ("up to"), because this Mac is not that Mac.
    public static let catalog: [SplashPackageOption] = [
        SplashPackageOption(
            id: "incoai/Qwen3.8-27B-Splash",
            displayName: "Qwen 3.8 27B Splash",
            detail: tr("Dense 27B for coding and agents. 4-bit with a trained DFlash 2 draft: about 2x the decode of the next-fastest engine."),
            downloadBytes: 17_400_000_000
        ),
        SplashPackageOption(
            id: "incoai/Qwen3.6-35B-A3B-Splash",
            displayName: "Qwen 3.6 35B-A3B Splash",
            detail: tr("The fastest: up to 2,000 tok/s prefill and 210 tok/s decode. 4-bit MoE with 3B active and a DFlash 2 draft."),
            downloadBytes: 20_900_000_000
        ),
    ]

    public static func option(for identifier: String) -> SplashPackageOption? {
        catalog.first { $0.id == identifier }
    }

    /// Where Splash keeps a verified package; see its install/paths.py.
    public var installURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Splash/models", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
    }

    /// True once Splash has verified the package against its manifest.
    ///
    /// A plain file check, deliberately: the picker renders on every open and
    /// must not shell out to find this.
    public var isInstalled: Bool {
        FileManager.default.fileExists(
            atPath: installURL.appendingPathComponent("manifest.json").path
        )
    }

    public var downloadSizeLabel: String {
        String(format: "%.1f GB", Double(downloadBytes) / 1e9)
    }
}
