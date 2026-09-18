import Foundation

// MARK: - LaunchTarget
//
// Mirrors `mtplx start <target>` — the user picks "where am I serving
// MTPLX to?" at the Play button. The picker drives a CommandBuilder
// preset (scheduler/batching/SSD knobs tuned for the surface) so the
// app's Play flow matches the CLI start flow.
//
// Skipping `swival` per user request.

public enum LaunchTarget: String, Codable, CaseIterable, Identifiable, Sendable {
    case chat        // In-app chat surface owned by MTPLXApp (native tool
                     // prompt, reasoning visible, web-search + file-attach
                     // tools, conversation persistence)
    case openWebUI   // Open WebUI streaming server
    case pi          // Pi terminal
    case omp         // Oh My Pi terminal
    case openCode    // OpenCode Desktop D3 MTP; reasoning follows MTPLXApp
    case hermes      // Hermes Agent launched in Terminal against MTPLX
    case benchmark   // AIME 2026 native benchmark overlay. The overlay
                     // starts a benchmark-tuned daemon if none is
                     // reachable, then runs against that daemon.
    case other       // Custom client (OpenAI- or Anthropic-compatible)

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .chat: return tr("Chat")
        case .openWebUI: return tr("Web UI")
        case .pi: return tr("Pi")
        case .omp: return tr("OMP")
        case .openCode: return tr("OpenCode")
        case .hermes: return tr("Hermes")
        case .benchmark: return tr("Benchmark")
        case .other: return tr("Other")
        }
    }

    /// One-line tagline shown in the picker.
    public var tagline: String {
        switch self {
        case .chat:
            return tr("Chat right here with web search and file uploads.")
        case .openWebUI:
            return tr("Chat in your browser.")
        case .pi:
            return tr("Use Pi in the terminal.")
        case .omp:
            return tr("Use Oh My Pi in the terminal.")
        case .openCode:
            return tr("Use OpenCode Desktop, powered by MTPLX.")
        case .hermes:
            return tr("Use Hermes Desktop, powered by MTPLX — terminal, file, web, browser, and messaging tools.")
        case .benchmark:
            return tr("Run AIME 2026.")
        case .other:
            return tr("Connect Cursor, Codex, Claude Code, or any compatible app.")
        }
    }

    /// SF Symbol identifier. Picked so every glyph reads at the same
    /// visual weight as a single-stroke monochrome shape:
    ///   - chat → single message bubble (`text.bubble`) — the dual
    ///     bubbles SF Symbol was visually heavier than its siblings.
    ///   - openCode → `square` is an outlined frame (the SF Symbol
    ///     equivalent of a square donut — outer thick stroke, empty
    ///     centre), matching the OpenCode mark.
    ///   - pi → bare mathematical π.
    ///   - benchmark → `function` reads as "this surface measures
    ///     mathematical reasoning"; the bullseye / target glyphs
    ///     visually clashed with the other launch icons.
    ///   - other → curly-braces ellipsis to suggest "custom client".
    public var systemImage: String {
        switch self {
        case .chat: return "text.bubble"
        case .openWebUI: return "globe"
        case .pi: return "pi"
        case .omp: return "option"
        case .openCode: return "square"
        case .hermes: return "sparkles"  // fallback only; launch picker renders the real Hermes logo
        case .benchmark: return "function"
        case .other: return "ellipsis.curlybraces"
        }
    }

    /// Whether picking this target opens an inline sub-form before
    /// starting the daemon. `.other` expands to ask for port + API key
    /// and shows the OpenAI / Anthropic endpoint URLs.
    public var hasInlineForm: Bool {
        self == .other
    }

    /// Whether picking this target spawns/connects a daemon target
    /// preset.
    public var spawnsDaemon: Bool {
        true
    }
}
