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
    case openCode    // OpenCode Desktop D3 MTP; reasoning follows MTPLXApp
    case hermes      // Hermes Agent launched in Terminal against MTPLX
    case claudeCode  // Claude Code CLI in Terminal; MTPLX-owned --settings
                     // file and per-launch env, the user's setup untouched
    case codex       // Codex app as MTPLX's own instance when installed,
                     // else the Codex CLI in Terminal (-c overrides)
    case deepseekHarness // DeepSeek Harness: its browser app by default, or
                     // its terminal app; MTPLX's own harness home with its
                     // telemetry off (mtplx connect deepseek-harness)
    case benchmark   // AIME 2026 native benchmark overlay. The overlay
                     // starts a benchmark-tuned daemon if none is
                     // reachable, then runs against that daemon.
    case other       // Custom client (OpenAI- or Anthropic-compatible)

    public var id: String { rawValue }

    // MARK: Release switch

    /// Claude Code, Codex and DeepSeek Harness arrive in MTPLX 3 (the founder, 22 November 2026). Off, the
    /// Start button, the client-opening settings and the settings search show none of them, and a saved choice
    /// of one falls back to Chat; their code stays. The CLI's twin is `V3_AGENT_CLIENTS` in
    /// mtplx/release_gates.py: turn both on together.
    public static let v3AgentClients = false

    /// Whether this release offers the target to pick (tests pass the switch to cover both states).
    public func isOffered(v3AgentClients: Bool = LaunchTarget.v3AgentClients) -> Bool {
        switch self {
        case .claudeCode, .codex, .deepseekHarness: return v3AgentClients
        default: return true
        }
    }

    /// The targets the Start button lists, in their order.
    public static func offered(v3AgentClients: Bool = LaunchTarget.v3AgentClients) -> [LaunchTarget] {
        allCases.filter { $0.isOffered(v3AgentClients: v3AgentClients) }
    }

    public var title: String {
        switch self {
        case .chat: return tr("Chat")
        case .openWebUI: return tr("Web UI")
        case .pi: return tr("Pi")
        case .openCode: return tr("OpenCode")
        case .hermes: return tr("Hermes")
        case .claudeCode: return tr("Claude Code")
        case .codex: return tr("Codex")
        case .deepseekHarness: return tr("DeepSeek Harness")
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
        case .openCode:
            return ClientOpenings.currentSurface(.openCode) == ClientOpenings.terminalChoice
                ? tr("Use OpenCode in the terminal, powered by MTPLX.")
                : tr("Use OpenCode Desktop, powered by MTPLX.")
        case .hermes:
            return ClientOpenings.currentSurface(.hermes) == ClientOpenings.terminalChoice
                ? tr("Use Hermes in the terminal, powered by MTPLX, with file, web, browser and messaging tools.")
                : tr("Use Hermes Desktop, powered by MTPLX — terminal, file, web, browser, and messaging tools.")
        case .claudeCode:
            return tr("Use Claude Code in the terminal, powered by MTPLX.")
        case .codex:
            // The Codex app as MTPLX's own instance when installed and not
            // set to a terminal, else the Codex CLI in the picked terminal
            // (CodexDesktopIntegration, ClientOpenings). The copy is a second
            // app in the Dock with its own history: say so up front.
            return ClientOpenings.currentSurface(.codex) == ClientOpenings.mtplxApp
                && CodexDesktopIntegration.findApp()?.supported == true
                ? tr("Opens a separate copy of the Codex app with its own history, powered by MTPLX.")
                : tr("Use Codex in the terminal, powered by MTPLX.")
        case .deepseekHarness:
            // Its browser app by default (`dsh web`, which MTPLX starts and
            // stops). With a terminal picked, the same browser app is started
            // from that terminal so its log shows there: 0.1.2 has no
            // terminal interface of its own.
            return ClientOpenings.currentSurface(.deepseekHarness) == ClientOpenings.terminalChoice
                ? tr("Starts DeepSeek Harness from the terminal and opens it in your browser, powered by MTPLX, telemetry off.")
                : tr("Use DeepSeek Harness in your browser, powered by MTPLX, telemetry off.")
        case .benchmark:
            return tr("Run AIME 2026.")
        case .other:
            return tr("Connect Cursor, Cline, or any compatible app.")
        }
    }

    /// SF Symbol identifier, shown where the target has no drawn mark (the
    /// app draws Claude Code, Codex, OpenCode and Hermes as one-colour
    /// monochrome artwork, `ClientMark`). Picked so every
    /// glyph reads at the same visual weight as a single-stroke
    /// monochrome shape:
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
    ///   - claudeCode → asterisk; codex → terminal prompt.
    public var systemImage: String {
        switch self {
        case .chat: return "text.bubble"
        case .openWebUI: return "globe"
        case .pi: return "pi"
        case .openCode: return "square"
        case .hermes: return "sparkles"
        case .claudeCode: return "asterisk"
        case .codex: return "terminal"
        case .deepseekHarness: return "fish"
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
