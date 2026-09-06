import Foundation

public struct ChatSlashCommandDefinition: Identifiable, Hashable, Sendable {
    public let name: String
    public let aliases: [String]
    public let title: String
    public let detail: String
    public let icon: String

    public var id: String { name }

    public init(
        name: String,
        aliases: [String] = [],
        title: String,
        detail: String,
        icon: String
    ) {
        self.name = name
        self.aliases = aliases
        self.title = title
        self.detail = detail
        self.icon = icon
    }

    public var commandText: String { "/\(name)" }
}

public struct ParsedChatSlashCommand: Hashable, Sendable {
    public let definition: ChatSlashCommandDefinition
    public let argument: String

    public init(definition: ChatSlashCommandDefinition, argument: String) {
        self.definition = definition
        self.argument = argument
    }
}

public enum ChatSlashCommands {
    /// The small, high-value native command set. It mirrors the useful
    /// desktop controls in the reference UI while keeping model generation
    /// separate from local app actions.
    public static let definitions: [ChatSlashCommandDefinition] = [
        .init(name: "help", aliases: ["?"], title: "Help", detail: "Show native chat commands", icon: "questionmark.circle"),
        .init(name: "new", aliases: [], title: "New chat", detail: "Start a fresh conversation", icon: "plus.bubble"),
        .init(name: "clear", aliases: [], title: "Clear chat", detail: "Remove messages from this conversation", icon: "trash"),
        .init(name: "model", aliases: [], title: "Model", detail: "Show or choose the loaded MTPLX model", icon: "cube"),
        .init(name: "plan", aliases: [], title: "Plan mode", detail: "Toggle planning mode for this chat", icon: "lightbulb"),
        .init(name: "reasoning", aliases: ["think"], title: "Reasoning", detail: "Set reasoning effort: auto, low, medium, or high", icon: "brain"),
        .init(name: "goal", aliases: [], title: "Goal", detail: "Set the active goal for this conversation", icon: "target"),
        .init(name: "memory", aliases: ["memories"], title: "Memories", detail: "Search the local MTPLX memory store", icon: "brain.head.profile"),
        .init(name: "workspace", aliases: [], title: "Workspace", detail: "Show or select the active local project", icon: "folder"),
        .init(name: "files", aliases: [], title: "Files", detail: "List files in the active workspace", icon: "doc.on.doc"),
        .init(name: "diff", aliases: [], title: "Diff", detail: "Show the active workspace Git diff", icon: "arrow.left.arrow.right"),
        .init(name: "test", aliases: [], title: "Test", detail: "Run the workspace test command with approval", icon: "checkmark.circle"),
        .init(name: "run", aliases: [], title: "Run", detail: "Run an explicit workspace command with approval", icon: "terminal"),
        .init(name: "resume", aliases: [], title: "Resume", detail: "Reload the latest durable run timeline", icon: "arrow.clockwise"),
        .init(name: "fork", aliases: [], title: "Fork", detail: "Create a new conversation from this chat", icon: "arrow.branch"),
        .init(name: "stop", aliases: [], title: "Stop", detail: "Stop the active model or agent run", icon: "stop.circle"),
        .init(name: "retry", aliases: [], title: "Retry", detail: "Retry the last failed model or delegated agent run", icon: "arrow.clockwise.circle"),
        .init(name: "review", aliases: [], title: "Delegate reviewer", detail: "Run a read-only reviewer in an isolated worktree", icon: "person.2"),
        .init(name: "skills", aliases: [], title: "Skills", detail: "List reusable local agent workflows", icon: "wand.and.stars"),
        .init(name: "mcp", aliases: [], title: "MCP and tools", detail: "Show the tools available to this chat", icon: "link"),
        .init(name: "status", aliases: [], title: "Status", detail: "Show model, workspace, and chat state", icon: "gauge.with.dots.needle.bottom.50percent"),
        .init(name: "usage", aliases: [], title: "Usage", detail: "Show the latest request usage and speed", icon: "chart.bar"),
        .init(name: "side", aliases: [], title: "Side chat", detail: "Start a side conversation", icon: "plus.bubble"),
        .init(name: "compact", aliases: [], title: "Compact", detail: "Rebuild context around the current conversation", icon: "arrow.down.right.and.arrow.up.left"),
        .init(name: "feedback", aliases: [], title: "Feedback", detail: "Save feedback with this chat", icon: "bubble.left.and.exclamationmark.bubble.right"),
    ]

    public static func definition(for token: String) -> ChatSlashCommandDefinition? {
        let normalized = token.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "/ "))
        return definitions.first { definition in
            definition.name == normalized || definition.aliases.contains(normalized)
        }
    }

    public static func parse(_ text: String) -> ParsedChatSlashCommand? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.first == "/" else { return nil }
        let pieces = trimmed.split(maxSplits: 1, whereSeparator: { $0.isWhitespace })
        guard let token = pieces.first, let definition = definition(for: String(token)) else {
            return nil
        }
        return ParsedChatSlashCommand(
            definition: definition,
            argument: pieces.count == 2 ? String(pieces[1]).trimmingCharacters(in: .whitespacesAndNewlines) : ""
        )
    }

    public static func suggestions(for text: String) -> [ChatSlashCommandDefinition] {
        guard text.first == "/", !text.contains(where: \.isWhitespace) else { return [] }
        let query = String(text.dropFirst()).lowercased()
        return definitions.filter { definition in
            query.isEmpty || definition.name.hasPrefix(query) || definition.title.lowercased().hasPrefix(query)
        }
    }

    public static var helpText: String {
        definitions.map { "\($0.commandText)  \($0.detail)" }.joined(separator: "\n")
    }
}
