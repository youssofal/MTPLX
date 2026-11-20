import MTPLXAppCore
import SwiftUI

/// Settings > Clients > Where clients open: one picker per client listing
/// the client's app and the terminals this Mac has. Every choice carries a
/// one-colour glyph in MTPLX's style (the client's mark for its app, a
/// terminal symbol for each terminal), never another app's own icon.
/// A pick is saved at once to the config file `mtplx start` reads
/// (`pi_opens_in`, ...) and applies to the next launch from either.
struct ClientOpeningRows: View {
    private struct Row: Identifiable {
        let item: SettingsItem
        let target: LaunchTarget
        var id: SettingsItem { item }
    }

    private static let rows = [
        Row(item: .piOpensIn, target: .pi),
        Row(item: .openCodeOpensIn, target: .openCode),
        Row(item: .hermesOpensIn, target: .hermes),
        Row(item: .claudeCodeOpensIn, target: .claudeCode),
        Row(item: .codexOpensIn, target: .codex),
        Row(item: .deepseekHarnessOpensIn, target: .deepseekHarness),
    ]

    @State private var terminals: [ClientOpenings.TerminalApp] = []
    @State private var choices: [String: String] = [:]
    @State private var saveError: String?

    private struct Option: Identifiable {
        let value: String
        let title: String
        /// The client whose mark this choice shows; nil for a terminal.
        let mark: LaunchTarget?
        let symbol: String
        var id: String { value }
    }

    var body: some View {
        Group {
            ForEach(Self.rows) { row in
                if let client = ClientOpenings.client(for: row.target) {
                    SettingsRow(item: row.item) {
                        picker(client: client, target: row.target)
                    }
                }
            }
            if let saveError {
                SettingsInfoRow(item: .piOpensIn) {
                    Label(saveError, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(Brand.warning)
                }
            }
        }
        .task { await load() }
    }

    private func load() async {
        let found = await Task.detached(priority: .utility) {
            ClientOpenings.installedTerminals()
        }.value
        terminals = found
        choices = ClientOpenings.savedChoices()
    }

    private func picker(client: ClientOpenings.Client, target: LaunchTarget) -> some View {
        let options = options(for: client, target: target)
        return Picker(client.name, selection: selection(for: client)) {
            ForEach(options) { option in
                Label {
                    Text(option.title)
                } icon: {
                    if let mark = option.mark {
                        Image(nsImage: ClientMarkImages.templateImage(for: mark))
                    } else {
                        Image(systemName: option.symbol)
                    }
                }
                .tag(option.value)
            }
        }
        .labelsHidden()
        .fixedSize()
        .accessibilityLabel(tr("%@ opens in", client.name))
    }

    /// The client's apps first, then every terminal found; a saved terminal
    /// that is no longer installed stays listed so the row tells the truth.
    private func options(for client: ClientOpenings.Client, target: LaunchTarget) -> [Option] {
        let appName = client.appName ?? client.name
        var options: [Option] = client.appSurfaces.map { surface in
            let title = surface == ClientOpenings.mtplxApp
                ? tr("MTPLX's own %@ app", appName)
                : tr("%@ app", appName)
            return Option(value: surface, title: title, mark: target, symbol: target.systemImage)
        }
        options += terminals.map {
            Option(value: $0.name, title: $0.name, mark: nil, symbol: "terminal")
        }
        let current = selectedValue(for: client)
        if !options.contains(where: { $0.value == current }) {
            options.append(Option(
                value: current,
                title: tr("%@ (not installed)", current),
                mark: nil,
                symbol: "questionmark.app"
            ))
        }
        return options
    }

    /// The option the saved choice resolves to, as `mtplx start` resolves it.
    private func selectedValue(for client: ClientOpenings.Client) -> String {
        let saved = choices[client.id]
        let opening = ClientOpenings.resolve(client: client.id, choice: saved, terminals: terminals)
        if !opening.opensInTerminal { return opening.surface }
        if let saved, opening.note != nil, saved != ClientOpenings.terminalChoice { return saved }
        return opening.terminalName
    }

    private func selection(for client: ClientOpenings.Client) -> Binding<String> {
        Binding(
            get: { selectedValue(for: client) },
            set: { value in
                do {
                    try ClientOpenings.save(choice: value, for: client.id)
                    choices[client.id] = value
                    saveError = nil
                } catch {
                    saveError = tr("Could not save where %@ opens: %@", client.name, error.localizedDescription)
                }
            }
        )
    }
}
