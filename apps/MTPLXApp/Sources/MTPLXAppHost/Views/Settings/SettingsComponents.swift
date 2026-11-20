import SwiftUI
import MTPLXAppCore

// MARK: - Settings page building blocks
//
// One shape for every row: label and one line of help on the left, the
// control on the right (or below, for paths and lists), rows of a group
// on one card with hairlines between them. Groups and rows hide
// themselves when a search does not match them, so the page never shows
// an empty card or a stray divider.

enum SettingsRowLayout {
    /// Control on the trailing edge of the row.
    case trailing
    /// Control under the text, full width: paths, lists, wide fields.
    case stacked
}

/// A titled card of settings rows.
struct SettingsGroup<Content: View>: View {
    let group: SettingsGroupID
    var footer: String? = nil
    @ViewBuilder let content: Content

    @Environment(\.settingsSearch) private var search

    var body: some View {
        if search.matches(group) {
            VStack(alignment: .leading, spacing: 8) {
                MicroHeader(group.title)
                    .padding(.leading, 2)
                // The 0.5 pt spacing shows the hairline-colored backing
                // between rows, so a divider sits only between rows that
                // are actually shown.
                VStack(alignment: .leading, spacing: Brand.hairline) {
                    content
                }
                .background {
                    ZStack {
                        Brand.cardSurface
                        Brand.separator
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                }
                if let footer {
                    Text(footer)
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 2)
                }
            }
        }
    }
}

/// The row surface every settings row and info line shares.
private struct SettingsRowSurface: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .frame(maxWidth: .infinity, minHeight: 46, alignment: .leading)
            .background(Brand.cardSurface)
    }
}

extension View {
    fileprivate func settingsRowSurface() -> some View {
        modifier(SettingsRowSurface())
    }
}

/// Label and help text of a row.
struct SettingsRowText: View {
    let label: String
    let caption: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Brand.typeBody)
                .fixedSize(horizontal: false, vertical: true)
            if let caption, !caption.isEmpty {
                Text(caption)
                    .font(.system(size: 11))
                    .foregroundStyle(Brand.typeTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// A setting: catalog label and help on the left, its control on the
/// right (or below with `.stacked`).
struct SettingsRow<Control: View>: View {
    let item: SettingsItem
    /// Replaces the catalog's help when it depends on the Mac or model.
    var caption: String? = nil
    var layout: SettingsRowLayout = .trailing
    @ViewBuilder let control: Control

    @Environment(\.settingsSearch) private var search

    var body: some View {
        if search.matches(item) {
            Group {
                switch layout {
                case .trailing:
                    HStack(alignment: .center, spacing: 16) {
                        SettingsRowText(label: item.label, caption: caption ?? item.caption)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        control
                    }
                case .stacked:
                    VStack(alignment: .leading, spacing: 10) {
                        SettingsRowText(label: item.label, caption: caption ?? item.caption)
                        control
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .settingsRowSurface()
            .id(item)
        }
    }
}

/// A switch row.
struct SettingsToggleRow: View {
    let item: SettingsItem
    @Binding var isOn: Bool
    var caption: String? = nil

    var body: some View {
        SettingsRow(item: item, caption: caption) {
            Toggle(item.label, isOn: $isOn)
                .toggleStyle(.switch)
                .labelsHidden()
        }
    }
}

/// A read-only line inside a group: what the engine is using now, a
/// status, a note. Shown and hidden with the item it explains.
struct SettingsInfoRow<Content: View>: View {
    let item: SettingsItem
    @ViewBuilder let content: Content

    @Environment(\.settingsSearch) private var search

    var body: some View {
        if search.matches(item) {
            content
                .font(.system(size: 11))
                .foregroundStyle(Brand.typeSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .settingsRowSurface()
        }
    }
}

/// A row that opens something (About, Logs): the whole row is the button.
struct SettingsLinkRow: View {
    let item: SettingsItem
    var trailingHint: String? = nil
    let action: () -> Void

    @Environment(\.settingsSearch) private var search
    @State private var hovering = false

    var body: some View {
        if search.matches(item) {
            Button(action: action) {
                HStack(alignment: .center, spacing: 16) {
                    SettingsRowText(label: item.label, caption: item.caption)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let trailingHint {
                        Text(verbatim: trailingHint)
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(Brand.typeTertiary)
                    }
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .settingsRowSurface()
            .overlay(Brand.wash.opacity(hovering ? 0.03 : 0).allowsHitTesting(false))
            .onHover { hovering = $0 }
            .accessibilityLabel(item.label)
        }
    }
}

// MARK: - Section header

struct SettingsSectionHeader<Trailing: View>: View {
    let section: SettingsSection
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: section.systemImage)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Brand.accentChrome)
                    Text(section.title.uppercased())
                        .font(.system(size: 13, weight: .heavy, design: .monospaced))
                        .tracking(2.4)
                        .foregroundStyle(Brand.typeHi)
                }
                Text(section.subtitle)
                    .font(.system(size: 12))
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            trailing
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

extension SettingsSectionHeader where Trailing == EmptyView {
    init(section: SettingsSection) {
        self.section = section
        self.trailing = EmptyView()
    }
}

// MARK: - Sidebar

struct SettingsSidebar: View {
    @Binding var selection: SettingsSection
    @Binding var query: String
    let search: SettingsSearch
    let unsaved: Set<SettingsSection>
    let onSelect: (SettingsSection) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SettingsSearchField(query: $query)
            VStack(alignment: .leading, spacing: 2) {
                ForEach(SettingsSection.allCases) { section in
                    if search.hasMatches(section) {
                        SettingsSidebarItem(
                            section: section,
                            isSelected: !search.isActive && selection == section,
                            hasUnsavedChanges: unsaved.contains(section)
                        ) {
                            onSelect(section)
                        }
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 16)
    }
}

private struct SettingsSidebarItem: View {
    let section: SettingsSection
    let isSelected: Bool
    let hasUnsavedChanges: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: section.systemImage)
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 18)
                    .foregroundStyle(isSelected ? Brand.accentChrome : Brand.typeSecondary)
                Text(section.title)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                    .tracking(0.4)
                    .foregroundStyle(isSelected ? Brand.typeHi : Brand.typeBody.opacity(0.8))
                    .lineLimit(1)
                Spacer(minLength: 4)
                if hasUnsavedChanges {
                    Circle()
                        .fill(Brand.warning)
                        .frame(width: 6, height: 6)
                        .accessibilityLabel(tr("Unsaved changes"))
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                    .fill(isSelected ? Brand.wash.opacity(0.07) : Brand.wash.opacity(hovering ? 0.035 : 0))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityLabel(section.title)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

/// Section chips for a narrow window, where a sidebar would crowd the rows.
struct SettingsSectionChips: View {
    @Binding var selection: SettingsSection
    let search: SettingsSearch
    let unsaved: Set<SettingsSection>
    let onSelect: (SettingsSection) -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(SettingsSection.allCases) { section in
                    if search.hasMatches(section) {
                        let selected = !search.isActive && selection == section
                        Button {
                            onSelect(section)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: section.systemImage)
                                    .font(.system(size: 11, weight: .semibold))
                                Text(section.title)
                                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                if unsaved.contains(section) {
                                    Circle().fill(Brand.warning).frame(width: 5, height: 5)
                                }
                            }
                            .foregroundStyle(selected ? Brand.typeHi : Brand.typeSecondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(Brand.wash.opacity(selected ? 0.08 : 0.03))
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(section.title)
                        .accessibilityAddTraits(selected ? .isSelected : [])
                    }
                }
            }
            .padding(.horizontal, 16)
        }
    }
}

struct SettingsSearchField: View {
    @Binding var query: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
            TextField(tr("Search settings"), text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(Brand.typeBody)
            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeTertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(tr("Clear"))
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 7)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                .fill(Brand.wash.opacity(0.05))
                .overlay {
                    RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                }
        }
    }
}

// MARK: - Save bar

/// Pinned under the Settings content while there are edits to save. One
/// place to save, whichever section the edits were made in.
struct SettingsSaveBar: View {
    let running: Bool
    let applying: Bool
    let error: String?
    let onSave: () -> Void
    let onRevert: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(error == nil ? Brand.warning : Brand.danger)
                .frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 2) {
                Text(error == nil ? tr("Unsaved changes") : tr("Last save error"))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Brand.typeHi)
                Text(error ?? tr("Your edits are kept while you use other tabs. Save them to apply, or revert to the saved values."))
                    .font(.system(size: 11))
                    .foregroundStyle(error == nil ? Brand.typeSecondary : Brand.danger)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Button(tr("Revert"), action: onRevert)
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .disabled(applying)
            Button(action: onSave) {
                if applying {
                    ProgressView().controlSize(.small)
                } else {
                    Label(
                        running ? tr("Apply + Restart") : tr("Save"),
                        systemImage: running ? "arrow.clockwise" : "checkmark.circle"
                    )
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(applying)
            .keyboardShortcut("s", modifiers: [.command])
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background {
            Brand.bgInner
                .overlay(alignment: .top) {
                    Rectangle().fill(Brand.separator).frame(height: Brand.hairline)
                }
        }
    }
}

// MARK: - Gear button (top bar)

/// Opens Settings; pressed again, goes back to the dashboard the user
/// came from. Same chrome puck as its neighbours, with a ring while
/// Settings is showing.
struct SettingsGearButton: View {
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        let showing = router.selection == .settings && router.primaryMode == .dashboard
        Button {
            if router.benchmarkOverlayPresented { router.closeBenchmark() }
            router.inferenceParamsPresented = false
            router.toggleSettings()
        } label: {
            Image(systemName: "gearshape")
                .font(.system(size: 13, weight: .semibold))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(showing ? Brand.typeHi : Brand.typeBody)
                .frame(width: Brand.controlSize, height: Brand.controlSize)
                .overlay {
                    Circle()
                        .strokeBorder(Brand.accentChrome.opacity(showing ? 0.7 : 0), lineWidth: 1)
                }
        }
        .buttonStyle(PremiumPuckStyle())
        // Says what opens, like the Model controls puck beside it: the two
        // glyphs alone both read as "settings".
        .help(
            tr("Settings") + " (⌘,)"
                + "\n" + tr("All of MTPLX's settings: model, performance, memory, cache, clients, cooling.")
        )
        .accessibilityLabel(tr("Settings"))
        .accessibilityHint(tr("All of MTPLX's settings: model, performance, memory, cache, clients, cooling."))
        .accessibilityAddTraits(showing ? .isSelected : [])
    }
}
