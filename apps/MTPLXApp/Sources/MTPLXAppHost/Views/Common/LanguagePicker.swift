import SwiftUI
import MTPLXAppCore

// MARK: - LanguagePickerList
//
// Searchable list of the app languages. Used full-size by the onboarding
// Language step and inside a popover by Settings > Language. The search
// field filters by native name, English name and code (case, diacritic
// and width insensitive, see AppLanguage.matching). Arrow keys move a
// keyboard cursor through the visible rows and Return picks it, so the
// whole flow works without a mouse.
//
// Picking writes LanguageStore.language, which activates the language in
// L10n before publishing; the caller's tree re-renders in the new
// language on the same run loop turn.

struct LanguagePickerList: View {
    @EnvironmentObject private var languageStore: LanguageStore
    @State private var query = ""
    @State private var cursor: AppLanguage?
    @FocusState private var searchFocused: Bool
    var onPick: ((AppLanguage) -> Void)? = nil

    private var results: [AppLanguage] {
        AppLanguage.matching(query)
    }

    var body: some View {
        VStack(spacing: 10) {
            searchField
            if results.isEmpty {
                Text(tr("No matching languages."))
                    .font(.system(size: 13))
                    .foregroundStyle(Brand.typeTertiary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView(.vertical, showsIndicators: true) {
                        LazyVStack(spacing: 2) {
                            ForEach(results) { language in
                                row(language)
                                    .id(language)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                    .onAppear {
                        proxy.scrollTo(languageStore.language, anchor: .center)
                    }
                    .onChange(of: cursor) { _, next in
                        if let next {
                            proxy.scrollTo(next, anchor: nil)
                        }
                    }
                }
            }
        }
        .onChange(of: query) { _, _ in
            // Keep the cursor on a visible row after every filter change.
            if let cursor, results.contains(cursor) { return }
            cursor = results.first
        }
        .onKeyPress(.downArrow) {
            moveCursor(by: 1)
            return .handled
        }
        .onKeyPress(.upArrow) {
            moveCursor(by: -1)
            return .handled
        }
        .onKeyPress(.return) {
            guard let cursor, results.contains(cursor) else { return .ignored }
            pick(cursor)
            return .handled
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(tr("Language"))
    }

    // MARK: - Pieces

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
                .accessibilityHidden(true)
            TextField(tr("Search languages"), text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundStyle(Brand.typeHi)
                .focused($searchFocused)
            if !query.isEmpty {
                Button {
                    query = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeTertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(tr("Clear search"))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(searchFocused ? Brand.separatorStrong : Brand.separator, lineWidth: Brand.hairlineStrong)
                )
        )
        .onAppear { searchFocused = true }
    }

    private func row(_ language: AppLanguage) -> some View {
        let selected = language == languageStore.language
        let focused = language == cursor
        return Button {
            pick(language)
        } label: {
            HStack(spacing: 12) {
                Text(language.flag)
                    .font(.system(size: 20))
                    .frame(width: 28)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    Text(language.nativeName)
                        .font(.system(size: 13, weight: selected ? .semibold : .medium))
                        .foregroundStyle(Brand.typeHi)
                    Text(language.englishName)
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeTertiary)
                }
                Spacer(minLength: 0)
                if selected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundStyle(Brand.typeHi)
                        .accessibilityHidden(true)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(selected ? Brand.raisedSurface : (focused ? Brand.cardSurface : Color.clear))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(focused ? Brand.separatorStrong : Color.clear, lineWidth: Brand.hairlineStrong)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            if hovering { cursor = language }
        }
        .accessibilityLabel(
            language.englishName == language.nativeName
                ? language.nativeName
                : tr("%@ (%@)", language.nativeName, language.englishName)
        )
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }

    // MARK: - Behaviour

    private func moveCursor(by delta: Int) {
        let visible = results
        guard !visible.isEmpty else { return }
        let current = cursor.flatMap { visible.firstIndex(of: $0) }
            ?? visible.firstIndex(of: languageStore.language)
            ?? (delta > 0 ? -1 : visible.count)
        let next = min(max(current + delta, 0), visible.count - 1)
        cursor = visible[next]
    }

    private func pick(_ language: AppLanguage) {
        cursor = language
        languageStore.language = language
        onPick?(language)
    }
}
