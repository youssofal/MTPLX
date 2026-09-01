import SwiftUI
import MTPLXAppCore

// MARK: - ForgeDiscoverView
//
// LazyVGrid of MTPLX-branded community models pulled live from
// Hugging Face, sorted by download count desc. Backed by
// `mtplx forge discover --json` via ForgeDiscoveryService.
//
// Three states the view rotates through:
//   • loading        — initial fetch in flight (skeleton + spinner)
//   • loaded(entries) — grid of DiscoveryCards (empty if HF returned
//                       zero matches)
//   • error(.hfUnreachable / .backendNotAvailable / .other) —
//                       explicit empty state with Retry button
//
// In-session in-memory cache: once the entries are fetched they stay
// in @State until the user clicks Refresh in the toolbar (or the
// view disappears and re-appears, since @State is per-instance).

struct ForgeDiscoverView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    enum LoadState: Equatable {
        case idle
        case loading
        case loaded([DiscoveryEntry])
        case error(ForgeDiscoveryError)
    }

    /// Discover-wall ownership filter. The default `.all` view sorts
    /// MTPLX's verified-creator models to the top of the otherwise
    /// downloads-descending HF list; `.mine` (rendered as
    /// "Verified") collapses to only entries published by the
    /// verified creator.
    enum OwnershipFilter: String, CaseIterable {
        case all
        case mine

        var label: String {
            switch self {
            case .all: return "All"
            case .mine: return "Verified"
            }
        }
    }

    @State private var state: LoadState = .idle
    @State private var filter: OwnershipFilter = .all
    @State private var searchText: String = ""

    /// MTPLX's verified creator handle. This is the project author —
    /// every model under `huggingface.co/youssofal/...` is the
    /// canonical, MTPLX-team-published artifact, so every user of
    /// the app sees these cards with the verified-blue badge. NOT a
    /// per-user setting; the badge is to MTPLX what the blue check
    /// is to a platform account.
    static let verifiedCreatorHandle = "youssofal"

    private let columns = [GridItem(.adaptive(minimum: 280, maximum: 360), spacing: 14)]

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider().overlay(Brand.separator)
            Group {
                switch state {
                case .idle, .loading:
                    loadingView
                case .loaded(let entries) where visibleEntries(from: entries).isEmpty:
                    emptyState(originalCount: entries.count)
                case .loaded(let entries):
                    grid(entries: visibleEntries(from: entries))
                case .error(let err):
                    errorView(err)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear { reloadIfNeeded() }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 10) {
            searchField
                .frame(maxWidth: 320)
            Spacer(minLength: 8)
            ownershipFilterPill
            Button {
                refresh()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10.5, weight: .semibold))
                    Text("Refresh")
                        .font(.system(size: 11, weight: .semibold))
                }
                .foregroundStyle(Brand.typeBody)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    Capsule(style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: 0.5)
                )
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .disabled(state == .loading)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }


    /// Search field for the discover list — filters the loaded results
    /// in-memory by model name, creator, or repo. Clears with the inline
    /// button. Jet Chrome: piano card surface, hairline border, no purple.
    private var searchField: some View {
        HStack(spacing: 7) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
            TextField("Search models, creators…", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 12))
                .foregroundStyle(Brand.typeBody)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Brand.typeTertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            Capsule(style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    Capsule(style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    /// Two-segment pill (All / Mine) sitting next to Refresh. The
    /// Mine segment is disabled when the user hasn't published yet
    /// (no `huggingFaceHandle` configured) — the tooltip explains
    /// where to set it instead of silently doing nothing.
    private var ownershipFilterPill: some View {
        HStack(spacing: 2) {
            ForEach(OwnershipFilter.allCases, id: \.self) { value in
                ownershipFilterSegment(value)
            }
        }
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.white.opacity(0.03))
                .overlay(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }

    @ViewBuilder
    private func ownershipFilterSegment(_ value: OwnershipFilter) -> some View {
        let isSelected = filter == value
        Button {
            withAnimation(.smooth(duration: 0.18)) {
                filter = value
            }
        } label: {
            Text(value.label)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .tracking(0.2)
                .foregroundStyle(isSelected ? Brand.typeHi : Brand.typeSecondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .frame(minWidth: 60)
                .background(
                    Group {
                        if isSelected {
                            RoundedRectangle(cornerRadius: 5, style: .continuous)
                                .fill(Brand.cardSurface)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                                        .stroke(Brand.separatorStrong, lineWidth: 0.5)
                                )
                        }
                    }
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Show \(value.label.lowercased())")
    }

    // MARK: - States

    private var loadingView: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.regular)
            Text("Loading models from Hugging Face…")
                .font(.caption)
                .foregroundStyle(Brand.typeSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Shown when the visible-entries list is empty. Distinct copy
    /// for the Verified-only path (HF returned results but none
    /// match the verified creator) vs the All path (HF returned an
    /// empty match list).
    @ViewBuilder
    private func emptyState(originalCount: Int) -> some View {
        if filter == .mine && originalCount > 0 {
            EmptyStateView(
                symbol: "checkmark.seal.fill",
                title: "No verified models right now",
                message: "Switch to All to browse community models."
            )
        } else {
            EmptyStateView(
                symbol: "globe",
                title: "No MTPLX models on Hugging Face yet",
                message: "Be the first — head to Create and build one."
            )
        }
    }

    @ViewBuilder
    private func errorView(_ error: ForgeDiscoveryError) -> some View {
        switch error {
        case .hfUnreachable:
            EmptyStateView(
                symbol: "wifi.exclamationmark",
                title: "Can't reach Hugging Face",
                message: "Check your connection and try again.",
                action: { refresh() },
                actionLabel: "Retry"
            )
        case .backendNotAvailable:
            EmptyStateView(
                symbol: "hammer.fill",
                title: "Forge isn't available",
                message: "Update MTPLX to use Discover."
            )
        case .malformedResponse(let detail), .subprocessFailed(_, let detail):
            EmptyStateView(
                symbol: "exclamationmark.triangle.fill",
                title: "Couldn't load models",
                message: detail.isEmpty ? "Something went wrong. Try again." : detail,
                action: { refresh() },
                actionLabel: "Retry"
            )
        }
    }

    private func grid(entries: [DiscoveryEntry]) -> some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 14) {
                ForEach(entries) { entry in
                    DiscoveryCard(
                        entry: entry,
                        isOwnedByUser: isVerifiedCreator(entry),
                        isInstalling: backend.pendingModelDownload?.repoID == entry.repo,
                        isInstalled: isAlreadyInstalled(entry: entry),
                        onInstall: { install(entry: entry) },
                        onOpenOnHF: { openOnHF(entry: entry) }
                    )
                }
            }
            .padding(20)
        }
    }

    // MARK: - Verified creator ownership

    /// `true` when an entry is published by MTPLX's verified creator
    /// (Youssofal). Drives the blue checkmark badge on the card AND
    /// the "yours-first" pinning in the All listing. Hardcoded —
    /// not a per-user setting — so every install of the app shows
    /// the same set of verified models with the same badge.
    private func isVerifiedCreator(_ entry: DiscoveryEntry) -> Bool {
        entry.owner.lowercased() == Self.verifiedCreatorHandle
    }

    /// Apply the ownership filter + the "verified first" ordering to
    /// the raw HF results. Backend always returns downloads-desc;
    /// this layer pins MTPLX's verified-creator models to the top
    /// of the All view (preserving their HF download order within
    /// the pinned group) or collapses to only verified entries when
    /// the Verified filter is active.
    private func visibleEntries(from entries: [DiscoveryEntry]) -> [DiscoveryEntry] {
        let searched = entries.filter(matchesSearch)
        let (verified, others) = partitionByVerification(searched)
        switch filter {
        case .all:
            return verified + others
        case .mine:
            return verified
        }
    }

    /// Case-insensitive substring match over the model's display name,
    /// creator handle, and repo slug. Empty query matches everything.
    private func matchesSearch(_ entry: DiscoveryEntry) -> Bool {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return true }
        return entry.brandedName.lowercased().contains(query)
            || entry.owner.lowercased().contains(query)
            || entry.repo.lowercased().contains(query)
    }

    private func partitionByVerification(
        _ entries: [DiscoveryEntry]
    ) -> (verified: [DiscoveryEntry], others: [DiscoveryEntry]) {
        var verified: [DiscoveryEntry] = []
        var others: [DiscoveryEntry] = []
        for entry in entries {
            if isVerifiedCreator(entry) {
                verified.append(entry)
            } else {
                others.append(entry)
            }
        }
        return (verified, others)
    }

    // MARK: - Actions

    private func reloadIfNeeded() {
        if case .idle = state { refresh() }
    }

    private func refresh() {
        state = .loading
        Task {
            do {
                let service = ForgeDiscoveryService()
                let entries = try await service.discover()
                await MainActor.run { state = .loaded(entries) }
            } catch let error as ForgeDiscoveryError {
                await MainActor.run { state = .error(error) }
            } catch {
                await MainActor.run {
                    state = .error(.subprocessFailed(exitCode: nil, stderrTail: error.localizedDescription))
                }
            }
        }
    }

    private func install(entry: DiscoveryEntry) {
        // Hand off to the shared model-download sheet (progress + cancel +
        // friendly errors), which on completion registers the model into
        // the top-left picker and starts/restarts the daemon against it.
        // This replaces the old fire-and-forget consumer that dropped all
        // progress events, swallowed failures, and got killed when the
        // user navigated away from the Discover wall.
        backend.presentModelDownload(
            repoID: entry.repo,
            displayName: entry.brandedName,
            totalBytes: entry.sizeBytes
        )
    }

    private func openOnHF(entry: DiscoveryEntry) {
        guard let url = URL(string: "https://huggingface.co/\(entry.repo)") else { return }
        NSWorkspace.shared.open(url)
    }

    /// "Installed" means the weights are actually on disk — not merely
    /// registered as a custom model — so the card badge is honest.
    private func isAlreadyInstalled(entry: DiscoveryEntry) -> Bool {
        if let opt = MTPLXModelOption.option(matching: entry.repo) {
            return opt.isInstalled(in: backend.configuration.modelLibrary)
        }
        return MTPLXModelOption.customHuggingFaceModel(repoID: entry.repo)?
            .isInstalled(in: backend.configuration.modelLibrary) ?? false
    }
}
