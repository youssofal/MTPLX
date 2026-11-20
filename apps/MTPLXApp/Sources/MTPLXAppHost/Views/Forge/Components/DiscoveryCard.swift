import SwiftUI
import MTPLXAppCore

// MARK: - DiscoveryCard
//
// One card on the Discover wall. Hairline-outline chip vocabulary
// matches the rest of the Forge surface (ModelPickerOverlay
// statusBadge, ModelPickStep badge, TuneStep pillTag). Two actions:
//   • Install — calls onInstall to spawn ModelDownloader for the
//     entry's HF repo
//   • View on HF — opens the public repo URL in the default browser

struct DiscoveryCard: View {
    let entry: DiscoveryEntry
    /// `true` when the entry is published by MTPLX's verified creator
    /// (Youssofal). Drives the blue verification tick + the slightly
    /// brighter owner-text treatment in the header. NOT a per-user
    /// flag — the badge is to MTPLX what the blue check is to a
    /// platform account, so every install sees the same set of
    /// verified cards.
    var isOwnedByUser: Bool = false
    var isInstalling: Bool = false
    var isInstalled: Bool = false
    let onInstall: () -> Void
    let onOpenOnHF: () -> Void

    @State private var hovering: Bool = false

    /// Twitter-style verification blue used for the "made by you"
    /// tick. Kept as a hard-coded literal (not a Brand token)
    /// because it's a single deliberate exception to the Jet Chrome
    /// palette — the badge needs to read as the universal
    /// verified-account blue at a glance.
    private static let verifiedBlue = Color(red: 29.0 / 255.0, green: 155.0 / 255.0, blue: 240.0 / 255.0)

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            chipRow
            Spacer(minLength: 4)
            actionRow
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground)
        .scaleEffect(hovering ? 1.015 : 1.0)
        .shadow(
            color: hovering ? Brand.shade.opacity(0.30) : .clear,
            radius: hovering ? 12 : 0,
            x: 0,
            y: hovering ? 4 : 0
        )
        .animation(.smooth(duration: 0.22), value: hovering)
        .onHover { hovering = $0 }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "hammer.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.typeTertiary)
                Text(entry.owner)
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(isOwnedByUser ? Brand.typeBody : Brand.typeTertiary)
                if isOwnedByUser {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Self.verifiedBlue)
                        .help(tr("Verified MTPLX creator"))
                        .accessibilityLabel(tr("Verified MTPLX creator"))
                }
                Spacer(minLength: 0)
                downloadsCount
            }
            Text(entry.brandedName)
                .font(.system(size: 13.5, weight: .semibold))
                .foregroundStyle(Brand.typeHi)
                .lineLimit(2)
        }
    }

    private var downloadsCount: some View {
        HStack(spacing: 3) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 9, weight: .semibold))
            Text(formatDownloads(entry.downloads))
                .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
        }
        .foregroundStyle(Brand.typeSecondary)
    }

    private var chipRow: some View {
        HStack(spacing: 5) {
            if let depth = entry.depth, depth > 0 {
                chip(text: tr("D%lld verified", depth))
            }
            if let multiplier = entry.multiplierVsAr, multiplier > 1.0 {
                chip(text: tr("%.2f× baseline", multiplier))
            }
            if let size = entry.sizeBytes {
                chip(text: formatGB(size))
            }
            if let license = entry.license, !license.isEmpty {
                chip(text: license)
            }
        }
    }

    private var actionRow: some View {
        HStack(spacing: 8) {
            installButton
            openOnHFButton
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var installButton: some View {
        if isInstalled {
            // A state, not a control: drawn as a disabled pill it read
            // as a button that did not work.
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.success)
                Text(tr("Installed"))
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(Brand.typeSecondary)
            }
            .padding(.vertical, 6)
            .padding(.trailing, 4)
            .accessibilityElement(children: .combine)
        } else {
            installControl
        }
    }

    private var installControl: some View {
        let (title, icon) = installButtonLabel
        return Button(action: onInstall) {
            HStack(spacing: 5) {
                if isInstalling {
                    ProgressView()
                        .controlSize(.mini)
                        .tint(Brand.bgOuter)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .semibold))
                }
                Text(title)
                    .font(.system(size: 11.5, weight: .semibold))
            }
            .foregroundStyle(Brand.bgOuter)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Capsule(style: .continuous).fill(Brand.typeBody))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(isInstalling)
    }

    /// Localized: these three labels used to reach every language in English.
    private var installButtonLabel: (String, String) {
        if isInstalling { return (tr("Installing…"), "arrow.down.circle") }
        return (tr("Install"), "arrow.down.circle.fill")
    }

    private var openOnHFButton: some View {
        Button(action: onOpenOnHF) {
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 10.5, weight: .semibold))
                Text(tr("View on HF"))
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundStyle(Brand.typeSecondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                Capsule(style: .continuous)
                    .strokeBorder(Brand.separator, lineWidth: 0.5)
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(tr("Open %@ on huggingface.co", entry.repo))
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 12, style: .continuous)
            .fill(Brand.raisedSurface)
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(Brand.separator, lineWidth: 0.5)
            )
    }

    // MARK: - Helpers

    private func chip(text: String, systemImage: String? = nil) -> some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .medium))
            }
            Text(text)
                .font(.system(size: 10, weight: .medium))
        }
        .foregroundStyle(Brand.typeSecondary)
        .padding(.horizontal, 6)
        .padding(.vertical, 1.5)
        .background(
            Capsule(style: .continuous)
                .strokeBorder(Brand.separator, lineWidth: 0.5)
        )
    }

    private func formatDownloads(_ count: Int) -> String {
        if count >= 1_000_000 { return tr("%.1fM", Double(count) / 1_000_000) }
        if count >= 1_000 { return String(format: "%.1fk", Double(count) / 1_000) }
        return String(count)
    }

    private func formatGB(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1 { return tr("%.1f GB", gb) }
        let mb = Double(bytes) / 1_048_576
        return tr("%.0f MB", mb)
    }
}
