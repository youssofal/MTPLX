import SwiftUI
import MTPLXAppCore

// MARK: - ChatHeaderView
//
// Slim row at the top of the chat surface (below the global
// TopChromeStrip). Sidebar collapse toggle on the left, conversation
// title in the middle, live TPS chip on the right pulled from the
// active chat request stream. This intentionally does not read the
// global dashboard headline: chat needs the number for the reply that
// is currently writing into the chat bubble.

struct ChatHeaderView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Binding var sidebarCollapsed: Bool
    @Binding var workspacePanelPresented: Bool

    /// Latched display state for the TPS chip. The daemon emits
    /// progress events several times per second; binding the chip
    /// directly to every raw event makes it flicker. We
    /// snapshot the reading on a 0.5s tick (matches the dashboard
    /// gauge's `speedDisplayInterval`) so the chip holds each value
    /// long enough to actually be read.
    @State private var displayedReading: HeadlineDecodeReading = .absent

    var body: some View {
        HStack(spacing: 12) {
            Button {
                withAnimation(.smooth(duration: 0.22)) {
                    sidebarCollapsed.toggle()
                }
            } label: {
                Image(systemName: sidebarCollapsed ? "sidebar.left" : "sidebar.squares.left")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Brand.typeSecondary)
                    .frame(width: 24, height: 24)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(sidebarCollapsed ? tr("Show sidebar") : tr("Hide sidebar"))
            .accessibilityLabel(tr("Toggle sidebar"))

            Text(title)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeHi)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer()

            Button {
                withAnimation(.smooth(duration: 0.2)) {
                    workspacePanelPresented.toggle()
                }
            } label: {
                Image(systemName: workspacePanelPresented ? "sidebar.right" : "folder")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(workspacePanelPresented ? Brand.accentChrome : Brand.typeSecondary)
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .help("Workspace and agent controls")
            .accessibilityLabel("Workspace and agent controls")

            tpsChip
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(minHeight: 40)
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(height: 0.5),
                    alignment: .bottom
                )
        )
        .task {
            // Seed once so the chip shows the current value
            // immediately instead of waiting half a second.
            displayedReading = viewModel.chatDecodeReading
            // Then poll every 0.5s. Using `.task` instead of a
            // `Timer.publish().autoconnect()` `let` because that
            // pattern silently drops the subscription on body
            // re-renders (SwiftUI tears down the publisher when the
            // View value is replaced) — which is exactly what
            // produced "chip stuck on one value".
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                if Task.isCancelled { return }
                let next = viewModel.chatDecodeReading
                if !Self.isEquivalent(displayedReading, next) {
                    displayedReading = next
                }
            }
        }
        .onChange(of: lifecyclePhase(of: viewModel.chatDecodeReading)) { _, _ in
            // Lifecycle transitions (live -> held, held -> absent,
            // absent -> live) are user-visible state changes — flip
            // immediately so the chip never lies about whether the
            // daemon is generating right now.
            displayedReading = viewModel.chatDecodeReading
        }
    }

    private var title: String {
        viewModel.current?.title ?? tr("Chat")
    }

    @ViewBuilder
    private var tpsChip: some View {
        if case .live(let value) = displayedReading {
            chipLabel(text: tr("%@ tok/s", Format.tps(value)), accent: Brand.accentChrome)
        } else if case .held(let value, _) = displayedReading {
            chipLabel(text: tr("%@ tok/s · last", Format.tps(value)), accent: Brand.typeSecondary)
        } else {
            EmptyView()
        }
    }

    /// Coarse lifecycle classifier used to detect changes the user
    /// MUST see immediately (no throttle), independent of value
    /// fluctuations within the same phase.
    private func lifecyclePhase(of reading: HeadlineDecodeReading) -> Int {
        switch reading {
        case .absent: return 0
        case .live: return 1
        case .held: return 2
        }
    }

    /// Treat tiny value drift within the same phase as equivalent so
    /// the chip doesn't re-render every tick when the daemon reports
    /// 50.1, 50.3, 50.0 — only when the rounded integer changes.
    private static func isEquivalent(
        _ lhs: HeadlineDecodeReading,
        _ rhs: HeadlineDecodeReading
    ) -> Bool {
        switch (lhs, rhs) {
        case (.absent, .absent):
            return true
        case (.live(let a), .live(let b)):
            return Int(a.rounded()) == Int(b.rounded())
        case let (.held(a, ta), .held(b, tb)):
            return Int(a.rounded()) == Int(b.rounded()) && ta == tb
        default:
            return false
        }
    }

    private func chipLabel(text: String, accent: Color) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(accent)
                .frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .tracking(0.2)
                .foregroundStyle(Brand.typeSecondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule(style: .continuous)
                .fill(Brand.wash.opacity(0.04))
                .overlay(
                    Capsule(style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
    }
}
