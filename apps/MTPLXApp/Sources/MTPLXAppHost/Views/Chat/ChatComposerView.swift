import SwiftUI
import AppKit
import UniformTypeIdentifiers
import MTPLXAppCore

// MARK: - ChatComposerView
//
// Pill-shaped composer at the bottom of the chat surface. Three rows:
//   1. Pending-attachments strip (only when non-empty)
//   2. `ComposerInputTextView` autosizing AppKit text view
//   3. Toolbar: paperclip (attach), globe (web-search toggle), spacer,
//      send/stop circle
//
// Max width matches the conversation column (768pt). Brand-themed
// pill container with subtle border and bgInner fill.

struct ChatComposerView: View {
    @ObservedObject var viewModel: ChatViewModel
    let daemonState: DaemonState
    let selectedModel: String
    let visionEnabled: Bool
    @State private var text: String = ""
    @State private var measuredHeight: CGFloat = 48
    @State private var sendButtonHovering = false

    private let minHeight: CGFloat = 48
    private let maxHeight: CGFloat = 144

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let status = engineStatusText {
                Text(status)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(Brand.typeSecondary)
                    .padding(.horizontal, 4)
                    .transition(.opacity)
            }
            if !viewModel.pendingAttachments.isEmpty {
                attachmentStrip
            }
            HStack(alignment: .bottom, spacing: 12) {
                ComposerInputTextView(
                    text: $text,
                    measuredHeight: $measuredHeight,
                    minHeight: minHeight,
                    maxHeight: maxHeight,
                    onSubmit: handleSubmit,
                    onFileDrop: { urls in
                        Task { await viewModel.attach(urls) }
                    }
                )
                .frame(height: measuredHeight)
            }
            HStack(spacing: 8) {
                attachButton
                webSearchToggle
                Spacer()
                sendOrStopButton
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: 768)
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Brand.bgInner)
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Brand.separatorStrong, lineWidth: 1.5)
                )
        )
    }

    // MARK: - Composer pieces

    private var attachmentStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(viewModel.pendingAttachments, id: \.id) { attachment in
                    AttachmentCard(
                        filename: attachment.filename,
                        fileExtension: extensionOf(attachment.filename),
                        sizeBytes: attachment.sizeBytes,
                        imageData: attachment.imageData,
                        errorMessage:
                            (attachment.imageData == nil
                                && attachment.extractedText.isEmpty)
                            ? "Could not read" : nil,
                        onRemove: { viewModel.removeAttachment(attachment) }
                    )
                }
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 2)
        }
    }

    private var attachButton: some View {
        Button {
            openFilePanel()
        } label: {
            Image(systemName: "paperclip")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Brand.typeSecondary)
                .frame(width: 32, height: 32)
                .background(
                    Circle()
                        .fill(Color.white.opacity(0.04))
                        .overlay(Circle().stroke(Brand.separator, lineWidth: 0.5))
                )
        }
        .buttonStyle(.plain)
        .help("Attach a file (PDF, docx, md, txt)")
        .accessibilityLabel("Attach file")
    }

    private var webSearchToggle: some View {
        let isOn = viewModel.webSearchEnabled
        return Button {
            viewModel.webSearchEnabled.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "globe")
                    .font(.system(size: 12, weight: .semibold))
                Text("Web")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .tracking(0.2)
            }
            .foregroundStyle(isOn ? Brand.accentChrome : Brand.typeSecondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(
                        isOn
                            ? Brand.accentChrome.opacity(0.12)
                            : Color.white.opacity(0.04)
                    )
                    .overlay(
                        Capsule(style: .continuous)
                            .stroke(
                                isOn ? Brand.accentChrome.opacity(0.45) : Brand.separator,
                                lineWidth: 0.5
                            )
                    )
            )
        }
        .buttonStyle(.plain)
        .help(isOn ? "Web search is on for this conversation" : "Enable web search for this conversation")
        .accessibilityLabel("Web search")
        .accessibilityValue(isOn ? "on" : "off")
    }

    private var sendOrStopButton: some View {
        Button {
            if viewModel.isStreaming {
                Task { await viewModel.cancel() }
            } else {
                handleSubmit()
            }
        } label: {
            ZStack {
                Circle()
                    .fill(Brand.accentChrome)
                    .opacity(canSend || viewModel.isStreaming ? 1.0 : 0.4)
                Image(systemName: viewModel.isStreaming ? "stop.fill" : "arrow.up")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
                    .symbolRenderingMode(.monochrome)
                    .contentTransition(.symbolEffect(.replace))
            }
            .frame(width: 32, height: 32)
            .scaleEffect(sendButtonHovering && canSend ? 1.04 : 1.0)
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.isStreaming && !canSend)
        .onHover { sendButtonHovering = $0 }
        .help(viewModel.isStreaming ? "Stop generating" : "Send")
        .accessibilityLabel(viewModel.isStreaming ? "Stop generating" : "Send message")
        .animation(.smooth(duration: 0.18), value: viewModel.isStreaming)
        .animation(.smooth(duration: 0.18), value: sendButtonHovering)
    }

    private var canSend: Bool {
        engineCanAcceptMessages
            && (
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || viewModel.hasSendablePendingAttachments
            )
    }

    private var engineCanAcceptMessages: Bool {
        daemonState.kind == .running
    }

    private var engineStatusText: String? {
        switch daemonState.kind {
        case .starting, .warming:
            return "Loading \(selectedModelName)…"
        case .stopping:
            return "Stopping MTPLX…"
        case .stopped:
            return "Start MTPLX to send."
        case .degraded, .crashed:
            return "Restart MTPLX to send."
        case .running:
            return nil
        }
    }

    private var selectedModelName: String {
        if let option = MTPLXModelOption.option(matching: selectedModel) {
            return option.shortName
        }
        let expanded = NSString(string: selectedModel).expandingTildeInPath
        let last = URL(fileURLWithPath: expanded).lastPathComponent
        return last.isEmpty ? selectedModel : last
    }

    // MARK: - Actions

    private func handleSubmit() {
        guard canSend, !viewModel.isStreaming else { return }
        let payload = text
        text = ""
        measuredHeight = minHeight
        viewModel.send(payload)
    }

    private func openFilePanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = Self.allowedContentTypes(
            includeImages: visionEnabled
        )
        panel.prompt = "Attach"
        panel.message = visionEnabled
            ? "Attach documents (PDF, docx, md, txt) or images (PNG, JPEG, WebP)."
            : "Attach files (PDF, docx, md, txt) to include their text in your message."
        if panel.runModal() == .OK {
            let urls = panel.urls
            Task { await viewModel.attach(urls) }
        }
    }

    private static func allowedContentTypes(includeImages: Bool) -> [UTType] {
        var types: [UTType] = []
        if let pdf = UTType(filenameExtension: "pdf") { types.append(pdf) }
        if let docx = UTType(filenameExtension: "docx") { types.append(docx) }
        if let md = UTType(filenameExtension: "md") { types.append(md) }
        if let txt = UTType(filenameExtension: "txt") { types.append(txt) }
        if includeImages {
            types.append(contentsOf: [.png, .jpeg, .webP])
        }
        return types
    }

    private func extensionOf(_ filename: String) -> String {
        guard let dot = filename.lastIndex(of: ".") else { return "" }
        return String(filename[filename.index(after: dot)...])
    }
}
