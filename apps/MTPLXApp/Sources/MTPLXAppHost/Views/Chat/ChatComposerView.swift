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
            if !commandSuggestions.isEmpty {
                commandPalette
            }
            HStack(alignment: .bottom, spacing: 12) {
                ComposerInputTextView(
                    text: $text,
                    measuredHeight: $measuredHeight,
                    minHeight: minHeight,
                    maxHeight: maxHeight,
                    onSubmit: handleSubmit,
                    onFileDrop: { urls in
                        Task { await viewModel.attach(urls, visionEnabled: visionEnabled) }
                    },
                    onImagePaste: { png in
                        Task {
                            await viewModel.attachPastedImage(
                                png,
                                filename: ComposerPasteClassifier.pastedImageFilename(),
                                visionEnabled: visionEnabled
                            )
                        }
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
                    let presentation = Self.cardPresentation(
                        for: attachment,
                        state: viewModel.extractionState(for: attachment)
                    )
                    AttachmentCard(
                        filename: attachment.filename,
                        fileExtension: extensionOf(attachment.filename),
                        sizeBytes: attachment.sizeBytes,
                        imageData: attachment.imageData,
                        errorMessage: presentation.errorMessage,
                        statusMessage: presentation.statusMessage,
                        isExtracting: presentation.isExtracting,
                        onRemove: { viewModel.removeAttachment(attachment) }
                    )
                }
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 2)
        }
    }

    /// What the card says about a pending attachment: extracting,
    /// failed (with the reason), truncated (with what was kept), or
    /// unreadable when extraction produced nothing to send.
    struct AttachmentCardPresentation: Equatable {
        var errorMessage: String?
        var statusMessage: String?
        var isExtracting = false
    }

    static func cardPresentation(
        for attachment: ChatAttachment,
        state: ChatViewModel.AttachmentExtractionState?
    ) -> AttachmentCardPresentation {
        switch state {
        case .extracting:
            return AttachmentCardPresentation(statusMessage: tr("Extracting…"), isExtracting: true)
        case .failed(let message):
            return AttachmentCardPresentation(errorMessage: message)
        case .ready(let truncation):
            if attachment.imageData == nil && attachment.extractedText.isEmpty {
                return AttachmentCardPresentation(errorMessage: tr("Could not read"))
            }
            return AttachmentCardPresentation(statusMessage: truncation?.summary)
        case nil:
            let unreadable = attachment.imageData == nil && attachment.extractedText.isEmpty
            return AttachmentCardPresentation(errorMessage: unreadable ? tr("Could not read") : nil)
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
                        .fill(Brand.wash.opacity(0.04))
                        .overlay(Circle().stroke(Brand.separator, lineWidth: 0.5))
                )
        }
        .buttonStyle(.plain)
        .help(tr("Attach a file (PDF, docx, md, txt)"))
        .accessibilityLabel(tr("Attach file"))
    }

    private var commandSuggestions: [ChatSlashCommandDefinition] {
        ChatSlashCommands.suggestions(for: text)
    }

    private var commandPalette: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(commandSuggestions.prefix(6)) { command in
                Button {
                    text = command.commandText + " "
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: command.icon)
                            .frame(width: 16)
                        Text(command.commandText)
                            .fontWeight(.semibold)
                        Text(command.detail)
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(1)
                        Spacer()
                    }
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(Brand.typeSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(5)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.bgOuter)
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Brand.separator, lineWidth: 0.5))
        )
    }

    private var webSearchToggle: some View {
        let isOn = viewModel.webSearchEnabled
        return Button {
            viewModel.webSearchEnabled.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "globe")
                    .font(.system(size: 12, weight: .semibold))
                Text(tr("Web"))
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
                            : Brand.wash.opacity(0.04)
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
        .help(isOn ? tr("Web search is on for this conversation") : tr("Enable web search for this conversation"))
        .accessibilityLabel(tr("Web search"))
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
                    .foregroundStyle(Brand.onAccent)
                    .symbolRenderingMode(.monochrome)
                    .contentTransition(.symbolEffect(.replace))
            }
            .frame(width: 32, height: 32)
            .scaleEffect(sendButtonHovering && canSend ? 1.04 : 1.0)
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.isStreaming && !canSend)
        .onHover { sendButtonHovering = $0 }
        .help(sendButtonHelp)
        .accessibilityLabel(viewModel.isStreaming ? tr("Stop generating") : tr("Send message"))
        .animation(.smooth(duration: 0.18), value: viewModel.isStreaming)
        .animation(.smooth(duration: 0.18), value: sendButtonHovering)
    }

    private var sendButtonHelp: String {
        if viewModel.isStreaming { return tr("Stop generating") }
        if viewModel.isExtractingAttachments { return tr("Extracting attachments…") }
        return tr("Send")
    }

    private var canSend: Bool {
        engineCanAcceptMessages
            // Send waits for the strip to settle so a file still being
            // read is never left behind for the next message.
            && !viewModel.isExtractingAttachments
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
            return tr("Loading %@…", selectedModelName)
        case .stopping:
            return tr("Stopping MTPLX…")
        case .stopped:
            return tr("Start MTPLX to send.")
        case .degraded, .crashed:
            return tr("Restart MTPLX to send.")
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
        panel.prompt = tr("Attach")
        panel.message = visionEnabled
            ? "Attach documents (PDF, docx, md, txt) or images (PNG, JPEG, WebP)."
            : tr("Attach files (PDF, docx, md, txt) to include their text in your message.")
        if panel.runModal() == .OK {
            let urls = panel.urls
            Task { await viewModel.attach(urls, visionEnabled: visionEnabled) }
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
