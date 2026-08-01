import SwiftUI
import AppKit
import MTPLXAppCore

struct HermesOverlay: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var hermes: HermesAgentStore
    let onCollapse: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            closeBar
            HermesPanel()
                .background(Brand.bgOuter)
        }
        // This is the root lifetime of the overlay, rather than a transient
        // child view that SwiftUI may recreate while updating the surface.
        .onDisappear {
            Task { await hermes.stop() }
        }
    }

    private var closeBar: some View {
        HStack(spacing: 8) {
            ChatCloseButton(action: collapseAndStop)
            Spacer()
            approvalToggle
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Brand.bgInner
                .overlay(
                    Rectangle()
                        .fill(Brand.separator)
                        .frame(height: Brand.hairline),
                    alignment: .bottom
                )
        )
    }

    private func collapseAndStop() {
        Task {
            await hermes.stop()
            onCollapse()
        }
    }

    /// Auto-approve ("YOLO") toggle. On = Hermes runs its tools without
    /// asking; off = Hermes pauses for your approval. Persisted, and
    /// applies the next time Hermes is started.
    private var approvalToggle: some View {
        let on = backend.configuration.hermesAutoApprove
        let tint = on ? Brand.warning : Brand.typeSecondary
        return Button {
            var config = backend.configuration
            config.hermesAutoApprove.toggle()
            try? backend.saveSettings(config)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: on ? "bolt.fill" : "hand.raised.fill")
                    .font(.system(size: 10, weight: .bold))
                Text(on ? "YOLO" : "ASK FIRST")
                    .font(.system(size: 10, weight: .heavy, design: .monospaced))
                    .tracking(0.8)
            }
            .foregroundStyle(tint)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(
                Capsule(style: .continuous)
                    .fill(tint.opacity(0.12))
                    .overlay(
                        Capsule(style: .continuous)
                            .stroke(tint.opacity(0.45), lineWidth: 0.75)
                    )
            )
        }
        .buttonStyle(.plain)
        .help(on
            ? "Auto-approve is on: Hermes runs tools without asking. Click to make it ask first. Applies next time Hermes starts."
            : "Hermes will ask before running tools. Click to auto-approve (YOLO). Applies next time Hermes starts.")
        .accessibilityLabel(on ? "Auto-approve on, tap to require approval" : "Approval required, tap to auto-approve")
    }
}

struct HermesPanel: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var hermes: HermesAgentStore
    @EnvironmentObject private var router: AppRouter

    @State private var composerText = ""
    @State private var pendingResponseText = ""
    @State private var pendingSubmissionID: String?
    @State private var createProfileName = ""
    @State private var creatingProfile = false
    @State private var localError: String?
    @State private var handledResumeIntent = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 318)
                .background(
                    Brand.bgInner
                        .overlay(
                            Rectangle()
                                .fill(Brand.separator)
                                .frame(width: Brand.hairline),
                            alignment: .trailing
                        )
                )

            VStack(spacing: 0) {
                header
                Divider().overlay(Brand.separator)
                transcript
                composer
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .task {
            await prepare()
        }
        .onChange(of: router.hermesLaunchIntent) { _, _ in
            Task { await prepare() }
        }
        .onChange(of: backend.daemonState.kind) { _, _ in
            hermes.refreshTerminalAgentState()
        }
        .onChange(of: hermes.activeReference) { _, reference in
            if let reference {
                remember(reference)
            }
        }
        .onChange(of: hermes.pendingRequest?.id) { _, _ in
            pendingResponseText = ""
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Hermes")
                    .font(.system(.title3, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeBody)
                statusText
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)

            if let status = hermes.installStatus {
                capabilityBlock(status)
            }

            if case .needsSetup = hermes.connectionState {
                setupBlock
            } else {
                profileList
                sessionList
                createProfileBlock
            }

            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var statusText: some View {
        switch hermes.connectionState {
        case .idle:
            Text(idleStatusLabel)
                .font(.caption)
                .foregroundStyle(idleStatusColor)
        case .checkingInstall:
            Text("Checking install")
                .font(.caption)
                .foregroundStyle(Brand.typeTertiary)
        case .needsSetup(let message):
            Text(message)
                .font(.caption)
                .foregroundStyle(Brand.warning)
                .fixedSize(horizontal: false, vertical: true)
        case .starting:
            Text("Starting gateway")
                .font(.caption)
                .foregroundStyle(Brand.typeTertiary)
        case .connected:
            Text("Gateway connected")
                .font(.caption)
                .foregroundStyle(Brand.success)
        case .failed:
            Text("Hermes connection failed")
                .font(.caption)
                .foregroundStyle(Brand.warning)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var idleStatusLabel: String {
        guard let status = hermes.installStatus, status.gatewayNeedsRepair else {
            return "Ready"
        }
        return status.gatewayHealth == .unavailable
            ? "Messaging unavailable"
            : "Messaging needs repair"
    }

    private var idleStatusColor: Color {
        guard let status = hermes.installStatus, status.gatewayNeedsRepair else {
            return Brand.typeTertiary
        }
        return Brand.warning
    }

    private var setupBlock: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let status = hermes.installStatus {
                Label(status.kind == .missing ? "Hermes isn't installed yet" : "Hermes needs an update",
                      systemImage: status.kind == .missing ? "shippingbox" : "arrow.down.circle")
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeBody)

                Text(status.kind == .missing
                    ? "Hermes is a separate command-line agent. Install it once, then come back and start it from here."
                    : "Your installed Hermes is too old for MTPLX. Update it, then recheck.")
                    .font(.caption)
                    .foregroundStyle(Brand.typeSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                if let command = status.updateCommand {
                    setupCommandRow(command)
                }

                Button {
                    Task { await hermes.prepare(configuration: backend.configuration) }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 10, weight: .bold))
                        Text("Recheck")
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface)
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.75)
                )
        )
        .padding(.horizontal, 16)
    }

    /// Install/update command shown as a copyable chip so the user can
    /// paste it into a terminal instead of retyping a raw command.
    private func setupCommandRow(_ command: String) -> some View {
        HStack(spacing: 8) {
            Text(command)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(command, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.typeSecondary)
            }
            .buttonStyle(.plain)
            .help("Copy command")
            .accessibilityLabel("Copy install command")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(Brand.bgInner)
                .overlay(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(Brand.separator, lineWidth: 0.75)
                )
        )
    }

    private func capabilityBlock(_ status: HermesInstallStatus) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Tools")
            capabilityRow("Tools", status.enabledToolsets.joined(separator: ", "))
            if let version = status.versionSummary {
                capabilityRow("Version", version)
            }
            if let update = status.updateSummary {
                capabilityRow("Update", update, color: Brand.warning)
            }
            if let updateCommand = status.updateCommand, status.updateSummary != nil {
                capabilityRow("Command", updateCommand)
            }
            if let gateway = status.gatewaySummary {
                capabilityRow("Messaging", gateway, color: gatewayColor(for: status.gatewayHealth))
            }
            if let repairMessage = hermes.gatewayRepairMessage {
                capabilityRow("Repair", repairMessage)
            }
            ForEach(status.integrationSummaries.prefix(3), id: \.self) { item in
                capabilityRow("Channels", item)
            }
            ForEach(status.warnings.prefix(2), id: \.self) { warning in
                capabilityRow("Heads up", warning, color: Brand.warning)
            }
            // Repair is a deliberate action, so it sits at the bottom as a
            // full-width button rather than wedged between status rows.
            if status.gatewayNeedsRepair {
                gatewayRepairButton
                    .padding(.top, 2)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Brand.cardSurface.opacity(0.72))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.75)
                )
        )
        .padding(.horizontal, 16)
    }

    private var gatewayRepairButton: some View {
        Button {
            Task { await hermes.repairGateway() }
        } label: {
            HStack(spacing: 7) {
                if hermes.gatewayRepairInFlight {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 10, weight: .bold))
                }
                Text(hermes.gatewayRepairInFlight ? "Reconnecting…" : "Reconnect messaging")
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(hermes.gatewayRepairInFlight)
    }

    private func gatewayColor(for health: HermesInstallStatus.GatewayHealth?) -> Color {
        switch health {
        case .healthy:
            return Brand.success
        case .warning, .unavailable:
            return Brand.warning
        case nil:
            return Brand.typeSecondary
        }
    }

    private func capabilityRow(
        _ label: String,
        _ value: String,
        color: Color = Brand.typeSecondary
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label.uppercased())
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .foregroundStyle(Brand.typeTertiary)
                .frame(width: 62, alignment: .leading)
            Text(value)
                .font(.system(size: 10, design: .rounded).weight(.medium))
                .foregroundStyle(color)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var profileList: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("Profiles")
            ForEach(hermes.profiles) { profile in
                Button {
                    Task { await selectProfile(profile) }
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: profile.isDefault ? "person.crop.circle" : "person.2.crop.square.stack")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(Brand.typeSecondary)
                            .frame(width: 16)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(profile.name)
                                .font(.system(.callout, design: .rounded).weight(.medium))
                                .foregroundStyle(Brand.typeBody)
                            Text(profile.path)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(Brand.typeTertiary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        Spacer(minLength: 0)
                        Text(routeLabel(for: hermes.profileRoutingStates[profile.id] ?? .external))
                            .font(.system(size: 9, weight: .heavy, design: .monospaced))
                            .foregroundStyle(routeColor(for: profile))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(profileSelectionBackground(profile))
                }
                .buttonStyle(.plain)
                .disabled(profileUnavailable(profile))
                .help(profileUnavailable(profile)
                    ? "This profile cannot be started safely."
                    : "Select \(profile.name) profile")
                .accessibilityLabel("\(profile.name), \(routeLabel(for: hermes.profileRoutingStates[profile.id] ?? .external))")
            }
        }
        .padding(.horizontal, 12)
    }

    private var sessionList: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionLabel("Agents")
                Spacer()
                Button {
                    Task { await startNew() }
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Brand.typeSecondary)
                        .frame(width: 24, height: 22)
                        .background(Capsule().stroke(Brand.separator, lineWidth: 0.75))
                }
                .buttonStyle(.plain)
                .help("New Agent")
                .accessibilityLabel("Create a new Hermes agent")
                .disabled(hermes.selectedProfile.map(profileUnavailable) ?? true)
            }
            if hermes.sessions.isEmpty {
                Text("No saved agents")
                    .font(.caption)
                    .foregroundStyle(Brand.typeTertiary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
            } else {
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(hermes.sessions) { session in
                            Button {
                                Task { await resume(session) }
                            } label: {
                                sessionRow(session)
                            }
                            .buttonStyle(.plain)
                            .help("Resume \(session.title.isEmpty ? "Untitled Agent" : session.title)")
                            .accessibilityLabel("\(session.title.isEmpty ? "Untitled Agent" : session.title), \(activityLabel(session.activity))")
                        }
                    }
                    .padding(.bottom, 4)
                }
                .frame(maxHeight: 270)
            }
        }
        .padding(.horizontal, 12)
    }

    private var createProfileBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            if creatingProfile {
                TextField("profile-name", text: $createProfileName)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.caption, design: .monospaced))
                HStack {
                    Button("Cancel") {
                        creatingProfile = false
                        createProfileName = ""
                    }
                    .buttonStyle(.plain)
                    Spacer()
                    Button("Create") {
                        Task {
                            guard await ensureDaemonReady() else { return }
                            await hermes.createProfile(
                                named: createProfileName,
                                configuration: backend.configuration
                            )
                            createProfileName = ""
                            creatingProfile = false
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(createProfileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .font(.caption)
                .foregroundStyle(Brand.typeSecondary)
            } else {
                Button {
                    creatingProfile = true
                } label: {
                    Label("Create Profile", systemImage: "person.badge.plus")
                        .font(.system(.caption, design: .rounded).weight(.semibold))
                        .foregroundStyle(Brand.typeSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(hermes.activeSessionTitle ?? "Hermes Agent")
                    .font(.system(.title3, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeBody)
                    .lineLimit(1)
                Text(hermes.selectedProfile?.name ?? "No profile")
                    .font(.caption)
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer()
            if let profile = hermes.selectedProfile {
                Text(routeLabel(for: hermes.profileRoutingStates[profile.id] ?? .external))
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .foregroundStyle(routeColor(for: profile))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Brand.cardSurface))
            }
            if let value = backend.headlineDecode.value {
                Text(String(format: "%.1f tok/s", value))
                    .font(.system(.caption, design: .monospaced).weight(.bold))
                    .foregroundStyle(backend.headlineDecode.isLive ? Brand.success : Brand.typeSecondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(Brand.cardSurface))
            }
            Button {
                Task { await hermes.interrupt() }
            } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(hermes.isStreaming ? Brand.warning : Brand.typeTertiary)
                    .frame(width: 28, height: 24)
                    .background(Capsule().stroke(Brand.separator, lineWidth: 0.75))
            }
            .buttonStyle(.plain)
            .disabled(!hermes.isStreaming)
            .help("Stop Hermes")
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 14)
        .background(Brand.bgInner)
    }

    private var transcript: some View {
        Group {
            if hermes.messages.isEmpty && localError == nil && hermes.toolTraces.isEmpty {
                emptyTranscript
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            if let localError {
                                systemBubble(localError)
                            }
                            ForEach(hermes.messages) { message in
                                messageBubble(message)
                                    .id(message.id)
                            }
                            if !hermes.toolTraces.isEmpty {
                                toolTimeline
                            }
                        }
                        .padding(.horizontal, 28)
                        .padding(.vertical, 24)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .onChange(of: hermes.messages.count) { _, _ in
                        if let last = hermes.messages.last {
                            withAnimation(.smooth(duration: 0.18)) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Brand.bgOuter)
    }

    @ViewBuilder
    private var composer: some View {
        if let pendingRequest = hermes.pendingRequest {
            pendingRequestCard(pendingRequest)
        } else if case .failed = hermes.connectionState {
            reconnectBlock
        } else if let reason = hermes.readOnlyReason {
            readOnlyComposer(reason: reason)
        } else {
            HStack(alignment: .bottom, spacing: 10) {
                TextEditor(text: $composerText)
                    .font(.system(.body, design: .rounded))
                    .foregroundStyle(Brand.typeBody)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 54, maxHeight: 110)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Brand.bgInner)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .stroke(Brand.separatorStrong, lineWidth: 1)
                            )
                    )
                    .disabled(!hermes.activeSessionWritable || !hermes.gatewayReady || hermes.pendingRequest != nil)
                    .accessibilityLabel("Message Hermes")
                Button {
                    Task { await send() }
                } label: {
                    Image(systemName: hermes.isStreaming ? "stop.fill" : "arrow.up")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 36, height: 36)
                        .background(Circle().fill(canSend ? Brand.accentChrome : Brand.typeTertiary.opacity(0.45)))
                }
                .buttonStyle(.plain)
                .disabled(!canSend && !hermes.isStreaming)
                .help(hermes.isStreaming ? "Stop Hermes" : composerDisabledHelp)
                .accessibilityLabel(hermes.isStreaming ? "Stop Hermes" : "Send message")
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(composerBackground)
        }
    }

    private var canSend: Bool {
        if hermes.isStreaming { return true }
        return hermes.activeSessionWritable
            && hermes.gatewayReady
            && hermes.pendingRequest == nil
            && !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var composerBackground: some View {
        Brand.bgInner
            .overlay(
                Rectangle()
                    .fill(Brand.separator)
                    .frame(height: Brand.hairline),
                alignment: .top
            )
    }

    private var composerDisabledHelp: String {
        if !hermes.gatewayReady { return "Hermes is not connected yet." }
        if hermes.pendingRequest != nil { return "Respond to the pending Hermes request first." }
        if hermes.readOnlyReason != nil { return "This session is read-only. Create a new agent to continue." }
        if composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a message to send." }
        return "A writable Hermes agent is required."
    }

    private func readOnlyComposer(reason: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "lock.fill")
                .foregroundStyle(Brand.warning)
            Text(reason)
                .font(.callout)
                .foregroundStyle(Brand.typeSecondary)
            Spacer(minLength: 0)
            Button("New Agent") {
                Task { await startNew() }
            }
            .buttonStyle(.bordered)
            .disabled(hermes.selectedProfile.map(profileUnavailable) ?? true)
            .help("Create a new writable Hermes agent")
            .accessibilityLabel("Create a new writable Hermes agent")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .background(composerBackground)
    }

    private var reconnectBlock: some View {
        HStack(spacing: 12) {
            Image(systemName: "wifi.exclamationmark")
                .foregroundStyle(Brand.warning)
            Text("Hermes disconnected. Your transcript is still available.")
                .font(.callout)
                .foregroundStyle(Brand.typeSecondary)
            Spacer(minLength: 0)
            Button("Reconnect") {
                Task { await reconnect() }
            }
            .buttonStyle(.bordered)
            .help("Reconnect to the selected Hermes profile")
            .accessibilityLabel("Reconnect to Hermes")
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .background(composerBackground)
    }

    @ViewBuilder
    private func pendingRequestCard(_ request: HermesPendingRequest) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: pendingRequestIcon(request.kind))
                    .foregroundStyle(Brand.warning)
                Text(pendingRequestTitle(request.kind))
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeBody)
            }
            Text(request.prompt)
                .font(.callout)
                .foregroundStyle(Brand.typeSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if hermes.approvalPipelineBlocked {
                HStack(spacing: 10) {
                    Text("Approval recovery is required before another response can be sent.")
                        .font(.caption)
                        .foregroundStyle(Brand.warning)
                    Spacer(minLength: 0)
                    Button("Reconnect") { Task { await reconnect() } }
                        .buttonStyle(.bordered)
                        .help("Reconnect before responding to this approval")
                }
            } else {
                pendingRequestControls(request)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .background(composerBackground)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Pending \(pendingRequestTitle(request.kind)) request")
    }

    @ViewBuilder
    private func pendingRequestControls(_ request: HermesPendingRequest) -> some View {
        switch request.kind {
        case .approval:
            HStack(spacing: 8) {
                ForEach(request.choices.filter { $0.lowercased() != "deny" }, id: \.self) { choice in
                    Button(choice.capitalized) {
                        Task { await submitPendingRequest(request, value: choice) }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!pendingRequestActionsEnabled(request))
                    .help("Allow this Hermes action: \(choice)")
                    .accessibilityLabel("Allow action: \(choice)")
                }
                Button("Deny") {
                    Task { await submitPendingRequest(request, value: "deny", isDenial: true) }
                }
                .buttonStyle(.bordered)
                .disabled(!pendingRequestActionsEnabled(request))
                .help("Deny this Hermes action")
                .accessibilityLabel("Deny Hermes action")
            }
        case .clarification:
            pendingTextResponse(
                label: "Clarification",
                placeholder: "Type your answer",
                sensitive: false,
                request: request
            )
        case .sudo:
            pendingTextResponse(
                label: "Sudo password",
                placeholder: "Enter password",
                sensitive: true,
                request: request
            )
        case .secret:
            pendingTextResponse(
                label: "Secret",
                placeholder: "Enter secret",
                sensitive: true,
                request: request
            )
        }
    }

    @ViewBuilder
    private func pendingTextResponse(
        label: String,
        placeholder: String,
        sensitive: Bool,
        request: HermesPendingRequest
    ) -> some View {
        HStack(spacing: 8) {
            if sensitive {
                SecureField(placeholder, text: $pendingResponseText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(!pendingRequestActionsEnabled(request))
                    .accessibilityLabel(label)
            } else {
                TextField(placeholder, text: $pendingResponseText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(!pendingRequestActionsEnabled(request))
                    .accessibilityLabel(label)
            }
            Button("Submit") {
                submitPendingTextResponse(request)
            }
            .buttonStyle(.borderedProminent)
            .disabled(pendingResponseText.isEmpty || !pendingRequestActionsEnabled(request))
            .keyboardShortcut(.defaultAction)
            .help("Submit \(label.lowercased()) to Hermes")
            .accessibilityLabel("Submit \(label.lowercased())")
        }
    }

    private func submitPendingTextResponse(_ request: HermesPendingRequest) {
        guard pendingRequestActionsEnabled(request) else { return }
        let value = pendingResponseText
        pendingResponseText = ""
        Task { await submitPendingRequest(request, value: value) }
    }

    private func pendingRequestActionsEnabled(_ request: HermesPendingRequest) -> Bool {
        pendingSubmissionID == nil
            && hermes.pendingRequest?.id == request.id
            && hermes.pendingRequest?.kind == request.kind
            && hermes.activeSessionWritable
            && hermes.gatewayReady
            && !hermes.approvalPipelineBlocked
    }

    private func submitPendingRequest(
        _ request: HermesPendingRequest,
        value: String,
        isDenial: Bool = false
    ) async {
        guard pendingRequestActionsEnabled(request) else { return }
        let submissionID = request.id
        pendingSubmissionID = submissionID
        defer {
            // A completion from an old request must never mutate the state of
            // a newer card. Its ID is the local submission lifecycle token.
            if pendingSubmissionID == submissionID {
                pendingSubmissionID = nil
            }
        }
        if isDenial {
            await hermes.denyPendingApproval()
        } else {
            await hermes.respondToPendingRequest(value: value)
        }
    }

    private var emptyTranscript: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Brand.accentChrome.opacity(0.12))
                    .overlay {
                        Circle().strokeBorder(Brand.accentChrome.opacity(0.30), lineWidth: Brand.hairline)
                    }
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Brand.accentChrome)
            }
            .frame(width: 72, height: 72)

            Text(hermes.selectedProfile == nil ? "Select a Hermes Profile" : "Start a Hermes Agent")
                .font(.system(.title3, design: .rounded).weight(.semibold))
                .foregroundStyle(Brand.typeBody)

            Text(hermes.selectedProfile == nil
                ? "Choose a readable profile to load its saved agents."
                : "Create a new agent or resume a saved one to chat directly in MTPLX.")
                .font(.callout)
                .foregroundStyle(Brand.typeSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 420)

            Button {
                Task { await startNew() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "plus")
                        .font(.system(size: 12, weight: .bold))
                    Text("New Agent")
                }
            }
            .buttonStyle(.mtplxPrimary)
            .padding(.top, 2)
            .disabled(hermes.selectedProfile.map(profileUnavailable) ?? true)
            .help(hermes.selectedProfile == nil ? "Select a profile first." : "Create a new Hermes agent")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        .padding(40)
    }

    private var toolTimeline: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(hermes.toolTraces) { trace in
                HStack(alignment: .top, spacing: 9) {
                    Image(systemName: icon(for: trace.status))
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(color(for: trace.status))
                        .frame(width: 16)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(trace.name)
                            .font(.system(.caption, design: .rounded).weight(.semibold))
                            .foregroundStyle(Brand.typeSecondary)
                        if !trace.detail.isEmpty {
                            Text(trace.detail)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(Brand.typeTertiary)
                                .lineLimit(4)
                        }
                    }
                }
                .padding(10)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Brand.cardSurface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.75)
                        )
                )
                .frame(maxWidth: 640, alignment: .leading)
            }
        }
    }

    private func prepare() async {
        await hermes.prepare(configuration: backend.configuration)
        guard case .needsSetup = hermes.connectionState else {
            guard router.hermesLaunchIntent == .resumeLast else {
                if let profile = hermes.selectedProfile,
                   await ensureDaemonReady() {
                    await hermes.loadSessions(profile: profile, configuration: backend.configuration)
                }
                return
            }
            guard !handledResumeIntent else { return }
            handledResumeIntent = true
            await resumeLast()
            return
        }
    }

    private func selectProfile(_ profile: HermesProfile) async {
        localError = nil
        guard !profileUnavailable(profile) else {
            localError = "This Hermes profile is unavailable."
            return
        }
        guard await ensureDaemonReady() else { return }
        await hermes.loadSessions(profile: profile, configuration: backend.configuration)
    }

    private func ensureDaemonReady() async -> Bool {
        guard backend.daemonState.kind != .running else { return true }
        await backend.startDaemon(target: nil)
        guard backend.daemonState.kind == .running else {
            localError = "MTPLX is not ready yet."
            return false
        }
        return true
    }

    private func startNew() async {
        guard let profile = hermes.selectedProfile else { return }
        localError = nil
        guard !profileUnavailable(profile) else {
            localError = "This Hermes profile is unavailable."
            return
        }
        guard await ensureDaemonReady() else { return }
        do {
            let reference = try await hermes.startNewAgent(
                profile: profile,
                configuration: backend.configuration
            )
            remember(reference)
        } catch {
            localError = "Hermes could not create a new agent. Try again."
        }
    }

    private func resume(_ session: HermesSavedSession) async {
        guard let profile = hermes.selectedProfile else { return }
        localError = nil
        guard await ensureDaemonReady() else { return }
        do {
            let reference = try await hermes.resume(
                session,
                profile: profile,
                configuration: backend.configuration
            )
            remember(reference)
        } catch {
            localError = "Hermes could not resume this agent. Try again."
        }
    }

    private func reconnect() async {
        localError = nil
        guard await ensureDaemonReady() else { return }
        do {
            try await hermes.reconnect(configuration: backend.configuration)
        } catch {
            localError = "Hermes could not reconnect. Try again."
        }
    }

    private func resumeLast() async {
        localError = nil
        guard await ensureDaemonReady() else { return }
        do {
            let reference = try await hermes.resumeLast(configuration: backend.configuration)
            remember(reference)
        } catch {
            localError = "Hermes could not resume the last agent. Try again."
            router.hermesLaunchIntent = .browse
        }
    }

    private func send() async {
        if hermes.isStreaming {
            await hermes.interrupt()
            return
        }
        let text = composerText
        composerText = ""
        await hermes.send(text)
    }

    private func remember(_ reference: HermesSessionReference) {
        var config = backend.configuration
        config.lastLaunchTarget = LaunchTarget.hermes.rawValue
        config.lastHermesProfile = reference.profileName
        config.lastHermesSessionID = reference.sessionID
        config.lastHermesSessionTitle = reference.title
        try? backend.saveSettings(config)
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 9, weight: .heavy, design: .monospaced))
            .tracking(1.5)
            .foregroundStyle(Brand.typeTertiary)
            .padding(.horizontal, 4)
    }

    private func profileSelectionBackground(_ profile: HermesProfile) -> some View {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
            .fill(hermes.selectedProfile?.id == profile.id ? Brand.cardSurface : Color.clear)
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(
                        hermes.selectedProfile?.id == profile.id ? Brand.separatorStrong : Color.clear,
                        lineWidth: 0.75
                    )
            )
    }

    private func profileUnavailable(_ profile: HermesProfile) -> Bool {
        if case .unavailable = hermes.profileRoutingStates[profile.id] { return true }
        return false
    }

    private func routeLabel(for route: HermesProfileRoutingState) -> String {
        switch route {
        case .mtplx: "MTPLX"
        case .external: "External"
        case .unavailable: "Unavailable"
        }
    }

    private func routeColor(for profile: HermesProfile) -> Color {
        switch hermes.profileRoutingStates[profile.id] ?? .external {
        case .mtplx: Brand.success
        case .external: Brand.typeSecondary
        case .unavailable: Brand.warning
        }
    }

    private func activityLabel(_ activity: HermesSessionActivityState) -> String {
        switch activity {
        case .ready: "Ready"
        case .runningInMTPLX: "Running in MTPLX"
        case .externallyActive: "Externally active"
        case .ownershipUnknown: "Ownership unknown"
        }
    }

    private func activityColor(_ activity: HermesSessionActivityState) -> Color {
        switch activity {
        case .ready: Brand.success
        case .runningInMTPLX: Brand.accentChrome
        case .externallyActive, .ownershipUnknown: Brand.warning
        }
    }

    private func pendingRequestTitle(_ kind: HermesPendingRequestKind) -> String {
        switch kind {
        case .approval: "Approval needed"
        case .clarification: "Clarification needed"
        case .sudo: "Sudo authentication needed"
        case .secret: "Secret needed"
        }
    }

    private func pendingRequestIcon(_ kind: HermesPendingRequestKind) -> String {
        switch kind {
        case .approval: "checkmark.shield"
        case .clarification: "questionmark.bubble"
        case .sudo: "lock.shield"
        case .secret: "key.fill"
        }
    }

    private func sessionRow(_ session: HermesSavedSession) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(session.title.isEmpty ? "Untitled Agent" : session.title)
                    .font(.system(.callout, design: .rounded).weight(.medium))
                    .foregroundStyle(Brand.typeBody)
                    .lineLimit(1)
                Spacer()
                Text("\(session.messageCount)")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Text(activityLabel(session.activity))
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(activityColor(session.activity))
            if !session.preview.isEmpty {
                Text(session.preview)
                    .font(.caption)
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(2)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(
                    backend.configuration.lastHermesSessionID == session.id
                        ? Brand.success.opacity(0.10)
                        : Brand.cardSurface.opacity(0.75)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.75)
                )
        )
    }

    private func messageBubble(_ message: HermesTranscriptMessage) -> some View {
        HStack {
            if message.role == .user { Spacer(minLength: 80) }
            Text(message.text.isEmpty && message.isStreaming ? "..." : message.text)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(Brand.typeBody)
                .textSelection(.enabled)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(message.role == .user ? Brand.accentChrome.opacity(0.18) : Brand.cardSurface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.75)
                        )
                )
                .frame(maxWidth: 720, alignment: message.role == .user ? .trailing : .leading)
            if message.role != .user { Spacer(minLength: 80) }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    private func systemBubble(_ text: String) -> some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(Brand.warning)
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Brand.warning.opacity(0.10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(Brand.warning.opacity(0.35), lineWidth: 0.75)
                    )
            )
            .frame(maxWidth: 680, alignment: .leading)
    }

    private func icon(for status: HermesToolStatus) -> String {
        switch status {
        case .running: return "gearshape.2"
        case .complete: return "checkmark.circle"
        case .approval: return "bolt.circle"
        case .waiting: return "questionmark.circle"
        case .failed: return "xmark.circle"
        }
    }

    private func color(for status: HermesToolStatus) -> Color {
        switch status {
        case .running: return Brand.accentChrome
        case .complete: return Brand.success
        case .approval: return Brand.warning
        case .waiting: return Brand.typeSecondary
        case .failed: return Brand.warning
        }
    }
}
