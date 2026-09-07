import AppKit
import SwiftUI
import MTPLXAppCore

struct AgentWorkspacePanel: View {
    @EnvironmentObject private var backend: MTPLXBackendStore
    @EnvironmentObject private var chatViewModel: ChatViewModel
    @State private var isCreatingWorkspace = false
    @State private var panelError: String?
    @State private var selectedGraphDefinitionID: String?
    @State private var graphInputsJSON = "{}"
    @State private var sideEffectRetryRun: AgentGraphRun?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    workspaceSection
                    modelSection
                    modeSection
                    goalSection
                    runSection
                    graphSection
                    delegationSection
                    approvalSection
                    if let panelError {
                        Text(panelError)
                            .font(.system(size: 10, design: .rounded))
                            .foregroundStyle(Brand.warning)
                    }
                }
                .padding(14)
            }
        }
        .background(
            Brand.bgInner
                .overlay(Rectangle().fill(Brand.separator).frame(width: 0.5), alignment: .leading)
        )
        .task {
            await backend.refreshAgentState()
        }
        .task(id: backend.activeRunID) {
            while !Task.isCancelled {
                await backend.refreshRun()
                try? await Task.sleep(for: .seconds(2))
            }
        }
        .task(id: backend.activeGraphRunID) {
            while !Task.isCancelled {
                await backend.refreshGraphState()
                try? await Task.sleep(for: .seconds(2))
            }
        }
        .confirmationDialog(
            "Retry side-effect node?",
            isPresented: Binding(
                get: { sideEffectRetryRun != nil },
                set: { if !$0 { sideEffectRetryRun = nil } }
            ),
            presenting: sideEffectRetryRun
        ) { run in
            Button("Retry exact failed node", role: .destructive) {
                sideEffectRetryRun = nil
                Task { await backend.retryGraphRun(run, allowSideEffectRetry: true) }
            }
            Button("Cancel", role: .cancel) {
                sideEffectRetryRun = nil
            }
        } message: { run in
            Text(
                "This can write files, run a command, run tests, or update memory. "
                + "MTPLX will preserve its idempotency guard when recovery requires it. "
                + "Run: \(run.id)"
            )
        }
    }

    private var header: some View {
        HStack {
            Text("Agent workspace")
                .font(.system(size: 10, weight: .heavy, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(Brand.typeTertiary)
            Spacer()
            Circle()
                .fill(backend.pendingApprovals.isEmpty ? Brand.success : Brand.warning)
                .frame(width: 6, height: 6)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .overlay(Rectangle().fill(Brand.separator).frame(height: 0.5), alignment: .bottom)
    }

    private var workspaceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("WORKSPACE")
            if backend.workspaces.isEmpty {
                Text("Choose a local project to give the agent a bounded working root.")
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(Brand.typeTertiary)
                Button {
                    chooseWorkspaceDirectory()
                } label: {
                    Label(isCreatingWorkspace ? "Adding project…" : "Add project", systemImage: "folder.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .tint(Brand.accentChrome)
                .disabled(isCreatingWorkspace)
            } else {
                Picker("Project", selection: workspaceSelection) {
                    Text("No project").tag(String?.none)
                    ForEach(backend.workspaces) { workspace in
                        Text(workspace.name).tag(Optional(workspace.id))
                    }
                }
                .labelsHidden()
                if let workspace = activeWorkspace {
                    Text(workspace.rootPath)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(2)
                        .textSelection(.enabled)
                    Text("read allow  ·  write ask  ·  terminal ask")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                }
                Button {
                    chooseWorkspaceDirectory()
                } label: {
                    Label("Add another project", systemImage: "plus")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.plain)
                .foregroundStyle(Brand.accentChrome)
            }
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("MODEL")
            let model = chatViewModel.current?.modelOverride
                ?? backend.agentModels?.models.first?.id
                ?? backend.health?.model
                ?? backend.configuration.model
            Text(model)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(2)
            if let info = backend.agentModels?.models.first {
                HStack(spacing: 8) {
                    capabilityPill(info.reasoning ? "reasoning" : "direct")
                    if info.vision { capabilityPill("vision") }
                    if let window = info.contextWindow {
                        capabilityPill("\(window / 1000)k ctx")
                    }
                }
            }
            Text("MTPLX local runtime")
                .font(.system(size: 10, design: .rounded))
                .foregroundStyle(Brand.typeTertiary)
        }
    }

    private var modeSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("CHAT MODE")
            Toggle("Plan mode", isOn: planModeBinding)
                .toggleStyle(.switch)
                .font(.system(size: 11, design: .rounded))
            Picker("Reasoning", selection: reasoningBinding) {
                Text("Auto").tag("auto")
                Text("Low").tag("low")
                Text("Medium").tag("medium")
                Text("High").tag("high")
            }
            .font(.system(size: 11, design: .rounded))
        }
    }

    private var goalSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionLabel("GOAL MODE")
                Spacer()
                Text(goalIsActive ? "ACTIVE" : "OFF")
                    .font(.system(size: 8, weight: .heavy, design: .monospaced))
                    .foregroundStyle(goalIsActive ? Brand.success : Brand.typeTertiary)
            }
            TextField(
                "Set a goal to keep pursuing",
                text: goalBinding,
                axis: .vertical
            )
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 10, design: .rounded))
            .lineLimit(2...4)
            if goalIsActive {
                HStack(spacing: 8) {
                    Text("Included in every agent turn.")
                        .font(.system(size: 9, design: .rounded))
                        .foregroundStyle(Brand.typeTertiary)
                    Spacer()
                    Button("Clear") {
                        chatViewModel.setGoal(nil)
                        chatViewModel.dismissCommandOutput()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 9, weight: .semibold, design: .rounded))
                    .foregroundStyle(Brand.accentChrome)
                }
            } else {
                Text("Set a durable objective for this conversation.")
                    .font(.system(size: 9, design: .rounded))
                    .foregroundStyle(Brand.typeTertiary)
            }
        }
    }

    private var approvalSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("APPROVALS")
            if backend.pendingApprovals.isEmpty {
                Text("No pending actions.")
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(Brand.typeTertiary)
            } else {
                ForEach(backend.pendingApprovals) { approval in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(approval.action)
                            .font(.system(size: 11, weight: .semibold, design: .rounded))
                            .foregroundStyle(Brand.typeSecondary)
                        Text(approval.target ?? approval.description)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(3)
                        Text("Risk: \(approval.risk)" + (approval.expiresAt.map { " · expires \($0.formatted(date: .omitted, time: .shortened))" } ?? ""))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(Brand.warning)
                        HStack(spacing: 6) {
                            Button("Allow") {
                                Task { await backend.resolveApproval(approval, decision: "approved") }
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(Brand.success)
                            Button("Deny") {
                                Task { await backend.resolveApproval(approval, decision: "denied") }
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.white.opacity(0.035))
                    )
                }
            }
        }
    }

    private var delegationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionLabel("DELEGATED AGENTS")
                Spacer()
                Button {
                    chatViewModel.send("/review")
                } label: {
                    Image(systemName: "person.2")
                }
                .buttonStyle(.plain)
                .foregroundStyle(Brand.accentChrome)
                .help("Delegate a reviewer")
            }
            if backend.delegations.isEmpty {
                Text("No delegated agents for this run.")
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(Brand.typeTertiary)
            } else {
                ForEach(backend.delegations) { delegation in
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(delegation.role) · \(delegation.status)")
                            .font(.system(size: 10, weight: .semibold, design: .monospaced))
                            .foregroundStyle(delegation.status == "completed" ? Brand.success : Brand.typeSecondary)
                        if let tokensUsed = delegation.tokensUsed {
                            Text(
                                "Budget \(tokensUsed)/\(delegation.budget) tok · "
                                    + "\(delegation.remainingTokenBudget) left · "
                                    + "attempt \(delegation.attempts ?? 0)"
                            )
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(
                                delegation.remainingTokenBudget < 256
                                    ? Brand.warning
                                    : Brand.typeTertiary
                            )
                        } else {
                            Text("Budget \(delegation.budget) tok")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(Brand.typeTertiary)
                        }
                        if let error = delegation.error {
                            Text(error)
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(Brand.warning)
                                .lineLimit(2)
                        } else if let path = delegation.worktreePath {
                            Text(path)
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(2)
                        }
                        HStack(spacing: 8) {
                            if delegation.status == "queued" || delegation.status == "running" {
                                Button("Stop") {
                                    Task { await backend.cancelDelegation(delegation) }
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                            if delegation.status == "paused" || delegation.status == "failed" || delegation.status == "cancelled" {
                                Button("Retry") {
                                    Task { await backend.retryDelegation(delegation) }
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                            if delegation.worktreePath != nil {
                                Button("Check worktree") {
                                    Task { await checkWorktree(delegation) }
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.mini)
                            }
                        }
                    }
                    .padding(.vertical, 3)
                }
            }
        }
    }

    private var runSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionLabel("RUN TIMELINE")
            if backend.agentRuns.isEmpty {
                Text("No durable agent runs yet.")
                    .font(.system(size: 11, design: .rounded))
                    .foregroundStyle(Brand.typeTertiary)
            } else {
                Picker("Run", selection: runSelection) {
                    ForEach(backend.agentRuns) { run in
                        Text("\(run.title) · \(run.status)").tag(Optional(run.id))
                    }
                }
                .labelsHidden()
                ForEach(Array(backend.activeRunEvents.suffix(10).reversed())) { event in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(event.kind.replacingOccurrences(of: "_", with: " ").uppercased())
                            .font(.system(size: 9, weight: .semibold, design: .monospaced))
                            .foregroundStyle(eventColor(event.kind))
                        Text(eventDetail(event))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(2)
                    }
                }
            }
        }
    }

    private var graphSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionLabel("GRAPHS")
                Spacer()
                Text("DURABLE")
                    .font(.system(size: 8, weight: .heavy, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            if !backend.graphDefinitions.isEmpty {
                Picker("Graph", selection: graphDefinitionSelection) {
                    ForEach(backend.graphDefinitions) { graph in
                        Text("\(graph.name) · r\(graph.revision)")
                            .tag(Optional(graph.id))
                    }
                }
                .labelsHidden()
                TextField("Inputs JSON", text: $graphInputsJSON)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 9, design: .monospaced))
                Button {
                    startSelectedGraph()
                } label: {
                    Label("Run Graph", systemImage: "play.fill")
                }
                .buttonStyle(.bordered)
                .disabled(selectedGraphDefinition == nil)
            }
            if backend.graphRuns.isEmpty {
                Text(
                    backend.graphDefinitions.isEmpty
                        ? "No Graph definitions for this workspace."
                        : "No Graph runs yet."
                )
                .font(.system(size: 11, design: .rounded))
                .foregroundStyle(Brand.typeTertiary)
            } else {
                Picker("Graph run", selection: graphRunSelection) {
                    ForEach(backend.graphRuns) { run in
                        Text("\(graphName(run.graphID)) · \(run.status)")
                            .tag(Optional(run.id))
                    }
                }
                .labelsHidden()
                if let run = backend.activeGraphRun {
                    HStack(spacing: 6) {
                        capabilityPill("r\(run.graphRevision)")
                        capabilityPill(run.runtimeProfile)
                        Text(run.status.uppercased())
                            .font(.system(size: 9, weight: .heavy, design: .monospaced))
                            .foregroundStyle(graphStatusColor(run.status))
                    }
                    Text(run.pinnedModel ?? "MTPLX runtime model")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(2)
                    graphRunControls(run)
                    if let summary = graphResourceSummary(run) {
                        Text(summary)
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundStyle(Brand.typeTertiary)
                            .lineLimit(3)
                    }
                    if let approvalID = run.pendingApprovalID {
                        if let approval = backend.activeGraphRunApprovals.first(where: {
                            $0.id == approvalID
                        }) {
                            graphApprovalCard(approval)
                        } else {
                            Text("Approval pending · \(approvalID)")
                                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                                .foregroundStyle(Brand.warning)
                                .lineLimit(2)
                        }
                    }
                    ForEach(graphNodeIDs(run), id: \.self) { nodeID in
                        graphNodeRow(run: run, nodeID: nodeID)
                    }
                    if let error = run.error, !error.isEmpty {
                        Text(error)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(Brand.warning)
                            .lineLimit(3)
                    }
                    ForEach(Array(backend.activeGraphRunEvents.suffix(6).reversed())) { event in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(event.kind.replacingOccurrences(of: "_", with: " ").uppercased())
                                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                                .foregroundStyle(eventColor(event.kind))
                            Text(eventDetail(event))
                                .font(.system(size: 8, design: .monospaced))
                                .foregroundStyle(Brand.typeTertiary)
                                .lineLimit(2)
                        }
                    }
                }
            }
        }
    }

    private var activeWorkspace: AgentWorkspace? {
        backend.workspaces.first { $0.id == backend.activeWorkspaceID }
    }

    private var workspaceSelection: Binding<String?> {
        Binding(
            get: { backend.activeWorkspaceID },
            set: { value in
                backend.activeWorkspaceID = value
                chatViewModel.setWorkspaceID(value)
                Task { await backend.selectWorkspace(value) }
            }
        )
    }

    private var planModeBinding: Binding<Bool> {
        Binding(
            get: { chatViewModel.current?.planModeEnabled ?? false },
            set: { value in
                chatViewModel.setPlanMode(value)
                chatViewModel.dismissCommandOutput()
            }
        )
    }

    private var goalBinding: Binding<String> {
        Binding(
            get: { chatViewModel.current?.goalText ?? "" },
            set: { value in
                chatViewModel.setGoal(value)
                chatViewModel.dismissCommandOutput()
            }
        )
    }

    private var goalIsActive: Bool {
        !(chatViewModel.current?.goalText ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
    }

    private var reasoningBinding: Binding<String> {
        Binding(
            get: { chatViewModel.current?.reasoningEffortRaw ?? "auto" },
            set: { value in chatViewModel.setReasoningEffort(value) }
        )
    }

    private var runSelection: Binding<String?> {
        Binding(
            get: { backend.activeRunID },
            set: { value in
                Task { await backend.selectRun(value) }
            }
        )
    }

    private var graphRunSelection: Binding<String?> {
        Binding(
            get: { backend.activeGraphRunID },
            set: { value in
                Task { await backend.selectGraphRun(value) }
            }
        )
    }

    private var graphDefinitionSelection: Binding<String?> {
        Binding(
            get: { selectedGraphDefinition?.id },
            set: { selectedGraphDefinitionID = $0 }
        )
    }

    private var selectedGraphDefinition: AgentGraphDefinition? {
        if let selectedGraphDefinitionID,
           let selected = backend.graphDefinitions.first(where: {
               $0.id == selectedGraphDefinitionID
           })
        {
            return selected
        }
        return backend.graphDefinitions.first
    }

    @ViewBuilder
    private func graphRunControls(_ run: AgentGraphRun) -> some View {
        HStack(spacing: 6) {
            switch run.status {
            case "queued", "running":
                Button("Pause") {
                    Task { await backend.pauseGraphRun(run) }
                }
                Button("Cancel", role: .destructive) {
                    Task { await backend.cancelGraphRun(run) }
                }
            case "waiting_approval":
                Button("Cancel", role: .destructive) {
                    Task { await backend.cancelGraphRun(run) }
                }
            case "paused":
                Button("Resume") {
                    Task { await backend.resumeGraphRun(run) }
                }
                Button("Cancel", role: .destructive) {
                    Task { await backend.cancelGraphRun(run) }
                }
            case "failed":
                if graphFailedNodeHasSideEffect(run) {
                    Button("Retry side effect", role: .destructive) {
                        sideEffectRetryRun = run
                    }
                } else {
                    Button("Retry") {
                        Task { await backend.retryGraphRun(run) }
                    }
                }
            default:
                EmptyView()
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    private func startSelectedGraph() {
        guard let graph = selectedGraphDefinition else { return }
        do {
            let data = Data(graphInputsJSON.utf8)
            let inputs = try JSONDecoder().decode(DynamicObject.self, from: data)
            panelError = nil
            Task { await backend.startGraphRun(graph, inputs: inputs) }
        } catch {
            panelError = "Graph inputs must be one valid JSON object: \(error.localizedDescription)"
        }
    }

    private func graphApprovalCard(_ approval: AgentApproval) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(approval.action)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(Brand.warning)
            Text(approval.target ?? approval.description)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(Brand.typeSecondary)
                .lineLimit(3)
            HStack(spacing: 6) {
                Text("Risk \(approval.risk)")
                if let hash = approval.argumentsSHA256 {
                    Text("args \(hash.prefix(10))")
                }
                if let expiry = approval.expiresAt {
                    Text("expires \(expiry.formatted(date: .omitted, time: .shortened))")
                }
            }
            .font(.system(size: 8, design: .monospaced))
            .foregroundStyle(Brand.typeTertiary)
            HStack(spacing: 6) {
                Button("Allow") {
                    Task { await backend.resolveApproval(approval, decision: "approved") }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
                Button("Deny") {
                    Task { await backend.resolveApproval(approval, decision: "denied") }
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }
        }
        .padding(7)
        .background(Brand.warning.opacity(0.08), in: RoundedRectangle(cornerRadius: 7))
    }

    private func graphFailedNodeHasSideEffect(_ run: AgentGraphRun) -> Bool {
        guard let nodeID = run.currentNodeID,
              let graph = backend.graphDefinitions.first(where: {
                  $0.id == run.graphID && $0.revision == run.graphRevision
              }),
              let node = graph.nodes.first(where: { $0.id == nodeID })
        else {
            return true
        }
        if ["memory_write", "memory_curate"].contains(node.type) {
            return true
        }
        if node.type == "tool" {
            let tool = node.config.values["tool"]?.stringValue ?? ""
            return ["write_file", "apply_patch", "run_tests", "run_command"].contains(tool)
        }
        if node.type == "loop",
           case .object(let body)? = node.config.values["body"],
           let type = body["type"]?.stringValue
        {
            if ["memory_write", "memory_curate"].contains(type) {
                return true
            }
            if type == "tool",
               case .object(let config)? = body["config"],
               let tool = config["tool"]?.stringValue
            {
                return ["write_file", "apply_patch", "run_tests", "run_command"].contains(tool)
            }
        }
        return false
    }

    private func graphName(_ graphID: String) -> String {
        backend.graphDefinitions.first(where: { $0.id == graphID })?.name ?? graphID
    }

    private func graphNodeIDs(_ run: AgentGraphRun) -> [String] {
        if let definition = backend.graphDefinitions.first(where: { graph in
            graph.id == run.graphID && graph.revision == run.graphRevision
        }) {
            return definition.nodes.map(\.id)
        }
        return run.nodeStates.keys.sorted()
    }

    @ViewBuilder
    private func graphNodeRow(run: AgentGraphRun, nodeID: String) -> some View {
        let state = run.nodeStates[nodeID]
        let status = state?.values["status"]?.stringValue ?? "unknown"
        let definition = backend.graphDefinitions.first(where: { graph in
            graph.id == run.graphID && graph.revision == run.graphRevision
        })
        let node = definition?.nodes.first(where: { $0.id == nodeID })
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 5) {
                Circle()
                    .fill(graphStatusColor(status))
                    .frame(width: 5, height: 5)
                Text(node?.name ?? nodeID)
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Brand.typeSecondary)
                Spacer()
                Text(status)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            if node?.type == "loop" {
                let completed = state?.values["iterations_completed"]?.intValue ?? 0
                let maximum = node?.config.values["max_iterations"]?.intValue ?? 0
                Text("Loop \(completed) of \(maximum)")
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
            }
            if let summary = graphNodeMetricSummary(state) {
                Text(summary)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(Brand.typeTertiary)
                    .lineLimit(2)
            }
            if let error = state?.values["error"]?.stringValue, !error.isEmpty {
                Text(error)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(Brand.warning)
                    .lineLimit(2)
            }
        }
    }

    private func graphNodeMetricSummary(_ state: DynamicObject?) -> String? {
        guard let state else { return nil }
        var parts: [String] = []
        let attempts = state.values["attempts"]?.intValue ?? 0
        if attempts > 1 {
            parts.append("attempt \(attempts)")
        }
        guard let metrics = state.values["metrics"]?.objectValue else {
            return parts.isEmpty ? nil : parts.joined(separator: " · ")
        }
        if let latency = metrics["latency_ms"]?.intValue ?? metrics["elapsed_ms"]?.intValue {
            parts.append("\(latency) ms")
        }
        if let wait = metrics["model_load_wait_ms"]?.intValue, wait > 0 {
            parts.append("model wait \(wait) ms")
        }
        if let usage = metrics["usage"]?.objectValue {
            let prompt = usage["prompt_tokens"]?.intValue ?? 0
            let completion = usage["completion_tokens"]?.intValue ?? 0
            if prompt + completion > 0 {
                parts.append("\(prompt)+\(completion) tok")
            }
        }
        if let decode = metrics["decode_tok_s"]?.doubleValue, decode > 0 {
            parts.append(String(format: "%.1f tok/s", decode))
        }
        if let acceptance = metrics["acceptance_rate"]?.doubleValue {
            parts.append(String(format: "%.0f%% accept", acceptance * 100))
        }
        if metrics["session_cache_hit"]?.boolValue == true {
            parts.append("session hit")
        }
        if metrics["ssd_cache_hit"]?.boolValue == true {
            parts.append("SSD hit")
        }
        if metrics["fallback_used"]?.boolValue == true {
            parts.append("model fallback")
        }
        if metrics["replayed"]?.boolValue == true {
            parts.append("idempotent replay")
        }
        if let guardValue = state.values["recovery_guard"]?.stringValue,
           !guardValue.isEmpty
        {
            parts.append(guardValue.replacingOccurrences(of: "_", with: " "))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func graphResourceSummary(_ run: AgentGraphRun) -> String? {
        guard let snapshot = run.resourceMetrics.values["latest_resource_snapshot"]?.objectValue else {
            return nil
        }
        var parts: [String] = []
        if let backend = snapshot["backend_id"]?.stringValue, !backend.isEmpty {
            parts.append(backend)
        }
        if let active = snapshot["active_memory_bytes"]?.doubleValue, active > 0 {
            parts.append("memory \(Self.byteText(active))")
        }
        if let pressure = snapshot["memory_pressure_level"]?.intValue, pressure > 0 {
            parts.append("pressure \(pressure)")
        }
        if snapshot["thermal_throttled"]?.boolValue == true {
            parts.append("thermal throttle")
        }
        if let context = snapshot["context_window"]?.intValue, context > 0 {
            parts.append("context \(context)")
        }
        if let elapsed = run.resourceMetrics.values["active_elapsed_seconds"]?.doubleValue,
           elapsed > 0
        {
            parts.append(String(format: "active %.2f s", elapsed))
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private static func byteText(_ value: Double) -> String {
        ByteCountFormatter.string(
            fromByteCount: Int64(value),
            countStyle: .memory
        )
    }

    private func graphStatusColor(_ status: String) -> Color {
        switch status {
        case "completed": return Brand.success
        case "failed", "cancelled": return Brand.warning
        case "waiting_approval": return Brand.warning
        case "running": return Brand.accentChrome
        default: return Brand.typeTertiary
        }
    }

    private func eventDetail(_ event: AgentRunEvent) -> String {
        let values = event.payload.values
        for key in ["tool", "path", "command", "target", "plan", "content", "result", "error"] {
            if let value = values[key]?.stringValue, !value.isEmpty {
                return String(value.prefix(180))
            }
        }
        return event.createdAt.formatted(date: .omitted, time: .shortened)
    }

    private func eventColor(_ kind: String) -> Color {
        switch kind {
        case "run_failed", "run_cancelled", "graph_failed", "graph_cancelled": return Brand.warning
        case "run_completed", "graph_completed", "test_completed", "file_changed": return Brand.success
        case "approval_requested": return Brand.warning
        default: return Brand.typeTertiary
        }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .heavy, design: .monospaced))
            .tracking(1.1)
            .foregroundStyle(Brand.typeTertiary)
    }

    private func capabilityPill(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .semibold, design: .monospaced))
            .foregroundStyle(Brand.typeTertiary)
            .padding(.horizontal, 5)
            .padding(.vertical, 3)
            .background(Capsule().fill(Color.white.opacity(0.05)))
    }

    private func chooseWorkspaceDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Project"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let name = url.lastPathComponent.isEmpty ? "Project" : url.lastPathComponent
        isCreatingWorkspace = true
        panelError = nil
        Task { @MainActor in
            defer { isCreatingWorkspace = false }
            do {
                let workspace = try await backend.createWorkspace(name: name, rootPath: url.path)
                chatViewModel.setWorkspaceID(workspace.id)
            } catch {
                panelError = error.localizedDescription
            }
        }
    }

    private func checkWorktree(_ delegation: AgentDelegation) async {
        do {
            let result = try await backend.apiClient.delegationWorktree(delegationID: delegation.id)
            let mergeable = result.values["merge_check"]
                .flatMap { value -> Bool? in
                    guard case .object(let object) = value else { return nil }
                    return object["mergeable"]?.boolValue
                }
            panelError = mergeable == true
                ? "\(delegation.role) worktree passed the mergeability check. No merge was performed."
                : "\(delegation.role) worktree needs review before merge."
        } catch {
            panelError = "Worktree check failed: \(error.localizedDescription)"
        }
    }
}
