import SwiftUI
import MTPLXAppCore

// MARK: - Settings catalog
//
// Every setting the Settings page shows, in one place: which section and
// group it lives in, its label, its one-line help and extra search words.
// Rows read their label and caption from here, so search, the sidebar's
// match counts and what is on screen can never disagree. Changing a label
// here never touches a saved setting: the stored keys live in
// `MTPLXAppConfiguration`, `ThemeStore` and `LanguageStore`.

/// The Settings page's sections, in sidebar order.
enum SettingsSection: String, CaseIterable, Identifiable, Hashable, Sendable {
    case general
    case model
    case performance
    case memory
    case cache
    case clients
    case cooling
    case advanced

    var id: String { rawValue }

    var title: String {
        switch self {
        case .general: return tr("General")
        case .model: return tr("Model")
        case .performance: return tr("Performance")
        case .memory: return tr("Memory")
        case .cache: return tr("Cache")
        case .clients: return tr("Clients")
        case .cooling: return tr("Cooling")
        case .advanced: return tr("Advanced")
        }
    }

    var subtitle: String {
        switch self {
        case .general: return tr("How MTPLX looks and starts.")
        case .model: return tr("Which model MTPLX serves, where models are kept, and how it generates.")
        case .performance: return tr("How MTPLX handles several requests at once. Changes apply after a restart.")
        case .memory: return tr("How much of this Mac the engine may plan with. Restart required.")
        case .cache: return tr("Saved prompts, so the next turn starts without reading everything again.")
        case .clients: return tr("How Pi, Hermes, OpenCode and other apps reach MTPLX.")
        case .cooling: return tr("Fan behavior while MTPLX generates.")
        case .advanced: return tr("Rarely needed. Change these with care.")
        }
    }

    var systemImage: String {
        switch self {
        case .general: return "gearshape"
        case .model: return "shippingbox"
        case .performance: return "speedometer"
        case .memory: return "memorychip"
        case .cache: return "internaldrive"
        case .clients: return "point.3.connected.trianglepath.dotted"
        case .cooling: return "fan"
        case .advanced: return "wrench.and.screwdriver"
        }
    }

    var groups: [SettingsGroupID] {
        SettingsGroupID.allCases.filter { $0.section == self }
    }

    var items: [SettingsItem] {
        SettingsItem.allCases.filter { $0.group.section == self }
    }
}

/// The titled groups inside each section, in page order.
enum SettingsGroupID: String, CaseIterable, Hashable, Sendable {
    case appearance, startup, info
    case modelIdentity, modelFolders, downloads
    case requests, batching
    case memoryLimits, kvQuantization
    case ramCache, ssdCache, cacheReset
    case connection, openIn, agents
    case fans
    case engine, retrieval, diagnostics

    var section: SettingsSection {
        switch self {
        case .appearance, .startup, .info: return .general
        case .modelIdentity, .modelFolders, .downloads: return .model
        case .requests, .batching: return .performance
        case .memoryLimits, .kvQuantization: return .memory
        case .ramCache, .ssdCache, .cacheReset: return .cache
        case .connection, .openIn, .agents: return .clients
        case .fans: return .cooling
        case .engine, .retrieval, .diagnostics: return .advanced
        }
    }

    var title: String {
        switch self {
        case .appearance: return tr("Interface")
        case .startup: return tr("Startup")
        case .info: return tr("Info")
        case .modelIdentity: return tr("Serving")
        case .modelFolders: return tr("Model libraries")
        case .downloads: return tr("Downloads")
        case .requests: return tr("Requests")
        case .batching: return tr("Batching")
        case .memoryLimits: return tr("Limits")
        case .kvQuantization: return tr("KV cache")
        case .ramCache: return tr("Memory Cache (RAM)")
        case .ssdCache: return tr("Persistent Cache (SSD)")
        case .cacheReset: return tr("Reset")
        case .connection: return tr("Connection")
        case .openIn: return tr("Where clients open")
        case .agents: return tr("Agents")
        case .fans: return tr("Fans")
        case .engine: return tr("Engine")
        case .retrieval: return tr("Retrieval endpoints")
        case .diagnostics: return tr("Diagnostics")
        }
    }

    var items: [SettingsItem] {
        SettingsItem.allCases.filter { $0.group == self }
    }
}

/// One setting (or one read-only line) on the Settings page.
enum SettingsItem: String, CaseIterable, Hashable, Sendable {
    // General
    case appearance, language, speedRecordSound, menuBarItem
    case launchOnOpen, autoRestart
    case about
    // Model
    case model, profile, generationMode, loadMTP, contextWindow
    case modelFolders
    case downloadMirror
    // Performance
    case schedulingMode, concurrencyCap
    case decodeBatchMax, admissionWindow, mtpCohorts
    // Memory
    case memoryLimit, allowSwap
    case kvQuantization
    // Cache
    case ramPolicy, ramBlockPrefix, ramMaxEntries, ramTotalCap, ramPerSessionCap
    case ssdPolicy, ssdFolder, ssdMaxSize, ssdMinPrompt
    case clearCache
    // Clients
    case hostPort, apiKey
    case piOpensIn, openCodeOpensIn, hermesOpensIn, claudeCodeOpensIn, codexOpensIn, deepseekHarnessOpensIn
    case agentWorkspace
    case hermes
    // Cooling
    case fanMode, thermalPolling
    // Advanced
    case executablePath, streamCadence, stallWatchdog
    case embeddingModels, rerankerModels, retrievalIdle, retrievalResident
    case logs, settingsFile

    var group: SettingsGroupID {
        switch self {
        case .appearance, .language, .speedRecordSound, .menuBarItem: return .appearance
        case .launchOnOpen, .autoRestart: return .startup
        case .about: return .info
        case .model, .profile, .generationMode, .loadMTP, .contextWindow: return .modelIdentity
        case .modelFolders: return .modelFolders
        case .downloadMirror: return .downloads
        case .schedulingMode, .concurrencyCap: return .requests
        case .decodeBatchMax, .admissionWindow, .mtpCohorts: return .batching
        case .memoryLimit, .allowSwap: return .memoryLimits
        case .kvQuantization: return .kvQuantization
        case .ramPolicy, .ramBlockPrefix, .ramMaxEntries, .ramTotalCap, .ramPerSessionCap: return .ramCache
        case .ssdPolicy, .ssdFolder, .ssdMaxSize, .ssdMinPrompt: return .ssdCache
        case .clearCache: return .cacheReset
        case .hostPort, .apiKey: return .connection
        case .piOpensIn, .openCodeOpensIn, .hermesOpensIn, .claudeCodeOpensIn, .codexOpensIn,
             .deepseekHarnessOpensIn:
            return .openIn
        case .agentWorkspace: return .agents
        case .hermes: return .agents
        case .fanMode, .thermalPolling: return .fans
        case .executablePath, .streamCadence, .stallWatchdog: return .engine
        case .embeddingModels, .rerankerModels, .retrievalIdle, .retrievalResident: return .retrieval
        case .logs, .settingsFile: return .diagnostics
        }
    }

    var section: SettingsSection { group.section }

    var label: String {
        switch self {
        case .appearance: return tr("Appearance")
        case .language: return tr("Language")
        case .speedRecordSound: return tr("Sound on new speed record")
        case .menuBarItem: return tr("Speed in the menu bar")
        case .launchOnOpen: return tr("Start MTPLX when the app opens")
        case .autoRestart: return tr("Restart this session's MTPLX after a crash")
        case .about: return tr("About MTPLX")
        case .model: return tr("Model")
        case .profile: return tr("Profile")
        case .generationMode: return tr("Generation mode")
        case .loadMTP: return tr("Load MTP head")
        case .contextWindow: return tr("Context window")
        case .modelFolders: return tr("Folders")
        case .downloadMirror: return tr("HF download mirror")
        case .schedulingMode: return tr("Mode")
        case .concurrencyCap: return tr("Requests at once")
        case .decodeBatchMax: return tr("Answers per step")
        case .admissionWindow: return tr("Wait to group requests")
        case .mtpCohorts: return tr("Faster parallel requests (experimental)")
        case .memoryLimit: return tr("Memory limit")
        case .allowSwap: return tr("Allow swap")
        case .kvQuantization: return tr("KV quantization")
        case .ramPolicy: return tr("Allocation policy")
        case .ramBlockPrefix: return tr("Block-prefix restore")
        case .ramMaxEntries: return tr("Max entries")
        case .ramTotalCap: return tr("Total RAM cap")
        case .ramPerSessionCap: return tr("Per-session cap")
        case .ssdPolicy: return tr("Policy")
        case .ssdFolder: return tr("Folder")
        case .ssdMaxSize: return tr("Max size")
        case .ssdMinPrompt: return tr("Save prompts ≥")
        case .clearCache: return tr("Clear prompt cache")
        case .hostPort: return tr("Host / Port")
        case .apiKey: return tr("API key (optional)")
        case .piOpensIn: return tr("Pi")
        case .openCodeOpensIn: return tr("OpenCode")
        case .hermesOpensIn: return tr("Hermes")
        case .claudeCodeOpensIn: return tr("Claude Code")
        case .codexOpensIn: return tr("Codex")
        case .deepseekHarnessOpensIn: return tr("DeepSeek Harness")
        case .agentWorkspace: return tr("Agent workspace")
        case .hermes: return "Hermes"
        case .fanMode: return tr("Fan Mode")
        case .thermalPolling: return tr("Enable thermal polling")
        case .executablePath: return tr("Executable path")
        case .streamCadence: return tr("Stream cadence")
        case .stallWatchdog: return tr("Stall watchdog")
        case .embeddingModels: return tr("Embedding models")
        case .rerankerModels: return tr("Reranker models")
        case .retrievalIdle: return tr("Release when idle")
        case .retrievalResident: return tr("Models kept resident")
        case .logs: return tr("Logs")
        case .settingsFile: return tr("Settings file")
        }
    }

    /// The row's one line of help. A few rows replace it with text that
    /// depends on the Mac or the model (`SettingsRow.caption`).
    var caption: String? {
        switch self {
        case .appearance: return tr("Jet black, warm cream, or follow macOS.")
        case .language: return tr("The language MTPLX uses across the app. Changes apply immediately.")
        case .speedRecordSound: return tr("Plays a soft chime when your Mac hits a new top speed.")
        case .menuBarItem: return tr("Shows whether MTPLX is running and, while it answers, how fast. Click it for quick controls.")
        case .launchOnOpen: return tr("Otherwise you start MTPLX from the toolbar manually.")
        case .autoRestart: return tr("Applies to daemons launched after enabling this setting. Retries up to three times with backoff; Stop cancels pending retries.")
        case .about: return tr("Version, build and licenses.")
        case .model: return tr("Path or Hugging Face id of the model MTPLX serves. The model menu in the top bar sets it for you.")
        case .profile: return tr("Auto picks the recommended profile for the selected model — Turbo for the 27B and Flash-Next models.")
        case .generationMode: return tr("MTP drafts several tokens per step. Baseline decodes one token at a time.")
        case .loadMTP: return tr("Disable to fall back to baseline (no speculation).")
        case .contextWindow: return tr("Empty serves the largest window this Mac fits.")
        case .modelFolders: return tr("Downloads and Forge write to the primary folder. Additional folders are searched in order.")
        case .downloadMirror: return tr("Hugging Face endpoint for model downloads. Your HF token is never sent to a mirror.")
        case .schedulingMode: return tr("Auto picks the best mode for what you're using. Pick a mode below to use it everywhere. Saved as soon as you pick it. Benchmark runs keep their own single-stream setup.")
        case .concurrencyCap: return tr("How many requests run together; the rest wait their turn. Used when MTPLX runs requests side by side: the modes “Handle multiple at once” and “Long agent tasks”, and Auto for a custom client. Flash-Next answers one request at a time whatever this says.")
        case .decodeBatchMax: return tr("How many answers share each GPU step when several requests run together. More serves more clients in total, each a little slower. Same modes as above; not used by Flash-Next.")
        case .admissionWindow: return tr("How long a new request waits for others so they can start together. 0 starts at once; any wait adds to that request's first token. Same modes as above; not used by Flash-Next.")
        case .mtpCohorts: return tr("A switch for a feature still in testing. In this version it changes nothing on any model; leave it off.")
        case .memoryLimit: return tr("Empty uses the engine's own plan, about three quarters of this Mac's memory. Raise it to serve longer contexts on a Mac with headroom.")
        case .allowSwap: return tr("Serve contexts larger than what fits in memory. macOS pages to SSD, so speed drops sharply, but long sessions stop being refused.")
        case .kvQuantization: return nil
        case .ramPolicy: return tr("Target default budgets half the RAM left after the model loads; bounded uses the limits below.")
        case .ramBlockPrefix: return tr("Restore a shared prompt prefix and prefill only the changed suffix.")
        case .ramMaxEntries: return tr("Number of warm prefixes allowed to stay in RAM.")
        case .ramTotalCap: return tr("Auto budgets half the RAM left after the model loads. Old prompts are evicted past the cap.")
        case .ramPerSessionCap: return tr("Maximum RAM cache held by one conversation. Auto keeps it at 2/3 of the total cap.")
        case .ssdPolicy: return nil
        case .ssdFolder: return tr("Defaults to ~/.mtplx/session-bank when blank.")
        case .ssdMaxSize: return tr("Auto scales with your Mac's RAM tier (16 GB to 100 GB). Old entries are evicted to stay under the cap.")
        case .ssdMinPrompt: return tr("Shorter prompts aren't worth the write churn.")
        case .clearCache: return tr("Clears every saved prompt cache. Anything mid-flight keeps running.")
        case .hostPort: return tr("Where clients reach MTPLX on this Mac.")
        case .apiKey: return tr("When set, clients must send this key.")
        case .piOpensIn: return tr("Pi runs in the terminal you pick.")
        case .openCodeOpensIn: return tr("OpenCode Desktop, or OpenCode in the terminal you pick.")
        case .hermesOpensIn: return tr("Hermes Desktop, or Hermes in the terminal you pick.")
        case .claudeCodeOpensIn: return tr("Claude Code runs in the terminal you pick.")
        case .codexOpensIn: return tr("MTPLX's own copy of the Codex app, with its own history, or Codex in the terminal you pick.")
        case .deepseekHarnessOpensIn: return tr("DeepSeek Harness in your browser, or started from the terminal you pick so you can watch its log. Telemetry is off either way.")
        case .agentWorkspace: return tr("Pi and Hermes terminal tools start in this folder.")
        case .hermes: return tr("Agent handoff, tools, and gateway state.")
        case .fanMode: return tr("Default uses Apple's curve. Smart boosts during requests. Max pins verified fans.")
        case .thermalPolling: return tr("Required to verify fan ramp before benchmarks.")
        case .executablePath: return tr("Defaults to mtplx in PATH if blank.")
        case .streamCadence: return nil
        case .stallWatchdog: return tr("Seconds a response may wait on a model that is making no progress before it is cancelled. 0 turns the watchdog off; long prompts and deep reasoning keep reporting progress and are never cut short by this.")
        case .embeddingModels: return tr("One per line — a Hugging Face id or local path, optionally REF=served-id. Serves /v1/embeddings.")
        case .rerankerModels: return tr("Serves /v1/rerank. Listing the same reference here and above loads one copy of the weights for both roles.")
        case .retrievalIdle: return tr("Minutes of inactivity before retrieval weights are unloaded. 0 keeps them resident; they reload on the next request.")
        case .retrievalResident: return tr("Retrieval models load on first request. Beyond this count the least recently used one is unloaded.")
        case .logs: return tr("Engine and app logs for this session.")
        case .settingsFile: return tr("Where these settings are saved.")
        }
    }

    /// Words a user may type that are not in the label or caption: units,
    /// abbreviations, the names other apps use for the same thing.
    var keywords: [String] {
        switch self {
        case .appearance: return ["dark", "light", "theme"]
        case .language: return ["locale"]
        case .speedRecordSound: return ["chime", "audio"]
        case .menuBarItem: return ["menu bar", "status", "tok/s", "tray", "menubar"]
        case .launchOnOpen: return ["auto", "launch", "open"]
        case .autoRestart: return ["crash", "recover"]
        case .about: return ["version"]
        case .model: return ["path", "huggingface", "hf"]
        case .profile: return ["turbo", "sustained"]
        case .generationMode: return ["mtp", "ar", "speculative"]
        case .loadMTP: return ["mtp", "speculation", "draft"]
        case .contextWindow: return ["context", "tokens", "length"]
        case .modelFolders: return ["folder", "ssd", "drive", "external", "directory", "library"]
        case .downloadMirror: return ["hf", "huggingface", "endpoint", "mirror"]
        case .schedulingMode: return ["scheduler", "latency", "throughput", "agent"]
        case .concurrencyCap: return ["parallel", "concurrency", "requests", "cap", "max active"]
        case .decodeBatchMax: return ["batch", "decode", "batch max"]
        case .admissionWindow: return ["batch", "wait", "ms", "admission", "window"]
        case .mtpCohorts: return ["mtp", "batch", "experimental", "cohorts"]
        case .memoryLimit: return ["ram", "gb", "limit"]
        case .allowSwap: return ["swap", "ssd", "pager"]
        case .kvQuantization: return ["kv", "q8", "q4", "int8"]
        case .ramPolicy, .ramBlockPrefix, .ramMaxEntries, .ramTotalCap, .ramPerSessionCap:
            return ["ram", "sessionbank", "cache", "prefix"]
        case .ssdPolicy, .ssdFolder, .ssdMaxSize, .ssdMinPrompt:
            return ["ssd", "disk", "session-bank", "cache", "persistent"]
        case .clearCache: return ["clear", "delete", "reset", "cache"]
        case .hostPort: return ["port", "host", "lan", "localhost", "127.0.0.1", "url"]
        case .apiKey: return ["key", "token", "auth", "password"]
        case .piOpensIn, .openCodeOpensIn, .hermesOpensIn, .claudeCodeOpensIn, .codexOpensIn:
            return ["terminal", "open in", "ghostty", "iterm", "warp", "desktop", "app"]
        case .deepseekHarnessOpensIn:
            return ["terminal", "open in", "browser", "web", "dsh", "deepseek", "telemetry"]
        case .agentWorkspace: return ["pi", "hermes", "folder", "project"]
        case .hermes: return ["gateway", "agent", "tools"]
        case .fanMode: return ["fan", "max", "smart", "thermal"]
        case .thermalPolling: return ["fan", "rpm", "thermal", "monitor"]
        case .executablePath: return ["cli", "binary", "path"]
        case .streamCadence: return ["refresh", "interval", "ms", "dashboard"]
        case .stallWatchdog: return ["timeout", "stall", "cancel"]
        case .embeddingModels: return ["embeddings", "rag"]
        case .rerankerModels: return ["rerank", "rag"]
        case .retrievalIdle, .retrievalResident: return ["retrieval", "embeddings", "rerank"]
        case .logs: return ["log", "debug"]
        case .settingsFile: return ["json", "file", "path"]
        }
    }

    /// Rows that appear only once this item is set a certain way: the RAM
    /// and SSD cache limits show under a non-default policy. Search shows
    /// the policy row for a match in any of them, so a search for "ssd size"
    /// lands on the control that reveals the row, not on an empty group.
    var reveals: [SettingsItem] {
        switch self {
        case .ramPolicy: return [.ramBlockPrefix, .ramMaxEntries, .ramTotalCap, .ramPerSessionCap]
        case .ssdPolicy: return [.ssdFolder, .ssdMaxSize, .ssdMinPrompt]
        default: return []
        }
    }

    /// Everything search reads for this item, in the current language.
    var searchText: [String] {
        [label, caption ?? "", group.title, section.title] + keywords
    }
}

// MARK: - Search

/// The query typed into the Settings search field. A row shows when every
/// word of the query appears somewhere in its label, help, group, section
/// or keywords.
struct SettingsSearch: Equatable, Sendable {
    var query: String = ""

    private var words: [String] {
        query
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
            .filter { !$0.isEmpty }
    }

    var isActive: Bool { !words.isEmpty }

    func matches(_ item: SettingsItem) -> Bool {
        let words = words
        guard !words.isEmpty else { return true }
        return ([item] + item.reveals).contains { candidate in
            let haystack = candidate.searchText
            return words.allSatisfy { word in
                haystack.contains { $0.localizedStandardContains(word) }
            }
        }
    }

    func matches(_ group: SettingsGroupID) -> Bool {
        !isActive || group.items.contains(where: matches)
    }

    /// Whether the sidebar lists this section: every section without a
    /// query, only the sections with a matching row during a search.
    func hasMatches(_ section: SettingsSection) -> Bool {
        !isActive || section.items.contains(where: matches)
    }
}

private struct SettingsSearchKey: EnvironmentKey {
    static let defaultValue = SettingsSearch()
}

extension EnvironmentValues {
    var settingsSearch: SettingsSearch {
        get { self[SettingsSearchKey.self] }
        set { self[SettingsSearchKey.self] = newValue }
    }
}
