import AppKit
import Combine
import Foundation
import QuartzCore
import os

// MARK: - UIStreamPerfProbe
//
// Frontend streaming-performance instrumentation. Answers, with numbers
// instead of feel, the three questions the 2026-07-31 perf hunt had to
// reconstruct from screen recordings and `sample`:
//
//   1. When did each SSE delta reach the view model, and when was it
//      APPLIED to the streaming document? (network vs UI attribution)
//   2. How often was the main thread too busy to run at all, and for
//      how long? (the stall census behind "freeze then vomit")
//   3. What did one turn cost end to end? (`ui_turn_render_summary` —
//      joinable against the engine's /metrics + persisted ZSTATSJSON
//      via request id.)
//
// Enablement: `MTPLX_UI_PERF=1` (or any AIME diagnostics run —
// `MTPLX_AIME_DIAGNOSTICS=1` — since the JSONL rides the same writer).
// `MTPLX_UI_PERF_HUD=1` additionally shows the live overlay chip.
// Fully inert when disabled: every hook early-returns on a stored Bool.
//
// The stall monitor measures MAIN-THREAD SCHEDULING GAPS (a 12 ms
// main-actor heartbeat and how late it runs), not display vsync. That
// is deliberate: a starved main thread is the shared cause of both
// coalesced streaming flushes and scroll jank, and the heartbeat works
// identically in CI/headless runs where there is no display.
@MainActor
public final class UIStreamPerfProbe: ObservableObject {

    // MARK: Enablement

    public static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        if flag(environment["MTPLX_UI_PERF"]) { return true }
        return AIMEDiagnostics.isEnabled(environment: environment)
    }

    public static func hudEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        flag(environment["MTPLX_UI_PERF_HUD"])
    }

    private static func flag(_ raw: String?) -> Bool {
        switch raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on": return true
        default: return false
        }
    }

    // MARK: HUD surface

    public struct HUDSnapshot: Equatable, Sendable {
        public var isStreaming = false
        public var flushesPerSecond = 0.0
        public var stallsLastSecond = 0
        public var worstStallMsLastSecond = 0.0
        public var lastAppendMs = 0.0
        public var documentBlocks = 0
        public var turnChars = 0
    }

    @Published public private(set) var hud = HUDSnapshot()

    public let enabled: Bool
    public let showsHUD: Bool

    /// Last-created enabled probe. The render-layer hooks below are called
    /// from NSView draw/apply paths that hold no reference to the chat view
    /// model; a weak static bridge wires them without new plumbing. One chat
    /// view model exists during perf runs, so last-wins is fine.
    public private(set) static weak var shared: UIStreamPerfProbe?

    // MARK: Turn ledger

    private struct FlushRecord {
        var t: Double            // uptime seconds
        var gapMs: Double        // since previous flush apply
        var drainedBytes: Int
        var applyMs: Double      // document append duration (both docs)
        var blocksAfter: Int
        var linesFinalized: Int  // lines finalized BY this flush
        var merges: Int          // segment merges performed by this flush
    }

    private struct StallRecord {
        var t: Double
        var ms: Double
        var streaming: Bool
    }

    private var turnActive = false
    private var turnStartedAt: Double = 0
    private var turnChars = 0
    private var chunkCount = 0
    private var chunkBytes = 0
    private var firstChunkAt: Double?
    private var lastChunkAt: Double = 0
    private var interChunkGaps: [Double] = []
    private var flushes: [FlushRecord] = []
    private var stalls: [StallRecord] = []
    private var scrollTicks = 0
    private var scrollPins = 0
    private var lastFlushAt: Double?
    private var lastLinesTotal = 0
    private var lastMergesTotal = 0

    // MARK: Stall monitor

    private var monitorTask: Task<Void, Never>?
    private static let heartbeat: Duration = .milliseconds(12)
    private static let heartbeatS = 0.012
    private static let stallThresholdMs = 50.0

    public init(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.enabled = Self.isEnabled(environment: environment)
        self.showsHUD = enabled && Self.hudEnabled(environment: environment)
        if enabled {
            Self.shared = self
            startStallMonitor()
            startPaintWatchdog()
        }
    }

    // No deinit: the monitor task holds `self` weakly and terminates on
    // its next tick once the probe deallocates.

    private func startStallMonitor() {
        monitorTask?.cancel()
        monitorTask = Task { @MainActor [weak self] in
            var expected = ProcessInfo.processInfo.systemUptime
            var lastHUDUpdate = expected
            var recentStalls: [(t: Double, ms: Double)] = []
            var recentFlushCount: [(t: Double, count: Int)] = []
            while !Task.isCancelled {
                try? await Task.sleep(for: Self.heartbeat)
                guard let self else { return }
                let now = ProcessInfo.processInfo.systemUptime
                let lateMs = (now - expected - Self.heartbeatS) * 1000
                expected = now
                if lateMs >= Self.stallThresholdMs {
                    self.recordStall(ms: lateMs, at: now)
                    recentStalls.append((now, lateMs))
                }
                // HUD refresh at 2 Hz from a rolling 1 s window.
                if self.showsHUD, now - lastHUDUpdate >= 0.5 {
                    lastHUDUpdate = now
                    recentStalls.removeAll { now - $0.t > 1.0 }
                    recentFlushCount.append((now, self.flushes.count))
                    recentFlushCount.removeAll { now - $0.t > 1.2 }
                    let flushDelta: Double
                    if let oldest = recentFlushCount.first, recentFlushCount.count > 1 {
                        let dt = now - oldest.t
                        flushDelta = dt > 0 ? Double(self.flushes.count - oldest.count) / dt : 0
                    } else {
                        flushDelta = 0
                    }
                    self.hud = HUDSnapshot(
                        isStreaming: self.turnActive,
                        flushesPerSecond: flushDelta,
                        stallsLastSecond: recentStalls.count,
                        worstStallMsLastSecond: recentStalls.map(\.ms).max() ?? 0,
                        lastAppendMs: self.flushes.last?.applyMs ?? 0,
                        documentBlocks: self.hud.documentBlocks,
                        turnChars: self.turnChars
                    )
                }
            }
        }
    }

    private var lastIdleStallEmit: Double = 0
    private var lastScrollEmit: Double = 0

    private func recordStall(ms: Double, at t: Double) {
        stalls.append(StallRecord(t: t, ms: ms, streaming: turnActive))
        // Stall events are the rare, high-signal record: emit each one
        // while a turn is live; cadence-limit them when idle.
        if !turnActive {
            guard t - lastIdleStallEmit >= 5 else { return }
            lastIdleStallEmit = t
        }
        AIMEDiagnostics.record(
            "ui_main_thread_stall",
            fields: [
                "stall_ms": .double((ms * 10).rounded() / 10),
                "streaming": .bool(turnActive),
                "turn_chars": .int(turnChars),
                "doc_blocks": .int(hud.documentBlocks)
            ],
            force: true
        )
    }

    // MARK: Pipeline hooks (all cheap no-ops when disabled)

    public func turnStarted() {
        guard enabled else { return }
        turnActive = true
        turnStartedAt = ProcessInfo.processInfo.systemUptime
        turnChars = 0
        chunkCount = 0
        chunkBytes = 0
        firstChunkAt = nil
        lastChunkAt = 0
        interChunkGaps = []
        flushes = []
        stalls = []
        scrollTicks = 0
        scrollPins = 0
        lastFlushAt = nil
        lastLinesTotal = 0
        lastMergesTotal = 0
        renderDurations = [:]
        renderTrace = []
        paintGaps = []
        paintGapTrace = []
        startPaintWatchdog()
        AIMEDiagnostics.record("ui_turn_started", fields: [:], force: true)
    }

    public func chunkArrived(bytes: Int) {
        guard enabled, turnActive else { return }
        let now = ProcessInfo.processInfo.systemUptime
        if firstChunkAt == nil {
            firstChunkAt = now
        } else {
            interChunkGaps.append((now - lastChunkAt) * 1000)
        }
        lastChunkAt = now
        chunkCount += 1
        chunkBytes += bytes
        turnChars += bytes
    }

    public func flushApplied(
        drainedBytes: Int,
        applyMs: Double,
        blocksAfter: Int,
        linesFinalizedTotal: Int = 0,
        mergesTotal: Int = 0
    ) {
        guard enabled, turnActive else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let gapMs = lastFlushAt.map { (now - $0) * 1000 } ?? 0
        lastFlushAt = now
        let lineDelta = max(0, linesFinalizedTotal - lastLinesTotal)
        let mergeDelta = max(0, mergesTotal - lastMergesTotal)
        lastLinesTotal = max(lastLinesTotal, linesFinalizedTotal)
        lastMergesTotal = max(lastMergesTotal, mergesTotal)
        flushes.append(FlushRecord(
            t: now,
            gapMs: gapMs,
            drainedBytes: drainedBytes,
            applyMs: applyMs,
            blocksAfter: blocksAfter,
            linesFinalized: lineDelta,
            merges: mergeDelta
        ))
        hud.documentBlocks = blocksAfter
        // The founder-reported failure shape is "freezes on NEW LINES":
        // emit a record for every flush that finalized at least one
        // line, so each line lands in the JSONL with its own apply cost
        // and the gap that preceded it. No sampling, no cadence.
        if lineDelta > 0 {
            AIMEDiagnostics.record(
                "ui_line_finalized",
                fields: [
                    "lines": .int(lineDelta),
                    "gap_ms": .double((gapMs * 10).rounded() / 10),
                    "apply_ms": .double((applyMs * 100).rounded() / 100),
                    "blocks_after": .int(blocksAfter),
                    "turn_chars": .int(turnChars)
                ],
                force: true
            )
        }
        // Segment merges restructure the block list — if merge flushes
        // spike apply cost, this record pins it directly.
        if mergeDelta > 0 {
            AIMEDiagnostics.record(
                "ui_segment_merge",
                fields: [
                    "merges": .int(mergeDelta),
                    "apply_ms": .double((applyMs * 100).rounded() / 100),
                    "blocks_after": .int(blocksAfter)
                ],
                force: true
            )
        }
        // Slow applies are the streaming-jank signal — record each one.
        if applyMs >= 8 {
            AIMEDiagnostics.record(
                "ui_flush_slow_apply",
                fields: [
                    "apply_ms": .double((applyMs * 10).rounded() / 10),
                    "drained_bytes": .int(drainedBytes),
                    "blocks_after": .int(blocksAfter),
                    "turn_chars": .int(turnChars)
                ],
                force: true
            )
        }
    }

    /// A synchronous bottom-pin ran inside the document-growth layout
    /// pass (the anti-sawtooth path). Counted per turn; the A/B gate is
    /// the video edge-tracker, this is the "did it engage" receipt.
    public func scrollPinned() {
        guard enabled else { return }
        scrollPins += 1
        os_signpost(.event, log: Self.renderSignpostLog, name: "ScrollPin")
    }

    // MARK: Render-layer probe (2026-08-19 streamwar)
    //
    // Every prior streaming regression shipped green because
    // instrumentation stopped at the document store: `apply_ms` measured
    // the string append, not the TextKit layout, glyph draw, or the
    // frames the window actually painted. These hooks close that gap.
    // Same contract as the rest of the probe: inert unless
    // MTPLX_UI_PERF=1, early-return on a stored Bool.

    public enum RenderSite: String, CaseIterable, Sendable {
        /// `StreamingAssistantMarkdownView.renderItems` — block list ->
        /// render items derivation (runs per document revision).
        case renderItems = "render_items"
        /// `StreamingCodeTextViewport.apply` — fragment diff + attributed
        /// string build + NSTextStorage mutation for the live code card.
        case applyRender = "apply_render"
        /// `LiveTailTextSurface.draw` — TextKit tail layout + glyph draw.
        case draw = "draw"
    }

    public static let renderSignpostLog = OSLog(
        subsystem: "com.mtplx.app", category: "RenderPerf"
    )

    /// Cached once for the same reason as `AIMEDiagnostics.isEnabled`:
    /// these wrappers sit on per-frame paths.
    public static let renderProbeEnabled: Bool = UIStreamPerfProbe.isEnabled()

    /// Time one render-layer site. Zero-cost passthrough when disabled;
    /// when enabled, emits an os_signpost interval (Instruments) and an
    /// in-memory sample (JSONL percentiles + slow-event records).
    /// `size` is the site's work-size proxy (blocks, fragments, or
    /// storage UTF-16 length) — the O(n) vs O(1) conviction evidence.
    @MainActor
    public static func renderTimed<T>(
        _ site: RenderSite,
        size: @autoclosure () -> Int = 0,
        _ body: () -> T
    ) -> T {
        guard renderProbeEnabled else { return body() }
        let signpostID = OSSignpostID(log: renderSignpostLog)
        let sizeValue = size()
        os_signpost(
            .begin,
            log: renderSignpostLog,
            name: "Render",
            signpostID: signpostID,
            "site=%{public}@ size=%{public}d",
            site.rawValue,
            sizeValue
        )
        let started = ProcessInfo.processInfo.systemUptime
        let result = body()
        let ms = (ProcessInfo.processInfo.systemUptime - started) * 1000
        os_signpost(
            .end,
            log: renderSignpostLog,
            name: "Render",
            signpostID: signpostID,
            "ms=%{public}.3f",
            ms
        )
        shared?.renderEvent(site, ms: ms, size: sizeValue)
        return result
    }

    private struct RenderRecord {
        var t: Double
        var site: RenderSite
        var ms: Double
        var size: Int
    }

    private var renderDurations: [RenderSite: [Double]] = [:]
    private var renderTrace: [RenderRecord] = []
    private var renderSlowLastIdleEmit: [RenderSite: Double] = [:]

    private func renderEvent(_ site: RenderSite, ms: Double, size: Int) {
        let now = ProcessInfo.processInfo.systemUptime
        if turnActive {
            renderDurations[site, default: []].append(ms)
            // Trace only the events worth plotting; the full distribution
            // lives in the turn-summary percentiles.
            if ms >= 4 {
                renderTrace.append(
                    RenderRecord(t: now, site: site, ms: ms, size: size)
                )
            }
        }
        guard ms >= 8 else { return }
        if !turnActive {
            guard now - (renderSlowLastIdleEmit[site] ?? 0) >= 5 else { return }
            renderSlowLastIdleEmit[site] = now
        }
        AIMEDiagnostics.record(
            "ui_render_slow",
            fields: [
                "site": .string(site.rawValue),
                "ms": .double((ms * 100).rounded() / 100),
                "size": .int(size),
                "streaming": .bool(turnActive)
            ],
            force: true
        )
    }

    // MARK: Paint-gap watchdog
    //
    // A CADisplayLink on the main run loop. Unlike the 12 ms heartbeat
    // above (scheduling gaps), this measures the display-frame cadence
    // the user's eye sees: a late tick means the main thread could not
    // service a vsync callback — a dropped paint. The frame-rate floor
    // keeps ProMotion from idling the link so gap math stays trivial.

    private var paintLink: CADisplayLink?
    private var paintGaps: [Double] = []
    private var paintGapTrace: [(t: Double, ms: Double)] = []
    private var lastPaintTick: Double = 0
    private var lastIdlePaintEmit: Double = 0
    private static let paintGapRecordMs = 50.0

    private func startPaintWatchdog() {
        guard paintLink == nil, let screen = NSScreen.main ?? NSScreen.screens.first
        else { return }
        // CADisplayLink retains its target; the probe already lives for
        // the app's lifetime (owned by the chat view model), so the cycle
        // is moot and the link never needs invalidation.
        let link = screen.displayLink(target: self, selector: #selector(paintTick(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30, maximum: 120, preferred: 60
        )
        link.add(to: .main, forMode: .common)
        paintLink = link
    }

    @objc private func paintTick(_ link: CADisplayLink) {
        let now = ProcessInfo.processInfo.systemUptime
        defer { lastPaintTick = now }
        guard lastPaintTick > 0 else { return }
        let gapMs = (now - lastPaintTick) * 1000
        if turnActive {
            paintGaps.append(gapMs)
        }
        guard gapMs >= Self.paintGapRecordMs else { return }
        if turnActive {
            paintGapTrace.append((t: now, ms: gapMs))
        } else {
            guard now - lastIdlePaintEmit >= 5 else { return }
            lastIdlePaintEmit = now
        }
        os_signpost(
            .event,
            log: Self.renderSignpostLog,
            name: "PaintGap",
            "ms=%{public}.1f",
            gapMs
        )
        AIMEDiagnostics.record(
            "ui_paint_gap",
            fields: [
                "gap_ms": .double((gapMs * 10).rounded() / 10),
                "streaming": .bool(turnActive),
                "turn_chars": .int(turnChars)
            ],
            force: true
        )
    }

    public func scrollTick(distanceToBottom: Double, userInitiated: Bool) {
        guard enabled else { return }
        scrollTicks += 1
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastScrollEmit >= 1 else { return }
        lastScrollEmit = now
        AIMEDiagnostics.record(
            "ui_scroll_tick",
            fields: [
                "distance_to_bottom": .double((distanceToBottom * 10).rounded() / 10),
                "user_initiated": .bool(userInitiated),
                "streaming": .bool(turnActive)
            ],
            force: true
        )
    }

    public func turnEnded(requestId: String?) {
        guard enabled, turnActive else { return }
        turnActive = false
        let now = ProcessInfo.processInfo.systemUptime
        let wallS = now - turnStartedAt
        let flushGaps = flushes.dropFirst().map(\.gapMs)
        let applies = flushes.map(\.applyMs)
        let turnStalls = stalls.filter { $0.streaming }
        var fields: [String: AIMEDiagnosticValue] = [
            "request_id": .string(requestId ?? ""),
            "wall_s": .double((wallS * 100).rounded() / 100),
            "chunks": .int(chunkCount),
            "chunk_bytes": .int(chunkBytes),
            "chunk_gap_ms_p50": .double(Self.percentile(interChunkGaps, 50)),
            "chunk_gap_ms_p95": .double(Self.percentile(interChunkGaps, 95)),
            "chunk_gap_ms_max": .double(interChunkGaps.max() ?? 0),
            "flushes": .int(flushes.count),
            "flush_gap_ms_p50": .double(Self.percentile(flushGaps, 50)),
            "flush_gap_ms_p95": .double(Self.percentile(flushGaps, 95)),
            "flush_gap_ms_max": .double(flushGaps.max() ?? 0),
            "apply_ms_p50": .double(Self.percentile(applies, 50)),
            "apply_ms_p95": .double(Self.percentile(applies, 95)),
            "apply_ms_max": .double(applies.max() ?? 0),
            "stalls_over_50ms": .int(turnStalls.count),
            "stall_ms_max": .double(turnStalls.map(\.ms).max() ?? 0),
            "stall_ms_total": .double(turnStalls.map(\.ms).reduce(0, +)),
            "scroll_ticks": .int(scrollTicks),
            "scroll_pins": .int(scrollPins),
            "lines_finalized": .int(flushes.map(\.linesFinalized).reduce(0, +)),
            "segment_merges": .int(flushes.map(\.merges).reduce(0, +)),
            "doc_blocks_final": .int(flushes.last?.blocksAfter ?? 0)
        ]
        for site in RenderSite.allCases {
            let values = renderDurations[site] ?? []
            fields["\(site.rawValue)_count"] = .int(values.count)
            fields["\(site.rawValue)_ms_p50"] = .double(Self.percentile(values, 50))
            fields["\(site.rawValue)_ms_p95"] = .double(Self.percentile(values, 95))
            fields["\(site.rawValue)_ms_max"] = .double(values.max() ?? 0)
        }
        fields["paint_ticks"] = .int(paintGaps.count)
        fields["paint_gap_ms_p95"] = .double(Self.percentile(paintGaps, 95))
        fields["paint_gap_ms_max"] = .double(paintGaps.max() ?? 0)
        fields["paint_gaps_over_50ms"] = .int(paintGaps.filter { $0 >= 50 }.count)
        fields["paint_gaps_over_100ms"] = .int(paintGaps.filter { $0 >= 100 }.count)
        AIMEDiagnostics.record(
            "ui_turn_render_summary",
            fields: fields,
            flushImmediately: true,
            force: true
        )
        dumpFlushTrace(requestId: requestId)
    }

    /// Full per-flush trace for offline cross-referencing against the
    /// engine's own per-request records. One file per turn, next to the
    /// aime JSONLs.
    private func dumpFlushTrace(requestId: String?) {
        let records = flushes
        let stallRecords = stalls
        let renderRecords = renderTrace
        let paintRecords = paintGapTrace
        guard !records.isEmpty else { return }
        let id = requestId ?? "unknown"
        // Uptime/wall anchor pair: StreamScope joins this trace against the
        // engine's visible-emit census (wall-clock) using this one line.
        let anchorUptime = ProcessInfo.processInfo.systemUptime
        let anchorWall = Date().timeIntervalSince1970
        Task.detached(priority: .utility) {
            let base = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first ?? URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent("Library/Application Support")
            let dir = base
                .appendingPathComponent("MTPLX", isDirectory: true)
                .appendingPathComponent("Diagnostics", isDirectory: true)
            try? FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true
            )
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "")
            let url = dir.appendingPathComponent("uistream-\(stamp).jsonl")
            var lines: [String] = []
            lines.reserveCapacity(
                records.count + stallRecords.count
                    + renderRecords.count + paintRecords.count + 1
            )
            lines.append(String(
                format: #"{"kind":"turn","request_id":"%@","t_uptime":%.4f,"t_wall":%.4f}"#,
                id, anchorUptime, anchorWall
            ))
            for r in records {
                lines.append(String(
                    format: #"{"kind":"flush","t":%.4f,"gap_ms":%.1f,"drained_bytes":%d,"apply_ms":%.2f,"blocks":%d,"lines":%d,"merges":%d}"#,
                    r.t, r.gapMs, r.drainedBytes, r.applyMs, r.blocksAfter,
                    r.linesFinalized, r.merges
                ))
            }
            for s in stallRecords {
                lines.append(String(
                    format: #"{"kind":"stall","t":%.4f,"ms":%.1f,"streaming":%@}"#,
                    s.t, s.ms, s.streaming ? "true" : "false"
                ))
            }
            for r in renderRecords {
                lines.append(String(
                    format: #"{"kind":"render","t":%.4f,"site":"%@","ms":%.2f,"size":%d}"#,
                    r.t, r.site.rawValue, r.ms, r.size
                ))
            }
            for p in paintRecords {
                lines.append(String(
                    format: #"{"kind":"paint_gap","t":%.4f,"ms":%.1f}"#,
                    p.t, p.ms
                ))
            }
            try? (lines.joined(separator: "\n") + "\n")
                .write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private nonisolated static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = min(
            sorted.count - 1,
            max(0, Int((Double(sorted.count) * p / 100).rounded(.down)))
        )
        return ((sorted[rank] * 10).rounded()) / 10
    }
}
