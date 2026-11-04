import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { Card } from "./Card";
import { useDashboardStore, useFilteredHistory, useFilteredLiveHistory } from "../state/store";
import { fmtTokS } from "../lib/utils";
import { autoAxisSize } from "../lib/uplotAxis";

export function TPSTimeSeries() {
  const history = useFilteredHistory();
  const liveHistory = useFilteredLiveHistory();
  const rolling = useDashboardStore((s) => s.rolling);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  const { data, maxPoint, minPoint } = useMemo(() => {
    // Live samples (every ~0.75 s while decoding) draw the line; completed
    // requests are dots on top. With long agent requests the 5-minute window
    // often holds a single completed request, which alone draws nothing.
    const byTime = new Map<number, [number | null, number | null]>();
    for (const point of liveHistory) byTime.set(point.t, [point.tok_s, null]);
    for (const point of history) {
      const row = byTime.get(point.t) ?? [null, null];
      row[1] = point.tok_s;
      byTime.set(point.t, row);
    }
    const xs = Array.from(byTime.keys()).sort((a, b) => a - b);
    const live = xs.map((t) => byTime.get(t)![0]);
    const done = xs.map((t) => byTime.get(t)![1]);
    let maxIdx = -1;
    let minIdx = -1;
    for (let i = 0; i < history.length; i += 1) {
      const point = history[i];
      if (maxIdx === -1 || point.tok_s > history[maxIdx].tok_s) maxIdx = i;
      if (minIdx === -1 || point.tok_s < history[minIdx].tok_s) minIdx = i;
    }
    return {
      data: [xs, live, done] as uPlot.AlignedData,
      maxPoint: maxIdx >= 0 ? history[maxIdx] : null,
      minPoint: minIdx >= 0 ? history[minIdx] : null,
    };
  }, [history, liveHistory]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const width = wrapper.clientWidth;
    const opts: uPlot.Options = {
      width,
      height: 220,
      padding: [8, 16, 8, 8],
      cursor: {
        drag: { x: false, y: false, setScale: false },
        focus: { prox: 24 },
        sync: { key: "tps", scales: ["x", null] },
      },
      scales: {
        x: {
          time: true,
          // Always the last 5 minutes up to now, also with zero or one sample.
          range: () => {
            const now = Date.now() / 1000;
            return [now - (useDashboardStore.getState().rolling?.window_s ?? 300), now];
          },
        },
        y: {
          range: (_self, dataMin, dataMax) =>
            dataMin == null || dataMax == null
              ? [0, 1]
              : [Math.max(0, dataMin * 0.9), dataMax * 1.05],
        },
      },
      axes: [
        {
          stroke: "rgba(200,210,220,0.55)",
          grid: { show: true, stroke: "rgba(255,255,255,0.04)", width: 1 },
          // One label per whole minute (HH:MM) instead of uPlot's mixed seconds/am-pm ticks.
          incrs: [60, 120, 300],
          values: (_self, ticks) =>
            ticks.map((t) =>
              new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }),
            ),
        },
        {
          stroke: "rgba(200,210,220,0.55)",
          grid: { show: true, stroke: "rgba(255,255,255,0.04)", width: 1 },
          values: (_self, ticks) => ticks.map((t) => `${t.toFixed(0)} tok/s`),
          size: autoAxisSize,
        },
      ],
      legend: { show: false },
      series: [
        {},
        {
          label: "decode tok/s",
          stroke: "rgba(0,214,143,0.9)",
          width: 2,
          points: { show: false },
          paths: uPlot.paths.spline?.(),
          fill: "rgba(0,214,143,0.10)",
          spanGaps: true,
        },
        {
          label: "completed request",
          stroke: "rgba(79,182,243,0.95)",
          paths: () => null,
          points: { show: true, size: 7, fill: "rgba(79,182,243,0.95)" },
        },
      ],
    };
    const plot = new uPlot(opts, data, wrapper);
    plotRef.current = plot;
    const handleResize = () => {
      plot.setSize({ width: wrapper.clientWidth, height: 220 });
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      plot.destroy();
      plotRef.current = null;
    };
  }, []);

  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    plot.setData(data);
  }, [data]);

  const sessionFilter = useDashboardStore((s) => s.sessionFilter);

  return (
    <Card
      title="Decode TPS (last 5 min)"
      subtitle={
        rolling
          ? `${rolling.count} completed · p50 ${fmtTokS(rolling.p50)} · p95 ${fmtTokS(rolling.p95)}${
              sessionFilter ? ` · filtered by ${sessionFilter}` : ""
            }`
          : "no completed requests yet"
      }
    >
      <div ref={wrapperRef} className="w-full" />
      {(maxPoint || minPoint) && (
        <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
          <div className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 flex items-center justify-between">
            <span className="text-[var(--text-muted)]">window max</span>
            <span className="text-[var(--accent-warm)] font-semibold tabular-nums">
              {fmtTokS(maxPoint?.tok_s ?? null)} tok/s
            </span>
          </div>
          <div className="rounded-md border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 flex items-center justify-between">
            <span className="text-[var(--text-muted)]">window min</span>
            <span className="text-[var(--accent-cool)] font-semibold tabular-nums">
              {fmtTokS(minPoint?.tok_s ?? null)} tok/s
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}
