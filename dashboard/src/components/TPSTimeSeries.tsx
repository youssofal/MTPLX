import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { Card } from "./Card";
import { useDashboardStore, useFilteredHistory } from "../state/store";
import { fmtTokS } from "../lib/utils";
import { autoAxisSize } from "../lib/uplotAxis";
import { cursorValueTooltip } from "../lib/uplotTooltip";

export function TPSTimeSeries() {
  const history = useFilteredHistory();
  const rolling = useDashboardStore((s) => s.rolling);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  const { data, maxPoint, minPoint } = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    let maxIdx = -1;
    let minIdx = -1;
    for (let i = 0; i < history.length; i += 1) {
      const point = history[i];
      xs.push(point.t);
      ys.push(point.tok_s);
      if (maxIdx === -1 || point.tok_s > history[maxIdx].tok_s) maxIdx = i;
      if (minIdx === -1 || point.tok_s < history[minIdx].tok_s) minIdx = i;
    }
    return {
      data: [xs, ys] as uPlot.AlignedData,
      maxPoint: maxIdx >= 0 ? history[maxIdx] : null,
      minPoint: minIdx >= 0 ? history[minIdx] : null,
    };
  }, [history]);

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
        x: { time: true },
        y: { range: (_self, dataMin, dataMax) => [Math.max(0, dataMin * 0.9), dataMax * 1.05] },
      },
      axes: [
        {
          stroke: "rgba(200,210,220,0.55)",
          grid: { show: true, stroke: "rgba(255,255,255,0.04)", width: 1 },
        },
        {
          stroke: "rgba(200,210,220,0.55)",
          grid: { show: true, stroke: "rgba(255,255,255,0.04)", width: 1 },
          values: (_self, ticks) => ticks.map((t) => `${t.toFixed(0)} tok/s`),
          size: autoAxisSize,
        },
      ],
      legend: { show: false },
      plugins: [cursorValueTooltip((value) => `${fmtTokS(value)} tok/s`)],
      series: [
        {},
        {
          label: "decode tok/s",
          stroke: "rgba(0,214,143,0.9)",
          width: 2,
          points: { show: false },
          paths: uPlot.paths.spline?.(),
          fill: "rgba(0,214,143,0.10)",
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
          ? `${rolling.count} samples · p50 ${fmtTokS(rolling.p50)} · p95 ${fmtTokS(rolling.p95)}${
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
