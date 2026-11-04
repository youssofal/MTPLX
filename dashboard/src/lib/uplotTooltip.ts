import type uPlot from "uplot";

/** Max distance in CSS pixels between the cursor and a point that still counts as hovered. */
const SNAP_PX = 40;

/**
 * Nearest non-null value of `seriesIdx` around data index `idx`, within SNAP_PX
 * of the cursor. Series with gaps (or sparse dots) are often null at the exact
 * index the cursor lands on, so walk outwards instead of reading data[idx].
 */
function nearestValue(
  u: uPlot,
  seriesIdx: number,
  idx: number,
  cursorLeft: number,
): number | null {
  const xs = u.data[0];
  const ys = u.data[seriesIdx];
  for (let step = 0; step < xs.length; step += 1) {
    let tooFar = true;
    for (const i of step === 0 ? [idx] : [idx - step, idx + step]) {
      if (i < 0 || i >= xs.length) continue;
      const distance = Math.abs(u.valToPos(xs[i], "x") - cursorLeft);
      if (distance > SNAP_PX) continue;
      tooFar = false;
      const value = ys[i];
      if (value !== null && value !== undefined) return value;
    }
    if (tooFar && step > 0) return null;
  }
  return null;
}

/**
 * uPlot plugin: a small label next to the crosshair with the time and the value
 * of every visible series at that point. Used where the uPlot legend is hidden.
 */
export function cursorValueTooltip(formatValue: (value: number) => string): uPlot.Plugin {
  let box: HTMLDivElement | null = null;

  return {
    hooks: {
      init: (u) => {
        box = document.createElement("div");
        Object.assign(box.style, {
          position: "absolute",
          display: "none",
          pointerEvents: "none",
          zIndex: "10",
          padding: "4px 8px",
          borderRadius: "6px",
          border: "1px solid var(--border-soft)",
          background: "var(--bg-elevated)",
          color: "var(--text-primary)",
          fontSize: "12px",
          lineHeight: "1.4",
          whiteSpace: "nowrap",
          fontVariantNumeric: "tabular-nums",
        });
        u.over.appendChild(box);
      },
      setCursor: (u) => {
        if (!box) return;
        const { left, top, idx } = u.cursor;
        if (idx === null || idx === undefined || left === undefined || left < 0 || top === undefined) {
          box.style.display = "none";
          return;
        }
        const rows: string[] = [];
        for (let s = 1; s < u.series.length; s += 1) {
          const series = u.series[s];
          if (!series.show) continue;
          const value = nearestValue(u, s, idx, left);
          if (value === null) continue;
          // uPlot turns a stroke string into a function after init; both give the line colour.
          const stroke = typeof series.stroke === "function" ? series.stroke(u, s) : series.stroke;
          const color = typeof stroke === "string" ? stroke : "currentColor";
          rows.push(
            `<div><span style="color:${color}">●</span> ${series.label ?? ""}: <b>${formatValue(value)}</b></div>`,
          );
        }
        if (rows.length === 0) {
          box.style.display = "none";
          return;
        }
        const time = new Date(u.posToVal(left, "x") * 1000).toLocaleTimeString([], { hour12: false });
        box.innerHTML = `<div style="color:var(--text-muted)">${time}</div>${rows.join("")}`;
        box.style.display = "block";
        // Keep the label inside the plot: flip to the left of the cursor near the right edge.
        const flip = left + 16 + box.offsetWidth > u.over.clientWidth;
        box.style.left = `${flip ? left - 12 - box.offsetWidth : left + 12}px`;
        box.style.top = `${Math.max(0, Math.min(top - box.offsetHeight - 8, u.over.clientHeight - box.offsetHeight))}px`;
      },
    },
  };
}
