import { afterAll, describe, expect, mock, test } from "bun:test";
import * as React from "react";
import type uPlot from "uplot";
import { autoAxisSize } from "../src/lib/uplotAxis";

const charts: uPlot.Options[] = [];
const previousWindow = globalThis.window;
const previousRatio = globalThis.devicePixelRatio;
Object.assign(globalThis, {
  devicePixelRatio: 2,
  window: { addEventListener() {}, removeEventListener() {} },
});
afterAll(() => {
  Object.assign(globalThis, { window: previousWindow, devicePixelRatio: previousRatio });
  mock.restore();
});

// Run each component's chart setup, retaining its actual axis options.
mock.module("react", () => ({
  ...React,
  useEffect: (effect: () => unknown) => effect(),
  useMemo: (compute: () => unknown) => compute(),
  useRef: () => ({ current: { clientWidth: 400 } }),
}));
mock.module("../src/components/Card", () => ({ Card: () => null }));
mock.module("../src/state/store", () => ({
  useFilteredHistory: () => [],
  useFilteredLiveHistory: () => [],
  useDashboardStore: (select: (state: object) => unknown) =>
    select({ rolling: null, sessionFilter: null }),
}));
mock.module("../src/hooks/usePolling", () => ({
  usePrefillHistory: () => ({ data: { history: [] } }),
}));
mock.module("uplot", () => ({
  default: class {
    static paths = { spline: () => undefined };
    constructor(options: uPlot.Options) { charts.push(options); }
    setData() {}
    setSize() {}
    destroy() {}
  },
}));

const { TPSTimeSeries } = await import("../src/components/TPSTimeSeries");
const { PrefillTPSSparkline } = await import("../src/components/PrefillTPSSparkline");

function plot(widths: Record<string, number>) {
  return {
    axes: [{ ticks: { size: 10 }, gap: 5, font: ["24px Arial", 24, 12], _size: 91 }],
    ctx: {
      font: "16px serif",
      measureText: (value: string) => ({ width: widths[value] ?? 0 }),
    },
  } as unknown as uPlot;
}

describe("value-axis labels", () => {
  test.each([
    ["decode", TPSTimeSeries, "52 tok/s"],
    ["prefill", PrefillTPSSparkline, "125000"],
  ] as const)("%s chart reserves space for the complete label", (_name, component, label) => {
    component();
    const axis = charts.at(-1)!.axes![1];
    const size = axis.size ?? 50; // uPlot's shipped default.
    const self = plot({ [label]: 108 });
    const actual = typeof size === "function" ? size(self, [label], 0, 1) : size;
    expect(actual).toBeGreaterThanOrEqual(10 + 5 + 108 / 2);
  });

  test("measures pixel width, not character count", () => {
    const self = plot({ "1111": 24, "888": 36 });
    expect(autoAxisSize(self, ["1111", "888"], 0, 1)).toBe(33);
  });

  test("restores the drawing font after measuring", () => {
    const self = plot({ "1111": 24 });
    expect(autoAxisSize(self, ["1111"], 0, 1)).toBe(27);
    expect(self.ctx.font).toBe("16px serif");
  });

  test("handles the initial empty sizing pass and converges without remeasuring", () => {
    const self = plot({ "100": 40 });
    expect(autoAxisSize(self, null, 0, 0)).toBe(15);
    expect(autoAxisSize(self, ["100"], 0, 2)).toBe(91);
  });
});
