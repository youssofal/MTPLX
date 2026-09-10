import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";
import { depthTotal, fmtNumber } from "../lib/utils";

export function DepthAcceptanceBars() {
  const latest = useDashboardStore((s) => s.latest);
  const accepted = latest?.accepted_by_depth ?? [];
  const drafted = latest?.drafted_by_depth ?? [];
  const meanProb = latest?.mean_accept_probability_by_depth ?? [];

  const acceptedTotal = depthTotal(latest?.accepted_drafts, accepted);
  const draftedTotal = depthTotal(latest?.drafted_tokens, drafted);

  const maxLen = Math.max(accepted.length, drafted.length, meanProb.length);
  const rows = Array.from({ length: maxLen }, (_, i) => {
    const a = accepted[i] ?? 0;
    const d = drafted[i] ?? Math.max(a, 1);
    return {
      depth: `D${i + 1}`,
      accepted: a,
      drafted: d,
      rate: d > 0 ? (a / d) * 100 : 0,
      meanProb: meanProb[i] != null ? meanProb[i] * 100 : null,
    };
  });

  return (
    <Card
      title="Per-depth acceptance"
      subtitle={
        rows.length > 0
          ? `${fmtNumber(latest?.verify_calls)} verify calls · ${fmtNumber(acceptedTotal)} accepted of ${fmtNumber(draftedTotal)} drafted`
          : "no completed generation yet"
      }
    >
      <div className="h-[260px]">
        {rows.length === 0 ? (
          <EmptyState />
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 8, right: 24, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="depth" stroke="rgba(200,210,220,0.6)" />
              <YAxis
                yAxisId="left"
                stroke="rgba(200,210,220,0.6)"
                tickFormatter={(v) => `${v}%`}
                domain={[0, 100]}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="rgba(240,180,41,0.7)"
                tickFormatter={(v) => `${v}%`}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                }}
                labelStyle={{ color: "var(--text-muted)" }}
                formatter={(value, name) => {
                  if (typeof value === "number") {
                    return [`${value.toFixed(1)}%`, String(name)];
                  }
                  return [String(value), String(name)];
                }}
              />
              <Bar
                yAxisId="left"
                dataKey="rate"
                fill="rgba(0,214,143,0.85)"
                name="accept rate"
                radius={[6, 6, 0, 0]}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="meanProb"
                stroke="rgba(240,180,41,0.95)"
                strokeWidth={2}
                dot={{ r: 4 }}
                name="mean P(accept)"
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
      Run a generation to populate per-depth acceptance.
    </div>
  );
}
