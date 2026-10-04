import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "./Card";
import { fmtSeconds } from "../lib/utils";
import { useDashboardStore } from "../state/store";

type Slice = { key: string; label: string; color: string; description: string };

const SLICES: Slice[] = [
  { key: "verify_forward_time_s", label: "verify forward", color: "rgba(0,214,143,0.85)", description: "Forward pass through the verify graph (target model)" },
  { key: "verify_logits_eval_time_s", label: "logits eval", color: "rgba(79,182,243,0.85)", description: "Logits evaluation against MTP draft tokens" },
  { key: "verify_hidden_eval_time_s", label: "hidden eval", color: "rgba(155,118,233,0.85)", description: "Hidden-state evaluation for downstream cache writes" },
  { key: "verify_target_distribution_time_s", label: "target dist", color: "rgba(245,158,11,0.85)", description: "Target distribution computation (probability ratio)" },
  { key: "verify_eval_unattributed_time_s", label: "unattributed", color: "rgba(244,114,182,0.75)", description: "Unaccounted-for eval cost; ideally near zero" },
  { key: "accept_time_s", label: "accept", color: "rgba(0,214,143,0.55)", description: "Acceptance sampling + residual correction" },
  { key: "repair_time_s", label: "repair", color: "rgba(239,68,68,0.85)", description: "Repair pass after rejection (lazy when 0)" },
  { key: "snapshot_time_s", label: "snapshot", color: "rgba(200,210,220,0.45)", description: "Cache snapshot/restore" },
  { key: "capture_commit_time_s", label: "capture/commit", color: "rgba(0,214,143,0.35)", description: "Capture-commit verifier overhead" },
  { key: "rollback_time_s", label: "rollback", color: "rgba(240,88,106,0.55)", description: "State rollback after reject" },
];

export function VerifyWaterfall() {
  const latest = useDashboardStore((s) => s.latest);

  const total = Number(latest?.verify_time_s ?? 0);
  const rows = SLICES.map((slice) => {
    const seconds = Number((latest as Record<string, unknown> | null)?.[slice.key] ?? 0) || 0;
    return {
      ...slice,
      seconds,
      pct: total > 0 ? (seconds / total) * 100 : 0,
    };
  });
  const hasData = rows.some((r) => r.seconds > 0);

  return (
    <Card
      title="Verify-cycle waterfall"
      subtitle={
        latest
          ? `verify total ${fmtSeconds(total)} · target forward ${fmtSeconds(
              latest?.target_forward_time_s,
            )} · draft ${fmtSeconds(latest?.draft_time_s)}`
          : "no completed verify cycle"
      }
    >
      <div className="h-[280px]">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={rows}
              margin={{ top: 4, right: 30, left: 110, bottom: 0 }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis
                type="number"
                stroke="rgba(200,210,220,0.6)"
                tickFormatter={(v) => `${(v * 1000).toFixed(0)}ms`}
              />
              <YAxis
                type="category"
                dataKey="label"
                stroke="rgba(200,210,220,0.7)"
                width={100}
                interval={0}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-soft)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: "var(--text-muted)" }}
                formatter={(value, _name, item) => {
                  const slice = SLICES.find((s) => s.label === item?.payload?.label);
                  if (typeof value !== "number") return [value, slice?.label ?? "—"];
                  return [
                    `${fmtSeconds(value)} · ${item?.payload?.pct?.toFixed(1) ?? "—"}%`,
                    slice?.description ?? slice?.label ?? "—",
                  ];
                }}
              />
              <Bar dataKey="seconds" radius={[0, 6, 6, 0]}>
                {rows.map((entry) => (
                  <Cell key={entry.key} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full grid place-items-center text-[var(--text-muted)] text-sm">
            Run a generation to capture the verify decomposition.
          </div>
        )}
      </div>
    </Card>
  );
}
