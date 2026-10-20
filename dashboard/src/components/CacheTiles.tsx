import { BigNumber, Card } from "./Card";
import { fmtNumber } from "../lib/utils";
import { useDashboardStore } from "../state/store";
import { countsTowardHitRate } from "../lib/requestCancel";

export function CumulativeCachedTokensTile() {
  const lifetime = useDashboardStore((s) => s.lifetime);
  const cached = lifetime?.cached_tokens_total ?? 0;
  const prompt = lifetime?.prompt_tokens_total ?? 0;
  const ratio = prompt > 0 ? (cached / prompt) * 100 : 0;
  return (
    <Card title="Cached tokens · lifetime" subtitle="cached / prompt across all requests">
      <BigNumber
        value={fmtNumber(cached)}
        unit="tokens"
        tone="accent"
        caption={`${ratio.toFixed(1)}% of ${fmtNumber(prompt)} prompt tokens`}
      />
    </Card>
  );
}

export function HitRateGauge() {
  const recent = useDashboardStore((s) => s.recent);
  // A cancelled request says nothing about the cache's hit rate.
  const window = recent.filter(countsTowardHitRate).slice(-32);
  const hits = window.filter((r) => r.session_cache_hit).length;
  const ratio = window.length > 0 ? (hits / window.length) * 100 : 0;
  const tone = ratio >= 70 ? "accent" : ratio >= 40 ? "warm" : "hot";
  return (
    <Card title="Session cache hit rate" subtitle={`last ${window.length} requests`}>
      <BigNumber
        value={`${ratio.toFixed(0)}%`}
        unit="hit"
        tone={tone}
        caption={`${hits} hits / ${window.length} requests`}
      />
    </Card>
  );
}

export function ContextUtilizationBar() {
  const latest = useDashboardStore((s) => s.latest);
  const contextWindow = useDashboardStore((s) => s.contextWindow);
  const len = latest?.context_len ?? 0;
  const pct = contextWindow ? Math.min(100, (len / contextWindow) * 100) : 0;
  const band =
    pct >= 95 ? "hot" : pct >= 75 ? "warm" : pct >= 50 ? "cool" : "accent";
  const color =
    band === "hot"
      ? "var(--accent-hot)"
      : band === "warm"
        ? "var(--accent-warm)"
        : band === "cool"
          ? "var(--accent-cool)"
          : "var(--accent)";
  return (
    <Card title="Context window utilization" subtitle={`${fmtNumber(len)} / ${fmtNumber(contextWindow ?? 0)} tokens`}>
      <div className="h-4 w-full rounded-full bg-[var(--bg-elevated)] overflow-hidden border border-[var(--border-soft)]">
        <div
          className="h-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <div className="flex justify-between mt-2 text-xs text-[var(--text-muted)] tabular-nums">
        <span>0</span>
        <span className="text-[var(--text-primary)] font-semibold">{pct.toFixed(0)}%</span>
        <span>{fmtNumber(contextWindow ?? 0)}</span>
      </div>
    </Card>
  );
}
