import { BigNumber, Card } from "./Card";
import { depthTotal, fmtNumber, fmtTokS } from "../lib/utils";
import { useDashboardStore } from "../state/store";

export function VerifyRatioTile() {
  const latest = useDashboardStore((s) => s.latest);
  const drafted = depthTotal(latest?.drafted_tokens, latest?.drafted_by_depth);
  const verifies = latest?.verify_calls ?? 0;
  const ratio = verifies > 0 ? drafted / verifies : null;
  return (
    <Card title="Drafted / verify call" subtitle="higher is faster">
      <BigNumber
        value={ratio === null ? "—" : ratio.toFixed(2)}
        unit="tok/call"
        tone={typeof ratio === "number" && ratio >= 3 ? "accent" : "default"}
        caption={`${fmtNumber(drafted)} drafted · ${fmtNumber(verifies)} verifies`}
      />
    </Card>
  );
}

export function CorrectionBonusTile() {
  const latest = useDashboardStore((s) => s.latest);
  const correction = latest?.correction_tokens ?? 0;
  const bonus = latest?.bonus_tokens ?? 0;
  return (
    <Card title="Correction vs bonus tokens" subtitle="dropped + reborn tokens">
      <div className="flex items-baseline gap-6">
        <div>
          <div className="text-2xl font-semibold text-[var(--accent-hot)] tabular-nums">
            {fmtNumber(correction)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            correction
          </div>
        </div>
        <div>
          <div className="text-2xl font-semibold text-[var(--accent)] tabular-nums">
            {fmtNumber(bonus)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            bonus
          </div>
        </div>
      </div>
      <div className="text-xs text-[var(--text-muted)] mt-3">
        bonus = accepted &gt; drafted at depth d; correction = residual fix-up
      </div>
    </Card>
  );
}

export function ServerTokSTile() {
  const latest = useDashboardStore((s) => s.latest);
  const requestTokS = latest?.request_tok_s ?? null;
  const decodeTokS = latest?.decode_tok_s ?? null;
  return (
    <Card title="Decode vs request tok/s" subtitle="decode excludes prefill">
      <div className="flex items-baseline gap-6">
        <div>
          <div className="text-2xl font-semibold text-[var(--accent)] tabular-nums">
            {fmtTokS(decodeTokS)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            decode tok/s
          </div>
        </div>
        <div>
          <div className="text-2xl font-semibold text-[var(--accent-cool)] tabular-nums">
            {fmtTokS(requestTokS)}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            request tok/s
          </div>
        </div>
      </div>
    </Card>
  );
}
