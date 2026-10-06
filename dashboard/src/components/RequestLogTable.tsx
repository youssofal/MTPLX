import { Fragment, useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Card } from "./Card";
import { useRecentRequests } from "../hooks/usePolling";
import { fmtNumber, fmtSeconds, fmtTokS, relativeTime, truncateMiddle } from "../lib/utils";
import { useDashboardStore, useFilteredRecent } from "../state/store";
import { refusalOf } from "../lib/requestRefusal";
import type { MetricsLatest } from "../lib/types";

export function RequestLogTable() {
  const polledRecent = useRecentRequests();
  const storeRecent = useFilteredRecent();
  const sessionFilter = useDashboardStore((s) => s.sessionFilter);

  // Polled data is fresher than the SSE snapshot's `recent`; reconcile.
  const rows = useMemo<MetricsLatest[]>(() => {
    const fromPolled = polledRecent.data?.recent ?? [];
    const merged = fromPolled.length > 0 ? fromPolled : storeRecent;
    if (!sessionFilter) return merged.slice().reverse();
    return merged.filter((r) => r.session_id === sessionFilter).reverse();
  }, [polledRecent.data?.recent, storeRecent, sessionFilter]);

  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  return (
    <Card
      title="Recent requests"
      subtitle={
        rows.length === 0
          ? "no requests yet"
          : `${rows.length} of ${polledRecent.data?.recent?.length ?? storeRecent.length}${
              sessionFilter ? ` · filtered by ${sessionFilter}` : ""
            }`
      }
    >
      {rows.length === 0 ? (
        <div className="text-sm text-[var(--text-muted)]">
          Drive a few requests against this server and they will appear here in
          order, most recent first.
        </div>
      ) : (
        <div className="overflow-x-auto -mx-3">
          <table className="min-w-full text-sm">
            <thead className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
              <tr>
                <Th />
                <Th>session</Th>
                <Th align="right">prompt</Th>
                <Th align="right">cached</Th>
                <Th align="right">gen</Th>
                <Th align="right">tok/s</Th>
                <Th align="right">ttft</Th>
                <Th align="right">verify</Th>
                <Th>cache</Th>
                <Th align="right">when</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const open = expanded.has(idx);
                return (
                  <Row
                    key={`${row.session_id ?? "x"}-${idx}`}
                    row={row}
                    isOpen={open}
                    onToggle={() =>
                      setExpanded((prev) => {
                        const next = new Set(prev);
                        if (next.has(idx)) {
                          next.delete(idx);
                        } else {
                          next.add(idx);
                        }
                        return next;
                      })
                    }
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function Th({
  children,
  align = "left",
}: {
  children?: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={`px-3 py-2 font-medium whitespace-nowrap ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  highlight = false,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  highlight?: boolean;
}) {
  return (
    <td
      className={`px-3 py-2 whitespace-nowrap ${
        align === "right" ? "text-right tabular-nums" : ""
      } ${highlight ? "text-[var(--accent)] font-medium" : "text-[var(--text-primary)]"}`}
    >
      {children}
    </td>
  );
}

export function Row({
  row,
  isOpen,
  onToggle,
}: {
  row: MetricsLatest;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const sessionId = row.session_id ?? "—";
  const refusal = refusalOf(row);
  const cacheBadge = refusal
    ? {
        label: refusal.label,
        color: "text-[var(--accent-hot)] bg-[var(--accent-hot)]/10",
      }
    : row.session_cache_hit
    ? { label: "HIT", color: "text-[var(--accent)] bg-[var(--accent)]/10" }
    : {
        label: (row.cache_miss_reason ?? "MISS").toUpperCase(),
        color: "text-[var(--accent-warm)] bg-[var(--accent-warm)]/10",
      };
  return (
    <>
      <tr className="border-t border-[var(--border-soft)] hover:bg-[var(--bg-elevated)]/60">
        <Td>
          <button
            type="button"
            onClick={onToggle}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
            aria-label={isOpen ? "Collapse" : "Expand"}
          >
            {isOpen ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
          </button>
        </Td>
        <Td>
          <span className="font-mono text-xs">{truncateMiddle(sessionId, 20)}</span>
        </Td>
        <Td align="right">{fmtNumber(row.prompt_tokens)}</Td>
        <Td align="right">{fmtNumber(row.cached_tokens)}</Td>
        <Td align="right">{refusal ? "—" : fmtNumber(row.completion_tokens)}</Td>
        <Td align="right" highlight>
          {refusal ? "—" : fmtTokS(row.decode_tok_s)}
        </Td>
        <Td align="right">{refusal ? "—" : fmtSeconds(row.ttft_s)}</Td>
        <Td align="right">{refusal ? "—" : fmtNumber(row.verify_calls)}</Td>
        <Td>
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wider ${cacheBadge.color}`}
          >
            {cacheBadge.label}
          </span>
        </Td>
        <Td align="right" highlight={false}>
          <span className="text-[var(--text-muted)] text-xs">
            {relativeTime(row.completed_at_s)}
          </span>
        </Td>
      </tr>
      {isOpen ? (
        <tr className="bg-[var(--bg-elevated)]/40">
          <td colSpan={10} className="px-3 py-3">
            {refusal && refusal.details.length > 0 ? (
              <dl className="mb-2 text-xs grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1">
                {refusal.details.map(([name, value]) => (
                  <Fragment key={name}>
                    <dt className="text-[var(--text-muted)]">{name}</dt>
                    <dd className="text-[var(--text-primary)]">{value}</dd>
                  </Fragment>
                ))}
              </dl>
            ) : null}
            <pre className="text-[11px] leading-relaxed text-[var(--text-muted)] overflow-x-auto max-h-[260px]">
              {JSON.stringify(row, null, 2)}
            </pre>
          </td>
        </tr>
      ) : null}
    </>
  );
}
