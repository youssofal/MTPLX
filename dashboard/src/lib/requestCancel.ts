import type { MetricsLatest } from "./types";
import { fmtNumber, fmtSeconds } from "./utils";

export type Cancellation = {
  /** Badge text, e.g. "cancelled · client · 3m 0s". */
  label: string;
  /** Why and after how long, in a sentence, plus the cache outcome. */
  tooltip: string;
  /** The cache outcome the request still had, e.g. "hit 92,636"; null when unknown. */
  cacheHint: string | null;
};

// Short names for the badge; the tooltip carries the raw reason. The keys are
// the server's cancellation reasons (the cancel origins in openai.py).
const REASON_LABELS: Record<string, string> = {
  client_disconnected: "client",
  client_disconnect: "client",
  stream_cancelled: "stream",
  stream_teardown: "stream",
  stream_stall: "stall",
  stop_sequence: "stop",
  early_tool_finish: "tool call",
  hidden_tool_guard: "tool guard",
  post_endpoint: "cancel call",
};

const REASON_SENTENCES: Record<string, string> = {
  client_disconnected: "the client closed the connection",
  client_disconnect: "the client closed the connection",
  stream_stall: "the stream stalled",
  stop_sequence: "a stop sequence ended the answer",
  early_tool_finish: "the server finished the tool call early",
  hidden_tool_guard: "the hidden-tool stream guard stopped it",
  post_endpoint: "POST /v1/mtplx/cancel stopped it",
};

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

/**
 * The cache outcome a cancelled row still reports; null when the server did
 * not know it yet. Older servers wrote `session_cache_hit: false` and no
 * `cached_tokens` on every cancelled row, so without a token count the
 * outcome is unknown as well.
 */
export function cancelledCacheHint(row: MetricsLatest): string | null {
  if (row.cached_tokens === null || row.cached_tokens === undefined) return null;
  if (row.session_cache_hit === true) {
    const source = row.cache_source && row.cache_source !== "ram" ? ` ${row.cache_source}` : "";
    return `hit${source} ${fmtNumber(row.cached_tokens)}`;
  }
  if (row.session_cache_hit === false) {
    return row.cache_miss_reason ? `miss: ${humanize(row.cache_miss_reason)}` : "miss";
  }
  return null;
}

/**
 * Describe a request that was cancelled before it finished, or null for one
 * that ran to the end. Its row would otherwise fall back to the MISS badge,
 * whatever the cache did.
 */
export function cancellationOf(row: MetricsLatest): Cancellation | null {
  if (!row.request_cancelled) return null;
  const reason = row.cancellation_reason ?? "";
  const elapsed = row.cancellation_elapsed_s ?? row.request_elapsed_s;
  const parts = ["cancelled"];
  if (reason) parts.push(REASON_LABELS[reason] ?? humanize(reason));
  if (typeof elapsed === "number" && Number.isFinite(elapsed)) parts.push(fmtSeconds(elapsed));
  const why = reason ? (REASON_SENTENCES[reason] ?? humanize(reason)) : "unknown reason";
  const after =
    typeof elapsed === "number" && Number.isFinite(elapsed) ? ` after ${fmtSeconds(elapsed)}` : "";
  const cacheHint = cancelledCacheHint(row);
  const cacheLine =
    cacheHint === null
      ? "Cache outcome unknown: the request ended before its prompt was read."
      : `Cache: ${cacheHint}.`;
  return {
    label: parts.join(" · "),
    tooltip: `Cancelled${after}: ${why}${reason ? ` (${reason})` : ""}.\n${cacheLine}\nNot counted in the hit rate.`,
    cacheHint,
  };
}

/** Rows that count toward a cache hit rate: requests that ran to the end. */
export function countsTowardHitRate(row: MetricsLatest): boolean {
  return !row.request_cancelled;
}
