import type { MetricsLatest } from "./types";

export type Refusal = {
  /** Badge text, e.g. "507 memory". */
  label: string;
  /** Name/value pairs for the expanded row. */
  details: [string, string][];
};

const KIND_LABELS: Record<string, string> = {
  memory_refusal: "memory",
  http_error: "error",
  engine_error: "error",
};

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

/**
 * Describe a request the server refused or failed (anything carrying an
 * `error_status`), or null for a request that ran. The request log row
 * carries zeros for the generation fields in that case, which would
 * otherwise read as a cache miss.
 */
export function refusalOf(row: MetricsLatest): Refusal | null {
  const status = row.error_status;
  if (typeof status !== "number" || !Number.isFinite(status)) return null;
  const kind = row.error_kind ? (KIND_LABELS[row.error_kind] ?? humanize(row.error_kind)) : "";
  const details: [string, string][] = [];
  if (row.refusal_reason) details.push(["refused because", humanize(row.refusal_reason)]);
  if (row.retry_when) details.push(["retry when", humanize(row.retry_when)]);
  if (row.error_detail) details.push(["detail", row.error_detail]);
  return { label: kind ? `${status} ${kind}` : String(status), details };
}
