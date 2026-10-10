import type { MetricsLatest } from "./types";

// Spans of the server's TTFT breakdown (mtplx/server/request_spans.py) in
// which a request waited for other work instead of doing its own.
export const WAIT_SPANS = [
  ["response_tail_wait_s", "previous turn's commit"],
  ["postcommit_wait_s", "postcommit"],
  ["scheduler_queue_s", "scheduler queue"],
  ["lock_wait_s", "generation lock"],
] as const;

export type RequestWait = {
  total_s: number;
  parts: { label: string; seconds: number }[];
};

// The time before a request's own work started. Read from the spans rather
// than `lock_wait_time_s`, which counts only part of it and, where it carries
// the queue wait, would count that span twice.
export function requestWait(row: MetricsLatest): RequestWait | null {
  const spans = row.ttft_spans?.exclusive_s;
  if (!spans) return null;
  const parts = WAIT_SPANS.flatMap(([key, label]) => {
    const seconds = spans[key];
    return typeof seconds === "number" && seconds > 0 ? [{ label, seconds }] : [];
  });
  return { total_s: parts.reduce((sum, part) => sum + part.seconds, 0), parts };
}
