import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function fmtBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Number(bytes);
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(idx === 0 ? 0 : value < 10 ? 2 : 1)} ${units[idx]}`;
}

export function fmtSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const s = Number(seconds);
  if (s < 1) return `${(s * 1000).toFixed(0)}ms`;
  if (s < 60) return `${s.toFixed(2)}s`;
  if (s < 3600) {
    const m = Math.floor(s / 60);
    const r = Math.round(s - m * 60);
    return `${m}m ${r}s`;
  }
  const h = Math.floor(s / 3600);
  const m = Math.round((s - h * 3600) / 60);
  return `${h}h ${m}m`;
}

export function fmtNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtTokS(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}`;
}

export function truncateMiddle(text: string | null | undefined, max = 24): string {
  if (!text) return "—";
  if (text.length <= max) return text;
  const head = Math.ceil((max - 3) / 2);
  const tail = Math.floor((max - 3) / 2);
  return `${text.slice(0, head)}...${text.slice(-tail)}`;
}

export function relativeTime(timestampS: number | null | undefined): string {
  if (!timestampS) return "—";
  const diff = Date.now() / 1000 - timestampS;
  if (diff < 1) return "just now";
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

// `_metrics_envelope` emits `accepted_by_depth` and `drafted_by_depth` but not
// the flat `accepted_drafts` / `drafted_tokens` totals, so the surfaces that
// show a total derive it from the per-depth arrays. A total emitted by the
// envelope still wins, so this keeps working if the server starts sending one.
export function depthTotal(
  emitted: number | null | undefined,
  byDepth: number[] | null | undefined,
): number {
  if (emitted !== null && emitted !== undefined) return emitted;
  return (byDepth ?? []).reduce((sum, n) => sum + n, 0);
}
