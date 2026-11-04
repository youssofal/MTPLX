/** What the request log shows for a request that did not hit the session cache. */
export type CacheMissBadge = {
  /** Badge text, e.g. "NEW CHAT". */
  label: string;
  /** Tailwind classes for the badge. */
  color: string;
  /** What happened, in a sentence; the raw reason is appended. */
  tooltip: string;
};

const GREY = "text-[var(--text-muted)] bg-[var(--text-muted)]/10";
const YELLOW = "text-[var(--accent-warm)] bg-[var(--accent-warm)]/10";
const ORANGE = "text-[var(--accent-orange)] bg-[var(--accent-orange)]/10";

/**
 * The server reports why a request found nothing to reuse in
 * `cache_miss_reason`. Most misses are normal, so only a failed SSD lookup or
 * restore is coloured as a problem.
 *
 * `ssd_prefix_miss` is what older servers reported for every normal miss (the
 * SSD is the last place searched); it stays readable and is shown neutrally.
 */
export function cacheMissBadge(reason: string | null | undefined): CacheMissBadge {
  const raw = reason ?? "";
  const withRaw = (text: string) => (raw ? `${text} (${raw})` : text);
  if (raw === "new_session") {
    return {
      label: "NEW CHAT",
      color: GREY,
      tooltip: withRaw("A new conversation: nothing earlier to reuse."),
    };
  }
  if (raw === "prefix_divergence_at_token") {
    return {
      label: "NO SHARED PREFIX",
      color: GREY,
      tooltip: withRaw(
        "Other conversations are stored, but this prompt shares too little with them to reuse (below the reuse minimum).",
      ),
    };
  }
  if (raw === "evicted") {
    return {
      label: "EVICTED",
      color: YELLOW,
      tooltip: withRaw(
        "This conversation's earlier turn was evicted from memory and is not on the SSD.",
      ),
    };
  }
  if (
    raw.startsWith("ssd_restore_error") ||
    raw.startsWith("ssd_missing_") ||
    raw === "ssd_prefix_no_recurrent_boundary"
  ) {
    return {
      label: "SSD ERROR",
      color: ORANGE,
      tooltip: withRaw("The SSD lookup or restore failed."),
    };
  }
  if (raw === "ssd_prefix_miss") {
    return {
      label: "NO SSD MATCH",
      color: GREY,
      tooltip: withRaw(
        "Nothing reusable in memory or on the SSD. Older servers report this for every normal miss.",
      ),
    };
  }
  return {
    label: (raw || "MISS").toUpperCase(),
    color: YELLOW,
    tooltip: raw ? `Cache miss: ${raw}` : "Cache miss",
  };
}
