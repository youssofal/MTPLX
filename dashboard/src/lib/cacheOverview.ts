import type { ColdTierStats, MetricsLatest, SessionBank } from "./types";

// Totals behind the Cache tab's overview: what the warm bank holds in RAM and
// what the cold tier holds on SSD. Pure functions over the snapshot, so the
// card stays a thin view and the arithmetic is testable on its own.

export type RamOverview = {
  usedBytes: number;
  // The cap eviction works against: the dynamic ceiling when the server
  // reports one, else the configured max_bytes.
  budgetBytes: number | null;
  configuredMaxBytes: number | null;
  usedFraction: number | null;
  snapshots: number;
  sessions: number;
  idleTtlS: number | null;
};

export function ramOverview(bank: SessionBank | null): RamOverview {
  const prefixes = bank?.prefixes ?? [];
  const configured = positiveOrNull(bank?.max_bytes);
  const budget = positiveOrNull(bank?.effective_max_bytes) ?? configured;
  const used = Math.max(0, Number(bank?.total_nbytes ?? 0));
  return {
    usedBytes: used,
    budgetBytes: budget,
    configuredMaxBytes: configured,
    usedFraction: budget ? Math.min(1, used / budget) : null,
    snapshots: bank?.entries ?? prefixes.length,
    sessions: new Set(prefixes.map((p) => p.session_id)).size,
    idleTtlS: positiveOrNull(bank?.idle_ttl_s),
  };
}

export type FollowUpHits = {
  followUps: number;
  ram: number;
  ssd: number;
  missed: number;
  ramFraction: number | null;
};

// Where follow-up turns found their prefix. A follow-up is a request that
// either hit the cache or missed for a reason other than being the first
// turn of its conversation; first turns have nothing to find.
export function followUpHits(recent: MetricsLatest[]): FollowUpHits {
  let ram = 0;
  let ssd = 0;
  let missed = 0;
  for (const row of recent) {
    if (!row.session_id) continue;
    if (row.session_cache_hit) {
      if (String(row.session_restore_mode ?? "").startsWith("ssd")) ssd += 1;
      else ram += 1;
    } else if (row.cache_miss_reason && row.cache_miss_reason !== "new_session") {
      missed += 1;
    }
  }
  const followUps = ram + ssd + missed;
  return {
    followUps,
    ram,
    ssd,
    missed,
    ramFraction: followUps > 0 ? ram / followUps : null,
  };
}

export type SsdOverview = {
  enabled: boolean;
  physicalBytes: number;
  logicalBytes: number;
  capBytes: number | null;
  usedFraction: number | null;
  dedupeRatio: number | null;
  snapshots: number;
  sessions: number | null;
  snapshotsWithoutSession: number;
  writerQueueDepth: number;
  writerBacklogBytes: number;
  // Something accepted for writing has not reached disk yet: until it does,
  // that conversation exists only in RAM.
  writePending: boolean;
  restoreHits: number;
  restoreFailures: number;
  oldestAgeS: number | null;
  newestAgeS: number | null;
  neverRestored: number | null;
};

export function ssdOverview(
  cold: ColdTierStats | null | undefined,
  nowS: number,
): SsdOverview {
  const physical = Number(cold?.physical_bytes ?? cold?.bytes ?? 0);
  const logical = Number(cold?.logical_bytes ?? physical);
  const cap = positiveOrNull(cold?.effective_max_bytes) ?? positiveOrNull(cold?.max_bytes);
  const queue = Number(cold?.writer_queue_depth ?? 0);
  const backlog = Number(cold?.writer_backlog_bytes ?? 0);
  return {
    enabled: Boolean(cold?.enabled),
    physicalBytes: physical,
    logicalBytes: logical,
    capBytes: cap,
    usedFraction: cap ? Math.min(1, physical / cap) : null,
    dedupeRatio: cold?.dedupe_ratio ?? null,
    snapshots: Number(cold?.entries ?? 0),
    sessions: cold?.sessions ?? null,
    snapshotsWithoutSession: Number(cold?.entries_without_session ?? 0),
    writerQueueDepth: queue,
    writerBacklogBytes: backlog,
    writePending: queue > 0 || backlog > 0,
    restoreHits: Number(cold?.restore_hits ?? 0),
    restoreFailures: Number(cold?.restore_failures ?? 0),
    oldestAgeS: ageOf(cold?.oldest_created_at_s, nowS),
    newestAgeS: ageOf(cold?.newest_created_at_s, nowS),
    neverRestored: cold?.never_restored_entries ?? null,
  };
}

// "3m", "5h", "2d": an age without the "ago", for "oldest 2d · newest 3m".
export function fmtAge(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function ageOf(createdAtS: number | null | undefined, nowS: number): number | null {
  if (createdAtS === null || createdAtS === undefined) return null;
  return Math.max(0, nowS - createdAtS);
}

function positiveOrNull(value: number | null | undefined): number | null {
  return value !== null && value !== undefined && value > 0 ? Number(value) : null;
}
