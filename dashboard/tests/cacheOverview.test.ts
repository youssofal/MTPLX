import { describe, expect, test } from "bun:test";
import { fmtAge, followUpHits, ramOverview, ssdOverview } from "../src/lib/cacheOverview";
import type { MetricsLatest, SessionBank, SessionBankPrefix } from "../src/lib/types";

const GiB = 1024 ** 3;

function prefix(session_id: string, prefix_len: number): SessionBankPrefix {
  return {
    session_id,
    prefix_len,
    hits: 0,
    nbytes: GiB,
    created_at_s: 0,
    last_access_s: 0,
  };
}

function bank(overrides: Partial<SessionBank>): SessionBank {
  return {
    max_entries: 24,
    total_nbytes: 0,
    last_miss_reason: null,
    prefixes: [],
    eviction_log: [],
    ...overrides,
  };
}

describe("ramOverview", () => {
  test("measures use against the dynamic ceiling and counts distinct sessions", () => {
    const ram = ramOverview(
      bank({
        max_bytes: 24 * GiB,
        effective_max_bytes: 16 * GiB,
        total_nbytes: 4 * GiB,
        entries: 3,
        prefixes: [prefix("a", 100), prefix("a", 200), prefix("b", 300)],
      }),
    );
    expect(ram.budgetBytes).toBe(16 * GiB);
    expect(ram.configuredMaxBytes).toBe(24 * GiB);
    expect(ram.usedFraction).toBe(0.25);
    expect(ram.snapshots).toBe(3);
    expect(ram.sessions).toBe(2);
  });

  test("falls back to max_bytes and to the prefix count", () => {
    const ram = ramOverview(
      bank({ max_bytes: 8 * GiB, total_nbytes: 2 * GiB, prefixes: [prefix("a", 1)] }),
    );
    expect(ram.budgetBytes).toBe(8 * GiB);
    expect(ram.snapshots).toBe(1);
  });

  test("an empty or missing bank has no fraction to show", () => {
    const ram = ramOverview(null);
    expect(ram.usedBytes).toBe(0);
    expect(ram.budgetBytes).toBeNull();
    expect(ram.usedFraction).toBeNull();
    expect(ram.sessions).toBe(0);
  });
});

describe("followUpHits", () => {
  const row = (fields: Partial<MetricsLatest>) => ({ session_id: "s", ...fields }) as MetricsLatest;

  test("splits follow-up turns into RAM, SSD and missed; first turns do not count", () => {
    const hits = followUpHits([
      row({ session_cache_hit: true, session_restore_mode: "clone" }),
      row({ session_cache_hit: true, session_restore_mode: "reference_lease" }),
      row({ session_cache_hit: true, session_restore_mode: "ssd_clone" }),
      row({ session_cache_hit: false, cache_miss_reason: "prefix_divergence_at_token" }),
      row({ session_cache_hit: false, cache_miss_reason: "new_session" }),
      row({ session_cache_hit: false, cache_miss_reason: null }),
      { session_cache_hit: true } as MetricsLatest,
    ]);
    expect(hits).toEqual({ followUps: 4, ram: 2, ssd: 1, missed: 1, ramFraction: 0.5 });
  });

  test("no follow-ups means no rate", () => {
    expect(followUpHits([]).ramFraction).toBeNull();
  });
});

describe("ssdOverview", () => {
  test("reads bytes against the effective cap, dedupe, ages and never-restored", () => {
    const ssd = ssdOverview(
      {
        enabled: true,
        max_bytes: 100 * GiB,
        effective_max_bytes: 80 * GiB,
        physical_bytes: 20 * GiB,
        logical_bytes: 60 * GiB,
        dedupe_ratio: 2 / 3,
        entries: 10,
        sessions: 4,
        entries_without_session: 1,
        oldest_created_at_s: 1_000,
        newest_created_at_s: 9_400,
        never_restored_entries: 7,
        writer_queue_depth: 0,
        writer_backlog_bytes: 0,
        restore_hits: 5,
        restore_failures: 1,
      },
      10_000,
    );
    expect(ssd.capBytes).toBe(80 * GiB);
    expect(ssd.usedFraction).toBe(0.25);
    expect(ssd.logicalBytes).toBe(60 * GiB);
    expect(ssd.sessions).toBe(4);
    expect(ssd.snapshotsWithoutSession).toBe(1);
    expect(ssd.oldestAgeS).toBe(9_000);
    expect(ssd.newestAgeS).toBe(600);
    expect(ssd.neverRestored).toBe(7);
    expect(ssd.writePending).toBe(false);
    expect(ssd.restoreHits).toBe(5);
    expect(ssd.restoreFailures).toBe(1);
  });

  test("a queued write or backlog bytes flag a pending write", () => {
    expect(ssdOverview({ enabled: true, writer_queue_depth: 1 }, 0).writePending).toBe(true);
    expect(ssdOverview({ enabled: true, writer_backlog_bytes: 512 }, 0).writePending).toBe(true);
  });

  test("an empty store or an older server leaves the new fields empty", () => {
    const empty = ssdOverview(
      { enabled: true, entries: 0, sessions: 0, oldest_created_at_s: null },
      10,
    );
    expect(empty.oldestAgeS).toBeNull();
    expect(empty.sessions).toBe(0);
    const older = ssdOverview({ enabled: true, entries: 3, bytes: GiB, max_bytes: 4 * GiB }, 10);
    expect(older.sessions).toBeNull();
    expect(older.neverRestored).toBeNull();
    expect(older.physicalBytes).toBe(GiB);
    expect(older.usedFraction).toBe(0.25);
  });

  test("a disabled or missing tier reports disabled", () => {
    expect(ssdOverview(undefined, 0).enabled).toBe(false);
    expect(ssdOverview({ enabled: false }, 0).enabled).toBe(false);
  });
});

test("fmtAge is a bare age", () => {
  expect(fmtAge(null)).toBe("—");
  expect(fmtAge(42)).toBe("42s");
  expect(fmtAge(180)).toBe("3m");
  expect(fmtAge(5 * 3600)).toBe("5h");
  expect(fmtAge(2 * 86400)).toBe("2d");
});
