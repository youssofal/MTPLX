import { useMutation, useQueryClient } from "@tanstack/react-query";
import { HardDrive, Info, MemoryStick, Snowflake, TriangleAlert, X, Zap } from "lucide-react";
import type { ReactNode } from "react";
import { api } from "../lib/api";
import { fmtAge, followUpHits, ramOverview, ssdOverview } from "../lib/cacheOverview";
import { fmtBytes, fmtNumber, fmtSeconds, relativeTime, truncateMiddle } from "../lib/utils";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";
import type {
  ColdTierStats,
  MetricsLatest,
  SessionBank,
  SessionBankPrefix,
  SessionRow,
} from "../lib/types";

const RAM_EVICTION_NOTE =
  "A snapshot leaves RAM when the bank goes over its byte budget or entry cap " +
  "(idle conversations go first), or when it has not been used for the idle TTL.";
const SSD_EVICTION_NOTE =
  "When a write would go over the byte cap, the least recently used snapshots " +
  "are deleted first. There is no age limit: an old snapshot stays until space is needed.";

export function SessionBankGrid() {
  const sessionBank = useDashboardStore((s) => s.sessionBank);
  const sessions = useDashboardStore((s) => s.sessions);
  const recent = useDashboardStore((s) => s.recent);
  const setSessionFilter = useDashboardStore((s) => s.setSessionFilter);
  const sessionFilter = useDashboardStore((s) => s.sessionFilter);
  const prefixes = sessionBank?.prefixes ?? [];
  const sessionByPrefix: Record<string, SessionRow> = (sessions?.sessions ?? []).reduce(
    (acc, row) => ((acc[row.session_id] = row), acc),
    {} as Record<string, SessionRow>,
  );

  const queryClient = useQueryClient();
  const clearMutation = useMutation({
    mutationFn: (sessionId: string) => api.postClearSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });

  return (
    <Card
      title="SessionBank · cache overview"
      subtitle={
        sessionBank?.last_miss_reason
          ? `what is cached in RAM and on SSD · last miss: ${sessionBank.last_miss_reason}`
          : "what is cached in RAM and on SSD"
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RamPanel bank={sessionBank} recent={recent} />
        <SsdPanel cold={sessionBank?.cold_tier} />
      </div>
      <div className="mt-4 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        In RAM now · {prefixes.length} of {sessionBank?.max_entries ?? "—"} slots used
      </div>
      {prefixes.length > 0 ? (
        <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-2">
          {prefixes.map((slot, idx) => (
            <SlotCard
              key={`${slot.session_id}:${slot.prefix_len}`}
              index={idx}
              slot={slot}
              session={sessionByPrefix[slot.session_id]}
              isFiltered={sessionFilter === slot.session_id}
              onClickSession={(sessionId) => setSessionFilter(sessionId)}
              onEvict={(sessionId) => clearMutation.mutate(sessionId)}
            />
          ))}
        </div>
      ) : (
        <p className="mt-2 text-xs text-[var(--text-muted)]">Nothing cached in RAM.</p>
      )}
    </Card>
  );
}

function RamPanel({ bank, recent }: { bank: SessionBank | null; recent: MetricsLatest[] }) {
  const ram = ramOverview(bank);
  const hits = followUpHits(recent);
  const caption =
    ram.budgetBytes === null
      ? "no byte budget reported"
      : ram.configuredMaxBytes !== null && ram.configuredMaxBytes !== ram.budgetBytes
        ? `of ${fmtBytes(ram.budgetBytes)} budget now (configured ${fmtBytes(ram.configuredMaxBytes)})`
        : `of ${fmtBytes(ram.budgetBytes)} budget`;
  return (
    <TierPanel
      icon={<MemoryStick className="size-3.5 text-[var(--accent-warm)]" />}
      label="RAM · warm bank"
      note={
        ram.idleTtlS !== null
          ? `${RAM_EVICTION_NOTE} Idle TTL: ${fmtSeconds(ram.idleTtlS)}.`
          : RAM_EVICTION_NOTE
      }
      usedBytes={ram.usedBytes}
      caption={caption}
      fraction={ram.usedFraction}
    >
      <Stat label="snapshots" value={fmtNumber(ram.snapshots)} />
      <Stat label="conversations" value={fmtNumber(ram.sessions)} />
      <Stat
        label="follow-ups from RAM"
        value={
          hits.ramFraction === null
            ? "—"
            : `${hits.ram} / ${hits.followUps} · ${(hits.ramFraction * 100).toFixed(0)}%`
        }
        title={
          `Last ${recent.length} requests: ${hits.ram} follow-up turns found their prefix ` +
          `in RAM, ${hits.ssd} on SSD, ${hits.missed} missed. First turns are not counted.`
        }
      />
      <Stat
        label="from SSD · missed"
        value={hits.followUps > 0 ? `${hits.ssd} · ${hits.missed}` : "—"}
        tone={hits.missed > 0 ? "warm" : undefined}
      />
    </TierPanel>
  );
}

function SsdPanel({ cold }: { cold: ColdTierStats | undefined }) {
  const ssd = ssdOverview(cold, Date.now() / 1000);
  if (!ssd.enabled) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3 text-xs text-[var(--text-muted)]">
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest">
          <HardDrive className="size-3.5" /> SSD · cold tier
        </div>
        <p className="mt-2">SSD cache off: conversations are cached in RAM only.</p>
      </div>
    );
  }
  const dedupe =
    ssd.logicalBytes > ssd.physicalBytes
      ? ` · ${fmtBytes(ssd.logicalBytes)} before dedupe`
      : "";
  return (
    <TierPanel
      icon={<HardDrive className="size-3.5 text-[var(--accent-cool)]" />}
      label="SSD · cold tier"
      note={SSD_EVICTION_NOTE}
      usedBytes={ssd.physicalBytes}
      caption={
        ssd.capBytes === null ? `on disk${dedupe}` : `of ${fmtBytes(ssd.capBytes)} cap${dedupe}`
      }
      fraction={ssd.usedFraction}
    >
      <Stat label="snapshots" value={fmtNumber(ssd.snapshots)} />
      <Stat
        label="conversations"
        value={
          ssd.sessions === null
            ? "—"
            : ssd.snapshotsWithoutSession > 0
              ? `${fmtNumber(ssd.sessions)} + ${fmtNumber(ssd.snapshotsWithoutSession)} unnamed`
              : fmtNumber(ssd.sessions)
        }
        title={
          ssd.snapshotsWithoutSession > 0
            ? `${ssd.snapshotsWithoutSession} snapshots were written without a session id.`
            : undefined
        }
      />
      <Stat
        label="waiting to write"
        value={
          ssd.writePending
            ? `${fmtNumber(ssd.writerQueueDepth)} · ${fmtBytes(ssd.writerBacklogBytes)}`
            : "nothing"
        }
        tone={ssd.writePending ? "warm" : undefined}
        icon={ssd.writePending ? <TriangleAlert className="size-3" /> : undefined}
        title="Snapshots accepted for writing that have not reached disk yet. Until they do, that conversation is cached in RAM only."
      />
      <Stat
        label="restores · failed"
        value={`${fmtNumber(ssd.restoreHits)} · ${fmtNumber(ssd.restoreFailures)}`}
        tone={ssd.restoreFailures > 0 ? "hot" : undefined}
      />
      <Stat label="oldest · newest" value={`${fmtAge(ssd.oldestAgeS)} · ${fmtAge(ssd.newestAgeS)}`} />
      <Stat
        label="never restored"
        value={
          ssd.neverRestored === null
            ? "—"
            : `${fmtNumber(ssd.neverRestored)} of ${fmtNumber(ssd.snapshots)}`
        }
      />
    </TierPanel>
  );
}

function TierPanel({
  icon,
  label,
  note,
  usedBytes,
  caption,
  fraction,
  children,
}: {
  icon: ReactNode;
  label: string;
  note: string;
  usedBytes: number;
  caption: string;
  fraction: number | null;
  children: ReactNode;
}) {
  const pct = fraction === null ? 0 : fraction * 100;
  const color =
    pct >= 95 ? "var(--accent-hot)" : pct >= 80 ? "var(--accent-warm)" : "var(--accent)";
  return (
    <div className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-3">
      <div className="flex items-center justify-between">
        <span className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
          {icon}
          {label}
        </span>
        <span
          title={note}
          className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-help"
        >
          <Info className="size-3.5" />
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-2">
        <span className="text-2xl font-semibold tabular-nums text-[var(--text-primary)]">
          {fmtBytes(usedBytes)}
        </span>
        <span className="text-xs text-[var(--text-muted)]">{caption}</span>
      </div>
      <div className="mt-2 h-2 w-full rounded-full bg-[var(--bg-card)] overflow-hidden border border-[var(--border-soft)]">
        <div
          className="h-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">{children}</dl>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
  icon,
  title,
}: {
  label: string;
  value: string;
  tone?: "warm" | "hot";
  icon?: ReactNode;
  title?: string;
}) {
  const color =
    tone === "hot"
      ? "text-[var(--accent-hot)]"
      : tone === "warm"
        ? "text-[var(--accent-warm)]"
        : "text-[var(--text-primary)]";
  return (
    <div className="flex items-baseline justify-between gap-2" title={title}>
      <dt className="text-[var(--text-muted)] uppercase tracking-wider text-[9px]">{label}</dt>
      <dd className={`inline-flex items-center gap-1 tabular-nums ${color}`}>
        {icon}
        {value}
      </dd>
    </div>
  );
}

function SlotCard({
  index,
  slot,
  session,
  isFiltered,
  onClickSession,
  onEvict,
}: {
  index: number;
  slot: SessionBankPrefix;
  session: SessionRow | undefined;
  isFiltered: boolean;
  onClickSession: (sessionId: string) => void;
  onEvict: (sessionId: string) => void;
}) {
  const ageS = Date.now() / 1000 - slot.last_access_s;
  const inFlight = Boolean(session?.in_flight);
  const hot = ageS < 30;
  return (
    <button
      type="button"
      onClick={() => onClickSession(slot.session_id)}
      className={
        "group relative text-left rounded-lg border bg-[var(--bg-elevated)] p-2.5 transition-colors " +
        (isFiltered
          ? "border-[var(--accent)] shadow-[0_0_0_1px_var(--accent)]"
          : hot
            ? "border-[var(--accent)]/40 hover:border-[var(--accent)]"
            : "border-[var(--border-soft)] hover:border-[var(--text-muted)]")
      }
    >
      <div className="flex items-center justify-between pr-5">
        <span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
          slot {index + 1}
        </span>
        {inFlight ? (
          <span className="inline-flex items-center gap-1 text-[10px] text-[var(--accent)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
            in flight
          </span>
        ) : hot ? (
          <Zap className="size-3 text-[var(--accent-warm)]" />
        ) : (
          <Snowflake className="size-3 text-[var(--accent-cool)]" />
        )}
      </div>
      <div className="text-xs font-mono text-[var(--text-primary)] mt-1 truncate">
        {truncateMiddle(slot.session_id, 24)}
      </div>
      <dl className="mt-1.5 grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
        <Tag label="prefix" value={fmtNumber(slot.prefix_len)} />
        <Tag label="hits" value={fmtNumber(slot.hits)} />
        <Tag label="bytes" value={fmtBytes(slot.nbytes)} />
        <Tag label="age" value={relativeTime(slot.last_access_s)} />
      </dl>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onEvict(slot.session_id);
        }}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 text-[var(--text-muted)] hover:text-[var(--accent-hot)] transition-opacity"
        title="Evict this slot"
      >
        <X className="size-3.5" />
      </button>
    </button>
  );
}

function Tag({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-1">
      <dt className="text-[var(--text-muted)] uppercase tracking-wider text-[9px]">{label}</dt>
      <dd className="text-[var(--text-primary)] tabular-nums">{value}</dd>
    </div>
  );
}
