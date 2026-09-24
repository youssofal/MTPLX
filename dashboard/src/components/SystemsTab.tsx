import { AlertTriangle, CheckCircle2, CircleOff, Network } from "lucide-react";
import { useRuntimeSystems } from "../hooks/usePolling";
import type { RuntimeSystemContract, RuntimeSystemState } from "../lib/types";
import { relativeTime } from "../lib/utils";
import { Card } from "./Card";

const STATE_KEYS = new Set(["available", "enabled", "wired"]);

export function SystemsTab() {
  const systems = useRuntimeSystems();

  if (systems.isLoading) {
    return (
      <Card title="Runtime systems" subtitle="Loading the server capability contract">
        <div className="text-sm text-[var(--text-muted)]">
          Waiting for <code>/v1/mtplx/systems</code>.
        </div>
      </Card>
    );
  }

  if (systems.isError || !systems.data) {
    const message = String(
      (systems.error as Error | undefined)?.message ?? "No response payload",
    );
    return (
      <Card title="Runtime systems" subtitle="This server does not expose the systems contract">
        <div className="flex items-start gap-2 text-sm text-[var(--accent-warm)]">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div>
            <p>
              <code>/v1/mtplx/systems</code> is unavailable. Other dashboard views remain
              operational.
            </p>
            <p className="mt-2 text-xs text-[var(--text-muted)]">{message}</p>
          </div>
        </div>
      </Card>
    );
  }

  const payload = systems.data;
  const entries = Object.entries(payload.systems ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  );

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <Card bodyClassName="pt-5">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 text-base font-semibold text-[var(--text-primary)]">
                <Network className="size-4 text-[var(--accent-cool)]" />
                Runtime systems
              </div>
              <p className="mt-1 max-w-3xl text-sm text-[var(--text-muted)]">
                Live capability state reported by <code>/v1/mtplx/systems</code>. The
                dashboard displays server data without enabling or changing any system.
              </p>
            </div>
            <div className="text-right text-xs text-[var(--text-muted)]">
              <div>{payload.system_count.toLocaleString()} reported</div>
              <div>revision {payload.revision.toLocaleString()}</div>
              <div>updated {relativeTime(payload.updated_at_s)}</div>
            </div>
          </div>
        </Card>
      </div>

      {entries.length ? (
        entries.map(([name, state]) => (
          <div key={name} className="col-span-12 lg:col-span-6">
            <SystemCard name={name} contract={state} />
          </div>
        ))
      ) : (
        <div className="col-span-12">
          <Card title="No systems reported">
            <p className="text-sm text-[var(--text-muted)]">
              The endpoint responded successfully but did not return any system contracts.
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}

function SystemCard({
  name,
  contract,
}: {
  name: string;
  contract: RuntimeSystemContract;
}) {
  const state = contract.status;
  const metrics = Object.entries(state)
    .filter(([key, value]) => !STATE_KEYS.has(key) && isScalar(value))
    .slice(0, 8);
  const status = systemStatus(state);

  return (
    <Card
      title={humanize(name)}
      subtitle={`Contract revision ${contract.revision.toLocaleString()}, updated ${relativeTime(contract.updated_at_s)}`}
      action={<StatusPill label={status.label} tone={status.tone} />}
    >
      {metrics.length ? (
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {metrics.map(([key, value]) => (
            <div
              key={key}
              className="rounded-lg border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2"
            >
              <dt className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
                {humanize(key)}
              </dt>
              <dd className="mt-1 truncate text-sm font-medium text-[var(--text-primary)]">
                {formatScalar(value)}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-sm text-[var(--text-muted)]">
          The capability was reported without scalar metrics.
        </p>
      )}
      <details className="mt-3 text-xs text-[var(--text-muted)]">
        <summary className="cursor-pointer select-none">Raw contract</summary>
        <pre className="mt-2 max-h-72 overflow-auto rounded-lg border border-[var(--border-soft)] bg-[var(--bg-card)] p-3 font-mono text-[11px] leading-relaxed text-[var(--text-primary)]">
          {JSON.stringify(contract, null, 2)}
        </pre>
      </details>
    </Card>
  );
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "ok" | "warn" | "muted";
}) {
  const Icon = tone === "ok" ? CheckCircle2 : tone === "warn" ? AlertTriangle : CircleOff;
  const className =
    tone === "ok"
      ? "border-[var(--accent-cool)]/30 bg-[var(--accent-cool)]/10 text-[var(--accent-cool)]"
      : tone === "warn"
        ? "border-[var(--accent-warm)]/30 bg-[var(--accent-warm)]/10 text-[var(--accent-warm)]"
        : "border-[var(--border-soft)] bg-[var(--bg-elevated)] text-[var(--text-muted)]";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-wider ${className}`}
    >
      <Icon className="size-3" />
      {label}
    </span>
  );
}

function systemStatus(state: RuntimeSystemState): {
  label: string;
  tone: "ok" | "warn" | "muted";
} {
  if (state.available === false) return { label: "unavailable", tone: "warn" };
  if (state.enabled === false) return { label: "disabled", tone: "muted" };
  if (state.wired === false) return { label: "not wired", tone: "warn" };
  if (state.wired === true) return { label: "wired", tone: "ok" };
  if (state.enabled === true) return { label: "enabled", tone: "ok" };
  return { label: "reported", tone: "muted" };
}

function isScalar(value: unknown): value is string | number | boolean | null {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function formatScalar(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

function humanize(value: string): string {
  const label = value.replaceAll("_", " ").trim();
  return label ? label[0].toUpperCase() + label.slice(1) : value;
}
