import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Copy, Search, Trash } from "lucide-react";
import { useLaunchInfo } from "../hooks/usePolling";
import { api } from "../lib/api";
import {
  REDACTED,
  SOURCE_HINTS,
  SOURCE_LABELS,
  buildRestartCommand,
  commandHasRedactions,
  filterEnvRows,
  launchFlags,
} from "../lib/launchInfo";
import type { LaunchEnvRow, LaunchEnvSource, LaunchInfo, MutableSettings } from "../lib/types";
import { Card } from "./Card";
import { useDashboardStore } from "../state/store";

type MutableKey = keyof MutableSettings;

const MUTABLE_SETTINGS_KEYS: MutableKey[] = [
  "depth",
  "temperature",
  "top_p",
  "top_k",
  "presence_penalty",
  "max_response_tokens",
  "stream_interval",
  "enable_thinking",
  "reasoning_parser",
];

const DEBOUNCE_MS = 250;
const CHANGED_ON_SERVER_HINT_MS = 6000;

export function ControlsSidebar() {
  const launch = useLaunchInfo();
  const info = launch.data?.ok ? launch.data : null;
  const unavailable = launch.isError || (launch.data !== undefined && !launch.data.ok);
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-7">
        <SectionHeading title="Live" caption="Changes apply to the next request." />
        <DefaultsCard />
      </div>
      <div className="col-span-12 lg:col-span-5">
        <SectionHeading title="Fixed until restart" caption="Read-only: how the server was started." />
        <FixedSettingsCard info={info} unavailable={unavailable} />
        <div className="mt-4">
          <RestartCommandCard info={info} unavailable={unavailable} />
        </div>
        <div className="mt-4">
          <AdminActions />
        </div>
      </div>
      {info?.env ? (
        <div className="col-span-12">
          <ServerParametersCard rows={info.env} />
        </div>
      ) : null}
    </div>
  );
}

function SectionHeading({ title, caption }: { title: string; caption: string }) {
  return (
    <div className="mb-2 px-1 flex items-baseline justify-between gap-3">
      <h2 className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">{title}</h2>
      <span className="text-xs text-[var(--text-muted)]">{caption}</span>
    </div>
  );
}

function DefaultsCard() {
  const settings = useDashboardStore((s) => s.settings);
  // `draft` is what the controls show. Every key the user is not editing
  // follows the server on each snapshot; only keys in `dirty` (edited here,
  // not yet applied) are ever sent. Posting a full diff against the live
  // settings used to write a stale draft back over values changed from the
  // native app, the CLI or another tab.
  const [draft, setDraft] = useState<MutableSettings | null>(settings);
  const [dirty, setDirty] = useState<ReadonlySet<MutableKey>>(() => new Set());
  const [changedOnServer, setChangedOnServer] = useState<MutableKey[]>([]);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const mergeSettings = useDashboardStore((s) => s.mergeSettings);
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: (payload: Partial<MutableSettings>) => api.postSettings(payload),
    onSuccess: (res) => {
      // The response carries the server's settings after the write (with
      // any clamping); adopt them now rather than waiting for the next
      // snapshot, so the controls never snap back to the pre-write values.
      const written: Partial<MutableSettings> = {};
      MUTABLE_SETTINGS_KEYS.forEach((key) => {
        if (key in res) (written as Record<string, unknown>)[key] = res[key];
      });
      mergeSettings(written);
      queryClient.invalidateQueries({ queryKey: ["snapshot"] });
    },
  });
  const [lastApplied, setLastApplied] = useState<Partial<MutableSettings> | null>(null);

  useEffect(() => {
    if (!settings) return;
    const shown = draftRef.current;
    if (!shown) {
      setDraft(settings);
      return;
    }
    // A key the user is not editing whose server value differs from what
    // the controls show moved underneath us: follow the server and say so.
    const moved = MUTABLE_SETTINGS_KEYS.filter(
      (key) => !dirty.has(key) && shown[key] !== settings[key],
    );
    if (moved.length === 0) return;
    setDraft((prev) => {
      if (!prev) return settings;
      const next = { ...prev };
      moved.forEach((key) => {
        (next as Record<string, unknown>)[key] = settings[key];
      });
      return next;
    });
    setChangedOnServer(moved);
  }, [settings, dirty]);

  useEffect(() => {
    if (changedOnServer.length === 0) return;
    const handle = window.setTimeout(() => setChangedOnServer([]), CHANGED_ON_SERVER_HINT_MS);
    return () => window.clearTimeout(handle);
  }, [changedOnServer]);

  const edit = <K extends MutableKey>(key: K, value: MutableSettings[K]) => {
    setDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
    setDirty((prev) => {
      if (prev.has(key)) return prev;
      const next = new Set(prev);
      next.add(key);
      return next;
    });
  };

  useEffect(() => {
    if (!draft || !settings || dirty.size === 0) return;
    // Send only the keys the user moved, and only the mutable ones: the
    // draft carries the full settings payload, whose informational keys
    // would trip the server's all-or-nothing unknown_settings 400.
    const payload: Partial<MutableSettings> = {};
    dirty.forEach((key) => {
      if (draft[key] !== settings[key]) {
        (payload as Record<string, unknown>)[key] = draft[key];
      }
    });
    const keys = Object.keys(payload) as MutableKey[];
    if (keys.length === 0) {
      // Every edit landed back on the server's value; nothing is pending.
      setDirty(new Set());
      return;
    }
    const handle = window.setTimeout(() => {
      mutation.mutate(payload, {
        onSuccess: (res) => {
          setLastApplied(res.applied ?? payload);
          // A key stays dirty if the user moved it again while the request
          // was in flight; the next debounce sends the newer value.
          setDirty((prev) => {
            const next = new Set(prev);
            keys.forEach((key) => {
              if (draftRef.current?.[key] === payload[key]) next.delete(key);
            });
            return next;
          });
        },
      });
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  if (!draft) {
    return (
      <Card title="Defaults" subtitle="loading server settings...">
        <div className="text-sm text-[var(--text-muted)]">
          Settings will appear once the dashboard receives its first snapshot.
        </div>
      </Card>
    );
  }

  return (
    <Card
      title="Defaults"
      subtitle="server-side defaults applied to every chat completion"
      action={
        mutation.isPending ? (
          <span className="text-xs text-[var(--text-muted)]">applying…</span>
        ) : lastApplied ? (
          <span className="text-xs text-[var(--text-muted)]">
            applied · {Object.keys(lastApplied).join(", ")}
          </span>
        ) : undefined
      }
    >
      <div className="space-y-4">
        <NumberField
          label="depth"
          value={draft.depth}
          min={0}
          max={5}
          onChange={(v) => edit("depth", v)}
        />
        <NumberField
          label="temperature"
          value={draft.temperature}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => edit("temperature", v)}
        />
        <NumberField
          label="top_p"
          value={draft.top_p}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => edit("top_p", v)}
        />
        <NumberField
          label="top_k"
          value={draft.top_k}
          min={0}
          max={2000}
          step={1}
          onChange={(v) => edit("top_k", v)}
        />
        <NumberField
          label="presence_penalty"
          value={draft.presence_penalty ?? 0}
          min={0}
          max={2}
          step={0.05}
          onChange={(v) => edit("presence_penalty", v)}
          description="0 is exact (best for coding); 0.5-1.5 discourages repetition."
        />
        <NumberField
          label="stream_interval"
          value={draft.stream_interval}
          min={1}
          max={32}
          step={1}
          onChange={(v) => edit("stream_interval", v)}
        />
        <ToggleField
          label="enable_thinking"
          value={draft.enable_thinking}
          onChange={(v) => edit("enable_thinking", v)}
          description="When on, requests default to including reasoning content."
        />
        <SelectField
          label="reasoning_parser"
          value={draft.reasoning_parser}
          options={["qwen3", "poolside_v1", "none"]}
          onChange={(v) => edit("reasoning_parser", v)}
        />
        {changedOnServer.length > 0 ? (
          <div className="text-xs text-[var(--accent-warm)]">
            updated on the server: {changedOnServer.join(", ")}
          </div>
        ) : null}
        {mutation.isError ? (
          <div className="text-xs text-[var(--accent-hot)]">
            {String((mutation.error as Error).message)}
          </div>
        ) : null}
      </div>
    </Card>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  description,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (next: number) => void;
  description?: string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between text-xs text-[var(--text-muted)]">
        <span>{label}</span>
        <span className="tabular-nums text-[var(--text-primary)]">{Number(value).toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full mt-1 accent-[var(--accent)]"
      />
      {description ? (
        <div className="mt-0.5 text-xs text-[var(--text-muted)]">{description}</div>
      ) : null}
    </label>
  );
}

function ToggleField({
  label,
  value,
  onChange,
  description,
}: {
  label: string;
  value: boolean;
  onChange: (next: boolean) => void;
  description?: string;
}) {
  return (
    <label className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm text-[var(--text-primary)]">{label}</div>
        {description ? (
          <div className="text-xs text-[var(--text-muted)]">{description}</div>
        ) : null}
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`h-5 w-9 rounded-full transition-colors relative shrink-0 ${
          value ? "bg-[var(--accent)]" : "bg-[var(--border-soft)]"
        }`}
        aria-pressed={value}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            value ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (next: string) => void;
}) {
  return (
    <label className="block">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 w-full bg-[var(--bg-elevated)] border border-[var(--border-soft)] rounded px-2 py-1.5 text-sm text-[var(--text-primary)]"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function FixedSettingsCard({
  info,
  unavailable,
}: {
  info: LaunchInfo | null;
  unavailable: boolean;
}) {
  const fixed = info?.fixed;
  if (!fixed) {
    return (
      <Card title="Startup settings">
        <div className="text-sm text-[var(--text-muted)]">
          {unavailable
            ? "This server does not report how it was started."
            : "Loading how the server was started..."}
        </div>
      </Card>
    );
  }
  const flags = launchFlags(info?.argv ?? []);
  const address = fixed.host && fixed.port ? `${fixed.host}:${fixed.port}` : "—";
  return (
    <Card title="Startup settings" subtitle={`started with ${info?.launcher ?? "—"}`}>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        <FixedRow label="profile" value={fixed.profile ?? "—"} />
        <FixedRow label="model" value={fixed.model_id ?? "—"} detail={fixed.model ?? undefined} />
        <FixedRow label="address" value={address} />
        <FixedRow
          label="generation"
          value={`${fixed.generation_mode ?? "—"} · MTP ${fixed.load_mtp ? "loaded" : "off"}`}
        />
        <FixedRow label="API key" value={fixed.api_key_required ? "required" : "not required"} />
        <FixedRow
          label="context window"
          value={fixed.context_window ? fixed.context_window.toLocaleString() : "model default"}
        />
      </dl>
      {flags.length > 0 ? (
        <div className="mt-4">
          <div className="text-xs text-[var(--text-muted)] mb-1.5">flags</div>
          <div className="flex flex-wrap gap-1.5">
            {flags.map(({ flag, value }, idx) => (
              <span
                key={`${flag}-${idx}`}
                className="rounded border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--text-primary)] max-w-full truncate"
                title={value === null ? flag : `${flag} ${value}`}
              >
                {flag}
                {value === null ? null : (
                  <span className="text-[var(--text-muted)]"> {value}</span>
                )}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </Card>
  );
}

function FixedRow({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <>
      <dt className="text-xs text-[var(--text-muted)] pt-0.5">{label}</dt>
      <dd className="min-w-0 text-[var(--text-primary)]">
        <div className="truncate" title={value}>
          {value}
        </div>
        {detail && detail !== value ? (
          <div className="truncate font-mono text-[11px] text-[var(--text-muted)]" title={detail}>
            {detail}
          </div>
        ) : null}
      </dd>
    </>
  );
}

function RestartCommandCard({
  info,
  unavailable,
}: {
  info: LaunchInfo | null;
  unavailable: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const restart = info?.restart;
  const command = restart ? buildRestartCommand(restart) : null;
  useEffect(() => {
    if (!copied) return;
    const handle = window.setTimeout(() => setCopied(false), 2000);
    return () => window.clearTimeout(handle);
  }, [copied]);
  const copy = () => {
    if (!command || typeof navigator === "undefined") return;
    navigator.clipboard?.writeText(command).then(() => setCopied(true), () => undefined);
  };
  return (
    <Card title="Restart command" subtitle="These settings only change after a server restart.">
      {command && restart ? (
        <>
          <p className="text-xs text-[var(--text-muted)] mb-3">
            Starts the server again with the same command and environment it runs with now.
            To restart with a different setting, edit the copied command.
          </p>
          <pre className="rounded border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-2 font-mono text-xs text-[var(--text-primary)] overflow-x-auto max-h-[260px] whitespace-pre">
            {command}
          </pre>
          {commandHasRedactions(restart) ? (
            <p className="mt-2 text-xs text-[var(--accent-warm)]">
              Values shown as {REDACTED} are secret and hidden; fill them in before running.
            </p>
          ) : null}
          <button
            type="button"
            onClick={copy}
            className="mt-3 inline-flex items-center gap-1.5 text-xs text-[var(--accent-cool)] hover:text-[var(--accent)]"
          >
            <Copy className="size-3.5" />
            {copied ? "copied" : "copy restart command"}
          </button>
        </>
      ) : (
        <div className="text-sm text-[var(--text-muted)]">
          {unavailable
            ? "This server does not report how it was started. Restart it with the command you used to start it."
            : "Loading how the server was started..."}
        </div>
      )}
    </Card>
  );
}

function ServerParametersCard({ rows }: { rows: LaunchEnvRow[] }) {
  const [query, setQuery] = useState("");
  const shown = filterEnvRows(rows, query);
  return (
    <Card
      title="Server parameters"
      subtitle="MTPLX_*, MLX_* and PYTHONPATH as the server runs with them, and where each value comes from"
      action={
        <span className="text-xs text-[var(--text-muted)] tabular-nums">
          {shown.length === rows.length ? rows.length : `${shown.length} of ${rows.length}`}
        </span>
      }
    >
      <div className="flex flex-wrap items-center gap-3 mb-3">
        <label className="relative flex-1 min-w-[220px] max-w-md">
          <Search className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search name, value or source"
            aria-label="Search server parameters"
            className="w-full bg-[var(--bg-elevated)] border border-[var(--border-soft)] rounded pl-8 pr-2 py-1.5 text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
          />
        </label>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
          {(Object.keys(SOURCE_LABELS) as LaunchEnvSource[]).map((source) => (
            <span key={source} title={SOURCE_HINTS[source]} className="inline-flex items-center gap-1.5">
              <SourceBadge source={source} />
            </span>
          ))}
        </div>
      </div>
      {shown.length === 0 ? (
        <div className="text-sm text-[var(--text-muted)]">No parameter matches "{query}".</div>
      ) : (
        <div className="overflow-x-auto -mx-3 max-h-[480px] overflow-y-auto">
          <table className="min-w-full text-sm">
            <thead className="text-[10px] uppercase tracking-widest text-[var(--text-muted)] sticky top-0 bg-[var(--bg-card)]">
              <tr>
                <th className="px-3 py-2 font-medium text-left">name</th>
                <th className="px-3 py-2 font-medium text-left">value</th>
                <th className="px-3 py-2 font-medium text-left">source</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((row) => (
                <tr key={row.name} className="border-t border-[var(--border-soft)]">
                  <td className="px-3 py-1.5 font-mono text-xs text-[var(--text-primary)] whitespace-nowrap">
                    {row.name}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-xs text-[var(--text-primary)] break-all">
                    {row.value}
                    {row.source === "both" && row.profile_value !== undefined && row.profile_value !== row.value ? (
                      <div className="text-[11px] text-[var(--text-muted)]">
                        profile asked for {row.profile_value}
                      </div>
                    ) : null}
                    {row.source === "both" && row.launch_value !== undefined && row.launch_value !== row.value ? (
                      <div className="text-[11px] text-[var(--text-muted)]">
                        launch value {row.launch_value} replaced by the profile
                      </div>
                    ) : null}
                  </td>
                  <td className="px-3 py-1.5 whitespace-nowrap" title={SOURCE_HINTS[row.source]}>
                    <SourceBadge source={row.source} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

const SOURCE_TONE: Record<LaunchEnvSource, string> = {
  launch: "text-[var(--accent)] border-[var(--accent)]/40",
  profile: "text-[var(--text-muted)] border-[var(--border-soft)]",
  both: "text-[var(--accent-warm)] border-[var(--accent-warm)]/40",
  flag: "text-[var(--accent-cool)] border-[var(--accent-cool)]/40",
};

function SourceBadge({ source }: { source: LaunchEnvSource }) {
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[11px] ${SOURCE_TONE[source]}`}>
      {SOURCE_LABELS[source]}
    </span>
  );
}
function AdminActions() {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const clearAll = useMutation({
    mutationFn: () => api.postClearCache(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setConfirming(false);
    },
  });
  return (
    <Card title="Admin actions" subtitle="bank-wide controls">
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="inline-flex items-center gap-2 text-sm text-[var(--accent-hot)] hover:bg-[var(--accent-hot)]/10 rounded px-3 py-2 transition-colors"
      >
        <Trash className="size-4" />
        Clear all SessionBank entries
      </button>
      {confirming ? (
        <div className="mt-3 p-3 rounded-md border border-[var(--accent-hot)]/40 bg-[var(--accent-hot)]/5 text-sm text-[var(--text-primary)]">
          <p>
            Evict every cached prefix? Future requests will pay full prefill
            until the cache refills.
          </p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={() => clearAll.mutate()}
              disabled={clearAll.isPending}
              className="text-xs px-3 py-1 rounded bg-[var(--accent-hot)] text-white disabled:opacity-50"
            >
              {clearAll.isPending ? "Clearing..." : "Yes, clear cache"}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="text-xs px-3 py-1 rounded border border-[var(--border-soft)] text-[var(--text-muted)]"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
