import { useMemo, useState, type ChangeEvent } from "react";
import { Check, Copy, ShieldCheck, Terminal } from "lucide-react";
import { useDashboardStore } from "../state/store";
import { Card } from "./Card";

const EVERYDAY_COMMANDS = [
  {
    name: "start",
    command: "mtplx start",
    summary: "Interactive setup, then open the local chat surface.",
  },
  {
    name: "tune",
    command: "mtplx tune",
    summary: "Find the fastest AR, D1, D2, or D3 depth for this Mac.",
  },
  {
    name: "help",
    command: "mtplx help",
    summary: "Open detailed command, flag, and topic help.",
  },
  {
    name: "setup",
    command: "mtplx setup",
    summary: "Prepare local configuration and the model cache.",
  },
  {
    name: "quickstart",
    command: "mtplx quickstart",
    summary: "Start the local compatible API server.",
  },
  {
    name: "connect",
    command: "mtplx connect",
    summary: "Print local endpoint and client connection settings.",
  },
  {
    name: "ask",
    command: 'mtplx ask "<prompt>"',
    summary: "Ask the verified local model once and exit.",
  },
] as const;

const BENCH_PROFILES = [
  "stable",
  "performance-cold",
  "sustained",
  "turbo",
  "exact",
  "max-diagnostic",
  "native-mtp-60",
] as const;

type BenchProfile = (typeof BENCH_PROFILES)[number];
type GenerationMode = "mtp" | "ar";
type CopyState = "idle" | "copied" | "error";

export function NativeTab() {
  const modelId = useDashboardStore((s) => s.modelId);
  const [profile, setProfile] = useState<BenchProfile>("sustained");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("mtp");
  const [strict, setStrict] = useState(true);
  const [dryRun, setDryRun] = useState(false);
  const [stockAr, setStockAr] = useState(false);

  const inspectCommand = `mtplx inspect ${shellToken(modelId ?? "<model>")} --require-mtp --json`;
  const benchCommand = useMemo(
    () =>
      [
        "mtplx bench run",
        `--profile ${profile}`,
        `--generation-mode ${generationMode}`,
        generationMode === "ar" && stockAr ? "--stock-ar" : null,
        strict ? "--strict" : null,
        dryRun ? "--dry-run" : null,
        "--json",
      ]
        .filter((part): part is string => Boolean(part))
        .join(" "),
    [dryRun, generationMode, profile, stockAr, strict],
  );

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12">
        <Card bodyClassName="pt-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <span className="inline-flex size-10 shrink-0 items-center justify-center rounded-xl border border-[var(--accent)]/30 bg-[var(--accent)]/10 text-[var(--accent)]">
                <Terminal className="size-5" />
              </span>
              <div>
                <div className="text-base font-semibold text-[var(--text-primary)]">
                  Native MTPLX operations
                </div>
                <div className="mt-1 text-sm text-[var(--text-muted)]">
                  Native MTP speculative decoding on Apple Silicon.
                </div>
              </div>
            </div>
            <span className="self-start rounded-full border border-[var(--border-soft)] bg-[var(--bg-elevated)] px-3 py-1 text-[10px] uppercase tracking-widest text-[var(--text-muted)] sm:self-auto">
              read-only command surface
            </span>
          </div>
        </Card>
      </div>

      <div className="col-span-12">
        <Card
          title="Everyday commands"
          subtitle="Native entry points exposed without adding shell execution to the dashboard"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {EVERYDAY_COMMANDS.map((item) => (
              <CommandTile
                key={item.name}
                name={item.name}
                command={item.command}
                summary={item.summary}
              />
            ))}
          </div>
        </Card>
      </div>

      <div className="col-span-12">
        <Card
          title="Diagnostics and benchmarks"
          subtitle="Deep inspection plus a native benchmark command builder"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <CommandTile
              name="doctor"
              command="mtplx doctor --deep --json"
              summary="Run deep install, runtime, and integration checks with machine-readable output."
            />
            <CommandTile
              name="inspect"
              command={inspectCommand}
              summary="Require a valid native-MTP model contract before execution."
            />
          </div>

          <div className="mt-4 rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-4">
            <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="text-sm font-semibold text-[var(--text-primary)]">
                  Benchmark builder
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Select the product profile and compare native MTP with target-only AR.
                </div>
              </div>
              <span className="mt-1 text-[10px] uppercase tracking-widest text-[var(--text-muted)] sm:mt-0">
                bench run
              </span>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
              <SelectControl
                label="profile"
                value={profile}
                options={BENCH_PROFILES}
                onChange={(value) => setProfile(value as BenchProfile)}
              />
              <SelectControl
                label="generation mode"
                value={generationMode}
                options={["mtp", "ar"] as const}
                onChange={(value) => {
                  const next = value as GenerationMode;
                  setGenerationMode(next);
                  if (next !== "ar") setStockAr(false);
                }}
              />
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <FlagToggle label="--strict" checked={strict} onChange={setStrict} />
              <FlagToggle label="--dry-run" checked={dryRun} onChange={setDryRun} />
              <FlagToggle
                label="--stock-ar"
                checked={stockAr}
                onChange={setStockAr}
                disabled={generationMode !== "ar"}
              />
            </div>

            <CommandLine value={benchCommand} className="mt-4" />
            <p className="mt-2 text-xs text-[var(--text-muted)]">
              AR is target-only by default. Enable <code>--stock-ar</code> only when the
              MTP sidecar must remain unloaded for the comparison.
            </p>
          </div>
        </Card>
      </div>

      <div className="col-span-12">
        <Card
          title="Boundary"
          subtitle="What this dashboard surface intentionally does not do"
        >
          <div className="flex items-start gap-3 text-sm text-[var(--text-muted)]">
            <ShieldCheck className="mt-0.5 size-4 shrink-0 text-[var(--accent-cool)]" />
            <p>
              Commands are copied for execution in the local terminal. The dashboard
              does not spawn processes, publish models, bypass validation gates, or add
              third-party integration controls.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}

function CommandTile({
  name,
  command,
  summary,
}: {
  name: string;
  command: string;
  summary: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--border-soft)] bg-[var(--bg-elevated)] p-4">
      <div className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">
        {name}
      </div>
      <p className="mt-2 min-h-10 text-sm text-[var(--text-muted)]">{summary}</p>
      <CommandLine value={command} className="mt-3" compact />
    </div>
  );
}

function CommandLine({
  value,
  className = "",
  compact = false,
}: {
  value: string;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-2 rounded-lg border border-[var(--border-soft)] bg-[var(--bg-card)] ${
        compact ? "px-2.5 py-2" : "px-3 py-2.5"
      } ${className}`}
    >
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-xs text-[var(--text-primary)]">
        {value}
      </code>
      <CopyButton value={value} />
    </div>
  );
}

function CopyButton({ value }: { value: string }) {
  const [state, setState] = useState<CopyState>("idle");

  async function handleCopy() {
    try {
      await writeClipboard(value);
      setState("copied");
      window.setTimeout(() => setState("idle"), 1400);
    } catch {
      setState("error");
      window.setTimeout(() => setState("idle"), 1800);
    }
  }

  const label = state === "copied" ? "Copied" : state === "error" ? "Failed" : "Copy";

  return (
    <button
      type="button"
      onClick={() => void handleCopy()}
      className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[10px] uppercase tracking-wider text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]"
      aria-label={`Copy command: ${value}`}
    >
      {state === "copied" ? <Check className="size-3" /> : <Copy className="size-3" />}
      {label}
    </button>
  );
}

function SelectControl<T extends readonly string[]>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T[number];
  options: T;
  onChange: (value: T[number]) => void;
}) {
  return (
    <label className="block">
      <span className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
        {label}
      </span>
      <select
        value={value}
        onChange={(event: ChangeEvent<HTMLSelectElement>) =>
          onChange(event.target.value as T[number])
        }
        className="mt-1 w-full rounded-lg border border-[var(--border-soft)] bg-[var(--bg-card)] px-3 py-2 text-sm text-[var(--text-primary)]"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function FlagToggle({
  label,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-colors ${
        checked
          ? "border-[var(--accent)]/50 bg-[var(--accent)]/10 text-[var(--accent)]"
          : "border-[var(--border-soft)] bg-[var(--bg-card)] text-[var(--text-muted)]"
      } ${disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer"}`}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.target.checked)}
        className="sr-only"
      />
      <span
        className={`size-1.5 rounded-full ${
          checked ? "bg-[var(--accent)]" : "bg-[var(--text-muted)]"
        }`}
      />
      <code>{label}</code>
    </label>
  );
}

function shellToken(value: string): string {
  if (/^<[^>]+>$/.test(value) || /^[A-Za-z0-9_./:@+-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", `'"'"'`)}'`;
}

async function writeClipboard(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard unavailable");
}
