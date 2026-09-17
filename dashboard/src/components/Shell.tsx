import { useState, type ReactNode } from "react";
import {
  Activity,
  Cpu,
  Database,
  HardDrive,
  Layers,
  Network,
  Settings,
  Sliders,
  Thermometer,
} from "lucide-react";
import { cn } from "../lib/utils";
import { useDashboardStore } from "../state/store";
import { ConnectionDot } from "./ConnectionDot";
import { ReconnectBanner } from "./ReconnectBanner";
import { SessionFilterDropdown } from "./SessionFilterDropdown";
import { SoundToggle } from "./SoundToggle";
import { ThemeToggle } from "./ThemeToggle";

export type TabId =
  | "overview"
  | "speculative"
  | "cache"
  | "memory"
  | "thermal"
  | "requests"
  | "systems"
  | "settings";

const TABS: { id: TabId; label: string; icon: typeof Activity }[] = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "speculative", label: "Speculative", icon: Layers },
  { id: "cache", label: "Cache", icon: Database },
  { id: "memory", label: "Memory", icon: HardDrive },
  { id: "thermal", label: "Thermal", icon: Thermometer },
  { id: "requests", label: "Requests", icon: Sliders },
  { id: "systems", label: "Systems", icon: Network },
  { id: "settings", label: "Settings", icon: Settings },
];

type ShellProps = {
  active: TabId;
  onSelect: (tab: TabId) => void;
  children: ReactNode;
  bottomBar?: ReactNode;
};

export function Shell({ active, onSelect, children, bottomBar }: ShellProps) {
  const modelId = useDashboardStore((s) => s.modelId);
  const profileName = useDashboardStore((s) => s.profileName);
  const activeRequests = useDashboardStore((s) => s.inFlight.length);
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-dvh flex flex-col bg-[var(--bg-canvas)] text-[var(--text-primary)]">
      <ReconnectBanner />
      <TopBar
        modelId={modelId}
        profileName={profileName}
        activeRequests={activeRequests}
      />
      <div className="flex-1 flex">
        <LeftRail
          active={active}
          onSelect={onSelect}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
        />
        <main className="flex-1 min-w-0 px-6 lg:px-8 py-6 lg:py-8 pb-24 overflow-x-hidden">
          {children}
        </main>
      </div>
      {bottomBar ? (
        <div className="fixed bottom-0 left-0 right-0 z-40 border-t border-[var(--border-soft)] bg-[var(--bg-elevated)]/90 backdrop-blur">
          {bottomBar}
        </div>
      ) : null}
    </div>
  );
}

function TopBar({
  modelId,
  profileName,
  activeRequests,
}: {
  modelId: string | null;
  profileName: string | null;
  activeRequests: number;
}) {
  return (
    <div className="h-14 px-4 lg:px-6 flex items-center justify-between border-b border-[var(--border-soft)] bg-[var(--bg-elevated)]">
      <div className="flex items-center gap-3 min-w-0">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-[var(--accent)] text-black font-bold text-sm">
          M
        </span>
        <div className="hidden sm:block leading-none">
          <div className="text-sm font-semibold">MTPLX</div>
          <div className="text-[10px] uppercase tracking-widest text-[var(--text-muted)]">
            Live Dashboard
          </div>
        </div>
        <div className="hidden md:flex items-center gap-2 ml-4 text-xs text-[var(--text-muted)] min-w-0">
          <Cpu className="size-3.5 shrink-0" />
          <span className="truncate max-w-[280px]">{modelId ?? "—"}</span>
          {profileName ? (
            <span className="px-2 py-0.5 rounded-full border border-[var(--border-soft)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
              {profileName}
            </span>
          ) : null}
          {activeRequests > 0 ? (
            <span className="px-2 py-0.5 rounded-full bg-[var(--accent)]/15 text-[var(--accent)] text-[10px] uppercase tracking-wider">
              {activeRequests} in flight
            </span>
          ) : null}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <SessionFilterDropdown />
        <SoundToggle />
        <ThemeToggle />
        <ConnectionDot />
      </div>
    </div>
  );
}

function LeftRail({
  active,
  onSelect,
  collapsed,
  setCollapsed,
}: {
  active: TabId;
  onSelect: (tab: TabId) => void;
  collapsed: boolean;
  setCollapsed: (next: boolean) => void;
}) {
  return (
    <nav
      className={cn(
        "shrink-0 border-r border-[var(--border-soft)] bg-[var(--bg-elevated)] flex flex-col py-3 transition-[width]",
        collapsed ? "w-14" : "w-56",
      )}
    >
      <div className="px-2 flex flex-col gap-1">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = active === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onSelect(tab.id)}
              className={cn(
                "group w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left text-sm transition-colors",
                isActive
                  ? "bg-[var(--bg-card)] text-[var(--text-primary)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card)]/60",
              )}
              title={collapsed ? tab.label : undefined}
            >
              <Icon className="size-4 shrink-0" />
              {!collapsed ? <span className="truncate">{tab.label}</span> : null}
              {isActive ? (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
              ) : null}
            </button>
          );
        })}
      </div>
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="mt-auto mx-2 mb-2 text-[10px] uppercase tracking-widest text-[var(--text-muted)] hover:text-[var(--text-primary)] py-2"
      >
        {collapsed ? "Expand" : "Collapse"}
      </button>
    </nav>
  );
}
