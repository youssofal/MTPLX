import { create } from "zustand";
import type {
  BusEvent,
  ConnectionState,
  DashboardSnapshot,
  InFlightSnapshot,
  LifetimeSnapshot,
  MachineInfo,
  MemSnapshot,
  MetricsLatest,
  MutableSettings,
  PrefillState,
  RollingMetricsSnapshot,
  RollingTPSPoint,
  SessionBank,
  SessionsPayload,
  ThermalSnapshot,
} from "../lib/types";

type ThemeName = "hippo" | "river" | "light" | "mono";

export type DashboardStore = {
  // ---- live state slices ----
  snapshot: DashboardSnapshot | null;
  latest: MetricsLatest | null;
  recent: MetricsLatest[];
  rolling: RollingMetricsSnapshot | null;
  lifetime: LifetimeSnapshot | null;
  inFlight: InFlightSnapshot[];
  sessionBank: SessionBank | null;
  sessions: SessionsPayload | null;
  mem: MemSnapshot | null;
  thermal: ThermalSnapshot | null;
  thermalWhenS: number;
  settings: MutableSettings | null;
  modelId: string | null;
  profileName: string | null;
  contextWindow: number | null;
  machine: MachineInfo | null;
  uptimeS: number;
  // Decode speed of the request generating right now, fed by progress
  // events. null whenever nothing is in flight: a finished request's speed
  // lives in `latest`, and must never be shown as current.
  liveTokS: number | null;
  liveProgressByRequest: Record<string, BusEvent>;
  // request_id -> active prefill state; entries removed on `completed`.
  activePrefillByRequest: Record<string, PrefillState & { request_id: string; session_id: string | null }>;
  lastCompletedPrefill: (PrefillState & { request_id: string; session_id: string | null; when_s: number }) | null;
  newMaxTPSEvent: { tok_s: number; when_s: number; session_id: string | null } | null;

  // ---- connection ----
  connection: ConnectionState;
  reconnectAttempts: number;
  lastSnapshotAtMs: number | null;
  // The server answered 401: it runs with an API key and this browser has no
  // valid session cookie. The page shows a sign-in state instead of
  // reporting a lost connection.
  authRequired: boolean;

  // ---- UI ----
  sessionFilter: string | null;
  theme: ThemeName;
  pauseStream: boolean;
  soundEnabled: boolean;

  // ---- actions ----
  applySnapshot: (snapshot: DashboardSnapshot) => void;
  applyEvent: (event: BusEvent) => void;
  // Adopt the mutable settings a settings write returned, ahead of the next
  // snapshot, so the controls reflect the server's post-write values at once.
  mergeSettings: (patch: Partial<MutableSettings>) => void;
  setConnection: (state: ConnectionState) => void;
  setAuthRequired: (required: boolean) => void;
  setSessionFilter: (sessionId: string | null) => void;
  setTheme: (theme: ThemeName) => void;
  cycleTheme: () => void;
  togglePauseStream: () => void;
  toggleSound: () => void;
  consumeNewMaxTPS: () => void;
};

const THEME_KEY = "mtplx.dashboard.theme";

function readPersistedTheme(): ThemeName {
  if (typeof window === "undefined") return "hippo";
  const raw = window.localStorage.getItem(THEME_KEY);
  if (raw === "hippo" || raw === "river" || raw === "light" || raw === "mono") {
    return raw;
  }
  return "hippo";
}

function persistTheme(theme: ThemeName): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
    window.document.documentElement.setAttribute("data-theme", theme);
  } catch {
    // ignore
  }
}

const THEME_ORDER: ThemeName[] = ["hippo", "river", "light", "mono"];

export const useDashboardStore = create<DashboardStore>((set, get) => ({
  snapshot: null,
  latest: null,
  recent: [],
  rolling: null,
  lifetime: null,
  inFlight: [],
  sessionBank: null,
  sessions: null,
  mem: null,
  thermal: null,
  thermalWhenS: 0,
  settings: null,
  modelId: null,
  profileName: null,
  contextWindow: null,
  machine: null,
  uptimeS: 0,
  liveTokS: null,
  liveProgressByRequest: {},
  activePrefillByRequest: {},
  lastCompletedPrefill: null,
  newMaxTPSEvent: null,

  connection: "idle",
  reconnectAttempts: 0,
  lastSnapshotAtMs: null,
  authRequired: false,

  sessionFilter: null,
  theme: readPersistedTheme(),
  pauseStream: false,
  soundEnabled: false,

  applySnapshot: (snapshot) => {
    if (get().pauseStream) return;
    // Rebuild active prefill map from the snapshot's in_flight handles
    // so polled state survives an SSE reconnect.
    const activePrefillByRequest: DashboardStore["activePrefillByRequest"] = {};
    (snapshot.in_flight ?? []).forEach((h) => {
      if (h.prefill_state) {
        activePrefillByRequest[h.request_id] = {
          ...h.prefill_state,
          request_id: h.request_id,
          session_id: h.session_id,
        };
      }
    });
    const inFlight = snapshot.in_flight ?? [];
    set((state) => ({
      snapshot,
      latest: snapshot.latest,
      recent: snapshot.recent ?? [],
      rolling: snapshot.rolling,
      lifetime: snapshot.lifetime,
      inFlight,
      sessionBank: snapshot.session_bank ?? null,
      sessions: snapshot.sessions ?? null,
      mem: snapshot.mem,
      thermal: snapshot.thermal,
      thermalWhenS: snapshot.thermal_when_s,
      settings: snapshot.settings,
      modelId: snapshot.model_id,
      profileName: snapshot.profile?.name ?? null,
      contextWindow: snapshot.context_window,
      machine: snapshot.machine,
      uptimeS: snapshot.uptime_s,
      activePrefillByRequest,
      // A snapshot's `latest` is the last completed request, never the one
      // decoding now; only progress events carry a current speed. Idle
      // server, no live reading.
      liveTokS: inFlight.length > 0 ? state.liveTokS : null,
      lastSnapshotAtMs: Date.now(),
    }));
  },

  applyEvent: (event) => {
    if (get().pauseStream) return;
    switch (event.kind) {
      case "progress": {
        const tokS = event.progress?.decode_tok_s;
        set((state) => ({
          liveTokS: typeof tokS === "number" && tokS > 0 ? tokS : state.liveTokS,
          liveProgressByRequest: {
            ...state.liveProgressByRequest,
            [event.request_id]: event,
          },
        }));
        break;
      }
      case "completed": {
        // The request is over: its speed now lives in `latest`, so the live
        // reading clears. Generation is serialized; if another request is
        // decoding, its next progress event restores the reading.
        set((state) => ({
          latest: event.envelope ?? state.latest,
          liveTokS: null,
        }));
        break;
      }
      case "new_max_tps": {
        set({
          newMaxTPSEvent: {
            tok_s: event.tok_s,
            when_s: event.when_s,
            session_id: event.session_id,
          },
        });
        break;
      }
      case "thermal": {
        set({ thermal: event.thermal, thermalWhenS: event.when_s });
        break;
      }
      case "prefill": {
        const key = event.request_id;
        const baseState = {
          phase: event.phase,
          tokens_done: event.tokens_done,
          tokens_total: event.tokens_total,
          cached_tokens: event.cached_tokens,
          new_prefill_tokens: event.new_prefill_tokens,
          elapsed_s: event.elapsed_s,
          prefill_tok_s: event.prefill_tok_s,
          chunk_size: event.chunk_size,
          cache_hit: event.cache_hit,
          started_s: event.started_s,
          request_id: key,
          session_id: event.session_id,
        };
        if (event.phase === "completed") {
          set((state) => {
            const next = { ...state.activePrefillByRequest };
            delete next[key];
            return {
              activePrefillByRequest: next,
              lastCompletedPrefill: { ...baseState, when_s: event.when_s },
            };
          });
        } else {
          set((state) => ({
            activePrefillByRequest: {
              ...state.activePrefillByRequest,
              [key]: baseState,
            },
          }));
        }
        break;
      }
      case "snapshot": {
        // Backend may push a periodic snapshot via this kind; fall through.
        get().applySnapshot(event);
        break;
      }
      default:
        break;
    }
  },

  mergeSettings: (patch) =>
    set((state) => (state.settings ? { settings: { ...state.settings, ...patch } } : {})),

  setConnection: (state) => {
    set((prev) => ({
      connection: state,
      reconnectAttempts: state === "reconnecting" ? prev.reconnectAttempts + 1 : 0,
    }));
  },

  setAuthRequired: (required) => set({ authRequired: required }),

  setSessionFilter: (sessionId) => set({ sessionFilter: sessionId }),

  setTheme: (theme) => {
    persistTheme(theme);
    set({ theme });
  },

  cycleTheme: () => {
    const current = get().theme;
    const next = THEME_ORDER[(THEME_ORDER.indexOf(current) + 1) % THEME_ORDER.length];
    persistTheme(next);
    set({ theme: next });
  },

  togglePauseStream: () => set((s) => ({ pauseStream: !s.pauseStream })),
  toggleSound: () => set((s) => ({ soundEnabled: !s.soundEnabled })),
  consumeNewMaxTPS: () => set({ newMaxTPSEvent: null }),
}));

// Hydrate the document with the persisted theme at module-eval time so the
// first paint already wears the right colors.
if (typeof window !== "undefined") {
  window.document.documentElement.setAttribute("data-theme", readPersistedTheme());
}

// Selectors -----------------------------------------------------------------
//
// Selectors that derive new arrays/objects on every call would cause an
// infinite render loop with Zustand's default `Object.is` equality check
// (a fresh reference on every selector run looks like "state changed",
// which schedules a re-render, which re-runs the selector, which yields
// another fresh reference, ad infinitum). These hooks use `useShallow`
// so the equality check sees structurally equal arrays as unchanged.

import { useShallow } from "zustand/react/shallow";

export function useFilteredSessionIds(): string[] {
  return useDashboardStore(
    useShallow((s) => {
      const ids = new Set<string>();
      s.rolling?.history.forEach((p) => {
        if (p.session_id) ids.add(p.session_id);
      });
      s.inFlight.forEach((h) => {
        if (h.session_id) ids.add(h.session_id);
      });
      return Array.from(ids).sort();
    }),
  );
}

export function useFilteredHistory(): RollingTPSPoint[] {
  return useDashboardStore(
    useShallow((s) => {
      if (!s.rolling) return [];
      const filter = s.sessionFilter;
      if (!filter) return s.rolling.history;
      return s.rolling.history.filter((p) => p.session_id === filter);
    }),
  );
}

export function useFilteredLiveHistory(): RollingTPSPoint[] {
  return useDashboardStore(
    useShallow((s) => {
      if (!s.rolling) return [];
      const live = s.rolling.live_history ?? [];
      const filter = s.sessionFilter;
      if (!filter) return live;
      return live.filter((p) => p.session_id === filter);
    }),
  );
}

export function useFilteredRecent(): MetricsLatest[] {
  return useDashboardStore(
    useShallow((s) => {
      if (!s.sessionFilter) return s.recent;
      return s.recent.filter((row) => row.session_id === s.sessionFilter);
    }),
  );
}

export type ActivePrefillStateView =
  | {
      active: true;
      request_id: string;
      session_id: string | null;
      tokens_done: number;
      tokens_total: number;
      cached_tokens: number;
      elapsed_s: number;
      prefill_tok_s: number | null;
      pct: number;
      eta_s: number | null;
    }
  | { active: false };

export function usePrimaryActivePrefill(): ActivePrefillStateView {
  return useDashboardStore(
    useShallow((s): ActivePrefillStateView => {
      const entries = Object.values(s.activePrefillByRequest);
      if (entries.length === 0) return { active: false };
      // Prefer the freshest one (most-recently-updated). For now the live
      // map only has one entry typically because generation is serialized,
      // but be defensive.
      const winner = entries.reduce((a, b) =>
        (b.elapsed_s ?? 0) > (a.elapsed_s ?? 0) ? b : a,
      );
      const total = Number(winner.tokens_total ?? 0);
      const done = Number(winner.tokens_done ?? 0);
      const elapsed = Number(winner.elapsed_s ?? 0);
      const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
      const tokS =
        typeof winner.prefill_tok_s === "number" && winner.prefill_tok_s > 0
          ? winner.prefill_tok_s
          : done > 0 && elapsed > 0
            ? done / elapsed
            : null;
      const remaining = Math.max(0, total - done);
      const etaS = tokS && tokS > 0 && remaining > 0 ? remaining / tokS : null;
      return {
        active: true,
        request_id: winner.request_id,
        session_id: winner.session_id,
        tokens_done: done,
        tokens_total: total,
        cached_tokens: Number(winner.cached_tokens ?? 0),
        elapsed_s: elapsed,
        prefill_tok_s: tokS,
        pct,
        eta_s: etaS,
      };
    }),
  );
}
