import type {
  DashboardSnapshot,
  HealthPayload,
  MutableSettings,
  PrefillHistoryPayload,
  RuntimeSystemsPayload,
  SessionsPayload,
} from "./types";
import { useDashboardStore } from "../state/store";

const BASE = "";  // same-origin; Vite's dev proxy handles `/v1`, `/admin`, `/health`, `/metrics`.

// The server answered 401: the browser has no valid session cookie for a
// server started with an API key. Every caller funnels through here so the
// page shows one sign-in state instead of a dozen unrelated errors.
export class UnauthorizedError extends Error {
  readonly status = 401;
  constructor(path: string) {
    super(`401 Unauthorized: ${path}`);
    this.name = "UnauthorizedError";
  }
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (response.status === 401) {
    useDashboardStore.getState().setAuthRequired(true);
    throw new UnauthorizedError(path);
  }
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${text || path}`);
  }
  return response.json() as Promise<T>;
}

// Probe used when the event stream drops: a 401 means the session cookie is
// missing or expired (sign in), anything else that answers means the server
// is up (reconnect), a network failure means it is not reachable.
export async function probeServer(): Promise<"ok" | "unauthorized" | "unreachable"> {
  try {
    const response = await fetch(`${BASE}/health`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    return response.status === 401 ? "unauthorized" : "ok";
  } catch {
    return "unreachable";
  }
}

// Exchange the API key for the browser session cookie the server sets on
// this same-origin POST; the key never goes into a URL. A wrong key throws
// UnauthorizedError.
export async function browserSignIn(apiKey: string): Promise<void> {
  const response = await fetch(`${BASE}/mtplx/browser-auth`, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey, next: "/dashboard/" }),
  });
  if (response.status === 401) throw new UnauthorizedError("/mtplx/browser-auth");
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${text || "sign-in failed"}`);
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return getJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
}

export const api = {
  getHealth: () => getJson<HealthPayload>("/health"),
  getMetrics: () =>
    getJson<{ latest: DashboardSnapshot["latest"]; recent: DashboardSnapshot["recent"] }>(
      "/metrics",
    ),
  getSessions: () => getJson<SessionsPayload>("/admin/sessions"),
  getPrefillHistory: () => getJson<PrefillHistoryPayload>("/v1/mtplx/prefill_history"),
  getSnapshot: () => getJson<DashboardSnapshot>("/v1/mtplx/snapshot"),
  getSystems: () => getJson<RuntimeSystemsPayload>("/v1/mtplx/systems"),
  // The response is the server's settings payload after the write: the
  // mutable keys at the top level (already clamped/normalised), plus
  // `applied` listing what this write changed.
  postSettings: (payload: Partial<MutableSettings>) =>
    postJson<MutableSettings & { ok: boolean; applied?: Partial<MutableSettings> }>(
      "/v1/mtplx/settings",
      payload,
    ),
  postCancel: (requestId: string) =>
    postJson<{ ok: boolean; cancelled: boolean; active_requests: number }>(
      `/v1/mtplx/cancel/${encodeURIComponent(requestId)}`,
      {},
    ),
  postClearSession: (sessionId: string) =>
    postJson<Record<string, unknown>>(
      `/admin/sessions/${encodeURIComponent(sessionId)}/clear`,
      {},
    ),
  postClearCache: () => postJson<Record<string, unknown>>("/admin/cache/clear", {}),
};

export type Api = typeof api;
