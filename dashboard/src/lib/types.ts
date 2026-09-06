// Type definitions for the MTPLX dashboard surface. These shapes mirror
// the Python-side `_mtplx_dashboard_snapshot` payload, `PUBLIC_MTPLX_STATS_KEYS`,
// `SessionBank.to_dict()`, and `EngineSessionManager.list_sessions()`.

export type MetricsLatest = {
  // ---- volume ----
  prompt_tokens?: number | null;
  cached_tokens?: number | null;
  new_prefill_tokens?: number | null;
  completion_tokens?: number | null;
  generated_tokens?: number | null;
  // ---- timing ----
  elapsed_s?: number | null;
  request_elapsed_s?: number | null;
  prompt_eval_time_s?: number | null;
  decode_elapsed_s?: number | null;
  ttft_s?: number | null;
  // ---- rates ----
  tok_s?: number | null;
  decode_tok_s?: number | null;
  prefill_tok_s?: number | null;
  request_tok_s?: number | null;
  server_tok_s?: number | null;
  sliding_decode_tok_s_first_32?: number | null;
  sliding_decode_tok_s_first_64?: number | null;
  sliding_decode_tok_s_last_32?: number | null;
  sliding_decode_tok_s_last_64?: number | null;
  // ---- speculative-decoding ----
  mtp_depth?: number | null;
  speculative_depth?: number | null;
  verify_calls?: number | null;
  accepted_drafts?: number | null;
  rejected_drafts?: number | null;
  drafted_tokens?: number | null;
  accepted_by_depth?: number[] | null;
  drafted_by_depth?: number[] | null;
  mean_accept_probability_by_depth?: number[] | null;
  correction_tokens?: number | null;
  bonus_tokens?: number | null;
  // ---- verify-decomposition (added by the dashboard) ----
  verify_time_s?: number | null;
  draft_time_s?: number | null;
  accept_time_s?: number | null;
  repair_time_s?: number | null;
  target_forward_time_s?: number | null;
  verify_forward_time_s?: number | null;
  verify_eval_time_s?: number | null;
  verify_logits_eval_time_s?: number | null;
  verify_hidden_eval_time_s?: number | null;
  verify_target_distribution_time_s?: number | null;
  verify_eval_unattributed_time_s?: number | null;
  snapshot_time_s?: number | null;
  commit_time_s?: number | null;
  capture_commit_time_s?: number | null;
  rollback_time_s?: number | null;
  graphbank?: Record<string, unknown> | null;
  repair_time_by_reject_depth_s?: Record<string, number> | null;
  // ---- cache ----
  session_cache_hit?: boolean | null;
  cache_miss_reason?: string | null;
  session_restore_mode?: string | null;
  session_id?: string | null;
  context_len?: number | null;
  lock_wait_time_s?: number | null;
  // ---- memory ----
  peak_memory_bytes?: number | null;
  // ---- reasoning ----
  reasoning_reentries?: number | null;
  reasoning_tokens?: number | null;
  answer_tokens?: number | null;
  // ---- server caps ----
  request_max_tokens?: number | null;
  server_max_response_tokens?: number | null;
  effective_max_tokens?: number | null;
  remaining_context_tokens?: number | null;
  server_cap_applied?: boolean | null;
  context_cap_applied?: boolean | null;
};

export type RollingTPSPoint = {
  t: number;
  tok_s: number;
  session_id: string | null;
};

export type RollingMetricsSnapshot = {
  window_s: number;
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  p50: number | null;
  p95: number | null;
  history: RollingTPSPoint[];
  live_history: RollingTPSPoint[];
  max_per_session: Record<string, number>;
  sticky_all_time_max: number;
  sticky_all_time_max_when_s: number;
  sticky_all_time_max_session_id: string | null;
  all_time_min: number | null;
};

export type LifetimeSnapshot = {
  started_at_s: number;
  uptime_s: number;
  prompt_tokens_total: number;
  completion_tokens_total: number;
  cached_tokens_total: number;
  tokens_total: number;
  requests_total: number;
  cancelled_total: number;
};

export type PrefillState = {
  phase: "started" | "chunk" | "completed";
  tokens_done?: number;
  tokens_total: number;
  cached_tokens?: number;
  new_prefill_tokens?: number;
  elapsed_s?: number;
  prefill_tok_s?: number | null;
  chunk_size?: number;
  cache_hit?: boolean;
  started_s?: number;
};

export type InFlightSnapshot = {
  request_id: string;
  started_s: number;
  age_s: number;
  session_id: string | null;
  model: string | null;
  prompt_preview: string;
  prompt_tokens: number | null;
  last_progress: Record<string, unknown>;
  prefill_state: PrefillState | null;
  cancelled: boolean;
};

export type SessionBankPrefix = {
  session_id: string;
  prefix_len: number;
  hits: number;
  nbytes: number;
  created_at_s: number;
  last_access_s: number;
  policy_fingerprint?: string | null;
  has_live_ref?: boolean;
};

export type EvictionEntry = {
  session_id?: string;
  reason: string;
  when_s: number;
};

export type SessionBank = {
  max_entries: number;
  total_nbytes: number;
  last_miss_reason: string | null;
  prefixes: SessionBankPrefix[];
  eviction_log: EvictionEntry[];
};

export type SessionRow = {
  session_id: string;
  prefix_len: number;
  bytes: number;
  in_flight?: boolean;
  last_finish_reason?: string | null;
  last_cache_miss_reason?: string | null;
  boundaries?: number[];
  last_access_s: number;
  age_s?: number;
};

export type SessionsPayload = {
  sessions: SessionRow[];
  count: number;
  session_bank?: SessionBank;
};

export type MemSnapshot = {
  ok: boolean;
  active_memory_bytes?: number | null;
  cache_memory_bytes?: number | null;
  peak_memory_bytes?: number | null;
  error?: string;
};

export type ThermalFan = {
  rpm: number;
  target_rpm?: number | null;
  actual_rpm?: number | null;
  max_capacity_rpm?: number | null;
  mode?: string | null;
};

export type ThermalSnapshot = {
  ok: boolean;
  min_rpm: number | null;
  max_rpm: number | null;
  fans: ThermalFan[];
};

export type MutableSettings = {
  depth: number;
  temperature: number;
  top_p: number;
  top_k: number;
  presence_penalty: number;
  max_response_tokens: number | null;
  stream_interval: number;
  enable_thinking: boolean;
  reasoning_parser: string;
};

export type MachineInfo = {
  chip: string | null;
  machine_model: string | null;
  unified_memory_bytes: number | null;
};

export type ProfileInfo = {
  name: string;
  [key: string]: unknown;
};

export type HealthPayload = {
  ok: boolean;
  model: string;
  model_path: string;
  generation_mode: string;
  load_mtp: boolean;
  mtp_enabled: boolean;
  depth: number;
  profile: ProfileInfo;
  context_window: number;
  max_response_tokens: number | null;
  warmup: Record<string, unknown>;
  foreground_active: number;
  active_requests: number;
  last_request_started_at: number;
  requests_completed: number;
  last_request_at: number;
  reasoning_parser: string;
  load_time_s?: number;
  machine_model: string | null;
  unified_memory_bytes: number | null;
  [key: string]: unknown;
};

export type DashboardSnapshot = {
  ts: number;
  model_id: string;
  profile: ProfileInfo;
  context_window: number;
  active_requests: number;
  in_flight: InFlightSnapshot[];
  latest: MetricsLatest | null;
  recent: MetricsLatest[];
  rolling: RollingMetricsSnapshot;
  lifetime: LifetimeSnapshot;
  sessions: SessionsPayload;
  session_bank: SessionBank;
  mem: MemSnapshot;
  thermal: ThermalSnapshot | null;
  thermal_when_s: number;
  settings: MutableSettings;
  machine: MachineInfo;
  uptime_s: number;
};

export type ProgressEvent = {
  kind: "progress";
  when_s: number;
  request_id: string;
  progress: {
    completion_tokens: number;
    decode_started_s?: number;
    decode_tok_s?: number | null;
    session_id?: string | null;
    request_id?: string;
  };
};

export type CompletedEvent = {
  kind: "completed";
  when_s: number;
  envelope: MetricsLatest;
};

export type NewMaxTPSEvent = {
  kind: "new_max_tps";
  when_s: number;
  tok_s: number;
  session_id: string | null;
};

export type ThermalEvent = {
  kind: "thermal";
  when_s: number;
  thermal: ThermalSnapshot;
};

export type PrefillEvent = {
  kind: "prefill";
  when_s: number;
  request_id: string;
  session_id: string | null;
  phase: "started" | "chunk" | "completed";
  tokens_done?: number;
  tokens_total: number;
  cached_tokens?: number;
  new_prefill_tokens?: number;
  elapsed_s?: number;
  prefill_tok_s?: number | null;
  chunk_size?: number;
  cache_hit?: boolean;
  started_s?: number;
};

export type SnapshotEvent = {
  kind: "snapshot";
  // Backend sends the whole snapshot under this kind
} & DashboardSnapshot;

export type BusEvent =
  | ProgressEvent
  | CompletedEvent
  | NewMaxTPSEvent
  | ThermalEvent
  | PrefillEvent
  | SnapshotEvent;

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "failed"
  | "unauthorized";

export type PrefillRow = {
  t: number;
  session_id: string | null;
  prompt_tokens: number;
  cached_tokens: number;
  new_prefill_tokens: number;
  prompt_eval_time_s: number;
  prefill_tok_s: number | null;
  ttft_s: number | null;
  session_cache_hit: boolean;
  cache_miss_reason: string | null;
  context_len: number;
  model_id: string;
};

export type PrefillHistoryPayload = {
  capacity: number;
  history: PrefillRow[];
};

export type RuntimeSystemState = {
  available?: boolean;
  enabled?: boolean;
  wired?: boolean;
  [key: string]: unknown;
};

export type RuntimeSystemContract = {
  revision: number;
  updated_at_s: number;
  status: RuntimeSystemState;
};

export type RuntimeSystemsPayload = {
  ts: number;
  revision: number;
  updated_at_s: number;
  system_count: number;
  systems: Record<string, RuntimeSystemContract>;
};
