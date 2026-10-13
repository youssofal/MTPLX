export type WaterfallMode = "request" | "round";

export type WaterfallScale = {
  /** Mode actually shown; "request" when per-round was asked but the round count is unknown. */
  mode: WaterfallMode;
  /** Verify rounds of the request (its `verify_calls`), or 0 when missing. */
  rounds: number;
  /** Divide each per-request total by this to get the shown value. */
  divisor: number;
};

/**
 * The request log carries verify times summed over every verify round of the
 * request. Per-round values divide those totals by the request's
 * `verify_calls`; without a positive round count the totals are shown as is.
 */
export function waterfallScale(
  requested: WaterfallMode,
  verifyCalls: number | null | undefined,
): WaterfallScale {
  const rounds = Number(verifyCalls);
  const known = Number.isFinite(rounds) && rounds > 0 ? Math.floor(rounds) : 0;
  if (requested === "round" && known > 0) return { mode: "round", rounds: known, divisor: known };
  return { mode: "request", rounds: known, divisor: 1 };
}
