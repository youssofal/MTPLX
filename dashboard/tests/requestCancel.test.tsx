import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { Row } from "../src/components/RequestLogTable";
import { cancellationOf, countsTowardHitRate } from "../src/lib/requestCancel";
import type { MetricsLatest } from "../src/lib/types";

// The receipt of a streamed turn the client closed after 180 s, after its
// prompt was restored from RAM.
const cancelledHit: MetricsLatest = {
  session_id: "c3ad535e-27bb-42b1-98d5-9e95ceaa8b6b",
  prompt_tokens: 92819,
  cached_tokens: 92636,
  completion_tokens: 11968,
  decode_tok_s: 67.9,
  ttft_s: 2.1,
  request_cancelled: true,
  cancellation_reason: "client_disconnected",
  cancellation_elapsed_s: 180.0005,
  stream_cancelled_by_client: true,
  session_cache_hit: true,
  cache_source: "ram",
  cache_miss_reason: null,
};

const html = (row: MetricsLatest) =>
  renderToStaticMarkup(
    <table>
      <tbody>
        <Row row={row} isOpen={false} onToggle={() => {}} />
      </tbody>
    </table>,
  );

describe("cancellationOf", () => {
  test("names the reason and the duration, and keeps the cache outcome", () => {
    const c = cancellationOf(cancelledHit);
    expect(c?.label).toBe("cancelled · client · 3m 0s");
    expect(c?.tooltip).toContain("client_disconnected");
    expect(c?.tooltip).toContain("the client closed the connection");
    expect(c?.cacheHint).toBe(`hit ${(92636).toLocaleString()}`);
  });

  test("a miss keeps its reason as the hint", () => {
    const c = cancellationOf({
      ...cancelledHit,
      session_cache_hit: false,
      cache_source: "none",
      cache_miss_reason: "evicted",
    });
    expect(c?.cacheHint).toBe("miss: evicted");
  });

  test("a cancel before the prompt was read has no cache outcome", () => {
    const c = cancellationOf({
      ...cancelledHit,
      session_cache_hit: null,
      cached_tokens: null,
      cache_source: null,
    });
    expect(c?.cacheHint).toBeNull();
    expect(c?.tooltip).toContain("Cache outcome unknown");
  });

  test("an unknown reason is shown as sent", () => {
    expect(
      cancellationOf({ request_cancelled: true, cancellation_reason: "some_new_origin" })?.label,
    ).toBe("cancelled · some new origin");
  });

  test("a request that ran to the end is not a cancellation", () => {
    expect(cancellationOf({ session_cache_hit: false })).toBeNull();
    expect(cancellationOf({ request_cancelled: false })).toBeNull();
  });
});

describe("request log row", () => {
  test("a cancelled row shows CANCELLED with the cache hint, not HIT or MISS", () => {
    const out = html(cancelledHit);
    expect(out).toContain("cancelled · client · 3m 0s");
    expect(out).toContain("title=");
    expect(out).toContain(`hit ${(92636).toLocaleString()}`);
    expect(out).not.toContain(">HIT<");
    expect(out).not.toContain("MISS");
  });

  test("an older receipt (session_cache_hit false, no cached_tokens) is CANCELLED, not a miss", () => {
    const old = { ...cancelledHit, session_cache_hit: false, cached_tokens: undefined };
    expect(cancellationOf(old)?.cacheHint).toBeNull();
    const out = html(old);
    expect(out).toContain("cancelled · client");
    expect(out).not.toContain("MISS");
    expect(out).not.toContain("miss");
  });

  test("an ordinary miss keeps its MISS badge", () => {
    expect(html({ session_cache_hit: false })).toContain("MISS");
  });
});

describe("hit rate", () => {
  test("cancelled requests do not count", () => {
    expect(countsTowardHitRate(cancelledHit)).toBe(false);
    expect(countsTowardHitRate({ session_cache_hit: true })).toBe(true);
  });
});
