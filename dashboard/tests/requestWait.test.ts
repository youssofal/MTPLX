import { describe, expect, test } from "bun:test";
import { requestWait } from "../src/lib/requestWait";

describe("requestWait", () => {
  test("counts the previous turn's commit as waiting", () => {
    const wait = requestWait({
      ttft_s: 141,
      ttft_spans: {
        exclusive_s: {
          encode_s: 0.2,
          response_tail_wait_s: 135,
          postcommit_wait_s: 0,
          scheduler_queue_s: 4.5,
          lock_wait_s: 0.5,
          engine_first_token_s: 0.8,
        },
      },
    });
    expect(wait?.total_s).toBe(140);
    expect(wait?.parts.map((part) => part.label)).toEqual([
      "previous turn's commit",
      "scheduler queue",
      "generation lock",
    ]);
  });

  test("does not add lock_wait_time_s on top of the spans", () => {
    const wait = requestWait({
      lock_wait_time_s: 4.5,
      ttft_spans: { exclusive_s: { scheduler_queue_s: 4.5 } },
    });
    expect(wait?.total_s).toBe(4.5);
  });

  test("a request without spans has no wait figure", () => {
    expect(requestWait({ ttft_s: 1 })).toBeNull();
  });

  test("a request that did not wait reads zero", () => {
    const wait = requestWait({ ttft_spans: { exclusive_s: { encode_s: 0.1 } } });
    expect(wait).toEqual({ total_s: 0, parts: [] });
  });
});
