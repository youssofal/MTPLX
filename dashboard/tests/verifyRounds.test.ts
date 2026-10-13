import { describe, expect, test } from "bun:test";
import { waterfallScale } from "../src/lib/verifyRounds";

describe("waterfallScale", () => {
  test("per round divides the request totals by its verify rounds", () => {
    const scale = waterfallScale("round", 7);
    expect(scale).toEqual({ mode: "round", rounds: 7, divisor: 7 });
    expect(0.752 / scale.divisor).toBeCloseTo(0.1074, 4);
  });

  test("per request keeps the totals and still reports the round count", () => {
    expect(waterfallScale("request", 7)).toEqual({ mode: "request", rounds: 7, divisor: 1 });
  });

  test("falls back to per request without a positive round count", () => {
    for (const calls of [0, null, undefined, -3, Number.NaN]) {
      expect(waterfallScale("round", calls)).toEqual({ mode: "request", rounds: 0, divisor: 1 });
    }
  });
});
