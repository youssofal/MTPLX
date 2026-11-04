import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { Row } from "../src/components/RequestLogTable";
import { cacheMissBadge } from "../src/lib/cacheMiss";

describe("cacheMissBadge", () => {
  test.each([
    ["new_session", "NEW CHAT", "text-muted"],
    ["prefix_divergence_at_token", "NO SHARED PREFIX", "text-muted"],
    ["evicted", "EVICTED", "accent-warm"],
    ["ssd_restore_error:OSError", "SSD ERROR", "accent-orange"],
    ["ssd_missing_mtp_history", "SSD ERROR", "accent-orange"],
    ["ssd_prefix_miss", "NO SSD MATCH", "text-muted"],
    ["model_mismatch", "MODEL_MISMATCH", "accent-warm"],
    [null, "MISS", "accent-warm"],
  ])("%s", (reason, label, colour) => {
    const badge = cacheMissBadge(reason);
    expect(badge.label).toBe(label);
    expect(badge.color).toContain(colour);
  });

  test("the tooltip carries the precise reason", () => {
    expect(cacheMissBadge("ssd_restore_error:OSError").tooltip).toContain(
      "ssd_restore_error:OSError",
    );
  });
});

describe("request log row", () => {
  const html = (row: object) =>
    renderToStaticMarkup(
      <table>
        <tbody>
          <Row row={row} isOpen={false} onToggle={() => {}} />
        </tbody>
      </table>,
    );

  test("a new conversation reads NEW CHAT with a tooltip, not SSD_PREFIX_MISS", () => {
    const out = html({ session_cache_hit: false, cache_miss_reason: "new_session" });
    expect(out).toContain("NEW CHAT");
    expect(out).toContain("title=");
    expect(out).not.toContain("SSD_PREFIX_MISS");
  });

  test("a hit stays HIT", () => {
    expect(html({ session_cache_hit: true })).toContain("HIT");
  });
});
