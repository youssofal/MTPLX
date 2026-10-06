import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { Row } from "../src/components/RequestLogTable";
import { refusalOf } from "../src/lib/requestRefusal";

const refused = {
  session_id: "anon-945f5c23332c58ae",
  prompt_tokens: 90005,
  completion_tokens: 0,
  decode_tok_s: 0,
  session_cache_hit: false,
  cache_miss_reason: null,
  error_kind: "memory_refusal",
  error_status: 507,
  error_detail: "insufficient memory: the other apps on this Mac leave 8.8 GiB free",
  refusal_reason: "system_memory_short_after_reclamation",
  retry_when: "after_other_apps_free_memory",
};

const html = (row: object, isOpen = false) =>
  renderToStaticMarkup(
    <table>
      <tbody>
        <Row row={row} isOpen={isOpen} onToggle={() => {}} />
      </tbody>
    </table>,
  );

describe("refusalOf", () => {
  test("names a memory refusal", () => {
    const r = refusalOf(refused);
    expect(r?.label).toBe("507 memory");
    expect(r?.details).toContainEqual([
      "refused because",
      "system memory short after reclamation",
    ]);
    expect(r?.details).toContainEqual(["retry when", "after other apps free memory"]);
  });

  test("generalises to other statuses", () => {
    expect(refusalOf({ error_status: 503, error_kind: "session_busy" })?.label).toBe(
      "503 session busy",
    );
    expect(refusalOf({ error_status: 429 })?.label).toBe("429");
  });

  test("ordinary requests are not refusals", () => {
    expect(refusalOf({ session_cache_hit: false, error_status: null })).toBeNull();
  });
});

describe("request row", () => {
  test("a refused row shows the badge and dashes, not MISS and zeros", () => {
    const out = html(refused);
    expect(out).toContain("507 memory");
    expect(out).toContain("accent-hot");
    expect(out).not.toContain("MISS");
    expect(out).not.toContain("0.0");
    expect(out.match(/>—</g)?.length).toBeGreaterThanOrEqual(4);
  });

  test("expanded details carry the reason and retry_when", () => {
    const out = html(refused, true);
    expect(out).toContain("system memory short after reclamation");
    expect(out).toContain("after other apps free memory");
  });

  test("a normal miss keeps its MISS badge", () => {
    const out = html({ prompt_tokens: 10, completion_tokens: 5, decode_tok_s: 20 });
    expect(out).toContain("MISS");
    expect(out).not.toContain("accent-hot");
  });
});
