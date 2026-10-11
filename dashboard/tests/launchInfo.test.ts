import { describe, expect, test } from "bun:test";
import {
  REDACTED,
  buildRestartCommand,
  commandHasRedactions,
  filterEnvRows,
  launchFlags,
  shellQuote,
} from "../src/lib/launchInfo";
import type { LaunchEnvRow } from "../src/lib/types";

describe("buildRestartCommand", () => {
  test("puts every launch variable in front of the operator's command", () => {
    const command = buildRestartCommand({
      env: [
        { name: "PYTHONPATH", value: "/opt/mlx:/opt/mtplx" },
        { name: "MTPLX_THINKING_BUDGET", value: "6144" },
      ],
      argv: ["/venv/bin/mtplx", "serve", "--model", "/models/q", "--port", "8123", "--no-auth"],
    });

    expect(command).toBe(
      [
        "PYTHONPATH=/opt/mlx:/opt/mtplx \\",
        "MTPLX_THINKING_BUDGET=6144 \\",
        "/venv/bin/mtplx serve --model /models/q --port 8123 --no-auth",
      ].join("\n"),
    );
  });

  test("without launch variables it is the command alone", () => {
    expect(buildRestartCommand({ env: [], argv: ["mtplx", "serve", "--mtp"] })).toBe(
      "mtplx serve --mtp",
    );
  });

  test("quotes values the shell would split or expand", () => {
    const command = buildRestartCommand({
      env: [{ name: "MTPLX_NOTE", value: "it's a test" }],
      argv: ["/My Apps/mtplx", "serve", "--pi-launch-command", "pi --model $M"],
    });

    expect(command).toBe(
      ["MTPLX_NOTE='it'\\''s a test' \\", "'/My Apps/mtplx' serve --pi-launch-command 'pi --model $M'"].join(
        "\n",
      ),
    );
  });

  test("leaves redacted secrets visible as a blank to fill in", () => {
    const restart = {
      env: [{ name: "MTPLX_API_KEY", value: REDACTED }],
      argv: ["mtplx", "serve", "--api-key", REDACTED],
    };

    expect(buildRestartCommand(restart)).toBe(
      `MTPLX_API_KEY=${REDACTED} \\\nmtplx serve --api-key ${REDACTED}`,
    );
    expect(commandHasRedactions(restart)).toBe(true);
    expect(commandHasRedactions({ env: [], argv: ["mtplx", "serve"] })).toBe(false);
  });

  test("shellQuote keeps plain tokens and empties visible", () => {
    expect(shellQuote("--port")).toBe("--port");
    expect(shellQuote("a=b,c:d/e")).toBe("a=b,c:d/e");
    expect(shellQuote("")).toBe("''");
  });
});

describe("launchFlags", () => {
  test("skips the program and pairs flags with their values", () => {
    expect(
      launchFlags(["/venv/bin/mtplx", "serve", "--model", "/m", "--port=8123", "--no-auth", "--mtp"]),
    ).toEqual([
      { flag: "--model", value: "/m" },
      { flag: "--port", value: "8123" },
      { flag: "--no-auth", value: null },
      { flag: "--mtp", value: null },
    ]);
  });

  test("skips python's own options on a direct server launch", () => {
    expect(
      launchFlags(["/py", "-P", "-m", "mtplx.server.openai", "--depth", "-1", "--yes"]),
    ).toEqual([
      { flag: "--depth", value: "-1" },
      { flag: "--yes", value: null },
    ]);
  });
});

describe("filterEnvRows", () => {
  const rows: LaunchEnvRow[] = [
    { name: "MTPLX_THINKING_BUDGET", value: "6144", source: "launch" },
    { name: "MTPLX_NAX_VERIFY", value: "1", source: "profile" },
    {
      name: "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
      value: "4096",
      source: "both",
      launch_value: "4096",
      profile_value: "2048",
    },
    { name: "MTPLX_PAGED_KV_QUANT", value: "off", source: "flag" },
    { name: "PYTHONPATH", value: "/opt/mlx", source: "launch" },
  ];
  const names = (query: string) => filterEnvRows(rows, query).map((row) => row.name);

  test("an empty query keeps every row", () => {
    expect(names("  ")).toHaveLength(rows.length);
  });

  test("matches names case-insensitively", () => {
    expect(names("prefill")).toEqual(["MTPLX_PREFILL_CHUNK_SIZE_DENSE"]);
  });

  test("matches values, including the profile's side of a pair", () => {
    expect(names("/opt/mlx")).toEqual(["PYTHONPATH"]);
    expect(names("2048")).toEqual(["MTPLX_PREFILL_CHUNK_SIZE_DENSE"]);
  });

  test("matches the source and its label", () => {
    expect(names("launch")).toEqual([
      "MTPLX_THINKING_BUDGET",
      "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
      "PYTHONPATH",
    ]);
    expect(names("from a flag")).toEqual(["MTPLX_PAGED_KV_QUANT"]);
  });

  test("every term must match", () => {
    expect(names("mtplx launch")).toEqual([
      "MTPLX_THINKING_BUDGET",
      "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
    ]);
    expect(names("nothing-like-this")).toEqual([]);
  });
});
