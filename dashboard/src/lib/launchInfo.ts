import type { LaunchEnvRow, LaunchEnvSource, LaunchRestart } from "./types";

// The server shows secret values as this placeholder; it stays unquoted in
// the restart command so it reads as a blank to fill in.
export const REDACTED = "•••";

const SHELL_SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;

export const SOURCE_LABELS: Record<LaunchEnvSource, string> = {
  launch: "launch",
  profile: "profile",
  both: "launch + profile",
  flag: "from a flag",
};

export const SOURCE_HINTS: Record<LaunchEnvSource, string> = {
  launch: "Set in the environment the server was started with.",
  profile: "Set by the profile.",
  both: "Set at launch and by the profile; the value shown is the one in effect.",
  flag: "Set by MTPLX from a command-line flag.",
};

export function shellQuote(value: string): string {
  if (value === "") return "''";
  if (SHELL_SAFE.test(value) || value.includes(REDACTED)) return value;
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

// One environment assignment per line, then the command itself, joined with
// backslash continuations so the block pastes into a shell as one command.
export function buildRestartCommand(restart: LaunchRestart): string {
  const lines = restart.env.map(({ name, value }) => `${name}=${shellQuote(value)}`);
  lines.push(restart.argv.map(shellQuote).join(" "));
  return lines.join(" \\\n");
}

export function commandHasRedactions(restart: LaunchRestart): boolean {
  return (
    restart.env.some(({ value }) => value.includes(REDACTED)) ||
    restart.argv.some((token) => token.includes(REDACTED))
  );
}

export type LaunchFlag = { flag: string; value: string | null };

// The flags of a launch command, without the program and subcommand in
// front (`mtplx serve`, `python -m mtplx.server.openai`).
export function launchFlags(argv: readonly string[]): LaunchFlag[] {
  const first = argv.findIndex((token) => token.startsWith("--"));
  if (first < 0) return [];
  const flags: LaunchFlag[] = [];
  for (let i = first; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("-")) continue;
    const eq = token.indexOf("=");
    if (token.startsWith("--") && eq > 0) {
      flags.push({ flag: token.slice(0, eq), value: token.slice(eq + 1) });
      continue;
    }
    const next = argv[i + 1];
    if (next !== undefined && (!next.startsWith("-") || /^-\d/.test(next))) {
      flags.push({ flag: token, value: next });
      i += 1;
    } else {
      flags.push({ flag: token, value: null });
    }
  }
  return flags;
}

// Every whitespace-separated term must appear in the name, the value (or
// either side of a launch + profile pair) or the source label.
export function filterEnvRows(rows: readonly LaunchEnvRow[], query: string): LaunchEnvRow[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [...rows];
  return rows.filter((row) => {
    const haystack = [
      row.name,
      row.value,
      row.launch_value ?? "",
      row.profile_value ?? "",
      row.source,
      SOURCE_LABELS[row.source],
    ]
      .join("\n")
      .toLowerCase();
    return terms.every((term) => haystack.includes(term));
  });
}
