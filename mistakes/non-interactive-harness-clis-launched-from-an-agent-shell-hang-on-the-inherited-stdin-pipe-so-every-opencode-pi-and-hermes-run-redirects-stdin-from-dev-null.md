# Non-interactive harness CLIs launched from an agent shell hang on the inherited stdin pipe, so every opencode/pi/hermes run redirects stdin from /dev/null

**Symptom:** `opencode run -m mtplx/... "<prompt>"` started from a
background shell initialised ("all LSPs are disabled … init") and then sat
for three minutes in state S without sending a single request to the daemon;
the harness step looked like a slow generation.

**Cause:** the shell that launches a harness step inherits a stdin pipe that
never reaches EOF. OpenCode's `run` (and Pi's `-p`, Hermes' `chat -q`) treat
a non-TTY stdin as extra input and block on it before doing anything.

**Fix / rule:** launch every harness command with `< /dev/null` (the harness
runner's `run_step` does it once for all steps). With stdin closed the same
OpenCode command answered "OK" in 4.3 s. Before trusting a silent step, check
the daemon's flight log (`~/.mtplx/metrics/flight-<port>.jsonl`) for live
`ev: s` events; no events means the client never called.
