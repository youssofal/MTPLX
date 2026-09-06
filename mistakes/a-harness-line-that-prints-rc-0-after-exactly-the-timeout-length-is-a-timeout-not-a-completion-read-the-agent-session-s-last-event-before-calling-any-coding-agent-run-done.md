# A harness line that prints rc=0 after exactly the timeout length is a timeout, not a completion: read the agent session's last event before calling any coding-agent run done

**Symptom:** the 2.11.2 release-day harness reported `pi rc=0 in 900 s`
for a Pi 0.85.1 Snake task and `pi-tetris done rc=143 in 3109 s` for a Pi
0.85.0 Tetris task, and both runs went into the founder report and the
release notes as validation ("18 turns, 95.9 % warm reuse, 19 tests green";
"the agent iterated to All 16 tests pass"). The second-opinion audit read the
Pi session log: the Snake run's last event was an unfinished tool call ten
minutes before the process was killed at the 900 s bound, and the Tetris run
was terminated by hand. The tests the agent wrote were real and green; the
agent never finished either task on its own.

**Cause:** I read the tests passing as the task completing, and I read the
harness's own `rc=` line as the process outcome without checking what
produced it. A bound of exactly 900 s with `rc=0` is the bound firing, not
the agent returning; the exit status printed after a `( … ) < /dev/null >
log` subshell is not the agent's status either. Pi's print mode also hangs
on the model's own curses smoke test, which is exactly why the run needed
a bound, and exactly why the bound firing was the likely outcome.

**Fix / rule:** run every agent CLI through
`scripts/run_harness_check.py` (or any runner that records the child's real
exit code and counts a timeout as a failure), and before writing "completed",
read the client's own session log: the last event must be the assistant's
final message with a stop reason, not a tool call. Tests passing is a
receipt for the code the agent wrote, never for the agent having finished.
An `rc=` line that lands on the timeout value is a timeout.
