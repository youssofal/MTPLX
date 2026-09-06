"""A CLI dispatch must not leave its profile environment for the next test.

`mtplx.profiles.apply_profile_env()` writes a runtime profile's whole env dict
into `os.environ` when no explicit mapping is passed — that is how the daemon
launch path hands the profile to the child process, and it is the right
behaviour for a short-lived CLI process. Inside a pytest session it is a leak:
every later test in the same process inherits the knobs. Measured on 2026-09-06,
one `mtplx run`/`ask`/`chat` dispatch left `MTPLX_BATCH_TARGET_ARRAYS`,
`MTPLX_DROP_EVENTS`, `MTPLX_LAZY_MTP_HISTORY_APPEND`,
`MTPLX_LAZY_TARGET_DISTRIBUTIONS`, `MTPLX_LAZY_VERIFY_LOGITS` and
`MTPLX_SKIP_VERIFY_SNAPSHOT` set, which changed the verify-call sequences that
`tests/test_generation_sustained.py` asserts, and six of its tests failed only
when `tests/test_public_cli.py` ran first.

These tests are deliberately order-dependent: the naming fixes their sequence, and
together they pin both halves of the guarantee — the mechanism is real, and the
suite-level guard contains it.
"""

from __future__ import annotations

import os

SENTINEL = "MTPLX_SUITE_ISOLATION_SENTINEL"


def test_a_writing_the_process_environment_is_visible_in_that_test():
    os.environ[SENTINEL] = "set-by-test-a"
    assert os.environ[SENTINEL] == "set-by-test-a"


def test_b_the_previous_test_environment_write_did_not_leak():
    assert SENTINEL not in os.environ


def test_c_profile_env_application_really_does_touch_the_process():
    from mtplx.profiles import apply_profile_env

    before = set(os.environ)
    apply_profile_env("stable")
    assert set(os.environ) - before, (
        "apply_profile_env('stable') no longer writes the process environment; "
        "if the CLI path stopped mutating os.environ, this file's premise is "
        "stale and the guard belongs in the CLI path instead"
    )


def test_d_profile_env_written_by_a_dispatch_did_not_leak_either():
    leaked = [
        "MTPLX_BATCH_TARGET_ARRAYS",
        "MTPLX_DROP_EVENTS",
        "MTPLX_LAZY_MTP_HISTORY_APPEND",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS",
        "MTPLX_LAZY_VERIFY_LOGITS",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
    ]
    present = [key for key in leaked if key in os.environ]
    assert not present, f"profile env leaked into the session: {present}"
