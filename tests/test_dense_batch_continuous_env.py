"""MTPLX_DENSE_BATCH_CONTINUOUS: the documented escape hatch must be openable.

`_run_sealed` takes the tuned solo path only when `not self.continuous`, and
both its comment and docs/dense-mtp-batch-contract.md offer `continuous=False`
to an operator whose traffic is serialised. Whether the service HONOURS that
setting is already covered by test_dense_mtp_batch_serving. What was missing is
whether an operator can set it at all: nothing passed `continuous`, so the lane
always took its constructor default and the hatch could not be opened.

These pin the parse, not the service behaviour. The specific regression they
guard is a reader swap: openai.py also carries a local `_env_bool_setting`
whose vocabulary is narrower (no enable/enabled/disable/disabled) and which
silently returns False for anything it does not recognise. Reading this var
through that one instead would turn continuous batching OFF for any operator
who wrote "disabled", and ON for one who wrote "disable", with no error either
way. tests/test_env_flag_parsing.py exists because that class of divergence has
already shipped here once.
"""

from __future__ import annotations

import pytest

from mtplx.runtime_options import env_bool

VAR = "MTPLX_DENSE_BATCH_CONTINUOUS"


def _resolve(monkeypatch, value: str | None) -> bool:
    """Resolve the var exactly as ServerState does."""
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    return env_bool(VAR, default=True)


def test_unset_keeps_continuous_batching_on(monkeypatch) -> None:
    """The default must not change: this flag exists to opt OUT."""

    assert _resolve(monkeypatch, None) is True


def test_empty_keeps_continuous_batching_on(monkeypatch) -> None:
    assert _resolve(monkeypatch, "   ") is True


@pytest.mark.parametrize(
    "spelling", ["0", "false", "FALSE", " no ", "off", "disable", "disabled"]
)
def test_false_vocabulary_turns_batching_off(monkeypatch, spelling: str) -> None:
    """Every false spelling the runtime honours, including the two the local
    narrow reader would silently mis-handle."""

    assert _resolve(monkeypatch, spelling) is False


@pytest.mark.parametrize(
    "spelling", ["1", "true", "TRUE", " yes ", "on", "enable", "enabled"]
)
def test_true_vocabulary_keeps_batching_on(monkeypatch, spelling: str) -> None:
    assert _resolve(monkeypatch, spelling) is True


@pytest.mark.parametrize("typo", ["flase", "of", "yess", "2", "maybe"])
def test_a_typo_raises_rather_than_silently_choosing(monkeypatch, typo: str) -> None:
    """A misspelling must not quietly decide the serving mode.

    Silently reading "flase" as False would disable continuous batching for a
    whole deployment and present as a throughput bug, not a config error.
    """

    monkeypatch.setenv(VAR, typo)
    with pytest.raises(ValueError) as excinfo:
        env_bool(VAR, default=True)
    assert VAR in str(excinfo.value)


def test_the_narrow_local_reader_would_disagree(monkeypatch) -> None:
    """Pins WHY the shared parser is used, so a future refactor cannot quietly
    swap in openai.py's local `_env_bool_setting` without failing here.

    That reader accepts only {1,true,yes,on} and treats everything else as
    False, so "disabled" and "disable" agree with the shared parser by
    accident, while "enabled" and "enable" do not: the operator asks for
    batching ON and gets it OFF.
    """

    from mtplx.server.openai import _env_bool_setting

    for spelling in ("enable", "enabled"):
        monkeypatch.setenv(VAR, spelling)
        assert env_bool(VAR, default=True) is True
        assert _env_bool_setting(VAR, default=True) is False, (
            "if this ever agrees, the local reader grew the shared vocabulary "
            "and this test can go"
        )


def test_server_state_wires_the_flag() -> None:
    """The parse is worthless if nothing passes the result to the lane.

    Constructing a ServerState in a CPU test is not possible (it loads a
    model), so this asserts the wiring at the source level. Crude, but the
    alternative is that the whole point of this change goes unverified: the
    bug being fixed is precisely that `continuous=` was never passed.
    """

    import inspect

    from mtplx.server import openai as server

    source = inspect.getsource(server.ServerState.__init__)
    assert "continuous=env_bool(" in source, "the lane is not given the setting"
    assert VAR in source, "the documented variable name is not read"
