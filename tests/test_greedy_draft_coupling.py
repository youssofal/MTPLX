"""Greedy target ⇒ greedy draft coupling (speed-only, output-invariant)."""

from __future__ import annotations

from mtplx.sampling import SamplerConfig
from mtplx.server.openai import _couple_draft_sampler_to_greedy_target


def _launch_draft():
    return SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)


def test_couples_at_temp0():
    obs: dict = {}
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=obs,
    )
    assert out.temperature == 0.0
    assert out.top_p == 0.95 and out.top_k == 20  # only greediness changes
    assert obs["draft_sampler_greedy_coupled"] is True


def test_untouched_at_sampled_target():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.6,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_explicit_draft_sampler_wins():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=True,
        target_temperature=0.0,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_none_target_temperature_untouched():
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=None,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_env_off_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_COUPLING", "off")
    out = _couple_draft_sampler_to_greedy_target(
        _launch_draft(),
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=None,
    )
    assert out.temperature == 0.6


def test_already_greedy_draft_passthrough():
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    obs: dict = {}
    out = _couple_draft_sampler_to_greedy_target(
        greedy,
        explicit_draft_sampler=False,
        target_temperature=0.0,
        request_observability=obs,
    )
    assert out is greedy
    assert "draft_sampler_greedy_coupled" not in obs
