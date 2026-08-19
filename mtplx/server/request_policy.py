"""Per-request policy resolution for the OpenAI/Anthropic serving endpoints.

One resolver owns the request "prologue" policy for /v1/chat/completions,
/v1/completions, and /v1/messages/count_tokens: tool policy, prompt
contracts, transcript canonicalization, thinking + reasoning effort,
generation mode and draft depth, client-control ownership, and sampler
resolution (including the OpenCode sampler/draft overrides and the
server-default fallback).

Contract notes:

- The observability envelope is pinned byte-for-byte by
  tests/test_request_observability_golden.py (13 client arms). Keys and
  values produced by ``RequestPolicy.as_observability`` are part of that
  contract; a drifted key is a behavior change, not a cosmetic one.
- Resolution ORDER is load-bearing: several steps raise HTTP 400 (invalid
  generation mode, out-of-range depth, non-finite sampler params) and the
  busy-background bypass raises :class:`BackgroundBusyBypass` before any
  further policy is resolved, so single-fault requests keep the same
  response they had when each endpoint carried its own copy.
- count_tokens deliberately keeps its historical shape: the REQUESTED
  toolset is encoded unfiltered and no prompt contracts are applied, so
  token counts stay identical to what that endpoint always returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from mtplx.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P
from mtplx.sampling import SamplerConfig


class BackgroundBusyBypass(Exception):
    """Background Open WebUI task admitted while foreground work is live.

    Raised at the exact point the busy check historically ran (right after
    transcript canonicalization, before thinking/effort/mode resolution) so
    a busy background request never reaches the later 400-raising steps.
    The endpoint converts this into the 503 bypass response.
    """


def _srv() -> Any:
    # The helpers this resolver orchestrates live in mtplx.server.openai,
    # which imports this module. Resolution is late-bound through the module
    # object so monkeypatched helpers (tests patch openai.<helper>) and the
    # runtime-import fallbacks in openai.py keep working.
    from mtplx.server import openai

    return openai


@dataclass(frozen=True)
class RequestPolicy:
    """Resolved per-request policy, immutable for the request's lifetime.

    Field names match the endpoint locals they feed so call sites can alias
    them 1:1. ``observability`` holds exactly the policy-owned entries the
    endpoints merge into ``request_observability``; endpoint-owned entries
    (session, vision, constraint, template) are not represented here.
    """

    endpoint: str

    # Tool policy.
    opencode_client: bool = False
    requested_tool_specs: list[dict[str, Any]] = field(default_factory=list)
    tool_specs: list[dict[str, Any]] = field(default_factory=list)
    tools_active: bool = False
    agent_transcript_tools_active: bool = False
    postcommit_tool_specs: list[dict[str, Any]] | None = None

    # Prompt contracts.
    read_only_force_answer_contract_active: bool = False
    no_tools_contract_active: bool = False
    post_tool_answer_contract_active: bool = False
    pi_convergence_contract_active: bool = False
    opencode_simple_chat_contract_active: bool = False
    opencode_prompt_contract_profile: str | None = None
    transient_suffix_contract_active: bool = False
    backend_chat_policy_active: bool = False

    # Canonicalized transcript.
    messages_for_generation: list[Any] = field(default_factory=list)
    transcript_stats: Any | None = None
    raw_messages_for_postcommit: list[Any] = field(default_factory=list)
    read_only_inspection_request: bool = False
    tool_result_history_present: bool = False
    background: bool = False

    # Reasoning.
    thinking_enabled: bool = False
    reasoning_effort: str | None = None
    aime_visible_working: bool = False

    # Control ownership.
    client_controls_allowed: bool = True

    # Pass-through for Pi: when True, MTPLX adds none of its own prompt
    # contracts (tool contract, convergence/no-tool/post-tool/read-only
    # suffixes, backend language policy). Set from per-request client
    # detection; the default keeps every non-Pi client byte-identical.
    disable_prompt_injections: bool = False

    # Tool prompt rendering.
    tool_prompt_mode: str | None = None
    template_tool_prompt_mode: str | None = None
    tool_prompt_mode_resolution: dict[str, Any] = field(default_factory=dict)
    postcommit_tool_prompt_mode: str | None = None

    # Generation mode and draft depth.
    request_generation_mode: str | None = None
    request_depth: int | None = None
    effective_request_depth: int | None = None
    short_depth_policy: dict[str, Any] | None = None
    long_context_depth_policy: dict[str, Any] | None = None

    # Sampler resolution (server defaults already applied).
    sampler_temperature: float | None = None
    sampler_top_p: float | None = None
    sampler_top_k: int | None = None
    sampler_presence_penalty: float | None = None
    sampler_frequency_penalty: float | None = None
    request_draft_sampler: SamplerConfig | None = None

    # Policy-owned observability entries.
    observability: dict[str, Any] = field(default_factory=dict)

    def as_observability(self) -> dict[str, Any]:
        """Policy-owned ``request_observability`` entries, ready to merge."""
        return dict(self.observability)

    def with_prompt_tokens(
        self,
        state: Any,
        request: Any,
        *,
        headers: dict[str, str],
        metadata: dict[str, Any],
        prompt_tokens: int,
    ) -> "RequestPolicy":
        """Depth refinements that need the encoded prompt length.

        Chat encodes the prompt after policy resolution (the committed-
        reasoning canonicalization seam sits on the encode), so the
        short-context and long-context depth policies run as a second
        stage. ``request`` is threaded through because the short-context
        policy inspects the raw depth field on the request body.
        """
        srv = _srv()
        request_depth, short_depth_policy = srv._opencode_short_context_depth_policy(
            request,
            headers=headers,
            metadata=metadata,
            generation_mode=self.request_generation_mode,
            request_depth=self.request_depth,
            prompt_tokens=prompt_tokens,
        )
        effective_request_depth, long_context_depth_policy = (
            srv._long_context_mtp_depth_policy_for_request(
                state,
                generation_mode=self.request_generation_mode,
                request_depth=request_depth,
                prompt_tokens=prompt_tokens,
            )
        )
        observability = dict(self.observability)
        observability["request_effective_mtp_depth"] = int(effective_request_depth)
        observability["long_context_mtp_depth_policy"] = long_context_depth_policy
        observability["opencode_short_context_depth_policy"] = short_depth_policy
        return replace(
            self,
            request_depth=request_depth,
            effective_request_depth=effective_request_depth,
            short_depth_policy=short_depth_policy,
            long_context_depth_policy=long_context_depth_policy,
            observability=observability,
        )


def _resolve_sampler(
    state: Any,
    request: Any,
    *,
    client_controls_allowed: bool,
    messages_for_generation: list[Any],
    tools_active: bool,
    observability: dict[str, Any],
) -> tuple[float, float, int, float | None, float | None, SamplerConfig | None]:
    """Sampler ownership, OpenCode overrides, and server-default fallback.

    Shared by chat and completions so both surface identical effective_*
    telemetry and identical draft-sampler resolution inputs.
    """
    srv = _srv()
    if client_controls_allowed:
        srv._reject_non_finite_sampler_controls(request)
    sampler_temperature = request.temperature if client_controls_allowed else None
    sampler_top_p = request.top_p if client_controls_allowed else None
    sampler_top_k = request.top_k if client_controls_allowed else None
    # Penalties follow the same control-ownership policy as the other
    # sampler fields; None falls through to the server default inside
    # _generation_params (request value > server default > 0.0).
    sampler_presence_penalty = (
        request.presence_penalty if client_controls_allowed else None
    )
    sampler_frequency_penalty = (
        request.frequency_penalty if client_controls_allowed else None
    )
    observability["request_temperature"] = request.temperature
    observability["request_top_p"] = request.top_p
    observability["request_top_k"] = request.top_k
    if request.presence_penalty is not None:
        observability["request_presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        observability["request_frequency_penalty"] = request.frequency_penalty
    ignored_sampler_fields = [
        name
        for name, value in (
            ("temperature", request.temperature),
            ("top_p", request.top_p),
            ("top_k", request.top_k),
            ("presence_penalty", request.presence_penalty),
            ("frequency_penalty", request.frequency_penalty),
        )
        if value is not None and not client_controls_allowed
    ]
    if ignored_sampler_fields:
        observability["client_sampler_fields_ignored"] = ignored_sampler_fields
    target_sampler_override = srv._opencode_default_sampler_override(
        messages=messages_for_generation,
        tools_active=tools_active,
        request_temperature=request.temperature,
        request_top_p=request.top_p,
        request_top_k=request.top_k,
        request_observability=observability,
        default_temperature=getattr(state.args, "temperature", DEFAULT_TEMPERATURE),
        default_top_p=getattr(state.args, "top_p", DEFAULT_TOP_P),
        default_top_k=getattr(state.args, "top_k", DEFAULT_TOP_K),
    )
    # The OpenCode normalization overrides the TARGET sampler only. The
    # draft sampler is deliberately NOT injected as a request value:
    # server-injected values are launch_default ownership, not
    # request_explicit, so the per-request draft resolution (family curve +
    # greedy coupling) still runs against the launch policy (F9).
    request_draft_sampler: SamplerConfig | None = None
    if target_sampler_override is not None:
        sampler_temperature = target_sampler_override.temperature
        sampler_top_p = target_sampler_override.top_p
        sampler_top_k = target_sampler_override.top_k
        srv._opencode_launch_default_draft_policy(state, observability)
    if sampler_temperature is None:
        default_temperature = getattr(state.args, "temperature", None)
        sampler_temperature = (
            DEFAULT_TEMPERATURE if default_temperature is None else default_temperature
        )
    if sampler_top_p is None:
        default_top_p = getattr(state.args, "top_p", None)
        sampler_top_p = DEFAULT_TOP_P if default_top_p is None else default_top_p
    if sampler_top_k is None:
        default_top_k = getattr(state.args, "top_k", None)
        sampler_top_k = DEFAULT_TOP_K if default_top_k is None else default_top_k
    observability["effective_temperature"] = float(sampler_temperature)
    observability["effective_top_p"] = float(sampler_top_p)
    observability["effective_top_k"] = int(sampler_top_k)
    observability["effective_presence_penalty"] = float(
        sampler_presence_penalty
        if sampler_presence_penalty is not None
        else getattr(state.args, "default_presence_penalty", 0.0) or 0.0
    )
    observability["effective_frequency_penalty"] = float(
        sampler_frequency_penalty
        if sampler_frequency_penalty is not None
        else getattr(state.args, "default_frequency_penalty", 0.0) or 0.0
    )
    return (
        sampler_temperature,
        sampler_top_p,
        sampler_top_k,
        sampler_presence_penalty,
        sampler_frequency_penalty,
        request_draft_sampler,
    )


def _control_ownership_observability(
    request: Any,
    *,
    client_controls_allowed: bool,
    observability: dict[str, Any],
) -> None:
    srv = _srv()
    observability["mtplx_control_owner"] = (
        "client" if client_controls_allowed else "server"
    )
    observability["client_controls_allowed"] = bool(client_controls_allowed)
    if not client_controls_allowed:
        ignored_fields = srv._ignored_client_control_fields(request)
        if ignored_fields:
            observability["client_control_fields_ignored"] = ignored_fields


def _resolve_completions_policy(
    state: Any,
    request: Any,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    prompt_tokens: int,
) -> RequestPolicy:
    srv = _srv()
    observability: dict[str, Any] = {}
    client_controls_allowed = srv._client_controls_allowed(headers, metadata)
    disable_prompt_injections = srv._is_pi_client(headers=headers, metadata=metadata)
    request_generation_mode = srv._request_generation_mode_for_generation(
        state,
        request,
        allow_client_controls=client_controls_allowed,
    )
    request_depth = srv._request_depth_for_generation(
        state,
        request,
        generation_mode=request_generation_mode,
        allow_client_controls=client_controls_allowed,
    )
    effective_request_depth, long_context_depth_policy = (
        srv._long_context_mtp_depth_policy_for_request(
            state,
            generation_mode=request_generation_mode,
            request_depth=request_depth,
            prompt_tokens=prompt_tokens,
        )
    )
    # Seed the client hint before sampler resolution: the OpenCode override
    # keys off it. A raw-prompt request has no transcript, so the override
    # can never fire here; routing through the shared resolver keeps the
    # telemetry and fallback identical to chat by construction.
    observability["request_client_hint"] = srv._request_client_hint_from_headers(
        headers, metadata
    )
    # Behavior branches key on per-request evidence only; the env-inclusive
    # hint above is an observability label (F1 — the launch env must never
    # steer another client's request).
    observability["request_client_evidence"] = srv._request_client_hint_from_request(
        headers, metadata
    )
    observability["request_generation_mode"] = request_generation_mode
    observability["request_depth"] = int(request_depth)
    observability["request_effective_mtp_depth"] = int(effective_request_depth)
    _control_ownership_observability(
        request,
        client_controls_allowed=client_controls_allowed,
        observability=observability,
    )
    (
        sampler_temperature,
        sampler_top_p,
        sampler_top_k,
        sampler_presence_penalty,
        sampler_frequency_penalty,
        request_draft_sampler,
    ) = _resolve_sampler(
        state,
        request,
        client_controls_allowed=client_controls_allowed,
        messages_for_generation=[],
        tools_active=False,
        observability=observability,
    )
    return RequestPolicy(
        endpoint="completions",
        client_controls_allowed=client_controls_allowed,
        disable_prompt_injections=disable_prompt_injections,
        request_generation_mode=request_generation_mode,
        request_depth=request_depth,
        effective_request_depth=effective_request_depth,
        long_context_depth_policy=long_context_depth_policy,
        sampler_temperature=sampler_temperature,
        sampler_top_p=sampler_top_p,
        sampler_top_k=sampler_top_k,
        sampler_presence_penalty=sampler_presence_penalty,
        sampler_frequency_penalty=sampler_frequency_penalty,
        request_draft_sampler=request_draft_sampler,
        observability=observability,
    )


def resolve_request_policy(
    state: Any,
    request: Any,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    endpoint: str = "chat",
    prompt_tokens: int | None = None,
) -> RequestPolicy:
    """Resolve the per-request policy for one serving endpoint.

    ``endpoint`` selects the profile:

    - ``"chat"`` — full pipeline. Raises :class:`BackgroundBusyBypass` for
      a busy background request, and HTTP 400 for invalid generation mode,
      depth, or non-finite sampler params (in that order). Depth policies
      that need the prompt length are applied later via
      :meth:`RequestPolicy.with_prompt_tokens`.
    - ``"count_tokens"`` — the encode-relevant front half only: unfiltered
      requested tools, plain canonicalization, backend chat policy,
      thinking/effort, tool prompt mode. Never raises for mode/depth/
      sampler fields because it never resolves them.
    - ``"completions"`` — no transcript or tool policy; resolves controls,
      mode, depth (``prompt_tokens`` is required and known up front), and
      the shared sampler block.
    """
    if endpoint == "completions":
        if prompt_tokens is None:
            raise ValueError("completions policy requires prompt_tokens")
        return _resolve_completions_policy(
            state,
            request,
            headers=headers,
            metadata=metadata,
            prompt_tokens=prompt_tokens,
        )
    if endpoint not in {"chat", "count_tokens"}:
        raise ValueError(f"unknown request-policy endpoint: {endpoint!r}")
    srv = _srv()
    chat = endpoint == "chat"
    observability: dict[str, Any] = {}

    opencode_client = srv._is_opencode_client(headers=headers, metadata=metadata)
    # Pi pass-through: when the client is Pi, MTPLX adds none of its own
    # prompt injections — Qwen receives exactly what Pi sent. Every
    # contract below is gated off (kept as dead code, not removed) so the
    # policy stays readable and the disable is a single observable switch.
    disable_prompt_injections = srv._is_pi_client(headers=headers, metadata=metadata)
    requested_tool_specs = srv._normalize_tool_specs(request.tools)
    if chat:
        tool_specs = srv._filter_tool_specs_for_request(
            requested_tool_specs,
            request.messages,
            tool_choice=request.tool_choice,
            client_manages_tools=opencode_client,
        )
        tools_active = srv._tools_active_for_request(tool_specs, request.tool_choice)
    else:
        # count_tokens counts the prompt a client would be billed for from
        # its own declared toolset: no bridge filtering, no contracts.
        tool_specs = requested_tool_specs
        tools_active = srv._tools_active_for_request(
            requested_tool_specs,
            request.tool_choice,
        )
    raw_tool_result_history_present = any(
        str(message.role).lower() == "tool" for message in request.messages
    )
    agent_transcript_tools_active = bool(
        tools_active
        or (
            chat
            and requested_tool_specs
            and raw_tool_result_history_present
            and srv._is_read_only_inspection_request(
                srv._last_user_text(request.messages)
            )
        )
    )
    read_only_force_answer_contract_active = bool(
        chat
        and not disable_prompt_injections
        and srv._request_should_force_answer_for_read_only_inspection(request.messages)
    )
    if read_only_force_answer_contract_active:
        if srv._tool_result_message_count(
            request.messages
        ) > 0 and srv._request_explicit_single_tool_then_answer(request.messages):
            # Explicit "use one tool then answer": the forced final turn
            # generates tool-free, and turn-level tool state/observability
            # must agree (zero remaining tools, read_only_force_answer:v1
            # policy version).
            tool_specs = []
            tools_active = False
        else:
            # Read-budget force answer: keep the REQUESTED toolset
            # byte-identical to every prior round of this loop. Filtering
            # to a read-only subset rewrote the rendered tool contract in
            # the system prompt, so the largest prompt of the session (the
            # forced final answer) diverged from every banked prefix at
            # token ~3 and re-prefilled fully cold (measured 2026-07-04:
            # 13.2k tokens, TTFT 21s). The appended force-answer user
            # message carries the "answer now, no more tools" conditioning;
            # prefix stability owns the toolset bytes.
            pass
    no_tools_contract_applies = bool(
        chat
        and not disable_prompt_injections
        and not read_only_force_answer_contract_active
        and srv._should_add_no_tool_contract(
            requested_tools=requested_tool_specs,
            tools_active=tools_active,
            messages=request.messages,
        )
    )
    # A tool loop's FINAL round (tool results already in this turn,
    # tools now disabled) must synthesize a full answer — it gets
    # the post-tool contract instead of the terse direct-reply one,
    # whose no-lists/no-analysis clauses clip searched answers.
    post_tool_answer_contract_active = bool(
        no_tools_contract_applies
        and srv._turn_tail_contains_tool_results(request.messages)
    )
    no_tools_contract_active = bool(
        no_tools_contract_applies and not post_tool_answer_contract_active
    )
    client_controls_allowed = srv._client_controls_allowed(headers, metadata)
    pi_convergence_contract_active = bool(
        chat
        and not disable_prompt_injections
        and not read_only_force_answer_contract_active
        and not no_tools_contract_active
        and not post_tool_answer_contract_active
        and srv._request_should_add_pi_convergence_contract(
            request.messages,
            headers=headers,
            metadata=metadata,
            tools_active=agent_transcript_tools_active,
        )
    )
    opencode_prompt_contract_profile = (
        srv._opencode_prompt_contract_profile(
            request.messages,
            headers=headers,
            metadata=metadata,
            tool_choice=request.tool_choice,
        )
        if chat
        else None
    )
    opencode_prompt_contract_system_prompt = (
        # Pi pass-through never replaces the client's leading system message.
        srv._opencode_prompt_contract_system_prompt(opencode_prompt_contract_profile)
        if chat and not disable_prompt_injections
        else None
    )
    opencode_simple_chat_contract_active = False
    if chat:
        messages_for_generation, transcript_stats = srv._canonicalize_agent_transcript(
            request.messages,
            tools_active=agent_transcript_tools_active,
            replace_simple_chitchat_system_prompt=False,
            initial_client_system_prompt=opencode_prompt_contract_system_prompt,
            strip_tool_call_preamble_text=opencode_client,
        )
    else:
        messages_for_generation, transcript_stats = srv._canonicalize_agent_transcript(
            request.messages,
            tools_active=tools_active,
        )
    if disable_prompt_injections:
        # Pi pass-through: no backend language-policy splice either.
        backend_chat_policy_active = False
    else:
        messages_for_generation, backend_chat_policy_active = (
            srv._with_backend_chat_policy(
                state,
                messages_for_generation,
            )
        )
    if chat:
        if read_only_force_answer_contract_active:
            messages_for_generation = (
                srv._with_mtplx_read_only_force_answer_contract(
                    messages_for_generation
                )
            )
        elif post_tool_answer_contract_active:
            messages_for_generation = srv._with_mtplx_post_tool_answer_contract(
                messages_for_generation
            )
        elif no_tools_contract_active:
            messages_for_generation = srv._with_mtplx_no_tool_contract(
                messages_for_generation
            )
        elif pi_convergence_contract_active:
            messages_for_generation = srv._with_mtplx_pi_convergence_contract(
                messages_for_generation
            )
    read_only_inspection_request = bool(
        chat
        and srv._is_read_only_inspection_request(
            srv._last_user_text(messages_for_generation)
        )
    )
    tool_result_history_present = any(
        str(message.role).lower() == "tool" for message in messages_for_generation
    )
    raw_messages_for_postcommit = (
        list(request.messages)
        if read_only_force_answer_contract_active
        else (
            list(messages_for_generation)
            if (
                no_tools_contract_active
                or post_tool_answer_contract_active
                or pi_convergence_contract_active
                or opencode_prompt_contract_profile is not None
                or backend_chat_policy_active
            )
            else list(request.messages)
        )
    )
    postcommit_tool_specs = (
        tool_specs
        if tools_active
        else (requested_tool_specs if agent_transcript_tools_active else None)
    )
    background = bool(
        chat
        and srv.is_background_request(
            messages=messages_for_generation,
            max_tokens=srv._request_max_tokens(request),
            headers=headers,
            metadata=metadata,
            main_system_hash=state.main_system_prompt_hash,
        )
    )
    if background and (
        state.has_foreground()
        or state.lock.locked()
        or srv._foreground_model_work_pending(state)
    ):
        raise BackgroundBusyBypass()
    thinking_enabled = srv._thinking_enabled_for_request(
        state,
        request,
        allow_client_controls=client_controls_allowed,
    )
    reasoning_effort = srv._reasoning_effort_for_state(
        state,
        thinking_enabled=thinking_enabled,
        request_effort=request.reasoning_effort,
        allow_client_controls=client_controls_allowed,
    )
    if (
        read_only_force_answer_contract_active
        and srv._reasoning_parser_for_state(state) == "gemma4"
    ):
        thinking_enabled = False
    aime_visible_working = bool(
        chat
        and srv._aime_visible_working_for_request(metadata)
        and thinking_enabled
        and srv._reasoning_parser_for_state(state) in {"qwen3", "step3p5"}
    )
    tool_prompt_mode, tool_prompt_mode_resolution = srv._tool_prompt_mode_for_request(
        state.args,
        headers=headers,
        metadata=metadata,
        tools_active=tools_active,
        backend=srv._backend_descriptor(state),
    )
    template_tool_prompt_mode = tool_prompt_mode
    if chat and read_only_force_answer_contract_active and tools_active:
        # Read-budget force-answer turns keep the SAME template mode as
        # every prior round. The earlier hybrid switch re-rendered the
        # system prompt with full "# Tools" schemas, so the forced final
        # turn's prompt diverged from all banked prefixes at token ~3 and
        # re-prefilled fully cold — the fingerprint compatibility shim
        # could not help because the BYTES differed (2026-07-04 fix).
        # The appended contract user message is a pure suffix and owns
        # the force-answer conditioning.
        tool_prompt_mode_resolution = {
            **tool_prompt_mode_resolution,
            "tool_prompt_mode_source": "read_only_force_answer_prefix_stable",
        }
    postcommit_tool_prompt_mode = tool_prompt_mode
    if chat and postcommit_tool_specs and not tools_active:
        postcommit_tool_prompt_mode, _ = srv._tool_prompt_mode_for_request(
            state.args,
            headers=headers,
            metadata=metadata,
            tools_active=True,
            backend=srv._backend_descriptor(state),
        )
    if not chat:
        return RequestPolicy(
            endpoint=endpoint,
            opencode_client=opencode_client,
            requested_tool_specs=requested_tool_specs,
            tool_specs=tool_specs,
            tools_active=tools_active,
            agent_transcript_tools_active=agent_transcript_tools_active,
            postcommit_tool_specs=postcommit_tool_specs,
            backend_chat_policy_active=backend_chat_policy_active,
            messages_for_generation=messages_for_generation,
            transcript_stats=transcript_stats,
            raw_messages_for_postcommit=raw_messages_for_postcommit,
            tool_result_history_present=tool_result_history_present,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            client_controls_allowed=client_controls_allowed,
            disable_prompt_injections=disable_prompt_injections,
            tool_prompt_mode=tool_prompt_mode,
            template_tool_prompt_mode=template_tool_prompt_mode,
            tool_prompt_mode_resolution=tool_prompt_mode_resolution,
            postcommit_tool_prompt_mode=postcommit_tool_prompt_mode,
        )
    request_generation_mode = srv._request_generation_mode_for_generation(
        state,
        request,
        allow_client_controls=client_controls_allowed,
    )
    request_depth = srv._request_depth_for_generation(
        state,
        request,
        generation_mode=request_generation_mode,
        allow_client_controls=client_controls_allowed,
    )
    transient_suffix_contract_active = bool(
        read_only_force_answer_contract_active
        or pi_convergence_contract_active
        # Suffix contracts since audit F11 #6: they no longer rewrite msg0,
        # so entries banked WITHOUT the contract stay byte-compatible
        # prefixes and restore/postcommit must use the contract-free
        # fingerprint on the flip turn (same mechanism as Pi convergence).
        or no_tools_contract_active
        or post_tool_answer_contract_active
    )

    observability["request_client_hint"] = srv._request_client_hint_from_headers(
        headers, metadata
    )
    observability["request_client_evidence"] = srv._request_client_hint_from_request(
        headers, metadata
    )
    observability["disable_prompt_injections"] = bool(disable_prompt_injections)
    server_reasoning_mode = getattr(state.args, "reasoning", None)
    if server_reasoning_mode not in {"auto", "on", "off"}:
        server_reasoning_mode = (
            "on" if bool(getattr(state.args, "enable_thinking", True)) else "off"
        )
    if not client_controls_allowed:
        request_reasoning_mode = (
            "off" if not thinking_enabled else server_reasoning_mode
        )
    elif request.enable_thinking is False:
        request_reasoning_mode = "off"
    elif request.enable_thinking is True and server_reasoning_mode == "auto":
        request_reasoning_mode = "on"
    else:
        request_reasoning_mode = server_reasoning_mode
    observability["request_reasoning_mode"] = request_reasoning_mode
    observability["request_enable_thinking"] = bool(thinking_enabled)
    observability["request_reasoning_effort"] = reasoning_effort
    observability["request_enable_thinking_override"] = (
        request.enable_thinking is not None and client_controls_allowed
    )
    _control_ownership_observability(
        request,
        client_controls_allowed=client_controls_allowed,
        observability=observability,
    )
    observability["request_reasoning_parser"] = srv._reasoning_parser_for_state(state)
    observability["request_read_only_inspection_force_answer"] = bool(
        read_only_force_answer_contract_active
    )
    observability["request_read_only_inspection_tool_result_count"] = (
        srv._tool_result_message_count(request.messages)
    )
    observability["request_read_only_inspection_force_answer_after_tools"] = (
        srv._read_only_inspection_force_answer_after_tools()
    )
    observability["request_pi_convergence_contract"] = bool(
        pi_convergence_contract_active
    )
    observability["request_pi_convergence_tool_result_count"] = (
        srv._tool_result_message_count(request.messages)
    )
    observability["request_pi_convergence_after_tools"] = (
        srv._pi_convergence_after_tools()
    )
    observability["opencode_simple_chat_contract_active"] = bool(
        opencode_simple_chat_contract_active
    )
    observability["opencode_prompt_contract_profile"] = (
        opencode_prompt_contract_profile or "none"
    )
    observability["backend_chat_policy_active"] = bool(backend_chat_policy_active)
    observability["request_effective_message_count"] = len(messages_for_generation)
    observability["request_effective_message_roles"] = [
        message.role for message in messages_for_generation
    ]
    observability["request_effective_message_chars"] = [
        len(srv._content_to_text(message.content))
        for message in messages_for_generation
    ]
    observability["preserve_thinking"] = getattr(
        state.args, "preserve_thinking", "auto"
    )
    observability["preserve_thinking_effective"] = srv._preserve_thinking_effective(
        state.args
    )
    observability["reasoning_history_mode"] = srv._reasoning_history_mode(state)
    observability["strip_assistant_reasoning_history"] = bool(
        state.args.strip_assistant_reasoning_history
    )
    observability.update(
        srv._bridge_policy_observability(
            tools_active=tools_active,
            tool_prompt_mode=template_tool_prompt_mode,
            no_tools_contract_active=no_tools_contract_active,
            read_only_force_answer_contract_active=read_only_force_answer_contract_active,
            pi_convergence_contract_active=pi_convergence_contract_active,
            post_tool_answer_contract_active=post_tool_answer_contract_active,
            disable_prompt_injections=disable_prompt_injections,
        )
    )
    observability.update(tool_prompt_mode_resolution)
    requested_tool_names = [
        name for tool in (request.tools or []) if (name := srv._tool_spec_name(tool))
    ]
    filtered_tool_names = srv._tool_names(tool_specs) if tools_active else []
    hidden_tool_names = [
        name for name in requested_tool_names if name not in filtered_tool_names
    ]
    observability.update(
        {
            "request_filtered_tool_count": len(filtered_tool_names),
            "request_filtered_tool_names": filtered_tool_names,
            "request_hidden_tool_names": hidden_tool_names,
            "request_tools_hidden_by_bridge": bool(hidden_tool_names),
        }
    )
    chat_template_report = getattr(state, "chat_template_report", {}) or {}
    observability.update(
        {
            "chat_template_profile": str(
                chat_template_report.get("profile")
                or getattr(
                    state, "chat_template_profile", srv._CHAT_TEMPLATE_PROFILE_LOCAL
                )
            ),
            "chat_template_source": chat_template_report.get("source"),
            "chat_template_path": chat_template_report.get("path"),
            "chat_template_hash": state.template_hash,
        }
    )
    observability.update(transcript_stats.to_metrics())
    (
        sampler_temperature,
        sampler_top_p,
        sampler_top_k,
        sampler_presence_penalty,
        sampler_frequency_penalty,
        request_draft_sampler,
    ) = _resolve_sampler(
        state,
        request,
        client_controls_allowed=client_controls_allowed,
        messages_for_generation=messages_for_generation,
        tools_active=tools_active,
        observability=observability,
    )
    return RequestPolicy(
        endpoint=endpoint,
        opencode_client=opencode_client,
        requested_tool_specs=requested_tool_specs,
        tool_specs=tool_specs,
        tools_active=tools_active,
        agent_transcript_tools_active=agent_transcript_tools_active,
        postcommit_tool_specs=postcommit_tool_specs,
        read_only_force_answer_contract_active=read_only_force_answer_contract_active,
        no_tools_contract_active=no_tools_contract_active,
        post_tool_answer_contract_active=post_tool_answer_contract_active,
        pi_convergence_contract_active=pi_convergence_contract_active,
        opencode_simple_chat_contract_active=opencode_simple_chat_contract_active,
        opencode_prompt_contract_profile=opencode_prompt_contract_profile,
        transient_suffix_contract_active=transient_suffix_contract_active,
        backend_chat_policy_active=backend_chat_policy_active,
        messages_for_generation=messages_for_generation,
        transcript_stats=transcript_stats,
        raw_messages_for_postcommit=raw_messages_for_postcommit,
        read_only_inspection_request=read_only_inspection_request,
        tool_result_history_present=tool_result_history_present,
        background=background,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
        aime_visible_working=aime_visible_working,
        client_controls_allowed=client_controls_allowed,
        disable_prompt_injections=disable_prompt_injections,
        tool_prompt_mode=tool_prompt_mode,
        template_tool_prompt_mode=template_tool_prompt_mode,
        tool_prompt_mode_resolution=tool_prompt_mode_resolution,
        postcommit_tool_prompt_mode=postcommit_tool_prompt_mode,
        request_generation_mode=request_generation_mode,
        request_depth=request_depth,
        effective_request_depth=request_depth,
        sampler_temperature=sampler_temperature,
        sampler_top_p=sampler_top_p,
        sampler_top_k=sampler_top_k,
        sampler_presence_penalty=sampler_presence_penalty,
        sampler_frequency_penalty=sampler_frequency_penalty,
        request_draft_sampler=request_draft_sampler,
        observability=observability,
    )
