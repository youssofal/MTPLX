"""The 400 for a prompt past the window tells the user what they can do on this Mac.

21 November (Bare Speed, simulated 24 GB, `mtplx serve`): a 32K prompt was refused at the 20,480-token window
with "serve with a larger --context-window", where that window is what the Mac's memory holds, so a larger one
would swap or be stopped mid-stream. The first sentence keeps OpenAI's wording and both numbers.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mtplx.server import openai as srv


def _refusal(window, prompt, plan):
    with pytest.raises(HTTPException) as e:
        srv._reject_prompt_over_context(SimpleNamespace(context_window=window, memory_plan=plan), prompt)
    assert e.value.status_code == 400 and e.value.detail["code"] == "context_length_exceeded"
    message = e.value.detail["message"]
    assert message.startswith(f"This model's maximum context length is {window} tokens, but the prompt alone has "
                              f"{prompt} tokens, leaving no room to generate. ")
    return message


def test_a_window_the_macs_memory_sets_asks_for_a_shorter_conversation():
    plan = SimpleNamespace(context_window_fit=20480, context_machine_bound=True, context_overcommitted=False)
    message = _refusal(20480, 32835, plan)
    assert message.endswith("This Mac's memory sets the window: shorten or compact the conversation.")
    assert "--context-window" not in message


def test_a_window_served_below_the_fit_can_be_raised_to_it():
    plan = SimpleNamespace(context_window_fit=163840, context_machine_bound=True, context_overcommitted=False)
    message = _refusal(32768, 40000, plan)
    assert message.endswith("Reduce the prompt or serve with a larger --context-window "
                            "(this Mac's memory fits 163840 tokens).")


def test_the_models_own_maximum_or_an_override_asks_for_a_shorter_prompt():
    model_bound = SimpleNamespace(context_window_fit=1048576, context_machine_bound=False, context_overcommitted=False)
    assert _refusal(1048576, 1048576, model_bound).endswith("Reduce the prompt.")
    overcommitted = SimpleNamespace(context_window_fit=112640, context_machine_bound=True, context_overcommitted=True)
    assert _refusal(262144, 262144, overcommitted).endswith("Reduce the prompt.")


def test_without_a_plan_the_message_is_unchanged():
    message = _refusal(4096, 6000, None)
    assert message.endswith("Reduce the prompt or serve with a larger --context-window.")
