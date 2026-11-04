"""The web chat at the server root takes pasted and dropped images (#601).

The page rendered markdown and had a system prompt field, but a screenshot
pasted into it went nowhere, so the browser chat could not use the vision
tower that the app's chat and every agent client reach. Images now go out as
OpenAI ``image_url`` parts next to the text, the shape agent clients send,
and a model without a vision tower says so in the status line instead of
sending parts the server would refuse.

The behaviour itself was driven in a real browser against a stub of the
page's three endpoints (receipts in the issues agent's report): a pasted PNG
becomes a thumbnail and then a ``[text, image_url]`` user turn, a follow-up
turn resends it, drop adds files, the remove button and New chat clear them,
and a text-only model attaches nothing. These tests pin the server side of
that contract: the page knows whether the model reads images, from the same
answer ``/v1/models`` gives.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import _chat_ui_html, create_app
from tests.test_server_openai import _fake_state

SETTINGS = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "depth": 3, "depth_max": 3}


def _page(**kwargs) -> str:
    return _chat_ui_html(
        model_id="m",
        server_url="http://127.0.0.1:8000",
        api_key_required=False,
        default_settings=SETTINGS,
        **kwargs,
    )


def test_the_page_says_whether_the_model_reads_images():
    assert "const VISION_INPUT = true;" in _page(vision_input=True)
    assert "const VISION_INPUT = false;" in _page(vision_input=False)
    # Callers that do not say keep a text-only page.
    assert "const VISION_INPUT = false;" in _page()
    for page in (_page(vision_input=True), _page()):
        assert "__VISION_INPUT_JSON__" not in page


def test_the_page_sends_images_as_image_url_parts_and_takes_paste_and_drop():
    page = _page(vision_input=True)
    assert '<div id="attachments" hidden></div>' in page
    assert 'parts.push({type: "image_url", image_url: {url: url}});' in page
    assert 'history.push({role: "user", content: userContent(text, attached)});' in page
    assert 'promptEl.addEventListener("paste"' in page
    assert 'form.addEventListener("drop"' in page
    assert "This model reads text only, so images are not sent" in page


def test_the_root_page_follows_the_served_models_vision_tower(monkeypatch):
    state = _fake_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_server_vision_spec", lambda _state: SimpleNamespace())
    assert "const VISION_INPUT = true;" in client.get("/").text
    monkeypatch.setattr(openai, "_server_vision_spec", lambda _state: None)
    assert "const VISION_INPUT = false;" in client.get("/").text
