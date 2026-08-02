from __future__ import annotations

from anthropic import Anthropic


client = Anthropic(
    api_key="local",
    base_url="http://127.0.0.1:8000",
)

message = client.messages.create(
    model="mtplx",
    max_tokens=256,
    system="Be concise.",
    messages=[
        {
            "role": "user",
            "content": "Write a tiny Python function that clamps a number.",
        }
    ],
)

for block in message.content:
    if block.type == "text":
        print(block.text)
