"""The terminal chat (`mtplx start cli`) streams an Inkling 2 Mini turn as its final text reads.

22 November 2026, the 2.13 tag wheel: on Inkling the terminal chat printed the model's control
tokens around the answer, `<|message_model|><|content_thinking|>...<|end_message|><|message_model|>
<|content_text|>4<|end_message|>`, because its streamer decoded the tokens plainly. `mtplx run` and
the server render the same turn as `<think>...</think>4` (`mtplx.models.inkling.protocol`).
"""

from __future__ import annotations

from mtplx.commands import public

CONTROL = (
    "<|message_model|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)
WORDS = {1: "The", 2: " user", 3: " asks.", 4: "4", 5: '{"name":"read","args":{"path":"a.txt"}}'}


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab = {name: 1000 + index for index, name in enumerate(CONTROL)}
        self._names = {value: key for key, value in self.vocab.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]

    def decode(self, ids, **_kwargs) -> str:
        return "".join(WORDS.get(int(i)) or self._names[int(i)] for i in ids)


def _ids(*names_or_words):
    tok = _Tokenizer()
    return [tok.vocab[x] if isinstance(x, str) else x for x in names_or_words]


def _stream(decoder, chunks) -> str:
    return "".join(decoder.feed(chunk) for chunk in chunks) + decoder.finish()


TURN = [
    _ids("<|message_model|>", "<|content_thinking|>", 1),
    _ids(2, 3, "<|end_message|>"),
    _ids("<|message_model|>", "<|content_text|>", 4, "<|end_message|>"),
]


def test_an_inkling_turn_streams_as_its_final_text_reads():
    text = _stream(public._InklingTerminalDecoder(_Tokenizer()), TURN)
    assert text == "<think>The user asks.</think>4"
    assert "<|" not in text


def test_the_plain_decoder_is_what_printed_the_control_tokens():
    text = _stream(public._QuickstartIncrementalTokenDecoder(_Tokenizer()), TURN)
    assert "<|message_model|>" in text and "<|content_text|>" in text


def test_an_inkling_tool_call_streams_as_a_tool_call():
    turn = [_ids("<|message_model|>", "<|content_invoke_tool_json|>", 5, "<|end_message|>")]
    text = _stream(public._InklingTerminalDecoder(_Tokenizer()), turn)
    assert text == '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>'


def test_the_terminal_streamer_takes_the_inkling_decoder_only_for_inkling():
    tok = _Tokenizer()
    assert isinstance(
        public._QuickstartTerminalStreamer(tok, label="MTPLX", inkling=True)._decoder,
        public._InklingTerminalDecoder,
    )
    assert isinstance(
        public._QuickstartTerminalStreamer(tok, label="MTPLX")._decoder,
        public._QuickstartIncrementalTokenDecoder,
    )
