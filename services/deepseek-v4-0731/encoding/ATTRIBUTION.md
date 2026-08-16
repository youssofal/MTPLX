# DeepSeek-V4-Flash-0731 encoding attribution

The files listed in `SHA256SUMS` are vendored byte-for-byte from the official
Hugging Face repository:

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Source revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Encoder: `encoding/encoding_dsv4.py`
- Vectors: `encoding/tests/test_input_{1..4}.json` and
  `encoding/tests/test_output_{1..4}.txt`
- Upstream owner: DeepSeek AI

The service verifies all nine files at its construction boundary. It never
downloads encoding code or falls back to a tokenizer chat template at runtime.
