"""The quality instrument: does a pack still give the model's answers, and at
what length?

Four measurements, all taken through an OpenAI-compatible server so they
cover what users run:

- **tasks passed**: fixed, checkable tasks (code with hidden tests, small
  agent projects driven through tools, tool calls, exact answers, pictures);
- **tokens written** to finish each task, thinking and answer apart; a badly
  quantized model thinks longer and loops more for the same result;
- **draft acceptance by step**, from the same requests;
- **divergence** from a reference (the full-precision model): the partition
  KL, top-1 agreement and log-likelihood change over fixed continuations the
  reference wrote, scored teacher-forced.

Rows are JSON lines; ``report`` builds every table from them. See ``cli``
for the commands.
"""
