# Optional interleaved n-gram row cache

David Tai proposed interleaving the Flash-Next n-gram table's weights, scales
and biases in issue #449. The existing planar table makes a cold quantized row
require three separate read ranges. The derived layout stores those same bits
in one record. Most records touch one page; records crossing a page boundary
can touch two. This is not a claim of a threefold model speedup.

The original model pack remains unchanged. Conversion writes a new cache,
compares every row bit-for-bit, checks that the source did not change, and only
then makes the final file available. The cache requires roughly as much disk
space as the original n-gram table. Conversion uses bounded working chunks.

Run with the Python environment containing MTPLX:

```sh
python -m mtplx.ngram_row_layout /path/to/model/ngram-table.safetensors \
  --out /path/to/cache/ngram.rows.safetensors
MTPLX_NGRAM_ROW_FILE=/path/to/cache/ngram.rows.safetensors \
  mtplx serve --model /path/to/model
```

Keep the derived file outside the model's weight directory so other engines
cannot mistake it for another weight shard. The startup log identifies the
interleaved layout. Unset `MTPLX_NGRAM_ROW_FILE` to return to the original path.
An explicit cache must match this exact source file identity and geometry;
rebuild after replacing or moving the source rather than reusing a stale file.
The resident-table path continues using the original model file.

This is an opt-in experiment, not a changed default. Compare actual cold and
warm row behavior, prefill and decode, memory pressure, and sampled outputs on
the intended hardware before enabling it broadly. This implementation changes
row layout; it does not introduce unmeasured hot-LRU seeding. The tested local
pack did not contain `ngram-hotness.npy`.
