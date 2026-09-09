# Table 5 — Long-context verdict at 255K

The 255K context is 261,120 tokens. The 262,144 context is not run: it is
above the knob. The knob is 107.374 GB. The table shows the peak memory at
255K for each arm. Every arm is an OOM at 255K. The prefill completes on every
arm; the failure is in the first decode step, and the binding allocation is the
speculative verify's KV-cache write. TensorOffsetKVCache.update_and_fetch wrote
the new rows with the functional mx.slice_update, which returns a new array and
so reallocates the whole [1, 2, capacity, 256] bf16 buffer per key and per value
tensor in each of the 12 full-attention layers: about 6.4 GB in one verify
command buffer. A probe of that cache measures 6.42 GB resident and, with the
write made in place, a 0.00 GB update transient. A second and smaller transient,
the unfused [24, S, 261120] attention score plane, sits in the same step; a
head-chunked build removed it alone and the cell still OOM'd at the same peak.
The steady peak is near 100.79 GB and stays below the knob; the KV reallocation
does not. These cells are void: they are not measured.

The fix arm (in-place verify KV write plus the head chunk) does fit: 261,120
tokens completes on all three cold seeds with native MTP on, at a 100.82 GB
peak, about 6.5 GB under the 107.374 GB knob. Receipts under
battery475/receipts/arm-F2-inplacekv/. That arm is a separate branch and is not
one of the four arms in this table.

| arm | 255K steady peak GB | result |
| --- | ---: | --- |
| A: release 2.11.2 | 100.82 | void — OOM (verify KV write) |
| C: +#475 aux | 100.79 | void — OOM (verify KV write) |
| D: +#478 @0.2 | 100.79 | void — OOM (verify KV write) |
| E: +#478 @0.09 | 100.79 | void — OOM (verify KV write) |

## W1 and W2 bisect probes on C at 255K

W1 sets MTPLX_QSA_SCORE_TILE_ROWS=256 to tile the QSA score prefill. W2 turns
the session cache off and tries to set the bank to zero. Each probe is one
cold seed. W1 runs its other two seeds only if the first seed fits. Neither
probe fit. Both cells are void by the allocation-failure rule. Both probes
target prefill, and the overflow is in decode, so neither could have fit; the
score tiler is additionally gated off under fixed_capacity
(qwen4_exp.py:2980), so it cannot reach the verify step either.

| probe | cell | steady peak GB | engaged? | classification |
| --- | --- | ---: | --- | --- |
| W1 | error after 3 tok | 100.79 | UNOBSERVABLE — reader is read-at-use (qwen4_exp.py:1599) but the tiled path emits no log or /health marker; inconclusive | void — OOM (exceeds knob, allocation-failure) |
| W2 | error after 3 tok | 100.79 | cache off = confirmed (/health enabled=False); bank 0 = REJECTED (invalid; fell back to 25769803776 bytes) | void — OOM (exceeds knob, allocation-failure) |

W1 is INCONCLUSIVE as an engagement question, but it is off-target either way:
the tiler bounds a prefill transient and the prefill completed. W2's cache-off
half engaged and the bank-zero half was rejected; the cell OOM'd. The verdict
stands for these four arms: 255K is not measurable under the 107.374 GB knob on
this pack until the verify KV write is made in place.
