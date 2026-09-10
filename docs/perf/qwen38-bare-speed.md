# Qwen3.8 Flash-Next: the Bare-Speed pack onto MTPLX 2.11.2

The Bare-Speed pack (`Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed`) has the
same geometry as Optimized-Speed and differs only in its quantization recipe:
base 4-bit group-64 (Optimized-Speed is group-32), `lm_head` Q4/g64 (not Q8/g64),
shared experts Q4/g64 (not Q8/g64), and routed experts Q4/g64 (not Q4/g32);
`mlp.gate` and the QSA indexer stay Q8/g64, the mixers and PLE projections stay
bf16, the n-gram sidecar stays Q4/g32.

Because the fixed-M4 predicate is pure geometry, the whole base + fixed-M4
decode stack already arms on Bare-Speed under upstream defaults. Two optimization
families were withheld, on two pack facts: FR-Spec (the `lm_head` is Q4/g64, not
Q8/g64) and the M=4 stage-3 combine tail (the shared experts are Q4/g64, not
Q8/g64, and the routed experts are Q4/g64, not Q4/g32). This change makes both
engage on Bare-Speed. Each optimization is default-on for a served Bare-Speed
pack (stamped in the `mtplx/server/openai.py` auto-arm block, gated on the
fixed-M4 config predicate) with a per-key `=0` opt-out through the existing pop
loop.

GPU parity numbers are measured later, under the flock, between battery windows;
this document records the engagement contract each optimization now satisfies.

## FR-Spec pruned draft head on a Q4/g64 lm_head

- **Problem.** FR-Spec's installer refused any native MTP draft head that was
  not Q8/g64, and the server auto-arm stamped FR-Spec only on a Q8/g64 `lm_head`.
  Bare-Speed ships `lm_head` Q4/g64, so FR-Spec (and the K20 pre-scatter that
  rides it) were withheld — even though the row-pruning is quant-generic.
- **Change.** Broaden the FR-Spec pack predicate
  (`_served_model_lm_head_is_frspec_capable`) and the installer's native-head
  contract to accept an affine g64 head at 4-bit as well as 8-bit. The pruned
  head is built as an `nn.QuantizedLinear` from the source head's own
  bits/group_size, so a Q4/g64 head prunes exactly like the Q8/g64 one; every
  other head layout still fails the model load.
- **Effect.** FR-Spec and its 65,536-row compact draft support arm on
  Bare-Speed. The decode effect (draft acceptance with a Q4 draft head) is a
  GPU-phase measurement.
- **Exactness.** Model outputs stay exact — the target verifies the full
  vocabulary. The draft distribution is the head's own Q4 output, so draft
  acceptance is rounding-class and is gated on the acceptance A/B.
- **Files.** `mtplx/server/openai.py`, `mtplx/frspec_draft.py`.
- **Switch.** `MTPLX_FRSPEC_DRAFT` (auto-armed for a served Bare-Speed pack;
  `=0` opt-out), with the `MTPLX_FRSPEC_VOCAB` companion.

> The K20 pre-scatter read-at-use fix is not part of this change: it belongs to
> the #475 base (the upstream auto-arm audit owns it). K20 engages on Bare-Speed
> as a consequence of the FR-Spec Q4/g64 head above — it rides the FR-Spec
> compact row.

## M=4 stage-3 combine on Q4/g64 shared experts

- **Problem.** The stage-3 combine tail pinned the shared-expert projections at
  Q8/g64 (both the server pack predicate and the installer contracts). Bare-Speed
  ships the shared experts Q4/g64, so the whole stage-3 optimization was withheld.
  (The router `mlp.gate` is Q8/g64 on both packs and is unchanged.)
- **Change.** The server pack predicate and the installer's shared-expert
  contracts (shared-expert gate, the fused gate+up owner, and the down
  projection) now accept Q4/g64 as well as Q8/g64. The combine reads each owner's
  own bits/group_size, so only the packed-weight column count differs between the
  two; every other geometry still raises at model load.
- **Effect.** The shared-expert half of the stage-3 combine engages on
  Bare-Speed. The routed-expert half needs the Q4/g64 routed kernels — the next
  optimization.
- **Exactness.** Bit-exact — the change is predicate-only; the combine math is
  unchanged and reads bits/group_size dynamically.
- **Files.** `mtplx/server/openai.py`, `mtplx/qwen4_m4_stage3.py`.
- **Switch.** `MTPLX_QWEN4_M4_STAGE3` (auto-armed for a served Bare-Speed pack;
  `=0` opt-out); its routed children `MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE`,
  `MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL`, and `MTPLX_QWEN4_M4_ROUTED_GLU` arm
  with it.

## Routed-expert Q4/g64 GLU and down kernels

- **Problem.** The paired routed GLU (`qwen4_m4_routed_glu`) and the routed-down
  reduce (`qwen4_m4_routed_down`) specialize MLX 0.32.2's affine q4/group-32
  arithmetic with `GROUP_SIZE = 32` baked in as a Metal constexpr. Bare-Speed's
  routed experts are Q4/g64, so the routed half of the stage-3 combine could not
  run on them.
- **Change.** Parameterize both kernels by group size (`bind(group_size)`), one
  compiled kernel cached per size. `GROUP_SIZE` is the only constexpr that
  changes: every scale/bias stride derives from it (`GROUPS_PER_ROW`,
  `BLOCK_SIZE / GROUP_SIZE`, `lane / (GROUP_SIZE / VALUES_PER_THREAD)`) and the
  4-bit packing is group-size-independent (`WEIGHT_BYTES_PER_ROW = HIDDEN / 2`).
  The installer selects the kernel from the pack's routed group size, and the
  routed contracts accept Q4/g64 (half the scale/bias group count of g32). The
  shared-add and residual-tail kernels are unchanged — they combine already
  computed activations, not quantized weights.
- **Effect.** The routed half of the stage-3 combine engages on Bare-Speed.
  Speed is a GPU-phase measurement.
- **Exactness.** Rounding-class. The existing construction-time self-check runs
  per layer and requires `dmax <= 2**-9` against the stock routed forward; the
  GPU parity probe (under the flock) signs the g64 kernels off. Until then the
  g64 path is fail-closed only when explicitly required
  (`MTPLX_STRICT_CLAIMS=1`); under default arming a self-check miss or a build
  failure declines the whole combine to the stock MoE forward and prints a
  verdict line, so the pack still serves. The g32 (Optimized-Speed) path keeps
  its raise-always contract.
- **Files.** `mtplx/kernels/qwen4_m4_routed_glu.py`,
  `mtplx/kernels/qwen4_m4_routed_down.py`, `mtplx/qwen4_m4_stage3.py`.
- **Switch.** `MTPLX_QWEN4_M4_ROUTED_GLU`,
  `MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE`,
  `MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL` (arm with `MTPLX_QWEN4_M4_STAGE3`;
  `=0` opt-out); `MTPLX_STRICT_CLAIMS=1` makes the g64 self-check fail closed.

## Deferred: the two-kernel route head

The route head (`MTPLX_QWEN4_ROUTE_KERNEL`) has a Q8/g64 shared-expert-gate GEMV
arm only, validated by `mx.array_equal`. On Bare-Speed the shared-expert gate is
Q4/g64, so the auto-arm leaves the route head off and the stock routing head runs
(`route=None`); the stage-3 combine ships without it. A bit-exact Q4/g64
route-head GEMV is a separate, later optimization, not part of this change.
