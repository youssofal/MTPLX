"""Headquarter-layout tape-capture kernel for the GDN verify recurrence.

Same I/O contract and arithmetic chain as
``mtplx_linear_gated_delta_from_conv_tape_v1`` (gdn_capture.py), different
execution allocation, following the A3B G3 "C1 headquarter" redesign that
measured 3.04x over the TGY layout at identical numerics:

  * incumbent: grid (32, Dv, B*Hv), tg (32, tgy, 1) -> Dv/tgy threadgroups per
    head, ONE dv row per simdgroup, q/k norm recomputed by every threadgroup
    (Dv/tgy times per head), two dependent simd_sums back-to-back per row.
  * here: grid (Simds*32, Quarters, B*Hv), tg (Simds*32, 1, 1) -> Quarters
    threadgroups per head, each simdgroup owns RPS = Dv/Quarters/Simds dv rows
    with the running fp32 state resident in registers, row-level ILP between
    the two simd_sum reductions, q/k norm computed once per threadgroup
    (Quarters times per head), one producer barrier per step plus a WAR
    barrier only when a next timestep exists.

Bit-exactness contract (must match the incumbent verbatim):
  * q/k rms-norm: per-lane 4 sequential fp32 square-adds -> simd_sum ->
    precise::rsqrt(sum/Dk + 1e-6) -> InT round -> *scale -> InT double round,
    identical lane->dk slice map (lane owns elements 4*lane .. 4*lane+3).
  * recurrence per dv row: state *= g; kv = sum(state*k) via 4 sequential
    adds + simd_sum; delta = (v - kv) * beta; state += k*delta; y = sum
    (state*q) same order; tape stores fp32 delta; state rounds through StT
    between steps; final_state written once after the T loop.

g/beta arrive precomputed ([B,T,Hv]) exactly as the incumbent takes them.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    auto n = thread_position_in_grid.z;           // b_idx*Hv + hv_idx
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    auto quarter = thread_position_in_grid.y;      // 0..Quarters-1
    uint tptg = thread_position_in_threadgroup.x;  // 0..(Simds*32-1)
    uint simd_id = tptg / 32u;                     // 0..Simds-1
    uint dk_idx = thread_index_in_simdgroup;       // 0..31
    constexpr int n_per_t = Dk / 32;               // fp32 elements per lane
    constexpr int QSIZE = Dv / Quarters;           // dv rows per threadgroup
    constexpr int RPS = QSIZE / Simds;             // dv rows per simdgroup
    int base_dv = int(quarter) * QSIZE + int(simd_id) * RPS;

    float inv_scale = 1.0f / metal::sqrt(float(Dk));
    float q_scale = inv_scale * inv_scale;
    float k_scale = static_cast<float>(static_cast<InT>(inv_scale));

    threadgroup float q_shared[Dk];
    threadgroup float k_shared[Dk];

    // Running fp32 state for this simdgroup's RPS rows, register resident.
    float S[RPS][n_per_t];
    for (int r = 0; r < RPS; ++r) {
      const device StT* s_ptr = state_in + (n * Dv + (base_dv + r)) * Dk;
      for (int i = 0; i < n_per_t; ++i) {
        S[r][i] = static_cast<float>(s_ptr[n_per_t * dk_idx + i]);
      }
    }

    for (int t = 0; t < T; ++t) {
      auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
      auto q_t = conv_t + hk_idx * Dk;
      auto k_t = conv_t + KeyDim + hk_idx * Dk;
      auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
      auto g_t = g + (b_idx * T + t) * Hv;
      auto beta_t = beta + (b_idx * T + t) * Hv;

      // Producer: simd 0 computes the shared normed q/k once per threadgroup.
      if (simd_id == 0u) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float q_raw[n_per_t];
        float k_raw[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          q_raw[i] = static_cast<float>(q_t[s_idx]);
          k_raw[i] = static_cast<float>(k_t[s_idx]);
          q_sum += q_raw[i] * q_raw[i];
          k_sum += k_raw[i] * k_raw[i];
        }
        q_sum = simd_sum(q_sum);
        k_sum = simd_sum(k_sum);
        float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
        float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
          auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
          q_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
          k_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float g_local = g_t[hv_idx];
      float beta_local = beta_t[hv_idx];
      float qloc[n_per_t];
      float kloc[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        qloc[i] = q_shared[s_idx];
        kloc[i] = k_shared[s_idx];
      }

      float kv[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] * g_local;
          acc += S[r][i] * kloc[i];
        }
        kv[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { kv[r] = simd_sum(kv[r]); }

      float delta[RPS];
      for (int r = 0; r < RPS; ++r) {
        delta[r] = (static_cast<float>(v_t[base_dv + r]) - kv[r]) * beta_local;
      }

      float out[RPS];
      for (int r = 0; r < RPS; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = S[r][i] + kloc[i] * delta[r];
          acc += S[r][i] * qloc[i];
        }
        out[r] = acc;
      }
      for (int r = 0; r < RPS; ++r) { out[r] = simd_sum(out[r]); }

      auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
      for (int r = 0; r < RPS; ++r) {
        int dv = base_dv + r;
        if (dk_idx == 0u) {
          y_t[dv] = static_cast<InT>(out[r]);
          tape[((b_idx * T + t) * Hv + hv_idx) * Dv + dv] = delta[r];
        }
      }

      // Inter-step state rounding through StT, exactly as the incumbent.
      for (int r = 0; r < RPS; ++r) {
        for (int i = 0; i < n_per_t; ++i) {
          S[r][i] = static_cast<float>(static_cast<StT>(S[r][i]));
        }
      }

      if (t + 1 < T) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // WAR guard on q/k_shared
      }
    }

    for (int r = 0; r < RPS; ++r) {
      device StT* state_t = final_state + (n * Dv + (base_dv + r)) * Dk;
      for (int i = 0; i < n_per_t; ++i) {
        state_t[n_per_t * dk_idx + i] = static_cast<StT>(S[r][i]);
      }
    }
"""


def _make_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="mtplx_linear_gated_delta_from_conv_tape_headquarter_v1",
        input_names=["conv_out", "g", "beta", "state_in", "T"],
        output_names=["y", "final_state", "tape"],
        source=_SOURCE,
    )


_KERNEL = _make_kernel()

_SIMDS = 8
_QUARTERS = 4


def headquarter_tape_capture(
    conv_out: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    gdn,
):
    """Drop-in alternative to the incumbent tape launch; None if ineligible."""
    if _KERNEL is None:
        return None
    B, T, conv_dim = conv_out.shape
    if int(conv_dim) != int(gdn.conv_dim):
        return None
    Dk = int(gdn.head_k_dim)
    Dv = int(gdn.head_v_dim)
    Hk = int(gdn.num_k_heads)
    Hv = int(gdn.num_v_heads)
    if Dk % 32 != 0:
        return None
    qsize = Dv // _QUARTERS
    if Dv % _QUARTERS != 0 or qsize % _SIMDS != 0:
        return None
    if Hv % Hk != 0:
        return None
    input_type = conv_out.dtype
    state_type = state.dtype
    return _KERNEL(
        inputs=[conv_out, g, beta, state, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("KeyDim", int(gdn.key_dim)),
            ("ConvDim", int(gdn.conv_dim)),
            ("Quarters", _QUARTERS),
            ("Simds", _SIMDS),
        ],
        grid=(_SIMDS * 32, _QUARTERS, B * Hv),
        threadgroup=(_SIMDS * 32, 1, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk), (B, T, Hv, Dv)],
        output_dtypes=[input_type, state_type, mx.float32],
    )
