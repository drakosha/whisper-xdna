//===- attn_fused.cc ------------------------------------------*- C++ -*-===//
//
// One attention head, fused: QK -> softmax -> PV with the scores never leaving
// the compute tile. Called once per K/V block; the running state lives in core
// memory between calls, which is what makes it a single pass over K and V.
//
// Online rescale rather than an exact full-row softmax. We had rejected that
// earlier on the grounds that AIE2 has no vector fp32 multiply -- and that was
// wrong: the running sum lives in an aie::accum<accfloat> and every
// multiply is bf16 into it, exactly the shapes the hardware has. The result is
// arithmetically the same softmax, just accumulated blockwise:
//
//   m'  = max(m, rowmax(S))
//   c   = exp(m - m')            correction for what is already accumulated
//   l'  = c*l + rowsum(exp(S - m'))
//   O'  = c*O  + exp(S - m') @ V
//
// and one divide by l at the end. No score matrix is ever written to DDR: the
// 12 GB per encoder pass that QK, softmax and AV move between them collapses to
// the 1.1 GB of Q, K, V and O.
//
// First version: correctness before speed. The two matmuls are written as
// straightforward vector reductions rather than aie::mmul tiles, so this is not
// the kernel to benchmark -- it is the kernel to check the dataflow and the
// arithmetic against numpy. Optimising them is the next step, and the program
// memory budget has room for it: the whole of AMD's reference is 11232 bytes of
// 16384.
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// Modifications Copyright (c) 2026 Mikhail Kostryukov
// Written against AMD's flash-attention examples; the originals live in
// https://github.com/Xilinx/mlir-air (MIT).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#ifndef ATTN_BQ
#define ATTN_BQ 48
#endif
#ifndef ATTN_BK
#define ATTN_BK 32
#endif
#ifndef ATTN_HD
#define ATTN_HD 64
#endif

using namespace aie;

// running state, one set per core, alive across the calls of one wave
static float acc_[ATTN_BQ][ATTN_HD];
static float m_[ATTN_BQ];
static float l_[ATTN_BQ];

extern "C" {

void attn_head_bf16(bfloat16 *restrict q, bfloat16 *restrict kv,
                    bfloat16 *restrict o, int32_t kb, int32_t nkb) {
  constexpr int N = 16;
  constexpr int BQ = ATTN_BQ, BK = ATTN_BK, HD = ATTN_HD;
  bfloat16 *k = kv;                 // packed: BK rows of K, then BK rows of V
  bfloat16 *v = kv + BK * HD;

  if (kb == 0) {
    for (int i = 0; i < BQ; i++) {
      m_[i] = -3.0e38f;
      l_[i] = 0.0f;
      for (int d = 0; d < HD; d++)
        acc_[i][d] = 0.0f;
    }
  }

  float s[BK];
  for (int i = 0; i < BQ; i++) {
    // S[i][j] = <q_i, k_j>
    float rowmax = -3.0e38f;
    for (int j = 0; j < BK; j++) {
      aie::accum<accfloat, N> a = aie::zeros<accfloat, N>();
      for (int d = 0; d < HD; d += N) {
        aie::vector<bfloat16, N> qv = aie::load_v<N>(q + i * HD + d);
        aie::vector<bfloat16, N> kvv = aie::load_v<N>(k + j * HD + d);
        a = aie::add(a, aie::mul(qv, kvv));
      }
      float t = aie::reduce_add(a.to_vector<float>());
      s[j] = t;
      if (t > rowmax)
        rowmax = t;
    }

    // online rescale of what is already accumulated
    float mnew = m_[i] > rowmax ? m_[i] : rowmax;
    float corr = 1.0f;
    if (m_[i] > -1.0e38f) {
      aie::vector<bfloat16, N> cm =
          aie::broadcast<bfloat16, N>((bfloat16)(m_[i] - mnew));
      corr = (float)to_v16bfloat16(getExpBf16(cm))[0];
    } else {
      corr = 0.0f;
    }
    float lsum = corr * l_[i];

    // p = exp(s - mnew), accumulated straight into acc
    for (int d = 0; d < HD; d++)
      acc_[i][d] *= corr;
    for (int j = 0; j < BK; j++) {
      aie::vector<bfloat16, N> sv =
          aie::broadcast<bfloat16, N>((bfloat16)(s[j] - mnew));
      float p = (float)to_v16bfloat16(getExpBf16(sv))[0];
      lsum += p;
      bfloat16 pb = (bfloat16)p;
      for (int d = 0; d < HD; d += N) {
        aie::vector<bfloat16, N> vv = aie::load_v<N>(v + j * HD + d);
        aie::accum<accfloat, N> a =
            aie::mul(vv, aie::broadcast<bfloat16, N>(pb));
        aie::vector<float, N> cur = aie::load_v<N>(&acc_[i][d]);
        aie::store_v(&acc_[i][d], aie::add(a, cur).to_vector<float>());
      }
    }
    m_[i] = mnew;
    l_[i] = lsum;
  }

  if (kb == nkb - 1) {
    for (int i = 0; i < BQ; i++) {
      float inv = 1.0f / l_[i];
      for (int d = 0; d < HD; d++)
        o[i * HD + d] = (bfloat16)(acc_[i][d] * inv);
    }
  }
}

} // extern "C"
