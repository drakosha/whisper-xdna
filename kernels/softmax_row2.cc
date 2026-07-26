//===- softmax_row2.cc ----------------------------------------*- C++ -*-===//
//
// Row-wise softmax for AIE2 with an fp32-accurate normalisation.
//
// v1 (softmax_row.cc) rounds twice on the way out: exp(x-max) is stored bf16,
// and it is then multiplied by a bf16 reciprocal. The reciprocal rounding is the
// damaging one: it is the SAME factor for every element of the row, so it does
// not average out -- it scales the whole attention row by up to ~0.4%. Measured
// end to end: whisper-tiny cosine 0.99913 (numpy softmax) -> 0.99844 (v1),
// under the 0.999 floor.
//
// AIE2 has no fp32 x fp32 vector multiply (aie::mul on float vectors fails to
// link: "undefined symbol: mul_elem_16_conf(float vector[16], ...)"); the vector
// unit multiplies bf16 into an fp32 accumulator. So the reciprocal is split into
// two bf16 halves, inv ~= inv_hi + inv_lo, and applied as two bf16 multiplies
// accumulated in fp32. That recovers ~fp32 precision on the scale factor using
// only the multiplies the hardware actually has.
//
// The final cast to bf16 stays: the following A@V matmul consumes bf16 whatever
// we do, and the numpy path pays exactly the same cast.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

extern "C" {

void softmax_row2_bf16(bfloat16 *restrict in, bfloat16 *restrict out,
                       int32_t n) {
  constexpr int N = 16;
  const int iters = n / N;

  // pass 1: row max, so exp cannot overflow
  auto it_m = aie::cbegin_vector<N>((bfloat16 *)in);
  aie::vector<bfloat16, N> vmax =
      aie::broadcast<bfloat16, N>((bfloat16)-30000.0f);
  for (int i = 0; i < iters; i++) {
    vmax = aie::max(vmax, *it_m++);
  }
  bfloat16 row_max = aie::reduce_max(vmax);
  aie::vector<bfloat16, N> vm = aie::broadcast<bfloat16, N>(row_max);

  // pass 2: exp into the output buffer, sum accumulated in fp32
  auto it_in = aie::cbegin_vector<N>((bfloat16 *)in);
  auto it_e = aie::begin_vector<N>((bfloat16 *)out);
  aie::accum<accfloat, N> acc = aie::zeros<accfloat, N>();
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> x = aie::sub(*it_in++, vm);
    aie::vector<bfloat16, N> e = to_v16bfloat16(getExpBf16(x));
    acc = add(acc, e);
    *it_e++ = e;
  }
  float sum = aie::reduce_add(acc.to_vector<float>());

  // fp32 reciprocal, split into two bf16 pieces so the scaling keeps ~fp32
  // precision through bf16-only vector multiplies
  float inv = aie::inv(sum);
  bfloat16 inv_hi = (bfloat16)inv;
  bfloat16 inv_lo = (bfloat16)(inv - (float)inv_hi);

  // pass 3: p = e * (inv_hi + inv_lo), accumulated in fp32, cast once
  auto it_s = aie::cbegin_restrict_vector<N>((bfloat16 *)out);
  auto it_o = aie::begin_restrict_vector<N>((bfloat16 *)out);
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> e = *it_s++;
    aie::accum<accfloat, N> p = aie::mul(e, inv_hi);
    p = aie::add(p, aie::mul(e, inv_lo));
    *it_o++ = p.to_vector<bfloat16>();
  }
}

} // extern "C"
