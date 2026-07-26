//===- softmax_row4_body.inc -----------------------------------*- C++ -*-===//
//
// The row-wise softmax v4 body, shared by softmax_row4.cc and its
// round-to-nearest-even twin softmax_row4_rne.cc. The two differed by a single
// aie::set_rounding call and were otherwise the same ninety lines.
//
// Three passes over the row: max, exp into the output buffer with the sum
// accumulated in fp32, then scale in place. v3's second getExpBf16 and its
// hi/lo split of e are gone -- the reciprocal alone carries the precision.
//
// Requires n % N == 0; rowops.py checks that on the host.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

  // pass 3: scale what pass 2 already stored. No second getExpBf16 and no
  // e hi/lo split -- see the header for the numbers that justify dropping them.
  //   p = e * (inv_hi + inv_lo),  two bf16 multiplies into one fp32 accumulator
  aie::vector<bfloat16, N> ihi = aie::broadcast<bfloat16, N>(inv_hi);
  aie::vector<bfloat16, N> ilo = aie::broadcast<bfloat16, N>(inv_lo);
  auto it_r = aie::cbegin_vector<N>((bfloat16 *)out);
  auto it_o = aie::begin_vector<N>((bfloat16 *)out);
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> e = *it_r++;
    aie::accum<accfloat, N> p = aie::mul(e, ihi);
    p = aie::add(p, aie::mul(e, ilo));
    *it_o++ = p.to_vector<bfloat16>();
  }
