//===- softmax_row.cc -----------------------------------------*- C++ -*-===//
//
// Row-wise softmax over a full attention row for AIE2 (Phoenix / XDNA1).
//
// The stock aie_kernels/aie2/softmax.cc reduces correctly over its whole
// `vector_size`, but the stock ml/softmax *design* feeds it fixed 1024-element
// tiles via transform_parallel, so a 512- or 1536-wide attention row is split
// across tiles and the normalisation is wrong. Here the design hands the kernel
// one complete row, and the kernel is given a max-subtraction pass that the
// stock one lacks (it calls getExpBf16 on the raw logits, which overflows once
// a score exceeds ~88).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

extern "C" {

// out[i] = exp(in[i] - max(in)) / sum_j exp(in[j] - max(in)), over the full row
void softmax_row_bf16(bfloat16 *restrict in, bfloat16 *restrict out,
                      int32_t n) {
  constexpr int N = 16;
  const int iters = n / N;

  // pass 1: row max (this is what makes it safe against large logits)
  auto it_m = aie::cbegin_vector<N>((bfloat16 *)in);
  aie::vector<bfloat16, N> vmax = aie::broadcast<bfloat16, N>((bfloat16)-30000.0f);
  for (int i = 0; i < iters; i++) {
    vmax = aie::max(vmax, *it_m++);
  }
  bfloat16 row_max = aie::reduce_max(vmax);
  aie::vector<bfloat16, N> vm = aie::broadcast<bfloat16, N>(row_max);

  // pass 2: exp(x - max), keep a float running sum
  auto it_in = aie::cbegin_vector<N>((bfloat16 *)in);
  auto it_out = aie::begin_vector<N>((bfloat16 *)out);
  aie::accum<accfloat, N> acc = aie::zeros<accfloat, N>();
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> x = aie::sub(*it_in++, vm);
    aie::vector<bfloat16, N> e = to_v16bfloat16(getExpBf16(x));
    acc = add(acc, e);
    *it_out++ = e;
  }
  float sum = aie::reduce_add(acc.to_vector<float>());
  bfloat16 inv = (bfloat16)aie::inv(sum);

  // pass 3: normalise
  auto it_s = aie::cbegin_restrict_vector<N>((bfloat16 *)out);
  auto it_o = aie::begin_restrict_vector<N>((bfloat16 *)out);
  for (int i = 0; i < iters; i++) {
    aie::accum<accfloat, N> v = aie::mul(*it_s++, inv);
    *it_o++ = v.to_vector<bfloat16>();
  }
}

} // extern "C"
