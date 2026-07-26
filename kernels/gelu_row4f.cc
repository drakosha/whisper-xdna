//===- gelu_row4f.cc -------------------------------------------*- C++ -*-===//
//
// gelu_row4 with an fp32 INPUT: the same polynomial, reading the fc1 GEMM's
// fp32 accumulator straight out of device memory and writing bf16 for fc2.
//
// Why the input side matters: with a bf16 input the fc1 result has to come back
// to the host, be converted, and be sent out again -- 241 ms of host conversion
// per encoder pass (mlp_stage) plus the readback. Reading fp32 here keeps the
// whole MLP on the device. The arithmetic is unchanged: the first thing the
// kernel does is round the input to bf16, which is exactly what the host
// conversion did, so this is bit-identical to gelu_row4 fed the same values.
//
// AIE2 has no fp32 vector multiply, but none is needed -- the fp32 vector is
// only LOADED and converted; every multiply below is still bf16 into an fp32
// accumulator.
//
// erf-GELU by piecewise polynomial, cheap form. Same reduction as gelu_row3.cc
//
//     GELU(x) = relu(x) + q(|x|),   q(s) = G(-s)
//
// so the linear part stays exact and only the bounded bump is approximated. What
// changes is the basis: each piece maps its interval onto u in [-1, 1] before
// the Horner run.
//
// That one substitution removes both of g3's costs at once (measured,
// scored through the real fc2 weight, which is what sets end-to-end cosine):
//
//   * g3 needed the fp32 Horner STATE carried as a bf16 hi/lo pair -- rounding
//     it to plain bf16 there cost 9x downstream (0.00663 -> 0.06098). On u it
//     costs nothing (0.00670), because the state never leaves O(1).
//   * g3 needed hi/lo COEFFICIENTS -- bf16 ones gave downstream 0.94.
//     On u they are fine (0.00681): the 0.94 was the monomial
//     basis on raw s being ill-conditioned, not a limit of bf16.
//   * With the state well behaved, three pieces of degree 4 match five
//     (0.00664 vs 0.00663), so it is 12 Horner steps instead of 20.
//
// Net per element: 12 steps of {convert, multiply, MAC} against 20 steps of
// {2 converts, subtract, 6 multiplies, 3 adds}.
//
// Coefficients come from gelu_poly4_coeffs.h (gen_gelu4_coeffs.py), the same
// numbers the numpy validator reads, so the kernel cannot silently diverge from
// its model. As in g3 there is no exp, no lookup table, no clamp and no x**3,
// and the result is finite on the whole bf16 range.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gelu_poly4_coeffs.h"
#include "gelu_poly4_eval.inc"

using namespace aie;

extern "C" {

void gelu_row4f_bf16(float *restrict in, bfloat16 *restrict out, int32_t n) {
  // The fp32 -> bf16 conversion below truncates toward zero under the default
  // rounding mode: measured against gelu_row4 on the same values, 94% of the
  // differing elements came out smaller in magnitude, mean -3.35e-03 -- against
  // g4's own bias of -1.9e-04 over the exact erf. Bias is precisely what
  // compounds over 32 layers, so round to nearest even instead.
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int N = 16;
  const int iters = n / N;

  const aie::vector<bfloat16, N> zero =
      aie::broadcast<bfloat16, N>((bfloat16)0.0f);
  const aie::vector<bfloat16, N> one =
      aie::broadcast<bfloat16, N>((bfloat16)1.0f);

  auto it_in = aie::cbegin_vector<N>((float *)in);
  auto it_out = aie::begin_restrict_vector<N>((bfloat16 *)out);

  for (int i = 0; i < iters; i++) {
    // load fp32, round to bf16 -- the only difference from gelu_row4.cc
    aie::accum<accfloat, N> xin;
    xin.from_vector(*it_in++);
    aie::vector<bfloat16, N> x = xin.to_vector<bfloat16>();

    *it_out++ = gelu_poly4_eval(x, zero, one);
  }
}

} // extern "C"
