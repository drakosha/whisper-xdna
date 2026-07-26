//===- gelu_row4_rne.cc ---------------------------------------*- C++ -*-===//
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

void gelu_row4_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  // Default rounding on this part is truncation toward zero. Measured on real
  // fc1 activations, switching this one line to round-to-nearest-even took the
  // kernel's bias from -2.0e-04 to +4.3e-05 and downstream from 0.00857 to
  // 0.00635 (measured) -- bias is what compounds over 32 layers.
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int N = 16;
  const int iters = n / N;

  const aie::vector<bfloat16, N> zero =
      aie::broadcast<bfloat16, N>((bfloat16)0.0f);
  const aie::vector<bfloat16, N> one =
      aie::broadcast<bfloat16, N>((bfloat16)1.0f);

  auto it_in = aie::cbegin_vector<N>((bfloat16 *)in);
  auto it_out = aie::begin_restrict_vector<N>((bfloat16 *)out);

  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> x = *it_in++;

    *it_out++ = gelu_poly4_eval(x, zero, one);
  }
}

} // extern "C"
