//===- gelu_row3.cc -------------------------------------------*- C++ -*-===//
//
// erf-GELU by piecewise polynomial, no exp/tanh table at all.
//
// Design path (all measured on real fc1 activations, projected through the real
// fc2 weight -- the metric that sets end-to-end cosine, not raw L2):
//
//  * The residual of the earlier tanh kernels is NOT the surrogate (its
//    downstream weight is 0.0012, it is large only in the deep tail that
//    carries ~0 output energy) and NOT an exp-table amplifier (sensitivity is
//    1.0x). It is the exp+Newton arithmetic. So drop the table, target erf.
//  * An fp64 piecewise poly reaches downstream 0.004 -- 4x better than g2's
//    0.0158 -- so the polynomial route has the headroom. The bf16-input floor
//    is downstream 0.0033.
//  * The bottleneck turned out to be COEFFICIENT precision: bf16 coefficients
//    give downstream 0.94 (!). Carrying each coefficient as a bf16 hi/lo pair
//    recovers it to 0.006, matching fp32 coefficients (0.006). The Horner state
//    is likewise kept in the fp32 accumulator, split hi/lo only for the
//    multiply -- the exact technique softmax_row3.cc uses for its reciprocal.
//
// Reduction: G(x) = relu(x) + q(|x|), q(s)=G(-s) a bounded bump (0 at 0, min
// ~-0.17 near 0.75, ~0 by s=4). relu is exact, so only q is approximated; q has
// no cancellation and no x^3 (no overflow branch). Coefficients in
// gelu_poly_coeffs.h (gen_gelu_coeffs.py), the same numbers the numpy validator
// uses so the kernel cannot silently diverge from its model.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gelu_poly_coeffs.h"

using namespace aie;

extern "C" {

void gelu_row3_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
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

    // s = |x|  (=-min(x,-x))
    aie::vector<bfloat16, N> negx = aie::sub(zero, x);
    aie::vector<bfloat16, N> s = aie::max(x, negx);

    // q(s): one Horner per piece, result selected by range. No array of vector
    // registers -- that lowered to garbage -- and no runtime-indexed vectors:
    // q and the accumulator are single values, coefficients are compile-time
    // scalars broadcast on use. Outer region first (s >= last break -> q = 0),
    // overwritten inward so the tightest piece wins.
    aie::vector<bfloat16, N> q = zero;
    for (int p = GELU_NPIECE - 1; p >= 0; p--) {
      // Horner with the state carried in the fp32 accumulator, split into a
      // bf16 hi/lo pair only for each multiply, and coefficients added as their
      // own hi/lo pair: acc <- acc*s + c, computed as
      //   (acc_hi + acc_lo)*s + (c_hi + c_lo)
      // four bf16 muls into one fp32 accumulate, no bf16 rounding of the state
      // between steps.
      aie::accum<accfloat, N> acc;
      acc = aie::add(aie::mul(aie::broadcast<bfloat16, N>(
                                  (bfloat16)GELU_COEF_HI[p][0]),
                              one),
                     aie::mul(aie::broadcast<bfloat16, N>(
                                  (bfloat16)GELU_COEF_LO[p][0]),
                              one));
      for (int k = 1; k <= GELU_DEG; k++) {
        aie::vector<bfloat16, N> ah = acc.to_vector<bfloat16>();
        aie::vector<bfloat16, N> al =
            aie::sub(acc, aie::mul(ah, one)).to_vector<bfloat16>();
        aie::accum<accfloat, N> a = aie::mul(ah, s);
        a = aie::add(a, aie::mul(al, s));
        a = aie::add(a, aie::mul(aie::broadcast<bfloat16, N>(
                                     (bfloat16)GELU_COEF_HI[p][k]),
                                 one));
        a = aie::add(a, aie::mul(aie::broadcast<bfloat16, N>(
                                     (bfloat16)GELU_COEF_LO[p][k]),
                                 one));
        acc = a;
      }
      aie::vector<bfloat16, N> qp = acc.to_vector<bfloat16>();
      aie::mask<N> inside =
          aie::lt(s, aie::broadcast<bfloat16, N>((bfloat16)GELU_BREAK[p]));
      q = aie::select(q, qp, inside);
    }

    // GELU = relu(x) + q(|x|)
    aie::vector<bfloat16, N> relu = aie::max(x, zero);
    *it_out++ = aie::add(relu, q);
  }
}

} // extern "C"
