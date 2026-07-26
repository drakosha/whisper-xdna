//===- softmax_row4.cc ----------------------------------------*- C++ -*-===//
//
// Row-wise softmax for AIE2, v4: the cheap third pass.
//
// v3 (softmax_row3.cc) makes THREE passes over the row, and the third one
// re-evaluates getExpBf16 from scratch so it can split the exponential into a
// bf16 hi/lo pair and keep both factors of e*inv at ~fp32:
//     p = (e_hi + e_lo) * (inv_hi + inv_lo)
// The LUT evaluation is the expensive part of that pass, and it buys very
// little. Measured on REAL attention scores against the quantity that matters,
// the attention output P@V (measured on 8 heads of large-v3):
//
//   hi/lo e * hi/lo inv (v3)      rel err of P@V  0.000445   worst row gain 0.0019
//   bf16  e * hi/lo inv (this)    rel err of P@V  0.000575   worst row gain 0.0023
//   bf16  e * bf16  inv           rel err of P@V  0.001692   worst row gain 0.0050
//
// So the RECIPROCAL split has to stay -- dropping it is 3.8x worse, because a
// bf16 reciprocal scales the whole row by one factor and that error does not
// average out. The EXPONENTIAL split does not: e is rounded to bf16 on the way
// out of pass 2 anyway, so carrying it as bf16 into pass 3 costs one rounding
// the result already pays. v4 therefore reads back what pass 2 stored instead
// of recomputing it: one LUT evaluation per row instead of two, and two
// multiplies instead of three.
//
// (The best variant measured was no on-device normalisation at all -- store the
// unnormalised exponentials, let the AV GEMM accumulate in fp32 and divide by
// the row sum on the host: rel err 0.000325, better than v3. It is not
// implementable here because the row-wise design has exactly one output buffer
// and no way to hand the 20x1536 row sums back beside the probabilities.)
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

extern "C" {

void softmax_row4_bf16(bfloat16 *restrict in, bfloat16 *restrict out,
                       int32_t n) {
#include "softmax_row4_body.inc"
}

} // extern "C"
