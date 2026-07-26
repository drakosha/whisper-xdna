//===- gelu_row2.cc -------------------------------------------*- C++ -*-===//
//
// GELU for AIE2 without the cancellation that makes the stock kernel unusable
// at depth.
//
// The stock kernel (aie_kernels/aie2/gelu.cc) computes 0.5*x*(1 + tanh(u)) and
// getTanhBf16 returns a bf16 VECTOR, so tanh arrives with bf16 relative
// precision. On the activations whisper's fc1 actually produces -- mean -1.99,
// 99.7% of the mass below x=1 -- that is fatal. At x=-2, tanh(u) = -0.96 with a
// bf16 ULP of ~0.004, so 1 + tanh(u) = 0.04 +/- 0.004: catastrophic
// cancellation, 10% error. Measured over real activations, the stock form is
// 95% wrong across x in [-6,-3), which is 32% of all elements, and end to end
// it drags large-v3 to cosine 0.95131 against 0.99649 for a numpy GELU.
//
// No rearrangement of the surrounding arithmetic fixes that, because the error
// is already inside tanh before anything else happens. So this uses the
// logistic form, which never subtracts nearly-equal quantities:
//
//     2u = sqrt(2/pi)*2*(x + 0.044715*x^3)
//     m  = exp(-|2u|)   in (0, 1]        d = 1 + m   in (1, 2]
//     gelu = x / d        when u >= 0
//     gelu = x * m / d    when u <  0
//
// d is a sum of positives. exp comes from getExpBf16, which returns an fp32
// accumulator, so m carries more than bf16 into it. The reciprocal is two
// Newton steps on h = d/2 in (0.5,1] from the standard 48/17 - 32/17*h seed
// (<=5.9% error, and squaring twice takes that to ~1e-5, well under bf16).
// Seeding on d directly would go negative at d=2 and diverge.
//
// x^2 is split hi/lo so x^3 keeps ~fp32 through a multiplier that only takes
// bf16 operands -- the same trick softmax_row3.cc uses for its reciprocal.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

extern "C" {

void gelu_row2_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  constexpr int N = 16;
  const int iters = n / N;

  const aie::vector<bfloat16, N> one =
      aie::broadcast<bfloat16, N>((bfloat16)1.0f);
  const aie::vector<bfloat16, N> half =
      aie::broadcast<bfloat16, N>((bfloat16)0.5f);
  const aie::vector<bfloat16, N> zero =
      aie::broadcast<bfloat16, N>((bfloat16)0.0f);
  // 2*sqrt(2/pi) and 2*sqrt(2/pi)*0.044715
  const aie::vector<bfloat16, N> vc =
      aie::broadcast<bfloat16, N>((bfloat16)1.5957691f);
  const aie::vector<bfloat16, N> vcb =
      aie::broadcast<bfloat16, N>((bfloat16)0.07135439f);
  const aie::vector<bfloat16, N> k48 =
      aie::broadcast<bfloat16, N>((bfloat16)(48.0f / 17.0f));
  const aie::vector<bfloat16, N> k32 =
      aie::broadcast<bfloat16, N>((bfloat16)(32.0f / 17.0f));
  const aie::vector<bfloat16, N> two =
      aie::broadcast<bfloat16, N>((bfloat16)2.0f);

  auto it_in = aie::cbegin_vector<N>((bfloat16 *)in);
  auto it_out = aie::begin_restrict_vector<N>((bfloat16 *)out);

  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, N> x = *it_in++;

    // x^2 as an fp32 accumulator, split into a bf16 pair so the cube keeps
    // precision the bf16-only multiplier could not carry on its own
    aie::accum<accfloat, N> a2 = aie::mul(x, x);
    aie::vector<bfloat16, N> x2h = a2.to_vector<bfloat16>();
    aie::vector<bfloat16, N> x2l =
        aie::sub(a2, aie::mul(x2h, one)).to_vector<bfloat16>();

    // x^3 = x*(x2h + x2l), accumulated in fp32
    aie::accum<accfloat, N> a3 = aie::mul(x, x2h);
    a3 = aie::add(a3, aie::mul(x, x2l));
    aie::vector<bfloat16, N> x3 = a3.to_vector<bfloat16>();

    // 2u = c*x + c*beta*x^3, one rounding at the end (the exp LUT wants bf16)
    aie::accum<accfloat, N> au = aie::mul(x, vc);
    au = aie::add(au, aie::mul(x3, vcb));
    aie::vector<bfloat16, N> u2 = au.to_vector<bfloat16>();

    // m = exp(-|2u|): -|v| is min(v, -v), no abs needed.
    //
    // The argument MUST be clamped. getExpBf16's table is only valid down to
    // about -142; past that it aliases rather than saturating, returning
    // garbage in periodic bands of the input. Measured: the kernel is exact for
    // |x| <= 12 (|2u| <= 142) and produces +-inf beyond, in stripes such as
    // x in [-30,-29.69] and [-28.94,-28.31] -- the signature of an index that
    // wraps, not one that clips. The clamp costs nothing mathematically: at
    // the clamp, m = exp(-80) = 1.8e-35 -- NOT zero in bf16 (bf16 underflows
    // near exp(-92.9); 1.8e-35 is a normal number), but the OUTPUT x*m/(1+m) =
    // x*1.8e-35 is negligible, which is the right answer since gelu of a large
    // negative x is ~0. A branch returning 0/x by sign would be cleaner than a
    // threshold; g3 (gelu_row3.cc) drops exp entirely and has no such edge.
    aie::vector<bfloat16, N> nu2 = aie::sub(zero, u2);
    aie::vector<bfloat16, N> nabs = aie::min(u2, nu2);
    nabs = aie::max(nabs, aie::broadcast<bfloat16, N>((bfloat16)-80.0f));
    aie::accum<accfloat, N> am = getExpBf16(nabs);
    aie::vector<bfloat16, N> m = am.to_vector<bfloat16>();

    // h = (1 + m)/2 in (0.5, 1]; halving is exact
    aie::accum<accfloat, N> ad = aie::add(am, one);
    aie::vector<bfloat16, N> d = ad.to_vector<bfloat16>();
    aie::vector<bfloat16, N> h = aie::mul(d, half).to_vector<bfloat16>();

    // r ~ 1/h: linear seed then two Newton steps r <- r*(2 - h*r)
    aie::vector<bfloat16, N> r =
        aie::sub(aie::mul(k48, one), aie::mul(k32, h)).to_vector<bfloat16>();
    for (int k = 0; k < 2; k++) {
      aie::vector<bfloat16, N> t =
          aie::sub(aie::mul(two, one), aie::mul(h, r)).to_vector<bfloat16>();
      r = aie::mul(r, t).to_vector<bfloat16>();
    }
    // 1/d = 0.5 * (1/h)
    aie::vector<bfloat16, N> rd = aie::mul(r, half).to_vector<bfloat16>();

    // numerator is m on the negative branch, 1 on the positive one
    aie::mask<N> neg = aie::lt(u2, zero);
    aie::vector<bfloat16, N> num = aie::select(one, m, neg);

    aie::vector<bfloat16, N> sig = aie::mul(num, rd).to_vector<bfloat16>();
    *it_out++ = aie::mul(x, sig).to_vector<bfloat16>();
  }
}

} // extern "C"
