//===- gelu_poly4_eval.inc -------------------------------------*- C++ -*-===//
//
// The erf-GELU bump, evaluated the way g4 does it: shared by gelu_row4.cc
// (bf16 in), gelu_row4_rne.cc (same, round-to-nearest-even) and gelu_row4f.cc
// (fp32 accumulator in). It used to be three copies of these forty lines, and
// a coefficient change had to be made three times to stay consistent.
//
// GELU(x) = relu(x) + q(|x|), q a degree-GELU4_DEG minimax fit of the exact
// erf-GELU on each of GELU4_NPIECE intervals, in a CENTRED basis: u = (s -
// shift) * scale maps the piece onto [-1, 1]. The monomial basis on raw s is
// ill-conditioned and its terms cancel, which is what made "bf16 coefficients
// are catastrophic" look like a hardware limit rather than a basis choice.
//
// Requires the caller to have included gelu_poly4_coeffs.h, and n % N == 0 --
// rowops.py enforces that on the host, where a bad shape is an error message
// instead of an unwritten tail.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Not a template: making N a template parameter turns every acc.to_vector<>
// into a dependent name ("use 'template' keyword to treat 'to_vector' as a
// dependent template name"), and all three kernels vectorise 16 wide anyway.
static constexpr int GELU4_N = 16;

static inline aie::vector<bfloat16, GELU4_N>
gelu_poly4_eval(const aie::vector<bfloat16, GELU4_N> x,
                const aie::vector<bfloat16, GELU4_N> zero,
                const aie::vector<bfloat16, GELU4_N> one) {
  constexpr int N = GELU4_N;
  // s = |x|  (= max(x, -x))
  aie::vector<bfloat16, N> s = aie::max(x, aie::sub(zero, x));

  // Outer region first (s past the last break -> q = 0), overwritten inward so
  // the tightest piece wins. No array of vector registers and no runtime
  // indexing: q is a single value and every coefficient is a compile-time
  // scalar broadcast where it is used (an aie::vector c[5] lowered to garbage).
  aie::vector<bfloat16, N> q = zero;
  for (int p = GELU4_NPIECE - 1; p >= 0; p--) {
    // u = (s - shift) * scale, the interval mapped onto [-1, 1]
    aie::vector<bfloat16, N> u =
        aie::mul(aie::sub(s, aie::broadcast<bfloat16, N>(
                                 (bfloat16)GELU4_SHIFT[p])),
                 aie::broadcast<bfloat16, N>((bfloat16)GELU4_SCALE[p]))
            .to_vector<bfloat16>();

    // Horner in u. The state lives in the fp32 accumulator between steps and is
    // rounded to bf16 only to be multiplied -- one conversion, not the hi/lo
    // pair g3 needed.
    aie::accum<accfloat, N> acc = aie::mul(
        aie::broadcast<bfloat16, N>((bfloat16)GELU4_COEF[p][0]), one);
    for (int k = 1; k <= GELU4_DEG; k++) {
      aie::vector<bfloat16, N> ah = acc.to_vector<bfloat16>();
      acc = aie::mul(aie::broadcast<bfloat16, N>((bfloat16)GELU4_COEF[p][k]),
                     one);
      acc = aie::mac(acc, ah, u);
    }
    aie::mask<N> inside =
        aie::lt(s, aie::broadcast<bfloat16, N>((bfloat16)GELU4_BREAK[p]));
    q = aie::select(q, acc.to_vector<bfloat16>(), inside);
  }

  return aie::add(aie::max(x, zero), q);
}
