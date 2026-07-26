//===- attn_mmul.cc -------------------------------------------*- C++ -*-===//
//
// One attention head on AMD's mmul-based kernel functions, wrapped in a SINGLE
// entry point.
//
// Why a single entry point: IRON compiles one object per declared
// ExternalFunction, from the same source, so declaring their six functions
// separately links six copies of the same file --
//   ld.lld: error: duplicate symbol: accum_sp_r_s
// Wrapping the whole per-block step in one function keeps it to one object and
// puts the running state where it belongs, in core memory.
//
// The step, exactly their sequence (attn_npu1.py, the inner loop):
//     zero_fill_g   -> g = 0
//     matmul_a_b    -> g = Q @ K^T          (mmul<4,8,4>, 128 MAC/instr)
//     fused_softmax -> scale by 1/sqrt(dk), running max, exp, row sums
//     mul_r_gp      -> gp *= correction
//     matmul_g_b    -> gp += g @ V
//     accum_sp_r_s  -> s += sp * r ; copy back into sp
// and once at the end div_gp_sp -> gp /= sp.
//
// NB the 1/sqrt(dk) scaling lives in scale_g_bf16, so Q and K must arrive
// UNSCALED -- our chain path scales both by hd**-0.25 on the host, which would
// double-apply it here.
//
// Operands arrive already in the 4x8 / 8x4 / 4x4 block layouts mmul wants; that
// tiling is done once on the host (blocklayout.py), where it replaces the
// per-head transpose the old chain was doing anyway.
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// Modifications Copyright (c) 2026 Mikhail Kostryukov
// Wrapper around AMD's attn_npu1.cc; the originals live in
// https://github.com/Xilinx/mlir-air (MIT).
//===----------------------------------------------------------------------===//

#include "attn_npu1.cc"
#include "attn_g_b_f32.inc"

// --- fp32 accumulator support -------------------------------------------
// Their gp is bf16 and is read-modify-written once per K/V block, so a head
// rounds the attention output 48 times. Measured cost: rel L2 0.12 against
// exact attention, per-row cosine 0.9957. Our budget is 0.00007 to
// the bf16 floor, so gp moves to fp32.
//
// The catch is that AIE2 has no fp32 vector multiply, and the rescale step
// gp *= r needs one. Same answer as everywhere else in this project: carry the
// fp32 value as a bf16 hi/lo pair for the multiply and accumulate in fp32.
// noinline on every helper below: without it the 219-line unrolled fp32
// matmul lands inside attn_step and the core's .text hits 21456 bytes against
// 16384 available -- the loader then rejects the ELF with
//   XAie_LoadElf failed with XAIE_INVALID_ELF
constexpr int GPVEC = 16;

__attribute__((noinline)) static void zero_fill_gp_f32(float *gp) {
  aie::vector<float, GPVEC> z = aie::zeros<float, GPVEC>();
  for (int i = 0; i < lqp * dv; i += GPVEC)
    aie::store_v(gp + i, z);
}

// gp *= r, with gp in fp32 and r one bf16 value per row.
// gp is 4x4-block tiled: block (rb, cb) at (cb*rowA + rb)*16, four rows of four
// inside. The multiplier for a block is r[rb*4+0..3] each repeated four times.
// Building that by writing vector ELEMENTS was the whole cost of the first
// attempt -- 1.05 ms/head became 7.98. Expanding r into memory once per call
// and loading it back as a vector is the same data for a fraction of the price.
static bfloat16 rexp_[lqp * 4];

__attribute__((noinline)) static void mul_r_gp_f32(bfloat16 *r, float *gp) {
  constexpr int rowA = lqp / 4;
  constexpr int colB = dv / 4;
  for (int i = 0; i < lqp; i++) {
    bfloat16 rv = r[i];
    rexp_[i * 4 + 0] = rv;
    rexp_[i * 4 + 1] = rv;
    rexp_[i * 4 + 2] = rv;
    rexp_[i * 4 + 3] = rv;
  }
  aie::vector<bfloat16, GPVEC> one =
      aie::broadcast<bfloat16, GPVEC>((bfloat16)1.0f);
  for (int rb = 0; rb < rowA; rb++) {
    aie::vector<bfloat16, GPVEC> rv = aie::load_v<GPVEC>(rexp_ + rb * GPVEC);
    for (int cb = 0; cb < colB; cb++) {
      float *p = gp + (cb * rowA + rb) * GPVEC;
      aie::accum<accfloat, GPVEC> a;
      a.from_vector(aie::load_v<GPVEC>(p));
      aie::vector<bfloat16, GPVEC> hi = a.to_vector<bfloat16>();
      aie::vector<bfloat16, GPVEC> lo =
          aie::sub(a, aie::mul(hi, one)).to_vector<bfloat16>();
      aie::accum<accfloat, GPVEC> q = aie::mul(hi, rv);
      q = aie::add(q, aie::mul(lo, rv));
      aie::store_v(p, q.to_vector<float>());
    }
  }
}

// The running row sum, in fp32. Their accum_sp_r_s keeps it in bf16 and
// read-modify-writes it once per block, so it collects 48 roundings -- and
// unlike a per-element error this one scales the WHOLE row through the final
// divide, exactly the reason softmax_row3.cc carries its reciprocal hi/lo.
// Measured: with gp already in fp32 the head still scored rel L2 0.072 against
// the chain's 0.0068, and this is what was left.
//   spf = spf * r + s
__attribute__((noinline)) static void zero_fill_sp_f32(float *spf) {
  aie::vector<float, GPVEC> z = aie::zeros<float, GPVEC>();
  for (int i = 0; i < lqp; i += GPVEC)
    aie::store_v(spf + i, z);
}

__attribute__((noinline)) static void accum_sp_f32(float *spf, bfloat16 *r, bfloat16 *sblk) {
  aie::vector<bfloat16, GPVEC> one =
      aie::broadcast<bfloat16, GPVEC>((bfloat16)1.0f);
  for (int i = 0; i < lqp; i += GPVEC) {
    aie::accum<accfloat, GPVEC> a;
    a.from_vector(aie::load_v<GPVEC>(spf + i));
    aie::vector<bfloat16, GPVEC> hi = a.to_vector<bfloat16>();
    aie::vector<bfloat16, GPVEC> lo =
        aie::sub(a, aie::mul(hi, one)).to_vector<bfloat16>();
    aie::vector<bfloat16, GPVEC> rv = aie::load_v<GPVEC>(r + i);
    aie::accum<accfloat, GPVEC> q = aie::mul(hi, rv);
    q = aie::add(q, aie::mul(lo, rv));
    q = aie::add(q, aie::mul(aie::load_v<GPVEC>(sblk + i), one));
    aie::store_v(spf + i, q.to_vector<float>());
  }
}

// o = gp / sp, fp32 in, bf16 out. Runs once per wave, not per block, so the
// scalar reciprocal per row is free -- and it removes the need for a fp32
// vector divide entirely.
__attribute__((noinline)) static void finalize_gp(float *sp, float *gp, bfloat16 *o) {
  constexpr int rowA = lqp / 4;
  constexpr int colB = dv / 4;
  for (int rb = 0; rb < rowA; rb++)
    for (int i = 0; i < 4; i++) {
      float inv = 1.0f / sp[rb * 4 + i];
      for (int cb = 0; cb < colB; cb++) {
        int base = (cb * rowA + rb) * GPVEC + i * 4;
        for (int j = 0; j < 4; j++)
          o[base + j] = (bfloat16)(gp[base + j] * inv);
      }
    }
}

// running state for one Q block, alive across the calls of one wave
static bfloat16 g_[lqp * lkp];
static float gp_[lqp * dv];
static float sp_[lqp];
static bfloat16 up_[lqp];
static bfloat16 r_[lqp];
static bfloat16 s_[lqp];

// Keys past the real context must not take probability mass. The GEMM pads
// ctx 1500 -> 1536, and a padded K row is ZERO, not -inf: its score is 0, and
// exp(0 - max) is not negligible. The chain solves this with a windowed
// softmax (1.4% of the average row's mass, 8.9% on the worst row, measured in
// measured); here the block that straddles the boundary is masked
// before the exp, which is exact rather than merely narrower.
#ifndef ATTN_NVALID
#define ATTN_NVALID lkp *(1 << 20)
#endif

// Keys past the real context must not take probability mass: the GEMM pads
// ctx 1500 -> 1536, and a padded K row is ZERO, so its score is 0 and
// exp(0 - max) is not negligible. Measured end to end, leaving it unmasked
// costs cosine 0.99486 -> 0.98552, while at a padding-free context (768) the
// fused path matches the chain to 0.0002 -- so this is the whole gap.
//
// Third mechanism, after two failures. Writing a sentinel into the scores
// BEFORE the softmax did not work with either -inf or -30000 (rel L2 1.0, and
// three times slower, at every active nvalid), so this does not touch the
// exponential at all: it zeroes the offending PROBABILITIES afterwards and
// recomputes the row sums with their own sum_g. Exact, and nothing but a zero
// store is involved.
static void drop_pad_cols(bfloat16 *g, int kb, bfloat16 *s) {
  constexpr int nvalid = ATTN_NVALID;
  constexpr int rowA = lqp / 4;
  int first = kb * lkp;
  if (first + lkp <= nvalid)
    return;
  bfloat16 zero = (bfloat16)0.0f;
  for (int c = 0; c < lkp; c++) {
    if (first + c < nvalid)
      continue;
    int cb = c / 4, c_in = c % 4;
    for (int rb = 0; rb < rowA; rb++)
      for (int r_in = 0; r_in < 4; r_in++)
        g[(cb * rowA + rb) * 16 + r_in * 4 + c_in] = zero;
  }
  sum_g(g, s);          // row sums again, now without the padded columns
}

extern "C" {

// layout probe: Q @ K^T only, so the block layouts can be checked on their own
void attn_qk(bfloat16 *restrict q, bfloat16 *restrict k,
             bfloat16 *restrict g) {
  zero_fill_g_bf16(g);
  matmul_a_b_bf16(q, k, g);
}

void attn_step(bfloat16 *restrict q, bfloat16 *restrict kv,
               bfloat16 *restrict o, int32_t kb, int32_t nkb) {
  bfloat16 *k = kv;
  bfloat16 *v = kv + lkp * dk;

  if (kb == 0) {
    zero_fill_gp_f32(gp_);
    zero_fill_sp_f32(sp_);
    neg_inf_fill_up_bf16(up_);
  }

  zero_fill_g_bf16(g_);
  matmul_a_b_bf16(q, k, g_);
  fused_softmax(g_, up_, s_, r_);
  drop_pad_cols(g_, kb, s_);
  mul_r_gp_f32(r_, gp_);
  matmul_g_b_f32(g_, v, gp_);
  accum_sp_f32(sp_, r_, s_);

  if (kb == nkb - 1)
    finalize_gp(sp_, gp_, o);
}

} // extern "C"
