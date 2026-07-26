//===- attn_g_b_f32.inc ----------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// Modifications Copyright (c) 2026 Mikhail Kostryukov
//
// Derived from matmul_g_b_bf16 in AMD's flash-attention example
// (programming_examples/flash_attention/, https://github.com/Xilinx/mlir-air).
// The one change is the accumulator: identical
// to the original except the C operand is fp32. Theirs rounds the attention
// accumulator to bf16 on every one of the 48 K/V blocks, which costs
// rel L2 0.12 over a head -- the same bf16-accumulation trap
// an earlier fused design hit. A and B stay bf16; only C changes.
__attribute__((noinline)) static void matmul_g_b_f32(bfloat16 *g_in, bfloat16 *b_in, float *out) {
  // Buffer shapes:
  // G: [lqp, lkp] (attention scores, column-major 4x4 tiled)
  // B: [lkp, dv] (V chunk, no software transpose)
  // Out: [lqp, dv] (attention output, column-major 4x4 tiled)
  //
  // G is in 4x4 column-major block layout (from QK matmul C output):
  //   Block [rb, cb] at g_in + rb * size_C + cb * rowA_C * size_C
  //   where size_C = r*t = 16, rowA_C = lqp/r = lqp/4.
  // But mmul<4,8,4> needs A in 4x8 block format (size_A = r*s = 32).
  // We load two adjacent column-blocks and interleave them into a 4x8 sub-tile.
  //
  // matmul: G[lqp, lkp] x V[lkp, dv] -> Out[lqp, dv]
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  constexpr unsigned rowA = lqp / r; // number of row-blocks of A/C
  constexpr unsigned colA = lkp / s; // number of k-blocks (A is 4x8)
  constexpr unsigned colB = dv / t;  // number of n-blocks of B/C
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

  // 4x4 C-block layout parameters
  constexpr unsigned size_C_blk = r * t; // 16 elements per 4x4 block
  constexpr unsigned col_block_stride =
      rowA * size_C_blk; // stride between column-blocks = 16*16 = 256

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      float *__restrict pC1 = out + (z)*MMUL::size_C;
      float *__restrict pC2 = out + ((z + 1)) * MMUL::size_C;
      float *__restrict pC3 = out + ((z + 2)) * MMUL::size_C;
      float *__restrict pC4 = out + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4) {
        const bfloat16 *__restrict pB1 = b_in + (j)*colA * MMUL::size_B;
        const bfloat16 *__restrict pB2 = b_in + ((j + 1)) * colA * MMUL::size_B;
        const bfloat16 *__restrict pB3 = b_in + ((j + 2)) * colA * MMUL::size_B;
        const bfloat16 *__restrict pB4 = b_in + ((j + 3)) * colA * MMUL::size_B;

        // Load A from 4x4 block format: read two 4x4 blocks, interleave to 4x8
        // For A sub-tile [z, i=0]: read C[rb=z, cb=0] and C[rb=z, cb=1]
        auto load_A_4x4 =
            [&](unsigned rb,
                unsigned kb) -> aie::vector<bfloat16, MMUL::size_A> {
          const bfloat16 *pLo =
              g_in + rb * size_C_blk + (2 * kb) * col_block_stride;
          const bfloat16 *pHi =
              g_in + rb * size_C_blk + (2 * kb + 1) * col_block_stride;
          aie::vector<bfloat16, 16> lo = aie::load_v<16>(pLo);
          aie::vector<bfloat16, 16> hi = aie::load_v<16>(pHi);
          // interleave_zip with step=4: takes alternating groups of 4 from lo,
          // hi lo = [r0c0..3 r1c0..3 r2c0..3 r3c0..3] hi = [r0c4..7 r1c4..7
          // r2c4..7 r3c4..7] result_lo = [r0c0..3 r0c4..7 r1c0..3 r1c4..7]
          // (rows 0-1, 8 cols) result_hi = [r2c0..3 r2c4..7 r3c0..3 r3c4..7]
          // (rows 2-3, 8 cols)
          auto [zlo, zhi] = aie::interleave_zip(lo, hi, 4);
          return aie::concat(zlo, zhi);
        };

        aie::vector<bfloat16, MMUL::size_A> A0 = load_A_4x4(z, 0);
        aie::vector<bfloat16, MMUL::size_A> A1 = load_A_4x4(z + 1, 0);
        aie::vector<bfloat16, MMUL::size_A> A2 = load_A_4x4(z + 2, 0);
        aie::vector<bfloat16, MMUL::size_A> A3 = load_A_4x4(z + 3, 0);

        aie::vector<bfloat16, MMUL::size_B> B0, B1, B2, B3;
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
        B2 = aie::load_v<MMUL::size_B>(pB3);
        B3 = aie::load_v<MMUL::size_B>(pB4);
        pB1 += MMUL::size_B;
        pB2 += MMUL::size_B;
        pB3 += MMUL::size_B;
        pB4 += MMUL::size_B;

        aie::vector<float, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<float, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C02 =
            aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C03 =
            aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

        aie::vector<float, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<float, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C12 =
            aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C13 =
            aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

        aie::vector<float, MMUL::size_C> acc_C20 =
            aie::load_v<MMUL::size_C>(pC3);
        aie::vector<float, MMUL::size_C> acc_C21 =
            aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C22 =
            aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C23 =
            aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

        aie::vector<float, MMUL::size_C> acc_C30 =
            aie::load_v<MMUL::size_C>(pC4);
        aie::vector<float, MMUL::size_C> acc_C31 =
            aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C32 =
            aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
        aie::vector<float, MMUL::size_C> acc_C33 =
            aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C02(acc_C02);
        MMUL C03(acc_C03);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C12(acc_C12);
        MMUL C13(acc_C13);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C22(acc_C22);
        MMUL C23(acc_C23);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);
        MMUL C32(acc_C32);
        MMUL C33(acc_C33);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
        C02.mac(A0, B2);
        C03.mac(A0, B3);
        C12.mac(A1, B2);
        C13.mac(A1, B3);
        C20.mac(A2, B0);
        C21.mac(A2, B1);
        C30.mac(A3, B0);
        C31.mac(A3, B1);
        C22.mac(A2, B2);
        C23.mac(A2, B3);
        C32.mac(A3, B2);
        C33.mac(A3, B3);

        for (unsigned i = 1; i < colA; ++i) {
          A0 = load_A_4x4(z, i);
          A1 = load_A_4x4(z + 1, i);
          A2 = load_A_4x4(z + 2, i);
          A3 = load_A_4x4(z + 3, i);

          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
          B2 = aie::load_v<MMUL::size_B>(pB3);
          B3 = aie::load_v<MMUL::size_B>(pB4);
          pB1 += MMUL::size_B;
          pB2 += MMUL::size_B;
          pB3 += MMUL::size_B;
          pB4 += MMUL::size_B;

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);
          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);
          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);
        }

        aie::store_v(pC1, C00.template to_vector<float>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C01.template to_vector<float>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C02.template to_vector<float>());
        pC1 += MMUL::size_C * rowA;
        aie::store_v(pC1, C03.template to_vector<float>());
        pC1 += MMUL::size_C * rowA;

        aie::store_v(pC2, C10.template to_vector<float>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C11.template to_vector<float>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C12.template to_vector<float>());
        pC2 += MMUL::size_C * rowA;
        aie::store_v(pC2, C13.template to_vector<float>());
        pC2 += MMUL::size_C * rowA;

        aie::store_v(pC3, C20.template to_vector<float>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C21.template to_vector<float>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C22.template to_vector<float>());
        pC3 += MMUL::size_C * rowA;
        aie::store_v(pC3, C23.template to_vector<float>());
        pC3 += MMUL::size_C * rowA;

        aie::store_v(pC4, C30.template to_vector<float>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C31.template to_vector<float>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C32.template to_vector<float>());
        pC4 += MMUL::size_C * rowA;
        aie::store_v(pC4, C33.template to_vector<float>());
        pC4 += MMUL::size_C * rowA;
      }
    }

  event1();
}
