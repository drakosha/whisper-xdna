//===- gelu_row.cc --------------------------------------------*- C++ -*-===//
//
// Size-parameterised GELU for AIE2. The stock aie_kernels/aie2/gelu.cc has a
// vectorised body that already takes a length, but its extern "C" wrapper
// hard-codes 1024 elements, which forces a 32x32 matmul tile when used as a
// fused epilogue. Measured at block level that tile choice costs more in extra
// design switching than the fused epilogue saves, so here GELU is exposed as a
// standalone row op that runs over a whole buffer in one launch.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#include "gelu.cc"   // provides gelu_tanh_approx_bf16(in, out, vector_size)

extern "C" {

void gelu_row_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  gelu_tanh_approx_bf16(in, out, n);
}

} // extern "C"
