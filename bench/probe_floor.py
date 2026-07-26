#!/usr/bin/env python3
"""Compute floor for the whisper-tiny encoder on XDNA1.

Measures each distinct GEMM shape with its own design run back-to-back (no
xclbin switching, buffers preallocated), then sums per the encoder's call
counts. This is the best case the NPU can do for this decomposition -- zero
switching, zero host overhead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib.util, os
import numpy as np, ml_dtypes
import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", "/opt/mlir-aie")
WA = os.path.join(MLIR_AIE, "programming_examples/basic/matrix_multiplication/"
                  "whole_array/whole_array.py")
spec = importlib.util.spec_from_file_location("wa", WA)
wa = importlib.util.module_from_spec(spec); sys.modules["wa"] = wa
spec.loader.exec_module(wa)
set_current_device(from_name("npu", n_cols=4))
bf = ml_dtypes.bfloat16

# (M, K, N, m, k, n, calls_per_encoder, label)
SHAPES = [
    (3072,  256,  384, 64, 64, 32,  1, "conv1 im2col"),
    (1536, 1152,  384, 64, 64, 32,  1, "conv2 im2col"),
    (1536,  384,  384, 64, 64, 32, 16, "q/k/v/out proj"),
    (1536,  384, 1536, 64, 64, 32,  4, "mlp fc1"),
    (1536, 1536,  384, 64, 64, 32,  4, "mlp fc2"),
    (1536,   64, 1536, 32, 64, 64, 24, "attn Q@K^T"),
    (1536, 1536,   64, 64, 64, 16, 24, "attn W@V"),
]

import os
only = os.environ.get("SHAPE_IDX")
if only is not None:
    SHAPES = [SHAPES[int(only)]]

total_ms = 0.0
total_gflop = 0.0
if only is None:
    print(f"{'shape':22s} {'tiles':10s} {'ms/call':>8s} {'GFLOPS':>8s} {'x':>4s} {'ms':>8s}  label")
for M, K, N, m, k, n, cnt, label in SHAPES:
    A = iron.tensor(np.zeros((M, K), dtype=bf), dtype=bf, device="npu")
    B = iron.tensor(np.zeros((K, N), dtype=bf), dtype=bf, device="npu")
    C = iron.zeros((M, N), dtype=np.float32, device="npu")
    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4,
              dtype_in_str="bf16", dtype_out_str="f32")
    for _ in range(5):
        wa.whole_array(A, B, C, **kw)
    s = []
    for _ in range(40):
        ret = wa.whole_array(A, B, C, **kw)
        s.append(getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", 0) / 1e6)
    lo, ms = float(np.min(s)), float(np.median(s))
    flop = 2.0 * M * K * N
    total_ms += lo * cnt
    total_gflop += flop * cnt / 1e9
    print(f"{f'{M}x{K}x{N}':22s} {f'{m}/{k}/{n}':10s} {lo:8.3f} {ms:8.3f} "
          f"{flop/(lo*1e-3)/1e9:8.1f} {cnt:4d} {lo*cnt:8.1f}  {label}")

if only is None:
    print(f"\nencoder GEMM floor: {total_ms:.1f} ms of NPU time for "
          f"{total_gflop:.1f} GFLOP (padded) -> "
          f"{total_gflop/(total_ms*1e-3)/1e3:.2f} TFLOPS avg")
    print("torch fp32 CPU, whole whisper-tiny encoder, 8 threads: 58.2 ms")
