#!/usr/bin/env python3
"""Is the per-call cost in the pipeline the GEMM, or switching xclbins?

Runs the same two designs (a) repeated back-to-back and (b) alternating,
in one process, reporting the runtime-reported npu_time for each call.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib.util, os, time
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

SHAPES = {
    "A_1536x384x384":   (1536, 384, 384, 64, 64, 32),
    "B_1536x1536x64":   (1536, 1536, 64, 64, 64, 16),
}
bufs = {}
for name, (M, K, N, m, k, n) in SHAPES.items():
    bufs[name] = (
        iron.tensor(np.zeros((M, K), dtype=bf), dtype=bf, device="npu"),
        iron.tensor(np.zeros((K, N), dtype=bf), dtype=bf, device="npu"),
        iron.zeros((M, N), dtype=np.float32, device="npu"),
    )


def call(name):
    M, K, N, m, k, n = SHAPES[name]
    A, B, C = bufs[name]
    t0 = time.perf_counter()
    ret = wa.whole_array(A, B, C, M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=4,
                         dtype_in_str="bf16", dtype_out_str="f32")
    e2e = (time.perf_counter() - t0) * 1e3
    ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", 0)
    return ns / 1e6, e2e


for name in SHAPES:                       # warm / JIT both designs
    for _ in range(3):
        call(name)

for name in SHAPES:
    s = [call(name) for _ in range(10)]
    print(f"repeated  {name:20s} npu={np.mean([x[0] for x in s]):6.3f} ms  "
          f"e2e={np.mean([x[1] for x in s]):6.3f} ms")

alt = {n: [] for n in SHAPES}
for i in range(20):
    name = list(SHAPES)[i % 2]
    alt[name].append(call(name))
for name in SHAPES:
    s = alt[name]
    print(f"alternating {name:18s} npu={np.mean([x[0] for x in s]):6.3f} ms  "
          f"e2e={np.mean([x[1] for x in s]):6.3f} ms")
