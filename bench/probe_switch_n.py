#!/usr/bin/env python3
"""Does per-call cost grow with the number of distinct designs in the rotation?

The block interleaves 4 matmul designs; moving layernorm, softmax and GELU onto
the NPU takes it to 7. Kernel time fell while wall time rose, which points at
xclbin switching rather than compute. This cycles N distinct designs and reports
cost per call as N grows, holding the arithmetic per call constant.
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

# 8 shapes with identical arithmetic (512x512x512) but distinct specialisations,
# so any cost difference is switching, not FLOPs.
BASE = dict(M=512, K=512, N=512, n_aie_cols=4,
            dtype_in_str="bf16", dtype_out_str="f32")
# only tile configs already proven to build on npu1 in this work
TILES = [(64, 64, 32), (32, 64, 64), (64, 64, 16), (32, 64, 32)]

bufs = []
for (m, k, n) in TILES:
    A = iron.tensor(np.zeros((512, 512), dtype=bf), dtype=bf, device="npu")
    B = iron.tensor(np.zeros((512, 512), dtype=bf), dtype=bf, device="npu")
    C = iron.zeros((512, 512), dtype=np.float32, device="npu")
    bufs.append((A, B, C, dict(BASE, m=m, k=k, n=n)))


def call(i):
    A, B, C, kw = bufs[i]
    t0 = time.perf_counter()
    ret = wa.whole_array(A, B, C, **kw)
    e2e = (time.perf_counter() - t0) * 1e3
    ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", 0)
    return ns / 1e6, e2e


for i in range(len(TILES)):        # warm / JIT every design
    for _ in range(3):
        call(i)

print(f"{'designs in rotation':>20}  {'npu ms/call':>12}  {'e2e ms/call':>12}")
for n_designs in [1, 2, 3, 4]:
    s = []
    for j in range(32):
        s.append(call(j % n_designs))
    npu = np.mean([x[0] for x in s])
    e2e = np.mean([x[1] for x in s])
    print(f"{n_designs:>20}  {npu:>12.3f}  {e2e:>12.3f}")
