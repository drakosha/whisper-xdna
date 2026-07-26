#!/usr/bin/env python3
"""Can one BO be written by design A and read by design B, in different contexts?

This is the load-bearing assumption of the QK -> softmax -> AV chain. BOs are
allocated against a kernel's group_id, and each xclbin gets its own hw_context,
so sharing is not obviously legal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import AOT as AOT_DIR
import numpy as np, pyxrt
from ml_dtypes import bfloat16

import rawxrt as R

AOT = str(AOT_DIR)
SYNC_TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
SYNC_FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

dev = pyxrt.device(0)

def load(name):
    xb = pyxrt.xclbin(f"{AOT}/{name}.xclbin")
    uu = dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, uu)
    kn = [k.get_name() for k in xb.get_kernels()][0]
    k = pyxrt.kernel(ctx, kn)
    ins = np.fromfile(f"{AOT}/{name}.bin", dtype=np.uint32)
    ib = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    ib.write(ins.tobytes(), 0); ib.sync(SYNC_TO)
    return ctx, k, ib, ins.size

# A: matmul 512x64x512 (the QK shape) -> writes a 512x512 result
QK = R.compile_shape(*R.padded_shape(512, 64, 512))    # compiles if absent
AV = R.compile_shape(*R.padded_shape(512, 512, 64))
ctxA, kA, ibA, nA = load(QK)
# B: a row-op that consumes 512x512 -- softmax over 9000x1504 is a different
# shape, so use gelu at a matching element count if present, else just prove
# the BO can be bound to a second kernel at all.
ctxB, kB, ibB, nB = load(AV)   # matmul 512x512x64: takes 512x512 as its A

M, K, N = 512, 64, 512
A = pyxrt.bo(dev, M*K*2, pyxrt.bo.host_only, kA.group_id(3))
B = pyxrt.bo(dev, K*N*2, pyxrt.bo.host_only, kA.group_id(4))
C = pyxrt.bo(dev, M*N*4, pyxrt.bo.host_only, kA.group_id(5))   # QK output, f32

rng = np.random.default_rng(0)
a = (rng.random((M, K))*2-1).astype(bfloat16)
b = (rng.random((K, N))*2-1).astype(bfloat16)
A.write(a.tobytes(), 0); A.sync(SYNC_TO)
B.write(b.tobytes(), 0); B.sync(SYNC_TO)
kA(3, ibA, nA, A, B, C).wait()
print("design A (qk) ran")

# now try to bind C -- allocated against kA's group_id -- as an input of kB
B2 = pyxrt.bo(dev, 512*64*2, pyxrt.bo.host_only, kB.group_id(4))
C2 = pyxrt.bo(dev, 512*64*4, pyxrt.bo.host_only, kB.group_id(5))
B2.write(np.zeros(512*64, dtype=bfloat16).tobytes(), 0); B2.sync(SYNC_TO)
try:
    r = kB(3, ibB, nB, C, B2, C2)
    st = r.wait()
    print(f"cross-design BO reuse: OK  ({st})")
except Exception as e:
    print(f"cross-design BO reuse FAILED: {type(e).__name__}: {e}")

print("group_id(3) A-design:", kA.group_id(3), " B-design:", kB.group_id(3))
print("group_id(5) A-design:", kA.group_id(5), " B-design:", kB.group_id(5))
