#!/usr/bin/env python3
"""Separate the sources of per-call latency at the raw XRT level.

Steps:
  1  single run, buffers already on device, no sync at all
  2  two sequential runs of the SAME overlay, output of #1 is input of #2,
     no intermediate sync / read / numpy copy
  3  the same two runs submitted as one pyxrt.runlist
  4  cost of the sync/read that the IRON path does between calls

Everything uses one hw_context and one xclbin, so nothing here can be
reconfiguration: any difference is submit/wait, cache maintenance, or copies.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import AOT as AOT_DIR
import time
import numpy as np
import pyxrt

import rawxrt as R

# Compiled on demand, so a fresh checkout does not need a pre-populated cache.
_NAME = R.compile_shape(*R.padded_shape(512, 512, 512))
XCLBIN = str(AOT_DIR / f"{_NAME}.xclbin")
INSTS = str(AOT_DIR / f"{_NAME}.bin")
M = K = N = 512
ITERS = 50


def stats(fn, iters=ITERS, warmup=5):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1e3)
    return float(np.min(s)), float(np.median(s))


dev = pyxrt.device(0)
xclbin = pyxrt.xclbin(XCLBIN)
uuid = dev.register_xclbin(xclbin)
ctx = pyxrt.hw_context(dev, uuid)

kname = None
for k in xclbin.get_kernels():
    kname = k.get_name()
    break
print("kernel:", kname)
kernel = pyxrt.kernel(ctx, kname)

insts = np.fromfile(INSTS, dtype=np.uint32)
insts_bo = pyxrt.bo(dev, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
insts_bo.write(insts.tobytes(), 0)
insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
n_insts = insts.size

el = np.dtype(np.float16).itemsize          # bf16 == 2 bytes
sz = M * K * el


def mkbo(gid):
    return pyxrt.bo(dev, sz, pyxrt.bo.host_only, kernel.group_id(gid))


A = mkbo(3)
B = mkbo(4)
C = mkbo(5)
D = mkbo(5)
for b in (A, B, C, D):
    b.write(np.zeros(M * K, dtype=np.float16).tobytes(), 0)
    b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

print(f"{'step':52s} {'min ms':>9} {'median ms':>10}")


# --- 1: single run, nothing else -------------------------------------------
def one():
    r = kernel(3, insts_bo, n_insts, A, B, C)
    r.wait()


lo, md = stats(one)
print(f"{'1  single run, no sync/read':52s} {lo:9.3f} {md:10.3f}")


# --- 2: two chained runs, same overlay, no sync between ---------------------
def two_chained():
    r1 = kernel(3, insts_bo, n_insts, A, B, C)
    r1.wait()
    r2 = kernel(3, insts_bo, n_insts, C, B, D)   # C feeds straight back in
    r2.wait()


lo2, md2 = stats(two_chained)
print(f"{'2  two chained runs, no sync between':52s} {lo2:9.3f} {md2:10.3f}"
      f"   (per run {md2/2:.3f})")


# --- 3: the same two runs as one runlist -----------------------------------
try:
    r1 = pyxrt.run(kernel)
    r1.set_arg(0, 3)
    r1.set_arg(1, insts_bo)
    r1.set_arg(2, n_insts)
    r1.set_arg(3, A)
    r1.set_arg(4, B)
    r1.set_arg(5, C)
    r2 = pyxrt.run(kernel)
    r2.set_arg(0, 3)
    r2.set_arg(1, insts_bo)
    r2.set_arg(2, n_insts)
    r2.set_arg(3, C)
    r2.set_arg(4, B)
    r2.set_arg(5, D)

    rl = pyxrt.runlist(ctx)
    rl.add(r1)
    rl.add(r2)

    def runlist_two():
        rl.execute()
        rl.wait()

    lo3, md3 = stats(runlist_two)
    print(f"{'3  same two runs via pyxrt.runlist':52s} {lo3:9.3f} {md3:10.3f}"
          f"   (per run {md3/2:.3f})")
except Exception as e:
    print(f"3  runlist FAILED: {type(e).__name__}: {e}")


# --- 4: what the IRON path adds between calls ------------------------------
host = np.zeros(M * N, dtype=np.float16)


def with_sync_and_read():
    r = kernel(3, insts_bo, n_insts, A, B, C)
    r.wait()
    C.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    buf = C.read(sz, 0)
    np.frombuffer(buf, dtype=np.float16).astype(np.float32)


lo4, md4 = stats(with_sync_and_read)
print(f"{'4  single run + sync + read + f32 cast':52s} {lo4:9.3f} {md4:10.3f}")
print(f"{'   -> cost of the round trip per call':52s} {'':9} {md4-md:10.3f}")
