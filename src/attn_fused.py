#!/usr/bin/env python3
"""Fused attention for one head: QK -> softmax -> PV without the scores ever
leaving a compute tile.

Why this shape and not the one we sketched before:

  * ONE head on the whole 4x4 array, not one head per column. AMD's own npu1
    driver says `num_heads_per_unroll=1 (4x4 array can only do one head at a
    time)`; a sketch of 4 heads in 4 columns was wrong.
  * The array is split along Q instead: 16 tiles x Bq rows, WAVES passes to
    cover the context.
  * K and V travel in ONE packed fifo object, not two. A compute tile has 2
    input and 2 output DMA channels, and Q + K + V + O would need three inputs:
        error: tile (0, 3) requires 3 input/1 output DMA channels, but only
        2 input/2 output available
    (the same failure an earlier fused layernorm hit). Packed KV keeps it at
    2 in / 1 out.
  * K/V are broadcast from each column's memtile to its four tiles, so the shim
    carries them once per column rather than once per tile.

L1 per compute tile, the constraint that sets Bq and Bk:

    Q      Bq x hd  bf16            6 KB   (48 x 64)
    KV     2 x Bk x hd bf16, depth 2   16 KB   (2 x 32 x 64, double buffered)
    O out  Bq x hd  bf16, depth 2    12 KB
    acc    Bq x hd  fp32 (local)     12 KB
    S      Bq x Bk  bf16 (local)      3 KB
                                    -------
                                     49 KB of 64 KB
"""
import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (CompileTime, In, Out, ObjectFifo, Program, Runtime,
                      Worker)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils.hostruntime import set_current_device

from repo_paths import AOT, KERNELS, air_includes

N_COLS, N_ROWS = 4, 4
N_TILES = N_COLS * N_ROWS


def build(ctx, hd, bq, bk, nvalid=0, sym="attn_step", source="attn_mmul.cc"):
    assert ctx % (N_TILES * bq) == 0, (ctx, bq)
    assert ctx % bk == 0, (ctx, bk)
    waves = ctx // (N_TILES * bq)
    nkb = ctx // bk

    # flat: every object is one block already in mmul layout, tiled on the host
    nv = nvalid or ctx
    q_ty = np.ndarray[(bq * hd,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * bk * hd,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(bq * hd,), np.dtype[bfloat16]]
    q_all = np.ndarray[(ctx, hd), np.dtype[bfloat16]]
    kv_all = np.ndarray[(2 * ctx, hd), np.dtype[bfloat16]]
    o_all = np.ndarray[(ctx, hd), np.dtype[bfloat16]]
    q_l2 = np.ndarray[(N_ROWS * bq, hd), np.dtype[bfloat16]]
    o_l2 = np.ndarray[(N_ROWS * bq, hd), np.dtype[bfloat16]]

    # One ExternalFunction only -- IRON emits an object per declared symbol from
    # the same source, and two of them link two copies of their file
    # (ld.lld: error: duplicate symbol: accum_sp_r_s).
    includes = air_includes()
    src = ((f"#define ATTN_NVALID {nv}\n" if nvalid else "")
           + f"#define lqp {bq}\n#define lkp {bk}\n"
           f"#define dk {hd}\n#define dk_full {hd}\n"
           f"#define dv {hd}\n#define dv_full {hd}\n"
           f'#include "{KERNELS / source}"\n')
    fn = ExternalFunction(sym, source_string=src,
                          arg_types=[q_ty, kv_ty, o_ty, np.int32, np.int32],
                          include_dirs=includes)

    def core_fn(qf, kvf, of, kernel):
        for _ in range_(waves):
            q = qf.acquire(1)
            o = of.acquire(1)
            for i in range_(nkb):
                kv = kvf.acquire(1)
                kernel(q, kv, o, i, nkb)
                kvf.release(1)
            qf.release(1)
            of.release(1)

    workers, q_prod, kv_prod, o_cons = [], [], [], []
    for c in range(N_COLS):
        qin = ObjectFifo(q_l2, name=f"Qin_{c}")
        qsub = qin.cons().split(
            [bq * hd * r for r in range(N_ROWS)],
            obj_types=[q_ty] * N_ROWS,
            names=[f"Q_{c}_{r}" for r in range(N_ROWS)])
        # one KV fifo per column, broadcast to its four tiles: the SAME object
        # is consumed by all of them, which is what keeps the shim at one
        # stream per column instead of one per tile
        kvin = ObjectFifo(kv_ty, name=f"KVin_{c}", depth=2)
        oout = ObjectFifo(o_l2, name=f"Oout_{c}")
        osub = oout.prod().join(
            [bq * hd * r for r in range(N_ROWS)],
            obj_types=[o_ty] * N_ROWS,
            names=[f"O_{c}_{r}" for r in range(N_ROWS)],
            depths=[2] * N_ROWS)
        for r in range(N_ROWS):
            workers.append(Worker(
                core_fn,
                [qsub[r].cons(), kvin.cons(), osub[r].prod(), fn]))
        q_prod.append(qin.prod())
        kv_prod.append(kvin.prod())
        o_cons.append(oout.cons())

    rows_per_col = ctx // N_COLS
    qtaps = TensorTiler2D.simple_tiler((ctx, hd), (rows_per_col, hd))
    otaps = TensorTiler2D.simple_tiler((ctx, hd), (rows_per_col, hd))
    # K/V are re-streamed once per wave: the cores consume waves*nkb objects,
    # and a size with stride 0 repeats the same region rather than needing the
    # host to send it twice. Split as [waves, nkb, 2*bk, hd] because a single
    # 2*ctx dimension is 3072, over the 1023 a BD dimension allows.
    kvtap = TensorAccessPattern((2 * ctx, hd), 0,
                                sizes=[waves, nkb, 2 * bk, hd],
                                strides=[0, 2 * bk * hd, hd, 1])

    rt = Runtime()
    with rt.sequence(q_all, kv_all, o_all) as (q, kv, o):
        rt.start(*workers)
        for c in range(N_COLS):
            rt.fill(q_prod[c], q, qtaps[c])
        for c in range(N_COLS):
            rt.fill(kv_prod[c], kv, kvtap)
        for c in range(N_COLS):
            rt.drain(o_cons[c], o, otaps[c], wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def attn_head(q_in: In, kv_in: In, o_out: Out, *,
              ctx: CompileTime[int], hd: CompileTime[int],
              bq: CompileTime[int], bk: CompileTime[int],
              nvalid: CompileTime[int] = 0):
    return build(ctx, hd, bq, bk, nvalid)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=1536)
    p.add_argument("--hd", type=int, default=64)
    p.add_argument("--bq", type=int, default=48)
    p.add_argument("--bk", type=int, default=32)
    p.add_argument("--nvalid", type=int, default=0)
    p.add_argument("--out", default=str(AOT))
    a = p.parse_args()
    set_current_device(from_name("npu", n_cols=N_COLS))
    name = (f"attnm_{a.ctx}x{a.hd}_q{a.bq}k{a.bk}"
            + (f"v{a.nvalid}" if a.nvalid else ""))
    spec = attn_head.specialize(ctx=a.ctx, hd=a.hd, bq=a.bq, bk=a.bk,
                                nvalid=a.nvalid)
    spec.compile(xclbin_path=f"{a.out}/{name}.xclbin",
                 inst_path=f"{a.out}/{name}.bin")
    print("compiled", name)
