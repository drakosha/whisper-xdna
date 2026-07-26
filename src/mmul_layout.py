#!/usr/bin/env python3
"""Smallest possible check that our host-side block layouts match AMD's mmul.

One compute tile, one of their functions (matmul_a_b_bf16 = Q@K^T), operands
tiled in numpy and DMA'd as a straight contiguous copy. If the result matches
numpy after untiling, the layout convention is confirmed and the rest of the
port can be built on it; if it does not, this fails in seconds instead of inside
a 16-tile design.
"""
import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime import set_current_device

from repo_paths import AOT, KERNELS, air_includes


def amd_source(bq, bk, hd):
    """Their kernel file with our block sizes baked in."""
    return (f"#define lqp {bq}\n#define lkp {bk}\n"
            f"#define dk {hd}\n#define dk_full {hd}\n"
            f"#define dv {hd}\n#define dv_full {hd}\n"
            f'#include "{KERNELS / "attn_mmul.cc"}"\n')


def includes():
    return air_includes()


def build(bq, bk, hd):
    q_ty = np.ndarray[(bq * hd,), np.dtype[bfloat16]]
    k_ty = np.ndarray[(bk * hd,), np.dtype[bfloat16]]
    g_ty = np.ndarray[(bq * bk,), np.dtype[bfloat16]]
    src = amd_source(bq, bk, hd)
    inc = includes()
    # ONE ExternalFunction: IRON emits an object per declared symbol from the
    # same source, and two of them link two copies of attn_npu1.cc
    qk = ExternalFunction("attn_qk", source_string=src,
                          arg_types=[q_ty, k_ty, g_ty], include_dirs=inc)

    fq = ObjectFifo(q_ty, name="Q")
    fk = ObjectFifo(k_ty, name="K")
    fg = ObjectFifo(g_ty, name="G")

    def core(qi, ki, go, mm):
        q = qi.acquire(1)
        k = ki.acquire(1)
        g = go.acquire(1)
        mm(q, k, g)
        qi.release(1)
        ki.release(1)
        go.release(1)

    w = Worker(core, [fq.cons(), fk.cons(), fg.prod(), qk])
    rt = Runtime()
    with rt.sequence(q_ty, k_ty, g_ty) as (q, k, g):
        rt.start(w)
        rt.fill(fq.prod(), q)
        rt.fill(fk.prod(), k)
        rt.drain(fg.cons(), g, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mmul_qk(q_in: In, k_in: In, g_out: Out, *,
            bq: CompileTime[int], bk: CompileTime[int], hd: CompileTime[int]):
    return build(bq, bk, hd)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bq", type=int, default=48)
    p.add_argument("--bk", type=int, default=32)
    p.add_argument("--hd", type=int, default=64)
    p.add_argument("--out", default=str(AOT))
    a = p.parse_args()
    set_current_device(from_name("npu", n_cols=1))
    name = f"mmulqk_{a.bq}x{a.bk}x{a.hd}"
    spec = mmul_qk.specialize(bq=a.bq, bk=a.bk, hd=a.hd)
    spec.compile(xclbin_path=f"{a.out}/{name}.xclbin",
                 inst_path=f"{a.out}/{name}.bin")
    print("compiled", name)
