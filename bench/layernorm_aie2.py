#!/usr/bin/env python3
"""Standalone layernorm on AIE2 (Phoenix / XDNA1), no learned affine.

The stock ml/norm design is aie2p-only; on aie2 it fails to link with
"ld.lld: error: undefined symbol: sqrtf". This uses layer_norm_gb.cc instead.

Standalone so the per-row cost can be measured at whisper shapes and fed into
the fused-encoder projection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import KERNELS
import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.device import from_name
from aie.helpers.taplib import TensorTiler2D
from aie.utils import config
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime import set_current_device

KERNEL = str(KERNELS / "layer_norm_gb.cc")
N_CORES = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def layer_norm(
    a_in: In,
    c_out: Out,
    *,
    sequence_length: CompileTime[int] = 1536,
    embedding_dim: CompileTime[int] = 384,
):
    device = iron.get_current_device()
    rows_per_core = sequence_length // N_CORES

    tensor_ty = np.ndarray[(sequence_length, embedding_dim), np.dtype[bfloat16]]
    chunk_ty = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    norm_fn = ExternalFunction(
        "layer_norm_bf16",
        source_file=KERNEL,
        arg_types=[chunk_ty, chunk_ty, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    of_ins = [ObjectFifo(chunk_ty, name=f"in_{i}") for i in range(N_CORES)]
    of_outs = [ObjectFifo(chunk_ty, name=f"out_{i}") for i in range(N_CORES)]
    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            kernel(ei, eo, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_fn,
            [
                of_ins[i].cons(),
                of_outs[i].prod(),
                norm_fn,
            ],
        )
        for i in range(N_CORES)
    ]

    taps = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, c):
        rt.start(*workers)
        for i in range(N_CORES):
            rt.fill(of_ins[i].prod(), a, taps[i])
        for i in range(N_CORES):
            rt.drain(of_outs[i].cons(), c, taps[i], wait=True)

    return Program(device, rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--sequence_length", type=int, default=1536)
    p.add_argument("-e", "--embedding_dim", type=int, default=384)
    p.add_argument("--iters", type=int, default=20)
    opts = p.parse_args()

    set_current_device(from_name("npu", n_cols=4))
    rows, cols = opts.sequence_length, opts.embedding_dim

    rng = np.random.default_rng(0)
    a_np = rng.normal(0, 1.0, size=(rows, cols)).astype(bfloat16)
    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    c_t = iron.zeros((rows, cols), dtype=bfloat16, device="npu")

    kw = dict(sequence_length=rows, embedding_dim=cols)
    bench = run_iters(layer_norm, a_t, c_t, warmup=3,
                      iters=opts.iters, **kw)

    out = c_t.numpy().reshape(rows, cols).astype(np.float32)
    x = a_np.astype(np.float32)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + 1e-5)

    d = out - ref
    rel = np.linalg.norm(d) / np.linalg.norm(ref)
    npu_min = bench.npu.min_us / 1e3 if bench.npu else float("nan")
    npu_avg = bench.npu.avg_us / 1e3 if bench.npu else float("nan")
    print(f"layer_norm {rows}x{cols}: npu min={npu_min:.3f} ms "
          f"avg={npu_avg:.3f} ms  e2e min={bench.e2e.min_us/1e3:.3f} ms")
    print(f"  rel L2 vs fp32 numpy = {rel:.6f}   max abs = {np.abs(d).max():.5f}")
    print("  PASS" if rel < 0.02 else "  FAIL")


if __name__ == "__main__":
    main()
