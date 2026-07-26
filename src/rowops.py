#!/usr/bin/env python3
"""Row-wise NPU kernels for the whisper encoder: layernorm and a correct softmax.

Both designs hand each core one COMPLETE row, so the reduction (mean/var, or
max/sum) spans the whole row. This is the part the stock ml/softmax design gets
wrong: it feeds fixed 1024-element tiles, so a 512- or 1536-wide attention row is
normalised per tile instead of per row.

All heads of a layer are softmaxed in one invocation by stacking the per-head
score matrices into (n_head*L, L) -- one design launch instead of n_head.
"""
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.utils import config

_KERNELS = Path(__file__).resolve().parents[1] / "kernels"
# 16 compute tiles, staged through the memtiles. A direct 16-worker design is
# rejected by the shim ("no ShimNOCTile has sufficient DMA capacity"): npu1 has
# 4 shim NOC tiles with 2 in / 2 out DMA channels each, so only 8 cores can be
# fed straight from it. Routing one fifo per column into a memtile, which then
# splits it across that column's 4 tiles, lifts the limit -- worth ~0.5 s on a
# large-v3 pass (GELU 1395 -> 882 ms, softmax 1033 -> 562 ms). Verified with the
# golden suite at 16. Set ROWOP_CORES=8 for the older shim-fed design.
N_CORES = int(os.environ.get("ROWOP_CORES", "16"))
N_COLS = 4


def _split_bd(n, hi_limit, lo_limit=1023, even_lo=False):
    """Factor n into (hi, lo) that fit two dma_bd size fields.

    The fields are not all the same width: dimension 0 is the BD's iteration
    count and caps at 64 ("Size 0 exceeds the [1:64] range"), the rest cap at
    1023. hi is taken as large as allowed so the outer stride stays small.

    even_lo is for the INNERMOST dimension: a BD moves whole 4-byte words, so a
    1-element bf16 innermost size is rejected outright --
      error: 'aie.dma_bd' op Transfer sizes must be multiples of 4 bytes.
      1 elements at 2 bytes each equal 2 bytes, which is not divisible by 4.
    which is what a 320-wide window produced (320 = 320 x 1).
    """
    for hi in range(min(n, hi_limit), 0, -1):
        if n % hi:
            continue
        lo = n // hi
        if lo <= lo_limit and (not even_lo or lo % 2 == 0):
            return hi, lo
    raise ValueError(f"cannot split {n} into dma_bd dimensions "
                     f"({hi_limit}, {lo_limit}, even_lo={even_lo})")


def _row_design(sym, source, rows, cols, needs_lut=False, valid=0,
                in_dtype=bfloat16, out_cols=0):
    """valid > 0 gives each core a `valid`-wide WINDOW out of each `cols`-wide
    row instead of the whole row.

    That is what a chained softmax needs. Its input is the QK GEMM's output
    buffer, whose row is padded from 1500 up to the design's 1536, and the pad
    holds q.0 = 0.0 rather than -inf -- so a full-row softmax normalises over 36
    columns of exp(0 - max) that are not attention at all (measured: 1.4% of the
    probability mass on average, 8.9% worst row). Reading a 1504-wide window
    leaves 4 such columns instead of 36. The columns outside the window are
    neither read nor written; the AV GEMM that consumes the result multiplies
    them by zero-padded rows of V, so their contents never reach the output.
    """
    w = cols if valid <= 0 else valid
    # Every row kernel steps 16 elements at a time and does not handle a tail:
    # a width that is not a multiple of 16 would leave the last elements of each
    # row unwritten, which is a wrong answer rather than a compile error.
    if w % 16:
        raise ValueError(f"{sym}: row width {w} is not a multiple of 16")
    # in_dtype lets a kernel read the fp32 accumulator a GEMM just wrote instead
    # of a bf16 copy the host had to make; out_cols > 0 makes the OUTPUT rows
    # `out_cols` wide, so a `w`-wide result lands as a column band of a wider
    # buffer -- which is how one fc1 column chunk writes into fc2's A operand.
    in_row_ty = np.ndarray[(w,), np.dtype[in_dtype]]
    row_ty = np.ndarray[(w,), np.dtype[bfloat16]]
    in_tensor_ty = np.ndarray[(rows, cols), np.dtype[in_dtype]]
    ocols = out_cols if out_cols > 0 else cols
    tensor_ty = np.ndarray[(rows, ocols), np.dtype[bfloat16]]
    # The wide layout needs the rows to split evenly over 4 columns and then over
    # the compute tiles under each; a row count that does not divide (the host
    # softmax path stacks 6 x 1500 = 9000) drops to the 8-core layout instead of
    # failing to compile.
    n_cores = N_CORES if rows % N_CORES == 0 else 8
    assert rows % n_cores == 0, (rows, n_cores)
    rows_per_core = rows // n_cores

    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    includes = [config.cxx_header_path(),
                str(Path(config.cxx_header_path()) / "aie_kernels"),
                str(Path(config.cxx_header_path()) / "aie_kernels" / "aie2"),
                str(runtime_dir)]
    if needs_lut:
        # getExpBf16 pulls in the exp lookup table; without lut_based_ops.cpp in
        # the same TU the link fails with
        #   ld.lld: error: undefined symbol: exp_flut_cd
        # This mirrors what aie.iron.kernels.activation._create_lut_kernel does
        # for the stock LUT kernels on aie2.
        src = (f'#include "{_KERNELS / source}"\n'
               f'#include "{runtime_dir / "lut_based_ops.cpp"}"\n')
        fn = ExternalFunction(sym, source_string=src,
                              arg_types=[in_row_ty, row_ty, np.int32],
                              include_dirs=includes)
    else:
        fn = ExternalFunction(sym, source_file=str(_KERNELS / source),
                              arg_types=[in_row_ty, row_ty, np.int32],
                              include_dirs=includes)

    def core_fn(of_in, of_out, kernel):
        for _ in range_(rows_per_core):
            a = of_in.acquire(1)
            c = of_out.acquire(1)
            kernel(a, c, w)
            of_in.release(1)
            of_out.release(1)

    if n_cores > N_COLS * 2:
        # Past 8 workers the shim runs out of DMA channels -- one fifo per core
        # straight to L3 needs N_CORES in + N_CORES out across 4 ShimNOC tiles:
        #   error: no ShimNOCTile has sufficient DMA capacity for 0 input/1
        #   output channels near centroid column 0
        # So a column takes ONE fifo from the shim, and its memtile splits the
        # object into a row per compute tile (and joins the results back). The
        # shim then carries 4 in + 4 out regardless of how many cores are fed,
        # and the memtile stays inside its 6 MM2S / 6 S2MM budget (1+4 each).
        per_col = n_cores // N_COLS
        rows_per_col = rows // N_COLS
        assert rows_per_col % per_col == 0, (rows_per_col, per_col)
        l2_ty = np.ndarray[(per_col * w,), np.dtype[bfloat16]]
        in_l2_ty = np.ndarray[(per_col * w,), np.dtype[in_dtype]]
        offs = [w * j for j in range(per_col)]
        workers = []
        prods, cons = [], []
        for cl_ in range(N_COLS):
            fin = ObjectFifo(in_l2_ty, name=f"iL2_{cl_}")
            subs_in = fin.cons().split(
                offs, obj_types=[in_row_ty] * per_col,
                names=[f"i{cl_}_{j}" for j in range(per_col)])
            fout = ObjectFifo(l2_ty, name=f"oL2_{cl_}")
            subs_out = fout.prod().join(
                offs, obj_types=[row_ty] * per_col,
                names=[f"o{cl_}_{j}" for j in range(per_col)],
                depths=[2] * per_col)
            for j in range(per_col):
                workers.append(Worker(
                    core_fn,
                    [subs_in[j].cons(), subs_out[j].prod(), fn]))
            prods.append(fin.prod())
            cons.append(fout.cons())
        if valid <= 0:
            taps = TensorTiler2D.simple_tiler((rows, cols),
                                              (rows_per_col, cols))
            otaps = (taps if ocols == cols else
                     [TensorAccessPattern(
                         (rows, ocols), i * rows_per_col * ocols,
                         sizes=[*_split_bd(rows_per_col, 64),
                                *_split_bd(w, 1023, even_lo=True)],
                         strides=[_split_bd(rows_per_col, 64)[1] * ocols, ocols,
                                  _split_bd(w, 1023, even_lo=True)[1], 1])
                      for i in range(N_COLS)])
        else:
            # Same 4-dimension windowed pattern as the direct path -- the fifo
            # chops the stream into per_col*w objects on its own, so staging
            # through L2 costs no extra dma_bd dimension.
            rh, rl = _split_bd(rows_per_col, 64)
            ch, cl = _split_bd(w, 1023, even_lo=True)
            taps = [TensorAccessPattern((rows, cols), i * rows_per_col * cols,
                                        sizes=[rh, rl, ch, cl],
                                        strides=[rl * cols, cols, cl, 1])
                    for i in range(N_COLS)]
            otaps = taps
        rt = Runtime()
        with rt.sequence(in_tensor_ty, tensor_ty) as (a, c):
            rt.start(*workers)
            for i in range(N_COLS):
                rt.fill(prods[i], a, taps[i])
            for i in range(N_COLS):
                rt.drain(cons[i], c, otaps[i], wait=True)
        return Program(iron.get_current_device(), rt).resolve_program()

    of_ins = [ObjectFifo(in_row_ty, name=f"i{i}") for i in range(n_cores)]
    of_outs = [ObjectFifo(row_ty, name=f"o{i}") for i in range(n_cores)]

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), fn])
        for i in range(n_cores)
    ]
    if valid <= 0:
        taps = TensorTiler2D.simple_tiler((rows, cols), (rows_per_core, cols))
    else:
        # A dma_bd dimension size must fit [0:1023], and both the per-core row
        # count (1152) and the window width (1504) are over that, so each is
        # split across two of the four available dimensions.
        rh, rl = _split_bd(rows_per_core, 64)
        ch, cl = _split_bd(w, 1023, even_lo=True)
        taps = [TensorAccessPattern((rows, cols), i * rows_per_core * cols,
                                    sizes=[rh, rl, ch, cl],
                                    strides=[rl * cols, cols, cl, 1])
                for i in range(n_cores)]

    assert ocols == cols, "narrow output only implemented for the wide layout"
    rt = Runtime()
    with rt.sequence(in_tensor_ty, tensor_ty) as (a, c):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_ins[i].prod(), a, taps[i])
        for i in range(n_cores):
            rt.drain(of_outs[i].cons(), c, taps[i], wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def layer_norm(a_in: In, c_out: Out, *,
               rows: CompileTime[int], cols: CompileTime[int]):
    return _row_design("layer_norm_bf16", "layer_norm_gb.cc", rows, cols)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row(a_in: In, c_out: Out, *,
                rows: CompileTime[int], cols: CompileTime[int]):
    return _row_design("softmax_row_bf16", "softmax_row.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row2(a_in: In, c_out: Out, *,
                 rows: CompileTime[int], cols: CompileTime[int]):
    return _row_design("softmax_row2_bf16", "softmax_row2.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row3(a_in: In, c_out: Out, *,
                 rows: CompileTime[int], cols: CompileTime[int]):
    return _row_design("softmax_row3_bf16", "softmax_row3.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row(a_in: In, c_out: Out, *,
             rows: CompileTime[int], cols: CompileTime[int]):
    return _row_design("gelu_row_bf16", "gelu_row.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row3_win(a_in: In, c_out: Out, *,
                     rows: CompileTime[int], cols: CompileTime[int],
                     valid: CompileTime[int]):
    """softmax_row3 over a `valid`-wide window of each `cols`-wide row."""
    return _row_design("softmax_row3_bf16", "softmax_row3.cc", rows, cols,
                       needs_lut=True, valid=valid)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row4(a_in: In, c_out: Out, *,
                 rows: CompileTime[int], cols: CompileTime[int]):
    """softmax_row3 without the second exp evaluation. See softmax_row4.cc."""
    return _row_design("softmax_row4_bf16", "softmax_row4.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row4_win(a_in: In, c_out: Out, *,
                     rows: CompileTime[int], cols: CompileTime[int],
                     valid: CompileTime[int]):
    """softmax_row4 over a `valid`-wide window of each `cols`-wide row."""
    return _row_design("softmax_row4_bf16", "softmax_row4.cc", rows, cols,
                       needs_lut=True, valid=valid)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row2(a_in: In, c_out: Out, *,
              rows: CompileTime[int], cols: CompileTime[int]):
    """GELU via the logistic form: no 1+tanh cancellation. See gelu_row2.cc."""
    return _row_design("gelu_row2_bf16", "gelu_row2.cc", rows, cols,
                       needs_lut=True)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row3(a_in: In, c_out: Out, *,
              rows: CompileTime[int], cols: CompileTime[int]):
    """erf-GELU by piecewise polynomial, no exp/tanh table. See gelu_row3.cc."""
    return _row_design("gelu_row3_bf16", "gelu_row3.cc", rows, cols,
                       needs_lut=False)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row4(a_in: In, c_out: Out, *,
              rows: CompileTime[int], cols: CompileTime[int]):
    """Same erf-GELU on a centred variable: 12 Horner steps, no hi/lo anything.
    See gelu_row4.cc."""
    return _row_design("gelu_row4_bf16", "gelu_row4.cc", rows, cols,
                       needs_lut=False)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row4f(a_in: In, c_out: Out, *,
               rows: CompileTime[int], cols: CompileTime[int],
               out_cols: CompileTime[int] = 0):
    """gelu_row4 reading an fp32 GEMM accumulator, writing bf16 into a column
    band of an `out_cols`-wide buffer. See gelu_row4f.cc."""
    return _row_design("gelu_row4f_bf16", "gelu_row4f.cc", rows, cols,
                       needs_lut=False, in_dtype=np.float32, out_cols=out_cols)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gelu_row4_rne(a_in: In, c_out: Out, *,
                  rows: CompileTime[int], cols: CompileTime[int]):
    """gelu_row4 with round-to-nearest-even instead of the default truncation."""
    return _row_design("gelu_row4_bf16", "gelu_row4_rne.cc", rows, cols,
                       needs_lut=False)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_row4r_win(a_in: In, c_out: Out, *,
                      rows: CompileTime[int], cols: CompileTime[int],
                      valid: CompileTime[int]):
    """softmax_row4 with round-to-nearest-even instead of truncation."""
    return _row_design("softmax_row4r_bf16", "softmax_row4_rne.cc", rows, cols,
                       needs_lut=True, valid=valid)
